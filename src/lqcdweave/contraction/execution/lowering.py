"""Conservative structural lowering for generic two-input contractions.

The lowering is deliberately independent of the physics frontend.  It only
describes how a binary explicit-index equation can be arranged as batched
matrix multiplication.  Numerical arrays are accepted only by
:func:`execute_matmul`; the bounded structural cache stores no arrays,
devices, or streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
import math
import operator
from typing import Any

import numpy as np
from lqcdweave.contracts.correlator.api import AxisPattern, parse_axis_pattern
from .diagnostics import debug_enabled
from .matrix_product import bind_pointer_gemm, _matrix_strides


_SUPPORTED_DTYPES = frozenset(
    np.dtype(dtype).str for dtype in (np.float32, np.float64, np.complex64, np.complex128)
)
_LOWERING_CACHE_SIZE = 256
# CuPy's strided-batched GEMM does not split long reductions: on V100 a
# (6, 6, 1e6) x (6, 1e6, 4) complex64 product takes 104 ms, per-matrix GEMM
# 3.5 ms. Small batches use per-matrix GEMM; products with few 32x32 output
# tiles split the reduction (with enough tiles the batched GEMM is faster).
_LONG_BATCHED_K = 4096
_SPLIT_K = 64
_SPLIT_MAX_TILES = 256


def _product(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


@dataclass(frozen=True, slots=True)
class MatmulLowering:
    """Immutable axis recipe for one binary batched-matmul contraction.

    The fields contain only axis records, integers, tuples and dtype strings. A
    result from :func:`execute_matmul` owns the matrix-product storage; any
    reshape/transpose view is therefore allowed to survive node release.
    """

    equation: AxisPattern
    left_subscript: tuple[int, ...]
    right_subscript: tuple[int, ...]
    output_subscript: tuple[int, ...]
    batch_labels: tuple[int, ...]
    left_labels: tuple[int, ...]
    right_labels: tuple[int, ...]
    reduction_labels: tuple[int, ...]
    left_permutation: tuple[int, ...]
    right_permutation: tuple[int, ...]
    output_permutation: tuple[int, ...]
    batch_shape: tuple[int, ...]
    left_free_shape: tuple[int, ...]
    reduction_shape: tuple[int, ...]
    right_free_shape: tuple[int, ...]
    dtype: str
    execution_permutations: tuple = field(init=False, repr=False)
    execution_shapes: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'execution_permutations', tuple(
            None if p == tuple(range(len(p))) else p
            for p in (self.left_permutation, self.right_permutation, self.output_permutation)))
        object.__setattr__(self, 'execution_shapes', (
            self.batch_shape + (self.matrix_m, self.matrix_k),
            self.batch_shape + (self.matrix_k, self.matrix_n), self.grouped_shape))

    @property
    def matrix_m(self) -> int:
        return _product(self.left_free_shape)

    @property
    def matrix_k(self) -> int:
        return _product(self.reduction_shape)

    @property
    def matrix_n(self) -> int:
        return _product(self.right_free_shape)

    @property
    def grouped_labels(self) -> tuple[int, ...]:
        return self.batch_labels + self.left_labels + self.right_labels

    @property
    def grouped_shape(self) -> tuple[int, ...]:
        return self.batch_shape + self.left_free_shape + self.right_free_shape


def _normalise_dtypes(operand_dtypes: tuple[Any, ...]) -> tuple[str, ...] | None:
    if len(operand_dtypes) != 2:
        return None
    try:
        normalized = tuple(np.dtype(value).str for value in operand_dtypes)
    except (TypeError, ValueError):
        return None
    if normalized[0] != normalized[1] or normalized[0] not in _SUPPORTED_DTYPES:
        return None
    return normalized


@lru_cache(maxsize=_LOWERING_CACHE_SIZE)
def _lower_binary_matmul_cached(
    equation: AxisPattern,
    operand_shapes: tuple[tuple[int, ...], ...],
    operand_dtypes: tuple[str, ...],
) -> MatmulLowering | None:
    """Build one immutable structural recipe, or cache ``None`` on fallback."""

    if len(operand_shapes) != 2 or len(operand_dtypes) != 2:
        return None
    if len(equation.inputs) != 2:
        return None
    left_text, right_text = equation.inputs
    output_text = equation.output
    left = tuple(left_text)
    right = tuple(right_text)
    output = tuple(output_text)
    if len(set(left)) != len(left) or len(set(right)) != len(right) or len(set(output)) != len(output):
        return None
    if len(left) != len(operand_shapes[0]) or len(right) != len(operand_shapes[1]):
        return None
    if any(any(int(dimension) <= 0 for dimension in shape) for shape in operand_shapes):
        return None
    if _normalise_dtypes(operand_dtypes) is None:
        return None

    left_set, right_set, output_set = set(left), set(right), set(output)
    if not output_set.issubset(left_set | right_set):
        return None
    # A label occurring on only one input must be retained.  Otherwise the
    # contraction contains a single-sided reduction, which is intentionally
    # left to the reference einsum path.
    if any(label not in output_set for label in left_set ^ right_set):
        return None

    common = left_set & right_set
    batch = tuple(label for label in output if label in common)
    left_free = tuple(label for label in output if label in left_set - right_set)
    right_free = tuple(label for label in output if label in right_set - left_set)
    reduction = tuple(label for label in left if label in common and label not in output_set)
    grouped = batch + left_free + right_free
    if set(grouped) != output_set or len(grouped) != len(output):
        return None
    if set(reduction) != common - output_set:
        return None

    dimensions: dict[int | str, int] = {}
    for labels, shape in ((left, operand_shapes[0]), (right, operand_shapes[1])):
        for label, dimension in zip(labels, shape):
            dimension = int(dimension)
            previous = dimensions.setdefault(label, dimension)
            if previous != dimension:
                return None

    left_group = batch + left_free + reduction
    right_group = batch + reduction + right_free
    left_permutation = tuple(left.index(label) for label in left_group)
    right_permutation = tuple(right.index(label) for label in right_group)
    output_permutation = tuple(grouped.index(label) for label in output)
    return MatmulLowering(
        equation=equation,
        left_subscript=left_text,
        right_subscript=right_text,
        output_subscript=output_text,
        batch_labels=batch,
        left_labels=left_free,
        right_labels=right_free,
        reduction_labels=reduction,
        left_permutation=left_permutation,
        right_permutation=right_permutation,
        output_permutation=output_permutation,
        batch_shape=tuple(dimensions[label] for label in batch),
        left_free_shape=tuple(dimensions[label] for label in left_free),
        reduction_shape=tuple(dimensions[label] for label in reduction),
        right_free_shape=tuple(dimensions[label] for label in right_free),
        dtype=operand_dtypes[0],
    )


def lower_binary_matmul(
    equation: AxisPattern | str,
    operand_shapes: tuple[tuple[int, ...], ...],
    operand_dtypes: tuple[Any, ...],
) -> MatmulLowering | None:
    """Return a cached matmul recipe for a supported binary equation.

    ``None`` means that the caller must use the existing reference contraction
    implementation.  The cache key is normalized to immutable equation,
    shape, and dtype descriptions before lookup.
    """

    if not isinstance(equation, (str, AxisPattern)):
        return None
    try:
        if isinstance(equation, str) and equation.count("->") != 1:
            return None
        equation = parse_axis_pattern(equation)
        shapes = tuple(
            tuple(operator.index(dimension) for dimension in shape)
            for shape in operand_shapes
        )
    except (TypeError, ValueError):
        return None
    dtypes = _normalise_dtypes(tuple(operand_dtypes))
    if dtypes is None:
        return None
    return _lower_binary_matmul_cached(equation, shapes, dtypes)


def _matrix_product(left, right, *, array_module, method, public=None, native=None):
    # The entry choice is frozen, but backend wrappers remain observable (for
    # measurement/failure injection); this lookup performs no layout analysis.
    current = getattr(array_module, method)
    if (native is not None and current is public and getattr(public, '_func', None) is native
            and not debug_enabled() and left.ndim > 2 and right.ndim > 2
            and left.shape[:-2] == right.shape[:-2] and left.dtype == right.dtype):
        # These matrices already have the GUFunc's default axes, identical
        # batch extents and dtype. Its default K-output route calls this same
        # CuPy routine; keep its precision policy, allocation and stream logic.
        return native(left, right)
    return current(left, right)


def bind_matrix_product(array_module, *, batched=False):
    """Bind the public 2-D CuPy entry; retain generic batch/CPU dispatch.

    Both inputs to this callable are already matrices in the selected layout.
    CuPy dot and matmul implement the same 2-D product, while dot avoids the
    generalized-ufunc broadcast/parser wrapper. No arrays/devices are stored.
    """
    dot = getattr(array_module, 'dot', None)
    method = ('dot' if not batched and getattr(array_module, '__name__', '') == 'cupy'
              and callable(dot) else 'matmul')
    public, native = None, None
    if batched and getattr(array_module, '__name__', '') == 'cupy':
        public = getattr(array_module, method, None)
        linalg = getattr(getattr(array_module, '_core', None), '_routines_linalg', None)
        candidate = getattr(linalg, 'matmul', None)
        if (type(public).__module__ == 'cupy._core._gufuncs'
                and getattr(public, '_signature', None) == '(n?,k),(k,m?)->(n?,m?)'
                and callable(candidate) and getattr(public, '_func', None) is candidate):
            native = candidate
    return partial(_matrix_product, array_module=array_module, method=method,
                   public=public, native=native)


def _bind_long_batched_product(lowering, array_module, multiply):
    """Route long-K CuPy batched products away from the strided-batched GEMM."""
    m, k, n = lowering.matrix_m, lowering.matrix_k, lowering.matrix_n
    if (getattr(array_module, '__name__', '') != 'cupy' or not lowering.batch_shape
            or k < _LONG_BATCHED_K or 1 in (m, n)):
        return multiply
    batch = _product(lowering.batch_shape)
    output_shape = lowering.batch_shape + (m, n)
    if batch <= 16 and (batch <= 4 or k >= 100_000 or m * n >= 16384):
        def per_matrix(left, right):
            left = array_module.reshape(left, (batch, m, k))
            right = array_module.reshape(right, (batch, k, n))
            result = array_module.empty((batch, m, n), dtype=left.dtype)
            for index in range(batch):
                array_module.dot(left[index], right[index], out=result[index])
            return array_module.reshape(result, output_shape)
        return per_matrix
    split = next((s for s in range(_SPLIT_K, k // _SPLIT_K + 1) if k % s == 0), None)
    if split is None or batch * -(-m // 32) * -(-n // 32) >= _SPLIT_MAX_TILES:
        return multiply

    def split_reduction(left, right):
        left = array_module.reshape(left, (batch, m, split, k // split)).transpose(0, 2, 1, 3)
        right = array_module.reshape(right, (batch, split, k // split, n))
        return array_module.reshape(array_module.matmul(left, right).sum(axis=1), output_shape)
    return split_reduction


def _execute_matmul(left, right, *, array_module, permutations, shapes, multiply):
    left_permutation, right_permutation, output_permutation = permutations
    left_shape, right_shape, grouped_shape = shapes
    left_view = left if left_permutation is None else array_module.transpose(left, left_permutation)
    right_view = right if right_permutation is None else array_module.transpose(right, right_permutation)
    left_matrix = left_view if left_view.shape == left_shape else array_module.reshape(left_view, left_shape)
    right_matrix = right_view if right_view.shape == right_shape else array_module.reshape(right_view, right_shape)
    product = multiply(left_matrix, right_matrix)
    grouped = product if product.shape == grouped_shape else array_module.reshape(product, grouped_shape)
    if output_permutation is not None:
        grouped = array_module.transpose(grouped, output_permutation)
    return grouped


def _bind_batched_inner_product(lowering, backend, fallback, shapes):
    """Reuse the bounded two-stage reduction for long double-precision dots."""
    dtype = np.dtype(lowering.dtype)
    multiply = fallback.keywords['multiply']
    if (not lowering.batch_shape or lowering.matrix_m != 1 or lowering.matrix_n != 1
            or lowering.matrix_k < 4096 or dtype not in (np.dtype('float64'), np.dtype('complex128'))
            or multiply.keywords.get('native') is None):
        return None
    cuda = getattr(backend, 'cuda', None)
    cublas = getattr(cuda, 'cublas', None)
    linalg = getattr(getattr(backend, '_core', None), '_routines_linalg', None)
    compute_type = getattr(linalg, 'get_compute_type', None)
    default_compute = getattr(linalg, 'COMPUTE_TYPE_DEFAULT', None)
    if (not callable(compute_type) or default_compute is None
            or not callable(getattr(cuda, 'Device', None))
            or not callable(getattr(cublas, 'getMathMode', None))
            or not hasattr(cublas, 'CUBLAS_DEFAULT_MATH')):
        return None
    from .reduction_kernel import bind_raw_reduction
    # Consume the original tensor strides: packing both inputs first would
    # spend the same workspace needed by the partials a second time.
    allowance = (sum(map(math.prod, shapes)) + math.prod(lowering.batch_shape)) * dtype.itemsize
    raw = bind_raw_reduction(lowering.equation, shapes, dtype, backend, allowance)
    if raw is None:
        return None
    public = multiply.keywords['public']

    def run(left, right):
        if (debug_enabled() or backend.matmul is not public
                or getattr(public, '_func', None) is not multiply.keywords['native']
                or compute_type(dtype) != default_compute
                or cublas.getMathMode(cuda.Device().cublas_handle) != cublas.CUBLAS_DEFAULT_MATH):
            return fallback(left, right)
        return raw(left, right)
    run.raw_batched_inner_product = True
    run.workspace_bytes = raw.workspace_bytes
    run.workspace_allowance = allowance
    return run


def _bind_lowered_matrix_product(lowering, array_module):
    """Reuse fixed GEMM after the required operand layout conversion.

    A tensor layout can require packing even when its resulting two matrices
    satisfy the existing C/F GEMM ABI. Keep that packing, then avoid the public
    dot wrapper. Unsupported/current precision modes retain the same wrapper.
    """
    multiply = bind_matrix_product(array_module, batched=bool(lowering.batch_shape))
    if lowering.batch_shape:
        return _bind_long_batched_product(lowering, array_module, multiply)
    shapes = lowering.execution_shapes[:2]
    pointer_product = bind_pointer_gemm(array_module, shapes, (1, 1), (None, None),
                                       lowering.dtype, (shapes[0][0], shapes[1][1]))
    if pointer_product is None:
        return multiply

    def run(left, right):
        result = pointer_product(left, right)
        return multiply(left, right) if result is None else result
    return run


def _bind_packed_matmul(lowering, backend):
    """Use the existing stride copy only for operands that need packing.

    Bindings contain geometry and bounded copy programs, never numerical owners.
    Singleton axes keep the original reshape path, where ignored strides can
    permit a view even when they do not match our exact C/F stride patterns.
    """
    logical_shapes = (lowering.left_free_shape + lowering.reduction_shape,
                      lowering.reduction_shape + lowering.right_free_shape)
    if (lowering.batch_shape or any(not s or 1 in s for s in logical_shapes)):
        return None
    permutations = (lowering.left_permutation, lowering.right_permutation)
    splits = (len(lowering.left_free_shape), len(lowering.reduction_shape))
    dtype = np.dtype(lowering.dtype)
    originals = tuple(tuple(s[p.index(i)] for i in range(len(s)))
                      for s, p in zip(logical_shapes, permutations))
    patterns = tuple(_matrix_strides(s, split, p, dtype.itemsize)
                     for s, split, p in zip(logical_shapes, splits, permutations))
    matrices = lowering.execution_shapes[:2]
    variants = {}
    for copied in ((False, True), (True, False), (True, True)):
        product = bind_pointer_gemm(backend,
            tuple(m if copy else s for copy, m, s in zip(copied, matrices, logical_shapes)),
            tuple(1 if copy else split for copy, split in zip(copied, splits)),
            tuple(None if copy else p for copy, p in zip(copied, permutations)),
            dtype, lowering.grouped_shape, lowering.execution_permutations[2])
        if product is None:
            return None
        public = partial(_execute_matmul, array_module=backend,
            permutations=tuple(None if copy else p for copy, p in
                zip(copied, lowering.execution_permutations[:2]))
                + (lowering.execution_permutations[2],),
            shapes=lowering.execution_shapes, multiply=bind_matrix_product(backend))
        variants[copied] = product, public
    # Import at binding time: the operation package also consumes lowering.
    from .operations.packing_kernel import bind_packing
    from .packed_values import pack_operand
    packers = tuple(bind_packing(backend, (shape,), (dtype,), 0, dtype,
                                input_layouts=(permutation,))
                    for shape, permutation in zip(logical_shapes, permutations))
    original_packers = tuple(getattr(pack, 'pack_original', None) for pack in packers)
    if not all(callable(pack) for pack in original_packers):
        return None

    def bind_layout(layouts):
        if any(shape != expected or actual_dtype != dtype
               for (shape, actual_dtype, _), expected in zip(layouts, originals)):
            return None  # Do not launch a fixed-size copy on a different input.
        copied = tuple(actual not in strides for (_, _, actual), strides in zip(layouts, patterns))
        choice = variants.get(copied)
        if choice is None:
            return None
        product, public = choice
        product_layouts = tuple(
            (shape, dtype, _matrix_strides(shape, 1, (0, 1), dtype.itemsize)[0])
            if copy else layout for copy, shape, layout in zip(copied, matrices, layouts))
        product = product.bind_layout(product_layouts)
        pack_recipes = tuple((dtype.str, shape, strides, permutation, matrix)
                             for (shape, _, strides), permutation, matrix in
                             zip(layouts, permutations, matrices))
        pack_bytes = tuple(math.prod(shape) * dtype.itemsize for shape in matrices)

        def allocate_and_pack(value, shape, pack):
            destination = backend.empty(shape, dtype=dtype, order='C')
            try:
                pack((value,), destination)
            except BaseException as error:
                # A failed copy may already have submitted work. Retain
                # inputs and all earlier packed owners through cleanup.
                try:
                    backend.cuda.get_current_stream().synchronize()
                except BaseException as cleanup:
                    error.add_note(f'Matrix packing cleanup also failed: {cleanup!r}')
                raise
            return destination

        def run(left, right):
            packed = []
            for copy, value, shape, pack, recipe, nbytes in zip(
                    copied, (left, right), matrices, original_packers, pack_recipes, pack_bytes):
                if copy:
                    packed.append(pack_operand(value, recipe, nbytes,
                        partial(allocate_and_pack, value, shape, pack)))
                else:
                    packed.append(value)
            result = product(*packed)
            return public(*packed) if result is None else result
        return run

    def run(left, right):
        if debug_enabled():
            return None
        bound = bind_layout(((left.shape, left.dtype, left.strides),
                             (right.shape, right.dtype, right.strides)))
        return None if bound is None else bound(left, right)
    run.bind_layout = bind_layout
    return run


def bind_matmul(lowering: MatmulLowering, array_module: Any):
    """Freeze matrix shapes and nontrivial permutations, without input storage.

    The callable resolves new operands on every invocation; it is safe to reuse
    across cfg/coordinate changes and stores no input arrays or stream state.
    """
    from .reduction import bind_precise_reduction
    # Recover original tensor shapes from the frozen execution permutation.
    logical = (lowering.batch_shape + lowering.left_free_shape + lowering.reduction_shape,
               lowering.batch_shape + lowering.reduction_shape + lowering.right_free_shape)
    permutations = (lowering.left_permutation, lowering.right_permutation)
    shapes = tuple(tuple(shape[permutation.index(i)] for i in range(len(shape)))
                   for shape, permutation in zip(logical, permutations, strict=True))
    precise = bind_precise_reduction(lowering.equation, shapes,
                                    (lowering.dtype, lowering.dtype), array_module)
    if precise is not None:
        return precise
    fallback = partial(_execute_matmul, array_module=array_module,
                   permutations=lowering.execution_permutations, shapes=lowering.execution_shapes,
                   multiply=_bind_lowered_matrix_product(lowering, array_module))
    inner_product = _bind_batched_inner_product(lowering, array_module, fallback, shapes)
    if inner_product is not None:
        return inner_product
    pointer_product = (None if lowering.batch_shape else bind_pointer_gemm(
        array_module, (lowering.left_free_shape + lowering.reduction_shape,
                       lowering.reduction_shape + lowering.right_free_shape),
        (len(lowering.left_free_shape), len(lowering.reduction_shape)),
        (lowering.left_permutation, lowering.right_permutation), lowering.dtype,
        lowering.grouped_shape, lowering.execution_permutations[2]))
    if pointer_product is None:
        return fallback
    packed_product = _bind_packed_matmul(lowering, array_module)
    routes = {}

    def run(left, right):
        if debug_enabled():
            return fallback(left, right)
        layout = ((left.shape, left.dtype, left.strides),
                  (right.shape, right.dtype, right.strides))
        route = routes.get(layout)
        if route is None:
            route = pointer_product.bind_layout(layout)
            if route is None and packed_product is not None:
                route = packed_product.bind_layout(layout)
            route = fallback if route is None else route
            # Array layouts can vary across reader/coordinate bindings. Bound
            # metadata stays small; no current input or result is retained.
            if len(routes) >= 32:
                routes.clear()
            routes[layout] = route
        result = route(left, right)
        return fallback(left, right) if result is None else result
    return run


def execute_matmul(lowering: MatmulLowering, left: Any, right: Any, array_module: Any):
    """Execute one previously lowered equation on NumPy or CuPy arrays."""
    from .reduction import bind_precise_reduction
    precise = bind_precise_reduction(lowering.equation, (left.shape, right.shape),
                                    (left.dtype, right.dtype), array_module)
    if precise is not None:
        return precise(left, right)
    return _execute_matmul(left, right, array_module=array_module,
                           permutations=lowering.execution_permutations, shapes=lowering.execution_shapes,
                           multiply=_bind_long_batched_product(lowering, array_module, bind_matrix_product(
                               array_module, batched=bool(lowering.batch_shape))))


def clear_lowering_cache() -> None:
    """Clear structural lowering recipes, primarily for deterministic tests."""

    _lower_binary_matmul_cached.cache_clear()


def lowering_cache_info():
    """Return bounded-cache statistics without exposing cache entries."""

    return _lower_binary_matmul_cached.cache_info()


__all__ = [
    "MatmulLowering",
    "bind_matmul",
    "clear_lowering_cache",
    "execute_matmul",
    "lower_binary_matmul",
    "lowering_cache_info",
]
