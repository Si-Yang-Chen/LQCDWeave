"""Execute saved Cartesian kernels using structural, array-free layout bindings."""
from dataclasses import dataclass, field
from functools import partial
from math import prod

import numpy as np
from .packing_kernel import bind_packing, _bind_packing_values
from .result_views import _DeferredResultView, bind_result_views, _array_pointer
from ..diagnostics import debug_enabled
from ..lowering import bind_matrix_product
from ..matrix_product import bind_pointer_gemm
from lqcdweave.contracts.correlator.precision import require_uniform_contraction_dtype


def _contiguous_concatenation(values, axis, shape, dtype, *, input_layouts=None):
    """Compatibility call; bound batch sides reuse the same metadata proof."""
    return _bind_contiguous_concatenation(axis, shape, input_layouts=input_layouts)(values, dtype)


def _bind_contiguous_concatenation(axis, shape, *, input_layouts=None):
    """Bind fixed geometry, proving every current owner and byte span on use.

    Only immutable geometry and one itemsize/byte-stride tuple are retained.
    Independent owners, gaps, overlap, reverse order, unsupported layouts and
    dtype promotion keep the ordinary packing path. No current array is cached.
    """
    shape = tuple(shape)
    rank, size = len(shape), prod(shape)
    element_strides = [1] * rank
    for index in range(rank - 2, -1, -1):
        element_strides[index] = element_strides[index + 1] * shape[index + 1]
    element_strides = tuple(element_strides)
    identity_layout = tuple(range(rank))
    input_layouts = (None if input_layouts is None else
                     tuple(tuple(layout) for layout in input_layouts))
    permutation = (None if input_layouts is None else
                   tuple(index for index in input_layouts[0] if index is not None))
    identity_permutation = (permutation == tuple(range(len(permutation)))
                            if permutation is not None else False)
    stride_cache = (None, None)

    def byte_strides(itemsize):
        nonlocal stride_cache
        cached_itemsize, strides = stride_cache
        if cached_itemsize != itemsize:
            strides = tuple(step * itemsize for step in element_strides)
            stride_cache = (itemsize, strides)
        return strides

    def borrow(values, dtype):
        first = values[0]
        first_dtype = first.dtype
        if dtype is not None and first_dtype != dtype:
            return None
        if len(values) == 1 and type(first) is not _DeferredResultView:
            if input_layouts is None:
                return first if first.shape == shape and first.flags.c_contiguous else None
            layout = input_layouts[0]
            if len(layout) != rank:
                return None
            actual_shape = first.shape
            for index, source_axis in enumerate(layout):
                extent = 1 if source_axis is None else actual_shape[source_axis]
                if extent != shape[index]:
                    return None
            actual_strides = first.strides
            strides = byte_strides(first_dtype.itemsize)
            for index, source_axis in enumerate(layout):
                step = 0 if source_axis is None else actual_strides[source_axis]
                if shape[index] > 1 and step != strides[index]:
                    return None
            if not identity_permutation or first.ndim != len(permutation):
                first = first.transpose(permutation)
            return first if first.shape == shape else first.reshape(shape)
        owner = first.owner.value if type(first) is _DeferredResultView else first
        owner_type = type(owner)
        while getattr(owner, 'base', None) is not None:
            base = owner.base
            if not isinstance(base, owner_type):
                break
            owner = base
        if not owner.flags.c_contiguous or owner.dtype != first_dtype:
            return None
        # Keep the inexpensive independent-owner rejection before shape,
        # stride and pointer work. These identities belong to this call only.
        for value in values:
            root = value.owner.value if type(value) is _DeferredResultView else value
            root_type = type(root)
            while getattr(root, 'base', None) is not None:
                base = root.base
                if not isinstance(base, root_type):
                    break
                root = base
            if root is not owner or value.dtype != first_dtype:
                return None
        itemsize = first_dtype.itemsize
        strides = byte_strides(itemsize)
        start = _array_pointer(first)
        position, extent = start, 0
        for value_index, value in enumerate(values):
            actual_shape, actual_strides = value.shape, value.strides
            if input_layouts is None:
                if len(actual_shape) != rank:
                    return None
                layout = identity_layout
            else:
                layout = input_layouts[value_index]
                if len(layout) != rank:
                    return None
            member_extent = 0
            for index, source_axis in enumerate(layout):
                length = 1 if source_axis is None else actual_shape[source_axis]
                step = 0 if source_axis is None else actual_strides[source_axis]
                if (index != axis and length != shape[index]
                        or length > 1 and step != strides[index]):
                    return None
                if index == axis:
                    member_extent = length
            if _array_pointer(value) != position:
                return None
            extent += member_extent
            position += member_extent * strides[axis]
        offset = start - _array_pointer(owner)
        if (extent != shape[axis] or offset < 0 or offset % itemsize
                or offset + size * itemsize > owner.nbytes):
            return None
        offset //= itemsize
        return owner.reshape(-1)[offset:offset + size].reshape(shape)

    return borrow


@dataclass(frozen=True)
class _BatchSide:
    operands: tuple
    shape: tuple
    axis: int
    prepared: bool
    packer: object = None
    original_packer: object = None
    stacked: bool = False
    prepend: bool = False
    input_layouts: tuple | None = None
    allow_fortran: bool = False
    _values: object = field(init=False, compare=False, repr=False)
    _borrow: object = field(init=False, compare=False, repr=False)
    _borrow_original: object = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_values', _bind_packing_values(self.operands))
        borrow = _bind_contiguous_concatenation(self.axis, self.shape)
        object.__setattr__(self, '_borrow', borrow)
        object.__setattr__(self, '_borrow_original',
            borrow if self.input_layouts is None else _bind_contiguous_concatenation(
                self.axis, self.shape, input_layouts=self.input_layouts))

    def _views(self, arrays):
        ordered = tuple(arrays[key] if permutation is None else arrays[key].transpose(permutation)
                        for key, permutation in self.operands)
        if self.stacked:
            ordered = tuple(value[None, ...] if self.prepend else value[..., None] for value in ordered)
        return ordered

    def pack(self, arrays, backend, dtype):
        if not self.prepared and not self.stacked and len(self.operands)==1 and len(self.shape)==2:
            source=self._values(arrays)[0]
            permutation=self.operands[0][1]
            value=(None if type(source) is _DeferredResultView else
                   source if permutation is None else source.transpose(permutation))
            # GEMM accepts C and F matrices; keep the current input owner alive
            # instead of allocating another C-contiguous packing snapshot.
            if (value is not None and value.shape==self.shape and (dtype is None or value.dtype==dtype)
                    and (value.flags.c_contiguous or self.allow_fortran and value.flags.f_contiguous)):
                return value
        sources = None
        if (not self.prepared and not self.stacked and self.original_packer is not None
                and len(self.operands) == 1 and self.operands[0][1] is None
                and not debug_enabled()):
            # Identity axes need only the array's existing C-layout proof.
            # Keep shape and promotion boundaries before borrowing this owner.
            sources = self._values(arrays)
            value = sources[0]
            if (type(value) is not _DeferredResultView and value.shape == self.shape
                    and value.dtype == dtype and value.flags.c_contiguous):
                return value
        if (not self.prepared and not debug_enabled() and self.original_packer is not None):
            if sources is None:
                sources = self._values(arrays)
            ready = self._borrow_original(sources, dtype)
            if ready is not None:
                return ready
            destination = backend.empty(self.shape, dtype=dtype, order='C')
            if destination.flags.c_contiguous:
                return self.original_packer(sources, destination)
            return self.packer(self._views(arrays), destination)

        ordered = self._views(arrays)
        if self.prepared:
            value = ordered[0]
            if not value.flags.c_contiguous or value.dtype != dtype:
                raise ValueError('batch producer did not emit its saved matrix layout: '
                    f'tensor={self.operands[0][0]!r}, expected_dtype={dtype}, '
                    f'actual_dtype={value.dtype}, shape={value.shape}, '
                    f'strides={value.strides}, c_contiguous={value.flags.c_contiguous}')
            return value
        # Reuse only a view of inputs already owned/accounted by the live table.
        # Saved packing_bytes remains a conservative upper bound, unchanged.
        ready = self._borrow(ordered, dtype)
        if ready is not None:
            return ready
        if dtype is not None:
            if self.packer is not None:
                return self.packer(ordered, backend.empty(self.shape, dtype=dtype, order='C'))
            if len(ordered) == 1:
                return backend.array(ordered[0], dtype=dtype, order='C', copy=True)
            destination = backend.empty(self.shape, dtype=dtype, order='C')
            backend.concatenate(ordered, axis=self.axis, out=destination)
            return destination
        return backend.concatenate(ordered, axis=self.axis) if len(ordered) > 1 else ordered[0]


@dataclass(frozen=True)
class _BatchLayout:
    sides: tuple
    dtype: object
    matrix_shapes: tuple
    output_shape: tuple
    reference_axes: tuple
    result_slices: tuple
    simple_matrix: bool
    multiply: object
    deferred_layouts: tuple | None = None
    pointer_product: object = None
    precise_product: object = None

    def contract(self, values, backend):
        if self.precise_product is not None or self.dtype is not None or (self.simple_matrix and
                (backend is np or all(v.flags.c_contiguous for v in values))):
            if self.pointer_product is not None and self.precise_product is None:
                product = self.pointer_product(*values)
                if product is not None:
                    return product
            left, right = (v if v.shape == shape else v.reshape(shape)
                           for v, shape in zip(values, self.matrix_shapes))
            product = (self.precise_product or self.multiply)(left, right)
            return product if product.shape == self.output_shape else product.reshape(self.output_shape)
        left_axes, right_axes, output_axes = self.reference_axes
        return backend.einsum(values[0], left_axes, values[1], right_axes, output_axes,
                              order='C', optimize=False)

    def run(self, arrays, *, backend, timed=None, device=False):
        if timed is None:
            values = tuple(side.pack(arrays, backend, self.dtype) for side in self.sides)
            output = self.contract(values, backend)
        else:
            values = []
            for side in self.sides:
                pack = partial(side.pack, arrays, backend, self.dtype)
                if not side.prepared and (self.dtype is not None or len(side.operands) > 1):
                    values.append(timed('dense_batch_pack', pack, device=device))
                else:
                    values.append(pack())
            output = timed('contract_kernel', partial(self.contract, values, backend), device=device)
        if (self.deferred_layouts is not None and timed is None and not debug_enabled()
                and callable(getattr(arrays, '_publish_result_views', None))):
            arrays._publish_result_views(output, self.deferred_layouts)
        else:
            arrays.update(self.views(output))
        return output

    def views(self, output):
        # Construct every view before publication, preserving failure atomicity.
        results = {}
        for key, bounds, squeeze in self.result_slices:
            value = output[bounds]
            results[key] = value.squeeze(axis=squeeze) if squeeze else value
        return results


def bind_dense_batch(batch, catalog, backend=np):
    """Bind one saved batch's metadata once, without storing any numerical input.

    New cfg/coordinate arrays are resolved by key at execution time. The binding
    is scoped to its saved batch and backend, not a shape-only numerical cache.
    """
    from lqcdweave.contracts.correlator.gemm_group import DenseGemmGroup
    if isinstance(batch, DenseGemmGroup):
        from .gemm_group import bind_dense_gemm_group
        return bind_dense_gemm_group(batch, catalog, backend)
    layout = bind_batch_layout(batch, catalog, backend)
    return partial(layout.run, backend=backend)


def bind_batch_layout(batch, catalog, backend=np, *, allow_fortran=True,
                      coalesce_reductions=True):
    """Bind an original Cartesian layout for ordinary or grouped execution."""
    dtype = require_uniform_contraction_dtype(
        {operand.tensor:catalog[operand.tensor] for operand in (*batch.left,*batch.right)})
    if batch.packing_dtype is not None and np.dtype(batch.packing_dtype) != dtype:
        raise ValueError('mixed-precision batch packing is unsupported; normalize inputs before planning')
    if backend is not np and getattr(backend, '__name__', '') != 'cupy':
        raise NotImplementedError('Cartesian dense kernel batches require NumPy or CuPy')
    if getattr(backend, '__name__', '') == 'cupy' and batch.packing_dtype is None:
        raise ValueError('CuPy batches require saved result-dtype packing')
    from ..batch_reduction import bind_precise_batch
    precise = None
    if (coalesce_reductions and all(batch.stacked_sides) and not debug_enabled()
            and getattr(backend, '__name__', '') == 'cupy'
            and dtype in (np.dtype('float32'), np.dtype('complex64'))):
        k = prod(catalog[batch.left[0].tensor].shape)
        precise = bind_precise_batch(batch, catalog,
            ((len(batch.left), k), (len(batch.right), k)), backend, right_rows=True)
    right_rows = precise is not None
    sides = []
    for side, axis, prepared, stacked, prepend in (
            (batch.left, batch.left_axis, batch.prepared_sides[0], batch.stacked_sides[0], True),
            (batch.right, 0 if right_rows else batch.right_axis,
             batch.prepared_sides[1], batch.stacked_sides[1], right_rows)):
        input_layouts = tuple(
            ((None, *o.permutation) if stacked and prepend else
             (*o.permutation, None) if stacked else o.permutation)
            for o in side)
        shapes = [tuple(catalog[o.tensor].shape[i] for i in o.permutation) for o in side]
        if stacked:
            shapes = [(1,) + shape if prepend else shape + (1,) for shape in shapes]
        merged = list(shapes[0])
        merged[axis] = sum(shape[axis] for shape in shapes)
        operands = tuple((o.tensor, None if o.permutation == tuple(range(len(o.permutation)))
                          else o.permutation) for o in side)
        packer = (bind_packing(backend, shapes, tuple(catalog[o.tensor].dtype for o in side),
                               axis, batch.packing_dtype, input_layouts=input_layouts)
                  if not prepared and batch.packing_dtype is not None else None)
        original_packer = getattr(packer, 'pack_original', None) if packer is not None else None
        sides.append(_BatchSide(operands, tuple(merged), axis, prepared, packer, original_packer,
                                stacked, prepend, input_layouts, allow_fortran))
    lf, kr, rf = batch.left_free_rank, batch.reduction_rank, batch.right_free_rank
    left_shape, right_shape = (side.shape for side in sides)
    right_free = right_shape[:1] if right_rows else right_shape[kr:]
    m, k, n = prod(left_shape[:lf]), prod(left_shape[lf:]), prod(right_free)
    matrix_shapes = ((m, k), (n, k) if right_rows else (k, n))
    axes = (tuple(range(lf + kr)), ((lf + kr,) + tuple(range(lf, lf + kr)) if right_rows
                                  else tuple(range(lf, lf + kr + rf))),
            tuple(range(lf)) + tuple(range(lf + kr, lf + kr + rf)))
    result_slices = []
    for result in batch.results:
        squeeze = result.squeeze_axes
        if squeeze and len(result.bounds) - len(squeeze) > 0:
            squeeze_axes = frozenset(squeeze)
            bounds = tuple(start if axis in squeeze_axes else slice(start, stop)
                           for axis, (start, stop) in enumerate(result.bounds))
            squeeze = ()
        else:
            bounds = tuple(slice(start, stop) for start, stop in result.bounds)
        result_slices.append((result.tensor, bounds, squeeze))
    dtype = None if batch.packing_dtype is None else np.dtype(batch.packing_dtype)
    output_shape = left_shape[:lf] + right_free
    result_slices = tuple(result_slices)
    deferred_layouts = (bind_result_views(output_shape, dtype, result_slices)
                        if getattr(backend, '__name__', '') == 'cupy' else None)
    if precise is None:
        precise = bind_precise_batch(batch, catalog, matrix_shapes, backend)
    layout = _BatchLayout(tuple(sides), dtype, matrix_shapes, output_shape, axes,
                          result_slices, lf == rf == 1, bind_matrix_product(backend),
                          deferred_layouts, (None if dtype is None or precise is not None else bind_pointer_gemm(
                              backend, (left_shape, right_shape), (lf, kr), (None, None),
                              dtype, output_shape)), precise)
    return layout


def contract_dense_batch(batch, arrays, *, backend=np, timed=None, device=False, bound_kernel=None):
    if bound_kernel is None:
        bound_kernel = bind_dense_batch(batch, arrays, backend)
    return bound_kernel(arrays, timed=timed, device=device)
