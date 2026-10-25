"""Dense contraction operation using the caller's preselected backend."""

from functools import lru_cache, partial
from operator import itemgetter
import numpy as np
from lqcdweave.contracts.correlator.api import MatrixKernelSpec, FactorizedWideReduction, MesonWholeKernel, parse_axis_pattern, AxisPattern

from ..indices import einsum_arguments
from ..lowering import bind_matmul, execute_matmul, lower_binary_matmul
from .transfer import copy_to_backend, owns_numpy_result
from ..matrix_kernels import contract_meson_matrix_pc, contract_meson_loop_pc
from ..diagnostics import debug_enabled
from .reference_matrix import bind_reference_matrix
from ..reduction import bind_precise_reduction
from .result_views import _DeferredResultView


@lru_cache(maxsize=256)
def _matmul_equation(equation):
    """Cache only equation eligibility, without arrays, shapes or devices.

    This rejects reference-only operations before dtype/shape normalization.
    Actual dimensions and dtypes are still validated by the existing lowering.
    """
    equation = parse_axis_pattern(equation)
    if len(equation.inputs) != 2:
        return False
    left_text, right_text = equation.inputs
    output_text = equation.output
    left, right, output = set(left_text), set(right_text), set(output_text)
    if any(len(text) != len(unique) for text, unique in
           ((left_text, left), (right_text, right), (output_text, output))):
        return False
    return bool((left & right) - output) and output <= left | right and (left ^ right) <= output


def _contract_value(equation, operands, backend, device, *, _device_scope=True, force_wide=False):
    """Return the value and whether the kernel guarantees independent storage."""
    equation = parse_axis_pattern(equation)
    if len(operands) == 1 and equation.inputs == (equation.output,):
        return operands[0], False
    precise = bind_precise_reduction(equation, tuple(value.shape for value in operands),
                                    tuple(value.dtype for value in operands), backend, force_matrix=force_wide)
    if precise is not None:
        return precise(*operands), True
    if force_wide:
        raise MemoryError('saved wide reduction cannot fit its bounded workspace')
    lowering = None
    if (
        device and len(operands) == 2
        and _matmul_equation(equation)
        and all(callable(getattr(backend, name, None))
                for name in ("transpose", "reshape", "matmul"))
    ):
        # This is an axis recipe for the already chosen binary action, not a
        # contraction-path search. Unsupported equations retain the reference
        # implementation. The structural cache holds no arrays.
        lowering = lower_binary_matmul(
            equation,
            tuple(tuple(value.shape) for value in operands),
            tuple(value.dtype for value in operands),
        )
    if lowering is not None and lowering.reduction_labels:
        # execute_matmul's output views retain only the newly allocated matrix
        # product. They never borrow an input or a reusable backend buffer.
        return execute_matmul(lowering, *operands, backend), True
    # V100 measurements favor the reference path for outer/elementwise
    # products. General einsum may also return an input view; keep its snapshot.
    return backend.einsum(*einsum_arguments(equation, operands), optimize=False), False


def _matrix_value(action, operands, backend):
    spec = action.core
    if debug_enabled() and (action.equation.normalized() != spec.equation.normalized()
            or tuple(value.shape for value in operands) != spec.action_shapes
            or tuple(value.dtype.name for value in operands) != spec.action_dtypes):
        raise ValueError('matrix Action runtime layouts differ from its fixed template')
    count = len(spec.shapes)
    if debug_enabled() and spec.partitions is not None:
        expected = np.zeros(spec.projection_shape,dtype=spec.dtype)
        for pc,a,b,c,d,weight in spec.projection_blocks:
            expected[pc,a:b,c:d] = weight
        for value in operands[count:]:
            actual = value if backend is np else backend.asnumpy(value)
            if not np.array_equal(actual,expected):
                raise ValueError('matrix Action PC coefficients differ from its declared projection')
        del expected
    if spec.kind == 'connected':
        value = contract_meson_matrix_pc(*operands[:count],partitions=spec.partitions,row_block=spec.row_block,
                                        backend=backend,mode_block=spec.mode_block)
    else:
        value = contract_meson_loop_pc(*operands[:count],partitions=spec.partitions,
                                       backend=backend,mode_block=spec.mode_block)
    return value, True


def bind_dense_action(action, catalog, backend, output_layout=None):
    """Bind the saved equation/layout once; the callable retains no arrays."""
    if type(action.core) is MesonWholeKernel:
        from ..meson_core import bind_meson_whole_action
        if output_layout is not None:
            raise ValueError('whole meson output cannot borrow a binary producer layout')
        return bind_meson_whole_action(action, catalog, backend)
    if type(action.core) is FactorizedWideReduction:
        from ..factorized_reduction import bind_factorized_reduction
        if output_layout is not None:
            raise ValueError('thin wide outputs cannot borrow a batched producer layout')
        return bind_factorized_reduction(action, catalog, backend)
    if isinstance(action.core, MatrixKernelSpec):
        return partial(_matrix_value, action, backend=backend)
    if action.core is not None:
        raise NotImplementedError('unsupported dense Action core')
    if len({catalog[key].dtype for key in action.operands}) > 1:
        raise ValueError('mixed-precision contraction is unsupported; normalize inputs before planning')
    equation = action.equation
    if len(action.operands) == 1 and equation.inputs == (equation.output,):
        return lambda operands: (operands[0], False)
    order = tuple(range(len(action.operands)))
    inverse = None
    if output_layout is not None:
        desired = tuple(equation.output[i] for i in output_layout)
        axes = equation.inputs
        common = set(axes[0]) & set(axes[1])
        left = tuple(i for i in desired if i in set(axes[0]) - common)
        right = tuple(i for i in desired if i in set(axes[1]) - common)
        if desired != left + right:
            axes, order = axes[::-1], order[::-1]
        equation = AxisPattern(axes, desired)
        inverse = tuple(output_layout.index(i) for i in range(len(output_layout)))
    specs = tuple(catalog[action.operands[i]] for i in order)
    precise = bind_precise_reduction(equation, tuple(s.shape for s in specs),
                                    tuple(s.dtype for s in specs), backend, force_matrix=action.wide_accumulation)
    if action.wide_accumulation and precise is None:
        raise MemoryError('saved wide reduction cannot fit its bounded workspace')
    lowering = None
    device = getattr(backend, '__name__', '') == 'cupy' or output_layout is not None
    if precise is None and (device or backend is np) and _matmul_equation(equation):
        specs = tuple(catalog[action.operands[i]] for i in order)
        lowering = lower_binary_matmul(equation, tuple(s.shape for s in specs), tuple(s.dtype for s in specs))
    matrix = None if lowering is None else bind_matmul(lowering, backend)
    reference_matrix = (bind_reference_matrix(equation,
        tuple(catalog[action.operands[i]] for i in order), backend)
        if precise is None and matrix is None else None)
    # The fallback's integer argument layout is structural too. Normalizing
    # on every execution rebuilt AxisPattern and all axis lists for each pair.
    local = equation.normalized()
    axes, output_axes = local.inputs, local.output
    natural_order = order == tuple(range(len(order)))

    def run(operands):
        ordered = operands if natural_order else tuple(operands[i] for i in order)
        if precise is not None:
            value, independent = precise(*ordered), True
        elif lowering is not None and lowering.reduction_labels:
            value, independent = matrix(*ordered), True
        elif reference_matrix is not None:
            value, independent = reference_matrix(*ordered), True
        else:
            arguments = tuple(item for operand, labels in zip(ordered, axes, strict=True)
                              for item in (operand, labels)) + (output_axes,)
            value = backend.einsum(*arguments, optimize=False)
            # Native NumPy reports ownership directly. Keep borrowed unary
            # views on the snapshot path, but do not copy an owned result again.
            independent = owns_numpy_result(value, backend)
        if inverse is not None:
            if debug_enabled() and (not independent or not value.flags.c_contiguous):
                raise ValueError('selected batch producer did not produce its saved layout')
            value = value.transpose(inverse)
        return value, independent
    run.reference_matrix_bound = reference_matrix is not None
    return run


def bind_dense_call(action, kernel, backend, *, shared_residency=False):
    """Freeze the normal-path store; resolve current arrays at each call."""
    keys, result = action.operands, action.tensor
    if len(keys) == 1:
        key, = keys
        def operands(arrays): return (arrays[key],)
    else:
        operands = itemgetter(*keys)

    def store(context):
        arrays = context[0]
        value, independent = kernel(operands(arrays))
        if not independent:
            # Same owned snapshot as contract_dense, without a timing adapter.
            value = backend.array(value, copy=True)
        arrays[result] = value

    if not shared_residency or len(keys) != 2:
        return store

    left_key, right_key = keys

    def store_nodes(context):
        if len(context) != 4:
            # Ordinary mappings and extensions retain their mapping protocol.
            return store(context)
        arrays, binding = context[2], context[3]
        left_id, right_id = binding[left_key], binding[right_key]
        left = arrays[left_id]
        if type(left) is _DeferredResultView:
            left = arrays[left_id] = left.array()
        right = arrays[right_id]
        if type(right) is _DeferredResultView:
            right = arrays[right_id] = right.array()
        value, independent = kernel((left, right))
        if not independent:
            value = backend.array(value, copy=True)
        arrays[binding[result]] = value

    return store_nodes


def contract_dense(action, arrays, backend, *, timed, device=False, copy_result=True, output_layout=None,
                   bound_kernel=None):
    """Execute one dense action and store its owned result in ``arrays``."""
    if action.reused:
        timed("reuse", lambda: None)
        return
    operands = tuple(arrays[key] for key in action.operands)
    if bound_kernel is not None:
        operation = partial(bound_kernel, operands)
    elif output_layout is not None:
        # The logical result is a transpose view of the newly generated matrix.
        # Its selected consumer obtains a contiguous view without another copy.
        desired = tuple(action.equation.output[i] for i in output_layout)
        axes = action.equation.inputs
        common = set(axes[0]) & set(axes[1])
        left = tuple(i for i in desired if i in set(axes[0]) - common)
        right = tuple(i for i in desired if i in set(axes[1]) - common)
        if desired != left + right:
            axes, operands = axes[::-1], operands[::-1]
        equation = AxisPattern(axes,desired)
        operation = partial(_contract_value,equation,operands,backend,True,
                            force_wide=action.wide_accumulation)
    elif isinstance(action.core,MatrixKernelSpec):
        operation = partial(_matrix_value,action,operands,backend)
    elif type(action.core) is MesonWholeKernel:
        from lqcdweave.contracts.correlator.api import TensorSpec
        dimensions = {a: n for axes, value in zip(action.core.private_equation.inputs, operands)
                      for a, n in zip(axes, value.shape)}
        dtype = operands[0].dtype
        if action.core.kind == 'loop' and action.core.representation == 'unit-spin' and dtype == np.dtype('complex64'):
            dtype = np.dtype('complex128')
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in zip(action.operands, operands)}
        catalog[action.tensor] = TensorSpec(tuple(dimensions[a] for a in action.core.private_equation.output), dtype)
        operation = partial(bind_dense_action(action, catalog, backend), operands)
    elif action.core is not None:
        raise NotImplementedError('unsupported dense Action core')
    else:
        operation = partial(_contract_value, action.equation, operands, backend, device,
                            force_wide=action.wide_accumulation)
    value, independent = timed(
        "contract_kernel",
        operation,
        device=device,
    )
    if output_layout is not None and bound_kernel is None:
        if not independent or not value.flags.c_contiguous:
            raise ValueError('selected batch producer did not lower to a contiguous matrix')
        inverse = tuple(output_layout.index(i) for i in range(len(output_layout)))
        value = value.transpose(inverse)
    if copy_result:
        value = (
            timed("result_snapshot", lambda: value, nbytes=0, device=device)
            if independent else
            copy_to_backend(value, backend, phase="result_snapshot", timed=timed, device=device)
        )
    arrays[action.tensor] = value


__all__ = ["contract_dense"]
