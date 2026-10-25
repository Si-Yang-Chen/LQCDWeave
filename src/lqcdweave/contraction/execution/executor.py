"""Synchronous NumPy/CuPy execution for a compiled contraction plan."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
import opt_einsum

from lqcdweave.contracts.correlator.api import (
    ContractionPlan,
    ExecutionResult,
    ExecutionStats,
    InputRef,
    NodeRef,
    Ref,
    AxisPattern,
)
from .lowering import execute_matmul, lower_binary_matmul
from .indices import einsum_arguments, einsum_equation
from .binding_cache import bind_cached
from .operations.transfer import owns_numpy_result


def _load_cupy() -> Any:
    """Import CuPy only when a caller requests or supplies the GPU backend."""

    try:
        return importlib.import_module("cupy")
    except Exception as exc:  # pragma: no cover - depends on optional install/runtime
        raise RuntimeError(
            "the CuPy backend was requested, but CuPy is not installed"
        ) from exc


_CUPY_EXPRESSION_CACHE_SIZE = 256


@lru_cache(maxsize=_CUPY_EXPRESSION_CACHE_SIZE)
def _cupy_contract_expression(
    equation: AxisPattern,
    operand_shapes: tuple[tuple[int, ...], ...],
    operand_dtypes: tuple[str, ...],
) -> Any:
    """Build one structural CuPy contraction expression.

    The cache key contains only immutable contraction structure.  In
    particular, it deliberately excludes bound arrays, CUDA devices, and
    streams: :func:`opt_einsum.contract_expression` receives shapes only and
    the returned expression is called with fresh arrays on each execution.
    ``operand_dtypes`` is part of the key even though the current planner does
    not choose a path from dtype; this prevents a future dtype-sensitive
    planner from silently reusing a mismatched structural entry.
    """

    del operand_dtypes
    return opt_einsum.contract_expression(
        einsum_equation(equation),
        *operand_shapes,
        optimize="auto",
        use_blas=True,
    )


@dataclass(frozen=True, slots=True)
class PreparedKernels:
    """Process-local structural kernels; never archived or bound to arrays."""

    plan: ContractionPlan
    operations: tuple[tuple[Any, Any], ...]


def prepare_plan_kernels(plan: ContractionPlan) -> PreparedKernels:
    """Retain all CuPy node expressions for a persisted worker's lifetime.

    This avoids repeated path construction if the dynamic global LRU is
    evicted by other workloads. Both lowering and expression construction
    use only shapes/dtypes; no device allocation or numerical binding occurs.
    """
    prepared = []
    for operation in plan.operations:
        specs = []
        for ref in operation.inputs:
            if isinstance(ref, InputRef):
                spec = plan.specs[ref.key]
                specs.append((spec.shape, spec.dtype))
            else:
                node = plan.operations[ref.index]
                specs.append((node.output_shape, node.output_dtype))
        shapes = tuple(shape for shape, _ in specs)
        dtypes = tuple(np.dtype(dtype).str for _, dtype in specs)
        lowering = (
            lower_binary_matmul(operation.equation, shapes, dtypes)
            if len(specs) == 2 else None
        )
        expression = (
            None if lowering is not None else
            _cupy_contract_expression(operation.equation, shapes, dtypes)
        )
        prepared.append((lowering, expression))
    return PreparedKernels(plan, tuple(prepared))


def _validate_binding_keys(plan: ContractionPlan, bindings: Mapping[object, object]) -> None:
    """Validate the exact input key set shared by CPU and GPU execution."""

    missing = [key for key in plan.specs if key not in bindings]
    extra = [key for key in bindings if key not in plan.specs]
    if missing:
        raise KeyError(f"missing bindings for input key(s): {missing!r}")
    if extra:
        raise KeyError(f"bindings contain unknown input key(s): {extra!r}")


def _validate_numpy_bindings(
    plan: ContractionPlan, bindings: Mapping[object, object]
) -> dict[InputRef, np.ndarray]:
    """Validate exact NumPy input shapes and dtypes without copying bindings."""

    validated: dict[InputRef, np.ndarray] = {}
    for key, spec in plan.specs.items():
        value = bindings[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"binding {key!r} must be a NumPy ndarray; only the synchronous NumPy CPU "
                "backend is supported"
            )
        if tuple(value.shape) != spec.shape:
            raise ValueError(
                f"binding {key!r} has shape {tuple(value.shape)!r}, expected {spec.shape!r}"
            )
        if value.dtype != spec.dtype:
            raise TypeError(
                f"binding {key!r} has dtype {value.dtype!r}, expected {spec.dtype!r}"
            )
        validated[InputRef(key)] = value
    return validated


def _validate_cupy_bindings(
    plan: ContractionPlan, bindings: Mapping[object, object], cupy: Any
) -> tuple[dict[InputRef, Any], int | None]:
    """Validate CuPy arrays and return their common device id."""

    validated: dict[InputRef, Any] = {}
    device_ids: set[int] = set()
    for key, spec in plan.specs.items():
        value = bindings[key]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(
                f"binding {key!r} is not a CuPy ndarray; mixed NumPy/CuPy bindings "
                "are unsupported"
            )
        if tuple(value.shape) != spec.shape:
            raise ValueError(
                f"binding {key!r} has shape {tuple(value.shape)!r}, expected {spec.shape!r}"
            )
        if value.dtype != spec.dtype:
            raise TypeError(
                f"binding {key!r} has dtype {value.dtype!r}, expected {spec.dtype!r}"
            )
        try:
            device_ids.add(int(value.device.id))
        except Exception as exc:  # pragma: no cover - device/runtime dependent
            raise RuntimeError(f"could not inspect the CUDA device for binding {key!r}") from exc
        validated[InputRef(key)] = value
    if len(device_ids) > 1:
        raise ValueError(
            f"CuPy bindings span multiple CUDA devices {sorted(device_ids)!r}; "
            "one execute call must use one device"
        )
    return validated, (next(iter(device_ids)) if device_ids else None)


def _requested_backend_name(backend: Any) -> str:
    """Normalize an explicit backend request to ``numpy`` or ``cupy``."""

    if isinstance(backend, str):
        name = backend.lower()
    else:
        module_name = getattr(backend, "__name__", "")
        name = module_name.rsplit(".", 1)[-1].lower() if module_name else ""
    if name in {"numpy", "np"}:
        return "numpy"
    if name in {"cupy", "cp"}:
        return "cupy"
    raise ValueError(
        f"unsupported contraction backend {backend!r}; expected None, 'numpy', or 'cupy'"
    )


def _resolve_backend(
    plan: ContractionPlan,
    bindings: Mapping[object, object],
    backend: Any,
) -> tuple[str, Any, dict[InputRef, Any], int | None]:
    """Infer or validate the backend without converting any bound arrays."""

    _validate_binding_keys(plan, bindings)
    requested = None if backend is None else _requested_backend_name(backend)
    values = [bindings[key] for key in plan.specs]

    cupy = None
    if requested is None:
        if all(isinstance(value, np.ndarray) for value in values):
            requested = "numpy"
        else:
            try:
                cupy = _load_cupy()
            except RuntimeError as exc:
                raise TypeError(
                    "bindings must contain NumPy arrays, or CuPy must be installed "
                    "for CuPy array bindings"
                ) from exc
            if all(isinstance(value, cupy.ndarray) for value in values):
                requested = "cupy"
            else:
                raise TypeError(
                    "bindings must contain only NumPy arrays or only CuPy arrays; "
                    "mixed or unsupported array backends are not accepted"
                )
    elif requested == "cupy":
        cupy = _load_cupy()

    if requested == "numpy":
        return requested, np, _validate_numpy_bindings(plan, bindings), None
    cupy_bindings, device_id = _validate_cupy_bindings(plan, bindings, cupy)
    return requested, cupy, cupy_bindings, device_id


def _decrement_use(
    ref: Ref,
    remaining_uses: dict[NodeRef, int],
    values: dict[NodeRef, Any],
) -> int:
    if not isinstance(ref, NodeRef):
        return 0
    count = remaining_uses[ref] - 1
    if count < 0:  # pragma: no cover - indicates a malformed compiler plan
        raise RuntimeError(f"negative logical use count for operation {ref.index}")
    remaining_uses[ref] = count
    if count == 0:
        released = values.pop(ref, None)
        return 0 if released is None else int(released.nbytes)
    return 0


def _bind_use_counts(plan):
    counts = {NodeRef(index): 0 for index in range(len(plan.operations))}
    for operation in plan.operations:
        for ref in operation.inputs:
            if isinstance(ref, NodeRef):
                counts[ref] += 1
    for term in plan.terms:
        if isinstance(term.output, NodeRef):
            counts[term.output] += 1
    return tuple(counts.items())


def _native_plan_binding(plan):
    from lqcdweave.contracts.correlator.ir import Operation, CompiledTerm
    return (type(plan) is ContractionPlan
        and all(type(operation) is Operation for operation in plan.operations)
        and all(type(term) is CompiledTerm for term in plan.terms))


def _execute_bound_plan(
    plan: ContractionPlan,
    input_values: Mapping[InputRef, Any],
    *,
    backend_name: str,
    array_module: Any,
    stream: Any = None,
    kernels: PreparedKernels | None = None,
) -> ExecutionResult:
    """Execute with a validated array module and no host-side conversions."""

    input_bytes = int(sum(array.nbytes for array in input_values.values()))

    # Count every logical dependency reference, including repeated use of the
    # same node in one operation and repeated output terms.  Input memory is
    # borrowed and therefore never enters this release table.
    remaining_uses = dict(bind_cached('ir-lifetimes', plan, array_module,
        lambda: _bind_use_counts(plan), lambda: _native_plan_binding(plan)))
    if backend_name == 'cupy' and kernels is None:
        kernels = bind_cached('ir-kernels', plan, array_module,
            lambda: prepare_plan_kernels(plan), lambda: _native_plan_binding(plan))

    node_values: dict[NodeRef, Any] = {}
    output_values: list[Any] = []
    node_live_bytes = 0
    output_live_bytes = 0
    peak_live_bytes = 0
    peak_intermediate_bytes = 0

    def record_peak() -> None:
        nonlocal peak_live_bytes, peak_intermediate_bytes
        live_bytes = node_live_bytes + output_live_bytes
        peak_intermediate_bytes = max(peak_intermediate_bytes, node_live_bytes)
        peak_live_bytes = max(peak_live_bytes, live_bytes)

    def resolve(ref: Ref) -> Any:
        if isinstance(ref, InputRef):
            return input_values[ref]
        try:
            return node_values[ref]
        except KeyError as exc:  # pragma: no cover - indicates a malformed plan
            raise RuntimeError(f"operation {ref.index} was released before its last use") from exc

    for operation in plan.operations:
        operands = tuple(resolve(ref) for ref in operation.inputs)
        lowering = None
        try:
            if backend_name == "numpy":
                # The public NumPy path remains the reference implementation.
                # Keep the reference einsum arithmetic. Its ownership follows
                # the same policy as scene kernels: views require snapshots.
                contracted = np.einsum(*einsum_arguments(operation.equation, operands), optimize=False)
                result = contracted if owns_numpy_result(contracted, np) else np.array(contracted, copy=True)
            else:
                operand_shapes = tuple(
                    tuple(int(size) for size in operand.shape) for operand in operands
                )
                operand_dtypes = tuple(np.dtype(operand.dtype).str for operand in operands)
                if kernels is not None:
                    lowering, expression = kernels.operations[operation.index]
                elif len(operands) == 2:
                    lowering = lower_binary_matmul(
                        operation.equation, operand_shapes, operand_dtypes
                    )
                if lowering is not None:
                    result = execute_matmul(lowering, operands[0], operands[1], array_module)
                    contracted = result
                else:
                    # Build/cache only the shape-level expression.  The call
                    # still supplies every fresh operand on every execution,
                    # so no input value, device, stream, or constant tensor is
                    # retained by the cache.  ``backend='cupy'`` keeps the
                    # operation on device.
                    if kernels is None:
                        expression = _cupy_contract_expression(
                            operation.equation,
                            operand_shapes,
                            operand_dtypes,
                        )
                    contracted = expression(*operands, backend="cupy")
                    result = array_module.array(contracted, copy=True)
        except Exception as exc:
            raise RuntimeError(
                f"{backend_name} failed operation {operation.index} ({operation.equation!r})"
            ) from exc
        if tuple(result.shape) != operation.output_shape:
            raise RuntimeError(
                f"operation {operation.index} produced shape {tuple(result.shape)!r}, "
                f"expected {operation.output_shape!r}"
            )
        node_ref = NodeRef(operation.index)
        node_values[node_ref] = result
        node_live_bytes += int(result.nbytes)
        # The result is logically live before dependencies are released.  The
        # temporary backend work area used to create it is deliberately omitted.
        record_peak()
        for ref in operation.inputs:
            node_live_bytes -= _decrement_use(ref, remaining_uses, node_values)
        record_peak()
        del operands, result, contracted

    for term in plan.terms:
        raw = resolve(term.output)
        if tuple(raw.shape) != term.output_shape:
            raise RuntimeError(
                f"term output has shape {tuple(raw.shape)!r}, expected {term.output_shape!r}"
            )
        # Multiplication is intentionally performed during assembly.  A fresh
        # output prevents a direct identity term or shared node from escaping
        # through ExecutionResult.values.
        try:
            output = array_module.empty(term.output_shape, dtype=term.output_dtype)
            array_module.multiply(raw, term.coefficient, out=output)
        except Exception as exc:
            raise RuntimeError(
                f"failed to assemble a coefficient-scaled {backend_name} term output"
            ) from exc
        if tuple(output.shape) != term.output_shape:
            raise RuntimeError(
                f"assembled term output has shape {tuple(output.shape)!r}, expected {term.output_shape!r}"
            )
        output_values.append(output)
        output_live_bytes += int(output.nbytes)
        record_peak()
        node_live_bytes -= _decrement_use(term.output, remaining_uses, node_values)
        record_peak()
        del raw

    if stream is not None:
        # All backend work and last-use releases were enqueued on this stream;
        # synchronize before returning so callers can time a complete call.
        stream.synchronize()

    stats = ExecutionStats(
        executed_operations=len(plan.operations),
        total_operations=plan.total_operations,
        unique_operations=plan.unique_operations,
        shared_operations=plan.shared_operations,
        peak_live_bytes=peak_live_bytes,
        peak_intermediate_bytes=peak_intermediate_bytes,
        input_bytes=input_bytes,
        output_bytes=int(sum(value.nbytes for value in output_values)),
        scratch_bytes=None,
    )
    return ExecutionResult(values=tuple(output_values), stats=stats)


def execute_plan(
    plan: ContractionPlan,
    bindings: Mapping[object, object],
    *,
    backend: Any = None,
    kernels: PreparedKernels | None = None,
    execution_cache=None,
) -> ExecutionResult:
    """Execute ``plan`` synchronously on NumPy or an optional CuPy backend.

    ``backend=None`` infers the backend from all bound arrays.  Explicit
    ``"numpy"`` and ``"cupy"`` requests are accepted (as are the respective
    imported modules).  A call must use one backend and, for CuPy, one CUDA
    device.  Arrays are borrowed without host conversion; intermediate and
    output arrays are owned by the selected backend.  CuPy work is enqueued on
    the current stream of the common device and synchronized before return;
    callers must ensure producer work for the input arrays is ready on that
    stream because cross-stream producer events are not inferred.
    """

    if not isinstance(plan, ContractionPlan):
        raise TypeError("plan must be a ContractionPlan returned by compile_plan")
    if execution_cache is not None:
        with execution_cache.scope(plan, backend):
            return execute_plan(plan, bindings, backend=backend, kernels=kernels)
    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping of input keys to NumPy or CuPy ndarrays")
    from lqcdweave.contracts.correlator.precision import require_uniform_contraction_dtype
    require_uniform_contraction_dtype(plan.specs)
    if kernels is not None and (
        not isinstance(kernels, PreparedKernels) or kernels.plan is not plan
        or len(kernels.operations) != len(plan.operations)
    ):
        raise ValueError("prepared kernels belong to another contraction plan")

    backend_name, array_module, input_values, device_id = _resolve_backend(
        plan, bindings, backend
    )
    if backend_name == "numpy":
        return _execute_bound_plan(
            plan,
            input_values,
            backend_name=backend_name,
            array_module=array_module,
            kernels=kernels,
        )

    # Enter the common device context before choosing a stream.  This also
    # handles a caller that supplied arrays on a non-current but consistent
    # device without copying them to host or another device.
    if device_id is None:
        return _execute_bound_plan(
            plan,
            input_values,
            backend_name=backend_name,
            array_module=array_module,
            kernels=kernels,
        )
    with array_module.cuda.Device(device_id):
        stream = array_module.cuda.get_current_stream()
        return _execute_bound_plan(
            plan,
            input_values,
            backend_name=backend_name,
            array_module=array_module,
            stream=stream,
            kernels=kernels,
        )
