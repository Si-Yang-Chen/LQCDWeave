"""Pure memory estimation and budget-cap composition for planning.

This module reads declared plan metadata and explicit memory observations. It
does not load arrays, inspect a numerical backend, or query process/device
resources; runtime callers collect observations before calling the resolver.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import numbers
from typing import Any, Literal

import numpy as np

from lqcdweave.contracts.memory import BudgetError, MemoryEstimate, MemoryLimits


GIB = 1024 ** 3
DEFAULT_GPU_RESERVE_BYTES = GIB
DEFAULT_GPU_RESERVE_FRACTION = 0.10
DEFAULT_HOST_AVAILABLE_FRACTION = 0.70


def _nonnegative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a non-negative integer or None")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return int(value)


def _first_budget_value(budget: Any, names: tuple[str, ...], *, name: str) -> int | None:
    if budget is None:
        return None
    for candidate in names:
        if hasattr(budget, candidate):
            value = getattr(budget, candidate)
            if callable(value):
                continue
            return _nonnegative_int(value, name=name)
    return None


def user_memory_cap(budget: Any, *, kind: str) -> int | None:
    """Read a user-specified host or device cap from a budget declaration."""

    if kind == "device":
        names = ("device_bytes",)
    elif kind == "host":
        names = ("host_bytes",)
    else:
        raise ValueError("kind must be 'host' or 'device'")
    return _first_budget_value(budget, names, name=f"{kind} memory cap")


def workspace_reserve(budget: Any) -> int:
    """Return the caller-declared backend/staging reserve in bytes."""

    value = _first_budget_value(
        budget,
        ("workspace_bytes",),
        name="workspace reserve",
    )
    return 0 if value is None else value


def resolve_budget_limits(
    budget: Any,
    *,
    target: Literal["cpu", "gpu"],
    host_available: int | None,
    device_info: tuple[int, int] | None = None,
) -> MemoryLimits:
    """Combine a declared budget with explicit host/device observations.

    This is deliberately a pure helper: ``host_available`` and ``device_info``
    are observations supplied by the caller, and this function never probes a
    process or imports a backend. The legacy 70% host allowance and device
    reserve calculation are retained, including truncating fractional reserve
    bytes and permitting a zero-byte device total.
    """

    if target not in ("cpu", "gpu"):
        raise ValueError("target must be 'cpu' or 'gpu'")

    host_cap = user_memory_cap(budget, kind="host")
    device_cap = user_memory_cap(budget, kind="device")
    observed_host = _nonnegative_int(host_available, name="host available memory")
    if observed_host is None:
        if host_cap is None:
            raise BudgetError(
                "cannot query available host memory; supply ExecutionBudget(host_bytes=...)"
            )
        resolved_host = host_cap
    else:
        default_host = int(observed_host * DEFAULT_HOST_AVAILABLE_FRACTION)
        resolved_host = default_host if host_cap is None else min(host_cap, default_host)

    free = total = reserve = None
    if target == "gpu":
        if device_info is None:
            raise BudgetError(
                "cannot query available GPU memory; supply device_info=(free_bytes, total_bytes)"
            )
        if len(device_info) != 2:
            raise ValueError("device_info must be a (free_bytes, total_bytes) pair")
        free, total = (
            _nonnegative_int(value, name="device memory") for value in device_info
        )
        assert free is not None and total is not None
        fraction = getattr(budget, "reserve_fraction", DEFAULT_GPU_RESERVE_FRACTION)
        minimum = getattr(budget, "min_reserve_bytes", DEFAULT_GPU_RESERVE_BYTES)
        if isinstance(fraction, bool) or not isinstance(fraction, numbers.Real):
            raise TypeError("reserve_fraction must be a non-negative number")
        if fraction < 0:
            raise ValueError("reserve_fraction must be non-negative")
        minimum = _nonnegative_int(minimum, name="min_reserve_bytes")
        assert minimum is not None
        reserve = max(int(total * float(fraction)), minimum)
        available_device = max(int(free) - reserve, 0)
        resolved_device = available_device if device_cap is None else min(device_cap, available_device)
    else:
        # A CPU logical device cap remains useful for CPU-only preflight checks.
        resolved_device = device_cap

    return MemoryLimits(
        host_bytes=resolved_host,
        device_bytes=resolved_device,
        host_available_bytes=observed_host,
        device_free_bytes=free,
        device_total_bytes=total,
        device_reserve_bytes=reserve,
    )


def estimate_plan_memory(
    plan: Any,
    specs: Mapping[Any, Any] | None = None,
    *,
    budget: Any = None,
    output_bytes: int | None = None,
    assembly_extra_bytes: int = 0,
    snapshot_inputs: bool = True,
) -> MemoryEstimate:
    """Conservatively estimate one compiled plan's logical memory.

    The estimator only uses declared shapes/dtypes and compiled operation
    metadata. It never asks a loader for an array. Unknown backend scratch is
    represented by ``budget.workspace_bytes`` and is intentionally not
    inferred.
    """

    if specs is None:
        specs = getattr(plan, "specs", {})
    if not isinstance(specs, Mapping):
        raise TypeError("specs must be a mapping")
    input_bytes = 0
    for _key, spec in specs.items():
        shape = tuple(int(dimension) for dimension in spec.shape)
        dtype = np.dtype(spec.dtype)
        input_bytes += math.prod(shape) * int(dtype.itemsize)

    # Mirror the executor's synchronous last-use accounting. Shared
    # intermediates stay alive until every operation and term consumer runs.
    from lqcdweave.contracts.correlator.api import InputRef, NodeRef

    operations = tuple(getattr(plan, "operations", ()))
    terms = tuple(getattr(plan, "terms", ()))
    node_bytes = {
        NodeRef(operation.index): math.prod(tuple(int(d) for d in operation.output_shape))
        * int(np.dtype(operation.output_dtype).itemsize)
        for operation in operations
    }
    remaining: dict[NodeRef, int] = dict.fromkeys(node_bytes, 0)
    for operation in operations:
        for ref in operation.inputs:
            if isinstance(ref, NodeRef):
                remaining[ref] += 1
    for term in terms:
        if isinstance(term.output, NodeRef):
            remaining[term.output] += 1
    live: dict[NodeRef, int] = {}
    live_bytes = 0
    intermediate_peak = 0
    pack_peak = 0
    for operation in operations:
        operation_bytes = node_bytes[NodeRef(operation.index)]
        operand_bytes = 0
        for ref in operation.inputs:
            if isinstance(ref, NodeRef):
                operand_bytes += node_bytes.get(ref, 0)
            elif isinstance(ref, InputRef):
                spec = specs[ref.key]
                operand_bytes += math.prod(tuple(int(d) for d in spec.shape)) * int(
                    np.dtype(spec.dtype).itemsize
                )
        # Account for packed operands and one eager contraction result.
        pack_peak = max(pack_peak, operand_bytes + operation_bytes)
        live[NodeRef(operation.index)] = operation_bytes
        live_bytes += operation_bytes
        intermediate_peak = max(intermediate_peak, live_bytes)
        for ref in operation.inputs:
            if isinstance(ref, NodeRef):
                remaining[ref] -= 1
                if remaining[ref] == 0:
                    live_bytes -= live.pop(ref, 0)
        intermediate_peak = max(intermediate_peak, live_bytes)

    if output_bytes is None:
        output_bytes = 0
        for term in terms:
            shape = tuple(int(dimension) for dimension in term.output_shape)
            output_bytes += math.prod(shape) * int(np.dtype(term.output_dtype).itemsize)
    output_bytes = _nonnegative_int(output_bytes, name="output bytes")
    assembly_extra_bytes = _nonnegative_int(
        assembly_extra_bytes, name="assembly extra bytes"
    ) or 0
    output_temp = max(
        (
            math.prod(tuple(int(d) for d in term.output_shape))
            * int(np.dtype(term.output_dtype).itemsize)
            for term in terms
        ),
        default=0,
    )
    return MemoryEstimate(
        input_bytes=input_bytes,
        snapshot_bytes=input_bytes if snapshot_inputs else 0,
        intermediate_bytes=intermediate_peak,
        output_bytes=output_bytes,
        assembly_extra_bytes=assembly_extra_bytes,
        workspace_bytes=workspace_reserve(budget) + max(pack_peak, output_temp),
    )


__all__ = [
    "DEFAULT_GPU_RESERVE_BYTES",
    "DEFAULT_GPU_RESERVE_FRACTION",
    "DEFAULT_HOST_AVAILABLE_FRACTION",
    "estimate_plan_memory",
    "resolve_budget_limits",
    "user_memory_cap",
    "workspace_reserve",
]
