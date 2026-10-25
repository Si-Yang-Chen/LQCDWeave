"""Finite structural schedules for persisted, memory-bounded workloads.

These candidates express an I/O versus shared-computation tradeoff.  Their
order is a conservative fallback policy, not a prediction of elapsed time.
Only representative end-to-end calibration can choose the fastest schedule.
"""

from __future__ import annotations

import math
import numbers
import time

from lqcdweave.contracts.memory import BudgetExceededError
from lqcdweave.contracts.correlator.api import (
    ExecutionBudget,
    PlanBatch,
    PlanBundle,
    WorkloadSpec,
)
from .compiler import compile_plan
from .memory import estimate_plan_memory


def _bytes(spec):
    return math.prod(spec.shape) * spec.dtype.itemsize


def list_plan_candidates(
    spec: WorkloadSpec,
    budget: ExecutionBudget | None = None,
    *,
    max_candidates: int = 4,
    optimize: str = "greedy",
    target: str | None = None,
) -> tuple[PlanBundle, ...]:
    """Build up to four feasible candidates without reading numerical inputs.

    Compare whole-workload, half-width and individual-output-group schedules
    using the greedy path planner.  Groups remain indivisible and retain their
    declared order.  Bounds apply to retained outputs across *all* batches,
    not just the current output group. Startup availability and reserve policy
    are resolved by the caller and supplied through ``budget``; this planner
    does not query a backend. When ``target=None``, the presence of
    ``budget.device_bytes`` retains the former ``backend=None`` CPU/GPU
    accounting inference. Runtime also checks current host/device availability.

    The widest feasible greedy schedule is returned first by default.  The
    smallest feasible schedule is retained when there is room, so calibration
    can measure the cost of rereading inputs versus lower residency.  Widths
    are finite samples: feasibility is not assumed monotonic with width.
    """
    if not isinstance(spec, WorkloadSpec):
        raise TypeError("spec must be a WorkloadSpec")
    if not spec.terms:
        raise ValueError("workload must contain at least one contraction term")
    budget = ExecutionBudget() if budget is None else budget
    if not isinstance(budget, ExecutionBudget):
        raise TypeError("budget must be an ExecutionBudget")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, numbers.Integral):
        raise TypeError("max_candidates must be an integer from 1 to 4")
    if not 1 <= max_candidates <= 4:
        raise ValueError("max_candidates must be from 1 to 4")
    if optimize == "joint":
        raise ValueError("optimize='joint' is retired; only 'greedy' schedules are supported")
    if optimize != "greedy":
        raise ValueError("optimize must be 'greedy'; optimize='joint' is retired")
    if target is None:
        target_is_gpu = budget.device_bytes is not None
    elif target == "cpu":
        target_is_gpu = False
    elif target == "gpu":
        target_is_gpu = True
    elif not isinstance(target, str):
        raise TypeError("target must be None, 'cpu', or 'gpu'")
    else:
        raise ValueError("target must be None, 'cpu', or 'gpu'")
    started = time.perf_counter()
    groups = spec.output_groups
    widths = tuple(dict.fromkeys((len(groups), (len(groups) + 1) // 2, 1))) if groups else (1,)
    # Compile singleton groups once.  Besides discovering output sizes, this
    # validates equations without constructing a full, potentially huge DAG.
    cache = {}

    def compiled(indices):
        key = indices
        if key not in cache:
            terms = tuple(spec.terms[index] for index in indices)
            used = {name for term in terms for name in term.inputs}
            inputs = {name: item for name, item in spec.specs.items() if name in used}
            cache[key] = compile_plan(inputs, terms, optimize="greedy")
        return cache[key]

    total_outputs = sum(
        math.prod(term.output_shape) * term.output_dtype.itemsize
        for group in groups for term in compiled(group).terms
    )
    assembly_bytes = spec.metadata.get("assembly_extra_bytes", 0)
    feasible = {}
    rejected = []
    for width in widths:
        indices_batches = tuple(
            tuple(index for group in groups[start:start + width] for index in group)
            for start in range(0, len(groups), width)
        )
        batches = []
        for indices in indices_batches:
            plan = compiled(indices)
            estimate = estimate_plan_memory(
                plan, budget=budget, output_bytes=total_outputs,
                assembly_extra_bytes=assembly_bytes,
            )
            # A declared device cap identifies GPU planning; the host cap
            # then protects staging/snapshots rather than GPU intermediates.
            host_need = (estimate.input_bytes + estimate.snapshot_bytes + budget.host_staging_bytes
                         if target_is_gpu
                         else estimate.logical_peak_bytes + budget.host_staging_bytes)
            violation = None
            if budget.host_bytes is not None and host_need > budget.host_bytes:
                violation = ("host", host_need, budget.host_bytes)
            if (target_is_gpu and budget.device_bytes is not None
                    and estimate.logical_peak_bytes > budget.device_bytes):
                violation = ("device", estimate.logical_peak_bytes, budget.device_bytes)
            if violation:
                rejected.append((width, indices, *violation))
                break
            memory = estimate.as_dict()
            memory["host_required_bytes"] = host_need
            batches.append(PlanBatch(plan, indices, memory))
        else:
            feasible[width] = tuple(batches)

    if not feasible:
        detail = "; ".join(
            f"indivisible group {indices} needs {need} {kind} bytes (limit {limit})"
            for width, indices, kind, need, limit in rejected if width == 1
        )
        raise BudgetExceededError("No sampled schedule fits the budget. " + detail)

    # Prefer the widest feasible schedule, retain the narrowest for calibration,
    # then include any intermediate sampled widths. Never sort by FLOPs.
    choices = list(feasible)
    order = [max(choices), min(choices), *choices]
    chosen = list(dict.fromkeys(order))[:max_candidates]
    elapsed = time.perf_counter() - started
    bundles = []
    for width in chosen:
        batches = feasible[width]
        bundles.append(PlanBundle(spec, batches, budget, {
            "candidate_id": f"greedy-groups-{width}",
            "optimize": "greedy",
            "requested_optimize": optimize,
            "batch_width": width,
            "calibration_status": "uncalibrated",
            "calibration": {"status": "uncalibrated"},
            "planning_seconds": elapsed,
            "static_metrics": {
                "logical_peak_bytes": max(
                    (batch.memory_estimate["logical_peak_bytes"] for batch in batches), default=0,
                ),
                "input_bytes_without_cross_batch_cache": sum(
                    _bytes(item) for batch in batches for item in batch.plan.specs.values()
                ),
                "unique_operations": sum(batch.plan.unique_operations for batch in batches),
                "retained_output_bytes": total_outputs,
            },
            "selection_basis": "bounded structural candidates; no elapsed-time ranking",
        }))
    return tuple(bundles)


def plan_workload(spec, budget=None, *, optimize="greedy", max_candidates=4, target=None):
    """Return the widest feasible uncalibrated greedy schedule.

    Use :func:`list_plan_candidates` and end-to-end calibration to compare
    the finite feasible batch widths by elapsed time.
    """
    return list_plan_candidates(
        spec, budget, optimize=optimize,
        max_candidates=max_candidates, target=target,
    )[0]


__all__ = ["list_plan_candidates", "plan_workload"]
