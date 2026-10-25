"""Domain validation and execution for bound contraction descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import nullcontext

import numpy as np

from lqcdweave.contracts.execution import (
    DirectExecution,
    ExecutionRequest,
    PathExecution,
    TermValue,
)
from lqcdweave.contracts.correlator.api import (
    ExpandedCorrelator,
    PlannedContraction,
    SceneStoragePlan,
)
from lqcdweave.contraction.planning.api import validate_execution_binding

from .diagnostics import debug_enabled, execution_debug
from .binding_cache import ExecutionBindingCache
from .device import device_execution, active_device_execution
from .direct import _NoSearchBackend, _resolve_backend, execute_direct_contraction
from .storage_runtime import replay_scene_storage, replay_storage_batches


@dataclass(frozen=True)
class BoundContractionExecution:
    """One expansion/path pair validated and executed by the contraction layer."""

    expanded: ExpandedCorrelator
    path: PlannedContraction
    debug: bool = False
    execution_cache: object = field(default_factory=ExecutionBindingCache, repr=False, compare=False)
    runtime_device_limit: int | None = None

    def __post_init__(self):
        _validate_execution_entry(self.expanded, self.path)

    def execute(self, *, backend: object = np, measure: bool = False,
                io_overlap: bool = True) -> PathExecution:
        return execute_contraction_path(self.expanded, self.path,
                                        backend=backend, measure=measure, io_overlap=io_overlap,
                                        debug=self.debug, execution_cache=self.execution_cache,
                                        runtime_device_limit=self.runtime_device_limit)


def execute_contraction_path(
    expanded: ExpandedCorrelator,
    path: PlannedContraction,
    *,
    backend: object = np,
    measure: bool = False,
    io_overlap: bool = True,
    debug: bool = False,
    execution_cache=None,
    runtime_device_limit=None,
    batch_consumer=None,
) -> PathExecution:
    """Execute one bound descriptor without searching or assembling its output.

    W may supply batch_consumer(index, ordered_raw_values, output_axes) for
    synchronous unmeasured consumption. No raw TermValues are then retained;
    the consumer must copy its final fields before returning. This is a backend
    handoff, not a user-configurable compute/assembly workflow.
    """
    _validate_execution_entry(expanded, path)
    if batch_consumer is not None:
        if not callable(batch_consumer):
            raise TypeError('batch_consumer must be callable')
        if measure or expanded.inputs is None or not expanded.inputs.task_batches:
            raise ValueError('batch_consumer requires unmeasured coordinate task batches')
    # PlannedContraction also owns the current expanded loader/readers. Cache
    # only the pure scene plans, never that runtime binding envelope.
    plans = (tuple(item.plan for item in path.batch_paths) if path.batch_paths else path.plan)
    scope = nullcontext() if execution_cache is None else execution_cache.scope(plans, backend)
    limit = path.memory_limit if runtime_device_limit is None else min(path.memory_limit, runtime_device_limit)
    with execution_debug(debug), scope, device_execution(
            backend, device_memory_limit=limit) as device:
        result = _execute_contraction_path(
            expanded, path, backend=backend, measure=measure, io_overlap=io_overlap,
            batch_consumer=batch_consumer,
        )
        if device is not None and expanded.inputs is not None and not expanded.inputs.task_batches:
            values = device.download(tuple(item.value for item in result.term_values))
            result = replace(result, term_values=tuple(
                replace(item, value=value) for item, value in zip(result.term_values, values, strict=True)))
        return result


def _execute_contraction_path(
    expanded: ExpandedCorrelator,
    path: PlannedContraction,
    *,
    backend: object,
    measure: bool,
    io_overlap: bool,
    batch_consumer=None,
) -> PathExecution:
    if debug_enabled():
        validate_execution_binding(expanded, path)
    _validate_execution_backend(path, backend)
    if type(measure) is not bool:
        raise TypeError("measure must be bool")
    if type(io_overlap) is not bool:
        raise TypeError("io_overlap must be bool")
    device = active_device_execution()
    if device is not None:
        eligible = (not debug_enabled() and path.execution_mode == 'planned'
                    and not expanded.inputs.task_batches
                    and not path.plan.retained_before and not path.plan.retained_after)
        device.configure_constants(path.plan if eligible else None,
            getattr(expanded.inputs.loader, 'preparations', None) if eligible else None)

    if path.execution_mode == "direct":
        if not expanded.terms:
            raise ValueError("direct mode is only valid for a correlator with Terms")
        result = (_execute_task_batch_plans(expanded, path, backend=backend, measure=measure,
                                           io_overlap=io_overlap, batch_consumer=batch_consumer)
                  if expanded.inputs is not None and expanded.inputs.task_batches
                  else execute_direct_contraction(expanded, path, backend=backend, measure=measure))
        if not isinstance(result, (DirectExecution, PathExecution)):
            raise TypeError("direct executor must return a typed DirectExecution or PathExecution")
        if batch_consumer is None:
            _check_term_values(expanded, result.term_values)
        return PathExecution(tuple(result.term_values), result.report)

    if path.execution_mode == "empty":
        if expanded.terms or path.plan is not None:
            raise ValueError("empty execution descriptor disagrees with expanded Terms")
        return PathExecution(())

    if path.execution_mode != "planned":
        raise ValueError(f"unknown execution mode {path.execution_mode!r}")
    if not expanded.terms:
        raise ValueError("planned mode requires at least one ContractionTerm")
    if path.plan is None:
        raise ValueError("a nonempty correlator cannot execute with plan=None")
    if path.plan_identity != id(path.plan):
        raise ValueError("plan object no longer matches the selected path descriptor")
    if expanded.inputs is None or expanded.inputs.loader is None:
        raise ValueError("expanded inputs have no loader")
    if expanded.inputs.task_batches:
        result = _execute_task_batch_plans(expanded, path, backend=backend, measure=measure,
                                           io_overlap=io_overlap, batch_consumer=batch_consumer)
        if batch_consumer is None:
            _check_term_values(expanded, result.term_values)
        return result
    if len(expanded.coordinates.points) != 1:
        raise NotImplementedError("planned execution currently supports one coordinate per invocation")
    report = replay_scene_storage(path.plan, expanded.inputs.loader,
                                  backend=backend, measure=measure, io_overlap=io_overlap)
    values = tuple(report.values) if measure else tuple(report)
    if len(values) != len(expanded.inputs.term_ids):
        raise ValueError("execution returned a different number of term results than planned")
    coordinate_key = expanded.coordinates.points[0].key
    term_values = tuple(
        TermValue(term_id, coordinate_key, value)
        for term_id, value in zip(expanded.inputs.term_ids, values, strict=True)
    )
    _check_term_values(expanded, term_values)
    return PathExecution(term_values, report if measure else None)


def _validate_execution_entry(expanded, path) -> None:
    """Check only the frozen descriptor identity and execution admission facts."""
    if not isinstance(expanded, ExpandedCorrelator):
        raise TypeError("expanded must be an ExpandedCorrelator")
    if not isinstance(path, PlannedContraction):
        raise TypeError("path must be a PlannedContraction")
    if not isinstance(path.expanded, ExpandedCorrelator):
        raise TypeError("path descriptor does not contain an ExpandedCorrelator")
    if path.expanded.identity_token is not expanded.identity_token:
        raise ValueError("path was selected for a different ExpandedCorrelator")
    if path.execution_mode == "empty":
        if expanded.terms or path.plan is not None:
            raise ValueError("empty execution descriptor disagrees with expanded Terms")
        return
    if path.execution_mode not in {"planned", "direct"}:
        raise ValueError(f"unknown execution mode {path.execution_mode!r}")
    if not isinstance(path.plan, SceneStoragePlan):
        raise ValueError("a nonempty path needs an audited SceneStoragePlan")
    if path.plan_identity != id(path.plan):
        raise ValueError("plan object no longer matches the selected path descriptor")
    if not path.plan.feasible:
        raise MemoryError("the stored path is not feasible under its audited memory limits")
    if expanded.inputs is None or expanded.inputs.problem is None:
        raise ValueError("nonempty plan has no matching DiagramInputs")
    if expanded.inputs.loader is None:
        raise ValueError("expanded inputs have no loader")


def _validate_execution_backend(path, backend) -> None:
    if path.execution_mode == "empty":
        return
    backend_name = backend if isinstance(backend, str) else getattr(backend, "__name__", None)
    if backend_name not in {"numpy", "cupy"}:
        raise TypeError("execution backend must be NumPy or CuPy")
    if backend_name != path.plan.state.problem.backend:
        raise ValueError("execution backend disagrees with the ContractionProblem")


def _execute_task_batch_plans(expanded, path, *, backend, measure, io_overlap=True, batch_consumer=None):
    """Replay one globally joined storage schedule for a task's compute batches."""
    if path.plan_identity != id(path.plan):
        raise ValueError("task path plan object no longer matches its selected descriptor")
    if path.expanded.identity_token is not expanded.identity_token:
        raise ValueError("path was selected for a different ExpandedCorrelator")
    task_batches = expanded.inputs.task_batches
    batch_paths = path.batch_paths
    mode = path.execution_mode
    if (mode not in {"planned", "direct"}
            or len(batch_paths) != len(task_batches)
            or any(item.plan is None for item in batch_paths)):
        raise ValueError("planned task batches do not match their executed coordinate groups")
    if any(item.execution_mode != mode
           or item.expanded.identity_token is not expanded.identity_token
           or item.plan_identity != id(item.plan)
           for item in batch_paths):
        raise ValueError("one task batch path is foreign, incomplete, or internally inconsistent")
    plans = tuple(item.plan for item in batch_paths)
    execution_backend = backend
    if mode == "direct":
        _, module = _resolve_backend(backend)
        execution_backend = _NoSearchBackend(module)
    loader = expanded.inputs.loader
    batch_context_index = [0]
    values, reports = [], []

    def activate_batch(plan):
        index = batch_context_index[0]
        if index >= len(plans) or plans[index] is not plan:
            raise ValueError("joined storage schedule order differs from the selected task batches")
        batch_context_index[0] += 1
        activate = getattr(loader, "activate", None)
        if activate is None:
            return nullcontext()
        if not callable(activate):
            raise TypeError("loader.activate must be callable when present")
        return activate(index)

    def consume(index, report):
        batch_inputs = task_batches[index]
        batch_points = batch_inputs.coordinate_batch.points
        raw_values = report.values if measure else report
        device = active_device_execution()
        if device is not None:
            # The replay output already owns stable packed allocations. Move
            # these owners once, then split coordinates on CPU; one take/copy
            # kernel per Term per lane would undo numerical time batching.
            raw_values = device.download(raw_values)
        else:
            raw_values = tuple(raw_values)
        output_axes = batch_inputs.batch_output_axes or tuple(None for _ in batch_inputs.term_ids)
        if len(raw_values) != len(batch_inputs.term_ids):
            raise ValueError("planned batch returned a different number of term results")
        if len(output_axes) != len(batch_inputs.term_ids):
            raise ValueError("planned batch output axes do not match its Terms")
        if batch_consumer is not None:
            # W consumes the whole joint batch synchronously. It may retain
            # assembled fields, never these raw term/device owners.
            batch_consumer(index, raw_values, output_axes)
            return
        for term_id, value, output_axis in zip(
                batch_inputs.term_ids, raw_values, output_axes, strict=True):
            if output_axis is None:
                for point in batch_points:
                    values.append(TermValue(term_id, point.key, np.array(value, copy=True)))
            else:
                if output_axis >= value.ndim or value.shape[output_axis] != len(batch_points):
                    raise ValueError(f"planned batch output axis for {term_id!r} has the wrong extent")
                for lane, point in enumerate(batch_points):
                    lane_value = np.take(value, lane, axis=output_axis)
                    values.append(TermValue(term_id, point.key, np.array(lane_value, copy=True)))
        if measure:
            reports.append(replace(report, values=tuple(raw_values)))

    replay_storage_batches(
        plans, loader, consume, backend=execution_backend, measure=measure,
        prejoined=True, batch_context=activate_batch,
        source_descriptions=tuple(batch_inputs.problem.sources for batch_inputs in task_batches),
        io_overlap=io_overlap,
    )
    report = None if not measure else {
        "execution_mode": f"{mode}-batches",
        "backend": getattr(backend, "__name__", str(backend)),
        "batch_reports": tuple(reports),
        "calibration_eligible": False if mode == "direct" else None,
    }
    return PathExecution(tuple(values), report)


def _check_term_values(expanded, term_values):
    """Keep cardinality cheap; internal coverage reconstruction is diagnostic."""
    inputs = expanded.inputs
    points = (() if inputs is None else
              inputs.coordinate_batch.points if inputs.coordinate_batch is not None
              else expanded.coordinates.points)
    count = 0 if inputs is None else len(inputs.term_ids) * len(points)
    if len(term_values) != count:
        raise ValueError("execution returned a different number of term results than requested")
    if debug_enabled():
        _validate_term_values(expanded, term_values)


def _validate_term_values(expanded: ExpandedCorrelator,
                          term_values: tuple[TermValue, ...]) -> None:
    if expanded.inputs is None:
        if term_values:
            raise ValueError("constant correlator cannot return term values")
        return
    points = (expanded.inputs.coordinate_batch.points
              if expanded.inputs.coordinate_batch is not None
              else expanded.coordinates.points)
    expected = {
        (term_id, point.key)
        for term_id in expanded.inputs.term_ids
        for point in points
    }
    actual = [(item.term_id, item.coordinate_key) for item in term_values]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            "direct/executor results must cover each requested (term_id, coordinate_key) once"
        )


__all__ = [
    "BoundContractionExecution",
    "execute_contraction_path",
]
