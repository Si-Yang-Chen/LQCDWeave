"""Deterministic contraction execution without a path search or storage planner."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from functools import partial
import importlib
import math
from collections.abc import Mapping
from types import MappingProxyType
from time import perf_counter
from typing import Any

import numpy as np

from lqcdweave.contracts.execution import DirectExecution, TermValue
from lqcdweave.contracts.measurement import ActionTiming, FileIOEvent, PhaseTiming, ReplayReport
from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ExpandedCorrelator,
    PlannedContraction,
    SourceLoader,
    SourceSpec,
    SourcePreparation,
    StorageBlock,
    validate_executable_resources,
)
from lqcdweave.contraction.planning.api import validate_execution_binding
from lqcdweave.contraction.performance.api import ExecutionMeasurement

from .diagnostics import debug_enabled
from .operations.contract import contract_dense
from .operations.baryon_join import contract_baryon_join
from .operations.load import read_block
from .operations.materialize import materialize_from_blocks
from .operations.output import copy_output, select_output_lane
from .operations.sparse_bridge import (
    action_bridges, contract_sparse_bridge, validate_sparse_bridge_parts,
)
from .storage_runtime import validate_coordinate_batch_workspace, _bind_loader_for_plan


class _NoSearchBackend:
    """Delegate array operations while forcing every sparse-kernel einsum to no-search."""

    def __init__(self, module):
        self._module = module

    @property
    def __name__(self):
        return self._module.__name__

    def __getattr__(self, name):
        return getattr(self._module, name)

    def einsum(self, subscripts, *operands, **kwargs):
        kwargs["optimize"] = False
        return self._module.einsum(subscripts, *operands, **kwargs)


def _resolve_backend(backend):
    if isinstance(backend, str):
        if backend == "numpy":
            backend = np
        elif backend == "cupy":
            backend = importlib.import_module("cupy")
        else:
            raise ValueError("backend must be numpy, cupy, or the corresponding array module")
    if backend is None or not callable(getattr(backend, "einsum", None)):
        raise TypeError("direct execution requires an explicit NumPy or CuPy backend")
    backend_name = getattr(backend, "__name__", None)
    if backend_name not in ("numpy", "cupy") or not isinstance(
        getattr(backend, "ndarray", None), type
    ):
        raise TypeError("direct execution backend must expose its ndarray type and einsum")
    return backend_name, backend


def _backend_array(value, module, description):
    """Accept an array or scalar tensor, rejecting implicit array-like coercion."""
    if isinstance(value, module.ndarray):
        return value
    if np.isscalar(value) or getattr(value, "shape", None) == ():
        try:
            return module.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{description} is not a valid {module.__name__} scalar") from exc
    raise TypeError(f"{description} must be a {module.__name__} ndarray or scalar tensor")


def _required_keys(problem: ContractionProblem) -> tuple[str, ...]:
    """Return the unique logical tensors and active sparse records in term order."""
    required = []
    for term_index, term in enumerate(problem.terms):
        for key in term.inputs:
            if key not in required:
                required.append(key)
        counts = {key: term.inputs.count(key) for key in set(term.inputs)}
        for bridge in problem.sparse_bridges:
            if bridge.scene not in (None, term_index):
                continue
            if counts.get(bridge.left, 0) < (2 if bridge.left == bridge.right else 1):
                continue
            if counts.get(bridge.right, 0) < 1:
                continue
            if bridge.record not in required:
                required.append(bridge.record)
    for join in problem.baryon_joins:
        for key in ((join.spin, join.elemental) if join.core is None else join.core.inputs):
            if key not in required:
                required.append(key)
    return tuple(required)


def _validate_input_catalog(inputs, problem, required) -> SourceLoader:
    """Validate every needed contract before asking the loader to read anything."""
    if not isinstance(inputs.blocks, Mapping) or not isinstance(inputs.preparations, Mapping):
        raise TypeError("direct execution requires block and preparation mappings")
    loader = inputs.loader
    if not callable(getattr(loader, "read_block", None)):
        raise NotImplementedError(
            "this adapter has no direct contraction implementation: inputs need "
            "SourceLoader.read_block"
        )
    close = getattr(loader, "close_block", None)
    if close is not None and not callable(close):
        raise TypeError("loader.close_block must be callable when present")
    for name in ("close_all", "drain_io_events", "set_measurement"):
        method = getattr(loader, name, None)
        if method is not None and not callable(method):
            raise TypeError(f"loader.{name} must be callable when present")

    for key in required:
        if key not in problem.sources:
            raise ValueError(f"direct input {key!r} has no SourceSpec")
        if not isinstance(problem.sources[key], SourceSpec):
            raise TypeError(f"direct input {key!r} has an invalid SourceSpec")
        if key not in inputs.preparations:
            raise ValueError(f"direct input {key!r} has no SourcePreparation")
        preparation = inputs.preparations[key]
        if not isinstance(preparation, SourcePreparation):
            raise TypeError(f"direct input {key!r} has an invalid SourcePreparation")
        if preparation.recipe is None and not callable(getattr(loader, "materialize", None)):
            raise NotImplementedError("opaque direct inputs require loader.materialize")
        for block_id in preparation.blocks:
            if block_id not in inputs.blocks:
                raise ValueError(f"direct input {key!r} refers to missing block {block_id!r}")
            if not isinstance(inputs.blocks[block_id], StorageBlock):
                raise TypeError(f"direct block {block_id!r} must be a StorageBlock")

    return loader


def _timed_for_action(measurement, enabled, phase_map):
    def for_action(action_index):
        if not enabled:
            return lambda _name, operation, **_kwargs: operation()

        def timed(name, operation, *, nbytes=None, device=False):
            value, phase = measurement.measure_phase(
                name,
                operation,
                nbytes=nbytes,
                device=device,
                parent_action_index=action_index,
            )
            phase_map[action_index].append(phase)
            return value

        return timed

    return for_action


def _drain_io_events(loader, action_index, tensor):
    if not hasattr(loader, "drain_io_events"):
        return ()
    return tuple(
        FileIOEvent(
            action_index,
            tensor,
            str(event["path"]),
            str(event["operation"]),
            int(event["bytes"]),
            float(event["observed_seconds"]),
        )
        for event in loader.drain_io_events()
    )


def _read_and_materialize(inputs, problem, required, module, opened, *, loader,
                          timed_for_action, load_action_indices, io_events):
    """Read each declared block once and keep borrowed storage alive through execution."""
    blocks = {}
    arrays = {}

    for key in required:
        for block_id in inputs.preparations[key].blocks:
            if block_id not in blocks:
                action_index = load_action_indices[key]
                blocks[block_id] = read_block(
                    loader,
                    block_id,
                    inputs.blocks[block_id],
                    timed=timed_for_action(action_index),
                )
                opened.append((block_id, blocks[block_id]))
                if io_events is not None:
                    io_events.extend(_drain_io_events(loader, action_index, key))

    for key in required:
        action_index = load_action_indices[key]
        timed = timed_for_action(action_index)
        source = problem.sources[key]
        dependencies = inputs.preparations[key].blocks
        borrowed = MappingProxyType({block_id: blocks[block_id] for block_id in dependencies})
        value = materialize_from_blocks(
            loader, key, source, borrowed, timed=timed,
            device=module.__name__ == "cupy",
        )
        if io_events is not None:
            io_events.extend(_drain_io_events(loader, action_index, key))
        if key in problem.specs:
            expected = problem.specs[key]
            if not isinstance(value, module.ndarray):
                nbytes = math.prod(expected.shape) * expected.dtype.itemsize
                value = timed(
                    "input_transfer",
                    lambda value=value, key=key: _backend_array(value, module, f"loader value for {key}"),
                    nbytes=nbytes,
                    device=module.__name__ == "cupy",
                )
            else:
                value = _backend_array(value, module, f"loader value for {key}")
            try:
                actual_shape = tuple(value.shape)
                actual_dtype = np.dtype(value.dtype)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError(f"loader returned an invalid tensor for {key}") from exc
            if actual_shape != expected.shape or actual_dtype != expected.dtype:
                raise ValueError(
                    f"loader returned a mismatched tensor for {key}: "
                    f"shape={actual_shape}, dtype={actual_dtype}; "
                    f"expected shape={expected.shape}, dtype={expected.dtype}"
                )
        elif key in {bridge.record for bridge in problem.sparse_bridges}:
            source_metadata = problem.sources[key].metadata
            if (isinstance(source_metadata, Mapping)
                    and source_metadata.get("family") == "p2p-coordinate-batch"):
                # Batch payloads are already a lane-keyed mapping; preserve
                # that public contract instead of iterating mapping keys.
                pass
            else:
                # The scalar sparse kernels iterate parts in input order;
                # freeze one-shot iterables without changing their semantics.
                try:
                    value = timed("materialize_records", lambda value=value: tuple(value))
                except TypeError as exc:
                    raise TypeError(f"sparse source {key!r} must materialize bridge parts") from exc
            value = validate_sparse_bridge_parts(problem, key, value)
        arrays[key] = value
    return arrays


def _execute_actions(state, arrays, module, *, timed_for_action, sample_memory=None):
    sparse_backend = _NoSearchBackend(module)
    is_gpu = module.__name__ == "cupy"
    for action_index, action in enumerate(state.actions):
        if action.kind == "load":
            continue
        if action.kind == "contract":
            if action.reused and action.tensor not in arrays:
                raise RuntimeError("fixed direct contraction reused a missing intermediate")
            contract_dense(
                action,
                arrays,
                module,
                timed=timed_for_action(action_index),
                device=is_gpu,
                copy_result=False,
            )
        elif action.kind == "baryon_join":
            if action.reused and action.tensor not in arrays:
                raise RuntimeError("fixed baryon join reused a missing intermediate")
            contract_baryon_join(
                action, arrays, module, timed=timed_for_action(action_index),
                device=is_gpu, no_search=True,
            )
        elif action.kind == "sparse_bridge":
            if action.reused and action.tensor not in arrays:
                raise RuntimeError("fixed sparse bridge reused a missing intermediate")
            contract_sparse_bridge(
                state,
                action,
                arrays,
                sparse_backend,
                timed=timed_for_action(action_index),
                device=is_gpu,
            )
        else:
            raise NotImplementedError(f"direct action kind {action.kind!r} is unsupported")

        if not action.reused and debug_enabled():
            value = arrays.get(action.tensor)
            expected = state.catalog.get(action.tensor)
            if expected is None:
                raise RuntimeError(f"direct action has no TensorSpec for {action.tensor}")
            value = _backend_array(value, module, f"direct kernel result for {action.tensor}")
            if tuple(value.shape) != expected.shape or np.dtype(value.dtype) != expected.dtype:
                raise ValueError(
                    f"direct kernel returned a mismatched tensor for {action.tensor}: "
                    f"shape={tuple(value.shape)}, dtype={np.dtype(value.dtype)}; "
                    f"expected shape={expected.shape}, dtype={expected.dtype}"
                )
            arrays[action.tensor] = value
        if sample_memory is not None:
            sample_memory()


def execute_direct_contraction(expanded, path, *, backend=np, measure=False) -> DirectExecution:
    """Execute all terms in fixed order using P's audited storage schedule.

    This executor consumes the fixed state selected by P without searching a
    contraction path. Dense terms contract left-to-right with
    ``einsum(optimize=False)``. Declared P2P bridges run before ordinary factors
    and reuse the sparse bridge kernels through a backend adapter that forces
    their internal einsums to no-search. The SceneStoragePlan must be feasible
    before any source read. Source blocks remain alive through result snapshots,
    then close on success or failure. Current/sampling coefficients remain in
    the output recipe.
    """
    if not isinstance(expanded, ExpandedCorrelator):
        raise TypeError("expanded must be an ExpandedCorrelator")
    if not isinstance(path, PlannedContraction):
        raise TypeError("path must be a PlannedContraction")
    if not isinstance(path.expanded, ExpandedCorrelator):
        raise TypeError("path descriptor does not contain an ExpandedCorrelator")
    if path.execution_mode != "direct":
        raise ValueError("direct executor requires an optimize='none' PlannedContraction")
    if path.expanded.identity_token is not expanded.identity_token:
        raise ValueError("path was selected for a different ExpandedCorrelator")
    if type(measure) is not bool:
        raise TypeError("measure must be bool")
    if debug_enabled():
        validate_execution_binding(expanded, path)
    if expanded.inputs is None or not expanded.terms:
        raise ValueError("direct execution requires an ExpandedCorrelator with Terms")
    from lqcdweave.contracts.correlator.precision import require_uniform_contraction_dtype
    require_uniform_contraction_dtype(expanded.problem.specs)
    if path.plan is None:
        raise ValueError("direct execution requires a budget-audited fixed execution plan")
    if path.plan_identity != id(path.plan):
        raise ValueError("direct execution plan no longer matches the selected path descriptor")
    # Direct replay has no resource-action dispatcher; reject such descriptors
    # before the input catalog can open readers or materialize blocks.
    plan = path.plan
    validate_executable_resources(plan)
    if not plan.feasible:
        raise MemoryError("direct execution plan exceeds its audited storage budget")
    inputs = expanded.inputs
    coordinate_batch = inputs.coordinate_batch
    if coordinate_batch is None:
        if expanded.coordinates.scan_shape or len(expanded.coordinates.points) != 1:
            raise NotImplementedError(
                "direct scan execution requires explicit CoordinateBatch metadata"
            )
        batch_points = expanded.coordinates.points
    else:
        batch_points = coordinate_batch.points
        for index, point in zip(coordinate_batch.point_indices, batch_points, strict=True):
            if index >= len(expanded.coordinates.points) or expanded.coordinates.points[index] != point:
                raise ValueError("direct CoordinateBatch does not match ExpandedCorrelator coordinates")
    problem = inputs.problem
    if not isinstance(problem, ContractionProblem):
        raise TypeError("direct inputs must contain a ContractionProblem")
    has_coordinate_batch_execution = False
    for action in path.plan.state.actions:
        if action.kind != "sparse_bridge" or action.reused:
            continue
        for bridge in action_bridges(path.plan.state, action):
            if bridge is None:
                continue
            source = problem.sources[bridge.record]
            if (isinstance(source.metadata, Mapping)
                    and source.metadata.get("family") == "p2p-coordinate-batch"):
                has_coordinate_batch_execution = True
                break
        if has_coordinate_batch_execution:
            break
    requested_backend = backend if isinstance(backend, str) else getattr(backend, "__name__", None)
    if has_coordinate_batch_execution and (
        problem.backend != "numpy" or requested_backend != "numpy"
    ):
        # Do not import/initialize an unsupported CuPy backend merely to reject
        # this NumPy-only batch execution later in the admission path.
        raise NotImplementedError(
            "coordinate-batched direct execution currently requires NumPy"
        )
    if len(problem.terms) != len(inputs.term_ids):
        raise ValueError("direct term IDs do not match the ContractionProblem")
    backend_name, module = _resolve_backend(backend)
    if problem.backend != backend_name:
        raise ValueError("execution backend disagrees with the ContractionProblem")
    validate_coordinate_batch_workspace(plan, module)
    if debug_enabled():
        validate_executable_resources(plan)
    if not plan.feasible:
        raise MemoryError("direct execution plan exceeds its X workspace admission budget")

    required = _required_keys(problem)
    loader = _validate_input_catalog(inputs, problem, required)
    loader = _bind_loader_for_plan(loader, plan, module)
    state = plan.state
    if not state.done:
        raise ValueError("direct execution state does not contain completed output Terms")
    if dict(plan.blocks) != dict(inputs.blocks):
        raise ValueError("direct execution block catalog differs from its storage audit")
    if dict(plan.preparations) != dict(inputs.preparations):
        raise ValueError("direct execution source preparations differ from its storage audit")

    load_action_indices = {
        action.tensor: index
        for index, action in enumerate(state.actions)
        if action.kind == "load"
    }
    if set(required) - load_action_indices.keys():
        raise RuntimeError("fixed direct state is missing a required load action")

    measurement = ExecutionMeasurement(module) if measure else None
    if measurement is not None:
        measurement.__enter__()
    process = None
    if measure:
        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            pass
    high_water = {"device": None, "host": None}

    def sample_memory():
        if backend_name == "cupy":
            module.cuda.get_current_stream().synchronize()
            allocated = int(module.get_default_memory_pool().total_bytes())
            high_water["device"] = max(high_water["device"] or 0, allocated)
        if process is not None:
            resident = int(process.memory_info().rss)
            high_water["host"] = max(high_water["host"] or 0, resident)

    phase_map = defaultdict(list) if measure else None
    timed_for_action = _timed_for_action(measurement, measure, phase_map)
    io_events = [] if measure else None
    cleanup_timings = [] if measure else None
    opened = []
    primary_error = None
    primary_traceback = None
    stream_synchronized = False
    hook_enabled = False
    measurement_hook = getattr(loader, "set_measurement", None) if measure else None
    hook_error = None
    measurement_exit_error = None
    close_errors = []
    sync_error = None
    elapsed_seconds = None
    values = []
    output_count = 0
    cleanup_action_index = len(state.actions)
    try:
        if measurement_hook is not None:
            if not callable(measurement_hook):
                raise TypeError("loader.set_measurement must be callable when present")
            measurement_hook(True)
            hook_enabled = True
        if measure:
            sample_memory()
        start = perf_counter() if measure else None
        arrays = _read_and_materialize(
            inputs,
            problem,
            required,
            module,
            opened,
            loader=loader,
            timed_for_action=timed_for_action,
            load_action_indices=load_action_indices,
            io_events=io_events,
        )
        if measure:
            sample_memory()
        _execute_actions(state, arrays, module, timed_for_action=timed_for_action,
                         sample_memory=sample_memory if measure else None)
        output_axes = inputs.batch_output_axes or tuple(None for _ in inputs.term_ids)
        if len(output_axes) != len(inputs.term_ids):
            raise ValueError("batch_output_axes must align with direct term IDs")
        output_timings = []
        output_index = 0
        is_gpu = backend_name == "cupy"
        for term_id, result_key, output_axis in zip(
                inputs.term_ids, state.results, output_axes, strict=True):
            if result_key is None or result_key not in arrays:
                raise RuntimeError(f"direct execution did not produce term {term_id!r}")
            result = arrays[result_key]
            if output_axis is None:
                for point in batch_points:
                    action_index = len(state.actions) + output_index
                    output_count = output_index + 1
                    timed = timed_for_action(action_index)
                    copied = copy_output(result_key, {result_key: result}, module,
                                         timed=timed, device=is_gpu)
                    values.append(TermValue(
                        term_id, point.key, copied,
                    ))
                    if measure:
                        phases = tuple(phase_map[action_index])
                        output_timings.append(ActionTiming(
                            "output_copy", 0., sum(p.observed_seconds for p in phases),
                            action_index, result_key, phases,
                        ))
                    output_index += 1
                    if measure:
                        sample_memory()
            else:
                if (output_axis >= result.ndim
                        or result.shape[output_axis] != len(batch_points)):
                    raise ValueError(
                        f"direct result batch axis for {term_id!r} does not match its CoordinateBatch"
                    )
                for lane, point in enumerate(batch_points):
                    action_index = len(state.actions) + output_index
                    output_count = output_index + 1
                    timed = timed_for_action(action_index)
                    lane_value = select_output_lane(
                        result, module, lane, output_axis, timed=timed, device=is_gpu,
                    )
                    copied = copy_output(
                        result_key, {result_key: lane_value}, module, timed=timed, device=is_gpu,
                    )
                    values.append(TermValue(
                        term_id, point.key, copied,
                    ))
                    if measure:
                        phases = tuple(phase_map[action_index])
                        output_timings.append(ActionTiming(
                            "output_copy", 0., sum(p.observed_seconds for p in phases),
                            action_index, result_key, phases,
                        ))
                    output_index += 1
                    if measure:
                        sample_memory()
        output_count = output_index
        cleanup_action_index = len(state.actions) + output_count
        if backend_name == "cupy":
            # The block loader may lend storage to materialized device arrays.
            # Drain only the current stream before releasing those blocks.
            if measure:
                _, phase = measurement.measure_phase(
                    "resource_stream_sync", lambda: None, device=True,
                    parent_action_index=cleanup_action_index,
                )
                cleanup_timings.append(phase)
            else:
                module.cuda.get_current_stream().synchronize()
            stream_synchronized = True
        if measure:
            elapsed_seconds = perf_counter() - start
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
        raise
    finally:
        cleanup_action_index = len(state.actions) + output_count
        if backend_name == "cupy" and not stream_synchronized:
            try:
                if measure:
                    _, phase = measurement.measure_phase(
                        "cleanup_stream_sync", lambda: None, device=True,
                        parent_action_index=cleanup_action_index,
                    )
                    cleanup_timings.append(phase)
                else:
                    module.cuda.get_current_stream().synchronize()
            except Exception as exc:
                sync_error = exc
        close = getattr(loader, "close_block", None)
        if close is not None:
            for block_id, value in reversed(opened):
                try:
                    if measure:
                        block = inputs.blocks[block_id]
                        _, phase = measurement.measure_phase(
                            "close_block",
                            partial(close, block_id, value),
                            nbytes=block.read_bytes,
                            device=bool(block.device_bytes),
                            parent_action_index=cleanup_action_index,
                        )
                        cleanup_timings.append(phase)
                    else:
                        close(block_id, value)
                except Exception as exc:
                    close_errors.append(exc)
        close_all = getattr(loader, "close_all", None)
        if close_all is not None:
            try:
                close_all()
            except Exception as exc:
                close_errors.append(exc)
        if hook_enabled:
            try:
                measurement_hook(False)
            except Exception as exc:
                hook_error = exc
        if measurement is not None:
            if primary_error is not None:
                for label, error in (("direct stream cleanup", sync_error),
                                     ("direct block cleanup", close_errors[0] if close_errors else None),
                                     ("loader measurement reset", hook_error)):
                    if error is not None:
                        measurement.result.setdefault("sample_errors", []).append(
                            f"{label}: {error!r}"
                        )
            try:
                measurement.__exit__(
                    type(primary_error) if primary_error is not None else None,
                    primary_error,
                    primary_traceback,
                )
            except BaseException as exc:
                measurement_exit_error = exc
                if primary_error is not None:
                    measurement.result.setdefault("sample_errors", []).append(
                        f"measurement collector cleanup: {exc!r}"
                    )
        if primary_error is None:
            cleanup_error = sync_error
            if cleanup_error is None and close_errors:
                cleanup_error = close_errors[0]
            if cleanup_error is None:
                cleanup_error = hook_error or measurement_exit_error
            if cleanup_error is not None:
                raise cleanup_error

    report = None
    if measure:
        action_timings = []
        for action_index, action in enumerate(state.actions):
            phases = tuple(phase_map[action_index])
            source = problem.sources.get(action.tensor) if action.kind == "load" else None
            model_features = (
                {"sparse_operation": "bounded_coo_join"}
                if action.join_id == "bounded_coo_join" else None
            )
            action_timings.append(ActionTiming(
                action.kind,
                0.,
                sum(phase.observed_seconds for phase in phases),
                action_index,
                action.tensor,
                phases,
                action.operands,
                source.region if source is not None else None,
                model_features,
            ))
        measurement.result["calibration_eligible"] = False
        host_delta = (None if high_water["host"] is None
                      else high_water["host"] - plan.peak_host_bytes)
        device_delta = (None if high_water["device"] is None
                        else high_water["device"] - plan.peak_device_bytes)
        profile_details = dict(measurement.result)
        profile_details.update({
            "planned_peak_host_bytes": plan.peak_host_bytes,
            "planned_peak_device_bytes": plan.peak_device_bytes,
            "observed_minus_planned_host_bytes": host_delta,
            "observed_minus_planned_device_bytes": device_delta,
            "coordinate_batch_workspace_reserve_host_bytes": 0,
            "observed_storage_budget_exceeded": any(
                value is not None and value > 0 for value in (host_delta, device_delta)
            ),
        })
        measurement_report = ReplayReport(
            tuple(item.value for item in values),
            tuple(action_timings) + ((ActionTiming(
                "release", 0., sum(p.observed_seconds for p in cleanup_timings),
                cleanup_action_index, phases=tuple(cleanup_timings),
            ),) if cleanup_timings else ()),
            high_water["device"] if is_gpu else None,
            high_water["host"],
            tuple(output_timings),
            tuple(io_events),
            profile_details,
        )
        report = {
            "execution_mode": "direct",
            "backend": backend_name,
            "term_count": len(inputs.term_ids),
            "coordinate_count": len(batch_points),
            "seconds": elapsed_seconds,
            "predicted_seconds": None,
            "measurement_report": measurement_report,
            "calibration_eligible": False,
        }
    return DirectExecution(tuple(values), report)


__all__ = ["DirectExecution", "execute_direct_contraction"]
