"""Choose and audit one contraction path using the existing scene planners."""

from dataclasses import replace
from lqcdweave.contracts.correlator.api import (
    validate_executable_resources,
    ContractionProblem,
    ExpandedCorrelator,
    PlannedContraction,
    SceneStoragePlan,
    SourcePreparation,
    StorageBlock,
)
from time import perf_counter

from ..performance.api import CostProfile, predict_plan_cost
from .optimizers.session import solve_storage
from .paths import rebind_scene_path
from .storage import (
    join_storage_batches, logical_cost_profile, logical_source_problem,
    plan_scene_storage,
)
from .state import ContractionState, _outer_product_axes
from .logical_index import LayoutIndex, reuse_indexes




class PathSearchExhausted(RuntimeError):
    """A bounded search did not find a feasible complete path; infeasibility is unproved."""


class NoLegalContractionPath(RuntimeError):
    """The selected exhaustive optimizer proved that no legal path exists."""


def plan_direct_execution(expanded, *, memory_limit, host_memory_limit=None,
                          profile=None, reserved_host_bytes=0,
                          reserved_device_bytes=0) -> PlannedContraction:
    """Build a fixed direct path and audit its complete physical memory plan."""
    if not isinstance(expanded, ExpandedCorrelator):
        raise TypeError("expanded must be an ExpandedCorrelator")
    if not isinstance(profile, (CostProfile, type(None))):
        raise TypeError("profile must be a CostProfile or None")
    _validate_budgets(memory_limit, host_memory_limit, reserved_host_bytes,
                      reserved_device_bytes, require_device=True)
    inputs = expanded.inputs
    if inputs is None or expanded.problem is None:
        raise ValueError("direct execution requires a nonempty expanded contraction")
    if inputs.task_batches:
        return _plan_direct_task_batches(
            expanded, memory_limit=memory_limit, host_memory_limit=host_memory_limit,
            profile=profile, reserved_host_bytes=reserved_host_bytes,
            reserved_device_bytes=reserved_device_bytes,
        )
    problem = inputs.problem
    if not isinstance(problem, ContractionProblem):
        raise TypeError("direct inputs must contain a ContractionProblem")
    if len(problem.terms) != len(inputs.term_ids):
        raise ValueError("direct term IDs do not match the ContractionProblem")
    loader = inputs.loader
    if not callable(getattr(loader, "read_block", None)):
        raise NotImplementedError("direct execution requires loader.read_block")

    profile = CostProfile() if profile is None else profile
    reserved_host_bytes += inputs.assembly_host_bytes
    reserved_device_bytes += inputs.assembly_device_bytes
    if problem.backend == "numpy" and reserved_device_bytes:
        raise ValueError("CPU plans cannot reserve device memory")
    if problem.meson_cores:
        started = perf_counter()
        from .meson_core import plan_meson_whole
        plan = plan_meson_whole(problem, inputs.blocks, inputs.preparations,
            profile=profile, memory_limit=memory_limit, host_memory_limit=host_memory_limit,
            reserved_host_bytes=reserved_host_bytes, reserved_device_bytes=reserved_device_bytes)
        prediction = predict_plan_cost((plan,), profile, workload={}, provenance={})
        prediction.update(execution_mode='whole-meson', planner=_planner_metadata('none', 'bounded', {}))
        # No search was performed, but the whole-kernel plan owns interleaved
        # loads/releases. Consume it with fixed scheduled replay, not the
        # legacy direct executor that materializes every Term input at once.
        return PlannedContraction(expanded, plan, None, prediction, perf_counter() - started,
            'planned', memory_limit, host_memory_limit, id(plan), profile,
            reserved_host_bytes, reserved_device_bytes)
    required = _direct_required_sources(problem)
    preparations, blocks = inputs.preparations, inputs.blocks
    for key in required:
        if key not in problem.sources:
            raise ValueError(f"direct input {key!r} has no SourceSpec")
        if key not in preparations:
            raise ValueError(f"direct input {key!r} has no SourcePreparation")
        preparation = preparations[key]
        if not isinstance(preparation, SourcePreparation):
            raise TypeError(f"direct preparation for {key!r} must be SourcePreparation")
        for block_key in preparation.blocks:
            if block_key not in blocks:
                raise ValueError(f"direct input {key!r} refers to missing block {block_key!r}")
            if not isinstance(blocks[block_key], StorageBlock):
                raise TypeError(f"direct block {block_key!r} must be StorageBlock")
    if (any(preparations[key].recipe is None for key in required)
            and not callable(getattr(loader, "materialize", None))):
        raise NotImplementedError(
            "opaque direct inputs require loader.materialize"
        )

    logical_problem = logical_source_problem(problem)
    logical_profile = logical_cost_profile(profile)
    started = perf_counter()
    state = ContractionState.initial(logical_problem)
    for key in required:
        state = state.load(key, logical_profile, 1 << 100)
    match_index = LayoutIndex(state) if reuse_indexes(state.pending) else None
    for term_index in range(len(problem.terms)):
        while state.pending[term_index] is not None:
            factors = state.pending[term_index].factors
            pair = _matching_direct_bridge_pair(problem, term_index, factors)
            if pair is None:
                join = next((join for join in problem.baryon_joins
                             if join.scene == term_index
                             and (join.scene, join.name) not in state.completed_joins), None)
                if join is not None:
                    state = state.contract_join(
                        join, logical_profile, 1 << 100, search_kernel_path=False,
                    )
                    if match_index is not None:
                        match_index.advance(state, (term_index,))
                    continue
                if len(factors) == 1:
                    pair = (0,)
                elif len(factors) >= 2:
                    pair = _first_direct_shared_label_pair(state.pending[term_index])
                else:
                    raise NotImplementedError("direct execution does not support zero-input terms")
            if match_index is None:
                state = state.contract(term_index, pair, logical_profile, 1 << 100)
            else:
                changes = []
                layout = match_index.layout(term_index, pair)
                state = state._contract_from_layout(
                    term_index, pair, layout, logical_profile, 1 << 100,
                    layout_index=match_index, changes=changes,
                )
                match_index.advance(state, (index for index, _, _ in changes))
    if not state.done:
        raise RuntimeError("fixed direct contraction sequence did not complete every term")

    plan = plan_scene_storage(
        state, blocks, preparations, profile=profile,
        memory_limit=memory_limit, host_memory_limit=host_memory_limit,
        reserved_host_bytes=reserved_host_bytes,
        reserved_device_bytes=reserved_device_bytes,
    )
    if not plan.feasible:
        raise MemoryError(
            "fixed direct path exceeds the audited physical storage budget: "
            f"host peak={plan.peak_host_bytes}, device peak={plan.peak_device_bytes}"
        )
    prediction = predict_plan_cost((plan,), profile, workload={}, provenance={})
    prediction.update({
        "execution_mode": "direct",
        "profile_version": profile.version,
        "hardware_fingerprint": profile.hardware_fingerprint,
        "memory_limit": memory_limit,
        "host_memory_limit": host_memory_limit,
        "planner": _planner_metadata("none", "bounded", {}),
    })
    result = PlannedContraction(
        expanded, plan, None, prediction, perf_counter() - started, "direct",
        memory_limit, host_memory_limit, id(plan), profile,
        reserved_host_bytes, reserved_device_bytes,
    )
    validate_execution_binding(expanded, result)
    return result


def _direct_required_sources(problem):
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


def _plan_direct_task_batches(expanded, *, memory_limit, host_memory_limit,
                              profile, reserved_host_bytes,
                              reserved_device_bytes):
    """Build fixed direct paths, then audit their shared raw lifetimes together."""
    inputs = expanded.inputs
    batches = tuple(inputs.task_batches)
    if not batches:
        raise ValueError("direct task batch planning requires at least one batch")
    profile = CostProfile() if profile is None else profile
    started = perf_counter()
    child_paths = []
    for batch_inputs in batches:
        batch_inputs = replace(
            batch_inputs,
            assembly_host_bytes=max(batch_inputs.assembly_host_bytes, inputs.assembly_host_bytes),
            assembly_device_bytes=max(batch_inputs.assembly_device_bytes, inputs.assembly_device_bytes),
        )
        child = replace(expanded, inputs=batch_inputs)
        child_path = plan_direct_execution(
            child, memory_limit=memory_limit, host_memory_limit=host_memory_limit,
            profile=profile, reserved_host_bytes=reserved_host_bytes,
            reserved_device_bytes=reserved_device_bytes,
        )
        child_paths.append(child_path)
    joined = join_storage_batches(tuple(item.plan for item in child_paths))
    if any(not item.feasible for item in joined):
        first_bad = next(index for index, item in enumerate(joined) if not item.feasible)
        raise MemoryError(
            f"full-task direct raw lifetimes exceed the memory budget at batch {first_bad}"
        )
    execution_mode = ('planned' if any(item.execution_mode == 'planned' for item in child_paths)
                      else 'direct')
    joined_paths = tuple(
        replace(child_path, plan=storage_plan, plan_identity=id(storage_plan), execution_mode=execution_mode)
        for child_path, storage_plan in zip(child_paths, joined, strict=True)
    )
    prediction = predict_plan_cost(joined, profile, workload={}, provenance={})
    prediction.update({
        "execution_mode": execution_mode,
        "profile_version": profile.version,
        "hardware_fingerprint": profile.hardware_fingerprint,
        "memory_limit": memory_limit,
        "host_memory_limit": host_memory_limit,
        # The reusable signature contains OutputNode recipes and complex
        # coefficients. Keep it in batch_shape_keys; predictions are strict
        # JSON reports, so describe only their numerical shapes here.
        "batch_shapes": tuple(dict(
            coordinate_count=item.coordinate_batch.actual_size,
            backend=item.problem.backend,
            tensors=tuple((spec.shape, spec.dtype.str) for spec in item.problem.specs.values()),
            term_count=len(item.problem.terms),
        ) for item in batches),
        "planner": _planner_metadata("none", "bounded", {}),
    })
    first = joined_paths[0]
    result = PlannedContraction(
        expanded, first.plan, None, prediction, perf_counter() - started, execution_mode,
        memory_limit, host_memory_limit, id(first.plan), profile,
        first.reserved_host_bytes, first.reserved_device_bytes,
        batch_paths=joined_paths,
        batch_shape_keys=tuple(_batch_shape_signature(item) for item in batches),
    )
    validate_execution_binding(expanded, result)
    return result


def _matching_direct_bridge_pair(problem, term_index, factors):
    for bridge in problem.sparse_bridges:
        if bridge.scene not in (None, term_index):
            continue
        left_positions = [i for i, factor in enumerate(factors) if factor.tensor == bridge.left]
        right_positions = [i for i, factor in enumerate(factors) if factor.tensor == bridge.right]
        if bridge.left == bridge.right:
            if len(left_positions) >= 2:
                return left_positions[0], left_positions[1]
            continue
        ordered = [(left, right) for left in left_positions for right in right_positions if left < right]
        if ordered:
            return ordered[0]
        if any(right < left for left in left_positions for right in right_positions):
            raise NotImplementedError(
                "direct P2P bridging requires term operand order to match SparseBridgeSpec.left/right"
            )
    return None


def _first_direct_shared_label_pair(term):
    """Prefer a summed shared axis, then the first legal non-outer pair."""
    factors = term.factors
    fallback = None
    for left in range(len(factors)):
        for right in range(left + 1, len(factors)):
            first, second = set(factors[left].labels), set(factors[right].labels)
            common = first & second
            remaining = set(term.output)
            for index, factor in enumerate(factors):
                if index not in (left, right):
                    remaining.update(factor.labels)
            if _outer_product_axes(first, second, remaining):
                continue
            if common - remaining:
                return left, right
            if fallback is None:
                fallback = left, right
    if fallback is None:
        raise ValueError("no contraction pair without an outer product")
    return fallback


def validate_execution_binding(expanded, path) -> None:
    """Validate binding and supported actions before any execution IO."""
    _validate_binding(expanded, path, executable=True)


def _validate_binding(expanded, path, *, executable):
    if not isinstance(expanded, ExpandedCorrelator):
        raise TypeError("expanded must be an ExpandedCorrelator")
    if not isinstance(path, PlannedContraction):
        raise TypeError("path must be a PlannedContraction")
    if not isinstance(path.expanded, ExpandedCorrelator):
        raise TypeError("path descriptor does not contain an ExpandedCorrelator")
    if path.expanded.identity_token is not expanded.identity_token:
        raise ValueError("path was selected for a different ExpandedCorrelator")
    if path.execution_mode == "empty":
        if expanded.problem is not None or path.plan is not None:
            raise ValueError("empty plan disagrees with the expanded correlator")
        return
    if path.execution_mode not in ("direct", "planned"):
        raise ValueError(f"unsupported execution mode {path.execution_mode!r}")
    if not isinstance(path.plan, SceneStoragePlan):
        raise ValueError("a nonempty path needs an audited SceneStoragePlan")
    if path.plan_identity != id(path.plan):
        raise ValueError("plan object no longer matches the selected path descriptor")
    if not path.plan.feasible:
        raise MemoryError("the stored path is not feasible under its audited memory limits")
    if (path.memory_limit != path.plan.memory_limit
            or path.host_memory_limit != path.plan.host_memory_limit
            or path.reserved_host_bytes != path.plan.reserved_host_bytes
            or path.reserved_device_bytes != path.plan.reserved_device_bytes):
        raise ValueError("path budget metadata differs from the audited storage plan")
    if path.profile is not None and path.plan.cost_profile != path.profile:
        raise ValueError("path profile differs from the audited storage plan")
    inputs = path.expanded.inputs
    if inputs is None or inputs.problem is None:
        raise ValueError("nonempty plan has no matching DiagramInputs")
    if inputs.task_batches:
        batches = tuple(inputs.task_batches)
        batch_paths = tuple(path.batch_paths)
        if len(batch_paths) != len(batches) or not batch_paths:
            raise ValueError("task batch plans do not cover each declared batch")
        if path.plan is not batch_paths[0].plan:
            raise ValueError("parent plan must identify the first globally joined batch plan")
        for index, (batch_inputs, batch_path) in enumerate(zip(batches, batch_paths, strict=True)):
            if batch_path.execution_mode != path.execution_mode:
                raise ValueError(f"batch {index} execution mode differs from the parent path")
            if batch_path.expanded.identity_token is not expanded.identity_token:
                raise ValueError(f"batch {index} belongs to a different expansion")
            _validate_one_plan(batch_inputs, batch_path, executable=executable)
        _validate_global_storage_events(batch_paths)
    else:
        if path.batch_paths:
            raise ValueError("path contains batch plans without task_batches")
        _validate_one_plan(inputs, path, executable=executable)
        if not path.plan.explicit_resources:
            _validate_global_storage_events((path,))


def _validate_one_plan(inputs, path, *, executable):
    if executable:
        validate_executable_resources(path.plan)
    if path.plan_identity != id(path.plan) or not path.plan.feasible:
        raise ValueError("batch plan identity or feasibility is invalid")
    if (path.memory_limit != path.plan.memory_limit
            or path.host_memory_limit != path.plan.host_memory_limit
            or path.reserved_host_bytes != path.plan.reserved_host_bytes
            or path.reserved_device_bytes != path.plan.reserved_device_bytes):
        raise ValueError("batch path budget metadata differs from its storage plan")
    if path.profile is not None and path.plan.cost_profile != path.profile:
        raise ValueError("batch profile differs from its storage plan")
    if len(inputs.term_ids) != len(path.plan.state.problem.terms):
        raise ValueError("plan term count differs from its bound term IDs")
    if not path.plan.state.done:
        raise ValueError("storage plan does not contain a complete contraction state")
    logical = logical_source_problem(inputs.problem)
    planned = path.plan.state.problem
    if (dict(logical.specs) != dict(planned.specs)
            or logical.terms != planned.terms
            or logical.backend != planned.backend
            or logical.sparse_bridges != planned.sparse_bridges
            or logical.baryon_joins != planned.baryon_joins
            or logical.meson_cores != planned.meson_cores
            or dict(logical.sources) != dict(planned.sources)):
        raise ValueError("plan problem differs from its expanded physical inputs")
    if path.execution_mode == "direct" and (
            path.plan.retained_before or path.plan.retained_after or path.plan.retained_releases):
        raise ValueError("direct plans cannot claim cross-batch logical tensor retention")


def _validate_global_storage_events(paths):
    expected_blocks = set()
    reads, releases = {}, {}
    for path in paths:
        inputs = path.expanded.inputs
        for action in path.plan.state.actions:
            if action.kind == "load":
                if action.tensor not in inputs.preparations:
                    raise ValueError(f"plan loads unknown physical source {action.tensor!r}")
                expected_blocks.update(inputs.preparations[action.tensor].blocks)
        for event in path.plan.events:
            if event.kind == "read":
                reads[event.block] = reads.get(event.block, 0) + 1
            elif event.kind == "release":
                releases[event.block] = releases.get(event.block, 0) + 1
            else:
                raise ValueError(f"unknown storage event kind {event.kind!r}")
    if any(block not in path.plan.blocks for path in paths for block in expected_blocks):
        raise ValueError("storage plan omits a required physical block")
    if set(reads) != expected_blocks or set(releases) != expected_blocks:
        raise ValueError("storage schedule does not cover every required physical block")
    if any(count != 1 for count in (*reads.values(), *releases.values())):
        raise ValueError("global storage schedule would repeat a physical read or release")


def find_contraction_path(expanded, *, memory_limit, optimize="anytime",
                          storage=None, profile=None, search=None, kernel_batching=False,
                          host_memory_limit=None, reserved_host_bytes=0,
                          reserved_device_bytes=0):
    """Search once, then audit the selected path against physical storage."""
    if type(kernel_batching) is not bool:
        raise TypeError('kernel_batching must be bool')
    if not isinstance(expanded, ExpandedCorrelator):
        raise TypeError("expanded must be an ExpandedCorrelator")
    if optimize == 'meson_matrix':
        optimize = 'anytime'
    if expanded.inputs is not None and expanded.inputs.matrix_kernels:
        expanded = replace(expanded, inputs=replace(expanded.inputs, matrix_kernels=()))
    options = dict(search or {})
    forbidden = {"problem", "profile", "memory_limit", "host_memory_limit", "optimizer",
                 "reserved_host_bytes", "reserved_device_bytes", "kernel_batching"}
    if options.keys() & forbidden:
        raise ValueError("search must contain search controls only")
    if profile is not None and not isinstance(profile, CostProfile):
        raise TypeError("low-level planning accepts CostProfile or None; resolve profile names and paths in the contraction frontend")

    if optimize != "none":
        if optimize not in ("anytime", "exact", "dynamic_programming"):
            raise ValueError("unknown unified optimizer")
        if storage not in (None, "bounded", "lifetime"):
            raise ValueError("storage must be 'bounded' or 'lifetime'")
        storage = "lifetime"
        from .optimizers.options import effective_search_options
        effective_search_options(optimize, options)
        if optimize in ("exact", "dynamic_programming") and expanded.inputs is not None and (
            expanded.inputs.batch_candidates or expanded.inputs.task_batches
        ):
            raise NotImplementedError("global exact storage planning requires one complete problem; task/auto batches are not yet supported")

    if optimize == "none" and search:
        raise ValueError("search controls cannot be used with optimize='none'")

    # A task with no contraction Terms is a no-op for every optimization mode.
    if expanded.problem is None:
        return PlannedContraction(expanded, None, None,
                                  {"predicted_seconds": 0.0, "operations": 0, "kernel_batching": kernel_batching}, 0.0,
                                  "empty", memory_limit, host_memory_limit, None)

    if optimize == "none":
        if search:
            raise ValueError("search controls cannot be used with optimize='none'")
        if storage not in (None, "bounded"):
            raise ValueError("direct mode supports storage=None or 'bounded'")
        if expanded.inputs is not None and expanded.inputs.batch_candidates:
            return _choose_auto_batch_candidate(
                expanded, memory_limit=memory_limit, optimize=optimize,
                storage=storage, profile=profile, search=search,
                kernel_batching=kernel_batching,
                host_memory_limit=host_memory_limit,
                reserved_host_bytes=reserved_host_bytes,
                reserved_device_bytes=reserved_device_bytes,
            )
        path = plan_direct_execution(
            expanded, memory_limit=memory_limit,
            host_memory_limit=host_memory_limit, profile=profile,
            reserved_host_bytes=reserved_host_bytes,
            reserved_device_bytes=reserved_device_bytes,
        )
        return replace(path, prediction=dict(path.prediction, kernel_batching=kernel_batching))

    if expanded.inputs is not None and expanded.inputs.batch_candidates:
        return _choose_auto_batch_candidate(
            expanded, memory_limit=memory_limit, optimize=optimize,
            storage=storage, profile=profile, search=search,
            kernel_batching=kernel_batching,
            host_memory_limit=host_memory_limit,
            reserved_host_bytes=reserved_host_bytes,
            reserved_device_bytes=reserved_device_bytes,
        )
    if expanded.inputs is not None and expanded.inputs.task_batches:
        return _find_task_batch_paths(
            expanded, memory_limit=memory_limit, optimize=optimize,
            storage=storage, profile=profile, search=search,
            kernel_batching=kernel_batching,
            host_memory_limit=host_memory_limit,
            reserved_host_bytes=reserved_host_bytes,
            reserved_device_bytes=reserved_device_bytes,
        )
    if storage is None:
        storage = "bounded"
    if storage not in ("bounded", "lifetime"):
        raise ValueError("storage must be 'bounded' or 'lifetime'")
    _validate_budgets(memory_limit, host_memory_limit,
                      reserved_host_bytes, reserved_device_bytes, require_device=True)

    inputs = expanded.inputs
    problem = inputs.problem
    reserved_host_bytes += inputs.assembly_host_bytes
    reserved_device_bytes += inputs.assembly_device_bytes
    if problem.backend == "numpy" and reserved_device_bytes:
        raise ValueError("CPU plans cannot reserve device memory")
    profile = CostProfile() if profile is None else profile
    budgets = dict(memory_limit=memory_limit, host_memory_limit=host_memory_limit,
                   reserved_host_bytes=reserved_host_bytes,
                   reserved_device_bytes=reserved_device_bytes)
    started = perf_counter()
    if problem.meson_cores:
        from .meson_core import plan_meson_whole
        result = None
        plan = plan_meson_whole(problem, inputs.blocks, inputs.preparations,
            profile=profile, **budgets)
    else:
        result = solve_storage(problem, inputs.blocks, inputs.preparations,
                               optimizer=optimize, profile=profile, kernel_batching=kernel_batching,
                               **budgets, **options)
        plan = result.plan
    if plan is None:
        if result.status == "INFEASIBLE":
            raise NoLegalContractionPath(
                "optimizer proved that no complete path fits the supplied constraints"
            )
        raise PathSearchExhausted(
            f"search did not find a complete feasible path: status={result.status}; "
            "this does not prove infeasibility"
        )
    if not plan.feasible:
        raise MemoryError("chosen path exceeds physical storage budget; no input data was read")
    prediction = predict_plan_cost((plan,), profile, workload={}, provenance={})
    prediction["planner"] = (_planner_metadata(optimize, storage, options) if not problem.meson_cores else
        dict(optimizer=optimize, search_performed=False, storage='lifetime',
             path_policy='closed whole-meson joins; original component arithmetic is local',
             private_paths_searched=False))
    prediction['kernel_batching'] = kernel_batching
    path = PlannedContraction(expanded, plan, result, prediction,
                              perf_counter() - started, "planned",
                              memory_limit, host_memory_limit, id(plan),
                              profile, reserved_host_bytes, reserved_device_bytes)
    _validate_binding(expanded, path, executable=False)
    return path


def _best_predicted_batch_width(comparisons):
    """Select the least predicted feasible candidate; smaller width breaks ties."""
    from math import isfinite

    feasible = [item for item in comparisons
                if item.get("feasible") is True
                and isinstance(item.get("predicted_seconds"), (int, float))
                and isfinite(float(item["predicted_seconds"]))]
    if not feasible:
        reasons = "; ".join(
            f"width {item.get('batch_width')}: {item.get('reason', 'no finite prediction')}"
            for item in comparisons
        )
        raise MemoryError(
            "no automatic batch candidate has a finite feasible prediction"
            + (f" ({reasons})" if reasons else "")
        )
    selected = min(feasible, key=lambda item: (
        float(item["predicted_seconds"]), int(item["batch_width"]),
    ))
    return int(selected["batch_width"])


def _choose_auto_batch_candidate(expanded, *, memory_limit, optimize, storage,
                                 profile, search, host_memory_limit,
                                 reserved_host_bytes, reserved_device_bytes, kernel_batching=False):
    """Compare bounded auto widths by CostProfile predictions, reusing shape paths."""
    profile = CostProfile() if profile is None else profile
    inputs = expanded.inputs
    template_cache = {}
    candidate_paths, comparison = {}, []
    total_planning_seconds = 0.0
    for width, task_batches in sorted(inputs.batch_candidates.items()):
        candidate_inputs = replace(
            inputs,
            problem=task_batches[0].problem,
            preparations=task_batches[0].preparations,
            loader=task_batches[0].loader,
            tensor_requests=task_batches[0].tensor_requests,
            task_batches=tuple(task_batches), batch_candidates={},
        )
        candidate_expanded = replace(expanded, inputs=candidate_inputs)
        try:
            if optimize == "none":
                path = find_contraction_path(
                    candidate_expanded, memory_limit=memory_limit,
                    optimize=optimize, storage=storage, profile=profile,
                    search=search, host_memory_limit=host_memory_limit,
                    kernel_batching=kernel_batching,
                    reserved_host_bytes=reserved_host_bytes,
                    reserved_device_bytes=reserved_device_bytes,
                )
                predicted = float(path.prediction["total_seconds"])
                prediction = dict(path.prediction, predicted_seconds=predicted,
                                  profile_version=profile.version,
                                  calibration_status=_profile_calibration(profile))
                path = replace(path, prediction=prediction)
            else:
                path = _find_task_batch_paths(
                    candidate_expanded, memory_limit=memory_limit,
                    optimize=optimize, storage=storage, profile=profile,
                    search=search, host_memory_limit=host_memory_limit,
                    kernel_batching=kernel_batching,
                    reserved_host_bytes=reserved_host_bytes,
                    reserved_device_bytes=reserved_device_bytes,
                    template_cache=template_cache,
                )
                predicted = (path.prediction["total_seconds"]
                             + _estimate_assembly_seconds(candidate_inputs, profile))
            total_planning_seconds += path.planning_seconds
            comparison.append({
                "batch_width": width,
                "batch_sizes": tuple(item.coordinate_batch.actual_size
                                      for item in task_batches),
                "predicted_seconds": float(predicted),
                "feasible": True,
                "planning_seconds": path.planning_seconds,
                "peak_host_bytes": path.prediction.get("peak_host_bytes"),
                "peak_device_bytes": path.prediction.get("peak_device_bytes"),
                "physical_reads": path.prediction.get("physical_reads"),
            })
            candidate_paths[width] = path
        except MemoryError as exc:
            comparison.append({
                "batch_width": width,
                "batch_sizes": tuple(item.coordinate_batch.actual_size
                                      for item in task_batches),
                "predicted_seconds": None,
                "feasible": False,
                "reason": str(exc),
            })
    selected_width = _best_predicted_batch_width(comparison)
    selected = candidate_paths[selected_width]
    selected_prediction = next(item["predicted_seconds"] for item in comparison
                               if item["batch_width"] == selected_width)
    selection = {
        "selection": "minimum CostProfile-predicted task time among feasible finite widths",
        "selected_width": selected_width,
        "profile_version": profile.version,
        "hardware_fingerprint": profile.hardware_fingerprint,
        "sample_count": profile.sample_count,
        "calibration_status": _profile_calibration(profile),
        "batch_width_calibration": "not modeled; uses scalar per-operation profile",
        "candidate_predictions": tuple(comparison),
        "search_seconds_included_in_objective": False,
        "selected_predicted_seconds": selected_prediction,
        "shape_searches_reused_across_width_candidates": optimize != "none",
    }
    prediction = dict(selected.prediction)
    prediction["batch_selection"] = selection
    if optimize == "none":
        prediction["predicted_seconds"] = comparison[
            next(index for index, item in enumerate(comparison)
                 if item["batch_width"] == selected_width)
        ]["predicted_seconds"]
        searches = selected.search
    else:
        searches = tuple(cached["path"].search for cached in template_cache.values())
    selected_path = replace(
        selected, search=searches, prediction=prediction,
        planning_seconds=total_planning_seconds,
    )
    validate_execution_binding(expanded, selected_path)
    return selected_path


def _estimate_assembly_seconds(inputs, profile):
    output_bytes = inputs.assembly_host_bytes + inputs.assembly_device_bytes
    return profile.snapshot_seconds(output_bytes)


def _profile_calibration(profile):
    return ("unmeasured" if profile.sample_count == 0
            or profile.hardware_fingerprint == "unmeasured" else "has_samples")


def _planner_metadata(optimize, storage, search):
    """Expose the effective search controls beside each selected plan."""
    if optimize == "none":
        return {
            "optimizer": "none",
            "search_performed": False,
            "storage": "bounded",
            "path_policy": "bridge first; first pair with a shared label eliminated now; non-outer pair fallback",
            "search_controls": {},
        }
    from .optimizers.options import effective_search_options
    controls = effective_search_options(optimize, search)
    return dict(optimizer=optimize, objective="modeled_compute_cost_under_storage_limits", search_performed=True,
                storage="lifetime", search_controls=controls)



def _find_task_batch_paths(expanded, *, memory_limit, optimize, storage, profile,
                           search, host_memory_limit, reserved_host_bytes,
                           reserved_device_bytes, template_cache=None, kernel_batching=False):
    """Search one operation order per batch shape, then rebind and join storage."""
    inputs = expanded.inputs
    batches = tuple(inputs.task_batches)
    if not batches:
        raise ValueError("task batch planning requires at least one compute batch")
    profile = CostProfile() if profile is None else profile
    shape_cache = {} if template_cache is None else template_cache
    used_searches, batch_paths, storage_plans, shape_keys = {}, [], [], []
    started = perf_counter()
    for child_inputs in batches:
        child_expanded = replace(expanded, inputs=child_inputs)
        signature = _batch_shape_signature(child_inputs)
        shape_keys.append(signature)
        cached = shape_cache.get(signature)
        if cached is None:
            path = find_contraction_path(
                child_expanded, memory_limit=memory_limit, optimize=optimize,
                storage=storage, profile=profile, search=search,
                kernel_batching=kernel_batching,
                host_memory_limit=host_memory_limit,
                reserved_host_bytes=reserved_host_bytes,
                reserved_device_bytes=reserved_device_bytes,
            )
            if path.execution_mode != "planned" or path.plan is None:
                raise ValueError("task batches require a complete planned path for each shape")
            template_state = path.plan.state
            if template_state is None:
                raise RuntimeError("path search returned no reusable operation state")
            cached = {"path": path, "template_state": template_state}
            shape_cache[signature] = cached
            batch_path = path
        else:
            template = cached["path"]
            blocks, preparations = child_inputs.blocks, child_inputs.preparations
            problem = child_inputs.problem
            reserved_host = reserved_host_bytes + child_inputs.assembly_host_bytes
            reserved_device = reserved_device_bytes + child_inputs.assembly_device_bytes
            required = {key for prep in preparations.values() for key in prep.blocks}
            host, device = reserved_host, reserved_device
            if storage in (None, "bounded"):
                host += sum(blocks[key].host_bytes for key in required)
                device += sum(blocks[key].device_bytes for key in required)
            headroom = memory_limit - (device if problem.backend == "cupy" else host)
            host_headroom = None if host_memory_limit is None else host_memory_limit - host
            if headroom <= 0 or (host_headroom is not None and host_headroom <= 0):
                raise MemoryError("raw inputs and reserves leave no contraction memory for a reused batch path")
            rebound = rebind_scene_path(
                cached["template_state"], logical_source_problem(problem),
                {key: key for key in problem.sources},
                profile=logical_cost_profile(profile), memory_limit=headroom,
                host_memory_limit=host_headroom,
            )
            if optimize == "anytime" and not problem.meson_cores:
                plan = _replay_macro_order(problem, blocks, preparations, rebound.actions,
                    profile=profile, memory_limit=memory_limit, host_memory_limit=host_memory_limit,
                    reserved_host_bytes=reserved_host, reserved_device_bytes=reserved_device,
                    disk_limit_bytes=dict(search or {}).get("disk_limit_bytes"))
            else:
                plan = plan_scene_storage(
                    rebound, blocks, preparations, profile=profile,
                    memory_limit=memory_limit, host_memory_limit=host_memory_limit,
                    reserved_host_bytes=reserved_host,
                    reserved_device_bytes=reserved_device,
                )
            if not plan.feasible:
                raise MemoryError("rebound batch path exceeds physical storage budget")
            batch_path = replace(
                template, expanded=child_expanded, plan=plan,
                plan_identity=id(plan), planning_seconds=0.0,
            )
        used_searches.setdefault(signature, cached["path"].search)
        batch_paths.append(batch_path)
        storage_plans.append(batch_path.plan)

    if any(plan.explicit_resources for plan in storage_plans):
        raise NotImplementedError("temporary storage across task batches requires a shared resource trajectory")
    joined = join_storage_batches(storage_plans)
    if any(not plan.feasible for plan in joined):
        first_bad = next(index for index, plan in enumerate(joined) if not plan.feasible)
        raise MemoryError(
            f"full-task shared raw lifetimes exceed the memory budget at batch {first_bad}"
        )
    joined_paths = tuple(
        replace(path, plan=plan, plan_identity=id(plan))
        for path, plan in zip(batch_paths, joined, strict=True)
    )
    prediction = predict_plan_cost(joined, profile, workload={}, provenance={})
    prediction["planner"] = _planner_metadata(optimize, storage, dict(search or {}))
    prediction['kernel_batching'] = kernel_batching
    elapsed = perf_counter() - started
    first = joined_paths[0]
    path = PlannedContraction(
        expanded, first.plan, tuple(used_searches.values()), prediction, elapsed, "planned",
        memory_limit, host_memory_limit, id(first.plan), profile,
        first.reserved_host_bytes, first.reserved_device_bytes,
        batch_paths=joined_paths, batch_shape_keys=tuple(shape_keys),
    )
    _validate_binding(expanded, path, executable=False)
    return path



def _replay_macro_order(problem, blocks, preparations, order, **physical):
    from .optimizers.macro_space import MacroSearchSpace
    from .optimizers.delivery import executable_delivery
    space = MacroSearchSpace(problem, blocks, preparations, **physical)
    cursor, actions = space.initial_state(), []
    from .canonical_operations import PREFIX
    computations = tuple(action for action in order if action.kind not in ("load", "release"))
    if computations and all(action.tensor.startswith(PREFIX) for action in computations):
        from .optimizers.storage_schedule import replay_fixed_dag
        # Rebinding already validated the selected logical actions. Keep their
        # order and the DAG pass's final-use lifetime policy for every batch.
        from .canonical_operations import replay
        symbolic = ContractionState.initial(space.problem)
        symbolic = replace(symbolic, loaded=frozenset(symbolic.catalog) | set(symbolic.problem.sources))
        for action in computations:
            symbolic = replay(symbolic, action, space.logical_profile)
        return replay_fixed_dag(symbolic, space)
    for old in order:
        if old.kind in ("load", "release"):
            continue
        if cursor.state.done:
            break
        chosen = next((step for step in space.logical_steps(cursor)
                       if step.actions[0].tensor == old.tensor), None)
        if chosen is None:
            # A shared join can already have completed several scenes.
            if old.reused:
                continue
            raise RuntimeError("reused compute order does not match the rebound problem")
        try:
            edge = space.evaluate(cursor, chosen)
        except MemoryError as exc:
            raise PathSearchExhausted("reused complete-step order exceeds storage limits") from exc
        cursor = edge.state
        actions.extend(edge.actions)
    if not cursor.state.done:
        raise RuntimeError("reused macro order did not complete every term")
    return executable_delivery(space.plan(cursor, actions))

def _batch_shape_signature(inputs):
    problem = inputs.problem
    return (
        inputs.batch_topology,
        inputs.coordinate_batch.actual_size,
        problem.backend,
        tuple((key, spec.shape, spec.dtype.str) for key, spec in problem.specs.items()),
        tuple((term.inputs, term.subscripts.normalized(), term.coefficient, term.kernel) for term in problem.terms),
        tuple((key, source.snapshot_bytes) for key, source in problem.sources.items()),
    )


def _validate_budgets(memory_limit, host_memory_limit, reserved_host_bytes,
                      reserved_device_bytes, *, require_device):
    if memory_limit is None and not require_device:
        pass
    elif isinstance(memory_limit, bool) or not isinstance(memory_limit, int) or memory_limit <= 0:
        raise ValueError("memory_limit must be a positive integer number of bytes")
    if host_memory_limit is not None and (
            isinstance(host_memory_limit, bool) or not isinstance(host_memory_limit, int)
            or host_memory_limit <= 0):
        raise ValueError("host_memory_limit must be a positive integer or None")
    for value in (reserved_host_bytes, reserved_device_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("memory reserves must be nonnegative integer bytes")


__all__ = [
    "NoLegalContractionPath",
    "PathSearchExhausted",
    "PlannedContraction",
    "find_contraction_path",
    "plan_direct_execution",
    "validate_execution_binding",
]
