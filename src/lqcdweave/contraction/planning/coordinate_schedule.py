"""Bounded coordinate batches of a fixed dense contraction DAG; no search."""
from dataclasses import dataclass, replace
from math import prod
from time import perf_counter

from lqcdweave.contracts.correlator.api import (
    Action, BatchSpec, FactorizedWideReduction, MesonWholeKernel, group_coordinate_batches, DEVICE_ALLOCATION_ALIGNMENT,
)
from lqcdweave.contracts.correlator.cores import TensorCoreSpec
from .coordinate_inputs import bind_coordinate_inputs, describe_coordinate_batch
from .coordinate_path import lift_coordinate_path
from .dense_batch_schedule import lower_dense_batches
from .gemm_groups import lower_independent_kernel_groups
from .storage import plan_scene_storage, join_storage_batches


@dataclass(frozen=True)
class CoordinateReplayPlan:
    batches: tuple
    plans: tuple
    width: int
    statistics: dict
    reader_aliases: tuple


@dataclass(frozen=True)
class CoordinateDeliverySelection:
    """One fixed coordinate delivery choice; only batched mode owns a replay."""
    mode: str
    replay: CoordinateReplayPlan | None
    statistics: dict


def _coordinate_work_candidates(template, binding, coordinates, grouping, profile):
    """Price widths from input identities and one selected DAG, without lifting it."""
    sources = tuple(template.state.problem.sources)
    all_sources = frozenset(sources)
    dependencies = {source: frozenset((source,)) for source in sources}
    work_by_dependency = {}
    for action in template.state.actions:
        if action.kind not in ('contract', 'baryon_join'):
            continue
        inputs = (action.operands + tuple(action.core.inputs)
                  if isinstance(action.core, TensorCoreSpec) else action.operands)
        members = tuple(dependencies.get(operand) for operand in inputs)
        # Unknown ownership cannot certify sharing; conservatively charge
        # the action to every scalar input of the selected template.
        required = (all_sources if any(member is None for member in members)
                    else frozenset().union(*members))
        dependencies[action.tensor] = required
        if action.reused:
            continue
        dtype = template.state.catalog[action.tensor].dtype
        call = min(action.seconds, profile.kernel_call_for_dtype(dtype))
        variable = max(0.0, action.seconds - call)
        entry = work_by_dependency.setdefault(required, [0.0, 0.0, 0])
        entry[0] += variable
        entry[1] += call
        entry[2] += 1

    signatures = {}
    for required in work_by_dependency:
        ordered = tuple(key for key in sources if key in required)
        signatures[required] = tuple(tuple(point[key] for key in ordered)
                                     for point in binding.source_names)

    axis = grouping.axis
    if axis is None and len(coordinates.scan_axes) > 1:
        axis = coordinates.scan_axes[-1]
    if coordinates.mode == 'pairs':
        axis = 'pair'
    largest = max(group.actual_size for group in group_coordinate_batches(
        coordinates, BatchSpec('full', axis)))
    widths = {1, largest}
    power = 1
    while power < largest:
        power *= 2
        widths.add(min(power, largest))
    # Constant source-identity runs expose natural widths such as five equal
    # source times in a flattened 5x5 scan. The grouping service still owns
    # the actual grid/pair boundaries and tail sizes.
    for source in sources:
        names = tuple(point[source] for point in binding.source_names)
        start = 0
        for index in range(1, len(names) + 1):
            if index == len(names) or names[index] != names[start]:
                if 1 < index - start <= largest:
                    widths.add(index - start)
                start = index

    results = []
    for width in sorted(widths):
        groups = group_coordinate_batches(coordinates, BatchSpec(width, axis))
        variable = calls = 0.0
        for required, (work, call, _) in work_by_dependency.items():
            values = signatures[required]
            static = set()
            for group in groups:
                members = tuple(values[index] for index in group.point_indices)
                if all(member == members[0] for member in members):
                    static.add(members[0])
                else:
                    variable += len(members) * work
                    calls += call
            variable += len(static) * work
            calls += len(static) * call
        stack_bytes = 0
        for group in groups:
            for source in sources:
                members = tuple(binding.source_names[index][source] for index in group.point_indices)
                if any(member != members[0] for member in members):
                    spec = template.state.problem.specs[source]
                    stack_bytes += 2 * len(members) * prod(spec.shape) * spec.dtype.itemsize
        results.append(dict(mode='batched', batch_width=width,
            batch_sizes=tuple(group.actual_size for group in groups),
            variable_seconds=variable, call_seconds=calls,
            stack_traffic_bytes=stack_bytes))

    variable = calls = 0.0
    for required, (work, call, _) in work_by_dependency.items():
        count = len(set(signatures[required]))
        variable += count * work
        calls += count * call
    results.append(dict(mode='scalar', batch_width=1,
        batch_sizes=(1,) * len(coordinates.points), variable_seconds=variable,
        call_seconds=calls, stack_traffic_bytes=0))
    calibrated = profile.sample_count > 0 and profile.hardware_fingerprint != 'unmeasured'
    for item in results:
        item['modeled_seconds'] = (item['variable_seconds'] + item['call_seconds']
                                   + profile.snapshot_seconds(item['stack_traffic_bytes']))
        item['selection_score'] = ((item['modeled_seconds'],) if calibrated else
            (round(item['variable_seconds'], 15), item['call_seconds'],
             item['stack_traffic_bytes']))
    results.sort(key=lambda item: (item['selection_score'],
        0 if item['mode'] == 'scalar' else 1, item['batch_width']))
    return results


def plan_coordinate_delivery(template, output, coordinates, requests, preparations, specs, *,
                             batch=BatchSpec(), read_metadata=None, reader_aliases=(),
                             memory_limit=None, host_memory_limit=None,
                             kernel_batching=False, debug=False, target_runtime=None,
                             allocator_reserve_bytes=0, output_host_reserve_bytes=None):
    """Choose scalar reuse or a fixed lane width after the DAG has been selected.

    Explicit widths always use the established batch replay. Auto selection
    prices source-identity reuse once and lifts only the chosen feasible width.
    """
    common = dict(batch=batch, read_metadata=read_metadata, reader_aliases=reader_aliases,
        memory_limit=memory_limit, host_memory_limit=host_memory_limit,
        kernel_batching=kernel_batching, debug=debug, target_runtime=target_runtime,
        allocator_reserve_bytes=allocator_reserve_bytes,
        output_host_reserve_bytes=output_host_reserve_bytes)
    if batch.size != 'auto':
        replay = plan_coordinate_batches(template, output, coordinates, requests,
            preparations, specs, **common)
        return CoordinateDeliverySelection('batched', replay,
            dict(mode='batched', selection='explicit coordinate batch', selected_width=replay.width,
                 batch_sizes=replay.statistics['batch_sizes']))
    profile = template.cost_profile
    binding = bind_coordinate_inputs(template, requests, preparations, specs, coordinates.points,
        read_metadata=read_metadata, reader_aliases=reader_aliases, debug=debug)
    comparisons = _coordinate_work_candidates(template, binding, coordinates, batch, profile)
    attempts = []
    for candidate in comparisons:
        if candidate['mode'] == 'scalar':
            return CoordinateDeliverySelection('scalar', None, dict(
                mode='scalar',
                selection='selected-DAG identity reuse and modeled work',
                selected_width=1, batch_sizes=candidate['batch_sizes'],
                candidate_costs=tuple(comparisons), candidate_attempts=tuple(attempts),
                profile_version=profile.version, profile_sample_count=profile.sample_count))
        try:
            replay = plan_coordinate_batches(template, output, coordinates, requests,
                preparations, specs, **dict(common, batch=BatchSpec(candidate['batch_width'], batch.axis)),
                _bound_inputs=binding)
        except MemoryError as error:
            attempts.append(dict(width=candidate['batch_width'], feasible=False,
                                 reason=str(error)))
            continue
        attempts.append(dict(width=candidate['batch_width'], feasible=True))
        return CoordinateDeliverySelection('batched', replay, dict(
            mode='batched',
            selection='selected-DAG identity reuse and modeled work',
            selected_width=replay.width, batch_sizes=replay.statistics['batch_sizes'],
            candidate_costs=tuple(comparisons), candidate_attempts=tuple(attempts),
            profile_version=profile.version, profile_sample_count=profile.sample_count))
    raise MemoryError('no automatic coordinate delivery candidate fits the selected resource budget')


def supports_coordinate_batches(template):
    """Capability of the fixed-DAG lane lift, without trial planning or IO."""
    problem=template.state.problem
    return (not problem.sparse_bridges
            and set(problem.sources) | {core.key for core in problem.meson_cores} == set(problem.specs)
            and all(type(action) is Action and action.resource is None
                    and (action.core is None or type(action.core) in
                         (FactorizedWideReduction, TensorCoreSpec, MesonWholeKernel))
                    and action.kind in ('load','release','contract','baryon_join')
                    for action in template.state.actions))


def plan_coordinate_batches(template, output, coordinates, requests, preparations, specs, *,
                            batch=BatchSpec(), read_metadata=None, reader_aliases=(),
                            memory_limit=None, host_memory_limit=None,
                            kernel_batching=False, debug=False, target_runtime=None,
                            allocator_reserve_bytes=0, output_host_reserve_bytes=None,
                            _bound_inputs=None):
    """Choose the largest admitted width and audit every actual tail/group.

    Only physical binding, leading-axis lifting and storage scheduling occur.
    All contraction dependencies and their selected order come from template.
    Auto widths are tried largest first, followed by powers of two down to one;
    a failed candidate is rejected before readers or device allocations run.
    ``allocator_reserve_bytes`` is an explicit empirical device-pool margin,
    additional to the template's caller reserve. It is not an allocation or a
    mathematical fragmentation bound. The hard pool cap is left unchanged.
    ``output_host_reserve_bytes`` is W's declared lifetime envelope for final
    fields and one-point R scratch. None preserves raw-term retention; a value
    requires synchronous per-batch assembly before raw owners are released.
    """
    started = perf_counter()
    if not template.state.done:
        raise ValueError('coordinate replay requires a complete selected path')
    if type(kernel_batching) is not bool or type(debug) is not bool:
        raise TypeError('kernel_batching and debug must be bool')
    if type(allocator_reserve_bytes) is not int or allocator_reserve_bytes < 0:
        raise ValueError('allocator_reserve_bytes must be a nonnegative integer')
    if (output_host_reserve_bytes is not None
            and (type(output_host_reserve_bytes) is not int or output_host_reserve_bytes < 0)):
        raise ValueError('output_host_reserve_bytes must be a nonnegative integer')
    gpu = template.state.problem.backend == 'cupy'
    if not gpu and allocator_reserve_bytes:
        raise ValueError('CPU coordinate plans cannot reserve device allocator memory')
    cap = template.memory_limit if memory_limit is None else min(template.memory_limit, memory_limit)
    reserved_device = template.reserved_device_bytes + allocator_reserve_bytes
    logical_cap = cap - reserved_device if gpu else cap
    if logical_cap <= 0:
        raise MemoryError('device allocator and caller reserves exhaust the selected hard budget')
    axis = batch.axis
    if axis is None and len(coordinates.scan_axes) > 1:
        axis = coordinates.scan_axes[-1]
    if coordinates.mode == 'pairs':
        axis = 'pair'
    grouping = BatchSpec(batch.size, axis)
    full_groups = group_coordinate_batches(coordinates, BatchSpec('full', axis))
    largest = max(group.actual_size for group in full_groups)
    if batch.size == 'auto':
        widths, power = {1, largest}, 1
        while power < largest:
            widths.add(power)
            power *= 2
        widths = sorted(widths, reverse=True)
    else:
        widths = [largest if batch.size == 'full' else batch.size]
    binding = (_bound_inputs if _bound_inputs is not None else
        bind_coordinate_inputs(template, requests, preparations, specs, coordinates.points,
            read_metadata=read_metadata, reader_aliases=reader_aliases, debug=debug))
    # The lane symbol is fresh in both source terms and every saved operation.
    used = {axis for term in template.state.problem.terms
            for axes in (*term.subscripts.inputs, term.subscripts.output) for axis in axes}
    used.update(axis for action in template.state.actions if action.equation is not None
                for axes in (*action.equation.inputs, action.equation.output) for axis in axes)
    lane_axis = max(used, default=-1) + 1
    host_cap = template.host_memory_limit
    if host_memory_limit is not None:
        host_cap = host_memory_limit if host_cap is None else min(host_cap, host_memory_limit)
    term_bytes = sum(prod(template.state.catalog[key].shape) * template.state.catalog[key].dtype.itemsize
                     for key in template.state.results)
    # Retained host results coexist with the full output recipe and the next
    # batch snapshot. The original reserve may include non-output work too.
    base_reserve = (len(coordinates.points) * (template.reserved_host_bytes + 2 * term_bytes)
                    if output_host_reserve_bytes is None else output_host_reserve_bytes)
    attempts = []
    for width in widths:
        described, plans = [], []
        # GPU output_copy covers the device snapshot. Its complete D2H host
        # owner coexists with coordinate copies; measured reports can retain
        # all raw snapshots too. Reserve those host owners before any reads.
        reserve = base_reserve + (min(width, largest) * term_bytes
            if template.state.problem.backend == 'cupy' else 0)
        try:
            groups = group_coordinate_batches(coordinates, BatchSpec(width, axis))
            for index, group in enumerate(groups):
                descriptor = describe_coordinate_batch(binding, group.point_indices,
                    lane_axis=lane_axis, key_prefix=f'coordinate-batch:{index}:')
                state = lift_coordinate_path(template.state, descriptor.problem,
                    descriptor.source_map, lane_axis=lane_axis, profile=template.cost_profile,
                    memory_limit=logical_cap, host_memory_limit=host_cap, debug=debug)
                plan = plan_scene_storage(state, binding.blocks, descriptor.preparations,
                    profile=template.cost_profile, memory_limit=cap, host_memory_limit=host_cap,
                    reserved_host_bytes=reserve,
                    reserved_device_bytes=reserved_device)
                if not plan.feasible:
                    raise MemoryError('lifted coordinate batch exceeds the selected resource budget: '
                        f'host={plan.peak_host_bytes}/{host_cap}, device={plan.peak_device_bytes}/{cap}, '
                        f'output_host_reserve={reserve}')
                if kernel_batching:
                    plan, _ = lower_dense_batches(plan, profile=template.cost_profile)
                if not plan.feasible:
                    raise MemoryError('coordinate kernel batching exceeds the selected resource budget')
                inputs = replace(descriptor.inputs, term_ids=output.term_ids, coordinates=coordinates,
                    coordinate_batch=replace(group, axis=grouping.axis), batch=grouping,
                    blocks=binding.blocks, read_requests=binding.requests)
                described.append(replace(descriptor, inputs=inputs))
                plans.append(plan)
            # A single fresh storage plan already closes/audits all its raw
            # lifetimes, rounding and budgets. Joining only adds cross-batch
            # lifetimes; avoid copying its entire memory trajectory again.
            plans = tuple(plans) if len(plans) == 1 else tuple(join_storage_batches(plans))
            lowering = ()
            if kernel_batching:
                grouped = tuple(lower_independent_kernel_groups(plan,
                    profile=template.cost_profile, target_runtime=target_runtime) for plan in plans)
                plans = tuple(item[0] for item in grouped)
                lowering = tuple(item[1] for item in grouped)
            if any(not plan.feasible for plan in plans):
                raise MemoryError('shared coordinate input lifetimes exceed the selected resource budget')
        except MemoryError as error:
            attempts.append(dict(width=width, feasible=False, reason=str(error)))
            if batch.size != 'auto':
                raise
            continue
        attempts.append(dict(width=width, feasible=True))
        statistics = dict(selection='largest admitted candidate width under host/device budgets',
            selected_width=width, batch_sizes=tuple(item.inputs.coordinate_batch.actual_size for item in described),
            candidate_attempts=tuple(attempts), path_searches=0,
            binding_and_storage_seconds=perf_counter()-started,
            peak_host_bytes=max(plan.peak_host_bytes for plan in plans),
            peak_device_bytes=max(plan.peak_device_bytes for plan in plans),
            allocator_reserve_bytes=allocator_reserve_bytes,
            allocator_reserve_policy='explicit empirical pool headroom; not a mathematical fragmentation bound',
            hard_device_limit_bytes=cap if gpu else 0,
            numerical_device_limit_bytes=cap-allocator_reserve_bytes if gpu else 0,
            caller_reserved_device_bytes=template.reserved_device_bytes,
            peak_modeled_device_bytes=max(plan.peak_device_bytes for plan in plans)-allocator_reserve_bytes,
            device_allocation_alignment=(DEVICE_ALLOCATION_ALIGNMENT
                if template.state.problem.backend == 'cupy' else 0),
            peak_allocation_rounding_device_bytes=max(point.allocation_rounding_device_bytes
                for plan in plans for point in plan.memory),
            kernel_batches=tuple(len(plan.kernel_batches) for plan in plans),
            independent_kernel_lowering=lowering,
            scalar_reduction_groups=tuple(len(plan.scalar_reduction_groups) for plan in plans),
            cartesian_reduction_groups=tuple(len(plan.cartesian_reduction_groups) for plan in plans),
            reserved_output_host_bytes=reserve,
            output_retention=('raw-terms' if output_host_reserve_bytes is None
                              else 'assembled-coordinates'))
        return CoordinateReplayPlan(tuple(described), plans, width, statistics, tuple(reader_aliases))
    raise MemoryError('no coordinate batch fits the selected resource budgets: ' + str(attempts))
