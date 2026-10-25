"""Group existing ready Cartesian wide reductions without changing Actions."""
from collections import Counter, defaultdict
from dataclasses import replace
from heapq import heappop, heappush

from lqcdweave.contracts.correlator.api import (
    Action, SceneStoragePlan, DenseKernelBatch, DenseGemmGroup,
    cartesian_reduction_geometry, describe_cartesian_reduction_group,
    cartesian_reduction_group_workspace, cartesian_reduction_group_memory, validate_cartesian_reduction_groups,
    device_allocation_bytes,
)
from .gemm_groups import _RangePeak


def _baseline_ready_times(plan, candidates):
    """Last original birth before each child, with no speculative early outputs.

    A live tensor has one current residency interval. At a child's saved start,
    its inputs' current births identify the interval that must contain any
    earlier group start. Requiring birth < group start then proves residency:
    no release can occur between that birth and this original child start.
    A release/reload updates the birth rather than reusing an older interval.
    Outputs of a batch at this index are published only *after* this snapshot.
    """
    starts = {batch.action_indices[0]: batch for batch in plan.kernel_batches}
    births, ready, regions = {}, {}, {}
    region = 0
    for index, action in enumerate(plan.state.actions):
        if index in candidates:
            batch = candidates[index][0]
            inputs = tuple(dict.fromkeys(o.tensor for o in (*batch.left, *batch.right)))
            if any(key not in births for key in inputs):
                return None
            ready[index] = max(births[key] + 1 for key in inputs)
            regions[index] = region
        batch = starts.get(index)
        if batch is not None:
            for result in batch.results:
                births.setdefault(result.tensor, index)
        if action.kind == 'release':
            births.pop(action.tensor, None)
        elif action.kind in ('load', 'contract', 'sparse_bridge', 'baryon_join'):
            births.setdefault(action.tensor, index)
        if action.kind not in ('load', 'release', 'contract') or action.core is not None:
            region += 1
    return ready, regions


def lower_cartesian_reduction_groups(plan, *, target_runtime, max_tasks=8):
    """Annotate ready whole batches; retain uncalibrated original costs.

    Ready-time heaps come from the *original* physical schedule. A new group's
    early outputs never make another group eligible, so binding-time fallback
    of one group cannot invalidate a downstream selection. Original packing,
    scratch and child ownership remain charged; only additional simultaneous
    packing, partials and early independent cubes enter the additive bucket.
    """
    stats = dict(applied=False, max_tasks=max_tasks,
                 cost_status='uncalibrated-original-member-sum', speedup_claimed=False)

    def unchanged(reason, **details):
        return plan, dict(stats, reason=reason, **details)

    if type(max_tasks) is not int or not 2 <= max_tasks <= 8:
        return unchanged('unsupported_group_width')
    if type(plan) is not SceneStoragePlan:
        return unchanged('unsupported_plan_type')
    if plan.cartesian_reduction_groups:
        return unchanged('existing_cartesian_groups')
    actions, catalog = plan.state.actions, plan.state.catalog
    if (target_runtime != 'cuda' or plan.state.problem.backend != 'cupy'
            or not plan.feasible or not plan.memory or plan.explicit_resources
            or plan.retention_planned or plan.retained_before or plan.retained_after
            or plan.retained_releases
            or any(type(batch) not in (DenseKernelBatch, DenseGemmGroup) for batch in plan.kernel_batches)
            or any(type(action) is not Action or action.kind not in
                   ('load', 'release', 'contract', 'sparse_bridge', 'baryon_join')
                   or action.resource is not None for action in actions)):
        return unchanged('unsupported_domain')
    if any(point.cartesian_reduction_device_bytes for point in plan.memory):
        return unchanged('nonzero_original_cartesian_memory')
    positions = [point.action_index for point in plan.memory]
    if any(left > right for left, right in zip(positions, positions[1:])):
        return unchanged('unordered_memory_trajectory')
    device_peak = max(point.device_bytes for point in plan.memory)
    host_peak = max(point.host_bytes for point in plan.memory)
    if (device_peak != plan.peak_device_bytes or host_peak != plan.peak_host_bytes
            or device_peak > plan.memory_limit
            or plan.host_memory_limit is not None and host_peak > plan.host_memory_limit
            or plan.disk_limit_bytes is not None and plan.peak_disk_bytes > plan.disk_limit_bytes):
        return unchanged('inconsistent_original_memory')
    contract_points = defaultdict(list)
    for position, point in enumerate(plan.memory):
        if point.phase == 'contract':
            contract_points[point.action_index].append(position)
    records, excluded = {}, Counter()
    for batch in plan.kernel_batches:
        if type(batch) is not DenseKernelBatch:
            excluded['existing_gemm_group'] += 1
            continue
        try:
            geometry = cartesian_reduction_geometry(batch, catalog)
        except (ValueError, TypeError, KeyError):
            excluded['unsupported_geometry'] += 1
            continue
        if any(actions[index].reused for index in batch.action_indices):
            excluded['reused_member'] += 1
            continue
        first = batch.action_indices[0]
        if len(contract_points[first]) != 1:
            return unchanged('missing_or_ambiguous_contract_point')
        records[first] = batch, geometry
    stats.update(candidates=len(records), excluded=dict(excluded))
    if len(records) < 2:
        return unchanged('insufficient_candidates')
    baseline = _baseline_ready_times(plan, records)
    if baseline is None:
        return unchanged('original_batch_input_not_resident')
    ready_times, regions = baseline
    if any(ready_times[index] > index for index in records):
        return unchanged('original_batch_dependency_not_ready')
    keys = {index: (regions[index], geometry.shapes, geometry.dtype.str,
                    geometry.allowance, geometry.blocks)
            for index, (_, geometry) in records.items()}
    pending = sorted((ready_times[index], index) for index in records)
    pending_position = 0
    heaps, consumed = defaultdict(list), set()
    operations = Counter()
    peak = _RangePeak([point.device_bytes for point in plan.memory])
    groups, attempts, rejections = [], 0, 0
    for index in sorted(records):
        while pending_position < len(pending) and pending[pending_position][0] <= index:
            _, member = pending[pending_position]
            pending_position += 1
            if member not in consumed:
                heappush(heaps[keys[member]], member)
                operations['heap_push'] += 1
        if index in consumed:
            continue
        heap, selected = heaps[keys[index]], [index]
        while heap and len(selected) < max_tasks:
            member = heappop(heap)
            operations['heap_pop'] += 1
            if member == index or member in consumed:
                continue
            if member < index:
                raise RuntimeError('ready Cartesian candidate precedes its original position')
            selected.append(member)
        snapshot = tuple(selected)
        start = contract_points[index][0]
        while len(selected) > 1:
            children = tuple(records[member][0] for member in selected)
            try:
                group = describe_cartesian_reduction_group(plan.kernel_batches,
                    tuple(child.action_indices for child in children), catalog,
                    sum(child.seconds for child in children))
            except ValueError:
                # For example, the group grid limit may require fewer children.
                selected.pop()
                continue
            changes = [(start, start + 1,
                        cartesian_reduction_group_workspace(children, catalog, group.workspace_bytes))]
            changes.extend((start, contract_points[member][0],
                            device_allocation_bytes(records[member][0].output_bytes))
                           for member in selected[1:])
            attempts += 1
            for begin, end, amount in changes:
                peak.add(begin, end, amount)
            if peak.maximum <= plan.memory_limit:
                groups.append(group)
                break
            for begin, end, amount in reversed(changes):
                peak.add(begin, end, -amount)
            rejections += 1
            selected.pop()
        consumed.update(selected)
        for member in snapshot[len(selected):]:
            heappush(heap, member)
            operations['heap_push'] += 1
    stats.update(resource_attempts=attempts, resource_rejections=rejections,
                 index_operations=dict(operations))
    if not groups:
        return unchanged('no_feasible_resident_group')
    candidate = replace(plan, cartesian_reduction_groups=tuple(groups))
    memory = cartesian_reduction_group_memory(candidate)
    final_device = max(point.device_bytes for point in memory)
    if final_device != peak.maximum:
        raise RuntimeError('incremental Cartesian resource bound differs from contract audit')
    candidate = replace(candidate, memory=memory, peak_device_bytes=final_device)
    validate_cartesian_reduction_groups(candidate)
    grouped = sum(len(group.batch_action_indices) for group in groups)
    return candidate, dict(stats, applied=True, groups=len(groups), grouped_batches=grouped,
        launches_before=2 * len(records), launches_after=2 * (len(records) - grouped + len(groups)),
        theoretical_launches_saved=2 * (grouped - len(groups)),
        original_seconds=plan.seconds, seconds=candidate.seconds,
        original_peak_device_bytes=plan.peak_device_bytes, peak_device_bytes=final_device,
        peak_host_bytes=host_peak, resource_audited=True)
