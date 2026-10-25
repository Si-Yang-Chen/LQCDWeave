"""Select ready resident terminal scalars without rewriting logical Actions."""
from collections import Counter, defaultdict
from dataclasses import replace
from heapq import heappop, heappush

from lqcdweave.contracts.correlator.api import (
    Action, SceneStoragePlan, DenseKernelBatch, DenseGemmGroup, device_allocation_bytes,
)
from lqcdweave.contracts.correlator.scalar_reduction_group import (
    ScalarReductionGroup, scalar_reduction_geometry, scalar_reduction_group_memory,
    validate_scalar_reduction_groups, scalar_reduction_group_limit,
)
from .dense_batch_schedule import _nodes
from .gemm_groups import _RangePeak


def lower_scalar_reduction_groups(plan, *, target_runtime, max_tasks=32):
    """Annotate bounded CUDA groups; measured speedups are deliberately unknown.

    W decides whether kernel batching is enabled before invoking this pass.
    This pass never loads a tensor, changes blocks, or replays/serializes a plan.
    Index counters and residency watchers feed geometry-partitioned heaps;
    every group takes a ready snapshot before publishing any selected output.
    """
    stats = dict(applied=False, cost_status='uncalibrated-original-member-sum',
                 speedup_claimed=False, max_tasks=max_tasks)

    def unchanged(reason, **details):
        return plan, dict(stats, reason=reason, **details)

    if type(max_tasks) is not int or not 2 <= max_tasks <= 32:
        return unchanged('unsupported_group_width')
    if type(plan) is not SceneStoragePlan:
        return unchanged('unsupported_plan_type')
    if plan.scalar_reduction_groups:
        return unchanged('existing_scalar_groups')
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
    if any(point.scalar_reduction_device_bytes for point in plan.memory):
        return unchanged('nonzero_original_scalar_memory')
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
    excluded = {index for batch in plan.kernel_batches for index in batch.action_indices}
    last_use, first_release, paid_producers, regions = {}, {}, Counter(), []
    region = 0
    for index, action in enumerate(actions):
        regions.append(region)
        if action.kind in ('contract', 'sparse_bridge', 'baryon_join'):
            for key in (*action.operands, *getattr(action.core, 'inputs', ())):
                last_use[key] = index
        if action.kind in ('load', 'contract', 'sparse_bridge', 'baryon_join') and not action.reused:
            paid_producers[action.tensor] += 1
        if action.kind == 'release':
            first_release.setdefault(action.tensor, index)
        if action.kind not in ('load', 'release', 'contract') or action.core is not None:
            region += 1

    records, rejected = {}, Counter()
    for index, action in enumerate(actions):
        if action.kind != 'contract' or action.reused:
            continue
        if index in excluded:
            rejected['existing_batch_member'] += 1
            continue
        if (last_use.get(action.tensor, -1) > index or paid_producers[action.tensor] != 1
                or first_release.get(action.tensor, len(actions)) < index):
            rejected['nonterminal_or_shared_result'] += 1
            continue
        try:
            geometry = scalar_reduction_geometry(action, catalog)
        except (ValueError, TypeError, KeyError):
            rejected['unsupported_geometry'] += 1
            continue
        if not scalar_reduction_group_limit(geometry.shapes):
            rejected['unsupported_parameter_envelope'] += 1
            continue
        if len(contract_points[index]) != 1:
            return unchanged('missing_or_ambiguous_contract_point')
        records[index] = (geometry, frozenset(action.operands),
            (regions[index], geometry.axes, geometry.shapes, geometry.dtype.str, geometry.blocks))
    stats.update(candidates=len(records), excluded=dict(rejected))
    if len(records) < 2:
        return unchanged('insufficient_candidates')

    graph = _nodes(plan.state, plan.kernel_batches)
    if graph is None:
        return unchanged('cyclic_physical_dag')
    nodes, parents, children = graph
    producers = {action.tensor: node for node, members in enumerate(nodes) for action in members}
    starts = {}
    for index, action in enumerate(actions):
        if action.kind in ('contract', 'sparse_bridge', 'baryon_join'):
            starts.setdefault(producers[action.tensor], index)
    at_start = {index: node for node, index in starts.items()}
    candidate_for_node = {producers[actions[index].tensor]: index for index in records}
    outputs = [tuple(dict.fromkeys(action.tensor for action in members)) for members in nodes]
    remaining = list(map(len, parents))
    missing = {index: len(record[1]) for index, record in records.items()}
    watchers = defaultdict(list)
    for index, (_, inputs, _) in records.items():
        for key in inputs:
            watchers[key].append(index)
    live, done = set(), set()
    versions, queued, heaps = Counter(), {}, defaultdict(list)
    operations = Counter()

    def enqueue(index):
        node = producers[actions[index].tensor]
        if node in done or remaining[node] or missing[index]:
            return
        version = versions[index]
        if queued.get(index) == version:
            return
        queued[index] = version
        heappush(heaps[records[index][2]], (index, version))
        operations['heap_push'] += 1

    def residency(key, present):
        if (key in live) == present:
            return
        live.add(key) if present else live.remove(key)
        for index in watchers[key]:
            if producers[actions[index].tensor] in done:
                continue
            missing[index] += -1 if present else 1
            versions[index] += 1
            enqueue(index)
            operations['residency_edge_update'] += 1

    def publish(node):
        if node in done:
            return
        done.add(node)
        for key in outputs[node]:
            residency(key, True)
        for child in children[node]:
            remaining[child] -= 1
            operations['dependency_edge_update'] += 1
            if child in candidate_for_node:
                enqueue(candidate_for_node[child])

    peak = _RangePeak([point.device_bytes for point in plan.memory])
    groups, attempts, budget_rejections = [], 0, 0
    for index, action in enumerate(actions):
        if action.kind == 'load':
            residency(action.tensor, True)
        elif action.kind == 'release':
            residency(action.tensor, False)
        node = at_start.get(index)
        if node is None or node in done:
            continue
        if remaining[node]:
            return unchanged('original_physical_dependency_not_ready')
        if index not in records:
            publish(node)
            continue
        if missing[index]:
            return unchanged('original_scalar_input_not_resident')
        heap, selected = heaps[records[index][2]], [index]
        limit = min(max_tasks, scalar_reduction_group_limit(records[index][0].shapes))
        while heap and len(selected) < limit:
            candidate, version = heappop(heap)
            operations['heap_pop'] += 1
            if queued.get(candidate) == version:
                queued.pop(candidate, None)
            candidate_node = producers[actions[candidate].tensor]
            if (candidate == index or candidate_node in done or version != versions[candidate]
                    or missing[candidate] or remaining[candidate_node]):
                continue
            if candidate < index:
                return unchanged('ready_queue_precedes_original_position')
            selected.append(candidate)
        snapshot = tuple(selected)
        start_position = contract_points[index][0]
        while len(selected) > 1:
            partial = sum(records[member][0].partial_bytes for member in selected)
            changes = [(start_position, start_position + 1, device_allocation_bytes(partial))]
            changes.extend((start_position, contract_points[member][0],
                            device_allocation_bytes(records[member][0].dtype.itemsize)) for member in selected[1:])
            attempts += 1
            for begin, end, amount in changes:
                peak.add(begin, end, amount)
            if peak.maximum <= plan.memory_limit:
                groups.append(ScalarReductionGroup(tuple(selected), partial,
                    sum(actions[member].seconds for member in selected)))
                break
            for begin, end, amount in reversed(changes):
                peak.add(begin, end, -amount)
            budget_rejections += 1
            selected.pop()
        # No newly published result can make another member enter this group.
        for member in selected:
            publish(producers[actions[member].tensor])
        for member in snapshot[len(selected):]:
            enqueue(member)
    stats.update(resource_attempts=attempts, resource_rejections=budget_rejections,
                 index_operations=dict(operations))
    if len(done) != len(nodes):
        return unchanged('incomplete_physical_dag')
    if not groups:
        return unchanged('no_feasible_resident_group')
    candidate = replace(plan, scalar_reduction_groups=tuple(groups))
    try:
        memory = scalar_reduction_group_memory(candidate)
        final_device = max(point.device_bytes for point in memory)
        if final_device != peak.maximum:
            return unchanged('incremental_memory_disagrees_with_contract')
        candidate = replace(candidate, memory=memory, peak_device_bytes=final_device)
        validate_scalar_reduction_groups(candidate)
    except (ValueError, TypeError, NotImplementedError, AttributeError) as error:
        return unchanged('contract_validation_unsupported_or_failed', detail=str(error))
    grouped = sum(len(group.action_indices) for group in groups)
    return candidate, dict(stats, applied=True, groups=len(groups), grouped_reductions=grouped,
        launches_before=2 * len(records), launches_after=2 * (len(records) - grouped + len(groups)),
        theoretical_launches_saved=2 * (grouped - len(groups)),
        original_seconds=plan.seconds, seconds=candidate.seconds,
        original_peak_device_bytes=plan.peak_device_bytes, peak_device_bytes=final_device,
        peak_host_bytes=host_peak, resource_audited=True)
