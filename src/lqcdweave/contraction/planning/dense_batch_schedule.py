"""Lower an audited dense route into atomic Cartesian kernel groups.

The logical Action stream remains independently replayable. Batch annotations
replace physical kernels only; search costs and exact bounds remain unchanged.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import math
from heapq import heappush, heappop

from lqcdweave.contracts.correlator.api import (
    AxisPattern, Factor, validate_executable_resources, dense_batch_memory,
)
from .canonical_operations import PREFIX, describe, replay
from .dense_batches import describe_dense_batch, find_dense_batches
from .state import ContractionState, _nbytes
from .storage import plan_scene_storage
from .allocation_rounding import account_allocation_rounding


def _nodes(state, batches):
    snippets = defaultdict(list)
    for action in state.actions:
        if action.kind in ('contract', 'sparse_bridge', 'baryon_join'):
            snippets[action.tensor].append(action)
    memberships = {r.tensor: i for i,b in enumerate(batches) for r in b.results}
    nodes, node_indices = [], {}
    for tensor, actions in snippets.items():
        key = ('batch',memberships[tensor]) if tensor in memberships else ('single',tensor)
        if key not in node_indices:
            node_indices[key]=len(nodes)
            nodes.append([])
        nodes[node_indices[key]].extend(actions)
    producers = {a.tensor:i for i,actions in enumerate(nodes) for a in actions}
    parents, children = [], [[] for _ in nodes]
    for i,actions in enumerate(nodes):
        dependencies = {producers[k] for a in actions for k in a.operands if k in producers}
        if i in dependencies:
            return None  # Unsafe group, including an indirect dependent edge.
        parents.append(dependencies)
        for parent in dependencies:
            children[parent].append(i)
    indegree = [len(p) for p in parents]
    ready=[i for i,n in enumerate(indegree) if n==0]
    visited=0
    while ready:
        i=ready.pop()
        visited+=1
        for child in children[i]:
            indegree[child]-=1
            if indegree[child]==0:
                ready.append(child)
    return (nodes, parents, children) if visited==len(nodes) else None


def _replay_lower_dense_batches(plan, *, profile):
    """Compile profitable NumPy/CuPy kernels, with bounded-resource fallback.

    Source snapshots and canonical binary costs are unchanged. A group is a
    scheduling node: all inputs are ready before any member is emitted. The
    final physical audit accounts for full output owners and packing scratch.
    """
    if (plan.kernel_batches or plan.state.problem.backend not in ('numpy','cupy') or plan.explicit_resources
            or plan.retention_planned or plan.retained_before or plan.retained_after
            or plan.retained_releases or any(a.kind not in ('load','release','contract')
            or (a.kind=='contract' and (a.core is not None or not a.tensor.startswith(PREFIX)))
            for a in plan.state.actions)):
        return plan, dict(applied=False,reason='unsupported_domain')
    batches=find_dense_batches(plan.state,profile)
    if not batches:
        return plan, dict(applied=False,reason='no_profitable_grid')
    prepared_inputs = {side[0].tensor:side[0].permutation for batch in batches
                       for side,prepared in zip((batch.left,batch.right),batch.prepared_sides) if prepared}
    graph=_nodes(plan.state,batches)
    if graph is None:
        return plan, dict(applied=False,reason='cyclic_grouping')
    nodes,parents,children=graph
    remaining=Counter(k for actions in nodes for a in actions for k in set(a.operands))
    sizes={k:_nbytes(spec) for k,spec in plan.state.catalog.items()}
    outputs=frozenset(plan.state.results)
    member_names={frozenset(r.tensor for r in b.results) for b in batches}
    lowered_batches=[]
    emitted=[]
    state=ContractionState.initial(plan.state.problem)
    uncached=object()

    def with_actions(current, actions):
        live_inputs=current.__dict__.get('live_inputs',uncached)
        current=current._seed_residency(replace(current,actions=actions))
        if live_inputs is not uncached:
            object.__setattr__(current,'live_inputs',live_inputs)
        return current

    def collect_actions(current):
        emitted.extend(current.actions)
        return with_actions(current,())

    indegree=[len(p) for p in parents]
    ready={i for i,n in enumerate(indegree) if n==0}
    while ready:
        def priority(index):
            references=Counter(k for a in nodes[index] for k in set(a.operands))
            freed=sum(sizes[k] for k,n in references.items()
                      if n==remaining[k] and k in state.loaded and k not in outputs)
            calls=sum(not a.reused for a in nodes[index])
            return -freed/max(1,calls), index
        index=min(ready,key=priority)
        ready.remove(index)
        actions=nodes[index]
        inputs={k for a in actions for k in a.operands}
        for key in sorted(inputs-state.loaded):
            if key not in state.problem.sources:
                raise ValueError('batch grouping consumes data before its producer')
            state=collect_actions(state.load(key,profile,1<<100))
        paid_indices=[]
        for old in actions:
            term=state.pending[old.term_index]
            if term is None:
                raise ValueError('batch grouping repeats a completed Term')
            positions=[]
            for tensor,axes in zip(old.operands,old.equation.inputs):
                matches=[i for i,f in enumerate(term.factors) if f.tensor==tensor and f.labels==axes and i not in positions]
                if not matches:
                    raise ValueError('batch grouping changed the selected expression')
                positions.append(matches[0])
            selected,*_=describe(state,old.term_index,tuple(sorted(positions)),profile=profile,
                                  output_order=old.equation.output)
            if selected.tensor!=old.tensor or selected.reused!=old.reused or selected.seconds!=old.seconds:
                raise ValueError('batch grouping changed the selected calculation or cost')
            if not selected.reused:
                paid_indices.append(len(emitted))
            state=collect_actions(replay(state,selected,profile))
        names=frozenset(a.tensor for a in actions)
        if names in member_names:
            batch=describe_dense_batch(emitted,tuple(paid_indices),state.catalog,profile,
                                       backend=state.problem.backend,prepared_inputs=prepared_inputs)
            if batch is None:
                raise ValueError('reordered batch lost its compatible kernel layout')
            lowered_batches.append(batch)
        for a in actions:
            remaining.subtract(set(a.operands))
        for key in sorted(state.loaded):
            if key not in outputs and not remaining[key] and key not in state.live_inputs:
                state=collect_actions(state.release(key))
        for child in children[index]:
            indegree[child]-=1
            if indegree[child]==0:
                ready.add(child)
    if (not state.done or state.results!=plan.state.results
            or not math.isclose(state.seconds,plan.state.seconds,rel_tol=1e-11,abs_tol=1e-14)):
        raise ValueError('batch lowering changed the complete logical plan')
    state=with_actions(state,tuple(emitted))
    base=plan_scene_storage(state,plan.blocks,plan.preparations,profile=profile,
        memory_limit=plan.memory_limit,host_memory_limit=plan.host_memory_limit,
        disk_limit_bytes=plan.disk_limit_bytes,reserved_host_bytes=plan.reserved_host_bytes,
        reserved_device_bytes=plan.reserved_device_bytes)
    lowered=replace(base,kernel_batches=tuple(lowered_batches))
    memory=dense_batch_memory(lowered)
    host=max(p.host_bytes for p in memory)
    device=max(p.device_bytes for p in memory)
    lowered=replace(lowered,memory=memory,peak_host_bytes=host,peak_device_bytes=device,
        feasible=(device if state.problem.backend=='cupy' else host)<=lowered.memory_limit and
        (lowered.host_memory_limit is None or host<=lowered.host_memory_limit))
    lowered=account_allocation_rounding(lowered)
    host,device=lowered.peak_host_bytes,lowered.peak_device_bytes
    if not lowered.feasible:
        return plan,dict(applied=False,reason='storage_limit',candidate_peak_host_bytes=host,
                         candidate_peak_device_bytes=device)
    if lowered.seconds>=plan.seconds:
        return plan,dict(applied=False,reason='no_total_cost_improvement')
    # Independently rebuild each member grid and validate the saved lifetime.
    for batch in lowered.kernel_batches:
        if describe_dense_batch(state.actions,batch.action_indices,state.catalog,profile,
                                backend=state.problem.backend,prepared_inputs=prepared_inputs)!=batch:
            raise ValueError('batch cost or kernel metadata failed independent audit')
    validate_executable_resources(lowered)
    return lowered,dict(applied=True,batches=len(lowered_batches),
        grouped_actions=sum(len(b.action_indices) for b in lowered_batches),
        logical_compute_seconds=base.compute_seconds,physical_compute_seconds=lowered.compute_seconds,
        packing_bytes=sum(b.packing_bytes for b in lowered_batches),
        producer_layouts=len(prepared_inputs),
        peak_host_bytes=host,peak_device_bytes=device,resource_audited=True)


def _lower_local_dense_batches(plan, *, profile):
    """Annotate uninterrupted ordinary regions without moving any Action.

    Mixed kernels are barriers. Keeping the original trajectory also keeps
    their audited workspaces, raw reads and selected numerical paths intact.
    """
    actions, batches = plan.state.actions, []
    outputs = frozenset(plan.state.results)
    begin = 0
    for end in range(len(actions) + 1):
        if end < len(actions) and actions[end].kind == 'contract' and actions[end].core is None:
            continue
        if end - begin >= 2:
            region = replace(plan.state, actions=actions[begin:end],
                results=tuple(action.tensor for action in actions[begin:end] if action.tensor in outputs))
            for batch in find_dense_batches(region, profile):
                names = {result.tensor for result in batch.results}
                if any(action.tensor not in names for action in
                       region.actions[batch.action_indices[0]:batch.action_indices[-1]+1]):
                    continue
                # Regional discovery cannot prove that a producer has no paid
                # occurrences elsewhere in the mixed plan. Keep its original
                # layout and price ordinary packing against the full catalog.
                selected = describe_dense_batch(actions, tuple(i+begin for i in batch.action_indices),
                    plan.state.catalog, profile, backend=plan.state.problem.backend)
                if selected is not None:
                    batches.append(selected)
        begin = end + 1
    if not batches:
        return plan, dict(applied=False, reason='no_profitable_local_grid')
    candidate = replace(plan, kernel_batches=tuple(batches))
    memory = dense_batch_memory(candidate)
    host, device = max(p.host_bytes for p in memory), max(p.device_bytes for p in memory)
    if ((device if plan.state.problem.backend == 'cupy' else host) > plan.memory_limit
            or (plan.host_memory_limit is not None and host > plan.host_memory_limit)):
        return plan, dict(applied=False, reason='storage_limit',
                          candidate_peak_host_bytes=host, candidate_peak_device_bytes=device)
    candidate = replace(candidate, memory=memory, peak_host_bytes=host, peak_device_bytes=device)
    candidate = account_allocation_rounding(candidate)
    host, device = candidate.peak_host_bytes, candidate.peak_device_bytes
    if not candidate.feasible:
        return plan, dict(applied=False, reason='storage_limit',
                          candidate_peak_host_bytes=host, candidate_peak_device_bytes=device)
    validate_executable_resources(candidate)
    return candidate, dict(applied=True, batches=len(batches), local_regions=True,
        grouped_actions=sum(len(batch.action_indices) for batch in batches),
        logical_compute_seconds=plan.compute_seconds, physical_compute_seconds=candidate.compute_seconds,
        packing_bytes=sum(batch.packing_bytes for batch in batches), producer_layouts=0,
        peak_host_bytes=host, peak_device_bytes=device, resource_audited=True)


def lower_dense_batches(plan, *, profile, debug=False):
    """Arrange audited dense Actions without replaying whole logical states.

    Only affected Terms and ready-node priorities are updated. Physical
    dependencies, preparation, workspace and ownership use the same fixed DAG
    lifecycle accounting as the path producer. Independent replay is debug.
    """
    if (plan.kernel_batches or plan.state.problem.backend not in ('numpy', 'cupy')
            or plan.explicit_resources or plan.retention_planned or plan.retained_before
            or plan.retained_after or plan.retained_releases):
        return plan, dict(applied=False, reason='unsupported_domain')
    if any(a.kind not in ('load', 'release', 'contract', 'sparse_bridge', 'baryon_join')
           for a in plan.state.actions):
        return plan, dict(applied=False, reason='unsupported_domain')
    if any(a.kind not in ('load', 'release', 'contract')
                   or (a.kind == 'contract' and (a.core is not None or not a.tensor.startswith(PREFIX)))
                   for a in plan.state.actions):
        return _lower_local_dense_batches(plan, profile=profile)
    batches = find_dense_batches(plan.state, profile)
    if not batches:
        return plan, dict(applied=False, reason='no_profitable_grid')
    graph = _nodes(plan.state, batches)
    if graph is None:
        return plan, dict(applied=False, reason='cyclic_grouping')
    nodes, parents, children = graph
    references = [Counter(k for a in actions for k in set(a.operands)) for actions in nodes]
    remaining = Counter()
    for counts in references:
        remaining.update(counts)
    calls = [max(1, sum(not a.reused for a in actions)) for actions in nodes]
    sizes = {k: _nbytes(spec) for k, spec in plan.state.catalog.items()}
    outputs = frozenset(plan.state.results)
    prepared_inputs = {side[0].tensor: side[0].permutation for batch in batches
        for side, prepared in zip((batch.left, batch.right), batch.prepared_sides) if prepared}
    member_names = {frozenset(r.tensor for r in b.results) for b in batches}
    initial = ContractionState.initial(plan.state.problem)
    pending = [list(term.factors) if term is not None else None for term in initial.pending]
    results = list(initial.results)
    loaded, emitted, groups = set(), [], []
    # A ready node's score changes only at its own reference-count threshold,
    # or when an operand enters/leaves residency. No full ready-set rescan.
    watchers = defaultdict(lambda: defaultdict(set))
    for index, counts in enumerate(references):
        for key, count in counts.items():
            watchers[key][count].add(index)
    indegree = [len(p) for p in parents]
    ready = {i for i, count in enumerate(indegree) if not count}
    versions, heap = [0] * len(nodes), []
    priority_updates = 0

    def enqueue(index):
        nonlocal priority_updates
        freed = sum(sizes[k] for k, count in references[index].items()
                    if count == remaining[k] and k in loaded and k not in outputs)
        versions[index] += 1
        priority_updates += 1
        heappush(heap, (-freed / calls[index], index, versions[index]))

    for index in ready:
        enqueue(index)
    visited = 0
    while heap:
        _, index, version = heappop(heap)
        if index not in ready or version != versions[index]:
            continue
        ready.remove(index)
        visited += 1
        actions = nodes[index]
        inputs = set(references[index])
        touched = inputs | {a.tensor for a in actions}
        before_counts = {key: remaining[key] for key in touched}
        if any(key not in initial.problem.sources for key in inputs-loaded):
            raise ValueError('batch grouping consumes data before its producer')
        loaded.update(inputs)
        for old in actions:
            factors = pending[old.term_index]
            if factors is None:
                raise ValueError('batch grouping repeats a completed Term')
            positions = []
            for tensor, axes in zip(old.operands, old.equation.inputs):
                matches = [i for i, factor in enumerate(factors)
                           if factor.tensor == tensor and factor.labels == axes and i not in positions]
                if not matches:
                    raise ValueError('batch grouping changed the selected expression')
                positions.append(matches[0])
            selected = tuple(sorted(positions))
            action = replace(old, positions=selected,
                operands=tuple(factors[i].tensor for i in selected),
                equation=AxisPattern(tuple(factors[i].labels for i in selected), old.equation.output))
            emitted.append(action)
            updated = [factor for i, factor in enumerate(factors) if i not in positions]
            updated.append(Factor(action.tensor, action.equation.output))
            if len(updated) == 1 and updated[0].labels == initial.pending[old.term_index].output:
                pending[old.term_index], results[old.term_index] = None, action.tensor
            else:
                pending[old.term_index] = updated
            loaded.add(action.tensor)
        names = frozenset(a.tensor for a in actions)
        if names in member_names:
            groups.append(names)
        remaining.subtract(references[index])
        for key in touched:
            if key not in outputs and not remaining[key]:
                loaded.discard(key)
        dirty = set()
        for key in touched:
            dirty.update(watchers[key].get(before_counts[key], ()))
            dirty.update(watchers[key].get(remaining[key], ()))
        for child in children[index]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.add(child)
                dirty.add(child)
        for node in dirty & ready:
            enqueue(node)
    if visited != len(nodes) or any(term is not None for term in pending) or tuple(results) != plan.state.results:
        raise ValueError('batch lowering changed the complete logical plan')
    # Workspaces were audited for these selected computations before lowering.
    workspace = {plan.state.actions[p.action_index].tensor:
        max(p.workspace_host_bytes, p.workspace_device_bytes)
        for p in plan.memory if p.phase == 'contract'
        and p.action_index < len(plan.state.actions) and not plan.state.actions[p.action_index].reused}
    spans = []
    for index, action in enumerate(emitted):
        if not spans or emitted[spans[-1][0]].tensor != action.tensor:
            spans.append((index, index+1))
        else:
            spans[-1] = (spans[-1][0], index+1)
    state = replace(plan.state, actions=tuple(emitted), loaded=frozenset(outputs),
        seconds=sum(a.seconds for a in emitted), peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0)
    from .optimizers.macro_space import MacroSearchSpace
    from .optimizers.storage_schedule import _Dag, _plan
    physical = MacroSearchSpace(state.problem, plan.blocks, plan.preparations,
        memory_limit=plan.memory_limit, host_memory_limit=plan.host_memory_limit,
        disk_limit_bytes=plan.disk_limit_bytes, profile=profile,
        reserved_host_bytes=plan.reserved_host_bytes, reserved_device_bytes=plan.reserved_device_bytes)
    dag = _Dag(state, spans, tuple(workspace.get(emitted[begin].tensor, 0) for begin, _ in spans),
               physical, audited_computations=not debug)
    node_indices = {emitted[begin].tensor: index for index, (begin, _) in enumerate(spans)}
    atomic_groups = tuple(tuple(sorted(node_indices[key] for key in names)) for names in groups)
    base = _plan(dag, range(len(dag.nodes)), audited_computations=True, debug=debug,
                 atomic_groups=atomic_groups)
    if not math.isclose(base.state.seconds, plan.state.seconds, rel_tol=1e-11, abs_tol=1e-14):
        raise ValueError('batch lowering changed selected computation or snapshot costs')
    paid = {a.tensor: i for i, a in enumerate(base.state.actions) if a.kind == 'contract' and not a.reused}
    lowered_batches = []
    for names in groups:
        batch = describe_dense_batch(base.state.actions, tuple(sorted(paid[key] for key in names)),
            base.state.catalog, profile, backend=base.state.problem.backend, prepared_inputs=prepared_inputs)
        if batch is None:
            raise ValueError('reordered batch lost its compatible kernel layout')
        lowered_batches.append(batch)
    lowered = replace(base, kernel_batches=tuple(lowered_batches))
    memory = dense_batch_memory(lowered)
    host, device = max(p.host_bytes for p in memory), max(p.device_bytes for p in memory)
    lowered = replace(lowered, memory=memory, peak_host_bytes=host, peak_device_bytes=device,
        feasible=(device if state.problem.backend == 'cupy' else host) <= lowered.memory_limit
        and (lowered.host_memory_limit is None or host <= lowered.host_memory_limit))
    lowered = account_allocation_rounding(lowered)
    host, device = lowered.peak_host_bytes, lowered.peak_device_bytes
    if not lowered.feasible:
        return plan, dict(applied=False, reason='storage_limit', candidate_peak_host_bytes=host,
                         candidate_peak_device_bytes=device)
    if lowered.seconds >= plan.seconds:
        return plan, dict(applied=False, reason='no_total_cost_improvement')
    if debug:
        reference, report = _replay_lower_dense_batches(plan, profile=profile)
        if (not report['applied'] or reference.kernel_batches != lowered.kernel_batches
                or reference.state.actions != lowered.state.actions
                or not math.isclose(reference.seconds, lowered.seconds, rel_tol=1e-11, abs_tol=1e-14)):
            raise RuntimeError('dense lowering differs from independent algebra replay')
        validate_executable_resources(lowered)
    return lowered, dict(applied=True, batches=len(lowered_batches),
        grouped_actions=sum(len(b.action_indices) for b in lowered_batches),
        logical_compute_seconds=base.compute_seconds, physical_compute_seconds=lowered.compute_seconds,
        packing_bytes=sum(b.packing_bytes for b in lowered_batches), producer_layouts=len(prepared_inputs),
        peak_host_bytes=host, peak_device_bytes=device, resource_audited=True,
        priority_updates=priority_updates)
