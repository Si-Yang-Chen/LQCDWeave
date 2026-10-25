"""Schedule shared physical reads around a separately chosen contraction path.

Block keys identify verified dataset selections, not logical tensors or just
filenames. Adapters must union overlapping selections before constructing this
catalog. This module neither discovers files nor changes numerical conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from .state import ContractionProblem, ContractionState, CostProfile, SourceSpec, _nbytes
from .actions import _apply_selected
from .sparse_workspace import prepare_coordinate_batch_workspace
from .logical_index import LayoutIndex, reuse_indexes
from lqcdweave.contracts.correlator.api import validate_executable_resources, MemoryPoint, SceneStoragePlan, SourcePreparation, StorageBlock, StorageEvent












def logical_source_problem(problem: ContractionProblem) -> ContractionProblem:
    """Remove physical reads/storage while retaining logical result allocations.

    Use with ``logical_cost_profile`` when choosing the compute path. Each
    logical source has its own zero-byte region; physical reuse is a separate
    relation and can connect tensors with different ranks and meanings.
    """
    sources = {
        key: SourceSpec(key, 0, snapshot_bytes=(source.snapshot_bytes if key in problem.specs
                                               else source.snapshot_bytes or source.region_bytes),
                        read_bytes=0, metadata=source.metadata)
        for key, source in problem.sources.items()
    }
    return replace(problem, sources=sources, single_read_regions=False)


def join_storage_batches(plans):
    """Recheck independent paths with raw lifetimes spanning all batches.

    All adapters must already use identical maximum-prefix blocks for a shared
    key. This validates a proposed order, not a search: infeasible returned
    batches need replanning with the additional resident storage accounted for.
    Output arrays must be consumed/released before executing the next batch.
    """
    plans = tuple(plans)
    if not plans:
        raise ValueError("at least one storage batch is required")
    blocks, identities, first, last = {}, {}, {}, {}
    backend = plans[0].state.problem.backend
    for batch, plan in enumerate(plans):
        validate_executable_resources(plan)
        if plan.state.problem.backend != backend:
            raise ValueError("storage batches must use one backend")
        for key, block in plan.blocks.items():
            if key in blocks and blocks[key] != block:
                raise ValueError("shared blocks must have identical maximum-prefix definitions")
            identity = (block.dataset, block.selection)
            if identity in identities and identities[identity] != key:
                raise ValueError("one physical selection must have one shared key")
            blocks[key], identities[identity] = block, key
        for event in plan.events:
            location = (batch, event.action_index)
            if event.kind == "read":
                first.setdefault(event.block, location)
            elif event.kind == "release":
                last[event.block] = location
    if first.keys() != last.keys():
        raise ValueError("batch paths must close every physical block")
    resident, joined = set(), []
    raw_host_bytes = raw_device_bytes = 0
    for batch, plan in enumerate(plans):
        initial_raw = tuple(resident)
        events = tuple(e for e in plan.events if
            (first if e.kind == "read" else last)[e.block] == (batch, e.action_index))
        releases = {}
        for event in events:
            if event.kind == "release":
                releases.setdefault(event.action_index, []).append(event.block)
        points_by_action = {}
        for point in plan.memory:
            points_by_action.setdefault(point.action_index, []).append(point)
        points = []
        for index in range(len(plan.state.actions) + 1):
            for point in points_by_action.get(index, ()):
                if point.phase.startswith("read:"):
                    key = point.phase[len("read:"):]
                    if first[key] != (batch, index):
                        continue  # neither staging allocation nor a second read
                    if key not in resident:
                        resident.add(key)
                        raw_host_bytes += blocks[key].host_bytes
                        raw_device_bytes += blocks[key].device_bytes
                points.append(replace(point,
                    raw_host_bytes=raw_host_bytes,
                    raw_device_bytes=raw_device_bytes))
            for key in releases.get(index, ()):
                resident.remove(key)
                raw_host_bytes -= blocks[key].host_bytes
                raw_device_bytes -= blocks[key].device_bytes
        host_peak = max(p.host_bytes for p in points)
        device_peak = max(p.device_bytes for p in points)
        primary = device_peak if backend == "cupy" else host_peak
        from .allocation_rounding import account_allocation_rounding
        joined.append(account_allocation_rounding(replace(plan, events=events, memory=tuple(points),
            peak_host_bytes=host_peak,
            peak_device_bytes=device_peak, feasible=primary <= plan.memory_limit and
            (plan.host_memory_limit is None or host_peak <= plan.host_memory_limit)),
            initial_raw=initial_raw, raw_blocks=blocks))
    if resident:
        raise ValueError("joined sequence leaves raw storage live")
    return tuple(joined)


def logical_cost_profile(profile: CostProfile) -> CostProfile:
    return profile


def plan_scene_storage(
    state: ContractionState,
    blocks: Mapping[str, StorageBlock],
    preparations: Mapping[str, SourcePreparation],
    *,
    profile: CostProfile,
    memory_limit: int,
    host_memory_limit: int | None = None,
    disk_limit_bytes: int | None = None,
    explicit_resources: bool = False,
    reserved_host_bytes: int = 0,
    reserved_device_bytes: int = 0,
    debug: bool = False,
) -> SceneStoragePlan:
    """Read just before first demand; release only after the last dependency.

    Future loads (including logical rematerialization) keep a raw block alive.
    Dense, whole meson and baryon operations consume only the selected
    allocations and local kernel workspace. Independent algebra replay is
    debug-only for these operations. Sparse bridges retain their descriptor
    replay until their opaque owner workspace is separately represented.
    This pass does not add load/release choices to the search.
    A compute-feasible path can be storage-infeasible; report it as such rather
    than silently reloading blocks or ignoring their resident memory.
    """
    if type(debug) is not bool:
        raise TypeError('debug must be bool')
    if explicit_resources:
        from .resources import audit_resource_trajectory
        return audit_resource_trajectory(state, blocks, preparations, profile=profile,
            memory_limit=memory_limit, host_memory_limit=host_memory_limit,
            disk_limit_bytes=disk_limit_bytes, reserved_host_bytes=reserved_host_bytes,
            reserved_device_bytes=reserved_device_bytes)
    validate_executable_resources(state)
    if disk_limit_bytes is not None and (type(disk_limit_bytes) is not int or disk_limit_bytes <= 0):
        raise ValueError("disk_limit_bytes must be positive integer bytes or None")
    if not state.done:
        raise ValueError("storage scheduling requires a complete compute path")
    for value in (reserved_host_bytes, reserved_device_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("reserved memory must be nonnegative integer bytes")
    if state.problem.backend == "numpy" and reserved_device_bytes:
        raise ValueError("CPU plans cannot reserve device memory")
    if memory_limit <= 0 or (host_memory_limit is not None and host_memory_limit <= 0):
        raise ValueError("memory limits must be positive")
    if any(s.region_bytes or s.host_region_bytes or s.staging_bytes or s.transfer_bytes
           or s.read_bytes != 0 for s in state.problem.sources.values()):
        raise ValueError("choose the path on logical_source_problem first")
    blocks, preparations = dict(blocks), dict(preparations)
    identities = [(b.dataset, b.selection) for b in blocks.values()]
    if len(set(identities)) != len(identities):
        raise ValueError("one physical selection must have one block key")
    if state.problem.backend == "numpy" and any(b.device_bytes or b.staging_device_bytes for b in blocks.values()):
        raise ValueError("CPU plans cannot contain device storage")
    if state.problem.backend == "numpy" and any(p.workspace_device_bytes for p in preparations.values()):
        raise ValueError("CPU preparations cannot contain device workspace")
    first, last = {}, {}
    borrowed = {}
    end = len(state.actions)
    for i, action in enumerate(state.actions):
        if action.kind == "load":
            if action.tensor not in preparations:
                raise ValueError(f"missing source preparation: {action.tensor}")
            prep = preparations[action.tensor]
            for key in prep.blocks:
                if key not in blocks:
                    raise ValueError(f"unknown storage block: {key}")
                first.setdefault(key, i)
                last[key] = i
            if action.tensor not in state.catalog:
                borrowed[action.tensor] = prep.blocks
        elif action.kind == "release":
            for key in borrowed.pop(action.tensor, ()):
                last[key] = max(last[key], i)
    for dependencies in borrowed.values():
        for key in dependencies:
            last[key] = end
    reads, releases = {}, {}
    for key in first:
        reads.setdefault(first[key], []).append(key)
        releases.setdefault(last[key], []).append(key)
    events, points, resident = [], [], set()
    from .storage_lifetimes import StorageLifetimes, supports_storage_lifetimes
    direct = not debug and supports_storage_lifetimes(state)
    logical = StorageLifetimes(state) if direct else ContractionState.initial(state.problem).with_indexed_rows()
    match_index = None if direct else LayoutIndex(logical) if reuse_indexes(logical.pending) else None
    model = logical_cost_profile(profile)
    preparation_costs = {}
    raw_host_bytes = raw_device_bytes = 0

    def point(i, phase, logical_host, logical_device, workspace_host=0, workspace_device=0):
        points.append(MemoryPoint(i, phase,
                                 raw_host_bytes,
                                 raw_device_bytes,
                                 logical_host, logical_device, workspace_host, workspace_device,
                                 reserved_host_bytes, reserved_device_bytes))

    for i, action in enumerate(state.actions):
        changes = []
        gpu = state.problem.backend == "cupy"
        for key in sorted(reads.get(i, ())):
            resident.add(key)
            block = blocks[key]
            raw_host_bytes += block.host_bytes
            raw_device_bytes += block.device_bytes
            events.append(StorageEvent(i, "read", key))
            point(i, "read:" + key, logical.resident_host_bytes,
                  logical.resident_bytes if gpu else 0,
                  block.staging_host_bytes, block.staging_device_bytes)
        if direct:
            previous_bytes=logical.resident_bytes
            workspace=logical.advance(action)
            if action.kind=='load':
                prep=preparations[action.tensor]
                preparation_costs[i]=prep.seconds(profile)
                point(i,'materialize:'+action.tensor,logical.resident_host_bytes,
                      logical.resident_bytes if gpu else 0,
                      prep.workspace_host_bytes,prep.workspace_device_bytes)
            elif action.kind in ('contract', 'baryon_join'):
                resident_compute=max(previous_bytes,logical.resident_bytes)
                point(i,action.kind,resident_compute if not gpu else 0,
                      resident_compute if gpu else 0,0 if gpu else workspace,
                      workspace if gpu else 0)
            for key in sorted(releases.get(i, ())):
                resident.remove(key)
                raw_host_bytes -= blocks[key].host_bytes
                raw_device_bytes -= blocks[key].device_bytes
                events.append(StorageEvent(i,'release',key))
            continue
        before = logical._seed_residency(replace(logical, actions=(), peak_bytes=0,
            peak_host_bytes=0, peak_device_bytes=0), 0)
        if "live_inputs" in logical.__dict__:
            object.__setattr__(before, "live_inputs", logical.live_inputs)
        if action.kind == "load":
            logical = before.load(action.tensor, model, 1 << 100)
            prep = preparations[action.tensor]
            preparation_costs[i] = prep.seconds(profile)
            point(i, "materialize:" + action.tensor, logical.resident_host_bytes,
                  logical.resident_bytes if gpu else 0,
                  prep.workspace_host_bytes, prep.workspace_device_bytes)
        elif action.kind == "release":
            logical = before.release(action.tensor)
        else:
            logical = _apply_selected(before, action.term_index, action, model, 1 << 100,
                                      layout_index=match_index, changes=changes)
            if logical is None or logical.actions[-1] != action:
                raise ValueError("compute path does not match its declared problem/profile")
            resident_compute = max(before.resident_bytes, logical.resident_bytes)
            workspace = max(0, logical.peak_bytes - resident_compute)
            point(i, action.kind, resident_compute if not gpu else logical.resident_host_bytes,
                  resident_compute if gpu else 0,
                  0 if gpu else workspace, workspace if gpu else 0)
        if match_index is not None:
            match_index.advance(logical, (index for index, _, _ in changes))
        for key in sorted(releases.get(i, ())):
            resident.remove(key)
            raw_host_bytes -= blocks[key].host_bytes
            raw_device_bytes -= blocks[key].device_bytes
            events.append(StorageEvent(i, "release", key))
    if direct:
        logical.finish()
    elif not logical.done or abs(logical.seconds - state.seconds) > 1e-8 * max(1, state.seconds):
        raise ValueError("logical plan costs disagree with the supplied profile")
    copies = sum(_nbytes(state.catalog[key]) for key in state.results)
    point(end, "output_copy", logical.resident_host_bytes + (copies if not gpu else 0),
          logical.resident_bytes + copies if gpu else 0)
    for key in sorted(releases.get(end, ())):
        resident.remove(key)
        events.append(StorageEvent(end, "release", key))
    host_peak = max(p.host_bytes for p in points)
    device_peak = max(p.device_bytes for p in points)
    primary_peak = device_peak if gpu else host_peak
    plan = SceneStoragePlan(state, MappingProxyType(blocks), MappingProxyType(preparations),
                            tuple(events), tuple(points), sum(preparation_costs.values()),
                            MappingProxyType(preparation_costs), host_peak, device_peak,
                            primary_peak <= memory_limit and (host_memory_limit is None or host_peak <= host_memory_limit),
                            memory_limit, host_memory_limit, reserved_host_bytes, reserved_device_bytes, profile,
                            disk_limit_bytes=disk_limit_bytes)
    from .packed_values import admit_packed_values
    from .allocation_rounding import account_allocation_rounding
    return admit_packed_values(account_allocation_rounding(prepare_coordinate_batch_workspace(plan)[0]))


def retain_storage_tensors(plans):
    """Keep owned logical arrays across batches under audited memory limits.

    Retention is explicit in each returned plan. Local release actions relinquish
    a batch reference, not the retained allocation. Eviction occurs only when
    the next batch's conservative headroom requires it; run teardown releases
    the remainder. Sparse borrowed records stay in the existing raw lifecycle.
    """
    plans=tuple(plans)
    if not plans: raise ValueError('at least one batch required')
    if any(p.retention_planned for p in plans):
        if not all(p.retention_planned for p in plans): raise ValueError('mixed retention schedules')
        return plans
    return tuple(iter_retain_storage_tensors(plans))


def iter_retain_storage_tensors(plans, *, prepare=None):
    """Apply the same cache policy lazily, before executing each batch."""
    specs={};costs={};source_definitions={};last_used={}
    live=set()
    for index,plan in enumerate(plans):
        if plan.kernel_batches:
            # Retention currently tracks individual logical allocations. The
            # offline cache planner selects ordinary kernels before rewriting
            # Actions, so no shared batch owner can be partly retained.
            plan=plan_scene_storage(plan.state,plan.blocks,plan.preparations,
                profile=plan.cost_profile,memory_limit=plan.memory_limit,
                host_memory_limit=plan.host_memory_limit,
                reserved_host_bytes=plan.reserved_host_bytes,
                reserved_device_bytes=plan.reserved_device_bytes)
        validate_executable_resources(plan)
        if prepare is not None: plan=prepare(plan)
        if plan.retention_planned: raise ValueError('stream requires unretained base plans')
        if not plan.feasible: raise MemoryError('base schedule exceeds memory limits')
        for action in plan.state.actions:
            key=action.tensor
            if action.kind not in ('load','contract','sparse_bridge','baryon_join') or key not in plan.state.catalog:
                continue
            spec=plan.state.catalog[key]
            if key in specs and specs[key]!=spec: raise ValueError('one tensor identity has incompatible shapes')
            specs[key]=spec
            if action.kind=='load':
                definition=(plan.state.problem.sources[key],plan.preparations[key])
                if key in source_definitions and source_definitions[key]!=definition:
                    raise ValueError('one source identity has incompatible definitions')
                source_definitions[key]=definition
            last_used[key]=index
            costs[key]=max(costs.get(key,0.),action.seconds+
                (plan.preparations[key].seconds(plan.cost_profile) if action.kind=='load' else 0.))
        gpu=plan.state.problem.backend=='cupy'
        headroom=plan.memory_limit-(plan.peak_device_bytes if gpu else plan.peak_host_bytes)
        if not gpu and plan.host_memory_limit is not None:
            headroom=min(headroom,plan.host_memory_limit-plan.peak_host_bytes)
        produced={a.tensor for a in plan.state.actions if a.kind in ('load','contract','sparse_bridge','baryon_join')
                  and a.tensor in specs}
        candidates=live|produced
        sizes={key:_nbytes(specs[key]) for key in candidates}
        # One admission decision, with no future-batch scan or path search.
        # If everything fits, preserve all storage without sorting/eviction.
        reserved=sum(sizes[key] for key in candidates)
        selected=set(candidates)
        if reserved>headroom:
            def eviction_order(key):
                return (key in produced, last_used.get(key,-1),
                        costs[key]/max(sizes[key],1),-sizes[key],key)
            for key in sorted(candidates,key=eviction_order):
                if reserved<=headroom: break
                selected.remove(key);reserved-=sizes[key]
        before=live&selected;available=set(before);actions=[];preparation_costs=dict(plan.preparation_costs)
        for action_index,action in enumerate(plan.state.actions):
            if action.kind in ('load','contract','sparse_bridge','baryon_join') and action.tensor in selected:
                if action.tensor in available:
                    action=replace(action,reused=True,seconds=0.)
                    if action.kind=='load': preparation_costs[action_index]=0.
                elif action.kind=='sparse_bridge' and not action.reused:
                    profile=plan.cost_profile
                    action=replace(action,seconds=action.seconds+profile.snapshot_seconds(sizes[action.tensor]))
                available.add(action.tensor)
            actions.append(action)
        # Deliberately reserve the whole selected cache for the whole batch.
        # This overcounts arrays already in local workspace, but guarantees a
        # feasible fixed path without hidden allocations or expensive replay.
        memory=tuple(replace(point,retained_host_bytes=0 if gpu else reserved,
                             retained_device_bytes=reserved if gpu else 0) for point in plan.memory)
        state=replace(plan.state,actions=tuple(actions),seconds=sum(a.seconds for a in actions))
        adjusted=replace(plan,state=state,memory=memory,
            preparation_costs=MappingProxyType(preparation_costs),preparation_seconds=sum(preparation_costs.values()),
            peak_host_bytes=max(p.host_bytes for p in memory),peak_device_bytes=max(p.device_bytes for p in memory),
            retained_before=tuple(sorted(before)),retained_after=tuple(sorted(selected)),retention_planned=True,
            retained_releases=tuple(sorted(live-before)))
        live=selected
        # Evicted tensors need no unbounded historical catalogue.
        specs={k:specs[k] for k in live};costs={k:costs[k] for k in live}
        source_definitions={k:v for k,v in source_definitions.items() if k in live}
        last_used={k:last_used[k] for k in live}
        yield adjusted


def iter_join_storage_batches(plans, demands):
    """Audit raw lifetimes lazily against an explicit full-task demand list.

    The caller supplies exact physical block sets per batch. A block is read at
    first use and released at its last action in its last batch. Later failures
    abort the stream; already consumed batches remain the caller's responsibility.
    """
    demands=tuple(frozenset(keys) for keys in demands)
    last={key:i for i,keys in enumerate(demands) for key in keys}
    resident=set();opened=set();definitions={};identities={};count=0;backend=None
    for batch,plan in enumerate(plans):
        initial_raw=tuple(resident)
        validate_executable_resources(plan)
        count+=1
        if batch>=len(demands): raise ValueError('more plans than declared demands')
        if backend is None: backend=plan.state.problem.backend
        if backend!=plan.state.problem.backend: raise ValueError('mixed execution backends')
        reads={e.block for e in plan.events if e.kind=='read'}
        closes={e.block for e in plan.events if e.kind=='release'}
        if reads!=demands[batch] or reads!=closes:
            raise ValueError('batch physical demand differs from declared full task')
        for key in reads|(resident&plan.blocks.keys()):
            block=plan.blocks[key]
            if key in definitions and definitions[key]!=block: raise ValueError('physical block changed')
            identity=(block.dataset,block.selection)
            if identity in identities and identities[identity]!=key: raise ValueError('duplicate physical selection')
            definitions[key]=block;identities[identity]=key
        new=reads-opened
        if (reads&opened)-resident: raise ValueError('attempt to reopen released input')
        events=tuple(e for e in plan.events if (e.block in new if e.kind=='read' else last[e.block]==batch))
        releases={}
        for event in events:
            if event.kind=='release': releases.setdefault(event.action_index,[]).append(event.block)
        points_by_action={}
        for point in plan.memory: points_by_action.setdefault(point.action_index,[]).append(point)
        points=[]
        for index in range(len(plan.state.actions)+1):
            for point in points_by_action.get(index,()):
                if point.phase.startswith('read:'):
                    key=point.phase[len('read:'):]
                    if key not in new: continue
                    resident.add(key);opened.add(key)
                points.append(replace(point,
                    raw_host_bytes=sum(definitions[k].host_bytes for k in resident),
                    raw_device_bytes=sum(definitions[k].device_bytes for k in resident)))
            for key in releases.get(index,()): resident.remove(key)
        host=max(p.host_bytes for p in points);device=max(p.device_bytes for p in points)
        feasible=(device if backend=='cupy' else host)<=plan.memory_limit and (plan.host_memory_limit is None or host<=plan.host_memory_limit)
        from .allocation_rounding import account_allocation_rounding
        joined=account_allocation_rounding(replace(plan,events=events,memory=tuple(points),
                      peak_host_bytes=host,peak_device_bytes=device,feasible=feasible),
                      initial_raw=initial_raw,raw_blocks=definitions)
        if not joined.feasible: raise MemoryError('fixed batch with shared inputs exceeds memory limits')
        yield joined
    if count!=len(demands) or resident: raise ValueError('incomplete streamed storage schedule')
