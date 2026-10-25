"""Pure resource transitions and fixed-trajectory audit; no IO or search."""
from dataclasses import replace
from collections import Counter
import math
from types import MappingProxyType
from lqcdweave.contracts.correlator.api import (
    ResourceAllocation, ResourceSnapshot, MemoryPoint, SceneStoragePlan, StorageEvent, Action,
    RESOURCE_ACTION_KINDS, COMPUTE_ACTION_KINDS, MatrixKernelSpec,
)
from .state import ContractionState, _nbytes
from .actions import _apply_selected, _live_inputs
from .sparse_workspace import prepare_coordinate_batch_workspace

_TRANSFERS = {"spill_host": ("device", "host"), "spill_disk": ("host", "disk"),
              "restore_host": ("disk", "host"), "restore_device": ("host", "device")}


def resource_point(snapshot, index, phase, *, workspace_host=0, workspace_device=0,
                   reserved_host=0, reserved_device=0):
    """Count every owner exactly once; borrowing contributes no extra bytes."""
    raw = [a for a in snapshot.allocations if a.domain == "block"]
    tensors = [a for a in snapshot.allocations if a.domain == "tensor"]
    return MemoryPoint(index, phase, sum(a.host_bytes for a in raw),
        sum(a.device_bytes for a in raw), sum(a.host_bytes for a in tensors),
        sum(a.device_bytes for a in tensors), workspace_host, workspace_device,
        reserved_host, reserved_device, temporary_disk_bytes=sum(a.disk_bytes for a in snapshot.allocations))


def _check(point, *, backend, memory_limit, host_memory_limit=None, disk_limit_bytes=None):
    if ((point.device_bytes if backend == "cupy" else point.host_bytes) > memory_limit
            or (host_memory_limit is not None and point.host_bytes > host_memory_limit)
            or (disk_limit_bytes is not None and point.temporary_disk_bytes > disk_limit_bytes)):
        error = MemoryError("resource transition exceeds declared storage limit")
        error.point = point
        raise error
    if backend == "numpy" and point.device_bytes:
        raise ValueError("CPU resources cannot occupy device memory")


def transition_resource(snapshot, action, *, future_needed=frozenset(),
                        recoverable_sources=frozenset(), backend, memory_limit,
                        host_memory_limit=None, disk_limit_bytes=None,
                        reserved_host_bytes=0, reserved_device_bytes=0, action_index=0):
    """Immutable move: validate source, charge overlap peak, then remove source.

    Recovery facts are supplied by the operation domain from live preparation
    dependencies, never by an Action flag. On error the input remains unchanged.
    """
    transition = action.resource
    if transition is None or action.kind not in _TRANSFERS.keys() | {"discard"}:
        raise ValueError("resource move requires a declared transition")
    if action != Action(action.kind, action.tensor, seconds=action.seconds, resource=transition):
        raise ValueError("resource action has extraneous compute fields")
    if action.seconds != 0.0:
        raise ValueError("resource actions have no predicted time cost")
    identity = (transition.domain, action.tensor)
    allocation = next((a for a in snapshot.allocations if (a.domain, a.key) == identity), None)
    if allocation is None:
        borrowed = dict(snapshot.borrowed).get(action.tensor) if transition.domain == "tensor" else None
        if (borrowed is None or action.kind != "discard" or transition.target != "absent"
                or transition.nbytes != 0):
            raise ValueError("resource action requires an owned allocation; borrowed views can only be discarded")
        if identity in future_needed and identity not in recoverable_sources:
            raise ValueError("future-needed borrowed view has no declared recovery route")
        peak = resource_point(snapshot, action_index, "discard:" + action.tensor,
            workspace_host=transition.workspace_host_bytes, workspace_device=transition.workspace_device_bytes,
            reserved_host=reserved_host_bytes, reserved_device=reserved_device_bytes)
        _check(peak, backend=backend, memory_limit=memory_limit,
               host_memory_limit=host_memory_limit, disk_limit_bytes=disk_limit_bytes)
        return replace(snapshot, borrowed=tuple(item for item in snapshot.borrowed if item[0] != action.tensor)), peak
    source_bytes = getattr(allocation, transition.source + "_bytes")
    total_bytes = allocation.host_bytes + allocation.device_bytes + allocation.disk_bytes
    if total_bytes == 0 and (action.kind != "discard" or transition.source !=
                            ("device" if backend == "cupy" else "host")):
        raise ValueError("zero-byte owners stay valid at the native tier and cannot be moved")
    if (source_bytes != transition.nbytes
            or total_bytes != source_bytes):
        raise ValueError("transition must move the complete declared allocation")
    borrowers = {key for key, owners in snapshot.borrowed if action.tensor in owners}
    if transition.domain == "block" and borrowers:
        raise ValueError("cannot move or discard a borrowed raw owner")
    if action.kind == "discard":
        if transition.target != "absent":
            raise ValueError("discard target must be absent")
        if identity in future_needed and identity not in recoverable_sources:
            raise ValueError("future-needed allocation has no declared recovery route")
        target = None
        transient = snapshot
    else:
        if (transition.source, transition.target) != _TRANSFERS[action.kind]:
            raise ValueError("resource move has an unsupported tier pair")
        target = replace(allocation, **{transition.source + "_bytes": 0,
                                      transition.target + "_bytes": source_bytes})
        # Temporary overlap is represented on the same owner, without a second IR.
        overlap = replace(allocation, **{transition.target + "_bytes": source_bytes})
        transient = replace(snapshot, allocations=tuple(overlap if a == allocation else a for a in snapshot.allocations))
    peak = resource_point(transient, action_index, action.kind + ":" + action.tensor,
        workspace_host=transition.workspace_host_bytes, workspace_device=transition.workspace_device_bytes,
        reserved_host=reserved_host_bytes, reserved_device=reserved_device_bytes)
    _check(peak, backend=backend, memory_limit=memory_limit,
           host_memory_limit=host_memory_limit, disk_limit_bytes=disk_limit_bytes)
    allocations = tuple(a for a in snapshot.allocations if a != allocation)
    if target is not None:
        allocations += (target,)
    return replace(snapshot, allocations=allocations), peak


def audit_resource_trajectory(state, blocks, preparations, *, profile, memory_limit,
                              host_memory_limit=None, disk_limit_bytes=None,
                              reserved_host_bytes=0, reserved_device_bytes=0,
                              checkpoint=lambda: None):
    """Replay the explicit action sequence, preserving every raw close and move.

    Raw reads/closes are explicit read_block/close_block Actions. Resource
    moves are checked for ownership and capacity, without assigning durations.
    The returned execution-facing descriptor is the existing SceneStoragePlan.
    """
    for name, value in (("memory_limit", memory_limit), ("host_memory_limit", host_memory_limit),
                        ("disk_limit_bytes", disk_limit_bytes)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be positive integer bytes or None")
    if type(memory_limit) is not int or memory_limit <= 0:
        raise ValueError("memory_limit is required")
    if any(type(v) is not int or v < 0 for v in (reserved_host_bytes, reserved_device_bytes)):
        raise ValueError("reserved memory must be nonnegative integer bytes")
    blocks, preparations = dict(blocks), dict(preparations)
    if len({(b.dataset, b.selection) for b in blocks.values()}) != len(blocks):
        raise ValueError("one physical selection must have one owner")
    if not state.done:
        raise ValueError("audit requires a complete trajectory")
    if any(s.region_bytes or s.host_region_bytes or s.staging_bytes or s.transfer_bytes
           or s.read_bytes != 0 for s in state.problem.sources.values()):
        raise ValueError("explicit audit requires logical_source_problem")
    from .storage import logical_cost_profile
    model = logical_cost_profile(profile)
    logical = ContractionState.initial(state.problem)
    from .canonical_operations import PREFIX
    # These two replay domains update exactly the named Term. All other
    # domains retain the general liveness scan, including bridges and joins.
    counted = (not state.problem.sparse_bridges and not state.problem.baryon_joins
        and all(a.kind not in COMPUTE_ACTION_KINDS or a.kind in ('load','release')
                or a.kind=='contract' and (a.tensor.startswith(PREFIX) or isinstance(a.core,MatrixKernelSpec))
                for a in state.actions))
    references = Counter(f.tensor for term in logical.pending if term is not None
                         for f in term.factors) if counted else None
    if references is not None:
        object.__setattr__(logical,'live_inputs',frozenset(references))
    owners = {}
    borrowed, opened = {}, set()
    totals = [0, 0, 0, 0, 0]
    points, events, costs = [], [], {}
    def set_owner(domain, key, allocation=None):
        identity = (domain, key)
        old = owners.pop(identity, None)
        offset = 0 if domain == 'block' else 2
        if old is not None:
            totals[offset] -= old.host_bytes
            totals[offset + 1] -= old.device_bytes
            totals[4] -= old.disk_bytes
        if allocation is not None:
            owners[identity] = allocation
            totals[offset] += allocation.host_bytes
            totals[offset + 1] += allocation.device_bytes
            totals[4] += allocation.disk_bytes
    def point(index, phase, wh=0, wd=0):
        p = MemoryPoint(index, phase, *totals[:4], wh, wd,
            reserved_host_bytes, reserved_device_bytes, temporary_disk_bytes=totals[4])
        points.append(p)
        return p
    def owner(domain, key):
        return owners.get((domain,key))
    def usable(key):
        allocation = owner("tensor", key)
        if allocation is not None:
            # Allocation presence carries validity. A zero-byte tensor is still
            # an owned, native-tier value; zero-byte moves are forbidden.
            native_bytes = allocation.device_bytes if state.problem.backend == "cupy" else allocation.host_bytes
            return (native_bytes > 0 or
                    allocation.host_bytes + allocation.device_bytes + allocation.disk_bytes == 0)
        dependencies = borrowed.get(key)
        # Borrowed records consume raw owners in the positions declared by the
        # adapter, which may include host payloads on a GPU scene. Empty owners
        # are valid too: byte count never stands in for owner existence.
        return dependencies is not None and all(owner("block", dep) is not None
            and owner("block", dep).host_bytes == blocks[dep].host_bytes
            and owner("block", dep).device_bytes == blocks[dep].device_bytes
            and owner("block", dep).disk_bytes == 0 for dep in dependencies)
    for index, action in enumerate(state.actions):
        checkpoint()
        if action.kind not in RESOURCE_ACTION_KINDS | COMPUTE_ACTION_KINDS:
            raise ValueError(f"unknown action {action.kind!r}")
        if action.kind in ("read_block", "close_block"):
            if action.resource is not None or action.tensor not in blocks:
                raise ValueError("raw lifecycle actions require a declared block and no move descriptor")
            if action != Action(action.kind, action.tensor, seconds=action.seconds):
                raise ValueError("raw lifecycle action has extraneous fields")
            if action.seconds != 0.0:
                raise ValueError("raw lifecycle actions have no predicted time cost")
            key, block = action.tensor, blocks[action.tensor]
            if action.kind == "read_block":
                if key in opened:
                    raise ValueError("original blocks may only be read once")
                opened.add(key)
                set_owner('block', key, ResourceAllocation("block", key, block.host_bytes, block.device_bytes))
                point(index, "read:" + key, block.staging_host_bytes, block.staging_device_bytes)
                events.append(StorageEvent(index, "read", key))
            else:
                allocation = owner("block", key)
                future_missing = {source for source in _live_inputs(logical)
                                  if owner("tensor", source) is None and source not in borrowed}
                if any(key in preparations[source].blocks for source in future_missing if source in preparations):
                    raise ValueError("close would remove a future source recovery dependency")
                if allocation is None or any(key in deps for deps in borrowed.values()):
                    raise ValueError("close requires a live raw owner without borrowers")
                if action.seconds != 0:
                    raise ValueError("raw close must have zero cost")
                set_owner('block', key)
                point(index, "close:" + key); events.append(StorageEvent(index, "release", key))
            continue
        if action.kind in _TRANSFERS.keys() | {"discard"}:
            required = _live_inputs(logical) | {key for key in logical.results if key is not None}
            future = {("tensor", key) for key in required}
            future.update(("block", dep) for key in required if key in preparations
                          for dep in preparations[key].blocks
                          if owner("tensor", key) is None and key not in borrowed)
            recoverable = {("tensor", key) for key in required if key in preparations and
                all(owner("block", dep) is not None for dep in preparations[key].blocks)}
            snapshot = ResourceSnapshot(tuple(owners.values()), tuple(borrowed.items()), frozenset(opened))
            snapshot, peak = transition_resource(snapshot, action, future_needed=future,
                recoverable_sources=recoverable, backend=state.problem.backend, memory_limit=memory_limit,
                host_memory_limit=host_memory_limit, disk_limit_bytes=disk_limit_bytes,
                reserved_host_bytes=reserved_host_bytes, reserved_device_bytes=reserved_device_bytes, action_index=index)
            identity=(action.resource.domain,action.tensor)
            allocation=next((a for a in snapshot.allocations if (a.domain,a.key)==identity),None)
            set_owner(*identity, allocation)
            borrowed = dict(snapshot.borrowed)
            points.append(peak)
            if action.kind == "discard" and action.resource.domain == "tensor":
                logical = replace(logical, loaded=logical.loaded - {action.tensor})
            continue
        if action.resource is not None:
            raise ValueError("compute actions cannot carry a resource move")
        # The supplied trajectory owns history. Independent algebra replay only
        # needs this action's fragment, not a copied prefix for every step.
        before = logical._seed_residency(replace(logical, actions=(),peak_bytes=0,
            peak_host_bytes=0, peak_device_bytes=0),0)
        if 'live_inputs' in logical.__dict__:
            object.__setattr__(before,'live_inputs',logical.live_inputs)
        if action.kind == "load":
            prep = preparations.get(action.tensor)
            if prep is None or any(dep not in blocks for dep in prep.blocks):
                raise ValueError("load requires declared preparation dependencies")
            if any(owner("block", dep) is None or owner("block", dep).disk_bytes
                   or owner("block", dep).host_bytes != blocks[dep].host_bytes
                   or owner("block", dep).device_bytes != blocks[dep].device_bytes for dep in prep.blocks):
                raise ValueError("load requires available raw blocks")
            logical = before.load(action.tensor, model, 1 << 100)
            costs[index] = prep.seconds(profile)
            size = logical._snapshot_bytes(action.tensor)
            if action.tensor in logical.catalog:
                set_owner('tensor', action.tensor, ResourceAllocation("tensor", action.tensor,
                    size if state.problem.backend == "numpy" else 0,
                    size if state.problem.backend == "cupy" else 0))
            else:
                borrowed[action.tensor] = prep.blocks
            point(index, "materialize:" + action.tensor, prep.workspace_host_bytes, prep.workspace_device_bytes)
        elif action.kind == "release":
            if action.tensor in _live_inputs(logical) or action.tensor in logical.results:
                raise ValueError("explicit release cannot destroy future-needed data")
            logical = before.release(action.tensor)
            set_owner('tensor', action.tensor)
            borrowed.pop(action.tensor, None)
            point(index, "release:" + action.tensor)
        else:
            if action.reused:
                if action.tensor not in before.loaded or not usable(action.tensor):
                    raise ValueError("cached output must be restored before reuse")
                # Replay the existing algebraic substitution with descriptor-only
                # operands. No input preparation, allocation or compute occurs.
                missing = set(action.operands) - before.loaded
                added = sum(before._snapshot_bytes(key) if key in before.problem.sources
                            else _nbytes(before.catalog[key]) for key in missing)
                symbolic = before._seed_residency(
                    replace(before, loaded=before.loaded | missing), added)
                logical = _apply_selected(symbolic, action.term_index, action, model, 1 << 100)
                if logical is not None:
                    logical = before._seed_residency(replace(logical, loaded=before.loaded | {action.tensor}),0)
            else:
                if any(not usable(key) for key in action.operands):
                    raise ValueError("compute operands must be restored before use")
                logical = _apply_selected(before, action.term_index, action, model, 1 << 100)
            if logical is None:
                raise ValueError("invalid selected compute action")
            if action.tensor not in before.loaded:
                size = _nbytes(logical.catalog[action.tensor])
                set_owner('tensor', action.tensor, ResourceAllocation("tensor", action.tensor,
                    size if state.problem.backend == "numpy" else 0,
                    size if state.problem.backend == "cupy" else 0))
            workspace = 0 if action.reused else max(0, logical.peak_bytes - max(before.resident_bytes, logical.resident_bytes))
            point(index, action.kind, workspace if state.problem.backend == "numpy" else 0,
                  workspace if state.problem.backend == "cupy" else 0)
        if logical.actions[-1] != action:
            raise ValueError("action fields disagree with selected transition")
        if references is not None:
            if action.kind not in ('load','release'):
                old,new=before.pending[action.term_index],logical.pending[action.term_index]
                for factor in (() if old is None else old.factors):
                    references[factor.tensor]-=1
                    if references[factor.tensor]==0:
                        del references[factor.tensor]
                references.update(f.tensor for f in (() if new is None else new.factors))
                object.__setattr__(logical,'live_inputs',frozenset(references))
            else:
                object.__setattr__(logical,'live_inputs',before.live_inputs)
    if (not logical.done or logical.pending != state.pending or logical.results != state.results
            or dict(logical.catalog) != dict(state.catalog) or logical.loaded != state.loaded
            or logical.completed_joins != state.completed_joins or logical.opened_regions != state.opened_regions):
        raise ValueError("trajectory final state differs from declared state")
    if not math.isclose(state.seconds, sum(a.seconds for a in state.actions), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("trajectory cumulative cost disagrees with actions")
    if any(not usable(key) for key in logical.results):
        raise ValueError("outputs must be restored before final assembly")
    copies = sum(_nbytes(logical.catalog[key]) for key in logical.results)
    checkpoint()
    point(len(state.actions), "output_copy", copies if state.problem.backend == "numpy" else 0,
          copies if state.problem.backend == "cupy" else 0)
    host = max(p.host_bytes for p in points); device = max(p.device_bytes for p in points)
    disk = max(p.temporary_disk_bytes for p in points)
    feasible = all((p.device_bytes if state.problem.backend == "cupy" else p.host_bytes) <= memory_limit
        and (host_memory_limit is None or p.host_bytes <= host_memory_limit)
        and (disk_limit_bytes is None or p.temporary_disk_bytes <= disk_limit_bytes) for p in points)
    if any(not math.isfinite(a.seconds) or a.seconds < 0 for a in state.actions):
        raise ValueError("all action costs must be finite and nonnegative")
    if state.problem.backend == "numpy" and any(p.device_bytes for p in points):
        raise ValueError("CPU trajectory contains device allocations")
    # Action seconds contain only snapshot preparation and numerical operations.
    plan = SceneStoragePlan(state, MappingProxyType(blocks), MappingProxyType(preparations),
        tuple(events), tuple(points), sum(costs.values()), MappingProxyType(costs),
        host, device, feasible, memory_limit, host_memory_limit, reserved_host_bytes,
        reserved_device_bytes, profile, peak_disk_bytes=disk, disk_limit_bytes=disk_limit_bytes,
        explicit_resources=True)
    from .packed_values import admit_packed_values
    from .allocation_rounding import account_allocation_rounding
    return admit_packed_values(account_allocation_rounding(prepare_coordinate_batch_workspace(plan)[0]))
