"""Represent supported fixed trajectories using existing execution events.

Temporary transfers keep their explicit metadata descriptor and remain blocked
by the execution/archive guards. No storage policy is reselected here.
"""
from dataclasses import replace
from types import MappingProxyType
from lqcdweave.contracts.correlator.api import Action, StorageEvent


def executable_delivery(plan):
    if not plan.explicit_resources:
        return plan
    if any(a.kind in ("spill_host", "spill_disk", "restore_host", "restore_device")
           or (a.kind == "discard" and a.seconds != 0) for a in plan.state.actions):
        return plan
    actions, events, remap, open_blocks = [], [], {}, set()
    for index, action in enumerate(plan.state.actions):
        if action.kind == "read_block":
            remap[index] = len(actions)
            events.append(StorageEvent(len(actions), "read", action.tensor))
            open_blocks.add(action.tensor)
        elif action.kind == "close_block" or (action.kind == "discard" and action.resource.domain == "block"):
            remap[index] = max(0, len(actions) - 1)
            events.append(StorageEvent(remap[index], "release", action.tensor))
            open_blocks.remove(action.tensor)
        else:
            remap[index] = len(actions)
            actions.append(Action("release", action.tensor) if action.kind == "discard" else action)
    if events and not actions:
        return plan  # A raw-only descriptor has no execution phase for reads.
    remap[len(plan.state.actions)] = len(actions)
    # Final cleanup occurs after output assembly; scene_runtime already calls
    # end_action(len(actions)). Caches are retained through the output peak.
    events.extend(StorageEvent(len(actions), "release", key) for key in sorted(open_blocks))
    state = replace(plan.state, actions=tuple(actions), seconds=sum(a.seconds for a in actions))
    memory = tuple(replace(point, action_index=remap[point.action_index]) for point in plan.memory)
    preparations = MappingProxyType({remap[index]: seconds for index, seconds in plan.preparation_costs.items()})
    lowered = replace(plan, state=state, events=tuple(events), memory=memory,
                      preparation_costs=preparations, explicit_resources=False)
    if lowered.seconds != plan.seconds:
        # Independent sums can differ at the last float bit; callers compare
        # with the same tolerance as the search/audit contract.
        import math
        if not math.isclose(lowered.seconds, plan.seconds, rel_tol=1e-11, abs_tol=1e-14):
            raise RuntimeError("delivery changed the audited total cost")
    # The explicit resource audit cannot charge ordinary CuPy request units
    # before read/close/discard descriptors become execution events. Refresh
    # optional packing headroom after that representation change as well.
    from ..allocation_rounding import account_allocation_rounding
    from ..packed_values import admit_packed_values
    if lowered.packed_value_cache_device_bytes:
        memory = tuple(replace(point, packed_value_cache_device_bytes=0)
                       for point in lowered.memory)
        lowered = replace(lowered, memory=memory, packed_value_cache_device_bytes=0,
                          peak_device_bytes=max((p.device_bytes for p in memory), default=0))
    return admit_packed_values(account_allocation_rounding(lowered))
