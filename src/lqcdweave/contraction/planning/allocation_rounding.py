"""Charge the known allocation owners in a fixed ordinary dense GPU route."""
from collections import Counter, defaultdict
from dataclasses import replace
from math import prod

from lqcdweave.contracts.correlator.api import device_allocation_bytes
from lqcdweave.contracts.correlator.gemm_group import dense_batch_leaves


def _padding(nbytes):
    return device_allocation_bytes(nbytes) - nbytes


def account_allocation_rounding(plan, *, initial_raw=(), raw_blocks=None):
    """Annotate CuPy request rounding, without changing algebra or lifetimes.

    Payload bytes and caller reserves retain their original meanings.  Each
    paid result/input/raw owner is charged once; shared result views charge
    their complete batch owner.  Output snapshots use the same distinct-key,
    shape/dtype groups as X.  This is not an allocator-fragmentation reserve.
    Opaque/core/sparse resource trajectories retain their declared envelopes.
    ``initial_raw`` and its complete ``raw_blocks`` registry are supplied by
    the cross-batch lifetime owner, including raw blocks skipped by this batch.
    """
    state = plan.state
    if (state.problem.backend != 'cupy' or plan.explicit_resources
            or any(a.kind not in ('load', 'release', 'contract') or a.core is not None
                   for a in state.actions)):
        return plan
    blocks = plan.blocks if raw_blocks is None else raw_blocks
    sizes = {key: prod(spec.shape) * spec.dtype.itemsize for key, spec in state.catalog.items()}
    allocations = {key: source.snapshot_bytes if source.snapshot_bytes is not None else sizes[key]
                   for key, source in state.problem.sources.items()}
    padding = {key: _padding(size) for key, size in sizes.items()}
    owned_padding = {**padding, **{key: _padding(size) for key, size in allocations.items()}}
    batches = tuple(dense_batch_leaves(plan.kernel_batches))
    starts = {batch.action_indices[0]: batch for batch in batches}
    outputs = {result.tensor: batch for batch in batches for result in batch.results}
    owners = Counter()
    points_by_action = defaultdict(list)
    for point in plan.memory:
        points_by_action[point.action_index].append(point)
    closes = defaultdict(list)
    for event in plan.events:
        if event.kind == 'release':
            closes[event.action_index].append(event.block)
    raw = set(initial_raw)
    raw_padding = sum(_padding(blocks[key].device_bytes) for key in raw)
    live, logical_padding, points = set(), 0, []
    snapshot_groups = Counter()
    for key in set(state.results):
        spec = state.catalog[key]
        snapshot_groups[spec.shape, spec.dtype.str] += sizes[key]
    snapshot_padding = sum(_padding(size) for size in snapshot_groups.values())

    for index in range(len(state.actions) + 1):
        before = logical_padding
        action = None if index == len(state.actions) else state.actions[index]
        batch = starts.get(index)
        if batch is not None:
            logical_padding += _padding(batch.output_bytes)
            owners[id(batch)] = len(batch.results)
        if action is not None:
            key = action.tensor
            if action.kind == 'load' or action.kind == 'contract' and not action.reused:
                if key not in outputs:
                    logical_padding += owned_padding[key]
                live.add(key)
            elif action.kind == 'release':
                if key in live:
                    live.remove(key)
                    owner = outputs.get(key)
                    if owner is None:
                        logical_padding -= owned_padding[key]
                    else:
                        owners[id(owner)] -= 1
                        if not owners[id(owner)]:
                            logical_padding -= _padding(owner.output_bytes)
        for point in points_by_action.get(index, ()):
            if point.phase.startswith('read:'):
                key = point.phase[len('read:'):]
                if key not in raw:
                    raw.add(key)
                    raw_padding += _padding(blocks[key].device_bytes)
                # Reads precede the Action/materialization at the same index.
                extra = raw_padding + before + _padding(blocks[key].staging_device_bytes)
            else:
                extra = raw_padding + logical_padding
                if point.phase == 'contract' and action is not None and not action.reused:
                    # Ordinary dense workspace permits one layout copy per
                    # input and one extra output. Round those slots separately.
                    extra += sum(padding[key] for key in action.operands)
                    if action.tensor not in outputs:
                        extra += padding[action.tensor]
                elif point.phase.startswith('materialize:'):
                    # A declared aggregate workspace is already a byte bound;
                    # this adds its one request unit, not invented temporaries.
                    extra += _padding(point.workspace_device_bytes)
                elif point.phase == 'output_copy':
                    extra += snapshot_padding
            points.append(replace(point, allocation_rounding_device_bytes=extra))
        for key in closes.get(index, ()):
            if key in raw:
                raw.remove(key)
                raw_padding -= _padding(blocks[key].device_bytes)
    memory = tuple(points)
    peak = max(point.device_bytes for point in memory)
    return replace(plan, memory=memory, peak_device_bytes=peak,
        feasible=peak <= plan.memory_limit and (plan.host_memory_limit is None
            or plan.peak_host_bytes <= plan.host_memory_limit))


__all__ = ['account_allocation_rounding']
