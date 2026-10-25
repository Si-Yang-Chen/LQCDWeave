"""Independent Cartesian reduction submissions, preserving each child owner.

This annotation groups existing complete DenseKernelBatch records, rather than
changing their logical Actions or merging their result cubes. Resource checks
remain pure metadata; execution must independently admit the native wide path.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import isfinite, prod

import numpy as np

from .dense_batch import DenseKernelBatch
from .allocation import device_allocation_bytes


@dataclass(frozen=True)
class CartesianReductionGroup:
    batch_action_indices: tuple[tuple[int, ...], ...]
    workspace_bytes: int
    seconds: float

    def __post_init__(self):
        batches = tuple(tuple(indices) for indices in self.batch_action_indices)
        object.__setattr__(self, 'batch_action_indices', batches)
        if (not 2 <= len(batches) <= 8
                or any(len(indices) < 2 or any(type(i) is not int or i < 0 for i in indices)
                       or any(a >= b for a, b in zip(indices, indices[1:])) for indices in batches)
                or any(a[0] >= b[0] for a, b in zip(batches, batches[1:]))
                or len({i for indices in batches for i in indices}) != sum(map(len, batches))):
            raise ValueError('Cartesian groups require 2–8 distinct ordered complete batch indices')
        if type(self.workspace_bytes) is not int or self.workspace_bytes <= 0:
            raise ValueError('Cartesian group workspace must be positive integer bytes')
        if type(self.seconds) not in (int, float) or not isfinite(self.seconds) or self.seconds < 0:
            raise ValueError('Cartesian group cost must be finite and nonnegative')

    @property
    def action_indices(self):
        return tuple(sorted(index for indices in self.batch_action_indices for index in indices))


@dataclass(frozen=True)
class CartesianReductionGeometry:
    shapes: tuple[tuple[int, int], tuple[int, int]]
    dtype: np.dtype
    allowance: int
    blocks: int
    partial_bytes: int
    output_shape: tuple[int, int]


def cartesian_reduction_geometry(batch, catalog):
    """Reproduce the original ik,jk wide 2×2 admission using metadata only.

    The allowance is the smallest *original* member's two inputs and output,
    not the packed matrices. Partial blocks and wide precision are unchanged.
    """
    if (type(batch) is not DenseKernelBatch or not all(batch.stacked_sides)
            or batch.packing_dtype is None or batch.left_free_rank != 1
            or batch.right_free_rank != 1):
        raise ValueError('Cartesian reduction groups require complete stacked DenseKernelBatch leaves')
    dtype = np.dtype(batch.packing_dtype)
    if dtype not in (np.dtype('float32'), np.dtype('complex64')):
        raise ValueError('Cartesian reduction groups require float32 or complex64')
    inner = None
    sizes = []
    for side in (batch.left, batch.right):
        current = []
        for operand in side:
            spec = catalog.get(operand.tensor)
            if (spec is None or np.dtype(spec.dtype) != dtype
                    or len(spec.shape) != len(operand.permutation)
                    or len(spec.shape) != batch.reduction_rank):
                raise ValueError('Cartesian input dtype or reduction rank differs from its batch')
            shape = tuple(spec.shape[i] for i in operand.permutation)
            if any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError('Cartesian reduction dimensions must be positive integers')
            if inner is not None and shape != inner:
                raise ValueError('Cartesian reduction inputs have different saved K geometry')
            inner = shape
            current.append(prod(shape) * dtype.itemsize)
        sizes.append(min(current))
    m, n, k = len(batch.left), len(batch.right), prod(inner)
    outputs = m * n
    if (k < 2 or k > 2**63 - 1 or outputs <= 1 or m > 65535 or n > 65535
            or batch.output_bytes != outputs * dtype.itemsize
            or any(prod(stop - start for start, stop in r.bounds) != 1 for r in batch.results)):
        raise ValueError('Cartesian reduction geometry is outside the native 2×2 domain')
    allowance = sum(sizes) + dtype.itemsize
    wide_bytes = 2 * dtype.itemsize
    blocks = min(1024, (k + 255) // 256,
                 (allowance - outputs * dtype.itemsize) // (outputs * wide_bytes))
    if blocks < 1 or outputs * blocks >= 2**31:
        raise ValueError('Cartesian reduction cannot retain its original wide block count')
    return CartesianReductionGeometry(((m, k), (n, k)), dtype, allowance, blocks,
                                      outputs * blocks * wide_bytes, (m, n))


def describe_cartesian_reduction_group(batches, batch_action_indices, catalog, seconds):
    """Describe matching top-level leaves; never unpack a DenseGemmGroup."""
    record = CartesianReductionGroup(batch_action_indices, 1, seconds)
    by_indices = {batch.action_indices: batch for batch in batches if type(batch) is DenseKernelBatch}
    try:
        children = tuple(by_indices[indices] for indices in record.batch_action_indices)
    except KeyError as error:
        raise ValueError('Cartesian group requires complete top-level batch members') from error
    geometry = tuple(cartesian_reduction_geometry(child, catalog) for child in children)
    if any(item != geometry[0] for item in geometry[1:]):
        raise ValueError('Cartesian group children have different native reduction geometry')
    # The grouped launch stacks child tile-row grids; individual admission alone
    # does not guarantee that the combined CUDA grid.z remains representable.
    if len(children) * ((geometry[0].output_shape[0] + 1) // 2) > 65535:
        raise ValueError('Cartesian group tile-row grid exceeds the CUDA limit')
    return replace(record, workspace_bytes=sum(item.partial_bytes for item in geometry))


def cartesian_reduction_group_workspace(children, catalog, workspace_bytes):
    """Additional group-start allocations above the unchanged child envelope.

    Partials share one owner. Each nonprepared packing side is a separate
    allocation, even when its size is smaller than the pool's request unit.
    The first child's payload packing already belongs to its original point;
    charge only its padding here. Existing allocation-rounding charges remain
    untouched, so this first-child padding can deliberately be counted twice.
    No assumption is made that original member scratch padding covers packing.
    """
    children = tuple(children)
    if not children:
        raise ValueError('Cartesian group workspace requires child batches')
    packed = []
    for child in children:
        dtype = np.dtype(child.packing_dtype)
        sizes = tuple(sum(prod(catalog[o.tensor].shape) for o in side) * dtype.itemsize
                      for side, prepared in zip((child.left, child.right), child.prepared_sides)
                      if not prepared)
        if sum(sizes) != child.packing_bytes:
            raise ValueError('Cartesian packing payload differs from its nonprepared sides')
        packed.append(sum(device_allocation_bytes(size) for size in sizes))
    return device_allocation_bytes(workspace_bytes) + sum(packed) - children[0].packing_bytes


def cartesian_reduction_group_memory(plan, *, check=False, subtract=False):
    """Add simultaneous packing/partials and early independent child cubes.

    The existing scratch, child packing and dense owner buckets are untouched.
    At the group contract point, all partials plus packing of later children are
    added, including conservative first-child packing padding. The common
    partial owner, each packing side and each child cube are rounded separately
    to device allocation units. Each later full cube
    stays in this bucket from that point until its original batch contract;
    that original owner's trajectory then takes over. Read phases precede the
    contract at their Action. Check accepts conservative cfg-filtered envelopes;
    construction is exact and subtraction preserves any conservative excess.
    """
    batches = {batch.action_indices: batch for batch in plan.kernel_batches}
    starts, changes = {}, defaultdict(int)
    for group in plan.cartesian_reduction_groups:
        try:
            children = tuple(batches[indices] for indices in group.batch_action_indices)
        except KeyError as error:
            raise ValueError('Cartesian memory references an absent complete batch') from error
        first = group.batch_action_indices[0][0]
        starts[first] = cartesian_reduction_group_workspace(children, plan.state.catalog, group.workspace_bytes)
        for batch in children[1:]:
            amount = device_allocation_bytes(batch.output_bytes)
            changes[first] += amount
            changes[batch.action_indices[0]] -= amount
    additions, live = {}, 0
    for index in range(len(plan.state.actions) + 1):
        before = live
        live += changes[index]
        additions[index] = before, live
    points = []
    for point in plan.memory:
        if type(point.action_index) is not int or point.action_index not in additions:
            raise ValueError('Cartesian memory point has an unknown Action index')
        required = additions[point.action_index][0 if point.phase.startswith('read:') else 1]
        if point.phase == 'contract':
            required += starts.get(point.action_index, 0)
        recorded = point.cartesian_reduction_device_bytes
        if type(recorded) is not int or recorded < 0:
            raise ValueError('Cartesian reduction memory bucket must be nonnegative integer bytes')
        if check:
            if recorded < required:
                raise ValueError('Cartesian reduction memory omits simultaneous packing/partials or early cubes')
            points.append(point)
        else:
            amount = recorded - required if subtract else required
            if amount < 0:
                raise ValueError('cannot subtract an absent Cartesian reduction envelope')
            points.append(replace(point, cartesian_reduction_device_bytes=amount))
    return tuple(points)


def validate_cartesian_reduction_groups(plan):
    """Audit inputs against original dense publication, before new grouping.

    Native execution may reject any new group and replay its original children.
    A later group must therefore never depend on this annotation's early output
    timing. Existing DenseGemmGroup publication is part of the original path.
    """
    groups = plan.cartesian_reduction_groups
    if not groups:
        cartesian_reduction_group_memory(plan, check=True)
        return
    if (plan.state.problem.backend != 'cupy' or plan.explicit_resources or plan.retention_planned
            or plan.retained_before or plan.retained_after or plan.retained_releases):
        raise NotImplementedError('Cartesian reduction groups require an unretained fixed CuPy plan')
    actions, catalog = plan.state.actions, plan.state.catalog
    batches = {batch.action_indices: batch for batch in plan.kernel_batches}
    scalar_indices = {index for group in plan.scalar_reduction_groups for index in group.action_indices}
    boundaries = [0]
    for action in actions:
        boundaries.append(boundaries[-1] + int(action.kind not in ('load', 'release', 'contract')
                          or action.core is not None or action.resource is not None))
    grouped_indices, grouped_batches, starts = set(), set(), {}
    for group in groups:
        if type(group) is not CartesianReductionGroup:
            raise TypeError('Cartesian groups require CartesianReductionGroup records')
        if grouped_indices.intersection(group.action_indices) or scalar_indices.intersection(group.action_indices):
            raise ValueError('one paid Action cannot belong to overlapping reduction groups')
        described = describe_cartesian_reduction_group(plan.kernel_batches, group.batch_action_indices,
                                                       catalog, group.seconds)
        if described.workspace_bytes != group.workspace_bytes:
            raise ValueError('Cartesian group partial workspace disagrees with original child blocks')
        if any(index >= len(actions) for index in group.action_indices):
            raise ValueError('Cartesian group references an unknown Action')
        for indices in group.batch_action_indices:
            child = batches[indices]
            for index, result in zip(indices, child.results, strict=True):
                action = actions[index]
                if (action.kind != 'contract' or action.reused or action.core is not None
                        or action.resource is not None or action.tensor != result.tensor):
                    raise ValueError('Cartesian group child contains a reused or nonordinary member')
        if boundaries[group.action_indices[-1] + 1] != boundaries[group.action_indices[0]]:
            raise ValueError('Cartesian group crosses a sparse, core or resource boundary')
        grouped_indices.update(group.action_indices)
        grouped_batches.update(group.batch_action_indices)
        starts[group.action_indices[0]] = tuple(batches[indices] for indices in group.batch_action_indices)

    # Other physical annotations retain their own early publication time. A
    # logical reused Action only hands off an owner at its explicit position.
    physical_indices = {i for batch in plan.kernel_batches for i in batch.action_indices} | scalar_indices
    dense_starts = {batch.action_indices[0]: batch for batch in plan.kernel_batches}
    scalar_starts = {group.action_indices[0]: group for group in plan.scalar_reduction_groups}
    live = set()
    for index, action in enumerate(actions):
        children = starts.get(index)
        batch, scalar = dense_starts.get(index), scalar_starts.get(index)
        if children is not None:
            required = tuple(o.tensor for child in children for o in (*child.left, *child.right))
            results = {r.tensor for child in children for r in child.results}
            if any(key not in live for key in required):
                raise ValueError('Cartesian group physical inputs must be resident before the first child')
            if results.intersection(required):
                raise ValueError('Cartesian group contains an internal data dependency')
        if batch is not None:
            inputs = tuple(o.tensor for o in (*batch.left, *batch.right))
            outputs = tuple(r.tensor for r in batch.results)
        elif scalar is not None:
            inputs = tuple(key for i in scalar.action_indices for key in actions[i].operands)
            outputs = tuple(actions[i].tensor for i in scalar.action_indices)
        else:
            inputs = outputs = ()
        if any(key not in live for key in inputs):
            raise ValueError('Cartesian group physical inputs must be resident before the first child')
        if set(inputs).intersection(outputs):
            raise ValueError('Cartesian group contains an internal data dependency')
        live.update(outputs)
        if index in physical_indices:
            continue
        if action.kind == 'release':
            live.discard(action.tensor)
        elif action.reused:
            live.add(action.tensor)
        elif action.kind in ('load', 'contract', 'sparse_bridge', 'baryon_join'):
            if any(key not in live for key in action.operands):
                raise ValueError('Cartesian group schedule consumes an absent allocation')
            live.add(action.tensor)
    phases = Counter((point.action_index, point.phase) for point in plan.memory)
    if any(phases[indices[0], 'contract'] != 1 for indices in grouped_batches):
        raise ValueError('Cartesian group memory omits an original child contract phase')
    cartesian_reduction_group_memory(plan, check=True)
    if (not plan.memory or plan.peak_host_bytes != max(p.host_bytes for p in plan.memory)
            or plan.peak_device_bytes != max(p.device_bytes for p in plan.memory)):
        raise ValueError('Cartesian group resource peaks disagree with memory points')
    feasible = (plan.peak_device_bytes <= plan.memory_limit
                and (plan.host_memory_limit is None or plan.peak_host_bytes <= plan.host_memory_limit)
                and (plan.disk_limit_bytes is None or plan.peak_disk_bytes <= plan.disk_limit_bytes))
    if plan.feasible != feasible:
        raise ValueError('Cartesian group feasibility disagrees with resource limits')
