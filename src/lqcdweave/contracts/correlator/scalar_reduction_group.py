"""Independent terminal scalar reductions and their additive memory envelope.

These are prospective physical annotations, not a planner or kernel binding.
Numerical execution must still satisfy the original wide-reduction admission.
"""
from dataclasses import dataclass, replace
from collections import Counter, defaultdict
from math import isfinite, prod

import numpy as np
from .allocation import device_allocation_bytes


def scalar_reduction_group_limit(shapes):
    """Largest scalar ABI capacity within the conservative 3072-byte envelope.

    Each fixed slot carries two pointers and the two byte-stride tuples;
    the final two words are split count and partial pointer. Matrix groups
    have a different ABI and retain their separate eight-member limit.
    """
    words = 2 + sum(map(len, shapes))
    return next((capacity for capacity in (32, 16, 8)
                 if (capacity * words + 2) * 8 <= 3072), 0)


@dataclass(frozen=True)
class ScalarReductionGroup:
    action_indices: tuple[int, ...]
    workspace_bytes: int
    seconds: float

    def __post_init__(self):
        object.__setattr__(self, 'action_indices', tuple(self.action_indices))
        indices = self.action_indices
        if (not 2 <= len(indices) <= 32
                or any(type(i) is not int or i < 0 for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))):
            raise ValueError('scalar groups require 2–32 strictly increasing paid Action indices')
        if type(self.workspace_bytes) is not int or self.workspace_bytes <= 0:
            raise ValueError('scalar group workspace must be positive integer bytes')
        if (type(self.seconds) not in (int, float) or not isfinite(self.seconds)
                or self.seconds < 0):
            raise ValueError('scalar group cost must be finite and nonnegative')


@dataclass(frozen=True)
class ScalarReductionGeometry:
    axes: tuple
    shapes: tuple
    dtype: np.dtype
    size: int
    allowance: int
    blocks: int
    partial_bytes: int


def scalar_reduction_geometry(action, catalog):
    """Describe a simple scalar's raw geometry without changing its precision.

    This describes possible raw partials only. In particular, it does not
    replace X's existing bounded-wide-reduction eligibility check.
    """
    if (action.kind != 'contract' or action.reused or action.core is not None
            or action.resource is not None or len(action.operands) != 2
            or action.equation is None):
        raise ValueError('scalar group member must be an ordinary paid binary Action')
    equation = action.equation.normalized()
    axes = equation.inputs
    if (equation.output or len(axes) != 2 or not axes[0]
            or any(len(set(side)) != len(side) for side in axes)
            or set(axes[0]) != set(axes[1])):
        raise ValueError('scalar groups require two equal label sets without repeated axes')
    try:
        left, right = (catalog[key] for key in action.operands)
        output = catalog[action.tensor]
    except KeyError as error:
        raise ValueError('scalar group tensor is absent from its catalog') from error
    dtype = np.dtype(left.dtype)
    if (dtype not in (np.dtype('float32'), np.dtype('complex64'))
            or right.dtype != dtype or output.dtype != dtype or output.shape != ()):
        raise ValueError('scalar group inputs and scalar output require one float32/complex64 dtype')
    shapes = left.shape, right.shape
    if (any(len(labels) != len(shape) for labels, shape in zip(axes, shapes))
            or any(n <= 0 for shape in shapes for n in shape)):
        raise ValueError('scalar group input shape disagrees with its labels')
    dimensions = dict(zip(axes[0], shapes[0]))
    if any(dimensions[label] != size for label, size in zip(axes[1], shapes[1])):
        raise ValueError('scalar group contracted dimensions disagree')
    size = prod(shapes[0])
    allowance = (sum(prod(shape) for shape in shapes) + 1) * dtype.itemsize
    wide_bytes = 2 * dtype.itemsize
    blocks = min(1024, (size + 255) // 256, (allowance - dtype.itemsize) // wide_bytes)
    if blocks < 1:
        raise ValueError('scalar group has no raw partial allowance')
    return ScalarReductionGeometry(axes, shapes, dtype, size, allowance, blocks, blocks * wide_bytes)


def describe_scalar_reduction_group(actions, action_indices, catalog, seconds):
    """Describe prospective shared geometry; full plan audit checks residency."""
    indices = tuple(action_indices)
    # Validate the record shape before indexing, including bool/negative IDs.
    ScalarReductionGroup(indices, 1, seconds)
    if any(index >= len(actions) for index in indices):
        raise ValueError('scalar group references an unknown Action')
    geometries = tuple(scalar_reduction_geometry(actions[index], catalog) for index in indices)
    if any(geometry != geometries[0] for geometry in geometries[1:]):
        raise ValueError('scalar group members have different normalized geometry')
    return ScalarReductionGroup(indices, len(indices) * geometries[0].partial_bytes, seconds)


def scalar_reduction_group_memory(plan, *, check=False, subtract=False):
    """Describe only partials and outputs born before their original Action.

    Every original scratch/packing/owner bucket is preserved. Each scalar has
    an independent result owner, charged at the device pool request granularity,
    so its extra bytes end at its original
    production point, when the unchanged logical trajectory takes over.
    Construction writes the exact bucket; check permits a conservative saved
    upper bound after cfg reuse filters out some groups. Subtraction removes
    only this annotation's required bytes, preserving any conservative excess.
    """
    groups = plan.scalar_reduction_groups
    actions, catalog = plan.state.actions, plan.state.catalog
    starts = {group.action_indices[0]: group for group in groups}
    changes = defaultdict(int)
    for group in groups:
        for index in group.action_indices[1:]:
            size = device_allocation_bytes(catalog[actions[index].tensor].dtype.itemsize)
            changes[group.action_indices[0]] += size
            changes[index] -= size
    additions, extra = {}, 0
    for index in range(len(actions) + 1):
        before = extra
        extra += changes[index]
        additions[index] = before, extra
    points = []
    for point in plan.memory:
        if type(point.action_index) is not int or point.action_index not in additions:
            raise ValueError('scalar memory point has an unknown Action index')
        extra = additions[point.action_index][0 if point.phase.startswith('read:') else 1]
        group = starts.get(point.action_index)
        if group is not None and point.phase == 'contract':
            extra += device_allocation_bytes(group.workspace_bytes)
        recorded = point.scalar_reduction_device_bytes
        if type(recorded) is not int or recorded < 0:
            raise ValueError('scalar reduction memory bucket must be nonnegative integer bytes')
        if check:
            if recorded < extra:
                raise ValueError('scalar reduction memory bucket omits partial/output lifetimes')
            points.append(point)
        else:
            amount = recorded - extra if subtract else extra
            if amount < 0:
                raise ValueError('cannot subtract an absent scalar reduction memory envelope')
            points.append(replace(point, scalar_reduction_device_bytes=amount))
    return tuple(points)


def validate_scalar_reduction_groups(plan):
    """Audit a fixed annotation, including physical dense-batch publication."""
    groups = plan.scalar_reduction_groups
    if not groups:
        scalar_reduction_group_memory(plan, check=True)
        return
    if (plan.state.problem.backend != 'cupy' or plan.explicit_resources
            or plan.retention_planned or plan.retained_before or plan.retained_after
            or plan.retained_releases):
        raise NotImplementedError('scalar groups require an unretained fixed CuPy plan')
    actions, catalog = plan.state.actions, plan.state.catalog
    dense_indices = {index for batch in plan.kernel_batches for index in batch.action_indices}
    producers, last_consumers, first_releases = Counter(), {}, {}
    boundaries = [0]
    for index, action in enumerate(actions):
        numerical = action.kind in ('contract', 'sparse_bridge', 'baryon_join')
        if action.kind == 'load' or numerical:
            if not action.reused:
                producers[action.tensor] += 1
        if numerical:
            for key in (*action.operands, *getattr(action.core, 'inputs', ())):
                last_consumers[key] = index
        if action.kind == 'release':
            first_releases.setdefault(action.tensor, index)
        boundaries.append(boundaries[-1] + int(action.kind not in ('load', 'release', 'contract')
                          or action.core is not None or action.resource is not None))
    indices, names, starts = set(), set(), {}
    for group in groups:
        if type(group) is not ScalarReductionGroup:
            raise TypeError('scalar groups require ScalarReductionGroup records')
        if indices.intersection(group.action_indices) or dense_indices.intersection(group.action_indices):
            raise ValueError('one paid Action cannot belong to multiple physical groups')
        described = describe_scalar_reduction_group(actions, group.action_indices, catalog, group.seconds)
        if group.workspace_bytes != described.workspace_bytes:
            raise ValueError('scalar group partial workspace disagrees with its geometry')
        indices.update(group.action_indices)
        starts[group.action_indices[0]] = group
        for index in group.action_indices:
            result = actions[index].tensor
            if result in names:
                raise ValueError('scalar groups cannot share a result owner')
            names.add(result)
            if last_consumers.get(result, -1) > index:
                raise ValueError('scalar group results must be terminal numerical outputs')
            if producers[result] != 1:
                raise ValueError('scalar group result has another paid producer')
            if first_releases.get(result, index) < index:
                raise ValueError('scalar group result is released before its original production')
        if boundaries[group.action_indices[-1] + 1] != boundaries[group.action_indices[0]]:
            raise ValueError('scalar group crosses a sparse, core or resource boundary')

    dense_starts = {batch.action_indices[0]: batch for batch in plan.kernel_batches}
    live = set()
    for index, action in enumerate(actions):
        batch = dense_starts.get(index)
        group = starts.get(index)
        if batch is not None:
            inputs = tuple(operand.tensor for operand in (*batch.left, *batch.right))
            outputs = tuple(result.tensor for result in batch.results)
        elif group is not None:
            inputs = tuple(key for i in group.action_indices for key in actions[i].operands)
            outputs = tuple(actions[i].tensor for i in group.action_indices)
        else:
            inputs = outputs = ()
        if any(key not in live for key in inputs):
            raise ValueError('scalar group physical inputs must be resident before its first member')
        if set(inputs).intersection(outputs):
            raise ValueError('scalar group contains an internal data dependency')
        live.update(outputs)
        if index in dense_indices or index in indices:
            continue  # Its physical start already read inputs and published outputs.
        if action.kind == 'release':
            live.discard(action.tensor)
        elif action.reused:
            # A reused Action is the explicit handoff of an already-owned
            # result, at this position only. X checks that current owner;
            # replaying its historical operands would be a different operation.
            live.add(action.tensor)
        elif action.kind in ('load', 'contract', 'sparse_bridge', 'baryon_join'):
            if any(key not in live for key in action.operands):
                raise ValueError('scalar group schedule consumes an absent allocation')
            live.add(action.tensor)

    # A scalar start must have its actual contract-phase peak in the archive;
    # absence cannot silently erase the temporary partial allocation.
    phases = Counter((point.action_index, point.phase) for point in plan.memory)
    for index in indices:
        if phases[index, 'contract'] != 1:
            raise ValueError('scalar group memory trajectory omits a member contract phase')
    scalar_reduction_group_memory(plan, check=True)
    if (not plan.memory or plan.peak_host_bytes != max(p.host_bytes for p in plan.memory)
            or plan.peak_device_bytes != max(p.device_bytes for p in plan.memory)):
        raise ValueError('scalar group resource peaks disagree with memory points')
    feasible = (plan.peak_device_bytes <= plan.memory_limit
                and (plan.host_memory_limit is None or plan.peak_host_bytes <= plan.host_memory_limit)
                and (plan.disk_limit_bytes is None or plan.peak_disk_bytes <= plan.disk_limit_bytes))
    if plan.feasible != feasible:
        raise ValueError('scalar group feasibility disagrees with its resource limits')
