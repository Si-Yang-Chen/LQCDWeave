"""Discover complete Cartesian kernels without changing logical computations.

Only genuine dense reductions are accepted. Shape equality alone never joins
unrelated edges: connected components use actual tensor/permutation identities.
"""
from collections import defaultdict
import math
from math import prod
import numpy as np

from lqcdweave.contracts.correlator.api import (
    DenseBatchOperand, DenseBatchResult, DenseKernelBatch,
    dense_batch_producer_eligible,
)


def _descriptor(action, catalog):
    if (action.kind != "contract" or action.reused or action.core is not None or action.wide_accumulation
            or action.resource is not None or action.equation is None
            or len(action.operands) != 2):
        return None
    axes, output = action.equation.inputs, action.equation.output
    if len(axes) != 2 or any(len(set(a)) != len(a) for a in axes):
        return None
    reduced = set(axes[0]) & set(axes[1])
    if not reduced or reduced & set(output):
        return None  # includes aligned multiplication and outer products
    leftfree = tuple(a for a in output if a in axes[0] and a not in reduced)
    rightfree = tuple(a for a in output if a in axes[1] and a not in reduced)
    if (len(set(output)) != len(output)
            or (set(axes[0]) | set(axes[1])) - reduced != set(output)):
        return None
    operands = action.operands
    if output not in (leftfree + rightfree, rightfree + leftfree):
        return None
    # A scalar output or a single free side does not determine operand order.
    # Term-factor positions can change when independent producers are moved.
    # Resolve that ambiguity from tensor identity, not those positions.
    swap = (output != leftfree + rightfree
            or ((not leftfree or not rightfree) and operands[1] < operands[0]))
    if swap:
        operands, axes = operands[::-1], axes[::-1]
        leftfree, rightfree = rightfree, leftfree
    contracted = tuple(a for a in axes[0] if a in reduced)
    permutations = (tuple(axes[0].index(a) for a in leftfree + contracted),
                    tuple(axes[1].index(a) for a in contracted + rightfree))
    shapes = [catalog[k].shape for k in operands]
    ordered = [tuple(s[p] for p in perm) for s, perm in zip(shapes, permutations)]
    inner = ordered[0][len(leftfree):]
    if inner != ordered[1][:len(contracted)] or not all(inner):
        return None
    return (tuple(DenseBatchOperand(k, p) for k, p in zip(operands, permutations)),
            ordered[0][:len(leftfree)] or (1,), inner,
            ordered[1][len(contracted):] or (1,), (not leftfree, not rightfree))


def _side(operands, shapes, *, offset):
    varying = [i for i in range(len(shapes[0])) if len({s[i] for s in shapes}) > 1]
    if len(varying) > 1 or any(not all(s) for s in shapes):
        return None
    axis = varying[0] if varying else len(shapes[0]) - 1
    bounds, start = {}, 0
    for operand, shape in zip(operands, shapes):
        bounds[operand] = tuple((start, start + n) if i == axis else (0, n)
                                for i, n in enumerate(shape))
        start += shape[axis]
    combined = list(shapes[0])
    combined[axis] = start
    return axis + offset, bounds, tuple(combined)


def describe_dense_batch(actions, action_indices, catalog, profile, *, backend='numpy', prepared_inputs=None):
    """Return a profitable compatible lowering, or None; cost is recomputed."""
    if backend not in ('numpy', 'cupy'):
        return None
    entries = [(index, actions[index], _descriptor(actions[index], catalog)) for index in action_indices]
    if any(descriptor is None for _, _, descriptor in entries):
        return None
    return _describe_dense_entries(entries, catalog, profile, backend=backend,
                                   prepared_inputs=prepared_inputs)


def _describe_dense_entries(entries, catalog, profile, *, backend, prepared_inputs=None):
    """Price current-call descriptors; public callers still derive their own."""
    if len(entries) < 2:
        return None
    left, right, pairs = {}, {}, set()
    inner = entries[0][2][2]
    stacked = entries[0][2][4]
    for _, action, d in entries:
        operands, lshape, kshape, rshape, stacking = d
        if kshape != inner or stacking != stacked or (operands[0], operands[1]) in pairs:
            return None
        left[operands[0]], right[operands[1]] = lshape, rshape
        pairs.add(tuple(operands))
    if (len(left) * len(right) != len(entries)
            or {o.tensor for o in left} & {o.tensor for o in right}):
        return None
    ls = _side(tuple(left), tuple(left.values()), offset=0)
    rs = _side(tuple(right), tuple(right.values()), offset=len(inner))
    if ls is None or rs is None:
        return None
    laxis, lbounds, lshape = ls
    raxis, rbounds, rshape = rs
    dtypes = {catalog[o.tensor].dtype for o in (*left, *right)}
    if len(dtypes) != 1:
        return None
    dtype = next(iter(dtypes))
    if any(catalog[a.tensor].dtype != dtype for _, a, _ in entries):
        return None
    input_bytes = sum(prod(catalog[o.tensor].shape) * catalog[o.tensor].dtype.itemsize
                      for o in (*left, *right))
    packing_bytes = sum(prod(catalog[o.tensor].shape) * catalog[o.tensor].dtype.itemsize
                        for side in (left, right) if len(side) > 1 for o in side)
    output_bytes = prod(lshape + rshape) * dtype.itemsize
    packing_dtype = None
    prepared_sides = (False,False)
    matrix_packing = backend == 'cupy' or (
        backend == 'numpy' and (len(lshape), len(inner), len(rshape)) != (1, 1, 1)
        and dtype.name in ('float32', 'float64', 'complex64', 'complex128'))
    if matrix_packing:
        if dtype.name not in ('float32', 'float64', 'complex64', 'complex128'):
            return None
        # Explicit matrix packing is a saved implementation choice, including
        # NumPy: high-rank einsum otherwise bypasses the multi-threaded BLAS.
        # One C-contiguous buffer per side keeps the uniform input dtype.
        # All allocations and copy costs are accounted below.
        packing_dtype = dtype.name
        prepared_inputs = {} if prepared_inputs is None else prepared_inputs
        prepared_sides = tuple(len(side)==1 and catalog[next(iter(side)).tensor].dtype == dtype
                               and prepared_inputs.get(next(iter(side)).tensor)==next(iter(side)).permutation
                               for side in (left,right))
        packing_bytes = sum(prod(catalog[o.tensor].shape)*dtype.itemsize
                            for side,prepared in zip((left,right),prepared_sides) if not prepared for o in side)
        prepared_bytes = sum(prod(catalog[o.tensor].shape)*dtype.itemsize
                             for side,prepared in zip((left,right),prepared_sides) if prepared for o in side)
        seconds = profile.contract_seconds(prod(lshape)*prod(rshape)*prod(inner),
            packing_bytes + prepared_bytes + output_bytes, dtype=dtype)
        for side, prepared in zip((left,right),prepared_sides):
            if prepared:
                continue
            source_bytes = sum(prod(catalog[o.tensor].shape)*catalog[o.tensor].dtype.itemsize for o in side)
            target_bytes = sum(prod(catalog[o.tensor].shape)*dtype.itemsize for o in side)
            seconds += profile.contract_seconds(0, source_bytes + target_bytes, dtype=dtype)
    else:
        seconds = profile.contract_seconds(prod(lshape) * prod(rshape) * prod(inner),
            input_bytes + output_bytes, dtype=dtype)
        for side in (left, right):
            if len(side) > 1:
                nbytes = sum(prod(catalog[o.tensor].shape) * catalog[o.tensor].dtype.itemsize for o in side)
                seconds += profile.contract_seconds(0, 2 * nbytes,
                    dtype=catalog[next(iter(side)).tensor].dtype)
    if seconds >= sum(a.seconds for _, a, _ in entries):
        return None
    squeeze = ((0,) if stacked[0] else ()) + ((len(lshape),) if stacked[1] else ())
    results = tuple(DenseBatchResult(a.tensor, lbounds[d[0][0]] + rbounds[d[0][1]], squeeze)
                    for _, a, d in entries)
    return DenseKernelBatch(tuple(index for index, _, _ in entries), tuple(left), tuple(right), len(lshape), len(inner),
        len(rshape), laxis, raxis, results, seconds, packing_bytes, output_bytes,
        packing_dtype, prepared_sides, stacked)


def _connected_indices(entries):
    """Visit each operand's incidence list once, including high-degree hubs."""
    neighbors = defaultdict(list)
    for i, (_, descriptor) in enumerate(entries):
        for side, operand in enumerate(descriptor[0]):
            neighbors[side, operand].append(i)
    pending = set(range(len(entries)))
    for seed in range(len(entries)):
        if seed not in pending:
            continue
        pending.remove(seed)
        found, stack, expanded = [], [seed], set()
        while stack:
            i = stack.pop()
            found.append(i)
            for side, operand in enumerate(entries[i][1][0]):
                key = side, operand
                if key in expanded:
                    continue
                expanded.add(key)
                for neighbor in neighbors[key]:
                    if neighbor in pending:
                        pending.remove(neighbor)
                        stack.append(neighbor)
        yield tuple(sorted(entries[i][0] for i in found))


def _subgrid_action_groups(entries):
    """Partition a complete grid when multiple free axes vary on a side.

    Each key fixes every free dimension except that side's last varying
    dimension. The returned groups preserve the source action order; the
    ordinary descriptor remains responsible for validating and costing each
    subgrid.
    """
    if len(entries) < 2:
        return ()

    left_shapes, right_shapes, pairs = {}, {}, set()
    inner = entries[0][2][2]
    stacked = entries[0][2][4]
    for _, _, descriptor in entries:
        operands, left_shape, inner_shape, right_shape, stacking = descriptor
        pair = tuple(operands)
        if (inner_shape != inner or stacking != stacked or pair in pairs):
            return ()
        left_shapes[operands[0]] = left_shape
        right_shapes[operands[1]] = right_shape
        pairs.add(pair)

    if (len(left_shapes) * len(right_shapes) != len(entries)
            or {operand.tensor for operand in left_shapes}
            & {operand.tensor for operand in right_shapes}):
        return ()
    all_shapes = (*left_shapes.values(), *right_shapes.values())
    if any(not all(shape) for shape in all_shapes):
        return ()

    def varying_axes(shapes):
        ranks = {len(shape) for shape in shapes}
        if len(ranks) != 1:
            return None
        return tuple(axis for axis in range(next(iter(ranks)))
                     if len({shape[axis] for shape in shapes}) > 1)

    left_varying = varying_axes(tuple(left_shapes.values()))
    right_varying = varying_axes(tuple(right_shapes.values()))
    if left_varying is None or right_varying is None:
        return ()
    if len(left_varying) <= 1 and len(right_varying) <= 1:
        return ()

    def group_key(shape, varying):
        if len(varying) <= 1:
            return ()
        keep_axis = varying[-1]
        return tuple(size for axis, size in enumerate(shape) if axis != keep_axis)

    left_keys = {operand: group_key(shape, left_varying)
                 for operand, shape in left_shapes.items()}
    right_keys = {operand: group_key(shape, right_varying)
                  for operand, shape in right_shapes.items()}
    groups = defaultdict(list)
    for index, _, descriptor in entries:
        left, right = descriptor[0]
        groups[left_keys[left], right_keys[right]].append(index)
    return tuple(tuple(indices) for indices in groups.values() if len(indices) >= 2)


def _partial_grid_action_groups(entries):
    """Cover existing edges with disjoint complete grids, without padding.

    Repeated pairs stay on the original path. At each row partition, consume
    only common neighbors, then divide remaining rows in stable degree order.
    Incidence storage is linear; each edge participates at most once per level.
    The ordinary descriptor still prices and checks each resulting candidate.
    """
    cells = {}
    for index, _, descriptor in entries:
        pair = descriptor[0]
        cells[pair] = None if pair in cells else index
    rows = {}
    for (left, right), index in cells.items():
        if index is not None and left.tensor != right.tensor:
            rows.setdefault(left, {})[right] = index
    pending = [sorted(rows, key=lambda left: -len(rows[left]))]
    groups = []
    while pending:
        active = [left for left in pending.pop() if rows[left]]
        if not active:
            continue
        common = set(rows[active[0]])
        for left in active[1:]:
            common.intersection_update(rows[left])
            if not common:
                break
        left_tensors = {left.tensor for left in active}
        common = {right for right in common if right.tensor not in left_tensors}
        if len(active) * len(common) >= 2:
            groups.append(tuple(sorted(rows[left][right] for left in active for right in common)))
            for left in active:
                for right in common:
                    del rows[left][right]
        active = [left for left in active if rows[left]]
        if len(active) > 1:
            midpoint = len(active) // 2
            pending.extend((active[midpoint:], active[:midpoint]))
    return tuple(groups)


def _price_producer_layouts(state, batches, profile, producers, eligibility, entries):
    """Reprice one complete batch candidate with its own producer layouts."""
    batch_outputs = {result.tensor for batch in batches for result in batch.results}
    prepared, conflicting = {}, set()
    for batch in batches:
        for side in (batch.left, batch.right):
            if len(side) != 1:
                continue
            operand = side[0]
            if (operand.tensor in batch_outputs
                    or state.catalog[operand.tensor].dtype != np.dtype(batch.packing_dtype)):
                continue
            if operand not in eligibility:
                eligibility[operand] = dense_batch_producer_eligible(
                    operand, producers[operand.tensor], state.catalog)
            if not eligibility[operand]:
                continue
            if operand.tensor in prepared and prepared[operand.tensor] != operand.permutation:
                conflicting.add(operand.tensor)
            prepared[operand.tensor] = operand.permutation
    prepared = {tensor: permutation for tensor, permutation in prepared.items()
                if tensor not in conflicting}

    repriced = []
    for batch in batches:
        candidate = _describe_dense_entries(
            [entries[index] for index in batch.action_indices], state.catalog, profile,
            backend='cupy', prepared_inputs=prepared)
        # The original descriptor has already passed the same legality and
        # profitability checks without producer layouts. Keep it if a prepared
        # layout cannot be represented by the saved descriptor.
        repriced.append(batch if candidate is None else candidate)
    return tuple(repriced)


def _batch_cost_delta(actions, batches):
    """Sum only the costs changed by replacing the listed paid Actions."""
    return math.fsum(
        batch.seconds - math.fsum(actions[index].seconds for index in batch.action_indices)
        for batch in batches)


def find_dense_batches(state, profile):
    """Connect exact operand edges, then accept only complete compatible grids."""
    stages = defaultdict(list)
    outputs = frozenset(state.results)
    backend = getattr(getattr(state, 'problem', None), 'backend', 'numpy')
    if backend not in ('numpy', 'cupy'):
        return ()
    described = {}
    for index, action in enumerate(state.actions):
        d = _descriptor(action, state.catalog)
        if d is None:
            continue
        described[index] = (index, action, d)
        operands, left, inner, right, stacked = d
        key = ("final", inner, len(left), len(right), str(state.catalog[action.tensor].dtype), stacked) if action.tensor in outputs else (
            "shared", operands[0], inner, len(left), len(right), str(state.catalog[action.tensor].dtype), stacked)
        stages[key].append((index, d))
    batches, baseline_batches, extra_batches = [], [], []
    for entries in stages.values():
        for indices in _connected_indices(entries):
            component = [described[index] for index in indices]
            batch = _describe_dense_entries(component, state.catalog, profile, backend=backend)
            if batch is not None:
                batches.append(batch)
                baseline_batches.append(batch)
                continue
            subgrids = _subgrid_action_groups(component) or _partial_grid_action_groups(component)
            for subgrid in subgrids:
                batch = _describe_dense_entries([described[index] for index in subgrid],
                                               state.catalog, profile, backend=backend)
                if batch is not None:
                    batches.append(batch)
                    extra_batches.append(batch)
    if backend == 'cupy':
        producers = defaultdict(list)
        for action in state.actions:
            if action.kind == 'contract' and not action.reused:
                producers[action.tensor].append(action)
        eligibility = {}
        baseline = _price_producer_layouts(
            state, baseline_batches, profile, producers, eligibility, described)
        if not extra_batches:
            return baseline
        augmented = _price_producer_layouts(
            state, batches, profile, producers, eligibility, described)
        if _batch_cost_delta(state.actions, augmented) < _batch_cost_delta(state.actions, baseline):
            return augmented
        return baseline
    return tuple(batches)
