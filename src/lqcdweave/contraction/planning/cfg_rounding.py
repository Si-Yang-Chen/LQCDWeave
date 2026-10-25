"""Allocation-request bounds for P's global cfg ownership reconstruction.

This module describes allocations, not allocator pool fragmentation or opaque
library workspace. Logical payloads and declared aggregate workspace remain
unchanged. The cfg interval builder and its independent replay audit apply the
same per-owner sizes to their separately constructed lifetimes.
"""
from collections import Counter
from math import prod

import numpy as np

from lqcdweave.contracts.correlator.api import device_allocation_bytes, dense_batch_leaves


def payload(spec):
    return prod(spec.shape) * spec.dtype.itemsize


def padding(size):
    return device_allocation_bytes(size) - size


def raw_device_bytes(block):
    # The native dense uploader owns one array. Sparse/p2p containers instead
    # retain their explicitly declared aggregate allocation bound.
    return (device_allocation_bytes(block.device_bytes)
            if block.read_kind == 'dense' else block.device_bytes)


def staging_device_bytes(block):
    return (device_allocation_bytes(block.staging_device_bytes)
            if block.read_kind == 'dense' else block.staging_device_bytes)


def constant_bytes(preparations, *, device=False):
    constants = {value for prep in preparations.values() for value in prep.recipe.constants}
    sizes = (prod(value.shape) * np.dtype(value.dtype).itemsize for value in constants)
    return sum(device_allocation_bytes(size) if device else size for size in sizes)


def snapshot_device_bytes(state):
    """Cover native grouped copies and the measured/debug per-result fallback.

    X groups unique result keys by shape/dtype on the fixed fast path, whereas
    its generic path copies every result occurrence. The latter is a safe
    bound for both, even for repeated result keys. It is charged only at the
    output slot, before final releases. X separately reserves queued downloads.
    """
    return sum(device_allocation_bytes(payload(state.catalog[key])) for key in state.results)


def native_snapshot_device_bytes(state):
    groups = Counter()
    for key in set(state.results):
        spec = state.catalog[key]
        groups[spec.shape, spec.dtype.str] += payload(spec)
    return sum(map(device_allocation_bytes, groups.values()))


def preparation_padding(preparations):
    """Conservative padding for the finite native preparation programs.

    Keep the declared logical/aggregate workspace. At most 511 extra bytes per
    explicit allocation covers any request size; counting all allocations in
    a recipe (rather than guessing simultaneous temporaries) is conservative.
    This includes the fused final owner, so some final-owner padding is charged
    twice. No array, input dtype/stride or backend probe is needed.

    A spin fallback has conversion, profile accumulator, conjugation, two
    potentially copying reshapes, final multiply/conversion and one spare
    destination slot (8), plus four ufunc results per profile row. Stack adds
    its output/conversion and every child program. Identity/constant leave
    zero temporary slots: their final owned copy is already a numeric owner.
    """
    counts = {}

    def count(key):
        if key not in counts:
            recipe = preparations[key].recipe
            if recipe.kind in {'stack', 'linear_combination'}:
                value = 2 + sum(count(member) for member in recipe.inputs)
            elif recipe.kind == 'spin_spatial':
                rows = recipe.constants[3].shape[0] if len(recipe.constants) == 4 else 0
                value = 8 + 4 * rows
            else:
                value = len(recipe.index_permutations) + 2 if recipe.index_permutations else 0
            counts[key] = value
        return counts[key]

    return {key: 511 * count(key) for key in preparations}


def action_workspace_padding(action, catalog, workspace):
    if not workspace:
        return 0
    if action.kind == 'contract' and action.core is None and action.resource is None:
        # Ordinary dense workspace permits independent operand layout copies
        # and an extra output. A sum rounded only once misses their padding.
        slots = (*action.operands, action.tensor)
        return max(padding(workspace), sum(padding(payload(catalog[key])) for key in slots))
    # Non-dense kernels retain their declared aggregate physical bound. Do not
    # infer a count of opaque temporaries from its size.
    return 0


def batch_packing_padding(batch, catalog):
    total = 0
    for child in dense_batch_leaves((batch,)):
        for side, prepared in zip((child.left, child.right), child.prepared_sides):
            if not prepared and (child.packing_dtype is not None or len(side) > 1):
                dtype = (np.dtype(child.packing_dtype) if child.packing_dtype is not None
                         else catalog[side[0].tensor].dtype)
                size = sum(prod(catalog[item.tensor].shape) * dtype.itemsize for item in side)
                total += padding(size)
    return total


__all__ = []
