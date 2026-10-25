"""Scalar width admission uses the same ABI bound and exact resource envelope."""
from dataclasses import replace

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, SourcePreparation,
    scalar_reduction_group_limit, validate_scalar_reduction_groups,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, plan_scene_storage, lower_scalar_reduction_groups,
)
from correlator.planning.test_scalar_reduction_groups import fixture


def ranked_fixture(rank, count=33):
    labels = 'abcdefghijklmnopqrstuvwxyz'[:rank]
    shape = (513,) + (1,)*(rank-1)
    profile = CostProfile()
    specs = {f'{side}{i}': TensorSpec(shape, np.complex64)
             for i in range(count) for side in ('a', 'b')}
    terms = tuple(ContractionTerm((f'a{i}', f'b{i}'), f'{labels},{labels}->')
                  for i in range(count))
    problem = ContractionProblem(specs, terms,
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend='cupy')
    state = ContractionState.initial(problem)
    for key in specs:
        state = state.load(key, profile, 1 << 30)
    for i in range(count):
        state = state.contract(i, (0, 1), profile, 1 << 30)
    return plan_scene_storage(state, {}, {k: SourcePreparation(()) for k in specs},
        profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)


@pytest.mark.parametrize('rank,limit,counts', [
    (4, 32, (32,)), (5, 16, (16, 16)), (10, 16, (16, 16)),
    (11, 8, (8, 8, 8, 8)), (22, 8, (8, 8, 8, 8)), (23, 0, ()),
])
def test_high_rank_caps_width_instead_of_disabling_old_eight(rank, limit, counts):
    original = ranked_fixture(rank)
    assert scalar_reduction_group_limit(((1,)*rank,)*2) == limit
    selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
    assert tuple(len(g.action_indices) for g in selected.scalar_reduction_groups) == counts
    assert selected.state is original.state
    if limit:
        validate_scalar_reduction_groups(selected)
        assert stats['grouped_reductions'] == 32
        for group in selected.scalar_reduction_groups:
            assert group.workspace_bytes == 3*16*len(group.action_indices)
    else:
        assert selected is original


@pytest.mark.parametrize('width,counts', [(8, (8,)*8), (16, (16,)*4), (32, (32,)*2)])
def test_same_actions_costs_and_partition_for_internal_width(width, counts):
    original, _ = fixture(64, shape=(513,), dtype=np.complex64)
    selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda', max_tasks=width)
    assert tuple(len(g.action_indices) for g in selected.scalar_reduction_groups) == counts
    assert selected.state is original.state
    assert selected.seconds == original.seconds
    assert stats['launches_before'] == 128
    assert stats['launches_after'] == 2*len(counts)
    validate_scalar_reduction_groups(selected)


def test_tight_budget_shrinks_before_binding_and_never_changes_original_splits():
    original, _ = fixture(32, shape=(65536,), dtype=np.complex64)
    generous, _ = lower_scalar_reduction_groups(original, target_runtime='cuda')
    assert len(generous.scalar_reduction_groups[0].action_indices) == 32
    cap = original.peak_device_bytes + 16*4096
    tight = replace(original, memory_limit=cap)
    selected, stats = lower_scalar_reduction_groups(tight, target_runtime='cuda')
    assert stats['resource_rejections'] > 0
    assert selected.scalar_reduction_groups
    assert all(len(g.action_indices) < 32 for g in selected.scalar_reduction_groups)
    assert selected.peak_device_bytes <= cap
    assert all(g.workspace_bytes == len(g.action_indices)*256*16
               for g in selected.scalar_reduction_groups)
    validate_scalar_reduction_groups(selected)
