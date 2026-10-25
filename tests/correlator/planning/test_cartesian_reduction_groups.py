"""Whole Cartesian grouping preserves baseline readiness and exact envelopes."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from contracts.test_cartesian_reduction_group import fixture, prepared_fixture
from lqcdweave.contracts.correlator.api import (
    Action, PlannedContraction, validate_executable_resources, dense_batch_memory,
    device_allocation_bytes,
)
from lqcdweave.contraction.planning.api import (
    CostProfile, lower_cartesian_reduction_groups,
)
from lqcdweave.contraction.planning.cartesian_reduction_groups import _baseline_ready_times
from lqcdweave.contraction.planning.gemm_groups import lower_independent_gemm_path


def assert_envelope(before, after):
    """Independent owner oracle, including reads before displaced children."""
    by_indices = {batch.action_indices: batch for batch in before.kernel_batches}
    for old, point in zip(before.memory, after.memory, strict=True):
        expected = 0
        for group in after.cartesian_reduction_groups:
            first = group.batch_action_indices[0][0]
            if point.action_index == first and point.phase == 'contract':
                # Enumerate physical allocations, independently of C's helper.
                expected += device_allocation_bytes(group.workspace_bytes)
                children = [by_indices[child] for child in group.batch_action_indices]
                for child in children:
                    for side, prepared in zip((child.left, child.right), child.prepared_sides):
                        if not prepared:
                            size = sum(int(np.prod(before.state.catalog[o.tensor].shape))
                                       * before.state.catalog[o.tensor].dtype.itemsize for o in side)
                            expected += device_allocation_bytes(size)
                expected -= children[0].packing_bytes
            for child in group.batch_action_indices[1:]:
                birth = child[0]
                if ((point.action_index > first or point.action_index == first and not point.phase.startswith('read:'))
                        and (point.action_index < birth or point.action_index == birth and point.phase.startswith('read:'))):
                    expected += device_allocation_bytes(by_indices[child].output_bytes)
        assert point.cartesian_reduction_device_bytes == expected
        assert replace(point, cartesian_reduction_device_bytes=0) == old
    assert after.peak_device_bytes == max(point.device_bytes for point in after.memory)
    validate_executable_resources(after)


@pytest.mark.parametrize('dtype', (np.float32, np.complex64))
def test_ready_whole_batches_keep_original_costs_blocks_and_owners(dtype):
    before = fixture(11, dtype=dtype)
    after, stats = lower_cartesian_reduction_groups(before, target_runtime='cuda')
    assert stats['applied'] and stats['grouped_batches'] == 11
    assert tuple(len(group.batch_action_indices) for group in after.cartesian_reduction_groups) == (8, 3)
    assert after.state is before.state and after.kernel_batches is before.kernel_batches
    assert after.blocks is before.blocks and after.events is before.events
    assert after.seconds == pytest.approx(before.seconds)
    assert stats['cost_status'] == 'uncalibrated-original-member-sum' and not stats['speedup_claimed']
    assert stats['theoretical_launches_saved'] == 18
    assert_envelope(before, after)
    same, reason = lower_cartesian_reduction_groups(after, target_runtime='cuda')
    assert same is after and reason['reason'] == 'existing_cartesian_groups'


def test_future_loads_cannot_be_pulled_into_a_group():
    before = fixture(5, upfront=False)
    after, stats = lower_cartesian_reduction_groups(before, target_runtime='cuda')
    assert after is before and not stats['applied']
    assert stats['reason'] == 'no_feasible_resident_group'


def test_birth_snapshot_tracks_reloads_and_does_not_publish_current_batch_first():
    # A small residency protocol independently names the expected ready times.
    # 'x' is produced by the first physical batch; a later use may not join it.
    batch = lambda start, inputs, output: SimpleNamespace(action_indices=(start,),
        left=tuple(SimpleNamespace(tensor=k) for k in inputs), right=(),
        results=(SimpleNamespace(tensor=output),))
    first = batch(2, ('a', 'b'), 'x')
    dependent = batch(4, ('x', 'b'), 'y')
    reloaded = batch(7, ('a', 'b'), 'z')
    actions = tuple(SimpleNamespace(kind=kind,tensor=key,core=None) for kind,key in (
        ('load','a'),('load','b'),('contract','x'),('release','a'),
        ('contract','y'),('load','a'),('contract','other'),('contract','z')))
    plan = SimpleNamespace(state=SimpleNamespace(actions=actions),kernel_batches=(first,dependent,reloaded))
    ready, _ = _baseline_ready_times(plan, {2:(first,None),4:(dependent,None),7:(reloaded,None)})
    assert ready == {2:2,4:3,7:6}
    assert ready[4] > 2 and ready[7] > 2


def test_sparse_or_core_boundaries_partition_original_ready_batches():
    before = fixture(4)
    original_batches = before.kernel_batches
    memory = dense_batch_memory(before, subtract=True)
    before = replace(before,kernel_batches=(original_batches[0],*original_batches[2:]),memory=memory)
    memory = dense_batch_memory(before)
    before = replace(before,memory=memory,peak_device_bytes=max(point.device_bytes for point in memory))
    boundary = original_batches[1].action_indices[0]
    for kind in ('core','sparse_bridge','baryon_join'):
        actions=list(before.state.actions)
        actions[boundary]=(replace(actions[boundary],core=SimpleNamespace(inputs=())) if kind=='core'
                           else replace(actions[boundary],kind=kind))
        original=replace(before,state=replace(before.state,actions=tuple(actions)))
        after,stats=lower_cartesian_reduction_groups(original,target_runtime='cuda')
        assert stats['applied']
        assert after.cartesian_reduction_groups[0].batch_action_indices == tuple(batch.action_indices for batch in original_batches[2:])
        assert_envelope(original,after)


def test_budget_shrinks_snapshot_and_restores_remaining_candidates():
    before=fixture(4)
    full,stats=lower_cartesian_reduction_groups(before,target_runtime='cuda')
    assert stats['applied']
    item=before.kernel_batches[0].packing_bytes
    bounded=replace(before,memory_limit=before.peak_device_bytes+item+1024)
    selected,stats=lower_cartesian_reduction_groups(bounded,target_runtime='cuda')
    assert stats['applied'] and stats['resource_rejections']>0
    assert max(map(lambda group:len(group.batch_action_indices),selected.cartesian_reduction_groups))<4
    assert selected.peak_device_bytes<=bounded.memory_limit
    assert_envelope(bounded,selected)
    tight=replace(before,memory_limit=before.peak_device_bytes)
    later,stats=lower_cartesian_reduction_groups(tight,target_runtime='cuda')
    # After earlier inputs are released, a later pair can fit even at the
    # original global peak. Rejecting an early group must not abandon it.
    assert stats['applied'] and stats['resource_rejections']>0
    assert later.cartesian_reduction_groups[0].batch_action_indices == tuple(
        batch.action_indices for batch in before.kernel_batches[-2:])
    assert later.peak_device_bytes<=tight.memory_limit
    assert_envelope(tight,later)
    pair=fixture(2)
    impossible=replace(pair,memory_limit=pair.peak_device_bytes)
    same,stats=lower_cartesian_reduction_groups(impossible,target_runtime='cuda')
    assert same is impossible and not stats['applied'] and stats['resource_rejections']>0


@pytest.mark.parametrize('k', (128, 129))
@pytest.mark.parametrize('prepared', (False, True))
def test_rounded_side_allocations_admit_exact_cap_and_reject_one_byte_less(k, prepared):
    before = prepared_fixture(k=k) if prepared else fixture(2, m=2, n=3, k=k)
    full, stats = lower_cartesian_reduction_groups(before, target_runtime='cuda')
    assert stats['applied']
    assert_envelope(before, full)
    exact = replace(before, memory_limit=full.peak_device_bytes)
    selected, stats = lower_cartesian_reduction_groups(exact, target_runtime='cuda')
    assert stats['applied'] and selected.peak_device_bytes == exact.memory_limit
    assert_envelope(exact, selected)
    smaller = replace(before, memory_limit=full.peak_device_bytes-1)
    kept, stats = lower_cartesian_reduction_groups(smaller, target_runtime='cuda')
    assert kept is smaller and not stats['applied'] and stats['resource_rejections'] > 0


@pytest.mark.parametrize('runtime,width', [('cpu',8),('hip',8),('cuda',1),('cuda',9),('cuda',True)])
def test_unsupported_domain_keeps_original_plan(runtime,width):
    before=fixture(2)
    same,stats=lower_cartesian_reduction_groups(before,target_runtime=runtime,max_tasks=width)
    assert same is before and not stats['applied']


def test_read_phases_and_other_physical_annotations_keep_their_buckets():
    before=fixture(4)
    points=[]
    starts={batch.action_indices[0] for batch in before.kernel_batches}
    for point in before.memory:
        if point.action_index in starts and point.phase=='contract':
            points.append(replace(point,phase='read:probe'))
        points.append(point)
    before=replace(before,memory=tuple(points))
    selected,stats=lower_cartesian_reduction_groups(before,target_runtime='cuda')
    assert stats['applied']
    assert_envelope(before,selected)


def test_new_path_handoff_uses_common_batching_switch_and_preserves_prediction():
    before=fixture(3)
    profile=CostProfile()
    path=PlannedContraction(None,before,None,
        {'kernel_batching':True,'predicted_seconds':before.seconds+1},0,
        profile=profile,plan_identity=id(before))
    selected=lower_independent_gemm_path(path,target_runtime='cuda')
    assert selected.plan.cartesian_reduction_groups
    assert selected.prediction['cartesian_reduction_lowering']['applied']
    assert selected.plan_identity==id(selected.plan)
    assert selected.prediction['predicted_seconds']==pytest.approx(before.seconds+1)
    for runtime,enabled in (('cuda',False),('hip',True),('cpu',True)):
        disabled=replace(path,prediction={'kernel_batching':enabled})
        with patch('lqcdweave.contraction.planning.cartesian_reduction_groups.lower_cartesian_reduction_groups',side_effect=AssertionError('must not lower')):
            assert lower_independent_gemm_path(disabled,target_runtime=runtime) is disabled
