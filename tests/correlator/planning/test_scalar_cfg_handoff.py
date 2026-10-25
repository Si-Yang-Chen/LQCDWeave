"""Scalar physical groups survive cfg remapping, reuse and fixed archives."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from contracts.test_scalar_reduction_group import fixture, annotated, describe, dense_producer_fixture
from lqcdweave.contracts.correlator.api import (
    ArrayConstant, PlannedContraction, SourcePreparation, TensorPreparation,
    describe_coordinates,
)
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.planning.api import CostProfile
from lqcdweave.contraction.planning.gemm_groups import lower_independent_gemm_path
from lqcdweave.contraction.execution import cfg_runtime
from lqcdweave.storage.contraction_plans.api import save_cfg_plan, load_cfg_plan


def cfg_fixture(*, partial=False, debug=False, count=4):
    original, indices = fixture(count)
    grouped = annotated(original, (describe(original, indices),))
    specs = grouped.state.problem.specs
    preparations = {}
    for n, (key, spec) in enumerate(specs.items()):
        # One pair aliases an earlier pair only in the partial-reuse case.
        value = n + 1 if not partial or n not in (2, 3) else n - 1
        preparations[key] = SourcePreparation((), recipe=TensorPreparation(
            'constant', constants=(ArrayConstant.from_array(np.full(spec.shape, value, spec.dtype)),)))
    points, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 1}])
    result = cfg_plan.build_cfg_plan(grouped, None, points.points, {}, preparations, specs,
                                     debug=debug)
    return grouped, indices, result


@pytest.mark.parametrize('count', (4, 32))
@pytest.mark.parametrize('debug', (False, True))
@pytest.mark.parametrize('partial', (False, True))
def test_cfg_resource_audit_and_remap_preserve_scalar_envelope(debug, partial, count):
    original, indices, cfg = cfg_fixture(partial=partial, debug=debug, count=count)
    template = cfg.templates[0].plan
    kept = [i for i, action in enumerate(original.state.actions) if action.kind != 'release']
    positions = {old: new for new, old in enumerate(kept)}
    positions[len(original.state.actions)] = len(kept)  # Terminal output ownership survives.
    assert template.scalar_reduction_groups[0].action_indices == tuple(positions[i] for i in indices)
    assert [point.scalar_reduction_device_bytes for point in template.memory] == [
        point.scalar_reduction_device_bytes for point in original.memory if point.action_index in positions]
    audit = cfg_plan.audit_cfg_plan(cfg)
    assert audit['peak_device_bytes'] == cfg.peak_device_bytes
    assert cfg.peak_device_bytes >= original.scalar_reduction_groups[0].workspace_bytes
    assert np.all(cfg.instances[1].reused.array())
    if partial:
        assert cfg.instances[0].reused.array()[template.scalar_reduction_groups[0].action_indices[1]]


def test_cfg_table_filters_whole_group_and_invalidates_on_annotations():
    _, _, full = cfg_fixture()
    _, _, partial = cfg_fixture(partial=True)
    cache = cfg_runtime._CfgTableCache(np)
    with patch.object(cfg_runtime, 'bind_scene_execution', wraps=cfg_runtime.bind_scene_execution) as bind:
        cfg_runtime._bind_cfg_tables(full, np, cache)
        calls = bind.call_args_list
        assert len(calls[0].kwargs['scalar_reduction_groups']) == 1
        assert calls[1].kwargs['scalar_reduction_groups'] == ()
        bind.reset_mock()
        cfg_runtime._bind_cfg_tables(partial, np, cache)
        assert all(call.kwargs['scalar_reduction_groups'] == () for call in bind.call_args_list)
        bind.reset_mock()
        # Same shape/state/masks with different physical records must rebind.
        changed_template = replace(partial.templates[0], plan=replace(
            partial.templates[0].plan, scalar_reduction_groups=()))
        changed = replace(partial, templates=(changed_template,))
        cfg_runtime._bind_cfg_tables(changed, np, cache)
        assert bind.call_count > 0


def test_partial_dense_group_reuse_cannot_leave_future_scalar_inputs_advanced():
    original, indices = dense_producer_fixture()
    group = describe(original, indices)
    state, dense = original.state, original.kernel_batches
    assert cfg_runtime._active_scalar_groups(state, dense, (group,)) == (group,)
    # The second Cartesian child now executes at its original position,
    # later than the saved scalar-group start. Neither scalar is reused.
    actual = (dense[0].batches[0],)
    actions = list(state.actions)
    skipped = dense[0].batches[1].action_indices[-1]
    actions[skipped] = replace(actions[skipped], reused=True, seconds=0)
    state = replace(state, actions=tuple(actions))
    assert all(not state.actions[i].reused for i in indices)
    assert cfg_runtime._active_scalar_groups(state, actual, (group,)) == ()
    assert cfg_runtime._active_scalar_groups(state, (), (group,)) == ()


@pytest.mark.parametrize('count', (8, 32))
def test_fixed_cfg_archive_preserves_annotations_and_budget_without_search(tmp_path, count):
    _, _, cfg = cfg_fixture(partial=True, debug=True, count=count)
    path = Path(tmp_path) / 'cfg.json.gz'
    save_cfg_plan(path, cfg, debug=True)
    with patch('lqcdweave.contraction.planning.scalar_reduction_groups.lower_scalar_reduction_groups',
               side_effect=AssertionError('fixed archive cannot lower again')):
        restored = load_cfg_plan(path)
    assert restored == cfg
    assert cfg_plan.audit_cfg_plan(restored)['peak_device_bytes'] == cfg.peak_device_bytes


def test_new_path_handoff_runs_scalar_pass_even_without_gemm_groups():
    original, _ = fixture(8)
    profile = CostProfile()
    path = PlannedContraction(None, original, None,
        {'kernel_batching': True, 'predicted_seconds': original.seconds + 1},
        0, profile=profile, plan_identity=id(original))
    result = lower_independent_gemm_path(path, target_runtime='cuda')
    assert result.plan.scalar_reduction_groups
    assert result.plan_identity == id(result.plan)
    assert result.prediction['scalar_reduction_lowering']['applied']
    assert result.plan.seconds == pytest.approx(path.plan.seconds)
    assert result.prediction['predicted_seconds'] == pytest.approx(path.plan.seconds + 1)
    disabled = replace(path, prediction={'kernel_batching': False})
    assert lower_independent_gemm_path(disabled, target_runtime='cuda') is disabled
    assert lower_independent_gemm_path(path, target_runtime='hip') is path
