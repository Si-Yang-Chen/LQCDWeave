"""Single-route cleanup preserves complete delivery and fallback semantics."""
from dataclasses import fields, replace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.planning.optimizers import storage_schedule as scheduler
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, SourcePreparation, StorageBlock,
    TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.storage import logical_source_problem
from correlator.planning.test_storage_peak_search import _chain_fixture
from correlator.planning.test_storage_schedule import _physical, _fixed_path, _ArrayLoader


def _equal_fields(actual, expected):
    for field in fields(expected):
        assert getattr(actual, field.name) == getattr(expected, field.name), field.name


def _ordered_choices(order):
    def choices(frontier, width, checkpoint):
        assert checkpoint()
        return iter(((next(index for index in order if index in frontier.ready), None),))
    return choices


def _gpu_fixture(dimensions=(200, 30, 4, 300)):
    state, _, _, physical, _ = _chain_fixture(dimensions)
    problem = replace(state.problem, backend='cupy')
    profile = CostProfile()
    state, spans, workspaces = _fixed_path(problem, profile,
        ((1, ('A', 'D')), (2, ('E', 'F')), (0, ('B', 'C')), (0, None)))
    blocks = {key: replace(block, device_bytes=block.host_bytes) for key, block in physical.blocks.items()}
    preparations = {key: replace(prep, recipe=TensorPreparation('identity', (key,)))
                    for key, prep in physical.preparations.items()}
    return state, spans, workspaces, _physical(problem, blocks, preparations, cap=1 << 30, profile=profile)


@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('deadline', [0, 100])
def test_single_route_keeps_its_original_delivery_without_undo(debug, deadline):
    state, spans, workspaces, physical, arrays = _chain_fixture()
    dag = scheduler._Dag(state, spans, workspaces, physical)
    expected = scheduler._plan(dag, tuple(range(len(spans))), audited_computations=True, debug=debug)
    # A foreign fallback can have extra metadata and cannot stand in for the
    # newly selected original route merely because its peaks/order match.
    foreign = replace(expected, feasible=False, retained_before=('foreign',))
    with patch.object(scheduler._Frontier, 'choices', _ordered_choices(range(len(spans)))), \
            patch.object(scheduler._Frontier, 'undo', side_effect=AssertionError('unneeded undo')), \
            patch.object(scheduler, '_plan', wraps=scheduler._plan) as emit:
        actual, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical,
            baseline=foreign, branch_width=1, deadline_seconds=deadline, max_states=1, debug=debug)
    _equal_fields(actual, expected)
    assert actual is not foreign
    assert stats['audited_schedules'] == emit.call_count == 1
    assert stats['backtracks'] == stats['optimization_expanded'] == 0
    assert stats['first_solution_expanded'] == len(spans)
    assert stats['stop_reason'] == ('deadline' if deadline == 0 else 'width_exhausted')
    references = (np.einsum('ij,jk,kl->il', arrays['A'], arrays['B'], arrays['C']),
                  arrays['A'] @ arrays['D'], arrays['E'] @ arrays['F'])
    for output, reference in zip(replay_scene_storage(actual, _ArrayLoader(arrays)), references):
        np.testing.assert_allclose(output, reference)


@pytest.mark.parametrize('debug', [False, True])
def test_payload_improvement_with_original_order_reuses_the_emitted_object(debug):
    state, spans, workspaces, physical = _gpu_fixture()
    expected = scheduler._plan(scheduler._Dag(state, spans, workspaces, physical),
        tuple(range(len(spans))), audited_computations=True, debug=debug)
    assert expected.packed_value_cache_device_bytes > 0
    emissions = []
    emit = scheduler._plan
    def recording(*args, **kwargs):
        result = emit(*args, **kwargs)
        emissions.append(result)
        return result
    with patch.object(scheduler._Frontier, 'choices', _ordered_choices(range(len(spans)))), \
            patch.object(scheduler, '_plan', recording):
        actual, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical,
            baseline=replace(expected, feasible=False, retained_before=('foreign',)),
            branch_width=1, debug=debug)
    _equal_fields(actual, expected)
    assert stats['complete_improvements'] == 1
    assert stats['audited_schedules'] == len(emissions) == 1
    assert actual is emissions[0]


@pytest.mark.parametrize('debug', [False, True])
def test_equal_peak_different_order_is_emitted_in_full(debug):
    state, spans, workspaces, physical = _gpu_fixture((1, 1, 1, 1))
    dag = scheduler._Dag(state, spans, workspaces, physical)
    original = scheduler._plan(dag, (0, 1, 2, 3), audited_computations=True, debug=debug)
    order = (0, 2, 1, 3)
    expected = scheduler._plan(dag, order, audited_computations=True, debug=debug)
    assert (expected.peak_host_bytes, expected.peak_device_bytes) == (original.peak_host_bytes, original.peak_device_bytes)
    assert expected.state.actions != original.state.actions
    with patch.object(scheduler._Frontier, 'choices', _ordered_choices(order)), \
            patch.object(scheduler, '_plan', wraps=scheduler._plan) as emit:
        actual, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical,
            baseline=replace(original, feasible=False), branch_width=1, debug=debug)
    _equal_fields(actual, expected)
    assert stats['audited_schedules'] == emit.call_count == 2
    assert stats['changed_order']


@pytest.mark.parametrize('width', [1, 2, 4])
def test_non_improving_route_keeps_the_provided_baseline_object(width):
    state, spans, workspaces, physical, _ = _chain_fixture()
    baseline = scheduler._plan(scheduler._Dag(state, spans, workspaces, physical),
        (1, 0, 2, 3), audited_computations=True)
    with patch.object(scheduler._Frontier, 'choices', _ordered_choices(range(len(spans)))):
        actual, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical,
            baseline=baseline, branch_width=width, deadline_seconds=0)
    assert actual is baseline
    assert stats['complete_improvements'] == 0
    assert stats['audited_schedules'] == 1 and stats['stop_reason'] == 'deadline'


@pytest.mark.parametrize('debug', [False, True])
def test_single_route_physical_prune_needs_no_restore_or_first_solution(debug):
    state, spans, workspaces, roomy, _ = _chain_fixture()
    physical = _physical(state.problem, roomy.blocks, roomy.preparations, cap=1)
    expected = scheduler._plan(scheduler._Dag(state, spans, workspaces, physical),
        tuple(range(len(spans))), audited_computations=True, debug=debug)
    with patch.object(scheduler._Frontier, 'undo', side_effect=AssertionError('no alternative')):
        actual, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical,
            branch_width=1, deadline_seconds=0, debug=debug)
    _equal_fields(actual, expected)
    assert not actual.feasible
    assert stats['peak_pruned'] == 1 and stats['backtracks'] == 0
    assert stats['first_solution_expanded'] is None
    assert stats['stop_reason'] == 'width_exhausted'


def test_single_route_final_copy_failure_preserves_the_infeasible_fallback():
    names = tuple('ABCDEFGHIJKLMNOP')
    problem = logical_source_problem(ContractionProblem(
        {key: TensorSpec((1,), 'float64') for key in names},
        tuple(ContractionTerm(names[i:i+2], 'i,i->') for i in range(0, 16, 2)),
        {key: SourceSpec(key, 0) for key in names}))
    blocks = {key: StorageBlock(key, 'full', 8, host_bytes=8) for key in names}
    preparations = {key: SourcePreparation((key,)) for key in names}
    state, spans, workspaces = _fixed_path(problem, CostProfile(), tuple((i, None) for i in range(8)))
    physical = _physical(problem, blocks, preparations, cap=127)
    expected = scheduler._plan(scheduler._Dag(state, spans, workspaces, physical),
        tuple(range(len(spans))), audited_computations=True, debug=True)
    with patch.object(scheduler._Frontier, 'undo', side_effect=AssertionError('no alternative')):
        actual, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical,
            branch_width=1, deadline_seconds=0, debug=True)
    _equal_fields(actual, expected)
    assert not actual.feasible and actual.peak_host_bytes == 128
    assert stats['first_solution_expanded'] is None and stats['backtracks'] == 0
    assert stats['stop_reason'] == 'width_exhausted'
