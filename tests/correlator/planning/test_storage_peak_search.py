"""Peak-bounded scheduling against exhaustive order and independent resources."""
from dataclasses import replace
from itertools import permutations
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, SourcePreparation, StorageBlock, TensorSpec,
)
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.storage import logical_source_problem
from lqcdweave.contraction.planning.optimizers import storage_schedule as scheduler
from lqcdweave.contraction.planning.optimizers.options import effective_search_options
from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from correlator.planning.test_storage_schedule import _fixed_path, _physical, _problem, _ArrayLoader


def _chain_fixture(dimensions=(200, 30, 4, 300)):
    m, k, n, r = dimensions
    shapes = {"A": (m, k), "B": (k, n), "C": (n, 1),
              "D": (k, 1), "E": (m, r), "F": (r, 1)}
    problem = logical_source_problem(ContractionProblem(
        {key: TensorSpec(shape, 'float64') for key, shape in shapes.items()},
        (ContractionTerm(('A', 'B', 'C'), 'ij,jk,kl->il'),
         ContractionTerm(('A', 'D'), 'ij,jl->il'),
         ContractionTerm(('E', 'F'), 'ij,jl->il')),
        {key: SourceSpec(key, 0) for key in shapes}))
    arrays = {key: np.arange(np.prod(shape), dtype=float).reshape(shape) / 17
              for key, shape in shapes.items()}
    blocks = {key: StorageBlock(key, 'full', array.nbytes, host_bytes=array.nbytes)
              for key, array in arrays.items()}
    preparations = {key: SourcePreparation((key,)) for key in shapes}
    profile = CostProfile()
    state, spans, workspaces = _fixed_path(problem, profile,
        ((1, ('A', 'D')), (2, ('E', 'F')), (0, ('B', 'C')), (0, None)))
    return state, spans, workspaces, _physical(problem, blocks, preparations, cap=10**9), arrays


def test_peak_excess_tries_lower_ranked_work_and_backtracks_to_previous_step():
    state, spans, workspaces, roomy, arrays = _chain_fixture()
    narrow, _ = scheduler.schedule_fixed_dag(state, spans, workspaces, roomy, branch_width=1)
    wide, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, roomy,
                                              branch_width=2, debug=True)
    assert wide.peak_host_bytes < narrow.peak_host_bytes
    assert stats['backtracks'] > 0 and stats['peak_pruned'] > 0
    assert stats['complete_improvements'] > 1
    assert [a.term_index for a in wide.state.actions if a.kind == 'contract'] == [2, 1, 0, 0]
    tight = _physical(state.problem, roomy.blocks, roomy.preparations, cap=wide.peak_host_bytes)
    failed, _ = scheduler.schedule_fixed_dag(state, spans, workspaces, tight, branch_width=1)
    rescued, _ = scheduler.schedule_fixed_dag(state, spans, workspaces, tight, branch_width=2)
    assert not failed.feasible and rescued.feasible
    assert rescued.peak_host_bytes == wide.peak_host_bytes
    expected = (np.einsum('ij,jk,kl->il', arrays['A'], arrays['B'], arrays['C']),
                arrays['A'] @ arrays['D'], arrays['E'] @ arrays['F'])
    for actual, reference in zip(replay_scene_storage(rescued, _ArrayLoader(arrays)), expected):
        np.testing.assert_allclose(actual, reference)


def test_small_dags_match_exhaustive_topological_order_reference():
    rng = np.random.default_rng(8192)
    for _ in range(12):
        state, spans, workspaces, physical, _ = _chain_fixture(tuple(rng.integers(2, 18, size=4)))
        dag = scheduler._Dag(state, spans, workspaces, physical)
        peaks = []
        for order in permutations(range(len(dag.nodes))):
            positions = {node: index for index, node in enumerate(order)}
            if all(positions[parent] < positions[node] for node in order for parent in dag.nodes[node].parents):
                # Each reference order also traverses the independent resource auditor.
                plan = scheduler._plan(dag, order, audited_computations=True, debug=True)
                peaks.append(plan.peak_host_bytes)
        actual, _ = scheduler.schedule_fixed_dag(state, spans, workspaces, physical,
                                                branch_width=4, max_states=10000, debug=True)
        assert actual.peak_host_bytes == min(peaks)


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('shared', [False, True])
def test_local_undo_restores_every_frontier_field(backend, shared):
    problem, _, blocks, preparations = _problem(shared=shared)
    problem = replace(problem, backend=backend)
    if backend == 'cupy':
        blocks = {key: replace(block, device_bytes=block.host_bytes, host_bytes=16,
                              staging_host_bytes=24, staging_device_bytes=32)
                  for key, block in blocks.items()}
    preparations = {key: replace(prep, workspace_host_bytes=40,
                                 workspace_device_bytes=48 if backend == 'cupy' else 0)
                    for key, prep in preparations.items()}
    steps = (((0, ('A', 'B')), (0, None), (1, None)) if shared else
             ((0, ('A', 'B')), (1, ('D', 'E')), (0, None), (1, None)))
    state, spans, workspaces = _fixed_path(problem, CostProfile(), steps)
    physical = _physical(problem, blocks, preparations)
    dag = scheduler._Dag(state, spans, workspaces, physical)
    frontier = scheduler._Frontier(dag)
    def snapshot():
        return (dict(frontier.remaining), tuple(frontier.indegree), set(frontier.ready),
                set(frontier.live), set(frontier.prepared), set(frontier.raw), set(frontier.opened),
                frontier.host, frontier.device, frontier.peak_host, frontier.peak_device,
                frontier.raw_host, frontier.raw_device, frontier.done, tuple(frontier.order))
    def visit():
        before = snapshot()
        for index in sorted(frontier.ready):
            change = frontier.advance(index, reversible=True)
            if frontier.ready:
                visit()
            else:
                frontier.finish()
            frontier.undo(change)
            assert snapshot() == before
    visit()
    scheduler.schedule_fixed_dag(state, spans, workspaces, physical, debug=True)


def test_final_output_copy_is_pruned_even_when_every_computation_fits():
    names = tuple('ABCDEFGHIJKLMNOP')
    problem = logical_source_problem(ContractionProblem(
        {key: TensorSpec((1,), 'float64') for key in names},
        tuple(ContractionTerm(names[i:i+2], 'i,i->') for i in range(0, 16, 2)),
        {key: SourceSpec(key, 0) for key in names}))
    blocks = {key: StorageBlock(key, 'full', 8, host_bytes=8) for key in names}
    preparations = {key: SourcePreparation((key,)) for key in names}
    state, spans, workspaces = _fixed_path(problem, CostProfile(), tuple((i, None) for i in range(8)))
    physical = _physical(problem, blocks, preparations, cap=127)
    plan, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical, debug=True)
    assert not plan.feasible and plan.peak_host_bytes == 128
    assert max(point.host_bytes for point in plan.memory if point.phase != 'output_copy') < 128
    assert stats['complete_improvements'] == 0


def test_complete_candidates_are_not_reaudited_during_search():
    state, spans, workspaces, physical, _ = _chain_fixture()
    with patch.object(scheduler, '_plan', wraps=scheduler._plan) as emit:
        plan, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, physical)
    assert plan.feasible and stats['complete_improvements'] > 1
    assert emit.call_count == stats['audited_schedules'] == 2


def test_shared_consumers_do_not_hide_the_next_ranked_operation():
    shapes = {'A': (100, 100), 'B': (60, 100), 'C': (20, 100),
              **{key: (100, 1) for key in ('X0','X1','Y0','Y1','Z0','Z1')}}
    pairs = (('A','X0'),('B','Y0'),('C','Z0'),('C','Z1'),('B','Y1'),('A','X1'))
    problem = logical_source_problem(ContractionProblem(
        {key:TensorSpec(shape,'float64') for key,shape in shapes.items()},
        tuple(ContractionTerm(pair,'ij,jk->ik') for pair in pairs),
        {key:SourceSpec(key,0) for key in shapes}))
    blocks = {key:StorageBlock(key,'full',int(np.prod(shape))*8,host_bytes=int(np.prod(shape))*8)
              for key,shape in shapes.items()}
    preps = {key:SourcePreparation((key,)) for key in shapes}
    state,spans,workspaces = _fixed_path(problem,CostProfile(),tuple(enumerate(pairs)))
    frontier = scheduler._Frontier(scheduler._Dag(state,spans,workspaces,_physical(problem,blocks,preps)))
    for index in range(3):
        frontier.advance(index)
    # A's raw block and owned snapshot both point at node 5. The next
    # operation must still consume B (4), ahead of the earlier node C (3).
    assert [index for index,_ in frontier.choices(2,lambda:True)] == [5,4]


@pytest.mark.parametrize('limited_domain', ['host', 'device'])
def test_gpu_metadata_enforces_each_budget_independently(limited_domain):
    state, _, _, physical, _ = _chain_fixture()
    problem = replace(state.problem, backend='cupy')
    blocks = {key: replace(block, device_bytes=2*block.host_bytes, staging_host_bytes=24)
              for key, block in physical.blocks.items()}
    state, spans, workspaces = _fixed_path(problem, CostProfile(),
        ((1, ('A', 'D')), (2, ('E', 'F')), (0, ('B', 'C')), (0, None)))
    roomy = MacroSearchSpace(problem, blocks, physical.preparations, memory_limit=10**9,
                            host_memory_limit=10**9)
    best, _ = scheduler.schedule_fixed_dag(state, spans, workspaces, roomy, debug=True)
    tight = MacroSearchSpace(problem, blocks, physical.preparations,
        memory_limit=best.peak_device_bytes if limited_domain == 'device' else 10**9,
        host_memory_limit=best.peak_host_bytes if limited_domain == 'host' else 10**9)
    plan, stats = scheduler.schedule_fixed_dag(state, spans, workspaces, tight, debug=True)
    assert plan.feasible and stats['peak_pruned'] > 0
    assert plan.peak_host_bytes <= tight.host_memory_limit
    assert plan.peak_device_bytes <= tight.memory_limit


@pytest.mark.parametrize('workers', [1, 2])
def test_public_width_reaches_shared_serial_and_parallel_storage_stage(workers):
    problem, _, blocks, preparations = _problem()
    result = solve_storage(problem, blocks, preparations, memory_limit=10**6,
        root_width=1, branch_width=1, component_workers=workers, storage_branch_width=2,
        deadline_seconds=10)
    assert result.statistics['storage_schedule']['branch_width'] == 2
    assert result.diagnostic['strategy_options']['storage_branch_width'] == 2


@pytest.mark.parametrize('width', [0, -1, True, 1.5])
def test_invalid_width_is_rejected_at_public_option_boundary(width):
    with pytest.raises((TypeError, ValueError)):
        effective_search_options('anytime', {'storage_branch_width': width})


def test_exact_mode_does_not_accept_heuristic_storage_width():
    with pytest.raises(ValueError, match='unsupported options'):
        effective_search_options('exact', {'storage_branch_width': 2})
