"""Incremental release priorities against a fresh full-ranking oracle."""
from dataclasses import replace
from functools import cmp_to_key

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, StorageBlock, SourcePreparation,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.storage import logical_source_problem
from lqcdweave.contraction.planning.optimizers import storage_schedule as scheduler
from correlator.planning.test_storage_schedule import _fixed_path, _physical, _problem
from correlator.planning.test_storage_peak_search import _chain_fixture


def full_ranking(frontier):
    goals = []
    for domain, keys in (('tensor', frontier.live-frontier.dag.outputs), ('block', frontier.raw)):
        for key in keys:
            if domain == 'tensor':
                size = frontier.dag.sizes.get(key, 0)
                mask = frontier.dag.release_masks.get(key, 0)
            else:
                block = frontier.dag.physical.blocks[key]
                size = block.host_bytes+block.device_bytes
                mask = frontier.dag.raw_release_masks.get(key, 0)
            mask &= ~frontier.done
            if size > 0 and mask:
                goals.append(scheduler._ReleaseGoal(domain, key, size, mask,
                    (mask & frontier.dag.computation_mask).bit_count()))
    return sorted(goals, key=cmp_to_key(scheduler._compare_goals))


def check_ranking(frontier):
    expected = full_ranking(frontier)
    assert frontier.release_goals(len(expected)+1) == expected
    for width in (1, 2):
        assert frontier.release_goals(width) == expected[:width]
    assert all(entry[0].obj.computations >= 0 for entry in frontier.ranking.entries.values())


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('late_initialization', [False, True])
def test_every_advance_and_undo_matches_full_ranking(backend, shared, late_initialization):
    problem, _, blocks, preparations = _problem(shared=shared)
    problem = replace(problem, backend=backend)
    if backend == 'cupy':
        blocks = {key:replace(block,device_bytes=block.host_bytes,host_bytes=16)
                  for key,block in blocks.items()}
    steps = (((0,('A','B')),(0,None),(1,None)) if shared else
             ((0,('A','B')),(1,('D','E')),(0,None),(1,None)))
    state,spans,workspaces = _fixed_path(problem,CostProfile(),steps)
    frontier = scheduler._Frontier(scheduler._Dag(state,spans,workspaces,
        _physical(problem,blocks,preparations)))
    first = frontier.advance(0,reversible=True) if late_initialization else None
    def visit():
        check_ranking(frontier)
        for index in sorted(frontier.ready):
            change=frontier.advance(index,reversible=True)
            visit()
            frontier.undo(change)
            check_ranking(frontier)
    visit()
    if first is not None:
        frontier.undo(first)
        check_ranking(frontier)


def test_indirect_prerequisite_changes_resident_priority():
    state,spans,workspaces,physical,_ = _chain_fixture()
    frontier=scheduler._Frontier(scheduler._Dag(state,spans,workspaces,physical))
    frontier.advance(0)
    check_ranking(frontier)
    identity=frontier.ranking.ids['tensor','A']
    before=frontier.ranking.entries[identity]
    assert before[0].obj.computations == 2
    assert 'A' not in dict(frontier.dag.nodes[2].references)
    change=frontier.advance(2,reversible=True)
    check_ranking(frontier)
    assert frontier.ranking.entries[identity][0].obj.computations == 1
    frontier.undo(change)
    check_ranking(frontier)
    assert frontier.ranking.entries[identity][0].obj.computations == 2


def test_new_large_intermediate_replaces_previous_release_target():
    shapes={'A':(2,),'D':(2,),'E':(2,),'B':(2,3),'C':(3,100)}
    problem=logical_source_problem(ContractionProblem(
        {key:TensorSpec(shape,'float64') for key,shape in shapes.items()},
        (ContractionTerm(('A','B','C'),'i,ik,kj->j'),ContractionTerm(('A','D'),'i,i->'),
         ContractionTerm(('E','B','C'),'i,ik,kj->j')),
        {key:SourceSpec(key,0) for key in shapes}))
    blocks={key:StorageBlock(key,'full',int(np.prod(shape))*8,host_bytes=int(np.prod(shape))*8)
            for key,shape in shapes.items()}
    preparations={key:SourcePreparation((key,)) for key in shapes}
    state,spans,workspaces=_fixed_path(problem,CostProfile(),
        ((1,('A','D')),(0,('B','C')),(2,None),(0,None)))
    frontier=scheduler._Frontier(scheduler._Dag(state,spans,workspaces,
        _physical(problem,blocks,preparations)))
    frontier.advance(0)
    previous=frontier.release_goal()
    assert previous.key == 'A' and previous.computations == 2
    frontier.advance(1)
    check_ranking(frontier)
    index,goal=next(frontier.choices(1,lambda:True))
    assert index == 2 and not previous.required & (1<<index)
    assert goal.key == frontier.dag.nodes[1].actions[0].tensor
    assert frontier.release_goal() == goal


def test_unaffected_priorities_are_reused_by_identity():
    shapes={'A':(100,100),'B':(60,100),'C':(20,100),
            **{key:(100,1) for key in ('X0','X1','Y0','Y1','Z0','Z1')}}
    pairs=(('A','X0'),('B','Y0'),('C','Z0'),('C','Z1'),('B','Y1'),('A','X1'))
    problem=logical_source_problem(ContractionProblem(
        {key:TensorSpec(shape,'float64') for key,shape in shapes.items()},
        tuple(ContractionTerm(pair,'ij,jk->ik') for pair in pairs),
        {key:SourceSpec(key,0) for key in shapes}))
    blocks={key:StorageBlock(key,'full',int(np.prod(shape))*8,host_bytes=int(np.prod(shape))*8)
            for key,shape in shapes.items()}
    preparations={key:SourcePreparation((key,)) for key in shapes}
    state,spans,workspaces=_fixed_path(problem,CostProfile(),tuple(enumerate(pairs)))
    frontier=scheduler._Frontier(scheduler._Dag(state,spans,workspaces,
        _physical(problem,blocks,preparations)))
    frontier.advance(0)
    frontier.advance(1)
    check_ranking(frontier)
    before=dict(frontier.ranking.entries)
    evaluations=frontier.target_evaluations
    change=frontier.advance(2,reversible=True)
    for identity,entry in before.items():
        assert frontier.ranking.entries[identity] is entry
    # Only C's owned snapshot and raw owner are newly ranked.
    assert frontier.target_evaluations-evaluations == 2
    check_ranking(frontier)
    frontier.undo(change)
    assert frontier.ranking.entries == before
    check_ranking(frontier)


def test_resource_replay_does_not_build_priority_index():
    state,spans,workspaces,physical,_=_chain_fixture()
    from unittest.mock import patch
    with patch.object(scheduler,'_ReleaseRanking',side_effect=AssertionError('ranking on replay')):
        scheduler._plan(scheduler._Dag(state,spans,workspaces,physical),range(len(spans)),
                        audited_computations=True,debug=True)


def test_budget_returns_complete_incumbent_without_forcing_a_suffix():
    state,spans,workspaces,physical,_=_chain_fixture()
    from unittest.mock import patch
    update=scheduler._ReleaseRanking.update
    calls=[]
    def track(ranking,index,tensors,delta):
        calls.append((len(ranking.frontier.order),delta))
        return update(ranking,index,tensors,delta)
    with patch.object(scheduler._ReleaseRanking,'update',track):
        plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,physical,max_states=1)
    assert plan.state.done and stats['suffix_steps']==0
    assert stats['first_solution_expanded']>=stats['nodes']
    assert stats['optimization_expanded']==1
    assert sum(delta==-1 for _,delta in calls)==stats['explored_frontiers']


def test_lazy_heap_is_bounded_under_repeated_backtracking():
    state,spans,workspaces,physical,_=_chain_fixture()
    frontier=scheduler._Frontier(scheduler._Dag(state,spans,workspaces,physical))
    frontier.advance(0)
    check_ranking(frontier)
    for _ in range(150):
        change=frontier.advance(2,reversible=True)
        frontier.undo(change)
    ranking=frontier.ranking
    assert ranking.compactions>0
    assert len(ranking.heap)<=2*len(ranking.entries)+128
    check_ranking(frontier)


def test_partial_rank_query_restores_heap_entries():
    state,spans,workspaces,physical,_=_chain_fixture()
    frontier=scheduler._Frontier(scheduler._Dag(state,spans,workspaces,physical))
    frontier.advance(0)
    expected=full_ranking(frontier)
    count=[0]
    def once():
        count[0]+=1
        return count[0]<=1
    assert frontier.release_goals(20,once)==expected[:1]
    check_ranking(frontier)
