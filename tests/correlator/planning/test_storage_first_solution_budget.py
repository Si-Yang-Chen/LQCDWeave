"""Storage budgets cover optimization after the first searched feasible order."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.planning.optimizers import storage_schedule as scheduler
from lqcdweave.contraction.execution.api import replay_scene_storage
from correlator.planning.test_storage_peak_search import _chain_fixture
from correlator.planning.test_storage_schedule import _physical, _ArrayLoader


def test_zero_deadline_cannot_stop_before_first_searched_path():
    state,spans,workspaces,physical,arrays=_chain_fixture()
    plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,physical,
        deadline_seconds=0,max_states=1,debug=True)
    assert plan.feasible and plan.state.done
    assert stats['first_solution_expanded']==len(spans)
    assert stats['optimization_expanded']==0
    assert stats['stop_reason']=='deadline' and stats['suffix_steps']==0
    expected=(np.einsum('ij,jk,kl->il',arrays['A'],arrays['B'],arrays['C']),
              arrays['A']@arrays['D'],arrays['E']@arrays['F'])
    for actual,reference in zip(replay_scene_storage(plan,_ArrayLoader(arrays)),expected):
        np.testing.assert_allclose(actual,reference)


def test_state_budget_is_fresh_after_first_solution():
    state,spans,workspaces,physical,_=_chain_fixture()
    plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,physical,max_states=1)
    assert plan.feasible
    assert stats['first_solution_expanded']==4
    assert stats['explored_frontiers']==5
    assert stats['optimization_expanded']==1 and stats['stop_reason']=='max_states'


def test_deadline_starts_at_first_path_and_does_not_include_its_search():
    state,spans,workspaces,physical,_=_chain_fixture()
    now=[0.0]
    advance=scheduler._Frontier.advance
    def slow(frontier,index,*args,**kwargs):
        result=advance(frontier,index,*args,**kwargs)
        if kwargs.get('reversible'):
            now[0]+=600
        return result
    with patch.object(scheduler._Frontier,'advance',slow):
        plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,physical,
            deadline_seconds=1,clock=lambda:now[0])
    assert plan.feasible
    assert stats['first_solution_seconds']==2400
    assert stats['optimization_seconds']==600
    assert stats['optimization_expanded']==1 and stats['stop_reason']=='deadline'


def test_non_improving_first_path_still_starts_budget():
    state,spans,workspaces,physical,_=_chain_fixture()
    dag=scheduler._Dag(state,spans,workspaces,physical)
    optimal=scheduler._plan(dag,(1,0,2,3),audited_computations=True)
    plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,physical,
        baseline=optimal,deadline_seconds=0,max_states=1)
    assert stats['first_solution_expanded']==4
    assert stats['complete_improvements']==0
    assert stats['stop_reason']=='deadline'
    assert plan is optimal


def test_later_improvements_do_not_reset_first_solution_or_budget_counters():
    state,spans,workspaces,physical,_=_chain_fixture()
    plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,physical,max_states=100)
    assert plan.feasible and stats['complete_improvements']>=2
    assert stats['first_solution_expanded']==4
    assert stats['optimization_expanded']==stats['explored_frontiers']-4
    assert stats['stop_reason']=='width_exhausted'


def test_width_exhaustion_without_feasible_path_is_not_a_budget_stop():
    state,spans,workspaces,roomy,_=_chain_fixture()
    tight=_physical(state.problem,roomy.blocks,roomy.preparations,cap=1)
    plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,tight,
        deadline_seconds=0,max_states=1)
    assert not plan.feasible
    assert stats['explored_frontiers']>1
    assert stats['first_solution_expanded'] is None
    assert stats['optimization_expanded']==0 and stats['stop_reason']=='width_exhausted'


def test_infeasible_fallback_cannot_block_a_feasible_incumbent():
    state,spans,workspaces,physical,_=_chain_fixture()
    dag=scheduler._Dag(state,spans,workspaces,physical)
    fallback=scheduler._plan(dag,range(len(spans)),audited_computations=True)
    invalid=replace(fallback,feasible=False,peak_host_bytes=0)
    plan,stats=scheduler.schedule_fixed_dag(state,spans,workspaces,physical,
        baseline=invalid,deadline_seconds=0)
    assert plan.feasible and stats['first_solution_expanded']==4
