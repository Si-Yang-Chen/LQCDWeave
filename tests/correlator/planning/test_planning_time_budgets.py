"""Separate one-hour defaults and explicit shorter search deadlines."""
from unittest.mock import patch

import pytest

from lqcdweave.contraction.planning.optimizers.budget import SearchBudget, BudgetExhausted
from lqcdweave.contraction.planning.optimizers.options import effective_search_options
from lqcdweave.contraction.planning.optimizers.search import solve_shortest_path, solve_layered_search
from lqcdweave.contraction.planning.optimizers.storage_schedule import schedule_fixed_dag, _Frontier
from lqcdweave.contraction.performance.api import CostProfile
from correlator.planning.test_search_space import _Space
from correlator.planning.test_storage_schedule import _problem, _fixed_path, _physical


def test_counted_checkpoint_matches_check_then_transition_at_limits():
    from lqcdweave.contraction.planning.optimizers.protocols import SearchEvent
    now = [0.0]
    left, right = (SearchBudget(deadline_seconds=10, max_transitions=2,
                               clock=lambda: now[0]) for _ in range(2))
    for _ in range(2):
        left.check(transition=True)
        right.check()
        right.consume(SearchEvent('transition'))
        assert left.transitions == right.transitions
    for budget in (left, right):
        with pytest.raises(BudgetExhausted, match='max_transitions'):
            budget.check(transition=True)
        assert budget.transitions == 2
    now[0] = 10.0
    for budget in (left, right):
        with pytest.raises(BudgetExhausted, match='deadline'):
            budget.check(transition=True)
        assert budget.transitions == 2


def test_default_budget_stops_at_one_hour_and_explicit_deadlines_still_work():
    now = [100.0]
    budget = SearchBudget(clock=lambda: now[0])
    now[0] = 3699.0
    budget.check()
    now[0] = 3700.0
    with pytest.raises(BudgetExhausted, match='deadline'):
        budget.check()
    shorter = SearchBudget(deadline_seconds=2, clock=lambda: now[0])
    now[0] += 2
    with pytest.raises(BudgetExhausted, match='deadline'):
        shorter.check()


@pytest.mark.parametrize('optimizer', ['anytime', 'exact', 'dynamic_programming'])
def test_workflow_option_resolution_keeps_one_hour_default_and_overrides(optimizer):
    assert effective_search_options(optimizer, {})['deadline_seconds'] == 3600
    assert effective_search_options(optimizer, {'deadline_seconds': 7})['deadline_seconds'] == 7


@pytest.mark.parametrize('solver', [solve_shortest_path, solve_layered_search])
def test_exact_search_can_finish_after_previous_ten_minute_default(solver):
    def run(**options):
        now = [0.0]
        space = _Space({'start': [('ready', 'a', 1)], 'ready': [('done', 'b', 1)]}, lambda _: 1)
        original = space.successors
        def successors(state):
            for candidate in original(state):
                now[0] += 700
                yield candidate
        space.successors = successors
        return solver(space, clock=lambda: now[0], **options)
    assert run().status == 'OPTIMAL'
    short = run(deadline_seconds=600)
    assert short.status == 'UNSOLVED'
    assert short.stop_reason == 'deadline'


def test_storage_gets_a_fresh_hour_after_compute_budget_expires():
    now = [0.0]
    clock = lambda: now[0]
    compute = SearchBudget(clock=clock)
    now[0] = 3600.0
    with pytest.raises(BudgetExhausted, match='deadline'):
        compute.check()
    problem, _, blocks, preparations = _problem()
    profile = CostProfile()
    state, spans, workspaces = _fixed_path(problem, profile,
        ((0, ('A', 'B')), (1, ('D', 'E')), (0, None), (1, None)))
    physical = _physical(problem, blocks, preparations, profile=profile)
    advance = _Frontier.advance
    def slow_advance(frontier, node, *args, **kwargs):
        result = advance(frontier, node, *args, **kwargs)
        if not args:
            now[0] += 30
        return result
    with patch.object(_Frontier, 'advance', slow_advance):
        plan, statistics = schedule_fixed_dag(state, spans, workspaces, physical, clock=clock)
        assert plan.feasible
        assert statistics['explored_frontiers'] >= statistics['nodes']
        _, short = schedule_fixed_dag(state, spans, workspaces, physical, clock=clock, deadline_seconds=1)
        assert short['first_solution_expanded'] >= short['nodes']
        assert short['optimization_seconds'] >= 1
        assert short['stop_reason'] == 'deadline'
