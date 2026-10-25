"""Ready-mask scheduling preserves the previous candidate and resource order."""
from heapq import nsmallest
from unittest.mock import patch

from .test_storage_schedule import _problem, _fixed_path, _physical
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.storage_schedule import (
    _Dag, _Frontier, _first_indices, schedule_fixed_dag,
)


def scanned_choices(self, width, checkpoint):
    """Original set-scanning selector, independent of the ready-mask cache."""
    seen, visited, limit = set(), set(), width
    while checkpoint():
        targets = self.release_goals(limit, checkpoint)
        for target in targets:
            identity = target.domain, target.key
            if identity in visited:
                continue
            visited.add(identity)
            for index in nsmallest(width, (i for i in self.ready if target.required & (1 << i))):
                if index not in seen:
                    seen.add(index)
                    yield index, target
                    if len(seen) == width:
                        return
        if len(targets) < limit:
            break
        limit *= 2
    for index in nsmallest(width, self.ready):
        if index not in seen:
            seen.add(index)
            yield index, None
            if len(seen) == width:
                return


def fixture(shared):
    problem, _, blocks, preparations = _problem(shared=shared)
    steps = (((0, ('A','B')), (0, None), (1, None)) if shared else
             ((0, ('A','B')), (1, ('D','E')), (0, None), (1, None)))
    state, spans, workspaces = _fixed_path(problem, CostProfile(), steps)
    return state, spans, workspaces, _physical(problem, blocks, preparations)


def test_ready_mask_stays_exact_through_forward_and_undo():
    for shared in (False, True):
        frontier = _Frontier(_Dag(*fixture(shared)))
        changes = []
        while frontier.ready:
            assert frontier.ready_mask == sum(1 << i for i in frontier.ready)
            for width in (1,4,16):
                actual = tuple(frontier.choices(width, lambda: True))
                expected = tuple(scanned_choices(frontier, width, lambda: True))
                assert actual == expected
            changes.append(frontier.advance(min(frontier.ready), reversible=True))
        assert frontier.ready_mask == 0
        for change in reversed(changes):
            frontier.undo(change)
            assert frontier.ready_mask == sum(1 << i for i in frontier.ready)


def test_full_schedule_matches_original_scanning_actions_and_resources():
    for shared in (False,True):
        for width in (1,4):
            args = fixture(shared)
            actual, stats = schedule_fixed_dag(*args, branch_width=width, max_states=64, debug=True)
            with patch.object(_Frontier, 'choices', scanned_choices):
                expected, old_stats = schedule_fixed_dag(*args, branch_width=width, max_states=64, debug=True)
            assert actual.state.actions == expected.state.actions
            assert actual.events == expected.events
            assert actual.memory == expected.memory
            assert actual.preparation_costs == expected.preparation_costs
            assert actual.feasible == expected.feasible
            assert actual.seconds == expected.seconds
            for key in ('nodes','explored_frontiers','backtracks','first_solution_expanded'):
                assert stats[key] == old_stats[key]


def test_sparse_high_indices_do_not_scan_the_ready_set():
    ready = {0,7,320,1017,20000,199999}
    required = sum(1 << i for i in (7,300,1017,199999))
    for width in (0,1,4,8):
        assert list(_first_indices(sum(1 << i for i in ready) & required,width)) == nsmallest(
            width, (i for i in ready if required & (1 << i)))
