"""Compact tail costs match the original stepwise domain and NumPy."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.terminal_completion import expand_completion
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from correlator.planning.test_integer_search import make_problem


def fixture(count=5):
    shapes, terms = {}, []
    for i in range(count):
        a, b = f'A{i}', f'B{i}'
        shapes.update({a: (2, 3), b: (3, 4)})
        terms.extend([((a, b), 'ij,jk->ki'), ((a, b), 'ab,bc->ac')])
    return make_problem(terms, shapes)


def stepwise(problem, profile):
    space = IntegerSpace(problem, profile=profile)
    operations, cost = [], 0.0
    while not space.is_done():
        op, seconds = next(space.search_candidates(lambda: None))
        space.apply(op, lambda: None)
        operations.append(op)
        cost += seconds
    return space.audit(operations, cost), cost


class TerminalCompletionTests(unittest.TestCase):
    def test_direct_summary_never_applies_or_undoes_and_reuses_whole_cost(self):
        space = IntegerSpace(fixture(80), profile=CostProfile())
        before = (tuple(space.terms), frozenset(space.available), space.fingerprint())
        with patch.object(space, 'apply', side_effect=AssertionError('summary applied a step')), \
             patch.object(space, 'undo', side_effect=AssertionError('summary undid a step')):
            summary = space.terminal_completion(lambda: None)
            self.assertIsNotNone(summary)
            self.assertIs(space.terminal_completion(lambda: None), summary)
        self.assertEqual((tuple(space.terms), frozenset(space.available), space.fingerprint()), before)
        self.assertEqual(space.terminal_statistics()['terminal_summary_hits'], 1)
        self.assertEqual(space.terminal_statistics()['terminal_summary_expansions'], 0)
        state = space.audit(expand_completion(summary, (), space.registry), summary.seconds)
        self.assertTrue(state.done)

    def test_summary_matches_stepwise_layout_sharing_and_complex_numpy_values(self):
        problem = fixture()
        profile = CostProfile()
        reference, cost = stepwise(problem, profile)
        space = IntegerSpace(problem, profile=profile)
        summary = space.terminal_completion(lambda: None)
        self.assertTrue(math.isclose(summary.seconds, cost, rel_tol=1e-12))
        state = space.audit(expand_completion(summary, (), space.registry), summary.seconds)
        rng = np.random.default_rng(1085)
        arrays = {k: rng.normal(size=s.shape)+1j*rng.normal(size=s.shape) for k,s in problem.specs.items()}
        left = replay(state, lambda k,s:arrays[k], shared_arrays=dict(arrays))
        right = replay(reference, lambda k,s:arrays[k], shared_arrays=dict(arrays))
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            expected = arrays[f'A{i//2}'] @ arrays[f'B{i//2}']
            if i%2 == 0:
                expected = expected.T
            np.testing.assert_allclose(a, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

    def test_nonterminal_work_keeps_original_search_and_winner_expands_once(self):
        problem = make_problem([(('A','B','C'),'ij,jk,kl->il'),
                                (('D','E'),'ij,jk->ki')],
                               {'A':(2,3),'B':(3,4),'C':(4,5),'D':(2,3),'E':(3,4)})
        space = IntegerSpace(problem, profile=CostProfile())
        self.assertIsNone(space.terminal_completion(lambda: None))
        engine = FingerprintDFS(space, root_width=4, branch_width=4)
        result = engine.run()
        self.assertEqual(result.status, 'FEASIBLE')
        self.assertEqual(result.statistics['terminal_summary_expansions'], 1)
        self.assertGreater(result.statistics['terminal_summary_builds'], 0)
        self.assertTrue(engine.best_plan.done)

    def test_undo_reuses_family_recipes_without_stale_availability(self):
        space = IntegerSpace(fixture(4), profile=CostProfile())
        original = space.terminal_completion(lambda: None)
        op, _ = next(space.search_candidates(lambda: None))
        token = space.apply(op, lambda: None)
        shorter = space.terminal_completion(lambda: None)
        self.assertIsNotNone(shorter)
        space.undo(token)
        self.assertIs(space.terminal_completion(lambda: None), original)
        # Restoring the completed prefix reuses the family recipes priced for
        # that same index instead of describing them again.
        self.assertGreater(space.terminal_statistics()['terminal_family_cache_hits'], 0)
        self.assertGreater(original.seconds, shorter.seconds)

    def test_partial_budget_does_not_publish_a_partial_summary(self):
        from lqcdweave.contraction.planning.optimizers.budget import BudgetExhausted
        space = IntegerSpace(fixture(4), profile=CostProfile())
        space._index(lambda: None)
        ticks = 0
        def checkpoint():
            nonlocal ticks
            ticks += 1
            if ticks == 5:
                raise BudgetExhausted('diagnostic')
        with self.assertRaises(BudgetExhausted):
            space.terminal_completion(checkpoint)
        self.assertFalse(space.is_done())
        self.assertFalse(space._terminal_completions.tails)
        self.assertIsNotNone(space.terminal_completion(lambda: None))

    def test_overlapping_terminal_families_fall_back(self):
        space = IntegerSpace(fixture(2), profile=CostProfile())
        space._index(lambda: None)
        a, b = sorted(space.terminal_members)[:2]
        # A malformed/overlapping family index is not eligible for the
        # independent-tail shortcut; the normal domain remains available.
        space.groups[b] = space.groups[a]
        self.assertIsNone(space.terminal_completion(lambda: None))
        self.assertEqual(space.terminal_statistics()['terminal_summary_fallbacks'], 1)

    def test_parallel_losing_roots_do_not_expand_execution_actions(self):
        from correlator.planning.test_parallel_roots import cycle
        from lqcdweave.contraction.planning.optimizers import parallel_components as pc
        problem, blocks, preparations = cycle()
        with patch.object(pc, '_materialize_winner', wraps=pc._materialize_winner) as expand:
            result = solve_storage(problem, blocks, preparations, memory_limit=10**7,
                component_workers=2, root_width=4, branch_width=4, max_states=1000,
                deadline_seconds=30)
        self.assertEqual(result.statistics['component_root_counts'], (4,))
        self.assertEqual(expand.call_count, 1)
        self.assertEqual(result.statistics['terminal_summary_expansions'], 1)
        self.assertTrue(result.plan.feasible)

    def test_unchanged_recipe_blocks_are_shared_between_different_tails(self):
        space = IntegerSpace(fixture(200), profile=CostProfile())
        original = space.terminal_completion(lambda: None)
        op, _ = next(space.search_candidates(lambda: None))
        space.apply(op, lambda: None)
        other = space.terminal_completion(lambda: None)
        common = {id(c) for c in original.calculations.chunks}.intersection(
                 id(c) for c in other.calculations.chunks)
        self.assertTrue(common)


if __name__ == '__main__':
    unittest.main()
