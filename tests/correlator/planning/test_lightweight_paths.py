"""Whole-task baryon first paths retain sharing and executable arithmetic."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from itertools import combinations
from dataclasses import replace

import numpy as np

from correlator.planning.test_candidate_metadata import join_fixture
from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem
from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionTerm, SourcePreparation, TensorSpec,
)
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.canonical_operations import describe
from lqcdweave.contraction.planning.optimizers.lightweight_paths import LightweightPaths
from lqcdweave.contraction.planning.optimizers.budget import SearchBudget
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.planning.path_planning import PathSearchExhausted


class LightweightPathTests(unittest.TestCase):
    def test_compiled_plain_pairs_match_independent_baryon_descriptions_after_updates(self):
        disjoint = join_fixture().problem
        # The ordinary fixture has three disconnected legs, so every plain
        # pair is correctly rejected as an outer product. Add one shared source
        # axis to exercise a legal pair before the whole baryon join.
        specs = dict(disjoint.specs)
        for name in ('a', 'b'):
            original = specs[name]
            specs[name] = TensorSpec((*original.shape, 2), original.dtype)
        shared = replace(disjoint, specs=specs,
                         terms=(ContractionTerm(('a', 'b', 'c'), 'adx,bex,cf->'),) * 3)
        for problem in (disjoint, sparse_spin_problem(), shared):
            with self.subTest(problem=len(problem.terms)):
                space = IntegerSpace(problem, profile=CostProfile())

                def compare_rows():
                    space._index(lambda: None)
                    for scene, term in enumerate(space.terms):
                        if term.done:
                            continue
                        plain = tuple(i for i, data in enumerate(term.factors)
                                      if space.registry.data[data].core is None)
                        expected = []
                        state = space._state(scene)
                        for positions in combinations(range(len(plain)), 2):
                            try:
                                action, spec, _, output, _ = describe(
                                    state, 0, positions, profile=space.profile)
                            except ValueError:
                                continue
                            expected.append((action.tensor, tuple(plain[i] for i in positions),
                                             tuple(output), spec.shape, spec.dtype))
                        actual = []
                        for identity, occurrence in space.rows[scene]:
                            if occurrence.join is not None:
                                continue
                            calculation = space.registry.calculations[identity]
                            if calculation.recipe.kind != 'contract':
                                continue
                            spec = space.registry.data[calculation.result].spec
                            actual.append((calculation.recipe.tensor, occurrence.positions,
                                           occurrence.output, spec.shape, spec.dtype))
                        self.assertEqual(sorted(actual, key=repr), sorted(expected, key=repr))

                compare_rows()
                pairs = [operation for operation, _ in space.candidates(lambda: None)
                         if space.registry.calculations[operation.calculation].recipe.kind == 'contract']
                if problem is shared:
                    self.assertTrue(pairs)
                if pairs:
                    token = space.apply(pairs[0], lambda: None)
                    compare_rows()
                    space.undo(token)
                    compare_rows()

    def test_scoped_join_reuses_checked_sparse_spin_entries(self):
        problem = sparse_spin_problem()
        original = problem.baryon_joins[0]
        with patch.object(BaryonJoinSpec, '__post_init__',
                          side_effect=AssertionError('checked join was reconstructed')):
            space = IntegerSpace(problem, profile=CostProfile())
        scoped = space.contexts[0].baryon_joins[0]
        self.assertIs(scoped.sparse_spin, original.sparse_spin)
        self.assertEqual(scoped.spin_labels,
                         tuple(space.symbols[0].index(label) for label in original.spin_labels))
        self.assertEqual(scoped.mode_labels,
                         tuple(space.symbols[0].index(label) for label in original.mode_labels))

    def test_core_rows_reuse_only_matching_scoped_descriptions(self):
        space = IntegerSpace(join_fixture().problem, profile=CostProfile())
        tuple(space.candidates(lambda: None))
        self.assertEqual(space.row_descriptions, 2)
        self.assertEqual(space.row_description_reuses, 1)
        joined = tuple(tuple(calculation for calculation, occurrence in space.rows[scene]
                             if occurrence.join is not None) for scene in range(3))
        self.assertEqual(joined[0], joined[1])
        self.assertNotEqual(joined[0], joined[2])

    def test_full_baryon_task_uses_one_workspace_and_matches_independent_contraction(self):
        problem = join_fixture().problem
        preparations = {key: SourcePreparation(()) for key in problem.sources}
        rng = np.random.default_rng(614)
        arrays = {key: rng.normal(size=spec.shape) for key, spec in problem.specs.items()}
        expected = tuple(np.einsum(
            'bac,def,ad,be,cf->' if join.spin_labels != problem.baryon_joins[0].spin_labels
            else 'abc,def,ad,be,cf->',
            arrays['spin'], arrays['mode'], arrays['a'], arrays['b'], arrays['c'])
            for join in problem.baryon_joins)
        with patch('lqcdweave.contraction.planning.optimizers.graph_session.FingerprintDFS',
                   side_effect=AssertionError('built a retained DFS graph')), patch.object(
                       IntegerSpace, '_part', side_effect=AssertionError('hashed an unused search fingerprint')):
            result = solve_storage(problem, {}, preparations, memory_limit=10**8,
                                   deadline_seconds=30, max_states=64)
        self.assertEqual(result.diagnostic['engine'], 'lightweight_integer_paths')
        self.assertTrue(result.plan.feasible)
        self.assertTrue(result.plan.state.done)
        self.assertEqual(result.statistics['retained_states'], 0)
        self.assertEqual(result.statistics['retained_connections'], 0)
        self.assertTrue(result.statistics['storage_schedule']['completed'])
        self.assertEqual(result.statistics['component_scenes'], ((0, 1, 2),))
        self.assertLessEqual(result.statistics['first_integer_operation_seconds'],
                             result.statistics['first_logical_complete_seconds'])
        self.assertLessEqual(result.statistics['first_logical_complete_seconds'],
                             result.first_solution_seconds)
        self.assertGreaterEqual(result.statistics['logical_audit_seconds'], 0)
        self.assertGreaterEqual(result.statistics['physical_schedule_seconds'], 0)
        np.testing.assert_allclose(
            replay(result.state, lambda key, source: arrays[key]), expected,
            rtol=1e-12, atol=1e-12)

    def test_exhausted_budget_does_not_publish_unverified_path(self):
        problem = join_fixture().problem
        preparations = {key: SourcePreparation(()) for key in problem.sources}
        with self.assertRaises(PathSearchExhausted) as caught:
            solve_storage(problem, {}, preparations, memory_limit=10**8,
                          deadline_seconds=30, max_states=0)
        report = caught.exception.search_result
        self.assertEqual(report.status, 'UNSOLVED')
        self.assertEqual(report.statistics['expanded_states'], 0)
        self.assertIsNone(report.first_solution_seconds)

    def test_last_successful_apply_is_admitted_after_deadline_crossing(self):
        now = [0.0]
        operation = SimpleNamespace(seconds=1.0)

        class LastStepSpace:
            done = False

            def reset(self):
                self.done = False

            def is_done(self):
                return self.done

            def rank_candidates(self, *args, **kwargs):
                return None

            def search_candidates(self, checkpoint):
                return ((operation, operation.seconds),)

            def apply(self, selected, checkpoint):
                self.done = True
                now[0] = 2.0
                return object()

            def undo(self, token):
                self.done = False

            def release(self):
                pass

        accepted = object()
        budget = SearchBudget(deadline_seconds=1, clock=lambda: now[0], max_states=1)
        search = LightweightPaths(LastStepSpace(), budget, None,
                                  root_width=1, branch_width=1,
                                  on_complete=lambda operations, cost: accepted).run()
        self.assertIs(search.best_plan, accepted)
        self.assertEqual(search.complete_routes, 1)
        self.assertEqual(search.first_solution_seconds, 2.0)


if __name__ == '__main__':
    unittest.main()
