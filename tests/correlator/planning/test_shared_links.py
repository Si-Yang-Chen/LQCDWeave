"""Shared connection updates preserve partial users and independent algebra."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from correlator.planning.test_integer_search import make_problem, solve
from correlator.planning.test_candidate_metadata import join_fixture
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers import integer_space
from lqcdweave.contraction.execution.api import replay


def signature(space):
    return sorted((space.registry.calculations[identity].recipe.tensor,
                   tuple((o.scene, o.positions, o.output, o.join, o.cores)
                         for o in users.values())) for identity, users in space.groups.items()
                  if identity not in space.layout_variants)


class SharedLinkTests(unittest.TestCase):
    def test_prediction_residency_matches_full_scan_through_apply_and_undo(self):
        from lqcdweave.contracts.correlator.api import SourceSpec
        original = join_fixture().problem
        for backend in ('numpy', 'cupy'):
            sources = {key: SourceSpec('shared-region', 1024, host_region_bytes=512,
                snapshot_bytes=128 if i % 2 else None)
                for i, key in enumerate(original.sources)}
            problem = replace(original, sources=sources, backend=backend)
            space = IntegerSpace(problem, profile=CostProfile())
            def check():
                for scene in range(len(space.terms)):
                    state = space._state(scene)
                    self.assertEqual(state._resident_totals, replace(state)._resident_totals)
            check()
            operation, _ = next(space.candidates(lambda: None))
            token = space.apply(operation, lambda: None)
            check()
            space.undo(token)
            check()

    def test_partial_scope_updates_do_not_consume_other_users(self):
        problem = make_problem([
            (('A', 'B', 'C'), 'ij,jk,ki->'),
            (('A', 'B', 'D'), 'ij,jk,ki->'),
            (('A', 'D'), 'ij,ji->'),
        ], {name: (3, 3) for name in 'ABCD'})
        space = IntegerSpace(problem, profile=CostProfile())
        original_terms = problem.terms
        initial = tuple(space.terms)
        choices = list(space.candidates(lambda: None))
        operation = next(op for op, _ in choices
                         if set(space.registry.calculations[op.calculation].recipe.operands) == {'A', 'B'})
        before = signature(space)
        token = space.apply(operation, lambda: None)
        space._index(lambda: None)
        self.assertEqual(space.terms[2], initial[2])
        self.assertIs(problem.terms, original_terms)
        result = space.registry.data[operation.result].name
        self.assertTrue(any(result in space.registry.calculations[i].recipe.operands
                            for i in space.groups))
        space.undo(token)
        with patch.object(space, '_dense_recipe', side_effect=AssertionError('re-estimated on undo')):
            space._index(lambda: None)
        self.assertEqual(signature(space), before)
        self.assertEqual(tuple(space.terms), initial)
        arrays = {key: np.arange(9).reshape(3, 3) + (i + 1j)
                  for i, key in enumerate('ABCD')}
        planned = solve(problem, root_width=1, branch_width=1, include_ties=False,
                        scorer='time_per_term', deadline_seconds=30)
        expected = (np.einsum('ij,jk,ki->', arrays['A'], arrays['B'], arrays['C']),
                    np.einsum('ij,jk,ki->', arrays['A'], arrays['B'], arrays['D']),
                    np.einsum('ij,ji->', arrays['A'], arrays['D']))
        np.testing.assert_allclose(replay(planned.state, lambda key, _: arrays[key]), expected)

    def test_unaffected_connection_survives_without_description(self):
        problem = make_problem([(('A', 'B', 'C', 'D'), 'ij,jk,kl,lm->im')],
                               {name: (2, 2) for name in 'ABCD'})
        space = IntegerSpace(problem, profile=CostProfile())
        candidates = list(space.candidates(lambda: None))
        chosen = next(op for op, _ in candidates
                      if space.registry.calculations[op.calculation].recipe.operands == ('A', 'B'))
        space.apply(chosen, lambda: None)
        with patch.object(space, '_dense_recipe', wraps=space._dense_recipe) as describe:
            space._index(lambda: None)
        self.assertEqual(describe.call_count, 1)  # X—C; C—D remains registered.
        self.assertGreater(space.links.reused_connections, 0)

    def test_baryon_class_is_priced_once_and_matches_reference(self):
        problem = join_fixture().problem
        space = IntegerSpace(problem, profile=CostProfile())
        with patch.object(integer_space, 'describe', wraps=integer_space.describe) as describe:
            list(space.candidates(lambda: None))
            self.assertEqual(describe.call_count, 2)  # two distinct spin bindings
            initial = signature(space)
            operation, _ = next(space.candidates(lambda: None))
            token = space.apply(operation, lambda: None)
            space._index(lambda: None)
            space.undo(token)
            space._index(lambda: None)
            self.assertEqual(signature(space), initial)
            self.assertEqual(describe.call_count, 2)

    def test_retained_shared_axis_matches_complete_descriptions(self):
        problem = make_problem([(('A', 'B', 'C'), 'bi,bi,bi->b')],
                               {name: (2, 3) for name in 'ABC'})
        fast, reference = (IntegerSpace(problem, profile=CostProfile(), debug=debug)
                           for debug in (False, True))
        for _ in range(3):
            left, right = list(fast.candidates(lambda: None)), list(reference.candidates(lambda: None))
            self.assertEqual(signature(fast), signature(reference))
            if not left:
                break
            operation = left[0][0]
            name = fast.registry.data[operation.result].name
            other = next(op for op, _ in right if reference.registry.data[op.result].name == name)
            fast.apply(operation, lambda: None)
            reference.apply(other, lambda: None)
        self.assertTrue(fast.is_done())


if __name__ == '__main__':
    unittest.main()
