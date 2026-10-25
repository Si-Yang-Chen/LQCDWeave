"""Weighted dense classes reproduce physical rows, ordering and numeric output."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers import compact_rows
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.integer_scoring import IntegerScorer, SCORERS
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contracts.correlator.api import SourcePreparation, OutputNode, OutputRecipe, describe_coordinates
from lqcdweave.contraction.results.api import TermValue, assemble_correlator
from correlator.planning.test_integer_row_reuse import repeated_problem
from correlator.planning.test_integer_search import make_problem


class CompactRowsTests(unittest.TestCase):
    def test_initial_index_interruption_does_not_publish_partial_groups(self):
        fast, reference = self.spaces(repeated_problem())
        before = fast.fingerprint(), tuple(fast.terms)
        checks = 0
        def checkpoint():
            nonlocal checks
            checks += 1
            if checks == 5:
                raise InterruptedError('initial row indexing')
        with self.assertRaises(InterruptedError):
            fast._index(checkpoint)
        self.assertEqual((fast.fingerprint(), tuple(fast.terms)), before)
        self.assertFalse(fast.rows)
        self.assertFalse(fast.groups)
        self.assertFalse(fast.group_ordinals)
        self.assertEqual(fast.next_group_ordinal, 0)
        self.compare(fast, reference)
        fast.reset()
        reference.reset()
        self.compare(fast, reference)

    def spaces(self, problem):
        with patch.object(compact_rows, 'COMPACT_DENSE_MIN_TERMS', 0):
            fast = IntegerSpace(problem, profile=CostProfile())
        self.assertIsNotNone(fast.scene_members)
        return fast, IntegerSpace(problem, profile=CostProfile(), debug=True)

    def compare(self, fast, reference):
        actual = list(fast.candidates(lambda: None))
        self.assertEqual(actual, list(reference.candidates(lambda: None)))
        self.assertEqual(fast.group_ordinals, reference.group_ordinals)
        self.assertEqual(fast.next_group_ordinal, reference.next_group_ordinal)
        self.assertEqual(fast.registry.data, reference.registry.data)
        self.assertEqual(fast.registry.calculations, reference.registry.calculations)
        for scene, members in enumerate(fast.scene_members):
            for physical in members:
                self.assertEqual(fast.terms[scene], reference.terms[physical])
        for identity, group in fast.groups.items():
            expanded = {}
            for scene, occurrence in group.items():
                for physical in fast.scene_members[scene]:
                    expanded[physical] = occurrence.positions, occurrence.output, occurrence.join, occurrence.cores
            expected = {scene: (o.positions, o.output, o.join, o.cores)
                        for scene, o in reference.groups[identity].items()}
            self.assertEqual(expanded, expected)
            self.assertEqual(fast.occurrence_count(identity), len(expected))
        for name in SCORERS:
            left, right = IntegerScorer(name), IntegerScorer(name)
            for operation, cost in actual:
                self.assertEqual(left(fast, operation, cost, lambda: None),
                                 right(reference, operation, cost, lambda: None), name)
                self.assertEqual(fast.tie_break(operation, lambda: None),
                                 reference.tie_break(operation, lambda: None))
        return actual

    def test_interleaved_classes_preserve_ordinals_all_scores_and_nested_undo(self):
        fast, reference = self.spaces(repeated_problem())
        self.assertEqual(fast._row_weights, (4, 4, 4, 4))

        def visit(depth):
            actual = self.compare(fast, reference)
            if not depth:
                return
            for operation, _ in actual[:3]:
                left = fast.apply(operation, lambda: None)
                right = reference.apply(operation, lambda: None)
                visit(depth - 1)
                fast.undo(left)
                reference.undo(right)
                self.compare(fast, reference)
        visit(3)

    def test_singletons_mixed_with_repeats_and_requested_output_layouts(self):
        terms = [(('Z', 'Y', 'X'), 'ij,jk,kl->il'),
                 (('Z', 'Y', 'W'), 'ij,jk,kl->li'),
                 (('Y', 'Z', 'X'), 'jk,ij,kl->li'),
                 (('Z', 'Y'), 'ij,jk->ki'),
                 (('X', 'Y', 'W'), 'ij,jk,kl->il')]
        terms = [terms[i] for i in (0, 1, 0, 2, 3, 1, 4, 0, 3, 0, 2)]
        fast, reference = self.spaces(make_problem(terms, {k: (2, 2) for k in 'WXYZ'}))
        while not fast.is_done():
            operations = self.compare(fast, reference)
            self.assertTrue(operations)
            operation = operations[0][0]
            fast.apply(operation, lambda: None)
            reference.apply(operation, lambda: None)
        self.compare(fast, reference)

    def test_incremental_heap_survives_position_only_updates_and_undo(self):
        for name in ('incremental_time', 'shared_savings', 'time_per_term',
                     'element_reduction', 'time_per_removed_element'):
            with self.subTest(scorer=name):
                fast, reference = self.spaces(repeated_problem())
                left_scorer, right_scorer = IntegerScorer(name), IntegerScorer(name)

                def visit(depth):
                    self.compare(fast, reference)
                    left_scorer.begin()
                    right_scorer.begin()
                    left = fast.rank_candidates(2, left_scorer, lambda: None)
                    right = reference.rank_candidates(2, right_scorer, lambda: None)
                    self.assertEqual(left.choices, right.choices)
                    if not depth:
                        return
                    for operation, _ in left.choices[:2]:
                        a = fast.apply(operation, lambda: None)
                        b = reference.apply(operation, lambda: None)
                        visit(depth-1)
                        fast.undo(a)
                        reference.undo(b)
                    self.compare(fast, reference)
                visit(3)

    def test_automatic_large_parallel_roots_preserve_every_physical_result(self):
        terms = [(('A', 'B', 'C'), 'ij,jk,kl->il'),
                 (('A', 'B', 'C'), 'ij,jk,kl->li')] * 2048
        problem = make_problem(terms, {'A': (2, 3), 'B': (3, 4), 'C': (4, 2)})
        preparations = {key: SourcePreparation(()) for key in problem.sources}
        options = dict(memory_limit=10**8, root_width=4, branch_width=4,
                       deadline_seconds=30, max_states=10000, kernel_batching=False)
        with patch.object(compact_rows, 'COMPACT_DENSE_MIN_TERMS', len(terms)+1):
            reference = solve_storage(problem, {}, preparations, component_workers=1, **options)
        result = solve_storage(problem, {}, preparations, component_workers=2, **options)
        self.assertEqual(result.state.seconds, reference.state.seconds)
        rng = np.random.default_rng(133)
        arrays = {key: rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
                  for key, spec in problem.specs.items()}
        values = replay(result.state, lambda key, source: arrays[key], shared_arrays=dict(arrays))
        self.assertEqual(len(values), len(terms))
        expected = arrays['A'] @ arrays['B'] @ arrays['C']
        for i, value in enumerate(values):
            np.testing.assert_allclose(value, expected if i % 2 == 0 else expected.T,
                                       rtol=1e-12, atol=1e-12)

    def test_complex_full_workflow_returns_every_original_output(self):
        problem = repeated_problem()
        options = dict(memory_limit=10**8, component_workers=1, root_width=2,
                       branch_width=2, deadline_seconds=20, max_states=10000,
                       kernel_batching=False)
        preparations = {key: SourcePreparation(()) for key in problem.sources}
        reference = solve_storage(problem, {}, preparations, debug=True, **options)
        with patch.object(compact_rows, 'COMPACT_DENSE_MIN_TERMS', 0):
            result = solve_storage(problem, {}, preparations, **options)
        self.assertEqual(result.state.seconds, reference.state.seconds)
        rng = np.random.default_rng(87)
        arrays = {key: rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
                  for key, spec in problem.specs.items()}
        actual = replay(result.state, lambda key, source: arrays[key], shared_arrays=dict(arrays))
        self.assertEqual(len(actual), len(problem.terms))
        references = []
        for value, term in zip(actual, problem.terms):
            args, labels = [], {}
            def number(axis):
                return labels.setdefault(axis, len(labels))
            for key, axes in zip(term.inputs, term.subscripts.inputs):
                args.extend((arrays[key], [number(axis) for axis in axes]))
            expected = np.einsum(*args, [number(axis) for axis in term.subscripts.output], optimize=False)
            np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
            references.append(expected)
        coordinates, _ = describe_coordinates()
        names = tuple(f'term-{i}' for i in range(len(actual)))
        coefficients = tuple((i+1)*0.13 + (i % 3)*0.2j for i in range(len(actual)))
        recipe = OutputRecipe(OutputNode('diagram_sum', tuple(zip(names, coefficients))),
                              names, coordinates)
        values = tuple(TermValue(name, coordinates.points[0].key, value)
                       for name, value in zip(names, actual))
        combined = assemble_correlator(SimpleNamespace(output_recipe=recipe, coordinates=coordinates), values)
        np.testing.assert_allclose(combined, sum(c*v for c, v in zip(coefficients, references)),
                                   rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
