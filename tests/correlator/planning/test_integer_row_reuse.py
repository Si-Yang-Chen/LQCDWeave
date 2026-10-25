"""Repeated dense-row descriptions agree with the original per-row path."""
from itertools import combinations
import unittest
from unittest.mock import patch

from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.shared_components import shared_operation_keys
from lqcdweave.contraction.planning.canonical_operations import logical_result_name
from lqcdweave.contraction.planning.logical_index import BridgeIndex
from lqcdweave.contraction.planning.state import ContractionState
from correlator.planning.test_integer_search import make_problem


def repeated_problem():
    terms = [(("A", "B", "C"), "ij,jk,kl->il"),
             (("A", "B", "D"), "ij,jk,kl->li"),
             (("A", "B"), "ij,jk->ki"),
             (("B", "B", "C"), "ij,jk,kl->il")]
    return make_problem(terms * 4, {key: (3, 3) for key in "ABCD"})


def key_classes(pairs):
    """Collapse operation keys to first-seen equivalence classes, in order.

    Classification only ever compares keys for equality, so the canonical
    dense tuple and the reference hashed name describe the same partition
    exactly when the class/scene sequence agrees.
    """
    classes = {}
    return tuple((classes.setdefault(key, len(classes)), scene) for key, scene in pairs)


def uncached_keys(problem):
    state = ContractionState.initial(problem)
    joins = {}
    for join in problem.baryon_joins:
        joins.setdefault(join.scene, []).append(join)
    bridges = BridgeIndex(problem.sparse_bridges)
    for scene, term in enumerate(state.pending):
        pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
        for pair in pairs:
            try:
                layout = state._contract_layout(scene, pair, scene_joins=joins,
                                                active_bridges=bridges.scoped(scene))
                yield logical_result_name(state, scene, pair, layout=layout), scene
            except ValueError:
                continue
        for join in joins.get(scene, ()):
            try:
                layout = state._join_layout(join, scene_joins=joins,
                                            scoped_bridges=bridges.scoped(scene))
                yield logical_result_name(state, scene, join=join, layout=layout), scene
            except ValueError:
                continue


class IntegerRowReuseTests(unittest.TestCase):
    def test_dense_contexts_are_lazy_and_match_normalized_reference(self):
        from lqcdweave.contraction.planning.optimizers import integer_space
        problem = repeated_problem()
        with patch.object(integer_space, 'component_problem',
                          side_effect=AssertionError('eager single-row problem')):
            space = IntegerSpace(problem, profile=CostProfile())
            list(space.candidates(lambda: None))
        reference = IntegerSpace(problem, profile=CostProfile(), debug=True)
        for scene in range(len(space.terms)):
            actual, expected = space._state(scene), reference._state(scene)
            self.assertEqual(actual.problem, expected.problem)
            self.assertEqual(actual.pending, expected.pending)
        context = space._dense_contexts[0]
        space.reset()
        self.assertIs(space._state(0).problem, context)
        space.release()
        self.assertFalse(space._dense_contexts)

    def test_long_compiled_binding_keeps_budget_checks_without_publishing_partial_rows(self):
        from lqcdweave.contraction.planning.optimizers.budget import BudgetExhausted
        # 92 equal axes compile to 4186 row slots, so the default 4096-slot
        # interval fires exactly twice: at ordinal 0 and at ordinal 4096. The
        # cold recipe path is stubbed because this test only covers the row
        # loop's own deadline checks and its all-or-nothing publication.
        names = tuple(f'A{i}' for i in range(92))
        problem = make_problem([(names, ','.join(['i']*92)+'->i')], {key: (2,) for key in names})
        space = IntegerSpace(problem, profile=CostProfile())
        space.registry.dense_pattern_choices(space.terms[0].pattern, lambda: None)
        before = space.fingerprint(), tuple(space.terms), frozenset(space.available)
        checks = 0
        def checkpoint():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise BudgetExhausted('deadline')
        with patch.object(space, '_dense_recipe', return_value=-1):
            with self.assertRaises(BudgetExhausted):
                space._dense_row(0, checkpoint)
        self.assertEqual(checks, 2)
        self.assertEqual((space.fingerprint(), tuple(space.terms), frozenset(space.available)), before)
        self.assertFalse(space.rows)
        self.assertFalse(space.groups)

    def test_dense_classification_never_builds_a_contraction_state(self):
        problem = repeated_problem()
        reference = list(uncached_keys(problem))
        with patch.object(ContractionState, 'initial', side_effect=AssertionError('temporary state')):
            self.assertEqual(key_classes(shared_operation_keys(problem)), key_classes(reference))

    def test_cost_prediction_is_shared_across_distinct_input_names(self):
        from lqcdweave.contraction.planning.optimizers import integer_space
        shapes, terms = {}, []
        for i in range(24):
            shapes.update({f'A{i}': (2, 3), f'B{i}': (3, 4)})
            terms.append(((f'A{i}', f'B{i}'), 'ij,jk->ik'))
        problem = make_problem(terms, shapes)
        fast = IntegerSpace(problem, profile=CostProfile())
        reference = IntegerSpace(problem, profile=CostProfile(), debug=True)
        with patch.object(integer_space, 'predict_dense', wraps=integer_space.predict_dense) as predict:
            actual = list(fast.candidates(lambda: None))
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(actual, list(reference.candidates(lambda: None)))
        self.assertEqual(fast.registry.calculations, reference.registry.calculations)
        fast.release()
        self.assertFalse(fast._dense_predictions)

    def test_initial_classification_preserves_all_keys_and_occurrence_order(self):
        from correlator.planning.test_candidate_metadata import join_fixture
        from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem
        for problem in (repeated_problem(), join_fixture().problem, sparse_spin_problem()):
            with self.subTest(terms=len(problem.terms)):
                self.assertEqual(key_classes(shared_operation_keys(problem)),
                                 key_classes(uncached_keys(problem)))

    def test_pass_local_reuse_matches_every_per_row_description_and_undo(self):
        problem = repeated_problem()
        cached, reference = (IntegerSpace(problem, profile=CostProfile()) for _ in range(2))
        rebuild = reference._rebuild
        with patch.object(reference, "_rebuild",
                          side_effect=lambda scene, checkpoint, memo: rebuild(scene, checkpoint)):

            def choices():
                actual = list(cached.search_candidates(lambda: None))
                self.assertEqual(actual, list(reference.search_candidates(lambda: None)))
                self.assertEqual(cached.registry.data, reference.registry.data)
                self.assertEqual(cached.registry.calculations, reference.registry.calculations)
                self.assertEqual(cached.rows, reference.rows)
                self.assertEqual(cached.group_ordinals, reference.group_ordinals)
                return actual

            def visit(depth):
                for operation, _ in choices()[:2]:
                    if not depth:
                        return
                    left = cached.apply(operation, lambda: None)
                    right = reference.apply(operation, lambda: None)
                    visit(depth - 1)
                    cached.undo(left)
                    reference.undo(right)
                    self.assertEqual(cached.fingerprint(), reference.fingerprint())
                    choices()

            visit(5)
        self.assertGreater(cached.row_description_reuses, 0)
        self.assertLess(cached.row_descriptions, reference.row_descriptions)
        self.assertEqual(reference.row_description_reuses, 0)

    def test_duplicate_rows_avoid_actual_description_calls(self):
        from lqcdweave.contraction.planning.optimizers import integer_space
        problem = repeated_problem()
        cached, reference = (IntegerSpace(problem, profile=CostProfile()) for _ in range(2))
        reference.debug = True
        with patch.object(integer_space, "describe", wraps=integer_space.describe) as describe:
            list(cached.candidates(lambda: None))
            fast_calls = describe.call_count
        rebuild = reference._rebuild
        with patch.object(reference, "_rebuild",
                          side_effect=lambda scene, checkpoint, memo: rebuild(scene, checkpoint)):
            with patch.object(integer_space, "describe", wraps=integer_space.describe) as describe:
                list(reference.candidates(lambda: None))
                slow_calls = describe.call_count
        self.assertGreaterEqual(slow_calls, 4 * fast_calls)
        self.assertEqual(fast_calls, 0)
        self.assertGreater(slow_calls, 0)
        self.assertGreater(cached.dense_recipe_builds, 0)
        self.assertGreater(cached.dense_recipe_hits, 0)

    def test_recipe_reuse_survives_apply_undo_without_reusing_availability(self):
        problem = repeated_problem()
        fast = IntegerSpace(problem, profile=CostProfile())
        reference = IntegerSpace(problem, profile=CostProfile(), debug=True)
        for _ in range(3):
            choices = list(fast.candidates(lambda: None))
            self.assertEqual(choices, list(reference.candidates(lambda: None)))
            operation = choices[0][0]
            left = fast.apply(operation, lambda: None)
            right = reference.apply(operation, lambda: None)
            self.assertEqual(list(fast.candidates(lambda: None)),
                             list(reference.candidates(lambda: None)))
            fast.undo(left)
            reference.undo(right)
        # Restoring a connection now bypasses even the recipe-key lookup.
        self.assertGreater(fast.dense_recipe_hits + fast.links.reused_connections,
                           fast.dense_recipe_builds)
        self.assertEqual(fast.rows, reference.rows)

    def test_compiled_patterns_match_original_with_diagonals_scalars_and_reordered_inputs(self):
        problem = make_problem([
            (("zz", "yy", "xx"), "ij,jk,kl->il"),
            (("yy", "zz", "xx"), "jk,ij,kl->li"),
            (("zz", "zz", "xx"), "ij,jk,ki->"),
            (("zz",), "ii->"), (("zz",), "ij->ji"),
            (("scalar", "zz"), ",ij->ij"),
        ], {"zz": (2, 2), "yy": (2, 2), "xx": (2, 2), "scalar": ()})
        fast = IntegerSpace(problem, profile=CostProfile())
        reference = IntegerSpace(problem, profile=CostProfile(), debug=True)

        def visit(depth):
            actual = list(fast.candidates(lambda: None))
            self.assertEqual(actual, list(reference.candidates(lambda: None)))
            self.assertEqual(fast.rows, reference.rows)
            self.assertEqual(fast.registry.calculations, reference.registry.calculations)
            if not depth:
                return
            for operation, _ in actual[:3]:
                left = fast.apply(operation, lambda: None)
                right = reference.apply(operation, lambda: None)
                visit(depth - 1)
                fast.undo(left)
                reference.undo(right)
        visit(3)
        self.assertTrue(fast.registry.dense_choices)

    def test_scoped_core_rows_share_only_equal_core_connections(self):
        from correlator.planning.test_candidate_metadata import join_fixture
        space = IntegerSpace(join_fixture().problem, profile=CostProfile())
        list(space.candidates(lambda: None))
        self.assertEqual(space.row_description_reuses, 1)
        self.assertEqual(space.rows[0][0][0], space.rows[1][0][0])
        self.assertNotEqual(space.rows[0][0][0], space.rows[2][0][0])


if __name__ == "__main__":
    unittest.main()
