"""Search-time occurrence bindings must reproduce the independent audit exactly."""
from unittest.mock import patch
import unittest

from lqcdweave.contraction.planning.optimizers import selected_path
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.selected_path import materialize_dense_path
from lqcdweave.contraction.performance.api import CostProfile
from correlator.planning.test_integer_search import make_problem, solve


ACTION_FIELDS = ("tensor", "operands", "equation", "positions", "used_np", "kernel_path", "seconds")


def _route(problem):
    """Run one deterministic DFS with binding capture and return its winner."""
    space = IntegerSpace(problem, profile=CostProfile())
    engine = FingerprintDFS(space, root_width=1, branch_width=1, capture_bindings=True)
    engine.advance()
    operations, bindings = engine.selected_operations_and_bindings()
    return space, operations, bindings, engine.best_cost


class SelectedPathBindingTests(unittest.TestCase):
    def test_bindings_reproduce_independent_recompute_field_by_field(self):
        cases = (
            ([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")] * 40,
             {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)}),
            (([(("A", "A", "B"), "ij,ji,ik->jk")]) * 16,
             {"A": (3, 3), "B": (3, 2)}),
            (([(("A", "B"), "ij,jk->ki")]) * 12 + [(("A", "B"), "zx,xy->zy")] * 12,
             {"A": (3, 3), "B": (3, 2)}),
            (([(("A", "B", "C"), "ij,jk,kl->li")]) * 12 +
             [(("A", "B", "C"), "xy,yz,zw->xw")] * 12,
             {"A": (2, 2), "B": (2, 2), "C": (2, 2)}),
        )
        for expressions, shapes in cases:
            with self.subTest(expressions=expressions[:1], terms=len(expressions)):
                problem = make_problem(expressions, shapes)
                space, operations, bindings, cost = _route(problem)
                self.assertTrue(any(bindings), "capture produced no usable binding")
                reference, reference_snippets = materialize_dense_path(
                    problem, operations, cost, space.registry, recompute=True)
                actual, snippets = materialize_dense_path(
                    problem, operations, cost, space.registry, bindings=bindings)
                self.assertEqual(snippets, reference_snippets)
                self.assertEqual(len(actual.actions), len(reference.actions))
                for got, want in zip(actual.actions, reference.actions):
                    for field in ACTION_FIELDS:
                        self.assertEqual(getattr(got, field), getattr(want, field), field)
                self.assertEqual(actual, reference)
                self.assertTrue(actual.done)

    def test_bindings_reproduce_audit_of_released_session_space(self):
        # The real owner releases the search workspace before the audit, so the
        # side table must be self-contained: (scene, positions, output) only.
        problem = make_problem([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")] * 30,
                               {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)})
        space, operations, bindings, cost = _route(problem)
        registry = space.registry
        space.release()
        reference = materialize_dense_path(problem, operations, cost,
                                           registry, recompute=True)[0]
        actual = materialize_dense_path(problem, operations, cost,
                                        registry, bindings=bindings)[0]
        self.assertEqual(actual.actions, reference.actions)
        self.assertEqual(actual.results, reference.results)
        self.assertEqual(actual.seconds, reference.seconds)

    def test_missing_bindings_fall_back_to_independent_match(self):
        problem = make_problem([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")] * 24,
                               {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)})
        space, operations, bindings, cost = _route(problem)
        reference, reference_snippets = materialize_dense_path(
            problem, operations, cost, space.registry, recompute=True)
        none_state, none_snippets = materialize_dense_path(
            problem, operations, cost, space.registry, bindings=None)
        blank = [None] * len(operations)
        blank_state, blank_snippets = materialize_dense_path(
            problem, operations, cost, space.registry, bindings=blank)
        sparse = [binding if index % 2 else None for index, binding in enumerate(bindings)]
        sparse_state, sparse_snippets = materialize_dense_path(
            problem, operations, cost, space.registry, bindings=sparse)
        for state, snippets in ((none_state, none_snippets), (blank_state, blank_snippets),
                                (sparse_state, sparse_snippets)):
            self.assertEqual(state.actions, reference.actions)
            self.assertEqual(state.results, reference.results)
            self.assertEqual(snippets, reference_snippets)

    def test_complete_bindings_take_the_bound_path_and_truncation_falls_back(self):
        problem = make_problem([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")] * 12,
                               {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)})
        space, operations, bindings, cost = _route(problem)
        reference = materialize_dense_path(problem, operations, cost, space.registry,
                                           recompute=True)[0]
        with patch.object(selected_path, "_materialize_bound_path",
                          wraps=selected_path._materialize_bound_path) as bound:
            actual = materialize_dense_path(problem, operations, cost, space.registry,
                                            bindings=bindings)[0]
            self.assertEqual(bound.call_count, 1)
            self.assertEqual(actual.actions, reference.actions)
            # A binding list that cannot cover every operation must not use it.
            truncated = list(bindings[:-1])
            fallback = materialize_dense_path(problem, operations, cost, space.registry,
                                              bindings=truncated)[0]
            self.assertEqual(bound.call_count, 1)
            self.assertEqual(fallback.actions, reference.actions)
            self.assertEqual(fallback.results, reference.results)

    def test_debug_verify_reopens_skipped_axis_matching(self):
        problem = make_problem([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")] * 20,
                               {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)})
        space, operations, bindings, cost = _route(problem)
        state, _ = materialize_dense_path(problem, operations, cost, space.registry,
                                          bindings=bindings, verify=True)
        reference = materialize_dense_path(problem, operations, cost, space.registry,
                                           recompute=True)[0]
        self.assertEqual(state.actions, reference.actions)
        corrupted = list(bindings)
        for index, binding in enumerate(bindings):
            if binding:
                corrupted[index] = tuple(binding[:-1])
                break
        else:
            self.fail("expected at least one multi-scene binding")
        with self.assertRaisesRegex(RuntimeError, "disagree with independent axis matching"):
            materialize_dense_path(problem, operations, cost, space.registry,
                                   bindings=tuple(corrupted), verify=True)

    def test_recompute_and_verify_require_booleans(self):
        problem = make_problem([(("A", "B"), "ij,jk->ik")], {"A": (2, 3), "B": (3, 2)})
        _, operations, _, cost = _route(problem)
        with self.assertRaisesRegex(ValueError, "recompute and verify must be bool"):
            materialize_dense_path(problem, operations, cost,
                                   IntegerSpace(problem, profile=CostProfile()).registry,
                                   recompute=1)

    def test_compacted_rows_expand_bindings_to_every_physical_scene(self):
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il")] * 4200,
                               {"A": (2, 2), "B": (2, 2), "C": (2, 2)})
        space, operations, bindings, cost = _route(problem)
        self.assertIsNotNone(space.scene_members)
        scenes = {scene for binding in bindings if binding for scene, *_ in binding}
        self.assertEqual(scenes, set(range(len(problem.terms))))
        reference = materialize_dense_path(problem, operations, cost, space.registry,
                                           recompute=True)[0]
        actual = materialize_dense_path(problem, operations, cost, space.registry,
                                        bindings=bindings)[0]
        self.assertEqual(actual.actions, reference.actions)
        self.assertEqual(actual.results, reference.results)

    def test_default_session_uses_global_scene_bindings_and_matches_debug(self):
        problem = make_problem(
            [(("A", "B", "C"), "ij,jk,kl->il")] * 20 +
            [(("D", "E", "F"), "xy,yz,zw->xw")] * 20,
            {name: (2, 2) for name in "ABCDEF"})
        captured = {}
        original = selected_path.materialize_dense_path

        def record(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(selected_path, "materialize_dense_path", record):
            default = solve(problem, root_width=1, branch_width=1)
        diagnostic = solve(problem, debug=True, root_width=1, branch_width=1)
        self.assertEqual(default.state.actions, diagnostic.state.actions)
        self.assertEqual(default.state.results, diagnostic.state.results)
        self.assertEqual(default.plan.compute_seconds, diagnostic.plan.compute_seconds)
        self.assertIsNotNone(captured.get("bindings"))
        self.assertTrue(captured["bindings"])
        self.assertTrue(any(binding for binding in captured["bindings"]))
        for binding in captured["bindings"]:
            if binding is None:
                continue
            for scene, positions, output, operands, inputs in binding:
                self.assertLess(scene, len(problem.terms))
                self.assertTrue(positions)
                self.assertTrue(operands)
                self.assertIsInstance(output, tuple)
                self.assertEqual(len(inputs), len(positions))


if __name__ == "__main__":
    unittest.main()
