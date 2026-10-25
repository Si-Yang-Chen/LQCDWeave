"""Independent verification replays: fast/reference agreement and tamper detection."""
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourcePreparation, SourceSpec, StorageBlock, TensorSpec)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.graph_session import solve_graph_storage
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerRegistry, IntegerSpace
from lqcdweave.contraction.planning.optimizers.selected_path import materialize_dense_path
from lqcdweave.contraction.planning.optimizers.verification import (
    VerificationError, capture_index_snapshot, compare_actions, compare_index_tables,
    rebuild_index_tables, reference_route, verify_route)
from fixtures.numeric_verification import verify_numeric_replay
from correlator.planning.test_candidate_metadata import join_fixture
from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem


def make_problem(terms, shapes, **options):
    return ContractionProblem({key: TensorSpec(shape, "complex128") for key, shape in shapes.items()},
        tuple(ContractionTerm(inputs, equation) for inputs, equation in terms),
        {key: SourceSpec(key, 0) for key in shapes}, **options)


def dense_problem():
    return make_problem([(("A", "B", "C"), "ij,jk,kl->il"), (("A", "B"), "ij,jk->ki")],
                        {key: (2, 2) for key in "ABC"})


def varied_problem():
    return make_problem([
        (("A", "B", "C"), "ij,jk,kl->il"),
        (("A", "B", "C"), "jk,ij,kl->li"),
        (("A", "B"), "ij,jk->ki"),
        (("B", "C", "D"), "ij,jk,kl->il"),
    ], {key: (2, 2) for key in "ABCD"})


def greedy_route(problem, profile, registry, debug=True):
    space = IntegerSpace(problem, profile=profile, registry=registry, debug=debug)
    route = []
    for _ in range(200):
        space._index(lambda: None)
        choices = list(space.candidates(lambda: None))
        if not choices:
            break
        operation = choices[0][0]
        route.append(operation)
        space.apply(operation, lambda: None)
        if space.is_done():
            break
    assert space.is_done(), "greedy route did not complete"
    return space, route


def solve(problem, **options):
    blocks = {key: StorageBlock(key, "all", 0) for key in problem.sources}
    preparations = {key: SourcePreparation(()) for key in problem.sources}
    options.setdefault("component_workers", 1)
    return solve_graph_storage(problem, blocks, preparations, memory_limit=10 ** 8, **options)


class ReferenceRouteTests(unittest.TestCase):
    def test_dense_fast_path_reference_replay_agrees(self):
        result = solve(dense_problem(), verify=True, verify_index_every=1)
        report = result.statistics["verification"]
        self.assertEqual(report["kind"], "route")
        self.assertEqual(report["index_samples"], report["steps"])
        self.assertEqual(report["reference_actions"], report["fast_actions"])
        self.assertEqual(report["registry_growth"], 0)
        self.assertTrue(report["limitations"])

    def test_debug_describe_path_reference_replay_agrees(self):
        result = solve(dense_problem(), debug=True, verify=True, verify_index_every=1)
        report = result.statistics["verification"]
        self.assertEqual(report["reference_actions"], report["fast_actions"])

    def test_verify_switch_preserves_the_delivered_result(self):
        problem = dense_problem()
        plain = solve(problem)
        verified = solve(problem, verify=True, verify_index_every=1)
        self.assertEqual(plain.plan.state.actions, verified.plan.state.actions)
        self.assertEqual(plain.plan.seconds, verified.plan.seconds)
        self.assertEqual(plain.plan.events, verified.plan.events)
        self.assertEqual(plain.statistics["logical_best_seconds"],
                         verified.statistics["logical_best_seconds"])

    def test_verify_without_index_sampling_still_replays_the_route(self):
        result = solve(dense_problem(), verify=True)
        report = result.statistics["verification"]
        self.assertGreater(report["steps"], 0)
        self.assertEqual(report["index_samples"], 0)

    def test_shared_node_with_scene_specific_operand_order_is_verified(self):
        # Regression: one canonical node is bound in two scenes with the
        # operands in opposite orders.  The reference must locate both
        # occurrences from the canonical name instead of the recipe's order.
        problem = make_problem([
            (("D", "E", "F"), "ij,jk,kl->il"),
            (("A", "B", "C"), "ij,jk,ki->"),
            (("C", "C", "A"), "ij,jk,kl->il"),
        ], {key: (2, 2) for key in "ABCDEF"})
        plain = solve(problem)
        verified = solve(problem, verify=True, verify_index_every=1)
        self.assertEqual(plain.plan.state.actions, verified.plan.state.actions)
        report = verified.statistics["verification"]
        self.assertEqual(report["reference_actions"], report["fast_actions"])
        self.assertGreater(report["steps"], 3)

    def test_repeated_factor_occurrences_keep_one_per_scene(self):
        # A term may bind the same computation at several factor positions.
        # The index keeps one occurrence per scene, so the reference must too.
        problem = make_problem([(("A", "A", "B"), "ij,ij,jk->ik"),
                                (("A", "B", "A"), "ij,jk,ki->i")],
                               {key: (2, 2) for key in "AB"})
        plain = solve(problem)
        verified = solve(problem, verify=True, verify_index_every=1)
        self.assertEqual(plain.plan.state.actions, verified.plan.state.actions)
        self.assertEqual(verified.statistics["verification"]["reference_actions"],
                         verified.statistics["verification"]["fast_actions"])

    def test_compacted_weighted_rows_are_verified(self):
        from lqcdweave.contraction.planning.optimizers import compact_rows
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il"),
                                (("A", "B"), "ij,jk->ki")] * 3 +
                               [(("A", "B", "C"), "ij,jk,kl->li")],
                               {key: (2, 2) for key in "ABC"})
        with patch.object(compact_rows, "COMPACT_DENSE_MIN_TERMS", 2):
            plain = solve(problem)
            verified = solve(problem, verify=True, verify_index_every=1)
        self.assertEqual(plain.plan.state.actions, verified.plan.state.actions)
        report = verified.statistics["verification"]
        self.assertEqual(report["reference_actions"], report["fast_actions"])
        self.assertEqual(report["registry_growth"], 0)

    def test_join_and_sparse_paths_reference_replay_agrees(self):
        for problem in (join_fixture().problem, sparse_spin_problem()):
            with self.subTest(terms=len(problem.terms)):
                result = solve(problem, verify=True)
                report = result.statistics["verification"]
                self.assertEqual(report["reference_actions"], report["fast_actions"])

    def test_reference_matches_materialized_dense_path_action_table(self):
        problem = dense_problem()
        profile = CostProfile()
        registry = IntegerRegistry(problem)
        _, route = greedy_route(problem, profile, registry)
        expected = sum(operation.seconds for operation in route)
        route_result = reference_route(problem, profile, registry, route)
        self.assertTrue(route_result.state.done)
        self.assertAlmostEqual(route_result.seconds, expected, places=14)
        fast, _ = materialize_dense_path(problem, route, expected, registry)
        self.assertTrue(compare_actions(route_result.actions, fast.actions))

    def test_sampled_full_rebuild_agrees_with_the_delivered_table(self):
        result = solve(dense_problem(), verify=True, verify_index_every=1, verify_full_every=2)
        report = result.statistics["verification"]
        self.assertEqual(report["steps"], report["index_samples"])
        self.assertEqual(report["reference_actions"], report["fast_actions"])

    def test_parallel_path_refuses_silent_skipping(self):
        with self.assertRaisesRegex(ValueError, "component_workers=1"):
            solve(dense_problem(), verify=True, component_workers=2)

    def test_verify_arguments_are_validated(self):
        problem = dense_problem()
        with self.assertRaises(ValueError):
            solve(problem, verify=1)
        with self.assertRaises(ValueError):
            solve(problem, verify_index_every=-1)
        with self.assertRaises(ValueError):
            solve(problem, verify_full_every=0)


class TamperDetectionTests(unittest.TestCase):
    def setUp(self):
        self.problem = dense_problem()
        self.profile = CostProfile()
        self.registry = IntegerRegistry(self.problem)
        _, self.route = greedy_route(self.problem, self.profile, self.registry)
        self.expected = sum(operation.seconds for operation in self.route)

    def verify(self, route):
        return verify_route(self.problem, self.profile, self.registry, route, self.expected)

    def test_intact_route_verifies(self):
        report = self.verify(self.route)
        self.assertEqual(report.steps, len(self.route))
        self.assertGreater(report.reference_actions, 0)

    def test_wrong_result_binding_is_detected(self):
        wrong = next(identity for identity in range(len(self.registry.data))
                     if identity != self.route[0].result)
        bad = list(self.route)
        bad[0] = replace(bad[0], result=wrong)
        with self.assertRaises(VerificationError):
            self.verify(bad)

    def test_wrong_reused_flag_is_detected(self):
        index = next(position for position, operation in enumerate(self.route) if not operation.reused)
        bad = list(self.route)
        bad[index] = replace(bad[index], reused=True, seconds=0.0)
        with self.assertRaises(VerificationError):
            self.verify(bad)

    def test_wrong_step_cost_is_detected(self):
        index = next(position for position, operation in enumerate(self.route) if not operation.reused)
        bad = list(self.route)
        bad[index] = replace(bad[index], seconds=bad[index].seconds * 1.5)
        with self.assertRaises(VerificationError):
            self.verify(bad)

    def test_duplicated_step_is_detected(self):
        bad = [self.route[0]] + list(self.route)
        with self.assertRaises((VerificationError, ValueError)):
            self.verify(bad)

    def test_action_field_mismatch_is_detected(self):
        route_result = reference_route(self.problem, self.profile, self.registry, self.route)
        first = route_result.actions[0]
        tampered = (replace(first, seconds=first.seconds + 1.0),) + route_result.actions[1:]
        with self.assertRaises(VerificationError):
            compare_actions(route_result.actions, tampered)


class IndexSelfCheckTests(unittest.TestCase):
    def test_snapshot_matches_from_scratch_rebuild_across_apply_and_undo(self):
        generator = random.Random(11)
        for debug in (False, True):
            with self.subTest(debug=debug):
                problem = varied_problem()
                profile = CostProfile()
                registry = IntegerRegistry(problem)
                space = IntegerSpace(problem, profile=profile, registry=registry, debug=debug)
                stack = []
                for step in range(12):
                    space._index(lambda: None)
                    choices = list(space.candidates(lambda: None))
                    snapshot = capture_index_snapshot(space, step)
                    tables = rebuild_index_tables(problem, profile, registry, snapshot, debug=debug)
                    compare_index_tables(snapshot, tables)
                    if choices and (not stack or generator.random() < 0.6):
                        stack.append(space.apply(generator.choice(choices)[0], lambda: None))
                    elif stack:
                        space.undo(stack.pop())
                    else:
                        break

    def test_initial_single_pass_index_matches_strict_ordinals(self):
        problem = varied_problem()
        profile = CostProfile()
        registry = IntegerRegistry(problem)
        space = IntegerSpace(problem, profile=profile, registry=registry)
        snapshot = capture_index_snapshot(space, 0)
        tables = rebuild_index_tables(problem, profile, registry, snapshot)
        self.assertTrue(compare_index_tables(snapshot, tables, strict_ordinals=True))

    def test_injected_index_drift_is_detected(self):
        problem = varied_problem()
        profile = CostProfile()
        registry = IntegerRegistry(problem)
        space = IntegerSpace(problem, profile=profile, registry=registry)
        snapshot = capture_index_snapshot(space, 0)
        tables = rebuild_index_tables(problem, profile, registry, snapshot)
        with self.assertRaises(VerificationError):
            compare_index_tables(replace(snapshot, groups={}), tables)
        with self.assertRaises(VerificationError):
            compare_index_tables(replace(snapshot, group_ordinals={}), tables)


class NumericReplayTests(unittest.TestCase):
    def test_numeric_replay_matches_oracle_and_reference_state(self):
        shapes = {"A": (2, 3), "B": (3, 4), "C": (4, 2)}
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il")], shapes)
        profile = CostProfile()
        registry = IntegerRegistry(problem)
        _, route = greedy_route(problem, profile, registry)
        expected = sum(operation.seconds for operation in route)
        reference = reference_route(problem, profile, registry, route)
        fast, _ = materialize_dense_path(problem, route, expected, registry)
        generator = np.random.default_rng(3)
        arrays = {key: generator.normal(size=shape) + 1j * generator.normal(size=shape)
                  for key, shape in shapes.items()}
        loader = lambda key, source: arrays[key]

        def oracle():
            return np.einsum("ij,jk,kl->il", arrays["A"], arrays["B"], arrays["C"])

        values = verify_numeric_replay(reference.state, loader, sources=arrays,
                                       reference_state=fast, oracle=oracle)
        self.assertEqual(len(values), 1)
        with self.assertRaises(VerificationError):
            verify_numeric_replay(reference.state, loader, sources=arrays,
                                  oracle=lambda: oracle() + 1.0)


if __name__ == "__main__":
    unittest.main()
