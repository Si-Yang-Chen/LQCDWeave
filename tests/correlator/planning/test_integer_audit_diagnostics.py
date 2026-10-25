"""Fixed-path construction is mandatory; expression diagnostics are opt-in."""
from dataclasses import replace
import pickle
import time
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import SourcePreparation
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.graph_session import solve_graph_storage
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.parallel_components import _pack, _search_component
from lqcdweave.contraction.planning.path_planning import PathSearchExhausted
from lqcdweave.contraction.planning.canonical_operations import describe, replay as replay_action
from lqcdweave.contraction.planning.state import ContractionState
from correlator.planning.test_integer_search import make_problem, solve


def _problem():
    return make_problem([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")],
                        {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)})


def _route(*, debug=False):
    space = IntegerSpace(_problem(), profile=CostProfile(), debug=debug)
    operations, cost = [], 0.0
    while not space.is_done():
        operation, seconds = next(space.search_candidates(lambda: None))
        space.apply(operation, lambda: None)
        operations.append(operation)
        cost += seconds
    return space, tuple(operations), cost


class IntegerAuditDiagnosticTests(unittest.TestCase):
    def test_selected_occurrences_share_only_exact_immutable_axis_layouts(self):
        from lqcdweave.contraction.planning.optimizers import selected_path
        problem = make_problem([
            (('A', 'B', 'C'), 'ij,jk,kl->il'),
            (('A', 'B', 'C'), 'xy,yz,zw->wx'),
        ] * 40, {name: (2, 2) for name in 'ABC'})
        space = IntegerSpace(problem, profile=CostProfile())
        operations, cost = [], 0.0
        while not space.is_done():
            op, seconds = next(space.search_candidates(lambda: None))
            space.apply(op, lambda: None)
            operations.append(op)
            cost += seconds
        reference = IntegerSpace(problem, profile=CostProfile(), registry=space.registry, debug=True)
        expected = reference.audit(operations, cost)
        with (patch.object(selected_path, 'AxisPattern', wraps=selected_path.AxisPattern) as patterns,
              patch.object(selected_path, '_occurrence', wraps=selected_path._occurrence) as matches):
            actual = space.audit(operations, cost)
        self.assertEqual(actual, expected)
        self.assertLess(patterns.call_count, len(actual.actions) // 10)
        self.assertLess(matches.call_count, len(actual.actions) // 10)
        layouts = {}
        for action in actual.actions:
            key = action.equation.inputs, action.equation.output
            if key in layouts:
                self.assertIs(action.equation, layouts[key])
            else:
                layouts[key] = action.equation

    def test_match_memo_ignores_only_unrelated_names_and_preserves_negative_matches(self):
        from lqcdweave.contraction.planning.optimizers import selected_path
        from lqcdweave.contracts.correlator.api import Action, AxisPattern, Factor, PendingTerm
        from lqcdweave.contraction.planning.canonical_operations import axis_key
        recipe = Action('contract', 'AB', ('A', 'B'), AxisPattern(((0, 1), (1, 2)), (0, 2)))
        pending = [
            PendingTerm((Factor('A', (0, 1)), Factor('B', (1, 2)), Factor(name, (2, 3))), (0, 3))
            for name in ('C', 'D', 'C')]
        pending += [PendingTerm((Factor('A', (0, 1)), Factor('B', (2, 3)), Factor('C', (1, 2))), (0, 3))] * 2
        pending += [PendingTerm((Factor('B', (1, 2)), Factor('A', (0, 1)), Factor('C', (2, 3))), (3, 0))]
        key = axis_key(recipe.operands, recipe.equation.inputs, recipe.equation.output)
        expected = [(scene, match) for scene, term in enumerate(pending)
                    if (match := selected_path._occurrence(term, recipe, key)) is not None]
        with patch.object(selected_path, '_occurrence', wraps=selected_path._occurrence) as calls:
            actual = selected_path._match_rows(pending, range(len(pending)), recipe, key)
        self.assertEqual(actual, expected)
        self.assertEqual(calls.call_count, 3)

    def test_selected_dense_path_skips_candidate_enumeration_and_repricing(self):
        space, operations, cost = _route()
        reference_space = IntegerSpace(space.problem, profile=CostProfile(),
                                       registry=space.registry, debug=True)
        reference = reference_space.audit(operations, cost)
        with (patch.object(space, "_index", side_effect=AssertionError("enumerated discarded candidates")),
              patch("lqcdweave.contraction.planning.optimizers.integer_space.describe",
                    side_effect=AssertionError("repeated recipe prediction")),
              patch("lqcdweave.contraction.planning.optimizers.integer_space.replay",
                    side_effect=AssertionError("independent diagnostic replay"))):
            actual = space.audit(operations, cost)
        self.assertEqual(actual, reference)
        self.assertEqual(space.snippets, reference_space.snippets)

    def test_selected_recipes_match_full_replay_for_repeated_axes_and_layouts(self):
        cases = (
            [(("A", "A", "B"), "ij,ji,ik->jk")],
            [(("A", "A"), "ij,jk->ik"), (("A", "A"), "ji,kj->ki")],
            [(("A",), "ii->"), (("A",), "ij->ji")],
            [(("A", "B"), "ij,jk->ki"), (("A", "B"), "zx,xy->zy"),
             (("B", "A"), "jk,ij->ik")],
            [(("A", "B", "C"), "ij,jk,kl->li"),
             (("A", "B", "C"), "xy,yz,zw->xw")],
        )
        for expressions in cases:
            with self.subTest(expressions=expressions):
                problem = make_problem(expressions, {name: (3, 3) for name in "ABC"})
                space = IntegerSpace(problem, profile=CostProfile())
                operations, cost = [], 0.0
                while not space.is_done():
                    operation, seconds = next(space.search_candidates(lambda: None))
                    space.apply(operation, lambda: None)
                    operations.append(operation)
                    cost += seconds
                reference_space = IntegerSpace(problem, profile=CostProfile(),
                                               registry=space.registry, debug=True)
                reference = reference_space.audit(operations, cost)
                actual = space.audit(operations, cost)
                self.assertEqual(actual, reference)
                self.assertEqual(space.snippets, reference_space.snippets)
                rng = np.random.default_rng(139)
                arrays = {name: rng.normal(size=spec.shape) + 1j*rng.normal(size=spec.shape)
                          for name, spec in problem.specs.items()}
                # The logical builder marks sources resident; the independent
                # numerical replay receives those same arrays explicitly.
                values = replay(actual, lambda key, source: arrays[key], shared_arrays=arrays)
                for value, (names, equation) in zip(values, expressions):
                    np.testing.assert_allclose(value,
                        np.einsum(equation, *(arrays[name] for name in names), optimize=False),
                        rtol=1e-12, atol=1e-12)

    def test_buffered_history_matches_original_per_occurrence_replay(self):
        problem = make_problem([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")] * 130,
                               {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)})
        space = IntegerSpace(problem, profile=CostProfile())
        operations, cost = [], 0.0
        while not space.is_done():
            operation, seconds = next(space.search_candidates(lambda: None))
            space.apply(operation, lambda: None)
            operations.append(operation)
            cost += seconds

        # Retain the original history-growing replay as an independent reference.
        reference_space = IntegerSpace(problem, profile=CostProfile(), registry=space.registry)
        reference = ContractionState.initial(problem)
        reference = replace(reference, loaded=frozenset(reference.catalog))
        snippets = []
        for operation in operations:
            begin = len(reference.actions)
            reference_space._index(lambda: None)
            occurrences = reference_space.groups[operation.calculation].values()
            for occurrence in sorted(occurrences, key=lambda o: o.scene):
                action, *_ = describe(reference, occurrence.scene, occurrence.positions,
                                      profile=CostProfile(),
                                      output_order=tuple(reference_space.symbols[occurrence.scene][a]
                                                         for a in occurrence.output))
                reference = replay_action(reference, action, CostProfile())
            reference_space.apply(operation, lambda: None)
            snippets.append((begin, len(reference.actions)))
        actual = space.audit(tuple(operations), cost)
        for attribute in ("actions", "results", "pending", "seconds", "peak_bytes",
                          "peak_host_bytes", "peak_device_bytes", "loaded", "catalog"):
            self.assertEqual(getattr(actual, attribute), getattr(reference, attribute), attribute)
        self.assertEqual(space.snippets, snippets)
        self.assertTrue(actual.done)
        self.assertIs(type(actual.pending), tuple)
        self.assertIs(type(actual.results), tuple)

    def test_indexed_replay_preserves_old_snapshots_and_tuple_reference(self):
        for backend in ("numpy", "cupy"):
            with self.subTest(backend=backend):
                problem = make_problem([(("A", "B", "C", "D"), "ij,jk,kl,lm->im")] * 130,
                                       {"A": (2, 3), "B": (3, 4), "C": (4, 3), "D": (3, 2)},
                                       backend=backend)
                reference = ContractionState.initial(problem)
                reference = replace(reference, loaded=frozenset(reference.catalog))
                actual = reference.with_indexed_rows()
                snapshots = []
                for scene in (0, 63, 64, 127, 128, 129):
                    for _ in range(3):
                        before = actual
                        snapshots.append((before, tuple(before.pending), tuple(before.results), before.search_key))
                        action, *_ = describe(reference, scene, (0, 1), profile=CostProfile())
                        reference = replay_action(reference, action, CostProfile())
                        actual = replay_action(actual, action, CostProfile())
                        for attribute in ("pending", "results", "actions", "catalog", "loaded",
                                          "seconds", "peak_bytes", "peak_host_bytes", "peak_device_bytes",
                                          "resident_bytes", "resident_host_bytes", "search_key"):
                            self.assertEqual(getattr(actual, attribute), getattr(reference, attribute), attribute)
                        # Every old state's rows and cached search key stay valid.
                        for snapshot, pending, results, key in snapshots:
                            self.assertEqual(snapshot.pending, pending)
                            self.assertEqual(snapshot.results, results)
                            self.assertEqual(snapshot.search_key, key)
                published = actual.with_tuple_rows()
                self.assertEqual(published, reference)
                self.assertIs(type(published.pending), tuple)
                self.assertIs(type(published.results), tuple)

    def test_indexed_replay_keeps_action_and_memory_validation(self):
        problem = make_problem([(("A", "B"), "ij,jk->ik")] * 65,
                               {"A": (2, 3), "B": (3, 4)})
        state = ContractionState.initial(problem)
        state = replace(state, loaded=frozenset(state.catalog)).with_indexed_rows()
        action, *_ = describe(state, 64, (0, 1), profile=CostProfile())
        with self.assertRaisesRegex(ValueError, "independently computed cost"):
            replay_action(state, replace(action, seconds=action.seconds + 1), CostProfile())
        with self.assertRaisesRegex(MemoryError, "memory cap"):
            replay_action(state, action, CostProfile(), memory_limit=1)
        self.assertIsNone(state.results[64])
        self.assertIsNotNone(state.pending[64])

    def test_default_path_skips_diagnostic_decodes_and_collision_bookkeeping(self):
        original = IntegerSpace._decode
        raw_calls = []

        def decode(space, scene, *, raw=False):
            if raw:
                raw_calls.append(scene)
            return original(space, scene, raw=raw)

        with (patch.object(IntegerSpace, "_decode", decode),
              patch("lqcdweave.contraction.planning.optimizers.integer_space.computation_key",
                    side_effect=AssertionError("diagnostic collision reconstruction"))):
            result = solve(_problem(), root_width=1, branch_width=1)
        self.assertEqual(raw_calls, [])
        self.assertTrue(result.state.done)
        self.assertEqual(result.statistics["audited_logical_steps"], 3)
        self.assertEqual(result.statistics["storage_schedule"]["nodes"], 3)
        self.assertTrue(result.statistics["storage_schedule"]["completed"])

    def test_debug_performs_step_and_final_scene_diagnostics(self):
        original = IntegerSpace._decode
        decoded = []

        def decode(space, scene, *, raw=False):
            term = original(space, scene, raw=raw)
            if raw:
                decoded.append(term)
            return term

        with patch.object(IntegerSpace, "_decode", decode):
            result = solve(_problem(), debug=True, root_width=1, branch_width=1)
        self.assertEqual(len(decoded), result.statistics["audited_logical_steps"] + 1)
        self.assertEqual(sum(term is not None for term in decoded), 2)

    def test_debug_detects_intermediate_expression_inconsistency(self):
        original = IntegerSpace._decode

        def inconsistent(space, scene, *, raw=False):
            term = original(space, scene, raw=raw)
            return replace(term, output=tuple(reversed(term.output))) if raw and term else term

        with patch.object(IntegerSpace, "_decode", inconsistent):
            with self.assertRaisesRegex(RuntimeError, "changed its full expression"):
                solve(_problem(), debug=True, root_width=1, branch_width=1)

    def test_debug_final_scene_pass_still_detects_inconsistency(self):
        space, operations, cost = _route(debug=True)
        original = space._decode

        def inconsistent(scene, *, raw=False):
            if raw and space.audited_steps == len(operations):
                return object()
            return original(scene, raw=raw)

        with patch.object(space, "_decode", inconsistent):
            with self.assertRaisesRegex(RuntimeError, "changed its final expression"):
                space.audit(operations, cost)

    def test_default_and_debug_keep_actions_snippets_and_independent_values(self):
        problem = _problem()
        default = solve(problem, root_width=1, branch_width=1)
        diagnostic = solve(problem, debug=True, root_width=1, branch_width=1)
        reference = solve(problem, optimizer="exact")
        self.assertEqual(default.state.actions, diagnostic.state.actions)
        self.assertEqual(default.state.results, diagnostic.state.results)
        self.assertEqual(default.plan.peak_host_bytes, diagnostic.plan.peak_host_bytes)
        rng = np.random.default_rng(12012)
        arrays = {key: rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)
                  for key, spec in problem.specs.items()}
        # A direct four-input contraction does not use the selected binary IR.
        expected = np.einsum("ij,jk,kl,lm->im", *(arrays[key] for key in "ABCD"),
                             optimize=False)
        for result in (default, diagnostic, reference):
            actual = replay(result.state, lambda key, source: arrays[key])[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        ordinary_space, operations, cost = _route()
        diagnostic_space = IntegerSpace(problem, profile=CostProfile(),
                                        registry=ordinary_space.registry, debug=True)
        ordinary = ordinary_space.audit(operations, cost)
        diagnostic = diagnostic_space.audit(operations, cost)
        self.assertEqual(ordinary.actions, diagnostic.actions)
        self.assertEqual(ordinary_space.snippets, diagnostic_space.snippets)
        self.assertEqual(ordinary_space.snippets[-1][1], len(ordinary.actions))

    def test_parallel_worker_propagates_debug_without_changing_audit_protocol(self):
        original = IntegerSpace._decode
        counts, reports = [], []
        for debug in (False, True):
            raw_calls = []

            def decode(space, scene, *, raw=False):
                if raw:
                    raw_calls.append(scene)
                return original(space, scene, raw=raw)

            options = dict(root_width=1, branch_width=1, debug=debug,
                           scorer="incremental_time", profile_scoring=False,
                           budget=dict(max_states=1000, max_transitions=None,
                                       search_memory_limit=None))
            with patch.object(IntegerSpace, "_decode", decode):
                report = pickle.loads(_search_component(_pack((
                    _problem(), CostProfile(), options, time.monotonic() + 10,
                ))))
            self.assertEqual(report["status"], "FEASIBLE")
            counts.append(len(raw_calls))
            reports.append(report)
        self.assertEqual(counts[0], 0)
        self.assertGreater(counts[1], 0)
        for key in ("actions", "snippets", "workspaces"):
            self.assertEqual(reports[0][key], reports[1][key])

    def test_generic_dfs_keeps_two_argument_audit_and_space_debug(self):
        space = IntegerSpace(_problem(), profile=CostProfile(), debug=True)
        original = space.audit
        calls = []

        def audit(operations, cost):
            calls.append(len(operations))
            return original(operations, cost)

        with patch.object(space, "audit", audit):
            engine = FingerprintDFS(space, root_width=1, branch_width=1, debug=True)
            engine.run()
        self.assertEqual(calls, [3])
        self.assertTrue(engine.best_plan.done)

    def test_default_completion_and_cost_checks_remain_mandatory(self):
        space, operations, cost = _route()
        for route, expected in ((operations[:-1], cost), (operations, cost + 1)):
            with self.subTest(steps=len(route), expected=expected):
                with self.assertRaisesRegex(RuntimeError, "completion or cost audit"):
                    space.audit(route, expected)

    def test_resource_limits_and_invalid_external_dimensions_still_fail(self):
        problem = _problem()
        preparations = {key: SourcePreparation(()) for key in problem.sources}
        for debug in (False, True):
            with self.subTest(debug=debug):
                with self.assertRaisesRegex(PathSearchExhausted, "storage budget"):
                    solve_graph_storage(problem, {}, preparations, memory_limit=1,
                                        debug=debug, root_width=1, branch_width=1)
                with self.assertRaisesRegex(ValueError, "incompatible dimensions"):
                    invalid = make_problem([(("A", "B"), "ij,jk->ik")],
                                           {"A": (2, 3), "B": (4, 5)})
                    solve(invalid, debug=debug)


if __name__ == "__main__":
    unittest.main()
