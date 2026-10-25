"""Boundary ties widen DFS without changing computation or numeric semantics."""
import hashlib
import unittest
from types import SimpleNamespace

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    OutputNode,
    OutputRecipe,
    SourcePreparation,
    SourceSpec,
    TensorSpec,
    describe_coordinates,
)
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.results.api import TermValue, assemble_correlator
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.options import resolve_strategy_options
from lqcdweave.contraction.planning.optimizers.session import solve_storage


class FallbackGraph:
    """Small graph-only domain with no rank_candidates implementation."""

    def __init__(self, graph, goals=(), *, overlaps=None):
        self.graph = graph
        self.goals = frozenset(goals)
        self.overlaps = dict(overlaps or {})
        self.current = "root"
        self.visits = []
        self.applied = []
        self.audit_calls = 0

    @staticmethod
    def identity(name):
        return hashlib.blake2b(name.encode(), digest_size=32).digest()

    def reset(self):
        self.current = "root"

    def release(self):
        self.current = None

    def fingerprint(self):
        return self.identity(self.current)

    def is_done(self):
        return self.current in self.goals

    def candidates(self, checkpoint):
        self.visits.append(self.current)
        for target, cost in self.graph.get(self.current, ()):
            checkpoint()
            yield (self.current, target, cost), cost

    def tie_break(self, operation, checkpoint):
        checkpoint()
        return self.overlaps.get(operation[1], 0)

    def apply(self, operation, checkpoint):
        checkpoint()
        source, target, cost = operation
        if source != self.current or (target, cost) not in self.graph.get(source, ()):
            raise RuntimeError("invalid operation replay")
        self.applied.append(target)
        self.current = target
        return source

    def undo(self, token):
        self.current = token

    def audit(self, operations, expected):
        self.audit_calls += 1
        current, cost = "root", 0.0
        for source, target, increment in operations:
            if source != current or (target, increment) not in self.graph.get(source, ()):
                raise RuntimeError("invalid final path")
            current, cost = target, cost + increment
        if current not in self.goals or cost != expected:
            raise RuntimeError("invalid final goal or cost")
        return cost


def run(graph, **options):
    engine = FingerprintDFS(graph, **options)
    return engine, engine.run()


class TiedWidthTests(unittest.TestCase):
    def test_nonroot_width_one_keeps_same_score_branches_after_dead_end(self):
        graph = FallbackGraph({"root": (("branch", 0),), "branch": (("dead", 1), ("live", 1)),
                               "live": (("goal", 1),)}, goals=("goal",))
        self.assertFalse(hasattr(graph, "rank_candidates"))
        engine, result = run(graph, root_width=1, branch_width=1)
        self.assertEqual(result.status, "FEASIBLE")
        self.assertEqual(engine.best_plan, 2)
        self.assertEqual(graph.visits, ["root", "branch", "dead", "live"])
        self.assertEqual(graph.applied[:3], ["branch", "dead", "live"])
        self.assertEqual(len(engine.branch_ticks), 1)
        self.assertEqual(graph.audit_calls, 1)

    def test_secondary_order_does_not_drop_other_primary_score_ties(self):
        graph = FallbackGraph(
            {"root": (("branch", 0),), "branch": (("preferred_dead", 10), ("goal", 9),
                      ("cheaper_goal", 8), ("lower_primary", 1))},
            goals=("goal", "cheaper_goal", "lower_primary"),
            overlaps={"preferred_dead": 9, "goal": 5,
                      "cheaper_goal": 2, "lower_primary": 100},
        )
        primary = {"branch": 0, "preferred_dead": 1, "goal": 1,
                   "cheaper_goal": 1, "lower_primary": 2}
        engine, result = run(
            graph, root_width=1, branch_width=1,
            scorer=lambda space, operation, cost, checkpoint: primary[operation[1]],
        )
        self.assertEqual(result.status, "FEASIBLE")
        self.assertEqual(engine.best_plan, 8)
        self.assertEqual(graph.applied[1:4], ["preferred_dead", "goal", "cheaper_goal"])
        self.assertNotIn("lower_primary", graph.applied)
        self.assertEqual(len(engine.branch_ticks), 1)

    def test_width_two_includes_the_entire_tie_crossing_the_boundary(self):
        graph = FallbackGraph({"root": (("branch", 0),), "branch": (("top0", 10), ("top1", 9), ("top2", 8),
                                        ("top3", 7), ("lower", 1))},
                              goals=("top0", "top1", "top2", "top3", "lower"))
        primary = {name: 1 if name.startswith("top") else 2
                   for name in ("top0", "top1", "top2", "top3", "lower")}
        primary['branch'] = 0
        engine, result = run(
            graph, root_width=1, branch_width=2,
            scorer=lambda space, operation, cost, checkpoint: primary[operation[1]],
        )
        self.assertEqual(result.status, "FEASIBLE")
        self.assertEqual(graph.applied, ["branch", "top0", "top1", "top2", "top3"])
        self.assertEqual(len(engine.branch_ticks), 1)
        self.assertEqual(engine.best_plan, 7)

    def test_include_ties_false_keeps_exact_width_but_uses_secondary_order(self):
        graph = FallbackGraph({"root": (("preferred", 10), ("next", 9),
                                        ("third", 8), ("lower", 1))},
                              goals=("preferred", "next", "third", "lower"),
                              overlaps={"preferred": 3, "next": 2, "third": 1, "lower": 99})
        primary = {"preferred": 1, "next": 1, "third": 1, "lower": 2}
        engine, result = run(
            graph, root_width=2, branch_width=1, include_ties=False,
            scorer=lambda space, operation, cost, checkpoint: primary[operation[1]],
        )
        self.assertEqual(result.status, "FEASIBLE")
        self.assertEqual(graph.applied, ["preferred", "next"])
        self.assertEqual(len(engine.branch_ticks), 2)
        self.assertEqual(engine.best_plan, 9)

    def test_budget_can_interrupt_inside_a_large_boundary_tie(self):
        graph = FallbackGraph({"root": tuple((f"leaf{i}", 1) for i in range(40))})
        engine, result = run(graph, root_width=1, branch_width=1,
                             include_ties=True, max_transitions=12)
        self.assertTrue(result.limited)
        self.assertEqual(result.stop_reason, "max_transitions")
        self.assertIsNone(engine.best_plan)
        self.assertEqual(graph.applied, [])
        self.assertEqual(graph.audit_calls, 0)
        self.assertEqual(graph.visits, ["root"])

    def test_random_fallback_top_n_matches_independent_boundary_tie_oracle(self):
        rng = np.random.default_rng(0xC0FFEE)
        cases = [
            # A later lower score must replace the old boundary and discard
            # its complete tie group.
            (["tie0", "tie1", "best", "other"],
             {"tie0": 0, "tie1": 0, "best": -2, "other": 1},
             {"tie0": 3, "tie1": 0, "best": 1, "other": 4}),
        ]
        for case in range(40):
            count = int(rng.integers(5, 17))
            names = [f"case{case}-op{index}" for index in range(count)]
            scores = {name: int(rng.integers(-2, 4)) for name in names}
            overlaps = {name: int(rng.integers(0, 5)) for name in names}
            rng.shuffle(names)
            cases.append((names, scores, overlaps))

        for case_index, (names, scores, overlaps) in enumerate(cases):
            graph_edges = tuple((name, 1) for name in names)
            for width in range(1, 5):
                for include_ties in (True, False):
                    with self.subTest(case=case_index, width=width,
                                      include_ties=include_ties):
                        graph = FallbackGraph({"root": graph_edges}, overlaps=overlaps)
                        scorer = lambda space, operation, cost, checkpoint: scores[operation[1]]
                        engine = FingerprintDFS(
                            graph, root_width=width, branch_width=1,
                            scorer=scorer, include_ties=include_ties,
                        )
                        engine.start()
                        selected_ids = engine.rank_candidates(width)
                        actual = tuple(engine.operations[identity] for identity in selected_ids)

                        enumerated = [
                            (("root", name, 1), 1) for name in names
                        ]
                        ranked = sorted(
                            enumerate(enumerated),
                            key=lambda row: (scores[row[1][0][1]],
                                             -overlaps[row[1][0][1]], row[0]),
                        )
                        expected = ranked[:width]
                        if include_ties and expected and len(ranked) > width:
                            boundary_score = scores[expected[-1][1][0][1]]
                            for row in ranked[width:]:
                                if scores[row[1][0][1]] != boundary_score:
                                    break
                                expected.append(row)
                        expected_operations = tuple(row[1][0] for row in expected)
                        self.assertEqual(actual, expected_operations)
                        if case_index == 0 and width == 1 and include_ties:
                            self.assertEqual(actual, (("root", "best", 1),))

    def test_include_ties_is_strict_bool_and_exact_modes_reject_it(self):
        graph = FallbackGraph({"root": (("goal", 1),)}, goals=("goal",))
        for value in (1, 0, None, "true"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                FingerprintDFS(graph, include_ties=value)
        self.assertIs(resolve_strategy_options("anytime", include_ties=True)["include_ties"], True)
        self.assertIs(resolve_strategy_options("anytime", include_ties=False)["include_ties"], False)
        with self.assertRaises(ValueError):
            resolve_strategy_options("anytime", include_ties=1)
        for optimizer in ("exact", "dynamic_programming"):
            with self.subTest(optimizer=optimizer), self.assertRaises(ValueError):
                resolve_strategy_options(optimizer, include_ties=True)

    def test_small_complex_four_and_eight_factor_traces_match_numpy(self):
        self._check_complex_traces(include_ties=False)

    def test_small_complex_traces_with_all_ties_match_numpy(self):
        self._check_complex_traces(include_ties=True)

    def _check_complex_traces(self, *, include_ties, component_workers=1):
        equations = []
        coefficients = (0.25 + 0.4j, -0.3 + 0.2j)
        terms = []
        specs, sources = {}, {}
        for count, coefficient in zip((4, 8), coefficients, strict=True):
            labels = "abcdefgh"[:count]
            keys = tuple(f"T{count}_{index}" for index in range(count))
            inputs = tuple(labels[index] + labels[(index + 1) % count]
                           for index in range(count))
            equation = ",".join(inputs) + "->"
            equations.append(equation)
            terms.append(ContractionTerm(keys, equation))
            for key in keys:
                specs[key] = TensorSpec((2, 2), "complex128")
                sources[key] = SourceSpec(key, 0)
        problem = ContractionProblem(specs, tuple(terms), sources)
        preparations = {key: SourcePreparation(()) for key in sources}
        result = solve_storage(
            problem, {}, preparations, memory_limit=10**8, profile=CostProfile(),
            root_width=1, branch_width=1, scorer="time_per_term", include_ties=include_ties,
            max_states=200, deadline_seconds=10,
            component_workers=component_workers,
        )
        self.assertTrue(result.plan.feasible)
        self.assertTrue(result.state.done)

        rng = np.random.default_rng(20261015)
        arrays = {key: rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)
                  for key, spec in specs.items()}
        actual = replay(result.state, lambda key, source: arrays[key])
        references = []
        for value, equation, term in zip(actual, equations, terms, strict=True):
            expected = np.einsum(
                equation, *(arrays[key] for key in term.inputs), optimize=False,
            )
            np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
            references.append(expected)

        # Physical scene terms carry unit coefficients; complex physics weights
        # are applied by the output recipe, matching the public contract.
        coordinates, _ = describe_coordinates()
        term_ids = ("trace-4", "trace-8")
        recipe = OutputRecipe(
            OutputNode("diagram_sum", tuple(zip(term_ids, coefficients, strict=True))),
            term_ids, coordinates,
        )
        values = tuple(TermValue(term_id, coordinates.points[0].key, value)
                       for term_id, value in zip(term_ids, actual, strict=True))
        combined = assemble_correlator(
            SimpleNamespace(output_recipe=recipe, coordinates=coordinates), values,
        )
        expected_combined = sum((coefficient * reference
                                 for coefficient, reference
                                 in zip(coefficients, references, strict=True)), 0j)
        np.testing.assert_allclose(combined, expected_combined, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
