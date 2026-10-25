"""Ranking never replaces costs, ownership, algebra or the search budget."""
import math
import unittest

import numpy as np

from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.integer_scoring import IntegerScorer, SCORERS
from lqcdweave.contraction.planning.optimizers.graph_session import solve_graph_storage
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.options import resolve_strategy_options
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.execution.indices import einsum_arguments
from lqcdweave.contracts.correlator.api import SourcePreparation
from correlator.planning.test_integer_search import make_problem, solve
from correlator.planning.test_fingerprint_dfs import ReversibleGraph


class IntegerScoringTests(unittest.TestCase):
    def test_default_prefers_shared_work_and_time_override_still_works(self):
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il"),
                                (("A", "B", "D"), "ij,jk,kl->il")],
                               {"A": (20, 3), "B": (3, 4), "C": (4, 2), "D": (4, 2)})
        options = dict(root_width=1, branch_width=1)
        default = solve(problem, **options)
        direct = solve_graph_storage(problem, {},
            {key: SourcePreparation(()) for key in problem.sources},
            memory_limit=10**8, **options)
        for result in (default, direct):
            self.assertEqual(result.diagnostic["scorer"], "shared_savings")
            first = next(a for a in result.state.actions if a.kind == "contract")
            self.assertEqual(set(first.operands), {"A", "B"})
        explicit = solve(problem, scorer="incremental_time", **options)
        self.assertEqual(explicit.diagnostic["scorer"], "incremental_time")
        first = next(a for a in explicit.state.actions if a.kind == "contract")
        self.assertIn(set(first.operands), ({"B", "C"}, {"B", "D"}))

    def test_negative_ranking_does_not_change_nonnegative_path_cost(self):
        graph = ReversibleGraph({"root": (("goal_a", 1), ("goal_b", 7))}, ("goal_a", "goal_b"))
        engine = FingerprintDFS(graph, root_width=1,
            scorer=lambda space, operation, cost, checkpoint: -cost, profile_scoring=True)
        result = engine.run()
        self.assertEqual(engine.best_plan, 7)
        self.assertEqual(result.statistics["logical_best_seconds"], 7)
        self.assertEqual(result.statistics["scoring_evaluations"], 2)
        self.assertGreater(result.statistics["scoring_seconds"], 0)
        self.assertEqual(result.statistics["first_solution_compute_seconds"], 7)

    def test_first_solution_metrics_survive_later_improvements(self):
        graph = ReversibleGraph({"root": (("goal_a", 9), ("goal_b", 3))}, ("goal_a", "goal_b"))
        engine = FingerprintDFS(graph, root_width=2, profile_scoring=True,
                                scorer=lambda space, operation, cost, checkpoint: -cost)
        result = engine.run()
        self.assertEqual(engine.best_plan, 3)
        self.assertEqual(result.statistics["first_solution_compute_seconds"], 9)
        self.assertLessEqual(result.statistics["first_solution_scoring_seconds"], result.statistics["scoring_seconds"])

    def test_invalid_ranking_and_strict_mode_options(self):
        for score in (math.nan, math.inf, -math.inf, (math.nan, 0), (0, math.inf)):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "score must be finite"):
                FingerprintDFS(ReversibleGraph({"root": (("goal", 1),)}),
                    scorer=lambda *args: score).run()
        for mode in ("exact", "dynamic_programming"):
            with self.assertRaises(ValueError):
                resolve_strategy_options(mode, scorer="storage_delta")
        with self.assertRaises(ValueError):
            resolve_strategy_options(scorer="unknown")
        with self.assertRaises(ValueError):
            resolve_strategy_options(profile_scoring=1)

    def test_ordered_pair_options_are_normalized_and_reject_ambiguous_forms(self):
        pair = ("sharing_count", "element_reduction")
        for value in (pair, list(pair)):
            self.assertEqual(resolve_strategy_options(scorer=value)["scorer"], pair)
            self.assertEqual(IntegerScorer(value).name, pair)
        for value in ((), ("sharing_count",), (*pair, "incremental_time"),
                      ("sharing_count", "unknown"), (pair, "incremental_time")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_strategy_options(scorer=value)

    def test_sharing_count_precedes_reduction_and_ignores_step_time(self):
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il"),
                                (("A", "B", "D"), "ij,jk,kl->il")],
                               {"A": (20, 3), "B": (3, 4), "C": (4, 2), "D": (4, 2)})
        space = IntegerSpace(problem, profile=CostProfile())
        choices = list(space.search_candidates(lambda: None))
        scorer = IntegerScorer(("sharing_count", "element_reduction"))
        shared = next(choice for choice in choices if len(space.groups[choice[0].calculation]) == 2)
        single = next(choice for choice in choices if len(space.groups[choice[0].calculation]) == 1)
        shared_score = scorer(space, *shared, lambda: None)
        single_score = scorer(space, *single, lambda: None)
        self.assertGreater(shared_score[1], single_score[1])  # Secondary alone prefers the other operation.
        self.assertLess(shared_score, single_score)
        self.assertEqual(scorer(space, shared[0], 0, lambda: None),
                         scorer(space, shared[0], 10**30, lambda: None))
        selected = space.rank_candidates(1, scorer, lambda: None)
        self.assertEqual(selected.choices, (shared,))

    def test_equal_sharing_count_uses_reduction_as_tiebreaker(self):
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il")] * 2,
                               {"A": (20, 3), "B": (3, 4), "C": (4, 2)})
        space = IntegerSpace(problem, profile=CostProfile())
        choices = list(space.search_candidates(lambda: None))
        scorer = IntegerScorer(("sharing_count", "element_reduction"))
        scores = [scorer(space, *choice, lambda: None) for choice in choices]
        self.assertEqual({score[0] for score in scores}, {-2})
        self.assertGreater(scores[0][1], min(score[1] for score in scores))
        selected = space.rank_candidates(1, scorer, lambda: None)
        self.assertEqual(selected.choices, (choices[scores.index(min(scores))],))

    def test_full_scan_pair_heap_preserves_both_priorities_and_stable_ties(self):
        goals = ("goal_a", "goal_b", "goal_c", "goal_d")
        priorities = dict(zip(goals, ((-3, 10**30), (-2, -10**30), (-3, -5), (-3, -5))))
        for width in (1, 2, 3):
            graph = ReversibleGraph({"root": tuple(zip(goals, (9, 3, 5, 7)))}, goals)
            engine = FingerprintDFS(graph,
                scorer=lambda space, operation, cost, checkpoint: priorities[operation[1]])
            selected = engine.rank_candidates(width, root=True)
            self.assertEqual([engine.operations[index][1] for index in selected],
                             ["goal_c", "goal_d", "goal_a"][:width])

    def test_storage_counts_shared_input_once_and_keeps_external_users(self):
        problem = make_problem([(("A", "B"), "ij,jk->ik"),
                                (("A", "B"), "ij,jk->ik"),
                                (("B",), "ij->ij")], {"A": (2, 3), "B": (3, 4)})
        space = IntegerSpace(problem, profile=CostProfile())
        operation, cost = next((op, cost) for op, cost in space.candidates(lambda: None)
                               if len(space.groups[op.calculation]) == 2)
        score = IntegerScorer("storage_delta")(space, operation, cost, lambda: None)
        self.assertEqual(score, (8 - 6) * 16)  # A dies once, B remains live.

    def test_logical_core_does_not_claim_a_dense_allocation(self):
        from correlator.planning.test_candidate_metadata import join_fixture
        space = IntegerSpace(join_fixture().problem, profile=CostProfile())
        core = space.join_cores[0][0]
        elements, nbytes = IntegerScorer("storage_delta").size(space, core)
        self.assertEqual(elements, 64)
        self.assertEqual(nbytes, 0)

    def test_lookahead_counts_only_one_surviving_shared_opportunity(self):
        problem = make_problem([(("A", "B"), "i,i->"), (("A", "B"), "i,i->"),
                                (("C", "D"), "i,i->"), (("C", "D"), "i,i->")],
                               {k: (3,) for k in "ABCD"})
        space = IntegerSpace(problem, profile=CostProfile())
        choices = list(space.candidates(lambda: None))
        before = space.fingerprint()
        scorer = IntegerScorer("sharing_lookahead")
        self.assertAlmostEqual(scorer(space, *choices[0], lambda: None), -sum(cost for _, cost in choices))
        self.assertEqual(space.fingerprint(), before)

    def test_all_scorers_keep_numerical_semantics_and_audited_compute_cost(self):
        problem = make_problem([(("A", "B", "D"), "ij,jk,kl->il"),
                                (("A", "B", "C"), "ij,jk,kl->il")],
                               {"A": (2, 3), "B": (3, 4), "C": (4, 2), "D": (4, 2)})
        rng = np.random.default_rng(7)
        arrays = {k: rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)
                  for k, spec in problem.specs.items()}
        for name in (None, *SCORERS, ("sharing_count", "element_reduction"),
                     ("storage_delta", "sharing_count"), ("incremental_time", "shared_savings")):
            with self.subTest(scorer=name):
                options = {} if name is None else {"scorer": name}
                result = solve(problem, **options, profile_scoring=True, root_width=1, branch_width=1)
                self.assertEqual(result.diagnostic["scorer"], name or "shared_savings")
                self.assertGreater(result.statistics["scoring_seconds"], 0)
                self.assertEqual(result.statistics["first_solution_compute_seconds"],
                                 result.statistics["improvement_curve"][0][1])
                actual = replay(result.state, lambda key, source: arrays[key])
                for value, term in zip(actual, problem.terms):
                    operands = tuple(arrays[key] for key in term.inputs)
                    expected = np.einsum(*einsum_arguments(term.subscripts, operands), optimize=False)
                    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
                compute = sum(a.seconds for a in result.state.actions if a.kind != "load")
                self.assertAlmostEqual(result.statistics["logical_best_seconds"], compute)


if __name__ == "__main__":
    unittest.main()
