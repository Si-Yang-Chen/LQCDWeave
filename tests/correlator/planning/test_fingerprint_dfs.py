"""Shared graph labels, final replay and reversible expression ownership."""
from dataclasses import fields
import hashlib
import math
import random
import unittest
from unittest.mock import patch

from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS, GraphNode
from lqcdweave.contraction.planning.optimizers.graph_session import solve_graph_storage
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.planning.api import PathSearchExhausted
from correlator.planning.test_macro_steps import fixture


class ReversibleGraph:
    def __init__(self, graph, goals=("goal",)):
        self.graph, self.goals, self.visits = graph, goals, []
        self.current = "root"
        self.audit_calls = self.apply_calls = 0

    @staticmethod
    def identity(name):
        return hashlib.blake2b(name.encode(), digest_size=32).digest()

    def reset(self): self.current = "root"
    def release(self): self.current = None
    def fingerprint(self): return self.identity(self.current)
    def is_done(self): return self.current in self.goals
    def candidates(self, checkpoint):
        self.visits.append(self.current)
        for target, cost in self.graph.get(self.current, ()):
            checkpoint()
            yield (self.current, target, cost), cost
    def apply(self, operation, checkpoint):
        checkpoint()
        source, target, cost = operation
        if source != self.current or (target, cost) not in self.graph.get(source, ()):
            raise RuntimeError("invalid operation replay")
        self.apply_calls += 1
        self.current = target
        return source
    def undo(self, token): self.current = token
    def audit(self, operations, expected):
        self.audit_calls += 1
        current, cost = "root", 0.0
        for source, target, increment in operations:
            if source != current or (target, increment) not in self.graph.get(source, ()):
                raise RuntimeError("invalid final path")
            current, cost = target, cost + increment
        if current not in self.goals or not math.isclose(cost, expected):
            raise RuntimeError("invalid final goal or cost")
        return cost


def search(graph, **options):
    engine = FingerprintDFS(graph, **options)
    return engine, engine.run()


class FingerprintGraphTests(unittest.TestCase):
    def test_backtracking_and_one_final_audit(self):
        graph = ReversibleGraph({"root": (("a", 1),), "a": (("dead", 1), ("b", 2)),
                                 "b": (("goal", 3),)})
        engine, result = search(graph, branch_width=2)
        self.assertEqual(engine.best_plan, 6)
        self.assertEqual(graph.audit_calls, 1)
        self.assertEqual(result.status, "FEASIBLE")
        self.assertEqual(graph.visits, ["root", "a", "dead", "b"])
        self.assertIsNone(graph.current)

    def test_cheaper_arrival_propagates_without_reenumeration(self):
        graph = ReversibleGraph({"root": (("a", 1), ("b", 2)), "a": (("x", 9),),
                                 "b": (("x", 1),), "x": (("goal", 1),)})
        engine, result = search(graph, root_width=2)
        self.assertEqual(engine.best_plan, 4)
        self.assertEqual(graph.visits.count("x"), 1)
        self.assertEqual([c for _, c in engine.curve], [11, 4])
        self.assertGreater(result.statistics["propagated_connections"], 0)
        x, a, b = map(graph.identity, ("x", "a", "b"))
        self.assertIn(x, engine.nodes[a].children)
        self.assertIn(x, engine.nodes[b].children)
        self.assertEqual(engine.nodes[x].parent, b)

    def test_nonbest_connection_can_become_best_again(self):
        graph = ReversibleGraph({"root": (("a", 1), ("b", 2)), "a": (("s", 9),),
            "s": (("t", 2),), "t": (("goal", 1),), "b": (("u", 1), ("v", 2)),
            "u": (("t", 5),), "v": (("s", 1),)})
        engine, _ = search(graph, root_width=2, branch_width=2)
        self.assertEqual(engine.best_plan, 8)
        t, s, u = map(graph.identity, ("t", "s", "u"))
        self.assertIn(t, engine.nodes[u].children)
        self.assertIn(t, engine.nodes[s].children)
        self.assertEqual(engine.nodes[t].parent, s)
        self.assertEqual(graph.visits.count("t"), 1)

    def test_closed_state_only_propagates_cheaper_arrivals(self):
        graph = ReversibleGraph({"root": (("a", 1), ("b", 2)), "a": (("x", 9),),
                                 "b": (("x", 1),), "x": (("goal", 1),)})
        original = graph.apply
        def apply(operation, checkpoint):
            if operation[0] == "b":
                self.assertTrue(engine.nodes[graph.identity("x")].closed)
            return original(operation, checkpoint)
        graph.apply = apply
        engine = FingerprintDFS(graph, root_width=2)
        result = engine.run()
        self.assertEqual(engine.best_plan, 4)
        self.assertEqual(graph.visits.count("x"), 1)
        self.assertEqual(result.statistics["state_replays"], 0)
        self.assertEqual(engine.frames, [])
        self.assertTrue(all(n.closed for n in engine.nodes.values()))

    def test_pruned_new_state_can_first_expand_through_a_cheaper_parent(self):
        graph = ReversibleGraph({"root": (("a", 1), ("b", 2)),
            "a": (("goal_a", 10), ("x", 20)), "x": (("goal_b", 1),),
            "b": (("x", 1),)}, ("goal_a", "goal_b"))
        engine, _ = search(graph, root_width=2, branch_width=2)
        self.assertEqual(engine.best_plan, 4)
        self.assertEqual([c for _, c in engine.curve], [11, 4])
        self.assertEqual(graph.visits.count("x"), 1)

    def test_parent_discount_activates_deferred_operation_without_reexpanding_parent(self):
        graph = ReversibleGraph({"root": (("a", 0), ("b", 1), ("c", 2)),
            "a": (("goal_a", 10),), "b": (("s", 8),), "c": (("s", 1),),
            "s": (("x", 4),), "x": (("goal_b", 1),)}, ("goal_a", "goal_b"))
        engine, result = search(graph, root_width=3)
        self.assertEqual(engine.best_plan, 8)
        self.assertEqual(graph.visits.count("s"), 1)
        self.assertEqual(graph.visits.count("x"), 1)
        self.assertEqual(result.statistics["state_replays"], 1)
        self.assertEqual(result.statistics["historical_reexpansions"], 0)
        self.assertGreater(result.statistics["cost_pruned"], 0)

    def test_search_reset_occurs_once_and_widen_is_rejected(self):
        graph = ReversibleGraph({"root": (("a", 1), ("b", 2)),
                                 "a": (("goal", 4),), "b": (("goal", 1),)})
        with patch.object(graph, "reset", wraps=graph.reset) as reset:
            engine, _ = search(graph, root_width=2)
        self.assertEqual(reset.call_count, 1)
        self.assertEqual(engine.best_plan, 3)
        with self.assertRaises(TypeError):
            FingerprintDFS(graph, widen=False)

    def test_fixed_width_never_returns_to_truncated_root_candidate(self):
        graph = ReversibleGraph({"root": (("a", 1), ("b", 2)), "a": (("goal_a", 100),),
                                 "b": (("goal_b", 0),)}, ("goal_a", "goal_b"))
        engine, result = search(graph, root_width=1, branch_width=1)
        self.assertEqual(engine.best_plan, 101)
        self.assertEqual([c for _, c in engine.curve], [101])
        self.assertNotIn("b", graph.visits)
        self.assertNotIn("widening_rounds", result.statistics)

    def test_root_branches_finish_serially_without_prefix_replays(self):
        graph = {"root": (("a0", 1), ("b", 2)), "b": (("goal_b", 10),)}
        graph.update({f"a{i}": ((f"a{i+1}", 0),) for i in range(400)})
        graph["a400"] = (("goal_a", 1),)
        engine, result = search(ReversibleGraph(graph, ("goal_a", "goal_b")), root_width=2)
        self.assertEqual([c for _, c in engine.curve], [2])
        self.assertTrue(all(t > 0 for t in result.statistics["branch_ticks"]))
        self.assertEqual(result.statistics["state_replays"], 0)
        self.assertGreater(engine.branch_ticks[0], 400)

    def test_zero_cycles_and_all_budgets(self):
        graph = {"root": (("a", 0),), "a": (("root", 0), ("goal", 1))}
        engine, result = search(ReversibleGraph(graph), branch_width=2)
        self.assertEqual(engine.best_plan, 1)
        self.assertLess(result.explored, 10)
        for options, reason in ((dict(max_states=0), "max_states"),
                                (dict(max_transitions=0), "max_transitions"),
                                (dict(deadline_seconds=0), "deadline"),
                                (dict(search_memory_limit=1), "search_memory_limit")):
            with self.subTest(reason=reason):
                engine, result = search(ReversibleGraph(graph), **options)
                self.assertTrue(result.limited)
                self.assertEqual(result.stop_reason, reason)

    def test_narrow_dead_end_is_not_a_solution(self):
        graph = ReversibleGraph({"root": (("dead", 1), ("goal", 2))})
        engine, result = search(graph, root_width=1)
        self.assertIsNone(engine.best_plan)
        self.assertEqual(result.status, "UNSOLVED")

    def test_historical_records_have_no_state_or_expression_fields(self):
        graph = ReversibleGraph({"root": (("a", 1),), "a": (("goal", 1),)})
        engine, _ = search(graph)
        self.assertFalse({"state", "expression", "actions", "delta"} & {f.name for f in fields(GraphNode)})
        self.assertTrue(all(isinstance(key, bytes) and len(key) == 32 for key in engine.nodes))
        self.assertTrue(all(not hasattr(n, "__dict__") for n in engine.nodes.values()))

    def test_deliberate_hash_collision_cannot_deliver_unchecked_goal(self):
        graph = ReversibleGraph({"root": (("a", 1),), "a": (("goal", 1),)})
        graph.fingerprint = lambda: b"root" if graph.current == "root" else b"same"
        with self.assertRaisesRegex(RuntimeError, "invalid final goal"):
            search(graph)

    def test_final_audit_runs_after_search_budget_expires(self):
        graph = ReversibleGraph({"root": (("a", 0), ("b", 1)),
                                 "a": (("goal", 10),), "b": (("goal", 0),)})
        engine, result = search(graph, root_width=2, max_states=2)
        self.assertEqual(engine.best_plan, 10)
        self.assertTrue(result.limited)
        self.assertEqual(graph.audit_calls, 1)

    def test_interrupted_cost_propagation_is_settled_before_final_replay(self):
        graph = ReversibleGraph({"root": (("a", 1), ("b", 2)), "a": (("x", 9),),
                                 "b": (("x", 1),), "x": (("y", 1),), "y": (("goal", 1),)})
        engine = FingerprintDFS(graph, root_width=2)
        original = graph.apply
        def apply(operation, checkpoint):
            token = original(operation, checkpoint)
            if operation[0] == "b":
                # Allow x's outgoing connection, then interrupt before y's.
                engine.budget.max_transitions = engine.budget.transitions + 1
            return token
        graph.apply = apply
        result = engine.run()
        self.assertTrue(result.limited)
        self.assertEqual(result.stop_reason, "max_transitions")
        self.assertEqual(engine.best_plan, 5)
        self.assertEqual(graph.audit_calls, 1)
        self.assertFalse(engine.relax_pending)

    def test_wide_graphs_match_independent_shortest_distances(self):
        rng = random.Random(701)
        for sample in range(20):
            names = ["root"] + [f"n{i}" for i in range(6)] + ["goal"]
            graph = {name: tuple((names[j], rng.randrange(0, 10))
                                for j in range(i + 1, len(names))
                                if j == i + 1 or rng.random() < .45)
                     for i, name in enumerate(names[:-1])}
            distances = {"root": 0}
            for name in names[:-1]:
                for target, increment in graph[name]:
                    distances[target] = min(distances.get(target, math.inf), distances[name] + increment)
            with self.subTest(sample=sample):
                engine, _ = search(ReversibleGraph(graph), root_width=64, branch_width=64)
                self.assertEqual(engine.best_plan, distances["goal"])


class PublicGraphDeliveryTests(unittest.TestCase):
    def test_public_delivery_and_impossible_allocation_budget(self):
        source = fixture()
        result = solve_storage(source.problem, source.blocks, source.preparations, memory_limit=10000,
                               root_width=1, branch_width=1)
        self.assertEqual(result.diagnostic["engine"], "integer_fingerprint_graph_dfs")
        self.assertTrue(result.plan.feasible)
        self.assertTrue(result.state.done)
        with self.assertRaisesRegex(PathSearchExhausted, "per-operation storage budget"):
            solve_graph_storage(source.problem, source.blocks, source.preparations,
                                memory_limit=1, root_width=1, branch_width=1)

    def test_wide_small_case_agrees_with_exact_when_storage_is_spacious(self):
        source = fixture()
        options = dict(memory_limit=10000, profile=source.profile)
        heuristic = solve_storage(source.problem, source.blocks, source.preparations,
                                  root_width=16, branch_width=16, **options)
        exact = solve_storage(source.problem, source.blocks, source.preparations,
                              optimizer="exact", **options)
        self.assertAlmostEqual(heuristic.plan.seconds, exact.plan.seconds)
        self.assertEqual(exact.status, "OPTIMAL")
        self.assertEqual(heuristic.diagnostic["proof_scope"], "none")


if __name__ == "__main__": unittest.main()
