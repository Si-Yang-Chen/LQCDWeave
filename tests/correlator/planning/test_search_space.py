"""Exact engines must not encode a cost schema or a list of atom kinds."""
from dataclasses import dataclass, replace
import math
import unittest

from lqcdweave.contraction.planning.api import solve_layered_search, solve_shortest_path


@dataclass(frozen=True)
class _Node:
    name: str
    seconds: float = 0.0
    peak: int = 0
    operations: tuple = ()


class _Space:
    def __init__(self, edges, pricing, *, cap=10, layers=None):
        self.edges, self.pricing, self.cap = edges, pricing, cap
        self.layers = layers or {"start": 2, "ready": 1, "done": 0}

    def initial_state(self):
        return _Node("start")

    def key(self, state):
        return state.name

    def cost(self, state):
        return state.seconds

    def layer(self, state):
        return self.layers[state.name]

    def is_feasible(self, state):
        return state.peak <= self.cap

    def finish(self, state):
        return state if state.name == "done" else None

    def successors(self, state):
        for target, operation, bytes_ in self.edges.get(state.name, ()):
            yield _Node(target, state.seconds + self.pricing(operation), max(state.peak, bytes_),
                        state.operations + (operation,))


class SearchSpaceTests(unittest.TestCase):
    def test_new_atom_and_restructured_cost_model_change_the_optimum(self):
        edges = {"start": [("ready", "decode", 4)], "ready": [("done", "contract", 8)]}
        prices = {"decode": 2.0, "contract": 5.0, "fused-decode-contract": 3.0}

        class RestructuredProfile:
            # Deliberately no CostProfile fields or load/contract methods.
            def __init__(self):
                self.kernels = {k: {"startup": v / 2, "execution": v / 2} for k, v in prices.items()}

            def __call__(self, atom):
                return sum(self.kernels[atom].values())

        for solver in (solve_shortest_path, solve_layered_search):
            for pricing in (prices.__getitem__, RestructuredProfile()):
                with self.subTest(solver=solver.__name__, pricing=type(pricing).__name__):
                    old = solver(_Space(edges, pricing))
                    self.assertEqual(old.state.seconds, 7)
                    extended = dict(edges, start=edges["start"] + [("done", "fused-decode-contract", 9)])
                    new = solver(_Space(extended, pricing))
                    self.assertEqual(new.status, "OPTIMAL")
                    self.assertEqual(new.state.operations, ("fused-decode-contract",))
                    self.assertEqual(new.state.seconds, 3)
                    constrained = solver(_Space(extended, pricing, cap=8))
                    self.assertEqual(constrained.state.seconds, 7)

    def test_zero_cost_cycles_and_infeasible_cheaper_history(self):
        edges = {"start": [("ready", "too-large", 11), ("ready", "small", 4)],
                 "ready": [("ready", "cycle", 4), ("done", "finish", 5)]}
        prices = {"too-large": 0, "small": 2, "cycle": 0, "finish": 1}
        for solver in (solve_shortest_path, solve_layered_search):
            result = solver(_Space(edges, prices.__getitem__))
            self.assertEqual(result.status, "OPTIMAL")
            self.assertEqual(result.state.operations, ("small", "finish"))
            self.assertEqual(result.state.seconds, 3)

    def test_layer_increasing_atom_uses_dijkstra_and_is_rejected_by_dp(self):
        space = _Space({"start": [("ready", "expand", 2)], "ready": [("done", "finish", 2)]},
                       lambda _: 1, layers={"start": 1, "ready": 2, "done": 0})
        self.assertEqual(solve_shortest_path(space).state.seconds, 2)
        with self.assertRaisesRegex(ValueError, "increased"):
            solve_layered_search(space)

    def test_dp_requires_zero_layer_goals(self):
        space = _Space({"start": [("done", "finish", 1)]}, lambda _: 1,
                       layers={"start": 2, "done": 1})
        with self.assertRaisesRegex(ValueError, "layer zero"):
            solve_layered_search(space)

    def test_invalid_costs_and_hidden_finalization_are_rejected(self):
        for solver in (solve_shortest_path, solve_layered_search):
            for cost in (-1, math.nan, math.inf):
                with self.subTest(solver=solver.__name__, cost=cost):
                    with self.assertRaisesRegex(ValueError, "finite costs"):
                        solver(_Space({"start": [("done", "bad", 1)]}, lambda _: cost))
            space = _Space({"start": [("done", "finish", 1)]}, lambda _: 1)
            space.finish = lambda state: replace(state, seconds=state.seconds + 1) if state.name == "done" else None
            with self.assertRaisesRegex(ValueError, "explicit search operation"):
                solver(space)

    def test_budget_exhaustion_and_infeasibility_are_distinct(self):
        for solver in (solve_shortest_path, solve_layered_search):
            space = _Space({}, lambda _: 0)
            self.assertEqual(solver(space).status, "INFEASIBLE")
            for budget in ({"max_states": 0}, {"deadline_seconds": 0}):
                result = solver(space, **budget)
                self.assertEqual(result.status, "UNSOLVED")
                self.assertTrue(result.limited)


if __name__ == "__main__":
    unittest.main()
