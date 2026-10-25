"""Behavioral checks for shared budgets, proof coverage and strategy composition."""
from dataclasses import dataclass
import math
import random
import unittest

from lqcdweave.contraction.planning.optimizers.bounds import LowerBound
from lqcdweave.contraction.planning.optimizers.search import ExactSearch
from lqcdweave.contraction.planning.optimizers.session import SearchSession, solve_storage
from lqcdweave.contraction.planning.batch import solve_storage_batch
from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, StorageBlock, SourcePreparation,
)


@dataclass(frozen=True)
class Node:
    name: int
    seconds: float = 0.0


class Graph:
    def __init__(self, edges, target): self.edges, self.target = edges, target
    def initial_state(self): return Node(0)
    def key(self, state): return state.name
    def cost(self, state): return state.seconds
    def is_feasible(self, state): return True
    def finish(self, state): return state if state.name == self.target else None
    def layer(self, state): return self.target - state.name
    def successors(self, state):
        for target, cost in self.edges.get(state.name, ()):
            yield Node(target, state.seconds + cost)




class UnifiedSearchTests(unittest.TestCase):
    def test_conservative_batch_never_promotes_local_proof_to_global_certificate(self):
        problem = ContractionProblem({k: TensorSpec((1,), "float64") for k in "AB"},
            tuple(ContractionTerm((k,), "i->") for k in "AB"), {k: SourceSpec(k, 0) for k in "AB"})
        blocks = {k: StorageBlock(k, "all", 104, host_bytes=104) for k in "AB"}
        preps = {k: SourcePreparation((k,)) for k in "AB"}
        self.assertEqual(solve_storage(problem, blocks, preps, memory_limit=150, optimizer="exact").status, "OPTIMAL")
        for cap, status in ((150, "UNSOLVED"), (4096, "FEASIBLE")):
            result = solve_storage_batch(problem, blocks, preps, memory_limit=cap, optimizer="exact")
            self.assertEqual(result.status, status)
            self.assertIsNone(result.lower_bound_seconds)
            self.assertEqual(result.diagnostic["proof_scope"], "conservative_batch")


    def test_partial_expansion_keeps_its_bound_and_incumbent(self):
        space = Graph({0: [(2, 10), (2, 1)]}, 2)
        result = SearchSession(space, ExactSearch(), max_transitions=1).run()
        self.assertEqual(result.status, "FEASIBLE")
        self.assertEqual(result.state.seconds, 10)
        self.assertEqual(result.lower_bound_seconds, 0)
        self.assertEqual(result.stop_reason, "max_transitions")
        self.assertEqual(result.statistics["evaluated_transitions"], 1)

    def test_inconsistent_admissible_bound_reopens_state(self):
        space = Graph({0: [(1, 2), (2, 1)], 1: [(3, 2)], 2: [(1, .5), (3, 100)]}, 3)
        bound = LowerBound(space, lambda s: 2.5 if s.name == 2 else 0, "known shortest residuals")
        result = SearchSession(space, ExactSearch(lower_bound=bound)).run()
        self.assertEqual(result.status, "OPTIMAL")
        self.assertEqual(result.state.seconds, 3.5)
        self.assertEqual(result.lower_bound_seconds, 3.5)
        self.assertEqual([cost for _, cost in result.statistics["improvement_curve"]], [4, 3.5])
        with self.assertRaisesRegex(ValueError, "different search domain"):
            SearchSession(Graph(space.edges, 3), ExactSearch(lower_bound=bound)).run()

    def test_random_small_graphs_match_independent_relaxation(self):
        rng = random.Random(709)
        for case in range(30):
            edges = {i: [(j, rng.randrange(6)) for j in range(i, 7) if rng.random() < .4]
                     for i in range(6)}
            distance = [0] + [math.inf] * 6
            for _ in range(7):
                for i, outgoing in edges.items():
                    for j, weight in outgoing:
                        distance[j] = min(distance[j], distance[i] + weight)
            for layered in (False, True):
                with self.subTest(case=case, layered=layered):
                    result = SearchSession(Graph(edges, 6), ExactSearch(layered=layered)).run()
                    self.assertEqual(result.status, "OPTIMAL" if math.isfinite(distance[6]) else "INFEASIBLE")
                    if result.state:
                        self.assertEqual(result.state.seconds, distance[6])



    def test_physical_rejection_does_not_publish_a_logical_proof(self):
        from lqcdweave.contraction.planning.optimizers.results import CandidateMemoryExceeded
        space = Graph({0: [(1, 1)]}, 1)
        def reject(state):
            raise CandidateMemoryExceeded("physical allocation exceeds cap")
        result = SearchSession(space, ExactSearch(), audit=reject).run()
        self.assertEqual(result.status, "UNSOLVED")
        self.assertIsNone(result.state)
        self.assertEqual(result.stop_reason, "physical_admission")
        self.assertEqual(result.statistics["rejected_physical_candidates"], 1)
        self.assertIsNone(result.lower_bound_seconds)
        class TwoGoals(Graph):
            def finish(self, state):
                return state if state.name in (1, 2) else None
        def admit_second(state):
            if state.name == 1:
                raise CandidateMemoryExceeded("first goal exceeds cap")
            return state
        accepted = SearchSession(TwoGoals({0: [(1, 1), (2, 2)]}, 2), ExactSearch(),
                                 audit=admit_second).run()
        self.assertEqual(accepted.status, "FEASIBLE")
        self.assertEqual(accepted.state.name, 2)
        self.assertIsNone(accepted.lower_bound_seconds)
        def out_of_memory(state):
            raise MemoryError("host allocation failed")
        with self.assertRaisesRegex(MemoryError, "host allocation failed"):
            SearchSession(space, ExactSearch(), audit=out_of_memory).run()

    def test_audit_error_propagates_and_late_audit_is_not_published(self):
        space = Graph({0: [(1, 1)]}, 1)
        def bad_audit(state): raise RuntimeError("invalid physical replay")
        with self.assertRaisesRegex(RuntimeError, "physical replay"):
            SearchSession(space, ExactSearch(), audit=bad_audit).run()
        now = [0.0]
        def late_audit(state):
            now[0] = 2
            return state
        result = SearchSession(space, ExactSearch(), audit=late_audit,
                               clock=lambda: now[0], deadline_seconds=1).run()
        self.assertEqual(result.status, "UNSOLVED")
        self.assertIsNone(result.state)
        self.assertEqual(result.stop_reason, "deadline")











if __name__ == "__main__":
    unittest.main()
