"""Independent bounds and storage audits for the modeled compute-cost objective."""
from collections import Counter
from dataclasses import replace
from itertools import permutations
import math
import unittest

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, SourcePreparation, StorageBlock, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState, logical_cost_profile, logical_source_problem, plan_scene_storage, solve_storage_exact,
)


METHODS = ("exact", "dynamic_programming")


def triangle():
    problem = ContractionProblem(
        {k: TensorSpec((2,), "float64") for k in "ABC"},
        tuple(ContractionTerm(tuple(pair), "i,i->") for pair in ("AB", "AC", "BC")),
        {k: SourceSpec(k, 0) for k in "ABC"})
    blocks = {k: StorageBlock(k, "all", 16) for k in "ABC"}
    return problem, blocks


class StorageExactTests(unittest.TestCase):
    def test_rematerialization_minimizes_preparation_and_compute_cost(self):
        problem, blocks = triangle()
        profile = CostProfile()
        for costs in ((1, 2, 3), (3, 2, 1)):
            preps = {k: SourcePreparation((k,), flops=int(value * 1e9)) for k, value in zip("ABC", costs)}
            # At 96 B a second contraction cannot coexist with all 3 inputs
            # and the first scalar result (104 B). One input must be reloaded.
            # A 4-load schedule exists, so choose the cheapest preparation to repeat.
            # Physical reads occur once but are not priced. Loads retain the
            # default owned-snapshot cost; preparation and contraction remain priced.
            load = 1e-6 + 16 * 1e-10
            compute = 1.5e-5 + 2 * 1e-12 + 40 * 5e-12
            preparation = (sum(costs) + min(costs)) * 1e-3
            expected = preparation + 4 * load + 3 * compute
            cheapest = "ABC"[costs.index(min(costs))]
            for method in METHODS:
                with self.subTest(costs=costs, method=method):
                    result = solve_storage_exact(problem, blocks, preps, profile=profile,
                                                 memory_limit=96, optimizer=method)
                    self.assertEqual(result.status, "OPTIMAL")
                    self.assertAlmostEqual(result.plan.seconds, expected, places=14)
                    loads = Counter(a.tensor for a in result.plan.state.actions if a.kind == "load")
                    self.assertEqual(loads, Counter({k: 2 if k == cheapest else 1 for k in "ABC"}))
                    self.assertEqual(sum(e.kind == "read" for e in result.plan.events), 3)
                    self.assertLessEqual(result.plan.peak_host_bytes, 96)

    def test_shared_block_lifetimes_fit_without_reserving_all_raw_inputs(self):
        keys = ("a", "c", "b", "d")
        problem = ContractionProblem({k: TensorSpec((1,), "float64") for k in keys},
                                     tuple(ContractionTerm((k,), "i->") for k in keys),
                                     {k: SourceSpec(k, 0) for k in keys})
        blocks = {k: StorageBlock(k, "all", 104, host_bytes=104) for k in "AB"}
        preps = {k: SourcePreparation(("A" if k in "ab" else "B",)) for k in keys}
        profile = CostProfile()
        expected = 4 * (1e-6 + 8e-10 + 1.5e-5 + 1e-12 + 16 * 5e-12)
        for method in METHODS:
            for cap, status in ((135, "INFEASIBLE"), (136, "OPTIMAL")):
                with self.subTest(method=method, cap=cap):
                    result = solve_storage_exact(problem, blocks, preps, memory_limit=cap, optimizer=method)
                    self.assertEqual(result.status, status)
                    if result.plan:
                        self.assertAlmostEqual(result.plan.seconds, expected, places=14)
                        self.assertLessEqual(result.plan.peak_host_bytes, cap)

    def test_small_all_once_schedules_match_independent_permutation_audit(self):
        # Every complete simple schedule has 2 loads, 2 unary operations, and
        # 2 logical releases. Enumerate all 720 orders, independently of search.
        problem = ContractionProblem({k: TensorSpec((2,), "float64") for k in "AB"},
                                     tuple(ContractionTerm((k,), "i->") for k in "AB"),
                                     {k: SourceSpec(k, 0) for k in "AB"})
        blocks = {k: StorageBlock(k, "all", 16, host_bytes=16) for k in "AB"}
        preps = {k: SourcePreparation((k,), workspace_host_bytes=8) for k in "AB"}
        profile = CostProfile()
        for cap in (47, 48, 64):
            reference = math.inf
            for order in permutations((("load", 0), ("contract", 0), ("release", 0),
                                       ("load", 1), ("contract", 1), ("release", 1))):
                state = ContractionState.initial(logical_source_problem(problem))
                try:
                    for kind, index in order:
                        tensor = "AB"[index]
                        if kind == "load":
                            state = state.load(tensor, logical_cost_profile(profile), 10000)
                        elif kind == "contract":
                            state = state.contract(index, (0,), logical_cost_profile(profile), 10000)
                        else:
                            state = state.release(tensor)
                except (MemoryError, ValueError):
                    continue
                plan = plan_scene_storage(state, blocks, preps, profile=profile, memory_limit=cap)
                if plan.feasible:
                    reference = min(reference, plan.seconds)
            for method in METHODS:
                result = solve_storage_exact(problem, blocks, preps, memory_limit=cap, optimizer=method)
                self.assertEqual(result.status, "OPTIMAL" if math.isfinite(reference) else "INFEASIBLE")
                if result.plan:
                    self.assertAlmostEqual(result.plan.seconds, reference, places=14)

    def test_preparation_staging_output_copies_and_reserves_are_hard_limits(self):
        problem = ContractionProblem({"A": TensorSpec((1,), "float64")},
                                     (ContractionTerm(("A",), "i->"),) * 4, {"A": SourceSpec("A", 0)})
        for method in METHODS:
            for cap, status in ((39, "INFEASIBLE"), (40, "OPTIMAL")):
                result = solve_storage_exact(problem, {"A": StorageBlock("A", "all", 8)},
                    {"A": SourcePreparation(("A",))}, memory_limit=cap, optimizer=method)
                self.assertEqual(result.status, status)
            gpu = replace(problem, backend="cupy")
            for block, prep in ((StorageBlock("A", "all", 8, staging_host_bytes=33), SourcePreparation(("A",))),
                                (StorageBlock("A", "all", 8), SourcePreparation(("A",), workspace_host_bytes=33))):
                result = solve_storage_exact(gpu, {"A": block}, {"A": prep},
                    memory_limit=1000, host_memory_limit=32, optimizer=method)
                self.assertEqual(result.status, "INFEASIBLE")
            result = solve_storage_exact(problem, {"A": StorageBlock("A", "all", 8)},
                {"A": SourcePreparation(("A",))}, memory_limit=40, reserved_host_bytes=1, optimizer=method)
            self.assertEqual(result.status, "INFEASIBLE")

    def test_budgets_do_not_turn_incomplete_search_into_a_certificate(self):
        problem, blocks = triangle()
        preps = {k: SourcePreparation((k,)) for k in "ABC"}
        for method in METHODS:
            for limit in ({"max_states": 0}, {"deadline_seconds": 0}):
                result = solve_storage_exact(problem, blocks, preps, memory_limit=96,
                                             optimizer=method, **limit)
                self.assertEqual(result.status, "UNSOLVED")
                self.assertTrue(result.limited)
                self.assertIsNone(result.plan)


if __name__ == "__main__":
    unittest.main()
