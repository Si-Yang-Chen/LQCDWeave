import unittest

import numpy as np

from lqcdweave.contraction import (
    ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, CostProfile,
    StorageBlock, SourcePreparation, solve_storage,
    logical_source_problem, logical_cost_profile, plan_scene_storage, replay_scene_storage,
)
from lqcdweave.contraction.planning.api import ContractionState, solve_storage_exact


class _StorageSearchFixture(unittest.TestCase):
    def case(self, backend="numpy"):
        keys = ("a", "c", "b", "d")
        problem = ContractionProblem({k: TensorSpec((1,), "float64") for k in keys},
                                     tuple(ContractionTerm((k,), "i->") for k in keys),
                                     {k: SourceSpec(k, 8) for k in keys}, backend=backend)
        blocks = {k: StorageBlock(k, "full", 104, **({"device_bytes": 104} if backend == "cupy"
                                                    else {"host_bytes": 104})) for k in ("A", "B")}
        preps = {k: SourcePreparation(("A" if k in ("a", "b") else "B",)) for k in keys}
        return problem, blocks, preps


class StorageSearchTests(_StorageSearchFixture):
    def test_both_exact_physical_plans_replay_with_one_read_per_block(self):
        problem, blocks, preps = self.case()
        for method in ("exact", "dynamic_programming"):
            result = solve_storage_exact(problem, blocks, preps, memory_limit=136, optimizer=method)
            self.assertEqual(result.status, "OPTIMAL")
            calls = []

            class Loader:
                def read_block(self, key, block):
                    calls.append(key)
                    return np.ones(13)

                def materialize(self, key, source, arrays):
                    return next(iter(arrays.values()))[:1]

            np.testing.assert_array_equal(replay_scene_storage(result.plan, Loader()), (1, 1, 1, 1))
            self.assertCountEqual(calls, ("A", "B"))
            self.assertLessEqual(result.plan.peak_host_bytes, 136)

    def test_search_reorders_to_fit_where_postcheck_fails(self):
        problem, blocks, preps = self.case()
        profile = CostProfile()
        model = logical_cost_profile(profile)
        state = ContractionState.initial(logical_source_problem(problem))
        for i, key in enumerate(("a", "c", "b", "d")):
            state = state.load(key, model, 150).contract(i, (0,), model, 150).release(key)
        previous = plan_scene_storage(state, blocks, preps, profile=profile, memory_limit=150)
        self.assertFalse(previous.feasible)
        result = solve_storage(problem, blocks, preps, profile=profile, memory_limit=150, optimizer="exact", deadline_seconds=5)
        self.assertEqual(result.status, "OPTIMAL")
        self.assertLessEqual(result.plan.peak_host_bytes, 150)
        self.assertGreater(result.rejected_memory, 0)
        self.assertTrue(all(p.host_bytes <= 150 for p in result.plan.memory))
        calls = []
        class Loader:
            def read_block(self, key, block):
                calls.append(key)
                return np.ones(13)
            def materialize(self, key, source, arrays):
                return next(iter(arrays.values()))[:1]
        np.testing.assert_array_equal(replay_scene_storage(result.plan, Loader()), (1, 1, 1, 1))
        self.assertCountEqual(calls, ("A", "B"))



if __name__ == "__main__":
    unittest.main()
