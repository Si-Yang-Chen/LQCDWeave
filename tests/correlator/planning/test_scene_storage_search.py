from lqcdweave.contraction.planning.api import PathSearchExhausted
"""Module tests split from test_scene_storage_search.py; shared numerical fixtures stay unchanged."""

from test_scene_storage_search import (
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorSpec,
    _StorageSearchFixture,
    solve_storage,
)


class StorageSearchTests(_StorageSearchFixture):
    def test_workspace_and_host_cap_participate_before_acceptance(self):
        problem, blocks, preps = self.case("cupy")
        blocks["A"] = StorageBlock("A", "full", 104, device_bytes=104, staging_host_bytes=200)
        with self.assertRaises(PathSearchExhausted):
            solve_storage(problem, blocks, preps, memory_limit=150, host_memory_limit=100)
        preps["a"] = SourcePreparation(("A",), workspace_device_bytes=100)
        with self.assertRaises(PathSearchExhausted):
            solve_storage(problem, blocks, preps, memory_limit=150, host_memory_limit=1000)

    def test_budget_and_repeatability(self):
        problem, blocks, preps = self.case()
        with self.assertRaises(PathSearchExhausted) as exc:
            solve_storage(problem, blocks, preps, memory_limit=150, max_states=0)
        self.assertTrue(exc.exception.search_result.limited)
        a = solve_storage(problem, blocks, preps, memory_limit=10000)
        b = solve_storage(problem, blocks, preps, memory_limit=10000)
        self.assertEqual(a.plan.state.actions, b.plan.state.actions)
        self.assertEqual(a.plan.events, b.plan.events)

    def test_storage_dependency_ordering_is_deferred_to_postcheck_or_exact(self):
        keys = ("a", "b", "p", "q")
        problem = ContractionProblem({k: TensorSpec((1,), "float64") for k in keys},
                                     tuple(ContractionTerm((k,), "i->") for k in keys),
                                     {k: SourceSpec(k, 8) for k in keys})
        blocks = {k: StorageBlock(k, "full", n, host_bytes=n) for k,n in (("X",60),("Y",60),("Z",80))}
        preps = {"a": SourcePreparation(("X",)), "b": SourcePreparation(("Y",)),
                 "p": SourcePreparation(("X", "Z")), "q": SourcePreparation(("Y", "Z"))}
        with self.assertRaisesRegex(PathSearchExhausted, "post-search storage"):
            solve_storage(problem, blocks, preps, memory_limit=180)
        result = solve_storage(problem, blocks, preps, memory_limit=180, optimizer="exact")
        self.assertEqual(result.status, "OPTIMAL")
        self.assertLessEqual(result.plan.peak_host_bytes, 180)

    def test_output_copies_participate_in_complete_step(self):
        problem, blocks, preps = self.case()
        first = solve_storage(problem, blocks, preps, memory_limit=10000)
        self.assertTrue(any(point.phase == "output_copy" for point in first.plan.memory))
        blocks = {k: StorageBlock(k, "full", 0) for k in blocks}
        with self.assertRaises(PathSearchExhausted):
            solve_storage(problem, blocks, preps, memory_limit=50)
