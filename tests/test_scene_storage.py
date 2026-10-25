import unittest
import weakref
import tempfile
from pathlib import Path
from dataclasses import replace

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourcePreparation, SourceSpec,
    StorageBlock, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import ContractionState, solve_storage
from lqcdweave.adapters.scene_storage import NpyBlockBinding, NpySceneBlockLoader
from lqcdweave.contraction.planning.api import (
    logical_source_problem, logical_cost_profile, plan_scene_storage,
)
from lqcdweave.contraction.execution.api import replay_scene_storage


class SceneStorageTests(unittest.TestCase):
    def problem(self, backend="numpy"):
        return ContractionProblem(
            {"a": TensorSpec((2, 3), "float64"), "b": TensorSpec((3, 2), "float64")},
            (ContractionTerm(("a",), "ij->"), ContractionTerm(("b",), "ij->")),
            {"a": SourceSpec("legacy-a", 500, read_bytes=1000),
             "b": SourceSpec("legacy-b", 500, read_bytes=1000)}, backend=backend,
        )

    def path(self, backend="numpy"):
        profile = CostProfile()
        model = logical_cost_profile(profile)
        s = ContractionState.initial(logical_source_problem(self.problem(backend)))
        s = s.load("a", model, 100000).contract(0, (0,), model, 100000).release("a")
        s = s.load("b", model, 100000).contract(1, (0,), model, 100000).release("b")
        return s, profile

    def test_one_read_shared_across_different_shapes_and_late_demand(self):
        s, profile = self.path()
        block = StorageBlock("dataset-v1", "rows=0:2,cols=0:3", 48, host_bytes=48)
        plan = plan_scene_storage(s, {"raw": block},
                                  {k: SourcePreparation(("raw",)) for k in ("a", "b")},
                                  profile=profile, memory_limit=10000)
        self.assertEqual([(e.kind, e.action_index) for e in plan.events], [("read", 0), ("release", 3)])
        self.assertEqual(plan.blocks["raw"].read_bytes, 48)
        # First kernel: raw 48 + source 48 + result 8 + workspace (48 + 8).
        # At second materialization: raw 48 + first result 8 + source 48.
        self.assertEqual(plan.peak_host_bytes, 160)
        self.assertEqual(plan.peak_device_bytes, 0)
        refs, events = [], []

        class Loader:
            def read_block(self, key, spec):
                events.append("read")
                value = np.arange(6, dtype=np.float64).reshape(2, 3).copy()
                refs.append(weakref.ref(value))
                return value

            def materialize(self, key, source, blocks):
                events.append(key)
                self_outer.assertIsNotNone(refs[0]())
                return blocks["raw"] if key == "a" else blocks["raw"].T

            def close_block(self, key, value):
                events.append("close")

        self_outer = self
        report = replay_scene_storage(plan, Loader(), measure=True)
        np.testing.assert_array_equal(report.values, (15, 15))
        self.assertEqual(events, ["read", "a", "b", "close"])
        self.assertIsNone(refs[0]())
        self.assertAlmostEqual(sum(t.estimated_seconds for t in report.timings), plan.seconds)

    def test_gpu_model_counts_raw_transfer_and_preparation_workspace(self):
        s, profile = self.path("cupy")
        plan = plan_scene_storage(s, {"raw": StorageBlock("v1", "selection", 48, host_bytes=48,
                                                         device_bytes=48, transfer_bytes=48,
                                                         staging_host_bytes=24)},
                                  {k: SourcePreparation(("raw",), workspace_device_bytes=64,
                                                        flops=100, kernel_calls=1) for k in ("a", "b")},
                                  profile=profile, memory_limit=167)
        # Five live request slots (raw, prepared input, prior scalar result,
        # new result and preparation/contract scratch) each occupy 512 bytes.
        self.assertEqual(plan.peak_device_bytes, 5 * 512)
        self.assertEqual(plan.peak_host_bytes, 72)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.blocks["raw"].transfer_bytes, 48)
        self.assertGreater(plan.preparation_seconds, 0)
        with self.assertRaises(MemoryError):
            replay_scene_storage(plan, None)

    def test_failure_drops_shared_cache(self):
        s, profile = self.path()
        plan = plan_scene_storage(s, {"raw": StorageBlock("v1", "all", 48, host_bytes=48)},
                                  {k: SourcePreparation(("raw",)) for k in ("a", "b")},
                                  profile=profile, memory_limit=10000)
        released = []

        class Loader:
            def read_block(self, key, block):
                return np.ones((2, 3))

            def materialize(self, key, source, blocks):
                raise RuntimeError("bad input")

            def close_block(self, key, value):
                released.append(key)

        with self.assertRaisesRegex(RuntimeError, "bad input"):
            replay_scene_storage(plan, Loader())
        self.assertEqual(released, ["raw"])

    def test_rejects_duplicate_physical_identity_and_legacy_billing(self):
        s, profile = self.path()
        block = StorageBlock("v1", "all", 48, host_bytes=48)
        prep = {k: SourcePreparation(("raw",)) for k in ("a", "b")}
        with self.assertRaisesRegex(ValueError, "one physical"):
            plan_scene_storage(s, {"raw": block, "alias": block}, prep, profile=profile, memory_limit=10000)
        with self.assertRaisesRegex(ValueError, "logical_source_problem"):
            plan_scene_storage(replace(s, problem=self.problem()), {"raw": block}, prep,
                               profile=profile, memory_limit=10000)

    def test_unused_block_is_never_read(self):
        s, profile = self.path()
        plan = plan_scene_storage(s, {"raw": StorageBlock("v1", "all", 48, host_bytes=48),
                                     "unused": StorageBlock("v2", "all", 10**9, host_bytes=10**9)},
                                  {k: SourcePreparation(("raw",)) for k in ("a", "b")},
                                  profile=profile, memory_limit=10000)
        self.assertTrue(plan.feasible)
        self.assertEqual({e.block for e in plan.events}, {"raw"})

    def test_actual_npy_maximum_selection_serves_transposed_views(self):
        s, profile = self.path()
        plan = plan_scene_storage(s, {"raw": StorageBlock("v1", "rows=0:2,cols=0:3", 48, host_bytes=48)},
                                  {k: SourcePreparation(("raw",)) for k in ("a", "b")},
                                  profile=profile, memory_limit=10000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npy"
            np.save(path, np.arange(20, dtype=np.float64).reshape(4, 5))
            binding = NpyBlockBinding(path, (slice(0, 2), slice(0, 3)), (2, 3), "float64")
            loader = NpySceneBlockLoader({"raw": binding}, {
                "a": lambda metadata, raw: raw["raw"],
                "b": lambda metadata, raw: raw["raw"].T,
            })
            original = loader.read_block
            calls = []
            def read(key, block):
                calls.append(key)
                return original(key, block)
            loader.read_block = read
            np.testing.assert_array_equal(replay_scene_storage(plan, loader), (21, 21))
            self.assertEqual(calls, ["raw"])

    def test_sparse_borrow_survives_until_record_release(self):
        from test_scene_basis_batch import _bridge_problem
        problem = logical_source_problem(_bridge_problem(2, 3))
        profile = CostProfile()
        preparations = {k: SourcePreparation((k,)) for k in problem.sources}
        blocks = {k: StorageBlock(k, "full", 10000, host_bytes=10000) for k in problem.sources}
        result = solve_storage(problem, blocks, preparations, profile=profile,
                               memory_limit=1 << 20, optimizer="anytime")
        self.assertIsNotNone(result.plan)
        plan = result.plan
        release = next(e.action_index for e in plan.events if e.block == "record" and e.kind == "release")
        last_use = max(i for i, action in enumerate(result.state.actions)
                       if "record" in action.operands)
        self.assertGreaterEqual(release, last_use)
        record_releases = [i for i, action in enumerate(result.state.actions)
                           if (action.kind, action.tensor) == ("release", "record")]
        if record_releases:
            self.assertEqual(len(record_releases), 1)
            self.assertGreaterEqual(release, record_releases[0])
        else:
            self.assertEqual(release, len(result.state.actions))
            self.assertIn("record", result.state.loaded)
        closed = []
        class Loader:
            def read_block(self, key, block):
                if key == "record":
                    return ((np.eye(4), 1.0, {"type": "identity"}),)
                return np.ones(problem.specs[key].shape, dtype=np.complex128)
            def materialize(self, key, source, data):
                return data[key]
            def close_block(self, key, value):
                closed.append(key)
        values = replay_scene_storage(plan, Loader())
        np.testing.assert_allclose(values[0], np.full((2, 2), 36))
        self.assertCountEqual(closed, problem.sources)

    def test_logical_eviction_does_not_reread_raw_block(self):
        s, profile = self.path()
        model = logical_cost_profile(profile)
        s = ContractionState.initial(s.problem).load("a", model, 100000).release("a")
        s = s.load("b", model, 100000).contract(1, (0,), model, 100000).release("b")
        s = s.load("a", model, 100000).contract(0, (0,), model, 100000).release("a")
        plan = plan_scene_storage(s, {"raw": StorageBlock("v1", "all", 48, host_bytes=48)},
                                  {k: SourcePreparation(("raw",)) for k in ("a", "b")},
                                  profile=profile, memory_limit=10000)
        self.assertEqual([(e.kind, e.action_index) for e in plan.events], [("read", 0), ("release", 5)])


if __name__ == "__main__":
    unittest.main()
