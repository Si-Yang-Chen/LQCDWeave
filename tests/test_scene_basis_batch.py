"""Maximum Ne/Np reads must serve every view of one backing block."""

import unittest
from dataclasses import replace

import numpy as np

from lqcdweave.adapters.scene_basis_region import MaxBasisRegionLoader
from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    SourceSpec,
    SparseBridgeSpec,
    TensorSpec,
    StorageBlock,
    SourcePreparation,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState,
    combine_basis_problems,
    solve_storage,
)
from lqcdweave.contraction.execution.api import replay, replay_scene_storage, sparse_pairs


def _dense_problem(ne, np_):
    spec = TensorSpec((ne, np_), "float64")
    return ContractionProblem(
        {"block": spec},
        (ContractionTerm(("block",), "ij->"),),
        {"block": SourceSpec("same-physical-block", ne * np_ * 8,
                             metadata={"used_ne": ne, "used_np": np_})},
    )


def _bridge_problem(ne, np_):
    specs = {
        "left": TensorSpec((ne, 4, np_, 3), "complex128"),
        "right": TensorSpec((4, np_, 3, ne), "complex128"),
    }
    sources = {
        key: SourceSpec(key, int(np.prod(spec.shape)) * spec.dtype.itemsize,
                        metadata={"used_ne": ne, "used_np": np_})
        for key, spec in specs.items()
    }
    sources["record"] = SourceSpec(
        "same-p2p-record", np_ * (3 * 3 * 16 + 8),
        metadata={"family": "p2p", "used_ne": ne, "used_np": np_},
    )
    return ContractionProblem(
        specs,
        (ContractionTerm(("left", "right"), "aipc,jqdb->ab"),),
        sources,
        sparse_bridges=(SparseBridgeSpec("left", "right", "record", np_, np_),),
    )



def _physical_batch(batch, read_max):
    view_loader = MaxBasisRegionLoader(batch, read_max)
    blocks = {region: StorageBlock(region, "maximum", batch.problem.sources[key].region_bytes,
                                  host_bytes=batch.problem.sources[key].region_bytes)
              for region, key in batch.representative_by_region.items()}
    preparations = {key: SourcePreparation((source.region,)) for key, source in batch.problem.sources.items()}
    class Loader:
        def read_block(self, region, block):
            key = batch.representative_by_region[region]
            return view_loader.load_region(region, batch.problem.sources[key])
        def materialize(self, key, source, arrays):
            return view_loader.view(next(iter(arrays.values())), key, batch.problem.sources[key])
    return blocks, preparations, Loader()


class SceneBasisBatchTests(unittest.TestCase):
    def test_anytime_reuses_physical_block_for_later_basis_views(self):
        batch = combine_basis_problems(
            ((2, 4, _dense_problem(2, 4)), (4, 2, _dense_problem(4, 2))),
            backing_identity="same-ordered-bases",
        )
        full = np.arange(16, dtype=np.float64).reshape(4, 4)
        for optimizer in ('anytime',):
            with self.subTest(optimizer=optimizer):
                reads = []
                def read_max(request):
                    reads.append(request.spec.shape)
                    return full.copy()
                blocks, preparations, loader = _physical_batch(batch, read_max)
                result = solve_storage(batch.problem, blocks, preparations, optimizer=optimizer,
                                                  memory_limit=1 << 20, deadline_seconds=2)
                self.assertEqual(result.status, 'FEASIBLE')
                values = replay_scene_storage(result.plan, loader)
                self.assertEqual(reads, [(4, 4)])
                np.testing.assert_array_equal(values, (full[:2, :4].sum(), full[:4, :2].sum()))

    def test_crossed_ne_np_requests_read_one_joint_maximum(self):
        batch = combine_basis_problems(
            ((2, 4, _dense_problem(2, 4)), (4, 2, _dense_problem(4, 2))),
            backing_identity="cfg10000:verified-ordered-mode-and-point-bases",
        )
        self.assertEqual(len(batch.problem.terms), 2)
        self.assertEqual(len(batch.maxima_by_region), 1)
        region = next(iter(batch.maxima_by_region))
        self.assertEqual(batch.maxima_by_region[region], (4, 4))
        self.assertEqual(batch.spec_by_region[region].shape, (4, 4))
        self.assertEqual({source.region_bytes for source in batch.problem.sources.values()}, {128})

        reads = []
        full = np.arange(16, dtype=np.float64).reshape(4, 4)

        def read_max(request):
            reads.append((request.used_ne, request.used_np, request.spec.shape))
            self.assertEqual(request.source.metadata["used_ne"], 4)
            self.assertEqual(request.source.metadata["used_np"], 4)
            return full.copy()

        for optimizer in ("exact", "dynamic_programming"):
            with self.subTest(optimizer=optimizer):
                reads.clear()
                blocks, preparations, loader = _physical_batch(batch, read_max)
                planned = solve_storage(
                    batch.problem, blocks, preparations, optimizer=optimizer, memory_limit=1 << 20,
                    max_states=10_000, deadline_seconds=10,
                )
                self.assertEqual(planned.status, "OPTIMAL")
                values = replay_scene_storage(planned.plan, loader)
                self.assertEqual(reads, [(4, 4, (4, 4))])
                np.testing.assert_array_equal(values, (full[:2, :4].sum(), full[:4, :2].sum()))

        first_key, second_key = tuple(batch.problem.sources)
        state = ContractionState.initial(batch.problem).load(first_key, CostProfile(), 1 << 20)
        state = state.release(first_key)
        with self.assertRaisesRegex(MemoryError, "second time"):
            state.load(second_key, CostProfile(), 1 << 20)

    def test_sparse_p2p_views_are_derived_from_one_maximum_record(self):
        batch = combine_basis_problems(
            ((2, 4, _bridge_problem(2, 4)), (4, 2, _bridge_problem(4, 2))),
            backing_identity="cfg10000:verified-ordered-mode-and-point-bases",
        )
        calls = []

        def read_max(request):
            calls.append((request.original_key, request.used_ne, request.used_np))
            if request.original_key == "left":
                return np.ones(request.spec.shape, dtype=np.complex128)
            if request.original_key == "right":
                return np.ones(request.spec.shape, dtype=np.complex128)
            self.assertIsNone(request.spec)
            return ((np.eye(4), 1, {
                "layout": "matched",
                "indices": np.array([0, 1, 3, 2], dtype=np.int64),
                "values": np.ones((4, 3, 3), dtype=np.complex128),
            }),)

        state = ContractionState.initial(batch.problem)
        for key in batch.problem.sources:
            state = state.load(key, CostProfile(), 1 << 22)
        for index in range(2):
            state = state.contract(index, (0, 1), CostProfile(), 1 << 22)
        self.assertTrue(state.done)

        values = replay(state, MaxBasisRegionLoader(batch, read_max))
        self.assertEqual(len(calls), 3)
        self.assertTrue(all((ne, np_) == (4, 4) for _, ne, np_ in calls))
        np.testing.assert_array_equal(values[0], np.full((2, 2), 4 * 9 * 4))
        np.testing.assert_array_equal(values[1], np.full((4, 4), 2 * 9 * 4))

    def test_legacy_negative_p2p_indices_follow_each_requested_np(self):
        batch = combine_basis_problems(
            ((2, 4, _bridge_problem(2, 4)), (4, 2, _bridge_problem(4, 2))),
            backing_identity="cfg10000:verified-ordered-mode-and-point-bases",
        )
        record = ((np.eye(4), 1, {
            "type": "sparse",
            "indices": np.array([[-1, -1], [3, 3]]),
            "values": np.ones((2, 3, 3), dtype=np.complex128),
        }),)
        loader = MaxBasisRegionLoader(batch, lambda request: record)
        small_key = "basisbatch:1:record"
        small = loader.view(record, small_key, batch.problem.sources[small_key])
        pairs, _ = sparse_pairs(small[0][2], 2)
        np.testing.assert_array_equal(pairs, np.array([[1, 1]]))

    def test_different_component_selections_need_an_explicit_view(self):
        first = _dense_problem(2, 4)
        second = _dense_problem(4, 2)
        first = replace(first, sources={
            "block": replace(first.sources["block"], metadata={"used_ne": 2, "used_np": 4, "slice": 0})
        })
        second = replace(second, sources={
            "block": replace(second.sources["block"], metadata={"used_ne": 4, "used_np": 2, "slice": 1})
        })
        batch = combine_basis_problems(
            ((2, 4, first), (4, 2, second)), backing_identity="same-verified-backing"
        )
        with self.assertRaisesRegex(ValueError, "component selections"):
            MaxBasisRegionLoader(batch, lambda request: np.zeros(request.spec.shape))


if __name__ == "__main__":
    unittest.main()
