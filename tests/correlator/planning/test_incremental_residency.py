"""Compare derived byte caches with a fresh scan, including shared backing."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, TensorSpec,
)
from lqcdweave.contraction.planning.state import ContractionState, CostProfile, _Catalog


def scan(state):
    sources = state.problem.sources
    regions, host_regions = {}, {}
    snapshots = intermediates = 0
    for key in state.loaded:
        if key in sources:
            source = sources[key]
            regions[source.region] = source.region_bytes
            host_regions[source.region] = source.host_region_bytes
            snapshots += (source.snapshot_bytes if source.snapshot_bytes is not None else
                          math.prod(state.catalog[key].shape) * state.catalog[key].dtype.itemsize
                          if key in state.catalog else 0)
        else:
            spec = state.catalog[key]
            intermediates += math.prod(spec.shape) * spec.dtype.itemsize
    total = sum(regions.values()) + snapshots + intermediates
    return total, total if state.problem.backend == "numpy" else sum(host_regions.values())


class IncrementalResidencyTests(unittest.TestCase):
    def problem(self, backend="numpy", *, single_read=False):
        return ContractionProblem(
            {k: TensorSpec((4,), "float64") for k in ("A", "B")},
            (ContractionTerm(("A", "B"), "i,i->i"), ContractionTerm(("A",), "i->i")),
            {k: SourceSpec("shared", 48, host_region_bytes=96) for k in ("A", "B")}
            | {"record": SourceSpec("borrowed", 20, host_region_bytes=30, snapshot_bytes=0)},
            backend=backend, single_read_regions=single_read,
        )

    def check(self, state):
        self.assertEqual((state.resident_bytes, state.resident_host_bytes), scan(state))

    def test_shared_region_snapshots_intermediates_and_last_release_match_scan(self):
        profile = CostProfile()
        for backend in ("numpy", "cupy"):
            with self.subTest(backend=backend):
                state = ContractionState.initial(self.problem(backend))
                self.check(state)
                for key in ("A", "B", "record"):
                    state = state.load(key, profile, 1000000, 1000000)
                    self.check(state)
                parent = state
                for scene, pair in ((0, (0, 1)), (1, (0,))):
                    state = state.contract(scene, pair, profile, 1000000)
                    self.check(state)
                for key in ("A", "B", "record"):
                    state = state.release(key)
                    self.check(state)
                self.assertEqual(state.resident_bytes, 64)
                self.assertEqual(state.resident_host_bytes, 64 if backend == "numpy" else 0)
                self.check(parent)

    def test_replacements_invalidate_cache_and_siblings_keep_their_own_bytes(self):
        state = ContractionState.initial(self.problem())
        state = state.load("A", CostProfile(), 1000000)
        self.check(state)
        sibling = state.load("B", CostProfile(), 1000000)
        self.check(sibling)
        empty = replace(state, loaded=frozenset())
        self.check(empty)
        resized = replace(state, catalog=state.catalog.updated({"A": TensorSpec((8,), "float64")}))
        self.check(resized)
        self.assertEqual(resized.resident_bytes - state.resident_bytes, 32)
        updated = replace(state, problem=replace(state.problem, sources={
            **state.problem.sources, "A": SourceSpec("shared", 48, host_region_bytes=96, snapshot_bytes=80),
        }))
        self.check(updated)
        self.assertEqual(updated.resident_bytes - state.resident_bytes, 48)
        self.check(state)

    def test_failed_transition_and_single_read_guard_preserve_parent_residency(self):
        state = ContractionState.initial(self.problem(single_read=True)).load("A", CostProfile(), 1000000)
        before = scan(state)
        with self.assertRaises(MemoryError):
            state.load("B", CostProfile(), state.resident_bytes)
        self.assertEqual(scan(state), before)
        state = state.load("B", CostProfile(), 1000000).release("A")
        state = state.load("A", CostProfile(), 1000000)
        self.check(state)
        state = state.release("A").release("B")
        self.check(state)
        with self.assertRaisesRegex(MemoryError, "second time"):
            state.load("A", CostProfile(), 1000000)

    def test_peak_reset_and_generated_successor_do_not_rescan_old_allocations(self):
        state = ContractionState.initial(self.problem())
        for key in ("A", "B"):
            state = state.load(key, CostProfile(), 1000000)
        self.check(state)
        with patch.object(ContractionState, "_snapshot_bytes", side_effect=AssertionError("rescan")):
            reset = state._without_peaks()
            successor = reset.contract(0, (0, 1), CostProfile(), 1000000)
            self.assertEqual(successor.resident_bytes, state.resident_bytes + 32)
        self.check(successor)

    def test_catalog_overlay_preserves_order_updates_and_all_parent_snapshots(self):
        initial = {str(i): TensorSpec((i,), "float32") for i in range(80)}
        catalog = _Catalog(initial)
        base = catalog._values
        reference = dict(initial)
        snapshots = []
        for i in range(160):
            snapshots.append((catalog, dict(reference)))
            entry = {"10": TensorSpec((i + 1,), "float64"), "node:" + str(i): TensorSpec((i,), "complex128")}
            reference.update(entry)
            catalog = catalog.updated(entry)
            self.assertEqual(list(catalog), list(reference))
            self.assertEqual(dict(catalog), reference)
            self.assertEqual(len(catalog), len(reference))
            self.assertLessEqual(len(catalog._overlay), catalog.overlay_limit)
            if i < 63:
                self.assertIs(catalog._values, base)
        for old, values in snapshots:
            self.assertEqual(dict(old), values)
        with self.assertRaises(TypeError):
            catalog._overlay["new"] = TensorSpec((), "float64")
        with self.assertRaises(AttributeError):
            catalog._size = 0


if __name__ == "__main__":
    unittest.main()
