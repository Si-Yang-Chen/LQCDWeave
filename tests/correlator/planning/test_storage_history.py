"""Keep fixed-storage algebra replay independent of the emitted Action prefix."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorSpec,
)
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning import storage
from lqcdweave.contraction.planning.api import logical_cost_profile, logical_source_problem
from lqcdweave.contraction.planning.state import ContractionState


def _resident_scan(state):
    regions = {}
    snapshots = 0
    intermediates = 0
    for key in state.loaded:
        if key in state.problem.sources:
            source = state.problem.sources[key]
            regions[source.region] = source.region_bytes
            size = source.snapshot_bytes
            if size is None:
                spec = state.catalog[key]
                size = math.prod(spec.shape) * spec.dtype.itemsize
            snapshots += size
        else:
            spec = state.catalog[key]
            intermediates += math.prod(spec.shape) * spec.dtype.itemsize
    total = sum(regions.values()) + snapshots + intermediates
    return total, total if state.problem.backend == "numpy" else 0


class _ArrayLoader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []

    def read_block(self, key, block):
        self.reads.append(key)
        return self.arrays[block.dataset]

    def materialize(self, key, source, blocks):
        return blocks[source.region]

    def close_all(self):
        pass


class StorageHistoryTests(unittest.TestCase):
    def test_short_replay_preserves_caches_numpy_values_and_storage_trajectory(self):
        specs = {
            "A": TensorSpec((3, 4), np.float64),
            "B": TensorSpec((4, 2), np.float64),
            "C": TensorSpec((2, 1), np.float64),
        }
        problem = logical_source_problem(ContractionProblem(
            specs,
            (ContractionTerm(("A", "B", "C"), "ij,jk,kl->il"),),
            {key: SourceSpec(key, 0) for key in specs},
        ))
        profile = CostProfile()
        model = logical_cost_profile(profile)
        state = ContractionState.initial(problem)
        for key in specs:
            state = state.load(key, model, 1 << 20)
        state = state.contract(0, (0, 1), model, 1 << 20)
        state = state.contract(0, (0, 1), model, 1 << 20)
        self.assertTrue(state.done)

        rng = np.random.default_rng(705)
        arrays = {key: rng.normal(size=spec.shape).astype(spec.dtype)
                  for key, spec in specs.items()}
        blocks = {key: StorageBlock(key, "full", arrays[key].nbytes,
                                    host_bytes=arrays[key].nbytes)
                  for key in specs}
        preparations = {key: SourcePreparation((key,)) for key in specs}
        original_actions = state.actions
        cache_from_parent = {}
        before_counts = []
        after_counts = []

        seed_residency = ContractionState._seed_residency

        def seed_and_record(state_before, successor, delta=0, host_delta=0):
            # A zero-history, zero-peak successor is the independent replay
            # fragment constructed before each storage transition.
            if (successor.actions == () and successor.peak_bytes == 0
                    and successor.peak_host_bytes == 0 and successor.peak_device_bytes == 0):
                live_inputs = state_before.live_inputs
                successor = seed_residency(state_before, successor, delta, host_delta)
                cache_from_parent[id(successor)] = live_inputs
                return successor
            return seed_residency(state_before, successor, delta, host_delta)

        def check_before(before):
            self.assertEqual(before.actions, ())
            self.assertEqual((before.peak_bytes, before.peak_host_bytes,
                              before.peak_device_bytes), (0, 0, 0))
            self.assertIn("_resident_totals", before.__dict__)
            self.assertEqual(before._resident_totals, _resident_scan(before))
            self.assertIn(id(before), cache_from_parent)
            self.assertIs(before.__dict__.get("live_inputs"), cache_from_parent[id(before)])
            before_counts.append(len(before.actions))

        def observe_transition(original):
            def wrapped(before, *args, **kwargs):
                check_before(before)
                after = original(before, *args, **kwargs)
                after_counts.append(len(after.actions))
                self.assertLessEqual(len(after.actions), 1)
                return after
            return wrapped

        apply_selected = storage._apply_selected

        def observe_selected(before, *args, **kwargs):
            check_before(before)
            after = apply_selected(before, *args, **kwargs)
            self.assertIsNotNone(after)
            after_counts.append(len(after.actions))
            self.assertLessEqual(len(after.actions), 1)
            return after

        with (
            patch.object(ContractionState, "_seed_residency", new=seed_and_record),
            patch.object(ContractionState, "load", new=observe_transition(ContractionState.load)),
            patch.object(ContractionState, "release", new=observe_transition(ContractionState.release)),
            patch.object(storage, "_apply_selected", new=observe_selected),
        ):
            plan = storage.plan_scene_storage(
                state, blocks, preparations, profile=profile, memory_limit=1 << 20,debug=True,
            )

        self.assertGreater(len(before_counts), 1)
        self.assertEqual(before_counts, [0] * len(before_counts))
        self.assertEqual(after_counts, [1] * len(after_counts))
        self.assertIs(plan.state, state)
        self.assertEqual(state.actions, original_actions)
        self.assertEqual(
            [(event.action_index, event.kind, event.block) for event in plan.events],
            [(index, kind, key) for index, key in enumerate(specs)
             for kind in ("read", "release")],
        )
        self.assertEqual(
            [(point.action_index, point.phase) for point in plan.memory],
            [(0, "read:A"), (0, "materialize:A"),
             (1, "read:B"), (1, "materialize:B"),
             (2, "read:C"), (2, "materialize:C"),
             (3, "contract"), (4, "contract"), (5, "output_copy")],
        )
        self.assertEqual(plan.seconds, state.seconds + plan.preparation_seconds)

        loader = _ArrayLoader(arrays)
        actual = replay_scene_storage(plan, loader, measure=True)
        expected = arrays["A"] @ arrays["B"] @ arrays["C"]
        np.testing.assert_allclose(actual.values[0], expected)
        self.assertEqual(loader.reads, list(specs))
        self.assertEqual(len(actual.timings), len(state.actions))


if __name__ == "__main__":
    unittest.main()
