"""Audited dense lowering avoids logical replay and preserves numerical work."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    TensorSpec,
)
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.planning import dense_batch_schedule
from lqcdweave.contraction.planning.api import CostProfile, lower_dense_batches, solve_storage
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contraction.planning.storage import plan_scene_storage


class _ArrayLoader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []

    def read_block(self, *args):
        raise AssertionError("the fixture has no physical reads")

    def materialize(self, key, *args):
        self.reads.append(key)
        return self.arrays[key]

    def close_all(self):
        pass


class DenseBatchHistoryTests(TestCase):
    def fixture(self):
        specs = {
            "a0": TensorSpec((2, 5), np.complex128),
            "a1": TensorSpec((3, 5), np.complex128),
            "b0": TensorSpec((5, 3), np.complex128),
            "b1": TensorSpec((5, 4), np.complex128),
        }
        terms = tuple(
            ContractionTerm((f"a{i}", f"b{j}"), "ij,jk->ik")
            for i in range(2)
            for j in range(2)
        )
        sources = {key: SourceSpec(key, 0, read_bytes=0) for key in specs}
        problem = ContractionProblem(specs, terms, sources)
        preparations = {key: SourcePreparation(()) for key in specs}
        profile = CostProfile()
        searched = solve_storage(
            problem,
            {},
            preparations,
            optimizer="anytime",
            memory_limit=10**8,
            deadline_seconds=0.1,
            max_states=10000,
            root_width=1,
            branch_width=1,
            component_workers=1,
        )
        plain = plan_scene_storage(
            searched.plan.state,
            searched.plan.blocks,
            searched.plan.preparations,
            profile=profile,
            memory_limit=10**8,
        )
        rng = np.random.default_rng(915)
        arrays = {
            key: (rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)).astype(spec.dtype)
            for key, spec in specs.items()
        }
        return plain, profile, arrays

    def test_transitions_keep_short_history_and_numpy_results(self):
        plain, profile, arrays = self.fixture()
        with (
            patch.object(ContractionState, 'load', side_effect=AssertionError('logical load replay')),
            patch.object(ContractionState, 'release', side_effect=AssertionError('logical release replay')),
            patch.object(dense_batch_schedule, 'describe', side_effect=AssertionError('cost repricing')),
            patch.object(dense_batch_schedule, 'replay', side_effect=AssertionError('algebra replay')),
            patch.object(dense_batch_schedule, 'plan_scene_storage', side_effect=AssertionError('storage replay')),
        ):
            lowered, report = lower_dense_batches(plain, profile=profile)

        self.assertTrue(report["applied"])
        self.assertTrue(lowered.kernel_batches)
        self.assertGreater(len(lowered.state.actions), 1)
        diagnostic, _ = lower_dense_batches(plain, profile=profile, debug=True)
        self.assertEqual(lowered.state.actions, diagnostic.state.actions)
        self.assertEqual(lowered.kernel_batches, diagnostic.kernel_batches)
        self.assertEqual(lowered.memory, diagnostic.memory)
        self.assertEqual(lowered.seconds, diagnostic.seconds)
        self.assertEqual(report['priority_updates'], 1)

        loader = _ArrayLoader(arrays)
        actual = replay_scene_storage(lowered, loader, measure=True)
        for term, value in zip(lowered.state.problem.terms, actual.values, strict=True):
            np.testing.assert_allclose(value, arrays[term.inputs[0]] @ arrays[term.inputs[1]])
        self.assertEqual(sorted(loader.reads), sorted(arrays))
        self.assertEqual(len(loader.reads), len(arrays))
