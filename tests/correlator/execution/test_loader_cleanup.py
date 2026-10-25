"""Public storage replay cleanup semantics for resource-owning loaders."""

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
from lqcdweave.contraction.execution.api import (
    replay_scene_storage,
    replay_storage_batches,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState,
    logical_cost_profile,
    logical_source_problem,
    plan_scene_storage,
    join_storage_batches,
)


def dot_product_plan():
    problem = ContractionProblem(
        {
            "left": TensorSpec((4,), "float64"),
            "right": TensorSpec((4,), "float64"),
        },
        (ContractionTerm(("left", "right"), "i,i->"),),
        {
            "left": SourceSpec("left-region", 32, read_bytes=32),
            "right": SourceSpec("right-region", 32, read_bytes=32),
        },
    )
    profile = CostProfile()
    model = logical_cost_profile(profile)
    state = ContractionState.initial(logical_source_problem(problem))
    state = state.load("left", model, 4096).load("right", model, 4096)
    state = state.contract(0, (0, 1), model, 4096).release("left").release("right")
    return plan_scene_storage(
        state,
        {
            "left-block": StorageBlock("left-file", "all", 32, host_bytes=32),
            "right-block": StorageBlock("right-file", "all", 32, host_bytes=32),
        },
        {
            # Both physical blocks remain live through the second materialize.
            "left": SourcePreparation(("left-block", "right-block")),
            "right": SourcePreparation(("left-block", "right-block")),
        },
        profile=profile,
        memory_limit=4096,
    )


class RecordingLoader:
    def __init__(self, *, close_block_error=None, close_all_error=None):
        self.events = []
        self.close_block_error = close_block_error
        self.close_all_error = close_all_error
        self._raised_close_block_error = False

    def read_block(self, key, block):
        self.events.append(("read", key))
        values = {
            "left-block": np.array([1.0, 2.0, 3.0, 4.0]),
            "right-block": np.array([5.0, 6.0, 7.0, 8.0]),
        }
        return values[key]

    def materialize(self, key, source, blocks):
        self.events.append(("materialize", key))
        return blocks[f"{key}-block"]

    def close_block(self, key, value):
        self.events.append(("close_block", key))
        if self.close_block_error is not None and not self._raised_close_block_error:
            self._raised_close_block_error = True
            raise self.close_block_error

    def close_all(self):
        self.events.append(("close_all",))
        if self.close_all_error is not None:
            raise self.close_all_error


class LoaderCleanupTests(unittest.TestCase):
    def setUp(self):
        self.plan = dot_product_plan()
        self.assertTrue(self.plan.feasible)
        self.assertEqual(
            [(event.kind, event.action_index) for event in self.plan.events],
            [("read", 0), ("read", 0), ("release", 1), ("release", 1)],
        )

    def test_replay_keeps_independent_dot_product_result(self):
        loader = RecordingLoader()

        result = replay_scene_storage(self.plan, loader)

        np.testing.assert_allclose(result, (70.0,))
        self.assertEqual(
            [event for event in loader.events if event[0] == "read"],
            [("read", "left-block"), ("read", "right-block")],
        )
        self.assertEqual(
            [event for event in loader.events if event[0] == "close_block"],
            [("close_block", "left-block"), ("close_block", "right-block")],
        )
        self.assertIn(("close_all",), loader.events)

    def test_first_close_block_error_still_attempts_close_all(self):
        close_block_error = RuntimeError("first close_block failed")
        close_all_error = RuntimeError("close_all also failed")
        loader = RecordingLoader(
            close_block_error=close_block_error,
            close_all_error=close_all_error,
        )

        with self.assertRaisesRegex(RuntimeError, "first close_block failed") as caught:
            replay_scene_storage(self.plan, loader)

        self.assertIs(caught.exception, close_block_error)
        self.assertIn(("close_block", "left-block"), loader.events)
        self.assertIn(("close_all",), loader.events)

    def test_close_all_error_propagates_when_execution_succeeds(self):
        close_all_error = RuntimeError("close_all failed")
        loader = RecordingLoader(close_all_error=close_all_error)

        with self.assertRaisesRegex(RuntimeError, "close_all failed") as caught:
            replay_scene_storage(self.plan, loader)

        self.assertIs(caught.exception, close_all_error)
        self.assertIn(("close_all",), loader.events)

    def test_primary_kernel_error_wins_over_cleanup_errors(self):
        primary_error = RuntimeError("primary kernel failed")
        loader = RecordingLoader(
            close_block_error=RuntimeError("cleanup close_block failed"),
            close_all_error=RuntimeError("cleanup close_all failed"),
        )
        original_einsum = np.einsum
        calls = 0

        def fail_second_kernel(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise primary_error
            return original_einsum(*args, **kwargs)

        with patch("numpy.einsum", side_effect=fail_second_kernel),patch.object(np,'matmul',
                side_effect=lambda a,b:fail_second_kernel(a,[0,1],b,[1,2],[0,2])):
            with self.assertRaisesRegex(RuntimeError, "primary kernel failed") as caught:
                replay_storage_batches(
                    join_storage_batches((self.plan, self.plan, self.plan)),
                    loader,
                    lambda index, report: None,
                )

        self.assertIs(caught.exception, primary_error)
        self.assertEqual(calls, 2)
        self.assertEqual(
            [event for event in loader.events if event[0] == "read"],
            [("read", "left-block"), ("read", "right-block")],
        )
        self.assertIn(("close_block", "left-block"), loader.events)
        self.assertIn(("close_block", "right-block"), loader.events)
        self.assertIn(("close_all",), loader.events)


if __name__ == "__main__":
    unittest.main()
