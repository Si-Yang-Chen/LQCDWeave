"""Saved-workload schedules execute against independent matrix references.

Candidate ordering, option validation and budget accounting live in
correlator/planning/test_workload_schedule.py.
"""

import unittest

import numpy as np

from lqcdweave.contraction.planning.api import (
    list_plan_candidates,
    plan_workload,
)
from lqcdweave.contraction.contraction_session import ContractionSession
from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    ExecutionBudget,
    TensorSpec,
    WorkloadSpec,
)
from lqcdweave.storage.contraction_plans.api import validate_bundle


class SchedulePlannerTests(unittest.TestCase):
    @staticmethod
    def _workload():
        arrays = {
            "a": np.arange(1, 7, dtype=np.float64).reshape(2, 3),
            "b": np.arange(2, 14, dtype=np.float64).reshape(3, 4),
            "c": np.arange(3, 11, dtype=np.float64).reshape(4, 2),
            "d": np.arange(4, 19, dtype=np.float64).reshape(3, 5),
            "e": np.arange(5, 15, dtype=np.float64).reshape(2, 5),
        }
        terms = (
            ContractionTerm(("a", "b"), "ij,jk->ik"),
            ContractionTerm(("b", "c"), "ij,jk->ik"),
            ContractionTerm(("a", "d"), "ij,jk->ik"),
            ContractionTerm(("c", "e"), "ij,jk->ik"),
        )
        workload = WorkloadSpec(
            specs={key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()},
            terms=terms,
            input_sources={key: {"family": key} for key in arrays},
            output_groups=((0, 1), (2,), (3,)),
        )
        return workload, arrays

    @staticmethod
    def _budget(host_bytes=1 << 20):
        return ExecutionBudget(host_bytes=host_bytes, min_reserve_bytes=0)

    def _assert_bundle_matches_reference(self, bundle, arrays):
        cfg = 2
        bound = {
            key: value + cfg * (index + 1)
            for index, (key, value) in enumerate(arrays.items())
        }

        def loader(requested_cfg, key, source):
            self.assertEqual(requested_cfg, cfg)
            self.assertEqual(source["family"], key)
            return bound[key]

        expected = (
            bound["a"] @ bound["b"],
            bound["b"] @ bound["c"],
            bound["a"] @ bound["d"],
            bound["c"] @ bound["e"],
        )
        session = ContractionSession.from_plan(
            bundle, backend="numpy", validate_bundle=validate_bundle,
        )
        actual = session.execute_cfg(cfg, loader).values
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, reference, rtol=0, atol=0)


    def test_plan_workload_and_multibatch_execution_match_independent_reference(self):
        workload, arrays = self._workload()
        budget = self._budget()
        selected = plan_workload(workload, budget)
        narrow = next(
            item for item in list_plan_candidates(workload, budget)
            if item.metadata["batch_width"] == 1
        )
        self.assertEqual(selected.metadata["batch_width"], 3)
        validate_bundle(selected)
        validate_bundle(narrow)
        for bundle in (selected, narrow):
            with self.subTest(width=bundle.metadata["batch_width"]):
                self._assert_bundle_matches_reference(bundle, arrays)

    def test_default_workload_planner_selects_narrow_feasible_groups(self):
        workload, arrays = self._workload()
        uncapped = list_plan_candidates(workload, self._budget())
        required = {
            item.metadata["batch_width"]: max(
                batch.memory_estimate["host_required_bytes"] for batch in item.batches
            )
            for item in uncapped
        }
        # This fixture's actual planner estimates are 1824 bytes at width 3,
        # 1448 at width 2, and 1104 at width 1. The fixed 1300-byte cap
        # therefore exercises the real budget filter without mocking it.
        budget_bytes = 1300
        self.assertGreater(required[3], budget_bytes)
        self.assertGreater(required[2], budget_bytes)
        self.assertLess(required[1], budget_bytes)

        limited = list_plan_candidates(workload, self._budget(budget_bytes))
        selected = plan_workload(workload, self._budget(budget_bytes))
        self.assertEqual([item.metadata["batch_width"] for item in limited], [1])
        self.assertEqual(selected.metadata["batch_width"], 1)
        self.assertEqual(
            [batch.term_indices for batch in selected.batches],
            [(0, 1), (2,), (3,)],
        )
        validate_bundle(selected)
        self._assert_bundle_matches_reference(selected, arrays)


if __name__ == "__main__":
    unittest.main()
