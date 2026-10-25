import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.calibration import probe_compute_profile
from lqcdweave.contracts.measurement import ActionTiming


class StepClock:
    def __init__(self, step):
        self.value = 0.0
        self.step = step

    def __call__(self):
        result = self.value
        self.value += self.step
        return result


class ComputeCalibrationProbeTests(unittest.TestCase):
    def test_numpy_probe_reports_verified_compute_and_snapshot_samples(self):
        report = probe_compute_profile(
            np,
            dtypes=("complex64", "complex128"),
            max_seconds=10.0,
            max_bytes=64 << 20,
            clock=StepClock(1e-6),
        )

        self.assertEqual(len(report.timings), 20)
        self.assertEqual(report.execution_profile["timing_mode"], "perf_counter")
        self.assertEqual(report.execution_profile["covered_operations"],
                         ("contract", "snapshot"))
        self.assertEqual(report.execution_profile["implementation_version"],
                         "unified-device-v1")
        self.assertIs(report.execution_profile["verified"], True)
        self.assertTrue(all(isinstance(row, ActionTiming) for row in report.timings))

        contractions = [row for row in report.timings if row.kind == "contract"]
        snapshots = [row for row in report.timings if row.kind == "load"]
        self.assertEqual(len(contractions), 10)
        self.assertEqual(len(snapshots), 10)
        first = contractions[0]
        self.assertEqual(first.model_features, {
            "flops": 8 ** 3,
            "traffic_bytes": 3 * 8 * 8 * np.dtype("complex64").itemsize,
            "kernel_calls": 1,
            "dtype": "complex64",
        })
        self.assertEqual(snapshots[0].phases[0].name, "snapshot")
        self.assertEqual(snapshots[0].phases[0].bytes,
                         8 * 8 * np.dtype("complex64").itemsize)

    def test_budget_stops_new_samples_but_keeps_submitted_contract(self):
        report = probe_compute_profile(
            np,
            dtypes=("complex64",),
            max_seconds=0.0025,
            max_bytes=1 << 20,
            clock=StepClock(0.001),
        )

        self.assertEqual(len(report.timings), 1)
        self.assertEqual(report.timings[0].kind, "contract")

    def test_insufficient_array_budget_returns_empty_report(self):
        with patch(
            "lqcdweave.contraction.execution.calibration._contract_value",
            side_effect=AssertionError("probe should not allocate or submit a kernel"),
        ):
            report = probe_compute_profile(
                np, dtypes=("complex128",), max_bytes=1, clock=StepClock(1e-6)
            )

        self.assertEqual(report.timings, ())
        self.assertIs(report.execution_profile["verified"], True)


if __name__ == "__main__":
    unittest.main()
