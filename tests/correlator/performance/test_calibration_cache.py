from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import ExecutionTarget
from lqcdweave.contracts.measurement import ReplayReport
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contraction.execution.calibration import probe_compute_profile
from lqcdweave.contraction.performance.calibration_cache import (
    IMPLEMENTATION_VERSION,
    calibrate_target_profile,
    load_target_calibration,
)
from lqcdweave.contraction.performance.profile_io import load_cost_profile_document


class StepClock:
    def __init__(self, step=1e-6):
        self.value = 0.0
        self.step = step

    def __call__(self):
        result = self.value
        self.value += self.step
        return result


def _target(fingerprint="cpu-fingerprint"):
    return ExecutionTarget(
        name="cpu-cache-test",
        backend="numpy",
        runtime="cpu",
        memory=TargetHardwareMemory(host_available_bytes=1 << 20),
        hardware_fingerprint=fingerprint,
    )


class TargetCalibrationCacheTests(unittest.TestCase):
    def setUp(self):
        self.report = probe_compute_profile(
            np,
            dtypes=("complex64",),
            max_seconds=10.0,
            max_bytes=64 << 20,
            clock=StepClock(),
        )
        self.document = load_cost_profile_document()

    def test_cache_hit_requires_same_key_and_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            fitted = calibrate_target_profile(
                _target(), self.document, self.report,
                cache_key="adapter-hardware-software-v1", directory=directory,
            )
            loaded = load_target_calibration(
                _target(), cache_key="adapter-hardware-software-v1",
                directory=directory,
            )

            self.assertTrue(fitted["calibration"]["calibrated"])
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["profile"], fitted["profile"])
            self.assertEqual(loaded["provenance"]["implementation_version"],
                             IMPLEMENTATION_VERSION)
            self.assertEqual(loaded["provenance"]["timing_mode"], "perf_counter")
            self.assertEqual(loaded["provenance"]["covered_operations"],
                             ["contract", "snapshot"])
            self.assertIsNone(load_target_calibration(
                _target(), cache_key="different-adapter-key", directory=directory
            ))

            cache_path = Path(fitted["calibration"]["cache_path"])
            cache_document = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_document["provenance"]["implementation_version"] = "older-probe"
            cache_path.write_text(json.dumps(cache_document), encoding="utf-8")
            self.assertIsNone(load_target_calibration(
                _target(), cache_key="adapter-hardware-software-v1",
                directory=directory,
            ))

    def test_unverified_or_empty_report_preserves_profile_and_lists_missing_items(self):
        unverified = replace(
            self.report,
            execution_profile={**self.report.execution_profile, "verified": False},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = calibrate_target_profile(
                _target(), self.document, unverified,
                cache_key="unverified-key", directory=directory,
            )
            self.assertFalse(result["calibration"]["calibrated"])
            self.assertEqual(result["profile"], self.document["profile"])
            self.assertIn("report verified=True", result["calibration"]["missing"])
            self.assertIsNone(load_target_calibration(
                _target(), cache_key="unverified-key", directory=directory
            ))

            empty = ReplayReport(
                (), (), None, None,
                execution_profile=self.report.execution_profile,
            )
            empty_result = calibrate_target_profile(
                _target(), self.document, empty,
                cache_key="empty-key", directory=directory,
            )
            self.assertFalse(empty_result["calibration"]["calibrated"])
            self.assertIn("nonempty contract/snapshot samples",
                          empty_result["calibration"]["missing"])


if __name__ == "__main__":
    unittest.main()
