"""Module tests split from test_execution_profiling.py; shared numerical fixtures stay unchanged."""

from test_execution_profiling import (
    ActionTiming,
    Path,
    PhaseTiming,
    ReplayReport,
    TemporaryDirectory,
    json,
    load_measurement,
    save_measurement,
    unittest,
)


class ExecutionProfilingTests(unittest.TestCase):
    def test_legacy_measurement_json_loads_without_parent_links(self):
        payload = {
            "timings": [{
                "kind": "contract", "estimated_seconds": 1.0, "observed_seconds": 2.0,
                "action_index": 5, "tensor": "out", "phases": [{
                    "name": "contract_kernel", "observed_seconds": 2.0,
                    "device_seconds": None, "bytes": 64,
                }], "operands": ["left", "right"], "region": None,
                "model_features": None,
            }],
            "output_timings": [], "io_events": [], "peak_device_bytes": None,
            "peak_host_rss_bytes": None, "execution_profile": None,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "old-measurement.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = load_measurement(path)
        phase = restored.timings[0].phases[0]
        self.assertIsNone(phase.parent_action_index)
        self.assertIsNone(phase.stream_id)
        self.assertIsNone(phase.timing_mode)
        self.assertEqual(restored.timings[0].operands, ("left", "right"))

    def test_new_stream_provenance_roundtrips(self):
        phase = PhaseTiming("kernel", .25, .2, 64, 7, "ptr:101", "gpu_event_synchronized")
        report = ReplayReport((), (ActionTiming(
            "contract", .2, .25, action_index=7, phases=(phase,), operands=("left", "right"),
        ),), None, None)
        with TemporaryDirectory() as directory:
            restored = load_measurement(save_measurement(report, Path(directory) / "stream.json"))
        self.assertEqual(restored.timings[0].phases[0], phase)
