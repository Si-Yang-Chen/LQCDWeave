import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import lqcdweave.contraction.performance.measurement as measurement_impl
from lqcdweave.contracts.correlator.api import ContractionProblem, ContractionTerm, SourceSpec, TensorSpec
from lqcdweave.contraction.performance.api import CostProfile, ExecutionMeasurement, PhaseCollector
from lqcdweave.contraction.planning.api import ContractionState
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contracts.measurement import ActionTiming, PhaseTiming, ReplayReport
from lqcdweave.runtime.measurement import load_measurement, save_measurement, summarize_measurement


class FakeClock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)




class ExecutionProfilingTests(unittest.TestCase):
    def test_unmeasured_replay_does_not_query_or_synchronize_gpu(self):
        p = ContractionProblem({'a': TensorSpec((3,), 'float64')},
                               (ContractionTerm(('a',), 'i->'),), {'a': SourceSpec('a', 24)})
        profile = CostProfile()
        state = ContractionState.initial(p).load('a', profile, 10000).contract(0, (0,), profile, 10000)

        class FakeCuda:
            def __init__(self):
                self.stream_queries = 0
                self.event_creations = 0
                self.synchronizations = 0

            def get_current_stream(self):
                self.stream_queries += 1
                return self

            def Event(self):
                self.event_creations += 1
                raise AssertionError("measurement-disabled replay created a CUDA event")

            def synchronize(self):
                self.synchronizations += 1

        cuda = FakeCuda()
        backend = SimpleNamespace(__name__="cupy", ndarray=np.ndarray, cuda=cuda,
                                  array=np.array, einsum=np.einsum)
        with patch("lqcdweave.contraction.execution.scene_runtime.ExecutionMeasurement",
                   side_effect=AssertionError("measurement-disabled replay started a collector")), \
             patch("lqcdweave.contraction.performance.measurement.time.perf_counter",
                   side_effect=AssertionError("measurement-disabled replay called the clock")):
            result = replay(state, lambda key, spec: np.arange(3.), backend=backend, measure=False)
        self.assertEqual(float(result[0]), 3.0)
        self.assertEqual((cuda.stream_queries, cuda.event_creations, cuda.synchronizations), (0, 0, 0))

    def test_switch_preserves_values_and_disabled_does_not_start_collection(self):
        p = ContractionProblem({'a': TensorSpec((3,), 'float64')},
                               (ContractionTerm(('a',), 'i->'),), {'a': SourceSpec('a', 24)})
        profile = CostProfile()
        s = ContractionState.initial(p).load('a', profile, 10000).contract(0, (0,), profile, 10000)
        with patch('lqcdweave.contraction.execution.scene_runtime.ExecutionMeasurement', side_effect=AssertionError):
            plain = replay(s, lambda k, spec: np.arange(3.))
        report = replay(s, lambda k, spec: np.arange(3.), measure=True)
        np.testing.assert_array_equal(plain, report.values)
        self.assertTrue(report.execution_profile['completed'])
        self.assertGreaterEqual(report.execution_profile['sample_count'], 2)
        self.assertNotIn('device_used', report.execution_profile['sampled_peak_bytes'])
        self.assertEqual(report.execution_profile['timing_mode'], 'synchronous_calibration')
        for timing in (*report.timings, *report.output_timings):
            self.assertTrue(timing.phases)
            self.assertTrue(all(phase.parent_action_index == timing.action_index
                                for phase in timing.phases))
        summary = summarize_measurement(report)
        self.assertAlmostEqual(sum(row['host_seconds'] for row in summary.values()),
                               sum(t.observed_seconds for t in (*report.timings, *report.output_timings)))
        json.dumps(report.execution_profile)
        with TemporaryDirectory() as directory:
            path = save_measurement(report, Path(directory) / "measurement.json")
            restored = load_measurement(path)
            self.assertEqual(restored.values, ())
            self.assertEqual(restored.timings, report.timings)
            self.assertEqual(restored.output_timings, report.output_timings)

    def test_failed_replay_attaches_completed_phases_without_replacing_error(self):
        p = ContractionProblem({'a': TensorSpec((3,), 'float64')},
                               (ContractionTerm(('a',), 'i->'),), {'a': SourceSpec('a', 24)})
        profile = CostProfile()
        state = ContractionState.initial(p).load('a', profile, 10000).contract(0, (0,), profile, 10000)
        error = RuntimeError("kernel failed")
        with patch("numpy.einsum", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                replay(state, lambda key, spec: np.arange(3.), measure=True)
        self.assertIs(caught.exception, error)
        self.assertFalse(error.execution_measurement["completed"])
        phases = error.execution_measurement["completed_phases"]
        self.assertEqual([phase["name"] for phase in phases], ["source_get", "snapshot"])
        self.assertTrue(all(phase["parent_action_index"] == 0 for phase in phases))


if __name__ == '__main__':
    unittest.main()
