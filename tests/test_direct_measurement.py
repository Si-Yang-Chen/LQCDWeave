"""Direct-executor measurement preserves its legacy report and cleanup contract."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import ContractionProblem, ContractionTerm, TensorSpec
from lqcdweave.contraction.planning.api import find_contraction_path
from lqcdweave.contraction.execution.api import execute_direct_contraction
from lqcdweave.contracts.measurement import ActionTiming
from lqcdweave.runtime.measurement import load_measurement, save_measurement
from test_direct_contraction import _MemoryLoader, _make_expanded, _sources_for


class _HookLoader(_MemoryLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement_modes = []

    def set_measurement(self, enabled):
        self.measurement_modes.append(bool(enabled))


class _CloseFailLoader(_HookLoader):
    def close_block(self, key, value):
        super().close_block(key, value)
        raise RuntimeError(f"close failed for {key}")


class DirectMeasurementTests(unittest.TestCase):
    def case(self, *, loader=None, backend="numpy"):
        arrays = {"A": np.arange(3., dtype=np.float64),
                  "B": np.arange(3., dtype=np.float64) + 1.}
        specs = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        problem = ContractionProblem(
            specs,
            (ContractionTerm(("A", "B"), "i,i->"),),
            _sources_for(specs),
            backend=backend,
        )
        loader = _MemoryLoader({"A": arrays["A"], "B": arrays["B"]}) if loader is None else loader
        expanded = _make_expanded(problem, ("T",), arrays, loader=loader)
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="none")
        return arrays, loader, expanded, path

    def test_measured_direct_keeps_legacy_fields_and_adds_typed_operation_phases(self):
        arrays = {"A": np.arange(3., dtype=np.float64),
                  "B": np.arange(3., dtype=np.float64) + 1.}
        loader = _HookLoader(arrays)
        _arrays, loader, expanded, path = self.case(loader=loader)
        result = execute_direct_contraction(expanded, path, measure=True)
        self.assertEqual(len(result.term_values), 1)
        self.assertEqual(float(result.term_values[0].value), 8.0)
        report = result.report
        self.assertEqual(report["execution_mode"], "direct")
        self.assertEqual(report["backend"], "numpy")
        self.assertEqual(report["term_count"], 1)
        self.assertEqual(report["coordinate_count"], 1)
        self.assertIsNone(report["predicted_seconds"])
        self.assertGreaterEqual(report["seconds"], 0.)
        self.assertTrue(report["measurement_report"].execution_profile["completed"])
        self.assertEqual(report["measurement_report"].execution_profile["timing_mode"],
                         "synchronous_calibration")
        self.assertFalse(report["calibration_eligible"])

        self.assertIsInstance(report["measurement_report"].timings[0], ActionTiming)
        load_phases = [phase.name for timing in report["measurement_report"].timings if timing.kind == "load"
                       for phase in timing.phases]
        self.assertEqual(load_phases, ["read_block", "materialize", "read_block", "materialize"])
        contract = next(timing for timing in report["measurement_report"].timings if timing.kind == "contract")
        self.assertEqual([phase.name for phase in contract.phases], ["contract_kernel"])
        output = report["measurement_report"].output_timings[0]
        self.assertEqual([phase.name for phase in output.phases], ["output_copy"])
        cleanup = next(timing for timing in report["measurement_report"].timings
                       if timing.kind == "release")
        self.assertEqual([phase.name for phase in cleanup.phases], ["close_block", "close_block"])
        self.assertTrue(all(
            phase.parent_action_index == timing.action_index
            for timing in (*report["measurement_report"].timings,
                           *report["measurement_report"].output_timings)
            for phase in timing.phases
        ))
        self.assertTrue(all(phase.parent_action_index == cleanup.action_index
                            for phase in cleanup.phases))
        self.assertEqual(report["measurement_report"].values, (result.term_values[0].value,))
        self.assertEqual(loader.closed, ["B", "A"])
        self.assertEqual(loader.measurement_modes, [True, False])
        with TemporaryDirectory() as directory:
            saved = save_measurement(report["measurement_report"], Path(directory) / "direct.json")
            restored = load_measurement(saved)
        self.assertEqual(restored.timings, report["measurement_report"].timings)
        self.assertEqual(restored.output_timings, report["measurement_report"].output_timings)

    def test_measure_false_does_not_construct_collector_or_call_clock(self):
        arrays = {"A": np.arange(3., dtype=np.float64),
                  "B": np.arange(3., dtype=np.float64) + 1.}
        loader = _HookLoader(arrays)
        _arrays, loader, expanded, path = self.case(loader=loader)
        with patch("lqcdweave.contraction.execution.direct.ExecutionMeasurement",
                   side_effect=AssertionError("collector constructed with measure=False")), \
             patch("lqcdweave.contraction.execution.direct.perf_counter",
                   side_effect=AssertionError("timer called with measure=False")), \
             patch("lqcdweave.contraction.performance.measurement.time.perf_counter",
                   side_effect=AssertionError("measurement clock called with measure=False")):
            result = execute_direct_contraction(expanded, path, measure=False)
        self.assertIsNone(result.report)
        self.assertEqual(float(result.term_values[0].value), 8.0)
        self.assertEqual(loader.measurement_modes, [])

    def test_unmeasured_fake_gpu_keeps_resource_fence_without_timing_events(self):
        class FakeStream:
            def __init__(self):
                self.synchronizations = 0

            def synchronize(self):
                self.synchronizations += 1

        class FakeCuda:
            def __init__(self):
                self.stream = FakeStream()
                self.events = 0

            def get_current_stream(self):
                return self.stream

            def Event(self):
                self.events += 1
                raise AssertionError("measure=False must not create a CUDA event")

        cuda = FakeCuda()
        backend = SimpleNamespace(
            __name__="cupy", ndarray=np.ndarray, array=np.array, asarray=np.asarray, einsum=np.einsum,
            take=np.take, cuda=cuda,
        )
        _arrays, _loader, expanded, path = self.case(backend="cupy")
        with patch("lqcdweave.contraction.execution.direct.ExecutionMeasurement",
                   side_effect=AssertionError("collector constructed with measure=False")), \
             patch("lqcdweave.contraction.execution.direct.perf_counter",
                   side_effect=AssertionError("timer called with measure=False")), \
             patch("lqcdweave.contraction.performance.measurement.time.perf_counter",
                   side_effect=AssertionError("measurement clock called with measure=False")):
            result = execute_direct_contraction(expanded, path, backend=backend, measure=False)
        self.assertEqual(float(result.term_values[0].value), 8.0)
        self.assertEqual(cuda.stream.synchronizations, 1)
        self.assertEqual(cuda.events, 0)

    def test_primary_materialization_error_keeps_completed_direct_phases(self):
        loader = _HookLoader({"A": np.arange(3.), "B": np.arange(3.) + 1.},
                             fail_materialize="B")
        _arrays, loader, expanded, path = self.case(loader=loader)
        with self.assertRaises(FileNotFoundError) as caught:
            execute_direct_contraction(expanded, path, measure=True)
        self.assertIn("missing materialized data for B", str(caught.exception))
        self.assertEqual(loader.closed, ["B", "A"])
        details = caught.exception.execution_measurement
        self.assertFalse(details["completed"])
        phases = details["completed_phases"]
        self.assertIn("read_block", [phase["name"] for phase in phases])
        self.assertIn("materialize", [phase["name"] for phase in phases])
        self.assertNotIn("contract_kernel", [phase["name"] for phase in phases])
        self.assertEqual(loader.measurement_modes, [True, False])

    def test_cleanup_failures_preserve_primary_and_propagate_without_one(self):
        payloads = {"A": np.arange(3.), "B": np.arange(3.) + 1.}
        primary_loader = _CloseFailLoader(payloads, fail_materialize="B")
        _arrays, primary_loader, expanded, path = self.case(loader=primary_loader)
        with self.assertRaises(FileNotFoundError) as caught:
            execute_direct_contraction(expanded, path, measure=True)
        self.assertIn("missing materialized data for B", str(caught.exception))
        self.assertTrue(any("close failed" in item
                            for item in caught.exception.execution_measurement["sample_errors"]))
        self.assertEqual(primary_loader.measurement_modes, [True, False])

        clean_loader = _CloseFailLoader(payloads)
        _arrays, _loader, expanded, path = self.case(loader=clean_loader)
        with self.assertRaisesRegex(RuntimeError, "close failed for B"):
            execute_direct_contraction(expanded, path, measure=True)
        self.assertEqual(clean_loader.measurement_modes, [True, False])

    def test_sparse_join_direct_reuses_shared_operation_and_marks_unmeasured_cost(self):
        from test_sparse_cycle_partitions import SparseCyclePartitionTests, _complex, _dense_oracle

        partition = (0, 1, 2, 0)
        _state, _profile, _first, _second = SparseCyclePartitionTests()._planned_cycle(partition)
        problem = _state.problem
        rng = np.random.default_rng(9274)
        n = 3
        left = _complex(rng, (4, 4, n, 3, n, 3))
        right = _complex(rng, (4, 4, n, 3, n, 3))
        gamma0, gamma1 = _complex(rng, (4, 4)), _complex(rng, (4, 4))
        indices0 = np.asarray(((0, 1), (0, 1), (1, 0), (2, 2)), dtype=np.int64)
        values0 = _complex(rng, (len(indices0), 3, 3))
        indices1 = np.asarray(((0, 1), (2, 0), (2, 0), (1, 1)), dtype=np.int64)
        values1 = _complex(rng, (len(indices1), 3, 3))
        record0 = {"type": "sparse", "indices": indices0, "values": values0}
        record1 = {"type": "sparse", "indices": indices1, "values": values1}
        expected, _network = _dense_oracle(
            partition, left, right, gamma0, gamma1, record0, record1, n,
        )
        payloads = {
            "p0": left, "p1": right,
            "r0": ((gamma0, 1., record0),),
            "r1": ((gamma1, 1., record1),),
        }
        loader = _MemoryLoader(payloads)
        expanded = _make_expanded(problem, ("T",), payloads, loader=loader)
        path = find_contraction_path(expanded, memory_limit=1 << 30, optimize="none")
        result = execute_direct_contraction(expanded, path, measure=True)
        np.testing.assert_allclose(result.term_values[0].value, expected, rtol=1e-11, atol=1e-10)
        sparse_action = next(timing for timing in result.report["measurement_report"].timings
                             if timing.kind == "sparse_bridge")
        self.assertEqual(sparse_action.model_features["sparse_operation"], "bounded_coo_join")
        self.assertEqual([phase.name for phase in sparse_action.phases], ["sparse_bridge_kernel"])


if __name__ == "__main__":
    unittest.main()
