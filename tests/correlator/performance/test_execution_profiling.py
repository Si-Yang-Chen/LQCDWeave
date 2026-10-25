"""Module tests split from test_execution_profiling.py; shared numerical fixtures stay unchanged."""

from test_execution_profiling import (
    ExecutionMeasurement,
    FakeClock,
    PhaseCollector,
    SimpleNamespace,
    measurement_impl,
    np,
    unittest,
)


class ExecutionProfilingTests(unittest.TestCase):
    def test_canonical_collector_export_and_cpu_phase_clock(self):
        self.assertIs(measurement_impl.ExecutionMeasurement, ExecutionMeasurement)
        self.assertIs(measurement_impl.PhaseCollector, PhaseCollector)
        collector = PhaseCollector(np, clock=FakeClock(10.25, 11.75))
        value, phase = collector.measure_phase(
            "controlled", lambda: 17, nbytes=8, parent_action_index=4,
        )
        self.assertEqual(value, 17)
        self.assertEqual(phase.name, "controlled")
        self.assertEqual(phase.observed_seconds, 1.5)
        self.assertIsNone(phase.device_seconds)
        self.assertEqual(phase.bytes, 8)
        self.assertEqual(phase.parent_action_index, 4)

    def test_gpu_phase_records_the_captured_stream(self):
        class FakeStream:
            def __init__(self, ptr):
                self.ptr = ptr
                self.sync_count = 0

            def synchronize(self):
                self.sync_count += 1

        class FakeEvent:
            def __init__(self):
                self.recorded_streams = []
                self.sync_count = 0

            def record(self, stream):
                self.recorded_streams.append(stream)

            def synchronize(self):
                self.sync_count += 1

        selected_stream = FakeStream(101)

        class FakeCuda:
            def __init__(self):
                self.events = []

            def get_current_stream(self):
                return selected_stream

            def Event(self):
                event = FakeEvent()
                self.events.append(event)
                return event

            @staticmethod
            def get_elapsed_time(start, end):
                return 2.5

        backend = SimpleNamespace(__name__="cupy", cuda=FakeCuda())
        collector = PhaseCollector(backend, clock=FakeClock(2.0, 2.2))
        value, phase = collector.measure_phase("kernel", lambda: "done", device=True,
                                               parent_action_index=3)
        self.assertEqual(value, "done")
        self.assertAlmostEqual(phase.observed_seconds, 0.2)
        self.assertEqual(phase.device_seconds, 0.0025)
        self.assertEqual(phase.stream_id, "ptr:101")
        self.assertEqual(phase.timing_mode, "gpu_event_synchronized")
        self.assertEqual(len(backend.cuda.events), 2)
        self.assertTrue(all(event.recorded_streams == [selected_stream]
                            for event in backend.cuda.events))
        self.assertEqual(backend.cuda.events[1].sync_count, 1)
        self.assertEqual(selected_stream.sync_count, 0)
        ran = []
        with self.assertRaisesRegex(ValueError, "only the active GPU stream"):
            collector.measure_phase("wrong_stream", lambda: ran.append(True),
                                    device=True, stream=FakeStream(202))
        self.assertEqual(ran, [])

    def test_exception_stops_sampling_thread(self):
        measurement = ExecutionMeasurement(np, sample_interval_seconds=10.)
        original = RuntimeError("example")
        worker = None
        original_join = None
        original_sample = measurement._sample

        def failed_join():
            raise RuntimeError("join cleanup failed")

        def failed_sample(*, baseline=False):
            raise RuntimeError("sample cleanup failed")

        with self.assertRaises(RuntimeError) as raised:
            with measurement:
                worker = measurement.thread
                original_join = worker.join
                worker.join = failed_join
                measurement._sample = failed_sample
                raise original
        worker.join = original_join
        original_join()
        measurement._sample = original_sample
        self.assertIs(raised.exception, original)
        self.assertFalse(measurement.thread.is_alive())
        self.assertFalse(measurement.result['completed'])
        self.assertIs(raised.exception.execution_measurement, measurement.result)
        self.assertTrue(any("join cleanup failed" in item
                            for item in measurement.result["sample_errors"]))
        self.assertTrue(any("sample cleanup failed" in item
                            for item in measurement.result["sample_errors"]))

    def test_gpu_cleanup_sync_failure_does_not_replace_primary_error(self):
        class FakeStream:
            def __init__(self, *, fail_on_call=None):
                self.ptr = 88
                self.calls = 0
                self.fail_on_call = fail_on_call

            def synchronize(self):
                self.calls += 1
                if self.calls == self.fail_on_call:
                    raise RuntimeError("sync cleanup failed")

        class FakePool:
            @staticmethod
            def used_bytes():
                return 0

            @staticmethod
            def total_bytes():
                return 0

        class FakeDevice:
            def __init__(self, _device):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class FakeRuntime:
            is_hip = False

            @staticmethod
            def getDevice():
                return 0

            @staticmethod
            def runtimeGetVersion():
                return 1

            @staticmethod
            def driverGetVersion():
                return 1

            @staticmethod
            def getDeviceProperties(_device):
                return {"name": b"fake"}

            @staticmethod
            def memGetInfo():
                return 100, 100

        def backend_for(stream):
            cuda = SimpleNamespace(runtime=FakeRuntime(), Device=FakeDevice,
                                   get_current_stream=lambda: stream)
            return SimpleNamespace(__name__="cupy", __version__="fake", cuda=cuda,
                                   get_default_memory_pool=lambda: FakePool())

        stream = FakeStream(fail_on_call=2)
        measurement = ExecutionMeasurement(backend_for(stream))
        original = RuntimeError("primary execution failed")
        with self.assertRaises(RuntimeError) as raised:
            with measurement:
                raise original
        self.assertIs(raised.exception, original)
        self.assertFalse(measurement.thread.is_alive())
        self.assertFalse(original.execution_measurement["completed"])
        self.assertTrue(any("sync cleanup failed" in item
                            for item in original.execution_measurement["sample_errors"]))

        clean_stream = FakeStream(fail_on_call=2)
        clean_measurement = ExecutionMeasurement(backend_for(clean_stream))
        with self.assertRaisesRegex(RuntimeError, "sync cleanup failed"):
            with clean_measurement:
                pass
