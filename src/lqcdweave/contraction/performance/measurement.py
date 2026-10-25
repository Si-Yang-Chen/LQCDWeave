"""Opt-in execution measurements; sampled memory is not a certified high-water mark."""
from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
from collections.abc import Callable

from lqcdweave.contracts.measurement import PhaseTiming


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _stream_pointer(stream):
    pointer = getattr(stream, "ptr", None)
    if pointer is None:
        return None
    try:
        return int(pointer)
    except (TypeError, ValueError, OverflowError):
        return None


class PhaseCollector:
    """Record one operation phase using an explicitly selected backend.

    The caller supplies the action index that owns each phase. Action timings
    are inclusive summaries of their child phases; callers must aggregate one
    level only.
    """

    def __init__(self, backend, *, clock: Callable[[], float] = time.perf_counter):
        self.backend = backend
        self.gpu = getattr(backend, "__name__", "") == "cupy"
        self.clock = clock
        self.phases: list[PhaseTiming] = []

    def measure_phase(
        self,
        name: str,
        operation: Callable[[], object],
        *,
        nbytes: int | None = None,
        device: bool = False,
        parent_action_index: int | None = None,
        stream=None,
    ):
        """Run one operation and return ``(value, PhaseTiming)``.

        GPU boundaries use the current stream captured at phase entry. An
        explicit stream is accepted only when it identifies that same active
        stream; other streams are rejected because this method does not change
        the operation's execution context. Device phases use CUDA events on
        that stream. Other GPU phases wait on that stream before closing the
        wall interval, preserving synchronous calibration semantics.
        """
        backend = self.backend
        phase_stream = None
        stream_id = None
        timing_mode = "host_wall"
        if self.gpu:
            current_stream = backend.cuda.get_current_stream()
            if stream is not None:
                requested_pointer = _stream_pointer(stream)
                current_pointer = _stream_pointer(current_stream)
                if requested_pointer is None or current_pointer is None:
                    same_stream = stream is current_stream
                else:
                    same_stream = requested_pointer == current_pointer
                if not same_stream:
                    raise ValueError("measurement currently supports only the active GPU stream")
            phase_stream = current_stream
            pointer = _stream_pointer(phase_stream)
            stream_id = None if pointer is None else f"ptr:{pointer}"
            timing_mode = "gpu_event_synchronized" if device else "gpu_stream_synchronized_wall"
        start_event = end_event = None
        if self.gpu and device:
            start_event = backend.cuda.Event()
            end_event = backend.cuda.Event()

        started = self.clock()
        if start_event is not None:
            start_event.record(phase_stream)
        value = operation()
        if self.gpu:
            if end_event is not None:
                end_event.record(phase_stream)
                end_event.synchronize()
                device_seconds = float(backend.cuda.get_elapsed_time(start_event, end_event)) / 1e3
            else:
                phase_stream.synchronize()
                device_seconds = None
        else:
            device_seconds = None
        observed_seconds = self.clock() - started
        phase = PhaseTiming(
            name,
            observed_seconds,
            device_seconds,
            nbytes,
            parent_action_index,
            stream_id,
            timing_mode,
        )
        self.phases.append(phase)
        return value, phase


class ExecutionMeasurement(PhaseCollector):
    """Measure a caller-selected backend without importing or selecting another one.

    Device usage is device-wide and can include other processes. Pool usage only
    describes the backend's default pool. Sampling may miss short-lived peaks;
    neither value certifies compliance with a memory limit. Setup is excluded
    from elapsed_seconds. No background work survives context exit or failure.
    """

    def __init__(self, backend, *, sample_interval_seconds=0.005,
                 clock: Callable[[], float] = time.perf_counter):
        if sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        super().__init__(backend, clock=clock)
        self.interval = float(sample_interval_seconds)
        self.stop = threading.Event()
        self.thread = None
        self.result = {}

    def __enter__(self):
        backend = self.backend
        hardware = {"hostname": platform.node(), "machine": platform.machine(),
                    "processor": platform.processor(), "python": platform.python_version(),
                    "backend": backend.__name__, "backend_version": backend.__version__}
        self.process = None
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            pass
        if self.gpu:
            self.device_id = backend.cuda.runtime.getDevice()
            self.stream = backend.cuda.get_current_stream()
            self.stream.synchronize()
            runtime = backend.cuda.runtime
            hardware.update(device_id=self.device_id, is_hip=bool(runtime.is_hip),
                            runtime_version=runtime.runtimeGetVersion(), driver_version=runtime.driverGetVersion(),
                            device_properties=_json_value(runtime.getDeviceProperties(self.device_id)))
        encoded = json.dumps(hardware, sort_keys=True).encode()
        self.result = {"schema_version": 1, "hardware": hardware,
                       "hardware_fingerprint": hashlib.sha256(encoded).hexdigest(),
                       "timing_mode": "synchronous_calibration",
                       "timing_note": "GPU phase observations wait for their selected stream; they do not measure natural asynchronous overlap.",
                       "sample_interval_seconds": self.interval, "sample_count": 0,
                       "sample_errors": [], "baseline_bytes": {}, "sampled_peak_bytes": {}}
        self._sample(baseline=True)
        self.started = self.clock()
        self.thread = threading.Thread(target=self._run, name="execution-memory-sampler", daemon=True)
        self.thread.start()
        return self

    def _sample(self, *, baseline=False):
        values = {}
        if self.process is not None:
            values["host_rss"] = int(self.process.memory_info().rss)
        if self.gpu:
            pool = self.backend.get_default_memory_pool()
            values.update(pool_used=int(pool.used_bytes()), pool_reserved=int(pool.total_bytes()))
            free, total = self.backend.cuda.runtime.memGetInfo()
            values["device_used"] = int(total - free)
            values['non_pool'] = max(0,values['device_used']-values['pool_reserved'])
            values['pool_idle'] = max(0,values['pool_reserved']-values['pool_used'])
            self.result["device_total_bytes"] = int(total)
        if baseline:
            self.result["baseline_bytes"] = values.copy()
        for key, value in values.items():
            peaks = self.result["sampled_peak_bytes"]
            peaks[key] = max(peaks.get(key, 0), value)
        self.result["sample_count"] += 1

    def _run(self):
        try:
            if self.gpu:
                with self.backend.cuda.Device(self.device_id):
                    self._loop()
            else:
                self._loop()
        except Exception as exc:
            self.result["sample_errors"].append(repr(exc))

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __exit__(self, exc_type, exc, tb):
        cleanup_errors = []

        def record_cleanup_error(label, error):
            self.result.setdefault("sample_errors", []).append(f"{label}: {error!r}")
            if exc_type is None:
                cleanup_errors.append(error)

        if self.gpu:
            try:
                self.stream.synchronize()
            except Exception as error:
                record_cleanup_error("final stream synchronize", error)
        try:
            self.stop.set()
        except Exception as error:
            record_cleanup_error("stop sampler", error)
        try:
            if self.thread is not None:
                self.thread.join()
        except Exception as error:
            record_cleanup_error("join sampler", error)
        try:
            self.result["elapsed_seconds"] = self.clock() - self.started
        except Exception as error:
            record_cleanup_error("read elapsed clock", error)
        try:
            self._sample()
        except Exception as error:
            # Final memory sampling has always been best-effort.
            self.result.setdefault("sample_errors", []).append(f"final memory sample: {error!r}")
        self.result["completed"] = exc_type is None and not cleanup_errors
        if exc_type is not None:
            self.result["completed_phases"] = [
                {
                    "name": phase.name,
                    "observed_seconds": phase.observed_seconds,
                    "device_seconds": phase.device_seconds,
                    "bytes": phase.bytes,
                    "parent_action_index": phase.parent_action_index,
                    "stream_id": phase.stream_id,
                    "timing_mode": phase.timing_mode,
                }
                for phase in self.phases
            ]
            try:
                setattr(exc, "execution_measurement", self.result)
            except Exception:
                # Preserve the executor's original exception for immutable exception types.
                pass
        if exc_type is None and cleanup_errors:
            raise cleanup_errors[0]
        return False


__all__ = ["ExecutionMeasurement", "PhaseCollector"]
