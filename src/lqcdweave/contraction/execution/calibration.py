"""Small, synthetic compute probes for an explicitly supplied backend."""

from __future__ import annotations

from math import isfinite
from time import perf_counter

import numpy as np

from lqcdweave.contracts.measurement import (
    ActionTiming,
    PhaseTiming,
    ReplayReport,
)
from .operations.contract import _contract_value


_SIZES = (8, 16, 32, 64, 128)
_MAX_DTYPES = 4
_IMPLEMENTATION_VERSION = "unified-device-v1"
_EQUATION = "ik,kj->ij"


def _dtype_names(dtypes):
    if isinstance(dtypes, str):
        values = (dtypes,)
    else:
        try:
            values = tuple(dtypes)
        except TypeError as exc:
            raise TypeError("dtypes must be a dtype name or iterable of dtype names") from exc
    names = []
    for value in values:
        try:
            dtype = np.dtype(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported calibration dtype: {value!r}") from exc
        if dtype.name not in {"float32", "float64", "complex64", "complex128"}:
            raise ValueError("calibration supports float32/64 and complex64/128")
        if dtype.name not in names:
            names.append(dtype.name)
    if not names:
        raise ValueError("at least one calibration dtype is required")
    if len(names) > _MAX_DTYPES:
        raise ValueError(f"at most {_MAX_DTYPES} calibration dtypes may be requested")
    return tuple(np.dtype(name) for name in names)


def _empty_report(timing_mode):
    return ReplayReport(
        values=(),
        timings=(),
        peak_device_bytes=None,
        peak_host_rss_bytes=None,
        execution_profile={
            "timing_mode": timing_mode,
            "covered_operations": (),
            "implementation_version": _IMPLEMENTATION_VERSION,
            "verified": True,
            "calibration_eligible": True,
        },
    )


def _estimated_array_bytes(n, dtype, *, device):
    # Setup verification keeps host inputs/reference/result alongside the
    # backend inputs and output. CPU inputs are shared with the host copies.
    array_count = 20 if device else 16
    return array_count * n * n * dtype.itemsize


def _synthetic_inputs(n, dtype):
    count = n * n
    left = np.fromiter(
        ((complex((index * 7 % 37) / 37, (index * 11 % 31) / 31) if dtype.kind == 'c'
          else (index * 7 % 37) / 37)
         for index in range(count)),
        dtype=dtype,
        count=count,
    ).reshape(n, n)
    right = np.fromiter(
        ((complex((index * 13 % 41) / 41, (index * 5 % 29) / 29) if dtype.kind == 'c'
          else (index * 13 % 41) / 41)
         for index in range(count)),
        dtype=dtype,
        count=count,
    ).reshape(n, n)
    return left, right


def _synchronize(backend):
    backend.cuda.get_current_stream().synchronize()


def _device_timed(backend, operation):
    """Time one operation on the caller's current device stream."""
    stream = backend.cuda.get_current_stream()
    start = backend.cuda.Event()
    end = backend.cuda.Event()
    start.record(stream)
    value = operation()
    end.record(stream)
    end.synchronize()
    seconds = float(backend.cuda.get_elapsed_time(start, end)) * 1e-3
    return value, seconds


def _numpy_tolerance(dtype):
    return 5e-5 if dtype.name in ("complex64", "float32") else 1e-10


def probe_compute_profile(
    backend,
    *,
    dtypes=("complex128",),
    max_seconds=10.0,
    max_bytes=64 << 20,
    clock=perf_counter,
):
    """Measure synthetic dense binary contraction and snapshot costs.

    The supplied backend is the only backend used. NumPy is used separately
    for reference results; no input files, device discovery, or CuPy imports
    are performed here. GPU timing records only events on the current stream,
    so input transfers and result readback stay outside each timed interval.
    """
    if backend is None:
        raise TypeError("backend must be supplied explicitly")
    device = backend is not np
    if device and not all(
        callable(getattr(getattr(backend, "cuda", None), name, None))
        for name in ("Event", "get_current_stream", "get_elapsed_time")
    ):
        raise TypeError("non-NumPy backend must provide the explicit CUDA event interface")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
        raise TypeError("max_seconds must be a finite nonnegative number")
    if not isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds must be a finite nonnegative number")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be a nonnegative integer")
    if max_bytes < 0:
        raise ValueError("max_bytes must be a nonnegative integer")
    if not callable(clock):
        raise TypeError("clock must be callable")

    dtypes = _dtype_names(dtypes)
    timing_mode = "same_stream_event" if device else "perf_counter"
    candidates = tuple(
        (n, dtype)
        for n in _SIZES
        for dtype in dtypes
        if _estimated_array_bytes(n, dtype, device=device) <= max_bytes
    )
    if not candidates or max_seconds == 0:
        return _empty_report(timing_mode)

    started = clock()
    if not isinstance(started, (int, float)) or not isfinite(started):
        raise ValueError("clock must return finite seconds")
    deadline = started + float(max_seconds)
    timings = []

    for n, dtype in candidates:
        if clock() >= deadline:
            break
        host_left, host_right = _synthetic_inputs(n, dtype)
        if device:
            left = backend.asarray(host_left)
            right = backend.asarray(host_right)
        else:
            left, right = host_left, host_right

        # Warm the selected route, then compare one independently calculated
        # NumPy reference for this size/dtype before collecting any sample.
        warm_result, _ = _contract_value(
            _EQUATION, (left, right), backend, device,
            _device_scope=False,
        )
        if device:
            _synchronize(backend)
            actual = backend.asnumpy(warm_result)
        else:
            actual = warm_result
        expected = np.einsum(_EQUATION, host_left, host_right, optimize=False)
        tolerance = _numpy_tolerance(dtype)
        if not np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
            raise ValueError(
                f"{dtype.name} contraction failed the independent NumPy reference at n={n}"
            )

        warm_snapshot = left.copy()
        if device:
            _synchronize(backend)
            del host_left, host_right, expected, actual, warm_result
        else:
            del expected, actual, warm_result
        del warm_snapshot

        # Use the injected clock for the budget check and CPU measurement.
        # An operation submitted before its deadline is allowed to finish.
        contract_start = clock()
        if contract_start >= deadline:
            break
        operation = lambda: _contract_value(
            _EQUATION, (left, right), backend, device,
            _device_scope=False,
        )[0]
        if device:
            output, observed = _device_timed(backend, operation)
        else:
            output = operation()
            observed = float(clock() - contract_start)
        timings.append(ActionTiming(
            kind="contract",
            estimated_seconds=0.0,
            observed_seconds=observed,
            model_features={
                "flops": n ** 3,
                "traffic_bytes": 3 * n * n * dtype.itemsize,
                "kernel_calls": 1,
                "dtype": dtype.name,
            },
        ))
        del output

        snapshot_start = clock()
        if snapshot_start >= deadline:
            break
        if device:
            snapshot, snapshot_seconds = _device_timed(backend, left.copy)
        else:
            snapshot = left.copy()
            snapshot_seconds = float(clock() - snapshot_start)
        byte_count = n * n * dtype.itemsize
        phase = PhaseTiming(
            name="snapshot",
            observed_seconds=snapshot_seconds,
            device_seconds=snapshot_seconds if device else None,
            bytes=byte_count,
            timing_mode=timing_mode,
        )
        timings.append(ActionTiming(
            kind="load",
            estimated_seconds=0.0,
            observed_seconds=snapshot_seconds,
            phases=(phase,),
        ))
        del snapshot
        del left, right
        if not device:
            del host_left, host_right

    return ReplayReport(
        values=(),
        timings=tuple(timings),
        peak_device_bytes=None,
        peak_host_rss_bytes=None,
        execution_profile={
            "timing_mode": timing_mode,
            "covered_operations": tuple(name for name, kind in (("contract", "contract"), ("snapshot", "load"))
                                        if any(row.kind == kind for row in timings)),
            "covered_dtypes": tuple(sorted({row.model_features['dtype'] for row in timings
                                            if row.kind == 'contract'})),
            "implementation_version": _IMPLEMENTATION_VERSION,
            "verified": True,
            "calibration_eligible": True,
        },
    )


__all__ = ["probe_compute_profile"]
