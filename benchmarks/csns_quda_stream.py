"""Compare eager and mode-block VSV with the real single-rank QUDA backend.

Uses the bundled weak_field gauge and eigenvectors. This tests matching outputs
and records bounded memory observations; it does not claim native CUDA peak.
"""

import os
import sys

_mpi4 = os.environ.get(
    "CSNS_OPENMPI4_LIB",
    "/dg_hpc/LQCD/software/x86_gpu/utilities/openmpi-4.1.6-ucx-cuda12.4/lib",
)
if (os.path.isdir(_mpi4) and os.environ.get("CSNS_QUDA_STREAM_MPI4_READY") != "1"
        and not os.environ.get("LD_LIBRARY_PATH", "").startswith(_mpi4)):
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = _mpi4 + ":" + environment.get("LD_LIBRARY_PATH", "")
    environment["CSNS_QUDA_STREAM_MPI4_READY"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)

import argparse
import gc
import json
from pathlib import Path
import statistics
from time import perf_counter
import tracemalloc
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lqcdweave.adapters.legacy_fields import DenseFileHandle
from lqcdweave.adapters.quda import QudaBackend
from lqcdweave.production.perambulator.api import PerambulatorGenerator, VectorBasis


class GaugeFile:
    def __init__(self, path):
        self.path = path

    def load(self, key):
        if key != "weak_field":
            raise ValueError("unexpected configuration")
        return SimpleNamespace(file=str(self.path))


def mode_handle(fixture):
    return DenseFileHandle(str(fixture) + os.sep, ".eigenvector.input.npy",
        format="npy", shape=(8,20,4,4,4,3), dtype="<c16", Ne=20)


def memory_snapshot(cp, pool):
    free, total = cp.cuda.runtime.memGetInfo()
    return dict(device_used_bytes=int(total-free), cupy_live_bytes=int(pool.used_bytes()),
                cupy_reserved_bytes=int(pool.total_bytes()))


def run_case(cp, fixture, times, streamed, sink_block_size=5,
             sink_time_block_size=1, sink_cache_limit_bytes=0):
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    modes = mode_handle(fixture)
    vector = VectorBasis(modes, 20)
    backend = QudaBackend(measure=True)
    snapshots = {t:np.empty((8,4,4,20,20),dtype=np.complex128) for t in times}
    starting = memory_snapshot(cp, pool)
    original_allocator = cp.cuda.get_allocator()
    peak_pool = starting["cupy_live_bytes"]
    def tracked_allocator(size):
        nonlocal peak_pool
        pointer = original_allocator(size)
        peak_pool = max(peak_pool, int(pool.used_bytes()))
        return pointer
    cp.cuda.set_allocator(tracked_allocator)
    tracemalloc.start()
    generator = None
    try:
        generator = PerambulatorGenerator((4,4,4,8), GaugeFile(fixture/"weak_field.lime"),
            mass=.09253, tol=1e-9, maxiter=1000, xi_0=4.8965, nu=.86679,
            clover_coeff_t=.8549165664, clover_coeff_r=2.32582045,
            t_boundary=-1, multigrid=False, contract_prec="<c16",
            MRHS=True, rhs_batch_size=4, sink_ne_block_size=sink_block_size,
            sink_time_block_size=sink_time_block_size,
            sink_cache_limit_bytes=sink_cache_limit_bytes,
            stream_eigenvectors=streamed, backend=backend, measure=True)
        constructed = memory_snapshot(cp, pool)
        load_started = perf_counter()
        generator.load("weak_field")
        loaded = memory_snapshot(cp, pool)
        load_seconds = perf_counter() - load_started
        # Warm the solver and source/layout buffers outside measured source rows.
        generator.calc(times[0], src=vector, sink=[vector])
        backend.synchronize()
        source_rows = []
        observed = [starting, constructed, loaded, memory_snapshot(cp, pool)]
        for source_time in times:
            backend.synchronize()
            started = perf_counter()
            result = generator.calc(source_time, src=vector, sink=[vector])[0]
            backend.synchronize()
            seconds = perf_counter() - started
            values = cp.asnumpy(result.values)
            if values.shape != (8,4,4,20,20) or not np.isfinite(values).all():
                raise ValueError("invalid QUDA VSV output")
            snapshots[source_time][...] = values
            source_rows.append(dict(source_time=source_time, seconds=seconds,
                metrics=dict(generator.last_metrics)))
            observed.append(memory_snapshot(cp, pool))
        host_peak = tracemalloc.get_traced_memory()[1]
        return dict(streamed=streamed, sink_block_size=sink_block_size,
            sink_time_block_size=sink_time_block_size,
            sink_cache_limit_bytes=sink_cache_limit_bytes,
            sink_cache_used_bytes=generator.sink_cache_used_bytes,
            load_seconds=load_seconds,
            source_rows=source_rows,
            median_source_seconds=statistics.median(row["seconds"] for row in source_rows),
            vector_scratch_bytes=generator.vector_scratch_bytes,
            host_tracemalloc_peak_bytes=host_peak,
            cupy_live_peak_bytes=peak_pool,
            device_used_boundary_max_bytes=max(row["device_used_bytes"] for row in observed),
            memory_boundaries=observed,
            backend_metrics={name:getattr(backend,name) for name in
                ("solve_seconds","to_quda_seconds","from_quda_seconds",
                 "solve_calls","rhs_count","multirhs_calls")}), snapshots
    finally:
        tracemalloc.stop()
        cp.cuda.set_allocator(original_allocator)
        backend.destroy()
        del generator, backend
        gc.collect()
        pool.free_all_blocks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "tests" / "data" / "weak_field")
    parser.add_argument("--source-times", nargs="+", type=int,
                        default=list(range(8)))
    parser.add_argument("--stream-shapes", nargs="+",
                        default=["20:1:0","20:1:122880","20:1:491520"],
                        help="mode-block:time-block:cache-bytes triples")
    args = parser.parse_args()
    times = tuple(args.source_times)
    if not times or len(set(times)) != len(times) or any(t not in range(8) for t in times):
        parser.error("source times must be unique values in 0..7")
    try:
        shapes = tuple(tuple(int(part) for part in item.split(":"))
                       for item in args.stream_shapes)
    except ValueError:
        parser.error("stream shapes must be mode:time integer pairs")
    if (not shapes or len(set(shapes)) != len(shapes) or
            any(len(shape) != 3 or shape[0] not in range(1,21) or
                shape[1] not in range(1,9) or shape[2] < 0 for shape in shapes)):
        parser.error("stream shapes require unique mode 1..20, time 1..8 and cache >=0")
    fixture = args.fixture.resolve()
    for name in ("weak_field.lime", "weak_field.eigenvector.input.npy"):
        if not (fixture / name).is_file():
            raise FileNotFoundError(f"required fixture is missing: {name}")
    import cupy as cp
    import pyquda
    pyquda.init(grid_size=[1,1,1,1], backend="cupy")
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = device["name"].decode() if isinstance(device["name"], bytes) else device["name"]
    if "V100" not in name:
        raise RuntimeError(f"expected a V100, got {name}")
    eager, eager_values = run_case(cp, fixture, times, False)
    streamed_cases, numerical_errors = {}, {}
    for mode_block, time_block, cache_bytes in shapes:
        streamed, streamed_values = run_case(
            cp, fixture, times, True, mode_block, time_block, cache_bytes)
        errors = {}
        for t in times:
            diff = streamed_values[t] - eager_values[t]
            scale = np.linalg.norm(eager_values[t].ravel())
            errors[t] = dict(max_abs=float(np.max(np.abs(diff))),
                             relative_l2=float(np.linalg.norm(diff.ravel())/scale))
            np.testing.assert_allclose(streamed_values[t], eager_values[t],
                                       atol=1e-6, rtol=1e-6)
        label = f"{mode_block}x{time_block}:{cache_bytes}"
        streamed_cases[label] = streamed
        numerical_errors[label] = errors
    report = dict(gpu=name, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                  fixture="weak_field", source_times=times, eager=eager,
                  streamed_cases=streamed_cases, numerical_errors=numerical_errors,
                  measurement_scope=("host tracemalloc covers Python/NumPy allocations; "
                    "CuPy peak covers its allocator; device-used maximum is sampled "
                    "at stage/source boundaries and may miss native QUDA transients"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(gpu=name, source_times=times,
        max_abs_error=max(value["max_abs"] for rows in numerical_errors.values()
                          for value in rows.values()),
        eager_median=eager["median_source_seconds"],
        streamed_medians={key:row["median_source_seconds"]
                          for key,row in streamed_cases.items()}), sort_keys=True))


if __name__ == "__main__":
    main()
