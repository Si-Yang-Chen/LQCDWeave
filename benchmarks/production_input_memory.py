"""Bounded CPU full-calculation allocation comparison for optional mode reads.

Inputs are allocated before tracing. Results are diagnostic for this synthetic
identity solve and local baryon contraction, not production memory guarantees.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
from time import perf_counter
import tracemalloc

import numpy as np

from lqcdweave.production.perambulator.api import PerambulatorGenerator, VectorBasis
from lqcdweave.production.elemental import BaryonElementalGenerator
from lqcdweave.production.elemental.api import BaryonElementalKernel as BaryonElementalGenerator


class Field:
    def __init__(self, value):
        self.value = value
        self.Ne = value.shape[1] if getattr(value, "ndim", None) == 6 else None

    def load(self, key):
        return self.value


class BorrowedModes:
    def __init__(self, value, capacity):
        self.value = value
        self.Ne = value.shape[1]
        self.buffer = np.empty((capacity, *value.shape[2:]), dtype=value.dtype)
        self.reads = 0

    def load(self, key):
        raise AssertionError("whole-basis load is forbidden in streamed mode")

    def read_modes(self, key, time, start, stop):
        self.reads += 1
        self.buffer[:stop-start] = self.value[time, start:stop]
        return self.buffer[:stop-start]


class IdentityBackend:
    array_module = np
    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)

    def load(self, gauge, parameters):
        return None

    def solve_into(self, rhs, solution, *, tol, maxiter):
        solution[...] = rhs


def measure(make, calculate):
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    try:
        generator, calculate_args = make()
        generator.load("cfg")
        load_peak = tracemalloc.get_traced_memory()[1]
        result = calculate(generator, *calculate_args)
        full_peak = tracemalloc.get_traced_memory()[1]
        elapsed = perf_counter() - started
    finally:
        tracemalloc.stop()
    metrics = getattr(generator,"last_metrics",{})
    return dict(load_peak_bytes=load_peak, full_peak_bytes=full_peak,
                elapsed_seconds=elapsed, vector_scratch_bytes=generator.vector_scratch_bytes,
                sink_cache_used_bytes=getattr(generator,"sink_cache_used_bytes",0),
                sink_cache_hits=metrics.get("sink_cache_hits"),
                sink_cache_misses=metrics.get("sink_cache_misses")), \
           np.array(result, copy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sink-time-block-size", type=int, default=1)
    parser.add_argument("--sink-cache-limit-bytes", type=int, default=0)
    args = parser.parse_args()
    if (args.repeats < 1 or args.sink_time_block_size not in range(1,5)
            or args.sink_cache_limit_bytes < 0):
        parser.error("repeats must be positive, time block 1..4 and cache nonnegative")
    sizes = (8, 4, 4, 4)
    lx, ly, lz, lt = sizes
    ne = 12
    rng = np.random.default_rng(926)
    vectors = rng.normal(size=(lt, ne, lz, ly, lx, 3)) + 1j * rng.normal(
        size=(lt, ne, lz, ly, lx, 3))
    gauge = np.broadcast_to(np.eye(3), (lt, lz, ly, lx, 4, 3, 3)).astype(
        np.complex128, copy=True)
    def propagator(stream):
        vector = VectorBasis(BorrowedModes(vectors, 2) if stream else Field(vectors), ne)
        generator = PerambulatorGenerator(sizes, Field(None), .2, 1e-10, 10,
            backend=IdentityBackend(),
            stream_eigenvectors=stream, sink_ne_block_size=2,
            sink_time_block_size=args.sink_time_block_size if stream else 1,
            sink_cache_limit_bytes=args.sink_cache_limit_bytes if stream else 0,
            MRHS=True)
        return generator, (vector,)
    def baryon(stream):
        return BaryonElementalGenerator(sizes, Field(gauge),
            BorrowedModes(vectors, 2) if stream else Field(vectors),
            usedNe=ne, ne_block_size=2, paths=[((), (), ())],
            stream_eigenvectors=stream, cache_bytes=0), ()
    rows = {}
    for name, make, calculate in (
        ("propagator", propagator,
         lambda generator, vector: generator.calc(
             0, src=vector, sink=[vector])[0].values),
        ("baryon", baryon, lambda generator: generator.calc_path_blocks(0)["vvv"])):
        measured = {}
        values = {}
        for mode, enabled in (("eager", False), ("streamed", True)):
            samples = [measure(lambda: make(enabled), calculate)
                       for _ in range(args.repeats)]
            values[mode] = samples[0][1]
            for _, value in samples[1:]:
                np.testing.assert_allclose(value, values[mode], atol=2e-12)
            measured[mode] = dict(
                elapsed_seconds=statistics.median(row["elapsed_seconds"] for row, _ in samples),
                elapsed_samples_seconds=[row["elapsed_seconds"] for row, _ in samples],
                load_peak_bytes=max(row["load_peak_bytes"] for row, _ in samples),
                full_peak_bytes=max(row["full_peak_bytes"] for row, _ in samples),
                vector_scratch_bytes=samples[0][0]["vector_scratch_bytes"],
                sink_cache_used_bytes=samples[0][0]["sink_cache_used_bytes"],
                sink_cache_hits=samples[0][0]["sink_cache_hits"],
                sink_cache_misses=samples[0][0]["sink_cache_misses"])
        eager, streamed = measured["eager"], measured["streamed"]
        eager_value, streamed_value = values["eager"], values["streamed"]
        error = float(np.max(np.abs(eager_value - streamed_value)))
        np.testing.assert_allclose(eager_value, streamed_value, atol=2e-12)
        rows[name] = dict(eager=eager, streamed=streamed, max_abs_difference=error)
    report = dict(lattice=list(sizes), Ne=ne, repeats=args.repeats,
                  sink_time_block_size=args.sink_time_block_size,
                  sink_cache_limit_bytes=args.sink_cache_limit_bytes,
                  input_vector_bytes=vectors.nbytes,
                  gauge_bytes=gauge.nbytes, dtype="complex128", source_time=0,
                  backend="numpy-identity", rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
