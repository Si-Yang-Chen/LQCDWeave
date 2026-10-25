"""Reproducible local point-subset read benchmark; no performance threshold."""

import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from lqcdweave.adapters.propagator import ArrayPropagator


def measured(call, repeats=12):
    for _ in range(2):
        call()
    seconds = []
    for _ in range(repeats):
        begin = perf_counter()
        call()
        seconds.append(perf_counter() - begin)
    return median(seconds) * 1000


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "psp.npy"
        shape = (2, 2, 4, 4, 64, 3, 64, 3)
        data = np.lib.format.open_memmap(path, mode="w+", dtype="complex128", shape=shape)
        data[:] = 1 + 1j
        data.flush()
        del data
        points = {"sink_point": list(range(0, 64, 8)),
                  "source_point": list(range(1, 64, 8))}
        with ArrayPropagator.from_npy({"psp": path}, time_convention="relative") as reader:
            def full():
                return float(reader.block(0, 1, "psp").real.sum())
            def selected():
                return float(reader.block(0, 1, "psp", point_indices=points).real.sum())
            def prefix():
                return float(reader.block(0, 1, "psp", usedNp=8).real.sum())
            dense = dict(full_ms=measured(full), exact_ms=measured(selected),
                         prefix_ms=measured(prefix),
                         full_bytes=reader.block(0, 1, "psp").nbytes,
                         selected_bytes=reader.block(0, 1, "psp", point_indices=points).nbytes)
        print(json.dumps(dict(dense=dense), sort_keys=True))


if __name__ == "__main__":
    main()
