"""Time every Baryon vertex lowering per row-family structure, without fallbacks.

Structures are those of ``tests/correlator/expansion/baryon_row_structures.py``
(split rank, fold, density, Dirac-Pauli data, mixed families, single rows, two
Phi per row). Each is a full-Wick uuu row matrix on random complex64 data of
``--ne`` modes (each Phi antisymmetrized over its three modes) at ``--sink-times`` x
``--source-times``. Every lowering is planned in the same process with the
default planner, then all plans run interleaved (alternating order) for
``--rounds`` rounds after one warm-up round; walls are synchronized, include
reader transfers and end at the host value. Values are compared with
``joined``. A lowering that fails to plan is recorded, not retried.

    python benchmarks/baryon_vertex_structure.py --backend cupy --ne 100 \
        --structures rank2-p16 generic-e64 --output structure.json
"""
import argparse
from hashlib import sha256
from itertools import permutations
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from lqcdweave.adapters.elemental import ArrayBaryonElemental  # noqa: E402
from lqcdweave.adapters.propagator import ArrayPropagator  # noqa: E402
from lqcdweave.contraction.correlator import ContractionTask  # noqa: E402
from correlator.expansion.baryon_row_structures import COMPONENTS, STRUCTURES, request, structure  # noqa: E402

LOWERINGS = ("auto", "pair", "nonzero", "dense", "joined")


def _random(seed, shape):
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(shape[-1])).astype(np.complex64)


def readers(ne, lt):
    """One shared reader of two Phi components and one propagator, fixed by ``ne`` and ``lt``."""
    phi = _random(ne, (len(COMPONENTS), 1, lt, ne, ne, ne))
    phi = sum((-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) * phi.transpose(
        0, 1, 2, *(3 + q for q in p)) for p in permutations(range(3))).astype(np.complex64)
    elemental = ArrayBaryonElemental({"vvv": phi}, COMPONENTS, cfg="structure",
                                     data_identity=f"structure-phi2-{ne}-{lt}")
    propagator = ArrayPropagator({"vsv": _random(ne + 1, (lt, lt, 4, 4, ne, ne))}, time_convention="absolute",
                                 cfg="structure", data_identity=f"structure-vsv-{ne}-{lt}")
    return elemental, propagator


def _predicted(task):
    """Predicted cfg-plan seconds (compute and preparation) and the form "auto" selected, if any."""
    prediction = task.plan_report["prediction"]
    selection = prediction.get("baryon_vertex_selection") or {}
    return dict(predicted_s=float(prediction.get("compute_seconds", 0.0) + prediction.get("preparation_seconds", 0.0)),
                selected=selection.get("selected"), candidates_s=selection.get("predicted_seconds"))


def _host(value, xp):
    value = getattr(value, "value", value)
    return np.asarray(xp.asnumpy(value) if xp is not np and isinstance(value, xp.ndarray) else value)


def _summary(walls):
    q1, median, q3 = np.quantile(walls, (0.25, 0.5, 0.75))
    return dict(median_s=float(median), q1_s=float(q1), q3_s=float(q3), min_s=float(min(walls)))


def measure(spec, *, elemental, propagator, args, xp):
    value, vertices = request(spec, sink_reader=elemental, source_reader=elemental)
    options = dict(vertex_map=vertices, propagator_map={"S^q": propagator, "S^q_\\mathrm{local}": propagator},
                   configuration="structure", time_axes={"sink": tuple(args.sink_times),
                                                         "source": tuple(args.source_times)},
                   backend=args.backend, dtype="complex64")
    row = dict(structure=spec, families=structure(spec), cells=int(np.size(value)), variants={})
    tasks = {}
    try:
        for lowering in args.lowerings:
            variant = row["variants"][lowering] = {}
            try:
                task = ContractionTask(value, **options, baryon_vertices=lowering)
                start = time.perf_counter()
                task.plan()
                variant["plan_s"] = time.perf_counter() - start
                plan = task._selected.plan
                variant.update(templates=len(plan.templates), instances=len(plan.instances),
                               peak_device_bytes=plan.peak_device_bytes, peak_host_bytes=plan.peak_host_bytes,
                               **_predicted(task))
                tasks[lowering] = task
            except Exception as error:
                variant["error"] = f"{type(error).__name__}: {error}"
                print(spec, lowering, "plan failed", variant["error"], flush=True)
        names, values, walls = list(tasks), {}, {name: [] for name in tasks}
        for round_ in range(args.rounds + 1):
            for name in (names if round_ % 2 == 0 else names[::-1]):
                if xp is not np:
                    xp.get_default_memory_pool().free_all_blocks()
                    xp.cuda.Device().synchronize()
                start = time.perf_counter()
                result = tasks[name].run()
                host = _host(result.value, xp)
                if xp is not np:
                    xp.cuda.Device().synchronize()
                result.close()
                wall = time.perf_counter() - start
                if round_ == 0:
                    values[name], row["variants"][name]["first_s"] = host, wall
                else:
                    walls[name].append(wall)
        reference = values.get("joined")
        for name in names:
            variant = row["variants"][name]
            variant.update(_summary(walls[name]), walls_s=walls[name], dtype=str(values[name].dtype))
            if reference is not None:
                difference = values[name].astype(np.complex128) - reference
                # Structures whose uuu Wick sum cancels leave only rounding: compare the scales too.
                variant.update(rel_l2_joined=float(np.linalg.norm(difference) / np.linalg.norm(reference)),
                               max_abs=float(np.abs(values[name]).max()),
                               max_abs_joined=float(np.abs(reference).max()))
    finally:
        for task in tasks.values():
            task.close()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    parser.add_argument("--ne", type=int, default=100)
    parser.add_argument("--sink-times", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument("--source-times", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--structures", nargs="+", default=list(STRUCTURES))
    parser.add_argument("--lowerings", nargs="+", default=list(LOWERINGS), choices=LOWERINGS)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    xp = np
    if args.backend == "cupy":
        import cupy as xp
        if int(xp.arange(16).sum().get()) != 120:
            raise AssertionError("CUDA arithmetic failed")
    lt = 1 + max(args.sink_times + args.source_times)
    elemental, propagator = readers(args.ne, lt)
    report = dict(backend=args.backend, ne=args.ne, sink_times=args.sink_times, source_times=args.source_times,
                  rounds=args.rounds, host=os.uname().nodename, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  device=(str(xp.cuda.runtime.getDeviceProperties(0)["name"]) if xp is not np else None),
                  harness_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), rows=[])
    for spec in args.structures:
        try:
            row = measure(spec, elemental=elemental, propagator=propagator, args=args, xp=xp)
        except Exception as error:
            row = dict(structure=spec, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        report["rows"].append(row)
        for name, variant in row.get("variants", {}).items():
            fields = (f"error={variant['error']}" if "error" in variant else
                      f"plan={variant['plan_s']:.2f}s median={variant.get('median_s', float('nan')):.4f}s "
                      f"iqr=[{variant.get('q1_s', float('nan')):.4f},{variant.get('q3_s', float('nan')):.4f}] "
                      f"rel={variant.get('rel_l2_joined', float('nan')):.1e}")
            print(f"{spec:28s} {name:8s} {fields}", flush=True)
        if "error" in row:
            print(spec, row["error"], flush=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
