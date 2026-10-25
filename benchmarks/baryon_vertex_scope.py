"""Time the baryon_vertices scope cases under every lowering.

The cases are those of
``tests/correlator/expansion/test_baryon_factor_network_scope.py``. Without
``--ne`` they keep the test fixtures (modes of order 2-3), so walls mostly show
planning and per-operation overhead, and values are compared with the
packing-free test reference. ``--ne N`` keeps each case's request, operators and
bindings but replaces every reader and propagator with random complex64 data of
N modes (baryon Phi antisymmetrized over its three mode indices, like an
epsilon contraction); glueball scalars are kept. Values are then compared with
``joined``, mode blocks with the unsplit ``joined`` matrix. Every lowering is
planned in the same process with the default planner, then all plans run
interleaved (alternating order) for ``--repeat`` rounds after one warm-up
round; walls are synchronized, include reader transfers and end at the host
value. The median and quartiles are reported.

``--phase plan`` plans every lowering away from the device (``--device-bytes``
declares the target, ``--profile`` its cost model) and saves complete cfg
archives under ``--archives``; ``--phase run`` on the device only loads and
runs them. The random data are seeded by each original reader's bytes, so both
phases rebuild identical readers.

    python benchmarks/baryon_vertex_scope.py --phase plan --backend cupy --ne 100 --cases rows-control \
        --device-bytes 34089730048 --memory-limit 25769803776 --profile V100.json --archives plans
    python benchmarks/baryon_vertex_scope.py --phase run --backend cupy --ne 100 --cases rows-control \
        --archives plans --output rows.json
"""
import argparse
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from lqcdweave.adapters.propagator import ArrayPropagator  # noqa: E402
from lqcdweave.contraction.binding.api import Baryon, Glueball, Meson  # noqa: E402
from lqcdweave.contraction.correlator import ContractionTask, calculate  # noqa: E402
from lqcdweave.contracts.memory import TargetHardwareMemory  # noqa: E402
from correlator.expansion.test_baryon_factor_network_scope import (  # noqa: E402
    LOWERINGS, SCOPE_CASES, mode_blocked, uuu_case,
)


def _predicted(task):
    """Predicted cfg-plan seconds (compute and preparation) and the form "auto" selected, if any."""
    prediction = task.plan_report["prediction"]
    selection = prediction.get("baryon_vertex_selection") or {}
    return dict(predicted_s=float(prediction.get("compute_seconds", 0.0) + prediction.get("preparation_seconds", 0.0)),
                selected=selection.get("selected"), candidates_s=selection.get("predicted_seconds"))


def _host(value):
    value = getattr(value, "value", value)
    return np.asarray(value.get() if hasattr(value, "get") else value)


def _relative(a, b):
    scale = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / (scale or 1.0)


def _antisymmetrize(values):
    result = np.zeros_like(values)
    lead = values.ndim - 3
    for p in permutations(range(3)):
        sign = (-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
        result += sign * values.transpose(*range(lead), *(lead + q for q in p))
    return result


def rescale(case, ne):
    """The same request and bindings on random data with ``ne`` modes; no reference."""
    readers, vertices = {}, {}

    def data(old, key, rng):
        modes = 3 if key == "vvv" else 2
        shape = old.shape[:-modes] + (ne,) * modes
        values = ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(ne)).astype(np.complex64)
        return _antisymmetrize(values) if key == "vvv" else values

    def reader(old):
        if id(old) not in readers:
            # Seeded by the original bytes: independent of binding order.
            digest = sha256(b"".join(np.ascontiguousarray(old.sectors[key]).tobytes()
                                     for key in sorted(old.sectors))).hexdigest()
            rng = np.random.default_rng(int(digest[:16], 16) ^ ne)
            sectors = {key: data(old.sectors[key], key, rng) for key in sorted(old.sectors)}
            identity = f"ne{ne}-{digest[:16]}"
            if isinstance(old, ArrayPropagator):
                new = ArrayPropagator(sectors, time_convention=old.time_convention, cfg=old.cfg, data_identity=identity)
            else:
                new = type(old)(sectors, old.components, phase_convention=old.phase_convention, cfg=old.cfg,
                                data_identity=identity)
            readers[id(old)] = old, new
        return readers[id(old)][1]

    def vertex(old):
        if id(old) not in vertices:
            if isinstance(old, Baryon):
                new = Baryon(reader(old.reader), old.operator, source=old.dagger, phase_convention=old.phase_convention,
                             spin_basis=old.spin_basis, boundary=old.boundary)
            elif isinstance(old, Glueball):  # one complex scalar per time: no modes to scale
                new = old
            else:
                assert isinstance(old, Meson), type(old)
                new = Meson(reader(old.reader), old.operator, source=old.dagger, phase_convention=old.phase_convention,
                            boundary=old.boundary)
            vertices[id(old)] = old, new
        return vertices[id(old)][1]

    for old in case.readers:  # readers a callable vertex_map binds lazily, created now
        reader(old)
    source = case.options["vertex_map"]
    vertex_map = ({key: vertex(value) for key, value in source.items()} if isinstance(source, dict)
                  else lambda row: vertex(source(row)))
    options = dict(case.options, vertex_map=vertex_map,
                   propagator_map={key: reader(value) for key, value in case.options["propagator_map"].items()})
    # Fixture budgets and mode counts were sized for the small data.
    for key in ("usedNe", "memory_limit", "host_memory_limit"):
        options.pop(key, None)
    scaled = replace(case, options=options, expected=None)
    scaled.by_identity = {new.data_identity: new for _, new in readers.values()}
    return scaled


def scaled_case(name, ne, *, reference=True):
    if name == "rows-mode-blocked":
        control = rescale(uuu_case(symmetry=None), ne)
        blocked = mode_blocked(control, partitions=(0, ne // 3, ne))
        blocked.by_identity = control.by_identity
        if not reference:  # the run phase reads the planned reference back
            return blocked
        blocked.expected = _host(calculate(control.request, **control.options, baryon_vertices="joined"))
        return blocked
    return rescale(SCOPE_CASES[name](), ne)


def _repeat(task, repeat):
    walls, value = [], None
    for _ in range(repeat + 1):
        start = time.perf_counter()
        value = _host(task.run())
        walls.append(time.perf_counter() - start)
    return value, dict(first_s=walls[0], warm_median_s=statistics.median(walls[1:]), dtype=str(value.dtype))


def plan(name, case, *, options, archives):
    """Plan and save every lowering; the description goes next to the archives."""
    rows = []
    for lowering in LOWERINGS:
        try:
            problem = calculate(case.request, **options, baryon_vertices=lowering, execute=False).expanded.problem
            start = time.perf_counter()
            with ContractionTask(case.request, **options, baryon_vertices=lowering) as task:
                task.plan()
                planned = time.perf_counter() - start
                task.save_plan(archives / f"{name}-{lowering}.json.gz")
                # Saved reader roles, by the data identity rebuilt in the run phase.
                roles = {role: reader.data_identity for role, reader in task._selected.options["readers"].items()}
        except Exception as error:  # record and keep planning the other lowerings
            rows.append(dict(case=name, lowering=lowering, error=f"{type(error).__name__}: {error}"))
            continue
        rows.append(dict(case=name, lowering=lowering, opens=bool(problem.factorized_vertices),
                         terms=len(problem.terms), plan_s=planned, readers=roles))
    if case.expected is not None:
        np.save(archives / f"{name}-reference.npy", case.expected)
    (archives / f"{name}-plan.json").write_text(json.dumps(rows, indent=1))
    return rows


def replay(name, case, *, options, archives, repeat):
    """Load the saved archives and run them; plan rows are read back."""
    rows, values = [], {}
    reference = archives / f"{name}-reference.npy"
    expected = np.load(reference) if reference.is_file() else None
    for row in json.loads((archives / f"{name}-plan.json").read_text()):
        rows.append(row)
        if "error" in row:
            continue
        try:
            readers = {role: case.by_identity[identity] for role, identity in row["readers"].items()}
            with ContractionTask.load_plan(archives / f"{name}-{row['lowering']}.json.gz",
                                           configuration=options["configuration"], readers=readers) as task:
                value, timing = _repeat(task, repeat)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            continue
        values[row["lowering"]] = case.reduce(value)
        row.update(timing)
    return _compare(rows, values, expected)


def measure(name, case, *, options, repeat, xp=np):
    """Plan every lowering, then run all plans interleaved in alternating order."""
    rows, values, tasks = {}, {}, {}
    try:
        for lowering in LOWERINGS:
            row = rows[lowering] = dict(case=name, lowering=lowering)
            try:
                task = ContractionTask(case.request, **options, baryon_vertices=lowering)
                start = time.perf_counter()
                task.plan()
                row["plan_s"] = time.perf_counter() - start
                plan = task._selected.plan
                row.update(templates=len(plan.templates), instances=len(plan.instances),
                           peak_device_bytes=plan.peak_device_bytes, peak_host_bytes=plan.peak_host_bytes,
                           **_predicted(task))
                tasks[lowering] = task
            except Exception as error:  # record and keep timing the other lowerings
                row["error"] = f"{type(error).__name__}: {error}"
        names, walls = list(tasks), {name: [] for name in tasks}
        for round_ in range(repeat + 1):
            for lowering in (names if round_ % 2 == 0 else names[::-1]):
                if xp is not np:
                    xp.get_default_memory_pool().free_all_blocks()
                    xp.cuda.Device().synchronize()
                start = time.perf_counter()
                value = _host(tasks[lowering].run())
                if xp is not np:
                    xp.cuda.Device().synchronize()
                wall = time.perf_counter() - start
                if round_ == 0:
                    values[lowering] = case.reduce(value)
                    rows[lowering].update(first_s=wall, dtype=str(value.dtype))
                else:
                    walls[lowering].append(wall)
        for lowering in names:
            q1, median, q3 = np.quantile(walls[lowering], (0.25, 0.5, 0.75))
            rows[lowering].update(warm_median_s=float(median), q1_s=float(q1), q3_s=float(q3),
                                  walls_s=walls[lowering])
    finally:
        for task in tasks.values():
            task.close()
    return _compare(list(rows.values()), values, case.expected)


def _compare(rows, values, expected):
    for row in rows:
        if row["lowering"] not in values:
            continue
        if expected is not None:
            row["rel_l2_reference"] = _relative(values[row["lowering"]], expected)
        if "joined" in values:
            row["rel_l2_joined"] = _relative(values[row["lowering"]], values["joined"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    parser.add_argument("--ne", type=int, help="random data with this many modes instead of the test fixtures")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", nargs="*", default=list(SCOPE_CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("both", "plan", "run"), default="both")
    parser.add_argument("--archives", type=Path, help="archive directory of the plan/run phases")
    parser.add_argument("--device-bytes", type=int, help="plan phase: declared device memory of the target")
    parser.add_argument("--host-bytes", type=int, default=12 << 30, help="plan phase: declared host memory")
    parser.add_argument("--memory-limit", type=int, help="device (cupy) or host (numpy) planning budget")
    parser.add_argument("--profile", help="plan phase: cost-model document of the target")
    args = parser.parse_args()
    xp = np
    if args.backend == "cupy":
        import cupy as xp
    if args.phase != "both" and args.archives is None:
        parser.error("--phase plan/run requires --archives")
    rows = []
    for name in args.cases:
        case = (SCOPE_CASES[name]() if args.ne is None
                else scaled_case(name, args.ne, reference=args.phase != "run"))
        options = dict(case.options)
        if args.backend != "numpy":
            options["backend"] = args.backend
        if args.memory_limit is not None:
            options["memory_limit"] = args.memory_limit
        if args.phase == "plan":
            args.archives.mkdir(parents=True, exist_ok=True)
            options.update(host_memory_limit=args.host_bytes, profile=args.profile,
                           target_hardware=TargetHardwareMemory(host_available_bytes=args.host_bytes,
                               device_total_bytes=args.device_bytes, device_free_bytes=args.device_bytes))
            results = plan(name, case, options=options, archives=args.archives)
        elif args.phase == "run":
            results = replay(name, case, options=options, archives=args.archives, repeat=args.repeat)
        else:
            results = measure(name, case, options=options, repeat=args.repeat, xp=xp)
        for row in results:
            rows.append(row)
            if "error" in row:
                print(f"{name:22s} {row['lowering']:8s} {row['error']}", flush=True)
                continue
            fields = ([f"opens={row['opens']!s:5s}", f"terms={row['terms']:3d}"] if "opens" in row else
                      [f"templates={row['templates']:3d}"]) + [f"plan={row['plan_s']:.3f}s"]
            fields += [f"{key[:-2]}={row[key]:.4f}s" for key in ("first_s", "warm_median_s", "q1_s", "q3_s")
                       if key in row]
            fields += [f"{key[7:]}={row[key]:.1e}" for key in ("rel_l2_reference", "rel_l2_joined") if key in row]
            print(f"{name:22s} {row['lowering']:8s} " + " ".join(fields), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(dict(backend=args.backend, ne=args.ne, phase=args.phase,
                                               repeat=args.repeat, rows=rows),
                                          indent=1))


if __name__ == "__main__":
    main()
