"""CSNS V100 correctness and throughput check for Laplace eigenvectors.

This is an opt-in cluster benchmark, not part of unittest discovery. It compares
the GPU path against the bundled weak_field fixture, then follows the legacy
CuPy+QUDA default dispatch and benchmarks it against the migrated explicit
QUDA backend on a production-size QIO gauge field. The legacy checkout and
production gauge are supplied separately at run time.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# PyQUDA on CSNS links against OpenMPI 4.1.6, while the default login module
# exposes OpenMPI 5 first. Re-exec before importing NumPy/CuPy so the dynamic
# loader starts with the compatible MPI libraries at the front of its path.
_mpi4 = os.environ.get(
    "CSNS_OPENMPI4_LIB",
    "/dg_hpc/LQCD/software/x86_gpu/utilities/openmpi-4.1.6-ucx-cuda12.4/lib",
)
if (os.path.isdir(_mpi4) and os.environ.get("CSNS_LAPLACE_MPI4_READY") != "1"
        and not os.environ.get("LD_LIBRARY_PATH", "").startswith(_mpi4)):
    _env = os.environ.copy()
    _env["LD_LIBRARY_PATH"] = _mpi4 + ":" + _env.get("LD_LIBRARY_PATH", "")
    _env["CSNS_LAPLACE_MPI4_READY"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], _env)

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lqcdweave.adapters.laplace_quda import QudaLaplaceBackend
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator


class GaugeHandle:
    def __init__(self, field, gauge_id, file=None):
        self.field = field
        self.gauge_id = gauge_id
        self.file = file

    def load(self, key):
        if self.field is None and self.file is not None:
            from pyquda_utils import io
            return io.readQIOGauge(str(self.file))
        return self.field


def sync(cp):
    cp.cuda.runtime.deviceSynchronize()


def arrays(value, cp):
    return cp.asnumpy(value) if hasattr(value, "__cuda_array_interface__") else np.asarray(value)


def subspace_error(reference, candidate):
    left = reference.reshape(reference.shape[0], -1)
    right = candidate.reshape(candidate.shape[0], -1)
    singular = np.linalg.svd(left.conj() @ right.T, compute_uv=False)
    return float(np.max(np.abs(1.0 - singular)))


def check_legacy_weak_field(cp, backend, fixture_dir, tol, nstep, rho):
    fixture_dir = Path(fixture_dir)
    gauge = np.load(fixture_dir / "weak_field.npy", mmap_mode="r")
    gauge = cp.asarray(gauge.reshape(4, 8, 4, 4, 4, 3, 3).transpose(1, 2, 3, 4, 0, 5, 6))
    expected_vectors = np.load(fixture_dir / "weak_field.eigenvector.ref.npy", mmap_mode="r")
    expected_values = np.load(fixture_dir / "weak_field.eigenvalue.npy", mmap_mode="r").real
    lt, lz, ly, lx = gauge.shape[:4]
    gen = LaplaceEigenvectorGenerator((lx, ly, lz, lt), GaugeHandle(gauge, "weak_field"),
                                     expected_values.shape[1], tol, array_module=cp,
                                     backend=backend).load("weak_field")
    smear_started = time.perf_counter()
    gen.stout_smear(nstep, rho)
    sync(cp)
    smear_seconds = time.perf_counter() - smear_started
    max_eval_error = 0.0
    max_vector_error = 0.0
    min_overlap = 1.0
    for t in range(lt):
        vectors, values = gen.calc(t)
        sync(cp)
        vectors, values = arrays(vectors, cp).reshape(expected_values.shape[1], -1), arrays(values, cp)
        reference = expected_vectors[t].reshape(expected_values.shape[1], -1)
        eval_error = float(np.max(np.abs(values - expected_values[t])))
        overlaps = np.sum(reference.conj() * vectors, axis=1)
        vector_error = np.linalg.norm(vectors * overlaps.conj()[:, None] - reference, axis=1)
        max_eval_error = max(max_eval_error, eval_error)
        max_vector_error = max(max_vector_error, float(vector_error.max()))
        min_overlap = min(min_overlap, float(np.abs(overlaps).min()))
    if max_eval_error > 1e-7 or max_vector_error > 1e-5 or min_overlap < 0.999:
        raise AssertionError(
            "legacy weak_field mismatch: "
            f"max_eval={max_eval_error:.3g}, max_vector={max_vector_error:.3g}, "
            f"min_overlap={min_overlap:.12g}"
        )
    return {"case": "weak_field_gpu_correctness", "shape": [lt, lz, ly, lx],
            "Ne": int(expected_values.shape[1]), "times": list(range(lt)),
            "stout_seconds": smear_seconds, "max_eigenvalue_abs_error": max_eval_error,
            "max_phase_aligned_vector_error": max_vector_error,
            "min_mode_overlap": min_overlap}


def benchmark_old_default_pipeline(cp, backend, legacy_root, gauge_path, lattice, ne, tol, nstep, rho, repeats):
    """Run the original EigenvectorGenerator and migrated generator end to end."""
    from pathlib import Path

    legacy_root = Path(legacy_root).resolve()
    if not (legacy_root / "lattice").is_dir():
        raise FileNotFoundError(f"legacy checkout must contain a lattice package: {legacy_root}")
    sys.path.insert(0, str(legacy_root))
    from lattice import EigenvectorGenerator as LegacyGenerator
    from lattice import GaugeFieldIldg, Nc, Nd, set_backend
    import lattice.backend as legacy_backend

    set_backend("cupy")
    legacy_backend.PYQUDA = True  # pyquda.init has already run in main()
    lx, ly, lz, lt = lattice
    path = Path(gauge_path)
    stem = str(path.with_suffix(""))
    cfg = stem.rsplit("_", 1)[-1]
    prefix = stem[:-len(cfg)]
    loader = GaugeFieldIldg(prefix, path.suffix, [lt, lz, ly, lx, Nd, Nc, Nc])

    started = time.perf_counter()
    legacy = LegacyGenerator(lattice, loader, ne, tol)
    legacy_init_seconds = time.perf_counter() - started
    migrated = LaplaceEigenvectorGenerator(
        lattice, GaugeHandle(None, "csns:cfg2000", file=str(gauge_path)), ne, tol,
        array_module=cp, backend=backend,
    )
    migrated_init_seconds = time.perf_counter() - started - legacy_init_seconds

    def run_legacy():
        started = time.perf_counter()
        # The legacy generator clears lazy data after load but leaves the same
        # filename cached. Re-arm that handle for repeated measurements of one cfg.
        loader.file = None
        loader.data = None
        legacy.load(cfg)
        sync(cp)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        legacy.stout_smear(nstep, rho)
        sync(cp)
        stout_seconds = time.perf_counter() - started
        started = time.perf_counter()
        vectors, values = legacy.calc(0)
        sync(cp)
        solve_seconds = time.perf_counter() - started
        return load_seconds, stout_seconds, solve_seconds, vectors, values

    def run_migrated():
        started = time.perf_counter()
        migrated.load("cfg2000")
        sync(cp)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        migrated.stout_smear(nstep, rho)
        sync(cp)
        stout_seconds = time.perf_counter() - started
        started = time.perf_counter()
        vectors, values = migrated.calc(0)
        sync(cp)
        solve_seconds = time.perf_counter() - started
        return load_seconds, stout_seconds, solve_seconds, vectors, values

    legacy_samples, migrated_samples = [], []
    legacy_vectors = legacy_values = migrated_vectors = migrated_values = None
    for _ in range(repeats):
        run_started = time.perf_counter()
        load_seconds, stout_seconds, solve_seconds, vectors, values = run_legacy()
        legacy_samples.append(dict(load=load_seconds, stout=stout_seconds, solve=solve_seconds,
                                  total=time.perf_counter() - run_started))
        legacy_vectors, legacy_values = arrays(vectors, cp), arrays(values, cp)
        legacy_links = cp.asarray(legacy._U).copy()

        run_started = time.perf_counter()
        load_seconds, stout_seconds, solve_seconds, vectors, values = run_migrated()
        migrated_samples.append(dict(load=load_seconds, stout=stout_seconds, solve=solve_seconds,
                                     total=time.perf_counter() - run_started))
        migrated_vectors, migrated_values = arrays(vectors, cp), arrays(values, cp)
        migrated_links = cp.asarray(migrated._links).copy()

    def summary(samples, key):
        values = [sample[key] for sample in samples]
        return {"seconds": values, "steady_state_median_seconds": float(np.median(values[1:]))
                if len(values) > 1 else values[0]}

    eigen_error = float(np.max(np.abs(migrated_values - legacy_values)))
    overlap_error = subspace_error(legacy_vectors, migrated_vectors)
    link_error = float(cp.max(cp.abs(migrated_links - legacy_links)).item())
    result = {
        "case": "actual_legacy_default_end_to_end",
        "lattice": list(lattice), "Ne": ne, "tol": tol, "time_slice": 0,
        "stout_steps": nstep, "rho": rho, "repeats": repeats,
        "legacy_constructor_seconds": legacy_init_seconds,
        "migrated_constructor_seconds": migrated_init_seconds,
        "legacy": {key: summary(legacy_samples, key) for key in ("load", "stout", "solve", "total")},
        "migrated": {key: summary(migrated_samples, key) for key in ("load", "stout", "solve", "total")},
        "full_path_ratio_new_over_legacy": (
            summary(migrated_samples, "total")["steady_state_median_seconds"]
            / summary(legacy_samples, "total")["steady_state_median_seconds"]
        ),
        "max_eigenvalue_abs_difference": eigen_error,
        "subspace_error": overlap_error,
        "max_gauge_link_abs_difference": link_error,
        "correctness_pass": eigen_error < 1e-6 and overlap_error < 1e-4 and link_error < 1e-12,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path,
                        default=PROJECT_ROOT / "tests" / "data" / "weak_field",
                        help="weak_field reference directory (defaults to bundled test data)")
    parser.add_argument("--legacy-root", type=Path, required=True,
                        help="external old-program checkout containing lattice/")
    parser.add_argument("--gauge", required=True, help="readable production QIO gauge file")
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--latt-size", default="16,16,16,128", help="Lx,Ly,Lz,Lt")
    parser.add_argument("--ne", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--nstep", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.12)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    import cupy as cp
    import pyquda

    pyquda.init(grid_size=[1, 1, 1, 1], backend="cupy")
    backend = QudaLaplaceBackend()
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    print(json.dumps({"device": device["name"].decode() if isinstance(device["name"], bytes) else device["name"],
                      "cuda_runtime": cp.cuda.runtime.runtimeGetVersion()}), flush=True)
    lattice = tuple(int(value) for value in args.latt_size.split(","))
    results = [check_legacy_weak_field(cp, backend, args.fixture_dir, args.tol, args.nstep, args.rho)]
    results.append(benchmark_old_default_pipeline(
        cp, backend, args.legacy_root, args.gauge, lattice, args.ne, args.tol, args.nstep, args.rho, args.repeats
    ))
    output = {"results": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    if not all(result.get("correctness_pass", result.get("solver_comparison_pass", True))
               for result in results):
        raise AssertionError("production-scale solver outputs differ beyond the configured thresholds")


if __name__ == "__main__":
    main()
