"""PyQUDA adapter for the reusable propagator front-end.

The adapter is optional: importing this module does not initialize CUDA or
PyQUDA.  It converts the front-end's logical full-x fields to QUDA's
checkerboard fields and writes solutions into caller-owned CuPy buffers.
"""

from time import perf_counter

import numpy as np


class QudaBackend:
    """Single-process PyQUDA/QUDA backend for ``PerambulatorGenerator``."""

    stream_ordered_solve = True
    gamma5_hermitian = True
    # Wilson/clover in the DeGrand--Rossi basis; the overall gamma5 sign cancels.
    gamma5 = np.diag([1., 1., -1., -1.]).astype(np.complex128)
    spin_basis = 'degrand_rossi'

    def __init__(self, *, measure=False):
        if not isinstance(measure, (bool, np.bool_)):
            raise ValueError('measure must be boolean')
        try:
            import cupy as cp
            from pyquda_utils import core
        except ImportError as exc:  # pragma: no cover - depends on CSNS runtime
            raise ImportError("QudaBackend requires CuPy and pyquda_utils") from exc
        self.array_module = cp
        self.measure = bool(measure)
        self._core = core
        self.empty = cp.empty
        self.zeros = cp.zeros
        self.dirac = None
        self.latt_info = None
        self._sources = []
        self._multi_sources = {}
        self._indices = None
        self.reset_metrics()

    def reset_metrics(self):
        self.solve_seconds = 0.0
        self.to_quda_seconds = 0.0
        self.from_quda_seconds = 0.0
        self.solve_calls = 0
        self.rhs_count = 0
        self.multirhs_calls = 0

    def synchronize(self):
        self.array_module.cuda.runtime.deviceSynchronize()

    def load(self, gauge, parameters):
        from pyquda_utils import io

        self.destroy()
        cp = self.array_module
        latt_size = tuple(parameters["latt_size"])
        self.latt_info = self._core.LatticeInfo(
            latt_size=latt_size,
            t_boundary=parameters["t_boundary"],
            anisotropy=parameters["xi_0"] / parameters["nu"],
        )
        if tuple(self.latt_info.size) != latt_size or any(self.latt_info.grid_coord):
            raise NotImplementedError("QudaBackend supports one MPI rank only")
        lx, ly, lz, lt = latt_size
        if lx % 2:
            raise ValueError("QUDA requires an even x extent")
        t, z, y, h = np.indices((lt, lz, ly, lx // 2))
        self._indices = cp.asarray(np.stack([
            (((t * lz + z) * ly + y) * lx + 2*h + (p-t-z-y) % 2)
            for p in (0, 1)
        ]).reshape(-1))
        self.dirac = self._core.getDirac(
            self.latt_info,
            parameters["mass"],
            parameters["tol"],
            parameters["maxiter"],
            parameters["xi_0"],
            parameters["clover_coeff_t"],
            parameters["clover_coeff_r"],
            parameters["multigrid"],
        )
        gauge_payload = io.readQIOGauge(gauge.file) if hasattr(gauge, "file") else gauge
        self.dirac.loadGauge(gauge_payload)
        self._tol, self._maxiter = parameters['tol'], parameters['maxiter']
        self.reset_metrics()

    def stout_smear(self, gauge, *, nstep: int, rho: float, dir_ignore: int):
        """Apply the legacy PyQUDA gauge transform at the adapter boundary.

        The native method mutates its gauge payload in place; returning the
        same object makes that ownership behavior explicit to production.
        """
        transform = getattr(gauge, "stoutSmear", None)
        if not callable(transform):
            raise NotImplementedError("this gauge payload does not provide PyQUDA stoutSmear")
        transform(nstep, rho, dir_ignore)
        return gauge

    def _to_quda(self, fields):
        cp = self.array_module
        while len(self._sources) < len(fields):
            self._sources.append(self._core.LatticeFermion(self.latt_info))
        for rhs, field in zip(fields, self._sources):
            cp.take(rhs.reshape(-1, 12), self._indices, axis=0,
                    out=field.data.reshape(-1, 12))
        return self._sources[:len(fields)]

    def _to_quda_multi(self, fields):
        """Fill one persistent QUDA multi-source container."""
        count = len(fields)
        multi = self._multi_sources.get(count)
        if multi is None:
            multi_type = getattr(self._core, "MultiLatticeFermion", None)
            if multi_type is None:
                from pyquda_utils.core import MultiLatticeFermion as multi_type
            multi = multi_type(self.latt_info, count)
            self._multi_sources[count] = multi
        cp = self.array_module
        for rhs, field in zip(fields, (multi[i] for i in range(count))):
            cp.take(rhs.reshape(-1, 12), self._indices, axis=0,
                    out=field.data.reshape(-1, 12))
        return multi

    def _from_quda(self, fields, output):
        for r, field in enumerate(fields):
            output[r].reshape(-1, 12)[self._indices] = field.data.reshape(-1, 12)

    def solve_into(self, rhs, solution, *, tol, maxiter):
        if self.dirac is None:
            raise RuntimeError("load an operator before solving")
        if rhs.ndim != 7 or solution.shape != rhs.shape:
            raise ValueError("rhs and solution must have shape (rhs,Lt,Lz,Ly,Lx,4,3)")
        if not np.isfinite(float(tol)) or not 0 < tol < 1:
            raise ValueError("tol must be in (0,1)")
        if tol != self._tol or maxiter != self._maxiter:
            raise ValueError("solve parameters must match the loaded operator")
        if self.measure:
            self.synchronize()
        started = perf_counter()
        use_multi = len(rhs) > 1 and hasattr(self.dirac, "invertMultiSrc")
        sources = self._to_quda_multi(rhs) if use_multi else self._to_quda(rhs)
        self.synchronize()
        self.to_quda_seconds += perf_counter() - started
        started = perf_counter()
        solved = (self.dirac.invertMultiSrc(sources) if use_multi
                  else [self.dirac.invert(source) for source in sources])
        self.synchronize()
        self.solve_seconds += perf_counter() - started
        self.solve_calls += 1
        self.multirhs_calls += int(use_multi)
        self.rhs_count += int(rhs.shape[0])
        started = perf_counter()
        self._from_quda(solved, solution)
        if self.measure:
            self.synchronize()
        self.from_quda_seconds += perf_counter() - started

    def destroy(self):
        if self.dirac is not None and hasattr(self.dirac, "destroy"):
            self.dirac.destroy()
        self.dirac = None
        self._sources = []
        self._multi_sources = {}
        self._indices = None
