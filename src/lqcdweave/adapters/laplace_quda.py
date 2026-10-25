"""Optional single-process PyQUDA adapter for Laplace eigenvectors.

Importing this module does not initialize CUDA or PyQUDA. Construct the adapter
inside an already configured single-process PyQUDA runtime.
"""

import numpy as np


class QudaLaplaceBackend:
    """Use QUDA for stout smearing and Laplacian applications or Lanczos."""

    def __init__(self):
        try:
            import cupy as cp
            from pyquda_utils import core
        except ImportError as exc:
            raise ImportError("QudaLaplaceBackend requires CuPy and pyquda_utils") from exc
        self.cp, self.core = cp, core
        self.invalidate()

    def invalidate(self):
        """Discard a cached native representation after external transforms."""
        self._native_gauge = None
        self._native_links = None

    def load(self, handle, key, latt_size):
        """Resolve native QUDA objects and QIO locators entirely at this boundary."""
        self.invalidate()
        lx, ly, lz, lt = latt_size
        try:
            payload = handle.load(key)
            if callable(getattr(payload, "stoutSmear", None)) and callable(getattr(payload, "lexico", None)):
                native = payload
            else:
                # A filename is only a locator, not a declaration of QIO format.
                # Dense NPY/raw and QDP handles also expose ``file``.
                path = None
                if getattr(handle, "format", None) in ("ildg", "qio"):
                    path = getattr(payload, "file", None) or getattr(handle, "file", None)
                native = self.load_gauge(path, latt_size) if path else None
            if native is not None:
                links = self.links_from_gauge(native, latt_size)
                self._native_gauge = native
                self._native_links = links
                return links
            raw = self.cp.asarray(payload[:])
            if raw.shape != (lt, lz, ly, lx, 4, 3, 3) or raw.dtype.kind != "c":
                raise ValueError("gauge must have complex shape (Lt,Lz,Ly,Lx,4,3,3)")
            return raw.transpose(4, 0, 1, 2, 3, 5, 6).astype(self.cp.complex128, copy=True)
        except Exception:
            self.invalidate()
            raise

    def _gauge(self, links, latt_size):
        cp, core = self.cp, self.core
        info = core.LatticeInfo(list(latt_size))
        if tuple(info.size) != tuple(latt_size) or any(info.grid_coord):
            raise NotImplementedError("QudaLaplaceBackend supports one MPI rank only")
        if latt_size[0] % 2:
            raise ValueError("QUDA requires an even x extent")
        gauge = core.LatticeGauge(info, cp.asarray(core.cb2(cp.asnumpy(links), [1, 2, 3, 4])))
        return info, gauge

    def load_gauge(self, gauge_file, latt_size):
        """Load a native QIO gauge object for the legacy QUDA stout path."""
        from pyquda_utils import io

        gauge = io.readQIOGauge(str(gauge_file))
        if tuple(gauge.latt_info.size) != tuple(latt_size) or any(gauge.latt_info.grid_coord):
            raise NotImplementedError("QudaLaplaceBackend supports one matching MPI rank only")
        return gauge

    def links_from_gauge(self, gauge, latt_size):
        """Copy native QUDA links to the generator's direction-first CuPy layout."""
        lx, ly, lz, lt = latt_size
        if tuple(gauge.latt_info.size) != tuple(latt_size) or any(gauge.latt_info.grid_coord):
            raise NotImplementedError("QudaLaplaceBackend supports one matching MPI rank only")
        return self.cp.asarray(gauge.lexico()).reshape(4, lt, lz, ly, lx, 3, 3).copy()

    def stout_smear(self, links, latt_size, nstep, rho, *, native_gauge=None, gauge_file=None):
        if native_gauge is not None:
            gauge = native_gauge
        elif gauge_file is not None:
            gauge = self.load_gauge(gauge_file, latt_size)
        elif self._native_gauge is not None and links is self._native_links:
            gauge = self._native_gauge
        else:
            _, gauge = self._gauge(links, latt_size)
        try:
            gauge.stoutSmear(nstep, rho, dir_ignore=3)
            links = self.links_from_gauge(gauge, latt_size)
            self._native_gauge = gauge
            self._native_links = links
            return links
        except Exception:
            self.invalidate()
            raise

    def solve(self, links, latt_size, t, count, tol, polynomial_degree, lambda_cut):
        lx, ly, lz, _ = latt_size
        info, gauge = self._gauge(links[:, t:t + 1], (lx, ly, lz, 1))
        if polynomial_degree:
            return self._solve_lanczos(info, gauge, (lx, ly, lz), count, tol, polynomial_degree, lambda_cut)
        return self._solve_cupyx(info, gauge, (lx, ly, lz), count, tol)

    def _solve_lanczos(self, info, gauge, spatial_size, count, tol, degree, lambda_cut):
        from pyquda.pyquda import QudaEigParam, eigensolveQuda
        from pyquda.enum_quda import QudaBoolean, QudaDslashType, QudaEigSpectrumType, QudaEigType

        lx, ly, lz = spatial_size
        dimension = 3 * lx * ly * lz
        krylov = min(max(2 * count, count + 32), dimension - 1)
        if krylov <= count:
            raise ValueError("QUDA Krylov dimension must exceed Ne")
        gauge.gauge_dirac.setVerbosity(0)
        invert = gauge.gauge_dirac.invert_param
        invert.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
        invert.mass = -1
        invert.kappa = 1 / 6
        invert.laplace3D = 3
        eig = QudaEigParam()
        eig.invert_param = invert
        eig.eig_type = QudaEigType.QUDA_EIG_TR_LANCZOS
        eig.use_dagger = QudaBoolean.QUDA_BOOLEAN_FALSE
        eig.use_norm_op = QudaBoolean.QUDA_BOOLEAN_FALSE
        eig.use_pc = QudaBoolean.QUDA_BOOLEAN_FALSE
        eig.compute_gamma5 = QudaBoolean.QUDA_BOOLEAN_FALSE
        eig.spectrum = QudaEigSpectrumType.QUDA_SPECTRUM_SR_EIG
        eig.n_ev = eig.n_conv = count
        eig.n_kr = krylov
        eig.tol = tol
        eig.max_restarts = 10 * dimension // (krylov - count)
        eig.vec_infile = eig.vec_outfile = b""
        eig.use_poly_acc = QudaBoolean.QUDA_BOOLEAN_TRUE
        eig.poly_deg = degree
        eig.a_min = lambda_cut / 6
        eig.a_max = 2
        vectors = self.core.MultiLatticeStaggeredFermion(info, count)
        values = np.zeros(count, dtype=np.complex128)
        gauge.gauge_dirac.loadGauge(gauge)
        try:
            eigensolveQuda(vectors.data_ptrs, values, eig)
        finally:
            gauge.gauge_dirac.freeGauge()
        return (self.cp.asarray(vectors.lexico()).reshape(count, lz, ly, lx, 3),
                self.cp.asarray(values * 6))

    def _solve_cupyx(self, info, gauge, spatial_size, count, tol):
        from cupyx.scipy.sparse import linalg

        cp, core = self.cp, self.core
        lx, ly, lz = spatial_size
        dimension = 3 * lx * ly * lz
        gauge.ensurePureGauge()
        gauge.pure_gauge.loadGauge(gauge)

        def apply(vector):
            columns = vector.reshape(dimension, -1)
            output = cp.empty_like(columns, dtype=cp.complex128)
            for index in range(columns.shape[1]):
                field = core.LatticeStaggeredFermion(info, columns[:, index])
                output[:, index] = gauge.pure_gauge.laplace(field, 3).data.reshape(dimension)
            return output[:, 0] if vector.ndim == 1 else output

        try:
            operator = linalg.LinearOperator((dimension, dimension), matvec=apply, matmat=apply, dtype=cp.complex128)
            values, columns = linalg.eigsh(operator, k=count, which="SA", tol=tol)
        finally:
            gauge.pure_gauge.freeGauge()
        packed = cp.asnumpy(columns.T.reshape(count, 2, 1, lz, ly, lx // 2, 3))
        vectors = np.empty((count, lz, ly, lx, 3), dtype=np.complex128)
        for index in range(count):
            vectors[index] = core.lexico(packed[index], [0, 1, 2, 3, 4]).reshape(lz, ly, lx, 3)
        return cp.asarray(vectors), values * 6
