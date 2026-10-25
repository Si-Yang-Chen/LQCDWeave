"""Low modes of the three-dimensional gauge-covariant Laplacian."""

from copy import deepcopy
from numbers import Integral

import numpy as np

from lqcdweave.contracts.laplace import LaplaceBackend
from lqcdweave.contracts.time_selection import TimeSelectionRecord, normalize_time_selection
from ..gauge import stout_smear
from .laplacian import spatial_laplacian


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value):
    """Read a NumPy or CuPy scalar without changing the active array module."""
    return value.item()


class LaplaceEigenvectorGenerator:
    """Solve one time slice at a time; arrays use the caller's array module.

    The gauge handle must return (Lt,Lz,Ly,Lx,4,3,3). ``calc`` returns
    (vectors[Ne,Lz,Ly,Lx,3], values[Ne]). All times are available after load;
    callers normally iterate ``range(Lt)``. Required ``backend`` selects
    ``numpy``, ``cupy``, ``quda`` or a caller-provided LaplaceBackend instance.
    Custom solvers use NumPy unless an explicit array_module is supplied.
    """

    def __init__(self, latt_size, gauge_field, Ne, tol, *,
                 backend: str | LaplaceBackend, array_module=None, selection=None,
                 existing_basis=None):
        sizes = tuple(_integer(x, "lattice extent", 1) for x in latt_size)
        if len(sizes) != 4:
            raise ValueError("latt_size must be (Lx,Ly,Lz,Lt)")
        dimension = 3 * sizes[0] * sizes[1] * sizes[2]
        count = _integer(Ne, "Ne", 1)
        if count >= dimension - 1:
            raise ValueError("Ne must be at most 3*Lx*Ly*Lz - 2")
        if not callable(getattr(gauge_field, "load", None)):
            raise ValueError("gauge_field must provide load(key)")
        tolerance = float(tol)
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("tol must be finite and in (0,1)")
        from lqcdweave.adapters.laplace import resolve_laplace_backend
        array_module, backend = resolve_laplace_backend(backend, array_module)
        if getattr(array_module, "__name__", None) not in ("numpy", "cupy"):
            raise ValueError("array_module must be numpy or cupy")
        if backend is not None and any(not callable(getattr(backend, name, None))
                                       for name in ("load", "invalidate", "stout_smear", "solve")):
            raise ValueError("backend must implement the LaplaceBackend boundary")
        self.latt_size, self.gauge_field = sizes, gauge_field
        self.time_selection = (TimeSelectionRecord(sizes[3], tuple(range(sizes[3])))
                               if selection is None else normalize_time_selection(selection, Lt=sizes[3]))
        self.Lt, self.Nt = sizes[3], self.time_selection.Nt
        self.times = self.time_selection.times
        self.Ne, self.tol = count, tolerance
        self.xp, self.backend = array_module, backend
        if existing_basis is not None and not any(callable(getattr(existing_basis, name, None))
                                                  for name in ('load', 'read_modes')):
            raise ValueError('existing_basis must provide load or read_modes')
        self.existing_basis = existing_basis
        self._existing_data = None
        self._extension = None
        self._links = None
        self._key = self._gauge_id = None
        self._transforms = []

    def load(self, key):
        self._existing_data = None
        self._extension = None
        self._links = None
        self._key = self._gauge_id = None
        self._transforms = []
        lx, ly, lz, lt = self.latt_size
        if self.backend is not None:
            self.backend.invalidate()
            try:
                links = self.xp.asarray(self.backend.load(self.gauge_field, key, self.latt_size))
                if links.shape != (4, lt, lz, ly, lx, 3, 3) or links.dtype.kind != "c":
                    raise ValueError("backend returned an invalid gauge layout/dtype")
            except Exception:
                self.backend.invalidate()
                raise
        else:
            payload = self.gauge_field.load(key)
            raw = self.xp.asarray(payload[:])
            if raw.shape != (lt, lz, ly, lx, 4, 3, 3) or raw.dtype.kind != "c":
                raise ValueError(
                    "gauge must have complex shape (Lt,Lz,Ly,Lx,4,3,3); "
                    f"got shape={raw.shape}, dtype={raw.dtype}"
                )
            # Own the direction-first complex array in one allocation.
            links = raw.transpose(4, 0, 1, 2, 3, 5, 6).astype(self.xp.complex128, copy=True)
        if self.existing_basis is not None:
            reader = self.existing_basis
            if not callable(getattr(reader, 'read_modes', None)):
                self._existing_data = reader.load(key)
        self._links = links
        self._key = key
        self._gauge_id = deepcopy(getattr(self.gauge_field, "gauge_id", None))
        return self

    def _require_loaded(self):
        if self._links is None:
            raise RuntimeError("load a configuration before transforming or solving")

    def project_SU3(self):
        """Polar-project spatial links to SU(3); temporal links are untouched."""
        self._require_loaded()
        self._extension = None
        xp = self.xp
        try:
            left, _, right = xp.linalg.svd(self._links[:3])
            unitary = left @ right
            determinant = xp.linalg.det(unitary)
            projected = unitary * xp.exp(-1j * xp.angle(determinant) / 3)[..., None, None]
            if not bool(_scalar(xp.isfinite(projected).all())):
                raise ValueError("SU(3) projection produced nonfinite links")
            self._links[:3] = projected
            if self.backend is not None:
                self.backend.invalidate()
        except Exception:
            self._links = None
            if self.backend is not None:
                self.backend.invalidate()
            raise
        self._transforms.append({"kind": "project_SU3"})
        return self

    def stout_smear(self, nstep, rho):
        """Apply spatial stout steps to the currently loaded links."""
        self._require_loaded()
        self._extension = None
        nstep = _integer(nstep, "nstep")
        rho = float(rho)
        if not np.isfinite(rho):
            raise ValueError("rho must be finite")
        if nstep == 0 or rho == 0:
            return self
        if self.backend is None and self.xp.__name__ != "numpy":
            raise NotImplementedError(
                "CuPy stout smearing requires an explicit QudaLaplaceBackend"
            )
        try:
            if self.backend is None:
                # The shared kernel retains the time axis, so its spatial roll
                # axes are 3 - direction (the old per-time view used 2 - direction).
                self._links = stout_smear(
                    self._links, nstep, rho, array_module=np, inplace=True
                )
            else:
                updated = self.xp.asarray(self.backend.stout_smear(
                    self._links, self.latt_size, nstep, rho))
                if updated.shape != self._links.shape or updated.dtype.kind != "c":
                    raise ValueError("backend stout smearing returned an unexpected gauge layout/dtype")
                updated[3] = self._links[3]
                self._links = updated
        except Exception:
            self._links = None
            if self.backend is not None:
                self.backend.invalidate()
            raise
        self._transforms.append({"kind": "stout", "nstep": nstep, "rho": rho})
        return self

    def _solve_array(self, links):
        xp = self.xp
        if xp.__name__ == "numpy":
            from scipy.sparse import linalg
        else:
            from cupyx.scipy.sparse import linalg
        dimension = 3 * self.latt_size[0] * self.latt_size[1] * self.latt_size[2]
        shape = (self.latt_size[2], self.latt_size[1], self.latt_size[0], 3)

        def apply(v):
            fields = v.reshape(*shape, -1)
            result = spatial_laplacian(fields, links, xp).reshape(dimension, -1)
            return result[:, 0] if v.ndim == 1 else result

        operator = linalg.LinearOperator((dimension, dimension), matvec=apply, matmat=apply, dtype=xp.complex128)
        values, columns = linalg.eigsh(operator, k=self.Ne, which="SA", tol=self.tol)
        return columns.T.reshape(self.Ne, *shape), values

    def calc(self, t, apply_renorm_phase=True, polynomial_degree=0, lambda_cut=0.0):
        self._extension = None
        self._require_loaded()
        t = _integer(t, "time")
        if t >= self.latt_size[3]:
            raise ValueError("time is outside lattice")
        if not isinstance(apply_renorm_phase, (bool, np.bool_)):
            raise ValueError("apply_renorm_phase must be boolean")
        polynomial_degree = _integer(polynomial_degree, "polynomial_degree")
        lambda_cut = float(lambda_cut)
        if not np.isfinite(lambda_cut) or lambda_cut < 0:
            raise ValueError("lambda_cut must be finite and nonnegative")
        if (polynomial_degree > 0) != (lambda_cut > 0):
            raise ValueError("polynomial_degree and lambda_cut must both be positive or both zero")
        if lambda_cut >= 12:
            raise ValueError("lambda_cut must be less than the Laplacian upper bound 12")
        if self.existing_basis is not None:
            if polynomial_degree:
                raise NotImplementedError('basis extension does not support Chebyshev acceleration')
            return self._extend(t, apply_renorm_phase)
        if polynomial_degree and self.backend is None:
            raise NotImplementedError("Chebyshev acceleration requires an explicit QUDA backend")
        links = self._links[:3, t]
        if self.backend is None:
            vectors, values = self._solve_array(links)
        else:
            vectors, values = self.backend.solve(
                self._links, self.latt_size, t, self.Ne, self.tol, polynomial_degree, lambda_cut
            )
        values = self.xp.asarray(values).real
        order = self.xp.argsort(values)
        values, vectors = values[order], self.xp.asarray(vectors)[order].astype(self.xp.complex128, copy=True)
        if apply_renorm_phase:
            flat = vectors.reshape(self.Ne, -1)
            phase = self.xp.angle(flat[:, 0])
            vectors *= self.xp.exp(-1j * phase)[:, None, None, None, None]
        return vectors, values

    def _extend(self, t, apply_renorm_phase):
        from .extension import extend_basis
        solve = None
        if self.backend is not None:
            method = getattr(self.backend, 'solve_complement', None)
            if not callable(method):
                raise NotImplementedError('this backend does not support solve_complement for basis extension')
            def solve(prefix, count):
                return method(self._links, self.latt_size, t, count, self.tol, prefix)
        reader = self.existing_basis
        selection = normalize_time_selection(getattr(reader, 'time_selection', None), Lt=self.Lt)
        slot = selection.slot(t)
        if callable(getattr(reader, 'read_modes', None)):
            count = _integer(getattr(reader, 'Ne', None), 'existing basis Ne', 1)
            if count > self.Ne:
                raise ValueError('existing basis contains more modes than requested Ne')
            prefix = reader.read_modes(self._key, t, 0, count)
        else:
            data = self._existing_data
            if len(data.shape) != 6 or data.shape[0] != selection.Nt:
                raise ValueError('existing basis time axis differs from its declared time selection')
            prefix = data[slot]
        declared_count = getattr(reader, 'Ne', None)
        if declared_count is not None:
            declared_count = _integer(declared_count, 'existing basis Ne', 1)
            shape = getattr(prefix, 'shape', ())
            if not shape or shape[0] != declared_count:
                raise ValueError('existing basis reader returned a different mode count')
        vectors, values, record = extend_basis(self._links[:3, t], prefix, self.Ne,
            self.tol, self.xp, solve=solve, fix_phase=apply_renorm_phase)
        labels = getattr(reader, 'mode_labels', None)
        if labels is not None:
            labels = tuple(value.item() if hasattr(value, 'item') else value for value in labels)
            if len(labels) != record['prefix_count']:
                raise ValueError('existing basis mode labels differ from its actual mode count')
        record.update(source_basis_id=deepcopy(getattr(reader, 'basis_id', None)),
                      source_mode_labels=deepcopy(labels),
                      physical_time=t)
        self._extension = record
        return vectors, values

    def iter_slices(self, times=None, *, selection=None, apply_renorm_phase=True,
                    polynomial_degree=0, lambda_cut=0.0):
        """Lazily yield ``(physical_t, vectors, values)`` in the selected order.

        With no selection configured, every physical time is visited. Explicit
        times or a selection override retain their order and cannot be mixed.
        Each result is owned, as in
        ``calc``; no output stack or saved state is created. Load and transform
        the gauge before iterating, not between yields. Errors propagate at the
        affected iteration, without switching solvers or reporting completion.
        """
        if times is not None and selection is not None:
            raise ValueError("times and selection are mutually exclusive")
        selected = (normalize_time_selection(selection, Lt=self.Lt).times
                    if selection is not None else self.times if times is None else times)
        for t in selected:
            vectors, values = self.calc(
                t, apply_renorm_phase=apply_renorm_phase,
                polynomial_degree=polynomial_degree, lambda_cut=lambda_cut,
            )
            yield t, vectors, values

    def describe(self):
        return {"latt_size": self.latt_size, "Ne": self.Ne, "tol": self.tol,
                "key": self._key, "gauge_id": deepcopy(self._gauge_id),
                "transforms": deepcopy(self._transforms), "loaded": self._links is not None,
                "time_selection": self.time_selection.document, "Nt": self.Nt,
                "extension": deepcopy(self._extension)}


