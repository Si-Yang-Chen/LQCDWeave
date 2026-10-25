"""Owned gauge preparation and reusable storage for elemental producers."""

from collections import OrderedDict
import numpy as np

from lqcdweave.contracts.elemental import integer
from lqcdweave.production.gauge import project_su3, stout_smear
from .transport import momentum_phase


class ElementalWorkspace:
    """One producer's gauge, geometric phases and named complex buffers.

    No vector inputs, operator definitions or contraction algorithms live here.
    Links and phases are borrowed read-only by convention; buffers are writable
    scratch. None may be retained across the owner's next load or calculation.
    An instance must not be shared by concurrently active producers.
    """

    def __init__(self, latt_size, gauge_field, *, array_module=np,
                 phase_convention="midpoint"):
        self.latt_size = tuple(integer(n, "extent", 1) for n in latt_size)
        if len(self.latt_size) != 4:
            raise ValueError("latt_size must be (Lx,Ly,Lz,Lt)")
        if phase_convention not in ("midpoint", "anchor", "none"):
            raise ValueError("invalid phase convention")
        self.gauge_field = gauge_field
        self.xp = array_module
        self.phase_convention = phase_convention
        self._links = None
        self._buffers = {}
        self._phase_cache = OrderedDict()

    def invalidate(self):
        """Prevent calculation after a failed load or partial gauge transform."""
        self._links = None

    def load(self, key):
        """Copy handle data before any other input can reuse its scratch memory."""
        self.invalidate()
        lx, ly, lz, lt = self.latt_size
        gauge = self.gauge_field.load(key)[:]
        if gauge.shape != (lt, lz, ly, lx, 4, 3, 3):
            raise ValueError("gauge must have shape (Lt,Lz,Ly,Lx,4,3,3)")
        self._links = self.xp.array(gauge.transpose(4, 0, 1, 2, 3, 5, 6),
                                    dtype=complex, copy=True, order="C")
        return self

    @property
    def links(self):
        """Borrow direction-major (4,Lt,Lz,Ly,Lx,3,3) links; do not mutate."""
        if self._links is None:
            raise ValueError("load inputs before calculation")
        return self._links

    def time(self, t):
        t = integer(t, "time", 0)
        if self._links is None:
            raise ValueError("load inputs before calculation")
        if t >= self.latt_size[3]:
            raise ValueError("time is outside lattice")
        return t

    def buffer(self, name, shape):
        """Borrow reusable complex128 storage, overwritten by its consumer."""
        if name not in self._buffers or self._buffers[name].shape != shape:
            self._buffers[name] = self.xp.empty(shape, dtype=complex)
        return self._buffers[name]

    def phase(self, momentum, displacement=(0, 0, 0), convention=None):
        """Borrow a geometric phase; independent of the loaded gauge field."""
        convention = self.phase_convention if convention is None else convention
        displacement = tuple(displacement[:3])
        key = (tuple(momentum), displacement, convention)
        if key in self._phase_cache:
            self._phase_cache.move_to_end(key)
            return self._phase_cache[key]
        phase = momentum_phase(self.latt_size, momentum, displacement, convention,
                               array_module=self.xp)
        self._phase_cache[key] = phase
        if len(self._phase_cache) > 64:
            self._phase_cache.popitem(last=False)
        return phase

    def stout_smear(self, nstep, rho, *, matrix_batch_size=4096):
        self.time(0)
        try:
            self._links = stout_smear(self._links, nstep, rho,
                matrix_batch_size=matrix_batch_size, array_module=self.xp, inplace=True)
        except Exception:
            self.invalidate()
            raise

    def project_SU3(self, *, matrix_batch_size=4096):
        self.time(0)
        try:
            self._links[:3] = project_su3(self._links[:3],
                matrix_batch_size=matrix_batch_size, array_module=self.xp)
        except Exception:
            self.invalidate()
            raise
