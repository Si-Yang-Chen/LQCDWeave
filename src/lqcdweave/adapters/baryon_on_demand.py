"""Explicit runtime production of selected baryon elemental blocks.

This reader borrows stable, already prepared gauge/basis inputs. Describing a
block never loads fields; reading it invokes the existing public producer.
"""

import numpy as np

from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.elemental import (
    BaryonComponent, elemental_operator_rows, integer,
)
from lqcdweave.production.elemental.api import BaryonElementalGenerator
from .selection import resolve_selections


_AXES = ("mode0", "mode1", "mode2")


def _is_array(value):
    # Field handles may advertise a shape too; their explicit IO protocol wins.
    if any(callable(getattr(value, name, None))
           for name in ("load", "read_time", "read_modes")):
        return False
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and callable(getattr(value, "__getitem__", None)))


class _SelectedModes:
    """Translate local contiguous tiles to ordered input positions.

    Copy each borrowed run immediately: providers may reuse the same buffer.
    The producer supplies independent scratch for the three legs.
    """

    def __init__(self, source, positions, time, xp):
        self.source, self.positions, self.time, self.xp = source, positions, time, xp
        self.Ne = len(positions)

    def read_modes(self, key, _time, start, stop):
        positions = self.positions[start:stop]
        if _is_array(self.source):
            return self.xp.asarray(self.source[self.time, list(positions)])
        runs, offset = [], 0
        while offset < len(positions):
            end = offset + 1
            while end < len(positions) and positions[end] == positions[end - 1] + 1:
                end += 1
            raw = self.source.read_modes(key, self.time, positions[offset], positions[end - 1] + 1)
            if raw.shape[0] != end - offset:
                raise ValueError("mode provider returned an incorrect run length")
            runs.append(self.xp.array(raw, dtype=complex, copy=True))
            offset = end
        return runs[0] if len(runs) == 1 else self.xp.concatenate(runs, axis=0)


class OnDemandBaryonElemental:
    """A Baryon reader that computes only the requested VVV block.

    Arrays use gauge (Lt,z,y,x,4,3,3), basis (Lt,Ne,z,y,x,3).
    Alternatively gauge exposes read_time(cfg,t) or load(cfg), and each basis
    exposes read_modes(cfg,t,start,stop). Handles require lattice_size and
    mode_counts. basis_bindings is three (actual ID, ordered labels) pairs,
    with None for unknown identities. Inputs remain stable until clear_cache
    or load; construct a fresh reader for another configuration or recipe.

    cache_bytes bounds one retained result, not total workspace/input memory.
    Numerical execution is synchronous; this reader is not thread safe.
    """

    background_read_safe = False

    def __init__(self, gauge_field, eigenvectors, momenta, operator_table, *,
                 lattice_size=None, mode_counts=None, basis_bindings=None, cfg=None,
                 array_module=np, phase_convention="anchor", path_boundary="periodic",
                 ne_block_size=16, spatial_block_size=256, cache_bytes=64 * 1024**2,
                 transport_cache_bytes=64 * 1024**2):
        self.xp, self.gauge_field, self.cfg = array_module, gauge_field, cfg
        shared = _is_array(eigenvectors) or callable(getattr(eigenvectors, "read_modes", None))
        self.eigenvectors = (eigenvectors,) * 3 if shared else tuple(eigenvectors)
        if len(self.eigenvectors) != 3:
            raise ValueError("one shared basis or three ordered bases required")
        if lattice_size is None:
            if not _is_array(gauge_field) or len(gauge_field.shape) != 7:
                raise ValueError("gauge handles require explicit lattice_size")
            lt, lz, ly, lx = gauge_field.shape[:4]
            lattice_size = (lx, ly, lz, lt)
        self.lattice_size = tuple(integer(n, "lattice extent", 1) for n in lattice_size)
        if len(self.lattice_size) != 4:
            raise ValueError("lattice_size must be (Lx,Ly,Lz,Lt)")
        lx, ly, lz, self.Lt = self.lattice_size
        if _is_array(gauge_field):
            if tuple(gauge_field.shape) != (self.Lt, lz, ly, lx, 4, 3, 3):
                raise ValueError("gauge array shape differs from lattice_size")
        elif not any(callable(getattr(gauge_field, name, None)) for name in ("read_time", "load")):
            raise TypeError("gauge requires an array, read_time or load")
        if mode_counts is None:
            if not all(_is_array(v) for v in self.eigenvectors):
                raise ValueError("basis handles require explicit mode_counts")
            mode_counts = tuple(v.shape[1] for v in self.eigenvectors)
        if isinstance(mode_counts, (int, np.integer)):
            mode_counts = (mode_counts,) * 3
        self.mode_counts = tuple(integer(n, "mode count", 1) for n in mode_counts)
        if len(self.mode_counts) != 3:
            raise ValueError("mode_counts must contain three counts")
        for v, n in zip(self.eigenvectors, self.mode_counts):
            if _is_array(v):
                if len(v.shape) != 6 or v.shape[0] != self.Lt or v.shape[1] < n or tuple(v.shape[2:]) != (lz, ly, lx, 3):
                    raise ValueError("basis array shape differs from lattice/mode coverage")
            elif not callable(getattr(v, "read_modes", None)):
                raise TypeError("basis handles require read_modes")
        if phase_convention not in ("anchor", "none") or path_boundary not in ("periodic", "open"):
            raise ValueError("baryon requires anchor/none phase and periodic/open paths")
        self.phase_convention, self.path_boundary = phase_convention, path_boundary
        self.momentum_list = tuple(tuple(integer(p, "momentum") for p in m) for m in momenta)
        if not self.momentum_list or any(len(m) != 3 for m in self.momentum_list):
            raise ValueError("nonempty XYZ momenta required")
        self.num_momentum = len(self.momentum_list)
        rows = elemental_operator_rows(operator_table, rank=3)
        if any(not isinstance(c, BaryonComponent) for c, _ in rows):
            raise TypeError("operator_table keys must be BaryonComponent objects")
        if any(step >= 6 for _, terms in rows for paths, _ in terms
               for path in paths for step in path.steps):
            raise ValueError("on-demand baryon supports spatial paths only")
        self.components = tuple(c for c, _ in rows)
        self._tables = {c: ({"component": c, "terms": tuple(
            {"paths": paths, "coefficient": coefficient} for paths, coefficient in terms)},)
            for c, terms in rows}
        self.ne_block_size = ne_block_size
        counts = (ne_block_size,) * 3 if isinstance(ne_block_size, (int, np.integer)) else tuple(ne_block_size)
        if len(counts) != 3:
            raise ValueError("ne_block_size requires three counts")
        self.ne_block_size = tuple(integer(n, "ne_block_size", 1) for n in counts)
        self.spatial_block_size = integer(spatial_block_size, "spatial_block_size", 1)
        self.cache_bytes = integer(cache_bytes, "cache_bytes", 0)
        self.transport_cache_bytes = integer(transport_cache_bytes, "transport_cache_bytes", 0)
        bindings = (None,) * 3 if basis_bindings is None else tuple(basis_bindings)
        if len(bindings) != 3:
            raise ValueError("basis_bindings requires three entries")
        # Delegate scalar-label and actual-ID validation to the neutral contract.
        known = {axis: b for axis, b in zip(_AXES, bindings) if b is not None}
        metadata = TensorAxisMetadata(_AXES, self.mode_counts, np.complex128,
                                      None, "on-demand-basis", axis_bindings=known)
        self.basis_bindings = tuple(metadata.axis_bindings.get(axis) for axis in _AXES)
        self.data_identity = None  # Runtime results are not registered datasets.
        self._closed = False
        self.clear_cache()

    def clear_cache(self):
        """Discard the single retained result after an input update."""
        self._cached_key = self._cached_value = None

    def load(self, key):
        """Validate the bound configuration and discard cached numeric values."""
        if self._closed:
            raise ValueError("elemental reader is closed")
        if key != self.cfg:
            raise ValueError("construct a fresh on-demand reader for another configuration")
        self.clear_cache()
        return self

    def _selection(self, component, momentum, t, sector, usedNe, mode_indices, usedNp, point_indices):
        if self._closed:
            raise ValueError("elemental reader is closed")
        momentum, t = integer(momentum, "momentum", 0), integer(t, "time", 0)
        if sector != "vvv" or component not in self._tables or momentum >= self.num_momentum or t >= self.Lt:
            raise KeyError("missing on-demand baryon coverage")
        selections = resolve_selections(_AXES, self.mode_counts, usedNe=usedNe,
                                        mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices)
        positions = tuple(tuple(range(n))[s] if isinstance(s, slice) else s
                          for axis, n in zip(_AXES, self.mode_counts)
                          for s in (selections.get(axis, slice(None)),))
        return momentum, t, positions

    def describe_block(self, component, momentum, t, sector="vvv", *, usedNe=None,
                       mode_indices=None, usedNp=None, point_indices=None):
        momentum, t, positions = self._selection(component, momentum, t, sector, usedNe, mode_indices, usedNp, point_indices)
        bindings = {axis: (binding[0], tuple(binding[1][p] for p in selected))
                    for axis, binding, selected in zip(_AXES, self.basis_bindings, positions)
                    if binding is not None}
        return TensorAxisMetadata(_AXES, tuple(map(len, positions)), np.complex128,
                                  data_identity=None,
                                  selection_identity=repr((component, momentum, t, positions)),
                                  axis_bindings=bindings)

    def block(self, component, momentum, t, sector="vvv", *, usedNe=None,
              mode_indices=None, usedNp=None, point_indices=None):
        momentum, t, positions = self._selection(component, momentum, t, sector, usedNe, mode_indices, usedNp, point_indices)
        key = (component, momentum, t, positions)
        if key == self._cached_key:
            return self._cached_value
        try:
            if _is_array(self.gauge_field):
                gauge = self.gauge_field[t]
            elif callable(getattr(self.gauge_field, "read_time", None)):
                gauge = self.gauge_field.read_time(self.cfg, t)
            else:
                gauge = self.gauge_field.load(self.cfg)[t]
            # Own gauge before a basis provider can reuse a shared input buffer.
            gauge = self.xp.array(gauge, dtype=complex, copy=True)[None]
            if not bool(self.xp.isfinite(gauge).all()):
                raise ValueError("gauge field must be finite")
            vectors = tuple(_SelectedModes(v, selected, t, self.xp)
                            for v, selected in zip(self.eigenvectors, positions))
            producer = BaryonElementalGenerator(
                gauge, vectors, (self.momentum_list[momentum],), self._tables[component],
                Ne=tuple(map(len, positions)), array_module=self.xp,
                lattice_size=(*self.lattice_size[:3], 1), stream_eigenvectors=True,
                ne_block_size=self.ne_block_size, spatial_block_size=self.spatial_block_size,
                phase_convention=self.phase_convention, path_boundary=self.path_boundary,
                cache_bytes=self.transport_cache_bytes,
            ).load(self.cfg)
            value = self.xp.array(producer.calc_path_blocks(0)["vvv"][0, 0], copy=True)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        except Exception:
            self.clear_cache()
            raise
        if value.nbytes <= self.cache_bytes:
            self._cached_key, self._cached_value = key, value
        return value

    def close(self):
        """Release cached output, leaving borrowed field providers open."""
        self.clear_cache()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
