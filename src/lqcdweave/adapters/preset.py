"""Legacy-style declarations over checked block-oriented readers.

The three Npy classes start from lattice/preset.py in the old EasyDistillation
(72a922d): keep prefix/cfg/suffix and positional arguments, replace FileData
with existing adapters. Paths and shapes never establish scientific identity.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from math import prod

import numpy as np

from lqcdweave.contracts.elemental import ElementalComponent, integer
from lqcdweave.contracts.time_selection import normalize_time_selection
from .elemental import ArrayBaryonElemental, ArrayElemental
from .legacy_fields import DenseFileHandle
from .propagator import ArrayPropagator


_FIELD_AXES = {
    "gauge": ("time", "z", "y", "x", "direction", "color_left", "color_right"),
    "eigenvector": ("time", "mode", "z", "y", "x", "color"),
    "eigenvalues": ("time", "mode"),
    "points": ("time", "point", "xyz"),
    "overlap": ("time", "mode", "point", "color"),
}
_PROPAGATORS = ("vsv", "psv", "vsp", "psp")
_ELEMENTALS = (*ArrayElemental.sector_ranks, "vvv")


def _io_axes(kind, axis_order, time_convention, components, phase_convention):
    """Check caller declarations; an NPY header supplies no scientific axes."""
    if not isinstance(kind, str) or kind not in (*_FIELD_AXES, *_PROPAGATORS, *_ELEMENTALS):
        raise ValueError(f"unsupported NPY kind: {kind!r}")
    if kind in _PROPAGATORS:
        if time_convention not in ("absolute", "relative"):
            raise ValueError("propagators require time_convention='absolute' or 'relative'")
        canonical = ("source_time", "sink_time")
    else:
        if time_convention is not None:
            raise ValueError("time_convention is only valid for propagators")
        canonical = _FIELD_AXES.get(kind, ("component", "momentum", "time"))
    if kind in _ELEMENTALS:
        if components is None or phase_convention is None or axis_order is None:
            raise ValueError("elementals require components, phase_convention and axis_order")
    elif components is not None or phase_convention is not None:
        raise ValueError("components and phase_convention are only valid for elementals")
    order = canonical if axis_order is None else tuple(axis_order)
    if kind == "points" and order != canonical:
        raise ValueError("points require canonical time,point,xyz axes")
    if len(order) != len(canonical) or set(order) != set(canonical):
        raise ValueError(f"axis_order must permute {','.join(canonical)}")
    return canonical, order


def _canonical_array(data, canonical, order):
    if data.ndim < len(canonical):
        raise ValueError("array rank is smaller than axis_order")
    return data.transpose((*[order.index(name) for name in canonical],
                           *range(len(canonical), data.ndim)))


def _check_field(data, kind):
    allowed = "iu" if kind == "points" else "f" if kind == "eigenvalues" else "c"
    if (data.ndim != len(_FIELD_AXES[kind]) or data.dtype.kind not in allowed
            or min(data.shape) < 1):
        raise ValueError(f"invalid {kind} shape/dtype")
    if kind == "gauge" and data.shape[-3:] != (4, 3, 3):
        raise ValueError("gauge direction/color extents must be 4,3,3")
    if kind in ("eigenvector", "points", "overlap") and data.shape[-1] != 3:
        raise ValueError(f"{kind} color/xyz extent must be three")


def _check_npy_header(array, path, shape, dtype):
    if shape is not None and array.shape != tuple(integer(n, "shape", 1) for n in shape):
        raise ValueError("NPY shape differs from declared shape")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise ValueError("NPY dtype differs from declared dtype")
    if path.stat().st_size != array.offset + array.nbytes:
        raise ValueError("NPY file length differs from its header")


def _propagator_selections(selection, source_selection, sink_selection):
    if selection is not None:
        if source_selection is not None or sink_selection is not None:
            raise ValueError("selection cannot be mixed with endpoint selections")
        source_selection = sink_selection = normalize_time_selection(selection)
    return source_selection, sink_selection


def _check_field_selection(value, kind, selection):
    if selection is None:
        return
    record = normalize_time_selection(selection)
    if kind == 'gauge':
        raise ValueError('gauge fields retain the complete physical temporal lattice')
    if value.shape[0] != record.Nt:
        raise ValueError('selection Nt differs from saved field time extent')


def load(path, *, kind, shape=None, dtype=None, axis_order=None,
         time_convention=None, components=None, phase_convention=None, cfg=None,
         selection=None, source_selection=None, sink_selection=None):
    """Load one dense NPY with a declared purpose and checked header/layout.

    Fields return read-only canonical NumPy views. Propagators and elementals
    return the existing block readers. ``shape``/``dtype`` check the on-disk
    array; they never reshape or cast it. See docs/modules/storage/preset.md for each layout.
    Caller-supplied scientific declarations are not recovered from the file.
    """
    canonical, order = _io_axes(kind, axis_order, time_convention, components,
                                phase_convention)
    if kind in _PROPAGATORS:
        source_selection, sink_selection = _propagator_selections(selection, source_selection, sink_selection)
    elif source_selection is not None or sink_selection is not None:
        raise ValueError("endpoint selections apply only to propagators")
    path = Path(path)
    if kind in _FIELD_AXES and cfg is not None:
        raise ValueError("cfg binds sector readers only; field arrays carry no cfg")
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("preset.load requires one dense NPY file")
        stream.seek(0)
        version = np.lib.format.read_magic(stream)
        # Only dense numeric dtypes are accepted here, so their headers are
        # ASCII even in NPY 3.0. Use NumPy's public readers across versions.
        if version == (1, 0):
            disk_shape, _, disk_dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            disk_shape, _, disk_dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version}")
        if disk_dtype.hasobject:
            raise ValueError("preset.load requires non-object NPY data")
        if path.stat().st_size != stream.tell() + prod(disk_shape) * disk_dtype.itemsize:
            raise ValueError("NPY file length differs from its header")
    if kind in _FIELD_AXES:
        raw = DenseFileHandle(path, "", format="npy").load("")
        try:
            _check_npy_header(raw, path, shape, dtype)
            result = _canonical_array(raw, canonical, order)
            _check_field(result, kind)
            _check_field_selection(result, kind, selection)
            return result
        except Exception:
            raw._mmap.close()
            raise
    if kind in _PROPAGATORS:
        result = ArrayPropagator.from_npy(
            {kind: path}, time_convention=time_convention, axis_order=order, cfg=cfg,
            source_selection=source_selection, sink_selection=sink_selection)
    else:
        cls = ArrayBaryonElemental if kind == "vvv" else ArrayElemental
        result = cls.from_npy({kind: path}, components, axis_order=order,
                              phase_convention=phase_convention, cfg=cfg, selection=selection)
    try:
        # Restore a view of the disk axes for header checks; no payload copy.
        raw = _canonical_array(result.sectors[kind], order, canonical)
        _check_npy_header(raw, path, shape, dtype)
        return result
    except Exception:
        result.close()
        raise


def save(path, data, *, kind, axis_order=None, time_convention=None,
         components=None, phase_convention=None, overwrite=False,
         selection=None, source_selection=None, sink_selection=None):
    """Save one dense host NumPy array to exactly ``path`` and return that Path.

    Preserve dtype, endian and input axis order; derive shape from the array.
    Scientific declarations are validated, NOT stored in NPY: retain them and
    pass them again on load. Existing files require ``overwrite=True``.
    """
    canonical, order = _io_axes(kind, axis_order, time_convention, components,
                                phase_convention)
    if kind in _PROPAGATORS:
        source_selection, sink_selection = _propagator_selections(selection, source_selection, sink_selection)
    elif source_selection is not None or sink_selection is not None:
        raise ValueError("endpoint selections apply only to propagators")
    if (not isinstance(data, np.ndarray) or np.ma.isMaskedArray(data)
            or data.dtype.hasobject):
        raise TypeError("save requires a dense non-object NumPy array")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    value = _canonical_array(data, canonical, order)
    if kind in _FIELD_AXES:
        _check_field(value, kind)
        _check_field_selection(value, kind, selection)
    elif kind in _PROPAGATORS:
        ArrayPropagator({kind: value}, time_convention=time_convention,
                        source_selection=source_selection, sink_selection=sink_selection)
    else:
        cls = ArrayBaryonElemental if kind == "vvv" else ArrayElemental
        cls({kind: value}, components, phase_convention=phase_convention, selection=selection)
    path = Path(path)
    if overwrite:
        # Keep the destination intact on write/replace failure. Close the
        # temporary before cleanup, including when np.save raises.
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                    suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, data, allow_pickle=False)
            try:
                temporary.replace(path)
            except PermissionError as error:
                raise PermissionError("cannot replace destination; release active file mappings "
                                      "and check write permissions") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    else:
        # Exclusive creation prevents a check-then-open overwrite race.
        with path.open("xb") as stream:
            try:
                np.save(stream, data, allow_pickle=False)
            except BaseException:
                stream.close()
                path.unlink(missing_ok=True)
                raise
    return path


class _NpySource:
    def __init__(self, prefix, suffix, shape, count, *, dtype=None):
        self.prefix = str(prefix)
        self.suffix = ".npy" if suffix is None else str(suffix)
        self.shape = None if shape is None else tuple(integer(n, "shape", 1) for n in shape)
        if self.shape == ():
            raise ValueError("shape must not be empty")
        self.Ne = None if count is None else integer(count, "Ne", 1)
        self._declared_ne = self.Ne
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._loaded = None
        self._signature = None

    def _open(self, key):
        path = Path(f"{self.prefix}{key}{self.suffix}").resolve()
        stat = path.stat()
        signature = (path, key, stat.st_size, stat.st_mtime_ns)
        if self._loaded is not None and signature == self._signature:
            return None, signature
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if array.dtype.kind != "c":
                raise ValueError("NPY field must contain complex values")
            if self.shape is not None and array.shape != self.shape:
                raise ValueError("NPY shape differs from declared shape")
            if self.dtype is not None and array.dtype != self.dtype:
                raise ValueError("NPY dtype differs from declared dtype")
            if stat.st_size != array.offset + array.nbytes:
                raise ValueError("NPY file length differs from its header")
            return array, signature
        except Exception:
            array._mmap.close()
            raise

    def _modes(self, array, axes):
        counts = tuple(array.shape[axis] for axis in axes)
        if self._declared_ne is not None and any(count != self._declared_ne for count in counts):
            raise ValueError("NPY mode extent differs from declared Ne")
        self.Ne = min(counts)

    def _remember(self, value, signature):
        self._loaded, self._signature = value, signature
        return value


class _Propagator(ArrayPropagator):
    def __getitem__(self, key):
        """Legacy disk-axis indexing returns an independent host array."""
        if not self.sectors:
            raise ValueError("propagator reader is closed")
        return np.array(self.sectors["vsv"][key], copy=True)

    @property
    def shape(self):
        return self.sectors["vsv"].shape

    @property
    def dtype(self):
        return self.sectors["vsv"].dtype


class PerambulatorNpy(_NpySource):
    """Source-first VSV NPY, relative sink time in the legacy preset.

    shape/totNe are optional checks; usedNe at calculation time selects modes.
    """

    def __init__(self, prefix, suffix=".npy", shape=None, totNe=None, *,
                 time_convention="relative", dtype=None, selection=None,
                 source_selection=None, sink_selection=None):
        super().__init__(prefix, suffix, shape, totNe, dtype=dtype)
        if time_convention not in ("relative", "absolute"):
            raise ValueError("declare absolute or relative sink time")
        self.time_convention = time_convention
        self.source_selection, self.sink_selection = _propagator_selections(
            selection, source_selection, sink_selection)

    def load(self, key):
        if self._loaded is not None and not self._loaded.sectors:
            self._loaded = None
        array, signature = self._open(key)
        if array is None:
            return self._loaded
        try:
            value = _Propagator({"vsv": array}, time_convention=self.time_convention,
                                cfg=key, owned_maps=(array,),
                                source_selection=self.source_selection, sink_selection=self.sink_selection)
            self._modes(array, (-2, -1))
            return self._remember(value, signature)
        except Exception:
            array._mmap.close()
            raise


class _Elemental(ArrayElemental):
    def __init__(self, sectors, components, *, mapping, **options):
        super().__init__(sectors, components, **options)
        self._mapping = mapping

    def close(self):
        super().close()
        if self._mapping is not None:
            self._mapping._mmap.close()
            self._mapping = None

    def __getitem__(self, key):
        """Legacy component,momentum,time indexing owns the selected values."""
        if not self.sectors:
            raise ValueError("elemental reader is closed")
        return np.array(self.sectors["v2v"][key], copy=True)

    @property
    def shape(self):
        return self.sectors["v2v"].shape

    @property
    def dtype(self):
        return self.sectors["v2v"].dtype


class ElementalNpy(_NpySource):
    """Dense V/V with component,momentum,time leading axes.

    Default components are opaque integer row labels, never inferred paths.
    Midpoint phase is a preset declaration, not a claim inferred from the file.
    """

    axis_order = ("component", "momentum", "time")

    def __init__(self, prefix, suffix=".npy", shape=None, totNe=None, *,
                 components=None, phase_convention="midpoint", dtype=None, selection=None,
                 momentum_list=None):
        super().__init__(prefix, suffix, shape, totNe, dtype=dtype)
        self.time_selection = None if selection is None else normalize_time_selection(selection)
        self.components = None if components is None else tuple(components)
        self.momenta = None if momentum_list is None else tuple(tuple(p) for p in momentum_list)
        if phase_convention not in ("midpoint", "anchor", "none"):
            raise ValueError("declare midpoint, anchor, or none phase")
        self.phase_convention = phase_convention

    def load(self, key):
        if self._loaded is not None and not self._loaded.sectors:
            self._loaded = None
        array, signature = self._open(key)
        if array is None:
            return self._loaded
        try:
            if array.ndim != 5:
                raise ValueError("elemental NPY must have five axes")
            axes = tuple(self.axis_order.index(name) for name in ("component", "momentum", "time"))
            mapped = array.transpose((*axes, 3, 4))
            components = self.components
            if components is None:
                components = tuple(ElementalComponent(i) for i in range(mapped.shape[0]))
            value = _Elemental({"v2v": mapped}, components, mapping=array,
                               phase_convention=self.phase_convention, cfg=key, selection=self.time_selection,
                               momentum_list=self.momenta)
            self._modes(mapped, (-2, -1))
            return self._remember(value, signature)
        except Exception:
            array._mmap.close()
            raise


class CurrentElementalV2V(ElementalNpy):
    """Legacy time,component,momentum NPY exposed in ElementalNpy order."""

    axis_order = ("time", "component", "momentum")

    def __init__(self, prefix, suffix="_v2v.npy", shape=None, Ne=None, **options):
        super().__init__(prefix, suffix, shape, Ne, **options)


class EigenvectorNpy(_NpySource):
    """Canonical time,mode,z,y,x,color NPY, or declared old flat-volume data."""

    def __init__(self, prefix, suffix=".npy", shape=None, totNe=None, *,
                 lattice_size=None, axis_order=None, dtype=None, selection=None):
        super().__init__(prefix, suffix, shape, totNe, dtype=dtype)
        self.time_selection = None if selection is None else normalize_time_selection(selection)
        self.lattice_size = None if lattice_size is None else tuple(integer(n, "lattice_size", 1) for n in lattice_size)
        if self.lattice_size is not None and len(self.lattice_size) != 4:
            raise ValueError("lattice_size must be (Lx,Ly,Lz,Lt)")
        if self.time_selection is not None and self.lattice_size is not None:
            normalize_time_selection(self.time_selection, Lt=self.lattice_size[3])
        self.Nt = None if self.time_selection is None else self.time_selection.Nt
        self.axis_order = None if axis_order is None else tuple(axis_order)
        if self.Ne is None and self.shape is not None:
            names = (("time", "mode", "z", "y", "x", "color") if len(self.shape) == 6
                     else ("mode", "time", "volume", "color") if len(self.shape) == 4 else ())
            order = names if self.axis_order is None else self.axis_order
            if not names or len(order) != len(names) or set(order) != set(names):
                raise ValueError("eigenvector shape/axis_order must declare six canonical or four flat axes")
            # Shape is a checked caller declaration, not a scientific identity.
            # With no shape/count the cfg-specific header is resolved by load.
            self.Ne = self.shape[order.index("mode")]


    def load(self, key):
        array, signature = self._open(key)
        if array is None:
            return self._loaded
        try:
            names = ("time", "mode", "z", "y", "x", "color")
            if array.ndim == 6:
                order = names if self.axis_order is None else self.axis_order
                if len(order) != 6 or set(order) != set(names):
                    raise ValueError("axis_order must permute time,mode,z,y,x,color")
                value = array.transpose(tuple(order.index(name) for name in names))
            elif array.ndim == 4:
                if self.lattice_size is None:
                    raise ValueError("old flat-volume eigenvectors require lattice_size")
                flat = ("time", "mode", "volume", "color")
                order = ("mode", "time", "volume", "color") if self.axis_order is None else self.axis_order
                if len(order) != 4 or set(order) != set(flat):
                    raise ValueError("flat axis_order must permute time,mode,volume,color")
                mapped = array.transpose(tuple(order.index(name) for name in flat))
                x, y, z, t = self.lattice_size
                nt = t if self.time_selection is None else self.time_selection.Nt
                if mapped.shape[0] != nt or mapped.shape[2:] != (x*y*z, 3):
                    raise ValueError("flat eigenvectors differ from declared lattice_size")
                value = mapped.view()
                # Setting shape rejects layouts requiring a copy before any
                # full-volume allocation, including on NumPy 1.x.
                value.shape = (nt, mapped.shape[1], z, y, x, 3)
            else:
                raise ValueError("eigenvector NPY must have six canonical or four flat axes")
            if value.shape[-1] != 3 or min(value.shape) < 1:
                raise ValueError("invalid eigenvector color/axis extent")
            if self.lattice_size is not None:
                x, y, z, t = self.lattice_size
                nt = t if self.time_selection is None else self.time_selection.Nt
                if (value.shape[0], *value.shape[2:]) != (nt, z, y, x, 3):
                    raise ValueError("eigenvectors differ from declared lattice_size")
            _check_field_selection(value, "eigenvector", self.time_selection)
            self.Nt = value.shape[0]
            self._modes(value, (1,))
            return self._remember(value, signature)
        except Exception:
            array._mmap.close()
            raise

    def read_modes(self, key, time, start, stop):
        value = self.load(key)
        time = integer(time, "time", 0)
        if self.time_selection is not None:
            time = self.time_selection.slot(time)
        start, stop = integer(start, "start", 0), integer(stop, "stop", 1)
        if time >= value.shape[0] or not start < stop <= value.shape[1]:
            raise ValueError("time/mode block is outside eigenvector coverage")
        return value[time, start:stop]


class Dataset:
    """Registered whole-cfg data under one shared dataset directory."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.Ne = None
        self._handle = None
        self._field_key = None
        self.basis_id = self.gauge_id = None

    def load(self, key):
        from lqcdweave.storage.dataset import DatasetReader
        from .meson_dataset import DatasetElemental, DatasetPropagator
        from .production_dataset import DatasetFieldHandle

        self._handle, self._field_key, self.Ne = None, None, None
        self.basis_id = self.gauge_id = None
        reader = DatasetReader(self.directory, cfg=key)
        try:
            document = reader.spec.document
            if document["configuration"]["configuration"] != key:
                raise ValueError("dataset configuration differs from requested cfg")
            kind = document["kind"]
            if kind in ("vsv", "psv", "vsp", "psp"):
                value = DatasetPropagator(reader)
            elif kind in ("elemental_vv", "elemental_projected"):
                value = DatasetElemental(reader)
            elif kind == "glueball":
                from .glueball import ArrayGlueball
                value = ArrayGlueball.from_dataset(reader)
            elif kind in ("laplace_basis", "gauge"):
                value = DatasetFieldHandle(reader)
            else:
                raise ValueError(f"preset.Dataset does not support kind {kind!r}; use its explicit adapter")
            if kind in ("laplace_basis", "gauge"):
                result = value.load(key)
                self._handle = value
                self._field_key = key
                self.Ne = value.Ne
                self.basis_id, self.gauge_id = value.basis_id, value.gauge_id
                return result
            return value
        except Exception:
            reader.close()
            raise

    def read_modes(self, key, time, start, stop):
        if self._handle is None or self._field_key != key:
            self.load(key)
        if self._handle is None:
            raise ValueError("read_modes requires a published basis")
        return self._handle.read_modes(key, time, start, stop)


def resolve_correlator_source(source, configuration):
    """Load supported declarations; explicit readers keep their existing binding.

    Format dispatch stays in adapters. Unknown sources are not implicitly
    loaded: callers can supply their reader or a custom DiagramData adapter.
    """
    if isinstance(source, (PerambulatorNpy, ElementalNpy, Dataset)):
        if configuration is None:
            raise ValueError("preset data sources require configuration=cfg")
        return source.load(configuration)
    return source


__all__ = ["PerambulatorNpy", "ElementalNpy", "CurrentElementalV2V", "EigenvectorNpy", "Dataset",
           "resolve_correlator_source", "load", "save"]
