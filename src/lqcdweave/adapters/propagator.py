"""Read existing VSV, PSV, VSP and PSP arrays with explicit time semantics."""

from pathlib import Path
from numbers import Integral
from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.time_selection import normalize_time_selection
from .selection import resolve_selections, host_arrays_resident
from .selection import select_axes, select_slab_axes, times, _slab_array_module


_TAILS = {
    left + 's' + right:
        (('sink_point', 'sink_color') if left == 'p' else ('sink_mode',))
        + (('source_point', 'source_color') if right == 'p' else ('source_mode',))
    for left in ('v', 'd', 'p') for right in ('v', 'd', 'p')
}


def _axis_bindings(sector, names, raw_shape, selections, endpoints):
    bindings = {}
    for side, kind in (('sink', sector[0]), ('source', sector[2])):
        entry = endpoints.get((sector, side))
        if entry is None:
            continue
        axis = side + ('_point' if kind == 'p' else '_mode')
        positions = selections.get(axis, slice(None))
        if isinstance(positions, slice):
            positions = range(*positions.indices(raw_shape[names.index(axis)]))
        bindings[axis] = (entry[0], tuple(entry[1][index] for index in positions))
    return bindings


def _time_sequence(values, extent, name):
    if isinstance(values, (bool, np.bool_)):
        raise TypeError(f"{name} must contain integer time coordinates")
    if isinstance(values, Integral):
        values = (values,)
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain integer time coordinates")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or an iterable of integers") from exc
    if not result:
        raise ValueError(f"{name} cannot be empty")
    normalized = []
    for value in result:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integer time coordinates")
        value = int(value)
        if not 0 <= value < extent:
            raise ValueError(f"{name} must lie in [0, {extent})")
        normalized.append(value)
    return tuple(normalized)


def _single_time(value, extent, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer time coordinate")
    value = int(value)
    if not 0 <= value < extent:
        raise ValueError(f"{name} must lie in [0, {extent})")
    return value


class ArrayPropagator:
    """Source-time-first data; the second axis is absolute or relative sink time.

    A sector may be one dense array or a list of source-time files with the
    source axis omitted. Alternatively a sector may map explicit
    ``(sink_slot, source_slot)`` pairs to blocks, with absolute time convention
    and explicit source/sink selections. Absent pairs have no readable header.
    Views borrow the arrays until close(). All reads use
    integer slots. Optional endpoint records retain the ordered physical mapping;
    relative layouts use cyclic slot difference and require matching records.
    With no records, slots coincide with canonical physical times.
    """

    @property
    def host_resident(self):
        return type(self) is ArrayPropagator and host_arrays_resident(self.sectors)

    @property
    def background_read_safe(self):
        """Host arrays can be read concurrently while the reader stays open."""
        return all(isinstance(value, np.ndarray)
                   for data in self.sectors.values()
                   for value in (data.values() if isinstance(data, Mapping) else
                                 data if isinstance(data, (tuple, list)) else (data,))
                   if value is not None)

    def read_slab_in_background(self, *args, **kwargs):
        if not self.background_read_safe:
            raise TypeError("background propagator reads require host NumPy arrays")
        return self.read_slab(*args, **kwargs)

    def __init__(self, sectors, *, time_convention, cfg=None, owned_maps=(),
                 data_identity=None, source_selection=None, sink_selection=None,
                 basis_endpoints=None):
        if time_convention not in ("absolute", "relative"):
            raise ValueError("declare absolute or relative sink time")
        self.time_convention, self.cfg = time_convention, cfg
        self.data_identity = data_identity
        self.sectors, self._maps, self._closed = dict(sectors), list(owned_maps), False
        # Freeze pair membership while continuing to borrow numerical arrays.
        self.sectors = {key: dict(value) if isinstance(value, Mapping) else value
                        for key, value in self.sectors.items()}
        if not self.sectors or set(self.sectors) - _TAILS.keys():
            raise ValueError("unknown or empty propagator sectors")
        explicit_selection = source_selection is not None or sink_selection is not None
        declared = source_selection if source_selection is not None else sink_selection
        physical_lt = None if declared is None else normalize_time_selection(declared).Lt
        source_record = (None if not explicit_selection else
                         normalize_time_selection(source_selection, Lt=physical_lt))
        sink_record = (None if not explicit_selection else
                       normalize_time_selection(sink_selection, Lt=physical_lt))
        if time_convention == 'relative' and source_record != sink_record:
            raise ValueError("relative slot layout requires identical source/sink selections")
        extents = None
        for sector, data in self.sectors.items():
            split = isinstance(data, (tuple, list))
            if isinstance(data, Mapping):
                if not explicit_selection or time_convention != 'absolute':
                    raise ValueError("pair blocks require explicit selections and absolute time")
                if not data:
                    raise ValueError("empty propagator pair blocks")
                if any(not isinstance(pair, tuple) or len(pair) != 2 or
                       type(pair[0]) is not int or type(pair[1]) is not int or
                       not 0 <= pair[0] < sink_record.Nt or
                       not 0 <= pair[1] < source_record.Nt for pair in data):
                    raise ValueError("propagator pair keys require (sink_slot, source_slot) within selections")
                sample = next(iter(data.values()))
                if any(a.shape != sample.shape or a.dtype != sample.dtype for a in data.values()):
                    raise ValueError("pair blocks have inconsistent shape/dtype")
                shape = (source_record.Nt, sink_record.Nt, *sample.shape)
                dtype = sample.dtype
            elif split:
                if not data or not any(a is not None for a in data):
                    raise ValueError("empty source-time files")
                sample = next(a for a in data if a is not None)
                shape = (len(data), *sample.shape)
                if any(a is not None and (a.shape != sample.shape or a.dtype != sample.dtype) for a in data):
                    raise ValueError("source-time files have inconsistent shape/dtype")
                dtype = sample.dtype
            else:
                shape, dtype = data.shape, data.dtype
            if len(shape) != 4 + len(_TAILS[sector]) or min(shape) < 1 or dtype.kind != "c":
                raise ValueError(f"invalid {sector} shape/dtype")
            if shape[2:4] != (4, 4):
                raise ValueError("propagator spin axes must both have size four")
            for index, name in enumerate(_TAILS[sector], 4):
                if name.endswith("color") and shape[index] != 3:
                    raise ValueError("point color axes must have size three")
            if extents is not None and shape[:2] != extents:
                raise ValueError("propagator sectors have different time coverage")
            extents = tuple(shape[:2])
        if explicit_selection:
            if extents != (source_record.Nt, sink_record.Nt):
                raise ValueError("propagator time shape differs from source/sink selections")
        elif extents[0] != extents[1]:
            raise ValueError("asymmetric time axes require explicit source/sink selections")
        self.source_Nt, self.sink_Nt = extents
        self.basis_endpoints = {} if basis_endpoints is None else dict(basis_endpoints)
        for key, binding in self.basis_endpoints.items():
            if (not isinstance(key, tuple) or len(key) != 2
                    or key[0] not in self.sectors or key[1] not in ('sink', 'source')
                    or not isinstance(binding, (tuple, list)) or len(binding) != 2
                    or not isinstance(binding[0], str) or not binding[0]):
                raise ValueError('basis_endpoints must bind (sector,sink|source) to actual ID and labels')
            sector, side = key
            data = self.sectors[sector]
            if isinstance(data, Mapping):
                tail = next(iter(data.values())).shape
            else:
                sample = next(a for a in data if a is not None) if isinstance(data, (tuple, list)) else data
                tail = sample.shape[1:] if isinstance(data, (tuple, list)) else sample.shape[2:]
            names = ('sink_spin', 'source_spin', *_TAILS[sector])
            axis = side + ('_point' if sector[0 if side == 'sink' else 2] == 'p' else '_mode')
            if len(binding[1]) != tail[names.index(axis)]:
                raise ValueError('propagator endpoint labels differ from the numerical axis extent')
            self.basis_endpoints[key] = (binding[0], tuple(binding[1]))
        self.Lt = self.Nt = max(extents)  # Legacy contraction extent; never physical Lt for compact data.
        self.physical_Lt = physical_lt if explicit_selection else self.Lt
        self.source_selection, self.sink_selection = source_record, sink_record
        self.time_selections = ({} if not explicit_selection else
                                {'source_time': source_record, 'sink_time': sink_record})
        self.time_selection = source_record if source_record == sink_record else None
        self.time_selection_identity = (None if not explicit_selection else
            tuple((axis, record.Lt, record.times)
                  for axis, record in sorted(self.time_selections.items())))

    @classmethod
    def from_npy(cls, files, *, time_convention, axis_order=("source_time", "sink_time"), cfg=None,
                 source_selection=None, sink_selection=None, basis_endpoints=None):
        if tuple(sorted(axis_order)) != ("sink_time", "source_time"):
            raise ValueError("axis_order must permute source_time and sink_time")
        maps, arrays = [], {}
        try:
            for sector, path in files.items():
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                maps.append(array)
                axes = (axis_order.index("source_time"), axis_order.index("sink_time"))
                arrays[sector] = array.transpose((*axes, *range(2, array.ndim)))
            return cls(arrays, time_convention=time_convention, cfg=cfg, owned_maps=maps,
                       source_selection=source_selection, sink_selection=sink_selection,
                       basis_endpoints=basis_endpoints)
        except Exception:
            for array in maps:
                array._mmap.close()
            raise

    @classmethod
    def from_binary(cls, files, *, shapes, dtypes, time_convention, offsets=None, cfg=None,
                    source_selection=None, sink_selection=None, basis_endpoints=None):
        """Headerless C-order data; shape, dtype and byte order are mandatory."""
        maps, arrays = [], {}
        try:
            for sector, path in files.items():
                shape = tuple(shapes[sector])
                dtype = np.dtype(dtypes[sector])
                offset = 0 if offsets is None else offsets.get(sector, 0)
                if dtype.hasobject or type(offset) is not int or offset < 0 or Path(path).stat().st_size != offset + int(np.prod(shape)) * dtype.itemsize:
                    raise ValueError(f"invalid {sector} binary shape/dtype/offset or file size")
                array = np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset)
                maps.append(array)
                arrays[sector] = array
            return cls(arrays, time_convention=time_convention, cfg=cfg, owned_maps=maps,
                       source_selection=source_selection, sink_selection=sink_selection,
                       basis_endpoints=basis_endpoints)
        except Exception:
            for array in maps:
                array._mmap.close()
            raise

    @classmethod
    def from_npy_timeslices(cls, files, *, time_convention, source_times=None, cfg=None,
                           source_selection=None, sink_selection=None, basis_endpoints=None):
        """Open selected source-slot files; records retain physical time metadata."""
        maps, arrays = [], {}
        try:
            for sector, paths in files.items():
                paths = tuple(paths)
                selected = times(range(len(paths)) if source_times is None else source_times,
                                 len(paths), "source_times")
                arrays[sector] = [None] * len(paths)
                for t in selected:
                    path = paths[t]
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    maps.append(array)
                    arrays[sector][t] = array
            return cls(arrays, time_convention=time_convention, cfg=cfg, owned_maps=maps,
                       source_selection=source_selection, sink_selection=sink_selection,
                       basis_endpoints=basis_endpoints)
        except Exception:
            for array in maps:
                array._mmap.close()
            raise

    def block(self, sink_time, source_time, sector="vsv", *, usedNe=None,
              mode_indices=None, usedNp=None, point_indices=None):
        if self._closed:
            raise ValueError("propagator reader is closed")
        if (sector not in self.sectors or type(sink_time) is not int or
                type(source_time) is not int or not 0 <= sink_time < self.sink_Nt or
                not 0 <= source_time < self.source_Nt):
            raise KeyError("missing propagator sector or time coverage")
        time = ((sink_time - source_time) % self.Lt
                if self.time_convention == "relative" else sink_time)
        data = self.sectors[sector]
        if isinstance(data, Mapping):
            value = data[sink_time, source_time]
        elif isinstance(data, (tuple, list)):
            if data[source_time] is None:
                raise KeyError(f"source time {source_time} was not opened")
            value = data[source_time][time]
        else:
            value = data[source_time, time]
        return select_axes(value, ("sink_spin", "source_spin", *_TAILS[sector]),
                           usedNe=usedNe, mode_indices=mode_indices,
                           usedNp=usedNp, point_indices=point_indices)

    def describe_block(self, sink_time, source_time, sector="vsv", *, usedNe=None,
                       mode_indices=None, usedNp=None, point_indices=None):
        """Describe selected axes and dtype from held array headers only."""
        if self._closed:
            raise ValueError("propagator reader is closed")
        if (sector not in self.sectors or type(sink_time) is not int or
                type(source_time) is not int or not 0 <= sink_time < self.sink_Nt or
                not 0 <= source_time < self.source_Nt):
            raise KeyError("missing propagator sector or time coverage")
        data = self.sectors[sector]
        if isinstance(data, Mapping):
            block = data[sink_time, source_time]
            raw_shape, dtype = tuple(block.shape), block.dtype
        elif isinstance(data, (tuple, list)):
            if data[source_time] is None:
                raise KeyError(f"source time {source_time} was not opened")
            raw_shape = tuple(data[source_time].shape[1:])
            dtype = data[source_time].dtype
        else:
            raw_shape = tuple(data.shape[2:])
            dtype = data.dtype
        names = ("sink_spin", "source_spin", *_TAILS[sector])
        selections = resolve_selections(names, raw_shape, usedNe=usedNe,
                                        mode_indices=mode_indices, usedNp=usedNp,
                                        point_indices=point_indices)
        shape = tuple(
            len(range(*selections[name].indices(size))) if isinstance(selections.get(name), slice)
            else len(selections[name]) if name in selections else size
            for name, size in zip(names, raw_shape, strict=True)
        )
        identity = (self.data_identity, self.time_selection_identity, sink_time, source_time, sector,
                    repr(usedNe), repr(mode_indices), repr(usedNp), repr(point_indices))
        bindings = _axis_bindings(sector, names, raw_shape, selections, self.basis_endpoints)
        if bindings:
            identity += (tuple(bindings.items()),)
        return TensorAxisMetadata(names, shape, dtype, self.data_identity, repr(identity),
                                  axis_bindings=bindings)

    def describe_slab(self, sink_times, source_time, sector="vsv", *, usedNe=None,
                      mode_indices=None, usedNp=None, point_indices=None):
        """Describe ordered sink slots without materializing tensor values.

        The returned shape has one leading ``coordinate_batch`` axis. Times may
        be unsorted or repeated; both coordinate order and repetitions are kept.
        """
        if self._closed:
            raise ValueError("propagator reader is closed")
        source_time = _single_time(source_time, self.source_Nt, "source_time")
        selected_sinks = _time_sequence(sink_times, self.sink_Nt, "sink_times")
        if sector not in self.sectors:
            raise KeyError(f"missing propagator sector {sector}")
        data = self.sectors[sector]
        if isinstance(data, Mapping):
            # Validate every requested pair using headers only, before any read.
            rows = [self.describe_block(t, source_time, sector, usedNe=usedNe,
                        mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices)
                    for t in selected_sinks]
            first = rows[0]
            return TensorAxisMetadata(
                ('coordinate_batch', *first.axes), (len(rows), *first.shape), first.dtype,
                self.data_identity, repr(tuple(row.selection_identity for row in rows)),
                axis_bindings=first.axis_bindings)
        if isinstance(data, (tuple, list)):
            source = data[source_time]
            if source is None:
                raise KeyError(f"source time {source_time} was not opened")
            if source.shape[0] != self.sink_Nt:
                raise ValueError("propagator sink-time coverage differs from its declared sink slots")
            raw_shape, dtype = tuple(source.shape[1:]), source.dtype
        else:
            if data.shape[1] != self.sink_Nt:
                raise ValueError("propagator sink-time coverage differs from its declared sink slots")
            raw_shape, dtype = tuple(data.shape[2:]), data.dtype
        tail_names = ("sink_spin", "source_spin", *_TAILS[sector])
        selections = resolve_selections(tail_names, raw_shape, usedNe=usedNe,
                                        mode_indices=mode_indices, usedNp=usedNp,
                                        point_indices=point_indices)
        selected_shape = tuple(
            len(range(*selections[name].indices(size))) if isinstance(selections.get(name), slice)
            else len(selections[name]) if name in selections else size
            for name, size in zip(tail_names, raw_shape, strict=True)
        )
        sink_indices = (selected_sinks if self.time_convention == "absolute" else
                        tuple((sink - source_time) % self.Lt for sink in selected_sinks))
        identity = (self.data_identity, self.time_selection_identity, "sink-slab", source_time, selected_sinks,
                    sink_indices, sector, repr(usedNe), repr(mode_indices),
                    repr(usedNp), repr(point_indices), self.time_convention)
        return TensorAxisMetadata(
            ("coordinate_batch", *tail_names),
            (len(selected_sinks), *selected_shape), dtype,
            self.data_identity, repr(identity),
            axis_bindings=_axis_bindings(sector, tail_names, raw_shape, selections,
                                        self.basis_endpoints),
        )

    def read_slab(self, sink_times, source_time, sector="vsv", *, usedNe=None,
                  mode_indices=None, usedNp=None, point_indices=None):
        """Read/gather one source-slot slab, without repeated ``block`` calls.

        Increasing contiguous sink indices borrow one slice of the source slab;
        other orders use one indexed gather and preserve the requested order.
        """
        metadata = self.describe_slab(
            sink_times, source_time, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices,
        )
        source_time = _single_time(source_time, self.source_Nt, "source_time")
        selected_sinks = _time_sequence(sink_times, self.sink_Nt, "sink_times")
        sink_indices = (selected_sinks if self.time_convention == "absolute" else
                        tuple((sink - source_time) % self.Lt for sink in selected_sinks))
        data = self.sectors[sector]
        if isinstance(data, Mapping):
            blocks = [self.block(t, source_time, sector, usedNe=usedNe,
                                 mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices)
                      for t in selected_sinks]
            return _slab_array_module(blocks[0]).stack(blocks)
        source_slab = data[source_time] if isinstance(data, (tuple, list)) else data[source_time]
        names = metadata.axes
        value = select_slab_axes(
            source_slab, names, sink_indices, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError("propagator slab shape/dtype differs from its metadata")
        return value

    def iter_blocks(self, sink_times, source_times, sector="vsv", *, usedNe=None,
                    mode_indices=None, usedNp=None, point_indices=None):
        """Yield selected slot pairs without materializing a time grid."""
        sinks = times(sink_times, self.sink_Nt, "sink_times")
        sources = times(source_times, self.source_Nt, "source_times")
        for source in sources:
            for sink in sinks:
                yield sink, source, self.block(sink, source, sector,
                                               usedNe=usedNe, mode_indices=mode_indices,
                                               usedNp=usedNp, point_indices=point_indices)

    def read_times(self, sink_times, source_times, sector="vsv", *, usedNe=None,
                   mode_indices=None, usedNp=None, point_indices=None):
        """Return only selected pairs in (source_time,sink_time,...) order."""
        sinks = times(sink_times, self.sink_Nt, "sink_times")
        sources = times(source_times, self.source_Nt, "source_times")
        return np.stack([np.stack([self.block(sink, source, sector,
                                              usedNe=usedNe, mode_indices=mode_indices,
                                              usedNp=usedNp, point_indices=point_indices)
                                   for sink in sinks]) for source in sources])

    def close(self):
        self.sectors = {}
        for array in self._maps:
            array._mmap.close()
        self._maps, self._closed = [], True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def select_propagator_times(reader, selection):
    """Borrow a lazy ordered time view of an existing propagator reader.

    ``selection`` names actual physical times, never parent array positions.
    The returned reader accepts selected integer slots, while its parent keeps
    interpreting its original absolute/relative layout. No values are read or
    written here; the parent must remain open until the view is consumed.
    """
    return _TimeSelectedPropagator(reader, selection)


class _TimeSelectedPropagator:
    """Selected slot endpoints with the original reader as numerical owner."""

    def __init__(self, reader, selection):
        physical_lt = getattr(reader, 'physical_Lt', None)
        if type(physical_lt) is not int or physical_lt < 1:
            raise ValueError('time selection requires a declared parent physical Lt')
        record = normalize_time_selection(selection, Lt=physical_lt)
        parent_records = getattr(reader, 'time_selections', {})
        mappings = []
        for axis in ('sink_time', 'source_time'):
            parent = parent_records.get(axis)
            if parent is None:
                if parent_records or getattr(reader, 'Lt', None) != physical_lt:
                    raise ValueError('parent propagator requires explicit endpoint time selections')
                parent = normalize_time_selection(None, Lt=physical_lt)
            else:
                parent = normalize_time_selection(parent, Lt=physical_lt)
            mappings.append(parent.require_times(record.times))
        self.reader = reader
        self._sink_slots, self._source_slots = mappings
        self._closed = False
        self.Lt = self.Nt = self.sink_Nt = self.source_Nt = record.Nt
        self.physical_Lt = physical_lt
        self.source_selection = self.sink_selection = self.time_selection = record
        self.time_selections = {'sink_time': record, 'source_time': record}
        # The view exposes absolute endpoint slots. Its parent's relative
        # divisor and ordered coordinates remain part of the archived contract.
        self.time_convention = 'absolute'
        self.time_selection_identity = tuple(
            (axis, record.Lt, record.times) for axis in sorted(self.time_selections))
        self.propagator_time_view = ('ordered-physical-time-v1', physical_lt,
            reader.time_convention, getattr(reader, 'time_selection_identity', None),
            getattr(reader, 'propagator_time_view', None))

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def _pair(self, sink_time, source_time):
        if self._closed:
            raise ValueError('propagator time view is closed')
        sink = _single_time(sink_time, self.sink_Nt, 'sink_time')
        source = _single_time(source_time, self.source_Nt, 'source_time')
        return self._sink_slots[sink], self._source_slots[source]

    def _slab(self, sink_times, source_time):
        if self._closed:
            raise ValueError('propagator time view is closed')
        sinks = _time_sequence(sink_times, self.sink_Nt, 'sink_times')
        source = _single_time(source_time, self.source_Nt, 'source_time')
        return tuple(self._sink_slots[sink] for sink in sinks), self._source_slots[source]

    def _metadata(self, metadata):
        return replace(metadata, selection_identity=repr((
            'propagator-time-view-v1', self.time_selection_identity,
            self.propagator_time_view, metadata.selection_identity)))

    def describe_block(self, sink_time, source_time, sector='vsv', **selection):
        sink, source = self._pair(sink_time, source_time)
        return self._metadata(self.reader.describe_block(sink, source, sector, **selection))

    def block(self, sink_time, source_time, sector='vsv', **selection):
        sink, source = self._pair(sink_time, source_time)
        return self.reader.block(sink, source, sector, **selection)

    def describe_slab(self, sink_times, source_time, sector='vsv', **selection):
        sinks, source = self._slab(sink_times, source_time)
        return self._metadata(self.reader.describe_slab(sinks, source, sector, **selection))

    def read_slab(self, sink_times, source_time, sector='vsv', **selection):
        sinks, source = self._slab(sink_times, source_time)
        return self.reader.read_slab(sinks, source, sector, **selection)

    def read_slab_in_background(self, sink_times, source_time, sector='vsv', **selection):
        sinks, source = self._slab(sink_times, source_time)
        return self.reader.read_slab_in_background(sinks, source, sector, **selection)

    def iter_blocks(self, sink_times, source_times, sector='vsv', **selection):
        sinks = times(sink_times, self.sink_Nt, 'sink_times')
        sources = times(source_times, self.source_Nt, 'source_times')
        for source in sources:
            for sink in sinks:
                yield sink, source, self.block(sink, source, sector, **selection)

    def read_times(self, sink_times, source_times, sector='vsv', **selection):
        sinks = times(sink_times, self.sink_Nt, 'sink_times')
        sources = times(source_times, self.source_Nt, 'source_times')
        rows = [self.read_slab(sinks, source, sector, **selection) for source in sources]
        return _slab_array_module(rows[0]).stack(rows)

    def close(self):
        # The parent can serve other borrowed views and owns its file maps.
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise ValueError('propagator time view is closed')
        return self

    def __exit__(self, *exc):
        self.close()


class PropagatorPairReader:
    """Expose six-representative PR datasets through the correlator reader API.

    ``reader`` is an existing production PropagatorReader. Actual endpoint
    pairs are inferred only when unique, or explicitly selected with
    ``endpoints={"vsd": (sink_endpoint, source_endpoint), ...}``.
    The wrapper borrows the underlying readers and never closes them.
    """

    def __init__(self, reader, *, endpoints=None):
        from lqcdweave.production.perambulator.io import PropagatorReader
        from lqcdweave.production.perambulator.products import reverse_product, SAVED_PRODUCTS
        from lqcdweave.contracts.dataset import stable_digest
        if not isinstance(reader, PropagatorReader):
            raise TypeError('PropagatorPairReader requires a production PropagatorReader')
        self.reader = reader
        self.Lt = self.Nt = self.physical_Lt = self.source_Nt = self.sink_Nt = reader.lt
        self.time_selection = None
        self.time_selections = {}
        self.time_selection_identity = None
        choices = {}
        for product, sink, source in reader._readers:
            choices.setdefault(product.lower(), []).append((sink, source))
            reverse = reverse_product(product)
            if reverse not in SAVED_PRODUCTS:
                choices.setdefault(reverse.lower(), []).append((source, sink))
        if endpoints is None:
            if any(len(set(values)) != 1 for values in choices.values()):
                raise ValueError('multiple actual endpoint pairs require explicit endpoints')
            endpoints = {sector: values[0] for sector, values in choices.items()}
        self.endpoints = dict(endpoints)
        if not self.endpoints or set(self.endpoints) - set(_TAILS):
            raise ValueError('pair endpoints must name known propagator sectors')
        ids = []
        for sector, pair in self.endpoints.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('pair endpoints must contain actual sink/source records')
            pair = tuple(pair)
            self.endpoints[sector] = pair
            if pair not in choices.get(sector, ()):
                raise ValueError('pair endpoint records differ from actual saved propagators')
            ids.append([sector, pair[0].document, pair[1].document])
        self.sectors = self.endpoints
        saved_ids = sorted({dataset.data_id for entries in reader._readers.values()
                            for dataset, _layout in entries})
        self.data_identity = stable_digest({'kind': 'correlator-pr-pairs', 'endpoints': ids,
                                           'parents': saved_ids})
        first_reader = next(iter(reader._readers.values()))[0][0]
        self.cfg = first_reader.spec.document['configuration']['configuration']
        self.background_read_safe = reader.array_module is np

    @property
    def dilution_identity(self):
        from lqcdweave.contracts.dataset import canonical_json
        records = [[sector, side, endpoint.document]
                   for sector, pair in sorted(self.endpoints.items())
                   for side, endpoint in zip(('sink', 'source'), pair, strict=True)
                   if endpoint.kind == 'D']
        return canonical_json(records) if records else None

    def _request(self, sink_time, source_time, sector, selection):
        from lqcdweave.production.perambulator.products import SAVED_PRODUCTS, reverse_product, product_shape
        sector = sector.lower()
        if sector not in self.endpoints:
            raise KeyError('missing actual propagator sector')
        sink_time = _single_time(sink_time, self.Lt, 'sink_time')
        source_time = _single_time(source_time, self.Lt, 'source_time')
        sink, source = self.endpoints[sector]
        if (sink_time, source_time) not in self.reader.coverage(sector, sink=sink, source=source):
            raise KeyError('missing actual endpoint or reverse physical-pair coverage')
        product = sector.upper()
        reverse = product not in SAVED_PRODUCTS
        key = (reverse_product(product), source, sink) if reverse else (product, sink, source)
        pair = (source_time, sink_time) if reverse else (sink_time, source_time)
        candidates = [(key, pair)]
        if not reverse and product[0] == product[2]:
            candidates.append(((product, source, sink), pair[::-1]))
        parent_reader, layout = next((r, item) for candidate, coordinate in candidates
                                    for r, item in self.reader._readers.get(candidate, ())
                                    if coordinate in item.positions)
        names = ('sink_spin', 'source_spin', *_TAILS[sector])
        raw_shape = product_shape(product, len(sink.indices), len(source.indices))
        selections = resolve_selections(names, raw_shape, **selection)
        shape = tuple(len(range(*selections[name].indices(size)))
                      if isinstance(selections.get(name), slice)
                      else len(selections[name]) if name in selections else size
                      for name, size in zip(names, raw_shape, strict=True))
        endpoints = {(sector, 'sink'): (sink.data_id, sink.indices),
                     (sector, 'source'): (source.data_id, source.indices)}
        bindings = _axis_bindings(sector, names, raw_shape, selections, endpoints)
        identity = repr(('pr-pair', parent_reader.data_id, sector, sink_time, source_time,
                         sink.document, source.document, tuple(selections.items())))
        metadata = TensorAxisMetadata(names, shape, parent_reader.spec.document['dtype'],
                                      parent_reader.data_id, identity, axis_bindings=bindings)
        return sector, sink, source, metadata

    def describe_block(self, sink_time, source_time, sector='vsv', *, usedNe=None,
                       mode_indices=None, usedNp=None, point_indices=None):
        return self._request(sink_time, source_time, sector, dict(
            usedNe=usedNe, mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices))[-1]

    def block(self, sink_time, source_time, sector='vsv', *, usedNe=None,
              mode_indices=None, usedNp=None, point_indices=None):
        selection = dict(usedNe=usedNe, mode_indices=mode_indices, usedNp=usedNp,
                         point_indices=point_indices)
        sector, sink, source, metadata = self._request(sink_time, source_time, sector, selection)
        block = self.reader.read(sector, sink_time, source_time, sink=sink, source=source)
        return select_axes(block.values, metadata.axes, **selection)

    def describe_slab(self, sink_times, source_time, sector='vsv', **selection):
        sinks = _time_sequence(sink_times, self.Lt, 'sink_times')
        rows = [self.describe_block(time, source_time, sector, **selection) for time in sinks]
        first = rows[0]
        if any(row.shape != first.shape or row.dtype != first.dtype for row in rows[1:]):
            raise ValueError('propagator pair slab headers differ in shape or dtype')
        return TensorAxisMetadata(('coordinate_batch', *first.axes),
                                  (len(rows), *first.shape), first.dtype,
                                  self.data_identity,
                                  repr(tuple(row.selection_identity for row in rows)),
                                  axis_bindings=first.axis_bindings)

    def read_slab(self, sink_times, source_time, sector='vsv', **selection):
        sinks = _time_sequence(sink_times, self.Lt, 'sink_times')
        return self.reader.array_module.stack([
            self.block(time, source_time, sector, **selection) for time in sinks])

    def read_slab_in_background(self, *args, **kwargs):
        if not self.background_read_safe:
            raise TypeError('background pair reads require host NumPy data')
        return self.read_slab(*args, **kwargs)
