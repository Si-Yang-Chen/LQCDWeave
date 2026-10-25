"""Backend-resident, ordered physical-dt and first-seen one-point reduction.

The caller selects the backend and owns stream ordering, transfer, packing and
publication. This numerical service neither reads device values on the host nor
retains source-by-sink result tensors. NumPy is accepted for protocol tests.
"""

from collections.abc import Mapping
import importlib
from math import prod
from numbers import Integral

import numpy as np

from lqcdweave.contracts.time_selection import positive_integer
from .reconstruction import periodic_time_difference


_MEAN_DTYPES = frozenset(np.dtype(name) for name in (
    'float16', 'float32', 'float64', 'complex64', 'complex128',
))
_ONE_POINT_DTYPES = _MEAN_DTYPES | frozenset(np.dtype(name) for name in (
    'bool', 'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
))


def _layout(fields, Lt, slots):
    """Validate small immutable metadata before any device allocation."""
    Lt = positive_integer(Lt, 'Lt')
    if isinstance(slots, bool) or not isinstance(slots, Integral) or slots < 0:
        raise ValueError('one_point_slots must be a nonnegative integer')
    fields = tuple(fields)
    paths = set()
    for field in fields:
        if field.path in paths:
            raise ValueError(f'duplicate output field path {field.path!r}')
        paths.add(field.path)
        dtype = np.dtype(field.dtype)
        supported = _MEAN_DTYPES if field.dt_offset >= 0 else _ONE_POINT_DTYPES
        if not dtype.isnative or dtype not in supported:
            raise TypeError(f'unsupported device accumulator dtype {dtype.str!r} '
                            f'for field {field.path!r}')
        if field.dt_offset < 0 and (not slots or field.time_role is None):
            raise ValueError('one-point fields require slots and an explicit time_role')
    return fields, Lt, int(slots)


def _coordinates(values, Lt):
    coordinates = tuple(values)
    if any(isinstance(dt, bool) or not isinstance(dt, Integral) or not 0 <= dt < Lt
           for dt in coordinates):
        raise ValueError('dt coordinates must contain integers in [0, Lt)')
    return coordinates


def estimate_device_accumulator_bytes(fields, Lt, slots, dt_coordinates):
    """Return the logical device peak before allocating any numerical buffers.

    The peak includes all resident sums/first-seen values and every owned final
    field returned by ``finalize``. Ufunc ``out`` writes and strided copies need
    zero additional numerical-array scratch. Counts/seen/coordinates are host
    metadata. Packed output, transfer buffers, allocator rounding and backend
    kernel infrastructure belong to the caller's separate resource envelope.
    """
    return sum(describe_device_accumulator_allocations(fields, Lt, slots, dt_coordinates))


def describe_device_accumulator_allocations(fields, Lt, slots, dt_coordinates):
    """Return independent resident/final owner payloads, without a backend.

    The caller applies its allocator policy to each request. Empty owners have
    zero bytes. This is the same lifetime bound as the logical estimate; no
    per-column divide/copy temporary is added because execution writes out=.
    """
    fields, Lt, slots = _layout(fields, Lt, slots)
    n = len(_coordinates(dt_coordinates, Lt))
    return tuple(size for field in fields
                 for size in (prod(field.shape) * np.dtype(field.dtype).itemsize *
                              (Lt if field.dt_offset >= 0 else slots),
                              prod(field.shape) * np.dtype(field.dtype).itemsize *
                              (n if field.dt_offset >= 0 else slots)))



class DeviceTimePairAccumulator:
    """Keep native-dtype sums on the explicitly supplied array backend.

    Each add launches ordinary in-place additions in submission order, without
    atomics or reassociation. All calls must use the same ordered stream, or the
    caller must establish stream dependencies. Inputs are borrowed only during
    submission; they must remain valid until that stream has consumed them.
    Only CPU metadata (counts, seen slots and coordinates) is inspected normally.
    ``debug=True`` additionally checks finiteness and may synchronize the device.

    Returned means and one-point arrays are independent, caller-owned C-order
    arrays with a trailing time axis. Closing releases internal references and
    never invalidates already returned arrays; it does not synchronize streams.
    """

    def __init__(self, fields, *, Lt, backend, one_point_slots=0, debug=False):
        fields, self.Lt, self.one_point_slots = _layout(fields, Lt, one_point_slots)
        if isinstance(backend, str):
            if backend not in ('numpy', 'cupy'):
                raise ValueError('backend must be explicitly NumPy or CuPy')
            backend = importlib.import_module(backend)
        if getattr(backend, '__name__', '') not in ('numpy', 'cupy'):
            raise ValueError('backend must be explicitly NumPy or CuPy')
        self.backend = backend
        self.debug = debug
        self.fields = fields
        self._fields = {field.path: field for field in fields}
        self._closed = False
        self.counts = np.zeros(self.Lt, dtype=np.int64)
        self.sums = {}
        self.loops = {}
        self.seen = {}
        try:
            for field in fields:
                if field.dt_offset >= 0:
                    self.sums[field.path] = backend.zeros(
                        (self.Lt, *field.shape), dtype=field.dtype)
                else:
                    self.loops[field.path] = backend.zeros(
                        (self.one_point_slots, *field.shape), dtype=field.dtype)
                    self.seen[field.path] = np.zeros(self.one_point_slots, dtype=bool)
        except BaseException:
            # The exception traceback owns this partially constructed object
            # while X unwinds/fences its device scope. Do not discard buffers
            # here: a prior zeros fill may still be queued on that stream.
            self._closed = True
            raise

    def _require_open(self):
        if self._closed:
            raise RuntimeError('device time-pair accumulator is closed')

    def add(self, source_time, sink_time, values, *, slots=None, reject_equal=False):
        """Consume a mapping of field paths to existing backend arrays.

        ``slots`` maps each one-point time role to its selected logical slot;
        these are deliberately distinct from physical endpoint coordinates.
        Cheap metadata checks precede mutation. Device execution errors make the
        accumulator unusable, since submitted contributions cannot roll back.
        Its resident buffers remain owned until the caller fences the stream
        and calls ``close``.
        """
        self._require_open()
        for time in (source_time, sink_time):
            if isinstance(time, bool) or not isinstance(time, Integral) or not 0 <= time < self.Lt:
                raise ValueError('physical time coordinates must be integers in [0, Lt)')
        dt = periodic_time_difference(int(source_time), int(sink_time), Lt=self.Lt)
        if reject_equal and dt == 0:
            raise ValueError('packed nonzero dt layout received an equal-time pair')
        if not isinstance(values, Mapping) or len(values) != len(self.fields):
            raise ValueError('values must map every declared field path exactly once')
        contributions = []
        for field in self.fields:
            value = values[field.path]
            if not isinstance(value, self.backend.ndarray):
                raise TypeError('contributions must already reside on the selected backend')
            if value.shape != field.shape or value.dtype != np.dtype(field.dtype):
                raise ValueError(f'output field {field.path!r} differs from saved layout')
            slot = None
            if field.dt_offset < 0:
                if slots is None or field.time_role not in slots:
                    raise ValueError(f'missing one-point slot for role {field.time_role!r}')
                slot = slots[field.time_role]
                if isinstance(slot, bool) or not isinstance(slot, Integral) or not 0 <= slot < self.one_point_slots:
                    raise ValueError('one-point slot outside declared coverage')
            if self.debug and value.dtype.kind in 'fc' and not bool(self.backend.isfinite(value).all()):
                raise ValueError('nonfinite correlator output')
            contributions.append((field, value, slot))
        try:
            for field, value, slot in contributions:
                if field.dt_offset >= 0:
                    # A length-one slice also handles scalar fields as a view.
                    target = self.sums[field.path][dt:dt + 1]
                    self.backend.add(target, value, out=target)
                elif not self.seen[field.path][slot]:
                    self.loops[field.path][slot:slot + 1] = value
                    self.seen[field.path][slot] = True
            self.counts[dt] += 1
        except BaseException:
            # X owns synchronization. Keep queued outputs alive while its
            # exception boundary drains the stream, then let close release them.
            self._closed = True
            raise
        return dt

    def means(self, path, dts, *, missing='nan'):
        """Return native-dtype means in requested trailing physical-dt order."""
        self._require_open()
        if missing not in ('raise', 'nan'):
            raise ValueError("missing must be 'raise' or 'nan'")
        coordinates = _coordinates(dts, self.Lt)
        if missing == 'raise':
            absent = [int(dt) for dt in coordinates if not self.counts[dt]]
            if absent:
                raise ValueError(f'no source pairs for dt={absent}')
        field = self._fields[path]
        sums = self.sums[path]
        out = self.backend.empty((*field.shape, len(coordinates)), dtype=field.dtype)
        for column, dt in enumerate(coordinates):
            if self.counts[dt]:
                source = self.backend.moveaxis(sums[dt:dt + 1], 0, -1)
                self.backend.divide(source, self.counts[dt], out=out[..., column:column + 1])
                if self.debug and not bool(self.backend.isfinite(out[..., column:column + 1]).all()):
                    raise ValueError('nonfinite reconstructed output')
            else:
                out[..., column] = np.nan
        return out

    def one_point(self, path):
        """Return the first contribution per logical slot; require all slots."""
        self._require_open()
        if not self.seen[path].all():
            raise ValueError('incomplete one-point time coverage')
        field = self._fields[path]
        out = self.backend.empty((*field.shape, self.one_point_slots), dtype=field.dtype)
        out[...] = self.backend.moveaxis(self.loops[path], 0, -1)
        return out

    def finalize(self, dts, *, missing='nan'):
        """Return an owned array per field without changing accumulated sums."""
        self._require_open()
        coordinates = _coordinates(dts, self.Lt)
        if any(not seen.all() for seen in self.seen.values()):
            raise ValueError('incomplete one-point time coverage')
        return {field.path: (self.means(field.path, coordinates, missing=missing)
                            if field.dt_offset >= 0 else self.one_point(field.path))
                for field in self.fields}

    def close(self):
        """Release internal device buffers on success, failure or cancellation."""
        self.sums.clear()
        self.loops.clear()
        self.seen.clear()
        self._closed = True


__all__ = ['DeviceTimePairAccumulator', 'estimate_device_accumulator_bytes',
           'describe_device_accumulator_allocations']
