"""Select and copy completed result values into caller-owned outputs."""

from functools import partial
from dataclasses import dataclass, field
from collections.abc import Sequence
from operator import index as integer_index, itemgetter
from math import prod
import numpy as np

from ..diagnostics import debug_enabled
from .transfer import copy_to_backend
from .output_kernel import bind_packed_output_copy, copy_packed_outputs


class PackedOutputs(tuple):
    """Stable output views backed by independent, same-layout snapshot owners.

    Packing never promotes dtypes or flattens a strided input. The tuple API
    stays unchanged; only the device delivery boundary uses the owner table.
    """
    def __new__(cls, owners, positions):
        # Scalar lanes need a zero-dimensional array view instead of NumPy's
        # scalar. Resolve this once per owner, not once per logical Term.
        scalar = tuple(owner.ndim == 1 for owner in owners)
        result = super().__new__(cls, (
            owners[group][lane:lane + 1].reshape(()) if scalar[group] else owners[group][lane]
            for group, lane in positions))
        result.owners, result.positions = tuple(owners), tuple(positions)
        return result

    def __getnewargs__(self):
        return self.owners, self.positions


class _BoundLayoutPackedOutputs(PackedOutputs):
    """Private tuple handoff carrying the frozen layout for one X execution."""

    def __new__(cls, owners, positions, provenance):
        result = super().__new__(cls, owners, positions)
        result.__provenance = provenance
        return result

    @property
    def provenance(self):
        return self.__provenance

    def __reduce__(self):
        return PackedOutputs, (self.owners, self.positions)

    def __reduce_ex__(self, protocol):
        return PackedOutputs, (self.owners, self.positions)


class _OutputOwnerSnapshot(Sequence):
    """X-local handoff: views exist only when a direct array argument needs one.

    The existing output completion event owns this snapshot. No current
    arrays or pointers escape into the bound output program's schema cache.
    """
    __slots__ = ('owners', 'provenance', '_views')

    def __init__(self, owners, provenance):
        self.owners = tuple(owners)
        self.provenance = provenance
        self._views = {}

    @property
    def positions(self):
        return self.provenance.positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('output snapshot index out of range')
        try:
            return self._views[index]
        except KeyError:
            group, lane = self.positions[index]
            owner = self.owners[group]
            value = (owner[lane:lane + 1].reshape(()) if owner.ndim == 1 else
                     owner[lane])
            self._views[index] = value
            return value

    def __reduce__(self):
        return PackedOutputs, (self.owners, self.positions)


@dataclass(frozen=True)
class BoundOutputLayout:
    groups: tuple
    positions: tuple
    nbytes: int
    validate: bool = True
    copy_programs: tuple = ()
    _value_getters: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, '_value_getters', tuple(
            (itemgetter(*keys) if keys else None, len(keys) == 1)
            for keys, _, _ in self.groups))

    def copy(self, arrays, backend, *, owner_snapshot=False):
        debug = debug_enabled()
        owners = []
        for group_index, (keys, shape, dtype) in enumerate(self.groups):
            copier = (self.copy_programs[group_index]
                      if group_index < len(self.copy_programs) else None)
            raw = (getattr(arrays, '_raw_output_value', None)
                   if not self.validate and not debug
                   and getattr(copier, 'accepts_deferred_views', False) else None)
            select, single = self._value_getters[group_index]
            bulk = getattr(arrays, '_raw_output_values', None) if raw is not None else None
            if select is None:
                values = ()
            elif callable(bulk):
                try:
                    values = bulk(select, single)
                except KeyError:
                    # Preserve the original first missing local key or owner.
                    values = tuple(raw(key) for key in keys)
            elif raw is not None:
                values = tuple(raw(key) for key in keys)
            elif debug:
                values = tuple(arrays[key] for key in keys)
            else:
                values = (select(arrays),) if single else select(arrays)
            if (self.validate or debug) and any(
                    tuple(value.shape) != shape or np.dtype(value.dtype) != dtype
                    for value in values):
                raise ValueError('result differs from its saved output layout')
            owner = backend.empty((len(values), *shape), dtype=dtype)
            copied = (copier(values, owner) if copier is not None and not debug else
                      copy_packed_outputs(values, owner, backend))
            if not copied:
                for lane, value in enumerate(values):
                    owner[lane] = value
            owners.append(owner)
        if (not self.validate and not debug
                and getattr(backend, '__name__', '') == 'cupy'):
            if owner_snapshot:
                return _OutputOwnerSnapshot(owners, self)
            return _BoundLayoutPackedOutputs(owners, self.positions, self)
        return PackedOutputs(owners, self.positions)


def bind_output_layout(results, catalog, *, backend=None, validate=True):
    """Copy each result key once; preserve logical Term order through positions.

    Distinct keys stay distinct even when their current arrays or values agree.
    Only metadata is retained, so every run snapshots the current result owners.
    """
    if type(validate) is not bool:
        raise TypeError('validate must be bool')
    groups, positions, by_layout, by_key = [], [], {}, {}
    nbytes = 0
    for key in results:
        position = by_key.get(key)
        if position is not None:
            positions.append(position)
            continue
        spec = catalog[key]
        layout = (spec.shape, spec.dtype)
        group = by_layout.get(layout)
        if group is None:
            group = by_layout[layout] = len(groups)
            groups.append(([], spec.shape, spec.dtype))
        keys = groups[group][0]
        position = by_key[key] = (group, len(keys))
        positions.append(position)
        keys.append(key)
        nbytes += prod(spec.shape) * spec.dtype.itemsize
    frozen_groups = tuple((tuple(keys), shape, dtype) for keys, shape, dtype in groups)
    copy_programs = (tuple(None for _ in frozen_groups) if backend is None else tuple(
        bind_packed_output_copy(backend, shape, dtype, len(keys))
        for keys, shape, dtype in frozen_groups))
    return BoundOutputLayout(frozen_groups, tuple(positions), nbytes,
                             validate, copy_programs)


def copy_output(key, arrays, backend, *, timed, device=False):
    return copy_to_backend(
        arrays[key], backend, phase="output_copy", timed=timed, device=device,
    )


def select_output_lane(value, backend, lane, axis, *, timed, device=False):
    return timed(
        "output_select",
        partial(backend.take, value, lane, axis=axis),
        device=device,
    )


__all__ = ["copy_output", "select_output_lane"]
