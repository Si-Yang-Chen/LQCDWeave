"""Private basis staging for propagator production."""
from numbers import Integral
import numpy as np
from lqcdweave.contracts.time_selection import normalize_time_selection
from lqcdweave.contracts.propagator import PropagatorEndpoint

def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _count(handle, requested, attribute, name):
    available = 0 if handle is None else getattr(handle, attribute)
    if available is None:
        raise ValueError(f"{attribute} is unknown; load the input handle for the configuration "
                         "or declare its available count before passing the basis to calc")
    available = _integer(available, attribute)
    count = available if requested is None else _integer(requested, name)
    if count > available:
        raise ValueError(f"{name}={count} exceeds available {attribute}={available}")
    return count


def _snapshot_endpoint(handle, kind, count, *, endpoint=None):
    if not count:
        return None
    endpoint = getattr(handle, 'endpoint', None) if endpoint is None else endpoint
    if endpoint is None:
        labels = tuple(getattr(handle, 'point_labels' if kind == 'P' else 'mode_labels', range(count)))[:count]
        metadata = {}
        source = getattr(handle, 'dilution_source', None) if kind == 'D' else None
        if source is not None:
            if getattr(handle, 'basis_id', None) != source.data_id:
                raise ValueError('D handle identity differs from its actual dilution source')
            metadata = dict(vector_labels=[list(source.vector_labels[index]) for index in labels],
                mode_groups=[list(group.parent_modes) for group in source.groups], parent_basis_id=source.parent_basis_id,
                realization_id=source.data_id, noise_pool_id=[list(pair) for pair in source.time_pools],
                normalization='raw', dilution_source=source.document)
        return PropagatorEndpoint(kind, getattr(handle, 'basis_id', None), labels, metadata=metadata)
    if not isinstance(endpoint, PropagatorEndpoint) or endpoint.kind != kind:
        raise ValueError('input endpoint record must match the explicit representation')
    source = getattr(handle, 'dilution_source', None) if kind == 'D' else None
    if source is not None and endpoint.data_id != source.data_id:
        raise ValueError('D endpoint identity differs from its actual dilution source')
    if len(endpoint.indices) < count:
        raise ValueError('input endpoint record does not cover the selected vector/point prefix')
    metadata = endpoint.metadata
    if kind == 'D' and 'vector_labels' in metadata:
        if len(metadata['vector_labels']) != len(endpoint.indices):
            raise ValueError('D vector labels differ from the actual endpoint indices')
        metadata['vector_labels'] = metadata['vector_labels'][:count]
    return PropagatorEndpoint(kind, endpoint.data_id, endpoint.indices[:count], metadata=metadata)


class InputReader:
    def _input_selection(self, handle, required):
        """Resolve declared input slots without inferring coverage from shape."""
        selection = normalize_time_selection(
            getattr(handle, 'time_selection', None), Lt=self.latt_size[3])
        selection.require_times(required.times)
        return selection

    def _read_vectors(self, handle, count, key, selection):
        if not count:
            return None
        lx, ly, lz, _ = self.latt_size
        mapping = self._input_selection(handle, selection)
        loaded = handle.load(key)
        if len(loaded.shape) != 6 or loaded.shape[0] != mapping.Nt:
            raise ValueError("vector time shape differs from declared time_selection")
        vectors = self.backend.empty((selection.Nt, count, lz, ly, lx, 3),
                                     np.dtype(self.contract_prec))
        for slot, time in enumerate(selection.times):
            input_slot = mapping.slot(time)
            for e in range(count):
                value = self.xp.asarray(loaded[input_slot, e])
                if value.shape != (lz, ly, lx, 3) or value.dtype.kind not in 'fc':
                    raise ValueError("eigenvectors must be indexed as [time_slot,mode,z,y,x,color]")
                vectors[slot, e] = value
        if not bool(self.xp.isfinite(vectors).all()):
            raise ValueError("eigenvectors must be finite and fit contract_prec")
        return vectors

    def _read_mode_block(self, handle, time, start, stop, scratch):
        """Own a borrowed handle block before another read can reuse it."""
        lx, ly, lz, lt = self.latt_size
        if not 0 <= time < lt:
            raise ValueError("vector time outside lattice")
        try:
            raw = handle.read_modes(self._key, time, start, stop)
            expected = (stop-start, lz, ly, lx, 3)
            if tuple(raw.shape) != expected or raw.dtype.kind not in 'fc':
                raise ValueError("vector block must have (mode,z,y,x,color) complex data")
            owned = scratch[:stop-start]
            owned[...] = self.xp.asarray(raw)
            if not bool(self.xp.isfinite(owned).all()):
                raise ValueError("vector mode block must be finite and fit contract_prec")
            return owned
        except Exception:
            self._sink_cache.clear()
            self.sink_cache_used_bytes = 0
            raise

    def _read_points(self, handle, count, key, selection):
        if not count:
            return None
        lx, ly, lz, _ = self.latt_size
        mapping = self._input_selection(handle, selection)
        loaded = handle.load(key)
        if (len(loaded.shape) != 3 or loaded.shape[0] != mapping.Nt or loaded.shape[2] != 3
                or loaded.shape[1] < count or loaded.dtype.kind not in 'iu'):
            raise ValueError("points must have shape (declared_time_slots,Np,3) and integer xyz coordinates")
        points = self.backend.empty((selection.Nt, count, 3), loaded.dtype)
        for slot, time in enumerate(selection.times):
            points[slot] = self.xp.asarray(loaded[mapping.slot(time), :count, :])
        if bool((points < 0).any()) or bool((points >= self.xp.asarray((lx, ly, lz))).any()):
            raise ValueError("point coordinates are outside the lattice")
        for slot in range(selection.Nt):
            coordinates = points[slot].astype(self.xp.int64, copy=False)
            site_ids = (coordinates[:, 0] * ly + coordinates[:, 1]) * lz + coordinates[:, 2]
            if len(self.xp.unique(site_ids)) != count:
                raise ValueError("points must be distinct at each time")
        return points

