"""Execution-local, budgeted numerical packing results.

Unlike binding caches this object deliberately owns numerical device buffers.
Only an audited X lifetime activates it; the structural GEMM binding simply
calls ``pack_operand`` and never retains this owner or a source array.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import weakref


_active = ContextVar('contraction_packed_values', default=None)
_MAX_ENTRIES = 8192


class PackedValueCache:
    """Hold packed copies while their exact, immutable X source remains live."""

    def __init__(self, capacity, *, array_type=None):
        if type(capacity) is not int or capacity < 0:
            raise ValueError('packing cache capacity must be nonnegative integer bytes')
        self.capacity = capacity
        self.array_type = array_type
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.hits = self.misses = self.evictions = self.source_releases = 0
        self.cleared_entries = 0
        self._stream = None

    def _remove(self, key, reference=None):
        entry = self.entries.get(key)
        if entry is not None and (reference is None or entry[0] is reference):
            del self.entries[key]
            self.bytes -= entry[2]
            return True
        return False

    def pack(self, value, recipe, nbytes, factory):
        if (not self.capacity or nbytes <= 0 or nbytes > self.capacity
                or self.array_type is None or type(value) is not self.array_type):
            self.misses += 1
            return factory()
        key = (id(value), recipe)
        entry = self.entries.get(key)
        if entry is not None and entry[0]() is value:
            self.hits += 1
            self.entries.move_to_end(key)
            return entry[1]
        self.misses += 1
        owner = weakref.ref(self)

        def released(reference):
            cache = owner()
            if cache is not None and cache._remove(key, reference):
                cache.source_releases += 1

        try:
            reference = weakref.ref(value, released)
        except TypeError:
            return factory()
        while self.entries and (self.bytes + nbytes > self.capacity
                                or len(self.entries) >= _MAX_ENTRIES):
            old_key = next(iter(self.entries))
            self._remove(old_key)
            self.evictions += 1
        packed = factory()  # Failed/partial copies are never published.
        self.entries[key] = (reference, packed, nbytes)
        self.bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return packed

    def clear(self):
        self.cleared_entries += len(self.entries)
        self.entries.clear()
        self.bytes = 0

    @contextmanager
    def activate(self, backend):
        # Reuse across cfg points is valid only within the same CUDA stream.
        stream = getattr(backend.cuda.get_current_stream(), 'ptr', None)
        if stream is None:
            # Extension/test backends without stable stream identity cannot
            # establish reuse ordering. Preserve their existing numerical path.
            token = _active.set(None)
            try:
                yield
            finally:
                _active.reset(token)
            return
        if self._stream != stream:
            self.clear()
            self._stream = stream
        token = _active.set(self if self.capacity else None)
        try:
            yield
        finally:
            _active.reset(token)

    def report(self):
        return {f'packed_value_cache_{key}': value for key, value in {
            'capacity_bytes': self.capacity, 'hits': self.hits, 'misses': self.misses,
            'peak_device_bytes': self.peak_bytes, 'live_device_bytes': self.bytes,
            'evictions': self.evictions, 'source_releases': self.source_releases,
            'cleared_entries': self.cleared_entries}.items()}


def pack_operand(value, recipe, nbytes, allocate_and_pack):
    """Use an admitted current-run copy; otherwise preserve the original call."""
    cache = _active.get()
    if cache is None:
        return allocate_and_pack()
    return cache.pack(value, recipe, nbytes, allocate_and_pack)


@contextmanager
def packed_values_scope(execution, backend):
    from .diagnostics import debug_enabled
    cache = getattr(execution, 'packed_value_cache', None)
    if cache is None or not cache.capacity or debug_enabled():
        # A disabled nested replay must not inherit its caller's numerical
        # cache or budget. Restore that caller even when the replay fails.
        token = _active.set(None)
        try:
            yield
        finally:
            _active.reset(token)
    else:
        with cache.activate(backend):
            yield
