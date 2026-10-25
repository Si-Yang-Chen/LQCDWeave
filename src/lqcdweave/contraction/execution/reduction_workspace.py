"""One admitted, stream-local partial buffer for a device execution scope."""
from threading import get_ident

import numpy as np


MAX_REDUCTION_WORKSPACE_BYTES = 8 * 1024**2
_ALIGNMENT = 512
_WIDE_DTYPES = (np.dtype('float64'), np.dtype('complex128'))
# Synchronization failures cannot safely return an in-flight allocation to its
# pool. Exceptional owners remain alive, potentially until process exit; this
# is not a reusable cache and is never probed or retried from the hot path.
_FAILED_WORKSPACES = []


class ReductionWorkspace:
    """Reuse only internal partials; never lend storage for published results.

    The fixed reservation is additional to P's original workspace envelope.
    A different stream/thread/device/context or dtype keeps ordinary allocation.
    No array is captured by a kernel binding or retained across device scopes.
    """
    def __init__(self, backend):
        self.backend = backend
        cuda = backend.cuda
        self._device = getattr(getattr(cuda, 'runtime', None), 'getDevice', None)
        self._context = getattr(getattr(cuda, 'driver', None), 'ctxGetCurrent', None)
        if getattr(getattr(cuda, 'runtime', None), 'is_hip', False):
            self._context = None  # CuPy HIP exposes only an unsupported stub.
        self._stream = getattr(cuda, 'get_current_stream', None)
        self._supported = all(callable(fn) for fn in (self._device, self._context, self._stream))
        self._failed_close = False
        self.capacity = 0
        self.buffer = self.dtype = self.identity = self.stream = None
        self._pool = None
        self.allocations = self.hits = self.misses = self.peak_bytes = 0

    def _current(self):
        if not self._supported:
            return None, None
        stream = self._stream()
        pointer = getattr(stream, 'ptr', None)
        context = self._context()
        if pointer is None or not context:
            return None, None
        return (get_ident(), self._device(), context, pointer), stream

    def configure(self, available, *, pool=None):
        capacity = min(MAX_REDUCTION_WORKSPACE_BYTES, max(0, available))
        capacity = capacity // _ALIGNMENT * _ALIGNMENT
        identity, stream = self._current()
        if identity is None:
            capacity = 0
        if self.buffer is not None:
            if capacity >= self.capacity and identity == self.identity:
                return self.capacity
            # A changed admission cannot keep an uncharged old owner. This
            # uncommon boundary fences its original stream before releasing.
            self.close()
        self.capacity, self.identity, self.stream = capacity, identity, stream
        # CuPy pooled arrays do not keep their private pool alive. A failed
        # synchronization must retain both the array and this allocation owner.
        self._pool = pool
        return capacity

    def borrow(self, nbytes, dtype):
        dtype = np.dtype(dtype)
        if (self._failed_close or not self.capacity or type(nbytes) is not int or not 0 < nbytes <= self.capacity
                or dtype not in _WIDE_DTYPES
                or nbytes % dtype.itemsize or self.dtype is not None and dtype != self.dtype):
            self.misses += 1
            return None
        # Context is thread-local and streams may change inside a device scope.
        # Check current identity, not an indefinitely cached readiness marker.
        identity, _ = self._current()
        if identity != self.identity:
            self.misses += 1
            return None
        if self.buffer is None:
            try:
                value = self.backend.empty((self.capacity // dtype.itemsize,), dtype=dtype)
            except MemoryError:
                self.capacity = 0
                self.misses += 1
                return None
            extent = getattr(getattr(getattr(value, 'data', None), 'mem', None), 'size', None)
            if type(extent) is not int or not value.nbytes <= extent <= self.capacity:
                self.capacity = 0
                self.misses += 1
                return None
            self.buffer, self.dtype = value, dtype
            self.allocations += 1
            self.peak_bytes = max(self.peak_bytes, extent)
        self.hits += 1
        return self.buffer

    def close(self, *, synchronized_identity=None):
        if self.buffer is not None and synchronized_identity != self.identity:
            # A stream switch does not make the old stream's partials complete.
            try:
                with self.backend.cuda.Device(self.identity[1]):
                    if self._context() != self.identity[2]:
                        raise RuntimeError('cannot release reduction workspace in another CUDA context')
                    self.stream.synchronize()
            except BaseException:
                if not self._failed_close:
                    self._failed_close = True
                    _FAILED_WORKSPACES.append(self)
                raise
        self.buffer = self.dtype = self.identity = self.stream = None
        self._pool = None
        self.capacity = 0
        if self._failed_close:
            _FAILED_WORKSPACES.remove(self)
            self._failed_close = False

    def report(self):
        return dict(reduction_workspace_reserved_bytes=self.capacity,
                    reduction_workspace_allocations=self.allocations,
                    reduction_workspace_hits=self.hits,
                    reduction_workspace_misses=self.misses,
                    reduction_workspace_peak_bytes=self.peak_bytes)
