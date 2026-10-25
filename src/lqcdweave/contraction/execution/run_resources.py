"""Handle-owned pools/constants; numerical inputs/results remain run-local."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from .constant_cache import DeviceConstantCache

_active_resources = ContextVar('contraction_run_resources', default=None)

def active_run_resources():
    return _active_resources.get()

def _device_key(backend):
    if getattr(backend, '__name__', '') != 'cupy':
        return None
    if getattr(backend.cuda.runtime, 'is_hip', False):
        return None  # HIP cannot supply a context identity for cross-run reuse.
    driver = getattr(backend.cuda, 'driver', None)
    context = getattr(driver, 'ctxGetCurrent', None)
    if not callable(context):
        return None  # Extension backends retain their original ownership.
    return (backend, backend.cuda.Device().id, context())

class RunResources:
    """Reuse a bounded pool/constants, with fresh tickets, streams and inputs."""
    def __init__(self):
        self._lock = RLock()
        self._key = self._pool = self._limit = None
        self._leased = False
        self._constants = DeviceConstantCache()

    @contextmanager
    def activate(self):
        from lqcdweave.runtime.memory import reusable_device_memory
        token = _active_resources.set(self)
        try:
            with reusable_device_memory(self.reusable_bytes):
                yield
        finally:
            _active_resources.reset(token)

    def reusable_bytes(self, backend):
        from .device import active_device_execution
        if active_device_execution() is not None:
            return 0
        if self._leased or self._pool is None or not self._lock.acquire(blocking=False):
            return 0
        try:
            if self._leased or self._key != _device_key(backend):
                return 0
            # This owner's idle constants can be reused or evicted before IO.
            return int(self._pool.free_bytes()) + self._constants.nbytes
        finally:
            self._lock.release()

    def _release(self):
        pool, key = self._pool, self._key
        self._pool = self._key = self._limit = None
        if pool is not None:
            with key[0].cuda.Device(key[1]):
                self._constants.clear()
                pool.free_all_blocks()

    def constants_for(self, pool):
        return self._constants if self._leased and pool is self._pool else None

    @contextmanager
    def pool(self, backend, limit):
        key = _device_key(backend)
        if key is None or self._leased or not self._lock.acquire(blocking=False):
            yield None
            return
        try:
            if self._key != key or self._limit != limit:
                self._release()
                pool = backend.cuda.MemoryPool()
                try:
                    pool.set_limit(size=limit)
                except BaseException:
                    pool.free_all_blocks()
                    raise
                self._pool, self._key, self._limit = pool, key, limit
            self._leased = True
            try:
                yield self._pool
            except BaseException as primary:
                try:
                    self._release()
                except BaseException as cleanup:
                    if callable(getattr(primary, 'add_note', None)):
                        primary.add_note(f'reusable device pool cleanup also failed: {cleanup!r}')
                raise
            finally:
                self._leased = False
        finally:
            self._lock.release()

    def close(self):
        """Release idle reserved storage; a subsequent run may recreate it."""
        with self._lock:
            if self._leased:
                raise RuntimeError('cannot close resources from inside their active execution')
            self._release()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Explicit close reports failures; finalization cannot.


