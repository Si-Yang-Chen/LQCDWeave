"""Common accelerator ownership and bounded asynchronous input uploads.

Adapted from the former matrix_device loader; no matrix-specific dispatch or
placement decisions remain here. P owns placement and the resource envelope.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, partial
from operator import attrgetter, methodcaller
from time import perf_counter
import sys
import numpy as np

from .preparation import bind_prepared_loader
from .operations.output import PackedOutputs, _OutputOwnerSnapshot
from .diagnostics import debug_enabled
from .reduction_workspace import ReductionWorkspace


_active = ContextVar('contraction_device', default=None)


@lru_cache(maxsize=16)
def _event_poller(event_type):
    """Bind completion API only; retain neither events nor device owners."""
    if callable(getattr(event_type, 'query', None)):
        return methodcaller('query')
    if getattr(event_type, 'done', None) is not None:
        return attrgetter('done')
    return None


def active_device_execution():
    return _active.get()


@dataclass(eq=False)
class UploadedRead:
    host: object
    value: object = None
    event: object = None
    pinned: object = None
    error: BaseException | None = None
    complete: bool = False


@dataclass(eq=False)
class DownloadedOutputs:
    """One in-flight output owner; host views become public only after its event."""
    device_values: object
    host_owners: tuple
    positions: tuple | None
    event: object
    pinned: tuple


@dataclass(eq=False)
class RetiredInput:
    """Current raw input kept alive until its last submitted compute use."""
    value: object
    event: object
    device_bytes: int


@dataclass
class _BorrowedUpload:
    value: object
    borrowers: set
    raw_open: bool = True


class DeviceExecution:
    def __init__(self, backend, pool, limit, *, constant_cache=None):
        self.backend, self.pool, self.limit = backend, pool, limit
        self._constant_cache = constant_cache
        self._constants_configured = False
        self.constant_reserve = 0
        self.reduction_workspace = ReductionWorkspace(backend)
        self._synchronized_compute_identity = None
        self.device_id = backend.cuda.Device().id
        self.upload_stream = backend.cuda.Stream(non_blocking=True)
        event_factory = backend.cuda.Event
        # The public CuPy CUDA/HIP Event API supports untimed dependency markers.
        # Bind once; keep injected/legacy zero-argument factories compatible.
        self._dependency_event = (
            partial(event_factory, disable_timing=True)
            if getattr(event_factory, '__module__', None) == 'cupy.cuda.stream'
            else event_factory)
        self._uploads = set()
        self._retired_inputs = set()
        self.download_stream = None
        self._downloads = set()
        self.statistics = dict(prefetched_device_reads=0, prefetched_device_bytes=0,
                               uploaded_bytes=0, downloaded_bytes=0,
                               output_download_calls=0, peak_pending_outputs=0,
                               output_download_wait_seconds=0.0)

    @property
    def supports_output_overlap(self):
        return callable(getattr(self.backend.ndarray, 'get', None))

    def reserve_reduction_workspace(self, peak, *, limit=None, output_reserve=0):
        limit = self.limit if limit is None else min(self.limit, limit)
        available = max(0, limit - peak - self.constant_reserve - output_reserve)
        return self.reduction_workspace.configure(available if not debug_enabled() else 0, pool=self.pool)

    def reduction_partial(self, nbytes, dtype):
        """Internal raw-kernel workspace; returned shape is intentionally flat."""
        return self.reduction_workspace.borrow(nbytes, dtype)

    def synchronize_compute(self):
        identity, stream = self.reduction_workspace._current()
        if stream is None:
            stream = self.backend.cuda.get_current_stream()
        stream.synchronize()
        self._synchronized_compute_identity = identity

    def close_reductions(self):
        self.statistics.update(self.reduction_workspace.report())
        self.reduction_workspace.close(synchronized_identity=self._synchronized_compute_identity)

    def configure_constants(self, plan=None, preparations=None):
        from .binding_cache import bind_cached, native_storage_binding
        from .constant_cache import describe_constants, MAX_CONSTANT_BYTES
        self._constants_configured = True
        cache = self._constant_cache
        if cache is None:
            return
        if plan is None:
            cache.clear()
            self.constant_reserve = 0
            return
        self.constant_reserve = min(MAX_CONSTANT_BYTES,
            max(0, min(self.limit, plan.memory_limit) - plan.peak_device_bytes))
        if not self.constant_reserve:
            cache.clear()
            return
        items = bind_cached('constant-catalog', plan, self.backend,
            lambda: describe_constants(preparations or {}, plan.preparations),
            lambda: native_storage_binding(plan))
        if not items:
            self.constant_reserve = 0
        cache.configure(items, self.constant_reserve)

    def recipe_constant(self, item, create):
        if self.constant_reserve:
            return self._constant_cache.get(item, create)
        return create()

    def upload(self, host, *, background=False):
        ticket = UploadedRead(host)
        if not isinstance(host, np.ndarray):
            ticket.value = host  # Sparse record containers have their own X kernel ABI.
            ticket.complete = True
            return ticket
        cp = self.backend
        try:
            with cp.cuda.Device(self.device_id), cp.cuda.using_allocator(self.pool.malloc):
                with self.upload_stream:
                    pinned = cp.cuda.alloc_pinned_memory(host.nbytes)
                    staging = np.frombuffer(pinned, dtype=host.dtype, count=host.size).reshape(host.shape)
                    np.copyto(staging, host)
                    ticket.pinned = pinned
                    self._uploads.add(ticket)
                    ticket.value = cp.asarray(staging)
                    event = self._dependency_event()
                    event.record(self.upload_stream)
                    # An unrecorded event does not prove the copy completed.
                    ticket.event = event
            self.statistics['uploaded_bytes'] += host.nbytes
            if background:
                self.statistics['prefetched_device_reads'] += 1
                self.statistics['prefetched_device_bytes'] += host.nbytes
        except BaseException as error:
            # A queued transfer may still own staging even when event creation failed.
            ticket.error = error
            try:
                self.complete_upload(ticket, wait=True)
            except BaseException as cleanup:
                if callable(getattr(error, 'add_note', None)):
                    error.add_note(f'upload cleanup also failed: {cleanup!r}')
        return ticket

    def complete_upload(self, ticket, *, wait=False):
        """Reclaim staging only after H2D completion, not after a stream wait.

        ReadAhead retains this ticket's reservation until this returns True.
        CuPy exposes done; injected query APIs are also accepted. Backends
        without either API use the blocking fallback in ready().
        """
        if ticket.complete:
            return True
        with self.backend.cuda.Device(self.device_id):
            if wait:
                if ticket.event is None:
                    # A failed enqueue may not have produced a usable event.
                    self.upload_stream.synchronize()
                else:
                    ticket.event.synchronize()
            else:
                query = _event_poller(type(ticket.event))
                if query is None or not query(ticket.event):
                    return False
        ticket.complete = True
        ticket.pinned = None
        self._uploads.discard(ticket)
        return True

    def ready(self, ticket, *, wait=False):
        """Order compute after an admitted upload; its owner keeps the ticket.

        Direct reads pass wait=True: their transient staging allowance cannot
        be extended beyond the read. Queued reads keep both reservations until
        complete_upload confirms completion, including after input release.
        """
        if ticket.error is not None:
            try:
                self.complete_upload(ticket, wait=True)
            except BaseException as cleanup:
                if callable(getattr(ticket.error, 'add_note', None)):
                    ticket.error.add_note(f'upload cleanup also failed: {cleanup!r}')
            raise ticket.error
        if wait or _event_poller(type(ticket.event)) is None:
            self.complete_upload(ticket, wait=True)
        elif not self.complete_upload(ticket):
            with self.backend.cuda.Device(self.device_id):
                self.backend.cuda.get_current_stream().wait_event(ticket.event)
        return ticket.value

    def close_uploads(self):
        self.upload_stream.synchronize()
        for ticket in self._uploads:
            ticket.complete = True
            ticket.pinned = None
        self._uploads.clear()

    def release_input(self, value, read_ahead=None):
        """Order future upload-pool reuse after the current compute stream.

        Device inputs are allocated by ``upload_stream`` and may be consumed by
        the current compute stream. A bound read queue can retain them inside
        its existing device allowance until their compute event completes.
        Otherwise the upload stream waits for that marker before pool reuse.
        """
        with self.backend.cuda.Device(self.device_id):
            compute_stream = self.backend.cuda.get_current_stream()
            try:
                event = self._dependency_event()
                event.record(compute_stream)
                if (read_ahead is not None and read_ahead.device_capacity
                        and not read_ahead._stopping and not debug_enabled()
                        and isinstance(value, self.backend.ndarray)
                        and _event_poller(type(event)) is not None):
                    from .allocation_failure import device_buffer_bytes
                    ticket = RetiredInput(value, event, device_buffer_bytes(value))
                    if self.complete_input_retirement(ticket):
                        return
                    if read_ahead.retire_device_input(self, ticket):
                        self._retired_inputs.add(ticket)
                        return
                self.upload_stream.wait_event(event)
            except BaseException as primary:
                # A failed admission/query must not free upload-pool storage
                # while the compute stream can still read it.
                try:
                    compute_stream.synchronize()
                except BaseException as cleanup:
                    if callable(getattr(primary, 'add_note', None)):
                        primary.add_note(f'input release cleanup also failed: {cleanup!r}')
                raise

    def complete_input_retirement(self, ticket, *, wait=False):
        with self.backend.cuda.Device(self.device_id):
            if wait:
                ticket.event.synchronize()
            elif not _event_poller(type(ticket.event))(ticket.event):
                return False
        ticket.value = None
        self._retired_inputs.discard(ticket)
        return True

    def close_input_retirements(self):
        errors = []
        for ticket in tuple(self._retired_inputs):
            try:
                self.complete_input_retirement(ticket, wait=True)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def bind_loader(self, loader, preparations=None, specs=None):
        if not self._constants_configured:
            self.configure_constants()  # Other execution paths keep no old constants.
        if isinstance(loader, DeviceSourceLoader):
            if (loader.device is self
                    and (preparations is None or preparations is loader.prepared.preparations)
                    and (specs is None or specs is loader.prepared.specs)):
                return loader
            if loader.host_owners or loader._borrowed_uploads:
                raise ValueError('cannot rebind a device loader with live input owners')
            loader = loader.prepared
        # Even a prepared loader from another scope using the same CuPy module
        # must rebuild constants in this scope's device/allocator. An empty
        # explicit cache prevents the same-backend identity fast path.
        prepared = bind_prepared_loader(loader,
            preparations if preparations is not None else loader.preparations,
            specs if specs is not None else loader.specs, backend=self.backend,
            constant_values={})
        prepared._device_constants = self
        return DeviceSourceLoader(prepared, self)

    def download(self, values):
        packed = isinstance(values, (PackedOutputs, _OutputOwnerSnapshot))
        owners = values.owners if packed else values
        downloaded = tuple(self.backend.asnumpy(value) for value in owners)
        result = (tuple(PackedOutputs(downloaded, values.positions))
                  if packed else downloaded)
        self.statistics['downloaded_bytes'] += sum(value.nbytes for value in result)
        self.statistics['output_download_calls'] += len(owners)
        return result

    def start_download(self, values):
        """Submit a packed D2H after compute, without fencing the next pair.

        The caller admits both output slots in host/device budgets beforehand.
        No host destination is reused: consumers may keep previously yielded
        results, including scalar views, for as long as they need.
        """
        if not self.supports_output_overlap:
            raise TypeError('backend does not provide asynchronous ndarray.get')
        cp = self.backend
        packed = isinstance(values, (PackedOutputs, _OutputOwnerSnapshot))
        owners = values.owners if packed else tuple(values)
        if any(not owner.flags.c_contiguous for owner in owners):
            # ndarray.get may otherwise create an implicit contiguous device
            # temporary on the caller's current stream, outside this ticket's
            # dependency and budget. Diagnostic/unpacked paths stay synchronous.
            raise ValueError('asynchronous outputs require C-contiguous snapshot owners')
        if self.download_stream is None:
            self.download_stream = cp.cuda.Stream(non_blocking=True)
        hosts, pinned = [], []
        try:
            ready = self._dependency_event()
            ready.record(cp.cuda.get_current_stream())
            self.download_stream.wait_event(ready)
            for owner in owners:
                memory = cp.cuda.alloc_pinned_memory(owner.nbytes)
                host = np.frombuffer(memory, dtype=owner.dtype, count=owner.size).reshape(owner.shape)
                pinned.append(memory)
                hosts.append(host)
                # MemoryPointer's explicit async copy is available on CuPy 12
                # as well as newer CUDA/HIP builds. ndarray.get(blocking=...)
                # was added later; pinned destinations and the existing event
                # still delimit ownership without a device-wide fence.
                copy_async = getattr(owner.data, 'copy_to_host_async', None)
                if callable(copy_async):
                    copy_async(host.ctypes.data, owner.nbytes, self.download_stream)
                else:
                    owner.get(stream=self.download_stream, out=host, blocking=False)
            event = self._dependency_event()
            event.record(self.download_stream)
        except BaseException:
            # A failed enqueue may already own host/device memory.
            self.download_stream.synchronize()
            raise
        ticket = DownloadedOutputs(values, tuple(hosts),
            values.positions if packed else None,
            event, tuple(pinned))
        self._downloads.add(ticket)
        self.statistics['downloaded_bytes'] += sum(owner.nbytes for owner in owners)
        self.statistics['output_download_calls'] += len(owners)
        self.statistics['peak_pending_outputs'] = max(
            self.statistics['peak_pending_outputs'], len(self._downloads))
        return ticket

    def finish_download(self, ticket):
        if ticket not in self._downloads:
            raise RuntimeError('output ticket is not owned by this execution')
        start = perf_counter()
        try:
            with self.backend.cuda.Device(self.device_id):
                ticket.event.synchronize()
        finally:
            self.statistics['output_download_wait_seconds'] += perf_counter() - start
        self._downloads.remove(ticket)
        result = (tuple(PackedOutputs(ticket.host_owners, ticket.positions))
                  if ticket.positions is not None else ticket.host_owners)
        ticket.device_values = None
        # NumPy's base ownership keeps pinned destinations alive after return.
        ticket.pinned = ()
        return result

    def close_outputs(self):
        if self.download_stream is not None:
            self.download_stream.synchronize()
        for ticket in self._downloads:
            ticket.device_values = None
            ticket.pinned = ()
        self._downloads.clear()


class DeviceSourceLoader:
    def __init__(self, prepared, device):
        self.prepared, self.device = prepared, device
        self.host_owners = {}
        self._uploaded_blocks = set()
        self._borrowed_uploads = {}
        self._read_ahead = None
        # X-only stages of the same saved read, not a new reader protocol.
        self._prefetch_stages = (self._prefetch_host,
                                 self._prepare_prefetch, self._close_prefetch_host)

    def _bind_device_read_ahead(self, queue):
        self._read_ahead = queue

    def bind_batch_preparations(self, preparations, specs):
        """Switch this task's logical batch view without replacing its raw owners."""
        prepared = bind_prepared_loader(self.prepared, preparations, specs,
                                        backend=self.device.backend, constant_values={})
        prepared._device_constants = self.device
        self.prepared = prepared

    def __getattr__(self, name):
        return getattr(self.prepared, name)

    def prefetch_host_bytes(self, key, block):
        return self.prepared.prefetch_host_bytes(key, block) + block.device_bytes

    def prefetch_device_bytes(self, key, block):
        from lqcdweave.contracts.correlator.api import device_allocation_bytes
        return device_allocation_bytes(block.device_bytes)

    def prefetch_block(self, key, block):
        return self.device.upload(self.prepared.prefetch_block(key, block), background=True)

    def _prepare_prefetch(self, key, block, host):
        return self.device.upload(host, background=True)

    def _prefetch_host(self, key, block):
        return self.prepared.prefetch_block(key, block)

    def _close_prefetch_host(self, key, host):
        return self.prepared.close_prefetch(key, host)

    def finish_prefetch(self, key, block, ticket):
        value = self.device.ready(ticket)
        host = self.prepared.finish_prefetch(key, block, ticket.host)
        self.host_owners[key] = host
        if host is not ticket.host:
            # Do not hold two uploads under one read reservation if the host
            # callback replaces its value. This uncommon path drains first.
            self.device.complete_upload(ticket, wait=True)
            ticket.value = value = None
            value = self.device.ready(self.device.upload(host), wait=True)
        if isinstance(host, np.ndarray):
            self._uploaded_blocks.add(key)
        return value

    def prefetch_complete(self, key, ticket, *, wait=False):
        return self.device.complete_upload(ticket, wait=wait)

    def close_prefetch(self, key, ticket):
        primary = None
        try:
            self.device.ready(ticket, wait=True)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                if key in self.host_owners:
                    self.prepared.close_block(key, self.host_owners.pop(key))
                else:
                    self.prepared.close_prefetch(key, ticket.host)
                self._uploaded_blocks.discard(key)
            except BaseException as cleanup:
                if primary is None:
                    raise
                if callable(getattr(primary, 'add_note', None)):
                    primary.add_note(f'host prefetch cleanup also failed: {cleanup!r}')

    def read_block(self, key, block):
        host = self.prepared.read_block(key, block)
        try:
            value = self.device.ready(self.device.upload(host), wait=True)
        except BaseException as primary:
            try:
                self.prepared.close_block(key, host)
            except BaseException as cleanup:
                if callable(getattr(primary, 'add_note', None)):
                    primary.add_note(f'host block cleanup also failed: {cleanup!r}')
            raise
        self.host_owners[key] = host
        if isinstance(host, np.ndarray):
            self._uploaded_blocks.add(key)
        return value

    def borrow_uploaded_block(self, key, borrower, value):
        """Share only this loader's own H2D allocation, never an injected array."""
        if key not in self._uploaded_blocks or not isinstance(value, self.backend.ndarray):
            return False
        lease = self._borrowed_uploads.get(key)
        if lease is None:
            lease = self._borrowed_uploads[key] = _BorrowedUpload(value, set())
        lease.borrowers.add(borrower)
        return True

    def release_uploaded_borrow(self, key, borrower):
        lease = self._borrowed_uploads[key]
        lease.borrowers.discard(borrower)
        if not lease.raw_open and not lease.borrowers:
            self.device.release_input(lease.value, self._read_ahead)
            del self._borrowed_uploads[key]

    def close_block(self, key, value):
        owner = self.host_owners.pop(key)
        self._uploaded_blocks.discard(key)
        try:
            lease = self._borrowed_uploads.get(key)
            if lease is None:
                self.device.release_input(value, self._read_ahead)
            else:
                lease.raw_open = False
                if not lease.borrowers:
                    self.device.release_input(lease.value, self._read_ahead)
                    del self._borrowed_uploads[key]
        except BaseException as primary:
            try:
                self.prepared.close_block(key, owner)
            except BaseException as cleanup:
                add_note = getattr(primary, 'add_note', None)
                if callable(add_note):
                    add_note(f'host block cleanup also failed: {cleanup!r}')
            raise
        self.prepared.close_block(key, owner)

    def materialize(self, key, source, blocks):
        return self.prepared.materialize(key, source, blocks)


@contextmanager
def device_execution(backend, *, device_memory_limit):
    """Enter a target's array pool; lifetime cleanup fences only at scope exit."""
    if backend.__name__ != 'cupy':
        yield None
        return
    if type(device_memory_limit) is not int or device_memory_limit <= 0:
        raise ValueError('device memory limit must be a positive integer')
    current = _active.get()
    if current is not None:
        if current.backend is not backend or device_memory_limit > current.limit:
            raise ValueError('nested device execution differs from its admitted target')
        yield current
        return
    from .run_resources import active_run_resources
    owner = None if debug_enabled() else active_run_resources()
    scope = owner.pool(backend, device_memory_limit) if owner is not None else None
    if scope is not None:
        with scope as reused_pool:
            with _device_pool_scope(backend, device_memory_limit, reused_pool) as device:
                yield device
    else:
        with _device_pool_scope(backend, device_memory_limit, None) as device:
            yield device


@contextmanager
def _device_pool_scope(backend, device_memory_limit, reused_pool):
    pool = backend.cuda.MemoryPool() if reused_pool is None else reused_pool
    try:
        pool.set_limit(size=device_memory_limit)
        from .run_resources import active_run_resources
        owner = None if reused_pool is None else active_run_resources()
        constants = None if owner is None else owner.constants_for(pool)
        device = DeviceExecution(backend, pool, device_memory_limit, constant_cache=constants)
    except BaseException as primary:
        try:
            pool.free_all_blocks()
        except BaseException as cleanup:
            add_note = getattr(primary, 'add_note', None)
            if callable(add_note):
                add_note(f'device pool setup cleanup also failed: {cleanup!r}')
        raise
    token = _active.set(device)
    try:
        with backend.cuda.using_allocator(pool.malloc):
            yield device
    finally:
        primary = sys.exc_info()[1]
        _active.reset(token)
        errors = []
        for close in (device.close_uploads,
                      device.synchronize_compute,
                      device.close_reductions,
                      device.close_input_retirements,
                      device.close_outputs):
            try:
                close()
            except BaseException as error:
                errors.append(error)
        if reused_pool is None:
            try:
                pool.free_all_blocks()
            except BaseException as error:
                errors.append(error)
        if primary is not None:
            for error in errors:
                if callable(getattr(primary, 'add_note', None)):
                    primary.add_note(f'device cleanup also failed: {error!r}')
        elif errors:
            raise errors[0]


__all__ = ['device_execution', 'active_device_execution', 'DeviceSourceLoader']
