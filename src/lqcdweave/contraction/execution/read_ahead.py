"""A bounded host read queue ahead of a fixed storage-event stream."""

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from time import perf_counter
from threading import Condition

from .plan_cursor import plan_operations


_METHODS = (
    "can_prefetch_block", "prefetch_host_bytes", "prefetch_block",
    "finish_prefetch", "close_prefetch",
)

# Byte reservations bound buffers; this also bounds completed-read metadata
# when very small uploads outpace the device. It is not another worker queue.
_MAX_DEFERRED_READS = 32
_MAX_PENDING_READS = 128
_MAX_DEVICE_PENDING_READS = 1024
_ADMISSION_BATCH = 64
_MAX_RETIRED_INPUTS = 32


class _ReadTicket:
    """One read result, sharing the queue's single completion condition."""
    __slots__ = ('condition', 'state', 'value', 'error', 'waiters', 'event',
                 'stages', 'host_only')

    def __init__(self, condition, event=None, stages=None):
        self.condition = condition
        self.state, self.value, self.error = 'queued', None, None
        self.waiters = 0
        self.event, self.stages = event, stages
        self.host_only = False

    def cancelled(self):
        return self.state == 'cancelled'

    def done(self):
        return self.state not in ('queued', 'running')

    def result(self):
        with self.condition:
            while self.state in ('queued', 'running'):
                self.waiters += 1
                try:
                    self.condition.wait()
                finally:
                    self.waiters -= 1
            if self.cancelled():
                raise CancelledError()
            if self.error is not None:
                raise self.error
            return self.value


def supports_batch_read_ahead(loader):
    """Future reads must be independent of the active batch's source binding."""
    return (getattr(loader, 'prefetch_across_batches', False) is True
            and all(callable(getattr(loader, name, None)) for name in _METHODS))


def _small_resident_inputs(plan, loader):
    readers = getattr(loader, 'input_readers', None)
    return bool(readers and all(getattr(reader, 'host_resident', False) is True
                                for reader in readers.values())
                and sum(block.read_bytes for block in plan.blocks.values()) <= 8 * 1024**2)


def prepare_read_ahead(plan, loader, *, enabled=True, measure=False,
                       device_capacity=None):
    """Reserve read-ahead capacity without moving computation or release.

    Spare capacity is reserved uniformly over the existing audited trajectory.
    Each queued read reserves its entire bound until it is consumed and any
    asynchronous transfer has completed.
    Calibration measurement deliberately retains synchronous phase timing.
    """
    if type(enabled) is not bool:
        raise TypeError("io_overlap must be bool")
    if not enabled or measure or not all(callable(getattr(loader, name, None)) for name in _METHODS):
        return plan, None, 0
    resident = _small_resident_inputs(plan, loader)
    small_resident = len(plan.blocks) <= 4 and resident
    inline = bool(resident and plan.state.problem.backend == 'cupy'
                  and callable(getattr(loader, 'prefetch_device_bytes', None)))
    if small_resident and not inline:
        # CPU resident views cannot hide IO; device uploads can still overlap.
        return plan, None, 0
    cap = plan.host_memory_limit
    if plan.state.problem.backend == "numpy":
        cap = plan.memory_limit if cap is None else min(cap, plan.memory_limit)
    spare = None if cap is None else max(0, cap - plan.peak_host_bytes)
    device_spare = _resolve_device_capacity(plan, device_capacity)
    sizes, device_sizes = _read_sizes(plan, loader, spare, device_spare)
    if not sizes:
        return plan, None, 0
    reserve = sum(sizes.values())
    if spare is not None:
        reserve = min(spare, reserve)
    device_reserve = min(device_spare, sum(device_sizes.values()))
    # The additive admission check uses a scalar budget. Do not duplicate the
    # entire MemoryPoint trajectory just to add the same number to each point.
    return plan, ReadAhead(
        plan, loader, sizes, reserve, device_sizes=device_sizes,
        device_capacity=device_reserve, inline=inline,
    ), reserve


def _resolve_device_capacity(plan, requested):
    if requested is not None:
        if type(requested) is not int or requested < 0:
            raise ValueError("device_capacity must be nonnegative integer bytes")
        return requested
    try:
        if plan.state.problem.backend != "cupy":
            return 0
        capacity = plan.memory_limit
        peak = getattr(plan, "peak_device_bytes", None)
        if peak is None:
            peak = max((point.device_bytes for point in plan.memory), default=0)
        from .device import active_device_execution
        device = active_device_execution()
        if device is not None:
            capacity = min(capacity, device.limit) - device.constant_reserve
    except (AttributeError, TypeError):
        return 0
    if type(capacity) is not int or type(peak) is not int:
        return 0
    return max(0, capacity - peak)


def _validated_device_sizes(sizes):
    values = dict(sizes or {})
    if any(type(size) is not int or size < 0 for size in values.values()):
        raise ValueError("device_sizes must contain nonnegative integer byte counts")
    return values


def _read_sizes(plan, loader, spare, device_spare):
    sizes, device_sizes = {}, {}
    for event in plan.events:
        if event.kind != "read" or event.block in sizes:
            continue
        block = plan.blocks[event.block]
        eligible = loader.can_prefetch_block(event.block, block)
        if type(eligible) is not bool:
            raise TypeError("can_prefetch_block must return bool")
        if not eligible:
            continue
        size = loader.prefetch_host_bytes(event.block, block)
        if type(size) is not int or size < block.host_bytes + block.staging_host_bytes:
            raise ValueError("prefetch_host_bytes must bound resident host storage and read workspace")
        if spare is not None and size > spare:
            continue
        device_size_fn = getattr(loader, "prefetch_device_bytes", None)
        device_size = (device_size_fn(event.block, block)
                       if callable(device_size_fn) else 0)
        if type(device_size) is not int or device_size < 0:
            raise ValueError("prefetch_device_bytes must return nonnegative integer bytes")
        if device_size > device_spare:
            continue
        sizes[event.block] = size
        device_sizes[event.block] = device_size
    return sizes, device_sizes


def prepare_batch_read_ahead(plans, loader, *, enabled=True, measure=False,
                             device_capacity=None):
    """Connect read events across already joined batches without activating one.

    Only an explicit task-scoped capability may read future batches: its read
    callback must be independent of ``activate`` and current source bindings.
    """
    if type(enabled) is not bool:
        raise TypeError("io_overlap must be bool")
    if (not enabled or measure or not plans
            or not supports_batch_read_ahead(loader)):
        return tuple(plans), None, (), 0
    blocks, events, offsets = {}, [], []
    offset = 0
    spare = None
    device_spares = []
    for plan in plans:
        offsets.append(offset)
        for key, block in plan.blocks.items():
            if key in blocks and blocks[key] != block:
                raise ValueError("cross-batch prefetched block has inconsistent descriptors")
            blocks[key] = block
        events.extend(replace(event, action_index=event.action_index + offset)
                      for event in plan.events if event.kind == "read")
        offset += len(plan.state.actions) + 1
        cap = plan.host_memory_limit
        if plan.state.problem.backend == "numpy":
            cap = plan.memory_limit if cap is None else min(cap, plan.memory_limit)
        if cap is not None:
            available = max(0, cap - plan.peak_host_bytes)
            spare = available if spare is None else min(spare, available)
        device_spares.append(_resolve_device_capacity(plan, None))
    device_spare = (min(device_spares, default=0) if device_capacity is None
                    else _resolve_device_capacity(None, device_capacity))
    stream = SimpleNamespace(blocks=blocks, events=events)
    sizes, device_sizes = _read_sizes(stream, loader, spare, device_spare)
    if not sizes:
        return tuple(plans), None, (), 0
    reserve = sum(sizes.values())
    if spare is not None:
        reserve = min(spare, reserve)
    device_reserve = min(device_spare, sum(device_sizes.values()))
    return tuple(plans), ReadAhead(
        stream, loader, sizes, reserve, device_sizes=device_sizes,
        device_capacity=device_reserve,
    ), tuple(offsets), reserve


class ReadAhead:
    """One read worker, or inline admission for small resident device inputs."""

    def __init__(self, plan, loader, sizes, capacity, *, device_sizes=None,
                 device_capacity=None, inline=False):
        self.plan, self.loader, self.sizes = plan, loader, dict(sizes)
        self.inline = inline
        self.capacity = capacity
        self.device_sizes = _validated_device_sizes(device_sizes)
        self.device_capacity = _resolve_device_capacity(plan, device_capacity)
        self.pending_read_limit = self._pending_limit(self.sizes, self.device_sizes)
        self.peak_pending_read_limit = self.pending_read_limit
        self.peak_pending_reads = 0
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0
        self.reserved_device_bytes = 0
        self.peak_device_bytes = 0
        self.submitted_reads = 0
        self.completed_reads = 0
        self.read_seconds = 0.0
        self.wait_seconds = 0.0
        self.reads = sorted((e for e in plan.events if e.kind == "read"),
                            key=lambda e: e.action_index)
        self.cursor = 0
        self.pool = None
        self._condition = Condition()
        self._work = deque()
        self._stopping = False
        self._prepare_work = deque()
        self._prepare_condition = Condition()
        self._prepare_stopping = False
        self.prepare_pool = None
        # One eligible read has no IO/CPU pipeline to overlap. Small resident
        # inputs keep the existing inline or single-worker staging path.
        self.split_preparation = (not inline and len(self.sizes) > 1
                                  and getattr(loader, '_prefetch_stages', None) is not None
                                  and not _small_resident_inputs(plan, loader))
        self.pending = {}
        self.deferred = {}
        self.retired_inputs = deque()
        self.retired_input_bytes = 0
        self.peak_retired_input_bytes = 0
        self.retired_input_count = 0
        self.blocks = dict(plan.blocks)
        self.owners = {e.block:loader for e in self.reads}
        self._bind_device_loader(loader)

    def _bind_device_loader(self, loader):
        bind = getattr(loader, '_bind_device_read_ahead', None)
        if callable(bind):
            bind(self)

    def _pending_limit(self, sizes, device_sizes):
        """Use spare buffer bytes for small uploads, with bounded metadata."""
        largest_device = max((device_sizes or {}).values(), default=0)
        if not largest_device or not self.device_capacity:
            return _MAX_PENDING_READS
        largest_host = max(sizes.values(), default=0)
        fitting = self.device_capacity // largest_device
        if largest_host:
            fitting = min(fitting, self.capacity // largest_host)
        return max(_MAX_PENDING_READS, min(_MAX_DEVICE_PENDING_READS, fitting))

    def retire_device_input(self, device, ticket):
        """Share the existing device allowance with still-used released inputs."""
        if self.retired_inputs:
            self._reclaim_retired_inputs()
        if (self._stopping or len(self.retired_inputs) >= _MAX_RETIRED_INPUTS
                or self.reserved_device_bytes + ticket.device_bytes > self.device_capacity):
            return False
        self.retired_inputs.append((device, ticket))
        self.reserved_device_bytes += ticket.device_bytes
        self.retired_input_bytes += ticket.device_bytes
        self.retired_input_count += 1
        self.peak_device_bytes = max(self.peak_device_bytes, self.reserved_device_bytes)
        self.peak_retired_input_bytes = max(self.peak_retired_input_bytes, self.retired_input_bytes)
        return True

    def _reclaim_retired_input(self, *, wait=False):
        if not self.retired_inputs:
            return False
        device, ticket = self.retired_inputs[0]
        start = perf_counter()
        try:
            complete = device.complete_input_retirement(ticket, wait=wait)
        finally:
            if wait:
                self.wait_seconds += perf_counter() - start
        if complete:
            self.retired_inputs.popleft()
            self.reserved_device_bytes -= ticket.device_bytes
            self.retired_input_bytes -= ticket.device_bytes
        return complete

    def _reclaim_retired_inputs(self):
        while self._reclaim_retired_input():
            pass

    def extend(self, plan, loader, sizes, *, device_sizes=None, offset=0, prefix=''):
        """Append already planned reads to the same worker and capacity.

        The owner thread binds cfg identities; the IO worker never claims work.
        Prefixes must distinguish actual cfg bindings, including repeat cfgs.
        """
        device_sizes = _validated_device_sizes(device_sizes)
        self._bind_device_loader(loader)
        events=sorted((e for e in plan.events if e.kind=='read'),key=lambda e:e.action_index)
        for event in events:
            key=prefix+event.block
            if key in self.owners or key in self.pending:
                raise ValueError('continuous read queue repeats a bound data key')
            self.reads.append(replace(event,block=key,action_index=event.action_index+offset))
            self.blocks[key]=plan.blocks[event.block]
            self.owners[key]=_PrefixedReader(loader,prefix)
            if event.block in sizes: self.sizes[key]=sizes[event.block]
            if device_sizes is not None and event.block in device_sizes:
                self.device_sizes[key]=device_sizes[event.block]
        self.split_preparation = (not self.inline and len(self.sizes) > 1
                                  and getattr(loader, '_prefetch_stages', None) is not None
                                  and not _small_resident_inputs(plan, loader))
        self.pending_read_limit = self._pending_limit(sizes, device_sizes)
        self.peak_pending_read_limit = max(self.peak_pending_read_limit, self.pending_read_limit)

    def _forget(self,key):
        self.owners.pop(key,None)
        self.blocks.pop(key,None)
        self.sizes.pop(key,None)
        self.device_sizes.pop(key,None)

    def _release_reservation(self, key):
        self.reserved_bytes -= self.sizes[key]
        self.reserved_device_bytes -= self.device_sizes.get(key, 0)
        self._forget(key)

    def _reclaim_oldest(self, *, wait=False):
        if not self.deferred:
            return False
        key = next(iter(self.deferred))
        start = perf_counter()
        try:
            complete = self.owners[key].prefetch_complete(
                key, self.deferred[key], wait=wait)
        finally:
            if wait:
                self.wait_seconds += perf_counter() - start
        if complete:
            del self.deferred[key]
            self._release_reservation(key)
        elif wait:
            raise RuntimeError('blocking prefetch reclamation did not complete')
        return complete

    def _reclaim_completed(self):
        while self._reclaim_oldest():
            pass
        if self.retired_inputs:
            self._reclaim_retired_inputs()

    def _enqueue_reads(self, work):
        if self.inline:
            # Host buffers are filled on this thread; H2D remains asynchronous.
            # Each ticket already owns the same audited reservations as a worker read.
            for ticket, owner, key, block in work:
                self._complete_read(ticket, owner, key, block)
            work.clear()
            return
        # One notification admits a bounded ordered packet to the same worker.
        # Every member already has both reservations and its own ticket.
        with self._condition:
            self._work.extend(work)
            self._condition.notify()
        work.clear()

    def start_after(self, index):
        if self._stopping:
            raise RuntimeError('read-ahead queue is closed')
        self._reclaim_completed()
        while self.cursor < len(self.reads) and self.reads[self.cursor].action_index <= index:
            key=self.reads[self.cursor].block
            if key not in self.pending and key not in self.deferred: self._forget(key)
            self.cursor += 1
        work = []
        try:
            while self.cursor < len(self.reads):
                if len(self.pending) >= self.pending_read_limit:
                    return
                event = self.reads[self.cursor]
                # A later reopening must not overwrite the first ticket/owner.
                # Once consumed, that occurrence may fall back to its saved read.
                if event.block in self.pending or event.block in self.deferred:
                    return
                size = self.sizes.get(event.block)
                device_size = self.device_sizes.get(event.block, 0)
                # Never skip an earlier synchronous read: it could use the same
                # reader. A full queue resumes once its next consumer frees space.
                if size is None:
                    return
                while (self.reserved_bytes + size > self.capacity
                       or self.reserved_device_bytes + device_size > self.device_capacity):
                    if not self.deferred:
                        if (self.retired_inputs and self.reserved_device_bytes + device_size > self.device_capacity):
                            self._reclaim_retired_input(wait=True)
                            continue
                        return
                    # Only resource pressure may block on a consumed upload.
                    self._reclaim_oldest(wait=True)
                if self.pool is None and not self.inline:
                    self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="contraction-io")
                    self.pool.submit(self._run_reads)
                block, owner = self.blocks[event.block], self.owners[event.block]
                stages = (getattr(owner, '_prefetch_stages', None)
                          if self.split_preparation else None)
                if stages is not None and self.prepare_pool is None:
                    self.prepare_pool = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="contraction-prepare")
                    self.prepare_pool.submit(self._run_preparations)
                future = _ReadTicket(self._condition, event, stages)
                self.pending[event.block] = future
                self.reserved_bytes += size
                self.peak_reserved_bytes=max(self.peak_reserved_bytes,self.reserved_bytes)
                self.reserved_device_bytes += device_size
                self.peak_device_bytes=max(self.peak_device_bytes,self.reserved_device_bytes)
                self.submitted_reads+=1
                self.cursor += 1
                work.append((future, owner, event.block, block))
                if len(work) == _ADMISSION_BATCH:
                    self._enqueue_reads(work)
        finally:
            # Early quota/synchronous boundaries and errors still publish every
            # accepted ticket, so cleanup cannot wait for an invisible read.
            if work:
                self._enqueue_reads(work)
            self.peak_pending_reads = max(self.peak_pending_reads, len(self.pending))
        if self.cursor>1024:
            del self.reads[:self.cursor]
            self.cursor=0

    def _work_items(self, work, condition, stopping):
        """Read the bounded admitted view of original events, without a DAG.

        Each item still references its original StorageEvent. These queues
        contain in-flight owners only, not a second list of planned steps.
        A resource worker waits on its dependency condition, leaving the other
        resource generators free to advance.
        """
        while True:
            with condition:
                while not work:
                    if stopping():
                        return
                    condition.wait()
                item = work.popleft()
            yield item
            del item

    def _run_reads(self):
        for ticket, owner, key, block in plan_operations(self._work_items(
                self._work, self._condition, lambda: self._stopping)):
            self._complete_read(ticket, owner, key, block)
            # Do not retain the last numerical owner while idle.
            del ticket, owner, key, block

    def _complete_read(self, ticket, owner, key, block):
        ticket.state = 'running'
        try:
            if ticket.stages is None:
                ticket.value = self._read(owner, key, block)
            else:
                start = perf_counter()
                try:
                    ticket.value = ticket.stages[0](key, block)
                    ticket.host_only = True
                finally:
                    self.read_seconds += perf_counter() - start
                with self._prepare_condition:
                    if not self._stopping:
                        self._prepare_work.append((ticket, owner, key, block))
                        self._prepare_condition.notify()
                        return
        except BaseException as error:
            ticket.error = error
        self._publish(ticket)

    def _run_preparations(self):
        for ticket, owner, key, block in plan_operations(self._work_items(
                self._prepare_work, self._prepare_condition,
                lambda: self._prepare_stopping)):
            try:
                ticket.value = ticket.stages[1](key, block, ticket.value)
                ticket.host_only = False
            except BaseException as error:
                ticket.error = error
            self._publish(ticket)
            del ticket, owner, key, block

    def _publish(self, ticket):
        with self._condition:
            ticket.state = 'complete'
            if ticket.waiters:
                self._condition.notify_all()

    def pending_read(self, key):
        """Existing ticket dependency for a saved read; never submit new work."""
        return self.pending.get(key)

    def _read(self,owner,key,block):
        start=perf_counter()
        try: return owner.prefetch_block(key,block)
        finally: self.read_seconds+=perf_counter()-start

    def take(self, key, block):
        if key not in self.pending:
            return False, None
        self._reclaim_completed()
        if len(self.deferred) >= _MAX_DEFERRED_READS:
            self._reclaim_oldest(wait=True)
        future = self.pending.pop(key)
        start=perf_counter()
        retained = False
        try:
            try:
                value = future.result()
            except BaseException as primary:
                if future.host_only:
                    try:
                        future.stages[2](key, future.value)
                    except BaseException as cleanup:
                        primary.add_note(f'host preparation cleanup also failed: {cleanup!r}')
                    finally:
                        future.host_only = False
                        future.value = None
                raise
            self.completed_reads+=1
            try:
                owner = self.owners[key]
                result = owner.finish_prefetch(key, block, value)
                complete = getattr(owner, 'prefetch_complete', None)
                if callable(complete) and not complete(key, value):
                    self.deferred[key] = value
                    retained = True
                return True, result
            except BaseException as primary:
                try:
                    self.owners[key].close_prefetch(key, value)
                except BaseException as cleanup:
                    complete = getattr(self.owners[key], 'prefetch_complete', None)
                    if callable(complete):
                        # Keep failed-transfer ownership until close retries its
                        # completion or the device scope performs final cleanup.
                        try:
                            retained = not complete(key, value)
                        except BaseException:
                            retained = True
                        if retained:
                            self.deferred[key] = value
                    add_note = getattr(primary, "add_note", None)
                    if callable(add_note):
                        add_note(f"prefetch cleanup also failed: {cleanup!r}")
                raise
        finally:
            self.wait_seconds+=perf_counter()-start
            if not retained:
                self._release_reservation(key)

    def close(self):
        pending, self.pending = self.pending, {}
        with self._condition:
            self._stopping = True
            while self._work:
                self._work.popleft()[0].state = 'cancelled'
            self._condition.notify_all()
        if self.pool is not None:
            # A running read cannot be cancelled safely. Join before closing
            # readers or releasing any buffers which their callbacks may use.
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        # IO has stopped producing preparations. Stop admission, then join
        # the one active staging call before releasing its host/device owners.
        with self._prepare_condition:
            self._prepare_stopping = True
            self._prepare_work.clear()
            self._prepare_condition.notify()
        if self.prepare_pool is not None:
            self.prepare_pool.shutdown(wait=True, cancel_futures=True)
            self.prepare_pool = None
        errors = []
        for key, future in pending.items():
            if not future.cancelled():
                try:
                    if future.host_only:
                        future.stages[2](key, future.value)
                        future.host_only = False
                    else:
                        self.owners[key].close_prefetch(key, future.result())
                except BaseException as exc:
                    errors.append(exc)
        for key, value in self.deferred.items():
            try:
                self.owners[key].prefetch_complete(key, value, wait=True)
            except BaseException as exc:
                errors.append(exc)
        self.deferred.clear()
        while self.retired_inputs:
            device, ticket = self.retired_inputs.popleft()
            try:
                device.complete_input_retirement(ticket, wait=True)
            except BaseException as exc:
                # DeviceExecution still owns a failed ticket until scope exit.
                errors.append(exc)
        self.retired_input_bytes = 0
        self.reserved_bytes = 0
        self.reserved_device_bytes = 0
        self.owners.clear(); self.blocks.clear(); self.sizes.clear(); self.device_sizes.clear(); self.reads.clear()
        self.loader=None
        if errors:
            raise errors[0]


class _PrefixedReader:
    def __init__(self,loader,prefix):
        self.loader,self.prefix=loader,prefix
        stages = getattr(loader, '_prefetch_stages', None)
        if stages is not None:
            self._prefetch_stages = (
                lambda key, block: stages[0](key[len(prefix):], block),
                lambda key, block, value: stages[1](key[len(prefix):], block, value),
                lambda key, value: stages[2](key[len(prefix):], value))
    def prefetch_block(self,key,block): return self.loader.prefetch_block(key[len(self.prefix):],block)
    def prefetch_device_bytes(self,key,block):
        method=getattr(self.loader,'prefetch_device_bytes',None)
        if not callable(method): return 0
        return method(key[len(self.prefix):],block)
    def finish_prefetch(self,key,block,value): return self.loader.finish_prefetch(key[len(self.prefix):],block,value)
    def close_prefetch(self,key,value): return self.loader.close_prefetch(key[len(self.prefix):],value)
    def prefetch_complete(self,key,value,*,wait=False):
        method=getattr(self.loader,'prefetch_complete',None)
        return True if not callable(method) else method(key[len(self.prefix):],value,wait=wait)
