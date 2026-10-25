"""Bounded CPU result tasks fed by the existing fixed-backend cfg generator.

IO remains in ReadAhead, device work remains on its saved compute stream, and
this single ordered consumer waits only for each result's D2H dependency.
There is no new schedule, backend selection or task graph here.
"""
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import CancelledError
from contextvars import ContextVar
from contextlib import contextmanager
from math import prod
from time import perf_counter
from .diagnostics import debug_enabled
from lqcdweave.contracts.correlator.api import device_allocation_bytes


_active = ContextVar('contraction_output_queue', default=None)


def active_output_queue():
    return _active.get()


@dataclass(frozen=True)
class OutputDownloadLayout:
    """Internal X/W description of independent arrays handed to D2H."""
    owner_bytes: tuple[int, ...]

    def __post_init__(self):
        if type(self.owner_bytes) is not tuple or any(
                type(size) is not int or size < 0 for size in self.owner_bytes):
            raise ValueError('download owner extents must be nonnegative integer bytes')

    @property
    def device_bytes(self):
        return sum(device_allocation_bytes(size) for size in self.owner_bytes)


def _snapshot_download_layout(state, *, packed=True):
    """Mirror output copying, without arrays, kernels or device initialization."""
    if not packed:
        return OutputDownloadLayout(tuple(prod(state.catalog[key].shape)
            * state.catalog[key].dtype.itemsize for key in state.results))
    groups = {}
    for key in dict.fromkeys(state.results):
        spec = state.catalog[key]
        layout = spec.shape, spec.dtype
        groups[layout] = groups.get(layout, 0) + prod(spec.shape) * spec.dtype.itemsize
    return OutputDownloadLayout(tuple(groups.values()))


def _snapshot_device_bytes(plan, packed, queue):
    previous = None if queue is None else queue._snapshot_cache
    if previous is not None and previous[0] is plan and previous[1] is packed:
        return previous[2]
    size = max((_snapshot_download_layout(template.plan.state, packed=packed).device_bytes
                for template in plan.templates), default=0)
    if queue is not None:
        from lqcdweave.contracts.correlator.api import CfgPlan, CfgTemplate, SceneStoragePlan, TensorSpec
        from lqcdweave.contraction.planning.api import has_native_catalog
        # Binding proofs do not validate arbitrary statistics replacements.
        # Cache only our own calculation over immutable output metadata.
        native = (type(plan) is CfgPlan and type(plan.templates) is tuple and all(
            type(template) is CfgTemplate and type(template.plan) is SceneStoragePlan
            and has_native_catalog(template.plan.state)
            and type(template.plan.state.results) is tuple and all(
                type(key) is str and type(template.plan.state.catalog[key]) is TensorSpec
                for key in template.plan.state.results) for template in plan.templates))
        queue._snapshot_cache = (plan, packed, size) if native else None
    return size


def output_queue_reserve(plan, device, host_limit, device_limit):
    """Reserve two rounded download slots from unused host/device headroom.

    The integer interface deliberately reserves the device bound in both
    domains. Pinned host payloads are no larger; no host alignment is implied.
    P's current synchronous snapshot is separate, so its overlap with these
    two complete slots is a conservative reservation, not an extra allocation.
    """
    queue = active_output_queue()
    instrumented = queue is not None and queue.output_bytes is not None
    def result(reserve, reason):
        if instrumented:
            queue.postprocess_queue_mode = reason
        return reserve
    if instrumented and queue.output_bytes == 0:
        return result(0, 'no_pair_downloads')
    if debug_enabled():
        return result(0, 'synchronous_debug')
    if device is None or not getattr(device, 'supports_output_overlap', False):
        return result(0, 'synchronous_transfer_protocol')
    if queue is not None and (queue.device_transform is not None or instrumented):
        layout = queue.download_layout
        if type(layout) is not OutputDownloadLayout:
            # An aggregate callback byte count cannot prove its owner geometry.
            return result(0, 'synchronous_unknown_owners')
        size = layout.device_bytes
    else:
        # Complete-cfg replay binds output layout copying. An injected backend
        # without empty cannot use it and retains individual-copy accounting.
        backend = getattr(device, 'backend', None)
        packed = backend is None or callable(getattr(backend, 'empty', None))
        size = _snapshot_device_bytes(plan, packed, queue)
    reserve = 2 * size
    host = plan.prefetch_host_bytes
    gpu = plan.prefetch_device_bytes
    if host_limit is not None:
        host = min(host, host_limit - plan.peak_host_bytes)
    current_limit = getattr(device, 'limit', None)
    if type(current_limit) is int:
        gpu = min(gpu, current_limit - plan.peak_device_bytes)
    if device_limit is not None:
        gpu = min(gpu, device_limit - plan.peak_device_bytes)
    if reserve == 0:
        return result(0, 'no_pair_downloads')
    if reserve > min(host, gpu):
        return result(0, 'synchronous_budget')
    return result(reserve, 'overlapped_downloads')


class OutputConsumerQueue:
    """Two admitted result tasks, one CPU consumer, deterministic delivery order."""
    def __init__(self, consume, *, output_bytes=None, device_transform=None):
        if device_transform is not None:
            output_bytes = getattr(device_transform, 'output_bytes', None)
        if output_bytes is not None and (type(output_bytes) is not int or output_bytes < 0):
            raise ValueError("output_bytes must be nonnegative integer bytes")
        self.output_bytes = output_bytes
        self.device_transform = device_transform
        self.download_layout = getattr(device_transform, '_download_layout', None)
        self._snapshot_cache = None
        self.postprocess_queue_mode = None
        self.postprocess_queue_modes = {}
        self.consume = consume
        self.pool = None
        self.pending = deque()
        self.seconds = 0.0
        self.wait_seconds = 0.0
        self.peak_tasks = 0
        self.tasks = 0
        self.device = None
        self.reserve = 0
        self.peak_reserve = 0
        self.failure = None

    def __enter__(self):
        self.token = _active.set(self)
        return self

    @contextmanager
    def suspended(self):
        """Release activation across a public yield, restoring the caller scope.

        The caller may enter another queue before resuming or closing this
        iterator. A fresh token then preserves that current context; retaining
        a token across the yield would overwrite it with an obsolete scope.
        """
        _active.reset(self.token)
        try:
            yield
        finally:
            self.token = _active.set(self)

    def wait_one(self):
        """Wait for the oldest submitted result without draining its successor."""
        start = perf_counter()
        try:
            if self.pending:
                self.pending.popleft().result()
            elif self.failure is not None:
                raise self.failure
        finally:
            self.wait_seconds += perf_counter() - start

    def configure(self, device, reserve):
        if self.pending:
            self.flush()
        self.device, self.reserve = device, reserve
        if self.postprocess_queue_mode is not None:
            mode = self.postprocess_queue_mode
            self.postprocess_queue_modes[mode] = self.postprocess_queue_modes.get(mode, 0) + 1
        self.peak_reserve = max(self.peak_reserve, reserve)
        if reserve and self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='contraction-results')

    def acquire(self):
        """Check errors and apply backpressure before producing the next result."""
        if self.failure is not None:
            raise self.failure
        start = perf_counter()
        while self.pending and (len(self.pending) >= 2 or self.pending[0].done()):
            self.pending.popleft().result()
        self.wait_seconds += perf_counter() - start

    def _consume(self, cfg, index, values, device):
        start = None
        try:
            if device is not None:
                with device.backend.cuda.Device(device.device_id):
                    values = device.finish_download(values)
            if self.failure is not None:
                return  # Drain transfer ownership without running later user callbacks.
            start = perf_counter()
            self.consume(cfg, index, values)
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise
        finally:
            if start is not None:
                self.seconds += perf_counter() - start

    def submit(self, cfg, index, values):
        if index is None:
            # Completion/publication and reader teardown follow every result.
            self.flush()
            self._consume(cfg, index, values, None)
        elif self.reserve:
            self.acquire()
            self.pending.append(self.pool.submit(self._consume, cfg, index, values, self.device))
            self.tasks += 1
            self.peak_tasks = max(self.peak_tasks, len(self.pending))
        else:
            self._consume(cfg, index, values, None)

    def flush(self, *, cancel=False):
        error = None
        if cancel:
            for future in self.pending:
                future.cancel()
        while self.pending:
            try:
                self.pending.popleft().result()
            except CancelledError:
                pass
            except BaseException as failure:
                if error is None:
                    error = failure
                for future in self.pending:
                    future.cancel()
        if error is not None:
            raise error

    def __exit__(self, kind, error, traceback):
        _active.reset(self.token)
        # Preserve the primary failure, but always join in-flight consumers
        # before the device scope or caller-owned writer can be torn down.
        try:
            if error is None:
                self.flush()
            else:
                try:
                    self.flush(cancel=True)
                except BaseException as cleanup:
                    if callable(getattr(error, 'add_note', None)):
                        error.add_note(f'output consumer cleanup also failed: {cleanup!r}')
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()

    def report(self):
        result = dict(consumer_seconds=self.seconds, output_queue_tasks=self.tasks,
                    output_queue_peak_tasks=self.peak_tasks,
                    output_queue_wait_seconds=self.wait_seconds,
                    output_buffer_host_bytes=self.peak_reserve,
                    output_buffer_device_bytes=self.peak_reserve)
        if self.output_bytes is not None:
            result['output_postprocess_queue_modes'] = dict(self.postprocess_queue_modes)
        return result
