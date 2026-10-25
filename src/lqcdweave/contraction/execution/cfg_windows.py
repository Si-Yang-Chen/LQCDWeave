"""Consume finalized P windows through the existing replay and continuous IO."""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace
import sys
import numpy as np

from lqcdweave.contracts.correlator.api import CfgPlan
from lqcdweave.contraction.planning.api import validate_cfg_allocation_envelope
from .cfg_runtime import _ContinuousQueue, _iter_execute_cfg_plan, _close_many
from .device import active_device_execution, device_execution
from .diagnostics import execution_debug
from .preparation import bind_prepared_loader
from .storage_runtime import validate_coordinate_batch_workspace
from .output_queue import active_output_queue, output_queue_reserve, OutputConsumerQueue
from contextlib import ExitStack


class _WindowQueue(_ContinuousQueue):
    def __init__(self, first, windows, cfg, factory, host_cap, device_cap,
                 host_prefetch, device_prefetch, device, backend):
        super().__init__(first, (), factory, first.prefetch_host_bytes,
                         first.prefetch_device_bytes, device)
        self.backend = backend
        self.iterator = iter(windows)
        self._first = first
        self.cfg = cfg
        self.host_cap, self.device_cap = host_cap, device_cap
        self.host_prefetch, self.device_prefetch = host_prefetch, device_prefetch
        self._plans, self._coordinates = {}, {}
        self._coordinate_end = 0
        self.planning_wait_seconds = 0.0
        self.peak_bound_windows = 0

    def claim(self, wait=True):
        if self.exhausted or len(self.claimed) >= 2:
            return False
        if self._first is not None:
            plan, self._first = self._first, None
        else:
            ready = getattr(self.iterator, 'next_ready', None)
            if not wait and ready is None:
                return False
            start = perf_counter()
            try:
                plan = next(self.iterator) if wait else ready()
            except StopIteration:
                self.exhausted = True
                return False
            finally:
                if wait:
                    self.planning_wait_seconds += perf_counter() - start
            if plan is None:
                return False
        if (not isinstance(plan, CfgPlan) or plan.backend != self.plan.backend
                or plan.execution_target != self.plan.execution_target
                or plan.implementation_version != self.plan.implementation_version
                or plan.output_mode != self.plan.output_mode):
            raise ValueError('planning window changes the saved execution target or output mode')
        if plan is not self.plan:
            validate_cfg_allocation_envelope(plan)
        if plan.statistics.get('coordinate_offset', self._coordinate_end) != self._coordinate_end:
            raise ValueError('planning windows omit or repeat coordinate positions')
        host_capacity, device_capacity = _capacities(plan, self.host_cap, self.device_cap,
                                                    self.host_prefetch, self.device_prefetch)
        if active_output_queue() is not None:
            reserve = output_queue_reserve(plan, self.device, self.host_cap, self.device_cap)
            host_capacity = max(0, host_capacity - reserve)
            device_capacity = max(0, device_capacity - reserve)
        # Admission can shrink while previous reads drain; no new reservation
        # exceeds the smaller bound. At the boundary all previous reads have
        # been consumed, so outstanding buffers belong to the next window.
        self.queue.capacity = min(self.queue.capacity, host_capacity)
        self.queue.device_capacity = min(self.queue.device_capacity, device_capacity)
        for template in plan.templates:
            validate_coordinate_batch_workspace(template.plan, self.backend)
        scope = ExitStack()
        try:
            loader = scope.enter_context(self.factory(plan, self.cfg))
            loader = (self.device.bind_loader(loader, plan.preparations, plan.specs)
                      if self.device is not None else
                      bind_prepared_loader(loader, plan.preparations, plan.specs))
            prefix = f'{self.serial}:'
            events = tuple(replace(event, action_index=event.action_index + instance.action_offset)
                           for instance in plan.instances for event in instance.events if event.kind == 'read')
            sizes = {key: loader.prefetch_host_bytes(key, plan.blocks[key])
                     for key, request in plan.requests.items()
                     if request.prefetch_host_bytes is not None and loader.can_prefetch_block(key, plan.blocks[key])}
            device_sizes = ({key: loader.prefetch_device_bytes(key, plan.blocks[key]) for key in sizes}
                            if self.device is not None else {})
            self.queue.extend(SimpleNamespace(blocks=plan.blocks, events=events), loader, sizes,
                              offset=self.offset, prefix=prefix, device_sizes=device_sizes)
            self.claimed.append((self.cfg, loader, scope, prefix, self.offset))
            self._plans[prefix] = plan
            self._coordinates[prefix] = self._coordinate_end
            self._coordinate_end += len(plan.instances)
            self.offset += plan.instances[-1].action_offset + len(plan.templates[
                plan.instances[-1].template].plan.state.actions) + 1
            self.serial += 1
            self.peak_bound_windows = max(self.peak_bound_windows, len(self.claimed))
        except BaseException:
            scope.close()
            raise
        return True

    def advance(self, index):
        self.queue.start_after(index)
        if self.queue.cursor >= len(self.queue.reads) and self.claim(wait=False):
            self.queue.start_after(index)

    def plan_for(self, prefix):
        return self._plans[prefix]

    def coordinate_offset(self, prefix):
        return self._coordinates[prefix]

    def release_plan(self, prefix):
        self._plans.pop(prefix, None)
        self._coordinates.pop(prefix, None)

    def close(self):
        try:
            super().close()
        finally:
            self._plans.clear()
            self._coordinates.clear()
            self._first = None


def _capacities(plan, host_cap, device_cap, host_prefetch, device_prefetch):
    if host_cap is not None and plan.peak_host_bytes > host_cap:
        raise MemoryError('planning window exceeds worker host memory')
    if device_cap is not None and plan.peak_device_bytes > device_cap:
        raise MemoryError('planning window exceeds worker device memory')
    host, device = plan.prefetch_host_bytes, plan.prefetch_device_bytes
    if host_cap is not None:
        host = min(host, host_cap - plan.peak_host_bytes)
    if device_cap is not None:
        device = min(device, device_cap - plan.peak_device_bytes)
    for name, value in (('prefetch_host_bytes', host_prefetch), ('prefetch_device_bytes', device_prefetch)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f'{name} must be nonnegative integer bytes or None')
    return (host if host_prefetch is None else min(host, host_prefetch),
            device if device_prefetch is None else min(device, device_prefetch))


def iter_execute_cfg_windows(windows, configuration, loader_factory, *, backend=np,
                             host_memory_limit=None, device_memory_limit=None,
                             prefetch_host_bytes=None, prefetch_device_bytes=None,
                             debug=False, report=None, device_transform=None):
    """Replay P windows for one cfg; publish completion only after the full stream.

    factory(window, cfg) binds readers to that window's local IDs. Each yielded
    point index is global within the cfg. An optional windows.next_ready() lets
    the same IO queue admit the next planned window without blocking compute.
    Closing or failing closes both the reader queue and the planning producer.
    """
    windows = iter(windows)
    owner = events = None
    first_start = perf_counter()
    try:
        try:
            first = next(windows)
        except StopIteration as error:
            raise ValueError('cfg execution requires at least one planned window') from error
        first_wait = perf_counter() - first_start
        if not isinstance(first, CfgPlan) or getattr(backend, '__name__', None) != first.backend:
            raise ValueError('execution backend differs from the planned window')
        validate_cfg_allocation_envelope(first)
        if device_transform is not None and first.backend != 'cupy':
            raise ValueError('device output postprocessing requires a GPU execution target')
        if device_memory_limit is None and first.backend == 'cupy':
            device_memory_limit = min(t.plan.memory_limit for t in first.templates)
        if host_memory_limit is None:
            host_memory_limit = first.statistics.get('host_memory_cap')
        _capacities(first, host_memory_limit, device_memory_limit, prefetch_host_bytes, prefetch_device_bytes)
        with execution_debug(debug), device_execution(backend, device_memory_limit=device_memory_limit):
            owner = _WindowQueue(first, windows, configuration, loader_factory,
                host_memory_limit, device_memory_limit, prefetch_host_bytes, prefetch_device_bytes,
                active_device_execution(), backend)
            details = {}
            events = _iter_execute_cfg_plan(first, (), None,
                host_memory_limit=host_memory_limit, device_memory_limit=device_memory_limit,
                prefetch_host_bytes=prefetch_host_bytes, prefetch_device_bytes=prefetch_device_bytes,
                report=details, _owner=owner, device_transform=device_transform)
            try:
                for cfg, index, values in events:
                    if index is not None:
                        yield cfg, index, values
            finally:
                events.close()
            details.update(completed=(configuration,), planned_windows=owner.serial,
                first_window_wait_seconds=first_wait,
                planning_wait_seconds=owner.planning_wait_seconds, peak_bound_windows=owner.peak_bound_windows)
            if report is not None:
                report.update(details)
            yield configuration, None, None
            # Publication may finish device means/packing/download inside the
            # completion callback, after the last numerical window ended.
            device = active_device_execution()
            if report is not None and device is not None:
                report['device'] = dict(device.statistics)
    finally:
        callbacks = []
        if events is not None:
            callbacks.append(events.close)
        if owner is not None:
            callbacks.append(owner.close)
        close = getattr(windows, 'close', None)
        if close is not None:
            callbacks.append(close)
        _close_many(callbacks, sys.exc_info()[1])


def execute_cfg_windows(windows, configuration, loader_factory, consume, **options):
    """Callback facade; consume(cfg, None, None) occurs once on complete success."""
    report = {}
    with OutputConsumerQueue(consume, device_transform=options.get("device_transform")) as output_queue:
        events = iter_execute_cfg_windows(windows, configuration, loader_factory, report=report, **options)
        try:
            for cfg, index, values in events:
                output_queue.submit(cfg, index, values)
        finally:
            _close_many((events.close,), sys.exc_info()[1])
    report.update(output_queue.report())
    return report


__all__ = ['execute_cfg_windows', 'iter_execute_cfg_windows']
