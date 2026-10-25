"""Resident upload ordering and quotas, using a CPU-backed device protocol."""
from threading import current_thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.execution.device import device_execution
from lqcdweave.contraction.execution.read_ahead import prepare_read_ahead
from lqcdweave.contracts.correlator.api import StorageBlock, StorageEvent
from correlator.execution.test_device import _FakeCupy, _Tracker, _Event, _Stream


def case(*, fail=(), device_limit=2048, host_limit=1024):
    cp, tracker = _FakeCupy(_Tracker()), None
    tracker = cp.tracker
    events, dependencies = [], []
    class Event(_Event):
        def __init__(self):
            super().__init__(tracker)
            self.done = False
            events.append(self)
        def query(self):
            return self.done
        def synchronize(self):
            self.done = True
    class Stream(_Stream):
        def wait_event(self, event):
            dependencies.append((self.name, event.stream.name))
    cp.cuda.Event = Event
    cp.cuda.Stream = lambda **kw: Stream(tracker, 'upload')
    cp.cuda.compute_stream = Stream(tracker, 'compute')
    values = {key: np.arange(2, dtype=np.float64) + index for index, key in enumerate(('a', 'b', 'c'))}
    blocks = {key: StorageBlock('fixture', key, 16, host_bytes=16,
                               device_bytes=16, staging_host_bytes=16) for key in values}
    plan = SimpleNamespace(blocks=blocks,
        events=tuple(StorageEvent(i, 'read', key) for i, key in enumerate(values)),
        state=SimpleNamespace(problem=SimpleNamespace(backend='cupy')),
        memory_limit=device_limit, host_memory_limit=host_limit,
        peak_host_bytes=0, peak_device_bytes=0)
    reads, closed = [], []
    def read(key, block):
        reads.append((key, current_thread().ident))
        if key in fail:
            raise RuntimeError('read failed: ' + key)
        return values[key].copy()
    physical = SimpleNamespace(preparations={}, specs={},
        input_readers={'host': SimpleNamespace(host_resident=True)},
        can_prefetch_block=lambda *a: True, prefetch_host_bytes=lambda *a: 48,
        prefetch_block=read, finish_prefetch=lambda key, block, value: value,
        close_prefetch=lambda key, value: closed.append(key),
        close_block=lambda key, value: closed.append(key))
    return cp, plan, physical, events, dependencies, reads, closed


def test_inline_uploads_run_without_worker_and_hold_staging_until_copy_finishes():
    cp, plan, physical, events, dependencies, reads, closed = case()
    with device_execution(cp, device_memory_limit=2048) as device:
        loader = device.bind_loader(physical)
        with patch('lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor',
                   side_effect=AssertionError('resident uploads must not start a worker')):
            _, queue, reserve = prepare_read_ahead(plan, loader)
            assert queue.inline and reserve == 3 * 64
            queue.start_after(-1)
            assert [key for key, _ in reads] == ['a', 'b', 'c']
            assert all(thread == current_thread().ident for _, thread in reads)
            found, value = queue.take('a', plan.blocks['a'])
            assert found
            np.testing.assert_array_equal(value, np.arange(2))
            assert ('compute', 'upload') in dependencies
            assert not any(event.done for event in events)
            assert queue.reserved_bytes == reserve
            assert queue.deferred and device._uploads
            for event in events:
                event.done = True
            queue.start_after(0)
            assert queue.reserved_bytes == 2 * 64
            assert queue.reserved_device_bytes == 2 * 512
            # Unconsumed tickets and consumed host owners have separate cleanup.
            queue.close()
            loader.close_block('a', value)
    assert set(closed) == {'a', 'b', 'c'}


def test_inline_failure_is_delivered_at_consumption_and_close_reclaims_all_reservations():
    cp, plan, physical, events, dependencies, reads, closed = case(fail=('b',))
    with device_execution(cp, device_memory_limit=2048) as device:
        loader = device.bind_loader(physical)
        _, queue, _ = prepare_read_ahead(plan, loader)
        queue.start_after(-1)
        with pytest.raises(RuntimeError, match='read failed: b'):
            queue.take('b', plan.blocks['b'])
        queue.close()
        assert queue.reserved_bytes == queue.reserved_device_bytes == 0
    assert set(closed) == {'a', 'c'}


@pytest.mark.parametrize('host_limit,device_limit', [(0, 1024), (1024, 0)])
def test_no_inline_prefetch_without_both_memory_allowances(host_limit, device_limit):
    cp, plan, physical, *_ = case(host_limit=host_limit, device_limit=device_limit)
    with device_execution(cp, device_memory_limit=1024) as device:
        loader = device.bind_loader(physical)
        _, queue, reserve = prepare_read_ahead(plan, loader)
        assert queue is None and reserve == 0
