"""Bounded upload ownership; CPU protocols do not claim real GPU validation."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from correlator.execution.test_device import _FakeCupy, _Tracker, _Device, _Pool
from correlator.execution.test_read_ahead_device_budget import _RecordingPool, _plan
from lqcdweave.contraction.execution.device import DeviceSourceLoader, device_execution
from lqcdweave.contraction.execution.read_ahead import ReadAhead


class _AsyncStream:
    def __init__(self, cuda, name):
        self.cuda, self.name = cuda, name
        self.commands, self.waits = [], []
        self.completed = 0
        self.synchronizations = 0

    def __enter__(self):
        self.previous = self.cuda.current
        self.cuda.current = self
        return self

    def __exit__(self, *_):
        self.cuda.current = self.previous

    def wait_event(self, event):
        self.waits.append(event)
        # A GPU stream wait is an ordered dependency, not a host event wait.
        self.commands.append(lambda: event.stream.finish(event.count))

    def finish(self, count=None):
        count = len(self.commands) if count is None else count
        while self.completed < count:
            self.commands[self.completed]()
            self.completed += 1

    def synchronize(self):
        self.synchronizations += 1
        self.finish()


class _AsyncEvent:
    def __init__(self, cuda):
        self.cuda = cuda
        self.synchronizations = 0
        self.fail_sync = self.fail_query = False

    def record(self, stream):
        if self.cuda.fail_recording:
            raise RuntimeError('event recording failed')
        self.stream, self.count = stream, len(stream.commands)
        self.cuda.events.append(self)

    def query(self):
        if self.fail_query:
            self.fail_query = False
            raise RuntimeError('event query failed')
        return self.stream.completed >= self.count

    def synchronize(self):
        self.synchronizations += 1
        if self.fail_sync:
            self.fail_sync = False
            raise RuntimeError('event wait failed')
        self.stream.finish(self.count)


class _AsyncCuda:
    def __init__(self, tracker):
        self.tracker = tracker
        self.events = []
        self.compute = _AsyncStream(self, 'compute')
        self.current = self.compute
        self.fail_record = False
        self.fail_recording = False

    def MemoryPool(self): return _Pool(self.tracker)
    def using_allocator(self, _): return nullcontext()
    def Device(self, *args): return _Device(*args)
    def Stream(self, *, non_blocking): return _AsyncStream(self, 'upload')
    def Event(self):
        if self.fail_record:
            raise RuntimeError('event creation failed')
        return _AsyncEvent(self)
    def alloc_pinned_memory(self, size): return bytearray(size)
    def get_current_stream(self): return self.current


class _AsyncCupy(_FakeCupy):
    def __init__(self):
        super().__init__(_Tracker())
        self.cuda = _AsyncCuda(self.tracker)

    def asarray(self, host):
        value = np.full_like(host, -999)
        self.cuda.current.commands.append(lambda: np.copyto(value, host))
        return value


class _Physical:
    def __init__(self, *, replacement=False, fail_finish=False, fail_close=False):
        self.replacement, self.fail_finish = replacement, fail_finish
        self.fail_close = fail_close
        self.host = np.arange(4, dtype=np.float64)
        self.prefetched, self.closed, self.closed_prefetch = [], [], []

    def prefetch_host_bytes(self, key, block): return self.host.nbytes
    def can_prefetch_block(self, key, block): return True
    def prefetch_block(self, key, block):
        self.prefetched.append(key)
        return self.host.copy()
    def finish_prefetch(self, key, block, host):
        if self.fail_finish:
            raise ValueError('host finish failed')
        return host + 10 if self.replacement else host
    def close_prefetch(self, key, host):
        self.closed_prefetch.append(key)
        if self.fail_close:
            raise RuntimeError('host cleanup failed')
    def read_block(self, key, block): return self.host.copy()
    def close_block(self, key, host): self.closed.append((key, host))


def _queue(device, physical, keys=('A', 'B'), *, capacity=None, device_capacity=None):
    plan = _plan(keys)
    for block in plan.blocks.values():
        block.host_bytes = block.device_bytes = physical.host.nbytes
    loader = DeviceSourceLoader(physical, device)
    size = 2 * physical.host.nbytes
    queue = ReadAhead(plan, loader, {key: size for key in keys},
        len(keys) * size if capacity is None else capacity,
        device_sizes={key: physical.host.nbytes for key in keys},
        device_capacity=(len(keys) * physical.host.nbytes
                         if device_capacity is None else device_capacity))
    return plan, loader, queue


class UploadLifecycleTests(unittest.TestCase):
    def test_cupy_done_property_retains_upload_until_complete_without_host_wait(self):
        class DoneEvent(_AsyncEvent):
            __module__ = 'cupy.cuda.stream'
            query = None
            def __init__(self, *, disable_timing=False):
                super().__init__(backend.cuda)
                self.disable_timing = disable_timing
            @property
            def done(self):
                return self.stream.completed >= self.count
        backend, physical = _AsyncCupy(), _Physical()
        backend.cuda.Event = DoneEvent
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',))
            try:
                queue.start_after(-1)
                ticket = queue.pending['A'].result()
                self.assertTrue(ticket.event.disable_timing)
                _, value = queue.take('A', plan.blocks['A'])
                self.assertEqual(ticket.event.synchronizations, 0)
                self.assertEqual(backend.cuda.compute.waits, [ticket.event])
                self.assertIn('A', queue.deferred)
                self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (64, 32))
                device.upload_stream.finish()
                queue.start_after(0)
                self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
                self.assertIsNone(ticket.pinned)
                np.testing.assert_array_equal(value, physical.host)
                loader.close_block('A', value)
            finally:
                queue.close()

    def setUp(self):
        self.worker = patch('lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor',
                            _RecordingPool)
        self.worker.start()
        self.addCleanup(self.worker.stop)

    def test_compute_wait_retains_staging_and_both_reservations_until_completion(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',))
            queue.start_after(-1)
            ticket = queue.pending['A'].result()
            pinned = ticket.pinned
            found, value = queue.take('A', plan.blocks['A'])
            self.assertTrue(found)
            self.assertEqual(ticket.event.synchronizations, 0)
            self.assertEqual(backend.cuda.compute.waits, [ticket.event])
            self.assertIs(ticket.pinned, pinned)
            self.assertIn(ticket, device._uploads)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (64, 32))
            self.assertEqual(set(queue.deferred), {'A'})
            loader.close_block('A', value)
            self.assertIs(device.upload_stream.waits[-1].stream, backend.cuda.compute)
            self.assertIs(ticket.pinned, pinned)
            device.upload_stream.finish()
            queue.start_after(0)
            np.testing.assert_array_equal(value, physical.host)
            self.assertIsNone(ticket.pinned)
            self.assertFalse(device._uploads)
            self.assertFalse(queue.deferred)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
            queue.close()

    def test_exhausted_host_or_device_budget_waits_only_oldest_and_recovers(self):
        for capacity, device_capacity in ((64, 64), (128, 32)):
            with self.subTest(host=capacity, device=device_capacity):
                backend, physical = _AsyncCupy(), _Physical()
                with device_execution(backend, device_memory_limit=4096) as device:
                    plan, loader, queue = _queue(device, physical,
                        capacity=capacity, device_capacity=device_capacity)
                    queue.start_after(-1)
                    first = queue.pending['A'].result()
                    _, value = queue.take('A', plan.blocks['A'])
                    self.assertEqual(first.event.synchronizations, 0)
                    queue.start_after(0)
                    second = queue.pending['B'].result()
                    self.assertEqual(first.event.synchronizations, 1)
                    self.assertEqual(second.event.synchronizations, 0)
                    self.assertIsNotNone(second.pinned)
                    self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (64, 32))
                    self.assertLessEqual(queue.peak_reserved_bytes, capacity)
                    self.assertLessEqual(queue.peak_device_bytes, device_capacity)
                    self.assertEqual(device.upload_stream.synchronizations, 0)
                    loader.close_block('A', value)
                    queue.close()

    def test_deferred_ticket_metadata_has_a_fixed_bound(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A', 'B', 'C', 'D'))
            queue.start_after(-1)
            tickets = {key: future.result() for key, future in queue.pending.items()}
            with patch('lqcdweave.contraction.execution.read_ahead._MAX_DEFERRED_READS', 2):
                for key in ('A', 'B', 'C'):
                    _, value = queue.take(key, plan.blocks[key])
                    loader.close_block(key, value)
                    self.assertLessEqual(len(queue.deferred), 2)
            self.assertEqual(tickets['A'].event.synchronizations, 1)
            self.assertEqual(tickets['B'].event.synchronizations, 0)
            self.assertEqual(set(queue.deferred), {'B', 'C'})
            queue.close()

    def test_changed_host_drains_old_upload_before_synchronous_reupload(self):
        backend, physical = _AsyncCupy(), _Physical(replacement=True)
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',))
            queue.start_after(-1)
            ticket = queue.pending['A'].result()
            _, value = queue.take('A', plan.blocks['A'])
            np.testing.assert_array_equal(value, physical.host + 10)
            self.assertTrue(ticket.complete)
            self.assertIsNone(ticket.value)
            self.assertFalse(device._uploads)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
            self.assertEqual([event.synchronizations for event in backend.cuda.events], [1, 1])
            loader.close_block('A', value)
            self.assertFalse(physical.closed_prefetch)
            np.testing.assert_array_equal(physical.closed[0][1], physical.host + 10)
            queue.close()

    def test_direct_reads_keep_synchronous_staging_fallback(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',))
            value = loader.read_block('A', plan.blocks['A'])
            np.testing.assert_array_equal(value, physical.host)
            self.assertEqual(backend.cuda.events[0].synchronizations, 1)
            self.assertFalse(backend.cuda.compute.waits)
            self.assertFalse(device._uploads)
            loader.close_block('A', value)
            queue.close()

    def test_legacy_event_without_query_uses_compatible_fallback(self):
        tracker, physical = _Tracker(), _Physical()
        with device_execution(_FakeCupy(tracker), device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',))
            queue.start_after(-1)
            _, value = queue.take('A', plan.blocks['A'])
            np.testing.assert_array_equal(value, physical.host)
            self.assertEqual(tracker.event_waits, 1)
            self.assertFalse(queue.deferred)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
            loader.close_block('A', value)
            queue.close()

    def test_early_close_drains_pending_and_consumed_uploads_without_double_close(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical)
            queue.start_after(-1)
            tickets = [future.result() for future in queue.pending.values()]
            _, value = queue.take('A', plan.blocks['A'])
            queue.close()
            self.assertFalse(device._uploads)
            self.assertTrue(all(ticket.pinned is None for ticket in tickets))
            self.assertEqual(physical.closed_prefetch, ['B'])
            self.assertIn('A', loader.host_owners)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
            loader.close_block('A', value)
            self.assertEqual([key for key, _ in physical.closed], ['A'])
            queue.close()

    def test_failed_host_finish_drains_transfer_and_preserves_primary_cleanup_error(self):
        backend, physical = _AsyncCupy(), _Physical(fail_finish=True, fail_close=True)
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, _, queue = _queue(device, physical, keys=('A',))
            queue.start_after(-1)
            ticket = queue.pending['A'].result()
            with self.assertRaisesRegex(ValueError, 'host finish failed') as caught:
                queue.take('A', plan.blocks['A'])
            self.assertTrue(any('host cleanup failed' in note for note in caught.exception.__notes__))
            self.assertIsNone(ticket.pinned)
            self.assertEqual(physical.closed_prefetch, ['A'])
            queue.close()
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))

    def test_failed_event_wait_keeps_ticket_and_reservation_for_close_retry(self):
        backend, physical = _AsyncCupy(), _Physical(fail_finish=True)
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, _, queue = _queue(device, physical, keys=('A',))
            queue.start_after(-1)
            ticket = queue.pending['A'].result()
            ticket.event.fail_sync = True
            with self.assertRaisesRegex(ValueError, 'host finish failed') as caught:
                queue.take('A', plan.blocks['A'])
            self.assertTrue(any('event wait failed' in note for note in caught.exception.__notes__))
            self.assertIsNotNone(ticket.pinned)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (64, 32))
            queue.close()
            self.assertIsNone(ticket.pinned)
            self.assertFalse(device._uploads)
            self.assertEqual(physical.closed_prefetch, ['A'])

    def test_failed_enqueue_without_event_drains_before_host_cleanup(self):
        for failure in ('fail_record', 'fail_recording'):
            with self.subTest(failure=failure):
                backend, physical = _AsyncCupy(), _Physical()
                with device_execution(backend, device_memory_limit=4096) as device:
                    plan, _, queue = _queue(device, physical, keys=('A',))
                    setattr(backend.cuda, failure, True)
                    queue.start_after(-1)
                    ticket = queue.pending['A'].result()
                    with self.assertRaisesRegex(RuntimeError, 'event .* failed'):
                        queue.take('A', plan.blocks['A'])
                    self.assertTrue(ticket.complete)
                    self.assertIsNone(ticket.event)
                    self.assertIsNone(ticket.pinned)
                    self.assertFalse(device._uploads)
                    self.assertEqual(device.upload_stream.synchronizations, 1)
                    self.assertEqual(physical.closed_prefetch, ['A'])
                    self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
                    queue.close()

    def test_query_failure_closes_prefetch_and_releases_reservations(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, _, queue = _queue(device, physical, keys=('A',))
            queue.start_after(-1)
            ticket = queue.pending['A'].result()
            ticket.event.fail_query = True
            with self.assertRaisesRegex(RuntimeError, 'event query failed'):
                queue.take('A', plan.blocks['A'])
            self.assertTrue(ticket.complete)
            self.assertIsNone(ticket.pinned)
            self.assertEqual(physical.closed_prefetch, ['A'])
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
            queue.close()

    def test_scope_exit_reclaims_remaining_transfer_owners(self):
        backend = _AsyncCupy()
        with device_execution(backend, device_memory_limit=4096) as device:
            ticket = device.upload(np.arange(4))
            device.ready(ticket)
            self.assertIsNotNone(ticket.pinned)
            self.assertEqual(ticket.event.synchronizations, 0)
        self.assertTrue(ticket.complete)
        self.assertIsNone(ticket.pinned)
        self.assertTrue(backend.tracker.pools[0].freed)

    def test_prefixed_cfg_reads_keep_completion_ownership_after_consumption(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, _ = _queue(device, physical, keys=('A',))
            queue = ReadAhead(SimpleNamespace(events=(), blocks={}), None, {}, 64,
                              device_capacity=32)
            queue.extend(plan, loader, {'A': 64}, device_sizes={'A': 32}, prefix='cfg-3:')
            queue.start_after(-1)
            ticket = queue.pending['cfg-3:A'].result()
            _, value = queue.take('cfg-3:A', plan.blocks['A'])
            self.assertIn('cfg-3:A', queue.owners)
            self.assertIn('cfg-3:A', queue.deferred)
            self.assertIn('A', loader.host_owners)
            queue.close()
            self.assertIsNone(ticket.pinned)
            loader.close_block('A', value)
            self.assertEqual([key for key, _ in physical.closed], ['A'])

    def test_changed_host_reupload_failure_closes_transferred_host_owner(self):
        backend = _AsyncCupy()

        class Replacing(_Physical):
            def finish_prefetch(self, key, block, host):
                backend.cuda.fail_record = True
                return host + 10

        physical = Replacing()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',))
            queue.start_after(-1)
            with self.assertRaisesRegex(RuntimeError, 'event creation failed'):
                queue.take('A', plan.blocks['A'])
            self.assertFalse(loader.host_owners)
            self.assertFalse(device._uploads)
            self.assertFalse(physical.closed_prefetch)
            np.testing.assert_array_equal(physical.closed[0][1], physical.host + 10)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
            queue.close()

    def test_reclamation_error_does_not_orphan_next_pending_read(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical)
            queue.start_after(-1)
            first = queue.pending['A'].result()
            _, value = queue.take('A', plan.blocks['A'])
            first.event.fail_query = True
            with self.assertRaisesRegex(RuntimeError, 'event query failed'):
                queue.take('B', plan.blocks['B'])
            self.assertIn('B', queue.pending)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (128, 64))
            queue.close()
            loader.close_block('A', value)
            self.assertEqual(physical.closed_prefetch, ['B'])

    def test_direct_read_failure_preserves_error_when_host_cleanup_also_fails(self):
        backend = _AsyncCupy()

        class FailingClose(_Physical):
            def close_block(self, key, host):
                raise ValueError('host close failed')

        physical = FailingClose()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',))
            backend.cuda.fail_recording = True
            with self.assertRaisesRegex(RuntimeError, 'event recording failed') as caught:
                loader.read_block('A', plan.blocks['A'])
            self.assertTrue(any('host close failed' in note for note in caught.exception.__notes__))
            self.assertFalse(device._uploads)
            self.assertFalse(loader.host_owners)
            queue.close()

    def test_query_capable_cfg_protocol_matches_numpy_across_configurations(self):
        from correlator.execution.test_device import (
            _Cuda, _Event, _Stream, _targets, _complete, _run, _assert_tree_allclose,
        )

        class QueryEvent(_Event):
            complete = False
            def query(self): return self.complete
            def synchronize(self):
                if self.stream.name == 'upload':
                    super().synchronize()
                # FakeCupy executes numerical operations eagerly. A compute
                # event needs no additional operation or whole-stream fence.
                self.complete = True

        class DependencyStream(_Stream):
            def wait_event(self, event):
                if self.name == 'compute':
                    self.tracker.compute_dependencies.append(event)
                else:
                    super().wait_event(event)

        class QueryCuda(_Cuda):
            def __init__(self, tracker):
                super().__init__(tracker)
                self.compute_stream = DependencyStream(tracker, 'compute')
            def Stream(self, *, non_blocking):
                return DependencyStream(self.tracker, 'upload')
            def Event(self): return QueryEvent(self.tracker)

        cpu_target, gpu_target = _targets()
        cpu_selected, cpu_complete = _complete(cpu_target)
        selected, complete = _complete(gpu_target)
        expected, _, _ = _run(cpu_complete, cpu_selected, np, _Tracker())
        tracker = _Tracker()
        tracker.compute_dependencies = []
        backend = _FakeCupy(tracker)
        backend.cuda = QueryCuda(tracker)
        actual, report, closed = _run(complete, selected, backend, tracker)
        _assert_tree_allclose(actual, expected)
        self.assertEqual(closed, ['a', 'b'])
        self.assertEqual(report['completed'], ('a', 'b'))
        self.assertTrue(tracker.compute_dependencies)
        self.assertTrue(all(event.stream.name == 'upload'
                            for event in tracker.compute_dependencies))
        self.assertEqual(tracker.stream_syncs.count('compute'), 1)
        self.assertTrue(tracker.pools[0].freed)


class RealCuPyUploadLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
        except ImportError as exc:
            raise unittest.SkipTest('real CuPy upload checks require CuPy and allocated CUDA/HIP') from exc
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest('real CuPy upload checks require an allocated GPU')
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest(f'real CuPy upload checks require an allocated GPU: {exc}') from exc
        cls.cupy = cupy

    def test_upload_to_distinct_compute_stream_and_safe_pool_reuse(self):
        cp = self.cupy

        class DifferentInputs(_Physical):
            def prefetch_block(self, key, block):
                return self.host.copy() if key == 'A' else self.host + 100

        physical = DifferentInputs()
        physical.host = np.arange(1 << 16, dtype=np.float64)
        delay = cp.RawKernel(r'''
            extern "C" __global__ void wait_cycles(unsigned long long cycles) {
                unsigned long long start = clock64();
                while (clock64() - start < cycles) {}
            }
        ''', 'wait_cycles')
        # Compile before timing-sensitive ordering checks. This is a bounded
        # device delay, not a benchmark or a host callback waiting for Python.
        delay((1,), (1,), (np.uint64(1),))
        cp.cuda.get_current_stream().synchronize()
        with device_execution(cp, device_memory_limit=8 << 20) as device:
            plan, loader, queue = _queue(device, physical, keys=('A', 'B'),
                capacity=2 * physical.host.nbytes, device_capacity=physical.host.nbytes)
            compute = cp.cuda.Stream(non_blocking=True)
            try:
                with device.upload_stream:
                    delay((1,), (1,), (np.uint64(100_000_000),))
                with patch('lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor',
                           _RecordingPool):
                    queue.start_after(-1)
                    first_upload = queue.pending['A'].result()
                with compute, patch.object(device, 'complete_upload',
                        wraps=device.complete_upload) as complete:
                    _, value = queue.take('A', plan.blocks['A'])
                    # The normal take must not introduce a blocking completion.
                    self.assertFalse(any(call.kwargs.get('wait', False)
                                         for call in complete.call_args_list))
                    delay((1,), (1,), (np.uint64(100_000_000),))
                    result = value * 3 + 7
                    loader.close_block('A', value)
                    del value
                    # A is released while compute still reads it. Any upload
                    # allocation reused for the different B data must wait for
                    # the compute marker recorded by close_block.
                    queue.start_after(0)
                    _, next_value = queue.take('B', plan.blocks['B'])
                    next_result = next_value * 5 - 2
                    loader.close_block('B', next_value)
                    del next_value
                compute.synchronize()
                queue.start_after(1)
                self.assertFalse(device._uploads)
                self.assertFalse(queue.deferred)
                self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
                np.testing.assert_array_equal(cp.asnumpy(result), physical.host * 3 + 7)
                np.testing.assert_array_equal(cp.asnumpy(next_result), (physical.host + 100) * 5 - 2)
                # Completed dependency events remain usable for ordering, but
                # deliberately carry no CUDA timing data.
                with self.assertRaises(cp.cuda.runtime.CUDARuntimeError):
                    cp.cuda.get_elapsed_time(first_upload.event, first_upload.event)
            finally:
                queue.close()


if __name__ == '__main__':
    unittest.main()
