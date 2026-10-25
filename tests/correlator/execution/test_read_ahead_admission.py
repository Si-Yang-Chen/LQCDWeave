"""Bounded admission packets preserve ordered reads, budgets and cleanup."""
from threading import Condition, current_thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .test_read_ahead_device_budget import _Loader, _plan, _RecordingPool
from lqcdweave.contraction.execution.read_ahead import ReadAhead


class _AdmissionProbe(Condition):
    def __init__(self):
        super().__init__()
        self.owner = current_thread()
        self.admission_notifications = 0

    def notify(self, n=1):
        if current_thread() is self.owner:
            self.admission_notifications += 1
        return super().notify(n)


class ReadAheadAdmissionCpuTests(unittest.TestCase):
    def test_small_device_inputs_admit_more_in_packets_with_exact_byte_bounds(self):
        keys = tuple(f'key-{index}' for index in range(1200))
        plan = _plan(keys)
        host, device = dict.fromkeys(keys, 8), dict.fromkeys(keys, 4)
        loader = _Loader(host, device)
        queue = ReadAhead(plan, loader, host, 8192, device_sizes=device, device_capacity=4096)
        condition = queue._condition = _AdmissionProbe()
        try:
            with patch('lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor', _RecordingPool):
                queue.start_after(-1)
                self.assertEqual(queue.pending_read_limit, 1024)
                self.assertEqual(len(queue.pending), 1024)
                self.assertEqual(condition.admission_notifications, 16)
                self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (8192, 4096))
                self.assertEqual(queue.peak_pending_reads, 1024)
                for index, key in enumerate(keys):
                    queue.start_after(index - 1)
                    self.assertEqual(queue.take(key, plan.blocks[key]), (True, 'value:' + key))
                    self.assertLessEqual(queue.reserved_bytes, queue.capacity)
                    self.assertLessEqual(queue.reserved_device_bytes, queue.device_capacity)
                self.assertEqual(queue.pool.submissions, 1)
            self.assertEqual(loader.prefetched, list(keys))
            self.assertEqual(queue.completed_reads, len(keys))
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
        finally:
            queue.close()

    def test_each_domain_limits_expansion_and_cpu_or_zero_device_keep_original_bound(self):
        cases = ((2400, 10000, 4, 1, 600), (10000, 300, 4, 1, 300),
                 (10, 10000, 4, 1, 128), (10000, 0, 4, 1, 128),
                 (10000, 10000, 4, 0, 128), (0, 0, 0, 0, 128))
        keys = tuple(f'key-{index}' for index in range(1100))
        plan = _plan(keys)
        for host_cap, device_cap, host_size, device_size, wanted in cases:
            with self.subTest(host_cap=host_cap, device_cap=device_cap, device_size=device_size):
                host, device = dict.fromkeys(keys, host_size), dict.fromkeys(keys, device_size)
                queue = ReadAhead(plan, _Loader(host, device), host, host_cap,
                                  device_sizes=device, device_capacity=device_cap)
                try:
                    queue.start_after(-1)
                    self.assertEqual(queue.pending_read_limit, wanted)
                    self.assertLessEqual(len(queue.pending), wanted)
                    self.assertLessEqual(queue.reserved_bytes, host_cap)
                    self.assertLessEqual(queue.reserved_device_bytes, device_cap)
                finally:
                    queue.close()

    def test_short_packet_is_flushed_at_quota_or_synchronous_boundary(self):
        plan = _plan(('A', 'B', 'C'))
        for sizes, cap in (({'A': 1, 'B': 1, 'C': 1}, 2), ({'A': 1, 'B': 1}, 3)):
            with self.subTest(sizes=sizes):
                loader = _Loader(sizes, dict.fromkeys(sizes, 0))
                queue = ReadAhead(plan, loader, sizes, cap, device_capacity=0)
                try:
                    queue.start_after(-1)
                    self.assertEqual(queue.submitted_reads, 2)
                    self.assertEqual(queue.take('A', plan.blocks['A']), (True, 'value:A'))
                    self.assertEqual(queue.take('B', plan.blocks['B']), (True, 'value:B'))
                    self.assertEqual(loader.prefetched, ['A', 'B'])
                finally:
                    queue.close()

    def test_admission_exception_flushes_accepted_tickets_and_keeps_first_error(self):
        keys = tuple(f'key-{index}' for index in range(70))
        plan = _plan(keys)
        sizes = dict.fromkeys(keys, 1)
        queue = ReadAhead(plan, _Loader(sizes, dict.fromkeys(keys, 0)), sizes, 70,
                          device_capacity=0)
        del queue.blocks[keys[67]]
        try:
            with self.assertRaises(KeyError) as caught:
                queue.start_after(-1)
            self.assertEqual(caught.exception.args, (keys[67],))
            self.assertEqual(queue.submitted_reads, 67)
            for key in keys[:67]:
                self.assertEqual(queue.take(key, plan.blocks[key]), (True, 'value:' + key))
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
        finally:
            queue.close()

    def test_extension_keeps_cfg_readers_isolated_and_recomputes_only_count_bound(self):
        keys = tuple(f'key-{index}' for index in range(140))
        plan = _plan(keys)
        host, device = dict.fromkeys(keys, 8), dict.fromkeys(keys, 4)

        class Loader(_Loader):
            def __init__(self, label):
                super().__init__(host, device)
                self.label = label

            def prefetch_block(self, key, block):
                return self.label + ':' + super().prefetch_block(key, block)

        queue = ReadAhead(SimpleNamespace(events=(), blocks={}), None, {}, 8192,
                          device_capacity=4096)
        try:
            queue.extend(plan, Loader('2000'), host, device_sizes=device, prefix='2000:')
            queue.extend(plan, Loader('2001'), host, device_sizes=device, prefix='2001:', offset=200)
            queue.start_after(-1)
            self.assertEqual(queue.pending_read_limit, 1024)
            self.assertEqual(queue.submitted_reads, 280)
            for cfg in ('2000', '2001'):
                for key in keys:
                    self.assertEqual(queue.take(cfg + ':' + key, plan.blocks[key]),
                                     (True, cfg + ':value:' + key))
            self.assertEqual(queue.capacity, 8192)
            self.assertEqual(queue.device_capacity, 4096)
        finally:
            queue.close()


class ReadAheadAdmissionCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
        except ImportError as error:
            raise unittest.SkipTest('requires CuPy') from error
        cupy.cuda.runtime.getDeviceProperties(0)
        cls.cp = cupy

    def test_expanded_queue_uploads_current_inputs_before_nondefault_stream_consumption(self):
        import numpy as np
        from lqcdweave.contraction.execution.device import device_execution
        from .test_upload_lifecycle import _Physical, _queue

        class Physical(_Physical):
            def prefetch_block(self, key, block):
                self.prefetched.append(key)
                return self.host + int(key)

        cp = self.cp
        keys = tuple(str(index) for index in range(560))
        physical = Physical()
        outputs = []
        with device_execution(cp, device_memory_limit=64 << 20) as device:
            plan, loader, queue = _queue(device, physical, keys=keys,
                capacity=64 * 1024, device_capacity=32 * 1024)
            try:
                queue.start_after(-1)
                self.assertEqual(queue.pending_read_limit, 1024)
                self.assertEqual(queue.peak_pending_reads, 560)
                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    for key in keys:
                        found, value = queue.take(key, plan.blocks[key])
                        self.assertTrue(found)
                        outputs.append(value * 2 + 5)
                        loader.close_block(key, value)
                # Tests inspect completion; production admission adds no sync.
                stream.synchronize()
                for index, output in enumerate(outputs):
                    np.testing.assert_array_equal(cp.asnumpy(output), (physical.host + index) * 2 + 5)
                self.assertLessEqual(queue.peak_reserved_bytes, queue.capacity)
                self.assertLessEqual(queue.peak_device_bytes, queue.device_capacity)
                self.assertEqual(queue.completed_reads, len(keys))
                self.assertFalse(loader.host_owners)
            finally:
                queue.close()
            self.assertEqual(queue.reserved_bytes, 0)
            self.assertEqual(queue.reserved_device_bytes, 0)
            self.assertFalse(device._uploads)
            self.assertFalse(device._retired_inputs)


if __name__ == '__main__':
    unittest.main()
