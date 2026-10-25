"""Event/ownership oracles for sharing the existing device read-ahead budget."""
import gc
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

from lqcdweave.contraction.execution.device import device_execution
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.read_ahead import ReadAhead
from lqcdweave.contracts.correlator.api import device_allocation_bytes
from .test_upload_lifecycle import _AsyncCupy, _Physical, _queue as _upload_queue


def _queue(device, physical, keys=('A', 'B'), *, device_capacity=None):
    # Keep the upload oracle's host tickets but model physical device owners:
    # one 32-byte fake array consumes one 512-byte CuPy allocation unit.
    plan, loader, description = _upload_queue(device, physical, keys=keys)
    sizes = {key: device_allocation_bytes(physical.host.nbytes) for key in keys}
    queue = ReadAhead(plan, loader, description.sizes, description.capacity,
        device_sizes=sizes, device_capacity=(sum(sizes.values())
            if device_capacity is None else device_capacity))
    return plan, loader, queue


def _empty_queue(capacity):
    return ReadAhead(SimpleNamespace(events=(), blocks={}), None, {}, 64,
                     device_capacity=capacity)


def _last_use(backend, value):
    reference = weakref.ref(value)
    def use():
        owner = reference()
        if owner is None:
            raise AssertionError('input was freed before its last compute use')
        owner[0] += 7
    backend.cuda.compute.commands.append(use)
    return reference


class InputRetirementCpuTests(unittest.TestCase):
    def test_current_owner_lives_until_compute_done_without_upload_barrier(self):
        backend = _AsyncCupy()
        with device_execution(backend, device_memory_limit=4096) as device:
            queue = _empty_queue(1024)
            value = np.arange(4, dtype=np.float64)
            reference = _last_use(backend, value)
            device.release_input(value, queue)
            ticket = queue.retired_inputs[0][1]
            del value
            gc.collect()
            self.assertIsNotNone(reference())
            self.assertEqual(queue.reserved_device_bytes, 512)
            self.assertFalse(device.upload_stream.waits)
            self.assertEqual(backend.cuda.compute.synchronizations, 0)
            backend.cuda.compute.finish()
            self.assertEqual(reference()[0], 7)
            queue.start_after(0)
            self.assertIsNone(ticket.value)
            self.assertIsNone(reference())
            self.assertFalse(device._retired_inputs)
            self.assertEqual(queue.reserved_device_bytes, 0)
            self.assertEqual(queue.peak_retired_input_bytes, 512)
            self.assertEqual(queue.retired_input_count, 1)
            queue.close()

    def test_following_upload_finishes_while_retired_compute_remains_pending(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A', 'B'), device_capacity=1024)
            first = loader.read_block('A', plan.blocks['A'])
            _last_use(backend, first)
            loader.close_block('A', first)
            event = queue.retired_inputs[0][1].event
            following = device.upload(physical.host + 100)
            device.upload_stream.finish()
            np.testing.assert_array_equal(device.ready(following), physical.host + 100)
            self.assertFalse(event.query())
            self.assertFalse(device.upload_stream.waits)
            queue.close()
            self.assertEqual(first[0], 7)
            self.assertEqual([key for key, _ in physical.closed], ['A'])

    def test_retirement_and_pending_upload_share_one_device_capacity(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, queue = _queue(device, physical, keys=('A',), device_capacity=1024)
            value = np.arange(4, dtype=np.float64)
            _last_use(backend, value)
            device.release_input(value, queue)
            queue.start_after(-1)
            self.assertEqual(queue.reserved_device_bytes, 1024)
            other = np.ones(4, dtype=np.float64)
            _last_use(backend, other)
            device.release_input(other, queue)
            self.assertEqual(len(queue.retired_inputs), 1)
            self.assertEqual(len(device.upload_stream.waits), 1)
            self.assertLessEqual(queue.peak_device_bytes, queue.device_capacity)
            queue.close()
            self.assertEqual(queue.reserved_device_bytes, 0)
            self.assertFalse(device._retired_inputs)

    def test_pressure_waits_only_retired_event_before_admitting_next_read(self):
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, _, queue = _queue(device, physical, keys=('A',), device_capacity=512)
            value = np.arange(4, dtype=np.float64)
            _last_use(backend, value)
            device.release_input(value, queue)
            event = queue.retired_inputs[0][1].event
            queue.start_after(-1)
            self.assertEqual(event.synchronizations, 1)
            self.assertEqual(backend.cuda.compute.synchronizations, 0)
            self.assertEqual(device.upload_stream.synchronizations, 0)
            self.assertEqual(set(queue.pending), {'A'})
            self.assertEqual(queue.reserved_device_bytes, 512)
            self.assertEqual(queue.peak_device_bytes, 512)
            queue.close()

    def test_zero_budget_debug_closed_queue_and_unbound_loader_keep_fence(self):
        for mode in ('zero', 'debug', 'closed', 'unbound'):
            with self.subTest(mode=mode):
                backend = _AsyncCupy()
                with device_execution(backend, device_memory_limit=4096) as device:
                    queue = _empty_queue(0 if mode == 'zero' else 1024)
                    if mode == 'closed':
                        queue.close()
                    value = np.arange(4, dtype=np.float64)
                    _last_use(backend, value)
                    with execution_debug(mode == 'debug'):
                        device.release_input(value, None if mode == 'unbound' else queue)
                    self.assertEqual(len(device.upload_stream.waits), 1)
                    self.assertFalse(queue.retired_inputs)
                    queue.close()

    def test_query_error_fences_compute_and_preserves_existing_owner_for_cleanup(self):
        backend = _AsyncCupy()
        with device_execution(backend, device_memory_limit=4096) as device:
            queue = _empty_queue(1024)
            first = np.arange(4, dtype=np.float64)
            _last_use(backend, first)
            device.release_input(first, queue)
            ticket = queue.retired_inputs[0][1]
            ticket.event.fail_query = True
            second = np.ones(4, dtype=np.float64)
            _last_use(backend, second)
            with self.assertRaisesRegex(RuntimeError, 'event query failed'):
                device.release_input(second, queue)
            self.assertEqual(backend.cuda.compute.synchronizations, 1)
            self.assertEqual((first[0], second[0]), (7, 8))
            self.assertIn(ticket, device._retired_inputs)
            queue.close()
            self.assertFalse(device._retired_inputs)

    def test_close_failure_keeps_device_owner_until_scope_cleanup(self):
        backend = _AsyncCupy()
        with device_execution(backend, device_memory_limit=4096) as device:
            queue = _empty_queue(512)
            value = np.arange(4, dtype=np.float64)
            reference = _last_use(backend, value)
            device.release_input(value, queue)
            ticket = queue.retired_inputs[0][1]
            ticket.event.fail_sync = True
            del value
            with self.assertRaisesRegex(RuntimeError, 'event wait failed'):
                queue.close()
            self.assertIsNotNone(reference())
            self.assertIn(ticket, device._retired_inputs)
        self.assertIsNone(ticket.value)
        self.assertIsNone(reference())
        self.assertFalse(device._retired_inputs)

    def test_cfg_extension_binds_same_budget_and_retirement_metadata_is_bounded(self):
        from unittest.mock import patch
        backend, physical = _AsyncCupy(), _Physical()
        with device_execution(backend, device_memory_limit=4096) as device:
            plan, loader, _ = _queue(device, physical, keys=('A',))
            queue = _empty_queue(1536)
            queue.extend(plan, loader, {'A': 64}, device_sizes={'A': 512}, prefix='cfg-7:')
            self.assertIs(loader._read_ahead, queue)
            with patch('lqcdweave.contraction.execution.read_ahead._MAX_RETIRED_INPUTS', 1):
                for _ in range(2):
                    value = np.arange(4, dtype=np.float64)
                    _last_use(backend, value)
                    device.release_input(value, queue)
            self.assertEqual(len(queue.retired_inputs), 1)
            self.assertEqual(queue.reserved_device_bytes, 512)
            self.assertEqual(len(device.upload_stream.waits), 1)
            queue.close()


class RealCuPyInputRetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
        except ImportError as error:
            raise unittest.SkipTest('requires CuPy') from error
        # Installed but unusable CUDA is an error, never a passing skip.
        cupy.cuda.runtime.getDeviceProperties(0)
        cls.cp = cupy

    def test_next_upload_completes_before_delayed_compute_and_owners_release(self):
        cp = self.cp
        probe = cp.RawKernel(r'''
extern "C" __global__ void ed_input_retirement_probe(double* x, unsigned long long delay) {
    const unsigned long long start = clock64();
    while (clock64() - start < delay) {}
    x[0] += 7.0;
}
''', 'ed_input_retirement_probe')
        properties = cp.cuda.runtime.getDeviceProperties(0)
        cycles = np.uint64(int(properties['clockRate']) * 250)
        # Compile before the event/overlap oracle, so JIT startup cannot hide it.
        probe.compile()
        physical = _Physical()
        with device_execution(cp, device_memory_limit=4 << 20) as device:
            plan, loader, queue = _queue(device, physical, keys=('A', 'B'), device_capacity=1024)
            first = loader.read_block('A', plan.blocks['A'])
            probe((1,), (1,), (first, cycles))
            loader.close_block('A', first)
            self.assertEqual(len(queue.retired_inputs), 1)
            retained = queue.retired_inputs[0][1]
            reference = weakref.ref(first)
            del first
            next_upload = device.upload(physical.host + 100)
            device.complete_upload(next_upload, wait=True)
            self.assertFalse(retained.event.done)
            np.testing.assert_array_equal(cp.asnumpy(next_upload.value), physical.host + 100)
            retained.event.synchronize()
            expected = physical.host.copy()
            expected[0] += 7
            np.testing.assert_array_equal(cp.asnumpy(retained.value), expected)
            queue._reclaim_completed()
            self.assertIsNone(retained.value)
            self.assertIsNone(reference())
            self.assertEqual(queue.reserved_device_bytes, 0)
            self.assertLessEqual(queue.peak_device_bytes, 1024)
            queue.close()
            self.assertFalse(device._retired_inputs)
