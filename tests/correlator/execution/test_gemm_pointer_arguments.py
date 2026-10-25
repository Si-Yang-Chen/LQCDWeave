"""Current GEMM table owners use the existing fixed raw-argument ABI."""
import gc
import importlib
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.gemm_group import (
    _bind_group_pointer_setup, _pointer_kernel, _pointer_source,
)
from lqcdweave.contraction.execution.raw_arguments import _BoundPointerArguments
from .test_raw_consumer_arguments import _SlotBackend


def _current(backend, lanes, mask):
    count = sum(1 if mask & (1 << side) else lanes for side in (1, 0))
    return (tuple(backend.empty((2, 3), np.float64) for _ in range(count)),
            backend.empty((64,), np.uint64), backend.empty((lanes, 2, 3), np.float64))


class GemmPointerArgumentsCpuTests(unittest.TestCase):
    def test_all_masks_and_widths_bind_fresh_ordered_addresses_without_retaining_owners(self):
        for lanes in (2, 8, 21):
            backend = _SlotBackend()
            setup = _bind_group_pointer_setup(backend,
                {mask: backend.RawKernel('', '') for mask in range(4)}, lanes)
            for mask in range(4):
                previous_addresses = None
                for generation in range(2):
                    with self.subTest(lanes=lanes, mask=mask, generation=generation):
                        packed, table, cube = _current(backend, lanes, mask)
                        references = tuple(weakref.ref(value) for value in (*packed, table, cube))
                        wanted = tuple(value.data.ptr for value in (*packed, table, cube))
                        setup(packed, mask, table, cube)
                        self.assertEqual(backend.launches[-1][2], wanted)
                        bound = setup.native_arguments[mask]
                        self.assertIsInstance(bound, _BoundPointerArguments)
                        self.assertEqual(tuple(bound._local.storage[0]), (0,) * len(wanted))
                        if previous_addresses is not None:
                            self.assertNotEqual(wanted, previous_addresses)
                        # Hold the previous allocation through the next call,
                        # so allocator recycling cannot make freshness vacuous.
                        kept = packed, table, cube
                        previous_addresses = wanted
                del kept, packed, table, cube
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))

    def test_address_and_submission_errors_clear_all_slots_and_preserve_first_error(self):
        backend = _SlotBackend()
        kernels = {0: backend.RawKernel('', '')}
        setup = _bind_group_pointer_setup(backend, kernels, 21)
        packed, table, cube = _current(backend, 21, 0)
        error = RuntimeError('setup submission failed')
        with patch.object(type(kernels[0]), '__call__', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                setup(packed, 0, table, cube)
        self.assertIs(caught.exception, error)
        slots = setup.native_arguments[0]._local.storage[0]
        self.assertEqual(tuple(slots), (0,) * 44)

        class Broken:
            @property
            def data(self):
                raise error

        launches = len(backend.launches)
        with self.assertRaises(RuntimeError) as caught:
            setup((*packed[:1], Broken(), *packed[2:]), 0, table, cube)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(backend.launches), launches)
        self.assertEqual(tuple(slots), (0,) * 44)
        setup(packed, 0, table, cube)
        self.assertEqual(backend.launches[-1][2], tuple(value.data.ptr for value in (*packed, table, cube)))

    def test_debug_and_unsupported_kernels_keep_original_argument_handoff(self):
        backend = _SlotBackend()
        for debug in (False, True):
            for mask in range(4):
                with self.subTest(debug=debug, mask=mask):
                    kernel = Mock()
                    # Capability fallback remains available to custom kernels.
                    setup = _bind_group_pointer_setup(backend, {mask: kernel}, 21)
                    packed, table, cube = _current(backend, 21, mask)
                    with execution_debug(debug):
                        setup(packed, mask, table, cube)
                    kernel.assert_called_once_with((1,), (1,), (*packed, table, cube))

        real = backend.RawKernel('', '')
        setup = _bind_group_pointer_setup(backend, {3: real}, 21)
        packed, table, cube = _current(backend, 21, 3)
        with (execution_debug(True), patch.object(type(real), '__call__') as original,
              patch.object(_BoundPointerArguments, '__call__', side_effect=AssertionError('diagnostic used slots'))):
            setup(packed, 3, table, cube)
        original.assert_called_once_with((1,), (1,), (*packed, table, cube))


class GemmPointerArgumentsCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no device')
        except Exception as error:
            raise unittest.SkipTest(f'CuPy device unavailable: {error}')

    def test_current_nondefault_stream_table_matches_all_max_width_masks(self):
        cp = self.cp
        lanes, m, k, n = 21, 2, 3, 4
        dtype = np.dtype(np.complex128)
        kernels = {mask: _pointer_kernel(cp.RawKernel, _pointer_source(lanes, m * n, dtype.itemsize,
            input_elements=(m * k, k * n), packed_mask=mask)) for mask in range(4)}
        setup = _bind_group_pointer_setup(cp, kernels, lanes)
        stream = cp.cuda.Stream(non_blocking=True)
        previous = None
        with stream:
            for mask in range(4):
                for generation in range(2):
                    with self.subTest(mask=mask, generation=generation):
                        b = ((cp.empty((lanes, k, n), dtype=dtype),) if mask & 2 else
                             tuple(cp.empty((k, n), dtype=dtype) for _ in range(lanes)))
                        a = ((cp.empty((lanes, m, k), dtype=dtype),) if mask & 1 else
                             tuple(cp.empty((m, k), dtype=dtype) for _ in range(lanes)))
                        table = cp.empty((64,), dtype=cp.uint64)
                        cube = cp.empty((lanes, m, n), dtype=dtype)
                        setup((*b, *a), mask, table, cube)
                        bound = setup.native_arguments[mask]
                        self.assertIsNotNone(bound)
                        self.assertEqual(tuple(bound._local.storage[0]), tuple(value.data.ptr for value in (*b, *a, table, cube)))
                        # Only the test waits to inspect completed table values.
                        stream.synchronize()
                        expected_b = tuple(b[0].data.ptr + lane * k * n * dtype.itemsize
                                           if mask & 2 else b[lane].data.ptr for lane in range(lanes))
                        expected_a = tuple(a[0].data.ptr + lane * m * k * dtype.itemsize
                                           if mask & 1 else a[lane].data.ptr for lane in range(lanes))
                        expected_c = tuple(cube.data.ptr + lane * m * n * dtype.itemsize for lane in range(lanes))
                        np.testing.assert_array_equal(cp.asnumpy(table[:3 * lanes]),
                            np.asarray((*expected_b, *expected_a, *expected_c), dtype=np.uint64))
                        if previous is not None:
                            self.assertNotEqual(cube.data.ptr, previous[-1].data.ptr)
                        previous = (*b, *a, table, cube)


if __name__ == '__main__':
    unittest.main()
