"""Shared GEMM/table storage alignment, isolation and ownership without CUDA."""
import gc
import unittest
import weakref

import numpy as np

from lqcdweave.contraction.execution.operations.gemm_group import _bind_group_output_allocation


class _Pointer:
    def __init__(self, owner, offset=0):
        self.owner, self.offset = owner, offset

    def __add__(self, offset):
        return _Pointer(self.owner, self.offset + offset)


class _Allocation:
    def __init__(self, shape, dtype):
        self.buffer = np.empty(shape, dtype=dtype)

    @property
    def data(self):
        return _Pointer(self.buffer)


class _Backend:
    def __init__(self):
        self.allocations = []

    def empty(self, shape, *, dtype):
        value = _Allocation(shape, dtype)
        self.allocations.append((tuple(shape), np.dtype(dtype), weakref.ref(value.buffer)))
        return value

    def ndarray(self, shape, *, dtype, memptr):
        return np.ndarray(shape, dtype=dtype, buffer=memptr.owner, offset=memptr.offset)


class GroupOutputAllocationTests(unittest.TestCase):
    def test_supported_types_odd_sizes_disjoint_regions_and_fresh_results(self):
        for scalar in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=np.dtype(scalar)):
                dtype, shape = np.dtype(scalar), (3, 1, 1)
                backend = _Backend()
                allocate = _bind_group_output_allocation(backend, shape, dtype)
                cube, table = allocate()
                total = 512 + 3 * dtype.itemsize
                self.assertEqual(backend.allocations[0][:2], ((total,), np.dtype(np.uint8)))
                self.assertEqual((cube.shape, cube.dtype, table.shape, table.dtype),
                                 (shape, dtype, (64,), np.dtype(np.uint64)))
                self.assertTrue(cube.flags.c_contiguous and table.flags.c_contiguous)
                self.assertEqual(cube.ctypes.data - table.ctypes.data, 512)
                self.assertEqual(cube.ctypes.data % dtype.alignment, 0)
                self.assertEqual(table.ctypes.data % np.dtype(np.uint64).alignment, 0)
                self.assertFalse(np.shares_memory(cube, table))
                table.fill(np.iinfo(np.uint64).max)
                cube.fill(7)
                np.testing.assert_array_equal(table, np.full(64, np.iinfo(np.uint64).max))
                next_cube, next_table = allocate()
                next_cube.fill(19)
                next_table.fill(0)
                np.testing.assert_array_equal(cube, np.full(shape, 7, dtype=dtype))
                self.assertFalse(np.shares_memory(cube, next_cube))
                self.assertEqual(len(backend.allocations), 2)

    def test_last_result_view_keeps_table_storage_and_then_releases_allocation(self):
        backend = _Backend()
        allocate = _bind_group_output_allocation(backend, (3, 2, 4), np.dtype(np.complex128))
        cube, table = allocate()
        reference = backend.allocations[0][2]
        cube.fill(2+3j)
        kept = cube[0, :, :2]
        del cube, table
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(reference().nbytes, 512+3*2*4*16)
        np.testing.assert_array_equal(kept, np.full((2, 2), 2+3j))
        del kept
        gc.collect()
        self.assertIsNone(reference())

    def test_array_header_failure_does_not_cache_the_failed_allocation(self):
        backend = _Backend()
        original = backend.ndarray
        calls = []

        def failing(shape, **kwargs):
            calls.append(shape)
            if len(calls) == 2:
                raise RuntimeError('cube header failed')
            return original(shape, **kwargs)

        backend.ndarray = failing
        allocate = _bind_group_output_allocation(backend, (3, 2, 4), np.dtype(np.float64))
        with self.assertRaisesRegex(RuntimeError, 'cube header failed'):
            allocate()
        gc.collect()
        self.assertIsNone(backend.allocations[0][2]())
        cube, table = allocate()
        self.assertIsNotNone(backend.allocations[-1][2]())
        self.assertEqual(cube.shape, (3, 2, 4))
        self.assertEqual(table.shape, (64,))


if __name__ == '__main__':
    unittest.main()
