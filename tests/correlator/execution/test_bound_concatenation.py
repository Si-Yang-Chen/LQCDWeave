"""Reusable geometry with a fresh zero-copy ownership proof on every call."""
import gc
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from lqcdweave.contraction.execution.operations.dense_batch import (
    _BatchSide, _bind_contiguous_concatenation,
)
from lqcdweave.contraction.execution.operations.result_views import (
    _DeferredResultView, _ResultOwner, bind_result_views,
)


class BoundConcatenationTests(unittest.TestCase):
    def test_binding_reuses_fixed_geometry_but_rechecks_new_owners_and_itemsize(self):
        borrow = _bind_contiguous_concatenation(0, (4, 3, 5),
            input_layouts=((None, 0, 1),) * 4)
        for dtype in (np.float32, np.complex128, np.float64, np.float32):
            owner = np.arange(60).reshape(4, 3, 5).astype(dtype)
            values = tuple(owner[index] for index in range(4))
            with patch('lqcdweave.contraction.execution.operations.dense_batch.prod',
                       side_effect=AssertionError('repeated shape product')):
                result = borrow(values, np.dtype(dtype))
            np.testing.assert_array_equal(result, np.stack(values))
            self.assertTrue(np.shares_memory(result, owner))
            self.assertIsNone(borrow(tuple(value.copy() for value in values), np.dtype(dtype)))
            self.assertIsNone(borrow(values, np.dtype('complex64')))

    def test_dynamic_shapes_strides_owner_order_and_precision_boundaries(self):
        owner = np.arange(36, dtype=np.float64).reshape(6, 6)
        borrow = _bind_contiguous_concatenation(0, (4, 6))
        good = (owner[1:3], owner[3:5])
        np.testing.assert_array_equal(borrow(good, None), owner[1:5])
        cases = (
            (owner[:2], owner[3:5]),
            (good[1], good[0]),
            (good[0], good[0]),
            (good[0], good[1].copy()),
            (good[0], good[1][..., ::-1]),
            (good[0].reshape(3, 4), good[1]),
            (good[0], good[1].view(np.int64)),
        )
        for values in cases:
            self.assertIsNone(borrow(values, None))
        single = _bind_contiguous_concatenation(0, (2, 6))
        self.assertIs(single((good[0],), None), good[0])
        self.assertIsNone(single((good[0].reshape(3, 4),), None))

    def test_singleton_virtual_axes_and_actual_permutations(self):
        for axis in (0, 1, 2):
            logical_shape = list((3, 5))
            logical_shape.insert(axis, 2)
            layout = list((1, 0))
            layout.insert(axis, None)
            owner = np.arange(30, dtype=np.float64).reshape(logical_shape)
            slices = []
            for index in range(2):
                selector = [slice(None)] * 3
                selector[axis] = index
                slices.append(owner[tuple(selector)].T)
            values = tuple(slices)
            borrow = _bind_contiguous_concatenation(axis, logical_shape,
                input_layouts=(tuple(layout),) * 2)
            result = borrow(values, None)
            np.testing.assert_array_equal(result, owner)
            self.assertTrue(np.shares_memory(result, owner))
        source = np.asfortranarray(np.arange(15, dtype=np.float32).reshape(3, 5))
        single = _bind_contiguous_concatenation(0, (1, 5, 3), input_layouts=((None, 1, 0),))
        np.testing.assert_array_equal(single((source,), None), source.T[None])
        self.assertIsNone(single((source[:, ::-1],), None))
        one = np.arange(12, dtype=np.float64).reshape(4, 1, 3)
        singleton = _bind_contiguous_concatenation(0, (4, 1, 3))
        np.testing.assert_array_equal(singleton((one[:2], one[2:]), None), one)

    def test_deferred_range_and_alignment_proofs_and_no_owner_cache(self):
        borrow = _bind_contiguous_concatenation(0, (4, 6))
        owner = np.arange(24, dtype=np.float64).reshape(4, 6).copy()
        owner_ref = weakref.ref(owner)
        layout = bind_result_views(owner.shape, owner.dtype,
            (('all', (slice(None), slice(None)), ()),))[0][1]
        reference = _DeferredResultView(_ResultOwner(owner), layout)
        with patch.object(_DeferredResultView, 'array', side_effect=AssertionError('materialized')):
            result = borrow((reference,), None)
            np.testing.assert_array_equal(result, owner)
            for offset in (-8, 1, 8):
                invalid = _DeferredResultView(reference.owner, replace(layout, offset=offset))
                self.assertIsNone(borrow((invalid,), None))
        del invalid, reference, owner
        gc.collect()
        self.assertIsNotNone(owner_ref())  # The returned view alone owns storage.
        del result
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_failure_preserves_exception_and_binding_can_retry_without_retaining_arrays(self):
        borrow = _bind_contiguous_concatenation(0, (4, 3))
        owner = np.arange(12, dtype=np.float64).reshape(4, 3).copy()
        values = (owner[:2], owner[2:])
        refs = [weakref.ref(value) for value in (*values, owner)]
        with patch('lqcdweave.contraction.execution.operations.dense_batch._array_pointer',
                   side_effect=RuntimeError('current address failed')):
            with self.assertRaisesRegex(RuntimeError, 'current address failed'):
                borrow(values, None)
        del owner, values
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        fresh = np.arange(12, dtype=np.float64).reshape(4, 3) + 10
        np.testing.assert_array_equal(borrow((fresh[:2], fresh[2:]), None), fresh)

    def test_byte_stride_cache_is_safe_for_interleaved_item_sizes(self):
        borrow = _bind_contiguous_concatenation(0, (4, 3))
        def run(index):
            dtype = (np.float32, np.float64, np.complex128)[index % 3]
            owner = np.full((4, 3), index, dtype=dtype)
            result = borrow((owner[:2], owner[2:]), None)
            np.testing.assert_array_equal(result, owner)
            self.assertTrue(np.shares_memory(result, owner))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run, range(90)))

    def test_batch_side_keeps_one_proof_per_fixed_input_mapping(self):
        side = _BatchSide((('a', None), ('b', None)), (2, 3, 5), 0, False,
            original_packer=Mock(side_effect=AssertionError('unexpected packing')),
            stacked=True, prepend=True, input_layouts=((None, 0, 1),) * 2)
        backend = Mock()
        for offset in (0, 100):
            owner = np.arange(30, dtype=np.float64).reshape(2, 3, 5) + offset
            with patch('lqcdweave.contraction.execution.operations.dense_batch._bind_contiguous_concatenation',
                       side_effect=AssertionError('repeated binding')):
                result = side.pack({'a': owner[0], 'b': owner[1]}, backend, np.dtype('float64'))
            np.testing.assert_array_equal(result, owner)
            self.assertTrue(np.shares_memory(result, owner))
        backend.empty.assert_not_called()


if __name__ == '__main__':
    unittest.main()
