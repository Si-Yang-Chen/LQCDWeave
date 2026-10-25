"""Submission reports use actual packing bindings, without probing CUDA."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    DenseBatchOperand, DenseBatchResult, DenseKernelBatch, TensorSpec,
    cartesian_reduction_geometry,
)
from lqcdweave.contraction.execution.cartesian_group_execution import _bind_combined_packing
from lqcdweave.contraction.execution.operations import packing_kernel
from lqcdweave.contraction.execution.operations.dense_batch import _BatchSide
from lqcdweave.contraction.execution.scene_runtime import _batch_features, _cartesian_group_features


class RecordingBackend:
    """Count dispatches only; these stubs do not compute or initialize CUDA."""
    __name__ = 'cupy'

    def __init__(self):
        self.raw_calls = []
        self.elementwise_calls = []

    def RawKernel(self, source, name, **kwargs):
        def call(grid, block, args):
            self.raw_calls.append((grid, block, len(args)))
        return call

    def ElementwiseKernel(self, *args, **kwargs):
        def call(*values, **kwargs):
            self.elementwise_calls.append(kwargs['size'])
        return call

    def empty(self, shape, dtype, **kwargs):
        return np.empty(shape, dtype, **kwargs)


def metadata_group(backend, *, count=2, m=2, n=3, k=80, dtype='complex64', prepared=()):
    """Construct only immutable geometry, without a planner or numerical arrays."""
    dtype = np.dtype(dtype)
    catalog, batches, layouts = {}, [], []
    for child in range(count):
        operands, sides = [], []
        for axis, rows in enumerate((m, n)):
            keys = tuple(f'{child}:{axis}:{row}' for row in range(rows))
            catalog.update((key, TensorSpec((k,), dtype)) for key in keys)
            operands.append(tuple(DenseBatchOperand(key, (0,)) for key in keys))
            packer = packing_kernel.bind_packing(backend, ((1, k),) * rows,
                (dtype,) * rows, 0, dtype, input_layouts=((None, 0),) * rows)
            sides.append(_BatchSide(tuple((key, (0,)) for key in keys), (rows, k), 0,
                (child, axis) in prepared, packer, getattr(packer, 'pack_original', None),
                True, True, ((None, 0),) * rows))
        results = tuple(DenseBatchResult(f'{child}:out:{i}:{j}',
            ((i, i+1), (j, j+1)), (0, 1)) for i in range(m) for j in range(n))
        catalog.update((result.tensor, TensorSpec((), dtype)) for result in results)
        flags = tuple((child, axis) in prepared for axis in range(2))
        packing = sum(rows*k*dtype.itemsize for rows, ready in zip((m, n), flags) if not ready)
        batch = DenseKernelBatch(tuple(range(child*m*n, (child+1)*m*n)), *operands,
            1, 1, 1, 0, 1, results, 1., packing, m*n*dtype.itemsize,
            dtype.name, flags, (True, True))
        batches.append(batch)
        layouts.append(SimpleNamespace(sides=tuple(sides), dtype=dtype))
    geometry = cartesian_reduction_geometry(batches[0], catalog)
    bound = SimpleNamespace(batches=tuple(batches), layouts=tuple(layouts),
        kernel=SimpleNamespace(partial_bytes=count*geometry.partial_bytes),
        packing=_bind_combined_packing(layouts, catalog, backend))
    return SimpleNamespace(catalog=catalog), bound


class CartesianGroupFeatureTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(packing_kernel, 'bind_pointer_arguments', return_value=None))

    def tearDown(self):
        packing_kernel._raw_kernel.cache_clear()

    def features(self, state, bound):
        return _cartesian_group_features(state, bound, first=True)

    def test_raw_chunk_helper_and_bound_packer_share_the_width(self):
        for width in (32, 16):
            with patch.object(packing_kernel, '_MAX_RAW_INPUTS', width):
                for count in (1, 32, 33, 40, 64):
                    backend = RecordingBackend()
                    packer = packing_kernel.bind_packing(backend, ((1, 4),)*count,
                        (np.float32,)*count, 0, np.float32, input_layouts=((None, 0),)*count)
                    packer.pack_original((np.zeros(4, np.float32),)*count,
                                         np.empty((count, 4), np.float32))
                    expected = (count + width - 1)//width
                    self.assertEqual(len(backend.raw_calls), expected)
                    self.assertEqual(packing_kernel.raw_packing_kernel_calls_upper_bound(count), expected)
                    self.assertEqual(packer.c_contiguous_kernel_calls_upper_bound, expected)
                    self.assertEqual(packer.pack_original.c_contiguous_kernel_calls_upper_bound, expected)

    def test_child_and_combined_widths_include_all_chunks_without_double_counting(self):
        # The 40x40 case was reported as 4 packs despite 6 raw submissions.
        cases = ((2, 32, 2, 4096), (2, 33, 2, 4096), (2, 40, 40, 4096),
                 (2, 64, 2, 4096), (8, 5, 8, 128))
        for dtype in ('float32', 'complex64'):
            for count, m, n, k in cases:
                with self.subTest(dtype=dtype, count=count, m=m, n=n):
                    backend = RecordingBackend()
                    state, bound = metadata_group(backend, count=count, m=m, n=n, k=k, dtype=dtype)
                    features = self.features(state, bound)
                    expected = (count*m+31)//32 + (count*n+31)//32
                    self.assertEqual(features['packing_kernel_calls_upper_bound'], expected)
                    self.assertEqual(features['kernel_calls'], expected+2)
                    self.assertEqual(features['kernel_calls_kind'], 'upper_bound_including_packing')
                    self.assertFalse(features['calibration_eligible'])
                    # Independent zero-stride owners prevent borrowing, without
                    # allocating large inputs or performing any numerical kernel.
                    values = {key: np.broadcast_to(np.empty(1, spec.dtype), spec.shape)
                              for key, spec in state.catalog.items() if spec.shape}
                    for _, side in bound.packing:
                        side.pack(values, backend)
                    self.assertEqual(len(backend.raw_calls), expected)
                    self.assertEqual(_cartesian_group_features(state, bound, first=False)['kernel_calls'], 0)

    def test_prepared_mixed_and_uncombined_sides_count_only_their_own_bindings(self):
        backend = RecordingBackend()
        state, bound = metadata_group(backend, count=4, m=1, n=40, k=128, prepared=((0, 0),))
        # Three combined left inputs and 160 combined right inputs.
        self.assertEqual(self.features(state, bound)['packing_kernel_calls_upper_bound'], 6)
        # Keep right packing per child; the combined left is not also counted
        # as three original sides, and the prepared first left stays zero.
        bound.packing = tuple(item for item in bound.packing if item[0] == 0)
        self.assertEqual(self.features(state, bound)['packing_kernel_calls_upper_bound'], 9)
        bound.packing = ()
        self.assertEqual(self.features(state, bound)['packing_kernel_calls_upper_bound'], 11)
        state, bound = metadata_group(backend, count=4, m=1, n=40, k=128,
                                      prepared=tuple((i, 0) for i in range(4)))
        self.assertEqual(self.features(state, bound)['packing_kernel_calls_upper_bound'], 5)

    def test_contiguous_owner_borrow_does_not_turn_upper_bound_into_observed_count(self):
        backend = RecordingBackend()
        state, bound = metadata_group(backend)
        before = self.features(state, bound)
        values = {}
        owners = []
        for axis, combined in bound.packing:
            owner = np.empty(combined.shape, combined.dtype)
            owners.append(owner)
            row = 0
            for child in combined.members:
                for key, _ in bound.layouts[child].sides[axis].operands:
                    values[key] = owner[row]
                    row += 1
            self.assertTrue(np.shares_memory(combined.pack(values, backend), owner))
        self.assertEqual(backend.raw_calls, [])
        self.assertEqual(before, self.features(state, bound))
        self.assertEqual(before['packing_kernel_calls_upper_bound'], 2)

    def test_elementwise_binding_uses_its_own_chunk_limit(self):
        backend = RecordingBackend()
        backend.RawKernel = None
        state, bound = metadata_group(backend, m=40, n=2, k=128)
        self.assertEqual(bound.packing, ())
        # Each child has ceil(40/8)+ceil(2/8) Elementwise submissions.
        self.assertEqual(self.features(state, bound)['packing_kernel_calls_upper_bound'], 12)
        side = bound.layouts[0].sides[0]
        side.packer((np.empty((1, 128), np.complex64),)*40, np.empty((40, 128), np.complex64))
        self.assertEqual(len(backend.elementwise_calls), 5)
        # High-rank CArray arguments tighten the fallback's ABI below eight
        # sources per submission; its bound comes from the actual binding.
        shape = (1,)*30
        packer = packing_kernel.bind_packing(backend, (shape,)*40, (np.float32,)*40, 0, np.float32)
        before = len(backend.elementwise_calls)
        packer((np.empty(shape, np.float32),)*40, np.empty((40, *(1,)*29), np.float32))
        self.assertEqual(len(backend.elementwise_calls)-before, 8)
        self.assertEqual(packer.c_contiguous_kernel_calls_upper_bound, 8)

    def test_unknown_packer_and_unbound_batches_are_estimates_not_claimed_bounds(self):
        backend = RecordingBackend()
        state, bound = metadata_group(backend)
        bound.packing = ()
        side = bound.layouts[0].sides[0]
        for packer in (None, lambda *args: None):
            bound.layouts[0].sides = (replace(side, original_packer=None, packer=packer),
                                      bound.layouts[0].sides[1])
            features = self.features(state, bound)
            self.assertEqual(features['kernel_calls_kind'], 'estimate_including_packing')
            self.assertNotIn('packing_kernel_calls_upper_bound', features)
            self.assertEqual(features['packing_kernel_calls_estimate'], 4)
        for batch in (bound.batches[0], replace(bound.batches[0], packing_dtype=None)):
            self.assertEqual(_batch_features(state, batch, first=True)['kernel_calls_kind'], 'estimate')


if __name__ == '__main__':
    unittest.main()
