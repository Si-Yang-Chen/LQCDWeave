"""CPU checks for the fixed result-view binding used by grouped GEMMs."""
from types import SimpleNamespace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    DenseBatchOperand,
    DenseBatchResult,
    DenseGemmGroup,
    DenseKernelBatch,
    TensorSpec,
)
from lqcdweave.contraction.execution.operations.gemm_group import (
    _bind_group_result_views,
    _bind_layouts,
    bind_dense_gemm_group,
)


_DTYPE = np.dtype(np.complex128)


def _multiaxis_batch(lane, catalog):
    left = f"multi_left_{lane}"
    right_a, right_b = f"multi_right_{lane}_a", f"multi_right_{lane}_b"
    out_a, out_b = f"multi_out_{lane}_a", f"multi_out_{lane}_b"
    catalog[left] = TensorSpec((2, 3, 4), _DTYPE)
    catalog[right_a] = TensorSpec((4, 5, 1), _DTYPE)
    catalog[right_b] = TensorSpec((4, 5, 1), _DTYPE)
    catalog[out_a] = TensorSpec((2, 3, 5, 1), _DTYPE)
    catalog[out_b] = TensorSpec((2, 3, 5, 1), _DTYPE)
    return DenseKernelBatch(
        action_indices=(lane * 2, lane * 2 + 1),
        left=(DenseBatchOperand(left, (0, 1, 2)),),
        right=(DenseBatchOperand(right_a, (0, 1, 2)), DenseBatchOperand(right_b, (0, 1, 2))),
        left_free_rank=2,
        reduction_rank=1,
        right_free_rank=2,
        left_axis=0,
        right_axis=2,
        results=(
            DenseBatchResult(out_a, ((0, 2), (0, 3), (0, 5), (0, 1))),
            DenseBatchResult(out_b, ((0, 2), (0, 3), (0, 5), (1, 2))),
        ),
        seconds=0.1,
        packing_bytes=64 * _DTYPE.itemsize,
        output_bytes=60 * _DTYPE.itemsize,
        packing_dtype=_DTYPE.name,
    )


def _flat_batch(lane, catalog):
    left = f"flat_left_{lane}"
    right_a, right_b = f"flat_right_{lane}_a", f"flat_right_{lane}_b"
    out_a, out_b = f"flat_out_{lane}_a", f"flat_out_{lane}_b"
    catalog[left] = TensorSpec((6, 4), _DTYPE)
    catalog[right_a] = TensorSpec((4, 5), _DTYPE)
    catalog[right_b] = TensorSpec((4, 5), _DTYPE)
    catalog[out_a] = TensorSpec((6, 5), _DTYPE)
    catalog[out_b] = TensorSpec((6, 5), _DTYPE)
    return DenseKernelBatch(
        action_indices=(lane * 2, lane * 2 + 1),
        left=(DenseBatchOperand(left, (0, 1)),),
        right=(DenseBatchOperand(right_a, (0, 1)), DenseBatchOperand(right_b, (0, 1))),
        left_free_rank=1,
        reduction_rank=1,
        right_free_rank=1,
        left_axis=0,
        right_axis=1,
        results=(
            DenseBatchResult(out_a, ((0, 6), (0, 5))),
            DenseBatchResult(out_b, ((0, 6), (5, 10))),
        ),
        seconds=0.1,
        packing_bytes=64 * _DTYPE.itemsize,
        output_bytes=60 * _DTYPE.itemsize,
        packing_dtype=_DTYPE.name,
    )


def _multiaxis_group(lanes=3):
    catalog = {}
    batches = tuple(_multiaxis_batch(lane, catalog) for lane in range(lanes))
    return DenseGemmGroup(batches, (6, 4, 10), 0.1 * lanes), catalog


def _stacked_batch(lane, catalog):
    left = f"stack_left_{lane}"
    right_a, right_b = f"stack_right_{lane}_a", f"stack_right_{lane}_b"
    out_a, out_b = f"stack_out_{lane}_a", f"stack_out_{lane}_b"
    catalog[left] = TensorSpec((4,), _DTYPE)
    catalog[right_a] = TensorSpec((4,), _DTYPE)
    catalog[right_b] = TensorSpec((4,), _DTYPE)
    catalog[out_a] = TensorSpec((), _DTYPE)
    catalog[out_b] = TensorSpec((), _DTYPE)
    return DenseKernelBatch(
        action_indices=(lane * 2, lane * 2 + 1),
        left=(DenseBatchOperand(left, (0,)),),
        right=(DenseBatchOperand(right_a, (0,)), DenseBatchOperand(right_b, (0,))),
        left_free_rank=1,
        reduction_rank=1,
        right_free_rank=1,
        left_axis=0,
        right_axis=1,
        results=(
            DenseBatchResult(out_a, ((0, 1), (0, 1)), (0, 1)),
            DenseBatchResult(out_b, ((0, 1), (1, 2)), (0, 1)),
        ),
        seconds=0.1,
        packing_bytes=12 * _DTYPE.itemsize,
        output_bytes=2 * _DTYPE.itemsize,
        packing_dtype=_DTYPE.name,
        stacked_sides=(True, True),
    )


def _stacked_group():
    catalog = {}
    batches = tuple(_stacked_batch(lane, catalog) for lane in range(2))
    return DenseGemmGroup(batches, (1, 4, 2), 0.2), catalog


class GroupResultViewBindingTests(unittest.TestCase):
    def test_multiaxis_views_reuse_fresh_output_cube(self):
        group, catalog = _multiaxis_group()
        batches, layouts, _, _ = _bind_layouts(group, catalog, np)
        self.assertEqual(layouts[0].output_shape, (2, 3, 5, 2))
        bound_views = _bind_group_result_views(layouts)
        self.assertIsNotNone(bound_views)

        first = np.arange(3 * 6 * 10, dtype=np.float64).astype(_DTYPE).reshape(3, 6, 10)
        first_results = bound_views(first)
        first_grouped = first.reshape(3, 2, 3, 5, 2)
        for lane, batch in enumerate(batches):
            left_view = first_results[batch.results[0].tensor]
            right_view = first_results[batch.results[1].tensor]
            np.testing.assert_array_equal(left_view, first_grouped[lane, :, :, :, :1])
            np.testing.assert_array_equal(right_view, first_grouped[lane, :, :, :, 1:])
            self.assertTrue(np.shares_memory(left_view, first))
            self.assertTrue(np.shares_memory(right_view, first))

        second = first + 1000
        second_results = bound_views(second)
        grouped_second = second.reshape(3, 2, 3, 5, 2)
        for lane, batch in enumerate(batches):
            second_view = second_results[batch.results[0].tensor]
            np.testing.assert_array_equal(second_view, grouped_second[lane, :, :, :, :1])
            self.assertTrue(np.shares_memory(second_view, second))
            self.assertFalse(np.shares_memory(second_view, first))

    def test_stacked_singleton_result_squeezes_to_scalar_view(self):
        group, catalog = _stacked_group()
        batches, layouts, _, _ = _bind_layouts(group, catalog, np)
        self.assertEqual(layouts[0].output_shape, (1, 2))
        bound_views = _bind_group_result_views(layouts)
        self.assertIsNotNone(bound_views)

        cube = np.arange(4, dtype=np.float64).astype(_DTYPE).reshape(2, 1, 2)
        results = bound_views(cube)
        for lane, batch in enumerate(batches):
            for result_index, result in enumerate(batch.results):
                view = results[result.tensor]
                self.assertEqual(view.shape, ())
                self.assertEqual(view, cube[lane, 0, result_index])
                self.assertTrue(np.shares_memory(view, cube))

    def test_heterogeneous_and_empty_layouts_keep_fallback(self):
        catalog = {}
        multi = _multiaxis_batch(0, catalog)
        flat = _flat_batch(1, catalog)
        group = DenseGemmGroup((multi, flat), (6, 4, 10), 0.2)
        batches, layouts, _, _ = _bind_layouts(group, catalog, np)
        self.assertEqual(tuple(layout.output_shape for layout in layouts), ((2, 3, 5, 2), (6, 10)))
        self.assertIsNone(_bind_group_result_views(layouts))

        rng = np.random.default_rng(918)
        arrays = {}
        expected = []
        for batch in batches:
            left_shape = catalog[batch.left[0].tensor].shape
            right_a_shape = catalog[batch.right[0].tensor].shape
            right_b_shape = catalog[batch.right[1].tensor].shape
            left = (rng.normal(size=left_shape) + 1j * rng.normal(size=left_shape)).astype(_DTYPE)
            right_a = (rng.normal(size=right_a_shape) + 1j * rng.normal(size=right_a_shape)).astype(_DTYPE)
            right_b = (rng.normal(size=right_b_shape) + 1j * rng.normal(size=right_b_shape)).astype(_DTYPE)
            arrays[batch.left[0].tensor] = left
            arrays[batch.right[0].tensor] = right_a
            arrays[batch.right[1].tensor] = right_b
            if len(left_shape) == 3:
                product = left.reshape(6, 4) @ np.concatenate((right_a, right_b), axis=2).reshape(4, 10)
                expected.append(product.reshape(2, 3, 5, 2))
            else:
                expected.append(left @ np.concatenate((right_a, right_b), axis=1))
        cube = bind_dense_gemm_group(group, catalog, np)(arrays)
        self.assertEqual(cube.shape, (2, 6, 10))
        for batch, reference in zip(batches, expected, strict=True):
            actual_a = arrays[batch.results[0].tensor]
            actual_b = arrays[batch.results[1].tensor]
            if reference.ndim == 4:
                np.testing.assert_allclose(actual_a, reference[:, :, :, :1], rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(actual_b, reference[:, :, :, 1:], rtol=2e-12, atol=2e-12)
            else:
                np.testing.assert_allclose(actual_a, reference[:, :5], rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(actual_b, reference[:, 5:], rtol=2e-12, atol=2e-12)

        empty_layout = SimpleNamespace(output_shape=(), result_slices=())
        self.assertIsNone(_bind_group_result_views((empty_layout, empty_layout)))

    def test_numpy_group_scalar_squeeze_remains_on_reference_path(self):
        group, catalog = _stacked_group()
        arrays = {}
        expected = []
        rng = np.random.default_rng(1908)
        for batch in group.batches:
            left = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(_DTYPE)
            right_a = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(_DTYPE)
            right_b = (rng.normal(size=4) + 1j * rng.normal(size=4)).astype(_DTYPE)
            arrays[batch.left[0].tensor] = left
            arrays[batch.right[0].tensor] = right_a
            arrays[batch.right[1].tensor] = right_b
            expected.append((left @ right_a, left @ right_b))
        cube = bind_dense_gemm_group(group, catalog, np)(arrays)
        self.assertEqual(cube.shape, (2, 1, 2))
        for lane, batch in enumerate(group.batches):
            for result_index, result in enumerate(batch.results):
                value = arrays[result.tensor]
                self.assertEqual(value.shape, ())
                self.assertAlmostEqual(value, expected[lane][result_index])
                self.assertTrue(np.shares_memory(value, cube))


if __name__ == "__main__":
    unittest.main()
