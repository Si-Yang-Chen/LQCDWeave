"""Prepared layout reuse: operation counts, fresh inputs and independent sums.

NumPy-backed device protocol tests are not a CUDA performance measurement.
"""
import gc
import importlib
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, DenseBatchOperand, DenseBatchResult, DenseKernelBatch, TensorSpec,
    parse_axis_pattern,
)
from lqcdweave.contraction.execution.lowering import bind_matmul, lower_binary_matmul
from lqcdweave.contraction.execution.operations.contract import bind_dense_action
from lqcdweave.contraction.execution.operations.dense_batch import (
    bind_dense_batch, contract_dense_batch, _BatchSide, _contiguous_concatenation,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug
from correlator.execution.test_dense_atomic_kernel import _inputs, _loop_reference, CASES, FALLBACK_CASES


def _backend(**overrides):
    return SimpleNamespace(**dict(__name__='cupy', transpose=np.transpose, reshape=np.reshape,
                           matmul=np.matmul, einsum=np.einsum, **{
                               'array': np.array, 'empty': np.empty,
                               'concatenate': np.concatenate}) | overrides)


def _batch(arrays, *, transpose=False, dtype='complex128'):
    permutation = (1, 0) if transpose else (0, 1)
    left = tuple(DenseBatchOperand(key, permutation) for key in ('a0', 'a1') if key in arrays)
    right = tuple(DenseBatchOperand(key, permutation) for key in ('b0', 'b1') if key in arrays)
    shapes = {key: value.transpose(permutation).shape for key, value in arrays.items()}
    results = []
    a = 0
    for i, l in enumerate(left):
        b = 0
        for j, r in enumerate(right):
            m, n = shapes[l.tensor][0], shapes[r.tensor][1]
            results.append(DenseBatchResult(f'c{i}{j}', ((a, a + m), (b, b + n))))
            b += n
        a += m
    batch = DenseKernelBatch(tuple(range(len(results))), left, right, 1, 1, 1, 0, 1,
                             tuple(results), 0.0, sum(v.size for v in arrays.values()) * np.dtype(dtype).itemsize,
                             a * b * np.dtype(dtype).itemsize, dtype)
    catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
    return batch, catalog


def _shared_inputs():
    left, right = _inputs(((5, 4), (4, 7)), np.complex128)
    return {'a0': left[:2], 'a1': left[2:], 'b0': right[:, :3], 'b1': right[:, 3:]}


def _stacked_permuted_batch(arrays):
    permutation = (1, 0)
    left = tuple(DenseBatchOperand(key, permutation) for key in ('a0', 'a1'))
    right = tuple(DenseBatchOperand(key, permutation) for key in ('b0', 'b1'))
    results = tuple(DenseBatchResult(
        f'c{i}{j}', ((i, i + 1), (j, j + 1)), (0, 1))
        for i in range(2) for j in range(2))
    batch = DenseKernelBatch(
        (0, 1, 2, 3), left, right, 1, 2, 1, 0, 2, results, 0.0,
        24 * np.dtype('complex128').itemsize, 4 * np.dtype('complex128').itemsize,
        'complex128', stacked_sides=(True, True))
    catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
    return batch, catalog


def _stacked_permuted_values(scale=1):
    return {
        'a0': (scale * np.arange(6, dtype=np.float32).reshape(3, 2)).astype('complex128')[:, ::-1],
        'a1': (scale * np.arange(6, 12, dtype=np.float64).reshape(3, 2)).astype('complex128'),
        'b0': (scale * np.arange(12, 18, dtype=np.float32).reshape(2, 3)).astype('complex128')[:, ::-1],
        'b1': (scale * np.arange(18, 24, dtype=np.float64).reshape(2, 3)).astype('complex128'),
    }


def _stacked_permuted_reference(arrays):
    left = np.concatenate([arrays[key].T[None, ...] for key in ('a0', 'a1')], axis=0).reshape(2, -1)
    right = np.concatenate([arrays[key].T[..., None] for key in ('b0', 'b1')], axis=2).reshape(-1, 2)
    return left @ right


class PreparedLayoutTests(unittest.TestCase):
    def test_cupy_2d_binding_skips_gufunc_and_preserves_batch_dispatch(self):
        for equation, shapes, batched in (
                ('ij,jk->ik', ((3, 4), (4, 5)), False),
                ('abc,cd->abd', ((2, 3, 4), (4, 5)), False),
                ('ij,ij->', ((3, 4), (3, 4)), False),
                ('bij,bjk->bik', ((2, 3, 4), (2, 4, 5)), True)):
            with self.subTest(equation=equation):
                arrays = _inputs(shapes, np.complex128)
                recipe = lower_binary_matmul(equation, shapes, (np.complex128,) * 2)
                dot, multiply = unittest.mock.Mock(wraps=np.dot), unittest.mock.Mock(wraps=np.matmul)
                backend = _backend(dot=dot, matmul=multiply)
                run = bind_matmul(recipe, backend)
                value = run(*arrays)
                np.testing.assert_allclose(value, _loop_reference(equation, arrays), rtol=1e-12, atol=1e-12)
                self.assertEqual(dot.call_count, int(not batched))
                self.assertEqual(multiply.call_count, int(batched))
                fresh = tuple(array + 0.125 for array in arrays)
                np.testing.assert_allclose(run(*fresh), _loop_reference(equation, fresh), rtol=1e-12, atol=1e-12)

    def test_saved_matrix_batch_uses_public_cupy_2d_binding(self):
        arrays = _shared_inputs()
        batch, catalog = _batch(arrays)
        dot = unittest.mock.Mock(wraps=np.dot)
        backend = _backend(dot=dot, matmul=lambda *args: (_ for _ in ()).throw(
            AssertionError('generic gufunc dispatch')))
        bind_dense_batch(batch, catalog, backend)(arrays)
        self.assertEqual(dot.call_count, 1)
        self.assert_products(arrays, batch)

    def assert_products(self, arrays, batch, *, transpose=False):
        for i, left in enumerate(batch.left):
            for j, right in enumerate(batch.right):
                l, r = arrays[left.tensor], arrays[right.tensor]
                if transpose:
                    l, r = l.T, r.T
                expected = _loop_reference('ij,jk->ik', (l, r))
                np.testing.assert_allclose(arrays[f'c{i}{j}'], expected, rtol=1e-12, atol=1e-12)

    def test_structural_cache_separates_equations_shapes_and_dtypes(self):
        recipe = lower_binary_matmul('ij,jk->ik', ((3, 4), (4, 2)), (np.complex128,) * 2)
        self.assertIs(recipe, lower_binary_matmul('ij,jk->ik', ((3, 4), (4, 2)), (np.complex128,) * 2))
        for equation, shapes, dtype in (
                ('ij,jk->ki', ((3, 4), (4, 2)), np.complex128),
                ('ij,jk->ik', ((3, 4), (4, 5)), np.complex128),
                ('ij,jk->ik', ((3, 4), (4, 2)), np.complex64)):
            other = lower_binary_matmul(equation, shapes, (dtype,) * 2)
            self.assertIsNot(recipe, other)
            inputs = _inputs(shapes, dtype)
            actual = bind_matmul(other, np)(*inputs)
            tolerance = 3e-5 if dtype == np.complex64 else 1e-12
            np.testing.assert_allclose(actual, _loop_reference(equation, inputs),
                                       rtol=tolerance, atol=tolerance)

    def test_bound_matmul_skips_identity_transpose_and_reshape_calls(self):
        left, right = _inputs(((3, 4), (4, 2)), np.complex128)
        recipe = lower_binary_matmul('ij,jk->ik', (left.shape, right.shape), (left.dtype, right.dtype))
        run = bind_matmul(recipe, np)
        with (patch.object(np, 'transpose', side_effect=AssertionError('identity transpose')),
              patch.object(np, 'reshape', side_effect=AssertionError('identity reshape')),
              patch.object(np, 'matmul', wraps=np.matmul) as matmul):
            actual = run(left, right)
        self.assertEqual(matmul.call_count, 1)
        np.testing.assert_allclose(actual, _loop_reference('ij,jk->ik', (left, right)), rtol=1e-12, atol=1e-12)

    def test_bound_dense_reference_layout_normalizes_only_at_bind(self):
        equation = 'ii,ij->j'
        operands = _inputs(((3, 3), (3, 4)), np.complex128)
        action = Action('contract', 'c', ('a', 'b'), parse_axis_pattern(equation))
        catalog = {k: TensorSpec(v.shape, v.dtype) for k, v in zip(action.operands, operands)}
        run = bind_dense_action(action, catalog, np)
        with patch('lqcdweave.contracts.correlator.indices.AxisPattern.normalized',
                   side_effect=AssertionError('repeated argument layout normalization')):
            for scale in (1, 3):
                fresh = tuple(scale * v for v in operands)
                actual, independent = run(fresh)
                self.assertTrue(independent)
                self.assertTrue(actual.flags.owndata)
                self.assertFalse(any(np.shares_memory(actual,value) for value in fresh))
                np.testing.assert_allclose(actual, _loop_reference(equation, fresh), rtol=1e-12, atol=1e-12)

    def test_bound_dense_axes_dtypes_and_negative_strides_against_index_sum(self):
        for equation, shapes in CASES + FALLBACK_CASES:
            for dtype in (np.complex64, np.complex128):
                with self.subTest(equation=equation, dtype=dtype):
                    operands = tuple(v[..., ::-1] for v in _inputs(shapes, dtype))
                    action = Action('contract', 'c', ('a', 'b'), parse_axis_pattern(equation))
                    catalog = {k: TensorSpec(v.shape, v.dtype) for k, v in zip(action.operands, operands)}
                    run = bind_dense_action(action, catalog, _backend())
                    with patch('lqcdweave.contraction.execution.operations.contract.lower_binary_matmul',
                               side_effect=AssertionError('repeated lowering')):
                        actual, _ = run(operands)
                    tolerance = 3e-5 if dtype == np.complex64 else 1e-12
                    np.testing.assert_allclose(actual, _loop_reference(equation, operands),
                                               rtol=tolerance, atol=tolerance)
                    self.assertEqual(actual.dtype, dtype)

    def test_shared_result_slabs_reuse_both_packed_sides_without_copy_launches(self):
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                arrays = _shared_inputs()
                if transpose:
                    arrays = {k: v.T for k, v in arrays.items()}
                batch, catalog = _batch(arrays, transpose=transpose)
                backend = _backend(
                    concatenate=lambda *a, **k: self.fail('unnecessary concatenate'),
                    empty=lambda *a, **k: self.fail('unnecessary packing allocation'),
                    array=lambda *a, **k: self.fail('unnecessary copy'))
                run = bind_dense_batch(batch, catalog, backend)
                with patch.object(backend, 'matmul', wraps=np.matmul) as matmul:
                    output = run(arrays)
                self.assertEqual(matmul.call_count, 1)
                self.assert_products(arrays, batch, transpose=transpose)
                self.assertEqual(output.nbytes, batch.output_bytes)
                for result in batch.results:
                    self.assertTrue(np.shares_memory(output, arrays[result.tensor]))
                    for key in ('a0', 'a1', 'b0', 'b1'):
                        self.assertFalse(np.shares_memory(arrays[result.tensor], arrays[key]))

    def test_single_contiguous_side_does_not_launch_snapshot_copy(self):
        arrays = {k: v.copy() for k, v in _shared_inputs().items() if k != 'a1'}
        batch, catalog = _batch(arrays)
        backend = _backend(array=lambda *a, **k: self.fail('unnecessary single input copy'))
        with patch.object(backend, 'concatenate', wraps=np.concatenate) as concatenate:
            bind_dense_batch(batch, catalog, backend)(arrays)
        self.assertEqual(concatenate.call_count, 1)
        self.assert_products(arrays, batch)

    def test_shared_storage_guards_reject_gaps_overlap_reverse_dtype_and_new_owners(self):
        base = np.arange(36, dtype=np.float64).reshape(6, 6)
        cases = (
            ((base[:2], base[3:5]), 0, (4, 6), None),  # gap
            ((base[:2], base[1:3]), 0, (4, 6), None),  # overlap
            ((base[2:4], base[:2]), 0, (4, 6), None),  # reverse order
            ((base[:2], base[2:4].copy()), 0, (4, 6), None),
            ((base[:2], base[2:4]), 0, (4, 6), np.dtype('complex128')),
            ((base[:2, ::-1], base[2:4, ::-1]), 0, (4, 6), None),
            ((base[:2, ::2], base[2:4, ::2]), 0, (4, 3), None),
        )
        for values, axis, shape, dtype in cases:
            with self.subTest(axis=axis, shape=shape, dtype=dtype):
                self.assertIsNone(_contiguous_concatenation(values, axis, shape, dtype))

    def test_independent_owners_reject_before_any_layout_or_pointer_reads(self):
        class OwnerOnly(np.ndarray):
            @property
            def shape(self):
                raise AssertionError('independent owners need no shape proof')

            @property
            def strides(self):
                raise AssertionError('independent owners need no stride proof')

            @property
            def __array_interface__(self):
                raise AssertionError('independent owners need no pointer proof')

        arrays = tuple(np.ones((2, 3)).view(OwnerOnly) for _ in range(2))
        for debug in (False, True):
            with self.subTest(debug=debug), execution_debug(debug):
                self.assertIsNone(_contiguous_concatenation(
                    arrays, 1, (2, 6), np.dtype('float64'),
                    input_layouts=((0, 1), (0, 1))))

    def test_unrelated_and_reversed_inputs_keep_bounded_pack_and_independent_values(self):
        for negative in (False, True):
            arrays = {k: v.copy() for k, v in _shared_inputs().items()}
            if negative:
                arrays = {k: v[:, ::-1] for k, v in arrays.items()}
            batch, catalog = _batch(arrays)
            packed = []
            def allocate(*args, **kwargs):
                value = np.empty(*args, **kwargs)
                packed.append(value)
                return value
            backend = _backend(empty=allocate)
            bind_dense_batch(batch, catalog, backend)(arrays)
            self.assertEqual(sum(v.nbytes for v in packed), batch.packing_bytes)
            self.assert_products(arrays, batch)

    def test_uniform_inputs_borrow_shared_slabs_and_keep_dtype(self):
        arrays = _shared_inputs()
        batch, catalog = _batch(arrays)
        backend = _backend()
        with patch.object(backend, 'concatenate', wraps=np.concatenate) as concatenate:
            run = bind_dense_batch(batch, catalog, backend)
            output = run(arrays)
        self.assertEqual(concatenate.call_count, 0)
        self.assertEqual(output.dtype, np.complex128)
        for i in range(2):
            for j in range(2):
                expected = _loop_reference('ij,jk->ik', (arrays[f'a{i}'], arrays[f'b{j}'].astype('complex128')))
                np.testing.assert_allclose(arrays[f'c{i}{j}'], expected, rtol=1e-12, atol=1e-12)

    def test_bound_batch_fresh_inputs_failure_retry_and_no_retained_owners(self):
        arrays = _shared_inputs()
        batch, catalog = _batch(arrays)
        backend = _backend()
        run = bind_dense_batch(batch, catalog, backend)
        first = run(arrays)
        saved = first.copy()
        references = tuple(weakref.ref(v.base) for k, v in arrays.items() if k[0] in 'ab')
        arrays.clear()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        fresh = {k: v * 3 for k, v in _shared_inputs().items()}
        with patch.object(backend, 'matmul', side_effect=RuntimeError('kernel failed')):
            with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                run(fresh)
        self.assertEqual(set(fresh), {'a0', 'a1', 'b0', 'b1'})
        with patch('lqcdweave.contraction.execution.operations.dense_batch.bind_dense_batch',
                   side_effect=AssertionError('repeated batch layout binding')):
            second = contract_dense_batch(batch, fresh, backend=backend, bound_kernel=run)
        self.assertFalse(np.shares_memory(first, second))
        np.testing.assert_array_equal(first, saved)
        self.assert_products(fresh, batch)

    def test_shared_subspan_and_singleton_axes_are_checked_without_extending_lifetime(self):
        root = np.arange(42, dtype=np.float64).reshape(7, 6)
        views = (root[1:3].reshape(1, 2, 6), root[3:5].reshape(1, 2, 6))
        actual = _contiguous_concatenation(views, 1, (1, 4, 6), None)
        self.assertTrue(np.shares_memory(actual, root))
        np.testing.assert_array_equal(actual, np.concatenate(views, axis=1))

    def test_mapped_permutation_and_stack_axes_keep_complete_owner_proof(self):
        prepend_root = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        prepend_values = (prepend_root[0], prepend_root[1])
        prepend_layout = ((None, 0, 1), (None, 0, 1))
        prepend = _contiguous_concatenation(
            prepend_values, 0, (2, 3, 4), None, input_layouts=prepend_layout)
        self.assertTrue(np.shares_memory(prepend, prepend_root))
        np.testing.assert_array_equal(prepend, prepend_root)

        append_root = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
        append_values = (append_root[:, :, 0], append_root[:, :, 1])
        append_layout = ((0, 1, None), (0, 1, None))
        append = _contiguous_concatenation(
            append_values, 2, (3, 4, 2), None, input_layouts=append_layout)
        self.assertTrue(np.shares_memory(append, append_root))
        np.testing.assert_array_equal(append, append_root)

        permuted_prepend_root = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        permuted_prepend_values = (permuted_prepend_root[0].T,
                                   permuted_prepend_root[1].T)
        permuted_prepend_layout = ((None, 1, 0), (None, 1, 0))
        permuted_prepend = _contiguous_concatenation(
            permuted_prepend_values, 0, (2, 3, 4), None,
            input_layouts=permuted_prepend_layout)
        self.assertTrue(np.shares_memory(permuted_prepend, permuted_prepend_root))
        np.testing.assert_array_equal(permuted_prepend, permuted_prepend_root)

        permuted_append_root = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
        permuted_append_values = (permuted_append_root[:, :, 0].T,
                                  permuted_append_root[:, :, 1].T)
        permuted_append_layout = ((1, 0, None), (1, 0, None))
        permuted_append = _contiguous_concatenation(
            permuted_append_values, 2, (3, 4, 2), None,
            input_layouts=permuted_append_layout)
        self.assertTrue(np.shares_memory(permuted_append, permuted_append_root))
        np.testing.assert_array_equal(permuted_append, permuted_append_root)

        gap_root = np.arange(36, dtype=np.float64).reshape(3, 3, 4)
        cases = (
            ((gap_root[0], gap_root[2]), (2, 3, 4), None),  # gap
            ((prepend_root[1], prepend_root[0]), (2, 3, 4), None),  # reverse order
            ((prepend_root[0], prepend_root[0]), (2, 3, 4), None),  # overlap
            ((prepend_root[0].copy(), prepend_root[1].copy()), (2, 3, 4), None),
            ((prepend_root[0, :, ::-1], prepend_root[1, :, ::-1]), (2, 3, 4), None),
            ((prepend_root[0], prepend_root[1]), (2, 3, 4), np.dtype('complex128')),
        )
        for values, shape, dtype in cases:
            with self.subTest(shape=shape, dtype=dtype):
                self.assertIsNone(_contiguous_concatenation(
                    values, 0, shape, dtype, input_layouts=prepend_layout))

    def test_single_mapped_f_owner_transpose_borrows_logical_view(self):
        root = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
        source = root
        packer, original_packer = Mock(), Mock()
        backend = _backend(empty=Mock())
        side = _BatchSide(
            operands=(('x', (1, 0)),), shape=(1, 3, 2), axis=0, prepared=False,
            packer=packer, original_packer=original_packer, stacked=True, prepend=True,
            input_layouts=((None, 1, 0),))

        actual = side.pack({'x': source}, backend, np.dtype('float64'))
        self.assertTrue(root.flags.f_contiguous)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertTrue(np.shares_memory(actual, root))
        np.testing.assert_array_equal(actual, root.T[None, ...])
        backend.empty.assert_not_called()
        packer.assert_not_called()
        original_packer.assert_not_called()

        negative = source[:, ::-1]
        self.assertIsNone(_contiguous_concatenation(
            (negative,), 0, (1, 3, 2), np.dtype('float64'),
            input_layouts=((None, 1, 0),)))
        self.assertIsNone(_contiguous_concatenation(
            (source,), 0, (1, 2, 3), np.dtype('float64'),
            input_layouts=((None, 1, 0),)))

    def test_raw_batch_packer_reads_original_permuted_and_stacked_inputs(self):
        compiled, launches = [], []

        def factory(source, name, **kwargs):
            side = len(compiled)

            def kernel(grid, block, arguments):
                inputs = arguments[:grid[1]] if len(grid) == 2 else arguments[:-1]
                destination = arguments[-1]
                launches.append((side, inputs))
                if side == 0:
                    logical = [value.T[None, ...] if value.ndim == 2 else value
                               for value in inputs]
                    destination[...] = np.concatenate(logical, axis=0).astype(destination.dtype)
                else:
                    logical = [value.T[..., None] if value.ndim == 2 else value
                               for value in inputs]
                    destination[...] = np.concatenate(logical, axis=2).astype(destination.dtype)

            compiled.append(source)
            return kernel

        backend = _backend(RawKernel=factory, ElementwiseKernel=Mock(), dot=np.dot)
        first = _stacked_permuted_values()
        batch, catalog = _stacked_permuted_batch(first)
        run = bind_dense_batch(batch, catalog, backend)
        for point, scale in enumerate((1, 3)):
            arrays = first if point == 0 else _stacked_permuted_values(scale)
            output = run(arrays)
            expected = _stacked_permuted_reference(arrays)
            np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(output.dtype, np.dtype('complex128'))
            for i in range(2):
                for j in range(2):
                    np.testing.assert_allclose(arrays[f'c{i}{j}'], expected[i, j],
                                               rtol=1e-12, atol=1e-12)
            left_launch, right_launch = launches[point * 2:point * 2 + 2]
            self.assertEqual(left_launch[0], 0)
            self.assertEqual(right_launch[0], 1)
            self.assertIs(left_launch[1][0], arrays['a0'])
            self.assertIs(left_launch[1][1], arrays['a1'])
            self.assertIs(right_launch[1][0], arrays['b0'])
            self.assertIs(right_launch[1][1], arrays['b1'])
        self.assertEqual(len(compiled), 2)

    def test_debug_batch_packer_keeps_the_logical_view_fallback(self):
        compiled, launches = [], []

        def factory(source, name, **kwargs):
            side = len(compiled)

            def kernel(grid, block, arguments):
                inputs = arguments[:grid[1]] if len(grid) == 2 else arguments[:-1]
                destination = arguments[-1]
                launches.append((side, inputs))
                if side == 0:
                    logical = [value.T[None, ...] if value.ndim == 2 else value
                               for value in inputs]
                    destination[...] = np.concatenate(logical, axis=0).astype(destination.dtype)
                else:
                    logical = [value.T[..., None] if value.ndim == 2 else value
                               for value in inputs]
                    destination[...] = np.concatenate(logical, axis=2).astype(destination.dtype)

            compiled.append(source)
            return kernel

        backend = _backend(RawKernel=factory, ElementwiseKernel=Mock(), dot=np.dot)
        arrays = _stacked_permuted_values()
        batch, catalog = _stacked_permuted_batch(arrays)
        run = bind_dense_batch(batch, catalog, backend)
        with execution_debug(True):
            output = run(arrays)
        expected = _stacked_permuted_reference(arrays)
        np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
        left_launch, right_launch = launches
        self.assertEqual(left_launch[0], 0)
        self.assertEqual(right_launch[0], 1)
        self.assertTrue(all(value.ndim == 3 for value in left_launch[1]))
        self.assertTrue(all(value.ndim == 3 for value in right_launch[1]))


class PreparedLayoutCupyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f'CUDA unavailable: {exc}') from exc

    def test_real_device_shared_slabs_and_fresh_inputs_skip_packing(self):
        cp = self.cp
        host_left, host_right = _inputs(((5, 4), (4, 7)), np.complex128)
        host = {'a0': host_left[:2], 'a1': host_left[2:],
                'b0': host_right[:, :3], 'b1': host_right[:, 3:]}
        batch, catalog = _batch(host)
        run = bind_dense_batch(batch, catalog, cp)
        outputs = []
        for scale in (1, 2):
            left, right = cp.asarray(scale * host_left), cp.asarray(host_right)
            arrays = {'a0': left[:2], 'a1': left[2:], 'b0': right[:, :3], 'b1': right[:, 3:]}
            with (patch.object(cp, 'concatenate', side_effect=AssertionError('unnecessary concatenate')),
                  patch.object(cp, 'array', side_effect=AssertionError('unnecessary snapshot'))):
                outputs.append(run(arrays))
            for i in range(2):
                for j in range(2):
                    expected = _loop_reference('ij,jk->ik', (scale * host[f'a{i}'], host[f'b{j}']))
                    np.testing.assert_allclose(cp.asnumpy(arrays[f'c{i}{j}']), expected,
                                               rtol=1e-12, atol=1e-12)
            left[...] = 0
            right[...] = 0
        np.testing.assert_allclose(cp.asnumpy(outputs[0]),
                                   _loop_reference('ij,jk->ik', (host_left, host_right)),
                                   rtol=1e-12, atol=1e-12)
        self.assertNotEqual(outputs[0].data.ptr, outputs[1].data.ptr)

    def test_real_device_negative_strides_uniform_precision_with_packing(self):
        cp = self.cp
        host = {key: value[:, ::-1].astype('complex128')
                for key, value in _shared_inputs().items()}
        arrays = {key: cp.asarray(value[:, ::-1])[:, ::-1] for key, value in host.items()}
        batch, catalog = _batch(host)
        with patch.object(cp, 'concatenate', wraps=cp.concatenate) as concatenate:
            output = bind_dense_batch(batch, catalog, cp)(arrays)
        self.assertEqual(concatenate.call_count, 0)
        self.assertEqual(output.dtype, np.complex128)
        for i in range(2):
            for j in range(2):
                expected = _loop_reference('ij,jk->ik', (host[f'a{i}'], host[f'b{j}'].astype('complex128')))
                np.testing.assert_allclose(cp.asnumpy(arrays[f'c{i}{j}']), expected,
                                           rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
