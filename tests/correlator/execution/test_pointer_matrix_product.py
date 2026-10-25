"""Fixed GEMM geometry against independent index sums and actual CUDA."""
import ctypes
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import (
    DenseBatchOperand, DenseBatchResult, DenseKernelBatch, TensorSpec,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.lowering import bind_matmul, lower_binary_matmul
from lqcdweave.contraction.execution.matrix_product import bind_pointer_gemm
from lqcdweave.contraction.execution.operations.dense_batch import bind_dense_batch
from correlator.execution.test_dense_atomic_kernel import _inputs, _loop_reference


class _Array:
    def __init__(self, host):
        self.host = host
        self.shape, self.strides, self.dtype = host.shape, host.strides, host.dtype
        self.flags, self.data = host.flags, SimpleNamespace(ptr=host.ctypes.data)
        self.nbytes = host.nbytes

    def reshape(self, shape):
        return _Array(self.host.reshape(shape))

    def __getitem__(self, index):
        return _Array(self.host[index])


def _read(pointer, count, dtype):
    primitive = ctypes.c_float if dtype.itemsize in (4, 8) and dtype != np.float64 else ctypes.c_double
    if dtype == np.complex128:
        primitive = ctypes.c_double
    length = count * (2 if dtype.kind == 'c' else 1)
    return np.ctypeslib.as_array((primitive * length).from_address(pointer)).view(dtype)


class _Backend:
    __name__ = 'cupy'

    def __init__(self):
        self.calls, self.allocations, self.wrappers = [], [], []
        self.device_calls, self.synchronizations = 0, 0
        self.pointer_mode, self.failure = 0, None
        self.compute_mode, self.math_mode = 0, 0
        self._core = SimpleNamespace(_routines_linalg=SimpleNamespace(
            COMPUTE_TYPE_DEFAULT=0, get_compute_type=lambda dtype: self.compute_mode))
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(is_hip=False), Device=self.device,
            get_current_stream=lambda: SimpleNamespace(synchronize=self.synchronize),
            cublas=SimpleNamespace(CUBLAS_OP_N=0, CUBLAS_OP_T=1,
                CUBLAS_POINTER_MODE_HOST=0, getPointerMode=lambda handle: self.pointer_mode,
                CUBLAS_DEFAULT_MATH=0, getMathMode=lambda handle: self.math_mode))
        for dtype, name in ((np.float32, 'sgemm'), (np.float64, 'dgemm'),
                            (np.complex64, 'cgemm'), (np.complex128, 'zgemm')):
            setattr(self.cuda.cublas, name, self.gemm(np.dtype(dtype)))

    def device(self):
        self.device_calls += 1
        return SimpleNamespace(cublas_handle=27)

    def synchronize(self):
        self.synchronizations += 1

    def gemm(self, dtype):
        def run(handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
            self.calls.append((ta, tb, m, n, k, lda, ldb, ldc, a, b, c))
            self.assert_geometry(ta, tb, m, n, k, lda, ldb, ldc)
            av = _read(a, lda * (m if ta else k), dtype).reshape((lda, -1), order='F')
            bv = _read(b, ldb * (k if tb else n), dtype).reshape((ldb, -1), order='F')
            av, bv = (av.T if ta else av), (bv.T if tb else bv)
            cv = _read(c, ldc * n, dtype).reshape((ldc, n), order='F')
            scalar = _read(alpha, 1, dtype)[0]
            if _read(beta, 1, dtype)[0] != 0:
                raise AssertionError('new output must not be read before initialization')
            cv[:m] = scalar * (av @ bv)
            if self.failure is not None:
                raise self.failure
        return run

    @staticmethod
    def assert_geometry(ta, tb, m, n, k, lda, ldb, ldc):
        if (lda != (k if ta else m) or ldb != (n if tb else k) or ldc != m):
            raise AssertionError('invalid column-major leading dimension')

    def empty(self, shape, dtype, order='C'):
        self.allocations.append((tuple(shape), np.dtype(dtype), order))
        return _Array(np.empty(shape, dtype=dtype, order=order))

    def reshape(self, value, shape):
        self.wrappers.append('reshape')
        return value.reshape(shape)

    def transpose(self, value, axes):
        self.wrappers.append('transpose')
        return _Array(value.host.transpose(axes))

    def dot(self, left, right):
        self.wrappers.append('dot')
        return _Array(np.dot(left.host, right.host))

    def matmul(self, left, right):
        self.wrappers.append('matmul')
        return _Array(np.matmul(left.host, right.host))

    def concatenate(self, values, axis, out):
        self.wrappers.append('concatenate')
        np.concatenate([value.host for value in values], axis=axis, out=out.host)
        return out


def _prepared_batch(dtype):
    shapes = ((2, 3, 4), (4, 2, 5))
    batch = DenseKernelBatch((0, 1), (DenseBatchOperand('a0', (0, 1, 2)),
                                      DenseBatchOperand('a1', (0, 1, 2))),
        (DenseBatchOperand('b', (0, 1, 2)),), 2, 1, 2, 0, 1,
        tuple(DenseBatchResult(f'c{i}', ((i, i + 1), (0, 3), (0, 2), (0, 5))) for i in range(2)),
        0.0, 24 * np.dtype(dtype).itemsize, 60 * np.dtype(dtype).itemsize, np.dtype(dtype).name,
        prepared_sides=(False, True))
    return batch, {'a0': TensorSpec((1, 3, 4), dtype), 'a1': TensorSpec((1, 3, 4), dtype),
                   'b': TensorSpec(shapes[1], dtype)}


class PointerMatrixCpuTests(unittest.TestCase):
    def test_arena_offsets_out_extent_aliases_and_failure_owners(self):
        backend = _Backend()
        a, b = _inputs(((2, 3), (3, 4)), np.complex128)
        arena = np.zeros(72, dtype='complex128')
        arena[:6], arena[32:44] = a.ravel(), b.ravel()
        left = right = _Array(arena)
        out = _Array(arena[64:72].reshape(2, 4))
        run = bind_pointer_gemm(backend, ((2, 3), (3, 4)), (1, 1), (None, None),
            'complex128', (2, 4), input_offsets=(0, 512))
        bound = run.bind_layout((((2, 3), np.dtype('complex128'), (48, 16)),
                                 ((3, 4), np.dtype('complex128'), (64, 16))))
        self.assertIs(bound(left, right, out=out), out)
        np.testing.assert_allclose(out.host, a @ b, rtol=1e-12, atol=1e-12)
        self.assertEqual(backend.allocations, [])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            bound(left, right, out=_Array(arena[:8].reshape(2, 4)))
        with self.assertRaisesRegex(ValueError, 'smaller'):
            bound(left, _Array(arena[:40]), out=out)
        for invalid in (np.empty((4, 2), 'complex128'), np.empty((2, 4), 'complex64'),
                        np.empty((2, 4), 'complex128', order='F')):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaisesRegex(ValueError, 'layout'):
                bound(left, right, out=_Array(invalid))
        for offsets in ((-16, 512), (0, 8), (0, 0)):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                bind_pointer_gemm(backend, ((2, 3), (3, 4)), (1, 1), (None, None),
                    'complex128', (2, 4), input_offsets=offsets)
        backend.math_mode = 1
        self.assertIsNone(bound(left, right, out=out))
        self.assertEqual(len(backend.calls), 1)
        backend.math_mode = 0
        refs = weakref.ref(left), weakref.ref(out)
        def fence():
            self.assertTrue(all(ref() is not None for ref in refs))
            backend.synchronizations += 1
        backend.cuda.get_current_stream = lambda: SimpleNamespace(synchronize=fence)
        error = backend.failure = RuntimeError('submitted arena GEMM failed')
        with self.assertRaisesRegex(RuntimeError, 'submitted arena'):
            bound(left, right, out=out)
        self.assertEqual(backend.synchronizations, 1)
        error.__traceback__ = None
        del left, right, out
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_all_dtypes_and_c_f_geometry_without_input_or_matrix_views(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            left, right = _inputs(((2, 3, 4), (4, 2, 5)), dtype)
            left_f = np.ascontiguousarray(left.reshape(6, 4).T).reshape(4, 2, 3).transpose(1, 2, 0)
            right_f = np.ascontiguousarray(right.reshape(4, 10).T).reshape(2, 5, 4).transpose(2, 0, 1)
            expected = _loop_reference('abk,kcd->abcd', (left, right))
            for a, b in ((left, right), (left_f, right), (left, right_f), (left_f, right_f)):
                with self.subTest(dtype=dtype, strides=(a.strides, b.strides)):
                    backend = _Backend()
                    recipe = lower_binary_matmul('abk,kcd->abcd', (a.shape, b.shape), (dtype,) * 2)
                    run = bind_matmul(recipe, backend)
                    self.assertEqual(backend.device_calls, 0)
                    value = run(_Array(a), _Array(b))
                    tolerance = 3e-5 if np.dtype(dtype).itemsize <= 8 and dtype != np.float64 else 1e-12
                    np.testing.assert_allclose(value.host, expected, rtol=tolerance, atol=tolerance)
                    self.assertEqual(backend.wrappers, [])
                    self.assertEqual(len(backend.calls), 1)
                    self.assertEqual(backend.allocations, [(expected.shape, np.dtype(dtype), 'C')])
                    self.assertEqual(backend.synchronizations, 0)
                    self.assertFalse(np.shares_memory(value.host, a))
                    self.assertFalse(np.shares_memory(value.host, b))

    def test_input_axis_permutations_output_order_and_scalar_results(self):
        for equation, shapes in (
            ('kab,cdk->abcd', ((4, 2, 3), (2, 5, 4))),
            ('abk,kcd->cdab', ((2, 3, 4), (4, 2, 5))),
            ('k,k->', ((4,), (4,))), ('ak,k->a', ((3, 4), (4,))),
            ('k,kb->b', ((4,), (4, 5)))):
            with self.subTest(equation=equation):
                inputs = _inputs(shapes, np.complex128)
                backend = _Backend()
                run = bind_matmul(lower_binary_matmul(equation, shapes, (np.complex128,) * 2), backend)
                value = run(*map(_Array, inputs))
                expected = _loop_reference(equation, inputs)
                np.testing.assert_allclose(value.host, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(len(backend.calls), 1)
                self.assertFalse(any(name in backend.wrappers for name in ('reshape', 'dot')))

    def test_negative_strides_and_debug_keep_array_based_fallback(self):
        recipe = lower_binary_matmul('abk,kcd->abcd', ((2, 3, 4), (4, 2, 5)), (np.complex128,) * 2)
        for debug, reverse in ((False, True), (True, False)):
            with self.subTest(debug=debug, reverse=reverse):
                inputs = _inputs(((2, 3, 4), (4, 2, 5)), np.complex128)
                if reverse:
                    inputs = inputs[0][..., ::-1], inputs[1][::-1]
                backend = _Backend()
                with execution_debug(debug):
                    value = bind_matmul(recipe, backend)(*map(_Array, inputs))
                np.testing.assert_allclose(value.host, _loop_reference('abk,kcd->abcd', inputs),
                                           rtol=1e-12, atol=1e-12)
                self.assertEqual(backend.calls, [])
                self.assertIn('dot', backend.wrappers)

    def test_current_inputs_not_retained_and_results_survive_later_calls(self):
        backend = _Backend()
        recipe = lower_binary_matmul('ij,jk->ik', ((3, 4), (4, 5)), (np.complex128,) * 2)
        run = bind_matmul(recipe, backend)
        results, references = [], []
        for scale in (1, 2):
            inputs = tuple(_Array(scale * value) for value in _inputs(((3, 4), (4, 5)), np.complex128))
            references.extend(weakref.ref(value) for value in inputs)
            results.append(run(*inputs))
        del inputs
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        np.testing.assert_allclose(results[1].host, 4 * results[0].host)
        self.assertNotEqual(backend.calls[0][-1], backend.calls[1][-1])

    def test_nondefault_precision_modes_keep_public_dot_after_binding(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            backend = _Backend()
            recipe = lower_binary_matmul('ij,jk->ik', ((3, 4), (4, 5)), (dtype,) * 2)
            run = bind_matmul(recipe, backend)
            inputs = _inputs(((3, 4), (4, 5)), dtype)
            for compute, math in ((1, 0), (2, 0), (0, 1)):
                with self.subTest(dtype=dtype, compute=compute, math=math):
                    backend.compute_mode, backend.math_mode = compute, math
                    value = run(*map(_Array, inputs))
                    np.testing.assert_allclose(value.host, _loop_reference('ij,jk->ik', inputs),
                                               rtol=3e-5, atol=3e-5)
                    self.assertEqual(backend.calls, [])
                    self.assertIn('dot', backend.wrappers)
            backend.compute_mode = backend.math_mode = 0
            run(*map(_Array, inputs))
            self.assertEqual(len(backend.calls), 1)

    def test_prepared_cartesian_batch_uses_grouped_output_directly(self):
        batch, catalog = _prepared_batch(np.complex128)
        host = _inputs(((2, 3, 4), (4, 2, 5)), np.complex128)
        backend = _Backend()
        arrays = {'a0': _Array(host[0][:1].copy()), 'a1': _Array(host[0][1:].copy()),
                  'b': _Array(host[1])}
        value = bind_dense_batch(batch, catalog, backend)(arrays)
        expected = _loop_reference('abk,kcd->abcd', host)
        for i in range(2):
            np.testing.assert_allclose(arrays[f'c{i}'].host, expected[i:i+1], rtol=1e-12, atol=1e-12)
            self.assertTrue(np.shares_memory(arrays[f'c{i}'].host, value.host))
        self.assertEqual(backend.wrappers, ['concatenate'])
        self.assertEqual(len(backend.allocations), 2)

    def test_pointer_mode_and_kernel_failure_preserve_first_error_and_cleanup(self):
        recipe = lower_binary_matmul('ij,jk->ik', ((3, 4), (4, 5)), (np.complex128,) * 2)
        for failure in ('pointer_mode', 'kernel'):
            backend = _Backend()
            sentinel = RuntimeError('submitted GEMM failed')
            if failure == 'pointer_mode':
                backend.pointer_mode = 1
            else:
                backend.failure = sentinel
            with self.assertRaises(RuntimeError) as raised:
                bind_matmul(recipe, backend)(*map(_Array, _inputs(((3, 4), (4, 5)), np.complex128)))
            self.assertEqual(backend.synchronizations, int(failure == 'kernel'))
            self.assertEqual(len(backend.allocations), int(failure == 'kernel'))
            if failure == 'kernel':
                self.assertIs(raised.exception, sentinel)

    def test_missing_cuda_hip_and_shared_batch_use_existing_dispatch(self):
        for backend in (np, SimpleNamespace(__name__='cupy'), _Backend()):
            if isinstance(backend, _Backend):
                backend.cuda.runtime.is_hip = True
            self.assertIsNone(bind_pointer_gemm(backend, ((3, 4), (4, 5)), (1, 1),
                                               (None, None), np.float64, (3, 5)))
        backend = _Backend()
        shapes = ((2, 3, 4), (2, 4, 5))
        inputs = _inputs(shapes, np.complex128)
        value = bind_matmul(lower_binary_matmul('bij,bjk->bik', shapes, (np.complex128,) * 2), backend)(
            *map(_Array, inputs))
        np.testing.assert_allclose(value.host, _loop_reference('bij,bjk->bik', inputs), rtol=1e-12, atol=1e-12)
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.wrappers, ['matmul'])


class PointerMatrixCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))
        cls.backend = cupy

    def test_cuda_fixed_geometry_current_stream_fresh_owners_and_fallbacks(self):
        cp = self.backend
        cases = (
            ('abk,kcd->abcd', ((2, 3, 4), (4, 2, 5))),
            ('kab,cdk->abcd', ((4, 2, 3), (2, 5, 4))),
            ('abk,kcd->cdab', ((2, 3, 4), (4, 2, 5))),
            ('k,k->', ((4,), (4,))),
            ('bij,bjk->bik', ((2, 3, 4), (2, 4, 5))))
        stream = cp.cuda.Stream(non_blocking=True)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for equation, shapes in cases:
                with self.subTest(dtype=dtype, equation=equation):
                    host = _inputs(shapes, dtype)
                    run = bind_matmul(lower_binary_matmul(equation, shapes, (dtype,) * 2), cp)
                    tolerance = 5e-5 if dtype in (np.float32, np.complex64) else 2e-12
                    values = []
                    with stream:
                        for scale in (1, 2):
                            arrays = tuple(cp.asarray(scale * a) for a in host)
                            values.append(run(*arrays))
                            arrays[0].fill(0)
                    stream.synchronize()
                    for value, scale in zip(values, (1, 2)):
                        np.testing.assert_allclose(cp.asnumpy(value),
                            scale ** 2 * _loop_reference(equation, host), rtol=tolerance, atol=tolerance)
                    if equation == 'abk,kcd->abcd':
                        with stream:
                            left, right = (cp.asarray(a) for a in host)
                            left = cp.ascontiguousarray(left.reshape(6, 4).T).reshape(4, 2, 3).transpose(1, 2, 0)
                            right = cp.ascontiguousarray(right.reshape(4, 10).T).reshape(2, 5, 4).transpose(2, 0, 1)
                            with patch.object(cp, 'reshape', side_effect=AssertionError('matrix header')):
                                product = run(left, right)
                            reverse = run(left[..., ::-1], right[::-1])
                            with execution_debug(True):
                                debug = run(left, right)
                        stream.synchronize()
                        expected = _loop_reference(equation, host)
                        for value in (product, reverse, debug):
                            np.testing.assert_allclose(cp.asnumpy(value), expected, rtol=tolerance, atol=tolerance)

    def test_cuda_prepared_batch_uses_saved_result_layout_and_preserves_order(self):
        cp = self.backend
        batch, catalog = _prepared_batch(np.complex128)
        host = _inputs(((2, 3, 4), (4, 2, 5)), np.complex128)
        arrays = {'a0': cp.asarray(host[0][:1]), 'a1': cp.asarray(host[0][1:]), 'b': cp.asarray(host[1])}
        run = bind_dense_batch(batch, catalog, cp)
        with patch.object(cp, 'dot', side_effect=AssertionError('matrix dispatch wrapper')):
            output = run(arrays)
        cp.cuda.get_current_stream().synchronize()
        expected = _loop_reference('abk,kcd->abcd', host)
        for i in range(2):
            np.testing.assert_allclose(cp.asnumpy(arrays[f'c{i}']), expected[i:i+1], rtol=2e-12, atol=2e-12)
        self.assertEqual(output.shape, expected.shape)
        self.assertEqual(output.nbytes, batch.output_bytes)
        self.assertEqual(arrays['c0'].data.ptr, output.data.ptr)

    def test_cuda_nondefault_compute_and_math_modes_preserve_public_policy(self):
        cp = self.backend
        linalg, cublas = cp._core._routines_linalg, cp.cuda.cublas
        handle = cp.cuda.Device().cublas_handle
        saved_math = cublas.getMathMode(handle)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                saved_compute = linalg.get_compute_type(dtype)
                host = _inputs(((3, 4), (4, 5)), dtype)
                arrays = tuple(cp.asarray(value) for value in host)
                run = bind_matmul(lower_binary_matmul('ij,jk->ik',
                                  ((3, 4), (4, 5)), (dtype,) * 2), cp)
                try:
                    for compute, math in (
                            (linalg.COMPUTE_TYPE_PEDANTIC, cublas.CUBLAS_DEFAULT_MATH),
                            (linalg.COMPUTE_TYPE_DEFAULT, cublas.CUBLAS_TENSOR_OP_MATH)):
                        linalg.set_compute_type(dtype, compute)
                        cublas.setMathMode(handle, math)
                        with patch.object(cp, 'dot', wraps=cp.dot) as dot:
                            value = run(*arrays)
                        self.assertEqual(dot.call_count, 1)
                        tolerance = 5e-5 if dtype in (np.float32, np.complex64) else 2e-12
                        np.testing.assert_allclose(cp.asnumpy(value),
                            _loop_reference('ij,jk->ik', host), rtol=tolerance, atol=tolerance)
                finally:
                    linalg.set_compute_type(dtype, saved_compute)
                    cublas.setMathMode(handle, saved_math)
