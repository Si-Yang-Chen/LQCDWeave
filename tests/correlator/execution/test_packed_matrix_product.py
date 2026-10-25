"""Required layout packing followed by the existing fixed GEMM ABI."""
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.lowering import bind_matmul, lower_binary_matmul
from correlator.execution.test_dense_atomic_kernel import _inputs, _loop_reference
from correlator.execution.test_pointer_matrix_product import _Array, _Backend


_SHAPES = ((2, 4, 3), (3, 4, 5))
_DTYPES = (np.float32, np.float64, np.complex64, np.complex128)


def _recipe(equation, dtype):
    return lower_binary_matmul(equation, _SHAPES, (dtype,) * 2)


class PackedMatrixProductCpuTests(unittest.TestCase):
    def test_interleaved_reductions_keep_packing_and_use_one_fixed_gemm(self):
        for dtype in _DTYPES:
            for equation in ('akb,bkc->ac', 'akb,bkc->ca'):
                with self.subTest(dtype=dtype, equation=equation):
                    host = _inputs(_SHAPES, dtype)
                    backend = _Backend()
                    run = bind_matmul(_recipe(equation, dtype), backend)
                    self.assertEqual(backend.device_calls, 0)
                    value = run(*map(_Array, host))
                    np.testing.assert_allclose(value.host, _loop_reference(equation, host),
                                               rtol=3e-5, atol=3e-5)
                    self.assertEqual(len(backend.calls), 1)
                    self.assertEqual(backend.wrappers.count('reshape'), 2)
                    self.assertIn('transpose', backend.wrappers)
                    self.assertNotIn('dot', backend.wrappers)
                    self.assertEqual(backend.allocations, [((2, 5), np.dtype(dtype), 'C')])
                    self.assertEqual(backend.synchronizations, 0)
                    self.assertFalse(any(np.shares_memory(value.host, v) for v in host))

    def test_nondefault_precision_and_debug_preserve_public_product_after_packing(self):
        backend = _Backend()
        host = _inputs(_SHAPES, np.complex128)
        run = bind_matmul(_recipe('akb,bkc->ca', np.complex128), backend)
        for debug, compute, math in ((False, 1, 0), (False, 2, 0), (False, 0, 1),
                                      (True, 0, 0)):
            with self.subTest(debug=debug, compute=compute, math=math):
                backend.compute_mode, backend.math_mode = compute, math
                backend.wrappers.clear()
                with execution_debug(debug):
                    value = run(*map(_Array, host))
                np.testing.assert_allclose(value.host, _loop_reference('akb,bkc->ca', host),
                                           rtol=1e-12, atol=1e-12)
                self.assertEqual(backend.calls, [])
                self.assertEqual(backend.wrappers.count('dot'), 1)
                self.assertEqual(backend.wrappers.count('reshape'), 2)

    def test_current_owners_release_and_old_outputs_survive_later_calls(self):
        backend = _Backend()
        run = bind_matmul(_recipe('akb,bkc->ac', np.complex128), backend)
        refs, results, expected = [], [], []
        for scale in (1, 2):
            arrays = tuple(_Array(scale * value) for value in _inputs(_SHAPES, np.complex128))
            refs.extend(weakref.ref(value) for value in arrays)
            refs.extend(weakref.ref(value.host) for value in arrays)
            expected.append(_loop_reference('akb,bkc->ac', tuple(v.host for v in arrays)))
            results.append(run(*arrays))
            del arrays
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        for value, reference in zip(results, expected, strict=True):
            np.testing.assert_allclose(value.host, reference, rtol=1e-12, atol=1e-12)
        self.assertNotEqual(backend.calls[0][-1], backend.calls[1][-1])

    def test_packed_gemm_failure_keeps_first_error_without_retry(self):
        for failure in ('pointer_mode', 'submitted_kernel'):
            with self.subTest(failure=failure):
                backend = _Backend()
                sentinel = RuntimeError('packed GEMM failed after submission')
                if failure == 'pointer_mode':
                    backend.pointer_mode = 1
                else:
                    backend.failure = sentinel
                run = bind_matmul(_recipe('akb,bkc->ac', np.complex128), backend)
                with self.assertRaises(RuntimeError) as error:
                    run(*map(_Array, _inputs(_SHAPES, np.complex128)))
                if failure == 'submitted_kernel':
                    self.assertIs(error.exception, sentinel)
                self.assertEqual(backend.synchronizations, int(failure == 'submitted_kernel'))
                self.assertNotIn('dot', backend.wrappers)
                self.assertEqual(len(backend.calls), int(failure == 'submitted_kernel'))

    def test_cpu_hip_and_missing_cuda_keep_existing_dispatch(self):
        host = _inputs(_SHAPES, np.complex128)
        expected = _loop_reference('akb,bkc->ac', host)
        np.testing.assert_allclose(bind_matmul(_recipe('akb,bkc->ac', np.complex128), np)(*host),
                                   expected, rtol=1e-12, atol=1e-12)
        for missing in (False, True):
            with self.subTest(missing_cuda=missing):
                backend = _Backend()
                if missing:
                    backend.cuda = None
                else:
                    backend.cuda.runtime.is_hip = True
                value = bind_matmul(_recipe('akb,bkc->ac', np.complex128), backend)(*map(_Array, host))
                np.testing.assert_allclose(value.host, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(backend.calls, [])
                self.assertIn('dot', backend.wrappers)


class PackedMatrixProductCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))
        cls.backend = cupy

    def test_cuda_required_packing_current_stream_and_fresh_results_without_dot(self):
        cp = self.backend
        linalg, cublas = cp._core._routines_linalg, cp.cuda.cublas
        handle = cp.cuda.Device().cublas_handle
        saved_math = cublas.getMathMode(handle)
        stream = cp.cuda.Stream(non_blocking=True)
        for dtype in _DTYPES:
            saved_compute = linalg.get_compute_type(dtype)
            try:
                linalg.set_compute_type(dtype, linalg.COMPUTE_TYPE_DEFAULT)
                cublas.setMathMode(handle, cublas.CUBLAS_DEFAULT_MATH)
                for equation in ('akb,bkc->ac', 'akb,bkc->ca'):
                    with self.subTest(dtype=dtype, equation=equation):
                        host = _inputs(_SHAPES, dtype)
                        run = bind_matmul(_recipe(equation, dtype), cp)
                        results = []
                        with stream, patch.object(cp, 'dot',
                                side_effect=AssertionError('packed GEMM used public dot')):
                            for scale in (1, 2):
                                arrays = tuple(cp.asarray(scale * v) for v in host)
                                results.append(run(*arrays))
                                arrays[0].fill(0)
                        stream.synchronize()
                        tolerance = 5e-5 if dtype in (np.float32, np.complex64) else 2e-12
                        for value, scale in zip(results, (1, 2), strict=True):
                            np.testing.assert_allclose(cp.asnumpy(value),
                                scale ** 2 * _loop_reference(equation, host),
                                rtol=tolerance, atol=tolerance)
                        self.assertNotEqual(int(results[0].data.ptr), int(results[1].data.ptr))
            finally:
                linalg.set_compute_type(dtype, saved_compute)
                cublas.setMathMode(handle, saved_math)

    def test_cuda_packed_layout_preserves_nondefault_precision_policy(self):
        cp = self.backend
        linalg, cublas = cp._core._routines_linalg, cp.cuda.cublas
        handle = cp.cuda.Device().cublas_handle
        saved_math = cublas.getMathMode(handle)
        for dtype in _DTYPES:
            saved_compute = linalg.get_compute_type(dtype)
            host = _inputs(_SHAPES, dtype)
            arrays = tuple(cp.asarray(value) for value in host)
            run = bind_matmul(_recipe('akb,bkc->ca', dtype), cp)
            try:
                for compute, math in ((linalg.COMPUTE_TYPE_PEDANTIC, cublas.CUBLAS_DEFAULT_MATH),
                                      (linalg.COMPUTE_TYPE_DEFAULT, cublas.CUBLAS_TENSOR_OP_MATH)):
                    with self.subTest(dtype=dtype, compute=compute, math=math):
                        linalg.set_compute_type(dtype, compute)
                        cublas.setMathMode(handle, math)
                        with patch.object(cp, 'dot', wraps=cp.dot) as dot:
                            value = run(*arrays)
                        self.assertEqual(dot.call_count, 1)
                        tolerance = 5e-5 if dtype in (np.float32, np.complex64) else 2e-12
                        np.testing.assert_allclose(cp.asnumpy(value),
                            _loop_reference('akb,bkc->ca', host), rtol=tolerance, atol=tolerance)
            finally:
                linalg.set_compute_type(dtype, saved_compute)
                cublas.setMathMode(handle, saved_math)


if __name__ == '__main__':
    unittest.main()
