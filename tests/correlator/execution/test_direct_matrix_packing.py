"""Required matrix copies from original strides, against independent sums."""
import gc
from types import SimpleNamespace
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
_PACKING = 'lqcdweave.contraction.execution.operations.packing_kernel.bind_packing'


def _recipe(dtype, equation='akb,bkc->ac', shapes=_SHAPES):
    return lower_binary_matmul(equation, shapes, (dtype,) * 2)


def _layouts(xp, left, right):
    # Same logical shapes, both borrowed C/F sides and required stride copies.
    left_f = xp.ascontiguousarray(left.reshape(2, 12).T).reshape(4, 3, 2).transpose(2, 0, 1)
    right_c = xp.ascontiguousarray(right.transpose(1, 0, 2)).transpose(1, 0, 2)
    return ((left, right), (left[..., ::-1], right_c),
            (left_f, right[:, ::-1]), (left[..., ::-1], right[:, ::-1]))


def _protocol_packer(backend, shapes, dtypes, axis, output_dtype, *, input_layouts):
    """Host copy substitute; actual generated copies are covered on CUDA."""
    def copy(values, destination):
        value, = values
        backend.pack_calls.append((value.data.ptr, value.shape, value.strides, destination.shape))
        logical = value.host.transpose(input_layouts[0])
        if logical.shape != shapes[0] or value.dtype != np.dtype(dtypes[0]) or axis != 0:
            raise AssertionError('copy descriptor does not match the original operand')
        destination.host[...] = logical.reshape(destination.shape)
        if backend.pack_failure is not None:
            raise backend.pack_failure
        return destination
    return SimpleNamespace(pack_original=copy)


class DirectMatrixPackingCpuTests(unittest.TestCase):
    def setUp(self):
        self.packing = patch(_PACKING, side_effect=_protocol_packer)
        self.factory = self.packing.start()
        self.addCleanup(self.packing.stop)

    @staticmethod
    def backend():
        backend = _Backend()
        backend.pack_calls, backend.pack_failure = [], None
        return backend

    def test_c_f_borrowed_sides_and_negative_strides_copy_only_required_operands(self):
        for dtype in _DTYPES:
            host = _inputs(_SHAPES, dtype)
            for index, values in enumerate(_layouts(np, *host)):
                for equation in ('akb,bkc->ac', 'akb,bkc->ca'):
                    with self.subTest(dtype=dtype, layout=index, equation=equation):
                        backend = self.backend()
                        run = bind_matmul(_recipe(dtype, equation), backend)
                        self.assertEqual(backend.device_calls, 0)
                        arrays = tuple(map(_Array, values))
                        result = run(*arrays)
                        np.testing.assert_allclose(result.host, _loop_reference(equation, values),
                                                   rtol=4e-5, atol=4e-5)
                        copies = 2 if index == 3 else 1
                        self.assertEqual(len(backend.pack_calls), copies)
                        self.assertEqual(len(backend.allocations), copies + 1)
                        current = {value.data.ptr for value in arrays}
                        self.assertTrue(all(call[0] in current for call in backend.pack_calls))
                        self.assertEqual(len(backend.calls), 1)
                        self.assertNotIn('reshape', backend.wrappers)
                        self.assertNotIn('dot', backend.wrappers)
                        self.assertEqual(backend.wrappers.count('transpose'), int(equation.endswith('ca')))
                        self.assertEqual(backend.synchronizations, 0)
                        self.assertFalse(any(np.shares_memory(result.host, v) for v in values))

    def test_precision_fallback_reuses_the_copy_and_debug_keeps_original_views(self):
        host = _inputs(_SHAPES, np.complex128)
        for debug, compute, math in ((False, 1, 0), (False, 0, 1), (True, 0, 0)):
            with self.subTest(debug=debug, compute=compute, math=math):
                backend = self.backend()
                backend.compute_mode, backend.math_mode = compute, math
                run = bind_matmul(_recipe(np.complex128), backend)
                with execution_debug(debug):
                    result = run(*map(_Array, host))
                np.testing.assert_allclose(result.host, _loop_reference('akb,bkc->ac', host),
                                           rtol=1e-12, atol=1e-12)
                self.assertEqual(len(backend.pack_calls), int(not debug))
                self.assertEqual(len(backend.allocations), int(not debug))
                self.assertEqual(backend.wrappers.count('reshape'), 2 if debug else 1)
                self.assertEqual(backend.wrappers.count('dot'), 1)
                self.assertEqual(backend.calls, [])
                self.assertEqual(backend.synchronizations, 0)

    def test_original_input_boundary_and_missing_copy_capability_keep_fallback(self):
        host = _inputs(_SHAPES, np.complex128)
        backend = self.backend()
        run = bind_matmul(_recipe(np.complex128), backend)
        with self.assertRaises(ValueError):
            run(_Array(host[0][:1]), _Array(host[1]))
        self.assertEqual(backend.pack_calls, [])
        changed_dtype = host[0].real.astype(np.float32), host[1]
        result = run(*map(_Array, changed_dtype))
        np.testing.assert_allclose(result.host, _loop_reference('akb,bkc->ac', changed_dtype),
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(backend.pack_calls, [])
        backend = self.backend()
        with patch(_PACKING, return_value=None):
            result = bind_matmul(_recipe(np.complex128), backend)(*map(_Array, host))
        np.testing.assert_allclose(result.host, _loop_reference('akb,bkc->ac', host),
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(backend.pack_calls, [])
        self.assertEqual(backend.wrappers.count('reshape'), 2)

    def test_singleton_cpu_and_hip_do_not_bind_raw_matrix_copies(self):
        shapes = ((2, 1, 3), (3, 1, 5))
        host = _inputs(shapes, np.complex128)
        backend = self.backend()
        run = bind_matmul(_recipe(np.complex128, shapes=shapes), backend)
        np.testing.assert_allclose(run(*map(_Array, host)).host,
                                   _loop_reference('akb,bkc->ac', host), rtol=1e-12, atol=1e-12)
        self.factory.assert_not_called()
        host = _inputs(_SHAPES, np.complex128)
        np.testing.assert_allclose(bind_matmul(_recipe(np.complex128), np)(*host),
                                   _loop_reference('akb,bkc->ac', host), rtol=1e-12, atol=1e-12)
        backend.cuda.runtime.is_hip = True
        result = bind_matmul(_recipe(np.complex128), backend)(*map(_Array, host))
        np.testing.assert_allclose(result.host, _loop_reference('akb,bkc->ac', host),
                                   rtol=1e-12, atol=1e-12)
        self.factory.assert_not_called()

    def test_current_inputs_release_and_old_outputs_survive_new_calls(self):
        backend = self.backend()
        run = bind_matmul(_recipe(np.complex128), backend)
        refs, results, expected = [], [], []
        for scale in (1, 2):
            arrays = tuple(_Array(scale * v) for v in _inputs(_SHAPES, np.complex128))
            refs.extend(weakref.ref(v) for v in arrays)
            refs.extend(weakref.ref(v.host) for v in arrays)
            expected.append(_loop_reference('akb,bkc->ac', tuple(v.host for v in arrays)))
            results.append(run(*arrays))
            del arrays
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        for value, reference in zip(results, expected, strict=True):
            np.testing.assert_allclose(value.host, reference, rtol=1e-12, atol=1e-12)
        self.assertNotEqual(results[0].data.ptr, results[1].data.ptr)

    def test_copy_and_gemm_failures_keep_first_error_and_never_retry(self):
        for stage in ('copy', 'gemm'):
            with self.subTest(stage=stage):
                backend = self.backend()
                first = RuntimeError('submitted ' + stage + ' failed')
                if stage == 'copy':
                    backend.pack_failure = first
                else:
                    backend.failure = first
                run = bind_matmul(_recipe(np.complex128), backend)
                with self.assertRaises(RuntimeError) as caught:
                    run(*map(_Array, _inputs(_SHAPES, np.complex128)))
                self.assertIs(caught.exception, first)
                self.assertEqual(backend.synchronizations, 1)
                self.assertEqual(len(backend.pack_calls), 1)
                self.assertEqual(len(backend.calls), int(stage == 'gemm'))
                self.assertNotIn('dot', backend.wrappers)


class DirectMatrixPackingCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))
        cls.backend = cupy

    def test_cuda_original_strides_current_stream_and_fresh_outputs_without_input_views(self):
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
                host = _inputs(_SHAPES, dtype)
                host_layouts = _layouts(np, *host)
                for index in range(4):
                    with self.subTest(dtype=dtype, layout=index):
                        run = bind_matmul(_recipe(dtype), cp)
                        outputs = []
                        with stream:
                            for scale in (1, 2):
                                arrays = _layouts(cp, *(cp.asarray(scale * v) for v in host))[index]
                                with (patch.object(cp, 'transpose', side_effect=AssertionError('input header')),
                                        patch.object(cp, 'reshape', side_effect=AssertionError('matrix header')),
                                        patch.object(cp, 'dot', side_effect=AssertionError('public GEMM'))):
                                    outputs.append(run(*arrays))
                                arrays[0].fill(0)
                        stream.synchronize()
                        tolerance = 6e-5 if dtype in (np.float32, np.complex64) else 2e-12
                        for value, scale in zip(outputs, (1, 2), strict=True):
                            np.testing.assert_allclose(cp.asnumpy(value), scale ** 2 *
                                _loop_reference('akb,bkc->ac', host_layouts[index]),
                                rtol=tolerance, atol=tolerance)
                        self.assertNotEqual(outputs[0].data.ptr, outputs[1].data.ptr)
            finally:
                linalg.set_compute_type(dtype, saved_compute)
                cublas.setMathMode(handle, saved_math)

    def test_cuda_nondefault_precision_and_debug_retain_public_policy(self):
        cp = self.backend
        linalg, cublas = cp._core._routines_linalg, cp.cuda.cublas
        handle = cp.cuda.Device().cublas_handle
        saved_math = cublas.getMathMode(handle)
        for dtype in _DTYPES:
            saved_compute = linalg.get_compute_type(dtype)
            host = _inputs(_SHAPES, dtype)
            run = bind_matmul(_recipe(dtype, 'akb,bkc->ca'), cp)
            try:
                for debug, compute, math in (
                        (False, linalg.COMPUTE_TYPE_PEDANTIC, cublas.CUBLAS_DEFAULT_MATH),
                        (False, linalg.COMPUTE_TYPE_DEFAULT, cublas.CUBLAS_TENSOR_OP_MATH),
                        (True, linalg.COMPUTE_TYPE_DEFAULT, cublas.CUBLAS_DEFAULT_MATH)):
                    with self.subTest(dtype=dtype, debug=debug, compute=compute, math=math):
                        linalg.set_compute_type(dtype, compute)
                        cublas.setMathMode(handle, math)
                        with execution_debug(debug), patch.object(cp, 'dot', wraps=cp.dot) as dot:
                            value = run(*(cp.asarray(v) for v in host))
                        self.assertEqual(dot.call_count, 1)
                        tolerance = 6e-5 if dtype in (np.float32, np.complex64) else 2e-12
                        np.testing.assert_allclose(cp.asnumpy(value), _loop_reference('akb,bkc->ca', host),
                                                   rtol=tolerance, atol=tolerance)
            finally:
                linalg.set_compute_type(dtype, saved_compute)
                cublas.setMathMode(handle, saved_math)
