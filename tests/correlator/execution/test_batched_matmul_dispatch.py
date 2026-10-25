"""Fixed batch geometry retains current arrays, layout and public overrides."""
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref
import numpy as np

from lqcdweave.contraction.execution.lowering import (
    bind_matrix_product, bind_matmul, lower_binary_matmul)
from lqcdweave.contraction.execution.diagnostics import execution_debug


class _GUFunc:
    __module__ = 'cupy._core._gufuncs'
    _signature = '(n?,k),(k,m?)->(n?,m?)'

    def __init__(self, native):
        self._func = native
        self.calls = 0

    def __call__(self, left, right):
        self.calls += 1
        return self._func(left, right)


class BatchedMatmulDispatchTests(unittest.TestCase):
    def backend(self):
        native = Mock(side_effect=np.matmul)
        public = _GUFunc(native)
        return SimpleNamespace(__name__='cupy', matmul=public, reshape=np.reshape,
            transpose=np.transpose, einsum=np.einsum,
            _core=SimpleNamespace(_routines_linalg=SimpleNamespace(matmul=native))), public, native

    def test_fresh_arrays_and_outputs_are_not_retained(self):
        backend, public, native = self.backend()
        run = bind_matrix_product(backend, batched=True)
        refs = []
        for scale in (1, 3):
            left = scale*np.arange(24.).reshape(2, 3, 4)
            right = scale*np.arange(40.).reshape(2, 4, 5)
            value = run(left, right)
            np.testing.assert_array_equal(value, np.matmul(left, right))
            self.assertTrue(value.flags.c_contiguous)
            refs.extend(map(weakref.ref, (left, right, value)))
            del left, right, value
        # Mock records arguments; the actual callable stores only metadata.
        native.reset_mock(); gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(public.calls, 0)

    def test_strides_remain_native_and_broadcast_dtype_debug_wrappers_stay_public(self):
        backend, public, native = self.backend()
        run = bind_matrix_product(backend, batched=True)
        a=np.arange(24.).reshape(2,3,4); b=np.arange(40.).reshape(2,4,5)
        for left,right in ((a[...,::-1], b[:,::-1]), (np.asfortranarray(a),b)):
            np.testing.assert_array_equal(run(left,right),np.matmul(left,right))
        self.assertEqual(public.calls, 0)
        for left,right in ((a[:1],b), (a.astype('float32'),b), (a[0],b[0])):
            np.testing.assert_array_equal(run(left,right),np.matmul(left,right))
        self.assertEqual(public.calls, 3)
        with execution_debug(True): run(a,b)
        self.assertEqual(public.calls, 4)
        wrapped=Mock(wraps=public)
        with patch.object(backend,'matmul',wrapped): run(a,b)
        self.assertEqual(wrapped.call_count,1)
        # Changing the underlying entry after binding must remain observable.
        public._func=Mock(side_effect=np.matmul)
        run(a,b)
        self.assertEqual(public._func.call_count,1)

    def test_long_inner_product_budget_current_modes_overrides_and_failure(self):
        backend, public, native = self.backend()
        state = dict(compute=0, math=0)
        backend._core._routines_linalg.get_compute_type = lambda dtype: state['compute']
        backend._core._routines_linalg.COMPUTE_TYPE_DEFAULT = 0
        backend.cuda = SimpleNamespace(Device=lambda: SimpleNamespace(cublas_handle=7),
            cublas=SimpleNamespace(getMathMode=lambda handle: state['math'], CUBLAS_DEFAULT_MATH=0))
        lowering = lower_binary_matmul('bk,bk->b', ((2,8193),)*2, ('complex128',)*2)
        raw = Mock(side_effect=lambda left,right: np.einsum('bk,bk->b',left,right))
        raw.workspace_bytes = 1024
        with patch('lqcdweave.contraction.execution.reduction_kernel.bind_raw_reduction', return_value=raw) as bind:
            run = bind_matmul(lowering, backend)
        self.assertTrue(run.raw_batched_inner_product)
        self.assertEqual(run.workspace_allowance, (2*2*8193+2)*16)
        self.assertLessEqual(run.workspace_bytes, run.workspace_allowance)
        self.assertEqual(bind.call_args.args[1], ((2,8193),(2,8193)))
        a=np.arange(2*8193,dtype='complex128').reshape(2,8193)
        b=np.ones_like(a)
        np.testing.assert_array_equal(run(a,b),np.einsum('bk,bk->b',a,b))
        self.assertEqual(raw.call_count,1)
        for key in ('compute','math'):
            state[key]=1
            np.testing.assert_array_equal(run(a,b),np.einsum('bk,bk->b',a,b))
            state[key]=0
        with execution_debug(True): run(a,b)
        with patch.object(backend,'matmul',Mock(wraps=public)) as wrapped:
            run(a,b)
            self.assertEqual(wrapped.call_count,1)
        self.assertEqual(raw.call_count,1)
        raw.side_effect=RuntimeError('reduction submission')
        with self.assertRaisesRegex(RuntimeError,'reduction submission'): run(a,b)
        # No retry on the numerical fallback after a failed submission.
        self.assertEqual(native.call_count,4)
        for size,dtype in ((4095,'complex128'),(8193,'complex64')):
            other=lower_binary_matmul('bk,bk->b',((2,size),)*2,(dtype,)*2)
            self.assertFalse(getattr(bind_matmul(other,backend),'raw_batched_inner_product',False))


    def test_long_reductions_leave_the_strided_batched_gemm(self):
        backend, public, native = self.backend()
        backend.empty = np.empty
        backend.dot = Mock(side_effect=np.dot)
        rng = np.random.default_rng(3)
        # Few matrices take one GEMM each, a few output tiles split K; short K
        # or enough output tiles stay batched.
        for (batch, m, n, k), route in (((3, 2, 3, 8192), "dot"), ((10, 2, 3, 8192), "split"),
                                        ((10, 2, 3, 4095), "batched"), ((300, 2, 3, 8192), "batched")):
            with self.subTest(route=route):
                lowering = lower_binary_matmul('bmk,bkn->bmn', ((batch, m, k), (batch, k, n)), ('complex64',)*2)
                run = bind_matmul(lowering, backend)
                left, right = ((rng.standard_normal(shape) + 1j*rng.standard_normal(shape)).astype('complex64')
                               for shape in ((batch, m, k), (batch, k, n)))
                native.reset_mock(); backend.dot.reset_mock(); public.calls = 0
                value = run(left, right)
                self.assertEqual(value.shape, (batch, m, n))
                np.testing.assert_allclose(value, np.matmul(left.astype('complex128'), right.astype('complex128')),
                                           rtol=1e-4, atol=1e-3)
                self.assertEqual((backend.dot.call_count, public.calls, native.call_count),
                                 {"dot": (batch, 0, 0), "split": (0, 1, 1), "batched": (0, 0, 1)}[route])


class BatchedMatmulCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('no CUDA device')
        except (ImportError,RuntimeError) as error:
            raise unittest.SkipTest(str(error))
        cls.cp=cp

    def test_default_route_matches_public_values_strides_and_current_compute_modes(self):
        cp=self.cp; linalg=cp._core._routines_linalg
        stream=cp.cuda.Stream(non_blocking=True)
        for dtype in (np.float32,np.float64,np.complex64,np.complex128):
            saved=linalg.get_compute_type(dtype)
            run=bind_matrix_product(cp,batched=True)
            self.assertIs(run.keywords['native'], linalg.matmul)
            try:
                for mode in (linalg.COMPUTE_TYPE_DEFAULT,linalg.COMPUTE_TYPE_PEDANTIC):
                    linalg.set_compute_type(dtype,mode)
                    for scale in (1,3):
                        with stream:
                            a=cp.asarray(np.arange(24.).reshape(2,3,4),dtype=dtype)*scale
                            b=cp.asarray(np.arange(40.).reshape(2,4,5),dtype=dtype)*(scale+.25)
                            if np.dtype(dtype).kind=='c': a+=1j; b-=.5j
                            variants=((a,b),(cp.asfortranarray(a),cp.asfortranarray(b)),
                                (cp.ascontiguousarray(a.swapaxes(-1,-2)).swapaxes(-1,-2),
                                 cp.ascontiguousarray(b.swapaxes(-1,-2)).swapaxes(-1,-2)),
                                (a[...,::-1],b[:,::-1]),
                                (cp.broadcast_to(a[:1],a.shape),b))
                            products=tuple((run(left,right),cp.matmul(left,right)) for left,right in variants)
                        stream.synchronize()
                        for value,reference in products:
                            np.testing.assert_array_equal(cp.asnumpy(value),cp.asnumpy(reference))
                            self.assertEqual(value.strides,reference.strides)
                            self.assertNotEqual(value.data.ptr,reference.data.ptr)
            finally:
                linalg.set_compute_type(dtype,saved)

    def test_long_double_inner_products_strides_current_stream_and_fresh_owners(self):
        cp=self.cp; rng=np.random.default_rng(193)
        stream=cp.cuda.Stream(non_blocking=True)
        for dtype in (np.float64,np.complex128):
            lowering=lower_binary_matmul('bik,bki->b',((3,4,2049),(3,2049,4)),(dtype,)*2)
            from lqcdweave.contraction.execution.reduction_kernel import bind_raw_reduction
            calls=[]
            def tracked(*args):
                raw=bind_raw_reduction(*args)
                def invoke(*values):
                    calls.append(None)
                    return raw(*values)
                invoke.workspace_bytes=raw.workspace_bytes
                return invoke
            with patch('lqcdweave.contraction.execution.reduction_kernel.bind_raw_reduction',side_effect=tracked):
                run=bind_matmul(lowering,cp)
            self.assertTrue(run.raw_batched_inner_product)
            self.assertLessEqual(run.workspace_bytes,run.workspace_allowance)
            retained=[]
            for scale in (1.,3.):
                host=[rng.normal(size=s).astype(dtype)*scale for s in ((3,4,2049),(3,2049,4))]
                if dtype==np.complex128:
                    host=[v+1j*rng.normal(size=v.shape) for v in host]
                with stream:
                    a,b=map(cp.asarray,host)
                    variants=((a,b),(cp.asfortranarray(a),cp.asfortranarray(b)),
                        (a[...,::-1],b[:,::-1]),(cp.broadcast_to(a[:1],a.shape),b))
                    with patch.object(cp,'reshape',side_effect=AssertionError('no packing')), patch.object(cp,'transpose',side_effect=AssertionError('no packing')):
                        results=[run(*values) for values in variants]
                stream.synchronize()
                self.assertEqual(len(calls),4 if scale==1. else 8)
                for values,value in zip(variants,results):
                    reference=np.einsum('bik,bki->b',*(cp.asnumpy(v) for v in values))
                    np.testing.assert_allclose(cp.asnumpy(value),reference,rtol=1e-11,atol=1e-11)
                    self.assertEqual(value.dtype,np.dtype(dtype))
                    retained.append((value,cp.asnumpy(value)))
                linalg=cp._core._routines_linalg
                saved=linalg.get_compute_type(dtype)
                try:
                    linalg.set_compute_type(dtype,linalg.COMPUTE_TYPE_PEDANTIC)
                    np.testing.assert_allclose(cp.asnumpy(run(a,b)),cp.asnumpy(cp.einsum('bik,bki->b',a,b)),rtol=1e-11,atol=1e-11)
                    self.assertEqual(len(calls),4 if scale==1. else 8)
                finally: linalg.set_compute_type(dtype,saved)
            for value,reference in retained:
                np.testing.assert_array_equal(cp.asnumpy(value),reference)
