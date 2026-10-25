"""Common matrix kernels reuse code while binding current geometry and strides."""
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern
from lqcdweave.contraction.execution.reduction_kernel import (
    _kernel, bind_raw_reduction, matrix_reduction_sources,
)
from correlator.execution.test_raw_precise_reduction import _SourceBackend


class MatrixReductionSourceTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()

    def test_same_sources_serve_different_shapes_axes_signed_strides_and_split_counts(self):
        backend=_SourceBackend()
        rng=np.random.default_rng(308)
        templates=[]
        def abi(*args,**kwargs):
            templates.append((args[2],args[4],args[5]))
            return None
        with patch('lqcdweave.contraction.execution.reduction_kernel.bind_pointer_arguments',side_effect=abi):
            for dtype in (np.float32,np.complex64):
                wide=np.float64 if dtype is np.float32 else np.complex128
                for m,k,n in ((2,513,3),(3,257,2),(1,800,5)):
                    a=rng.normal(size=(m,k)).astype(dtype)
                    b=rng.normal(size=(k,n)).astype(dtype)
                    if dtype is np.complex64:
                        a=a+1j*a[:,::-1]; b=b-.25j*b[::-1]
                    a,b=a[::-1,::-1],b[::-1,::-1]
                    reference=(a.astype(wide)@b.astype(wide)).astype(dtype)
                    for equation,left,right in (('ik,kj->ij',a,b),('ik,jk->ij',a,b.T),('ki,jk->ij',a.T,b.T)):
                        fn=bind_raw_reduction(parse_axis_pattern(equation),(left.shape,right.shape),np.dtype(dtype),backend,8192)
                        self.assertEqual(fn.raw_source_family,'matrix_runtime_geometry')
                        self.assertLessEqual(fn.workspace_bytes,8192)
                        np.testing.assert_allclose(fn(left,right),reference,rtol=1e-6,atol=1e-6)
                        np.testing.assert_allclose(fn(2*left,right),2*reference,rtol=1e-6,atol=1e-6)
        self.assertEqual(len(backend.sources),4)
        self.assertEqual(backend.launches,72)
        first=[item for item in templates if len(item[1])==11]
        self.assertTrue(first)
        self.assertTrue(any(any(value<0 for value in template[6:10]) for _,template,_ in first))
        for grid,template,positions in first:
            self.assertEqual(len(grid),3)
            self.assertEqual(positions,((0,0),(1,1)))
            self.assertTrue(all(type(value) is np.int64 for value in template[2:-1]))
            self.assertIsNone(template[-1])
        for source in backend.sources:
            self.assertNotIn(' / ',source)
            self.assertNotIn(' % ',source)

    def test_grid_dimension_boundary_keeps_general_kernel_without_invalid_launch(self):
        backend=_SourceBackend()
        small=bind_raw_reduction(parse_axis_pattern('ik,kj->ij'),((65535,2),(2,1)),np.dtype('float32'),backend,1<<24)
        large=bind_raw_reduction(parse_axis_pattern('ik,kj->ij'),((65536,2),(2,1)),np.dtype('float32'),backend,1<<24)
        self.assertEqual(small.raw_source_family,'matrix_runtime_geometry')
        self.assertEqual(large.raw_source_family,'static_geometry')
        self.assertEqual(backend.launches,0)


class MatrixReductionSourcesCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()<1:
                raise RuntimeError('no CUDA device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')
        cls.cp=cp

    def test_cuda_runtime_dimensions_byte_strides_and_partial_counts_share_kernels(self):
        cp=self.cp
        rng=np.random.default_rng(816)
        for dtype in (np.float32,np.complex64):
            for m,k,n in ((2,257,3),(3,513,2),(2,134400,4),(1,230400,3)):
                a=rng.normal(size=(m,k)).astype(dtype)
                b=rng.normal(size=(k,n)).astype(dtype)
                if dtype is np.complex64:
                    a=a+.5j*a[:,::-1]; b=b-.3j*b[::-1]
                left=cp.asarray(a)[::-1,::-1]; right=cp.asarray(b)[::-1,::-1]
                wide=np.float64 if dtype is np.float32 else np.complex128
                reference=(a[::-1,::-1].astype(wide)@b[::-1,::-1].astype(wide)).astype(dtype)
                for equation,x,y in (('ik,kj->ij',left,right),('ik,jk->ij',left,right.T)):
                    fn=bind_raw_reduction(parse_axis_pattern(equation),(x.shape,y.shape),np.dtype(dtype),cp,1<<20)
                    self.assertEqual(fn.raw_source_family,'matrix_runtime_geometry')
                    np.testing.assert_allclose(cp.asnumpy(fn(x,y)),reference,rtol=3e-6,atol=1e-5)
        first,finish=matrix_reduction_sources('complex64')
        self.assertEqual(first.count('__syncthreads();'),1)
        self.assertEqual(finish.count('__syncthreads();'),1)


if __name__=='__main__':
    unittest.main()
