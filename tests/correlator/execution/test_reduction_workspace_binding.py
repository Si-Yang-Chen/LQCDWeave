"""Raw reductions borrow only internal partials from an admitted device scope."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern
from lqcdweave.contraction.execution.reduction_kernel import _kernel, bind_raw_reduction
from correlator.execution.test_raw_precise_reduction import _SourceBackend


class _Allocations(_SourceBackend):
    def __init__(self):
        super().__init__()
        self.sizes=[]

    def empty(self,shape,dtype):
        value=np.empty(shape,dtype=dtype)
        self.sizes.append(value.nbytes)
        return value


class ReductionWorkspaceBindingTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()

    def test_optional_cuda_protocol_missing_keeps_workspace_disabled(self):
        from lqcdweave.contraction.execution.reduction_workspace import ReductionWorkspace
        workspace=ReductionWorkspace(SimpleNamespace(cuda=SimpleNamespace()))
        self.assertEqual(workspace.configure(4096),0)
        self.assertIsNone(workspace.borrow(16,np.complex128))
        workspace.close()

    def test_flat_partial_owner_reused_but_published_outputs_remain_independent(self):
        rng=np.random.default_rng(513)
        for equation,shapes in (('ik,jk->ij',((3,513),(5,513))),('i,i->',((513,),)*2)):
            backend=_Allocations()
            partials=np.full(4096,complex(41,-7),np.complex128)
            borrows=[]
            def borrow(nbytes,dtype):
                self.assertEqual(dtype,partials.dtype)
                self.assertLessEqual(nbytes,partials.nbytes)
                borrows.append(nbytes)
                return partials
            device=SimpleNamespace(backend=backend,reduction_partial=borrow)
            values=tuple((rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64) for shape in shapes)
            with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
                run=bind_raw_reduction(parse_axis_pattern(equation),shapes,np.dtype('complex64'),backend,8192)
                first=run(*values)
                saved=first.copy()
                second=run(*(2*v for v in values))
            np.testing.assert_allclose(first,np.einsum(equation,*values,dtype=np.complex128).astype(np.complex64),rtol=1e-6,atol=1e-6)
            np.testing.assert_array_equal(first,saved)
            np.testing.assert_allclose(second,4*saved,rtol=1e-6,atol=1e-6)
            self.assertFalse(np.shares_memory(first,second))
            self.assertEqual(backend.sizes,[first.nbytes,second.nbytes])
            self.assertEqual(len(borrows),2)
            self.assertTrue(np.all(partials[borrows[-1]//partials.itemsize:]==complex(41,-7)))

    def test_missing_workspace_or_different_backend_keeps_original_allocations_and_errors(self):
        for kind in ('no-scope','no-reservation','other-backend','legacy-protocol'):
            backend=_Allocations()
            device=(None if kind=='no-scope' else SimpleNamespace(
                backend=backend if kind=='no-reservation' else object(),reduction_partial=lambda *args:None))
            if kind=='legacy-protocol':
                device=SimpleNamespace(backend=backend)
            with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
                run=bind_raw_reduction(parse_axis_pattern('i,i->'),((513,),)*2,np.dtype('float32'),backend,8192)
                values=(np.ones(513,np.float32),)*2
                self.assertEqual(run(*values),np.float32(513))
                self.assertEqual(len(backend.sizes),2)
                backend.fail_at=backend.launches+2
                with self.assertRaisesRegex(RuntimeError,'submission failed'):
                    run(*values)
                self.assertEqual(backend.synchronize.call_count,1)


class ReductionWorkspaceBindingCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount()<1:
                raise RuntimeError('no CUDA device')
        except Exception as error:
            raise unittest.SkipTest(f'CuPy device unavailable: {error}')
        cls.cp=cp

    def test_actual_static_and_tiled_reductions_reuse_partials_with_independent_results(self):
        from lqcdweave.contraction.execution.device import device_execution
        cp=self.cp
        rng=np.random.default_rng(1248)
        for dtype in (np.float32,np.complex64):
            wide=np.float64 if dtype is np.float32 else np.complex128
            outputs=[];references=[]
            with device_execution(cp,device_memory_limit=1<<26) as device:
                self.assertGreater(device.reserve_reduction_workspace(1<<20),0)
                for equation,shapes in (('ik,jk->ij',((3,513),(5,513))),
                        ('iik,oki->o',((5,5,49),(3,49,5))),('i,i->',((513,),)*2)):
                    host=tuple(rng.normal(size=shape).astype(dtype) for shape in shapes)
                    if dtype is np.complex64:
                        host=tuple(value+.3j*value[...,::-1] for value in host)
                    run=bind_raw_reduction(parse_axis_pattern(equation),shapes,np.dtype(dtype),cp,8192)
                    for scale in (1,2):
                        current=tuple(scale*value for value in host)
                        outputs.append(run(*(cp.asarray(value) for value in current)))
                        references.append(np.einsum(equation,*current,dtype=wide).astype(dtype))
                self.assertEqual(device.reduction_workspace.allocations,1)
                self.assertEqual(device.reduction_workspace.hits,6)
                self.assertEqual(device.reduction_workspace.misses,0)
                # DeviceExecution owns a private pool. Download while its
                # owner is live, as the common Calculation path does.
                for value,expected in zip(outputs,references,strict=True):
                    np.testing.assert_allclose(cp.asnumpy(value),expected,rtol=3e-6,atol=1e-5)
            self.assertIsNone(device.reduction_workspace.buffer)

    def test_other_stream_and_zero_headroom_keep_original_allocation_path(self):
        from lqcdweave.contraction.execution.device import device_execution
        cp=self.cp
        host=np.linspace(-1,1,513,dtype=np.float32)
        with device_execution(cp,device_memory_limit=1<<26) as device:
            device.reserve_reduction_workspace(1<<20)
            value=cp.asarray(host)
            run=bind_raw_reduction(parse_axis_pattern('i,i->'),(host.shape,)*2,host.dtype,cp,8192)
            first=run(value,value)
            cp.cuda.get_current_stream().synchronize()
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                second=run(value,value)
            stream.synchronize()
            self.assertEqual(device.reduction_workspace.hits,1)
            self.assertEqual(device.reduction_workspace.misses,1)
            third=run(value,value)
        for result in (first,second,third):
            np.testing.assert_allclose(cp.asnumpy(result),np.einsum('i,i->',host,host,dtype=np.float64).astype(np.float32),rtol=1e-6)
        # Pooled arrays do not keep CuPy's private pool alive after its owner
        # is replaced. Neither retain inputs nor read old results across pools.
        expected=np.einsum('i,i->',host,host,dtype=np.float64).astype(np.float32)
        del first,second,third,value,result
        with device_execution(cp,device_memory_limit=1<<26) as device:
            self.assertEqual(device.reserve_reduction_workspace(1<<26),0)
            value=cp.asarray(host)
            result=run(value,value)
            self.assertEqual(device.reduction_workspace.allocations,0)
            self.assertEqual(device.reduction_workspace.hits,0)
            observed=cp.asnumpy(result)
        np.testing.assert_allclose(observed,expected,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
