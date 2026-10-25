"""CPU source/ABI interpretation; real CUDA correctness is an explicit gate."""
import ctypes
import os
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern
from lqcdweave.contraction.execution import scalar_reduction_group as module
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.reduction_kernel import _kernel, _wide_parts, reduction_sources


def _tree(values):
    """Independent NumPy model of the existing eight-warp sum order."""
    lanes = values.reshape(8, 32).copy()
    for step in (16, 8, 4, 2, 1):
        previous = lanes.copy()
        lanes[:, :32-step] = previous[:, :32-step] + previous[:, step:]
    lanes = lanes[:, 0].copy()
    for step in (4, 2, 1):
        previous = lanes.copy()
        lanes[:8-step] = previous[:8-step] + previous[step:]
    return lanes[0]


def _reference(left, right, axes, blocks):
    wide = np.dtype('complex128' if left.dtype.kind == 'c' else 'float64')
    right_order = tuple(axes[1].index(label) for label in axes[0])
    terms = left.astype(wide).ravel() * right.transpose(right_order).astype(wide).ravel()
    partials = np.empty(blocks, dtype=wide)
    for split in range(blocks):
        lanes = np.zeros(256, dtype=wide)
        for start in range(split*256, terms.size, blocks*256):
            take = terms[start:start+256]
            lanes[:take.size] += take
        partials[split] = _tree(lanes)
    lanes = np.zeros(256, dtype=wide)
    for start in range(0, blocks, 256):
        take = partials[start:start+256]
        lanes[:take.size] += take
    return partials, np.asarray(_tree(lanes), dtype=left.dtype)


class _SourceBackend:
    __name__ = 'cupy'

    def __init__(self, *, metadata_only=False):
        self.sources = []; self.launches = []; self.allocations = []
        self.partial_snapshot = None; self.fail_at = None; self.fail_allocation = None
        self.metadata_only = metadata_only
        self.synchronize = Mock()
        self.cuda = SimpleNamespace(runtime=SimpleNamespace(is_hip=False),
            get_current_stream=lambda: SimpleNamespace(synchronize=self.synchronize))

    def empty(self, shape, dtype):
        if self.fail_allocation == len(self.allocations)+1:raise MemoryError('allocation failed')
        value = np.empty(shape, dtype=dtype)
        self.allocations.append(weakref.ref(value))
        return value

    def RawKernel(self, source, name, **options):
        assert options['options'] == ('--std=c++17', '--fmad=false')
        self.sources.append((source, name))
        def kernel(grid, threads, arguments):
            self.launches.append((grid, threads, name))
            if self.fail_at == len(self.launches):raise RuntimeError('submission failed')
            if self.metadata_only:return
            if name.endswith('_partial'):
                count, blocks = grid[1], int(arguments[-2])
                capacity = len(re.findall(r'case \d+:', source))
                rank_words = (len(arguments)-2)//capacity
                size = int(re.search(r'i < (\d+)LL', source)[1])
                offsets = re.findall(r'p([01]) \+ \((.*?)\)\)\);', source)
                assert len(offsets)==2
                expressions=[compile(expr.replace('LL','').replace('/','//'),'<group offset>','eval') for _,expr in offsets]
                stride_names=re.search(r'\n    long long (s0_0[^;]*);',source)[1].split(', ')
                partials=arguments[-1].reshape(-1)[:count*blocks].reshape(count,blocks)
                for task in range(count):
                    base=task*rank_words;values=arguments[base:base+2]
                    strides=dict(zip(stride_names,map(int,arguments[base+2:base+rank_words]),strict=True))
                    for split in range(blocks):
                        lanes=np.zeros(256,dtype=partials.dtype)
                        for start in range(split*256,size,blocks*256):
                            for thread,i in enumerate(range(start,min(size,start+256))):
                                loaded=[]
                                for value,expr in zip(values,expressions,strict=True):
                                    offset=eval(expr,{'__builtins__':{}},dict(strides,i=i))
                                    raw=ctypes.string_at(value.ctypes.data+offset,value.itemsize)
                                    loaded.append(partials.dtype.type(np.frombuffer(raw,dtype=value.dtype)[0]))
                                lanes[thread]+=loaded[0]*loaded[1]
                        partials[task,split]=_tree(lanes)
                self.partial_snapshot=partials.copy()
            else:
                count,blocks=grid[0],int(arguments[-2])
                partials=arguments[0].reshape(-1)[:count*blocks].reshape(count,blocks)
                outputs=(arguments[-1],*arguments[1:count])
                for task,result in enumerate(outputs):
                    lanes=np.zeros(256,dtype=partials.dtype)
                    for start in range(0,blocks,256):
                        take=partials[task,start:start+256];lanes[:take.size]+=take
                    result[()]=_tree(lanes)
        return kernel


def _allowance(shapes,dtype):
    return sum(np.prod(shape,dtype=np.int64).item()*np.dtype(dtype).itemsize for shape in shapes)+np.dtype(dtype).itemsize


def bind_shared(*args, **kwargs):
    return module.bind_scalar_reduction_group(*args, **kwargs, share_small_outputs=True)


class ScalarReductionGroupTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()
        module._FAILED_SUBMISSIONS.clear()

    def test_emitted_offsets_and_independent_tree_cover_permutations_negative_zero_strides(self):
        rng=np.random.default_rng(91)
        equation=parse_axis_pattern('abcd,cdab->').normalized()
        shapes=((2,7,3,13),(3,13,2,7))
        for dtype in (np.float32,np.complex64):
            for count in (2,4,8):
                tasks=[]
                for task in range(count):
                    values=[]
                    for shape in shapes:
                        value=rng.normal(size=shape).astype(dtype)
                        if dtype==np.complex64:value+=1j*rng.normal(size=shape).astype(np.float32)
                        if task%3==1:value=value[...,::-1]
                        if task%3==2:value=np.broadcast_to(value[:1],shape)
                        values.append(value)
                    tasks.append(tuple(values))
                backend=_SourceBackend()
                bound=bind_shared(equation,shapes,dtype,(_allowance(shapes,dtype),)*count,backend)
                outputs=bound(tasks)
                self.assertEqual(len(backend.launches),2)
                self.assertEqual(backend.launches[0][0],(bound.blocks,count))
                self.assertEqual(backend.launches[1][0],(count,))
                self.assertEqual(len(set(out.ctypes.data for out in outputs)),count)
                self.assertTrue(all(out.shape==() and out.base is not None for out in outputs))
                for i,((a,b),out) in enumerate(zip(tasks,outputs,strict=True)):
                    partial,reference=_reference(a,b,equation.inputs,bound.blocks)
                    self.assertEqual(backend.partial_snapshot[i].tobytes(),partial.tobytes())
                    self.assertEqual(out.tobytes(),reference.tobytes())
                    wide=np.complex128 if dtype==np.complex64 else np.float64
                    oracle=np.einsum('abcd,cdab->',a,b,dtype=wide).astype(dtype)
                    np.testing.assert_allclose(out,oracle,rtol=2e-6,atol=2e-6)
                for source,_ in backend.sources:
                    self.assertIn(_wide_parts(dtype)[5],source)
                    self.assertEqual(source.count('switch (task)'),1)

    def test_source_reused_for_counts_strides_and_split_counts_and_fresh_pointers(self):
        backend=_SourceBackend();shape=(513,)
        for count in (2,4,8):
            bound=bind_shared('i,i->',(shape,shape),np.float32,(4096,)*count,backend)
            for step in (1,-1):
                tasks=[(np.arange(513,dtype=np.float32)[::step],np.full(513,i+1,np.float32)) for i in range(count)]
                values=bound(tasks)
                for i,value in enumerate(values):self.assertEqual(value,np.arange(513,dtype=np.float64).sum()*(i+1))
            # Layout bindings retain integers, not prior input/output owners.
            refs=[weakref.ref(x) for pair in tasks for x in pair]+[weakref.ref(x) for x in values]
            del tasks,values,value
            self.assertTrue(all(ref() is None for ref in refs))
        other=bind_shared('i,i->',(shape,shape),np.float32,(4+2*8,)*2,backend)
        self.assertEqual(other.blocks,2)
        other([(np.ones(shape,np.float32),np.ones(shape,np.float32))]*2)
        self.assertEqual(len(backend.sources),2)

    def test_rejects_unsupported_geometry_precision_budget_and_runtime_mismatch(self):
        backend=_SourceBackend();shapes=((513,),)*2
        bind=lambda **kw:bind_shared(kw.get('equation','i,i->'),kw.get('shapes',shapes),kw.get('dtype',np.complex64),kw.get('allowances',(8200,8200)),kw.get('backend',backend))
        for kwargs in (dict(allowances=(0,0)),dict(allowances=(24,40)),dict(allowances=(8200,)),dict(allowances=(8200,)*33),
                       dict(dtype=np.complex128),dict(equation='i,i->i'),dict(equation='ii,ii->',shapes=((8,8),)*2),
                       dict(backend=np),dict(shapes=((513,),(512,)))):
            self.assertIsNone(bind(**kwargs),kwargs)
        with execution_debug(True):self.assertIsNone(bind())
        backend.cuda.runtime.is_hip=True;self.assertIsNone(bind());backend.cuda.runtime.is_hip=False
        labels='abcdefghijklmnopqrstuvw'
        self.assertIsNone(bind(equation=f'{labels},{labels}->',shapes=((2,)+(1,)*22,)*2))
        run=bind();a=np.ones(513,np.complex64)
        for tasks in ([(a,a)],[(a,a,a)]*2,[(a.astype(np.complex128),a)]*2,[(a[:-1],a)]*2):
            with self.assertRaises(ValueError):run(tasks)
        invalid=SimpleNamespace(shape=a.shape,dtype=a.dtype,strides=(2**63,),data=SimpleNamespace(ptr=123))
        with self.assertRaisesRegex(ValueError,'strides'):run([(invalid,a)]*2)
        invalid.strides=a.strides;invalid.data.ptr=0
        with self.assertRaisesRegex(ValueError,'pointer'):run([(invalid,a)]*2)
        self.assertEqual(backend.launches,[])
        backend.ndarray=type('DeviceArray',(),{})
        with self.assertRaisesRegex(ValueError,'bound array backend'):bind()([(a,a)]*2)

    def test_wide_cancellation_single_borrow_independent_results_and_failures(self):
        backend=_SourceBackend();a=np.tile(np.array([128.125,128.125,-128.125,-128.125],np.float32),200)
        a[0]=np.nextafter(a[0],np.float32(np.inf));b=np.full_like(a,1.125)
        run=bind_shared('i,i->',(a.shape,)*2,a.dtype,(_allowance((a.shape,)*2,a.dtype),)*4,backend)
        buffer=np.empty(8*1024**2//8,np.float64)
        borrow=Mock(return_value=buffer);device=SimpleNamespace(backend=backend,reduction_partial=borrow)
        with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
            outputs=run([(a,b)]*4)
        borrow.assert_called_once_with(run.partial_bytes,np.dtype('float64'))
        self.assertEqual(len(backend.allocations),1)
        for out in outputs:self.assertEqual(out,np.einsum('i,i->',a,b,dtype=np.float64).astype(np.float32))
        for relative in (1,2):
            backend.fail_at=len(backend.launches)+relative
            with self.assertRaisesRegex(RuntimeError,'submission failed'):run([(a,b)]*4)
        self.assertEqual(backend.synchronize.call_count,2)
        backend.synchronize.side_effect=RuntimeError('sync failed')
        backend.fail_at=len(backend.launches)+1
        pool=object();device.pool=pool
        with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
            with self.assertRaisesRegex(RuntimeError,'submission failed') as context:run([(a,b)]*4)
        self.assertTrue(context.exception.__notes__)
        self.assertEqual(len(module._FAILED_SUBMISSIONS),1)
        held=module._FAILED_SUBMISSIONS[0]
        self.assertIs(held[1][0],a);self.assertEqual(len(held[3]),4)
        self.assertIs(held[4],pool)

    def test_native_argument_protocol_refreshes_current_addresses(self):
        backend=_SourceBackend(metadata_only=True);calls=[];templates=[]
        def bind(backend,kernel,grid,block,template,positions,*,pointer_getter):
            templates.append(template)
            def launch(inputs,result):
                words=list(template)
                for position,index in positions:words[position]=pointer_getter(inputs[index])
                words[-1]=module._pointer(result)
                calls.append((grid,tuple(words)))
            return launch
        with patch.object(module,'bind_pointer_arguments',side_effect=bind):
            run=bind_shared('i,i->',((513,),)*2,np.float32,(4108,)*2,backend)
            old=[(np.ones(513,np.float32),np.ones(513,np.float32)) for _ in range(2)]
            outputs1=run(old)
            new=[(np.full(513,2,np.float32),np.full(513,3,np.float32)) for _ in range(2)]
            outputs2=run(new)
        self.assertEqual(len(calls),4);self.assertEqual(len(templates),2)
        self.assertEqual(calls[0][1][0],old[0][0].ctypes.data)
        self.assertEqual(calls[2][1][0],new[0][0].ctypes.data)
        self.assertNotEqual(calls[0][1][0],calls[2][1][0])
        self.assertEqual(calls[1][1][1],outputs1[1].ctypes.data)
        self.assertEqual(calls[3][1][-1],outputs2[0].ctypes.data)

    def test_borrow_miss_other_backend_and_allocation_error_keep_fallback_bounded(self):
        backend=_SourceBackend();a=np.ones(513,np.complex64)
        run=bind_shared('i,i->',(a.shape,)*2,a.dtype,(8200,)*2,backend)
        borrow=Mock(return_value=None)
        with patch('lqcdweave.contraction.execution.device.active_device_execution',
                   return_value=SimpleNamespace(backend=backend,reduction_partial=borrow)):
            result=run([(a,a)]*2)
        borrow.assert_called_once_with(run.partial_bytes,np.dtype('complex128'))
        self.assertEqual(len(backend.allocations),2)
        self.assertLessEqual(run.workspace_bytes,run.workspace_allowance)
        borrow.reset_mock()
        with patch('lqcdweave.contraction.execution.device.active_device_execution',
                   return_value=SimpleNamespace(backend=object(),reduction_partial=borrow)):
            result=run([(a,a)]*2)
        borrow.assert_not_called()
        backend.fail_allocation=len(backend.allocations)+2
        with self.assertRaisesRegex(MemoryError,'allocation failed'):run([(a,a)]*2)
        self.assertEqual(backend.synchronize.call_count,1)


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE')=='1','explicit CUDA gate; no device probe in CPU suite')
class ScalarReductionGroupCuPyTests(unittest.TestCase):
    def test_partial_and_final_bytes_match_original_single_kernels(self):
        import cupy as cp
        rng=np.random.default_rng(115)
        for dtype in (np.float32,np.complex64):
            for shapes in (((513,),)*2,((4,70,4,120),(4,120,4,70)),((4,120,4,120),)*2):
                equation=parse_axis_pattern('i,i->' if len(shapes[0])==1 else 'abcd,cdab->').normalized()
                for count in (2,4,8):
                    tasks=[]
                    for task in range(count):
                        pair=[]
                        for shape in shapes:
                            host=rng.normal(size=shape).astype(dtype)
                            if dtype==np.complex64:host+=1j*rng.normal(size=shape).astype(np.float32)
                            value=cp.asarray(host)
                            if task%3==1:value=value[...,::-1]
                            if task%3==2:value=cp.broadcast_to(value[:1],shape)
                            pair.append(value)
                        tasks.append(tuple(pair))
                    bound=bind_shared(equation,shapes,dtype,(_allowance(shapes,dtype),)*count,cp)
                    wide=np.dtype('complex128' if dtype==np.complex64 else 'float64')
                    partials=cp.empty((count,bound.blocks),dtype=wide)
                    device=SimpleNamespace(backend=cp,reduction_partial=lambda n,d:partials)
                    with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
                        outputs=bound(tasks)
                    for task,(values,result) in enumerate(zip(tasks,outputs,strict=True)):
                        first,finish,_=reduction_sources(equation.inputs,(),shapes,tuple(v.strides for v in values),np.dtype(dtype),bound.blocks)
                        original=cp.empty(bound.blocks,dtype=wide);reference=cp.empty((),dtype=dtype)
                        _kernel(cp.RawKernel,first,'ed_precise_partial')((bound.blocks,),(256,),(*values,original))
                        _kernel(cp.RawKernel,finish,'ed_precise_finish')((1,),(256,),(original,reference))
                        self.assertEqual(cp.asnumpy(partials[task]).tobytes(),cp.asnumpy(original).tobytes())
                        self.assertEqual(cp.asnumpy(result).tobytes(),cp.asnumpy(reference).tobytes())
                    self.assertEqual(len(set(out.data.ptr for out in outputs)),count)


if __name__=='__main__':unittest.main()
