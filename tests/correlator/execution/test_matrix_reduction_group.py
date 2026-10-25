"""Unintegrated matrix grouping: CPU source/ABI model and explicit CUDA gate."""
import ctypes
import os
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contraction.execution import matrix_reduction_group as module
from lqcdweave.contraction.execution import reduction_group_runtime as runtime
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.reduction_kernel import (
    _kernel, _wide_parts, bind_raw_reduction, tiled_matrix_reduction_source,
)
from lqcdweave.contracts.correlator.api import parse_axis_pattern
from correlator.execution.test_scalar_reduction_group import _SourceBackend, _tree, _reference


class MatrixBackend(_SourceBackend):
    def RawKernel(self,source,name,**options):
        assert options['options']==('--std=c++17','--fmad=false')
        self.sources.append((source,name))
        def kernel(grid,threads,args):
            self.launches.append((grid,threads,name))
            if self.fail_at==len(self.launches):raise RuntimeError('submission failed')
            if self.metadata_only:return
            if name.endswith('_partial'):
                reciprocal='const unsigned long long magic' in source
                if reciprocal:k,m,n,blocks,magic=map(int,args[-6:-1])
                else:k,m,n,blocks=map(int,args[-5:-1])
                tile_rows=(m+1)//2
                count=grid[2]//tile_rows
                partials=args[-1].reshape(-1)[:count*m*n*blocks].reshape(count,m,n,blocks)
                expressions=re.findall(r'p([01]) \+ (row \* lm \+ i \* lk(?: \+ lm)?|i \* rk \+ col \* rn(?: \+ rn)?)\)',source)
                assert len(expressions)==4
                compiled=[(int(side),compile(expr,'<matrix load>','eval')) for side,expr in expressions]
                if reciprocal:
                    assert 'const unsigned int tile_rows = (static_cast<unsigned int>(m) + 1U) >> 1;' in source
                    assert 'const unsigned int task = (static_cast<unsigned long long>(blockIdx.z) * magic) >> 32;' in source
                    assert 'row = 2LL * (blockIdx.z - task * tile_rows)' in source
                else:
                    # Interpret the prior d3d/8ce generated ABI for external
                    # same-process baseline comparisons without changing it.
                    assert 'task = blockIdx.z / tile_rows;' in source
                    assert 'row = 2LL * (blockIdx.z % tile_rows)' in source
                assert 'partials += task * m * n * blocks' in source
                assert 'i = split * 256LL + t; i < k; i += blocks * 256LL' in source
                # Interpret emitted address expressions with the actual host ABI,
                # then perform every original lane chain and warp tree.
                for z in range(grid[2]):
                    if reciprocal:
                        task=np.uint32((np.uint64(z)*np.uint64(magic))>>np.uint64(32))
                        remainder=np.uint32(z)-task*np.uint32(tile_rows)
                    else:task,remainder=divmod(np.uint32(z),np.uint32(tile_rows))
                    assert (int(task),int(remainder))==divmod(z,tile_rows)
                    task,row=int(task),2*int(remainder)
                    a,b=args[task*6:task*6+2]
                    lm,lk,rn,rk=map(int,args[task*6+2:task*6+6])
                    for col_tile in range(grid[1]):
                        col=col_tile*2
                        terms=np.zeros((2,2,k),dtype=partials.dtype)
                        for i in range(k):
                            loaded=[]
                            for load,(side,expr) in enumerate(compiled):
                                valid=(row+1<m if load==1 else col+1<n if load==3 else True)
                                value=(a,b)[side]
                                offset=eval(expr,{'__builtins__':{}},dict(row=row,col=col,i=i,lm=lm,lk=lk,rn=rn,rk=rk))
                                raw=ctypes.string_at(value.ctypes.data+offset,value.itemsize) if valid else None
                                loaded.append(partials.dtype.type(np.frombuffer(raw,dtype=value.dtype)[0]) if valid else 0)
                            for r in range(2):
                                for c in range(2):terms[r,c,i]=loaded[r]*loaded[2+c]
                        for split in range(grid[0]):
                            for r in range(min(2,m-row)):
                                for c in range(min(2,n-col)):
                                    lanes=np.zeros(256,dtype=partials.dtype)
                                    for start in range(split*256,k,blocks*256):
                                        take=terms[r,c,start:start+256];lanes[:take.size]+=take
                                    partials[task,row+r,col+c,split]=_tree(lanes)
                self.partial_snapshot=partials.copy()
            else:
                blocks,outputs=map(int,args[-3:-1]);count=grid[1]
                assert grid[0]==outputs
                assert 'partials += task * outputs * blocks' in source
                partials=args[0].reshape(-1)[:count*outputs*blocks].reshape(count,outputs,blocks)
                results=(args[-1],*args[1:count])
                for task,result in enumerate(results):
                    for output in range(outputs):
                        lanes=np.zeros(256,dtype=partials.dtype)
                        for start in range(0,blocks,256):
                            take=partials[task,output,start:start+256];lanes[:take.size]+=take
                        result.flat[output]=_tree(lanes)
        return kernel


def tasks_for(m,k,n,dtype,count):
    rng=np.random.default_rng(812)
    tasks=[]
    for task in range(count):
        values=[]
        for rows in (m,n):
            value=rng.normal(size=(rows,k*2)).astype(dtype)
            if np.dtype(dtype).kind=='c':value+=1j*rng.normal(size=value.shape).astype(np.float32)
            value=value[:,::2]  # A noncontiguous positive stride is also real input.
            if task%3==1:value=value[::-1,::-1]
            if task%3==2:value=np.broadcast_to(value[:1,:],value.shape)
            values.append(value)
        tasks.append(tuple(values))
    return tasks


def bind_shared(*args, **kwargs):
    return module.bind_matrix_reduction_group(*args, **kwargs, share_small_outputs=True)


class MatrixReductionGroupTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear();runtime.FAILED_SUBMISSIONS.clear()

    def test_partial_and_final_match_independent_wide_tree_edges_strides_and_budgets(self):
        for dtype in (np.float32,np.complex64):
            wide=np.dtype('complex128' if dtype==np.complex64 else 'float64')
            for m,k,n in ((1,513,1),(1,37,3),(3,37,1),(3,513,5),(8,37,8)):
                for count in (2,4,8):
                    for blocks in sorted({1,(k+255)//256}):
                        tasks=tasks_for(m,k,n,dtype,count);backend=MatrixBackend()
                        allowance=m*n*(blocks*wide.itemsize+np.dtype(dtype).itemsize)
                        run=bind_shared(((m,k),(n,k)),dtype,(allowance,)*count,backend)
                        self.assertEqual(run.blocks,blocks)
                        outputs=run(tasks)
                        self.assertEqual(backend.launches[0][0],(blocks,(n+1)//2,count*((m+1)//2)))
                        self.assertEqual(backend.launches[1][0],(m*n,count))
                        self.assertEqual(run.parameter_bytes,(432,88))
                        self.assertEqual(run.workspace_bytes,count*allowance)
                        self.assertEqual(len(backend.allocations),run.output_owner_count+1)
                        self.assertEqual(len({out.ctypes.data for out in outputs}),count)
                        self.assertTrue(all(out.shape==(m,n) for out in outputs))
                        for task,((a,b),out) in enumerate(zip(tasks,outputs,strict=True)):
                            for row in range(m):
                                for col in range(n):
                                    partial,expected=_reference(a[row],b[col],((0,),(0,)),blocks)
                                    self.assertEqual(backend.partial_snapshot[task,row,col].tobytes(),partial.tobytes())
                                    self.assertEqual(out[row,col].tobytes(),expected.tobytes())
                            np.testing.assert_allclose(out,np.einsum('ik,jk->ij',a,b,dtype=wide).astype(dtype),rtol=2e-6,atol=2e-6)

    def test_source_is_dtype_shared_and_keeps_production_arithmetic_body(self):
        for dtype in ('float32','complex64'):
            first,second=module.matrix_group_sources(dtype)
            original=tiled_matrix_reduction_source(dtype)
            self.assertIn(original[original.index('    const bool has_row'):],first)
            self.assertEqual(first.count('__syncthreads();'),1)
            self.assertIn(_wide_parts(dtype)[5],second)
            self.assertEqual(first.count('switch (task)'),1)
        backend=MatrixBackend(metadata_only=True)
        for m,k,n,count,blocks in ((1,513,1,2,1),(3,1025,5,4,5),(2,257,3,8,2)):
            allowance=m*n*(blocks*16+8)
            run=bind_shared(((m,k),(n,k)),np.complex64,(allowance,)*count,backend)
            run(tasks_for(m,k,n,np.complex64,count))
        self.assertEqual(len(backend.sources),2)

    def test_reciprocal_decode_matches_divmod_at_every_admitted_row_size(self):
        # Every admitted m/count, every child and extreme/middle remainders.
        # In particular d=1 needs magic=2**32, not a truncated uint32 zero.
        for count in range(2,9):
            sizes=np.arange(1,65536,dtype=np.int64)
            sizes=sizes[count*((sizes+1)//2)<=65535]
            rows=(sizes+1)//2
            magic=((1<<32)+rows-1)//rows
            self.assertEqual(int(magic[0]),1<<32)
            for child in range(count):
                for remainder in (np.zeros_like(rows),np.minimum(1,rows-1),rows//2,np.maximum(0,rows-2),rows-1):
                    z=child*rows+remainder
                    old_task,old_tile=np.divmod(z,rows)
                    task=(z.astype(np.uint64)*magic.astype(np.uint64))>>np.uint64(32)
                    tile=z.astype(np.uint32)-task.astype(np.uint32)*rows.astype(np.uint32)
                    np.testing.assert_array_equal(task,old_task)
                    np.testing.assert_array_equal(tile,old_tile)
                    self.assertTrue(np.all(z*(rows-1)<1<<32))
                    self.assertTrue(np.all(2*tile<sizes))
        first,_=module.matrix_group_sources('complex64')
        self.assertIn('const long long row = 2LL * (blockIdx.z - task * tile_rows)',first)
        self.assertIn('partials += task * m * n * blocks;',first)
        self.assertIn('const long long m, const long long n,',first)
        prologue=first[first.index('    const int t'):first.index('    const bool has_row')]
        self.assertNotIn('/',prologue);self.assertNotIn('%',prologue)

    def test_prior_division_abi_is_still_interpreted_for_byte_comparison(self):
        for dtype in (np.float32,np.complex64):
            first,finish=module.matrix_group_sources(np.dtype(dtype).name)
            previous=first.replace('const long long blocks, const unsigned long long magic,',
                'const long long blocks,').replace(
                '(static_cast<unsigned int>(m) + 1U) >> 1', '(static_cast<unsigned int>(m) + 1U) / 2U').replace(
                '(static_cast<unsigned long long>(blockIdx.z) * magic) >> 32','blockIdx.z / tile_rows').replace(
                '(blockIdx.z - task * tile_rows)','(blockIdx.z % tile_rows)')
            for m,n in ((1,1),(3,5),(8,8)):
                k,blocks,count=37,1,4
                allowance=m*n*(16 if dtype==np.complex64 else 8)+m*n*np.dtype(dtype).itemsize
                backend=MatrixBackend();tasks=tasks_for(m,k,n,dtype,count)
                current=bind_shared(((m,k),(n,k)),dtype,(allowance,)*count,backend)
                actual=current(tasks);partial=backend.partial_snapshot.copy()
                original=runtime.bind_group_runtime(backend,((m,k),(n,k)),np.dtype(dtype),(allowance,)*count,
                    blocks,(m,n),(previous,finish),
                    ('ed_precise_matrix_group_partial','ed_precise_matrix_group_finish'),
                    ((blocks,(n+1)//2,count*((m+1)//2)),(m*n,count)),
                    ((k,m,n,blocks),(blocks,m*n)),module.bind_pointer_arguments)
                expected=original(tasks)
                self.assertEqual(original.parameter_bytes,(424,88))
                self.assertEqual(partial.tobytes(),backend.partial_snapshot.tobytes())
                self.assertEqual(tuple(x.tobytes() for x in actual),tuple(x.tobytes() for x in expected))

    def test_rejects_unrepresentable_geometry_budget_and_current_metadata(self):
        backend=MatrixBackend(metadata_only=True)
        bind=lambda shapes=((3,513),(5,513)),allowances=(10000,)*2,dtype=np.complex64,b=backend:bind_shared(shapes,dtype,allowances,b)
        for shapes in (((3,513),(5,512)),((3,0),(5,0)),((3,513),),((3,1,513),(5,513)),
                       ((65536,513),(1,513)),((20000,513),(1,513)),((16000,230400),(16000,230400)),
                       ((1,2**63),(1,2**63)),((3,True),(5,True))):
            self.assertIsNone(bind(shapes=shapes,allowances=(1<<50,)*8))
        for allowances in ((0,0),(240,360),(360,600),(10000,),(10000,)*9,(-1,-1)):
            self.assertIsNone(bind(allowances=allowances))
        self.assertIsNone(bind(dtype=np.float64));self.assertIsNone(bind(b=np))
        with execution_debug(True):self.assertIsNone(bind())
        backend.cuda.runtime.is_hip=True;self.assertIsNone(bind());backend.cuda.runtime.is_hip=False
        run=bind();a,b=tasks_for(3,513,5,np.complex64,1)[0]
        for tasks in (((a,b),),((a,b,b),)*2,((a.astype(np.complex128),b),)*2,((a[:-1],b),)*2):
            with self.assertRaises(ValueError):run(tasks)
        invalid=SimpleNamespace(shape=a.shape,dtype=a.dtype,strides=(2**63,8),data=SimpleNamespace(ptr=123))
        with self.assertRaisesRegex(ValueError,'strides'):run(((invalid,b),)*2)
        invalid.strides=a.strides;invalid.data.ptr=0
        with self.assertRaisesRegex(ValueError,'pointer'):run(((invalid,b),)*2)
        backend.ndarray=type('OnlyDevice',(),{})
        with self.assertRaisesRegex(ValueError,'bound array backend'):bind()(((a,b),)*2)
        self.assertEqual(backend.launches,[])

    def test_arena_fallback_fresh_pointers_lru_and_failed_pool_owners(self):
        backend=MatrixBackend(metadata_only=True);shapes=((3,513),(5,513));tasks=tasks_for(3,513,5,np.complex64,4)
        run=bind_shared(shapes,np.complex64,(10000,)*4,backend)
        scratch=np.empty(run.partial_bytes//16,np.complex128);borrow=Mock(return_value=scratch);pool=object()
        device=SimpleNamespace(backend=backend,reduction_partial=borrow,pool=pool)
        with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
            outputs=run(tasks)
            borrow.assert_called_once_with(run.partial_bytes,np.dtype('complex128'))
            self.assertEqual(len(backend.allocations),1)
            borrow.return_value=None;run(tasks);self.assertEqual(len(backend.allocations),3)
            backend.fail_at=len(backend.launches)+2
            backend.synchronize.side_effect=RuntimeError('sync failed')
            with self.assertRaisesRegex(RuntimeError,'submission failed'):run(tasks)
        held=runtime.FAILED_SUBMISSIONS[0]
        self.assertIs(held[4],pool);self.assertIs(held[1][0],tasks[0][0]);self.assertEqual(len(held[3]),4)
        runtime.FAILED_SUBMISSIONS.clear();backend.synchronize.side_effect=None;backend.fail_at=None
        backend.fail_allocation=len(backend.allocations)+2
        with self.assertRaisesRegex(MemoryError,'allocation failed'):run(tasks)
        self.assertGreaterEqual(backend.synchronize.call_count,2)
        backend.fail_allocation=None
        calls=[];bindings=[]
        def binder(backend,kernel,grid,block,template,positions,*,pointer_getter):
            bindings.append(grid)
            def launch(inputs,result):
                words=list(template)
                for pos,index in positions:words[pos]=pointer_getter(inputs[index])
                words[-1]=runtime.current_pointer(result);calls.append(words)
            return launch
        with patch.object(module,'bind_pointer_arguments',side_effect=binder):
            current=bind_shared(shapes,np.complex64,(10000,)*4,backend)
        old=current(tasks);new_tasks=tasks_for(3,513,5,np.complex64,4);new=current(new_tasks)
        self.assertEqual(calls[0][0],tasks[0][0].ctypes.data)
        self.assertEqual(calls[2][0],new_tasks[0][0].ctypes.data)
        self.assertNotEqual(calls[0][0],calls[2][0])
        self.assertEqual(calls[1][-1],old[0].ctypes.data)
        self.assertEqual(calls[3][-1],new[0].ctypes.data)
        refs=[weakref.ref(x) for pair in new_tasks for x in pair]+[weakref.ref(x) for x in new]
        del new_tasks,new
        self.assertTrue(all(ref() is None for ref in refs))
        # More than 64 layouts stays bounded and never recompiles dtype source.
        b=np.ones((5,513),np.complex64)
        for step in range(1,67):
            a=np.ones((3,513*step),np.complex64)[:,::step]
            current(((a,b),)*4)
        bound_count=len(bindings);current(tasks)
        self.assertEqual(len(bindings),bound_count+2) # The old layout was evicted.
        self.assertEqual(len(backend.sources),2)

    def test_wide_cancellation_is_retained_in_all_tile_lanes(self):
        for dtype in (np.float32,np.complex64):
            a=np.tile(np.array([128.125,128.125,-128.125,-128.125],dtype),(3,256))
            a[:,0]=np.nextafter(np.float32(128.125),np.float32(np.inf))
            b=np.full((5,1024),1.125+(0.25j if dtype==np.complex64 else 0),dtype=dtype)
            wide=np.complex128 if dtype==np.complex64 else np.float64
            expected=np.einsum('ik,jk->ij',a,b,dtype=wide).astype(dtype)
            self.assertTrue(np.all(expected!=0))
            run=bind_shared((a.shape,b.shape),dtype,(2000,)*2,MatrixBackend())
            for out in run(((a,b),)*2):np.testing.assert_array_equal(out,expected)

    def test_request_over_arena_capacity_keeps_bounded_independent_allocation(self):
        backend=MatrixBackend(metadata_only=True);m,n,k,blocks,count=16,16,230400,900,8
        allowance=m*n*(blocks*16+8)
        run=bind_shared(((m,k),(n,k)),np.complex64,(allowance,)*count,backend)
        self.assertGreater(run.partial_bytes,8*1024**2)
        a=SimpleNamespace(shape=(m,k),dtype=np.dtype('complex64'),strides=(k*8,8),data=SimpleNamespace(ptr=123))
        b=SimpleNamespace(shape=(n,k),dtype=a.dtype,strides=a.strides,data=SimpleNamespace(ptr=456))
        borrow=Mock(side_effect=lambda size,dtype:None if size>8*1024**2 else self.fail('unexpected fit'))
        with patch('lqcdweave.contraction.execution.device.active_device_execution',
                   return_value=SimpleNamespace(backend=backend,reduction_partial=borrow)):
            outputs=run(((a,b),)*count)
        borrow.assert_called_once_with(run.partial_bytes,np.dtype('complex128'))
        self.assertEqual(len(backend.allocations),run.output_owner_count+1)
        self.assertEqual(run.workspace_bytes,count*allowance)
        self.assertTrue(all(value.shape==(m,n) for value in outputs))


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE')=='1',
                     'explicit CUDA gate; no device probe in CPU suite')
class MatrixReductionGroupCuPyTests(unittest.TestCase):
    def test_group_partial_final_bytes_match_production_raw_and_owner_guards(self):
        import cupy as cp
        equation=parse_axis_pattern('ik,jk->ij')
        for dtype in (np.float32,np.complex64):
            wide=np.dtype('complex128' if dtype==np.complex64 else 'float64')
            for m,k,n in ((1,513,1),(1,513,5),(3,513,1),(3,1025,5),(2,134400,3),(3,230400,2)):
                for count in (2,4,8):
                    host=tasks_for(m,k,n,dtype,count);tasks=[]
                    for task,(a,b) in enumerate(host):
                        pair=[cp.asarray(value) for value in (a,b)]
                        if task%3==1:pair=[value[::-1,::-1] for value in pair]
                        if task%3==2:pair=[cp.broadcast_to(value[:1],value.shape) for value in pair]
                        tasks.append(tuple(pair))
                    for blocks in sorted({1,min(1024,(k+255)//256)}):
                        allowance=m*n*(blocks*wide.itemsize+np.dtype(dtype).itemsize)
                        run=bind_shared(((m,k),(n,k)),dtype,(allowance,)*count,cp)
                        owner=cp.full(count*m*n*blocks+2,19,dtype=wide);partial=owner[1:-1]
                        device=SimpleNamespace(backend=cp,reduction_partial=lambda n,d:partial)
                        with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
                            outputs=run(tasks)
                        self.assertEqual(len({out.data.ptr for out in outputs}),count)
                        for task,pair in enumerate(tasks):
                            reference_partial=cp.empty((m*n,blocks),dtype=wide)
                            reference_device=SimpleNamespace(backend=cp,reduction_partial=lambda n,d:reference_partial)
                            with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=reference_device):
                                original=bind_raw_reduction(equation,tuple(x.shape for x in pair),np.dtype(dtype),cp,allowance)
                                expected=original(*pair)
                            actual=partial.reshape(count,m*n,blocks)[task]
                            self.assertEqual(cp.asnumpy(actual).tobytes(),cp.asnumpy(reference_partial).tobytes())
                            self.assertEqual(cp.asnumpy(outputs[task]).tobytes(),cp.asnumpy(expected).tobytes())
                        np.testing.assert_array_equal(cp.asnumpy(owner[[0,-1]]),[19,19])


if __name__=='__main__':unittest.main()
