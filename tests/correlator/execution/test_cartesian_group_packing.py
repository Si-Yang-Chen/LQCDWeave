"""Combine compatible child copies using the existing original-pointer packer."""
from dataclasses import replace
from types import SimpleNamespace
import gc
import os
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec, device_allocation_bytes
from lqcdweave.contraction.execution import cartesian_group_execution as module
from lqcdweave.contraction.execution import scene_runtime, bound_program, matrix_reduction_group
from lqcdweave.contraction.execution.operations import packing_kernel
from lqcdweave.contraction.execution.operations.dense_batch import _BatchSide
from lqcdweave.contraction.execution.operations.result_views import _DeferredResultView,_ResultOwner,bind_result_views
from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from correlator.execution.test_cartesian_group_execution import fixture,Backend as MatrixBackend
from correlator.execution.test_mixed_stride_stack_packing import _Kernel as PackingKernel


class Backend(MatrixBackend):
    def __init__(self):
        super().__init__()
        self.packing=SimpleNamespace(calls=[],fail=False)
        self.pack_sources=[];self.allocations=[]
    def ElementwiseKernel(self,*args,**kwargs):raise AssertionError('raw stack must remain eligible')
    def RawKernel(self,source,name,**kwargs):
        if name=='ed_dense_batch_pack_raw':
            self.pack_sources.append(source)
            return PackingKernel(source,self.packing)
        return super().RawKernel(source,name,**kwargs)
    def empty(self,shape,dtype,**kwargs):
        value=super().empty(shape,dtype,**kwargs)
        self.allocations.append((tuple(shape),np.dtype(dtype),weakref.ref(value)))
        return value


def bind(plan,backend):
    with patch.object(bound_program,'bind_output_layout',return_value=None):
        tables=scene_runtime.bind_scene_execution(plan.state,plan.kernel_batches,backend,
            events=(),cartesian_reduction_groups=plan.cartesian_reduction_groups)
    return next(iter(tables.cartesian_starts.values()))


def results(bound,arrays):
    return tuple(arrays[result.tensor] for batch in bound.batches for result in batch.results)


class CartesianGroupPackingTests(unittest.TestCase):
    def tearDown(self):
        module._FAILED_SUBMISSIONS.clear()
        packing_kernel._raw_kernel.cache_clear()

    def test_2_4_8_groups_two_packs_original_bytes_fresh_signed_zero_inputs(self):
        for count in (2,4,8):
            for dtype in (np.float32,np.complex64):
                plan,inputs=fixture(count,k=80,dtype=dtype);backend=Backend();group=bind(plan,backend)
                self.assertEqual(len(group.packing),2)
                for phase in (0,1):
                    values={key:value*(phase+1) for key,value in inputs.items()}
                    if phase:
                        values={key:value[::-1] if i%2 else np.broadcast_to(value[:1],value.shape)
                                for i,(key,value) in enumerate(values.items())}
                    expected=dict(values);replace(group,packing=()).execute(expected)
                    previous=len(backend.packing.calls);reductions=len(backend.launches)
                    actual=dict(values);group.execute(actual)
                    self.assertEqual(len(backend.packing.calls)-previous,2)
                    self.assertEqual(len(backend.launches)-reductions,2)
                    self.assertEqual(tuple(v.tobytes() for v in results(group,actual)),
                                     tuple(v.tobytes() for v in results(group,expected)))
                for axis,combined in group.packing:
                    total=combined.shape[0]*np.prod(combined.shape[1:])*np.dtype(dtype).itemsize
                    old=sum(device_allocation_bytes(int(np.prod(group.layouts[i].sides[axis].shape)*np.dtype(dtype).itemsize))
                            for i in combined.members)
                    self.assertLessEqual(device_allocation_bytes(int(total)),old)

    def test_prepared_side_mixed_children_and_disjoint_shared_outputs(self):
        plan,inputs=fixture(4,m=1,n=3,k=80,prepared_side='a');backend=Backend();group=bind(plan,backend)
        self.assertEqual([axis for axis,_ in group.packing],[1])
        arrays=dict(inputs);group.execute(arrays)
        self.assertEqual(len(backend.packing.calls),1)
        # One prepared left is changed only in this internal test descriptor;
        # every other left remains unprepared, and the prepared row stays borrowed.
        plan,inputs=fixture(4,m=1,n=3,k=80);group=bind(plan,backend)
        first=group.layouts[0]
        prepared=replace(first.sides[0],prepared=True)
        layouts=(replace(first,sides=(prepared,first.sides[1])),*group.layouts[1:])
        packing=module._bind_combined_packing(layouts,plan.state.catalog,backend)
        mixed=replace(group,layouts=layouts,packing=packing)
        self.assertEqual(dict(packing)[0].members,(1,2,3))
        tasks=[];original=mixed.kernel
        def capture(values):
            tasks.extend(values);return original(values)
        arrays=dict(inputs);replace(mixed,kernel=capture).execute(arrays)
        self.assertTrue(np.shares_memory(tasks[0][0],inputs['0a0']))
        roots=[]
        for batch in mixed.batches:
            owner=arrays[batch.results[0].tensor]
            while isinstance(owner.base,np.ndarray):owner=owner.base
            roots.append(owner)
        self.assertEqual(len({id(root) for root in roots}),1)
        results=[arrays[result.tensor] for batch in mixed.batches for result in batch.results]
        self.assertEqual(len({value.ctypes.data for value in results}),len(results))
        self.assertTrue(np.shares_memory(tasks[1][0],tasks[3][0].base))

    def test_multiple_pointer_chunks_keep_offsets_bytes_and_owner_budget(self):
        for m,n in ((5,3),(8,8)):
            for dtype in (np.float32,np.complex64):
                plan,inputs=fixture(8,m=m,n=n,k=128,dtype=dtype)
                inputs={key:value+np.asarray((index+1)/16,dtype=dtype)
                        for index,(key,value) in enumerate(inputs.items())}
                backend=Backend();group=bind(plan,backend)
                self.assertEqual(len(group.packing),2)
                for phase in (0,1):
                    values={key:value*(phase+1) for key,value in inputs.items()}
                    if phase:
                        values={key:value[::-1] if i%2 else np.broadcast_to(value[:1],value.shape)
                                for i,(key,value) in enumerate(values.items())}
                    expected=dict(values);replace(group,packing=()).execute(expected)
                    before=len(backend.packing.calls);actual=dict(values);group.execute(actual)
                    calls=backend.packing.calls[before:]
                    self.assertEqual(len(calls),(8*m+31)//32+(8*n+31)//32)
                    self.assertTrue(all(grid[1]<=32 for grid,_,_ in calls))
                    self.assertEqual(tuple(v.tobytes() for v in results(group,actual)),
                                     tuple(v.tobytes() for v in results(group,expected)))
                for axis,combined in group.packing:
                    payload=int(np.prod(combined.shape))*np.dtype(dtype).itemsize
                    original=sum(device_allocation_bytes(int(np.prod(group.layouts[i].sides[axis].shape))*np.dtype(dtype).itemsize)
                                 for i in combined.members)
                    self.assertLessEqual(device_allocation_bytes(payload),original)

    def test_pointer_only_inputs_combined_owner_borrow_and_range(self):
        plan,inputs=fixture(4,k=80);backend=Backend();group=bind(plan,backend)
        binding={key:index for index,key in enumerate(inputs)};store={}
        for axis,combined in group.packing:
            owner=np.stack([inputs[key] for child in combined.members
                for key,_ in group.layouts[child].sides[axis].operands])
            offset=0
            for child in combined.members:
                for key,_ in group.layouts[child].sides[axis].operands:
                    layout=bind_result_views(owner.shape,owner.dtype,((key,(offset,slice(None)),()),))[0][1]
                    store[binding[key]]=_DeferredResultView(_ResultOwner(owner),layout);offset+=1
            before=len(backend.packing.calls)
            packed=combined.pack(_NodeView(binding,store),backend)
            self.assertEqual(len(backend.packing.calls),before)
            self.assertTrue(np.shares_memory(packed,owner))
            self.assertEqual(packed.nbytes,owner.nbytes)
        # Different unrelated child owners may require the one combined copy;
        # no dynamic subset or changed kernel plan is selected.

    def test_full_inner_geometry_permutations_and_binding_limits(self):
        plan,inputs=fixture(2,k=80);backend=Backend();group=bind(plan,backend)
        catalog=dict(plan.state.catalog);layouts=[];values={}
        for child,layout in enumerate(group.layouts):
            sides=[]
            for side in layout.sides:
                for key,_ in side.operands:
                    catalog[key]=TensorSpec((10,8),np.complex64)
                    values[key]=inputs[key].reshape(10,8)
                operands=tuple((key,(1,0)) for key,_ in side.operands)
                input_layouts=((None,1,0),)*len(operands)
                packer=packing_kernel.bind_packing(backend,((1,8,10),)*len(operands),
                    (np.complex64,)*len(operands),0,np.complex64,input_layouts=input_layouts)
                sides.append(replace(side,shape=(len(operands),8,10),operands=operands,
                    input_layouts=input_layouts,packer=packer,original_packer=packer.pack_original))
            layouts.append(replace(layout,sides=tuple(sides)))
        combined=module._bind_combined_packing(layouts,catalog,backend)
        self.assertEqual(len(combined),2)
        for axis,side in combined:
            expected=np.concatenate([layout.sides[axis].pack(values,backend,np.dtype('complex64')) for layout in layouts])
            self.assertEqual(side.pack(values,backend).tobytes(),expected.tobytes())
        reference=dict(values);replace(group,layouts=tuple(layouts),packing=()).execute(reference)
        actual=dict(values);replace(group,layouts=tuple(layouts),packing=combined).execute(actual)
        self.assertEqual(tuple(v.tobytes() for v in results(group,actual)),
                         tuple(v.tobytes() for v in results(group,reference)))
        # Equal flat K is insufficient if the inner dimensions differ.
        changed=replace(layouts[1].sides[0],shape=(2,4,20))
        mismatch=(layouts[0],replace(layouts[1],sides=(changed,layouts[1].sides[1])))
        self.assertEqual([axis for axis,_ in module._bind_combined_packing(mismatch,catalog,backend)],[1])
        # Crossing32 pointers uses two bounded chunks in the same owner.
        wide_plan,wide_inputs=fixture(8,m=5,n=2,k=80);wide=bind(wide_plan,backend)
        self.assertEqual([axis for axis,_ in wide.packing],[0,1])
        reference=dict(wide_inputs);replace(wide,packing=()).execute(reference)
        before=len(backend.packing.calls);actual=dict(wide_inputs);wide.execute(actual)
        self.assertEqual(len(backend.packing.calls)-before,3)
        self.assertEqual(tuple(v.tobytes() for v in results(wide,actual)),
                         tuple(v.tobytes() for v in results(wide,reference)))
        # Worst-case mixed-stride ABI beyond3KiB is a bind-time rejection.
        oversized=replace(layouts[0].sides[0],shape=(2,*(2,)*11))
        high=tuple(replace(layout,sides=(oversized,layout.sides[1])) for layout in layouts)
        self.assertEqual([axis for axis,_ in module._bind_combined_packing(high,catalog,backend)],[1])
        with patch.object(module,'bind_packing',return_value=None):
            self.assertEqual(module._bind_combined_packing(layouts,catalog,backend),())

    def test_metadata_failure_and_launch_failure_keep_owner_without_retry(self):
        plan,inputs=fixture(2,k=80);backend=Backend();group=bind(plan,backend)
        for value in (np.empty((79,),np.complex64),np.empty((80,),np.complex128)):
            bad=dict(inputs);bad['0a0']=value
            with self.assertRaisesRegex(ValueError,'saved shape/dtype'):group.execute(bad)
            self.assertEqual(backend.packing.calls,[])
        backend.packing.fail=True;backend.synchronize.side_effect=RuntimeError('sync failed')
        pool=object()
        with patch('lqcdweave.contraction.execution.device.active_device_execution',
                   return_value=SimpleNamespace(backend=backend,pool=pool)):
            with self.assertRaisesRegex(RuntimeError,'packing launch failed'):group.execute(dict(inputs))
        gc.collect()
        self.assertEqual(len(backend.packing.calls),1);self.assertEqual(backend.launches,[])
        self.assertIs(module._FAILED_SUBMISSIONS[-1][-2],pool)
        # The failed pack allocation has not returned yet, but was registered
        # explicitly before submission; retention does not rely on traceback.
        self.assertTrue(any(ref() is not None for _,_,ref in backend.allocations))


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE')=='1','explicit CUDA gate; CPU suite does not probe device')
class CartesianGroupPackingCuPyTests(unittest.TestCase):
    def test_combined_packs_launch_counts_bytes_owner_ranges_and_budget(self):
        import cupy as cp
        from lqcdweave.contraction.execution.raw_arguments import bind_pointer_arguments
        rows=[]
        geometries=((2,2,3),(4,2,3),(8,2,3),(8,5,3),(8,8,8))
        for count,m,n,dtype in ((count,m,n,dtype) for count,m,n in geometries
                                for dtype in (np.float32,np.complex64)):
            plan,inputs=fixture(count,m=m,n=n,k=800,dtype=dtype);submissions=[]
            # Distinguish every source and child: repeated fixture values could
            # otherwise hide an incorrect cross-chunk pointer/task mapping.
            inputs={key:value+np.asarray((index+1)/16,dtype=dtype)
                    for index,(key,value) in enumerate(inputs.items())}
            def counted(*args,**kwargs):
                call=bind_pointer_arguments(*args,**kwargs)
                if call is None:return None
                name=args[1].name
                def submit(*values):
                    submissions.append(name)
                    return call(*values)
                return submit
            with patch.object(packing_kernel,'bind_pointer_arguments',side_effect=counted),patch.object(
                    matrix_reduction_group,'bind_pointer_arguments',side_effect=counted):
                group=bind(plan,cp)
                self.assertEqual(len(group.packing),2)
                for phase in (0,1):
                    values={key:cp.asarray(value*(phase+1)) for key,value in inputs.items()}
                    if phase:
                        values={key:value[::-1] if i%2 else cp.broadcast_to(value[:1],value.shape)
                                for i,(key,value) in enumerate(values.items())}
                    expected=dict(values);replace(group,packing=()).execute(expected)
                    cp.cuda.get_current_stream().synchronize();submissions.clear()
                    tasks=[];original=group.kernel
                    def capture(values):tasks.extend(values);return original(values)
                    actual=dict(values);replace(group,kernel=capture).execute(actual)
                    cp.cuda.get_current_stream().synchronize()
                    packs=sum(name=='ed_dense_batch_pack_raw' for name in submissions)
                    reductions=sum(name.startswith('ed_precise_matrix_group_') for name in submissions)
                    self.assertEqual(packs,(count*m+31)//32+(count*n+31)//32)
                    self.assertEqual(reductions,2)
                    self.assertEqual(tuple(cp.asnumpy(v).tobytes() for v in results(group,actual)),
                                     tuple(cp.asnumpy(v).tobytes() for v in results(group,expected)))
                    for axis,combined in group.packing:
                        views=[tasks[i][axis] for i in combined.members]
                        start=views[0].data.ptr;step=views[0].nbytes
                        self.assertEqual([view.data.ptr-start for view in views],[i*step for i in range(count)])
                        total=count*step
                        self.assertLessEqual(device_allocation_bytes(total),sum(device_allocation_bytes(view.nbytes) for view in views))
                        owner=views[0].base
                        self.assertIsNotNone(owner)
                        self.assertTrue(all(view.base is owner for view in views))
                        self.assertGreaterEqual(start,owner.data.ptr)
                        self.assertLessEqual(start+total,owner.data.ptr+owner.nbytes)
                    rows.append(dict(count=count,m=m,n=n,dtype=np.dtype(dtype).name,phase=phase,packing_submissions=packs,
                        reduction_submissions=reductions,bitwise_equal=True))
        import json
        print(json.dumps(dict(cartesian_group_packing_cuda=rows)),flush=True)


if __name__=='__main__':unittest.main()
