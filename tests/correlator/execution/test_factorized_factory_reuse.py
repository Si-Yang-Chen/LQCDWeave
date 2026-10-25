"""Exact immutable FWR factories share metadata, never Actions or values."""
from dataclasses import replace
import gc
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contraction.execution.scene_runtime import bind_scene_execution
from lqcdweave.contraction.execution.factorized_reduction import bind_factorized_reduction
from lqcdweave.contraction.execution.diagnostics import execution_debug
from .test_factorized_wide_reduction import tiled_fixture


def paired():
    _,completed,action,_=tiled_fixture(tile=4)
    names=tuple('other:'+name for name in action.operands)
    second=replace(action,operands=names,tensor='other:out',core=replace(action.core))
    catalog=dict(completed.catalog)
    catalog.update((name,catalog[old]) for name,old in zip(names,action.operands,strict=True))
    catalog[second.tensor]=catalog[action.tensor]
    return SimpleNamespace(actions=(action,second),catalog=catalog)


class FactorizedFactoryReuseTests(unittest.TestCase):
    def test_equal_cores_share_factory_with_current_values_and_separate_owners(self):
        state=paired();backend=np
        self.assertIsNot(state.actions[0].core,state.actions[1].core)
        bound=bind_scene_execution(state,(),backend)
        self.assertIs(bound.kernels[0],bound.kernels[1])
        actions=tuple(weakref.ref(action) for action in state.actions)
        rng=np.random.default_rng(261051);refs=[]
        for cfg in range(2):
            values=tuple((rng.normal(size=shape)+1j*rng.normal(size=shape)).astype('complex64')
                         for shape in state.actions[0].core.shapes)
            refs.extend(weakref.ref(value) for value in values)
            actual,owned=bound.kernels[cfg](values)
            self.assertTrue(owned)
            equation=state.actions[0].equation.normalized()
            args=tuple(item for value,axes in zip(values,equation.inputs,strict=True)
                       for item in (value.astype('complex128'),axes))+(equation.output,)
            expected=np.einsum(*args,optimize=True)
            limit=5e-5*np.maximum(np.maximum(np.abs(actual),np.abs(expected)),1e-12)+2e-8
            self.assertTrue(np.all(np.abs(actual-expected)<=limit))
            del values,actual,args
        gc.collect();self.assertTrue(all(ref() is None for ref in refs))
        del state
        gc.collect();self.assertTrue(all(ref() is None for ref in actions))

    def test_cache_hit_checks_each_action_contract_and_output_layout(self):
        state=paired();cache={}
        bind_scene_execution(SimpleNamespace(actions=state.actions[:1],catalog=state.catalog),(),np,kernel_cache=cache)
        action=state.actions[1]
        invalid=[replace(action,wide_accumulation=False),
                 replace(action,equation=replace(action.equation,output=action.equation.output[::-1]))]
        for current in invalid:
            with self.assertRaisesRegex(ValueError,'declared layouts'):
                bind_scene_execution(SimpleNamespace(actions=(current,),catalog=state.catalog),(),np,kernel_cache=cache)
        for key,spec in ((action.operands[0],TensorSpec((1,1),'complex64')),
                         (action.operands[1],replace(state.catalog[action.operands[1]],dtype=np.dtype('complex128'))),
                         (action.tensor,TensorSpec((1,1),'complex64')),
                         (action.tensor,replace(state.catalog[action.tensor],dtype=np.dtype('complex128')))):
            with self.assertRaisesRegex(ValueError,'declared layouts'):
                bind_scene_execution(SimpleNamespace(actions=(action,),catalog={**state.catalog,key:spec}),(),np,kernel_cache=cache)
        producer=SimpleNamespace(left=(SimpleNamespace(tensor=action.tensor,permutation=(0,1)),),
            right=(SimpleNamespace(tensor='unused',permutation=(0,1)),),prepared_sides=(True,False))
        with self.assertRaisesRegex(ValueError,'cannot borrow'):
            bind_scene_execution(SimpleNamespace(actions=(action,),catalog=state.catalog),(),np,
                                 kernel_cache=cache,producer_batches=(producer,))

    def test_core_tile_and_backend_identity_and_debug_do_not_alias(self):
        state=paired();action=state.actions[1]
        changed=replace(action,core=replace(action.core,tile_elements=2))
        state.actions=state.actions[:1]+(changed,)
        cache={};bound=bind_scene_execution(state,(),np,kernel_cache=cache)
        self.assertIsNot(bound.kernels[0],bound.kernels[1])
        backend=SimpleNamespace(__name__='cupy',RawKernel=object)
        gpu=bind_scene_execution(state,(),backend,kernel_cache=cache)
        self.assertIsNot(gpu.kernels[0],bound.kernels[0])
        state=paired()
        with execution_debug(True):
            debug=bind_scene_execution(state,(),np)
        self.assertIsNot(debug.kernels[0],debug.kernels[1])
        state=paired();action=state.actions[1]
        wide=replace(action,core=replace(action.core,output_dtype='complex128'))
        state.actions=(state.actions[0],wide)
        state.catalog[wide.tensor]=replace(state.catalog[wide.tensor],dtype=np.dtype('complex128'))
        precision=bind_scene_execution(state,(),np)
        self.assertIsNot(precision.kernels[0],precision.kernels[1])
        values=tuple(np.zeros(shape,'complex64') for shape in action.core.shapes)
        self.assertEqual(precision.kernels[0](values)[0].dtype,np.dtype('complex64'))
        self.assertEqual(precision.kernels[1](values)[0].dtype,np.dtype('complex128'))

    def test_gpu_warm_call_does_not_recompute_core_geometry(self):
        state=paired();action=state.actions[0]
        class Kernel:
            def __call__(self,*args):pass
        backend=SimpleNamespace(__name__='cupy',RawKernel=lambda *a,**k:Kernel(),empty=np.empty)
        run=bind_factorized_reduction(action,state.catalog,backend)
        values=tuple(np.zeros(shape,'complex64') for shape in action.core.shapes)
        run(values)  # Establish the ordinary byte-stride layout once.
        core_type=type(action.core)
        with patch.object(core_type,'dimensions',property(lambda self:(_ for _ in ()).throw(
                AssertionError('warm geometry recomputation')))):
            for _ in range(2):run(values)
        with self.assertRaisesRegex(ValueError,'saved layouts'):
            run(values[:-1])

    def test_shared_stride_cache_remains_bounded_without_input_owners(self):
        import inspect
        state=paired();action=state.actions[0]
        class Kernel:
            def __call__(self,*args):pass
        class Input:
            def __init__(self,shape,stride):
                self.shape,self.dtype=shape,np.dtype('complex64')
                self.strides=tuple(stride*(j+1)*8 for j in range(len(shape)))
        backend=SimpleNamespace(__name__='cupy',RawKernel=lambda *a,**k:Kernel(),empty=np.empty)
        run=bind_factorized_reduction(action,state.catalog,backend);refs=[]
        for stride in range(1,71):
            values=tuple(Input(shape,stride) for shape in action.core.shapes)
            refs.extend(weakref.ref(value) for value in values)
            run(values)
            del values
        self.assertEqual(len(inspect.getclosurevars(run).nonlocals['layouts']),64)
        gc.collect();self.assertTrue(all(ref() is None for ref in refs))


class FactorizedFactoryReuseCuPyTests(unittest.TestCase):
    def test_shared_factory_concurrent_current_values_streams_and_release(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        state=paired();bound=bind_scene_execution(state,(),cp)
        self.assertIs(bound.kernels[0],bound.kernels[1])
        barrier=threading.Barrier(2);errors=[];refs=[];gates=[]
        def worker(index):
            try:
                rng=np.random.default_rng(261052+index)
                with cp.cuda.Device(0):
                    stream=cp.cuda.Stream(non_blocking=True)
                    host=tuple((rng.normal(size=s)+1j*rng.normal(size=s)).astype('complex64')
                               for s in state.actions[0].core.shapes)
                    with stream:
                        values=tuple(cp.asarray(v[...,::-1])[...,::-1] if index else cp.asarray(v) for v in host)
                        refs.extend(weakref.ref(value) for value in values)
                        barrier.wait(timeout=30)
                        result,_=bound.kernels[index](values);actual=result.get()
                    equation=state.actions[0].equation.normalized()
                    args=tuple(item for value,axes in zip(host,equation.inputs,strict=True)
                               for item in (value.astype('complex128'),axes))+(equation.output,)
                    expected=np.einsum(*args,optimize=True)
                    limit=5e-5*np.maximum(np.maximum(np.abs(actual),np.abs(expected)),1e-12)+2e-8
                    gates.append(float(np.max(np.abs(actual-expected)/limit)))
                    stream.synchronize();del values,result
            except BaseException as error:
                errors.append(error)
        threads=[threading.Thread(target=worker,args=(index,)) for index in range(2)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(timeout=60)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertFalse(errors,errors);self.assertEqual(len(gates),2)
        self.assertTrue(all(gate<=1 for gate in gates),gates)
        gc.collect();self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(cp.get_default_memory_pool().used_bytes(),0)
