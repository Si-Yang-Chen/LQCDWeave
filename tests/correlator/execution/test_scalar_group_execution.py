"""Public scalar grouping preserves fallback, events and current ownership."""
from dataclasses import replace
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution import bound_program, scene_runtime, scalar_group_execution
from lqcdweave.contraction.execution.binding_cache import ExecutionBindingCache, native_storage_binding
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.storage_runtime import _BlockExecution, replay_scene_storage
from lqcdweave.contraction.execution.operations.result_views import (
    _DeferredResultView, _ResultOwner, bind_result_views,
)
from lqcdweave.contraction.planning.api import lower_scalar_reduction_groups
from lqcdweave.contracts.correlator.api import Action, StorageEvent, describe_scalar_reduction_group
from lqcdweave.contracts.measurement import PhaseTiming
from correlator.planning.test_scalar_reduction_groups import fixture
from correlator.execution.test_scalar_reduction_group import _SourceBackend as GroupBackend
from correlator.execution.test_raw_precise_reduction import _SourceBackend as SingleBackend


class Backend(GroupBackend):
    def __init__(self):
        super().__init__()
        self.single = SingleBackend()
    def __getattr__(self,name):return getattr(np,name)
    def RawKernel(self,source,name,**kwargs):
        if name.startswith('ed_precise_scalar_group'):
            return super().RawKernel(source,name,**kwargs)
        return self.single.RawKernel(source,name,**kwargs)
    def get_default_memory_pool(self):return SimpleNamespace(total_bytes=lambda:0)


def prepared(count=4,shape=(513,),dtype=np.float32):
    plan,_=fixture(count,shape=shape,dtype=dtype)
    plan,_=lower_scalar_reduction_groups(plan,target_runtime='cuda')
    rng=np.random.default_rng(391)
    arrays={key:rng.normal(size=spec.shape).astype(spec.dtype) for key,spec in plan.state.problem.specs.items()}
    return plan,arrays


class ArrayLoader:
    preparation_bound=True
    def __init__(self,arrays,backend):self.arrays,self.backend=arrays,backend
    def read_block(self,key,block):raise AssertionError('fixture has no physical blocks')
    def materialize(self,key,source,blocks):return self.arrays[key]


def execution(plan,tables,inputs):
    record=[];retained={}
    return SimpleNamespace(plan=plan,execution_tables=tables,record=record,retained=retained,
        source=lambda key:plan.state.problem.sources[key],sources=plan.state.problem.sources,
        materialize=lambda key,source:inputs[key],materializes_owned=lambda key:True,
        begin_action=lambda index,timed:record.append(('before',index)),
        end_action=lambda index,timed:record.append(('after',index)),
        retains=lambda key:False,retain_tensor=lambda key,value:retained.__setitem__(key,value),
        has_retained=lambda key:key in retained,retained_value=lambda key:retained[key],
        action_seconds=lambda index:plan.state.actions[index].seconds,
        model_features=lambda key:{'operation_domain':'test_load'})


class ScalarGroupExecutionTests(unittest.TestCase):
    def test_first_member_publishes_four_nonoverlapping_values_with_rounded_budget(self):
        plan, inputs = prepared(count=4, shape=(513,))
        backend = Backend()
        tables = self.bind(plan, backend)
        group = plan.scalar_reduction_groups[0]
        keys = tuple(plan.state.actions[index].tensor for index in group.action_indices)
        arrays = {}
        run = execution(plan, tables, inputs)
        calls = iter(tables.program.calls)
        for operation in calls:
            operation((arrays, run))
            if all(key in arrays for key in keys):
                break
        else:
            self.fail('first scalar group did not publish every result')
        self.assertEqual(len(backend.launches), 2)
        values = tuple(arrays[key] for key in keys)
        self.assertEqual(len({value.ctypes.data for value in values}), 4)
        self.assertTrue(all(value.shape == () and value.base is not None for value in values))
        for index, value in zip(group.action_indices, values, strict=True):
            action = plan.state.actions[index]
            expected = np.einsum('i,i->', *(inputs[key] for key in action.operands), dtype=np.float64)
            np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-6)
        point = next(point for point in plan.memory if point.phase == 'contract'
                     and point.action_index == group.action_indices[0])
        # The unchanged conservative budget covers four scalar owners: one original,
        # three born early, plus one independent 96-byte partial allocation.
        self.assertEqual(point.scalar_reduction_device_bytes, 3*512+512)
        self.assertGreater(point.allocation_rounding_device_bytes, 0)
        for operation in calls:
            operation((arrays, run))

    def bind(self,plan,backend,**kwargs):
        # Packed output-copy kernels are tested separately; use the same final
        # independent copies while interpreting only reduction sources here.
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            return scene_runtime.bind_scene_execution(plan.state,plan.kernel_batches,backend,
                events=kwargs.pop('events',plan.events),scalar_reduction_groups=plan.scalar_reduction_groups,**kwargs)

    def test_many_member_sets_keep_their_warm_stride_bindings_without_eviction(self):
        # A production template has many same-geometry groups. Their combined
        # stride tuples differ, even though each particular group is stable.
        # Cycling more groups than the 64-layout cache must not rebind ABI
        # tables on every warm replay.
        count=70
        plan,_=fixture(2*count,shape=(513,))
        plan,_=lower_scalar_reduction_groups(plan,target_runtime='cuda',max_tasks=2)
        backend=Backend();backend.metadata_only=True
        tables=self.bind(plan,backend)
        base=np.arange(513*(count+1),dtype=np.float32)
        arrays={}
        for index,bound in enumerate(tables.scalar_starts.values()):
            value=base[:513*(index+1):index+1]
            for pair in bound.inputs:
                arrays.update((key,value) for key in pair)
        self.assertEqual(len(tables.scalar_starts),count)
        observed=[]
        def trace(frame,event,arg):
            if (event=='call' and frame.f_code.co_name=='bind_layout'
                    and frame.f_code.co_filename.replace('\\','/').rsplit('/',1)[-1]
                    in ('scalar_reduction_group.py','reduction_group_runtime.py')):
                observed.append(1)
        previous=sys.getprofile()
        try:
            sys.setprofile(trace)
            for bound in tables.scalar_starts.values():bound.execute(arrays)
            cold=len(observed);observed.clear()
            for bound in tables.scalar_starts.values():bound.execute(arrays)
        finally:
            sys.setprofile(previous)
        self.assertEqual(cold,count)
        self.assertEqual(len(observed),0)

    def test_bound_program_current_inputs_fallback_kernels_and_event_retention_order(self):
        plan,inputs=prepared()
        actions=list(plan.state.actions)
        first=plan.scalar_reduction_groups[0].action_indices[0]
        actions.insert(first+1,Action('release','a0'))
        indices=tuple(i for i,a in enumerate(actions) if a.kind=='contract')
        group=describe_scalar_reduction_group(actions,indices,plan.state.catalog,1.0)
        plan=replace(plan,state=replace(plan.state,actions=tuple(actions)),scalar_reduction_groups=(group,))
        events=(StorageEvent(first,'read','test'),StorageEvent(first+1,'release','test'),
                StorageEvent(indices[-1],'release','later'),StorageEvent(len(actions),'release','final'))
        backend=Backend()
        tables=self.bind(plan,backend,events=events,retained=(actions[indices[-1]].tensor,))
        self.assertEqual(tables.scalar_groups,(group,))
        self.assertEqual(set(indices),set(tables.kernels))
        self.assertFalse(tables.starts);self.assertFalse(tables.members)
        for scale in (1,3):
            current={key:value*scale for key,value in inputs.items()}
            run=execution(plan,tables,current);arrays={}
            actual=tables.program.run(run,arrays)
            for term,value in zip(plan.state.problem.terms,actual,strict=True):
                expected=np.einsum('i,i->',*(current[k] for k in term.inputs),dtype=np.float64)
                np.testing.assert_allclose(value,expected,rtol=2e-6,atol=2e-6)
            self.assertNotIn('a0',arrays)
            self.assertIs(run.retained[actions[indices[-1]].tensor],arrays[actions[indices[-1]].tensor])
            self.assertEqual(run.record,[('before',0),('before',first),('after',first+1),
                                         ('after',indices[-1]),('after',len(actions))])
            outputs=[arrays[a.tensor] for a in actions if a.kind=='contract']
            self.assertTrue(all(value.base is outputs[0].base for value in outputs))
            self.assertEqual(len({value.ctypes.data for value in outputs}),len(outputs))
        self.assertEqual(len(backend.launches),4)
        self.assertEqual(backend.single.launches,0)

    def test_tiny_debug_unsupported_and_partially_reused_groups_fall_back_whole(self):
        for mode in ('tiny','debug','no_raw','hip','partial'):
            plan,inputs=prepared(shape=(2,3) if mode=='tiny' else (513,))
            backend=Backend()
            if mode=='no_raw':backend.RawKernel=None
            if mode=='hip':backend.cuda.runtime.is_hip=True
            if mode=='partial':
                member=plan.scalar_reduction_groups[0].action_indices[1]
                actions=list(plan.state.actions);actions[member]=replace(actions[member],reused=True)
                plan=replace(plan,state=replace(plan.state,actions=tuple(actions)))
            with execution_debug(mode=='debug'):
                with patch.object(scalar_group_execution,'bind_scalar_reduction_group',side_effect=AssertionError('unqualified fusion')):
                    tables=self.bind(plan,backend)
                self.assertEqual(tables.scalar_groups,())
                self.assertEqual(tables.scalar_starts,{})
                expected={i for i,a in enumerate(plan.state.actions) if a.kind=='contract' and not a.reused}
                self.assertEqual(set(tables.kernels),expected)
        # An invalid saved resource allowance is a binding error, not silently
        # enlarged workspace or a partially executed group.
        plan,_=prepared()
        plan=replace(plan,scalar_reduction_groups=(replace(plan.scalar_reduction_groups[0],workspace_bytes=1),))
        with self.assertRaisesRegex(ValueError,'saved workspace'):self.bind(plan,Backend())

    def test_direct_cfg_context_resolves_deferred_views_again_for_each_execution(self):
        plan,inputs=prepared(count=2)
        state=replace(plan.state,actions=tuple(replace(a,reused=True) if a.kind=='load' else a for a in plan.state.actions))
        plan=replace(plan,state=state);backend=Backend()
        tables=self.bind(plan,backend,shared_residency=True)
        binding={key:i for i,key in enumerate(state.catalog)}
        class DirectArrays:
            def __init__(self,store):self.store=store
            def _dense_call_context(self):return self.store,binding
            def __getitem__(self,key):
                if key in inputs:raise AssertionError('group used mapping instead of direct current values')
                return self.store[binding[key]]
        for scale in (1,2):
            store={}
            for key,value in inputs.items():
                owner=_ResultOwner(np.expand_dims(value*scale,0))
                layout=bind_result_views(owner.value.shape,value.dtype,((key,(0,slice(None)),()),))[0][1]
                store[binding[key]]=_DeferredResultView(owner,layout)
            actual=tables.program.run(execution(plan,tables,inputs),DirectArrays(store))
            self.assertTrue(all(type(store[binding[key]]) is np.ndarray for key in inputs))
            for term,value in zip(state.problem.terms,actual,strict=True):
                expected=np.einsum('i,i->',*(inputs[k]*scale for k in term.inputs),dtype=np.float64)
                np.testing.assert_allclose(value,expected,rtol=2e-6,atol=2e-6)
            member=plan.scalar_reduction_groups[0].action_indices[1]
            actions=list(state.actions);actions[member]=replace(actions[member],reused=True)
            partial=replace(plan,state=replace(state,actions=tuple(actions)))
            fallback=self.bind(partial,backend,shared_residency=True)
            kept=store[binding[actions[member].tensor]]
            repeated=fallback.program.run(execution(partial,fallback,inputs),DirectArrays(store))
            self.assertEqual(fallback.scalar_groups,())
            self.assertIs(store[binding[actions[member].tensor]],kept)
            for left,right in zip(repeated,actual,strict=True):
                self.assertEqual(np.asarray(left).tobytes(),np.asarray(right).tobytes())

    def test_measure_reports_one_actual_group_and_zero_work_for_later_members(self):
        plan,inputs=prepared(count=3);backend=Backend();tables=self.bind(plan,backend)
        run=execution(plan,tables,inputs)
        class Measurement:
            def measure_phase(self,name,call,**kwargs):
                return call(),PhaseTiming(name,0.001,kwargs.get('nbytes'),None)
        report=scene_runtime._replay(plan.state,object(),backend=backend,measure=True,
            source_execution=run,measurement=Measurement())
        indices=plan.scalar_reduction_groups[0].action_indices
        first=report.timings[indices[0]]
        self.assertEqual(first.model_features['kernel_calls'],2)
        self.assertEqual(first.model_features['operation_domain'],'scalar_reduction_group')
        self.assertEqual([phase.name for phase in first.phases],['contract_kernel'])
        for index in indices[1:]:
            self.assertEqual(report.timings[index].model_features['kernel_calls'],0)
            self.assertEqual(report.timings[index].phases,())
        self.assertEqual(len(backend.launches),2)
        self.assertEqual(len(report.timings),len(plan.state.actions))
        with execution_debug(True):
            scene_runtime._replay(plan.state,object(),backend=backend,source_execution=run)
        self.assertEqual(len(backend.launches),2) # Cached normal table cannot fuse in debug.

    def test_storage_passes_annotations_audits_debug_budget_and_checks_backend_and_cache(self):
        plan,inputs=prepared(count=2);backend=Backend()
        self.assertTrue(native_storage_binding(plan))
        class Extension(type(plan.scalar_reduction_groups[0])):pass
        extended=Extension(**vars(plan.scalar_reduction_groups[0]))
        self.assertFalse(native_storage_binding(replace(plan,scalar_reduction_groups=(extended,))))
        with self.assertRaisesRegex(ValueError,'backend disagrees'):
            _BlockExecution(plan,object(),np,io_overlap=False)
        loader=SimpleNamespace(backend=backend,preparation_bound=True)
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            run=_BlockExecution(plan,loader,backend,io_overlap=False)
        self.assertEqual(run.execution_tables.scalar_groups,plan.scalar_reduction_groups)
        indices=plan.scalar_reduction_groups[0].action_indices
        self.assertEqual(run.action_seconds(indices[0]),plan.scalar_reduction_groups[0].seconds)
        self.assertTrue(all(run.action_seconds(i)==0 for i in indices[1:]))
        bad=replace(plan,memory=tuple(replace(point,scalar_reduction_device_bytes=0) for point in plan.memory))
        with execution_debug(True):
            with self.assertRaisesRegex(ValueError,'scalar reduction memory bucket'):
                replay_scene_storage(bad,loader,backend=backend,io_overlap=False)

    def test_public_storage_replay_and_identity_cache_refresh(self):
        plan,inputs=prepared(count=2);backend=Backend();cache=ExecutionBindingCache()
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            for scale in (1,3):
                current={key:value*scale for key,value in inputs.items()}
                grouped=replay_scene_storage(plan,ArrayLoader(current,backend),backend=backend,
                    execution_cache=cache,io_overlap=False)
                original=replace(plan,scalar_reduction_groups=())
                reference=replay_scene_storage(original,ArrayLoader(current,backend),backend=backend,
                    execution_cache=cache,io_overlap=False)
                for actual,expected in zip(grouped,reference,strict=True):
                    self.assertEqual(np.asarray(actual).tobytes(),np.asarray(expected).tobytes())
            # A fresh immutable description invalidates an existing binding.
            self.assertIs(cache._binding[0],original)
            wrong=dict(inputs);wrong['a0']=inputs['a0'][:-1]
            with self.assertRaisesRegex(ValueError,'mismatched tensor'):
                replay_scene_storage(plan,ArrayLoader(wrong,backend),backend=backend,
                    execution_cache=cache,io_overlap=False)


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE')=='1',
                     'explicit CUDA gate; no device probe in CPU suite')
class ScalarGroupPublicCuPyTests(unittest.TestCase):
    def test_public_storage_bytes_current_arrays_and_metadata_invalidation(self):
        import cupy as cp
        for dtype in (np.float32,np.complex64):
            plan,host=prepared(count=4,shape=(4,513),dtype=dtype)
            original=replace(plan,scalar_reduction_groups=())
            cache=ExecutionBindingCache();previous=None;held=[]
            for iteration in range(3):
                arrays={key:cp.asarray(value*(iteration+1)+(0.3j if dtype==np.complex64 else 0.3))
                        .astype(dtype) for key,value in host.items()}
                if iteration==1:arrays={key:value[:,::-1] for key,value in arrays.items()}
                if iteration==2:arrays={key:cp.broadcast_to(value[:1],value.shape) for key,value in arrays.items()}
                # Keep old owners alive to prove this is a genuinely new pointer set.
                held.append(arrays)
                pointers=tuple(value.data.ptr for value in arrays.values())
                if previous is not None:self.assertNotEqual(pointers,previous)
                previous=pointers
                grouped=replay_scene_storage(plan,ArrayLoader(arrays,cp),backend=cp,
                    execution_cache=cache,io_overlap=False)
                reference=replay_scene_storage(original,ArrayLoader(arrays,cp),backend=cp,io_overlap=False)
                for actual,expected in zip(grouped,reference,strict=True):
                    self.assertEqual(cp.asnumpy(actual).tobytes(),cp.asnumpy(expected).tobytes())
            changed=replace(plan,scalar_reduction_groups=())
            replay_scene_storage(changed,ArrayLoader(arrays,cp),backend=cp,execution_cache=cache,io_overlap=False)
            self.assertIs(cache._binding[0],changed)
            bad=dict(arrays);bad['a0']=cp.empty((4,512),dtype=dtype)
            with self.assertRaisesRegex(ValueError,'mismatched tensor'):
                replay_scene_storage(plan,ArrayLoader(bad,cp),backend=cp,execution_cache=cache,io_overlap=False)
            bad=dict(arrays);bad['a0']=arrays['a0'].astype(np.complex128 if dtype==np.complex64 else np.float64)
            with self.assertRaisesRegex(ValueError,'mismatched tensor'):
                replay_scene_storage(plan,ArrayLoader(bad,cp),backend=cp,execution_cache=cache,io_overlap=False)

    def test_cfg_direct_deferred_arrays_and_partial_reuse_fallback(self):
        import cupy as cp
        plan,host=prepared(count=3,dtype=np.complex64)
        state=replace(plan.state,actions=tuple(replace(a,reused=True) if a.kind=='load' else a
                                               for a in plan.state.actions))
        plan=replace(plan,state=state)
        binding={key:index for index,key in enumerate(state.catalog)}
        class Arrays:
            def __init__(self,store):self.store=store
            def _dense_call_context(self):return self.store,binding
            def __getitem__(self,key):return self.store[binding[key]]
        tables=scene_runtime.bind_scene_execution(state,(),cp,events=(),shared_residency=True,
            scalar_reduction_groups=plan.scalar_reduction_groups)
        self.assertEqual(tables.scalar_groups,plan.scalar_reduction_groups)
        reference_tables=scene_runtime.bind_scene_execution(state,(),cp,events=(),shared_residency=True)
        for scale in (1,2):
            current={key:cp.asarray(value*scale) for key,value in host.items()}
            store={}
            for key,value in current.items():
                owner=_ResultOwner(value[None,:])
                layout=bind_result_views(owner.value.shape,value.dtype,((key,(0,slice(None)),()),))[0][1]
                store[binding[key]]=_DeferredResultView(owner,layout)
            actual=tables.program.run(execution(plan,tables,current),Arrays(store))
            expected=reference_tables.program.run(execution(plan,reference_tables,current),
                Arrays({binding[key]:value for key,value in current.items()}))
            for left,right in zip(actual,expected,strict=True):
                self.assertEqual(cp.asnumpy(left).tobytes(),cp.asnumpy(right).tobytes())
            self.assertTrue(all(isinstance(store[binding[key]],cp.ndarray) for key in current))
            member=plan.scalar_reduction_groups[0].action_indices[1]
            actions=list(state.actions);actions[member]=replace(actions[member],reused=True)
            partial=replace(state,actions=tuple(actions))
            fallback=scene_runtime.bind_scene_execution(partial,(),cp,events=(),shared_residency=True,
                scalar_reduction_groups=plan.scalar_reduction_groups)
            self.assertEqual(fallback.scalar_groups,())
            kept=store[binding[actions[member].tensor]]
            reused={binding[key]:value for key,value in current.items()}
            reused[binding[actions[member].tensor]]=kept
            values=fallback.program.run(execution(replace(plan,state=partial),fallback,current),Arrays(reused))
            self.assertIs(reused[binding[actions[member].tensor]],kept)
            for left,right in zip(values,expected,strict=True):
                self.assertEqual(cp.asnumpy(left).tobytes(),cp.asnumpy(right).tobytes())


if __name__=='__main__':unittest.main()
