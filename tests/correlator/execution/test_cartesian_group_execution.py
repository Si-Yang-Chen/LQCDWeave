"""Public Cartesian grouping preserves packing, events and independent cubes."""
from dataclasses import replace
from contextlib import contextmanager, nullcontext
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, SourcePreparation,
    StorageEvent, ArrayConstant, dense_batch_memory, describe_cartesian_reduction_group,
    cartesian_reduction_group_memory,
)
from lqcdweave.contraction.planning.api import ContractionState, CostProfile, describe_dense_batch, plan_scene_storage
from lqcdweave.contraction.execution import scene_runtime, bound_program, cartesian_group_execution as module
from lqcdweave.contraction.execution.binding_cache import ExecutionBindingCache, native_storage_binding
from lqcdweave.contraction.execution.cfg_runtime import _NodeView, _active_cartesian_groups
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.storage_runtime import _BlockExecution, replay_scene_storage
from lqcdweave.contraction.execution.operations.result_views import _DeferredResultView, _ResultOwner, bind_result_views
from lqcdweave.contraction.execution.operations.dense_batch import _BatchSide, _BatchLayout
from lqcdweave.contraction.execution.reduction_kernel import _kernel
from lqcdweave.contracts.measurement import PhaseTiming
from correlator.execution.test_matrix_reduction_group import MatrixBackend
from correlator.execution.test_raw_precise_reduction import _SourceBackend as SingleBackend
from correlator.execution.test_scalar_group_execution import execution, ArrayLoader


class Backend(MatrixBackend):
    def __init__(self):super().__init__();self.single=SingleBackend()
    def __getattr__(self,name):return getattr(np,name)
    def empty(self,shape,dtype,**kwargs):
        if kwargs:return np.empty(shape,dtype,**kwargs)
        return super().empty(shape,dtype)
    def RawKernel(self,source,name,**kwargs):
        if name.startswith('ed_precise_matrix_group'):return super().RawKernel(source,name,**kwargs)
        return self.single.RawKernel(source,name,**kwargs)
    def get_default_memory_pool(self):return SimpleNamespace(total_bytes=lambda:0)


def fixture(count=2, *, m=2,n=3,k=80,dtype=np.complex64,prepared_side=None):
    profile=CostProfile();specs={};values={};terms=[]
    for child in range(count):
        for side,rows in (('a',m),('b',n)):
            for row in range(rows):
                key=f'{child}{side}{row}';specs[key]=TensorSpec((k,),dtype)
                values[key]=((np.arange(k,dtype=np.float32)%7-3)*(row+1)).astype(dtype)
                if np.dtype(dtype).kind=='c':values[key]+=0.5j*values[key]
        terms.extend(ContractionTerm((f'{child}a{i}',f'{child}b{j}'),'i,i->') for i in range(m) for j in range(n))
    problem=ContractionProblem(specs,tuple(terms),{key:SourceSpec(key,0,read_bytes=0) for key in specs},backend='cupy')
    state=ContractionState.initial(problem)
    for key in specs:state=state.load(key,profile,1<<27)
    batches=[]
    for child in range(count):
        indices=[]
        for term in range(child*m*n,(child+1)*m*n):
            indices.append(len(state.actions));state=state.contract(term,(0,1),profile,1<<27)
        ready={f'{child}{prepared_side}0':(0,)} if prepared_side else None
        priced=tuple(replace(a,seconds=10.) if a.kind=='contract' else a for a in state.actions)
        batches.append(describe_dense_batch(priced,tuple(indices),state.catalog,profile,
                                             backend='cupy',prepared_inputs=ready))
    plan=plan_scene_storage(state,{}, {key:SourcePreparation(()) for key in specs},profile=profile,
                            memory_limit=1<<27,host_memory_limit=1<<27)
    plan=replace(plan,kernel_batches=tuple(batches))
    plan=replace(plan,memory=dense_batch_memory(plan))
    group=describe_cartesian_reduction_group(batches,tuple(b.action_indices for b in batches),state.catalog,1.)
    plan=replace(plan,cartesian_reduction_groups=(group,))
    memory=cartesian_reduction_group_memory(plan)
    plan=replace(plan,memory=memory,peak_host_bytes=max(p.host_bytes for p in memory),
                 peak_device_bytes=max(p.device_bytes for p in memory))
    return plan,values


class CartesianGroupExecutionTests(unittest.TestCase):
    def tearDown(self):_kernel.cache_clear();module._FAILED_SUBMISSIONS.clear()
    def bind(self,plan,backend,**kwargs):
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            return scene_runtime.bind_scene_execution(plan.state,plan.kernel_batches,backend,
                events=kwargs.pop('events',plan.events),cartesian_reduction_groups=plan.cartesian_reduction_groups,**kwargs)
    def check(self,plan,values,actual):
        for term,value in zip(plan.state.problem.terms,actual,strict=True):
            wide=np.complex128 if value.dtype.kind=='c' else np.float64
            expected=np.einsum('i,i->',*(values[key] for key in term.inputs),dtype=wide).astype(value.dtype)
            self.assertEqual(value.tobytes(),expected.tobytes())

    def test_public_program_uses_current_inputs_and_original_layouts_events_and_cubes(self):
        for m,n,prepared in ((2,3,None),(1,3,'a'),(2,1,'b')):
            plan,values=fixture(m=m,n=n,prepared_side=prepared);backend=Backend()
            indices=plan.cartesian_reduction_groups[0].action_indices
            events=(StorageEvent(indices[0],'read','first'),StorageEvent(indices[1],'release','middle'),
                    StorageEvent(indices[-1],'release','last'))
            tables=self.bind(plan,backend,events=events,retained=(plan.state.actions[indices[-1]].tensor,))
            self.assertEqual(tables.cartesian_groups,plan.cartesian_reduction_groups)
            bound=next(iter(tables.cartesian_starts.values()))
            for batch,layout in zip(bound.batches,bound.layouts):
                self.assertIs(layout,tables.kernels[batch.action_indices[0]].func.__self__)
            for scale in (1,2):
                current={key:value*scale for key,value in values.items()};arrays={}
                run=execution(plan,tables,current);actual=tables.program.run(run,arrays)
                self.check(plan,current,actual)
                self.assertEqual(run.record,[('before',0),('before',indices[0]),('after',indices[1]),('after',indices[-1])])
                owners=[]
                def root(value):
                    while isinstance(value.base,np.ndarray):value=value.base
                    return value
                for batch in plan.kernel_batches:
                    children=[root(arrays[result.tensor]) for result in batch.results]
                    self.assertTrue(all(value is children[0] for value in children[1:]))
                    owners.append(children[0])
                self.assertIs(owners[0],owners[1])
                results=[arrays[result.tensor] for batch in plan.kernel_batches for result in batch.results]
                self.assertEqual(len({value.ctypes.data for value in results}),len(results))
                self.assertIs(run.retained[plan.state.actions[indices[-1]].tensor],arrays[plan.state.actions[indices[-1]].tensor])
            self.assertEqual(len(backend.launches),4);self.assertEqual(backend.single.launches,0)

    def test_debug_cpu_hip_missing_raw_and_unqualified_child_bind_whole_fallback(self):
        for mode in ('debug','cpu','hip','no_raw','group_unsupported','partial'):
            plan,values=fixture();backend=np if mode=='cpu' else Backend()
            if mode=='hip':backend.cuda.runtime.is_hip=True
            if mode=='no_raw':backend.RawKernel=None
            if mode=='partial':
                index=plan.kernel_batches[-1].action_indices[-1];actions=list(plan.state.actions)
                actions[index]=replace(actions[index],reused=True);plan=replace(plan,state=replace(plan.state,actions=tuple(actions)))
            with execution_debug(mode=='debug'):
                if mode=='group_unsupported':
                    with patch.object(module,'bind_matrix_reduction_group',return_value=None):tables=self.bind(plan,backend)
                else:tables=self.bind(plan,backend)
            self.assertEqual(tables.cartesian_groups,())
            self.assertEqual(set(tables.starts),set(tables.kernels))
        plan,_=fixture();group=replace(plan.cartesian_reduction_groups[0],workspace_bytes=1)
        with self.assertRaisesRegex(ValueError,'saved workspace'):
            self.bind(replace(plan,cartesian_reduction_groups=(group,)),Backend())

    def test_direct_cfg_deferred_inputs_and_child_owners_preserve_zero_copy_protocol(self):
        plan,values=fixture(m=1,n=3,prepared_side='a');backend=Backend()
        state=replace(plan.state,actions=tuple(replace(a,reused=True) if a.kind=='load' else a for a in plan.state.actions))
        plan=replace(plan,state=state);tables=self.bind(plan,backend,shared_residency=True)
        binding={key:index for index,key in enumerate(state.catalog)}
        for scale in (1,3):
            store={}
            for key,value in values.items():
                owner=_ResultOwner(np.expand_dims(value*scale,0))
                view=bind_result_views(owner.value.shape,value.dtype,((key,(0,slice(None)),()),))[0][1]
                store[binding[key]]=_DeferredResultView(owner,view)
            arrays=_NodeView(binding,store);published=[]
            original=_NodeView._publish_result_views
            def publish(view,value,layouts):
                original(view,value,layouts)
                published.append(tuple(store[binding[key]] for key,_ in layouts))
            with patch.object(_NodeView,'_publish_result_views',new=publish):
                actual=tables.program.run(execution(plan,tables,values),arrays)
            self.check(plan,{key:value*scale for key,value in values.items()},actual)
            self.assertEqual(len(published),2)
            for results in published:
                self.assertTrue(all(type(value) is _DeferredResultView for value in results))
                self.assertTrue(all(value.owner is results[0].owner for value in results))
            self.assertIsNot(published[0][0].owner,published[1][0].owner)

    def test_measure_reports_aggregate_and_skips_later_numerical_actions(self):
        plan,values=fixture(count=3);backend=Backend();tables=self.bind(plan,backend)
        class Measurement:
            def measure_phase(self,name,call,**kwargs):return call(),PhaseTiming(name,.001,kwargs.get('nbytes'),None)
        report=scene_runtime._replay(plan.state,object(),backend=backend,measure=True,
            source_execution=execution(plan,tables,values),measurement=Measurement())
        indices=plan.cartesian_reduction_groups[0].action_indices
        first=report.timings[indices[0]]
        self.assertEqual(first.model_features['reduction_kernel_calls'],2)
        self.assertEqual(first.model_features['cartesian_batches'],3)
        self.assertEqual(first.model_features['operation_domain'],'cartesian_reduction_group')
        self.assertEqual([p.name for p in first.phases],['dense_batch_pack']*6+['contract_kernel'])
        for index in indices[1:]:
            self.assertEqual(report.timings[index].phases,())
            self.assertEqual(report.timings[index].model_features['kernel_calls'],0)
        self.assertEqual(len(backend.launches),2)

    def test_pack_failure_holds_preceding_owners_and_private_pool_when_sync_fails(self):
        plan,values=fixture();backend=Backend();bound=next(iter(self.bind(plan,backend).cartesian_starts.values()))
        original=_BatchSide.pack;calls=[];refs=[]
        def pack(side,*args):
            calls.append(side)
            if len(calls)==3:raise RuntimeError('pack failed')
            value=original(side,*args);refs.append(weakref.ref(value));return value
        pool=object();backend.synchronize.side_effect=RuntimeError('sync failed')
        with patch.object(_BatchSide,'pack',new=pack),patch(
                'lqcdweave.contraction.execution.device.active_device_execution',
                return_value=SimpleNamespace(backend=backend,pool=pool)):
            with self.assertRaisesRegex(RuntimeError,'pack failed'):bound.execute(dict(values))
        held=module._FAILED_SUBMISSIONS[-1]
        self.assertIs(held[-2],pool);self.assertTrue(all(ref() is not None for ref in refs))
        self.assertEqual(len(calls),3);self.assertEqual(len(backend.launches),0)
        self.assertEqual(set(values),set(held[1]))

    def test_ordinary_view_publication_is_atomic_and_allocation_errors_propagate(self):
        plan,values=fixture();backend=Backend();bound=next(iter(self.bind(plan,backend).cartesian_starts.values()))
        original=_BatchLayout.views;calls=[];arrays=dict(values)
        def views(layout,output):
            calls.append(layout)
            if len(calls)==2:raise RuntimeError('view failed')
            return original(layout,output)
        with patch.object(_BatchLayout,'views',new=views):
            with self.assertRaisesRegex(RuntimeError,'view failed'):bound.execute(arrays)
        self.assertEqual(set(arrays),set(values))
        self.assertGreaterEqual(backend.synchronize.call_count,1)
        backend.fail_allocation=len(backend.allocations)+1
        with self.assertRaisesRegex(MemoryError,'allocation failed'):bound.execute(arrays)
        self.assertEqual(set(arrays),set(values));self.assertEqual(backend.single.launches,0)

    def test_storage_cache_budget_and_same_geometry_group_identity(self):
        plan,values=fixture();backend=Backend();cache=ExecutionBindingCache()
        self.assertTrue(native_storage_binding(plan))
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            run=_BlockExecution(plan,ArrayLoader(values,backend),backend,io_overlap=False)
        group=plan.cartesian_reduction_groups[0]
        self.assertEqual(run.execution_tables.cartesian_groups,(group,))
        self.assertEqual(run.action_seconds(group.action_indices[0]),group.seconds)
        self.assertTrue(all(run.action_seconds(i)==0 for i in group.action_indices[1:]))
        bad=replace(plan,memory=tuple(replace(p,cartesian_reduction_device_bytes=0) for p in plan.memory))
        bad=replace(bad,peak_device_bytes=max(p.device_bytes for p in bad.memory))
        with execution_debug(True):
            with self.assertRaisesRegex(ValueError,'Cartesian'):
                replay_scene_storage(bad,ArrayLoader(values,backend),backend=backend,io_overlap=False)
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            for scale in (1,2):
                current={key:value*scale for key,value in values.items()}
                actual=replay_scene_storage(plan,ArrayLoader(current,backend),backend=backend,
                                             io_overlap=False,execution_cache=cache)
                self.check(plan,current,actual)

    def test_stable_group_identity_cache_and_cfg_table_signature(self):
        from lqcdweave.contraction.execution import cfg_runtime
        plan,values=fixture(count=4);children=plan.kernel_batches
        groups=tuple(describe_cartesian_reduction_group(children,tuple(b.action_indices for b in children[i:i+2]),
            plan.state.catalog,1.) for i in (0,2))
        plan=replace(plan,cartesian_reduction_groups=groups)
        backend=Backend();cache={}
        first=self.bind(plan,backend,kernel_cache=cache)
        again=self.bind(plan,backend,kernel_cache=cache)
        one,two=tuple(first.cartesian_starts.values())
        self.assertIsNot(one.kernel,two.kernel)
        self.assertTrue(all(a.kernel is b.kernel for a,b in zip(first.cartesian_starts.values(),again.cartesian_starts.values())))
        mask=ArrayConstant.from_array(np.zeros(len(plan.state.actions),dtype=bool))
        def cfg(scene,batches=children):
            return SimpleNamespace(templates=(SimpleNamespace(keys=tuple(scene.state.catalog),plan=scene),),
                instances=(SimpleNamespace(template=0,reused=mask,events=scene.events,kernel_batches=batches),))
        owner=cfg_runtime._CfgTableCache(backend)
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            _,tables=cfg_runtime._bind_cfg_tables(cfg(plan),backend,owner)
            original=next(iter(tables.values()))
            _,tables=cfg_runtime._bind_cfg_tables(cfg(plan),backend,owner)
            self.assertIs(original,next(iter(tables.values())))
            _,tables=cfg_runtime._bind_cfg_tables(cfg(replace(plan,cartesian_reduction_groups=())),backend,owner)
            changed=next(iter(tables.values()))
            self.assertIsNot(original,changed);self.assertEqual(changed.cartesian_groups,())
            _,tables=cfg_runtime._bind_cfg_tables(cfg(plan,children[:-1]),backend,owner)
            self.assertEqual(next(iter(tables.values())).cartesian_groups,(groups[0],))

    def test_partial_reuse_falls_back_without_overwriting_retained_scalar(self):
        plan,values=fixture();backend=Backend();actions=list(plan.state.actions)
        actions=[replace(a,reused=True) if a.kind=='load' else a for a in actions]
        member=plan.kernel_batches[-1].action_indices[-1]
        actions[member]=replace(actions[member],reused=True,seconds=0.)
        state=replace(plan.state,actions=tuple(actions));plan=replace(plan,state=state,kernel_batches=plan.kernel_batches[:1])
        tables=self.bind(plan,backend,shared_residency=True)
        self.assertEqual(tables.cartesian_groups,())
        binding={key:index for index,key in enumerate(state.catalog)}
        store={binding[key]:value for key,value in values.items()}
        term=state.problem.terms[-1]
        kept=np.einsum('i,i->',*(values[k] for k in term.inputs),dtype=np.complex128).astype(np.complex64)
        store[binding[actions[member].tensor]]=kept
        actual=tables.program.run(execution(plan,tables,values),_NodeView(binding,store))
        self.check(plan,values,actual)
        self.assertIs(store[binding[actions[member].tensor]],kept)

    def test_runtime_launch_and_prepared_metadata_errors_never_fall_back(self):
        plan,values=fixture();backend=Backend();bound=next(iter(self.bind(plan,backend).cartesian_starts.values()))
        backend.fail_at=1
        arrays=dict(values)
        with self.assertRaisesRegex(RuntimeError,'submission failed'):bound.execute(arrays)
        self.assertEqual(backend.single.launches,0)
        self.assertEqual(set(arrays),set(values))
        plan,values=fixture(m=1,n=3,prepared_side='a');backend=Backend()
        bound=next(iter(self.bind(plan,backend).cartesian_starts.values()))
        values['1a0']=values['1a0'].astype(np.complex128)
        with self.assertRaisesRegex(ValueError,'saved matrix layout'):bound.execute(values)
        self.assertEqual(backend.single.launches,0);self.assertEqual(len(backend.launches),0)

    def test_full_cfg_protocol_reuses_public_tables_and_preserves_filtered_envelope(self):
        from lqcdweave.contracts.correlator.api import TensorPreparation, describe_coordinates
        from lqcdweave.contraction.planning.api import build_cfg_plan, audit_cfg_plan
        from lqcdweave.contraction.execution import cfg_runtime
        from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value
        plan,values=fixture()
        values={key:value+np.complex64(10*(index+1)) for index,(key,value) in enumerate(values.items())}
        preparations={key:SourcePreparation((),recipe=TensorPreparation('constant',
            constants=(ArrayConstant.from_array(value),))) for key,value in values.items()}
        plan=replace(plan,preparations=preparations)
        points,_=describe_coordinates(time_pairs=[{0:0},{0:0}])
        cfg=build_cfg_plan(plan,None,points.points,{},preparations,plan.state.problem.specs,prefetch_host_bytes=0)
        audit=audit_cfg_plan(cfg)
        self.assertEqual(audit['peak_device_bytes'],cfg.peak_device_bytes)
        cfg=decode_plan_value(encode_plan_value(cfg))
        backend=Backend();local={key:prep.recipe.constants[0].array() for key,prep in cfg.preparations.items()}
        observed=[];original=cfg_runtime._CfgBlockExecution
        def make_execution(local_plan,*args,**kwargs):
            observed.append(local_plan)
            return original(local_plan,*args,**kwargs)
        @contextmanager
        def factory(configuration):yield ArrayLoader(local,backend)
        outputs=[]
        device=SimpleNamespace(backend=backend,statistics={},
            bind_loader=lambda loader,preparations,specs:loader,
            download=lambda values:tuple(np.array(value,copy=True) for value in values))
        # Only hardware scope and transfer are replaced: actual cfg planning,
        # archive, queue, loader, execution tables, releases and kernels run.
        with patch.object(cfg_runtime,'device_execution',return_value=nullcontext()),patch.object(
                cfg_runtime,'active_device_execution',return_value=device),patch.object(
                cfg_runtime,'_CfgBlockExecution',side_effect=make_execution),patch.object(
                bound_program,'bind_output_layout',return_value=None),patch(
                'lqcdweave.contraction.planning.api.find_contraction_path',side_effect=AssertionError('fixed replay searched')):
            report=cfg_runtime.execute_cfg_plan(cfg,['cfg'],factory,
                lambda name,index,result:outputs.append(result) if index is not None else None,
                backend=backend,prefetch_host_bytes=0)
        self.assertEqual(report['completed'],('cfg',));self.assertEqual(len(outputs),2)
        for result in outputs:self.check(plan,values,result)
        self.assertEqual(len(observed[0].cartesian_reduction_groups),1)
        self.assertEqual(observed[1].cartesian_reduction_groups,())
        self.assertTrue(any(p.cartesian_reduction_device_bytes for p in observed[1].memory))
        self.assertEqual(len(backend.launches),2)

    def test_cfg_filter_requires_whole_actual_children_and_baseline_ready_inputs(self):
        plan,_=fixture();group=plan.cartesian_reduction_groups[0]
        self.assertEqual(_active_cartesian_groups(plan.state,plan.kernel_batches,(group,)),(group,))
        self.assertEqual(_active_cartesian_groups(plan.state,plan.kernel_batches[:-1],(group,)),())
        action_index=group.batch_action_indices[-1][-1];actions=list(plan.state.actions)
        actions[action_index]=replace(actions[action_index],reused=True)
        self.assertEqual(_active_cartesian_groups(replace(plan.state,actions=tuple(actions)),plan.kernel_batches,(group,)),())
        # The saved group cannot pull a not-yet-published input forward. This
        # models the same lost producer readiness after partial cfg reuse.
        child=plan.kernel_batches[-1];delayed=child.left[0].tensor
        actions=list(plan.state.actions);load=next(i for i,a in enumerate(actions) if a.tensor==delayed and a.kind=='load')
        actions[load]=Action('release',delayed)
        self.assertEqual(_active_cartesian_groups(replace(plan.state,actions=tuple(actions)),plan.kernel_batches,(group,)),())


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE')=='1','explicit CUDA gate; no device probe in CPU suite')
class CartesianGroupPublicCuPyTests(unittest.TestCase):
    def test_full_cfg_archive_device_scope_reuse_and_release_match_original_children(self):
        import cupy as cp
        from lqcdweave.contracts.correlator.api import TensorPreparation, describe_coordinates
        from lqcdweave.contraction.planning.api import build_cfg_plan, audit_cfg_plan
        from lqcdweave.contraction.execution import cfg_runtime
        from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value

        for partial_reuse in (False,True):
            with self.subTest(partial_reuse=partial_reuse):
                plan,values=fixture(k=800)
                values={key:value+np.complex64(10*(index+1))
                        for index,(key,value) in enumerate(values.items())}
                if partial_reuse:
                    # Identical physical recipes genuinely reuse one row's
                    # contractions within the first point; no mask is forged.
                    values['1a1']=values['1a0'].copy()
                preparations={key:SourcePreparation((),recipe=TensorPreparation('constant',
                    constants=(ArrayConstant.from_array(value),))) for key,value in values.items()}
                plan=replace(plan,preparations=preparations)
                points,_=describe_coordinates(time_pairs=[{0:0},{0:0}])
                def archive(saved):
                    cfg=build_cfg_plan(saved,None,points.points,{},preparations,
                        plan.state.problem.specs,prefetch_host_bytes=0)
                    audit_cfg_plan(cfg)
                    return decode_plan_value(encode_plan_value(cfg))
                grouped=archive(plan)
                ordinary=archive(replace(plan,cartesian_reduction_groups=()))
                observed=[];stores=[];original=cfg_runtime._CfgBlockExecution
                def observe(local_plan,*args,**kwargs):
                    execution=original(local_plan,*args,**kwargs)
                    observed.append((local_plan,execution.execution_tables.cartesian_groups))
                    stores.append(kwargs['arrays'])
                    return execution
                @contextmanager
                def factory(configuration):
                    # Recipes prepare/upload these constants inside the real
                    # private device scope. No physical IO is required.
                    yield ArrayLoader({},cp)
                def run(cfg):
                    outputs=[]
                    report=cfg_runtime.execute_cfg_plan(cfg,['cfg'],factory,
                        lambda name,index,result:outputs.append(result) if index is not None else None,
                        backend=cp,prefetch_host_bytes=0)
                    return report,outputs
                # This wrapper only observes the actual constructed tables and
                # shared store. Device scope, packing, uploads, downloads,
                # binding, numerical replay and release are all unmodified.
                with patch.object(cfg_runtime,'_CfgBlockExecution',side_effect=observe):
                    report,actual=run(grouped)
                reference_report,reference=run(ordinary)
                self.assertEqual(report['completed'],('cfg',))
                self.assertEqual(reference_report['completed'],('cfg',))
                self.assertGreater(report['device']['output_download_calls'],0)
                self.assertEqual(len(actual),2);self.assertEqual(len(observed),2)
                self.assertEqual(len(observed[0][1]),0 if partial_reuse else 1)
                self.assertEqual(observed[0][0].cartesian_reduction_groups,observed[0][1])
                self.assertEqual(observed[1][1],())
                self.assertTrue(all(action.reused for action in observed[1][0].state.actions
                                    if action.kind=='contract'))
                if partial_reuse:
                    flags=[action.reused for action in observed[0][0].state.actions if action.kind=='contract']
                    self.assertTrue(any(flags));self.assertFalse(all(flags))
                self.assertTrue(any(point.cartesian_reduction_device_bytes for point in observed[1][0].memory))
                self.assertTrue(all(not store for store in stores))
                expected=tuple(np.einsum('i,i->',*(values[key] for key in term.inputs),
                    dtype=np.complex128).astype(np.complex64) for term in plan.state.problem.terms)
                for result,baseline in zip(actual,reference,strict=True):
                    self.assertEqual(tuple(value.tobytes() for value in result),
                                     tuple(value.tobytes() for value in baseline))
                    self.assertEqual(tuple(value.tobytes() for value in result),
                                     tuple(value.tobytes() for value in expected))

    def test_public_storage_current_pointers_metadata_and_packing_match_individual_bytes(self):
        import cupy as cp
        for dtype in (np.float32,np.complex64):
            plan,inputs=fixture(k=800,dtype=dtype);plain=replace(plan,cartesian_reduction_groups=())
            cache=ExecutionBindingCache();held=[];previous=None
            for scale in (1,2,3):
                values={key:cp.asarray(value*scale) for key,value in inputs.items()}
                if scale==2:values={key:value[::-1] for key,value in values.items()}
                if scale==3:values={key:cp.broadcast_to(value[:1],value.shape) for key,value in values.items()}
                held.append(values);pointers=tuple(value.data.ptr for value in values.values())
                if previous is not None:self.assertNotEqual(previous,pointers)
                previous=pointers
                actual=replay_scene_storage(plan,ArrayLoader(values,cp),backend=cp,io_overlap=False,execution_cache=cache)
                expected=replay_scene_storage(plain,ArrayLoader(values,cp),backend=cp,io_overlap=False)
                self.assertEqual(tuple(cp.asnumpy(v).tobytes() for v in actual),tuple(cp.asnumpy(v).tobytes() for v in expected))
            changed=replace(plan,cartesian_reduction_groups=())
            replay_scene_storage(changed,ArrayLoader(values,cp),backend=cp,execution_cache=cache,io_overlap=False)
            self.assertIs(cache._binding[0],changed)
            key=next(iter(values));bad=dict(values);bad[key]=cp.empty((799,),dtype=dtype)
            with self.assertRaisesRegex(ValueError,'mismatched tensor'):
                replay_scene_storage(plan,ArrayLoader(bad,cp),backend=cp,execution_cache=cache,io_overlap=False)
            bad=dict(values);bad[key]=values[key].astype(np.complex128 if dtype==np.complex64 else np.float64)
            with self.assertRaisesRegex(ValueError,'mismatched tensor'):
                replay_scene_storage(plan,ArrayLoader(bad,cp),backend=cp,execution_cache=cache,io_overlap=False)

    def test_cfg_direct_prepared_deferred_owners_and_partial_child_fallback(self):
        import cupy as cp
        for m,n,prepared in ((1,3,'a'),(2,1,'b')):
            plan,inputs=fixture(m=m,n=n,k=800,prepared_side=prepared)
            state=replace(plan.state,actions=tuple(replace(a,reused=True) if a.kind=='load' else a for a in plan.state.actions))
            plan=replace(plan,state=state)
            tables=scene_runtime.bind_scene_execution(state,plan.kernel_batches,cp,events=(),shared_residency=True,
                cartesian_reduction_groups=plan.cartesian_reduction_groups)
            reference=scene_runtime.bind_scene_execution(state,plan.kernel_batches,cp,events=(),shared_residency=True)
            self.assertEqual(tables.cartesian_groups,plan.cartesian_reduction_groups)
            binding={key:i for i,key in enumerate(state.catalog)}
            for scale in (1,2):
                values={key:cp.asarray(value*scale) for key,value in inputs.items()};store={}
                for key,value in values.items():
                    owner=_ResultOwner(value[None,:])
                    layout=bind_result_views(owner.value.shape,value.dtype,((key,(0,slice(None)),()),))[0][1]
                    store[binding[key]]=_DeferredResultView(owner,layout)
                expected=reference.program.run(execution(plan,reference,values),_NodeView(binding,{binding[k]:v for k,v in values.items()}))
                outputs=tables.program.run(execution(plan,tables,values),_NodeView(binding,store))
                self.assertEqual(tuple(cp.asnumpy(v).tobytes() for v in outputs),tuple(cp.asnumpy(v).tobytes() for v in expected))
                member=plan.kernel_batches[-1].action_indices[-1];actions=list(state.actions)
                actions[member]=replace(actions[member],reused=True,seconds=0.)
                partial=replace(state,actions=tuple(actions))
                fallback=scene_runtime.bind_scene_execution(partial,plan.kernel_batches[:1],cp,events=(),shared_residency=True,
                    producer_batches=plan.kernel_batches,cartesian_reduction_groups=plan.cartesian_reduction_groups)
                self.assertEqual(fallback.cartesian_groups,())
                kept=store[binding[actions[member].tensor]];reused={binding[k]:v for k,v in values.items()}
                reused[binding[actions[member].tensor]]=kept
                output=fallback.program.run(execution(replace(plan,state=partial),fallback,values),_NodeView(binding,reused))
                self.assertIs(reused[binding[actions[member].tensor]],kept)
                self.assertEqual(tuple(cp.asnumpy(v).tobytes() for v in output),tuple(cp.asnumpy(v).tobytes() for v in expected))


if __name__=='__main__':unittest.main()
