"""Cfg lifetimes are built from one selected template without replaying it."""
import unittest
from unittest.mock import patch
import numpy as np
from test_structured_correlator import fixture,task
from lqcdweave.contracts.correlator.api import DiagramBindings,describe_coordinates
from lqcdweave.contracts.correlator.api import (
    ArrayConstant,ExecutionTarget,SourcePreparation,SourceReadRequest,
    SourceSpec,StorageBlock,TensorAxisMetadata,TensorPreparation,
)
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contraction.expansion.api import (
    describe_meson_matrix,bind_meson_matrix_catalog,describe_input_reads,
)
from lqcdweave.contraction.planning.api import plan_matrix_actions,CostProfile,build_cfg_plan


def selected_template():
    binding,elemental,propagator=fixture()
    bindings=DiagramBindings(vertex_map=binding,configuration='cfg',usedNe=3,
        propagator_map={'S^q':propagator,r'S^q_\mathrm{local}':propagator})
    expanded=bind_meson_matrix_catalog(describe_meson_matrix(task(structured=True,flavor='I0'),
        bindings=bindings),bindings=bindings)
    inputs=expanded.inputs
    plan=plan_matrix_actions(inputs.problem,inputs.matrix_kernels,inputs.blocks,inputs.preparations,
        profile=CostProfile(),memory_limit=1<<30,reserved_host_bytes=inputs.assembly_host_bytes)
    return expanded,plan,elemental,propagator


class CfgPlanTests(unittest.TestCase):
    def build(self,*,mode='memory'):
        expanded,plan,elemental,propagator=selected_template()
        points,_=describe_coordinates(time_pairs=[{0:1,1:0},{0:0,1:1},{0:1,1:0}])
        loader=expanded.inputs.loader
        requests=describe_input_reads(expanded.inputs)
        with (patch('lqcdweave.contraction.planning.storage.plan_scene_storage',
                    side_effect=AssertionError('replay')),
              patch.object(elemental,'block',side_effect=AssertionError('IO'))):
            result=build_cfg_plan(plan,expanded.output_recipe,points.points,requests,
                loader.preparations,loader.specs,output_mode=mode)
        return result

    def test_shared_template_packed_bindings_and_global_last_use(self):
        plan=self.build()
        self.assertEqual(len(plan.templates),1)
        self.assertEqual(len(plan.instances),3)
        self.assertLess(plan.statistics['compute_nodes'],9)
        self.assertGreaterEqual(plan.statistics['compute_nodes'],2)
        reads=[e.block for i in plan.instances for e in i.events if e.kind=='read']
        closes=[e.block for i in plan.instances for e in i.events if e.kind=='release']
        self.assertEqual(len(reads),len(set(reads)))
        self.assertCountEqual(reads,closes)
        self.assertTrue(any(np.any(i.reused.array()) for i in plan.instances[1:]))
        released=[key for i in plan.instances for key in i.releases]
        self.assertEqual(len(released),len(set(released)))

    def test_streaming_reserves_one_point_and_does_not_rearrange_compute(self):
        memory=self.build(); streaming=self.build(mode='dataset')
        self.assertEqual(memory.statistics['compute_nodes'],streaming.statistics['compute_nodes'])
        self.assertEqual(memory.output_host_bytes,3*streaming.output_host_bytes)
        self.assertLess(streaming.peak_host_bytes,memory.peak_host_bytes)

    def test_coordinate_roles_are_explicit(self):
        expanded,_,_,_=selected_template()
        requests=list(expanded.inputs.read_requests.values())
        prop=next(r for r in requests if r.family=='propagator' and r.arguments[0]!=r.arguments[1])
        self.assertEqual(len(prop.coordinate_arguments),2)
        actual=prop.at({'0':2,'1':1})
        for index,role,offset,period in prop.coordinate_arguments:
            self.assertEqual(actual.arguments[index],({'0':2,'1':1}[role]+offset)%period)

    def test_mutable_momentum_declarations_support_shared_cfg_reads(self):
        from dataclasses import replace
        expanded,plan,_,_=selected_template()
        requests={key:replace(request,conventions=(*request.conventions,('momenta',[[0,0,0],[1,0,0]])))
                  if request.family=='elemental' else request
                  for key,request in describe_input_reads(expanded.inputs).items()}
        points,_=describe_coordinates(time_pairs=[{0:1,1:0},{0:1,1:0}])
        result=build_cfg_plan(plan,expanded.output_recipe,points.points,requests,
                             expanded.inputs.loader.preparations,expanded.inputs.loader.specs)
        self.assertTrue(all(result.instances[1].reused.array()))

    def test_independent_audit_rejects_missing_release_and_unknown_event(self):
        from dataclasses import replace
        from lqcdweave.contraction.planning.api import audit_cfg_plan
        from lqcdweave.contracts.correlator.api import StorageEvent
        plan=self.build()
        last=plan.instances[-1]
        with self.assertRaisesRegex(ValueError,'leaves storage live'):
            audit_cfg_plan(replace(plan,instances=(*plan.instances[:-1],replace(last,releases=()))))
        with self.assertRaisesRegex(ValueError,'unknown cfg resource event'):
            audit_cfg_plan(replace(plan,instances=(replace(plan.instances[0],
                events=(StorageEvent(0,'unknown','0'),)),*plan.instances[1:])))

    @staticmethod
    def dense_fixture(backend='numpy'):
        from dataclasses import replace
        from correlator.planning.test_dense_batches import DenseBatchIntegrationTests
        from lqcdweave.contraction.planning.api import solve_storage
        plan=DenseBatchIntegrationTests().plan().plan
        if backend != plan.state.problem.backend:
            plan=solve_storage(replace(plan.state.problem,backend=backend),
                plan.blocks,plan.preparations,optimizer='anytime',kernel_batching=True,memory_limit=10**8,
                deadline_seconds=0.1,max_states=10000,root_width=1,branch_width=1, component_workers=1).plan
        preparations={}
        for key,spec in plan.state.problem.specs.items():
            values=np.arange(np.prod(spec.shape),dtype=np.float64).reshape(spec.shape)
            values=(values+1j*(values+1)).astype(spec.dtype)
            constant=ArrayConstant.from_array(values)
            preparations[key]=SourcePreparation((),recipe=TensorPreparation(
                'constant',constants=(constant,)))
        return plan,preparations,plan.state.problem.specs

    def test_dense_cpu_preserves_selected_batch_and_audits(self):
        from lqcdweave.contraction.planning.api import audit_cfg_plan
        plan,preparations,specs=self.dense_fixture()
        points,_=describe_coordinates()
        result=build_cfg_plan(plan,None,points.points,{},preparations,specs,debug=True)
        self.assertEqual(result.backend,'numpy')
        self.assertEqual(len(result.instances),1)
        self.assertEqual(result.instances[0].kernel_batches,plan.kernel_batches)
        self.assertEqual(result.peak_device_bytes,0)
        self.assertGreater(result.peak_host_bytes,0)
        self.assertEqual(audit_cfg_plan(result)['batches'],1)

    def test_offline_cupy_multi_coordinate_target_has_dual_budget_without_gpu(self):
        plan,preparations,specs=self.dense_fixture('cupy')
        points,_=describe_coordinates(time_pairs=[{0:0,1:0},{0:1,1:1}])
        target=ExecutionTarget('offline-cuda','cupy','cuda',TargetHardwareMemory(
            host_available_bytes=1<<20,device_total_bytes=1<<21,device_free_bytes=1<<21))
        result=build_cfg_plan(plan,None,points.points,{},preparations,specs,
            execution_target=target,
            target_summary={'host_memory_limit':1<<20,'device_memory_limit':1<<21},
            prefetch_device_bytes=4096,debug=True)
        self.assertEqual(result.backend,'cupy')
        self.assertEqual(result.execution_target,target)
        self.assertEqual(len(result.instances),2)
        self.assertGreater(result.peak_host_bytes,0)
        self.assertGreater(result.peak_device_bytes,0)
        self.assertLessEqual(result.peak_device_bytes+result.prefetch_device_bytes,1<<21)

    def test_batch_membership_accepts_equal_copies_but_rejects_changed_semantics(self):
        from dataclasses import replace
        plan, preparations, specs = self.dense_fixture()
        points, _ = describe_coordinates()
        cfg = build_cfg_plan(plan, None, points.points, {}, preparations, specs)
        instance = cfg.instances[0]
        batch = instance.kernel_batches[0]
        copied = replace(batch)
        self.assertIsNot(copied, batch)
        self.assertEqual(replace(cfg, instances=(replace(instance, kernel_batches=(copied,)),)), cfg)
        for changed in (replace(batch, packing_bytes=batch.packing_bytes+1),
                        replace(batch, seconds=batch.seconds+1),
                        replace(batch, action_indices=tuple(i+1000 for i in batch.action_indices))):
            with self.assertRaisesRegex(ValueError, 'batch absent'):
                replace(cfg, instances=(replace(instance, kernel_batches=(changed,)),))
        reused = instance.reused.array().copy()
        reused[batch.action_indices[0]] = True
        with self.assertRaisesRegex(ValueError, 'reused Action'):
            replace(cfg, instances=(replace(instance, reused=ArrayConstant.from_array(reused)),))

    def test_partial_batch_reuse_falls_back_and_keeps_numeric_mask(self):
        from dataclasses import replace
        plan,preparations,specs=self.dense_fixture()
        raw='raw-a0'
        spec=specs['a0']
        amount=int(np.prod(spec.shape))*spec.dtype.itemsize
        metadata=TensorAxisMetadata(('i','k'),spec.shape,spec.dtype,'a0-data','a0-selection')
        request=SourceReadRequest('a-reader','elemental',(0,),(),metadata,
                                  coordinate_arguments=((0,'0',0,None),))
        preparations['a0']=SourcePreparation((raw,),recipe=TensorPreparation(
            'identity',inputs=(raw,)))
        sources=dict(plan.state.problem.sources)
        sources['a0']=SourceSpec(raw,amount,host_region_bytes=amount,
                                 snapshot_bytes=amount,read_bytes=amount)
        problem=replace(plan.state.problem,sources=sources)
        plan=replace(plan,state=replace(plan.state,problem=problem),
                     blocks={raw:StorageBlock('a0-data','a0-selection',amount,host_bytes=amount)})
        points,_=describe_coordinates(time_pairs=[{0:0,1:0},{0:1,1:1}])
        result=build_cfg_plan(plan,None,points.points,{raw:request},preparations,specs,debug=True)
        ordinary=build_cfg_plan(plan,None,points.points,{raw:request},preparations,specs)
        self.assertEqual(replace(ordinary,statistics={}),replace(result,statistics={}))
        self.assertEqual(len(result.instances[0].kernel_batches),1)
        self.assertEqual(result.instances[1].kernel_batches,())
        actions=result.templates[0].plan.state.actions
        second=result.instances[1]
        non_reused={action.tensor for action,reused in zip(actions,second.reused.array(),strict=True)
                    if not reused}
        reused={action.tensor for action,reused in zip(actions,second.reused.array(),strict=True)
                if reused}
        self.assertIn('a0',non_reused)
        self.assertTrue({'a1','b0','b1'} <= reused)
        compute_mask={action.term_index:bool(reused_flag) for action,reused_flag in zip(
            actions,second.reused.array(),strict=True) if action.kind=='contract'}
        self.assertEqual(compute_mask,{0:False,1:False,2:True,3:True})

    def test_cfg_checks_host_and_device_caps_and_describes_baryon_recipe(self):
        from dataclasses import replace
        plan,preparations,specs=self.dense_fixture()
        points,_=describe_coordinates()
        with self.assertRaisesRegex(MemoryError,'host bytes'):
            build_cfg_plan(replace(plan,memory_limit=1),None,points.points,{},preparations,specs)

        gpu_plan,gpu_preps,gpu_specs=self.dense_fixture('cupy')
        gpu_target=ExecutionTarget('tiny-cuda','cupy','cuda',TargetHardwareMemory(
            host_available_bytes=1<<20,device_total_bytes=1,device_free_bytes=1))
        with self.assertRaisesRegex(MemoryError,'device bytes'):
            build_cfg_plan(gpu_plan,None,points.points,{},gpu_preps,gpu_specs,
                           execution_target=gpu_target)

        from correlator.planning.test_planning_api import _expanded
        from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem
        from lqcdweave.contraction.planning.api import plan_direct_execution
        expanded=_expanded(sparse_spin_problem())
        baryon=plan_direct_execution(expanded,memory_limit=1<<30).plan
        baryon_preps={}
        for key,spec in expanded.inputs.problem.specs.items():
            constant=ArrayConstant.from_array(np.ones(spec.shape,dtype=spec.dtype))
            baryon_preps[key]=SourcePreparation((),recipe=TensorPreparation(
                'constant',constants=(constant,)))
        joins=[action for action in baryon.state.actions if action.kind=='baryon_join']
        self.assertEqual(len(joins),1)
        baryon=replace(baryon,preparations=baryon_preps)
        baryon_cfg=build_cfg_plan(baryon,expanded.output_recipe,expanded.coordinates.points,
            {},baryon_preps,expanded.inputs.problem.specs,debug=True)
        self.assertEqual(baryon_cfg.statistics['compute_nodes'],len(baryon.state.results))


if __name__=='__main__': unittest.main()
