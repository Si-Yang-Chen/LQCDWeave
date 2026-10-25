"""Coordinate rebinding shares physical grouping without another path search."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, TensorPreparation, describe_coordinates,
)
from lqcdweave.contraction.planning.api import plan_coordinate_batches, solve_storage
from lqcdweave.contraction.planning import coordinate_schedule, gemm_groups
from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value
from lqcdweave.contraction.execution.storage_runtime import replay_scene_storage
from lqcdweave.contraction.execution import bound_program
from correlator.execution.test_cartesian_group_execution import fixture as cartesian_fixture, Backend as MatrixBackend
from correlator.execution.test_scalar_group_execution import Backend as ScalarBackend, ArrayLoader
from correlator.planning.test_scalar_reduction_groups import fixture as scalar_fixture
from correlator.planning.test_coordinate_inputs import fixture as physical_fixture


def static(kind='cartesian', count=2):
    if kind=='cartesian':
        plan,values=cartesian_fixture(count,k=800)
    else:
        plan,_=scalar_fixture(count,shape=(800,),dtype=np.complex64)
        values={key:np.full(spec.shape,i+1,spec.dtype)
                for i,(key,spec) in enumerate(plan.state.problem.specs.items())}
    values={key:value+np.complex64(10*(index+1)) for index,(key,value) in enumerate(values.items())}
    preps={key:SourcePreparation((),recipe=TensorPreparation('constant',
        constants=(ArrayConstant.from_array(value),))) for key,value in values.items()}
    plan=replace(plan,preparations=preps)
    coordinates,batch=describe_coordinates(time_pairs=[{0:0},{0:1}],batch=1)
    args=(plan,SimpleNamespace(term_ids=tuple(map(str,range(len(plan.state.problem.terms))))),
          coordinates,{},preps,plan.state.problem.specs)
    return args,dict(batch=batch,kernel_batching=True,target_runtime='cuda'),values


@pytest.mark.parametrize('kind,count',[('cartesian',2),('scalar',4)])
def test_static_joined_plans_group_archive_and_numerical_reference(kind,count):
    args,kwargs,values=static(kind,count)
    with patch.object(coordinate_schedule,'lower_dense_batches',wraps=coordinate_schedule.lower_dense_batches) as dense, patch(
            'lqcdweave.contraction.planning.path_planning.find_contraction_path',
            side_effect=AssertionError('fixed replay searched')):
        selected=plan_coordinate_batches(*args,**kwargs)
    assert dense.call_count==2
    backend=MatrixBackend() if kind=='cartesian' else ScalarBackend()
    for plan in selected.plans:
        assert not plan.retention_planned and not plan.retained_before and not plan.retained_after
        groups=plan.cartesian_reduction_groups if kind=='cartesian' else plan.scalar_reduction_groups
        assert len(groups)==1
        restored=decode_plan_value(encode_plan_value(plan))
        assert restored==plan
        inputs={key:prep.recipe.constants[0].array() for key,prep in restored.preparations.items()}
        # The CPU source interpreters implement the real reduction ABI, not
        # CUDA output-copy kernels. Use the ordinary output-copy fallback.
        with patch.object(bound_program,'bind_output_layout',return_value=None):
            actual=replay_scene_storage(restored,ArrayLoader(inputs,backend),backend=backend,io_overlap=False)
        expected=tuple(np.einsum('i,i->',*(values[key] for key in term.inputs),dtype=np.complex128).astype(np.complex64)
                       for term in args[0].state.problem.terms)
        assert tuple(value.tobytes() for value in actual)==tuple(value.tobytes() for value in expected)
    stats=selected.statistics
    assert stats['peak_device_bytes']==max(plan.peak_device_bytes for plan in selected.plans)
    key=kind+'_reduction_groups'
    assert stats[key]==(1,1)
    assert all(item[kind+'_reduction_lowering']['applied'] for item in stats['independent_kernel_lowering'])


@pytest.mark.parametrize('runtime',[None,'cpu','hip'])
def test_unknown_and_non_cuda_targets_keep_dense_but_disable_groups(runtime):
    args,kwargs,_=static()
    selected=plan_coordinate_batches(*args,**dict(kwargs,target_runtime=runtime))
    assert all(len(plan.kernel_batches)==2 for plan in selected.plans)
    assert all(not plan.scalar_reduction_groups and not plan.cartesian_reduction_groups for plan in selected.plans)


def test_false_does_not_call_any_physical_lowering():
    args,kwargs,_=static()
    with patch.object(coordinate_schedule,'lower_dense_batches',side_effect=AssertionError('dense called')), patch.object(
            coordinate_schedule,'lower_independent_kernel_groups',side_effect=AssertionError('group called')):
        selected=plan_coordinate_batches(*args,**dict(kwargs,kernel_batching=False))
    assert selected.statistics['independent_kernel_lowering']==()
    assert all(not plan.kernel_batches and not plan.cartesian_reduction_groups for plan in selected.plans)


def test_path_wrapper_and_coordinates_call_same_plan_service():
    args,kwargs,_=static()
    with patch.object(coordinate_schedule,'lower_independent_kernel_groups',wraps=gemm_groups.lower_independent_kernel_groups) as common:
        selected=plan_coordinate_batches(*args,**kwargs)
    assert common.call_count==2
    from lqcdweave.contracts.correlator.api import PlannedContraction
    base=plan_coordinate_batches(*args,**dict(kwargs,target_runtime=None)).plans[0]
    # A non-CUDA path exits before the service; no implicit runtime guess.
    path=PlannedContraction(None,base,None,{'kernel_batching':True},0.,'planned',
        base.memory_limit,base.host_memory_limit,id(base),base.cost_profile,0,0)
    assert gemm_groups.lower_independent_gemm_path(path,target_runtime=None) is path
    with patch.object(gemm_groups,'lower_independent_kernel_groups',wraps=gemm_groups.lower_independent_kernel_groups) as common:
        lowered=gemm_groups.lower_independent_gemm_path(path,target_runtime='cuda')
    assert common.call_count==1
    assert len(lowered.plan.cartesian_reduction_groups)==1
    assert lowered.prediction['cartesian_reduction_lowering']['applied']


def physical():
    raw,requests,preps,specs,_,metadata,readers,_=physical_fixture('cupy')
    template=solve_storage(raw.state.problem,raw.blocks,preps,memory_limit=1<<27,host_memory_limit=1<<27,
        deadline_seconds=1,max_states=100,root_width=1,branch_width=1,component_workers=1,kernel_batching=False).plan
    coords,batch=describe_coordinates(time_axes={'source':3,'sink':(0,1)},batch=1)
    return (template,SimpleNamespace(term_ids=('dynamic','static')),coords,requests,preps,specs),dict(
        batch=batch,kernel_batching=True,target_runtime='cuda',read_metadata=metadata),readers


def test_shared_raw_lifetimes_and_dynamic_lane_are_not_retention_or_scalar_groups():
    args,kwargs,readers=physical()
    selected=plan_coordinate_batches(*args,**kwargs)
    assert [(sum(e.kind=='read' for e in p.events),sum(e.kind=='release' for e in p.events))
            for p in selected.plans]==[(4,3),(3,4)]
    assert all(not p.retention_planned and not p.retained_before and not p.retained_after for p in selected.plans)
    from lqcdweave.contracts.correlator.api import BatchSpec
    wide=plan_coordinate_batches(*args,**dict(kwargs,batch=BatchSpec('full','sink')))
    plan=wide.plans[0]
    # The dynamic result keeps its lane. It must not be flattened into a scalar.
    assert any(plan.state.catalog[result].shape==(2,) for result in plan.state.results)
    assert not plan.scalar_reduction_groups and not plan.cartesian_reduction_groups
    assert sum(reader.reads for reader in readers.values())==0


def test_group_budget_uses_joined_peak_and_keeps_caller_reserves():
    args,kwargs,_=static()
    cap=150000  # Admits original two batches but not their combined packing owners.
    base=plan_coordinate_batches(*args,**dict(kwargs,target_runtime=None,memory_limit=cap))
    limited=plan_coordinate_batches(*args,**dict(kwargs,memory_limit=cap))
    assert all(not plan.cartesian_reduction_groups for plan in limited.plans)
    assert all(plan.peak_device_bytes<=cap for plan in limited.plans)
    assert any(item['cartesian_reduction_lowering']['resource_rejections']
               for item in limited.statistics['independent_kernel_lowering'])
    for original,plan in zip(base.plans,limited.plans,strict=True):
        assert plan.reserved_device_bytes==original.reserved_device_bytes
        assert plan.reserved_host_bytes==original.reserved_host_bytes


def test_auto_width_budget_and_allocator_reserve_survive_common_grouping():
    from correlator.planning.test_coordinate_allocator_reserve import template
    plan,coords,requests,preps,specs,metadata,readers=template()
    args=(plan,SimpleNamespace(term_ids=('loop','static')),coords,requests,preps,specs)
    kwargs=dict(read_metadata=metadata,memory_limit=16000,kernel_batching=True,target_runtime='cuda')
    base=plan_coordinate_batches(*args,**kwargs)
    reserved=plan_coordinate_batches(*args,**kwargs,allocator_reserve_bytes=4000)
    assert reserved.width<base.width
    assert any(not row['feasible'] for row in reserved.statistics['candidate_attempts'])
    assert len(reserved.statistics['independent_kernel_lowering'])==len(reserved.plans)
    assert all(plan.reserved_device_bytes==4000 and plan.peak_device_bytes<=16000 for plan in reserved.plans)
    assert reserved.statistics['peak_device_bytes']==max(plan.peak_device_bytes for plan in reserved.plans)
    assert sum(reader.reads for reader in readers.values())==0


def test_workflow_passes_only_existing_declared_target_runtime():
    from correlator.execution.test_coordinate_replay import selected
    from lqcdweave.contraction import correlator_coordinates as workflow
    calculation,readers=selected()
    for runtime in (None,'hip','cuda'):
        target=None if runtime is None else SimpleNamespace(runtime=runtime,name='declared-target')
        selected_calculation=replace(calculation,options=dict(calculation.options,execution_target=target))
        # CPU metadata fixture; runtime classification itself must not probe CUDA.
        with patch.object(workflow,'check_execution_target',return_value=np),patch.object(
                workflow,'plan_coordinate_batches',wraps=plan_coordinate_batches) as service:
            result=workflow.select_coordinate_replay(selected_calculation,time_axes={'source':3,'sink':(0,1)})
        assert service.call_args.kwargs['target_runtime']==runtime
        assert result.path is not None
    assert sum(reader.reads for reader in readers.values())==0
