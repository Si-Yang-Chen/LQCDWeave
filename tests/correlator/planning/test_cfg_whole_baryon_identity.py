"""Whole and ordered baryon cfg values keep one owner across proof occurrences."""
from contextlib import contextmanager
from dataclasses import replace
import os
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, AxisPattern, BaryonJoinSpec, ContractionProblem,
    ContractionTerm, CoreComponent, SourcePreparation, SourceReadRequest,
    SourceSpec, StorageBlock, TensorAxisMetadata, TensorCoreSpec,
    TensorPreparation, TensorSpec, VertexKernelDescription, describe_coordinates,
)
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.planning.api import plan_direct_execution
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.contraction.execution.api import execute_cfg_plan
from lqcdweave.storage.contraction_plans.api import save_cfg_plan, load_cfg_plan
from correlator.planning.test_planning_api import _expanded


def fixture(dtype='complex64', backend='numpy'):
    rng = np.random.default_rng(6284)
    entries = (((0, 1, 2, .7+.2j), (3, 2, 1, -.4j)),
               ((1, 2, 0, -.3+.6j),), ((2, 0, 3, .5-.2j),))
    shapes = {**{f'S{i}': (4, 4, 4) for i in range(3)},
              **{f'E{i}': (2, 2, 2) for i in range(3)},
              'P': (4, 2, 2), 'Q': (4, 2, 2), 'R': (4, 2)}
    values = {key: (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(dtype)
              for key, shape in shapes.items()}
    for i, row in enumerate(entries):
        values[f'S{i}'].fill(0)
        for a, b, c, coefficient in row:
            values[f'S{i}'][a, b, c] = coefficient
    specs = {key: TensorSpec(value.shape, value.dtype) for key, value in values.items()}
    core = TensorCoreSpec('baryon', ('s0','s1','s2','v0','v1','v2'),
        TensorSpec((4,4,4,2,2,2),dtype), tuple(CoreComponent((f'S{i}', f'E{i}'),
        coefficient, entries[i]) for i, coefficient in enumerate((1+.2j, -.4+.3j, .6-.1j))))
    changed_weight = replace(core, components=(replace(core.components[0], coefficient=.3-.8j), *core.components[1:]))
    changed_sparse = replace(core, components=(replace(core.components[0],
        sparse_spin=((0,1,2,-.2+.9j), (3,2,1,-.4j))), *core.components[1:]))
    cores = (core, core, core, changed_weight, changed_sparse, core)
    ports = (('P','Q','R'), ('P','Q','R'), ('Q','P','R'),
             ('P','Q','R'), ('P','Q','R'), ('P','Q','R'))
    labels=((0,3,6),(1,4,7),(2,5))
    terms = tuple(ContractionTerm(inputs, AxisPattern(
        tuple(tuple(axis+(100 if i==1 else 0) for axis in row) for row in labels),
        tuple(axis+(100 if i==1 else 0) for axis in ((7,6) if i==5 else (6,7)))))
        for i,inputs in enumerate(ports))
    joins = tuple(BaryonJoinSpec(i, f'baryon-vertex:{i}', 'S0', 'E0',
        tuple(axis+(100 if i==1 else 0) for axis in (0,1,2)),
        tuple(axis+(100 if i==1 else 0) for axis in (3,4,5)),
        sparse_spin=entries[0], core=item,
        kernel_description=VertexKernelDescription('baryon','ordered-join-v1'))
        for i, item in enumerate(cores))
    problem = ContractionProblem(specs, terms,
        {key: SourceSpec(key, value.nbytes) for key,value in values.items()},
        baryon_joins=joins, backend=backend)
    metadata = {key: TensorAxisMetadata(tuple(f'axis{i}' for i in range(value.ndim)),
        value.shape, value.dtype, 'data:'+key, 'full')
        for key,value in values.items() if not key.startswith('S')}
    blocks = {key: StorageBlock('data:'+key,'full',values[key].nbytes,host_bytes=values[key].nbytes)
              for key in metadata}
    preps = {key: SourcePreparation((),dtype=dtype,recipe=TensorPreparation('constant',
        constants=(ArrayConstant.from_array(value),))) if key.startswith('S') else
        SourcePreparation((key,),dtype=dtype,recipe=TensorPreparation('identity',(key,)))
        for key,value in values.items()}
    requests = {key: SourceReadRequest(key,'elemental',(0,),(),record,
        coordinate_arguments=((0,'0',0,None),)) for key,record in metadata.items()}
    selected = plan_direct_execution(_expanded(problem,top_blocks=blocks,top_preparations=preps),
        memory_limit=1<<27,host_memory_limit=1<<27).plan
    return selected, preps, requests, metadata, values, cores, ports


def test_native_whole_baryon_identity_preserves_every_value_parameter():
    selected,preps,*_=fixture()
    action=next(item for item in selected.state.actions if item.core is not None)
    occurrence=replace(action,term_index=91,positions=(3,2,1))
    operands=tuple(range(len(action.operands)))
    canonical=cfg_plan._computation_identity(action,operands)
    assert cfg_plan._computation_identity(occurrence,operands)==canonical
    assert cfg_plan._computation_identity(action,operands[::-1])!=canonical
    from lqcdweave.contraction.planning.canonical_operations import result_name
    axes=action.equation
    renamed=AxisPattern(tuple(tuple(axis+100 for axis in row) for row in axes.inputs),
                        tuple(axis+100 for axis in axes.output))
    equivalent=(replace(occurrence,join_id='other-proof-name'),
                replace(occurrence,equation=renamed))
    for item in equivalent:
        assert result_name(item)==result_name(action)
        assert cfg_plan._computation_identity(item,operands)==canonical
    changed=replace(action.core,components=(replace(action.core.components[0],coefficient=.1j),*action.core.components[1:]))
    sparse=replace(action.core,components=(replace(action.core.components[0],sparse_spin=((1,0,2,.3j),)),*action.core.components[1:]))
    for variant in (replace(action,core=changed),replace(action,core=sparse),
                    replace(action,core=replace(action.core,components=action.core.components[::-1])),
                    replace(action,equation=AxisPattern(action.equation.inputs,action.equation.output[::-1])),
                    replace(action,equation=AxisPattern(tuple(
                        tuple(axes.inputs[0][0] if axis==axes.inputs[0][1] else axis for axis in row)
                        for row in axes.inputs),axes.output)),
                    replace(action,kernel_path=((0,1),)),
                    replace(action,kernel_description=None)):
        assert cfg_plan._computation_identity(variant,operands)!=canonical
    ordered=replace(action,core=None)
    ordered_identity=cfg_plan._computation_identity(ordered,operands)
    assert cfg_plan._computation_identity(replace(ordered,term_index=91,positions=(3,2,1)),operands)==ordered_identity
    assert ordered_identity!=canonical
    assert cfg_plan._computation_identity(replace(ordered,sparse_spin=((1,0,2,.3j),)),operands)!=ordered_identity
    with pytest.raises(ValueError, match='wide accumulation requires a dense contraction'):
        replace(action,wide_accumulation=True)
    actions=(action,occurrence,*equivalent)
    identities=cfg_plan._bind_computation_identities(actions,debug=False)
    assert tuple(bind(operands) for bind in identities)==(canonical,)*len(actions)
    assert cfg_plan._bind_local_computation_reuses(actions,identities,
        tuple(cfg_plan._bind_operands(a.operands) for a in actions),preps,debug=False)==(False,True,True,True)
    class ExtendedAction(Action):
        pass
    opaque=ExtendedAction(**{field:getattr(action,field) for field in action.__dataclass_fields__})
    assert cfg_plan._computation_identity(opaque,operands)!=cfg_plan._computation_identity(replace(opaque,term_index=91),operands)


@pytest.mark.parametrize('debug',(False,True))
def test_actual_input_aliases_preserve_selected_physical_baryon_owners(debug):
    selected,preps,requests,metadata,*_=fixture()
    # P and Q are separate logical inputs in the saved plan, but these actual
    # declarations explicitly alias the same current physical source.
    requests['Q']=replace(requests['Q'],metadata=replace(metadata['Q'],
        data_identity=metadata['P'].data_identity))
    points,_=describe_coordinates()
    plan=cfg_plan.build_cfg_plan(selected,None,points.points,requests,preps,
        selected.state.problem.specs,debug=debug)
    instance,=plan.instances
    bindings=dict(zip(plan.templates[0].keys,map(int,instance.node_ids.array()),strict=True))
    assert bindings['P']==bindings['Q']
    first=next(a for a in selected.state.actions if a.core is not None and a.term_index==0)
    swapped=next(a for a in selected.state.actions if a.core is not None and a.term_index==2)
    assert first.tensor!=swapped.tensor
    actual_inputs=lambda action:tuple(bindings[key] for key in action.operands)
    assert cfg_plan._computation_identity(first,actual_inputs(first))==cfg_plan._computation_identity(swapped,actual_inputs(swapped))
    assert bindings[first.tensor]!=bindings[swapped.tensor]
    assert plan.statistics['compute_nodes']==len({a.tensor for a in selected.state.actions
        if a.kind not in ('load','release')})
    cfg_plan.audit_cfg_plan(plan)


@pytest.mark.parametrize('backend',('numpy','cupy'))
@pytest.mark.parametrize('dtype',('complex64','complex128'))
@pytest.mark.parametrize('debug',(False,True))
def test_repeated_whole_baryon_current_cfg_archive_and_numerical_ports(tmp_path,backend,dtype,debug):
    if backend=='cupy':
        if os.environ.get('ED_TEST_GPU')!='1': pytest.skip('real GPU gate only')
        xp=pytest.importorskip('cupy')
        xp.zeros(1).sum().item()
    else:
        xp=np
    selected,preps,requests,metadata,values,cores,ports=fixture(dtype,backend)
    points,_=describe_coordinates(time_pairs=[{0:0},{0:1},{0:0}])
    # Cover repeated coordinates both within a window and across boundaries.
    for window in (4_000_000,1):
        with patch.object(cfg_plan,'_CFG_BINDING_WINDOW_SLOTS',window):
            plan=cfg_plan.build_cfg_plan(selected,None,points.points,requests,preps,
                selected.state.problem.specs,debug=debug)
        cfg_plan.audit_cfg_plan(plan)
        position=plan.templates[0].keys.index(selected.state.results[0])
        first,last=(int(plan.instances[i].node_ids.array()[position]) for i in (0,2))
        assert (first==last)==(window>1)
        for instance in plan.instances:
            bound=set(map(int,instance.node_ids.array()))
            released=set(instance.releases)
            if instance.tensor_releases is not None:
                released.update(map(int,instance.tensor_releases.array()[:,1]))
            assert released<=bound
        path=tmp_path/f'whole-baryon-{window}.json.gz';save_cfg_plan(path,plan)
        with patch.object(cfg_plan,'build_cfg_plan',side_effect=AssertionError('archive replanning')):
            restored=load_cfg_plan(path)
        collected={}
        class Reader:
            def __init__(self,key,configuration): self.key,self.cfg=key,configuration
            def describe_block(self,time):
                return replace(metadata[self.key],data_identity=metadata[self.key].data_identity+':'+self.cfg)
            def block(self,time):
                return (values[self.key]*((1. if self.cfg=='a' else 1.3)+.1*time)).astype(dtype)
        @contextmanager
        def factory(configuration):
            yield restore_input_loader(restored.requests,
                {key:Reader(key,configuration) for key in metadata},restored.preparations,
                restored.specs,configuration=configuration)
        execute_cfg_plan(restored,('a','b'),factory,
            lambda configuration,index,result:collected.update({(configuration,index):result}),
            backend=xp,debug=debug,prefetch_host_bytes=0)
        for configuration in ('a','b'):
            for index,time in enumerate((0,1,0)):
                current={key:Reader(key,configuration).block(time).astype('complex128') for key in metadata}
                expected=[]
                for scene,(core,legs) in enumerate(zip(cores,ports,strict=True)):
                    total=0
                    for component in core.components:
                        spin=np.zeros((4,4,4),dtype='complex128')
                        for a,b,c,weight in component.sparse_spin: spin[a,b,c]=weight
                        total=total+component.coefficient*np.einsum('ijk,abc,iam,jbn,kc->mn',spin,
                            current[component.inputs[1]],*(current[key] for key in legs),optimize=True)
                    expected.append(total.T if scene==5 else total)
                assert len(collected[configuration,index])==len(expected)
                for actual,reference in zip(collected[configuration,index],expected,strict=True):
                    np.testing.assert_allclose(actual,reference,rtol=5e-5 if dtype=='complex64' else 5e-12,
                                               atol=5e-5 if dtype=='complex64' else 5e-12)
                assert not np.allclose(expected[0],expected[2])
                assert not np.allclose(expected[0],expected[3])
                assert not np.allclose(expected[0],expected[4])
