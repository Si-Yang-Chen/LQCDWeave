"""Thin wide integer reductions: precision, admission and fixed replay."""
from dataclasses import replace
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionTerm, ContractionProblem, SourceSpec, SourcePreparation,
    TensorSpec, FactorizedWideReduction,
)
from lqcdweave.contraction.planning.api import ContractionState, CostProfile
from lqcdweave.contraction.planning.canonical_operations import describe, apply_description
from lqcdweave.contraction.planning.factorized_reduction import (
    lower_factorized_reductions, replay_factorized_action,
)
from lqcdweave.contraction.planning.storage import plan_scene_storage
from lqcdweave.contraction.execution.operations.contract import bind_dense_action
from lqcdweave.storage.contraction_plans.api import _encode_plan_value, _decode_plan_value


def fixture(*, barriers=True):
    shapes=((2,65),(3,65),(7,2),(11,3))
    specs={str(i):TensorSpec(shape,np.complex64) for i,shape in enumerate(shapes)}
    term=ContractionTerm(tuple(specs),'ak,bk,ca,db->cd')
    axes=term.subscripts.inputs
    if barriers:
        term=replace(term,sum_barriers=(((axes[0][1],),(axes[0][0],axes[1][0])),))
    problem=ContractionProblem(specs,(term,),{key:SourceSpec(key,0,read_bytes=0) for key in specs},backend='cupy')
    state=ContractionState.initial(problem);profile=CostProfile()
    for key in specs:state=state.load(key,profile,1<<25)
    for pair in ((0,2),(0,1),(0,1)):
        action,spec,workspace,output,positions=describe(state,0,pair,profile=profile)
        state=apply_description(state,action,spec,workspace,output,positions,memory_limit=1<<25)
    for key in sorted(state.loaded-set(state.results)):state=state.release(key)
    preps={key:SourcePreparation((),dtype='complex64') for key in specs}
    plan=plan_scene_storage(state,{},preps,profile=profile,memory_limit=1<<25,host_memory_limit=1<<25)
    return plan,profile


def tiled_fixture(*, loop=False, tile=4):
    """Generic non-spin geometry with coefficient-only cancelling sums."""
    from lqcdweave.contraction.planning.factorized_reduction import describe_factorized_action
    shapes = ((3,7),(5,7),(2,3,3),(2,5,2)) if not loop else ((3,7),(3,7),(2,3,3))
    equation = 'ak,bk,cax,dby->cd' if not loop else 'ak,ak,cax->c'
    specs={str(i):TensorSpec(shape,'complex64') for i,shape in enumerate(shapes)}
    term=ContractionTerm(tuple(specs),equation)
    axes=term.subscripts.inputs;spatial=(axes[0][-1],)
    coupled=tuple(dict.fromkeys(a for labels in axes[2:] for a in labels[1:]))
    barriers=((spatial,coupled),*((((a,),(*spatial,*(b for b in coupled if b!=a))))
                                for a in coupled if a not in spatial))
    term=replace(term,sum_barriers=barriers)
    problem=ContractionProblem(specs,(term,),{k:SourceSpec(k,0) for k in specs},backend='cupy')
    state=ContractionState.initial(problem);profile=CostProfile()
    for key in specs:state=state.load(key,profile,1<<25)
    action,_,_,_=describe_factorized_action(state,0,tuple(range(len(shapes))),(0,1),
        profile=profile,output_order=term.subscripts.output,tile_elements=tile)
    completed=replay_factorized_action(state,action,profile)
    return state,completed,action,profile


class FactorizedWideTests(unittest.TestCase):
    def test_internal_loop_owner_stays_wide_until_published_field_cast(self):
        from lqcdweave.contracts.correlator.api import OutputNode, OutputRecipe, TermContribution, describe_coordinates
        from lqcdweave.contraction.planning.optimizers.session import solve_storage
        from lqcdweave.contraction.execution.api import replay
        from lqcdweave.contraction.results.api import bind_output_assembly
        specs = {key: TensorSpec((1, 2), 'complex64') for key in ('P0', 'V0', 'P1', 'V1')}
        specs['C'] = TensorSpec((1, 1, 2), 'complex64')
        term = ContractionTerm(('P0', 'V0', 'C'), 'ak,ak,cax->c')
        a, k = term.subscripts.inputs[0]
        x = term.subscripts.inputs[2][-1]
        barriers = (((k,), (a, x)), ((a,), (k, x)), ((x,), (k, a)))
        terms = (replace(term, sum_barriers=barriers),
                 replace(term, inputs=('P1', 'V1', 'C'), sum_barriers=barriers))
        problem = ContractionProblem(specs, terms,
            {key: SourceSpec(key, 0, read_bytes=0) for key in specs})
        selected = solve_storage(problem, {}, {key: SourcePreparation(()) for key in specs},
            memory_limit=1 << 20, host_memory_limit=1 << 20, component_workers=1,
            root_width=1, branch_width=1, include_ties=False, deadline_seconds=2, max_states=100)
        arrays = {'P0': np.array([[4096+2**-11, 0]], 'complex64'),
                  'V0': np.array([[4096+2**-11, 0]], 'complex64'),
                  'P1': np.array([[4096, 0]], 'complex64'),
                  'V1': np.array([[4096+2**-10, 0]], 'complex64'),
                  'C': np.array([[[1, 0]]], 'complex64')}
        values = replay(selected.state, lambda key, _source: arrays[key])
        self.assertTrue(all(value.dtype == np.dtype('complex128') for value in values))
        coordinates, _ = describe_coordinates()
        leaves = tuple(OutputNode('term', TermContribution(str(i), sign)) for i, sign in enumerate((1, -1)))
        loop = OutputNode('add', children=leaves)
        root = OutputNode('dict', keys=('loop', 'disconnected'), children=(loop,
            OutputNode('multiply', children=(OutputNode('constant', .5), loop))))
        recipe = OutputRecipe(root, ('0', '1'), coordinates, dtype='complex64', intermediate_dtype='complex128')
        actual = bind_output_assembly(recipe)(values)
        reference = sum(sign*np.einsum('ak,ak,cax->c', *(arrays[key].astype('complex128') for key in t.inputs))
                        for sign, t in zip((1, -1), terms, strict=True))
        self.assertGreater(abs(reference[0]), 2e-8)
        for key, weight in (('loop', 1), ('disconnected', .5)):
            self.assertEqual(actual[key].dtype, np.dtype('complex64'))
            np.testing.assert_array_equal(actual[key], (reference*weight).astype('complex64'))
        # The previous cast loses a term's low bits before the shared R node.
        old = bind_output_assembly(recipe)(tuple(value.astype('complex64') for value in values))
        self.assertGreater(abs(old['loop'][0]-reference[0]), 2e-8)

    def test_wide_output_identity_and_old_missing_field_semantics(self):
        from lqcdweave.contraction.planning.factorized_reduction import factorized_result_name
        from lqcdweave.contraction.planning.factorized_reduction import describe_factorized_action
        from lqcdweave.contraction.planning.paths import rebind_scene_path
        before, completed, action, profile = tiled_fixture(loop=True)
        self.assertEqual(action.core.result_dtype, 'complex128')
        self.assertEqual(completed.catalog[action.tensor].dtype, np.dtype('complex128'))
        legacy = replace(action.core, output_dtype=None)
        payload = _encode_plan_value(legacy)
        payload['value'].pop('output_dtype')
        restored = _decode_plan_value(payload)
        self.assertEqual(restored.result_dtype, 'complex64')
        self.assertEqual(restored, legacy)
        self.assertNotEqual(factorized_result_name(action.operands, legacy), action.tensor)
        old_action, _, _, _ = describe_factorized_action(before, 0, action.positions,
            restored.large_operands, profile=profile, output_order=restored.equation.output,
            tile_elements=restored.tile_elements, output_dtype=restored.output_dtype)
        self.assertEqual(old_action.core, restored)
        old_state = replay_factorized_action(before, old_action, profile)
        rebound = rebind_scene_path(old_state, before.problem,
            {key: key for key in before.problem.sources}, profile=profile, memory_limit=1 << 25)
        self.assertEqual(rebound.catalog[rebound.results[0]].dtype, np.dtype('complex64'))

    def test_owner_core_matching_requires_actual_reduction_trigger_coverage(self):
        from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
        from lqcdweave.contraction.planning.optimizers.selected_path import _matches
        before, _, _, _ = tiled_fixture(loop=True)
        term = before.problem.terms[0]
        a, k = term.subscripts.inputs[0]
        x = term.subscripts.inputs[2][-1]
        # Both whole-Term contractions close a wide boundary, but only the
        # first declares the core's k reduction as a coupled wide trigger.
        other = replace(term, sum_barriers=(((x,), (a, k)), ((x,), (a, k))))
        problem = replace(before.problem, terms=(term, other))
        space = IntegerSpace(problem, profile=CostProfile())
        operation = next(op for op, _ in space.candidates(lambda: None)
                         if type(space.registry.calculations[op.calculation].recipe.core) is FactorizedWideReduction)
        initial = ContractionState.initial(problem)
        matches = _matches(initial.pending, (0, 1), operation.calculation, space.registry,
                           problem=problem, catalog=initial.catalog)
        self.assertEqual([scene for scene, _ in matches], [0])

    def test_independent_materialization_keeps_terminal_output_and_repeated_roles(self):
        from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
        from lqcdweave.contraction.planning.optimizers.selected_path import materialize_dense_path
        for repeated in (False,True):
            if repeated:
                specs={'z':TensorSpec((3,7),'complex64'),'a':TensorSpec((2,4,3,3),'complex64')}
                term=ContractionTerm(('z','z','a'),'ak,ak,cdax->dc')
                a,k=term.subscripts.inputs[0];x=term.subscripts.inputs[2][-1]
                term=replace(term,sum_barriers=(((k,),(a,x)),((a,),(k,x)),((x,),(k,a))))
                problem=ContractionProblem(specs,(term,),{name:SourceSpec(name,0) for name in specs})
            else:
                before,_,_,_=tiled_fixture()
                term=before.problem.terms[0]
                problem=replace(before.problem,terms=(replace(term,
                    subscripts=replace(term.subscripts,output=term.subscripts.output[::-1])),))
            space=IntegerSpace(problem,profile=CostProfile())
            operation=next(op for op,_ in space.candidates(lambda:None)
                if type(space.registry.calculations[op.calculation].recipe.core) is FactorizedWideReduction)
            # No search-time bindings: exercise exactly the owner's independent
            # matcher after the worker route has crossed its serialization ABI.
            import pickle
            registry,operation=pickle.loads(pickle.dumps((space.registry,operation)))
            materialized,_=materialize_dense_path(problem,(operation,),operation.seconds,registry,recompute=True)
            self.assertEqual(materialized.actions[-1].equation.output,problem.terms[0].subscripts.output)
            rng=np.random.default_rng(261045)
            arrays={name:(rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)).astype('complex64')
                    for name,spec in problem.specs.items()}
            action=materialized.actions[-1]
            actual,_=bind_dense_action(action,materialized.catalog,np)(tuple(arrays[name] for name in action.operands))
            equation=problem.terms[0].subscripts.normalized()
            args=tuple(item for name,axes in zip(problem.terms[0].inputs,equation.inputs,strict=True)
                       for item in (arrays[name].astype('complex128'),axes))+(equation.output,)
            np.testing.assert_allclose(actual,np.einsum(*args,optimize=True),rtol=2e-6,atol=2e-6)

    def test_shared_six_factor_boundary_reduces_before_expanding(self):
        from lqcdweave.contraction.planning.optimizers.session import solve_storage
        from lqcdweave.contraction.execution.api import replay
        shapes=((2,2,3,3),(2,2,3,3),(3,3,3),(3,3,3),(2,3,2,2,3),(2,3,2,2,2))
        specs={str(i):TensorSpec(shape,'complex64') for i,shape in enumerate(shapes)}
        term=ContractionTerm(tuple(specs),'abnm,cdpq,ump,vqn,iubcx,jvday->ij')
        a,b,n,m=term.subscripts.inputs[0];c,d,p,q=term.subscripts.inputs[1]
        u,v=term.subscripts.inputs[2][0],term.subscripts.inputs[3][0]
        x,y=term.subscripts.inputs[4][-1],term.subscripts.inputs[5][-1]
        spatial=(n,m,p,q);coupled=(a,b,c,d,u,v,x,y)
        term=replace(term,sum_barriers=((spatial,coupled),
            *((((axis,),(*spatial,*(other for other in coupled if other!=axis)))) for axis in coupled)))
        problem=ContractionProblem(specs,(term,term),{key:SourceSpec(key,0) for key in specs},backend='cupy')
        plan=solve_storage(problem,{}, {key:SourcePreparation((),dtype='complex64') for key in specs},
            memory_limit=1<<20,host_memory_limit=1<<20,root_width=1,branch_width=1,
            storage_branch_width=1,include_ties=False,component_workers=1,deadline_seconds=2.,max_states=100)
        self.assertTrue(plan.plan.feasible)
        contracts=[action for action in plan.state.actions if action.kind=='contract']
        self.assertTrue(any(type(action.core) is FactorizedWideReduction for action in contracts))
        first=contracts[0]
        self.assertLess(len(first.equation.output),len(set().union(*map(set,first.equation.inputs))))
        rng=np.random.default_rng(261041)
        arrays={key:(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype('complex64')
                for key,shape in zip(specs,shapes,strict=True)}
        actual=replay(plan.state,lambda key,source:arrays[key])
        equation=term.subscripts.normalized()
        args=tuple(item for key,axes in zip(specs,equation.inputs,strict=True)
                   for item in (arrays[key].astype('complex128'),axes))+(equation.output,)
        expected=np.einsum(*args,optimize=True)
        for value in actual:
            limit=5e-5*np.maximum(np.maximum(np.abs(value),np.abs(expected)),1e-12)+2e-8
            self.assertTrue(np.all(np.abs(value-expected)<=limit))

    def test_direct_wide_candidate_avoids_impossible_binary_expansion(self):
        from lqcdweave.contraction.planning.optimizers.session import solve_storage
        shapes=((65,65,4),(65,65,4),(1000,65),(1000,65))
        specs={str(i):TensorSpec(shape,'complex64') for i,shape in enumerate(shapes)}
        term=ContractionTerm(tuple(specs),'abk,abk,ca,db->cd')
        a,b,k=term.subscripts.inputs[0]
        term=replace(term,sum_barriers=(((k,),(a,b)),((a,),(b,k)),((b,),(a,k))))
        problem=ContractionProblem(specs,(term,),{key:SourceSpec(key,0) for key in specs},backend='cupy')
        plan=solve_storage(problem,{}, {key:SourcePreparation((),dtype='complex64') for key in specs},
            memory_limit=32<<20,host_memory_limit=32<<20,root_width=1,branch_width=1,
            storage_branch_width=1,include_ties=False,component_workers=1,deadline_seconds=1.,max_states=100)
        self.assertTrue(plan.state.done)
        self.assertTrue(plan.plan.feasible)
        self.assertTrue(any(type(a.core) is FactorizedWideReduction for a in plan.state.actions))
        self.assertLessEqual(plan.plan.peak_device_bytes,32<<20)

    def test_tiled_component_sums_tail_budget_and_direct_integer_admission(self):
        from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
        for loop in (False,True):
            before,completed,action,profile=tiled_fixture(loop=loop)
            core=action.core
            self.assertEqual(core.tiles,4 if not loop else 1)
            self.assertEqual(core.tile_width,4 if not loop else 3)
            self.assertLessEqual(core.partial_bytes,4*core.blocks*16)
            bound=(before.resident_bytes+np.prod(core.output_shape)*np.dtype(core.result_dtype).itemsize
                   +core.workspace_bytes)
            self.assertTrue(replay_factorized_action(before,action,profile,int(bound)).done)
            with self.assertRaises(MemoryError):
                replay_factorized_action(before,action,profile,int(bound)-1)
            self.assertEqual(_decode_plan_value(_encode_plan_value(completed)),completed)
            # The candidate must exist without first materializing the much
            # larger component-expanded binary products. Audit raw labels too.
            for debug in (False,True):
                space=IntegerSpace(before.problem,profile=profile,debug=debug)
                choices=list(space.candidates(lambda:None))
                operation=next(op for op,_ in choices
                    if type(space.registry.calculations[op.calculation].recipe.core) is FactorizedWideReduction)
                space.apply(operation,lambda:None)
                self.assertTrue(space.is_done())
                audited=space.audit((operation,),operation.seconds)
                self.assertTrue(audited.done)
                self.assertIsInstance(audited.actions[-1].core,FactorizedWideReduction)
            from correlator.planning.test_coordinate_path import batch_problem,LANE
            from lqcdweave.contraction.planning.coordinate_path import lift_coordinate_path
            batched,names=batch_problem(before.problem,{'0','2'},2,rename=True)
            batched=replace(batched,terms=tuple(replace(t,sum_barriers=old.sum_barriers)
                for t,old in zip(batched.terms,before.problem.terms,strict=True)))
            lifted=lift_coordinate_path(completed,batched,names,lane_axis=LANE,profile=profile,
                memory_limit=1<<25,debug=True)
            current=next(a for a in lifted.actions if a.core is not None)
            rng=np.random.default_rng(261043)
            values=tuple((rng.normal(size=s)+1j*rng.normal(size=s)).astype('complex64')[...,::-1]
                         for s in current.core.shapes)
            actual,_=bind_dense_action(current,lifted.catalog,np)(values)
            equation=current.equation.normalized()
            args=tuple(item for value,axes in zip(values,equation.inputs,strict=True)
                       for item in (value.astype('complex128'),axes))+(equation.output,)
            np.testing.assert_allclose(actual,np.einsum(*args,optimize=True),rtol=2e-6,atol=2e-6)

    def test_old_core_archive_defaults_and_oversized_untiled_rejected(self):
        plan,profile=fixture();lowered,_=lower_factorized_reductions(plan,profile=profile,target_runtime='cuda')
        core=next(a.core for a in lowered.state.actions if a.core is not None)
        encoded=_encode_plan_value(core);encoded['value'].pop('tile_elements')
        self.assertEqual(_decode_plan_value(encoded),core)
        from lqcdweave.contracts.correlator.api import AxisPattern
        pattern=AxisPattern(((0,2),(1,2),(3,0),(4,1)),(3,4))
        shapes=((65,2),(65,2),(1,65),(1,65))
        with self.assertRaisesRegex(ValueError,'bounded'):
            FactorizedWideReduction(pattern,shapes,(0,1))
        tiled=FactorizedWideReduction(pattern,shapes,(0,1),tile_elements=4096)
        self.assertEqual(tiled.tiles,2)
        self.assertEqual(tiled.partial_bytes,4096*tiled.blocks*16)
        from lqcdweave.contracts.correlator.api import Action
        specs={str(i):TensorSpec(shape,'complex64') for i,shape in enumerate(shapes)}
        specs['out']=TensorSpec(tiled.output_shape,'complex64')
        large_action=Action('contract','out',tuple(str(i) for i in range(4)),pattern,0.,False,
                            core=tiled,wide_accumulation=True)
        rng=np.random.default_rng(261040)
        values=tuple((rng.normal(size=s)+1j*rng.normal(size=s)).astype('complex64')[...,::-1]
                     for s in shapes)
        actual,_=bind_dense_action(large_action,specs,np)(values)
        expected=np.einsum('ak,bk,ca,db->cd',*(value.astype('complex128') for value in values),optimize=True)
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
        before,completed,action,profile=tiled_fixture(tile=2)
        rng=np.random.default_rng(261041)
        values=tuple((rng.normal(size=s)+1j*rng.normal(size=s)).astype('complex64') for s in action.core.shapes)
        actual,_=bind_dense_action(action,completed.catalog,np)(values)
        equation=action.equation.normalized()
        args=tuple(item for value,axes in zip(values,equation.inputs,strict=True)
                   for item in (value.astype('complex128'),axes))+(equation.output,)
        np.testing.assert_allclose(actual,np.einsum(*args,optimize=True),rtol=2e-6,atol=2e-6)
    def test_retained_coordinate_lanes_preserve_fixed_core_and_independent_replay(self):
        from correlator.planning.test_coordinate_path import batch_problem,LANE
        from lqcdweave.contraction.planning.coordinate_path import lift_coordinate_path
        plan,profile=fixture();lowered,_=lower_factorized_reductions(plan,profile=profile,target_runtime='cuda')
        template=lowered.state;rng=np.random.default_rng(261037)
        for dynamic in ({'0','1'},{'0'},{'2'},{'0','2','3'}):
            batched,names=batch_problem(template.problem,dynamic,2,rename=True)
            batched=replace(batched,terms=tuple(replace(term,sum_barriers=old.sum_barriers)
                for term,old in zip(batched.terms,template.problem.terms,strict=True)))
            lifted=lift_coordinate_path(template,batched,names,lane_axis=LANE,profile=profile,
                memory_limit=1<<25,host_memory_limit=1<<25,debug=True)
            self.assertTrue(lifted.done)
            self.assertEqual(_decode_plan_value(_encode_plan_value(lifted)),lifted)
            values={key:(rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)).astype(np.complex64)
                    for key,spec in batched.specs.items()}
            live={}
            for action in lifted.actions:
                if action.kind=='load':live[action.tensor]=values[action.tensor]
                elif action.kind=='release':del live[action.tensor]
                elif not action.reused:
                    live[action.tensor]=bind_dense_action(action,lifted.catalog,np)(tuple(live[key] for key in action.operands))[0]
            equation=batched.terms[0].subscripts.normalized()
            args=tuple(item for key,axes in zip(batched.terms[0].inputs,equation.inputs,strict=True)
                       for item in (values[key].astype(np.complex128),axes))+(equation.output,)
            expected=np.einsum(*args,optimize=True).astype(np.complex64)
            np.testing.assert_allclose(live[lifted.results[0]],expected,rtol=2e-6,atol=2e-6)

    def test_saved_schedule_rejects_changed_precision_contract_and_accepts_legacy_empty(self):
        import json,tempfile
        from pathlib import Path
        from lqcdweave.storage.contraction_plans.api import (
            save_storage_schedule,load_storage_schedule,PlanArchiveError)
        hardware={'backend':'cupy','memory':1<<25}
        for constrained in (True,False):
            plan,profile=fixture(barriers=constrained)
            expected=plan.state.problem
            shapes={key:spec.shape for key,spec in expected.specs.items()}
            changed=replace(expected,terms=tuple(replace(term,sum_barriers=()) if constrained else
                replace(term,sum_barriers=(((term.subscripts.inputs[0][1],),
                    (term.subscripts.inputs[0][0],term.subscripts.inputs[1][0])),)) for term in expected.terms))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'selected.json'
                save_storage_schedule(path,(plan,),shapes=shapes,hardware=hardware)
                if not constrained:
                    document=json.loads(path.read_text())
                    def remove(value):
                        if isinstance(value,dict):
                            if value.get('type','').endswith('.ContractionTerm'):
                                value['value'].pop('sum_barriers',None)
                            for child in value.values():remove(child)
                        elif isinstance(value,list):
                            for child in value:remove(child)
                    remove(document);path.write_text(json.dumps(document))
                (loaded,),_=load_storage_schedule(path,expected=expected,hardware=hardware,
                    blocks=plan.blocks,preparations=plan.preparations)
                self.assertEqual(loaded.state.problem.terms,expected.terms)
                with self.assertRaisesRegex(PlanArchiveError,'sum barriers'):
                    load_storage_schedule(path,expected=changed,hardware=hardware,
                        blocks=plan.blocks,preparations=plan.preparations)

    def test_same_layout_kernel_cache_preserves_declared_wide_semantics(self):
        from types import SimpleNamespace
        from lqcdweave.contracts.correlator.api import Action
        from lqcdweave.contraction.execution.scene_runtime import bind_scene_execution
        equation=AxisPattern(((0,1),(1,2)),(0,2))
        actions=tuple(Action('contract',key,('left','right'),equation,wide_accumulation=wide)
                      for key,wide in (('ordinary',False),('wide',True)))
        catalog={key:TensorSpec(shape,np.complex64) for key,shape in
                 (('left',(2,3)),('right',(3,2)),('ordinary',(2,2)),('wide',(2,2)))}
        state=SimpleNamespace(actions=actions,catalog=catalog)
        cache={};bound=bind_scene_execution(state,(),np,kernel_cache=cache)
        self.assertEqual(len(cache),2)
        values=(np.array([[1e8,1,-1e8]]*2,dtype=np.complex64),np.ones((3,2),dtype=np.complex64))
        actual,_=bound.kernels[1](values)
        np.testing.assert_array_equal(actual,np.ones((2,2),dtype=np.complex64))

    def test_default_path_lowers_without_enabling_optional_kernel_batching(self):
        from lqcdweave.contraction.planning.gemm_groups import lower_independent_gemm_path
        from lqcdweave.contracts.correlator.api import PlannedContraction
        plan,profile=fixture()
        path=PlannedContraction(expanded=None,plan=plan,search=None,planning_seconds=0.,
            profile=profile,prediction={'kernel_batching':False})
        selected=lower_independent_gemm_path(path,target_runtime='cuda')
        self.assertFalse(selected.prediction['kernel_batching'])
        self.assertTrue(selected.prediction['factorized_wide_lowering']['applied'])
        self.assertTrue(any(type(a.core) is FactorizedWideReduction for a in selected.plan.state.actions))

    def test_generic_nonspin_dimensions_lower_archive_replay_and_independent_reference(self):
        plan,profile=fixture()
        lowered,report=lower_factorized_reductions(plan,profile=profile,target_runtime='cuda')
        self.assertTrue(report['applied'],report)
        actions=[a for a in lowered.state.actions if a.kind=='contract']
        self.assertEqual(len(actions),1)
        action=actions[0];core=action.core
        self.assertIs(type(core),FactorizedWideReduction)
        self.assertEqual(core.gram_elements,6)
        self.assertEqual(core.partial_bytes,96)
        self.assertEqual(core.output_shape,(7,11))
        self.assertLessEqual(core.partial_bytes,core.workspace_bytes)
        self.assertLess(lowered.peak_device_bytes,plan.peak_device_bytes)
        self.assertEqual(_decode_plan_value(_encode_plan_value(lowered)),lowered)
        from lqcdweave.contraction.planning.paths import rebind_scene_path
        names={key:'other:'+key for key in lowered.state.problem.sources}
        problem=replace(lowered.state.problem,
            specs={names[key]:spec for key,spec in lowered.state.problem.specs.items()},
            sources={names[key]:replace(source,region=names[key]) for key,source in lowered.state.problem.sources.items()},
            terms=tuple(replace(term,inputs=tuple(names[key] for key in term.inputs))
                        for term in lowered.state.problem.terms))
        rebound=rebind_scene_path(lowered.state,problem,names,profile=profile,memory_limit=1<<25)
        self.assertTrue(rebound.done)
        self.assertEqual(next(a.core for a in rebound.actions if a.core is not None),core)
        current=ContractionState.initial(lowered.state.problem)
        for key in current.problem.sources:current=current.load(key,profile,1<<25)
        replayed=replay_factorized_action(current,action,profile)
        self.assertTrue(replayed.done)
        with self.assertRaisesRegex(ValueError,'independent'):
            replay_factorized_action(current,replace(action,seconds=action.seconds+1),profile)
        with self.assertRaises(MemoryError):
            replay_factorized_action(current,action,profile,current.resident_bytes)
        rng=np.random.default_rng(261034)
        for repeat in range(2):
            values=tuple((rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)[:,::-1]
                         for shape in core.shapes)
            actual,owned=bind_dense_action(action,lowered.state.catalog,np)(values)
            self.assertTrue(owned)
            equation=core.equation.normalized()
            arguments=tuple(item for value,axes in zip(values,equation.inputs,strict=True)
                            for item in (value.astype(np.complex128),axes))+(equation.output,)
            expected=np.einsum(*arguments,optimize=True).astype(np.complex64)
            np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
            self.assertTrue(all(not np.shares_memory(actual,value) for value in values))

    def test_old_unmarked_and_unsupported_targets_remain_ordinary(self):
        plan,profile=fixture()
        self.assertIs(lower_factorized_reductions(plan,profile=profile,target_runtime='hip')[0],plan)
        unmarked=replace(plan,state=replace(plan.state,actions=tuple(replace(a,wide_accumulation=False) for a in plan.state.actions)))
        self.assertIs(lower_factorized_reductions(unmarked,profile=profile,target_runtime='cuda')[0],unmarked)
        self.assertFalse(any(a.core is not None for a in plan.state.actions))
        encoded=_encode_plan_value(plan)
        decoded=_decode_plan_value(encoded)
        self.assertFalse(any(a.core is not None for a in decoded.state.actions))

    def test_malformed_geometry_and_missing_precision_contract_rejected(self):
        plan,profile=fixture();lowered,_=lower_factorized_reductions(plan,profile=profile,target_runtime='cuda')
        action=next(a for a in lowered.state.actions if a.core is not None)
        with self.assertRaises(ValueError):replace(action.core,dtype='complex128')
        with self.assertRaises(ValueError):replace(action.core,large_operands=(0,0))
        problem=replace(lowered.state.problem,terms=tuple(replace(t,sum_barriers=()) for t in lowered.state.problem.terms))
        state=ContractionState.initial(problem)
        for key in problem.sources:state=state.load(key,profile,1<<25)
        with self.assertRaisesRegex(ValueError,'boundary'):
            replay_factorized_action(state,action,profile)


class FactorizedWideCuPyTests(unittest.TestCase):
    def test_tiled_component_cancellation_negative_strides_and_lane_archive(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        from correlator.planning.test_coordinate_path import batch_problem,LANE
        from lqcdweave.contraction.planning.coordinate_path import lift_coordinate_path
        import gc,weakref
        for loop in (False,True):
            before,completed,action,profile=tiled_fixture(loop=loop,tile=2)
            rng=np.random.default_rng(261042)
            host=[(rng.normal(size=s)+1j*rng.normal(size=s)).astype('complex64') for s in action.core.shapes]
            host[0][0,:]=1e8;host[0][1,:]=1;host[0][2,:]=-1e8
            host[1][...]=1
            host[2][...,0]=1e8;host[2][...,1]=1;host[2][...,2]=-1e8
            # A narrow pre-sum erases the entire first small factor.
            self.assertTrue(np.all(host[2].sum(axis=-1,dtype='complex64')==0))
            self.assertNotEqual(complex(host[0].sum(dtype='complex64')),7+0j)
            for lanes in (False,True):
                if lanes:
                    batched,names=batch_problem(before.problem,{'0','2'},2,rename=True)
                    batched=replace(batched,terms=tuple(replace(t,sum_barriers=old.sum_barriers)
                        for t,old in zip(batched.terms,before.problem.terms,strict=True)))
                    state=lift_coordinate_path(completed,batched,names,lane_axis=LANE,profile=profile,
                        memory_limit=1<<25,debug=True)
                    current=next(a for a in state.actions if a.core is not None)
                    arrays={names[str(i)]:(np.stack((v,2*v)) if str(i) in ('0','2') else v)
                            for i,v in enumerate(host)}
                    values=tuple(arrays[k] for k in current.operands)
                else:
                    state,current,values=completed,action,tuple(host)
                device=tuple(cp.asarray(v[...,::-1])[...,::-1] for v in values)
                refs=tuple(weakref.ref(v) for v in device)
                actual,_=bind_dense_action(current,state.catalog,cp)(device)
                # The independently known sums are sum_a A[a,k]=1 and
                # sum_x C[c,a,x]=1. A generic full einsum can multiply the two
                # 1e8 cancellations into 1e16 and lose the reference in FP64.
                # Use their analytic c128 result, without the kernel's Gram
                # tiling, partials, strides or coordinate-label machinery.
                expected=(np.full(action.core.output_shape,7,dtype='complex128') if loop else
                    np.broadcast_to(7*host[3].astype('complex128').sum(axis=(1,2)),action.core.output_shape))
                if lanes:
                    expected=np.stack((expected,4*expected),axis=-1)
                    # batch_problem's lane is retained in its declared output
                    # position, independently of any raw integer axis names.
                    position=current.equation.output.index(LANE)
                    expected=np.moveaxis(expected,-1,position)
                np.testing.assert_allclose(actual.get(),expected,rtol=2e-6,atol=2e-6)
                self.assertEqual(_decode_plan_value(_encode_plan_value(current)),current)
                del actual,device
                cp.cuda.get_current_stream().synchronize();gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(cp.get_default_memory_pool().used_bytes(),0)
    def test_raw_strides_current_values_and_owned_release(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        import gc,weakref
        plan,profile=fixture()
        lowered,report=lower_factorized_reductions(plan,profile=profile,target_runtime='cuda')
        self.assertTrue(report['applied'],report)
        action=next(a for a in lowered.state.actions if a.core is not None)
        kernel=bind_dense_action(action,lowered.state.catalog,cp)
        rng=np.random.default_rng(261036);refs=[]
        for repeat in range(2):
            host=tuple((rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)[:,::-1]
                       for shape in action.core.shapes)
            values=tuple(cp.asarray(value[:,::-1])[:,::-1] for value in host)
            refs.extend(weakref.ref(value) for value in values)
            result,owned=kernel(values)
            self.assertTrue(owned)
            self.assertTrue(all(result.data.mem is not value.data.mem for value in values))
            equation=action.equation.normalized()
            args=tuple(item for value,axes in zip(host,equation.inputs,strict=True)
                       for item in (value.astype(np.complex128),axes))+(equation.output,)
            expected=np.einsum(*args,optimize=True).astype(np.complex64)
            np.testing.assert_allclose(result.get(),expected,rtol=2e-6,atol=2e-6)
            del result,values
        cp.cuda.get_current_stream().synchronize();gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(cp.get_default_memory_pool().used_bytes(),0)

    def test_public_default_two_cfg_fixed_archive_and_preparation_carry(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        self._check_public_archive('cupy')

    def test_cpu_fixture_retains_sources_and_rebinds_two_cfgs(self):
        self._check_public_archive('numpy')

    def test_cpu_complex_spatial_projection_archive_uses_independent_c128(self):
        self._check_public_archive('numpy',complex_space=True)

    def test_gpu_complex_spatial_projection_archive_uses_independent_c128(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        self._check_public_archive('cupy',complex_space=True)

    def _check_public_archive(self,backend,complex_space=False):
        from contextlib import contextmanager,ExitStack
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from test_structured_correlator import task
        from correlator.execution.test_cfg_runtime import SafeReader
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.operators.single_hadron.api import Operator,OperatorTerm
        from lqcdweave.contraction.correlator import calculate,save_calculation_plan
        from lqcdweave.contraction.correlator_cfg import complete_calculation
        from lqcdweave.contracts.correlator.api import describe_coordinates
        from lqcdweave.contraction.planning import cfg_plan
        rng=np.random.default_rng(261035)
        def random(shape):return (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
        n=4 if complex_space else 8;component=ElementalComponent(0)
        components=tuple(ElementalComponent(i) for i in range(3)) if complex_space else (component,)
        names=tuple('g'+str(i) for i in range(4 if complex_space else 16))
        elemental=ArrayElemental({'v2v':random((len(components),1,3,n,n))},components,cfg='cfg',phase_convention='anchor',data_identity='thin-wide-elemental')
        propagator=ArrayPropagator({'vsv':random((3,3,4,4,n,n))},time_convention='absolute',cfg='cfg',data_identity='thin-wide-propagator')
        operators={name:Operator.from_terms(name,
            (OperatorTerm(i,component,0,1),OperatorTerm((i+15)%16,component,0,.3+.7j),
             OperatorTerm(i,components[1],0,.2j),OperatorTerm((i+1)%16,components[2],0,-.4))
            if complex_space else (OperatorTerm(i,component,0,1),)) for i,name in enumerate(names)}
        def binding(reader):
            return lambda row:Meson(reader,operators[row.hadron_name],row.dagger,phase_convention='anchor')
        options=dict(vertex_map=binding(elemental),propagator_map={'S^q':propagator},configuration='cfg',usedNe=n,
            memory_limit=1<<28,host_memory_limit=1<<28)
        selected=calculate(task(structured=True,names=names),backend=backend,execute=False,time_axes={0:0,1:1},
            **options)
        self.assertTrue(selected.path.plan.state.actions)
        pairs=({0:0,1:1},{0:0,1:2},{0:1,1:0},{0:1,1:2})
        coordinates,_=describe_coordinates(time_pairs=pairs)
        with patch.object(cfg_plan,'_CFG_BINDING_WINDOW_SLOTS',1):
            complete=complete_calculation(selected,coordinates=coordinates)
        self.assertGreater(complete.plan.statistics.get('cross_window_reused_preparations',0),0,complete.plan.statistics)
        expected={}
        for cfg,scale in (('a',1),('b',2)):
            e=ArrayElemental({k:v.astype('complex128')*scale for k,v in elemental.sectors.items()},components,cfg=cfg,phase_convention='anchor')
            p=ArrayPropagator({k:v.astype('complex128')*scale for k,v in propagator.sectors.items()},cfg=cfg,time_convention='absolute')
            expected[cfg]=calculate(task(structured=False,names=names),optimize='none',time_pairs=pairs,
                vertex_map=binding(e),propagator_map={'S^q':p},configuration=cfg,usedNe=n,memory_limit=1<<28).value
        if backend=='cupy':
            lifted=selected.run(time_pairs=pairs,batch=2).value
            reference=expected['a']
            error=np.abs(lifted.astype(np.complex128)-reference.astype(np.complex128))
            limit=5e-5*np.maximum(np.maximum(np.abs(lifted),np.abs(reference)),1e-12)+2e-8
            self.assertTrue(np.all(error<=limit),float(np.max(error/limit)))
        records=[]
        @contextmanager
        def factory(cfg):
            yield {request.reader_role:SafeReader(elemental if request.family=='elemental' else propagator,
                cfg,1 if cfg=='a' else 2,records) for primary in complete.plan.requests.values()
                for request in (primary,*primary.aliases)}
        with tempfile.TemporaryDirectory() as directory:
            if backend=='numpy':
                from lqcdweave.storage.contraction_plans.api import (
                    save_contraction_plan,load_contraction_plan)
                path=Path(directory)/'selected-path.json'
                hardware={'backend':'numpy','memory':1<<28}
                save_contraction_plan(path,selected.path,hardware=hardware)
                load_contraction_plan(path,expected=selected.expanded,hardware=hardware,memory_limit=1<<28)
            archive=Path(directory)/'thin-wide.json.gz';save_calculation_plan(archive,complete)
            with ExitStack() as blocked:
                for name in ('lqcdweave.contraction.correlator.expand_correlator',
                             'lqcdweave.contraction.expansion.matrix_inputs._bind_matrix_base',
                             'lqcdweave.contraction.correlator.find_contraction_path',
                             'lqcdweave.contraction.planning.cfg_plan._bind_cfg_coordinates',
                             'lqcdweave.contraction.planning.cfg_preparation_carry.retain_cfg_preparations',
                             'lqcdweave.contraction.planning.factorized_reduction.lower_factorized_reductions'):
                    blocked.enter_context(patch(name,side_effect=AssertionError(name)))
                for repeat in range(2):
                    values=dict(calculate(plan_archive=archive,configuration=iter(('a','b')),data=factory).value)
                    for cfg in expected:
                        error=np.abs(values[cfg].astype(np.complex128)-expected[cfg].astype(np.complex128))
                        limit=5e-5*np.maximum(np.maximum(np.abs(values[cfg]),np.abs(expected[cfg])),1e-12)+2e-8
                        self.assertTrue(np.all(error<=limit),float(np.max(error/limit)))
        self.assertTrue(records)
        if backend=='cupy':
            import cupy as cp
            cp.cuda.get_current_stream().synchronize()
            self.assertEqual(cp.get_default_memory_pool().used_bytes(),0)


if __name__=='__main__':unittest.main()
