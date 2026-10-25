"""Fixed templates use the existing physical plan/replay, with no second IR."""
from dataclasses import replace
import unittest
from unittest.mock import Mock
import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, SourcePreparation,
    StorageBlock, MatrixKernelSpec,
)
from lqcdweave.contraction.execution.indices import einsum_arguments
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, logical_source_problem, logical_cost_profile,
    describe_matrix_action, replay_matrix_action, estimate_matrix_action, plan_scene_storage,
    plan_matrix_actions,
    rebind_scene_path,
)
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.execution.scene_runtime import replay, _model_features
from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value


def fixture(kind,partitions,repeat=False):
    rng = np.random.default_rng(13)
    n,cl,cr,nl,nr = 3,2,3,4,5
    shapes = (((4,4,n,n),)*2+((cl,4,4,n,n),(cr,4,4,n,n),(nl,cl),(nr,cr))
              if kind=='connected' else ((4,4,n,n),(cl,4,4,n,n),(nl,cl)))
    arrays = [rng.normal(size=shape)+1j*rng.normal(size=shape) for shape in shapes]
    spec = MatrixKernelSpec(kind,shapes,tuple(a.dtype.name for a in arrays),partitions,2)
    if partitions:
        pc = np.zeros(spec.projection_shape,dtype=spec.dtype)
        for index,a,b,c,d,w in spec.projection_blocks:
            pc[index,a:b,c:d] = w
        arrays += [pc.copy() for _ in range(2 if kind=='connected' else 1)]
    arrays = {f'in:{i}':a for i,a in enumerate(arrays)}
    specs = {key:TensorSpec(a.shape,a.dtype) for key,a in arrays.items()}
    sources = {key:SourceSpec(key,0,snapshot_bytes=a.nbytes,read_bytes=0) for key,a in arrays.items()}
    term = ContractionTerm(tuple(arrays),spec.equation)
    problem = ContractionProblem(specs,(term,term) if repeat else (term,),sources)
    profile = CostProfile()
    state = ContractionState.initial(logical_source_problem(problem))
    model = logical_cost_profile(profile)
    for key in arrays:
        state = state.load(key,model,1 << 40)
    return arrays,spec,profile,state


class MatrixActionTests(unittest.TestCase):
    def test_fixed_actions_rebind_without_generic_contraction_and_reprice(self):
        for kind in ('connected','loop'):
            with self.subTest(kind=kind):
                arrays,spec,profile,state = fixture(kind,(0,1,2,3),repeat=True)
                for index in range(2):
                    action,_,_ = describe_matrix_action(state,index,spec,profile=profile)
                    state = replay_matrix_action(state,action,profile)
                names = {key:'new:'+key for key in state.problem.sources}
                problem = replace(state.problem,
                    specs={names[key]:layout for key,layout in state.problem.specs.items()},
                    sources={names[key]:replace(source,region='new:'+source.region)
                             for key,source in state.problem.sources.items()},
                    terms=tuple(replace(term,inputs=tuple(names[key] for key in term.inputs))
                                for term in state.problem.terms))
                model = replace(profile,contract_scale=2)
                from unittest.mock import patch
                with patch.object(ContractionState,'contract',side_effect=AssertionError('generic contract')):
                    rebound = rebind_scene_path(state,problem,names,profile=model,memory_limit=1<<40)
                self.assertTrue(rebound.done)
                self.assertEqual(rebound.results[0],rebound.results[1])
                self.assertNotEqual(rebound.results,state.results)
                computations = [action for action in rebound.actions if action.kind=='contract']
                self.assertEqual([a.core for a in computations],[spec,spec])
                self.assertEqual(computations[0].seconds,2*state.actions[-2].seconds)
                self.assertTrue(computations[1].reused)
                self.assertEqual(computations[1].seconds,0)
                with self.assertRaises(MemoryError):
                    rebind_scene_path(state,problem,names,profile=model,memory_limit=1)
                relation = problem.terms[0].subscripts
                changed_relation = AxisPattern(relation.inputs, relation.output[::-1])
                changed = replace(problem,terms=(replace(problem.terms[0],subscripts=changed_relation),problem.terms[1]))
                with self.assertRaisesRegex(ValueError,'algebra'):
                    rebind_scene_path(state,changed,names,profile=model,memory_limit=1<<40)

    def plan(self,kind,partitions):
        arrays,spec,profile,state = fixture(kind,partitions)
        model = logical_cost_profile(profile)
        action,output,workspace = describe_matrix_action(state,0,spec,profile=model)
        final = replay_matrix_action(state,action,model)
        for key in arrays:
            final = final.release(key)
        blocks = {key:StorageBlock(key,'all',a.nbytes,host_bytes=a.nbytes) for key,a in arrays.items()}
        preps = {key:SourcePreparation((key,)) for key in arrays}
        plan = plan_scene_storage(final,blocks,preps,profile=profile,memory_limit=1 << 40)
        return arrays,spec,profile,action,plan

    def test_existing_physical_audit_and_execution(self):
        for kind in ('connected','loop'):
            for partitions in (None,(0,1,2,3)):
                with self.subTest(kind=kind,partitions=partitions):
                    arrays,spec,profile,action,plan = self.plan(kind,partitions)
                    class Loader:
                        def __init__(self):self.reads=[]
                        def read_block(self,key,block):self.reads.append(key);return arrays[key]
                        def materialize(self,key,source,raw):return raw[key]
                    loader = Loader()
                    report = replay_scene_storage(plan,loader,io_overlap=False)
                    operands = tuple(np.asarray(a, dtype=spec.dtype) for a in arrays.values())
                    expected = np.einsum(
                        *einsum_arguments(spec.equation, operands), optimize=True,
                    )
                    np.testing.assert_allclose(report[0],expected,rtol=2e-12,atol=2e-11)
                    self.assertEqual(len(loader.reads),len(arrays))
                    estimate = estimate_matrix_action(spec,profile=profile)
                    self.assertGreaterEqual(plan.peak_host_bytes,estimate.peak_array_bytes)
                    self.assertEqual(plan.state.catalog[action.tensor].shape,spec.output_shape)
                    features = _model_features(plan.state,action)
                    self.assertEqual(features['flops'],sum(s.flops for s in estimate.stages))
                    self.assertEqual(features['kernel_calls'],sum(s.kernel_calls for s in estimate.stages))
                    # Only registered read/materialization costs augment the
                    # unchanged selected computation cost.
                    self.assertGreaterEqual(plan.seconds,action.seconds)

    def test_independent_cost_audit_tampering_and_no_parent_mutation(self):
        arrays,spec,profile,state = fixture('connected',(0,1,2,3))
        action,_,_ = describe_matrix_action(state,0,spec,profile=profile)
        with self.assertRaisesRegex(ValueError,'computed cost'):
            replay_matrix_action(state,replace(action,seconds=action.seconds+1),profile)
        with self.assertRaises(MemoryError):
            replay_matrix_action(state,action,profile,memory_limit=state.resident_bytes)
        self.assertIsNotNone(state.pending[0])
        self.assertNotIn(action.tensor,state.catalog)
        self.assertNotIn(action,state.actions)

    def test_duplicate_terms_reuse_one_result(self):
        arrays,spec,profile,state = fixture('loop',None,repeat=True)
        first,_,_ = describe_matrix_action(state,0,spec,profile=profile)
        state = replay_matrix_action(state,first,profile)
        second,_,workspace = describe_matrix_action(state,1,spec,profile=profile)
        self.assertTrue(second.reused)
        self.assertEqual(second.seconds,0)
        self.assertEqual(workspace,0)
        final = replay_matrix_action(state,second,profile)
        self.assertTrue(final.done)
        self.assertEqual(final.results,(first.tensor,first.tensor))

    def test_compile_with_existing_blocks_and_budget_failure_before_io(self):
        arrays,spec,profile,state = fixture('connected',(0,1,2,3),repeat=True)
        blocks = {key:StorageBlock(key,'all',a.nbytes,host_bytes=a.nbytes) for key,a in arrays.items()}
        preps = {key:SourcePreparation((key,)) for key in arrays}
        plan = plan_matrix_actions(state.problem,(spec,spec),blocks,preps,profile=profile,memory_limit=1 << 40)
        contracts = [a for a in plan.state.actions if a.kind=='contract']
        self.assertEqual([a.reused for a in contracts],[False,True])
        self.assertEqual(len([a for a in plan.state.actions if a.kind=='load']),len(arrays))
        self.assertEqual(len([e for e in plan.events if e.kind=='read']),len(arrays))
        self.assertEqual(plan.state.loaded,frozenset(plan.state.results))
        with self.assertRaisesRegex(MemoryError,'before IO'):
            plan_matrix_actions(state.problem,(spec,spec),blocks,preps,profile=profile,memory_limit=1)
        with self.assertRaises(ValueError):
            plan_matrix_actions(state.problem,(spec,),blocks,preps,profile=profile,memory_limit=1 << 40)

    def test_projection_values_cannot_be_ignored(self):
        from lqcdweave.contraction.execution.diagnostics import execution_debug
        arrays,spec,profile,action,plan = self.plan('loop',(0,1,2,3))
        arrays['in:3'][0,0,0] += 1
        class Loader:
            def read_block(self,key,block):return arrays[key]
            def materialize(self,key,source,raw):return raw[key]
        with execution_debug(True), self.assertRaisesRegex(ValueError,'PC coefficients'):
            replay_scene_storage(plan,Loader(),io_overlap=False)

    def test_unsupported_backend_rejected_before_io_and_matrix_action_roundtrips(self):
        arrays,spec,profile,action,plan = self.plan('loop',None)
        loader = Mock()
        with self.assertRaisesRegex(NotImplementedError,'NumPy'):
            replay(plan.state,loader,backend=object())
        self.assertEqual(loader.mock_calls,[])
        restored = decode_plan_value(encode_plan_value(action))
        self.assertEqual(restored, action)
        self.assertEqual(restored.core, spec)


if __name__ == '__main__':unittest.main()
