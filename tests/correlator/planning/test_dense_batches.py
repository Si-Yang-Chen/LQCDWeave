from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, TensorSpec, ContractionProblem, ContractionTerm,
    SourceSpec, SourcePreparation, validate_executable_resources,
)
from lqcdweave.contraction.planning.api import CostProfile, describe_dense_batch, find_dense_batches
from lqcdweave.contraction.execution.operations.dense_batch import contract_dense_batch
from lqcdweave.contraction.planning.api import solve_storage, lower_dense_batches
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.storage.contraction_plans.api import (
    save_storage_schedule, load_storage_schedule, encode_plan_value, decode_plan_value,
)
from lqcdweave.contraction.performance.api import predict_plan_cost


class ArrayLoader:
    def __init__(self,arrays):
        self.arrays,self.reads=arrays,[]
    def read_block(self,*args):
        raise AssertionError('this fixture has no physical reads')
    def materialize(self,key,*args):
        self.reads.append(key)
        return self.arrays[key]
    def close_all(self):
        pass


class DenseBatchTests(unittest.TestCase):
    def test_numpy_high_rank_batches_use_saved_matrix_packing_and_blas(self):
        rng = np.random.default_rng(531)
        pattern = AxisPattern(((0, 2, 1, 3), (3, 5, 2, 4)), (0, 1, 4, 5))
        for left_dtype, right_dtype in ((np.float32, np.float32), (np.float64, np.float64),
                (np.complex64, np.complex64), (np.complex128, np.complex128)):
            with self.subTest(left_dtype=left_dtype, right_dtype=right_dtype):
                specs, inputs, actions = {}, {}, []
                dtype = np.result_type(left_dtype, right_dtype)
                for key, shape, kind in (
                        ('a0', (2, 4, 3, 5), left_dtype), ('a1', (2, 4, 4, 5), left_dtype),
                        ('b0', (5, 6, 4, 3), right_dtype), ('b1', (5, 7, 4, 3), right_dtype)):
                    value = rng.normal(size=shape)
                    if np.issubdtype(kind, np.complexfloating):
                        value = value + 1j * rng.normal(size=shape)
                    value = value.astype(kind)
                    value = value[..., ::-1] if key.startswith('a') else value[::-1]
                    specs[key] = TensorSpec(shape, kind)
                    inputs[key] = value
                expected = {}
                for i in range(2):
                    for j in range(2):
                        key = f'c{i}{j}'
                        value = np.einsum(inputs[f'a{i}'], pattern.inputs[0], inputs[f'b{j}'],
                                          pattern.inputs[1], pattern.output, optimize=False)
                        specs[key] = TensorSpec(value.shape, dtype)
                        expected[key] = value
                        actions.append(Action('contract', key, (f'a{i}', f'b{j}'), pattern, seconds=1))
                batch = describe_dense_batch(actions, (0, 1, 2, 3), specs, CostProfile())
                self.assertEqual(batch.packing_dtype, dtype.name)
                self.assertEqual(batch.packing_bytes, sum(a.size for a in inputs.values()) * dtype.itemsize)
                arrays = dict(inputs)
                with patch.object(np, 'einsum', side_effect=AssertionError('serial high-rank fallback')), \
                     patch.object(np, 'matmul', wraps=np.matmul) as multiply:
                    contract_dense_batch(batch, arrays)
                self.assertEqual(multiply.call_count, 1)
                tolerance = 5e-5 if dtype.itemsize <= 8 and dtype != np.float64 else 2e-12
                for key, value in expected.items():
                    self.assertEqual(arrays[key].dtype, dtype)
                    np.testing.assert_allclose(arrays[key], value, rtol=tolerance, atol=tolerance)
                del arrays['c00']
                np.testing.assert_allclose(arrays['c11'], expected['c11'], rtol=tolerance, atol=tolerance)
                # Old saved records keep the high-rank reference kernel.
                legacy = replace(batch, packing_dtype=None,
                    packing_bytes=sum(a.nbytes for a in inputs.values()))
                original = dict(inputs)
                with patch.object(np, 'matmul', side_effect=AssertionError('legacy plan changed implementation')):
                    contract_dense_batch(legacy, original)
                for key, value in expected.items():
                    np.testing.assert_allclose(original[key], value, rtol=tolerance, atol=tolerance)

    def fixture(self, *, ragged=True, dtype=np.complex128):
        specs, actions = {}, []
        for i, m in enumerate((2, 3) if ragged else (2, 2)):
            specs[f"a{i}"] = TensorSpec((m, 5), dtype)
        for j, n in enumerate((3, 4) if ragged else (3, 3)):
            specs[f"b{j}"] = TensorSpec((5, n), dtype)
        profile = CostProfile()
        for i in range(2):
            for j in range(2):
                shape = specs[f"a{i}"].shape[0], specs[f"b{j}"].shape[1]
                specs[f"c{i}{j}"] = TensorSpec(shape, dtype)
                traffic = sum(np.prod(specs[k].shape).item()*specs[k].dtype.itemsize
                              for k in (f"a{i}", f"b{j}", f"c{i}{j}"))
                seconds = profile.contract_seconds(np.prod(shape).item()*5, traffic, dtype=dtype)
                actions.append(Action("contract", f"c{i}{j}", (f"a{i}", f"b{j}"),
                    AxisPattern(((0,1),(1,2)),(0,2)), seconds=seconds))
        return actions, specs, profile

    def test_cartesian_numeric_and_owned_views(self):
        actions, specs, profile = self.fixture()
        batch = describe_dense_batch(actions, (0,1,2,3), specs, profile)
        self.assertIsNotNone(batch)
        rng = np.random.default_rng(24)
        arrays = {k:rng.normal(size=s.shape)+1j*rng.normal(size=s.shape)
                  for k,s in specs.items() if k.startswith(("a","b"))}
        contract_dense_batch(batch, arrays)
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(arrays[f"c{i}{j}"],arrays[f"a{i}"]@arrays[f"b{j}"])
        owner = arrays['c00'].base
        self.assertTrue(all(arrays[f'c{i}{j}'].base is owner for i in range(2) for j in range(2)))
        self.assertEqual(owner.nbytes, batch.output_bytes)
        del arrays['c00']
        self.assertEqual(arrays['c11'].shape, (3,4))

    def test_cost_includes_two_genuine_pack_calls(self):
        actions, specs, profile = self.fixture()
        batch = describe_dense_batch(actions, (0,1,2,3), specs, profile)
        ibytes = (5*5 + 5*7)*16
        expected = profile.contract_seconds(5*7*5, ibytes + 5*7*16,dtype=np.complex128)
        expected += profile.contract_seconds(0,2*5*5*16,dtype=np.complex128)
        expected += profile.contract_seconds(0,2*5*7*16,dtype=np.complex128)
        self.assertEqual(batch.seconds, expected)
        self.assertEqual(batch.packing_bytes, ibytes)

    def test_missing_edge_is_not_computed(self):
        actions, specs, profile = self.fixture()
        self.assertIsNone(describe_dense_batch(actions,(0,1,2),specs,profile))

    def test_shape_equal_disjoint_edges_are_separate(self):
        actions, specs, profile = self.fixture(ragged=False)
        state = SimpleNamespace(actions=(actions[0],actions[3]),catalog=specs,results=('c00','c11'))
        self.assertEqual(find_dense_batches(state,profile),())

    def test_retained_alignment_not_a_reduction_batch(self):
        actions, specs, profile = self.fixture()
        actions[0] = replace(actions[0],equation=AxisPattern(((0,1),(1,2)),(0,1,2)))
        self.assertIsNone(describe_dense_batch(actions,(0,1,2,3),specs,profile))

    def test_transposed_single_left_multi_free_right(self):
        specs = {'a':TensorSpec((3,2,5),np.complex128),
                 'b0':TensorSpec((4,5,6),np.complex128),
                 'b1':TensorSpec((4,5,7),np.complex128),
                 'c0':TensorSpec((2,3,4,6),np.complex128),
                 'c1':TensorSpec((2,3,4,7),np.complex128)}
        actions = [Action('contract',f'c{i}',('a',f'b{i}'),
                   AxisPattern(((1,0,2),(3,2,4)),(0,1,3,4)),seconds=1) for i in range(2)]
        batch = describe_dense_batch(actions,(0,1),specs,CostProfile())
        self.assertEqual(batch.packing_dtype, 'complex128')
        self.assertEqual(batch.packing_bytes,(3*2*5+4*5*6+4*5*7)*16)
        rng = np.random.default_rng(3)
        arrays = {k:rng.normal(size=specs[k].shape).astype(specs[k].dtype) for k in ('a','b0','b1')}
        contract_dense_batch(batch,arrays)
        for i in range(2):
            np.testing.assert_allclose(arrays[f'c{i}'],np.einsum('ijk,lkp->jilp',arrays['a'],arrays[f'b{i}']))

    def test_scalar_and_single_free_side_batch_survive_operand_reordering(self):
        rng = np.random.default_rng(907)
        for output in ((), (2,)):
            for backend in ('numpy', 'cupy'):
                with self.subTest(output=output, backend=backend):
                    shape = (3, 5) if not output else (3, 5, 2)
                    right_axes = (0, 1) if not output else (0, 1, 2)
                    specs = {'a': TensorSpec((5, 3), np.complex128),
                             **{f'b{i}': TensorSpec(shape, np.complex128) for i in range(2)},
                             **{f'c{i}': TensorSpec(() if not output else (2,), np.complex128)
                                for i in range(2)}}
                    actions = [Action('contract', f'c{i}', ('a', f'b{i}'),
                        AxisPattern(((1, 0), right_axes), output), seconds=1) for i in range(2)]
                    profile = CostProfile()
                    original = describe_dense_batch(actions, (0, 1), specs, profile, backend=backend)
                    self.assertIsNotNone(original)
                    # Moving one independent producer reverses the factors in
                    # one Term; the equation and contraction are still identical.
                    actions[1] = replace(actions[1], operands=actions[1].operands[::-1],
                        equation=AxisPattern(actions[1].equation.inputs[::-1], output))
                    batch = describe_dense_batch(actions, (0, 1), specs, profile, backend=backend)
                    self.assertEqual(batch, original)
                    arrays = {k: rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)
                              for k, s in specs.items() if k.startswith(('a', 'b'))}
                    expected = [np.einsum(arrays['a'], (1, 0), arrays[f'b{i}'], right_axes, output)
                                for i in range(2)]
                    contract_dense_batch(batch, arrays, backend=np)
                    for i, value in enumerate(expected):
                        np.testing.assert_allclose(arrays[f'c{i}'], value)

    def test_duplicate_edge_or_unprofitable_group_rejected(self):
        actions, specs, profile = self.fixture()
        self.assertIsNone(describe_dense_batch(actions,(0,0,1,2),specs,profile))
        actions = [replace(a,seconds=0) for a in actions]
        self.assertIsNone(describe_dense_batch(actions,(0,1,2,3),specs,profile))

    def test_no_partial_publication_on_missing_input(self):
        actions, specs, profile = self.fixture()
        batch = describe_dense_batch(actions,(0,1,2,3),specs,profile)
        arrays={'a0':np.ones((2,5))}
        with self.assertRaises(KeyError):
            contract_dense_batch(batch,arrays)
        self.assertEqual(set(arrays),{'a0'})

    def test_gpu_packing_uses_uniform_dtype_and_gemm(self):
        actions,specs,profile=self.fixture()
        batch=describe_dense_batch(actions,(0,1,2,3),specs,profile,backend='cupy')
        expected_bytes=(5*5+5*7)*16
        expected_cost=profile.contract_seconds(5*7*5,expected_bytes+5*7*16,dtype=np.complex128)
        expected_cost+=profile.contract_seconds(0,2*5*5*16,dtype=np.complex128)
        expected_cost+=profile.contract_seconds(0,5*7*(16+16),dtype=np.complex128)
        self.assertEqual(batch.packing_dtype,'complex128')
        self.assertEqual(batch.packing_bytes,expected_bytes)
        self.assertEqual(batch.seconds,expected_cost)
        rng=np.random.default_rng(74)
        arrays={k:(rng.normal(size=s.shape)+1j*rng.normal(size=s.shape)).astype(s.dtype)
                for k,s in specs.items() if k.startswith(('a','b'))}
        saved={k:v.copy() for k,v in arrays.items()}
        packs=[]
        def concatenate(values,*,axis,out):
            packs.append(out)
            return np.concatenate(values,axis=axis,out=out)
        backend=SimpleNamespace(__name__='cupy',array=np.array,empty=np.empty,
                                concatenate=concatenate,matmul=np.matmul)
        phases=[]
        output=contract_dense_batch(batch,arrays,backend=backend,
            timed=lambda name,fn,**kw: (phases.append((name,kw)),fn())[1],device=True)
        self.assertEqual([name for name,_ in phases],['dense_batch_pack','dense_batch_pack','contract_kernel'])
        self.assertTrue(all(kw['device'] for _,kw in phases))
        self.assertEqual(sum(value.nbytes for value in packs),expected_bytes)
        self.assertTrue(all(value.flags.c_contiguous and value.dtype==np.complex128 for value in packs))
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(arrays[f'c{i}{j}'],saved[f'a{i}']@saved[f'b{j}'])
                self.assertTrue(np.shares_memory(arrays[f'c{i}{j}'],output))
        for k,v in saved.items():
            np.testing.assert_array_equal(arrays[k],v)

    def test_legacy_numpy_batch_defaults_to_no_promoted_pack(self):
        actions,specs,profile=self.fixture()
        batch=describe_dense_batch(actions,(0,1,2,3),specs,profile)
        encoded=encode_plan_value(batch)
        encoded['value'].pop('packing_dtype')
        encoded['value'].pop('prepared_sides')
        self.assertEqual(decode_plan_value(encoded),batch)


class DenseBatchIntegrationTests(unittest.TestCase):
    def plan(self, *, three=False, backend='numpy', high_rank=False):
        _,catalog,_=DenseBatchTests().fixture()
        specs={k:s for k,s in catalog.items() if not k.startswith('c')}
        if high_rank:
            specs = {**{f'a{i}': TensorSpec((2, m, 5), np.complex128)
                        for i, m in enumerate((3, 4))},
                     **{f'b{j}': TensorSpec((5, 2, n), np.complex128)
                        for j, n in enumerate((6, 7))}}
            terms = tuple(ContractionTerm((f'a{i}', f'b{j}'), 'ijk,klm->ijlm')
                          for i in range(2) for j in range(2))
        elif three:
            specs['a0']=TensorSpec((64,5),np.complex128)
            specs.update(d0=TensorSpec((3,7),np.complex128),d1=TensorSpec((4,7),np.complex128))
            terms=tuple(ContractionTerm(('a0',f'b{i}',f'd{i}'),'ij,jk,kl->il') for i in range(2))
        else:
            terms=tuple(ContractionTerm((f'a{i}',f'b{j}'),'ij,jk->ik') for i in range(2) for j in range(2))
        problem=ContractionProblem(specs,terms,{k:SourceSpec(k,0,read_bytes=0) for k in specs},backend=backend)
        preps={k:SourcePreparation(()) for k in specs}
        result=solve_storage(problem,{},preps,optimizer='anytime',kernel_batching=True,memory_limit=10**8,
                            deadline_seconds=1.0,max_states=10000,root_width=1,branch_width=1,
                            component_workers=1)
        return result

    def arrays(self,plan):
        rng=np.random.default_rng(49)
        return {k:(rng.normal(size=s.shape)+1j*rng.normal(size=s.shape)).astype(s.dtype)
                for k,s in plan.state.problem.specs.items()}

    def test_public_anytime_and_execution(self):
        result=self.plan()
        plan=result.plan
        self.assertEqual(len(plan.kernel_batches),1)
        self.assertLess(plan.compute_seconds,result.statistics['logical_best_seconds'])
        self.assertTrue(result.statistics['dense_kernel_lowering']['resource_audited'])
        arrays=self.arrays(plan)
        loader=ArrayLoader(arrays)
        values=replay_scene_storage(plan,loader,measure=True)
        for term,actual in zip(plan.state.problem.terms,values.values):
            np.testing.assert_allclose(actual,arrays[term.inputs[0]]@arrays[term.inputs[1]])
        self.assertEqual(set(loader.reads),set(arrays))
        self.assertEqual(len(loader.reads),len(arrays))
        predicted=sum(t.estimated_seconds for t in values.timings)
        self.assertAlmostEqual(predicted,plan.seconds)
        self.assertFalse(values.execution_profile['calibration_eligible'])
        members=plan.kernel_batches[0].action_indices
        self.assertEqual(values.timings[members[0]].model_features['kernel_calls'],3)
        self.assertTrue(all(values.timings[i].model_features['kernel_calls']==0 for i in members[1:]))
        report=predict_plan_cost((plan,),CostProfile(),workload={},provenance={})
        self.assertAlmostEqual(report['total_seconds'],plan.seconds)
        self.assertAlmostEqual(report['parts_seconds']['dense_contraction'],plan.compute_seconds)
        self.assertEqual(report['dense_kernel_batches'],1)

    def test_archive_load_and_execute_without_planning(self):
        plan=self.plan().plan
        shapes={k:list(s.shape) for k,s in plan.state.problem.specs.items()}
        hardware={'backend':'numpy','memory':10**8}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'batch.json'
            save_storage_schedule(path,(plan,),shapes=shapes,hardware=hardware)
            with (patch('lqcdweave.contraction.planning.api.plan_scene_storage',side_effect=AssertionError('no planning')),
                    patch('lqcdweave.contraction.planning.api.lower_dense_batches',side_effect=AssertionError('no lowering'))):
                restored,_=load_storage_schedule(path,shapes=shapes,hardware=hardware,
                    expected=(plan.state.problem,),blocks={},preparations=plan.preparations)
                arrays=self.arrays(plan)
                actual=replay_scene_storage(restored[0],ArrayLoader(arrays))
        self.assertEqual(restored[0].kernel_batches,plan.kernel_batches)
        for term,value in zip(plan.state.problem.terms,actual):
            np.testing.assert_allclose(value,arrays[term.inputs[0]]@arrays[term.inputs[1]])

    def test_numpy_high_rank_matrix_choice_survives_archive_without_lowering(self):
        plan = self.plan(high_rank=True).plan
        self.assertTrue(plan.kernel_batches)
        self.assertTrue(all(batch.packing_dtype == 'complex128' for batch in plan.kernel_batches))
        shapes = {k: list(s.shape) for k, s in plan.state.problem.specs.items()}
        hardware = {'backend': 'numpy', 'memory': 10**8}
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'high-rank.json'
            save_storage_schedule(path, (plan,), shapes=shapes, hardware=hardware)
            with patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                       side_effect=AssertionError('saved implementation must not be reselected')):
                restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
                    expected=(plan.state.problem,), blocks={}, preparations=plan.preparations)
                arrays = self.arrays(plan)
                actual = replay_scene_storage(restored[0], ArrayLoader(arrays))
        self.assertEqual(restored[0].kernel_batches, plan.kernel_batches)
        for term, value in zip(plan.state.problem.terms, actual):
            expected = np.einsum(arrays[term.inputs[0]], [0, 1, 2],
                                 arrays[term.inputs[1]], [2, 3, 4], [0, 1, 3, 4], optimize=False)
            np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)

    def test_numpy_high_rank_packing_budget_rejects_lowering_offline(self):
        from lqcdweave.contraction.planning.api import plan_scene_storage
        lowered = self.plan(high_rank=True).plan
        plain = plan_scene_storage(lowered.state, lowered.blocks, lowered.preparations,
            profile=CostProfile(), memory_limit=10**8)
        limited = replace(plain, memory_limit=plain.peak_host_bytes)
        selected, stats = lower_dense_batches(limited, profile=CostProfile())
        self.assertIs(selected, limited)
        self.assertEqual(stats['reason'], 'storage_limit')
        self.assertGreater(stats['candidate_peak_host_bytes'], limited.memory_limit)

    def test_shared_allocation_survives_individual_view_release(self):
        from lqcdweave.contraction.planning.api import ContractionState, plan_scene_storage
        from lqcdweave.contraction.planning.canonical_operations import describe, replay
        original=self.plan(three=True).plan
        state=ContractionState.initial(original.state.problem)
        profile=CostProfile()
        for key in dict.fromkeys(k for term in state.problem.terms for k in term.inputs):
            state=state.load(key,profile,1<<100)
        for scene,positions in ((0,(0,1)),(1,(0,1)),(0,(0,1)),(1,(0,1))):
            action,*_=describe(state,scene,positions,profile=profile,
                output_order=state.pending[scene].output if len(state.pending[scene].factors)==2 else None)
            state=replay(state,action,profile)
        plain=plan_scene_storage(state,{},original.preparations,profile=profile,memory_limit=10**8)
        plan,_=lower_dense_batches(plain,profile=profile)
        self.assertTrue(plan.kernel_batches)
        self.assertTrue(any(p.retained_host_bytes>0 for p in plan.memory))
        arrays=self.arrays(plan)
        actual=replay_scene_storage(plan,ArrayLoader(arrays))
        for term,value in zip(plan.state.problem.terms,actual):
            np.testing.assert_allclose(value,arrays[term.inputs[0]]@arrays[term.inputs[1]]@arrays[term.inputs[2]])

    def test_corrupt_slice_or_memory_refused_before_materialization(self):
        plan=self.plan().plan
        batch=plan.kernel_batches[0]
        bad=replace(batch.results[0],bounds=((1,3),(0,3)))
        for corrupt in (replace(plan,kernel_batches=(replace(batch,results=(bad,*batch.results[1:])),)),
                replace(plan,memory=tuple(replace(p,retained_host_bytes=0) for p in plan.memory))):
            loader=ArrayLoader(self.arrays(plan))
            with self.assertRaises(ValueError):
                replay_scene_storage(corrupt,loader)
            self.assertEqual(loader.reads,[])

    def test_storage_cap_fallback_preserves_existing_route(self):
        lowered=self.plan().plan
        from lqcdweave.contraction.planning.api import plan_scene_storage
        plain=plan_scene_storage(lowered.state,lowered.blocks,lowered.preparations,
            profile=CostProfile(),memory_limit=10**8)
        limit=plain.peak_host_bytes
        candidate,stats=lower_dense_batches(replace(plain,memory_limit=limit),profile=CostProfile())
        self.assertGreater(lowered.peak_host_bytes,limit)
        self.assertEqual(stats['reason'],'storage_limit')
        self.assertIs(candidate.state,plain.state)
        self.assertFalse(candidate.kernel_batches)

    def test_exact_keeps_its_binary_proof_domain(self):
        heuristic=self.plan().plan
        with patch('lqcdweave.contraction.planning.dense_batch_schedule.lower_dense_batches',
                   side_effect=AssertionError('exact must not lower into the new kernel domain')):
            result=solve_storage(heuristic.state.problem,{},heuristic.preparations,
                optimizer='exact',memory_limit=10**8,deadline_seconds=5,max_states=10000)
        self.assertEqual(result.status,'OPTIMAL')
        self.assertFalse(result.plan.kernel_batches)
        self.assertGreater(result.plan.compute_seconds,heuristic.compute_seconds)

    def test_device_backend_mismatch_refused_before_io(self):
        plan=self.plan().plan
        loader=ArrayLoader(self.arrays(plan))
        with self.assertRaises(ValueError):
            replay_scene_storage(plan,loader,backend=SimpleNamespace(__name__='cupy'))
        self.assertFalse(loader.reads)

    def test_cupy_public_lowering_and_device_capacity_audit(self):
        plan=self.plan(backend='cupy').plan
        self.assertTrue(plan.kernel_batches)
        self.assertTrue(all(b.packing_dtype=='complex128' for b in plan.kernel_batches))
        self.assertEqual(plan.peak_host_bytes,0)
        self.assertGreater(plan.peak_device_bytes,0)
        self.assertTrue(any(p.retained_device_bytes for p in plan.memory))
        validate_executable_resources(plan)
        from lqcdweave.contraction.planning.api import plan_scene_storage
        plain=plan_scene_storage(plan.state,{},plan.preparations,profile=CostProfile(),memory_limit=10**8)
        if plan.peak_device_bytes>plain.peak_device_bytes:
            selected,stats=lower_dense_batches(replace(plain,memory_limit=plain.peak_device_bytes),profile=CostProfile())
            self.assertEqual(stats['reason'],'storage_limit')
            self.assertFalse(selected.kernel_batches)
        bad=replace(plan,memory=tuple(replace(p,retained_device_bytes=0) for p in plan.memory))
        with self.assertRaises(ValueError):
            validate_executable_resources(bad)
        loader=ArrayLoader(self.arrays(plan))
        with self.assertRaisesRegex(ValueError,'backend disagrees'):
            replay_scene_storage(plan,loader,backend=np)
        self.assertFalse(loader.reads)

    def test_cupy_protocol_replays_archived_plan_without_host_contractions(self):
        plan=self.plan(backend='cupy').plan
        arrays=self.arrays(plan)
        calls=[]
        def matmul(a,b):
            calls.append((a.dtype,b.dtype,a.flags.c_contiguous,b.flags.c_contiguous))
            return np.matmul(a,b)
        backend=SimpleNamespace(__name__='cupy',array=np.array,empty=np.empty,
            concatenate=np.concatenate,matmul=matmul)
        restored=decode_plan_value(encode_plan_value(plan))
        with patch('lqcdweave.contraction.planning.api.lower_dense_batches',side_effect=AssertionError('no lowering')):
            actual=replay_scene_storage(restored,ArrayLoader(arrays),backend=backend,io_overlap=False)
        self.assertEqual(len(calls),1)
        self.assertTrue(all(c and d for _,_,c,d in calls))
        for term,value in zip(plan.state.problem.terms,actual):
            np.testing.assert_allclose(value,arrays[term.inputs[0]]@arrays[term.inputs[1]])

    def test_producer_layout_eliminates_single_side_pack_and_preserves_logical_axes(self):
        specs={'vertex':TensorSpec((2,3,5),np.complex128),'coef':TensorSpec((4,2),np.complex128)}
        specs.update({f'b{i}':TensorSpec((5,3+i),np.complex128) for i in range(4)})
        problem=ContractionProblem(specs,tuple(ContractionTerm(('vertex','coef',f'b{i}'),'klm,nk,mx->nlx')
                                   for i in range(4)),{k:SourceSpec(k,0,read_bytes=0) for k in specs},backend='cupy')
        result=solve_storage(problem,{}, {k:SourcePreparation(()) for k in specs},optimizer='anytime',kernel_batching=True,
            memory_limit=10**8,deadline_seconds=.1,max_states=1000,root_width=1,branch_width=1, component_workers=1)
        plan=result.plan
        self.assertEqual(result.statistics['dense_kernel_lowering']['producer_layouts'],1)
        batch=plan.kernel_batches[0]
        self.assertEqual(batch.prepared_sides,(True,False))
        arrays=self.arrays(plan)
        packs=[]
        def concatenate(values,**kwargs):
            packs.append(kwargs['out'])
            return np.concatenate(values,**kwargs)
        backend=SimpleNamespace(__name__='cupy',array=np.array,empty=np.empty,
            transpose=np.transpose,reshape=np.reshape,concatenate=concatenate,matmul=np.matmul)
        actual=replay_scene_storage(plan,ArrayLoader(arrays),backend=backend,io_overlap=False)
        self.assertEqual(len(packs),1)
        for i,value in enumerate(actual):
            np.testing.assert_allclose(value,np.einsum('klm,nk,mx->nlx',arrays['vertex'],arrays['coef'],arrays[f'b{i}']))
        # A source has no known generated layout; such an annotation must fail.
        source_plan=self.plan(backend='cupy').plan
        bad=replace(source_plan.kernel_batches[0],prepared_sides=(True,False))
        with self.assertRaisesRegex(ValueError,'generated'):
            validate_executable_resources(replace(source_plan,kernel_batches=(bad,)))


class DenseBatchCupyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
            cls.cp=cupy
        except (ImportError,RuntimeError) as exc:
            raise unittest.SkipTest(f'CUDA unavailable: {exc}') from exc

    def test_real_cupy_uniform_dtype_ragged_pack_and_output_lifetime(self):
        actions,specs,profile=DenseBatchTests().fixture()
        batch=describe_dense_batch(actions,(0,1,2,3),specs,profile,backend='cupy')
        rng=np.random.default_rng(337)
        host={k:(rng.normal(size=s.shape)+1j*rng.normal(size=s.shape)).astype(s.dtype)
              for k,s in specs.items() if k.startswith(('a','b'))}
        arrays={k:self.cp.asarray(v)[...,::-1] for k,v in host.items()}
        output=contract_dense_batch(batch,arrays,backend=self.cp)
        self.cp.cuda.get_current_stream().synchronize()
        for i in range(2):
            for j in range(2):
                expected=host[f'a{i}'][:,::-1]@host[f'b{j}'][:,::-1]
                np.testing.assert_allclose(self.cp.asnumpy(arrays[f'c{i}{j}']),expected,rtol=2e-12,atol=2e-12)
        saved=self.cp.asnumpy(arrays['c11'])
        for key in tuple(host):
            arrays[key][...]=0
            arrays.pop(key)
        arrays.pop('c00')
        del output
        np.testing.assert_array_equal(self.cp.asnumpy(arrays['c11']),saved)

    def test_real_cupy_saved_plan_and_cuda_phase_measurement(self):
        helper=DenseBatchIntegrationTests()
        plan=helper.plan(backend='cupy').plan
        arrays=helper.arrays(plan)
        shapes={k:list(s.shape) for k,s in plan.state.problem.specs.items()}
        hardware={'backend':'cupy','device_memory':10**8}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'gpu.json'
            save_storage_schedule(path,(plan,),shapes=shapes,hardware=hardware)
            restored,_=load_storage_schedule(path,shapes=shapes,hardware=hardware,
                expected=(plan.state.problem,),blocks={},preparations=plan.preparations)
            report=replay_scene_storage(restored[0],ArrayLoader(arrays),backend=self.cp,
                                       io_overlap=False,measure=True)
        for term,actual in zip(plan.state.problem.terms,report.values):
            np.testing.assert_allclose(self.cp.asnumpy(actual),arrays[term.inputs[0]]@arrays[term.inputs[1]])
        phases=[p for t in report.timings for p in t.phases if p.name in ('dense_batch_pack','contract_kernel')]
        self.assertEqual(len(phases),3)
        self.assertTrue(all(p.device_seconds is not None for p in phases))
        self.assertAlmostEqual(sum(t.estimated_seconds for t in report.timings),plan.seconds)


if __name__ == '__main__':
    unittest.main()
