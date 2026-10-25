"""Saved full-Gram strategy: axis proof, explicit resources and current values."""
from dataclasses import replace
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern,FactorizedWideReduction,TensorSpec,ContractionTerm,ContractionProblem,
    SourceSpec,SourcePreparation,Action,
)
from lqcdweave.contracts.correlator.allocation import device_allocation_bytes
from lqcdweave.contraction.planning.api import ContractionState,CostProfile
from lqcdweave.contraction.planning.factorized_reduction import (
    describe_factorized_action,replay_factorized_action,factorized_result_name,
)
from lqcdweave.contraction.planning.factorized_gemm import _lower_plan
from lqcdweave.contraction.execution.factorized_reduction import bind_factorized_reduction
from lqcdweave.storage.contraction_plans.api import encode_plan_value,decode_plan_value


def fixture(*,m=96,n=96,k=256,strategy='reduction',output_dtype=None,output_shape=(2,3)):
    from lqcdweave.contraction.planning.storage import plan_scene_storage
    shapes=((m,k),(n,k),(output_shape[0],m),(output_shape[1],n))
    term=ContractionTerm(tuple(str(i) for i in range(4)),'ak,bk,ca,db->cd')
    a,kaxis=term.subscripts.inputs[0];b=term.subscripts.inputs[1][0]
    term=replace(term,sum_barriers=(((kaxis,),(a,b)),((a,),(kaxis,b)),((b,),(kaxis,a))))
    specs={str(i):TensorSpec(s,'complex64') for i,s in enumerate(shapes)}
    problem=ContractionProblem(specs,(term,),{key:SourceSpec(key,0,read_bytes=0) for key in specs},backend='cupy')
    state=ContractionState.initial(problem);profile=CostProfile()
    for key in specs:state=state.load(key,profile,1<<30)
    action,_,_,_=describe_factorized_action(state,0,tuple(range(4)),(0,1),profile=profile,
        output_order=term.subscripts.output,tile_elements=4096,output_dtype=output_dtype,
        gram_strategy=strategy)
    before=state;state=replay_factorized_action(state,action,profile)
    for key in specs:state=state.release(key)
    plan=plan_scene_storage(state,{}, {key:SourcePreparation((),dtype='complex64') for key in specs},
        profile=profile,memory_limit=1<<30,host_memory_limit=1<<30)
    return before,action,plan,profile


class FactorizedGemmCpuTests(unittest.TestCase):
    def test_retile_and_gemm_preserve_separate_saved_numerical_identities(self):
        from lqcdweave.contraction.planning.factorized_tiles import retile_factorized_reductions
        from lqcdweave.contraction.planning.storage import plan_scene_storage
        before,_,old,profile=fixture(m=192,n=192,output_shape=(15,15))
        whole,stats=retile_factorized_reductions(old,profile=profile,target_runtime='cuda')
        self.assertTrue(stats['applied'])
        self.assertEqual(next(a.core for a in whole.state.actions if a.core is not None).tile_width,36864)
        gemm,stats=_lower_plan(whole,profile)
        self.assertTrue(stats['applied'])
        core=next(a.core for a in gemm.state.actions if a.core is not None)
        self.assertEqual((core.gram_strategy,core.kernel_calls),('gemm',3))
        self.assertEqual(decode_plan_value(encode_plan_value(gemm)).state.results,gemm.state.results)
        # Re-describing a mixed selected route must retain an existing GEMM
        # numerical identity when another reduction changes its saved tile.
        problem=replace(before.problem,terms=before.problem.terms*2)
        state=ContractionState.initial(problem)
        for key in problem.sources:state=state.load(key,profile,1<<30)
        for index,strategy in enumerate(('gemm','reduction')):
            action,_,_,_=describe_factorized_action(state,index,(0,1,2,3),(0,1),
                profile=profile,output_order=problem.terms[index].subscripts.output,
                tile_elements=4096,gram_strategy=strategy)
            state=replay_factorized_action(state,action,profile)
        for key in problem.sources:state=state.release(key)
        mixed=plan_scene_storage(state,{},old.preparations,profile=profile,
            memory_limit=1<<30,host_memory_limit=1<<30)
        selected,stats=retile_factorized_reductions(mixed,profile=profile,target_runtime='cuda')
        self.assertTrue(stats['applied'])
        cores=[a.core for a in selected.state.actions if a.core is not None]
        self.assertEqual(tuple(c.gram_strategy for c in cores),('gemm','reduction'))
        self.assertEqual(selected.state.results[0],mixed.state.results[0])
        self.assertNotEqual(selected.state.results[1],mixed.state.results[1])

    def test_large_lowered_plan_publishes_serializable_tuple_rows(self):
        from lqcdweave.contraction.planning.storage import plan_scene_storage
        from lqcdweave.contraction.planning.state import _RowTable
        before, _, plan, profile = fixture(m=192,n=192,output_shape=(15,15))
        problem = replace(before.problem, terms=before.problem.terms * (_RowTable.block_size + 1))
        state = ContractionState.initial(problem)
        for key in problem.sources:
            state = state.load(key, profile, 1 << 30)
        for index in range(len(problem.terms)):
            action, _, _, _ = describe_factorized_action(
                state, index, tuple(range(4)), (0, 1), profile=profile,
                output_order=problem.terms[index].subscripts.output, tile_elements=4096)
            state = replay_factorized_action(state, action, profile)
        for key in problem.sources:
            state = state.release(key)
        plan = plan_scene_storage(state, {}, plan.preparations, profile=profile,
            memory_limit=1 << 30, host_memory_limit=1 << 30)
        candidate, stats = _lower_plan(plan, profile)
        self.assertTrue(stats['applied'])
        self.assertIs(type(candidate.state.pending), tuple)
        self.assertIs(type(candidate.state.results), tuple)
        restored = decode_plan_value(encode_plan_value(candidate))
        self.assertEqual(restored.state.results, candidate.state.results)

    def test_failed_library_submission_fences_stream_and_does_not_capture_arrays(self):
        import gc,weakref
        from types import SimpleNamespace
        from unittest.mock import patch
        _,action,plan,_=fixture(strategy='gemm')
        error=RuntimeError('GEMM submission failure');fenced=[]
        def fail(*args,**kwargs):raise error
        backend=SimpleNamespace(__name__='cupy',RawKernel=object,matmul=fail,
            empty=lambda *args,**kwargs:SimpleNamespace(data=0),
            ndarray=lambda shape,dtype,**kwargs:np.zeros(shape,dtype),
            cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(
                synchronize=lambda:fenced.append(True))))
        with patch('lqcdweave.contraction.execution.factorized_reduction._kernel',return_value=object()), \
             patch('lqcdweave.contraction.execution.factorized_reduction.bind_pointer_arguments',return_value=None), \
             patch('lqcdweave.contraction.execution.operations.packing_kernel.bind_partition_packing',
                   return_value=lambda *args:None):
            run=bind_factorized_reduction(action,plan.state.catalog,backend)
            values=tuple(np.zeros(s,'complex64') for s in action.core.shapes)
            refs=tuple(weakref.ref(value) for value in values)
            with self.assertRaisesRegex(RuntimeError,'submission failure'):run(values)
        self.assertEqual(fenced,[True])
        error.__traceback__=None;del values;gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_coordinate_lift_rebudgets_gemm_to_reduction_and_keeps_explicit_identity(self):
        from correlator.planning.test_coordinate_path import batch_problem,LANE
        from lqcdweave.contraction.planning.coordinate_path import lift_coordinate_path
        _,_,plan,profile=fixture(strategy='gemm')
        problem,names=batch_problem(plan.state.problem,set(plan.state.problem.specs),2,rename=True)
        problem=replace(problem,terms=tuple(replace(new,sum_barriers=old.sum_barriers)
            for new,old in zip(problem.terms,plan.state.problem.terms,strict=True)))
        state=lift_coordinate_path(plan.state,problem,names,lane_axis=LANE,profile=profile,
            memory_limit=1<<30,debug=True)
        action=next(a for a in state.actions if a.core is not None)
        self.assertEqual(action.core.gram_strategy,'reduction')
        self.assertIn(LANE,action.equation.output)

    def test_geometry_identity_rounding_exact_admission_and_legacy_codec(self):
        before,action,_,profile=fixture(strategy='gemm')
        core=action.core
        reduction=replace(core,gram_strategy='reduction')
        self.assertNotEqual(core,reduction)
        self.assertNotEqual(factorized_result_name(action.operands,core),
                            factorized_result_name(action.operands,reduction))
        self.assertIsNot(core._geometry,reduction._geometry)
        self.assertEqual((core.blocks,core.tiles,core.tile_width,core.kernel_calls),(1,1,96*96,3))
        self.assertEqual(core.partial_bytes,96*96*16)
        output=core.output_elements*np.dtype(core.result_dtype).itemsize
        packed=sum(device_allocation_bytes(np.prod(core.shapes[i]).item()*16) for i in core.large_operands)
        cuda_peak=packed+device_allocation_bytes(core.partial_bytes)+device_allocation_bytes(output)
        reference=core.gram_elements*16+core.output_elements*16+sum(np.prod(s).item()*16 for s in core.shapes)
        self.assertEqual(core.workspace_bytes,max(cuda_peak-output,reference))
        bound=before.resident_bytes+output+core.workspace_bytes
        self.assertTrue(replay_factorized_action(before,action,profile,int(bound)).done)
        with self.assertRaises(MemoryError):replay_factorized_action(before,action,profile,int(bound)-1)
        self.assertEqual(decode_plan_value(encode_plan_value(core)),core)
        old=encode_plan_value(reduction)
        # The public codec keeps optional missing fields compatible.
        def omit(value):
            if isinstance(value,dict):
                value.pop('gram_strategy',None)
                for item in value.values():omit(item)
            elif isinstance(value,list):
                for item in value:omit(item)
        omit(old)
        self.assertEqual(decode_plan_value(old),reduction)

    def test_planner_selects_only_cost_and_resource_feasible_strategy(self):
        _,_,plan,profile=fixture(m=192,n=192,output_shape=(15,15))
        candidate,stats=_lower_plan(plan,profile)
        self.assertTrue(stats['applied'],stats)
        actions=[a for a in candidate.state.actions if a.core is not None]
        self.assertEqual(len(actions),1)
        self.assertEqual(actions[0].core.gram_strategy,'gemm')
        self.assertNotEqual(candidate.state.results,plan.state.results)
        self.assertEqual(tuple(a.kind for a in candidate.state.actions),tuple(a.kind for a in plan.state.actions))
        self.assertEqual(plan.state.actions[4].core.gram_strategy,'reduction')
        tight=replace(plan,memory_limit=plan.peak_device_bytes)
        selected,stats=_lower_plan(tight,profile)
        self.assertIs(selected,tight)
        self.assertFalse(stats['applied'])

    def test_small_output_domain_keeps_selected_reduction_and_its_budget(self):
        _,_,plan,profile=fixture()
        candidate,stats=_lower_plan(plan,profile)
        self.assertIs(candidate,plan)
        self.assertEqual(stats['reason'],'no_amortized_matrix_gram')

    def test_split_shared_lane_small_or_overlapping_small_axes_are_rejected(self):
        _,action,_,_=fixture()
        core=action.core
        for changes in (dict(tile_elements=None),dict(shapes=((96,2048),(96,2048),(2,96),(3,96))),
                        dict(shapes=((96,7),(96,7),(2,96),(3,96)))):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                replace(core,gram_strategy='gemm',**changes)
        aligned=FactorizedWideReduction(AxisPattern(((0,1,2),(0,1,2),(3,0),(4,1)),(3,4)),
            ((65,65,256),(65,65,256),(2,65),(3,65)),(0,1),tile_elements=4096)
        with self.assertRaises(ValueError):replace(aligned,gram_strategy='gemm')
        shared=replace(core,equation=AxisPattern(((9,0,2),(9,1,2),(3,0),(4,1)),(9,3,4)),
            shapes=((2,96,256),(2,96,256),(2,96),(3,96)))
        with self.assertRaises(ValueError):replace(shared,gram_strategy='gemm')

    def test_public_strategy_alias_and_archive_cannot_introduce_unbudgeted_gemm(self):
        from lqcdweave.contraction.gram_strategy import resolve_gram_strategy,new_gram_strategy,archived_gram_strategy
        from lqcdweave.contraction.calculation import Calculation
        self.assertEqual(new_gram_strategy(resolve_gram_strategy(None,None),'cuda'),'gemm')
        self.assertEqual(new_gram_strategy(resolve_gram_strategy(None,False),'cuda'),'reduction')
        self.assertEqual(new_gram_strategy(resolve_gram_strategy(None,True),'cuda'),'tiled')
        self.assertEqual(new_gram_strategy(None,'cpu'),'reduction')
        self.assertEqual(new_gram_strategy(None,'hip'),'reduction')
        self.assertEqual(resolve_gram_strategy(None,True),'tiled')
        self.assertEqual(resolve_gram_strategy('reduction',False),'reduction')
        with self.assertRaises(ValueError):resolve_gram_strategy('gemm',True)
        with self.assertRaises(ValueError):resolve_gram_strategy('other',None)
        with self.assertRaises(ValueError):archived_gram_strategy(Calculation(),'gemm')

    def test_saved_gemm_archive_uses_tiled_fallback_in_its_owner_only(self):
        from unittest.mock import patch
        from lqcdweave.contraction.calculation import Calculation
        from lqcdweave.contraction.correlator import calculate
        from lqcdweave.contraction.execution.binding_cache import factorized_gram_tiling_enabled
        _, _, plan, _ = fixture(strategy='gemm')
        original = Calculation(_session_plan=plan)
        with patch('lqcdweave.contraction.correlator.load_calculation_plan', return_value=original):
            selected = calculate(plan_archive='fixed.json.gz', execute=False)
            with self.assertRaises(ValueError):
                calculate(plan_archive='fixed.json.gz', execute=False, factorized_gram_strategy='reduction')
        self.assertEqual(selected.options['factorized_gram_strategy'], 'gemm')
        self.assertTrue(selected.options['factorized_gram_tiling'])
        self.assertFalse(original._execution_bindings.factorized_gram_tiling)
        with selected._execution_bindings.factory_scope():
            self.assertTrue(factorized_gram_tiling_enabled())
        self.assertFalse(factorized_gram_tiling_enabled())


class FactorizedGemmCuPyTests(unittest.TestCase):
    def test_saved_strategy_large_tail_negative_strides_wide_output_stream_and_release(self):
        import gc,weakref
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:self.skipTest(f'CUDA unavailable: {error}')
        rng=np.random.default_rng(261059)
        for dtype in (None,'complex128'):
            _,action,plan,_=fixture(m=65,n=65,output_dtype=dtype,strategy='gemm')
            core=action.core;self.assertGreater(core.epilogue_elements,4096)
            run=bind_factorized_reduction(action,plan.state.catalog,cp)
            from types import SimpleNamespace
            from unittest.mock import patch
            from lqcdweave.contraction.execution.device import _active
            from lqcdweave.contraction.execution.reduction_workspace import ReductionWorkspace
            refs=[];streams=tuple(cp.cuda.Stream(non_blocking=True) for _ in range(2))
            workspace=ReductionWorkspace(cp)
            token=_active.set(SimpleNamespace(backend=cp,reduction_partial=workspace.borrow))
            linalg,cublas=cp._core._routines_linalg,cp.cuda.cublas
            handle=cp.cuda.Device().cublas_handle
            saved_compute=linalg.get_compute_type(np.complex128)
            saved_math=cublas.getMathMode(handle)
            try:
                for cfg,stream in enumerate(streams):
                    host=tuple((rng.normal(size=s)+1j*rng.normal(size=s)).astype('complex64')/np.float32(8) for s in core.shapes)
                    with stream:
                        if cfg==0:workspace.configure(core.workspace_bytes)
                        values=tuple(cp.asarray(v[...,::-1])[...,::-1] for v in host)
                        refs.extend(weakref.ref(v) for v in values)
                        if cfg:
                            if dtype is None:linalg.set_compute_type(np.complex128,linalg.COMPUTE_TYPE_PEDANTIC)
                            else:cublas.setMathMode(handle,cublas.CUBLAS_TENSOR_OP_MATH)
                        with patch.object(cp,'matmul',wraps=cp.matmul) as fallback:
                            result,owned=run(values);actual=result.get()
                        self.assertEqual(fallback.call_count,cfg)
                        if cfg:
                            self.assertEqual(fallback.call_args.kwargs['out'].shape,(65,65))
                        del fallback
                        linalg.set_compute_type(np.complex128,saved_compute)
                        cublas.setMathMode(handle,saved_math)
                    self.assertTrue(owned);self.assertEqual(result.dtype,np.dtype(core.result_dtype))
                    labels={axis:index for index,axis in enumerate(core.dimensions)}
                    args=tuple(item for value,axes in zip(host,core.equation.inputs,strict=True)
                        for item in (value.astype('complex128'),tuple(labels[a] for a in axes)))+(
                            tuple(labels[a] for a in core.equation.output),)
                    high=np.einsum(*args,optimize=True)
                    error=np.abs(actual-high);limit=5e-5*np.maximum(np.maximum(np.abs(actual),np.abs(high)),1e-12)+2e-8
                    self.assertTrue(np.all(error<=limit),float(np.max(error/limit)))
                    del values,result
                self.assertGreaterEqual(workspace.hits,1)
                self.assertGreaterEqual(workspace.misses,1)
            finally:
                linalg.set_compute_type(np.complex128,saved_compute)
                cublas.setMathMode(handle,saved_math)
                for stream in streams:stream.synchronize()
                workspace.close();_active.reset(token)
            gc.collect();self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(cp.get_default_memory_pool().used_bytes(),0)
