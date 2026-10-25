"""Matrix Gram reuse keeps saved tiled partial coordinates and arithmetic."""
from collections import Counter
from dataclasses import replace
import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import Action,AxisPattern,FactorizedWideReduction,TensorSpec
from lqcdweave.contraction.execution.factorized_gram import matrix_gram_geometry,matrix_gram_partial
from lqcdweave.contraction.execution.factorized_reduction import bind_factorized_reduction,factorized_sources


def core_for(m,n,*,width=4096,lanes=1,three=False):
    if three:
        equation=AxisPattern(((4,0,2),(1,2),(3,0,1)),(3,4))
        shapes=((lanes,m,7),(n,7),(3,m,n))
    else:
        equation=AxisPattern(((4,0,2),(4,1,2),(3,0),(5,1)),(5,4,3))
        shapes=((lanes,m,7),(lanes,n,7),(2,m),(3,n))
    return FactorizedWideReduction(equation,shapes,(0,1),tile_elements=width)


class FactorizedGramCpuTests(unittest.TestCase):
    def test_window_output_applies_option_until_binding_and_restores_scope(self):
        from tempfile import TemporaryDirectory
        from test_cfg_window_workflow import CfgWindowWorkflowTests
        from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation
        from lqcdweave.contraction.execution import cfg_runtime
        from lqcdweave.contraction.execution.binding_cache import factorized_gram_tiling_enabled
        selected,coordinates,_,factory,_=CfgWindowWorkflowTests().setup_case()
        selected=replace(selected,options=dict(selected.options,factorized_gram_tiling=True))
        seen=[];bind=cfg_runtime.bind_scene_execution
        def record(*args,**kwargs):
            seen.append(factorized_gram_tiling_enabled())
            return bind(*args,**kwargs)
        with TemporaryDirectory() as directory,patch.object(cfg_runtime,'bind_scene_execution',record):
            result=run_cfg_window_calculation(selected,coordinates=coordinates,configuration='a',
                source_factory=factory,output=directory,window_size=1)
            self.assertEqual(result.report['completed'],('a',))
        self.assertTrue(seen and all(seen))
        self.assertFalse(factorized_gram_tiling_enabled())

    def test_public_fixed_option_reaches_binding_and_keeps_other_calls_default(self):
        from test_common_execution_bindings import CommonExecutionBindingsTests
        from lqcdweave.contraction.execution import storage_runtime
        from lqcdweave.contraction.execution.binding_cache import factorized_gram_tiling_enabled
        helper=CommonExecutionBindingsTests();seen=[];bind=storage_runtime.bind_scene_execution
        def record(*args,**kwargs):
            seen.append(factorized_gram_tiling_enabled())
            return bind(*args,**kwargs)
        with patch.object(storage_runtime,'bind_scene_execution',record):
            selected=helper.calculation(factorized_gram_tiling=True)
            selected.run()
            self.assertTrue(seen and all(seen))
            seen.clear();helper.calculation().run()
            self.assertTrue(seen and not any(seen))
        self.assertFalse(factorized_gram_tiling_enabled())

    def test_complete_archive_iterator_forwards_option_without_archiving_it(self):
        import gzip
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from test_cfg_execution_binding_cache import ExecutionBindingCacheTests
        from lqcdweave.contraction.correlator import calculate,save_calculation_plan
        from lqcdweave.contraction.execution import cfg_runtime
        from lqcdweave.contraction.execution.binding_cache import factorized_gram_tiling_enabled
        calculation,_,_=ExecutionBindingCacheTests().fixture()
        reference=calculation.run().value['two_point']
        seen=[];bind=cfg_runtime.bind_scene_execution
        def record(*args,**kwargs):
            seen.append(factorized_gram_tiling_enabled())
            return bind(*args,**kwargs)
        with TemporaryDirectory() as directory:
            archive=Path(directory)/'fixed.json.gz'
            save_calculation_plan(archive,replace(calculation,options=dict(calculation.options,
                factorized_gram_tiling=True)))
            self.assertNotIn(b'factorized_gram_tiling',gzip.decompress(archive.read_bytes()))
            with patch.object(cfg_runtime,'bind_scene_execution',record):
                selected=calculate(plan_archive=archive,configuration=iter(('a','b')),
                    data=calculation.options['source_factory'],factorized_gram_tiling=True,execute=False)
                actual=dict(selected.run().value)
            self.assertTrue(seen)
            self.assertTrue(all(seen))
            np.testing.assert_array_equal(actual['a']['two_point'],reference)
            np.testing.assert_allclose(actual['b']['two_point'],16*reference,rtol=3e-12,atol=3e-12)
            self.assertFalse(factorized_gram_tiling_enabled())

    def test_public_archive_option_is_execution_only_and_owner_scoped(self):
        from dataclasses import replace
        from lqcdweave.contraction.calculation import Calculation
        from lqcdweave.contraction.correlator import calculate
        from lqcdweave.contraction.execution.binding_cache import factorized_gram_tiling_enabled
        original=Calculation(options={})
        with patch('lqcdweave.contraction.correlator.load_calculation_plan',return_value=original):
            enabled=calculate(plan_archive='fixed.json.gz',execute=False,factorized_gram_tiling=True)
            disabled=calculate(plan_archive='fixed.json.gz',execute=False)
        self.assertIs(enabled.plan,original.plan)
        self.assertIs(disabled._execution_bindings,original._execution_bindings)
        self.assertIsNot(enabled._execution_bindings,original._execution_bindings)
        self.assertTrue(enabled.options['factorized_gram_tiling'])
        self.assertFalse(factorized_gram_tiling_enabled())
        with enabled._execution_bindings.scope(object(),np):
            self.assertTrue(factorized_gram_tiling_enabled())
            with disabled._execution_bindings.scope(object(),np):
                self.assertFalse(factorized_gram_tiling_enabled())
            self.assertTrue(factorized_gram_tiling_enabled())
        self.assertFalse(factorized_gram_tiling_enabled())
        self.assertIs(replace(enabled,options={})._execution_bindings.factorized_gram_tiling,False)
        with self.assertRaises(TypeError):
            calculate(plan_archive='fixed.json.gz',execute=False,factorized_gram_tiling=1)

    def test_cfg_binding_applies_owner_option_and_restores_after_failure(self):
        from lqcdweave.contraction.execution.cfg_runtime import CfgExecutionCache
        from lqcdweave.contraction.execution.binding_cache import factorized_gram_tiling_enabled
        cache=CfgExecutionCache(factorized_gram_tiling=True)
        def fail(*args):
            self.assertTrue(factorized_gram_tiling_enabled())
            raise RuntimeError('binding failure')
        with patch('lqcdweave.contraction.execution.cfg_runtime._bind_cfg_tables',fail):
            with self.assertRaisesRegex(RuntimeError,'binding failure'):
                cache._bind(object(),np,None)
        self.assertFalse(factorized_gram_tiling_enabled())

    def test_default_general_and_opt_in_matrix_binding_use_separate_factories(self):
        from types import SimpleNamespace
        from lqcdweave.contraction.execution.binding_cache import ExecutionBindingCache
        from lqcdweave.contraction.execution.scene_runtime import bind_scene_execution
        core=core_for(65,65)
        specs={str(i):TensorSpec(shape,'complex64') for i,shape in enumerate(core.shapes)}
        specs['out']=TensorSpec(core.output_shape,core.result_dtype)
        action=Action('contract','out',tuple(str(i) for i in range(len(core.shapes))),
            core.equation,0.,False,core=core,wide_accumulation=True)
        state=SimpleNamespace(actions=(action,),catalog=specs)
        backend=SimpleNamespace(__name__='cupy',RawKernel=object,empty=np.empty)
        shared={};sources=[]
        def kernel(factory,source,name):
            if name=='ed_precise_partial':sources.append(source)
            return name
        values=tuple(np.zeros(s,'complex64') for s in core.shapes)
        with patch('lqcdweave.contraction.execution.factorized_reduction._kernel',kernel), \
             patch('lqcdweave.contraction.execution.factorized_reduction.bind_pointer_arguments',
                   return_value=lambda *args:None):
            for enabled in (False,True,False):
                with ExecutionBindingCache(factorized_gram_tiling=enabled).factory_scope():
                    table=bind_scene_execution(state,(),backend,kernel_cache=shared)
                    table.kernels[0](values)
        self.assertEqual(len(shared),2)
        self.assertEqual(len(sources),2)
        self.assertNotIn('const long long row =',sources[0])
        self.assertIn('const long long row =',sources[1])

    def test_grid_partitions_every_valid_tile_slot_once_including_retained_lanes(self):
        for m,n,width,lanes in ((3,5,8,1),(65,65,4096,1),(9,3072,4096,1),(3,5,8,2)):
            with self.subTest(m=m,n=n,width=width,lanes=lanes):
                core=core_for(m,n,width=width,lanes=lanes)
                _,_,_,rows,columns,row_groups,column_groups=matrix_gram_geometry(core)
                for start in range(0,m*n,core.tile_width):
                    count=min(core.tile_width,m*n-start);seen=Counter()
                    for z in range(lanes*row_groups):
                        batch=z//row_groups;row=start//columns+2*(z%row_groups)
                        for y in range(column_groups):
                            col=2*y
                            for r in range(2):
                                for c in range(2):
                                    point=(row+r)*columns+col+c
                                    if row+r<rows and col+c<columns and start<=point<start+count:
                                        seen[batch*core.tile_width+point-start]+=1
                    self.assertEqual(set(seen),{b*core.tile_width+e for b in range(lanes) for e in range(count)})
                    self.assertTrue(all(count==1 for count in seen.values()))

    def test_nonexclusive_interleaved_legacy_and_grid_limits_keep_general_kernel(self):
        core=core_for(3,5)
        self.assertIsNone(matrix_gram_geometry(replace(core,tile_elements=None)))
        self.assertIsNone(matrix_gram_geometry(core_for(3,5,width=4)))
        self.assertIsNone(matrix_gram_geometry(core_for(3,5,lanes=65536)))
        # A common internal axis is aligned, rather than an exclusive row/column.
        aligned=FactorizedWideReduction(AxisPattern(((0,1,2),(0,1,2),(3,0),(4,1)),(3,4)),
            ((3,5,7),(3,5,7),(2,3),(2,5)),(0,1),tile_elements=4096)
        self.assertIsNone(matrix_gram_geometry(aligned))
        interleaved=FactorizedWideReduction(AxisPattern(((0,1,2),(1,3,2),(4,0),(5,1,3)),(4,5)),
            ((3,5,7),(5,4,7),(2,3),(2,5,4)),(0,1),tile_elements=4096)
        self.assertIsNone(matrix_gram_geometry(interleaved))


class FactorizedGramCuPyTests(unittest.TestCase):
    def test_matrix_tail_retained_lanes_three_four_inputs_current_stream_and_release(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        rng=np.random.default_rng(261056)
        for three in (False,True):
            core=core_for(65,65,lanes=2,three=three)
            if three:core=replace(core,output_dtype='complex128')
            self.assertEqual(core.tiles,2)
            specs={str(i):TensorSpec(shape,'complex64') for i,shape in enumerate(core.shapes)}
            specs['out']=TensorSpec(core.output_shape,core.result_dtype)
            action=Action('contract','out',tuple(str(i) for i in range(len(core.shapes))),
                core.equation,0.,False,core=core,wide_accumulation=True)
            run=bind_factorized_reduction(action,specs,cp,factorized_gram_tiling=True)
            stream=cp.cuda.Stream(non_blocking=True);refs=[]
            for cfg in range(2):
                host=tuple((rng.normal(size=s)+1j*rng.normal(size=s)).astype('complex64') for s in core.shapes)
                with stream:
                    values=tuple(cp.asarray(v[...,::-1])[...,::-1] for v in host)
                    refs.extend(weakref.ref(v) for v in values)
                    if cfg==0:
                        strides=tuple(v.strides for v in values)
                        original=factorized_sources(core,strides)[0]
                        optimized,grid=matrix_gram_partial(core,strides)
                        old=cp.RawKernel(original,'ed_precise_partial',options=('--std=c++17','--fmad=false'))
                        new=cp.RawKernel(optimized,'ed_precise_partial',options=('--std=c++17','--fmad=false'))
                        a=cp.empty((2,core.tile_width,core.blocks),'complex128');b=cp.empty_like(a)
                        for start in range(0,core.epilogue_elements,core.tile_width):
                            count=min(core.tile_width,core.epilogue_elements-start)
                            old((core.partial_elements*core.blocks,),(256,),
                                (*(values[i] for i in core.large_operands),np.int64(start),np.int64(count),a))
                            new(grid,(256,),(*(values[i] for i in core.large_operands),np.int64(start),np.int64(count),b))
                            self.assertTrue(bool(cp.array_equal(a[:,:count],b[:,:count]).get()))
                        del a,b
                    actual,_=run(values);result=actual.get()
                args=tuple(item for v,axes in zip(host,core.equation.inputs,strict=True)
                    for item in (v.astype('complex128'),axes))+(core.equation.output,)
                expected=np.einsum(*args,optimize=True)
                error=np.abs(result-expected);limit=5e-5*np.maximum(np.maximum(np.abs(result),np.abs(expected)),1e-12)+2e-8
                self.assertTrue(np.all(error<=limit),float(np.max(error/limit)))
                del values,actual
            stream.synchronize();gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(cp.get_default_memory_pool().used_bytes(),0)
