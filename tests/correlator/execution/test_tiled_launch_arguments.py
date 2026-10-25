"""Bounded tile launch metadata and current pointer/scalar ABI regressions."""
import gc
import threading
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.raw_arguments import bind_pointer_arguments
from lqcdweave.contraction.execution.factorized_reduction import bind_factorized_reduction
from lqcdweave.contracts.correlator.api import Action, AxisPattern, FactorizedWideReduction, TensorSpec

from .test_raw_driver_arguments import _FakeBackend, _FakeRawKernel, _FakeArray
from .test_factorized_wide_reduction import tiled_fixture


class TiledLaunchArgumentsCpuTests(unittest.TestCase):
    def test_dynamic_scalar_is_thread_local_during_overlapping_submissions(self):
        backend=_FakeBackend(barrier=threading.Barrier(2))
        constant=_FakeArray(backend.registry,[10,20,30])
        binding=bind_pointer_arguments(backend,_FakeRawKernel(backend),(1,),(32,),
            (None,None,constant,None),((0,0),),scalar_positions=((1,0),))
        outputs,errors=[],[]
        def worker(scalar):
            try:
                source=_FakeArray(backend.registry,[1,2,3])
                result=_FakeArray(backend.registry,[0,0,0])
                binding((source,),result,(scalar,))
                outputs.append((scalar,result.array.copy()))
            except BaseException as error:
                errors.append(error)
        threads=[threading.Thread(target=worker,args=(scalar,)) for scalar in (-7,4096)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(timeout=15)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertFalse(errors,errors)
        self.assertEqual(len(outputs),2)
        for scalar,output in outputs:
            np.testing.assert_array_equal(output,[11+scalar,22+scalar,33+scalar])
        addresses=[record['parameter_addresses'] for record in backend.driver.records]
        self.assertTrue(set(addresses[0]).isdisjoint(addresses[1]))

    def test_dynamic_signed_scalar_uses_current_thread_slots_and_fallback_cleanup(self):
        for driver in (False, True):
            backend = _FakeBackend(launch=driver)
            kernel = _FakeRawKernel(backend)
            constant = _FakeArray(backend.registry, [10, 20, 30])
            binding = bind_pointer_arguments(backend, kernel, (1,), (32,),
                (None, None, constant, None), ((0, 0),), scalar_positions=((1, 0),))
            refs = []
            for scalar in (-7, 4096):
                source = _FakeArray(backend.registry, [1, 2, 3])
                result = _FakeArray(backend.registry, [0, 0, 0])
                refs.extend((weakref.ref(source), weakref.ref(result)))
                binding((source,), result, (scalar,))
                np.testing.assert_array_equal(result.array, [11+scalar, 22+scalar, 33+scalar])
                if not driver:
                    self.assertEqual(tuple(binding._local.storage[0]), tuple(binding.fixed_words))
                del source, result
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            with self.assertRaises(IndexError):
                binding((), constant, ())
            if not driver:
                self.assertEqual(tuple(binding._local.storage[0]), tuple(binding.fixed_words))

    def test_full_and_tail_bindings_are_bounded_and_keep_original_launch_order(self):
        from types import SimpleNamespace
        for tile in (1, 4, 100):
            _, completed, action, _ = tiled_fixture(tile=tile)
            core = action.core
            bound, calls = [], []
            class Kernel:
                def __init__(self, name): self.name = name
                def __call__(self, *args): raise AssertionError('unexpected ordinary launch')
            def binder(backend, kernel, grid, block, template, positions, **options):
                bound.append((kernel.name, template, positions, options))
                def launch(inputs, result, scalars=()):
                    start_position = 2 if kernel.name == 'ed_precise_partial' else len(core.small_operands)+2
                    start = scalars[0] if options['scalar_positions'] else int(template[start_position])
                    calls.append((kernel.name, start, int(template[start_position+1])))
                return launch
            backend = SimpleNamespace(__name__='cupy', RawKernel=object,
                empty=lambda shape,dtype: np.empty(shape,dtype=dtype))
            values = tuple(np.zeros(shape,'complex64') for shape in core.shapes)
            with patch('lqcdweave.contraction.execution.factorized_reduction._kernel',
                       side_effect=lambda factory, source, name: Kernel(name)), \
                 patch('lqcdweave.contraction.execution.factorized_reduction.bind_pointer_arguments', binder):
                run = bind_factorized_reduction(action, completed.catalog, backend)
                run(values)
                run(values)
            self.assertLessEqual(len(bound), 4)
            expected = [(name,start,min(core.tile_width,core.epilogue_elements-start))
                        for start in range(0,core.epilogue_elements,core.tile_width)
                        for name in ('ed_precise_partial','ed_factorized_wide_finish')]
            self.assertEqual(calls, expected*2)
            self.assertEqual(run.workspace_bytes,core.workspace_bytes)

    def test_tiled_launch_failure_fences_current_stream_and_preserves_exception(self):
        from types import SimpleNamespace
        _,completed,action,_=tiled_fixture()
        error=RuntimeError('tiled finish failure')
        fenced=[]
        backend=SimpleNamespace(__name__='cupy',RawKernel=object,
            empty=lambda shape,dtype:np.empty(shape,dtype=dtype),
            cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(
                synchronize=lambda:fenced.append(True))))
        def binder(backend,kernel,*args,**kwargs):
            def launch(*args):
                if kernel=='ed_factorized_wide_finish':raise error
            return launch
        with patch('lqcdweave.contraction.execution.factorized_reduction._kernel',
                   side_effect=lambda factory,source,name:name), \
             patch('lqcdweave.contraction.execution.factorized_reduction.bind_pointer_arguments',binder):
            run=bind_factorized_reduction(action,completed.catalog,backend)
            with self.assertRaises(RuntimeError) as caught:
                run(tuple(np.zeros(shape,'complex64') for shape in action.core.shapes))
        self.assertIs(caught.exception,error)
        self.assertEqual(fenced,[True])
        error.__traceback__=None


class TiledLaunchArgumentsCuPyTests(unittest.TestCase):
    def test_three_four_operand_large_tail_current_inputs_nondefault_stream_and_release(self):
        try:
            import cupy as cp
            cp.arange(2).sum().get()
        except Exception as error:
            self.skipTest(f'CUDA unavailable: {error}')
        rng=np.random.default_rng(261050)
        for four in (False,True):
            pattern=(AxisPattern(((0,2),(1,2),(3,0),(4,1)),(3,4)) if four else
                     AxisPattern(((0,1,2),(0,1,2),(3,0,1)),(3,)))
            shapes=(((65,2),(65,2),(2,65),(3,65)) if four else
                    ((65,65,2),(65,65,2),(2,65,65)))
            core=FactorizedWideReduction(pattern,shapes,(0,1),tile_elements=4096)
            self.assertEqual(core.tiles,2)
            specs={str(i):TensorSpec(shape,'complex64') for i,shape in enumerate(shapes)}
            specs['out']=TensorSpec(core.output_shape,'complex64')
            action=Action('contract','out',tuple(str(i) for i in range(len(shapes))),pattern,0.,False,
                          core=core,wide_accumulation=True)
            run=bind_factorized_reduction(action,specs,cp)
            refs=[]
            stream=cp.cuda.Stream(non_blocking=True)
            for cfg in range(2):
                host=tuple((rng.normal(size=shape)+1j*rng.normal(size=shape)).astype('complex64')
                           for shape in shapes)
                with stream:
                    values=tuple(cp.asarray(value[...,::-1])[...,::-1] for value in host)
                    refs.extend(weakref.ref(value) for value in values)
                    actual,_=run(values)
                    result=actual.get()
                args=tuple(item for value,axes in zip(host,pattern.inputs,strict=True)
                           for item in (value.astype('complex128'),axes))+(pattern.output,)
                expected=np.einsum(*args,optimize=True)
                error=np.abs(result-expected)
                limit=5e-5*np.maximum(np.maximum(np.abs(result),np.abs(expected)),1e-12)+2e-8
                self.assertTrue(np.all(error<=limit),float(np.max(error/limit)))
                del values,actual
            stream.synchronize();gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(cp.get_default_memory_pool().used_bytes(),0)
