"""Packed ownership and bounded CPU/GPU delivery; fake protocols are not GPU timing."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import tempfile
import threading
import copy
import pickle
import unittest
from unittest.mock import Mock, patch
import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec, describe_coordinates
from lqcdweave.contraction.execution.operations.output import bind_output_layout, PackedOutputs, _OutputOwnerSnapshot
from lqcdweave.contraction.execution.device import device_execution
from lqcdweave.contraction.execution.output_queue import OutputConsumerQueue, output_queue_reserve
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.cfg_runtime import execute_cfg_plan
from lqcdweave.contraction.execution.cfg_windows import execute_cfg_windows
from lqcdweave.contraction.correlator_cfg import complete_calculation, source_loader_factory
from correlator.execution.test_device import _FakeCupy, _Tracker, _Cuda, _Stream, _complete, _targets, _factory


class _OutputArray(np.ndarray):
    def get(self, *, stream, out, blocking):
        if blocking:
            raise AssertionError('output queue requested blocking transfer')
        stream.copies.append((out, self))


class _OutputStream(_Stream):
    def __init__(self, tracker, name):
        super().__init__(tracker, name)
        self.copies = []

    def wait_event(self, event):
        self.tracker.dependencies.append((self.name, event.stream.name))

    def synchronize(self):
        super().synchronize()
        self.finish(len(self.copies))

    def finish(self, count):
        for out, value in self.copies[:count]:
            if out is not None:
                np.copyto(out, value)
        self.copies[:count] = [(None, None)] * count


class _OutputEvent:
    def __init__(self, tracker):
        self.stream = None

    def record(self, stream):
        self.stream = stream
        self.count = len(stream.copies)

    def synchronize(self):
        self.stream.finish(self.count)


class _OutputCuda(_Cuda):
    def __init__(self, tracker):
        super().__init__(tracker)
        self.compute_stream = _OutputStream(tracker, 'compute')
        self.count = 0

    def Stream(self, *, non_blocking):
        self.count += 1
        return _OutputStream(self.tracker, 'upload' if self.count == 1 else 'download')

    def Event(self):
        return _OutputEvent(self.tracker)


class _OutputCupy(_FakeCupy):
    ndarray = _OutputArray

    def __init__(self):
        tracker = _Tracker()
        tracker.dependencies = []
        super().__init__(tracker)
        self.cuda = _OutputCuda(tracker)

    def asarray(self, value, dtype=None, order=None, *, copy=None):
        return super().asarray(value, dtype=dtype, order=order, copy=copy).view(_OutputArray)

    def array(self, value, *args, **kwargs):
        return np.array(value, *args, **kwargs).view(_OutputArray)

    def empty(self, *args, **kwargs):
        return np.empty(*args, **kwargs).view(_OutputArray)

    def stack(self, values, axis=0, **options):
        return np.stack(values, axis=axis, **options).view(_OutputArray)

    def asnumpy(self, value):
        self.tracker.asnumpy_calls += 1
        return np.asarray(value).copy()


def _plan(target):
    selected, _ = _complete(target)
    coordinates, _ = describe_coordinates(time_pairs=[{0:1,1:0},{0:0,1:1},{0:1,1:1}])
    complete = complete_calculation(selected, coordinates=coordinates,
        readers=selected.expanded.inputs.loader.input_readers)
    return selected, complete


class OutputPackingTests(unittest.TestCase):
    def test_legacy_get_signature_uses_async_memory_pointer_and_retains_ownership(self):
        import ctypes
        class Pointer:
            def __init__(self, owner):
                self.owner = owner
            def copy_to_host_async(self, pointer, size, stream):
                destination = np.ctypeslib.as_array((ctypes.c_ubyte*size).from_address(pointer))
                source = self.owner.view(np.ndarray).view(np.uint8).reshape(-1)
                stream.copies.append((destination, source))
        class LegacyArray(_OutputArray):
            @property
            def data(self):
                return Pointer(self)
            def get(self, *, stream=None, out=None):
                raise AssertionError('legacy get must not receive modern keywords')
        class LegacyBackend(_OutputCupy):
            ndarray = LegacyArray
            def array(self, value, *args, **kwargs):
                return np.array(value, *args, **kwargs).view(LegacyArray)
        backend = LegacyBackend()
        expected = np.arange(12).reshape(3, 4)*(1+2j)
        with device_execution(backend, device_memory_limit=4096) as device:
            value = backend.array(expected)
            ticket = device.start_download((value,))
            self.assertIn(('download', 'compute'), backend.tracker.dependencies)
            actual, = device.finish_download(ticket)
            value[...] = 0
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(device.statistics['output_download_calls'], 1)
        np.testing.assert_array_equal(actual, expected)

    def test_downloads_keep_snapshot_owners_without_resolving_device_lanes(self):
        import gc
        import weakref
        backend = _OutputCupy()
        positions = ((0,3), (1,1), (0,0), (0,3))
        owners = (backend.array(np.arange(4)+1j), backend.array(np.arange(6).reshape(2,3)-2j))
        snapshot = _OutputOwnerSnapshot(owners, SimpleNamespace(positions=positions))
        owner_ref = weakref.ref(owners[0])
        with device_execution(backend,device_memory_limit=4096) as device:
            with patch.object(_OutputOwnerSnapshot,'__getitem__',side_effect=AssertionError(
                    'D2H must not construct per-Term device views')):
                synchronous = device.download(snapshot)
                ticket = device.start_download(snapshot)
                del owners, snapshot
                gc.collect()
                self.assertIsNotNone(owner_ref())
                asynchronous = device.finish_download(ticket)
            gc.collect()
            self.assertIsNone(owner_ref())
            self.assertIsNone(ticket.device_values)
            self.assertEqual(device.statistics['output_download_calls'],4)
            for values in (synchronous,asynchronous):
                self.assertIs(type(values),tuple)
                self.assertIsInstance(values[0],np.ndarray)
                self.assertEqual(values[0].shape,())
                np.testing.assert_array_equal(values[0],3+1j)
                np.testing.assert_array_equal(values[1],np.array([3,4,5])-2j)
                np.testing.assert_array_equal(values[2],1j)
                np.testing.assert_array_equal(values[3],3+1j)

    def test_output_cleanup_failure_preserves_primary_and_releases_pool(self):
        backend=_OutputCupy()
        with self.assertRaisesRegex(ValueError,'primary consumer error') as caught:
            with device_execution(backend,device_memory_limit=4096) as device:
                device.download_stream=SimpleNamespace(synchronize=Mock(
                    side_effect=RuntimeError('output cleanup failed')))
                raise ValueError('primary consumer error')
        self.assertTrue(backend.tracker.pools[0].freed)
        self.assertTrue(any('output cleanup failed' in note for note in caught.exception.__notes__))
        backend=_OutputCupy()
        with self.assertRaisesRegex(RuntimeError,'output cleanup failed'):
            with device_execution(backend,device_memory_limit=4096) as device:
                device.download_stream=SimpleNamespace(synchronize=Mock(
                    side_effect=RuntimeError('output cleanup failed')))
        self.assertTrue(backend.tracker.pools[0].freed)

    def test_device_loader_rebinding_does_not_reuse_changed_recipe_or_old_scope_constants(self):
        from lqcdweave.contracts.correlator.api import ArrayConstant,SourcePreparation,TensorPreparation
        from lqcdweave.contraction.execution.preparation import bind_prepared_loader
        backend=_OutputCupy();specs={'a':TensorSpec((2,),np.float64)}
        def recipes(scale):
            return {'a':SourcePreparation((),recipe=TensorPreparation('constant',(),
                (ArrayConstant.from_array(np.array([1.,2.])*scale),)))}
        first=recipes(1)
        prepared=bind_prepared_loader(SimpleNamespace(),first,specs,backend=backend)
        old=prepared.materialize('a',None,{})
        with device_execution(backend,device_memory_limit=4096) as device:
            loaded=device.bind_loader(prepared,first,specs)
            new=loaded.materialize('a',None,{})
            self.assertIsNot(new,old)
            rebound=device.bind_loader(loaded,recipes(3),specs)
            np.testing.assert_array_equal(rebound.materialize('a',None,{}),[3.,6.])
            loaded.host_owners['raw']=object()
            with self.assertRaisesRegex(ValueError,'live input owners'):
                device.bind_loader(loaded,recipes(4),specs)
            loaded.host_owners.clear()

    def test_noncontiguous_download_rejected_before_any_stream_or_copy(self):
        backend=_OutputCupy()
        with device_execution(backend,device_memory_limit=4096) as device:
            value=backend.array(np.arange(12).reshape(3,4)).T
            with self.assertRaisesRegex(ValueError,'C-contiguous'):
                device.start_download((value,))
            self.assertIsNone(device.download_stream)
            self.assertEqual(device._downloads,set())
            with execution_debug(True):
                spec=TensorSpec((4,),np.complex128)
                plan=SimpleNamespace(templates=(SimpleNamespace(plan=SimpleNamespace(
                    state=SimpleNamespace(results=('a',),catalog={'a':spec}))),),
                    prefetch_host_bytes=128,prefetch_device_bytes=128,peak_host_bytes=0,peak_device_bytes=0)
                self.assertEqual(output_queue_reserve(plan,device,None,None),0)

    def test_enqueue_failure_fences_partial_downloads_and_keeps_primary_error(self):
        backend=_OutputCupy()
        with device_execution(backend,device_memory_limit=4096) as device:
            outputs=(backend.array(np.arange(3)),backend.array(np.arange(3)))
            original=_OutputArray.get
            calls=0
            def broken(value,**options):
                nonlocal calls
                calls+=1
                if calls==2:
                    raise RuntimeError('download failed deliberately')
                return original(value,**options)
            with patch.object(_OutputArray,'get',broken):
                with self.assertRaisesRegex(RuntimeError,'download failed deliberately'):
                    device.start_download(outputs)
            self.assertEqual(device._downloads,set())
            self.assertIn('download',backend.tracker.stream_syncs)
    def test_packed_carrier_supports_copy_and_pickle_without_losing_owners(self):
        packed=PackedOutputs((np.arange(12).reshape(2,2,3),),((0,1),(0,0)))
        for actual in (copy.copy(packed),copy.deepcopy(packed),pickle.loads(pickle.dumps(packed))):
            self.assertEqual(actual.positions,packed.positions)
            for left,right in zip(actual,packed,strict=True):
                np.testing.assert_array_equal(left,right)
    def test_endian_scalar_and_empty_layouts_preserve_exact_bytes(self):
        for dtype in ('>c16','>f8','<c8','int64','bool'):
            for shape in ((),(0,3),(2,3)):
                with self.subTest(dtype=dtype,shape=shape):
                    value=np.ones(shape,dtype=dtype)
                    layout=bind_output_layout(('a','b'),{'a':TensorSpec(shape,value.dtype),
                        'b':TensorSpec(shape,value.dtype)})
                    outputs=layout.copy({'a':value,'b':value},np)
                    for actual in outputs:
                        self.assertEqual(actual.dtype,value.dtype)
                        self.assertEqual(actual.shape,value.shape)
                        self.assertEqual(actual.tobytes(),value.tobytes())

    def test_same_layout_snapshots_share_one_copy_preserve_dtype_shape_and_order(self):
        arrays = {'a': np.arange(12).reshape(3,4).T.astype(np.complex64),
                  'b': (np.arange(12).reshape(3,4).T + 2j).astype(np.complex64),
                  'c': np.array(3-4j, dtype=np.complex128),
                  'd': np.arange(4, dtype=np.float32)[::-1]}
        originals = {key: value.copy() for key,value in arrays.items()}
        order = ('b','c','a','d')
        layout = bind_output_layout(order, {key: TensorSpec(value.shape,value.dtype) for key,value in arrays.items()})
        backend = SimpleNamespace(empty=Mock(wraps=np.empty))
        outputs = layout.copy(arrays,backend)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(backend.empty.call_count, 3)
        self.assertEqual(len(outputs.owners),3)
        self.assertEqual(sum(owner.nbytes for owner in outputs.owners),layout.nbytes)
        for key,value in arrays.items():
            value[...] = 0
        for key,output in zip(order,outputs,strict=True):
            self.assertEqual(output.dtype, originals[key].dtype)
            self.assertEqual(output.shape, originals[key].shape)
            np.testing.assert_array_equal(output, originals[key])
        outputs[0][...] = 17
        np.testing.assert_array_equal(outputs[2], originals['a'])

    def test_saved_layout_mismatch_precedes_copy(self):
        layout = bind_output_layout(('a',), {'a':TensorSpec((2,),np.complex64)})
        backend = SimpleNamespace(empty=Mock(wraps=np.empty))
        with self.assertRaisesRegex(ValueError,'saved output layout'):
            layout.copy({'a':np.ones(2,dtype=np.complex128)},backend)
        backend.empty.assert_not_called()

    def test_async_copy_waits_for_dependency_and_old_views_survive_next_download(self):
        backend = _OutputCupy()
        original = np.array([2+3j,4-5j],dtype=np.complex64)
        with device_execution(backend,device_memory_limit=4096) as device:
            outputs = PackedOutputs((backend.stack((original,original*2)),),((0,0),(0,1)))
            ticket = device.start_download(outputs)
            self.assertEqual(len(device._downloads),1)
            result = device.finish_download(ticket)
            self.assertIs(type(result),tuple)
            self.assertEqual(len(device._downloads),0)
            newer = device.start_download(PackedOutputs((backend.stack((original*7,)),),((0,0),)))
            device.finish_download(newer)
            np.testing.assert_array_equal(result[0],original)
            np.testing.assert_array_equal(result[1],original*2)
            self.assertEqual(device.statistics['output_download_calls'],2)
            self.assertEqual(backend.tracker.dependencies,[('download','compute'),('download','compute')])
        np.testing.assert_array_equal(result[0],original)

    def test_two_slot_reservation_requires_both_domains_and_caps(self):
        spec = TensorSpec((4,),np.complex128)
        plan = SimpleNamespace(templates=(SimpleNamespace(plan=SimpleNamespace(
            state=SimpleNamespace(results=('a',),catalog={'a':spec}))),),
            prefetch_host_bytes=1024,prefetch_device_bytes=1024,peak_host_bytes=500,peak_device_bytes=600)
        device=SimpleNamespace(supports_output_overlap=True)
        self.assertEqual(output_queue_reserve(plan,device,1524,1624),1024)
        self.assertEqual(output_queue_reserve(plan,device,1523,1624),0)
        self.assertEqual(output_queue_reserve(plan,device,1524,1623),0)
        plan.prefetch_device_bytes=1023
        self.assertEqual(output_queue_reserve(plan,device,None,None),0)


class OutputQueueIntegrationTests(unittest.TestCase):
    def test_ordinary_cfg_requests_owner_snapshot_and_returns_host_tuples(self):
        from lqcdweave.contraction.execution import cfg_runtime
        received=[]
        with patch.object(cfg_runtime,'replay',wraps=cfg_runtime.replay) as replay:
            with patch.object(_OutputOwnerSnapshot,'__getitem__',side_effect=AssertionError(
                    'ordinary cfg output should download owners directly')):
                self.run_gpu(lambda cfg,index,values: received.append(values) if index is not None else None)
        self.assertTrue(all(call.kwargs['_owner_snapshot'] for call in replay.call_args_list))
        self.assertEqual(len(received),3)
        self.assertTrue(all(type(value) is tuple for value in received))

    def test_window_boundaries_drain_before_reader_close_and_complete_once(self):
        from lqcdweave.contraction.expansion.api import describe_input_preparations, describe_input_reads
        from lqcdweave.contraction.planning.api import iter_cfg_plan_windows
        selected,complete=_plan(_targets()[1]); backend=_OutputCupy()
        coords,_=describe_coordinates(time_pairs=[{0:1,1:0},{0:0,1:1},{0:1,1:1}])
        preparations,specs=describe_input_preparations(selected.expanded.inputs)
        windows=iter_cfg_plan_windows(selected.plan,complete.plan.templates[0].output,coords.points,
            describe_input_reads(selected.expanded.inputs),preparations,specs,window_size=1,
            execution_target=complete.plan.execution_target,target_summary=complete.plan.target_summary)
        closed=[];values=[]
        calculation=replace(complete,options=dict(complete.options,
            source_factory=_factory(complete,selected,backend.tracker,closed),readers=None))
        base=source_loader_factory(calculation)
        @contextmanager
        def factory(window,cfg):
            with base(window,cfg) as loader:
                yield loader
                self.assertEqual(len(values),window.statistics['coordinate_offset']+1)
        report=execute_cfg_windows(windows,'a',factory,
            lambda cfg,index,value:values.append(value) if index is not None else None,
            backend=backend)
        self.assertEqual(report['completed'],('a',))
        self.assertEqual(report['planned_windows'],3)
        self.assertEqual(report['output_queue_tasks'],3)
        self.assertEqual(closed,['a','a','a'])

    def test_dataset_consumer_queue_preserves_binding_and_publishes_after_results(self):
        from lqcdweave.contraction.correlator_output import load_correlator_output
        from lqcdweave.storage.dataset import read_dataset_metadata
        from correlator.execution.test_device import _assert_tree_allclose
        selected,complete=_plan(_targets()[1]); backend=_OutputCupy()
        coords,_=describe_coordinates(time_pairs=[{0:1,1:0},{0:0,1:1},{0:1,1:1}])
        closed=[]
        with tempfile.TemporaryDirectory() as directory:
            dataset=complete_calculation(selected,coordinates=coords,output_mode='dataset',output=directory)
            dataset=replace(dataset,options=dict(dataset.options,configuration='a',readers=None,
                source_factory=_factory(dataset,selected,backend.tracker,closed)))
            with (patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',return_value=1<<40),
                  patch('lqcdweave.adapters.hardware_budget._observed_device_memory',return_value=(1<<30,1<<30))):
                actual=dataset.run(backend=backend)
            loaded=load_correlator_output(Path(directory),configuration='a')
            cpu_selected,cpu_complete=_plan(_targets()[0]); cpu_closed=[]
            cpu_complete=replace(cpu_complete,options=dict(cpu_complete.options,configuration='a',readers=None,
                source_factory=_factory(cpu_complete,cpu_selected,_Tracker(),cpu_closed)))
            _assert_tree_allclose(loaded,cpu_complete.run().value)
            self.assertEqual(actual.report['output_queue_tasks'],3)
            self.assertEqual(closed,['a'])
            self.assertEqual(read_dataset_metadata(directory,cfg='a')['spec'].document['kind'],'correlator')
            self.assertTrue((Path(directory)/'a.h5').exists())
    def test_worker_failure_latch_prevents_already_queued_successor_callback(self):
        entered=threading.Event(); release=threading.Event(); calls=[]
        def consume(cfg,index,values):
            calls.append(index)
            if index==0:
                entered.set()
                if not release.wait(5):
                    raise AssertionError('test never released failing consumer')
                raise RuntimeError('first callback failed')
        with self.assertRaisesRegex(RuntimeError,'first callback failed'):
            with OutputConsumerQueue(consume) as queue:
                queue.pool=__import__('concurrent.futures').futures.ThreadPoolExecutor(max_workers=1)
                queue.reserve=1
                queue.submit('a',0,())
                self.assertTrue(entered.wait(5))
                queue.submit('a',1,())
                release.set()
                queue.flush()
        self.assertEqual(calls,[0])
    def run_gpu(self, consume, *, limited=False):
        selected, complete = _plan(_targets()[1])
        backend = _OutputCupy()
        closed=[]
        physical_factory = _factory(complete,selected,backend.tracker,closed)
        calculation = replace(complete,options=dict(complete.options,source_factory=physical_factory,readers=None))
        options = dict(backend=backend)
        if limited:
            options.update(host_memory_limit=complete.plan.peak_host_bytes,
                           device_memory_limit=complete.plan.peak_device_bytes)
        report = execute_cfg_plan(complete.plan,('a',),source_loader_factory(calculation),consume,**options)
        self.assertEqual(closed,['a'])
        return report, backend

    def test_consumer_overlaps_next_gpu_task_with_bounded_ordered_results(self):
        started=threading.Event(); next_task=threading.Event(); received=[]
        from lqcdweave.contraction.execution import cfg_runtime
        original=cfg_runtime.replay
        calls=0
        def replay(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:
                self.assertTrue(started.wait(5),'CPU queue never started')
                next_task.set()
            return original(*args,**kwargs)
        def consume(cfg,index,values):
            received.append((cfg,index,values,threading.current_thread().name))
            if index==0:
                started.set()
                self.assertTrue(next_task.wait(5),'producer did not advance during CPU accumulation')
        with patch.object(cfg_runtime,'replay',replay):
            report,backend=self.run_gpu(consume)
        self.assertEqual([row[1] for row in received],[0,1,2,None])
        self.assertTrue(all(row[3].startswith('contraction-results') for row in received[:-1]))
        self.assertEqual(report['output_queue_tasks'],3)
        self.assertLessEqual(report['output_queue_peak_tasks'],2)
        self.assertLessEqual(report['device']['peak_pending_outputs'],2)
        self.assertGreater(report['output_buffer_host_bytes'],0)
        self.assertEqual(backend.tracker.asnumpy_calls,0)
        # Compare with the unchanged synchronous NumPy recipe/kernel path.
        selected,complete=_plan(_targets()[0]); expected=[]
        closed=[]; tracker=_Tracker()
        calculation=replace(complete,options=dict(complete.options,
            source_factory=_factory(complete,selected,tracker,closed),readers=None))
        execute_cfg_plan(complete.plan,('a',),source_loader_factory(calculation),
            lambda cfg,index,values: expected.append(values) if index is not None else None)
        for row,reference in zip(received[:-1],expected,strict=True):
            for value,truth in zip(row[2],reference,strict=True):
                np.testing.assert_allclose(value,truth,rtol=3e-12,atol=3e-12)

    def test_tight_envelope_uses_synchronous_delivery_without_extra_buffers(self):
        threads=[]
        report,_=self.run_gpu(lambda cfg,index,values:threads.append(threading.current_thread().name),limited=True)
        self.assertEqual(report['output_buffer_host_bytes'],0)
        self.assertEqual(report['output_queue_tasks'],0)
        self.assertEqual(set(threads),{'MainThread'})

    def test_consumer_error_joins_before_device_and_reader_cleanup(self):
        events=[]
        def consume(cfg,index,values):
            events.append(index)
            if index==0:
                raise RuntimeError('consumer failed deliberately')
        with self.assertRaisesRegex(RuntimeError,'consumer failed deliberately'):
            self.run_gpu(consume)
        self.assertNotIn(None,events)
        self.assertFalse(any(t.name.startswith('contraction-results') for t in threading.enumerate()))

    def test_backpressure_prevents_a_third_unconsumed_task(self):
        entered=threading.Event(); release=threading.Event(); submissions=[]
        with OutputConsumerQueue(lambda cfg,index,values: (entered.set(),release.wait(5))) as queue:
            queue.pool=__import__('concurrent.futures').futures.ThreadPoolExecutor(max_workers=1)
            queue.reserve=1
            # A simple test dependency avoids CUDA; queue ownership is tested separately above.
            queue.device=None
            queue.submit('a',0,()); queue.submit('a',1,())
            self.assertTrue(entered.wait(5))
            def third():
                queue.acquire(); submissions.append(3)
            waiter=threading.Thread(target=third); waiter.start()
            self.assertEqual(submissions,[])
            release.set(); waiter.join(5)
            self.assertEqual(submissions,[3])
        self.assertEqual(queue.peak_tasks,2)


class RealCuPyOutputQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount()<1:
                raise RuntimeError('no device')
        except Exception as error:
            raise unittest.SkipTest(f'real CuPy output queue requires a working device: {error}')
        cls.backend=cupy

    def test_real_strided_inputs_packed_downloads_and_retained_outputs(self):
        cp=self.backend
        expected=np.arange(24,dtype=np.float64).reshape(4,6).T*(1+2j)
        with device_execution(cp,device_memory_limit=16<<20) as device:
            arrays={'a':cp.asarray(expected),'b':cp.asarray(expected*2),
                    'c':cp.asarray(np.array(3+4j,dtype=np.complex64))}
            layout=bind_output_layout(tuple(arrays),{key:TensorSpec(value.shape,value.dtype)
                for key,value in arrays.items()})
            packed=layout.copy(arrays,cp)
            self.assertTrue(all(owner.flags.c_contiguous for owner in packed.owners))
            ticket=device.start_download(packed)
            result=device.finish_download(ticket)
            for value in arrays.values():
                value[...] = 0
            second=device.start_download(layout.copy(arrays,cp))
            device.finish_download(second)
            np.testing.assert_array_equal(result[0],expected)
            np.testing.assert_array_equal(result[1],2*expected)
            self.assertEqual(result[2].dtype,np.dtype('complex64'))
            self.assertEqual(device.statistics['output_download_calls'],4)
        np.testing.assert_array_equal(result[0],expected)


if __name__=='__main__':
    unittest.main()
