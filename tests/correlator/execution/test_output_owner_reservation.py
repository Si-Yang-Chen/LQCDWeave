"""CPU owner geometry and bounded-queue admission; no CUDA performance claim."""
from collections import Counter
from dataclasses import replace
from unittest.mock import patch
from math import prod
from types import SimpleNamespace
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec, OutputNode, device_allocation_bytes
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.output_queue import (
    OutputConsumerQueue, OutputDownloadLayout, _snapshot_download_layout, output_queue_reserve,
)
from lqcdweave.contraction.execution.output_postprocessing import describe_device_download_layout
from lqcdweave.contraction.execution import output_program
from lqcdweave.sampling.temporal.api import (
    DeviceTimePairAccumulator, describe_device_accumulator_allocations, estimate_device_accumulator_bytes,
)
from correlator.execution.test_device_output_program import _fixture, _term, _RecordingBackend
from sampling.temporal.test_device_reconstruction import _field, _DeviceBackend, _DeviceArray


def _plan(state):
    return SimpleNamespace(templates=(SimpleNamespace(plan=SimpleNamespace(state=state)),),
        prefetch_host_bytes=4096, prefetch_device_bytes=4096,
        peak_host_bytes=100, peak_device_bytes=200)


class OutputOwnerReservationTests(unittest.TestCase):
    def test_packed_distinct_keys_layout_and_individual_measurement_geometry(self):
        state = SimpleNamespace(results=('a', 'a', 'b', 'c', 'd'), catalog={
            'a': TensorSpec((), 'float32'), 'b': TensorSpec((), 'float32'),
            'c': TensorSpec((), 'complex64'), 'd': TensorSpec((2,), 'float32')})
        native = _snapshot_download_layout(state)
        self.assertEqual(native.owner_bytes, (8, 8, 8))
        self.assertEqual(native.device_bytes, 1536)
        measured = _snapshot_download_layout(state, packed=False)
        self.assertEqual(measured.owner_bytes, (4, 4, 4, 8, 8))
        self.assertEqual(measured.device_bytes, 2560)
        device = SimpleNamespace(supports_output_overlap=True, backend=np)
        self.assertEqual(output_queue_reserve(_plan(state), device, None, None), 3072)
        with execution_debug(True):
            self.assertEqual(output_queue_reserve(_plan(state), device, None, None), 0)
        self.assertEqual(output_queue_reserve(_plan(state), None, None, None), 0)

    def test_two_slots_use_current_device_limit_and_both_headrooms(self):
        state = SimpleNamespace(results=('a',),catalog={'a':TensorSpec((), 'float32')})
        plan = _plan(state)
        device = SimpleNamespace(supports_output_overlap=True, backend=np, limit=1224)
        self.assertEqual(output_queue_reserve(plan, device, 1124, 1224), 1024)
        self.assertEqual(output_queue_reserve(plan, device, 1123, 1224), 0)
        self.assertEqual(output_queue_reserve(plan, device, 1124, 1223), 0)
        device.limit = 1223
        self.assertEqual(output_queue_reserve(plan, device, None, 999999), 0)
        device.limit = 1224
        plan.prefetch_device_bytes = 1023
        self.assertEqual(output_queue_reserve(plan, device, None, None), 0)

    def test_known_transform_download_is_packed_after_mixed_field_assembly(self):
        layout = SimpleNamespace(output_layout='pairs', output_storage_shape=(3,))
        descriptor = describe_device_download_layout(layout)
        self.assertEqual(descriptor.owner_bytes, (24,))
        transform = SimpleNamespace(output_bytes=24, _download_layout=descriptor)
        plan = _plan(SimpleNamespace(results=(), catalog={}))
        device = SimpleNamespace(supports_output_overlap=True)
        with OutputConsumerQueue(lambda *row:None, device_transform=transform) as queue:
            self.assertEqual(output_queue_reserve(plan, device, None, None), 1024)
            self.assertEqual(queue.postprocess_queue_mode, 'overlapped_downloads')
        layout.output_layout = 'dt_packed'
        descriptor = describe_device_download_layout(layout)
        self.assertEqual(descriptor.owner_bytes, ())
        with OutputConsumerQueue(lambda *row:None, device_transform=SimpleNamespace(
                output_bytes=0, _download_layout=descriptor)):
            self.assertEqual(output_queue_reserve(plan, device, None, None), 0)

    def test_opaque_transform_never_guesses_owner_count(self):
        plan = _plan(SimpleNamespace(results=(), catalog={}))
        device = SimpleNamespace(supports_output_overlap=True)
        for transform in (SimpleNamespace(), SimpleNamespace(output_bytes=8),
                          SimpleNamespace(output_bytes=8, _download_layout=(8,))):
            with self.subTest(transform=transform):
                with OutputConsumerQueue(lambda *row:None, device_transform=transform):
                    self.assertEqual(output_queue_reserve(plan, device, None, None), 0)
        with self.assertRaises(ValueError):
            OutputDownloadLayout((True,))

    def test_opaque_device_transform_preserves_public_results_synchronously(self):
        from correlator.execution.test_output_queue import _plan as make_plan, _targets, _OutputCupy
        from correlator.execution.test_device import _factory
        from lqcdweave.contraction.correlator_cfg import source_loader_factory
        from lqcdweave.contraction.execution.cfg_runtime import execute_cfg_plan
        selected, complete = make_plan(_targets()[1])
        backend = _OutputCupy()
        closed, received, transformed = [], [], []
        calculation = replace(complete, options=dict(complete.options, readers=None,
            source_factory=_factory(complete,selected,backend.tracker,closed)))
        def transform(cfg,index,values,template):
            transformed.append(index)
            return tuple(value*2 for value in values)
        report = execute_cfg_plan(complete.plan,('a',),source_loader_factory(calculation),
            lambda cfg,index,values: received.append(values) if index is not None else None,
            backend=backend,device_transform=transform)
        self.assertEqual(transformed,list(range(len(complete.plan.instances))))
        self.assertEqual(len(received),len(transformed))
        self.assertEqual(report['output_queue_tasks'],0)
        self.assertEqual(closed,['a'])
        baseline=[]
        execute_cfg_plan(complete.plan,('a',),source_loader_factory(calculation),
            lambda cfg,index,values: baseline.append(values) if index is not None else None,
            backend=backend)
        for original, current in zip(baseline,received,strict=True):
            for left,right in zip(original,current,strict=True):
                np.testing.assert_array_equal(left*2,right)

    def test_output_program_owner_extents_match_actual_recorded_requests(self):
        root = OutputNode('dict', keys=('a','b'), children=(_term(0),_term(1,2.5)))
        recipe, problem = _fixture(root,np.array(1,np.float32),np.array(2,np.complex64))
        metadata = output_program.describe_device_output_reservation(recipe, problem)
        backend = _RecordingBackend()
        program = output_program.bind_device_output_assembly(recipe,problem,backend)
        program((np.array(1,np.float32),np.array(2,np.complex64)))
        actual = [array.nbytes for array in backend.uploads]
        actual += [prod(shape)*dtype.itemsize for shape,dtype in backend.allocations]
        self.assertEqual(Counter(metadata['device_owner_bytes']),Counter(actual))
        self.assertEqual(sum(actual),metadata['device_bytes'])
        self.assertGreater(sum(map(device_allocation_bytes,actual)),device_allocation_bytes(sum(actual)))

    def test_native_queue_cache_ignores_external_statistics_and_keeps_dynamic_cap(self):
        from correlator.execution.test_output_queue import _plan as make_plan, _targets
        from lqcdweave.contraction.execution import output_queue
        _, complete = make_plan(_targets()[1])
        plan = complete.plan
        device = SimpleNamespace(supports_output_overlap=True, backend=np)
        with OutputConsumerQueue(lambda *row:None) as queue, patch.object(
                output_queue, '_snapshot_download_layout', wraps=_snapshot_download_layout) as describe:
            reserve = output_queue_reserve(plan,device,None,None)
            calls = describe.call_count
            self.assertGreater(reserve,0)
            self.assertEqual(output_queue_reserve(plan,device,None,None),reserve)
            self.assertEqual(describe.call_count,calls)
            self.assertEqual(output_queue_reserve(plan,device,None,plan.peak_device_bytes),0)
            self.assertEqual(describe.call_count,calls)
            forged = replace(plan, statistics=dict(plan.statistics,
                native_output_snapshot_device_bytes=0, output_snapshot_device_bytes=0))
            self.assertEqual(output_queue_reserve(forged,device,None,None),reserve)
            self.assertGreater(describe.call_count,calls)
        state = SimpleNamespace(results=('a',),catalog={'a':TensorSpec((), 'float32')})
        extension = _plan(state)
        with OutputConsumerQueue(lambda *row:None):
            self.assertEqual(output_queue_reserve(extension,device,None,None),1024)
            state.catalog['a']=TensorSpec((200,), 'float32')
            self.assertEqual(output_queue_reserve(extension,device,None,None),2048)

    def test_workflow_rounds_assembly_temporal_and_final_packing_owners(self):
        from lqcdweave.contraction.correlator_postprocessing import describe_gpu_postprocessing_reservation
        root = OutputNode('dict',keys=('a','b'),children=(_term(0),_term(1)))
        recipe, problem = _fixture(root,np.array(1,np.float32),np.array(2,np.complex64))
        fields = tuple(_field(path,shape,dtype) for path,shape,dtype in
            output_program.describe_device_output_reservation(recipe,problem)['fields'])
        layout = SimpleNamespace(output_fields=fields, output_layout='pairs_and_dt',
            physical_lt=3, one_point_slots=0, dt_coordinates=(1,),
            output_storage_shape=(2,),dt_storage_shape=(2,))
        selected = SimpleNamespace(state=SimpleNamespace(problem=problem,
            catalog=problem.specs,results=('t0','t1')))
        result=describe_gpu_postprocessing_reservation(selected,recipe,layout)
        # 2 copies of (2 term owners + 2 field owners), 4 accumulator
        # owners, 2 packed record owners and one largest strided field copy.
        self.assertEqual(result['postprocess_device_bytes'],15*512)
        self.assertEqual(result['postprocess_host_bytes'],32+3*8)

    def test_legacy_direct_and_window_envelopes_fail_before_device_or_reader(self):
        from correlator.execution.test_output_queue import _plan as make_plan, _targets
        from lqcdweave.contraction.execution import cfg_runtime, cfg_windows
        _, complete = make_plan(_targets()[1])
        stats = dict(complete.plan.statistics)
        stats.pop('cfg_allocation_rounding_version')
        legacy = replace(complete.plan,statistics=stats,peak_device_bytes=1)
        for name, entry, args in (
            ('cfg_runtime',cfg_runtime.iter_execute_cfg_plan,(legacy,('cfg',),None)),
            ('cfg_windows',cfg_windows.iter_execute_cfg_windows,((legacy,),'cfg',None))):
            with self.subTest(entry=name), patch(
                    'lqcdweave.contraction.execution.'+name+'.device_execution',
                    side_effect=AssertionError('device opened before legacy resource gate')):
                events=entry(*args,backend=SimpleNamespace(__name__='cupy'))
                with self.assertRaisesRegex(MemoryError,'existing selected'):
                    next(events)
                events.close()

    def test_empty_snapshot_has_no_allocation_charge(self):
        state=SimpleNamespace(results=(),catalog={})
        self.assertEqual(_snapshot_download_layout(state).owner_bytes,())
        self.assertEqual(output_queue_reserve(_plan(state),SimpleNamespace(
            supports_output_overlap=True),None,None),0)


class TemporalOwnerReservationTests(unittest.TestCase):
    def test_owner_description_matches_numpy_protocol_and_logical_estimate(self):
        fields=(_field(('mean',),(), 'float32'),
                _field(('loop',),(2,), 'complex64',dt=False,role='t'))
        for Lt, slots, dts in ((3,2,(1,)), (129,3,(2,1,2)), (4,1,())):
            with self.subTest(Lt=Lt, slots=slots, dts=dts):
                backend=_DeviceBackend()
                accumulator=DeviceTimePairAccumulator(fields,Lt=Lt,backend=backend,one_point_slots=slots)
                for slot in range(slots):
                    accumulator.add(0,1,{('mean',):_DeviceArray(np.array(1,np.float32)),
                        ('loop',):_DeviceArray(np.ones(2,np.complex64))},slots={'t':slot})
                accumulator.finalize(dts)
                extents=describe_device_accumulator_allocations(fields,Lt,slots,dts)
                actual=tuple(prod(shape)*dtype.itemsize for shape,dtype in backend.allocations)
                self.assertEqual(Counter(extents),Counter(actual))
                self.assertEqual(sum(extents),estimate_device_accumulator_bytes(fields,Lt,slots,dts))
                accumulator.close()
        self.assertEqual(describe_device_accumulator_allocations((),3,0,()),())
        self.assertEqual(estimate_device_accumulator_bytes((),3,0,()),0)
