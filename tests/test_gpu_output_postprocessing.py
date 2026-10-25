"""Dataset postprocessing handoffs, CPU ABI checks, and optional real CuPy.

NumPy-backed protocol tests check ownership, downloads, budgets and publication;
their independent field oracles are computed from synthetic inputs. Only
RealCuPyDatasetPostprocessingTests execute GPU numerical kernels.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import CfgOutputField, OutputNode, describe_coordinates
from lqcdweave.contraction.execution import output_postprocessing as output
from lqcdweave.contraction.results.api import bind_output_assembly
from lqcdweave.sampling.temporal.api import DeviceTimePairAccumulator
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, read_dataset_metadata
from lqcdweave.storage.correlator_results import (
    CfgOutputWriter, PackedDtOutputWriter, load_correlator_output,
)
from lqcdweave.storage.contraction_plans.api import load_cfg_plan
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import CfgSources, complete_calculation
from lqcdweave.contraction.correlator_postprocessing import GpuDatasetPostprocessing
from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation
from correlator.execution.test_device import _targets, _factory
from correlator.execution.test_output_queue import _OutputCupy, _OutputArray
from test_execution_target_frontend import inputs


class _ProtocolBackend(_OutputCupy):
    """CPU-only protocol double, including packed allocation and strided copy."""
    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs).view(_OutputArray)

    def ascontiguousarray(self, value, *args, **kwargs):
        return np.ascontiguousarray(value, *args, **kwargs).view(_OutputArray)


def _device_tree(value, backend):
    if isinstance(value, dict):
        return {key: _device_tree(item, backend) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_tree(item, backend) for item in value)
    if isinstance(value, list):
        return [_device_tree(item, backend) for item in value]
    return backend.array(value)


def _protocol_assembly(recipe, problem, backend, *, debug=False):
    """Substitute CPU R solely to test integration around X's GPU assembler."""
    assemble = bind_output_assembly(recipe, debug=debug)
    return lambda values: _device_tree(assemble(values), backend)


def _synthetic_case(layout):
    fields = (
        CfgOutputField(('two_point',), (2, 2), '<c16', 0, 8, 0),
        CfgOutputField(('connected',), (3,), '<f4', 8, 2, 8),
        CfgOutputField(('vacuum_sink',), (), '<i8', 10, 1, -1, '0'),
        CfgOutputField(('vacuum_source',), (2,), '<f4', 11, 1, -1, '1'),
    )
    points = ({'0': 1, '1': 0}, {'0': 2, '1': 1}, {'0': 0, '1': 2})
    plan = SimpleNamespace(
        coordinates=points, instances=tuple(SimpleNamespace(coordinate=tuple(p.items())) for p in points),
        output_fields=fields, output_words=12, dt_words=10,
        output_storage_shape=(1, 12), dt_storage_shape=(1, 10),
        dt_packed_storage_shape=(1, 30), one_point_storage_shape=(1, 6),
        physical_lt=11, physical_times=(2, 5, 8), time_roles=('0', '1'),
        coordinate_shape=(3,), output_layout=layout, dt_coordinates=(3, 5, 1),
        one_point_slots=3 if layout == 'dt_packed' else 0,
        output_definition={'fixture': 'frozen-gpu-postprocessing-abi-v1'},
    )
    values = []
    for number, point in enumerate(points):
        values.append({
            'two_point': np.array([[number + 1 + 2j, 3 - number * 1j],
                                   [-2.5 + number, 7 + number * 0.25j]], dtype='complex128'),
            'connected': np.array([number + 0.25, -number - 0.5, 3.5], dtype='float32'),
            'vacuum_sink': np.array(2**60 + point['0'] + 3, dtype='int64'),
            'vacuum_source': np.array([point['1'] + 0.25, point['1'] + 0.5], dtype='float32'),
        })
    binding = CfgSources({}, plan.physical_times, plan.physical_lt,
        provenance={'fixture': 'independent pre-change writer'},
        lattice=(2, 2, 2, 11), ensemble='synthetic-abi')
    return plan, binding, values


def _snapshot(directory, *, configuration):
    with DatasetReader(directory, cfg=configuration) as reader:
        fields = {}
        for key in sorted(reader.fields):
            path = json.loads(key)
            value = np.asarray(reader.read_field(path))
            fields[key] = dict(shape=list(value.shape), dtype=value.dtype.str,
                               bytes=value.tobytes().hex())
        return fields, reader.spec.document


def _expected_native_fields(plan, binding, values, layout):
    def snapshot(value):
        value = np.asarray(value)
        return dict(shape=list(value.shape), dtype=value.dtype.str, bytes=value.tobytes().hex())

    result = {}
    def save(path, value):
        result[json.dumps(list(path), sort_keys=True, separators=(',', ':'))] = snapshot(value)

    counts = np.zeros(plan.physical_lt, dtype=np.int64)
    sums = {field.path: np.zeros((plan.physical_lt, *field.shape), dtype=np.dtype(field.dtype))
            for field in plan.output_fields if field.dt_offset >= 0}
    sink_role, source_role = plan.time_roles
    for index, point in enumerate(plan.coordinates):
        dt = (binding.physical_times[point[sink_role]] -
              binding.physical_times[point[source_role]]) % plan.physical_lt
        counts[dt] += 1
        for field in plan.output_fields:
            value = values[index][field.path[0]]
            if layout == 'pairs_and_dt':
                save(('pairs', *field.path), np.stack(
                    [item[field.path[0]] for item in values], axis=-1))
            if field.dt_offset >= 0:
                sums[field.path][dt] += value
    for field in plan.output_fields:
        if layout == 'pairs_and_dt' and field.dt_offset >= 0:
            coordinates = range(plan.physical_lt)
        elif layout == 'dt_packed' and field.dt_offset >= 0:
            coordinates = plan.dt_coordinates
        elif layout == 'dt_packed':
            loop = np.empty((*field.shape, plan.one_point_slots), dtype=np.dtype(field.dtype))
            seen = np.zeros(plan.one_point_slots, dtype=bool)
            for point, value in zip(plan.coordinates, values, strict=True):
                slot = point[field.time_role]
                if not seen[slot]:
                    loop[..., slot] = value[field.path[0]]
                    seen[slot] = True
            if not seen.all():
                raise AssertionError('independent one-point reference has incomplete time coverage')
            save(('one_point', *field.path), loop)
            continue
        else:
            continue
        mean = np.full((*field.shape, len(coordinates)), np.nan, dtype=np.dtype(field.dtype))
        for index, dt in enumerate(coordinates):
            if counts[dt]:
                mean[..., index] = sums[field.path][dt] / counts[dt]
        save(('dt', *field.path), mean)
    return result


def _assert_cfg_unpublished(test_case, directory, configuration):
    directory = Path(directory)
    test_case.assertFalse((directory / f'{configuration}.h5').exists())
    with test_case.assertRaises((FileNotFoundError, KeyError)):
        read_dataset_metadata(directory, cfg=configuration)
    with test_case.assertRaises((FileNotFoundError, KeyError)):
        DatasetReader(directory, cfg=configuration)


def _processor_case(layout='dt_packed'):
    plan, binding, values = _synthetic_case(layout)
    backend = _ProtocolBackend()
    accumulator = DeviceTimePairAccumulator(
        tuple(f for f in plan.output_fields if layout == 'dt_packed' or f.dt_offset >= 0),
        Lt=plan.physical_lt, backend=backend, one_point_slots=plan.one_point_slots)
    downloads = []
    def download(items):
        downloads.append(tuple(value.shape for value in items))
        return tuple(np.asarray(value).copy() for value in items)
    device = SimpleNamespace(backend=backend, download=download, statistics={})
    processor = output.DeviceOutputProcessor(plan, binding.physical_times, device, accumulator)
    template = SimpleNamespace(output=object(), plan=SimpleNamespace(state=SimpleNamespace(problem=object())))
    return plan, binding, values, backend, processor, template, downloads


class PrepackedWriterReferenceTests(unittest.TestCase):
    def test_cpu_and_prepacked_writers_match_independent_native_field_bytes(self):
        for layout in ('pairs_and_dt', 'dt_packed'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                plan, binding, values, backend, processor, template, downloads = _processor_case(layout)
                writer_class = PackedDtOutputWriter if layout == 'dt_packed' else CfgOutputWriter
                cpu_root = Path(directory) / 'cpu'
                prepared_root = Path(directory) / 'prepared'
                cpu = writer_class(plan, 'reference', cpu_root, binding, debug=True)
                prepared = writer_class(plan, 'reference', prepared_root, binding,
                                        debug=True, prepacked=True)
                publication = []
                original_publish = DatasetWriter.publish
                def publish(writer):
                    publication.append(writer.directory.name)
                    return original_publish(writer)
                try:
                    with patch.object(output, 'bind_device_output_assembly',
                                      return_value=lambda items: items[0]):
                        for index, value in enumerate(values):
                            cpu.consume(index, value)
                            packed = processor.process(index, (_device_tree(value, backend),), template)
                            self.assertEqual(downloads, [], 'process downloaded an individual term or pair')
                            if layout == 'pairs_and_dt':
                                self.assertEqual(len(packed), 1)
                                prepared.consume_packed(index, np.asarray(packed[0]).copy())
                            else:
                                self.assertEqual(packed, ())
                            _assert_cfg_unpublished(self, prepared_root, 'reference')
                    cpu.publish()
                    with patch.object(DatasetWriter, 'publish', publish):
                        if layout == 'dt_packed':
                            prepared.publish_prepacked(*processor.packed_records())
                            self.assertEqual(downloads, [((1, 30), (1, 6))])
                            self.assertEqual(publication, ['prepared'])
                        else:
                            prepared.publish_prepacked(processor.counts, processor.pair_records())
                            self.assertEqual(downloads, [((1, 10),), ((1, 10),)])
                            self.assertEqual(publication, ['prepared'])
                    expected = _expected_native_fields(plan, binding, values, layout)
                    cpu_fields, cpu_spec = _snapshot(cpu_root, configuration='reference')
                    prepared_fields, prepared_spec = _snapshot(prepared_root, configuration='reference')
                    self.assertEqual(cpu_fields, expected)
                    self.assertEqual(prepared_fields, expected)
                    self.assertEqual(cpu_fields, prepared_fields)
                    self.assertEqual(cpu_spec, prepared_spec)
                    self.assertEqual(cpu_spec['axes'], [])
                    self.assertIsNone(cpu_spec['dtype'])
                    self.assertEqual(cpu_spec['definition']['encoding'], 'native-hdf5-fields-v1')
                    # Retain mixed precision and an integer larger than float64's exact range.
                    integer = load_correlator_output(prepared_root, ('vacuum_sink',), configuration='reference')
                    self.assertEqual(integer.dtype, np.int64)
                    self.assertEqual(set(integer.tolist()), {2**60 + 3, 2**60 + 4, 2**60 + 5})
                finally:
                    processor.close()

    def test_incomplete_processor_never_publishes_root(self):
        for layout in ('pairs_and_dt', 'dt_packed'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                plan, binding, values, backend, processor, template, _ = _processor_case(layout)
                writer_class = PackedDtOutputWriter if layout == 'dt_packed' else CfgOutputWriter
                directory = str(Path(directory) / 'dataset')
                writer = writer_class(plan, 'reference', directory, binding, prepacked=True)
                with patch.object(output, 'bind_device_output_assembly', return_value=lambda items: items[0]):
                    packed = processor.process(0, (_device_tree(values[0], backend),), template)
                if packed:
                    writer.consume_packed(0, np.asarray(packed[0]).copy())
                manager = GpuDatasetPostprocessing(plan, {'reference': binding})
                manager.active['reference'] = processor
                with self.assertRaisesRegex(ValueError, 'incomplete cfg coverage') as caught:
                    manager.publish('reference', writer)
                writer.abort(caught.exception)
                self.assertTrue(processor.closed)
                self.assertEqual(manager.active, {})
                _assert_cfg_unpublished(self, directory, 'reference')

    def test_prepacked_record_count_and_dtype_fail_without_root_publication(self):
        plan, binding, _ = _synthetic_case('pairs_and_dt')
        for record in (np.zeros(plan.dt_storage_shape, dtype='float64'),):
            with tempfile.TemporaryDirectory() as directory:
                directory = str(Path(directory) / 'dataset')
                writer = CfgOutputWriter(plan, 'reference', directory, binding, prepacked=True)
                for index in range(3):
                    writer.consume_packed(index, np.zeros(plan.output_storage_shape, dtype='int64'))
                counts = np.zeros(11, dtype='int64'); counts[3] = 1
                with self.assertRaisesRegex(ValueError, 'host byte layout') as caught:
                    writer.publish_prepacked(counts, iter((record,)))
                writer.abort(caught.exception)
                _assert_cfg_unpublished(self, directory, 'reference')
        with tempfile.TemporaryDirectory() as directory:
            directory = str(Path(directory) / 'dataset')
            writer = CfgOutputWriter(plan, 'reference', directory, binding, prepacked=True)
            for index in range(3):
                writer.consume_packed(index, np.zeros(plan.output_storage_shape, dtype='int64'))
            counts = np.zeros(11, dtype='int64'); counts[3] = 1
            try:
                with self.assertRaises(ValueError) as caught:
                    writer.publish_prepacked(counts, iter(()))
            finally:
                writer.abort(None)
            _assert_cfg_unpublished(self, directory, 'reference')


class OutputProcessorLifecycleTests(unittest.TestCase):
    def test_pending_owner_is_retained_until_event_and_released_on_close(self):
        plan, binding, values, backend, processor, template, _ = _processor_case()
        value = _device_tree(values[0], backend)
        reference = weakref.ref(value['two_point'])
        with patch.object(output, 'bind_device_output_assembly', return_value=lambda items: items[0]):
            processor.process(0, (value,), template)
        del value
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertIsNotNone(processor.pending)
        processor.close()
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(processor.pending)
        self.assertEqual(processor.programs, [])
        self.assertEqual(processor.accumulator.sums, {})
        processor.close()

    def test_failed_event_wait_fences_stream_before_releasing_owners(self):
        _, _, values, backend, processor, template, _ = _processor_case()
        value = _device_tree(values[0], backend)
        reference = weakref.ref(value['two_point'])
        with patch.object(output, 'bind_device_output_assembly', return_value=lambda items: items[0]):
            processor.process(0, (value,), template)
        del value
        event = processor.pending[0]
        observed = []
        def fence():
            observed.append(reference() is not None)
        with patch.object(event, 'synchronize', side_effect=RuntimeError('event failed')), patch.object(
                backend.cuda.compute_stream, 'synchronize', side_effect=fence):
            with self.assertRaisesRegex(RuntimeError, 'event failed'):
                processor.close()
        gc.collect()
        self.assertEqual(observed, [True])
        self.assertIsNone(reference())
        self.assertTrue(processor.closed)
        self.assertEqual(processor.accumulator.sums, {})

    def test_next_coordinate_releases_previous_owner_and_keeps_one_template(self):
        _, _, values, backend, processor, template, _ = _processor_case()
        references = []
        with patch.object(output, 'bind_device_output_assembly', return_value=lambda items: items[0]):
            for index, point in enumerate(values):
                value = _device_tree(point, backend)
                references.append(weakref.ref(value['two_point']))
                # Model independent window templates, which must not accumulate.
                template = SimpleNamespace(output=object(), plan=template.plan)
                processor.process(index, (value,), template)
                del value
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references[:-1]))
                self.assertIsNotNone(references[-1]())
                self.assertEqual(len(processor.programs), 1)
        processor.close()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_indirect_argument_tables_stay_alive_until_their_event(self):
        _, _, values, backend, processor, template, _ = _processor_case()
        references = []
        class Bound:
            metadata_bytes = 8
            scratch_bytes = 16
            last_stage_launches = 2
            def __call__(self, inputs):
                table = backend.array([len(references)], dtype='uint64')
                references.append(weakref.ref(table))
                self.call_owners = (table,)
                return inputs[0]
        def bind(*args, **kwargs):
            return Bound()
        with patch.object(output, 'bind_device_output_assembly', side_effect=bind):
            for index, point in enumerate(values):
                processor.process(index, (_device_tree(point, backend),), template)
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references[:-1]))
                self.assertIsNotNone(references[-1]())
                self.assertIs(processor.pending[1][-1][0], references[-1]())
                self.assertEqual(processor.device.statistics['output_postprocess_stage_launches'], 2*(index+1))
                self.assertEqual(processor.device.statistics['output_metadata_upload_calls'], index+1)
                self.assertEqual(processor.device.statistics['output_metadata_upload_bytes'], 8*(index+1))
                self.assertEqual(processor.device.statistics['output_postprocess_max_scratch_bytes'], 16)
                self.assertEqual(processor.device.statistics['output_postprocess_bindings'], 1)
        processor.close()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_event_record_error_fences_current_stream_before_unwinding(self):
        _, _, values, backend, processor, template, _ = _processor_case()
        value = _device_tree(values[0], backend)
        observed = []
        event = SimpleNamespace(record=Mock(side_effect=RuntimeError('event record failed')))
        with patch.object(output, 'bind_device_output_assembly', return_value=lambda items: items[0]), patch.object(
                backend.cuda, 'Event', return_value=event), patch.object(
                backend.cuda.compute_stream, 'synchronize', side_effect=lambda: observed.append(value['two_point'].copy())):
            with self.assertRaisesRegex(RuntimeError, 'event record failed'):
                processor.process(0, (value,), template)
        self.assertGreaterEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0], value['two_point'])
        self.assertEqual(processor.consumed, 0)
        processor.close()

    def test_publication_error_stays_primary_when_processor_cleanup_also_fails(self):
        manager = GpuDatasetPostprocessing(SimpleNamespace(output_layout='dt_packed'), {})
        processor = SimpleNamespace(
            packed_records=Mock(side_effect=OSError('primary finalization failed')),
            close=Mock(side_effect=RuntimeError('secondary cleanup failed')))
        manager.active['cfg'] = processor
        with self.assertRaisesRegex(OSError, 'primary finalization failed') as caught:
            manager.publish('cfg', Mock())
        self.assertTrue(any('secondary cleanup failed' in note
                            for note in getattr(caught.exception, '__notes__', ())))
        self.assertEqual(manager.active, {})

    def test_repeated_or_missing_coordinate_fails_before_assembly(self):
        _, _, values, backend, processor, template, _ = _processor_case()
        with patch.object(output, 'bind_device_output_assembly', return_value=lambda items: items[0]) as bind:
            with self.assertRaisesRegex(ValueError, 'repeats or omits'):
                processor.process(1, (_device_tree(values[0], backend),), template)
            self.assertEqual(bind.call_count, 0)
            processor.process(0, (_device_tree(values[0], backend),), template)
            with self.assertRaisesRegex(ValueError, 'repeats or omits'):
                processor.process(0, (_device_tree(values[0], backend),), template)
        processor.close()

    def test_manager_closes_every_cfg_even_if_first_cleanup_fails(self):
        manager = GpuDatasetPostprocessing(SimpleNamespace(output_layout='dt_packed'), {})
        first = SimpleNamespace(close=Mock(side_effect=RuntimeError('first close failed')))
        second = SimpleNamespace(close=Mock())
        manager.active.update(a=first, b=second)
        with self.assertRaisesRegex(RuntimeError, 'first close failed'):
            manager.close()
        first.close.assert_called_once(); second.close.assert_called_once()
        self.assertEqual(manager.active, {})


def _selected(target):
    options = inputs()
    options['correlator'] = {'two_point': options['correlator']}
    options['search'] = {'max_states': 128, 'deadline_seconds': 0.1, 'component_workers': 1}
    selected = calculate(**options, target_hardware=target)
    root = selected.expanded.output_recipe.root
    recipe = replace(selected.expanded.output_recipe, root=replace(root,
        keys=(*root.keys, 'vacuum_sink', 'vacuum_source'),
        children=(*root.children, OutputNode('constant_array', np.array([1+2j, 3+4j]), shape=(2,)),
                  OutputNode('constant_array', np.array([5+6j, 7+8j]), shape=(2,)))))
    expanded = replace(selected.expanded, output_recipe=recipe)
    return replace(selected, expanded=expanded, path=replace(selected.path, expanded=expanded))


def _coords():
    return describe_coordinates(time_pairs=[{0: 1, 1: 0}, {0: 0, 1: 1}])[0]


def _physical_factory(complete, selected, backend, closed):
    raw = _factory(complete, selected, getattr(backend, 'tracker', None) or _ProtocolBackend().tracker, closed)
    @contextmanager
    def bind(cfg):
        with raw(cfg) as binding:
            yield replace(binding, physical_times=(4, 19), physical_lt=128)
    return bind


@contextmanager
def _protocol_runtime():
    with ExitStack() as scope:
        scope.enter_context(patch.object(output, 'bind_device_output_assembly', side_effect=_protocol_assembly))
        scope.enter_context(patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                                  return_value=1 << 40))
        scope.enter_context(patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                                  return_value=(1 << 30, 1 << 30)))
        yield


class GpuDatasetProtocolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu_selected = _selected(_targets()[0])
        cls.gpu_selected = _selected(_targets()[1])

    def test_default_cpu_is_unchanged_and_never_binds_device_postprocessing(self):
        implicit = complete_calculation(self.cpu_selected, coordinates=_coords(), output_mode='dataset', physical_lt=128)
        explicit = complete_calculation(self.cpu_selected, coordinates=_coords(), output_mode='dataset',
                                        physical_lt=128, postprocess_backend='cpu')
        for plan in (implicit.plan, explicit.plan):
            self.assertNotIn('postprocess_backend', plan.statistics)
        self.assertEqual(implicit.plan.output_fields, explicit.plan.output_fields)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                output, 'bind_device_output_assembly', side_effect=AssertionError('GPU path used by default')):
            outputs = []
            for name, calculation in (('implicit', implicit), ('explicit', explicit)):
                closed = []
                calculation = replace(calculation, options=dict(calculation.options,
                    configuration='a', output=str(Path(directory)/name),
                    source_factory=_physical_factory(calculation, self.cpu_selected, np, closed)))
                outputs.append(calculation.run())
                self.assertEqual(closed, ['a'])
            implicit_fields, implicit_spec = _snapshot(Path(directory)/'implicit', configuration='a')
            explicit_fields, explicit_spec = _snapshot(Path(directory)/'explicit', configuration='a')
            self.assertEqual(implicit_fields, explicit_fields)
            self.assertEqual(implicit_spec, explicit_spec)

    def test_cpu_target_and_memory_mode_are_rejected_before_planning_or_reader(self):
        with patch('lqcdweave.contraction.correlator_cfg.build_cfg_plan',
                   side_effect=AssertionError('planning started before validation')):
            with self.assertRaisesRegex(ValueError, 'CuPy execution target'):
                complete_calculation(self.cpu_selected, output_mode='dataset', postprocess_backend='gpu')
            with self.assertRaisesRegex(ValueError, 'dataset output'):
                complete_calculation(self.gpu_selected, output_mode='memory', postprocess_backend='gpu')
            for value in ('GPU', 'cuda', None):
                with self.assertRaisesRegex(ValueError, 'postprocess_backend'):
                    complete_calculation(self.cpu_selected, postprocess_backend=value)
        source = Mock(side_effect=AssertionError('opened reader before validation'))
        with self.assertRaisesRegex(ValueError, 'CuPy execution target'):
            run_cfg_window_calculation(self.cpu_selected, coordinates=_coords(), output='unused',
                                       source_factory=source, postprocess_backend='gpu')
        source.assert_not_called()

    def test_gpu_reserve_is_offline_saved_and_rejects_tight_device_cap_first(self):
        import builtins
        original_import = builtins.__import__
        def no_cupy(name, *args, **kwargs):
            if name == 'cupy' or name.startswith('cupy.'):
                raise AssertionError('offline postprocessing reserve imported CuPy')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=no_cupy), patch(
                'lqcdweave.contraction.correlator_postprocessing.DeviceTimePairAccumulator',
                side_effect=AssertionError('offline planning allocated device buffers')):
            complete = complete_calculation(self.gpu_selected, coordinates=_coords(), output_mode='dataset',
                physical_lt=128, output_layout='dt_packed', postprocess_backend='gpu')
            self.assertEqual(complete.plan.statistics['postprocess_backend'], 'gpu')
            self.assertGreater(complete.plan.statistics['postprocess_device_bytes'], 0)
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / 'gpu.json.gz'
                save_calculation_plan(archive, complete)
                saved = load_cfg_plan(archive)
                for key in ('postprocess_backend', 'postprocess_device_bytes', 'postprocess_host_bytes'):
                    self.assertEqual(saved.statistics[key], complete.plan.statistics[key])
            tight = replace(self.gpu_selected, path=replace(self.gpu_selected.path,
                plan=replace(self.gpu_selected.plan, memory_limit=(self.gpu_selected.plan.peak_device_bytes
                    - self.gpu_selected.plan.packed_value_cache_device_bytes))))
            with patch('lqcdweave.contraction.correlator_cfg.build_cfg_plan',
                       side_effect=AssertionError('planner started before fixed buffer reserve')):
                with self.assertRaisesRegex(MemoryError, 'postprocessing buffers'):
                    complete_calculation(tight, coordinates=_coords(), output_mode='dataset', physical_lt=128,
                                         output_layout='dt_packed', postprocess_backend='gpu')

    def test_loaded_gpu_plan_rejects_stale_reserve_before_any_device_allocation(self):
        complete = complete_calculation(self.gpu_selected, coordinates=_coords(), output_mode='dataset',
            physical_lt=128, output_layout='dt_packed', postprocess_backend='gpu')
        plan = replace(complete.plan, statistics=dict(complete.plan.statistics, postprocess_device_bytes=0))
        with patch('lqcdweave.contraction.correlator_postprocessing.DeviceTimePairAccumulator',
                   side_effect=AssertionError('allocated before saved reserve revalidation')):
            with self.assertRaisesRegex(MemoryError, 'saved GPU postprocessing reserve'):
                GpuDatasetPostprocessing(plan, {})

    def test_window_device_budget_fails_before_reader_or_planner(self):
        selected = replace(self.gpu_selected, path=replace(self.gpu_selected.path,
            plan=replace(self.gpu_selected.plan, memory_limit=(self.gpu_selected.plan.peak_device_bytes
                    - self.gpu_selected.plan.packed_value_cache_device_bytes))))
        source = Mock(side_effect=AssertionError('opened reader before GPU reserve check'))
        with _protocol_runtime(), patch(
                'lqcdweave.contraction.correlator_window_run.plan_cfg_windows',
                side_effect=AssertionError('started window planner before GPU reserve check')):
            with self.assertRaisesRegex(MemoryError, 'postprocessing buffers'):
                run_cfg_window_calculation(selected, coordinates=_coords(), output='unused',
                    source_factory=source, backend=_ProtocolBackend(), physical_lt=128,
                    output_layout='dt_packed', postprocess_backend='gpu')
        source.assert_not_called()

    def test_writer_failure_closes_pending_processor_readers_and_device(self):
        for layout in ('pairs_and_dt', 'dt_packed'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                backend = _ProtocolBackend(); closed = []; processors = []
                complete = complete_calculation(self.gpu_selected, coordinates=_coords(), output_mode='dataset',
                    physical_lt=128, output_layout=layout, postprocess_backend='gpu')
                complete = replace(complete, options=dict(complete.options, configuration='a', output=directory,
                    source_factory=_physical_factory(complete, self.gpu_selected, backend, closed)))
                original = output.DeviceOutputProcessor
                def create(*args, **kwargs):
                    processor = original(*args, **kwargs)
                    processors.append(processor)
                    return processor
                def fail(writer, path, value, **options):
                    raise OSError('disk write failed')
                with _protocol_runtime(), patch(
                        'lqcdweave.contraction.correlator_postprocessing.DeviceOutputProcessor',
                        side_effect=create), patch.object(DatasetWriter, 'write_field', fail):
                    with self.assertRaisesRegex(OSError, 'disk write failed'):
                        complete.run(backend=backend)
                self.assertEqual(len(processors), 1)
                self.assertTrue(processors[0].closed)
                self.assertIsNone(processors[0].pending)
                self.assertEqual(processors[0].accumulator.sums, {})
                self.assertEqual(closed, ['a'])
                self.assertTrue(backend.tracker.pools[0].freed)
                _assert_cfg_unpublished(self, directory, 'a')

    def test_writer_failure_stays_primary_when_final_cleanup_also_fails(self):
        for windows in (False, True):
            with self.subTest(windows=windows), tempfile.TemporaryDirectory() as directory:
                backend = _ProtocolBackend(); closed = []
                complete = complete_calculation(self.gpu_selected, coordinates=_coords(), output_mode='dataset',
                    physical_lt=128, output_layout='pairs_and_dt', postprocess_backend='gpu')
                factory = _physical_factory(complete, self.gpu_selected, backend, closed)
                complete = replace(complete, options=dict(complete.options, configuration='a', output=directory,
                    source_factory=factory))
                original_close = output.DeviceOutputProcessor.close
                def fail_close(processor):
                    original_close(processor)
                    raise RuntimeError('secondary output cleanup failed')
                with _protocol_runtime(), patch.object(DatasetWriter, 'write_field', side_effect=OSError('primary disk failed')), patch.object(
                        output.DeviceOutputProcessor, 'close', fail_close):
                    with self.assertRaisesRegex(OSError, 'primary disk failed') as caught:
                        if windows:
                            run_cfg_window_calculation(self.gpu_selected, coordinates=_coords(), output=directory,
                                source_factory=factory, configuration='a', backend=backend, physical_lt=128,
                                postprocess_backend='gpu', window_size=1)
                        else:
                            complete.run(backend=backend)
                self.assertTrue(any('secondary output cleanup failed' in note
                                    for note in getattr(caught.exception, '__notes__', ())))
                self.assertEqual(closed, ['a'])
                _assert_cfg_unpublished(self, directory, 'a')

    def test_complete_both_layouts_match_cpu_and_dt_packed_downloads_only_final_records(self):
        for layout in ('dt_packed', 'pairs_and_dt'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                for name, selected, backend, mode in (
                        ('cpu', self.cpu_selected, np, 'cpu'),
                        ('protocol', self.gpu_selected, _ProtocolBackend(), 'gpu')):
                    complete = complete_calculation(selected, coordinates=_coords(), output_mode='dataset',
                        physical_lt=128, output_layout=layout, postprocess_backend=mode)
                    closed = []
                    complete = replace(complete, options=dict(complete.options, configuration=iter(('a', 'b')),
                        output=str(Path(directory)/name),
                        source_factory=_physical_factory(complete, selected, backend, closed)))
                    with _protocol_runtime():
                        result = complete.run(backend=backend)
                    self.assertEqual(closed, ['a', 'b'])
                    self.assertEqual(result.report['completed'], ('a', 'b'))
                    if mode == 'gpu':
                        # Counts record transfers: final dt plus loops per cfg; pair mode
                        # transfers two persisted pairs plus two physical-dt records per cfg.
                        self.assertEqual(result.report['device']['output_download_calls'],
                                         4 if layout == 'dt_packed' else 8)
                        self.assertEqual(result.report['postprocess_backend'], 'gpu')
                        if layout == 'dt_packed':
                            expected_bytes = 2 * 8 * (np.prod(complete.plan.dt_packed_storage_shape)
                                + np.prod(complete.plan.one_point_storage_shape))
                            self.assertEqual(result.report['device']['downloaded_bytes'], expected_bytes)
                        self.assertTrue(backend.tracker.pools[0].freed)
                for cfg in ('a', 'b'):
                    for field in ('two_point', 'vacuum_sink', 'vacuum_source'):
                        left = load_correlator_output(Path(directory)/'cpu', (field,), configuration=cfg)
                        right = load_correlator_output(Path(directory)/'protocol', (field,), configuration=cfg)
                        np.testing.assert_allclose(left, right, rtol=3e-12, atol=3e-12, equal_nan=True)
                    expected = read_dataset_metadata(Path(directory)/'cpu',cfg=cfg)['spec'].document
                    actual = read_dataset_metadata(Path(directory)/'protocol',cfg=cfg)['spec'].document
                    self.assertEqual(actual, expected)
                    self.assertTrue((Path(directory)/'protocol'/f'{cfg}.h5').exists())
                self.assertEqual({path.name for path in (Path(directory)/'protocol').glob('*.h5')},
                                 {'a.h5','b.h5'})

    def test_window_both_layouts_match_complete_numeric_bytes_and_schema(self):
        for layout in ('pairs_and_dt', 'dt_packed'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                backend = _ProtocolBackend(); closed = []
                # Default host postprocessing on the same selected contraction is
                # independent of the device-transform/temporal/prepacked path.
                complete = complete_calculation(self.gpu_selected, coordinates=_coords(), output_mode='dataset',
                    physical_lt=128, output_layout=layout, postprocess_backend='cpu')
                factory = _physical_factory(complete, self.gpu_selected, backend, closed)
                complete = replace(complete, options=dict(complete.options, configuration='a',
                    output=str(Path(directory)/'cpu'), source_factory=factory))
                with _protocol_runtime():
                    complete.run(backend=backend)
                    run_cfg_window_calculation(self.gpu_selected, coordinates=_coords(),
                        output=Path(directory)/'gpu-protocol', configuration='a', source_factory=factory,
                        backend=backend, physical_lt=128, output_layout=layout,
                        postprocess_backend='gpu', window_size=1)
                cpu_fields, cpu_spec = _snapshot(Path(directory)/'cpu', configuration='a')
                gpu_fields, gpu_spec = _snapshot(Path(directory)/'gpu-protocol', configuration='a')
                self.assertEqual(cpu_fields, gpu_fields)
                self.assertEqual(cpu_spec, gpu_spec)
                self.assertEqual(closed, ['a', 'a'])

    def test_window_protocol_preserves_coverage_and_cancellation_never_publishes(self):
        selected = self.gpu_selected
        complete = complete_calculation(selected, coordinates=_coords(), output_mode='dataset', physical_lt=128)
        for cancel in (False, True):
            backend = _ProtocolBackend(); closed = []; events = []
            def progress(kind, values):
                events.append((kind, values['count']))
                if cancel and kind == 'point':
                    raise RuntimeError('cancelled after first device result')
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as directory, _protocol_runtime():
                kwargs = dict(coordinates=_coords(), output=directory, configuration='a', backend=backend,
                    source_factory=_physical_factory(complete, selected, backend, closed), physical_lt=128,
                    output_layout='dt_packed', postprocess_backend='gpu', window_size=1, progress=progress)
                if cancel:
                    with self.assertRaisesRegex(RuntimeError, 'cancelled after first'):
                        run_cfg_window_calculation(selected, **kwargs)
                    _assert_cfg_unpublished(self, directory, 'a')
                    self.assertNotIn(('published', 2), events)
                else:
                    result = run_cfg_window_calculation(selected, **kwargs)
                    self.assertEqual(result.report['completed'], ('a',))
                    self.assertEqual(result.report['postprocess_backend'], 'gpu')
                    self.assertEqual(result.report['device']['output_download_calls'], 2)
                    self.assertEqual([n for kind, n in events if kind == 'point'], [1, 2])
                    self.assertEqual(events[-1], ('published', 2))
                    actual = load_correlator_output(Path(directory), ('two_point',), configuration='a')
                    self.assertEqual(actual.shape[-1], 127)
                self.assertEqual(closed, ['a'])
                self.assertTrue(backend.tracker.pools[0].freed)


class RealCuPyDatasetPostprocessingTests(unittest.TestCase):
    """Actual small CUDA integration; never substituted by the protocol double."""
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as error:
            raise unittest.SkipTest(f'real CuPy dataset postprocessing requires a working device: {error}')
        cls.backend = cupy
        cls.cpu_selected = _selected(_targets()[0])
        cls.gpu_selected = _selected(_targets()[1])

    def test_both_dataset_layouts_match_cpu_on_actual_device(self):
        for layout in ('pairs_and_dt', 'dt_packed'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                for name, selected, backend, mode in (
                        ('cpu', self.cpu_selected, np, 'cpu'), ('gpu', self.gpu_selected, self.backend, 'gpu')):
                    complete = complete_calculation(selected, coordinates=_coords(), output_mode='dataset',
                        physical_lt=128, output_layout=layout, postprocess_backend=mode)
                    closed = []
                    complete = replace(complete, options=dict(complete.options, configuration='a',
                        output=str(Path(directory)/name),
                        source_factory=_physical_factory(complete, selected, backend, closed)))
                    result = complete.run(backend=backend)
                    self.assertEqual(result.report['completed'], ('a',))
                    self.assertEqual(closed, ['a'])
                for field in ('two_point', 'vacuum_sink', 'vacuum_source'):
                    expected = load_correlator_output(Path(directory)/'cpu', (field,), configuration='a')
                    actual = load_correlator_output(Path(directory)/'gpu', (field,), configuration='a')
                    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12, equal_nan=True)
                expected = read_dataset_metadata(Path(directory)/'cpu',cfg='a')['spec'].document
                actual = read_dataset_metadata(Path(directory)/'gpu',cfg='a')['spec'].document
                self.assertEqual(actual, expected)


    def test_incremental_dt_packed_matches_complete_on_actual_device(self):
        with tempfile.TemporaryDirectory() as directory:
            closed = []
            complete = complete_calculation(self.gpu_selected, coordinates=_coords(), output_mode='dataset',
                physical_lt=128, output_layout='dt_packed', postprocess_backend='gpu')
            factory = _physical_factory(complete, self.gpu_selected, self.backend, closed)
            complete = replace(complete, options=dict(complete.options, configuration='a',
                output=str(Path(directory)/'complete'), source_factory=factory))
            complete.run(backend=self.backend)
            run_cfg_window_calculation(self.gpu_selected, coordinates=_coords(),
                output=Path(directory)/'windows', configuration='a', source_factory=factory,
                backend=self.backend, physical_lt=128, output_layout='dt_packed',
                postprocess_backend='gpu', window_size=1)
            for field in ('two_point', 'vacuum_sink', 'vacuum_source'):
                expected = load_correlator_output(Path(directory)/'complete', (field,), configuration='a')
                actual = load_correlator_output(Path(directory)/'windows', (field,), configuration='a')
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12, equal_nan=True)
            self.assertEqual(closed, ['a', 'a'])


if __name__ == '__main__':
    unittest.main()
