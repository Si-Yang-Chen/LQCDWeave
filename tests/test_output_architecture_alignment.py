"""Native field byte/budget parity and independent temporal-service handoffs."""
from pathlib import Path
from types import SimpleNamespace as N
from dataclasses import dataclass, field, replace
import hashlib, json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from lqcdweave.contracts.correlator.api import CfgOutputField
from lqcdweave.storage.dataset import DatasetReader, read_dataset_metadata
from lqcdweave.storage.correlator_results import save_correlator, load_correlator
from lqcdweave.contraction.correlator_output import load_correlator_output
from lqcdweave.storage.correlator_results import CfgOutputWriter, PackedDtOutputWriter
from lqcdweave.storage.dataset import DatasetWriter
from lqcdweave.contraction.planning.api import configure_cfg_output_layout
from lqcdweave.sampling.temporal.api import (
    TimePairAccumulator, reconstruct_two_point_from_coordinates,
    pair_time_differences, mean_time_pairs, count_time_pairs,
)
from lqcdweave.contraction.correlator_dt_output import configure_packed_dt

@dataclass
class Plan:
    output_fields: tuple
    instances: tuple
    output_mode: str='dataset'
    physical_lt: int=8
    output_words: int=5
    dt_words: int=3
    output_storage_shape: tuple=(1,5)
    dt_storage_shape: tuple=(1,3)
    time_roles: tuple=('sink','source')
    coordinate_shape: tuple=(6,)
    output_definition: dict=field(default_factory=lambda:{'task':'fixed-codec-parity','nested':{'axes':[1,2]}})
    output_layout: str='pairs_and_dt'
    dt_coordinates: tuple=()
    dt_packed_storage_shape: tuple=(0,0)
    one_point_storage_shape: tuple=(0,0)
    one_point_slots: int=0
    peak_host_bytes: int=10000
    output_host_bytes: int=4000
    prefetch_host_bytes: int=2000
    statistics: dict=field(default_factory=dict)
    backend: str='numpy'
    templates: tuple=field(default_factory=lambda:(N(plan=N(host_memory_limit=20000,memory_limit=20000)),))

    @property
    def coordinates(self):
        return tuple(instance.coordinate for instance in self.instances)

@dataclass
class Calc:
    plan: object

fields=(CfgOutputField(('two_point',),(2,),'<c8',0,2,0),
        CfgOutputField(('real',),(),'>f4',2,1,2),
        CfgOutputField(('vacuum_sink',),(1,),'<c8',3,1,-1,'sink'),
        CfgOutputField(('vacuum_source',),(),'<f8',4,1,-1,'source'))
coords=tuple(N(coordinate=(('source',s),('sink',t))) for s in range(3) for t in range(3) if s!=t)
plan=Plan(fields,coords)
binding=N(physical_lt=8,physical_times=(6,0,2),ensemble='synthetic',lattice=(1,1,1,8),parents=(),provenance={'source':'fixed-codec-parity'})


def contribution(index, item):
    q = dict(item.coordinate)
    return {'two_point': np.array([index+0.25j, 1e5+index*0.5-3j], dtype='<c8'),
            'real': np.array(index/3, dtype='>f4'),
            'vacuum_sink': np.array([q['sink']+1j], dtype='<c8'),
            'vacuum_source': np.array(q['source']+0.125, dtype='<f8')}


def writer_for(root, layout):
    configured = configure_cfg_output_layout(plan, layout, None, host_cap=20000)
    cls = CfgOutputWriter if layout == 'pairs_and_dt' else PackedDtOutputWriter
    writer = cls(configured, 'fixed', root, binding, debug=True)
    for index, item in enumerate(coords):
        writer.consume(index, contribution(index, item))
    return configured, writer


def _reference_native_fields(configured, layout):
    values = [contribution(index, item) for index, item in enumerate(configured.instances)]
    points = [dict(item.coordinate) for item in configured.instances]
    counts = np.zeros(configured.physical_lt, dtype=np.int64)
    differences = []
    sink_role, source_role = configured.time_roles
    for point in points:
        dt = (binding.physical_times[point[sink_role]] -
              binding.physical_times[point[source_role]]) % configured.physical_lt
        differences.append(dt)
        counts[dt] += 1

    result = {}
    for field in configured.output_fields:
        if layout == 'pairs_and_dt':
            result[('pairs', *field.path)] = np.stack(
                [value[field.path[-1]] for value in values], axis=-1).astype(
                    np.dtype(field.dtype), copy=False)
        if field.dt_offset >= 0:
            sums = np.zeros((configured.physical_lt, *field.shape), dtype=np.dtype(field.dtype))
            for dt, value in zip(differences, values, strict=True):
                sums[dt] += value[field.path[-1]]
            means = np.full(sums.shape, np.nan, dtype=np.dtype(field.dtype))
            for dt, count in enumerate(counts):
                if count:
                    np.divide(sums[dt:dt+1], count, out=means[dt:dt+1])
            means = np.moveaxis(means, 0, -1)
            coordinates = (range(configured.physical_lt) if layout == 'pairs_and_dt'
                           else configured.dt_coordinates)
            result[('dt', *field.path)] = np.take(means, coordinates, axis=-1)
        elif layout == 'dt_packed':
            slots = np.empty((configured.one_point_slots, *field.shape), dtype=np.dtype(field.dtype))
            seen = np.zeros(configured.one_point_slots, dtype=bool)
            for point, value in zip(points, values, strict=True):
                slot = point[field.time_role]
                if not seen[slot]:
                    slots[slot] = value[field.path[-1]]
                    seen[slot] = True
            if not seen.all():
                raise AssertionError('independent one-point reference has incomplete time coverage')
            result[('one_point', *field.path)] = np.moveaxis(slots, 0, -1)
    return result


def _native_snapshot(root, configuration='fixed'):
    with DatasetReader(root, cfg=configuration) as reader:
        fields = {}
        for key in sorted(reader.fields):
            value = np.asarray(reader.read_field(json.loads(key)))
            fields[key] = dict(shape=list(value.shape), dtype=value.dtype.str,
                               bytes=value.tobytes().hex())
        return fields


class OutputArchitectureTests(unittest.TestCase):
    def test_native_field_bytes_and_budgets_and_both_load_frontends(self):
        from lqcdweave.storage.correlator_results import load_correlator_output as storage_load
        self.assertIs(storage_load, load_correlator_output)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {}
            for layout in ('pairs_and_dt', 'dt_packed'):
                configured, writer = writer_for(root/layout, layout)
                writer.publish()
                self.assertEqual({path.name for path in (root/layout).glob('*.h5')}, {'fixed.h5'})
                report[layout] = {'budgets': {k: getattr(configured, k) for k in (
                    'peak_host_bytes', 'output_host_bytes', 'prefetch_host_bytes',
                    'dt_packed_storage_shape', 'one_point_storage_shape', 'one_point_slots')},
                    'statistics': dict(configured.statistics)}
                with DatasetReader(root/layout, cfg='fixed') as reader:
                    spec = reader.spec.document
                    definition = spec['definition']
                    self.assertEqual(spec['axes'], [])
                    self.assertIsNone(spec['dtype'])
                    self.assertEqual(definition['encoding'], 'native-hdf5-fields-v1')
                    self.assertEqual(definition['output_layout'], layout)
                    self.assertEqual(definition['coordinate_shape'], list(configured.coordinate_shape))
                    self.assertEqual(definition['coordinates'], [dict(item.coordinate) for item in coords])
                    self.assertEqual(definition['physical_times'], list(binding.physical_times))
                    self.assertEqual(definition['physical_dt'], list(configured.dt_coordinates)
                        if layout == 'dt_packed' else list(range(configured.physical_lt)))
                    self.assertEqual(definition['fields'], [dict(path=list(f.path), shape=list(f.shape),
                        dtype=f.dtype, time_role=f.time_role, averaged_dt=f.dt_offset >= 0)
                        for f in configured.output_fields])
                    self.assertTrue(all(set(field) == {'path','shape','dtype','time_role','averaged_dt'}
                                        for field in definition['fields']))
                    expected_counts = np.zeros(configured.physical_lt, dtype=np.int64)
                    sink_role, source_role = configured.time_roles
                    for item in configured.instances:
                        point = dict(item.coordinate)
                        dt = (binding.physical_times[point[sink_role]] -
                              binding.physical_times[point[source_role]]) % configured.physical_lt
                        expected_counts[dt] += 1
                    self.assertEqual(definition['measured_pair_counts'], expected_counts.tolist())
                    for field in definition['fields']:
                        if layout == 'pairs_and_dt':
                            role = 'pairs'
                            extra_shape = tuple(definition['coordinate_shape'])
                        elif field['averaged_dt']:
                            role = 'dt'
                            extra_shape = (len(definition['physical_dt']),)
                        else:
                            role = 'one_point'
                            extra_shape = (len(definition['physical_times']),)
                        value = reader.read_field((role, *field['path']))
                        self.assertEqual(value.shape, tuple(field['shape']) + extra_shape)
                        self.assertEqual(value.dtype.str, field['dtype'])
                    expected_fields = _reference_native_fields(configured, layout)
                    self.assertEqual({tuple(json.loads(key)) for key in reader.fields}, set(expected_fields))
                    for path, expected in expected_fields.items():
                        actual = reader.read_field(path)
                        self.assertEqual(actual.shape, expected.shape, path)
                        self.assertEqual(actual.dtype.str, expected.dtype.str, path)
                        if actual.dtype.kind in 'fc':
                            self.assertEqual(np.isnan(actual).sum(), np.isnan(expected).sum(), path)
                        self.assertEqual(actual.tobytes(), expected.tobytes(), path)
                    report[layout]['fields'] = _native_snapshot(root/layout)
                for f in fields:
                    array = storage_load(root/layout, f.path, dt=f.dt_offset >= 0, configuration='fixed')
                    self.assertEqual(array.dtype.newbyteorder('='), np.dtype(f.dtype).newbyteorder('='))
            value = {'data': np.arange(18, dtype=np.float32).reshape(2,3,3)*(1+2j),
                     'scalar': np.asarray(3, dtype='>i2')}
            save_correlator(root/'structured', value)
            restored = load_correlator(root/'structured')
            np.testing.assert_array_equal(restored.value['data'], value['data'])
            self.assertEqual(restored.value['scalar'].dtype.str, '>i2')
            report['structured_fields'] = _native_snapshot(root/'structured', configuration='_unknown')
            serialized = json.dumps(report, sort_keys=True, separators=(',', ':'))
            self.assertEqual(hashlib.sha256(serialized.encode()).hexdigest(),
                'fd1fe727018d3bf7cc72584f9c6132e528bf03595107f177bab3912da1b16cb3')

    def test_field_write_or_publish_failure_never_registers_cfg(self):
        for layout in ('pairs_and_dt', 'dt_packed'):
            for operation in ('write_field', 'publish'):
                with self.subTest(layout=layout, operation=operation), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)/'output'
                    _, writer = writer_for(root, layout)
                    def fail_write(storage_writer, *args, **kwargs):
                        raise OSError('cfg output publication failed')
                    def fail_publish(storage_writer, *args, **kwargs):
                        raise OSError('cfg output publication failed')
                    failure = fail_write if operation == 'write_field' else fail_publish
                    with patch.object(DatasetWriter, operation, failure):
                        with self.assertRaisesRegex(OSError, 'cfg output publication failed') as caught:
                            writer.publish()
                    writer.abort(caught.exception)
                    self.assertFalse((root/'fixed.h5').exists())
                    with self.assertRaises((FileNotFoundError, KeyError)):
                        read_dataset_metadata(root,cfg='fixed')
                    with self.assertRaises((FileNotFoundError, KeyError)):
                        DatasetReader(root,cfg='fixed')

    def test_pure_capacity_calculation_is_idempotent_and_caps_prefetch(self):
        before = dict(plan.statistics)
        result = configure_cfg_output_layout(plan, 'dt_packed', None, host_cap=11000)
        self.assertEqual((result.peak_host_bytes, result.output_host_bytes,
                          result.prefetch_host_bytes), (10382, 4382, 618))
        again = configure_cfg_output_layout(result, 'dt_packed', None, host_cap=11000)
        self.assertEqual(dict(again.statistics), dict(result.statistics))
        self.assertEqual(plan.statistics, before)
        with self.assertRaisesRegex(MemoryError, '10382 host bytes'):
            configure_cfg_output_layout(plan, 'dt_packed', None, host_cap=10381)
        self.assertIs(configure_cfg_output_layout(plan, 'pairs_and_dt', None, host_cap=None), plan)
        with patch('lqcdweave.contraction.correlator_cfg._host_cap', side_effect=AssertionError('unneeded cap')):
            original = Calc(plan)
            self.assertIs(configure_packed_dt(original, 'pairs_and_dt', None), original)

    def test_three_consumers_keep_original_complex64_dtype_and_count_policy(self):
        sources, sinks = (2, 6), (0, 6, 2)
        data = np.array([[[1e8+3j, 3+1j, -1e8+2j], [7-4j, 2+9j, -5+8j]],
                         [[6+4j, 8+7j, 11-6j], [4+5j, -9+2j, 13-3j]]], dtype=np.complex64)
        expected_counts = np.zeros(8, dtype=np.int64)
        sums = np.zeros((2,8), dtype=np.complex128)
        native = np.zeros((2,8), dtype=np.complex64)
        accumulator = TimePairAccumulator({'value': ((2,), np.complex64)}, Lt=8)
        with patch('numpy.stack', side_effect=AssertionError('pair matrix stack')):
            for i, source in enumerate(sources):
                for j, sink in enumerate(sinks):
                    dt = (sink-source) % 8
                    expected_counts[dt] += 1
                    sums[:,dt] += data[:,i,j]
                    native[:,dt] += data[:,i,j]
                    accumulator.add(source, sink, [('value', data[:,i,j])])
        # Resident state is Lt times each leaf shape, independent of pair count.
        self.assertEqual(accumulator.sums['value'].shape, (8,2))
        self.assertEqual(set(vars(accumulator)), {'Lt','counts','sums'})
        expected = np.full((2,8), np.nan, dtype=np.complex128)
        streaming_expected = np.full((2,8), np.nan, dtype=np.complex64)
        for dt, count in enumerate(expected_counts):
            if count:
                expected[:,dt] = sums[:,dt]/count
                streaming_expected[:,dt] = native[:,dt]/count
        geometry = pair_time_differences(sources, sinks, Lt=8)
        reconstructed = reconstruct_two_point_from_coordinates(data, Lt=8,
            tsrc_times=sources, dt_values=geometry, tsrc_axis=1, dt_axis=2, missing='nan')
        np.testing.assert_array_equal(reconstructed.counts, expected_counts)
        np.testing.assert_array_equal(reconstructed.values, expected)
        self.assertEqual(reconstructed.values.dtype, np.dtype('complex128'))
        with self.assertRaisesRegex(ValueError, 'no source pairs'):
            mean_time_pairs(data, geometry, Lt=8, source_axis=1, other_axis=2)
        streamed = np.empty((2,8), dtype=np.complex64)
        accumulator.write_means('value', range(8), streamed)
        np.testing.assert_array_equal(streamed, streaming_expected)
        np.testing.assert_array_equal(accumulator.counts, expected_counts)
        with tempfile.TemporaryDirectory() as directory:
            options = dict(axes=('source','sink'), coordinates={'source':sources,'sink':sinks},
                average_tsrc=True, source_axis='source', sink_axis='sink', physical_lt=8,
                boundary='periodic', coordinate_domain='physical')
            with self.assertRaisesRegex(ValueError, 'missing physical dt'):
                save_correlator(Path(directory)/'missing', data, **options)
            save_correlator(Path(directory)/'mean', data, missing='nan', **options)
            result = load_correlator(Path(directory)/'mean')
            self.assertEqual(result.value.dtype, np.dtype('complex64'))
            np.testing.assert_array_equal(result.value, expected.astype(np.complex64))
            np.testing.assert_array_equal(result.reduction['counts'], expected_counts)


if __name__ == '__main__':
    unittest.main()
