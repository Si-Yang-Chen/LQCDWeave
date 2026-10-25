"""Numerical result storage and explicit periodic source averaging."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.correlator.api import CorrelatorResult
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter
from lqcdweave.storage.correlator_results import load_correlator, save_correlator


class CorrelatorResultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def roundtrip(self, value, **options):
        path = self.root / 'result'
        identity = save_correlator(path, value, **options)
        result = load_correlator(path)
        self.assertIsInstance(result, CorrelatorResult)
        self.assertEqual(identity, result.data_id)
        return result

    def test_unreduced_nested_arrays_scalars_keys_empty_and_object_containers(self):
        objects = np.empty((2, 1), dtype=object)
        objects[0, 0] = {'sub': np.array([2-3j, 1+4j], dtype=np.complex64)}
        objects[1, 0] = (np.float32(4), [])
        raw = {('channel', 2): [np.arange(12, dtype='>i2').reshape(3, 4)[:, ::-1],
                (np.array(5, dtype=np.uint8), 2+3j, np.int16(7), True)],
            3: objects, 'empty': np.empty((0, 3)), 'nothing': {}}
        result = self.roundtrip(raw, metadata={'ensemble': 'synthetic', 'cfg': 7}, provenance={'source': 'unit test'})
        loaded = result.value
        self.assertEqual(list(loaded), list(raw))
        self.assertIs(type(loaded[('channel', 2)]), list)
        self.assertIs(type(loaded[('channel', 2)][1]), tuple)
        self.assertIs(type(loaded[('channel', 2)][0]), np.ndarray)
        self.assertEqual(loaded[('channel', 2)][0].dtype.str, '>i2')
        np.testing.assert_array_equal(loaded[('channel', 2)][0], raw[('channel', 2)][0])
        self.assertEqual(loaded[('channel', 2)][1][0].shape, ())
        self.assertIs(type(loaded[('channel', 2)][1][1]), complex)
        self.assertIs(type(loaded[('channel', 2)][1][2]), np.int16)
        self.assertIs(type(loaded[('channel', 2)][1][3]), bool)
        self.assertEqual(loaded[3].shape, (2, 1))
        self.assertEqual(loaded[3].dtype, object)
        np.testing.assert_array_equal(loaded[3][0, 0]['sub'], objects[0, 0]['sub'])
        self.assertIs(type(loaded[3][1, 0]), tuple)
        self.assertEqual(loaded['empty'].shape, (0, 3))
        self.assertEqual(loaded['nothing'], {})
        self.assertIsNone(result.reduction)
        self.assertEqual(result.metadata, {'ensemble': 'synthetic', 'cfg': 7})
        self.assertEqual(result.provenance, {'source': 'unit test'})
        loaded[('channel', 2)][0][0, 0] = 99
        self.assertNotEqual(raw[('channel', 2)][0][0, 0], 99)
        self.assertNotEqual(load_correlator(self.root / 'result').value[('channel', 2)][0][0, 0], 99)

    def test_shared_integer_time_keys_and_heterogeneous_operator_metadata(self):
        raw = {'matrix': np.ones((2, 4, 3, 2), dtype=np.complex128),
               'vector': [np.ones((5, 3, 2), dtype=np.float32)]}
        result = self.roundtrip(raw, axes=(0, 1), coordinates={0: [5, 1, 3], 1: [2, 0]},
            field_axes={('matrix',): ('sink_op', 'source_op', 0, 1)},
            field_coordinates={('matrix',): {'sink_op': ['a', 'b'], 'source_op': ['w', 'x', 'y', 'z']}})
        self.assertEqual(result.axes, (0, 1))
        self.assertEqual(result.coordinates, {0: [5, 1, 3], 1: [2, 0]})
        self.assertEqual(result.field_axes[('matrix',)], ('sink_op', 'source_op', 0, 1))
        self.assertEqual(result.field_axes[('vector', 0)], ('operator_axis_0', 0, 1))
        self.assertEqual(result.field_coordinates[('matrix',)]['sink_op'], ['a', 'b'])
        self.assertEqual(result.value['vector'][0].dtype, np.float32)
        np.testing.assert_array_equal(result.value['matrix'], raw['matrix'])

    def test_absolute_full_times_reference_nonfixed_sink_and_operator_axes(self):
        lt = 5
        sources, sinks = [4, 1, 0, 3, 2], [2, 4, 1, 0, 3]
        values = np.empty((2, 3, lt, lt), dtype=np.complex128)
        for op in np.ndindex((2, 3)):
            for i, source in enumerate(sources):
                for j, sink in enumerate(sinks):
                    values[op+(i, j)] = (source+1)*(sink+2) + 7*op[0] + 3*op[1] + 1j*(2*source-sink)
        result = self.roundtrip(values, axes=('source', 'sink'), coordinates={'source': sources, 'sink': sinks},
            average_tsrc=True, source_axis='source', sink_axis='sink', relative_axis='delta',
            physical_lt=lt, boundary='periodic', coordinate_domain='physical')
        expected = np.empty((2, 3, lt), dtype=np.complex128)
        for op in np.ndindex((2, 3)):
            for dt in range(lt):
                expected[op+(dt,)] = sum(values[op+(sources.index(s), sinks.index((s+dt)%lt))] for s in range(lt))/lt
        np.testing.assert_allclose(result.value, expected)
        self.assertEqual(result.axes, ('delta',))
        self.assertEqual(result.reduction['counts'], [lt]*lt)
        self.assertEqual(result.reduction['mode'], 'absolute')
        self.assertEqual(result.field_axes[()], ('operator_axis_0', 'operator_axis_1', 'delta'))

    def test_partial_absolute_counts_and_explicit_nan_gaps_reference(self):
        # dt=0, 2, 4, 6 only, with unequal numbers of covered contributions.
        sources, sinks, lt = [6, 0], [0, 2, 6], 8
        values = np.array([[4, 10], [20, 31], [43, 57]], dtype=np.float32)
        options = dict(axes=('sink', 'source'), coordinates={'sink': sinks, 'source': sources},
            average_tsrc=True, source_axis='source', sink_axis='sink', physical_lt=lt, boundary='periodic', coordinate_domain='physical')
        with self.assertRaisesRegex(ValueError, 'missing physical dt'):
            save_correlator(self.root / 'reject', values, **options)
        self.assertFalse((self.root / 'reject').exists())
        result = self.roundtrip(values, missing='nan', **options)
        expected, counts = np.full(lt, np.nan, dtype=np.float32), [0]*lt
        for dt in range(lt):
            contributions = [values[i, j] for i, sink in enumerate(sinks) for j, source in enumerate(sources)
                             if (sink-source) % lt == dt]
            counts[dt] = len(contributions)
            if contributions:
                expected[dt] = sum(contributions)/len(contributions)
        np.testing.assert_allclose(result.value, expected, equal_nan=True)
        self.assertEqual(result.value.dtype, np.float32)
        self.assertEqual(result.reduction['counts'], counts)
        self.assertEqual(result.coordinates, {'dt': list(range(lt))})
        self.assertTrue(np.isnan(result.value[1::2]).all())

    def test_relative_axis_only_averages_explicit_source_and_preserves_dt_labels(self):
        values = np.arange(2*3*2, dtype=np.float64).reshape(2, 3, 2)+1j
        result = self.roundtrip(values, axes=(11, 12), coordinates={11: [3, 0, 1], 12: [2, 0]},
            average_tsrc=True, source_axis=11, relative_axis=12, physical_lt=4,
            boundary='periodic', coordinate_domain='physical', missing='nan')
        expected = np.full((2, 4), np.nan, dtype=np.complex128)
        expected[:, [2, 0]] = values.mean(axis=1)
        np.testing.assert_allclose(result.value, expected, equal_nan=True)
        self.assertEqual(result.axes, (12,))
        self.assertEqual(result.reduction['counts'], [3, 0, 3, 0])

    def test_reduced_result_resave_retains_history_but_no_second_average(self):
        result = self.roundtrip(np.arange(4).reshape(2, 2), axes=('s', 'r'), coordinates={'s': [0, 2], 'r': [0, 2]},
            average_tsrc=True, source_axis='s', sink_axis='r', physical_lt=4, boundary='periodic', coordinate_domain='physical', missing='nan')
        path = self.root / 'resaved'
        save_correlator(path, result)
        loaded = load_correlator(path)
        self.assertEqual(result.reduction, loaded.reduction)
        np.testing.assert_allclose(result.value, loaded.value, equal_nan=True)
        with self.assertRaisesRegex(ValueError, 'already reduced'):
            save_correlator(self.root / 'twice', result, average_tsrc=True)
        corrupt = replace(result, value=np.zeros_like(result.value))
        with self.assertRaisesRegex(ValueError, 'NaN gaps'):
            save_correlator(self.root / 'filled', corrupt)

    def test_plain_time_names_never_trigger_reduction(self):
        values = np.arange(6).reshape(2, 3)
        result = self.roundtrip(values, axes=('tsrc', 'dt'), coordinates={'tsrc': [5, 2], 'dt': [0, 1, 2]})
        np.testing.assert_array_equal(result.value, values)
        self.assertIsNone(result.reduction)

    def test_invalid_reduction_semantics_and_time_coverage_reject_before_io(self):
        defaults = dict(axes=('sink', 'source'), coordinates={'sink': [0, 1], 'source': [0, 1]},
            average_tsrc=True, source_axis='source', sink_axis='sink', physical_lt=2, boundary='periodic', coordinate_domain='physical')
        invalid = [dict(coordinate_domain=None), dict(coordinate_domain='slot'), dict(source_axis=None), dict(sink_axis=None), dict(physical_lt=None), dict(physical_lt=True),
                   dict(boundary=None), dict(boundary='open'), dict(boundary='antiperiodic'),
                   dict(source_axis='sink'), dict(relative_axis='source'), dict(missing='zero'),
                   dict(coordinates={'sink': [0, 2], 'source': [0, 1]}),
                   dict(coordinates={'sink': [0., 1.], 'source': [0, 1]}),
                   dict(coordinates={'sink': [0, 0], 'source': [0, 1]})]
        for index, change in enumerate(invalid):
            path = self.root / f'invalid-{index}'
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                save_correlator(path, np.ones((2, 2)), **dict(defaults, **change))
            self.assertFalse(path.exists())
        with self.assertRaisesRegex(ValueError, 'exactly source/sink'):
            save_correlator(self.root / 'extra', np.ones((2, 2, 2)),
                **dict(defaults, axes=('sink', 'current', 'source'),
                       coordinates={'sink': [0, 1], 'current': [0, 1], 'source': [0, 1]}))

    def test_invalid_field_metadata_dtype_cycle_and_unknown_leaf_reject(self):
        invalid = [dict(axes=('t',), coordinates={'t': [0]}),
                   dict(field_axes={(): ('x', 'y')}),
                   dict(field_axes={('missing',): ('x',)}),
                   dict(axes=('t',), coordinates={'t': [0, 1]}, field_coordinates={(): {'t': [1, 0]}}),
                   dict(metadata={1: 'not JSON native'})]
        for index, options in enumerate(invalid):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                save_correlator(self.root / f'field-{index}', np.ones(2), **options)
        loop = []
        loop.append(loop)
        for index, raw in enumerate((loop, {'x': object()}, np.array(['string']))):
            with self.subTest(raw=type(raw)), self.assertRaises((TypeError, ValueError)):
                save_correlator(self.root / f'value-{index}', raw)

    def test_unreduced_nonfinite_repeated_coordinates_and_frozen_metadata_roundtrip(self):
        from types import MappingProxyType
        value = np.array([np.nan, np.inf, -np.inf])
        result = self.roundtrip(value, axes=('t',), coordinates={'t': [1, 1, 0]},
            metadata=MappingProxyType({'nested': MappingProxyType({'labels': ('a', 'b')})}))
        np.testing.assert_array_equal(result.value, value)
        self.assertEqual(result.coordinates['t'], [1, 1, 0])
        self.assertEqual(result.metadata, {'nested': {'labels': ['a', 'b']}})

    def test_empty_tree_and_zero_byte_numeric_arrays_roundtrip(self):
        result = self.roundtrip({'a': [], 'b': (), 'c': np.empty((0, 2), dtype=np.complex64)})
        self.assertEqual(result.value['a'], [])
        self.assertEqual(result.value['b'], ())
        self.assertEqual(result.value['c'].shape, (0, 2))

    def test_published_dataset_immutable_and_incomplete_write_not_loadable(self):
        values = np.arange(6)
        self.roundtrip(values)
        path = self.root / 'result'
        original = (path / 'dataset.json').read_bytes()
        with self.assertRaises(FileExistsError):
            save_correlator(path, values+1)
        self.assertEqual(original, (path / 'dataset.json').read_bytes())
        failed = self.root / 'failed'
        with patch.object(DatasetWriter, 'write_field', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                save_correlator(failed, values)
        self.assertFalse((failed / 'dataset.json').exists())
        with self.assertRaises((ValueError, FileNotFoundError)):
            load_correlator(failed)

    def test_numeric_payload_mutation_and_truncation_detected(self):
        import h5py
        self.roundtrip(np.arange(20, dtype=np.float64))
        path = self.root / 'result'
        with DatasetReader(path) as reader:
            field = next(iter(reader.fields.values()))
            payload = path / reader._storage['path']
            dataset = field['dataset']
        original = payload.read_bytes()
        with h5py.File(payload, 'r+') as stored:
            values = stored[dataset][()]
            values.flat[1] += 1
            stored[dataset][()] = values
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_correlator(path)
        payload.write_bytes(original[:-1])
        with self.assertRaises((ValueError, OSError)):
            load_correlator(path)

    def repackage(self, source, destination, change):
        with DatasetReader(source) as reader:
            document = reader.spec.document
            fields = [(field['path'], np.array(reader.read_field(field['path']), copy=True))
                      for field in document['definition']['fields']]
        change(document['definition'])
        with DatasetWriter(destination, DatasetSpec(document), provenance={},
                           format='hdf5', structured=True) as writer:
            for path, value in fields:
                writer.write_field(path, value)
            writer.publish()

    def test_valid_generic_manifests_with_corrupt_typed_metadata_rejected(self):
        self.roundtrip({'x': np.arange(6).reshape(2, 3)}, axes=(0,), coordinates={0: [2, 0, 1]})
        changes = [lambda d: d['fields'][0].update(path=[{'scalar': 'missing'}]),
                   lambda d: d['fields'][0].update(shape=[6]),
                   lambda d: d['fields'][0].update(dtype='O'),
                   lambda d: d['tree']['children'][0].update(index=99),
                   lambda d: d['tree'].update(kind='unknown'),
                   lambda d: d['fields'][0].update(value_kind='float'),
                   lambda d: d['coordinates'][0].update(values=[1, 0, 2])]
        for index, change in enumerate(changes):
            destination = self.root / f'corrupt-{index}'
            self.repackage(self.root / 'result', destination, change)
            with DatasetReader(destination):
                pass
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_correlator(destination)

    def test_wrong_reduction_counts_and_semantics_rejected_even_in_valid_dataset(self):
        self.roundtrip(np.arange(4).reshape(2, 2), axes=(0, 1), coordinates={0: [0, 1], 1: [0, 1]},
            average_tsrc=True, source_axis=0, sink_axis=1, physical_lt=2, boundary='periodic', coordinate_domain='physical')
        changes = [lambda d: d['reduction'].update(counts=[1, 1]),
                   lambda d: d['reduction'].update(boundary='open'),
                   lambda d: d['reduction'].update(mode='relative')]
        for index, change in enumerate(changes):
            destination = self.root / f'corrupt-mean-{index}'
            self.repackage(self.root / 'result', destination, change)
            with DatasetReader(destination):
                pass
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_correlator(destination)

    def test_native_hdf5_field_preserves_real_shape_and_dtype_without_packing_axes(self):
        values = np.arange(10000, dtype=np.float32).reshape(100, 100)
        self.roundtrip(values)
        with DatasetReader(self.root / 'result') as reader:
            definition = reader.spec.document['definition']
            field = definition['fields'][0]
            stored = reader.read_field(field['path'])
            self.assertEqual(reader.spec.shape, ())
            self.assertIsNone(reader.spec.document['dtype'])
            self.assertEqual(field['shape'], [100, 100])
            self.assertEqual(field['dtype'], values.dtype.str)
            self.assertEqual(stored.shape, values.shape)
            self.assertEqual(stored.dtype.str, values.dtype.str)
            self.assertEqual(set(field), {'path', 'shape', 'dtype', 'value_kind', 'axes', 'coordinates', 'sha256'})


if __name__ == '__main__':
    unittest.main()


def test_source_average_explicit_dt_subset_roundtrip(tmp_path):
    from lqcdweave.contraction.correlator import save_correlator, load_correlator
    value=np.array([[10.,20.]])
    save_correlator(tmp_path,value,configuration="subset",axes=("source","sink"),
        coordinates={"source":[0],"sink":[1,2]},average_tsrc=True,
        source_axis="source",sink_axis="sink",physical_lt=4,boundary="periodic",
        coordinate_domain="physical",dt_coordinates=[2,1])
    saved=load_correlator(tmp_path,configuration="subset")
    np.testing.assert_array_equal(saved.value,[20.,10.])
    assert saved.coordinates=={"dt":[2,1]}
    assert saved.reduction["counts"]==[0,1,1,0]
