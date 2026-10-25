"""Whole-cfg storage, shared definitions, native fields and interrupted production."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, inspect_dataset
from lqcdweave.storage.registration import read_registration


def spec(cfg, *, kind='test_array'):
    return DatasetSpec(dict(schema=1, kind=kind,
        configuration=dict(ensemble='test', configuration=str(cfg), lattice=[1, 1, 1, 2]),
        axes=[dict(name='time', coordinates=[1, 0]), dict(name='mode', coordinates=[2, 5, 7])],
        dtype='complex128', bases=dict(mode=dict(data_id='basis-'+str(cfg), indices=[2, 5, 7])),
        definition=dict(recipe='same scientific definition'), conventions={}, parents=['basis-'+str(cfg)]))


class DatasetFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'data'
        self.addCleanup(self.temporary.cleanup)

    def write(self, cfg, *, format='npy'):
        values = np.arange(6).reshape(2, 3).astype(np.complex128) + int(cfg)*1j
        with DatasetWriter(self.root, spec(cfg), provenance={}, format=format) as writer:
            for row in writer.missing:
                writer.write(row, values[row])
            identity = writer.publish()
        return identity, values

    def test_two_cfgs_share_definition_and_have_whole_npy_files(self):
        first, a = self.write(2000)
        second, b = self.write(2010)
        self.assertNotEqual(first, second)
        self.assertEqual({p.name for p in self.root.iterdir()}, {'dataset.json', '2000.npy', '2010.npy'})
        document = json.loads((self.root/'dataset.json').read_text())
        self.assertEqual(document['common']['identity']['spec']['definition'], spec(2000).document['definition'])
        records = read_registration(self.root)
        self.assertEqual(records['2000']['identity']['spec'], spec(2000).document)
        self.assertEqual(records['2010']['identity']['spec'], spec(2010).document)
        with DatasetReader(self.root, cfg='2010') as reader:
            np.testing.assert_array_equal(reader.read(time=1, select={'mode': [7, 2]}), b[0, [2, 0]])
            self.assertEqual(reader.data_id, second)
        np.testing.assert_array_equal(np.load(self.root/'2000.npy'), a)

    def test_interrupted_cfg_resumes_in_one_file_without_published_partial_data(self):
        values = np.arange(6).reshape(2, 3).astype(np.complex128)
        with DatasetWriter(self.root, spec(2000), provenance={}) as writer:
            writer.write(0, values[0])
        self.assertEqual(inspect_dataset(self.root, cfg='2000')['completed'], (0,))
        self.assertFalse((self.root/'dataset.json').exists())
        self.assertFalse((self.root/'2000.npy').exists())
        with DatasetWriter(self.root, spec(2000), provenance={}, resume=True) as writer:
            self.assertEqual(writer.missing, (1,))
            writer.write(1, values[1]); writer.publish()
        np.testing.assert_array_equal(np.load(self.root/'2000.npy'), values)
        self.assertFalse((self.root/'.state').exists())

    def test_independent_cfg_publications_do_not_lose_registration_entries(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(self.write, (2000, 2010)))
        self.assertEqual(set(read_registration(self.root)), {'2000', '2010'})

    def test_hdf5_single_array_has_same_selection_and_identity_contract(self):
        identity, values = self.write(2000, format='hdf5')
        with DatasetReader(self.root, cfg='2000') as reader:
            selected = reader.read(time=0, select={'mode': [7, 2]})
            np.testing.assert_array_equal(selected, values[1, [2, 0]])
            self.assertTrue(selected.flags.owndata)
            self.assertEqual(reader.data_id, identity)

    def test_named_fields_preserve_actual_correlator_shapes_and_complex_bits(self):
        first_path = ['I0', 'A1_Pp_Cp_Rp', 'row_0', 'two_point']
        vacuum_path = ['I0', 'A1_Pp_Cp_Rp', 'row_0', 'vacuum_sink']
        values = np.zeros((9, 9, 10, 10, 127), dtype=np.complex128)
        values[0, 0, 0, 0, 0] = 4679.063032856879 + 1.468121628714796e-6j
        values[1, 2, 0, 0, 0] = complex(np.nan, -0.)
        vacuum = np.full((9, 10, 40), -68.45950644680936-2.6861291276192915e-8j, dtype=np.complex128)
        with DatasetWriter(self.root, spec(2000, kind='correlator'), provenance={},
                           format='hdf5', structured=True) as writer:
            writer.write_field(first_path, values)
            writer.write_field(vacuum_path, vacuum)
            writer.publish()
        with DatasetReader(self.root, cfg='2000') as reader:
            actual = reader.read_field(first_path)
            self.assertEqual(actual.shape, values.shape)
            self.assertEqual(actual.dtype, values.dtype)
            np.testing.assert_array_equal(actual.reshape(-1).view(np.uint8), values.reshape(-1).view(np.uint8))
            np.testing.assert_array_equal(reader.read_field(vacuum_path), vacuum)
            small = reader.read_field(first_path, selection=(1, 2, slice(None), slice(None), slice(None)))
            np.testing.assert_array_equal(small.reshape(-1).view(np.uint8), values[1, 2].reshape(-1).view(np.uint8))
