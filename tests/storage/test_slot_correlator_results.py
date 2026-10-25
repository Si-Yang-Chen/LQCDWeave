"""Persist compact coordinates and reconstruct through the existing serializer."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.storage.correlator_results import save_correlator, load_correlator


class SlotResultTests(unittest.TestCase):
    def test_raw_and_reduced_nested_roundtrip_with_ordered_maps(self):
        record = TimeSelectionRecord(8, (6, 0, 1, 2))
        raw = np.arange(32).reshape(2, 4, 4).astype(complex) + 2j
        value = {'matrix': [raw, (2*raw,)]}
        opts = dict(axes=('source', 'sink'), coordinates={'source': range(4), 'sink': range(4)},
                    coordinate_domain='slot', selection=record, source_axis='source', sink_axis='sink', boundary='periodic')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_correlator(root/'raw', value, **opts)
            loaded = load_correlator(root/'raw')
            np.testing.assert_array_equal(loaded.value['matrix'][0], raw)
            self.assertEqual(loaded.metadata['time_declaration']['selections']['source'], record.document)
            save_correlator(root/'mean', loaded, average_tsrc=True, **opts)
            reduced = load_correlator(root/'mean')
            expected = np.empty((2, 8), complex)
            counts = []
            for dt in range(8):
                pairs = [(i, j) for i, s in enumerate(record.times) for j, t in enumerate(record.times) if (t-s)%8==dt]
                counts.append(len(pairs))
                expected[:, dt] = sum(raw[:, i, j] for i, j in pairs)/len(pairs)
            np.testing.assert_allclose(reduced.value['matrix'][0], expected)
            np.testing.assert_allclose(reduced.value['matrix'][1][0], 2*expected)
            self.assertEqual(reduced.field_axes[('matrix', 0)], ('operator_axis_0', 'dt'))
            self.assertEqual(reduced.reduction['counts'], counts)
            save_correlator(root/'copy', reduced)
            np.testing.assert_allclose(load_correlator(root/'copy').value['matrix'][0], expected)
            with self.assertRaisesRegex(ValueError, 'already reduced'):
                save_correlator(root/'twice', reduced, average_tsrc=True, **opts)

    def test_asymmetric_missing_nan_cyclic_and_invalid_metadata(self):
        source, sink = TimeSelectionRecord(8, (6, 0)), TimeSelectionRecord(8, (0, 2, 6))
        raw = np.arange(6).reshape(2, 3)
        opts = dict(axes=(0, 1), coordinates={0: [0, 1], 1: [0, 1, 2]},
                    source_axis=0, sink_axis=1, boundary='periodic', coordinate_domain='slot',
                    source_selection=source, sink_selection=sink, average_tsrc=True)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'missing physical dt'):
                save_correlator(root/'missing', raw, **opts)
            save_correlator(root/'nan', raw, missing='nan', **opts)
            result = load_correlator(root/'nan')
            self.assertEqual(sum(result.reduction['counts']), 6)
            self.assertTrue(np.isnan(result.value[1::2]).all())
            for changes in ({'physical_lt': 7}, {'coordinate_domain': 'physical'}, {'time_layout': 'cyclic_difference'},
                            {'coordinates': {0: [0, 2], 1: [0, 1, 2]}}):
                with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                    save_correlator(root/'invalid', raw, **dict(opts, missing='nan', **changes))

    def test_explicit_cyclic_save_matches_absolute_and_preserves_pair_counts(self):
        record = TimeSelectionRecord(8, (6, 0, 1, 2))
        absolute = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        cyclic = np.stack([absolute[:, i, (i+np.arange(4))%4] for i in range(4)], axis=1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            common = dict(average_tsrc=True, source_axis='source', boundary='periodic',
                          coordinate_domain='slot', selection=record)
            save_correlator(root/'absolute', absolute, axes=('source', 'sink'),
                coordinates={'source': range(4), 'sink': range(4)}, sink_axis='sink', **common)
            save_correlator(root/'cyclic', cyclic, axes=('source', 'difference'),
                coordinates={'source': range(4), 'difference': range(4)}, relative_axis='difference',
                time_layout='cyclic_difference', **common)
            left, right = load_correlator(root/'absolute'), load_correlator(root/'cyclic')
            np.testing.assert_array_equal(left.value, right.value)
            self.assertEqual(left.reduction['counts'], right.reduction['counts'])
            self.assertEqual(right.value.dtype, np.float32)
