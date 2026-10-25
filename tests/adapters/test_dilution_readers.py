"""Actual V/D/P endpoints, persistence and six-representative PR consumption."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.adapters.propagator import ArrayPropagator, PropagatorPairReader
from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.contracts.dilution import DILUTION_WEIGHT_SCHEME, DilutionBasis, DilutionGroup
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS, elemental_tail_axes
from lqcdweave.production.perambulator.api import SAVED_PRODUCTS, PropagatorReader
from lqcdweave.planning.production import (
    elemental_dataset_document, point_elemental_dataset_document,
)
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter
from lqcdweave.runtime.datasets import DatasetView


def source(data_id='actual-d', pool='random-A'):
    return DilutionBasis(data_id, 'actual-parent', (
        DilutionGroup((2, 3), ((0, 0),)),
        DilutionGroup((4, 5, 6), ((1, 0), (1, 1))),
    ), ((0, pool), (1, pool + '-1'), (2, pool + '-2')))


def dilution_dataset_document(*, weighted, include_scheme):
    conventions = dict(phase='anchor', normalization='unscaled',
        elemental_weighted=weighted, dilution_endpoints={'field': source().document})
    if include_scheme:
        conventions['dilution_weight_scheme'] = DILUTION_WEIGHT_SCHEME
    return dict(schema=1, kind='elemental_vv',
        configuration=dict(ensemble='toy', configuration='cfg', lattice=[2, 1, 1, 2]),
        axes=[dict(name='time', coordinates=[0, 1]),
              dict(name='component', coordinates=[{'path': []}]),
              dict(name='momentum', coordinates=[[0, 0, 0]]),
              dict(name='sink_mode', coordinates=[11, 5]),
              dict(name='source_mode', coordinates=[0, 1, 2])],
        dtype='complex128', bases={
            'sink_mode': dict(data_id='actual-v', indices=[11, 5]),
            'source_mode': dict(data_id='actual-d', indices=[0, 1, 2])},
        definition=dict(gauge_id='actual-gauge', logical_sector='v2d', path_boundary='periodic'),
        conventions=conventions,
        parents=['actual-gauge', 'actual-v', 'actual-d'])


def publish_elemental(path, document):
    spec = DatasetSpec(document)
    values = np.zeros(spec.shape, dtype=complex)
    with DatasetWriter(path, spec, provenance={'test': 'dilution weight scheme'}, format='npy') as writer:
        for time in range(len(spec.coordinates('time'))):
            writer.write(time, values[time])
        writer.publish()
    return DatasetReader(path)


class DilutionElementalReaderTests(unittest.TestCase):
    def test_nine_sectors_retain_distinct_endpoints_and_order(self):
        rng = np.random.default_rng(401)
        vectors = {'v': rng.normal(size=(2, 6)) + 1j * rng.normal(size=(2, 6)),
                   'd': rng.normal(size=(3, 6)) + 1j * rng.normal(size=(3, 6)),
                   'p': np.eye(6)}
        tails = {'v': (2,), 'd': (3,), 'p': (2, 3)}
        arrays, endpoints, dilution = {}, {}, {}
        for sector in ELEMENTAL_SECTORS:
            left, right = sector[0], sector[2]
            gram = vectors[left].conj() @ vectors[right].T
            arrays[sector] = gram.reshape((1, 1, 1, *tails[left], *tails[right]))
            for side, kind in (('bar', left), ('field', right)):
                labels = (11, 5) if kind in ('v', 'p') else (0, 1, 2)
                endpoints[sector, side] = ('actual-' + kind, labels)
                if kind == 'd':
                    dilution[sector, side] = source()
        component = ElementalComponent.path(())
        reader = ArrayElemental(arrays, (component,), phase_convention='anchor',
                                basis_endpoints=endpoints, dilution_endpoints=dilution)
        self.assertEqual(reader.dilution_weight_scheme, DILUTION_WEIGHT_SCHEME)
        for sector in ELEMENTAL_SECTORS:
            with self.subTest(sector=sector):
                metadata = reader.describe_block(component, 0, 0, sector)
                self.assertEqual(metadata.axes, elemental_tail_axes(sector))
                self.assertEqual(metadata.structure, 'general')
                np.testing.assert_array_equal(reader.block(component, 0, 0, sector), arrays[sector][0, 0, 0])
                self.assertEqual(reader.dilution_basis(component, 0, 0, sector, 'bar'),
                                 dilution.get((sector, 'bar')))
        selected = reader.describe_block(component, 0, 0, 'v2d',
                                         mode_indices={'mode_left': (1, 0), 'mode_right': (2, 1)})
        self.assertEqual(selected.axis_bindings['mode_left'], ('actual-v', (5, 11)))
        self.assertEqual(selected.axis_bindings['mode_right'], ('actual-d', (2, 1)))
        previous = reader.dilution_identity
        self.assertEqual(previous, canonical_json({
            'weight_scheme': DILUTION_WEIGHT_SCHEME,
            'endpoints': [[sector, side, value.document]
                          for (sector, side), value in sorted(dilution.items())],
        }))
        reader.dilution_endpoints['v2d', 'field'] = source(pool='changed-pool')
        self.assertNotEqual(previous, reader.dilution_identity)
        unproven = ArrayElemental({'d2d': arrays['d2d']}, (component,), phase_convention='anchor')
        self.assertEqual(unproven.dilution_weight_scheme, DILUTION_WEIGHT_SCHEME)
        self.assertIsNone(unproven.dilution_identity)

    def test_cross_actual_bases_never_keep_diagonal_proof(self):
        component = ElementalComponent.path(())
        values = np.eye(3, dtype=complex)[None, None, None]
        reader = ArrayElemental({'d2d': values}, (component,), phase_convention='anchor',
            elemental_structures={('d2d', component, 0): 'diagonal'},
            dilution_endpoints={('d2d', 'bar'): source(), ('d2d', 'field'): source('other-d')})
        self.assertEqual(reader.describe_block(component, 0, 0, 'd2d').structure, 'general')
        with self.assertRaisesRegex(ValueError, 'data_id'):
            ArrayElemental({'d2d': values}, (component,), phase_convention='anchor',
                basis_endpoints={('d2d', 'bar'): ('wrong-d', (0, 1, 2))},
                dilution_endpoints={('d2d', 'bar'): source()})

    def test_registered_dilution_sources_survive_view_reorder_and_persistence(self):
        document = dilution_dataset_document(weighted=True, include_scheme=True)
        spec = DatasetSpec(document)
        values = np.arange(np.prod(spec.shape)).reshape(spec.shape).astype(complex)
        with TemporaryDirectory() as temporary:
            with DatasetWriter(temporary, spec, provenance={'test': 'actual-D'}, format='npy') as writer:
                for time in range(2):
                    writer.write(time, values[time])
                writer.publish()
            with DatasetReader(temporary) as stored:
                reader = DatasetElemental({'v2d': stored})
                component = reader.components[0]
                self.assertTrue(reader.elemental_weighted)
                self.assertEqual(reader.dilution_weight_scheme, DILUTION_WEIGHT_SCHEME)
                self.assertEqual(reader.dilution_basis(component, 0, 1, 'v2d', 'field'), source())
                self.assertEqual(reader.dilution_identity, canonical_json({
                    'weight_scheme': DILUTION_WEIGHT_SCHEME,
                    'endpoints': [['v2d', 'field', source().document]],
                }))
                np.testing.assert_array_equal(reader.block(component, 0, 1, 'v2d'), values[1, 0, 0])
                view_document = spec.document
                for axis in view_document['axes']:
                    if axis['name'] == 'source_mode':
                        axis['coordinates'] = [2, 1]
                view_document['bases']['source_mode']['indices'] = [2, 1]
                view = DatasetView(stored, DatasetSpec(view_document), subset_axes=('source_mode',))
                subset = DatasetElemental({'v2d': view})
                metadata = subset.describe_block(component, 0, 0, 'v2d')
                self.assertEqual(metadata.axis_bindings['source_mode'], ('actual-d', (2, 1)))
                self.assertEqual(subset.dilution_basis(component, 0, 0, 'v2d', 'field'), source())
                np.testing.assert_array_equal(subset.block(component, 0, 0, 'v2d'), values[0, 0, 0][:, [2, 1]])
                with self.assertRaisesRegex(ValueError, 'logical sector'):
                    DatasetElemental({'v2v': stored})

    def test_raw_legacy_d_reads_as_current_scheme_but_weighted_legacy_is_rejected(self):
        raw_identity = None
        for weighted, include_scheme in ((False, False), (False, True), (True, True)):
            with self.subTest(weighted=weighted, include_scheme=include_scheme), TemporaryDirectory() as temporary:
                stored = publish_elemental(temporary, dilution_dataset_document(
                    weighted=weighted, include_scheme=include_scheme))
                try:
                    with patch.object(stored, 'read', side_effect=AssertionError('numeric read during metadata binding')):
                        reader = DatasetElemental({'v2d': stored})
                    component = reader.components[0]
                    self.assertEqual(reader.dilution_weight_scheme, DILUTION_WEIGHT_SCHEME)
                    self.assertEqual(reader.dilution_basis(component, 0, 0, 'v2d', 'field'), source())
                    expected = canonical_json({
                        'weight_scheme': DILUTION_WEIGHT_SCHEME,
                        'endpoints': [['v2d', 'field', source().document]],
                    })
                    self.assertEqual(reader.dilution_identity, expected)
                    if not weighted:
                        if raw_identity is None:
                            raw_identity = reader.dilution_identity
                        else:
                            self.assertEqual(reader.dilution_identity, raw_identity)
                finally:
                    stored.close()
        with TemporaryDirectory() as temporary:
            stored = publish_elemental(temporary, dilution_dataset_document(
                weighted=True, include_scheme=False))
            try:
                with patch.object(stored, 'read', side_effect=AssertionError('numeric read during rejection')):
                    with self.assertRaisesRegex(ValueError, 'explicit dilution_weight_scheme'):
                        DatasetElemental({'v2d': stored})
            finally:
                stored.close()

    def test_production_planners_write_scheme_for_mode_and_point_d_sectors(self):
        configuration = dict(ensemble='toy', configuration='cfg', lattice=[2, 1, 1, 2])

        def spec(kind, axes, conventions, dtype='complex128'):
            return DatasetSpec(dict(schema=1, kind=kind, configuration=configuration,
                axes=[dict(name=name, coordinates=list(values)) for name, values in axes],
                dtype=dtype, bases={}, parents=[], definition={}, conventions=conventions))

        gauge = spec('gauge', (('time', range(2)), ('z', range(1)), ('y', range(1)),
            ('x', range(2)), ('direction', range(4)), ('color_row', range(3)),
            ('color_column', range(3))), {})
        d_basis = spec('explicit_basis', (('time', range(2)), ('mode', range(3)),
            ('z', range(1)), ('y', range(1)), ('x', range(2)), ('color', range(3))),
            dict(basis_orthogonal=True, dilution_source=source().document))
        mode_document = elemental_dataset_document(gauge, d_basis, gauge_id='actual-gauge',
            basis_id='actual-d', components=[dict(path=[])], momenta=((0, 0, 0),),
            modes=(0, 1, 2), phase_convention='anchor', product='d2d')
        self.assertEqual(mode_document['conventions']['dilution_weight_scheme'], DILUTION_WEIGHT_SCHEME)

        points = spec('spatial_points', (('time', range(2)), ('point', range(1)),
            ('xyz', ('x', 'y', 'z'))),
            dict(representation='coordinate_delta', normalization='unit_delta'), 'int64')
        for product, side in (('d2p', 'bar'), ('p2d', 'field')):
            with self.subTest(product=product):
                document = point_elemental_dataset_document(gauge, d_basis, points,
                    product=product, gauge_id='actual-gauge', basis_id='actual-d',
                    points_id='actual-points', components=[dict(path=[])],
                    momenta=((0, 0, 0),), modes=(0, 1, 2), point_labels=(0,))
                self.assertEqual(document['conventions']['dilution_weight_scheme'], DILUTION_WEIGHT_SCHEME)
                self.assertEqual(set(document['conventions']['dilution_endpoints']), {side})


class DilutionPropagatorReaderTests(unittest.TestCase):
    def test_dense_arrays_share_mode_selection_with_actual_d_identity(self):
        values = np.arange(2 * 2 * 4 * 4 * 2 * 3).reshape(2, 2, 4, 4, 2, 3).astype(complex)
        reader = ArrayPropagator({'vsd': values}, time_convention='absolute',
            basis_endpoints={('vsd', 'sink'): ('actual-v', (11, 5)),
                             ('vsd', 'source'): ('actual-d', (0, 1, 2))})
        metadata = reader.describe_block(1, 0, 'vsd', mode_indices={'source_mode': (2, 1)})
        self.assertEqual(metadata.axis_bindings['source_mode'], ('actual-d', (2, 1)))
        np.testing.assert_array_equal(reader.block(1, 0, 'vsd', mode_indices={'source_mode': (2, 1)}),
                                      values[0, 1][..., [2, 1]])

    def test_six_saved_products_feed_nine_correlator_sectors_without_numeric_metadata_reads(self):
        from production.perambulator.test_dilution import DilutionTests
        fixture = DilutionTests('test_six_saved_products_read_nine_and_noncanonical_writes_swap_real_times')
        fixture.setUp()
        pairs = ((2, 0), (0, 2), (0, 0), (2, 2))
        with TemporaryDirectory() as temporary:
            stored = []
            try:
                for product in SAVED_PRODUCTS:
                    path = Path(temporary) / product
                    with DatasetWriter(path, fixture.spec(product, pairs),
                                       provenance={'fixture': 'dense'}, format='npy') as writer:
                        for index, pair in enumerate(pairs):
                            writer.write(index, fixture.oracle(product, *pair))
                        writer.publish()
                    stored.append(DatasetReader(path))
                reader = PropagatorPairReader(PropagatorReader(stored))
                for sector in reader.sectors:
                    with self.subTest(sector=sector):
                        for dataset in stored:
                            dataset.original_read, dataset.read = dataset.read, lambda *_a, **_kw: self.fail('metadata read numeric values')
                        metadata = reader.describe_block(0, 2, sector)
                        for dataset in stored:
                            dataset.read = dataset.original_read
                        np.testing.assert_allclose(reader.block(0, 2, sector), fixture.oracle(sector.upper(), 0, 2), atol=4e-14)
                        sink, source_endpoint = reader.endpoints[sector]
                        for side, endpoint in (('sink', sink), ('source', source_endpoint)):
                            axis = side + ('_point' if endpoint.kind == 'P' else '_mode')
                            self.assertEqual(metadata.axis_bindings[axis], (endpoint.data_id, endpoint.indices))
                self.assertEqual(len(reader.sectors), 9)
                selected = reader.block(0, 2, 'dsd', mode_indices={'sink_mode': (2, 0), 'source_mode': (1,)})
                expected = fixture.oracle('DSD', 0, 2).take((2, 0), axis=2).take((1,), axis=3)
                np.testing.assert_allclose(selected, expected, atol=4e-14)
            finally:
                for dataset in stored:
                    dataset.close()


if __name__ == '__main__':
    unittest.main()
