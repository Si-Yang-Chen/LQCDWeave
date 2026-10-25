"""Pure scientific gates for registered full-time Diagram output requests."""

from copy import deepcopy
import unittest

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.contracts.correlator.api import TensorDescription
from lqcdweave.contracts.time_selection import TimeSelectionRecord, bind_time_selections, dataset_time_selection
from lqcdweave.planning.diagram_measurement import (
    diagram_measurement_document, validate_diagram_endpoint_descriptions,
    validate_diagram_input_specs,
)


class DiagramMeasurementPlanningTests(unittest.TestCase):
    def propagator_document(self, *, mass=.4, gauge='dirac-links'):
        return dict(schema=1, kind='vsv',
            configuration=dict(ensemble='ensemble', configuration='cfg', lattice=[4, 1, 1, 3]),
            axes=[dict(name=name, coordinates=coordinates) for name, coordinates in (
                ('source_time', [2, 0, 1]), ('sink_time', [0, 1, 2]),
                ('sink_spin', [0, 1, 2, 3]), ('source_spin', [0, 1, 2, 3]),
                ('sink_mode', [8, 3]), ('source_mode', [8, 3]))],
            dtype='complex128', bases={axis: dict(data_id='actual-basis', indices=[8, 3])
                for axis in ('sink_mode', 'source_mode')}, parents=['actual-basis', gauge],
            definition=dict(gauge_id=gauge, dirac=dict(action='Wilson', mass=mass),
                            solver_tolerance=1e-11),
            conventions=dict(normalization='unscaled', spin_basis='degrand_rossi',
                             temporal_boundary='antiperiodic'))

    def test_full_times_and_different_flavor_masses_do_not_force_equal_gauges(self):
        forward = DatasetSpec(self.propagator_document())
        reverse = DatasetSpec(self.propagator_document(mass=.9, gauge='other-dirac-links'))
        configuration, times = validate_diagram_input_specs(dict(forward=forward, reverse=reverse))
        self.assertEqual(times, (0, 1, 2))
        self.assertEqual(configuration, forward.document['configuration'])
        self.assertEqual(forward.coordinates('source_time'), [2, 0, 1])

    def test_unknown_scientific_declarations_are_refused(self):
        changes = (
            ('ensemble', lambda d: d['configuration'].update(ensemble=None)),
            ('configuration', lambda d: d['configuration'].update(configuration=None)),
            ('gauge', lambda d: d['definition'].update(gauge_id=None)),
            ('basis', lambda d: d['bases']['sink_mode'].update(data_id=None)),
            ('Dirac action', lambda d: d['definition']['dirac'].update(action=None)),
            ('Dirac mass', lambda d: d['definition']['dirac'].update(mass=None)),
            ('solver tolerance', lambda d: d['definition'].update(solver_tolerance=None)),
            ('spin', lambda d: d['conventions'].update(spin_basis='other')),
            ('temporal boundary', lambda d: d['conventions'].update(temporal_boundary=None)),
        )
        for label, change in changes:
            with self.subTest(declaration=label):
                document = self.propagator_document()
                change(document)
                with self.assertRaises(ValueError):
                    validate_diagram_input_specs(dict(line=DatasetSpec(document)))

    def test_missing_time_and_configuration_conflicts_are_refused(self):
        incomplete = self.propagator_document()
        incomplete['axes'][0]['coordinates'] = [0, 2]
        with self.assertRaisesRegex(ValueError, 'all physical times'):
            validate_diagram_input_specs(dict(line=DatasetSpec(incomplete)))
        changed = self.propagator_document()
        changed['configuration']['configuration'] = 'other-cfg'
        with self.assertRaisesRegex(ValueError, 'configuration or lattice'):
            validate_diagram_input_specs(dict(line=DatasetSpec(self.propagator_document()),
                                              other=DatasetSpec(changed)))

    def test_spin_coordinate_types_cannot_masquerade_as_integer_labels(self):
        for labels in ([False, True, 2, 3], [0.0, 1.0, 2.0, 3.0]):
            with self.subTest(labels=labels):
                document = self.propagator_document()
                document['axes'][2]['coordinates'] = labels
                with self.assertRaisesRegex(ValueError, 'spin axes'):
                    validate_diagram_input_specs(dict(line=DatasetSpec(document)))

    def test_sparse_point_rows_keep_the_declared_same_actual_pool(self):
        cfg = self.propagator_document()['configuration']
        points = DatasetSpec(dict(schema=1, kind='spatial_points', configuration=cfg,
            axes=[dict(name='time', coordinates=[2, 0, 1]),
                  dict(name='point', coordinates=[9, 3, 14, 8]),
                  dict(name='xyz', coordinates=['x', 'y', 'z'])],
            dtype='int64', bases={}, parents=[], definition={},
            conventions=dict(representation='coordinate_delta', normalization='unit_delta')))
        document = dict(schema=1, kind='elemental_p2p_row_sparse', configuration=cfg,
            axes=[dict(name='time', coordinates=[0, 1, 2]),
                  dict(name='component', coordinates=[dict(path=[0])]),
                  dict(name='momentum', coordinates=[[1, 0, 0]]),
                  dict(name='point_left', coordinates=[9, 3, 14, 8]),
                  dict(name='color_left', coordinates=[0, 1, 2]),
                  dict(name='color_right', coordinates=[0, 1, 2])],
            dtype='complex128', bases=dict(point_left=dict(data_id='actual-points',
                indices=[9, 3, 14, 8])), parents=['elemental-links', 'actual-points'],
            definition=dict(gauge_id='elemental-links', path_boundary='periodic',
                point_right_id='actual-points', point_right_selection='same_ordered_prefix_as_point_left'),
            conventions=dict(normalization='unscaled', phase='anchor',
                point_pair_layout='left-point-row-slots-v1'))
        self.assertEqual(validate_diagram_input_specs(
            dict(points=points, p2p=DatasetSpec(document)))[1], (0, 1, 2))
        for labels in ([False, True, 2], [0.0, 1.0, 2.0]):
            with self.subTest(color_labels=labels):
                wrong = deepcopy(document)
                wrong['axes'][4]['coordinates'] = labels
                with self.assertRaisesRegex(ValueError, 'color axes'):
                    validate_diagram_input_specs(dict(p2p=DatasetSpec(wrong)))
        document['definition']['point_right_id'] = 'different-actual-points'
        with self.assertRaisesRegex(ValueError, 'same-pool'):
            validate_diagram_input_specs(dict(p2p=DatasetSpec(document)))

    def description(self, *, status='verified', knowledge='both-bound',
                    propagator=('actual-basis', (8, 3)), meson=('actual-basis', (8, 3))):
        checks = {side: dict(status=status, knowledge=knowledge,
                            propagator=propagator, meson=meson)
                  for side in ('source', 'sink')}
        return TensorDescription(role='propagator', bindings=dict(endpoint_axis_bindings=checks))

    def test_persistent_gate_refuses_unknown_or_missing_endpoints(self):
        validate_diagram_endpoint_descriptions((self.description(),))
        for knowledge in ('one-unbound', 'both-unbound'):
            with self.subTest(knowledge=knowledge), self.assertRaisesRegex(ValueError, 'unknown'):
                validate_diagram_endpoint_descriptions((self.description(
                    status='unknown', knowledge=knowledge, meson=None),))
        with self.assertRaisesRegex(ValueError, 'both endpoint'):
            validate_diagram_endpoint_descriptions((TensorDescription(role='propagator'),))
        with self.assertRaisesRegex(ValueError, 'at least one'):
            validate_diagram_endpoint_descriptions((TensorDescription(role='elemental'),))

    def test_endpoint_gate_checks_actual_ids_order_and_coordinate_types(self):
        mismatches = (('other-basis', (8, 3)), ('actual-basis', (3, 8)),
                      ('actual-basis', (8.0, 3)))
        for binding in mismatches:
            with self.subTest(binding=binding), self.assertRaisesRegex(ValueError, 'mismatch'):
                validate_diagram_endpoint_descriptions((self.description(meson=binding),))

    def test_output_snapshots_detach_and_resolve_through_existing_requests(self):
        spec = DatasetSpec(self.propagator_document())
        cfg = spec.document['configuration']
        references = dict(line=dict(request='line'))
        science = dict(operator=dict(coefficient=[.6, .4]), source_dagger=True,
                       sampling=dict(pool_id='explicit-pool', population=4, sample=4))
        document = diagram_measurement_document(cfg, (0, 1, 2), dict(line=spec), references, science)
        request = DatasetRequest(document)
        self.assertEqual(request.dependencies, ('line',))
        actual = request.resolve(dict(line='actual-propagator'))
        self.assertEqual(actual.axis_names, ('source_time', 'sink_time'))
        self.assertEqual(actual.shape, (3, 3))
        self.assertEqual(actual.document['parents'], ['actual-propagator'])
        self.assertEqual(actual.document['definition']['input_roles'], dict(line='actual-propagator'))
        science['operator']['coefficient'][0] = 99
        references['line']['request'] = 'other'
        cfg['configuration'] = 'changed'
        self.assertEqual(document['definition']['science']['operator']['coefficient'], [.6, .4])
        self.assertEqual(document['definition']['input_roles'], dict(line=dict(request='line')))
        self.assertEqual(document['configuration']['configuration'], 'cfg')
        changed = deepcopy(document)
        changed['definition']['science']['operator']['coefficient'][0] = .8
        self.assertNotEqual(DatasetRequest(changed).resolve(dict(line='actual-propagator')).definition_id,
                            actual.definition_id)
        with self.assertRaisesRegex(ValueError, 'all physical times'):
            diagram_measurement_document(spec.document['configuration'], (0, 2),
                dict(line=spec), dict(line='actual-propagator'), {})
        with self.assertRaisesRegex(ValueError, 'actual input identity'):
            diagram_measurement_document(spec.document['configuration'], (0, 1, 2),
                dict(line=spec), dict(line=None), {})


    def test_selected_slot_contract_preserves_order_and_rejects_mixed_domains(self):
        selection=TimeSelectionRecord(3,(2,0))
        def selected(source=selection,sink=selection):
            document=self.propagator_document()
            document['axes'][0]['coordinates']=list(source.times)
            document['axes'][1]['coordinates']=list(sink.times)
            return DatasetSpec(bind_time_selections(document,{'source_time':source,'sink_time':sink}))
        spec=selected()
        configuration,times=validate_diagram_input_specs({'line':spec})
        self.assertEqual(times,(0,1))
        document=diagram_measurement_document(configuration,times,{'line':spec},{'line':'actual'}, {})
        result=DatasetSpec(document)
        self.assertEqual(result.shape,(2,2))
        self.assertEqual(result.document['configuration']['lattice'][3],3)
        self.assertEqual(dataset_time_selection(result,'source_time'),selection)
        self.assertEqual(result.document['conventions']['time_coverage'],'selected_physical')
        invalid=[{'line':spec,'other':DatasetSpec(self.propagator_document())},
                 {'line':spec,'other':selected(TimeSelectionRecord(3,(0,2)))},
                 {'line':selected(sink=TimeSelectionRecord(3,(1,)))}]
        for views in invalid:
            with self.subTest(views=tuple(views)),self.assertRaisesRegex(ValueError,'identical ordered'):
                validate_diagram_input_specs(views)
            with self.assertRaisesRegex(ValueError,'identical ordered'):
                diagram_measurement_document(configuration,times,views,{role:'actual' for role in views},{})


if __name__ == '__main__':
    unittest.main()
