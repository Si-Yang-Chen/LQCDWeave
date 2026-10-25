"""Pure registration gates retain both actual ends of rectangular P/P rows."""

from copy import deepcopy
import unittest

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.diagram_measurement import (
    diagram_measurement_document, validate_diagram_input_specs,
)


class IndependentPointDiagramPlanningTests(unittest.TestCase):
    def document(self):
        return dict(schema=1, kind='elemental_p2p_row_sparse',
            configuration=dict(ensemble='ensemble', configuration='cfg', lattice=[4, 1, 1, 3]),
            axes=[dict(name=name, coordinates=labels) for name, labels in (
                ('time', [0, 1, 2]), ('component', [dict(path=[0])]),
                ('momentum', [[1, 0, 0]]), ('point_left', [9, 3, 14]),
                ('color_left', [0, 1, 2]), ('color_right', [0, 1, 2]))],
            dtype='complex128',
            bases=dict(point_left=dict(data_id='actual-left-points', indices=[9, 3, 14])),
            parents=['elemental-links', 'actual-left-points', 'actual-right-points'],
            definition=dict(gauge_id='elemental-links', path_boundary='periodic',
                point_right_id='actual-right-points',
                point_right_selection='explicit_ordered_labels', point_right_labels=[24, 7]),
            conventions=dict(normalization='unscaled', phase='anchor',
                point_pair_layout='left-point-row-slots-v1'))

    def test_rectangular_endpoints_retain_actual_ids_and_right_order(self):
        spec = DatasetSpec(self.document())
        cfg, times = validate_diagram_input_specs(dict(p2p=spec))
        self.assertEqual(times, (0, 1, 2))
        self.assertEqual(spec.coordinates('point_left'), [9, 3, 14])
        self.assertEqual(spec.document['definition']['point_right_labels'], [24, 7])
        first = DatasetSpec(diagram_measurement_document(
            cfg, times, dict(p2p=spec), dict(p2p='actual-p2p'), {}))
        changed = self.document()
        changed['definition']['point_right_labels'].reverse()
        second = DatasetSpec(diagram_measurement_document(
            cfg, times, dict(p2p=DatasetSpec(changed)), dict(p2p='actual-p2p'), {}))
        self.assertNotEqual(first.definition_id, second.definition_id)
        self.assertEqual(first.document['definition']['input_views']['p2p']['parents'],
                         ['elemental-links', 'actual-left-points', 'actual-right-points'])

    def test_incomplete_or_ambiguous_right_declarations_are_refused(self):
        changes = (
            lambda d: d['definition'].pop('point_right_labels'),
            lambda d: d['definition'].update(point_right_labels=[]),
            lambda d: d['definition'].update(point_right_labels=[24, 24]),
            lambda d: d['definition'].update(point_right_labels=[False, 7]),
            lambda d: d['definition'].update(point_right_labels=[24.0, 7.0]),
            lambda d: d['definition'].update(point_right_selection='guess-by-shape'),
            lambda d: d['parents'].remove('actual-right-points'),
        )
        for change in changes:
            with self.subTest(change=change):
                document = self.document()
                change(document)
                with self.assertRaisesRegex(ValueError, 'P/P'):
                    validate_diagram_input_specs(dict(p2p=DatasetSpec(document)))

    def test_same_id_can_declare_distinct_right_labels_without_same_pool_guess(self):
        document = self.document()
        document['definition']['point_right_id'] = 'actual-left-points'
        document['definition']['point_right_labels'] = [14, 9]
        document['parents'].remove('actual-right-points')
        validate_diagram_input_specs(dict(p2p=DatasetSpec(document)))
        legacy = deepcopy(document)
        legacy['definition']['point_right_selection'] = 'same_ordered_prefix_as_point_left'
        with self.assertRaisesRegex(ValueError, 'same-pool'):
            validate_diagram_input_specs(dict(p2p=DatasetSpec(legacy)))


if __name__ == '__main__':
    unittest.main()
