"""Named symbolic times retain identity independently of printable row names."""
import unittest

from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, Tag, gen_correlator,
)


class NamedTimeKeyTests(unittest.TestCase):
    def hadron(self, key=0, *, dagger=False):
        row = HadronIrrep('pi', (0, 0, 0), 'A_1', -1, Tag(0, key), dagger)[0]
        return Hadron(row, Flavor('ud', key))

    def test_distinct_times_are_not_overwritten_and_rewrite_keeps_dagger(self):
        first = self.hadron('source', dagger=True)
        second = self.hadron('current', dagger=True)
        self.assertNotEqual(first.irrep_row, second.irrep_row)
        self.assertEqual(first.irrep_row.tag.time, 'source')
        moved = first.set_time('sink')
        self.assertEqual(moved.irrep_row.tag.time, 'sink')
        self.assertEqual(moved.flavor_structure.time, 'sink')
        self.assertTrue(moved.irrep_row.dagger)
        self.assertEqual(first.irrep_row.tag.time, 'source')

    def test_named_equal_and_mixed_time_keys_survive_wick(self):
        for times in (('sink', 'source'), ('current', 'current'), (0, '0')):
            with self.subTest(times=times):
                value = gen_correlator([[self.hadron()], [self.hadron()]], times, [False, True])
                diagram, = value[0, 0].atoms(Diagram)
                self.assertEqual(set(diagram.time_list), set(times))
                self.assertEqual({row.tag.time for row in diagram.vertex_list}, set(times))
                local = times[0] == times[1]
                self.assertEqual(any('local' in str(prop) for prop in diagram.propagator_list), local)
        self.assertNotEqual(Flavor('ud', 0), Flavor('ud', '0'))

    def test_three_point_and_none_keep_declared_particle_times(self):
        keys = ('sink', 'current', 'source')
        hadrons = [[Hadron(HadronIrrep(name, (0, 0, 0), 'A_1', -1, Tag(0, key))[0],
                           Flavor(flavor, key))]
                   for name, key, flavor in zip(('pi', 'J', 'pi'), keys, ('ud', 'dd', 'ud'))]
        value = gen_correlator(hadrons, [None, None, None], [False, False, True])
        diagrams = value[0, 0, 0].atoms(Diagram)
        self.assertEqual({time for diagram in diagrams for time in diagram.time_list}, set(keys))
        self.assertEqual({row.tag.time for diagram in diagrams for row in diagram.vertex_list}, set(keys))

    def test_coordinates_keep_declaration_order_and_shared_key(self):
        coordinates, _ = describe_coordinates(
            time_axes={'sink': (2, 0), 'current': 1, 'source': (1, 0, 2)})
        self.assertEqual(coordinates.scan_axes, ('sink', 'source'))
        self.assertEqual(coordinates.scan_shape, (2, 3))
        self.assertEqual(coordinates.points[0].values,
                         (('sink', 2), ('current', 1), ('source', 1)))
        shared, _ = describe_coordinates(time_axes={'current': range(3)})
        self.assertEqual(shared.scan_shape, (3,))

    def test_sympy_integer_labels_from_existing_diagrams_are_normalized(self):
        import sympy as sp
        import numpy as np
        from lqcdweave.contracts.correlator.api import normalize_time_key
        coordinates, _ = describe_coordinates(time_axes={sp.Integer(0): (0, 1), sp.Integer(1): 0})
        self.assertEqual(coordinates.scan_axes, (0,))
        self.assertIs(coordinates.axes[0][0], sp.Integer(0))
        self.assertEqual(normalize_time_key(np.int64(2)), 2)
        source, sink = sp.symbols('t_source t_sink')
        symbolic, _ = describe_coordinates(time_axes={source: [0, 1], sink: 0})
        self.assertIs(symbolic.scan_axes[0], source)
        paired, _ = describe_coordinates(time_pairs=[{source: 0, sink: 1}])
        self.assertIs(paired.points[0].values[0][0], source)
        advanced, _ = describe_coordinates(time_axes={('insertion', 2): [0, 1], 1.5: 0})
        self.assertEqual(advanced.scan_axes, (('insertion', 2),))
        with self.assertRaises(TypeError):
            normalize_time_key(np.bool_(True))

    def test_invalid_scalar_time_keys_fail_at_symbolic_boundary(self):
        for key in (True, 1.5, -1, '', ' '):
            with self.subTest(key=key):
                with self.assertRaises((TypeError, ValueError)):
                    self.hadron(key)


if __name__ == '__main__':
    unittest.main()
