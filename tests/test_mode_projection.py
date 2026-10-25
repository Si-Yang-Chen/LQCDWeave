"""Fused symbolic projection checked against independent numerical pairing."""
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Add, Mul

from lqcdweave.operators.correlator.api import (
    Diagram, QuarkDiagram, split_eigenmodes,
    split_meson_mode_pairs,
)
from lqcdweave.contraction.correlator import calculate
from test_mode_block_correlator import fixture


def pair_reference(value, mode_axis, vertices):
    """Direct sum over orientations with independent real pair weights."""
    count = value.shape[mode_axis]
    pairs = [(a, b) for a in range(count) for b in range(a, count)]
    shape = (*value.shape[:mode_axis], *((len(pairs),) * vertices),
             *value.shape[mode_axis + 2 * vertices:])
    result = np.empty(shape, dtype=complex)
    from itertools import product
    for paired in np.ndindex((len(pairs),) * vertices):
        orientations = [[pairs[row]] if pairs[row][0] == pairs[row][1]
                        else [pairs[row], pairs[row][::-1]] for row in paired]
        blocks = [value[(slice(None),) * mode_axis
                        + tuple(axis for pair in choices for axis in pair)]
                  for choices in product(*orientations)]
        result[(slice(None),) * mode_axis + paired] = sum(blocks) / np.sqrt(len(blocks))
    return result


class ModeProjectionTests(unittest.TestCase):
    def test_four_pairs_keep_full_physical_terms_and_complex_values(self):
        diagram, loops, vertices, elemental, propagator = fixture(ne=4)
        full = -(1 + .2j) * diagram + .4 * loops[0] * loops[1]
        matrix = np.array([[full, diagram], [loops[0] * loops[1], 0]], dtype=object)
        fused = split_meson_mode_pairs(matrix, partitions=(0, 1, 2, 3, 4))
        self.assertEqual(fused.shape, (2, 2, 10, 10))
        with patch.object(elemental, 'block', side_effect=AssertionError('metadata read')), \
                patch.object(propagator, 'block', side_effect=AssertionError('metadata read')):
            planned = calculate(fused, vertex_map=vertices, propagator_map={'q': propagator},
                                configuration='cfg', usedNe=4, optimize='none', execute=False)
        # Connected 256 blocks and 16 independent blocks for each loop are kept.
        self.assertEqual(len(planned.expanded.problem.terms), 288)
        actual = planned.run().value
        ordered = calculate(split_eigenmodes(matrix, partitions=(0, 1, 2, 3, 4)),
                            vertex_map=vertices, propagator_map={'q': propagator},
                            configuration='cfg', usedNe=4, optimize='none').value
        np.testing.assert_allclose(actual, pair_reference(ordered, 2, 2), rtol=8e-13, atol=8e-13)

    def test_disconnected_projection_remains_product_of_projected_loops(self):
        _, loops, *_ = fixture()
        projected = split_meson_mode_pairs(loops[0] * loops[1], partitions=(0, 2, 7))
        mixed = projected[1, 1]
        self.assertIsInstance(mixed, Mul)
        self.assertEqual(sum(isinstance(factor, Add) for factor in mixed.args), 2)
        self.assertEqual(len(mixed.atoms(Diagram)), 4)

    def test_repeated_fields_reuse_projection_and_own_array_slots(self):
        diagram, loops, *_ = fixture()
        full = diagram + loops[0] * loops[1]
        from lqcdweave.operators.correlator.mode_projection import _project_symbolic_pairs
        with patch('lqcdweave.operators.correlator.mode_projection._project_symbolic_pairs',
                   wraps=_project_symbolic_pairs) as project:
            result = split_meson_mode_pairs(
                {'full': full, 'connected': diagram, 'disconnected': loops[0] * loops[1],
                 'again': diagram}, partitions=(0, 2, 7))
        self.assertEqual(project.call_count, 3)
        result['connected'][0, 0] = 0
        self.assertNotEqual(result['again'][0, 0], 0)

    def test_reversed_vertex_order_nested_zeros_and_unequal_bands(self):
        diagram, loops, vertices, _, propagator = fixture(ne=7)
        reverse = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [1, 0], ['b', 'a'], [None, 'q'])
        original = {'matrix': np.array([[reverse, diagram], [0, .3j * reverse]], dtype=object),
                    'loops': ([loops[1], loops[0]],), 'constant': 3}
        fused = split_meson_mode_pairs(original, partitions=(0, 1, 3, 7))
        self.assertEqual(fused['matrix'].shape, (2, 2, 6, 6))
        self.assertIsInstance(fused['loops'], tuple)
        self.assertIsInstance(fused['loops'][0], list)
        self.assertEqual(fused['constant'], 3)
        actual = calculate(fused, vertex_map=vertices, propagator_map={'q': propagator},
                           configuration='cfg', usedNe=7, optimize='none',
                           time_axes={0: [0, 1], 1: [1, 0]}).value
        ordered = calculate(split_eigenmodes(original, partitions=(0, 1, 3, 7)),
                            vertex_map=vertices, propagator_map={'q': propagator},
                            configuration='cfg', usedNe=7, optimize='none',
                            time_axes={0: [0, 1], 1: [1, 0]}).value
        np.testing.assert_allclose(actual['matrix'], pair_reference(ordered['matrix'], 2, 2),
                                   rtol=8e-13, atol=8e-13)
        for actual_loop, ordered_loop in zip(actual['loops'][0], ordered['loops'][0]):
            np.testing.assert_allclose(actual_loop, pair_reference(ordered_loop, 0, 1),
                                       rtol=8e-13, atol=8e-13)

    def test_invalid_expressions_use_the_same_rejection_domain(self):
        diagram, loops, *_ = fixture()
        repeated = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 0], ['a', 'a'], [None, 'q'])
        for expression in (diagram ** 2, loops[0] + loops[1], repeated):
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                split_meson_mode_pairs(expression, partitions=(0, 2, 7))
        with self.assertRaisesRegex(ValueError, 'already mode-partitioned'):
            split_meson_mode_pairs(split_eigenmodes(diagram, partitions=(0, 2, 7)),
                                   partitions=(0, 2, 7))
        with self.assertRaises(ValueError):
            split_meson_mode_pairs(diagram, partitions=(0, 2, 2, 7))


if __name__ == '__main__':
    unittest.main()
