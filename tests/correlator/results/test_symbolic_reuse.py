"""Reuse immutable symbolic subtrees without changing output ownership."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from lqcdweave.contraction.results import core
from lqcdweave.contracts.correlator.api import (
    ContractionTerm, OutputNode, OutputRecipe, TensorSpec,
    TermContribution, describe_coordinates,
)
from lqcdweave.contracts.execution import TermValue
from test_mode_block_correlator import fixture


class SymbolicReuseTests(unittest.TestCase):
    def test_retention_reuses_estimates_but_counts_every_output_occurrence(self):
        coordinates, _ = describe_coordinates(time_axes={0: [0, 1]})
        problem = SimpleNamespace(
            terms=(ContractionTerm(('input',), 'i->'),),
            specs={'input': TensorSpec((3,), np.complex128)},
        )

        def subtree():
            return OutputNode('multiply', children=(
                OutputNode('term', TermContribution('t')),
                OutputNode('constant', .5),
            ))

        shared = subtree()
        recipe = OutputRecipe(OutputNode('list', children=(shared,) * 3), ('t',), coordinates)
        with patch.object(core.np, 'broadcast_shapes', wraps=np.broadcast_shapes) as shapes:
            reserved = core.estimate_output_retention_bytes(recipe, problem, (None,), coordinates)
        self.assertEqual(shapes.call_count, 1)
        # Two complex TermValues, three independently assembled two-lane
        # products (term + real weight + result), and three list pointers.
        expected = 2 * 16 + 3 * (2 * 16 + 2 * 8 + 2 * 16) + 3 * np.dtype(object).itemsize
        self.assertEqual(reserved, expected)
        copied = OutputRecipe(OutputNode('list', children=tuple(subtree() for _ in range(3))),
                              ('t',), coordinates)
        self.assertEqual(core.estimate_output_retention_bytes(copied, problem, (None,), coordinates),
                         reserved)

    def test_immutable_numeric_weights_are_shared_and_arrays_are_independent(self):
        from sympy import Add, Float, Mul
        diagram, *_ = fixture(ne=1)
        weight = Float(.25)
        weighted = Mul(weight, diagram, evaluate=False)
        shifted = Add(weight, weighted, evaluate=False)
        raw = np.array([1., 2.])
        coordinates, _ = describe_coordinates()
        recipe = core.build_output_recipe(
            [weighted, shifted, raw, raw],
            {diagram: (TermContribution('term-0', 2 + 1j),)}, ('term-0',), coordinates,
        )
        self.assertIs(recipe.root.children[0].children[0], recipe.root.children[1].children[0])
        arrays = [child.value for child in recipe.root.children[2:]]
        self.assertFalse(np.shares_memory(*arrays))
        raw[:] = -1
        np.testing.assert_array_equal(arrays[0], [1., 2.])
        np.testing.assert_array_equal(arrays[1], [1., 2.])
        actual = core.assemble_correlator(
            SimpleNamespace(output_recipe=recipe, coordinates=coordinates),
            [TermValue('term-0', coordinates.points[0].key, 3 - 2j)],
        )
        self.assertEqual(actual[0], .25 * (2 + 1j) * (3 - 2j))
        self.assertEqual(actual[1], .25 + .25 * (2 + 1j) * (3 - 2j))

    def test_repeated_diagram_recipe_is_validated_once(self):
        diagram, *_ = fixture(ne=1)
        leaf = OutputNode('term', TermContribution('term-0'))
        coordinates, _ = describe_coordinates()
        with patch.object(core, '_validate_diagram_recipe',
                          wraps=core._validate_diagram_recipe) as validate:
            recipe = core.build_output_recipe(
                [diagram, diagram, np.array([diagram]*10, dtype=object)], {}, ('term-0',),
                coordinates, diagram_recipes={diagram: leaf},
            )
        self.assertEqual(validate.call_count, 1)
        self.assertIs(recipe.root.children[0], recipe.root.children[1])
        self.assertEqual(core.collect_diagrams([diagram, diagram]), (diagram,))

    def test_shared_containers_preserve_order_and_cycles_still_fail(self):
        diagram, loops, *_ = fixture(ne=1)
        shared = [diagram, loops[0]]
        self.assertEqual(core.collect_diagrams([shared, shared, loops[1]]),
                         (diagram, loops[0], loops[1]))
        coordinates, _ = describe_coordinates()
        circular = [shared]
        circular.append(circular)
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            core.collect_diagrams(circular)
        circular = [1]
        circular.append(circular)
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            core.build_output_recipe(circular, {}, (), coordinates)


if __name__ == '__main__':
    unittest.main()
