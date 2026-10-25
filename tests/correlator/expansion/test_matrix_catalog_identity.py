"""Request-local label formatting preserves exact catalog and binding identity."""
from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp

from test_structured_correlator import fixture, task
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.expansion import matrix_catalog
from lqcdweave.contraction.expansion.api import MatrixCatalogBinder, describe_meson_matrix
from lqcdweave.contracts.correlator.api import DiagramBindings, MatrixLinearAxis
from lqcdweave.operators.correlator.api import (
    Flavor, Hadron, HadronIrrep, Tag, gen_correlator, remove_disconneted_diagram,
)


def original_axis(builder, labels, rows, time_label, *, selection=None):
    """Frozen pre-memoization algorithm, including its scientific expression key."""
    key = tuple(labels), time_label, selection
    if key not in builder.axis_ids:
        offsets, ids, coefficients = [0], [], []
        for choices in rows:
            combined = {}
            for row, position_weight in choices:
                for component, weight in builder.row(row, time_label):
                    combined[component] = combined.get(component, 0) + complex(position_weight) * weight
            for component, weight in combined.items():
                if weight != 0:
                    ids.append(component)
                    coefficients.append(weight)
            offsets.append(len(ids))
        builder.axis_ids[key] = len(builder.axes)
        builder.axes.append(MatrixLinearAxis(tuple(map(str, labels)), offsets, ids, coefficients))
    return builder.axis_ids[key]


class MatrixCatalogIdentityTests(unittest.TestCase):
    def setUp(self):
        self.bind, self.elemental, self.propagator = fixture()
        self.bindings = DiagramBindings(vertex_map=self.bind, usedNe=3, configuration='cfg',
            propagator_map={'S^q': self.propagator, r'S^q_\mathrm{local}': self.propagator})

    def compare_reference(self, request, bindings):
        with patch.object(matrix_catalog._CatalogBuilder, 'axis', original_axis):
            expected = describe_meson_matrix(request, bindings=bindings)
        actual = describe_meson_matrix(request, bindings=bindings)
        a, b = actual.catalog, expected.catalog
        self.assertEqual(a.components, b.components)
        self.assertEqual(a.networks, b.networks)
        self.assertEqual(a.groups, b.groups)
        self.assertEqual(a.output, b.output)
        self.assertEqual(len(a.axes), len(b.axes))
        for left, right in zip(a.axes, b.axes, strict=True):
            self.assertEqual(left.labels, right.labels)
            for name in ('offsets', 'component_ids', 'coefficients'):
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
        for left, right in zip(actual.component_bindings, expected.component_bindings, strict=True):
            self.assertIs(left.elemental, right.elemental)
            self.assertEqual(left.operator.name, right.operator.name)
            self.assertEqual(left.operator.terms, right.operator.terms)
            self.assertEqual(left.dagger, right.dagger)
            self.assertEqual(left.mode_indices, right.mode_indices)
        # The executable task equations, ordered inputs and output recipes are
        # unchanged as well; this is metadata-only and performs no array reads.
        with patch.object(self.elemental, 'block', side_effect=AssertionError('numeric IO')):
            left = MatrixCatalogBinder(actual, bindings=bindings, _factorize_spin=True).bind()
            right = MatrixCatalogBinder(expected, bindings=bindings, _factorize_spin=True).bind()
        self.assertEqual(left.problem, right.problem)
        self.assertEqual(left.output_recipe, right.output_recipe)
        self.assertEqual(left.inputs.read_requests, right.inputs.read_requests)
        return actual

    def mixed_request(self):
        a = HadronIrrep('a', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0]
        z = HadronIrrep('z', (0, 0, 0), 'A_1', -1, Tag(0, 0))[0]
        expression = (1+sp.I)*a + sp.sqrt(2)*z
        axis = [Hadron(expression, Flavor('uu')+Flavor('dd')),
                Hadron(2*expression, Flavor('uu')+Flavor('dd'))]
        full = gen_correlator([axis.copy(), axis.copy()], [0, 1], [False, True],
                              structured=True, degenerate=True)
        selected = {}
        def bind(row):
            if row not in selected:
                modes = (0,) if row.hadron_name == 'a' else (1, 2)
                selected[row] = self.bind(row).with_mode_indices((modes, modes))
            return selected[row]
        return {'full': full, 'again': full, 'connected': remove_disconneted_diagram(
            full, [r'S^q_\mathrm{local}'])}, replace(self.bindings, vertex_map=bind)

    def test_mixed_axes_format_each_object_once_per_request_and_match_original(self):
        request, bindings = self.mixed_request()
        calls = Counter()
        def formatted(label):
            calls[id(label)] += 1
            return str(label)
        with patch.object(matrix_catalog, 'str', formatted, create=True):
            described = describe_meson_matrix(request, bindings=bindings)
        self.assertTrue(calls)
        self.assertEqual(set(calls.values()), {1})
        self.assertGreater(sum(len(axis.labels) for axis in described.catalog.axes), len(calls))
        self.compare_reference(request, bindings)
        # A new request has its own cache; no cfg or numeric data can persist.
        calls.clear()
        with patch.object(matrix_catalog, 'str', formatted, create=True):
            describe_meson_matrix(request, bindings=bindings)
        self.assertEqual(set(calls.values()), {1})

    def test_equal_display_rows_and_different_coefficients_retain_distinct_axes(self):
        rows = [HadronIrrep('a', (0, 0, 0), 'A_1', -1, Tag(tag, 0))[0] for tag in (0, 1)]
        self.assertEqual(str(rows[0]), str(rows[1]))
        self.assertNotEqual(rows[0], rows[1])
        requests = []
        for expression in (rows[0], rows[1], (1+sp.I)*rows[0], (1-sp.I)*rows[0]):
            axis = [Hadron(expression, Flavor('ud'))]
            requests.append(gen_correlator([axis.copy(), axis.copy()], [0, 1], [False, True],
                                          structured=True, degenerate=True))
        def bind(row):
            return self.bind(row).with_mode_indices(((row.tag.tag,), (row.tag.tag,)))
        result = self.compare_reference(requests, replace(self.bindings, vertex_map=bind))
        self.assertEqual(len(result.catalog.axes), 8)
        self.assertEqual(result.catalog.axes[0].labels, result.catalog.axes[2].labels)
        self.assertNotEqual(result.catalog.groups[0].axes, result.catalog.groups[1].axes)
        self.assertEqual({v.mode_indices for v in result.component_bindings},
                         {((0,), (0,)), ((1,), (1,))})

    def test_unmixed_original_catalog_and_binding_are_unchanged(self):
        for flavor in ('charged', 'I0', 'I1'):
            with self.subTest(flavor=flavor):
                self.compare_reference(task(structured=True, flavor=flavor), self.bindings)


if __name__ == '__main__':
    unittest.main()
