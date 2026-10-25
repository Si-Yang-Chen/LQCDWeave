"""Frozen result recipes consume ordered values with optional deep diagnostics."""
import unittest
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.results import core
from lqcdweave.contraction.results.bound import bind_fixed_output
from lqcdweave.contraction.results.api import (
    OutputNode,
    OutputRecipe,
    TermValue,
    assemble_correlator,
    bind_output_assembly,
)
from lqcdweave.contracts.correlator.api import (
    TermContribution,
    describe_coordinates,
)


def _matrix_recipe(coordinates):
    shared = OutputNode("term", TermContribution("beta", 2 + 1j))
    matrix = OutputNode(
        "transpose",
        (1, 0),
        (OutputNode(
            "reshape",
            children=(OutputNode("term", TermContribution("alpha")),),
            shape=(2, 2),
        ),),
        shape=(2, 2),
    )
    root = OutputNode(
        "dict",
        keys=("repeated", "linear", "matrix"),
        children=(
            OutputNode("add", children=(shared, shared)),
            OutputNode("diagram_sum", (("alpha", -0.5j), ("beta", 2))),
            matrix,
        ),
    )
    return OutputRecipe(root, ("beta", "alpha"), coordinates)


class BoundOutputAssemblyTests(unittest.TestCase):
    def test_joint_layout_metadata_counts_term_and_arithmetic_owners(self):
        from lqcdweave.contraction.results.api import describe_output_layout
        from lqcdweave.contracts.correlator.api import (
            ContractionProblem, ContractionTerm, SourceSpec, TensorSpec,
        )
        coordinates, _ = describe_coordinates()
        term = OutputNode('term', TermContribution('x'))
        root = OutputNode('dict', keys=('a', 'b'), children=(term,
            OutputNode('multiply', children=(OutputNode('constant', 2), term))))
        recipe = OutputRecipe(root, ('x',), coordinates)
        problem = ContractionProblem({'v': TensorSpec((2, 3), 'complex128')},
            (ContractionTerm(('v',), 'ij->ij'),), {'v': SourceSpec('v', 0)})
        retention, fields = describe_output_layout(recipe, problem)
        self.assertEqual(fields, ((('a',), (2, 3), '<c16'), (('b',), (2, 3), '<c16')))
        # One term snapshot, three term/arithmetic owners and two dict pointers.
        self.assertEqual(retention, 4 * 2 * 3 * 16 + 2 * np.dtype(object).itemsize)
        source = np.arange(6, dtype='complex128').reshape(2, 3)
        value = bind_output_assembly(recipe)((source,))
        np.testing.assert_array_equal(value['a'], source)
        np.testing.assert_array_equal(value['b'], 2 * source)

    def test_shared_scalar_dag_preserves_cancellation_and_mixed_precision_order(self):
        coordinates, _ = describe_coordinates()
        term = OutputNode('term', TermContribution('x', 1-2j))
        shared = OutputNode('multiply', children=(OutputNode('constant', 0.125), term,
            OutputNode('constant', -3.0)))
        root = OutputNode('tuple', children=(
            OutputNode('add', children=(OutputNode('constant', 1e16), shared,
                OutputNode('constant', -1e16), OutputNode('constant', 1.0))),
            OutputNode('power', -2, (shared,)),
            OutputNode('multiply', children=(OutputNode('constant', np.float32(0.1)),
                OutputNode('constant', np.float32(0.3)), shared)), shared))
        recipe = OutputRecipe(root, ('x',), coordinates)
        actual, reference = bind_output_assembly(recipe), bind_output_assembly(recipe, debug=True)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            for value in (dtype(1.25), dtype(-0.25)):
                left, right = actual((np.array(value),)), reference((np.array(value),))
                for a, b in zip(left, right, strict=True):
                    self.assertEqual(type(a), type(b))
                    np.testing.assert_array_equal(a, b)

    def test_shared_array_arithmetic_keeps_independent_owners_and_no_call_cache(self):
        coordinates, _ = describe_coordinates()
        shared = OutputNode('multiply', children=(OutputNode('constant', 2+1j),
            OutputNode('term', TermContribution('x', -0.5j))))
        recipe = OutputRecipe(OutputNode('list', children=(shared, shared)), ('x',), coordinates)
        assemble = bind_output_assembly(recipe)
        original = np.arange(4, dtype=np.complex128)
        result = assemble((original,))
        expected = (2+1j)*(-0.5j*original)
        np.testing.assert_array_equal(result[0], expected)
        np.testing.assert_array_equal(result[1], expected)
        self.assertFalse(np.shares_memory(result[0], result[1]))
        result[0][0] = 99
        np.testing.assert_array_equal(result[1], expected)
        source_ref, output_ref = weakref.ref(original), weakref.ref(result[1])
        del original, result
        gc.collect()
        self.assertIsNone(source_ref())
        self.assertIsNone(output_ref())
        np.testing.assert_array_equal(assemble((np.arange(4)+3j,))[0],
            (2+1j)*(-0.5j*(np.arange(4)+3j)))

    def test_literal_array_copies_and_arithmetic_errors_remain_per_call(self):
        coordinates, _ = describe_coordinates()
        value = np.array([1., 2.])
        node = OutputNode('constant_array', value, shape=(2,))
        assemble = bind_output_assembly(OutputRecipe(OutputNode('tuple', children=(node, node)), (), coordinates))
        first, second = assemble(())
        first[0] = 99
        np.testing.assert_array_equal(second, value)
        np.testing.assert_array_equal(assemble(())[0], value)
        invalid = OutputNode('power', -1, (OutputNode('constant', 0),))
        fail = bind_output_assembly(OutputRecipe(invalid, (), coordinates))
        with self.assertRaises(ZeroDivisionError): fail(())

    def test_shared_arithmetic_accepts_indexed_values_with_changing_rank(self):
        coordinates, _ = describe_coordinates()
        term = OutputNode('term', TermContribution('x', -0.5j))
        shared = OutputNode('multiply', children=(OutputNode('constant', 2+1j), term))
        recipe = OutputRecipe(OutputNode('tuple', children=(shared, shared)), ('x',), coordinates)
        assemble = bind_output_assembly(recipe)
        reference = bind_output_assembly(recipe, debug=True)

        class IndexedValues:
            def __init__(self, value): self.value = value
            def __len__(self): return 1
            def __getitem__(self, index):
                if index != 0: raise IndexError(index)
                return self.value

        for value in (np.array(1+2j), np.arange(6).reshape(2, 3)+1j,
                      np.complex64(-3+0.25j), np.arange(3)*0.5j):
            values = IndexedValues(value)
            actual, expected = assemble(values), reference(values)
            for left, right in zip(actual, expected, strict=True):
                self.assertEqual(type(left), type(right))
                np.testing.assert_array_equal(left, right)
            if isinstance(actual[0], np.ndarray):
                self.assertFalse(np.shares_memory(*actual))

    def test_multiply_keeps_initial_one_literal_order_and_errors(self):
        coordinates, _ = describe_coordinates()
        term = OutputNode('term', TermContribution('x'))
        nodes = tuple(OutputNode('multiply', children=(
            OutputNode('constant', left), OutputNode('constant', right), term,
            OutputNode('constant', 0.125)))
            for left, right in ((-0.0, 3.0), (1e-200, 1e200), (2+3j, -0.5j)))
        recipe = OutputRecipe(OutputNode('tuple', children=nodes), ('x',), coordinates)
        actual, reference = bind_output_assembly(recipe), bind_output_assembly(recipe, debug=True)
        for value in (np.array(-0.0), np.complex64(1-2j), np.array([1+2j, -3j])):
            for left, right in zip(actual((value,)), reference((value,)), strict=True):
                self.assertEqual(type(left), type(right))
                left, right = np.asarray(left), np.asarray(right)
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.tobytes(), right.tobytes())
        invalid = OutputNode('power', -1, (OutputNode('constant', 0),))
        fail = bind_output_assembly(OutputRecipe(OutputNode('multiply', children=(
            OutputNode('constant', 0), invalid, term)), ('x',), coordinates))
        with self.assertRaises(ZeroDivisionError): fail((1,))

    def test_binding_releases_term_index_workspace_without_waiting_for_gc(self):
        coordinates, _ = describe_coordinates()
        recipe = OutputRecipe(OutputNode('term', TermContribution('x', 2j)), ('x',), coordinates)

        class Indexes(dict): pass

        indexes = Indexes(x=0)
        reference = weakref.ref(indexes)
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            assemble = bind_fixed_output(recipe, indexes)
            del indexes
            self.assertIsNone(reference())
            self.assertEqual(assemble((3+1j,)), 2j*(3+1j))
        finally:
            if was_enabled: gc.enable()

    def test_multiply_constant_prefix_keeps_dynamic_boundary_and_error_time(self):
        coordinates, _ = describe_coordinates()
        term = OutputNode('term', TermContribution('x'))
        node = OutputNode('multiply', children=(OutputNode('constant', 1e308), term,
            OutputNode('constant', 1e-308)))
        recipe = OutputRecipe(node, ('x',), coordinates)
        actual, reference = bind_output_assembly(recipe), bind_output_assembly(recipe, debug=True)
        with np.errstate(over='ignore', invalid='ignore'):
            for value in (np.array(1e308), np.array([1e308, -0.0]), np.float32(0.25)):
                left, right = np.asarray(actual((value,))), np.asarray(reference((value,)))
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.tobytes(), right.tobytes())

        node = OutputNode('multiply', children=(OutputNode('constant', 10**400),
            OutputNode('constant', 1j), term))
        recipe = OutputRecipe(node, ('x',), coordinates)
        actual, reference = bind_output_assembly(recipe), bind_output_assembly(recipe, debug=True)
        for evaluate in (actual, reference, actual):
            with self.assertRaises(OverflowError):
                evaluate((1,))

    def test_shared_nodes_coefficients_and_matrix_layout_match_independent_values(self):
        coordinates, _ = describe_coordinates(time_axes={"source": 2, "sink": 1})
        recipe = _matrix_recipe(coordinates)
        assemble = bind_output_assembly(recipe)
        beta = np.array([1 + 2j, -3 + 0.5j, 2 - 1j, 4 + 3j])
        alpha = np.array([2 - 1j, 3 + 4j, -2 + 0.25j, 5 - 2j])

        for current_beta, current_alpha in (
            (beta, alpha),
            (beta * (1 - 2j), alpha + (3 + 1j)),
        ):
            result = assemble((current_beta, current_alpha))
            weighted_beta = (2 + 1j) * current_beta
            np.testing.assert_allclose(result["repeated"], weighted_beta + weighted_beta)
            np.testing.assert_allclose(
                result["linear"], -0.5j * current_alpha + 2 * current_beta
            )
            np.testing.assert_array_equal(
                result["matrix"], current_alpha.reshape(2, 2).T
            )

    def test_default_binding_skips_deep_audit_and_term_value_wrapping(self):
        coordinates, _ = describe_coordinates()
        recipe = OutputRecipe(
            OutputNode("add", children=(
                OutputNode("term", TermContribution("left", 2 + 1j)),
                OutputNode("term", TermContribution("right", -0.5j)),
            )),
            ("left", "right"),
            coordinates,
        )
        with patch.object(core, "_validate_output_node",
                          wraps=core._validate_output_node) as validate:
            assemble = bind_output_assembly(recipe)
            self.assertEqual(validate.call_count, 0)
            with patch.object(core, "TermValue", side_effect=AssertionError(
                    "the bound path must not construct TermValue objects")):
                self.assertEqual(assemble((3 + 2j, -4j)), (2 + 1j) * (3 + 2j) - 0.5j * (-4j))
                self.assertEqual(assemble((1 - 1j, 7 + 2j)), (2 + 1j) * (1 - 1j) - 0.5j * (7 + 2j))
            self.assertEqual(validate.call_count, 0)

        with self.assertRaisesRegex(ValueError, "ordered term value count mismatch"):
            bind_output_assembly(recipe, debug=True)((1,))

    def test_debug_rejects_inconsistent_frozen_recipe_and_extra_values(self):
        coordinates, _ = describe_coordinates()
        term = OutputNode("term", TermContribution("x"))
        duplicate = OutputRecipe(term, ("x", "x"), coordinates)
        with self.assertRaisesRegex(ValueError, "term_ids must be unique"):
            bind_output_assembly(duplicate, debug=True)
        assemble = bind_output_assembly(OutputRecipe(term, ("x",), coordinates), debug=True)
        with self.assertRaisesRegex(ValueError, "ordered term value count mismatch"):
            assemble((1, 2))

    def test_debug_uses_deep_assembly_and_requires_a_boolean_flag(self):
        coordinates, _ = describe_coordinates()
        recipe = OutputRecipe(
            OutputNode("term", TermContribution("value", 3 - 2j)),
            ("value",),
            coordinates,
        )
        with self.assertRaisesRegex(TypeError, "debug must be bool"):
            bind_output_assembly(recipe, debug=1)

        with patch.object(core, "_validate_output_node",
                          wraps=core._validate_output_node) as validate:
            assemble = bind_output_assembly(recipe, debug=True)
            binding_calls = validate.call_count
            self.assertEqual(assemble((2 + 4j,)), (3 - 2j) * (2 + 4j))
            self.assertGreater(validate.call_count, binding_calls)

    def test_debug_binding_rejects_scanned_coordinates_even_for_one_point(self):
        cases = (
            describe_coordinates(time_pairs=({"source": 0, "sink": 1},))[0],
            describe_coordinates(time_pairs=(
                {"source": 0, "sink": 1},
                {"source": 2, "sink": 3},
            ))[0],
        )
        for coordinates in cases:
            recipe = OutputRecipe(
                OutputNode("term", TermContribution("value")),
                ("value",),
                coordinates,
            )
            with self.subTest(point_count=len(coordinates.points)):
                with self.assertRaisesRegex(ValueError, "fixed single-point"):
                    bind_output_assembly(recipe, debug=True)

    def test_ordinary_assembly_keeps_coverage_validation(self):
        coordinates, _ = describe_coordinates()
        recipe = OutputRecipe(
            OutputNode("term", TermContribution("value")),
            ("value",),
            coordinates,
        )
        expanded = SimpleNamespace(output_recipe=recipe, coordinates=coordinates)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            assemble_correlator(expanded, ())
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            assemble_correlator(
                expanded,
                (TermValue("unexpected", coordinates.points[0].key, 2),),
            )


if __name__ == "__main__":
    unittest.main()
