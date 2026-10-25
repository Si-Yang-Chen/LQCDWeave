"""Direct independent checks for result recipes and coordinate restoration."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    TensorSpec,
    TermContribution,
    WorkloadSpec,
    describe_coordinates,
)
from lqcdweave.contraction.results.api import (
    OutputNode,
    OutputRecipe,
    TermValue,
    assemble_correlator,
    assemble_workload_outputs,
    build_output_recipe,
    collect_diagrams,
)
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram


def expanded(root, term_ids, coordinates):
    return SimpleNamespace(
        output_recipe=OutputRecipe(root, tuple(term_ids), coordinates),
        coordinates=coordinates,
    )


def term_node(term_id, coefficient=1):
    return OutputNode("diagram_sum", ((term_id, coefficient),))


def output_node(*, root, term_ids, coordinates):
    return expanded(root, term_ids, coordinates)


class ResultStructureTests(unittest.TestCase):
    def test_exact_exchange_radicals_are_numeric_constants_but_diagram_powers_remain_integer(self):
        import sympy as sp
        diagram = Diagram(QuarkDiagram([[1]]), [0], ['a'], [None, 'q'])
        coordinates, _ = describe_coordinates()
        recipe = build_output_recipe((sp.sqrt(3)+sp.I/2)*diagram,
                                     {diagram: (TermContribution('term'),)}, ('term',), coordinates)
        result = assemble_correlator(expanded(recipe.root, ('term',), coordinates),
                                    (TermValue('term', 'bound', 2-3j),))
        np.testing.assert_allclose(result, (np.sqrt(3)+.5j)*(2-3j))
        with self.assertRaisesRegex(TypeError, 'integer exponents'):
            build_output_recipe(sp.sqrt(diagram), {diagram: (TermContribution('term'),)},
                                ('term',), coordinates)

    def test_persisted_workload_recipes_multiply_complex_values_in_order(self):
        keys = ("left", "middle", "right")
        workload = WorkloadSpec(
            specs={key: TensorSpec((1,), np.complex128) for key in keys},
            terms=tuple(ContractionTerm((key,), "i->") for key in keys),
            convention="meson-simple-v1",
            output_groups=((0, 1), (2,)),
            metadata={
                "adapter": "meson-simple-v1",
                "output_recipes": [[0, 1], [2]],
                "assembly_extra_bytes": 96,
            },
        )
        values = (2 + 3j, -1 + 0.5j, 7 - 2j)

        result = assemble_workload_outputs(workload, values, np)

        expected = np.array([values[0] * values[1], values[2]])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (2,))

    def test_generic_workload_assembly_preserves_the_owned_array_tuple(self):
        value = np.array([1.0, 2.0])
        workload = WorkloadSpec(
            specs={"input": TensorSpec((2,), np.float64)},
            terms=(ContractionTerm(("input",), "i->i"),),
        )

        result = assemble_workload_outputs(workload, (value,), np)

        self.assertIsInstance(result, tuple)
        self.assertIs(result[0], value)

    def test_axes_restore_requested_scan_order_and_nested_structure(self):
        coordinates, batch = describe_coordinates(
            time_axes={"sink": (4, 1, 7), "source": 5, "current": (9, 0)},
            batch={"size": 2, "axis": "sink"},
        )
        self.assertEqual(batch.size, 2)
        self.assertEqual(coordinates.fixed_axes, ("source",))
        self.assertEqual(coordinates.scan_axes, ("sink", "current"))
        self.assertEqual(coordinates.scan_shape, (3, 2))
        self.assertEqual(
            [tuple(value for _, value in point.values) for point in coordinates.points],
            [(4, 5, 9), (4, 5, 0), (1, 5, 9), (1, 5, 0), (7, 5, 9), (7, 5, 0)],
        )

        diagram = term_node("left", 2 + 1j)
        diagram = OutputNode(
            "diagram_sum",
            (("left", 2 + 1j), ("right", -.5j)),
        )
        scalar = OutputNode("constant", 3)
        offset = OutputNode("constant", .5 - 1j)
        product = OutputNode("multiply", children=(OutputNode("constant", 1 - 2j), diagram))
        squared = OutputNode("power", 2, (OutputNode("add", children=(diagram, offset)),))
        formula = OutputNode("add", children=(product, squared))
        vector = OutputNode("array", children=(diagram, scalar), shape=(2,))
        nested = OutputNode(
            "list",
            children=(diagram, OutputNode("tuple", children=(diagram, scalar))),
        )
        fixed = np.array([1.0, 2.0])
        constant_array = OutputNode("constant_array", fixed, shape=fixed.shape)
        root = OutputNode(
            "dict",
            keys=("vector", "formula", "nested", "array_constant", "empty_dict", "empty_list", "empty_tuple"),
            children=(
                vector,
                formula,
                nested,
                constant_array,
                OutputNode("dict", keys=(), children=()),
                OutputNode("list"),
                OutputNode("tuple"),
            ),
        )

        values = []
        expected_diagram = []
        for point in coordinates.points:
            point_map = dict(point.values)
            left = complex(10 * point_map["sink"] + point_map["current"],
                           point_map["source"] - point_map["current"])
            right = complex(point_map["sink"] - 2 * point_map["current"],
                            point_map["source"] + 3 * point_map["current"])
            expected_diagram.append((2 + 1j) * left - .5j * right)
            values.append(TermValue("left", point.key, left))
            values.append(TermValue("right", point.key, right))
        # Results arrive in a batch/interleaved order. Coordinate keys, rather
        # than arrival order, determine the restored scan axes.
        values = values[::2][::-1] + values[1::2]
        result = assemble_correlator(output_node(root=root, term_ids=("left", "right"),
                                                 coordinates=coordinates), values)
        expected_diagram = np.asarray(expected_diagram).reshape(coordinates.scan_shape)
        expected_vector = np.stack(
            (expected_diagram, np.full(coordinates.scan_shape, 3, dtype=complex)), axis=0
        )
        np.testing.assert_allclose(result["vector"], expected_vector)
        np.testing.assert_allclose(
            result["formula"], (1 - 2j) * expected_diagram + (expected_diagram + (.5 - 1j)) ** 2
        )
        self.assertEqual(type(result["nested"]), list)
        self.assertEqual(type(result["nested"][1]), tuple)
        np.testing.assert_allclose(result["nested"][0], expected_diagram)
        np.testing.assert_allclose(result["nested"][1][1], np.full(coordinates.scan_shape, 3))
        self.assertEqual(result["empty_dict"], {})
        self.assertEqual(result["empty_list"], [])
        self.assertEqual(result["empty_tuple"], ())
        self.assertEqual(result["array_constant"].shape, (2, 3, 2))
        np.testing.assert_array_equal(
            result["array_constant"],
            np.broadcast_to(fixed.reshape(2, 1, 1), (2, 3, 2)),
        )

    def test_time_pairs_keep_arbitrary_order_interleaving_and_tail_batch(self):
        pairs = (
            {"source": 8, "sink": 3},
            {"source": 1, "sink": 9},
            {"source": 7, "sink": 0},
            {"source": 2, "sink": 5},
            {"source": 6, "sink": 4},
        )
        coordinates, batch = describe_coordinates(time_pairs=pairs, batch={"size": 2, "axis": "pair"})
        self.assertEqual(batch.axis, "pair")
        self.assertEqual(batch.size, 2)
        values = []
        expected = []
        for start in range(0, len(coordinates.points), batch.size):
            tail = coordinates.points[start:start + batch.size]
            for point in reversed(tail):
                point_map = dict(point.values)
                raw = 100 * point_map["source"] + point_map["sink"]
                values.append(TermValue("only", point.key, raw))
                expected.append((point.key, raw))
        # Add a second term in a different order to ensure pairing is by key.
        for point in reversed(coordinates.points):
            point_map = dict(point.values)
            raw = point_map["source"] - 2 * point_map["sink"]
            values.append(TermValue("other", point.key, raw))
        root = OutputNode(
            "diagram_sum",
            (("only", 1.25 - .5j), ("other", -2j)),
        )
        result = assemble_correlator(output_node(root=root, term_ids=("only", "other"),
                                                 coordinates=coordinates), values)
        by_key = {key: raw for key, raw in expected}
        oracle = np.array([
            (1.25 - .5j) * by_key[point.key]
            - 2j * (dict(point.values)["source"] - 2 * dict(point.values)["sink"])
            for point in coordinates.points
        ])
        self.assertEqual(result.shape, (5,))
        np.testing.assert_allclose(result, oracle)

    def test_fixed_axes_do_not_add_dimensions_and_constant_scalar_broadcasts(self):
        coordinates, _ = describe_coordinates(time_axes={"source": 3, "sink": (4, 2)})
        root = OutputNode(
            "tuple",
            children=(term_node("t"), OutputNode("constant", 7)),
        )
        values = [TermValue("t", point.key, 10 + point.values[1][1])
                  for point in coordinates.points]
        result = assemble_correlator(output_node(root=root, term_ids=("t",),
                                                 coordinates=coordinates), values)
        self.assertEqual(result[0].shape, (2,))
        np.testing.assert_array_equal(result[0], (14, 12))
        np.testing.assert_array_equal(result[1], (7, 7))

        fixed, _ = describe_coordinates(time_axes={"source": 3, "sink": 4})
        scalar = assemble_correlator(
            output_node(root=term_node("t"), term_ids=("t",), coordinates=fixed),
            [TermValue("t", fixed.points[0].key, 12 + 3j)],
        )
        self.assertEqual(scalar, 12 + 3j)
        self.assertEqual(np.asarray(scalar).shape, ())

    def test_array_valued_term_keeps_value_axes_before_scan_axes(self):
        coordinates, _ = describe_coordinates(
            time_axes={"sink": (4, 1), "source": (2, 0, 3)}
        )
        values = []
        ordered_vectors = []
        for index, point in enumerate(coordinates.points):
            vector = np.array([10 + index, -3 * index], dtype=np.complex128)
            ordered_vectors.append(vector)
            values.append(TermValue("tensor", point.key, vector))
        result = assemble_correlator(
            output_node(root=term_node("tensor"), term_ids=("tensor",), coordinates=coordinates),
            values,
        )
        oracle = np.stack(ordered_vectors).reshape((*coordinates.scan_shape, 2)).transpose(2, 0, 1)
        self.assertEqual(result.shape, (2, 2, 3))
        np.testing.assert_array_equal(result, oracle)

    def test_nested_object_arrays_keep_containers_and_unequal_shapes(self):
        coordinates, _ = describe_coordinates(time_pairs=(
            {"source": 0, "sink": 2}, {"source": 3, "sink": 1},
        ))
        unequal = OutputNode("array", children=(
            term_node("variable"),
            OutputNode("list", children=(OutputNode("constant", 5), term_node("variable"))),
        ), shape=(2,))
        values = [
            TermValue("variable", coordinates.points[0].key, np.array([1, 2])),
            TermValue("variable", coordinates.points[1].key, np.array([3, 4, 5])),
        ]
        result = assemble_correlator(output_node(root=unequal, term_ids=("variable",),
                                                 coordinates=coordinates), values)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, object)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape, (2,))
        np.testing.assert_array_equal(result[0][0], np.array([1, 2]))
        np.testing.assert_array_equal(result[0][1], np.array([3, 4, 5]))
        self.assertIsInstance(result[1], list)
        np.testing.assert_array_equal(result[1][0], (5, 5))

        # Empty native containers retain their declared type and dimensions.
        empty = np.empty((0, 2), dtype=object)
        empty_recipe = OutputNode("array", shape=empty.shape)
        empty_result = assemble_correlator(
            output_node(root=empty_recipe, term_ids=(), coordinates=coordinates), ()
        )
        self.assertEqual(empty_result.shape, (0, 2))
        self.assertEqual(empty_result.dtype, object)

    def test_equal_shaped_array_elements_remain_nested_object_cells(self):
        source = np.empty((2,), dtype=object)
        source[0] = np.array([1, 2])
        source[1] = np.array([3, 4])

        fixed, _ = describe_coordinates()
        fixed_recipe = build_output_recipe(source, {}, (), fixed)
        fixed_result = assemble_correlator(
            SimpleNamespace(output_recipe=fixed_recipe, coordinates=fixed), ()
        )
        self.assertEqual(fixed_result.shape, (2,))
        self.assertEqual(fixed_result.dtype, object)
        np.testing.assert_array_equal(fixed_result[0], (1, 2))
        np.testing.assert_array_equal(fixed_result[1], (3, 4))

        scanned, _ = describe_coordinates(time_pairs=(
            {"source": 0, "sink": 1}, {"source": 2, "sink": 3},
        ))
        scanned_recipe = build_output_recipe(source, {}, (), scanned)
        scanned_result = assemble_correlator(
            SimpleNamespace(output_recipe=scanned_recipe, coordinates=scanned), ()
        )
        self.assertEqual(scanned_result.shape, (2,))
        self.assertEqual(scanned_result.dtype, object)
        self.assertEqual(scanned_result[0].shape, (2, 2))
        self.assertEqual(scanned_result[1].shape, (2, 2))
        np.testing.assert_array_equal(scanned_result[0], np.array([[1, 1], [2, 2]]))
        np.testing.assert_array_equal(scanned_result[1], np.array([[3, 3], [4, 4]]))

        nested_source = np.empty((2,), dtype=object)
        nested_source[0] = np.array([1, 2], dtype=object)
        nested_source[1] = np.array([3, 4], dtype=object)
        nested_recipe = build_output_recipe(nested_source, {}, (), fixed)
        nested_result = assemble_correlator(
            SimpleNamespace(output_recipe=nested_recipe, coordinates=fixed), ()
        )
        self.assertEqual(nested_result.shape, (2,))
        self.assertEqual(nested_result.dtype, object)
        np.testing.assert_array_equal(nested_result[0], (1, 2))
        np.testing.assert_array_equal(nested_result[1], (3, 4))

    def test_term_coefficient_is_applied_once_and_repeated_diagram_reads_reuse_value(self):
        coordinates, _ = describe_coordinates(time_pairs=(
            {"source": 4, "sink": 7}, {"source": 1, "sink": 0},
        ))
        contribution = OutputNode("term", TermContribution("t", 2 + 3j))
        repeated = OutputNode("add", children=(contribution, contribution))
        root = OutputNode(
            "multiply",
            children=(OutputNode("constant", 1 - 2j), repeated),
        )
        values = [TermValue("t", point.key, index + 1j)
                  for index, point in enumerate(coordinates.points)]
        result = assemble_correlator(output_node(root=root, term_ids=("t",),
                                                 coordinates=coordinates), values)
        expected = (1 - 2j) * 2 * (2 + 3j) * np.array([0 + 1j, 1 + 1j])
        np.testing.assert_allclose(result, expected)

    def test_repeated_diagram_is_collected_once_and_reused_across_output_positions(self):
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ["a", "b"], [None, "q"])
        self.assertEqual(collect_diagrams([diagram, diagram]), (diagram,))
        coordinates, _ = describe_coordinates()
        recipe = build_output_recipe(
            [diagram, diagram],
            {diagram: (TermContribution("shared", 2 + 1j),)},
            ("shared",), coordinates,
        )
        result = assemble_correlator(
            SimpleNamespace(output_recipe=recipe, coordinates=coordinates),
            [TermValue("shared", coordinates.points[0].key, 3 - 2j)],
        )
        self.assertEqual(result, [((2 + 1j) * (3 - 2j)), ((2 + 1j) * (3 - 2j))])

    def test_missing_extra_duplicate_and_unknown_term_results_are_rejected(self):
        coordinates, _ = describe_coordinates(time_pairs=(
            {"source": 0, "sink": 1}, {"source": 2, "sink": 3},
        ))
        expanded_value = output_node(root=term_node("t"), term_ids=("t",), coordinates=coordinates)
        complete = [TermValue("t", point.key, index)
                    for index, point in enumerate(coordinates.points)]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            assemble_correlator(expanded_value, complete[:1])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            assemble_correlator(expanded_value, complete + [TermValue("extra", "0", 4)])
        with self.assertRaisesRegex(ValueError, "duplicate term result"):
            assemble_correlator(expanded_value, complete + [complete[0]])
        with self.assertRaisesRegex(ValueError, "unknown term_id"):
            bad = output_node(root=term_node("unknown"), term_ids=("t",), coordinates=coordinates)
            assemble_correlator(bad, complete)

        duplicate_coordinates = replace(coordinates, points=(coordinates.points[0], coordinates.points[0]))
        duplicate_keys = output_node(root=term_node("t"), term_ids=("t",),
                                     coordinates=duplicate_coordinates)
        with self.assertRaisesRegex(ValueError, "unique nonempty point keys"):
            assemble_correlator(duplicate_keys, complete)

    def test_cycles_in_supported_containers_are_rejected(self):
        cycle = []
        cycle.append(cycle)
        with self.assertRaisesRegex(ValueError, "cyclic correlator containers"):
            collect_diagrams(cycle)
        coordinates, _ = describe_coordinates()
        with self.assertRaisesRegex(ValueError, "cyclic correlator containers"):
            build_output_recipe(cycle, {}, (), coordinates)


if __name__ == "__main__":
    unittest.main()
