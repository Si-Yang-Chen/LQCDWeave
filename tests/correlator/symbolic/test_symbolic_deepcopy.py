"""Independent symbolic snapshots must preserve identity without aliasing."""

from copy import deepcopy
import unittest
import numpy as np

from sympy import Symbol

from lqcdweave.operators.correlator.api import (
    Diagram, HadronIrrep, Propagator, QuarkDiagram, Tag,
)


class SymbolicDeepcopyTests(unittest.TestCase):
    def test_graph_equality_and_hash_normalize_array_coordinates_and_links(self):
        left = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ['a', 'b'], [None, 'q'])
        right = Diagram(QuarkDiagram(np.array([[0, 1], [1, 0]]), validate=False),
                        np.arange(2), ['a', 'b'], [None, 'q'])
        self.assertIs(left == right, True)
        self.assertEqual(hash(left), hash(right))
        self.assertEqual(len({left, right}), 1)
        changed = Diagram(QuarkDiagram([[1, 0], [0, 1]]), np.arange(2), ['a', 'b'], [None, 'q'])
        self.assertIs(left == changed, False)
        # Ragged mixed meson/baryon entries cannot be coerced to a dense array.
        mixed = [[0, [[1, 0, 0]]], [[[0], [2], [0]], np.zeros((3, 3), dtype=int)]]
        mixed_copy = deepcopy(mixed)
        first = Diagram(QuarkDiagram(mixed, validate=False), [0, 1], ['m', 'b'], [None, 'q'])
        second = Diagram(QuarkDiagram(mixed_copy, validate=False), np.arange(2), ['m', 'b'], [None, 'q'])
        self.assertIs(first == second, True)
        self.assertEqual(hash(first), hash(second))

    def test_propagator_copy_preserves_assumptions_and_detaches_metadata(self):
        original = Propagator('u', Tag(4201, 0), Tag(4202, 1), positive=True)
        original.annotation = {'labels': [8, 3]}
        clone = deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone, original)
        self.assertEqual(hash(clone), hash(original))
        self.assertEqual(clone.assumptions0, original.assumptions0)
        self.assertEqual(clone.source_tag, original.source_tag)
        self.assertEqual(clone.sink_tag, original.sink_tag)
        self.assertEqual(clone.flavor, original.flavor)
        clone.annotation['labels'].reverse()
        clone.flavor = 'd'
        self.assertEqual(original.annotation['labels'], [8, 3])
        self.assertEqual(original.flavor, 'u')

    def test_diagram_copy_detaches_graph_lists_and_preserves_repeated_line_links(self):
        source, sink = Symbol('copy-source'), Symbol('copy-sink')
        line = Propagator('s', Tag(4203, 0), Tag(4204, 1))
        original = Diagram(QuarkDiagram([[0, [1]], [2, 0]]),
                           [source, sink], ['left', 'right'], [None, line, line])
        original.value = {'coordinates': [0, 2]}
        graph = original.diagram
        original_times = original.time_list
        original_lines = original.propagator_list
        clone = deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone, original)
        self.assertEqual(hash(clone), hash(original))
        self.assertEqual(clone.assumptions0, original.assumptions0)
        self.assertIsNot(clone.diagram, graph)
        self.assertIsNot(clone.time_list, original_times)
        self.assertIsNot(clone.propagator_list, original_lines)
        self.assertIsNot(clone.propagator_list[1], line)
        self.assertIs(clone.propagator_list[1], clone.propagator_list[2])
        clone.diagram.adjacency_matrix[0][1].append(7)
        clone.time_list.reverse()
        clone.vertex_list[0] = 'changed'
        clone.value['coordinates'].append(1)
        self.assertEqual(graph.adjacency_matrix, [[0, [1]], [2, 0]])
        self.assertIs(original.diagram, graph)
        self.assertIs(original.time_list, original_times)
        self.assertIs(original.propagator_list, original_lines)
        self.assertEqual(original.time_list, [source, sink])
        self.assertEqual(original.vertex_list, ['left', 'right'])
        self.assertEqual(original.value, {'coordinates': [0, 2]})

    def test_graph_copy_accepts_public_hadron_vertex_labels(self):
        hadron = HadronIrrep('copy-pi', [0, 0, 0], 'A_1', -1, Tag(4205, 0))
        row = hadron[0]
        for label in (hadron, row):
            with self.subTest(type=type(label).__name__):
                label.annotation = {'coordinates': [8, 3]}
                original = Diagram(QuarkDiagram([[0, 1], [2, 0]]),
                                   [0, 1], [label, label], [None, 'forward', 'reverse'])
                clone = deepcopy(original)
                copied_label = clone.vertex_list[0]
                self.assertIsNot(copied_label, label)
                self.assertIs(copied_label, clone.vertex_list[1])
                self.assertEqual(copied_label, label)
                self.assertEqual(hash(copied_label), hash(label))
                self.assertEqual(copied_label.tag, label.tag)
                self.assertEqual(copied_label.momentum, label.momentum)
                copied_label.annotation['coordinates'].append(1)
                self.assertEqual(label.annotation['coordinates'], [8, 3])


if __name__ == '__main__':
    unittest.main()
