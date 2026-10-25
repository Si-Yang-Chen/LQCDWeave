"""S API, legacy symbolic behavior and Wick matching tests."""

import unittest
from unittest.mock import patch

import sympy as sp
from lqcdweave.operators.correlator import hadron

from lqcdweave.operators.correlator.api import (
    Diagram,
    Flavor,
    Hadron,
    HadronFlavorStructure,
    HadronIrrep,
    HadronIrrepRow,
    QuarkDiagram,
    Tag,
    diagram_simplify,
    gen_correlator,
    hadron_little_group_projection,
    quark_contract,
    remove_disconneted_diagram,
)


class SymbolicFrontendTests(unittest.TestCase):
    def test_numeric_wick_polynomial_skips_general_simplification(self):
        connected = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ['a', 'b'], [None, 'q'])
        left = Diagram(QuarkDiagram([[1]]), [0], ['a'], [None, 'q'])
        right = Diagram(QuarkDiagram([[1]]), [1], ['b'], [None, 'q'])
        expressions = (sp.S.Zero, connected, sp.Rational(3, 2) * connected,
                       -connected + 2 * left * right)
        references = tuple(sp.simplify(value) for value in expressions)
        with patch.object(hadron.sp, 'simplify', side_effect=AssertionError('redundant simplify')):
            actual = tuple(hadron._finish_correlator(value) for value in expressions)
        for value, reference in zip(actual, references, strict=True):
            self.assertEqual(sp.expand(value - reference), 0)

    def test_other_correlator_algebra_keeps_general_simplification(self):
        diagram = Diagram(QuarkDiagram([[1]]), [0], ['a'], [None, 'q'])
        z = sp.Symbol('z')
        expressions = ((z + 1) * diagram, sp.sqrt(2) * diagram, diagram ** 2,
                       (1 + sp.I) * diagram)
        for value in expressions:
            with self.subTest(value=value), patch.object(hadron.sp, 'simplify',
                                                       wraps=sp.simplify) as simplify:
                self.assertEqual(hadron._finish_correlator(value), sp.simplify(value))
                self.assertGreaterEqual(simplify.call_count, 2)

    def test_correlator_package_exports_s_api_objects(self):
        from lqcdweave.operators.correlator import (
            Diagram as PackageDiagram,
            Hadron as PackageHadron,
            gen_correlator as package_gen_correlator,
        )

        self.assertIs(PackageHadron, Hadron)
        self.assertIs(PackageDiagram, Diagram)
        self.assertIs(package_gen_correlator, gen_correlator)

    def test_public_flavor_alias_and_none_time_preservation(self):
        from lqcdweave.operators.correlator import Flavor as PackageFlavor

        self.assertIs(Flavor, HadronFlavorStructure)
        self.assertIs(PackageFlavor, Flavor)
        source_irrep = HadronIrrep("pi", (0, 0, 0), "A_1", -1, Tag(0, 2))[0]
        sink_irrep = HadronIrrep("pi", (0, 0, 0), "A_1", -1, Tag(0, 5))[0]
        expression = gen_correlator(
            [[Hadron(source_irrep, Flavor("ud", 2))],
             [Hadron(sink_irrep, Flavor("ud", 5))]],
            time_slice_list=[None, None],
            dagger_list=[False, True],
        )
        self.assertEqual(expression.shape, (1, 1))
        diagram, = expression[0, 0].atoms(Diagram)
        self.assertEqual(diagram.time_list, [2, 5])
        self.assertEqual([vertex.dagger for vertex in diagram.vertex_list], [False, True])

    def test_projection_retains_each_particle_time_and_dagger(self):
        irrep = HadronIrrep("pi", (0, 0, 0), "A_1", -1, Tag(4, 7), True)
        expressions = hadron_little_group_projection([irrep], "A_1", 0, parity=-1)
        self.assertTrue(expressions)
        rows = set().union(*(expression.atoms(HadronIrrepRow) for expression in expressions))
        self.assertTrue(rows)
        self.assertEqual({row.tag.time for row in rows}, {7})
        self.assertEqual({row.dagger for row in rows}, {True})

    def test_diagram_name_interning_and_quark_link_check(self):
        with self.assertRaisesRegex(ValueError, "Vertex 0"):
            QuarkDiagram([[0, 1], [0, 0]])
        source = HadronIrrepRow("pi", [0, 0, 0], "A_1", 0, -1, Tag(0, 0))
        sink = source.conjugate()
        fields = [source, sink]
        props = [None, "S^q"]
        first = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], fields, props)
        second = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], fields, props)
        self.assertIs(first, second)

    def test_same_visible_names_do_not_alias_distinct_insertion_tags(self):
        def generate(tag):
            row=HadronIrrepRow('tagged', [0,0,0], 'A_1', 0, -1, Tag(tag,0))
            return gen_correlator([[Hadron(row,Flavor('ud'))],[Hadron(row,Flavor('ud'))]],
                [0,1],[False,True])[0,0]
        first=generate(1)
        first_rows=tuple(tuple(graph.vertex_list) for graph in first.atoms(Diagram))
        second=generate(2)
        self.assertEqual(first_rows,tuple(tuple(graph.vertex_list) for graph in first.atoms(Diagram)))
        self.assertNotEqual(first,second)
        self.assertTrue(all(row.tag.tag==1 for graph in first.atoms(Diagram) for row in graph.vertex_list))
        self.assertTrue(all(row.tag.tag==2 for graph in second.atoms(Diagram) for row in graph.vertex_list))
        self.assertNotEqual(sp.expand(first-second),0)

    def test_source_sink_time_rewrite_uses_legacy_half_order(self):
        row = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
        template = Hadron(row, Flavor("ud"))
        halves = [[template], [template]]
        expression = gen_correlator(halves, [0, 1], [False, True])
        self.assertEqual(expression.shape, (1, 1))
        diagram, = expression[0, 0].atoms(Diagram)
        self.assertEqual(diagram.time_list, [0, 1])
        self.assertEqual([v.dagger for v in diagram.vertex_list], [False, True])
        # Time labels remain independent even when visible row names agree.
        self.assertEqual([h.irrep_row.tag.time for half in halves for h in half], [0, 1])

    def test_same_visible_row_retains_independent_time(self):
        source = HadronIrrepRow("pi", [0, 0, 0], "A_1", 0, -1, Tag(0, 0))
        insertion = HadronIrrepRow("pi", [0, 0, 0], "A_1", 0, -1, Tag(0, 1))
        self.assertEqual(str(source), str(insertion))
        self.assertIsNot(source, insertion)
        self.assertNotEqual(source, insertion)
        self.assertEqual(source.tag.time, 0)
        self.assertEqual(insertion.tag.time, 1)

    def test_identical_hadrons_follow_little_group_projection(self):
        pion = HadronIrrep("pi", [0, 0, 0], "A_1", -1, Tag(0, 0))
        rows = hadron_little_group_projection([pion, pion], "A_1", 0, parity=1)
        self.assertEqual(rows, [1])

    def test_moving_pair_projection_matches_reference_row_count(self):
        pair = [HadronIrrep("D_star", [0, 1, 1], "B_2", None, Tag(0, 0)),
                HadronIrrep("D", [0, -1, 0], "A_2", None, Tag(0, 0))]
        rows = hadron_little_group_projection(pair, "E", 0, parity=None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0].as_ordered_terms()), 2)

    def test_three_point_keeps_all_symbolic_times(self):
        rows = [HadronIrrep(name, [0, 0, 0], "A_1", -1, Tag(0, 0))[0]
                for name in ("pi_source", "insertion", "pi_sink")]
        halves = [[Hadron(row, Flavor(flavor))]
                  for row, flavor in zip(rows, ("ud", "dd", "ud"))]
        correlator = gen_correlator(halves, [0, 1, 2], [False, False, True])
        self.assertEqual(correlator.shape, (1, 1, 1))
        times = {time for diagram in correlator[0, 0, 0].atoms(Diagram)
                 for time in diagram.time_list}
        self.assertEqual(times, {0, 1, 2})

    def test_neutral_flavor_graphs_and_disconnected_filter(self):
        flavor = Flavor("uu", 0) * Flavor("uu", 1)
        expression = quark_contract(flavor, [0, 1])
        matrices = {repr(diagram.diagram.adjacency_matrix)
                    for diagram in expression.atoms(Diagram)}
        self.assertEqual(matrices, {"[[0, 2], [2, 0]]", "[[1, 0], [0, 1]]"})
        connected = remove_disconneted_diagram(
            diagram_simplify(expression), [r"S^q_\mathrm{local}"])
        self.assertEqual(
            {repr(diagram.diagram.adjacency_matrix) for diagram in connected.atoms(Diagram)},
            {"[[0, 1], [1, 0]]"},
        )

    def test_baryon_antibaryon_graph_retains_ordered_flavor_lines(self):
        flavor = Flavor("bar{uds}", 0) * Flavor("uds", 1)
        expression = diagram_simplify(quark_contract(flavor, [0, 1], degenerate=False))
        diagram, = expression.atoms(Diagram)
        self.assertEqual(diagram.time_list, [0, 1])
        self.assertEqual(diagram.propagator_list, [None, "S^d", "S^s", "S^u"])
        self.assertEqual(diagram.diagram.adjacency_matrix[0][1],
                         [[3, 0, 0], [0, 1, 0], [0, 0, 2]])

    def test_unmatched_flavor_content_is_a_symbolic_zero(self):
        self.assertEqual(quark_contract(Flavor("uds"), [0], degenerate=False), 0)
        self.assertEqual(quark_contract(Flavor("ud"), [0], degenerate=False), 0)
        self.assertEqual(
            quark_contract(Flavor("bar{uds}") * Flavor("ud"), [0, 1], degenerate=False),
            0,
        )

    def test_invalid_flavor_and_projection_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            Flavor("ux")
        with self.assertRaises(ValueError):
            Flavor("uds", -1)
        with self.assertRaises(ValueError):
            hadron_little_group_projection([], "A_1", 0)
        with self.assertRaises(ValueError):
            HadronIrrep("pi", (0, 0), "A_1", -1, Tag(0, 0))


if __name__ == "__main__":
    unittest.main()
