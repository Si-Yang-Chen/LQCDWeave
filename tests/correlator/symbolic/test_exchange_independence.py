"""Independent finite-group rank oracles for the unified contraction quotient."""

from itertools import permutations, product
import unittest

import sympy as sp

from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, QuarkDiagram, QuarkExchange, Tag, gen_correlator,
    diagram_simplify, iter_structured_wick_entries,
)
from lqcdweave.operators.correlator.quark_exchange_matrix import reduce_matrix_exchange
from correlator.symbolic.test_matrix_quark_exchange import family


def scalar(name, symmetry, flavor='uuu'):
    row = HadronIrrep(name, (0, 0, 0), 'G1', 1, Tag(0, 0))[0]
    return Hadron(row, Flavor(flavor), quark_symmetry=symmetry)


def all_pairings(left, right):
    """All pointwise kernels; deliberately no Wick weights or simplification."""
    result = []
    for permutation in permutations(range(3)):
        crossing = [[int(j == permutation[i]) for j in range(3)] for i in range(3)]
        adjacency = [[[[0]*3 for _ in range(3)], [[0]*3 for _ in range(3)]],
                     [crossing, [[0]*3 for _ in range(3)]]]
        for a, b in product(left, right):
            result.append(Diagram(QuarkDiagram(adjacency), [0, 1], [a.irrep_row, b.irrep_row], [None, 'S^q']))
    return result


def relation_rank_oracle(graphs, families, vertex_generators=()):
    """Nullity of explicit equations K(pE,a)=sum_b D(p^-1)[b,a]K(E,b).

    Construct every raw graph/component variable and every generating equation.
    This does not use canonical orbits, group averaging, or pivot columns.
    """
    states = tuple(product(*(range(len(family)) for family in families)))
    index = {(edges, state): i for i, (edges, state) in enumerate(product(graphs, states))}
    equations = []
    for edges, state in index:
        for vertex, family in enumerate(families):
            declaration = family[0].irrep_row.quark_exchange
            for p, _ in declaration.active:
                if p == (0, 1, 2):
                    continue
                moved = tuple(sorted((s, p[a] if s == vertex else a,
                                      t, p[b] if t == vertex else b, prop)
                                     for s, a, t, b, prop in edges))
                inverse = tuple(p.index(i) for i in range(3))
                matrix = declaration.representation.matrices[inverse]
                row = {index[moved, state]: sp.S.One}
                for partner in range(len(family)):
                    replaced = (*state[:vertex], partner, *state[vertex+1:])
                    col = index[edges, replaced]
                    row[col] = row.get(col, 0)-matrix[partner, state[vertex]]
                equations.append(row)
        for p in vertex_generators:
            moved = tuple(sorted((p[s], a, p[t], b, prop) for s, a, t, b, prop in edges))
            replaced = [None]*len(state)
            for i, j in enumerate(p):
                replaced[j] = state[i]
            a, b = index[moved, tuple(replaced)], index[edges, state]
            row = {a: sp.S.One}
            row[b] = row.get(b, 0)-1
            equations.append(row)
    relations = sp.MutableSparseMatrix(len(equations), len(index), {})
    for i, row in enumerate(equations):
        for j, coefficient in row.items():
            relations[i, j] = coefficient
    return len(index)-relations.rank()


def colored_pairings(families):
    """Enumerate flavor-compatible labeled matchings, independently of Wick."""
    half = len(families)//2
    words = [family[0].irrep_row.quark_exchange.flavors for family in families]
    sinks = tuple((v, a, f) for v in range(half) for a, f in enumerate(words[v]))
    sources = tuple((v, a, f) for v in range(half, 2*half) for a, f in enumerate(words[v]))
    props = {f: i+1 for i, f in enumerate(sorted(set(f for _, _, f in sinks)))}
    graphs = []
    for matching in permutations(sinks):
        if all(source[2] == sink[2] for source, sink in zip(sources, matching)):
            graphs.append(tuple(sorted((s, a, t, b, props[f])
                                       for (s, a, f), (t, b, _) in zip(sources, matching))))
    diagrams = []
    for edges in graphs:
        adjacency = [[[[0]*3 for _ in range(3)] for _ in families] for _ in families]
        for s, a, t, b, prop in edges:
            adjacency[s][t][a][b] = prop
        for chosen in product(*families):
            diagrams.append(Diagram(QuarkDiagram(adjacency), [0]*half+[1]*half,
                                    [h.irrep_row for h in chosen], [None, *props]))
    return tuple(graphs), diagrams


class ExchangeIndependenceTests(unittest.TestCase):
    def test_all_three_quark_components_and_all_two_quark_pairs_are_available(self):
        rep = QuarkExchange('all')
        self.assertEqual(rep.dimension, 4)
        self.assertEqual(rep.matrices[(1, 0, 2)], sp.diag(1, -1, 1, -1))
        for slots in ((0, 1), (0, 2), (1, 2)):
            for name, dimension in (('symmetric', 1), ('antisymmetric', 1), ('all', 2)):
                with self.subTest(slots=slots, name=name):
                    value = QuarkExchange.pair(name, slots=slots)
                    self.assertEqual(value.dimension, dimension)
                    self.assertEqual(len(value.elements), 2)

    def test_joint_scalar_and_matrix_relations_select_one_common_kernel(self):
        rep = QuarkExchange().tensor_product(QuarkExchange())
        for matrix_at_sink in (False, True):
            mixed, antisymmetric = family('mixed', rep), scalar('scalar', 'antisymmetric')
            axes = ([list(mixed), [antisymmetric]] if matrix_at_sink else [[antisymmetric], list(mixed)])
            value = gen_correlator([list(axis) for axis in axes])
            nonzero = [item for item in value.flat if item != 0]
            self.assertEqual(len(nonzero), 2)
            self.assertEqual(sp.expand(sum(nonzero)), 0)
            self.assertEqual(len(set().union(*(item.atoms(Diagram) for item in value.flat))), 1)
            structured = gen_correlator([list(axis) for axis in axes], structured=True)
            atoms = {diagram for _, entries in iter_structured_wick_entries(structured)
                     for _, diagrams in entries for diagram in diagrams}
            self.assertEqual(len(atoms), 1)

    def test_whole_projected_output_matrix_uses_joint_component_relations(self):
        rep = QuarkExchange().tensor_product(QuarkExchange())
        mixed = family('mixed', rep)
        source = scalar('source', 'antisymmetric')
        projected = [Hadron(expression, mixed[0].flavor_structure) for expression in
                     (mixed[1].irrep_row+mixed[2].irrep_row,
                      mixed[1].irrep_row+sp.I*mixed[2].irrep_row,
                      2*mixed[1].irrep_row-mixed[2].irrep_row)]
        value = gen_correlator([projected, [source]])
        self.assertEqual(value[0, 0], 0)
        self.assertEqual(sp.expand(3*value[1, 0]-(1-sp.I)*value[2, 0]), 0)
        self.assertEqual(len(set().union(*(item.atoms(Diagram) for item in value.flat))), 1)
        structured = gen_correlator([projected, [source]], structured=True)
        for index, entries in iter_structured_wick_entries(structured):
            actual = sp.Add(*(weight*sp.Mul(*diagrams) for weight, diagrams in entries))
            self.assertEqual(sp.expand(actual-value[index]), 0)

    def test_structured_radical_projection_cancels_before_numeric_conversion(self):
        ms, ma = family('b', 'mixed', 'udu')
        projected = Hadron(ms.irrep_row-sp.sqrt(3)*ma.irrep_row, ms.flavor_structure)
        plain = scalar('plain', None, 'udu')
        self.assertEqual(gen_correlator([[projected], [plain]])[0, 0], 0)
        structured = gen_correlator([[projected], [plain]], structured=True)
        self.assertEqual(tuple(iter_structured_wick_entries(structured)), (((0, 0), ()),))

    def test_pointwise_single_mixed_endpoint_has_exactly_two_independent_kernels(self):
        mixed = family('mixed')
        plain = [scalar('plain', None).conjugate()]
        memo = {}
        images = [reduce_matrix_exchange(value, cache=memo) for value in all_pairings(mixed, plain)]
        basis = sorted(set().union(*(value.atoms(Diagram) for value in images)), key=str)
        self.assertEqual(len(basis), 2)
        matrix = sp.Matrix([[value.coeff(diagram) for diagram in basis] for value in images])
        self.assertEqual(matrix.rank(), 2)
        for value in images:
            self.assertEqual(sp.expand(reduce_matrix_exchange(value)-value), 0)

    def test_both_endpoints_match_independent_character_inner_product_rank(self):
        # Frobenius reciprocity: the pointwise quotient for three equal-flavor
        # lines has dimension <chi_left, chi_right>. This oracle does not use
        # the reducer's orbit, stabilizer projector, or pivot selection.
        names = ('symmetric', 'antisymmetric', 'mixed', 'all')
        for l_name, r_name in product(names, repeat=2):
            with self.subTest(left=l_name, right=r_name):
                l_rep, r_rep = QuarkExchange(l_name), QuarkExchange(r_name)
                left, right = family('left', l_rep), [h.conjugate() for h in family('right', r_rep)]
                scalar_product = sum(sp.trace(l)*sp.trace(r_rep.matrices[p])
                                     for p, l in l_rep.elements)/6
                expected = int(scalar_product)
                memo = {}
                images = [reduce_matrix_exchange(value, cache=memo) for value in all_pairings(left, right)]
                basis = sorted(set().union(*(value.atoms(Diagram) for value in images)), key=str)
                coefficients = sp.Matrix([[sp.expand(value).coeff(d) for d in basis] for value in images])
                self.assertEqual(len(basis), expected)
                self.assertEqual(coefficients.rank(), expected)

    def test_two_identical_odd_baryon_insertions_cancel_under_vertex_exchange(self):
        baryon = scalar('b', 'antisymmetric')
        composite = Hadron(baryon.irrep_row*baryon.irrep_row,
                           baryon.flavor_structure*baryon.flavor_structure)
        result = gen_correlator([[composite], [composite]])
        self.assertEqual(result[0, 0], 0)

    def assert_relation_rank(self, families, vertex_generators=(), *, factorized=False):
        graphs, diagrams = colored_pairings(families)
        expected = relation_rank_oracle(graphs, families, vertex_generators)
        memo = {}
        images = [reduce_matrix_exchange(diagram, cache=memo) for diagram in diagrams]
        basis = sorted(set().union(*(image.atoms(Diagram) for image in images)), key=str)
        coefficients = sp.Matrix([[sp.expand(image).coeff(d) for d in basis] for image in images])
        self.assertEqual(len(basis), expected)
        self.assertEqual(coefficients.rank(), expected)
        for image in images:
            self.assertEqual(sp.expand(reduce_matrix_exchange(image)-image), 0)
        if factorized:
            # The public generator splits disconnected graphs before exchange
            # reduction. Its polynomial monomials must have the same quotient
            # dimension as the independent unsplit linear-equation oracle.
            factored = [reduce_matrix_exchange(diagram_simplify(diagram), cache=memo) for diagram in diagrams]
            primitives = sorted(set().union(*(value.atoms(Diagram) for value in factored)), key=str)
            polynomials = [sp.Poly(value, *primitives) for value in factored]
            monomials = sorted(set().union(*(polynomial.monoms() for polynomial in polynomials)))
            coefficients = sp.Matrix([[polynomial.coeff_monomial(monomial) for monomial in monomials]
                                      for polynomial in polynomials])
            self.assertEqual(coefficients.rank(), expected)
            for full, separate in zip(images, factored):
                self.assertEqual(sp.expand(reduce_matrix_exchange(diagram_simplify(full), cache=memo)-separate), 0)

    def test_changed_complex_basis_and_heterogeneous_subgroups_match_relation_rank(self):
        change = sp.Matrix([[1, sp.I], [0, 2]])
        mixed = QuarkExchange({p: change.inv()*m*change for p, m in QuarkExchange().elements})
        for slots in ((0, 1), (0, 2), (1, 2)):
            with self.subTest(slots=slots):
                pair = QuarkExchange.pair(slots=slots)
                self.assert_relation_rank((family('left', mixed),
                                           tuple(h.conjugate() for h in family('right', pair))))

    def test_all_same_flavor_pair_placements_match_relation_rank_with_distinct_flavor_lines(self):
        for flavor, slots in (('uud', (0, 1)), ('udu', (0, 2)), ('duu', (1, 2))):
            with self.subTest(flavor=flavor):
                self.assert_relation_rank((family('left', 'all', flavor), tuple(h.conjugate() for h in
                                           family('right', QuarkExchange.pair(slots=slots), flavor))))

    def test_multiple_identical_vertices_match_full_relation_rank(self):
        # Four baryon vertices, 48 colored matchings. Identical vertices may
        # exchange while leg symmetries operate at every insertion.
        left = family('b', QuarkExchange.pair('antisymmetric'), 'uud')
        right = tuple(h.conjugate() for h in left)
        self.assert_relation_rank((left, left, right, right), ((1, 0, 2, 3), (0, 1, 3, 2)))

    def test_zero_insertions_are_preserved_in_both_frontends(self):
        nonzero = scalar('b', 'antisymmetric')
        for position, flavor in ((sp.S.Zero, nonzero.flavor_structure),
                                 (nonzero.irrep_row, sp.S.Zero)):
            zero = Hadron(position, flavor)
            self.assertEqual(gen_correlator([[zero], [nonzero]])[0, 0], 0)
            structured = gen_correlator([[zero], [nonzero]], structured=True)
            self.assertEqual(tuple(iter_structured_wick_entries(structured)), (((0, 0), ()),))

    def test_multi_baryon_output_has_exact_odd_vertex_exchange_signs(self):
        values = family('b', 'mixed', 'uds')
        axis = [Hadron(a.irrep_row*b.irrep_row, a.flavor_structure*b.flavor_structure)
                for a, b in product(values, repeat=2)]
        matrix = gen_correlator([list(axis), list(axis)], degenerate=False)
        self.assertNotEqual(matrix[1, 1], 0)
        for i in (0, 3):
            self.assertTrue(all(matrix[i, j] == matrix[j, i] == 0 for j in range(4)))
        self.assertEqual(sp.expand(matrix[1, 1]+matrix[2, 1]), 0)
        self.assertEqual(sp.expand(matrix[1, 1]+matrix[1, 2]), 0)
        self.assertEqual(sp.expand(matrix[1, 1]-matrix[2, 2]), 0)


if __name__ == '__main__':
    unittest.main()
