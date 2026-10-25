"""Exact matrix exchange declarations and coupled symbolic/structured orbits."""

from copy import deepcopy
import pickle
import unittest

import sympy as sp

from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, QuarkExchange, Tag, gen_correlator,
    iter_structured_wick_entries,
)


def family(name="b", representation="mixed", flavor="uuu"):
    rep = QuarkExchange(representation)
    plain = tuple(Hadron(HadronIrrep(f"{name}{i}", (0, 0, 0), "G1", 1, Tag(0, 0))[0],
                         Flavor(flavor)) for i in range(rep.dimension))
    return Hadron.multiplet(plain, quark_symmetry=rep)


class MatrixQuarkExchangeTests(unittest.TestCase):
    def test_standard_ms_ma_matrices_close_s3_and_do_mix_components(self):
        rep = QuarkExchange()
        p, q = rep.matrices[(1, 0, 2)], rep.matrices[(0, 2, 1)]
        self.assertEqual(p, sp.diag(1, -1))
        self.assertNotEqual(q[0, 1], 0)
        self.assertEqual(sp.simplify((p*q)**3), sp.eye(2))
        self.assertEqual(len(rep.elements), 6)
        with self.assertRaises(TypeError):
            rep.matrices[(0, 1, 2)] = sp.zeros(2)

    def test_reject_inconsistent_dimensions_relations_and_inexact_matrices(self):
        for value in ({(1, 0, 2): [[1, 1], [0, 1]]},
                      {(1, 2, 0): -sp.eye(2)}, {(1, 0, 2): [[1.0]]},
                      {(1, 0, 2): sp.eye(2), (0, 2, 1): sp.eye(3)},
                      {(0, 0, 2): sp.eye(2)}, {(1, 0, 2): [[sp.Symbol('x')]]}):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                QuarkExchange(value)
        with self.assertRaisesRegex(ValueError, 'one baryon'):
            Hadron.multiplet(family()[:1])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            row = Hadron(HadronIrrep('b', (0, 0, 0), 'G1', 1, Tag(0, 0))[0], Flavor('uuu'))
            Hadron.multiplet([row, row])

    def test_pure_mixed_cancels_with_one_or_both_ends_declared(self):
        tagged = family()
        plain = Hadron(HadronIrrep('plain', (0, 0, 0), 'G1', 1, Tag(0, 0))[0], Flavor('uuu'))
        for source in ([plain], list(family('source'))):
            result = gen_correlator([list(tagged), source])
            self.assertTrue(all(value == 0 for value in result.flat))
        scalar = Hadron(plain.irrep_row, plain.flavor_structure, quark_symmetry='antisymmetric')
        self.assertTrue(all(value == 0 for value in gen_correlator([list(tagged), [scalar]]).flat))

    def test_actual_flavor_stabilizer_retains_ma_but_cancels_ms(self):
        values = family(flavor='uud')
        plain = Hadron(HadronIrrep('plain', (0, 0, 0), 'G1', 1, Tag(0, 0))[0], Flavor('uud'))
        result = gen_correlator([list(values), [plain]])
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(len(result[1, 0].atoms(Diagram)), 1)
        both = gen_correlator([list(values), list(values)])
        self.assertTrue(all(both[i, j] == 0 for i, j in ((0, 0), (0, 1), (1, 0))))
        self.assertEqual(len(both[1, 1].atoms(Diagram)), 1)
        distinct = family(flavor='uds')
        result = gen_correlator([list(distinct), list(distinct)])
        self.assertTrue(all(len(value.atoms(Diagram)) == 1 for value in result.flat))

    def test_baryon_flavor_adjoint_reverses_order_together_with_exchange_slots(self):
        for word in ('uud', 'udu', 'duu', 'uds'):
            value = Flavor(word)
            self.assertEqual(value.conjugate().flavor_str, 'bar{'+word[::-1]+'}')
            self.assertEqual(value.conjugate().conjugate(), value)
        value = family(flavor='uud')[1].conjugate()
        self.assertEqual(value.flavor_structure.flavor_str, 'bar{duu}')
        self.assertIn((0, 2, 1), dict(value.irrep_row.quark_exchange.active))

    def test_copy_time_rotation_pickle_and_complex_adjoint_keep_family_identity(self):
        change = sp.Matrix([[1, sp.I], [0, 1]])
        rep = QuarkExchange({p: change.inv()*m*change for p, m in QuarkExchange().elements})
        original = family(representation=rep)[0]
        for value in (original.copy(), deepcopy(original), pickle.loads(pickle.dumps(original)),
                      original.set_time('new')):
            self.assertEqual(value.irrep_row.quark_exchange, original.irrep_row.quark_exchange)
        adjoint = original.conjugate()
        expected = {tuple(2-p[2-i] for i in range(3)): m.conjugate() for p, m in rep.elements}
        self.assertTrue(all(sp.simplify(m - expected[p]) == sp.zeros(2)
                            for p, m in adjoint.irrep_row.quark_exchange.representation.elements))
        self.assertEqual(adjoint.conjugate().irrep_row, original.irrep_row)
        rotation = next(iter(original.irrep_row.little_group_matrix))
        rotated = original.irrep_row.transform(rotation)
        self.assertTrue(all(r.quark_exchange == original.irrep_row.quark_exchange
                            for r in rotated.atoms(type(original.irrep_row))))
        with self.assertRaisesRegex(ValueError, 'ordered flavor'):
            Hadron(original.irrep_row, Flavor('uud'))
        with self.assertRaisesRegex(ValueError, 'scalar or matrix'):
            Hadron(original.irrep_row, original.flavor_structure, quark_symmetry='antisymmetric')

    def test_structured_and_symbolic_agree_for_nonzero_tensor_product_multiplets(self):
        rep = QuarkExchange().tensor_product(QuarkExchange())
        a, b = family('a', rep), family('b', rep)
        symbolic = gen_correlator([list(a), list(b)])
        structured = gen_correlator([list(a), list(b)], structured=True)
        for index, entries in iter_structured_wick_entries(structured):
            value = sum(weight*sp.Mul(*diagrams) for weight, diagrams in entries)
            difference = sp.expand(value-symbolic[index])
            self.assertTrue(all(abs(complex(difference.coeff(diagram))) < 1e-12
                                for diagram in difference.atoms(Diagram)))
        self.assertTrue(any(value != 0 for value in symbolic.flat))


if __name__ == '__main__':
    unittest.main()
