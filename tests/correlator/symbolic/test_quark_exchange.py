"""Scalar baryon exchange characters, independent of numerical bindings."""

from copy import deepcopy
import pickle
import unittest

import sympy as sp

from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, Tag, gen_correlator,
    iter_structured_wick_entries, quark_contract, structured_wick_templates,
)


def hadron(symmetry=None, flavor="uuu"):
    row = HadronIrrep("b", (0, 0, 0), "G1", 1, Tag(0, 0))[0]
    return Hadron(row, Flavor(flavor), quark_symmetry=symmetry)


def two_point(first, second=None, *, structured=False):
    return gen_correlator([[first], [first if second is None else second]],
                          ["sink", "source"], [False, True], structured=structured)


class QuarkExchangeSymbolTests(unittest.TestCase):
    def test_full_antisymmetry_six_wick_graphs_become_one_with_multiplicity(self):
        plain = two_point(hadron())[0, 0]
        reduced = two_point(hadron("antisymmetric"))[0, 0]
        self.assertEqual(len(plain.atoms(Diagram)), 6)
        self.assertEqual(len(reduced.atoms(Diagram)), 1)
        coefficient, _ = reduced.as_coeff_Mul()
        self.assertEqual(abs(coefficient), 6)
        one_end = two_point(hadron("antisymmetric"), hadron())[0, 0]
        self.assertEqual(len(one_end.atoms(Diagram)), 1)
        self.assertEqual(abs(one_end.as_coeff_Mul()[0]), 6)

    def test_symmetric_identical_quark_coefficient_cancels(self):
        self.assertEqual(two_point(hadron("symmetric"))[0, 0], 0)
        self.assertEqual(two_point(hadron("symmetric"), hadron("antisymmetric"))[0, 0], 0)

    def test_partial_symmetry_and_actual_flavor_stabilizer(self):
        first_pair = {(1, 0, 2): -1}
        self.assertEqual(len(two_point(hadron(first_pair))[0, 0].atoms(Diagram)), 2)
        plain = two_point(hadron(flavor="uud"))[0, 0]
        reduced = two_point(hadron(first_pair, "uud"), hadron(flavor="uud"))[0, 0]
        self.assertEqual(len(plain.atoms(Diagram)), 2)
        self.assertEqual(len(reduced.atoms(Diagram)), 1)
        # Degenerate u/d propagators do not license permuting actual flavors.
        distinct = two_point(hadron("antisymmetric", "uds"))[0, 0]
        self.assertEqual(len(distinct.atoms(Diagram)), 1)
        self.assertEqual(abs(distinct.as_coeff_Mul()[0]), 1)

    def test_mixed_baryon_meson_graph_orbits(self):
        plain = Flavor("uuu", 0) * Flavor("uu", 1) * Flavor("bar{uuu}", 2)
        tagged = (Flavor("uuu", 0, "antisymmetric") * Flavor("uu", 1)
                  * Flavor("bar{uuu}", 2, "antisymmetric"))
        self.assertEqual(len(quark_contract(plain, (0, 1, 2)).atoms(Diagram)), 24)
        self.assertEqual(len(quark_contract(tagged, (0, 1, 2)).atoms(Diagram)), 2)

    def test_dagger_time_copy_rotation_and_pickle_preserve_labels(self):
        original = hadron({(1, 0, 2): -1})
        adjoint = original.conjugate()
        self.assertIn(((0, 2, 1), -1), adjoint.irrep_row.quark_symmetry)
        self.assertEqual(adjoint.irrep_row.quark_symmetry, adjoint.flavor_structure.quark_symmetry)
        restored = adjoint.conjugate()
        self.assertEqual(restored.irrep_row, original.irrep_row)
        for value in (original.copy(), deepcopy(original), original.set_time("new"),
                      pickle.loads(pickle.dumps(original))):
            self.assertEqual(value.quark_symmetry, original.quark_symmetry)
            self.assertEqual(value.irrep_row.quark_symmetry, original.irrep_row.quark_symmetry)
            self.assertEqual(value.flavor_structure.quark_symmetry, original.flavor_structure.quark_symmetry)
        rotation = next(iter(original.irrep_row._little_group_matrix))
        rotated = original.irrep_row.transform(rotation)
        self.assertTrue(all(row.quark_symmetry == original.irrep_row.quark_symmetry
                            for row in rotated.atoms(type(original.irrep_row))))

    def test_label_is_part_of_cache_and_structured_identity(self):
        plain, tagged = hadron(), hadron("antisymmetric")
        self.assertNotEqual(plain.irrep_row, tagged.irrep_row)
        self.assertNotEqual(plain.flavor_structure, tagged.flavor_structure)
        symbolic = gen_correlator([[plain, tagged], [plain, tagged]], structured=False)
        structured = gen_correlator([[plain, tagged], [plain, tagged]], structured=True)
        for index, entries in iter_structured_wick_entries(structured):
            expression = sum(coefficient * sp.Mul(*diagrams) for coefficient, diagrams in entries)
            self.assertEqual(sp.expand(expression - symbolic[index]), 0)
        self.assertEqual([len(entries) for _, entries in structured_wick_templates(structured)],
                         [6, 1, 1, 1])

    def test_invalid_and_inconsistent_declarations_fail_at_construction(self):
        for value in ("mixed", {(0, 0, 2): 1}, {(0, 1, 2): -1},
                      {(1, 2, 0): -1}, {(1, 0, 2): 1, (0, 2, 1): -1},
                      {(1, 0, 2): True}, {(1, 0, 2): 1.0}):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    hadron(value)
        with self.assertRaisesRegex(ValueError, "baryon"):
            hadron("antisymmetric", "ud")
        self.assertEqual(hadron().irrep_row, hadron({}).irrep_row)

    def test_composite_preserves_each_baryons_own_label(self):
        tagged, plain = hadron({(1, 0, 2): -1}), hadron()
        composite = Hadron(tagged.irrep_row * plain.irrep_row,
                           tagged.flavor_structure * plain.flavor_structure)
        moved = composite.set_time("together")
        self.assertEqual({row.quark_symmetry for row in moved.irrep_row.atoms(type(tagged.irrep_row))},
                         {(), tagged.quark_symmetry})
        with self.assertRaisesRegex(ValueError, "each single baryon"):
            Hadron(composite.irrep_row, composite.flavor_structure, quark_symmetry="antisymmetric")

    def test_do_not_apply_a_combined_row_character_to_unverified_individual_rows(self):
        rows = HadronIrrep("b", (0, 0, 0), "G1", 1, Tag(0, 0))
        with self.assertRaisesRegex(ValueError, "one complete baryon row"):
            Hadron(rows[0] + rows[1], Flavor("uuu"), quark_symmetry="antisymmetric")
        # Separately verified scalar eigenrows can still be composed.
        first = Hadron(rows[0], Flavor("uuu"), quark_symmetry="antisymmetric")
        second = Hadron(rows[1], Flavor("uuu"), quark_symmetry="antisymmetric")
        combined = Hadron(first.irrep_row + second.irrep_row, first.flavor_structure)
        self.assertTrue(all(row.quark_symmetry == first.quark_symmetry
                            for row in combined.irrep_row.atoms(type(rows[0]))))
        product = Hadron(first.irrep_row * second.irrep_row,
                         first.flavor_structure * second.flavor_structure)
        self.assertEqual(product.conjugate().conjugate().flavor_structure, product.flavor_structure)
        unverified = Hadron(rows[1], Flavor("uuu"))
        with self.assertRaisesRegex(ValueError, "same ordered quark symmetry"):
            Hadron(first.irrep_row + unverified.irrep_row, first.flavor_structure)
        with self.assertRaisesRegex(ValueError, "same ordered quark symmetry"):
            Hadron(rows[0], Flavor("uuu", 0, "antisymmetric"))

    def test_zero_insertions_survive_label_and_composition_checks(self):
        tagged = hadron("antisymmetric")
        for position, flavor in ((sp.S.Zero, tagged.flavor_structure), (tagged.irrep_row, sp.S.Zero)):
            with self.subTest(position=position, flavor=flavor):
                value = Hadron(position, flavor, quark_symmetry="antisymmetric")
                self.assertEqual(value.copy().irrep_row, position)
                self.assertEqual(value.copy().flavor_structure, flavor)
                restored = value.conjugate().conjugate()
                self.assertEqual(restored.irrep_row, position)
                self.assertEqual(restored.flavor_structure, flavor)


if __name__ == "__main__":
    unittest.main()
