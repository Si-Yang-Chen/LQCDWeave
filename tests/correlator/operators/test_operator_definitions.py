"""O-only insertion-to-operator construction and endpoint tests."""

import unittest

from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent
from lqcdweave.operators.single_hadron.api import (
    ConservedVectorInsertion,
    ConservedVectorOperator,
    DerivativeName,
    GammaName,
    Insertion,
    InsertionTerm,
    Operator,
    OperatorTerm,
    ProjectionName,
    SparseSpinTensor,
    gamma,
    spin_basis_matrix,
)
from lqcdweave.operators.single_hadron.mom_dict import momDict_mom1


class OperatorDefinitionTests(unittest.TestCase):
    def test_legacy_meson_insertion_call_sequence(self):
        insertion = Insertion(
            GammaName.PI,
            DerivativeName.IDEN,
            ProjectionName.A1,
            momDict_mom1,
        )
        operator = Operator("pi", [insertion[0](0, 0, 0)], [1.0])
        self.assertEqual([(term.gamma, term.momentum) for term in operator.terms], [(15, 0)])

    def test_generic_insertion_builds_ordered_three_leg_operator(self):
        component = BaryonComponent.paths(((0,), (1,), (2,)))
        spin = SparseSpinTensor(((0, 1, 2, 1 / 2), (2, 1, 0, -1j)))
        insertion = Insertion.from_rows(
            [[InsertionTerm(component, coefficient=2j, spin=spin)]],
            {0: (0, 0, 0)},
        )
        operator = Operator("n", [insertion[0](0, 0, 0)], [-3])
        term, = operator.terms
        self.assertEqual(term.rank, 3)
        self.assertIsNone(term.gamma)
        self.assertFalse(hasattr(term, "flavors"))
        self.assertIs(term.component, component)
        self.assertIs(term.spin, spin)
        self.assertEqual(term.momentum, 0)
        self.assertEqual(term.coefficient, -6j)
        self.assertEqual(term.endpoint_time_offsets, (0, 0, 0))
        self.assertEqual(tuple(step for path in term.component.key for step in path.steps), (0, 1, 2))

    def test_conserved_vector_operator_matches_manual_operator(self):
        self.assertTrue(issubclass(ConservedVectorOperator, Operator))
        current = ConservedVectorOperator(3, momentum_index=2, wilson_r=1.25)
        self.assertTrue(all(term.current_term is not None for term in current.terms))
        self.assertEqual([len(group) for group in current.current_term_ranges], [2, 2])
        self.assertEqual(current.current_terms[0].field_offset, (0, 0, 0, 1))
        self.assertEqual(current.current_terms[1].bar_offset, (0, 0, 0, 1))
        self.assertEqual(current.current_terms[0].displacement, (0, 0, 0, 1))
        self.assertEqual(current.current_terms[1].displacement, (0, 0, 0, -1))
        self.assertFalse(current.current_terms[0].link_dagger)
        self.assertTrue(current.current_terms[1].link_dagger)
        self.assertEqual({term.momentum for term in current.terms}, {2})
        manual = Operator.from_terms("manual", tuple(current.terms))
        self.assertEqual(manual.terms, current.terms)
        self.assertEqual(manual.current_terms, current.current_terms)
        self.assertEqual(manual.current_term_ranges, current.current_term_ranges)
        compatible = Operator("legacy", [ConservedVectorInsertion(3, 2, 1.25)], [1])
        self.assertEqual(compatible.terms, current.terms)
        weighted = Operator("weighted", [ConservedVectorInsertion(3, 2, 1.25)], [2j])
        self.assertEqual(
            tuple(term.coefficient for term in weighted.terms),
            tuple(2j * term.coefficient for term in current.terms),
        )
        self.assertEqual(
            tuple(term.current_term.coefficient for term in weighted.terms),
            tuple(2j * term.current_term.coefficient for term in current.terms),
        )

    def test_spin_math_helpers_and_mesonic_term_rank(self):
        from lqcdweave.operators.single_hadron import InsertionTerm as PackageInsertionTerm

        self.assertEqual(spin_basis_matrix("degrand_rossi").shape, (4, 4))
        self.assertEqual(gamma(0).shape, (4, 4))
        self.assertEqual(OperatorTerm(0, ElementalComponent(0), 0).rank, 2)
        self.assertIs(PackageInsertionTerm, InsertionTerm)

    def test_rank_and_profile_errors_are_reported_at_construction(self):
        component = BaryonComponent.paths()
        spin = SparseSpinTensor(((0, 1, 2, 1),))
        with self.assertRaises(ValueError):
            OperatorTerm(None, ElementalComponent(0), 0)
        with self.assertRaises(ValueError):
            InsertionTerm(component, profile=object(), spin=spin)
        mixed = (
            OperatorTerm(0, ElementalComponent(0), 0),
            OperatorTerm.from_baryon(component, spin),
        )
        with self.assertRaisesRegex(ValueError, "mix"):
            Operator.from_terms("mixed", mixed)


if __name__ == "__main__":
    unittest.main()
