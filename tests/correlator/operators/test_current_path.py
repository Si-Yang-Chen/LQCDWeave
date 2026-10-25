"""Module tests split from test_diagram_expansion.py; shared numerical fixtures stay unchanged."""

from test_diagram_expansion import (
    ConservedVectorInsertion,
    OperatorPath,
    _DiagramExpansionMetadataFixture,
)


class DiagramExpansionMetadataTests(_DiagramExpansionMetadataFixture):
    def test_operator_path_keeps_original_current_endpoints_and_term_groups(self):
        current = ConservedVectorInsertion(direction=3, wilson_r=1.5)
        operator = OperatorPath.from_current(current, direction=3, momentum_index=2)
        self.assertEqual(operator.current_terms, current.terms)
        self.assertEqual(len(operator.current_term_ranges), 2)
        self.assertEqual(tuple(map(len, operator.current_term_ranges)), (2, 2))
        forward, backward = operator.current_terms
        self.assertEqual(forward.field_offset, (0, 0, 0, 1))
        self.assertEqual(backward.bar_offset, (0, 0, 0, 1))
        self.assertTrue(backward.link_dagger)
        self.assertEqual(operator.path_terms[0][3], forward.coefficient * forward.wilson_r)
