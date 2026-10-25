"""Module tests split from test_directed_current_contract.py; shared numerical fixtures stay unchanged."""

from test_directed_current_contract import (
    ConservedVectorCurrent,
    ConservedVectorInsertion,
    PublicConservedVectorInsertion,
    unittest,
)


class DirectedCurrentContractTests(unittest.TestCase):
    def test_dedicated_insertion_keeps_direction_momentum_and_endpoints(self):
        self.assertIs(PublicConservedVectorInsertion, ConservedVectorInsertion)
        insertion = ConservedVectorInsertion(3, momentum_index=2, wilson_r=1.25)
        self.assertEqual(insertion.terms, ConservedVectorCurrent(wilson_r=1.25).terms[6:8])
        operator = insertion.operator()
        self.assertEqual({term.momentum for term in operator.terms}, {2})
        self.assertEqual([term.anchor_time_offset for term in operator.terms],
                         [0, 0, 1, 1])
        self.assertEqual([term.component.time_offsets for term in operator.terms],
                         [(0, 1), (0, 1), (0, -1), (0, -1)])
        with self.assertRaises(ValueError):
            ConservedVectorInsertion(4)
        with self.assertRaises(ValueError):
            ConservedVectorInsertion(3, momentum_index=-1)
        with self.assertRaises(ValueError):
            ConservedVectorInsertion(3, wilson_r=1 + 1j)
