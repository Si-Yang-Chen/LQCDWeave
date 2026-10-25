"""Directed current contract and endpoint checks from the old current suite."""

import unittest

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.contracts.current import (
    CURRENT_ELEMENTAL_SCHEMA, DirectedCurrentBasis, build_current_raw_contract,
    current_raw_cache_key, validate_current_raw_contract,
)
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.contraction.directed_current import resolve_current_term_endpoints
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.single_hadron import ConservedVectorInsertion as PublicConservedVectorInsertion
from lqcdweave.operators.single_hadron.current import (
    CurrentTerm, ConservedVectorCurrent, ConservedVectorInsertion,
    coerce_current_term,
)




class DirectedCurrentContractTests(unittest.TestCase):
    def test_term_fields_and_boundary_resolution(self):
        forward, backward = ConservedVectorCurrent().terms[6:8]
        self.assertEqual(forward.term_time_offsets(), (0, 1))
        self.assertEqual(backward.term_time_offsets(), (1, 0))
        self.assertEqual(coerce_current_term(forward.as_dict()), forward)
        self.assertEqual(resolve_current_term_endpoints(
            forward, anchor_time=2, temporal_extent=3, boundary="periodic")["field_time"], 0)
        self.assertEqual(resolve_current_term_endpoints(
            backward, anchor_time=2, temporal_extent=3, boundary="periodic")["bar_time"], 0)
        with self.assertRaises(IndexError):
            resolve_current_term_endpoints(
                forward, anchor_time=2, temporal_extent=3, boundary="open")
        self.assertEqual(resolve_current_term_endpoints(
            forward, anchor_time=2, boundary="unbounded")["field_time"], 3)
        with self.assertRaises(ValueError):
            CurrentTerm(1, 0, (1, 0, 0, 1), 1, temporal_point_split=True)

    def test_insertion_operator_uses_existing_elemental_reader_protocol(self):
        components = tuple(ElementalComponent.path(GaugePath((step,))) for step in (6, 7))
        values = np.ones((2, 1, 3, 2, 2), dtype=np.complex128)
        reader = ArrayElemental({"v2v": values}, components,
                                phase_convention="midpoint", cfg="cfg")
        insertion = ConservedVectorInsertion(3)
        factors = tuple(Meson(reader, insertion.operator()).load("cfg").factors(0))
        self.assertEqual(len(factors), 4)
        self.assertEqual([factor.endpoint_times for factor in factors],
                         [(0, 1), (0, 1), (1, 0), (1, 0)])
        with self.assertRaisesRegex(ValueError, "phase conventions differ"):
            Meson(ArrayElemental({"v2v": values}, components,
                                 phase_convention="anchor", cfg="cfg"),
                  insertion.operator()).load("cfg")


if __name__ == "__main__":
    unittest.main()
