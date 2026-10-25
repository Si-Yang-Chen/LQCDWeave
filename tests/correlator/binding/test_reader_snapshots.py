"""Reader snapshots preserve vertex conventions and isolate mutable core registries."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental, ArrayBaryonElemental
from lqcdweave.contraction.binding.api import Baryon, Current
from lqcdweave.contracts.elemental import ElementalComponent, BaryonComponent
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, SparseSpinTensor


class ReaderSnapshotTests(unittest.TestCase):
    def test_same_unknown_reader_retains_explicit_load_key(self):
        component = ElementalComponent(0)
        operator = Operator.from_terms("local", [OperatorTerm(1, component, 0)])
        reader = ArrayElemental({"v2v": np.ones((1, 1, 2, 2, 2), complex)},
                                (component,), phase_convention="midpoint")
        loaded = Current(reader, operator).load("explicit", usedNe=1)
        snapshot = loaded.with_reader(reader)
        self.assertEqual((snapshot.key, snapshot.usedNe), ("explicit", 1))
        self.assertIsNone(reader.cfg)  # Declaring a key does not invent scientific identity.

    def test_unbound_current_mode_view_keeps_source_and_selected_metadata(self):
        component = ElementalComponent(0)
        operator = Operator.from_terms("local", [OperatorTerm(1, component, 0)])
        declaration = object()
        template = Current(declaration, operator, True, boundary="open")
        selected = template.with_mode_indices(((1,), (0, 1)))
        self.assertIsInstance(selected, Current)
        self.assertIs(selected.elemental, declaration)
        reader = ArrayElemental({"v2v": np.ones((1, 1, 2, 2, 2), complex)},
                                (component,), phase_convention="midpoint", cfg="a")
        selected.usedNe, selected.usedNp = 2, 7
        with patch.object(reader, "block", side_effect=AssertionError("snapshot read data")):
            bound = selected.with_reader(reader, configuration="a")
            description, = bound.describe_factors(0)
        self.assertIsInstance(bound, Current)
        self.assertEqual(bound.mode_indices, ((0, 1), (1,)))
        self.assertEqual(description.mode_indices, bound.mode_indices)
        self.assertEqual((bound.usedNe, bound.usedNp, bound.boundary), (2, 7, "open"))
        self.assertIsNone(selected.reader)
        self.assertIsNone(selected.key)
        with self.assertRaisesRegex(ValueError, "cfg"):
            selected.with_reader(reader, configuration="b")
        wrong_phase = ArrayElemental({"v2v": np.ones((1, 1, 2, 2, 2), complex)},
                                    (component,), phase_convention="anchor", cfg="a")
        with self.assertRaisesRegex(ValueError, "phase"):
            selected.with_reader(wrong_phase)
        self.assertIsNone(selected.reader)

    def test_baryon_reader_view_keeps_spin_basis_cuts_and_independent_cores(self):
        component = BaryonComponent.paths()
        spin = SparseSpinTensor(((0, 1, 2, 1j),))
        operator = Operator.from_terms("baryon", [OperatorTerm.from_baryon(component, spin)])
        reader = ArrayBaryonElemental({"vvv": np.ones((1, 1, 2, 2, 2, 2), complex)},
                                      (component,), cfg="a")
        template = Baryon(object(), operator, True, spin_basis="dirac_pauli", boundary="open")
        template.usedNe = (1, 2, 1)
        template.cores["old"] = object()
        with patch.object(reader, "block", side_effect=AssertionError("snapshot read data")):
            first = template.with_reader(reader, configuration="a")
            second = template.with_reader(reader, configuration="a")
            first.describe_factors(0)
        self.assertIs(first.operator, operator)
        self.assertEqual((first.usedNe, first.spin_basis, first.boundary, first.dagger),
                         ((1, 2, 1), "dirac_pauli", "open", True))
        self.assertEqual(first.cores, {})
        first.cores["new"] = object()
        self.assertEqual(second.cores, {})
        self.assertEqual(tuple(template.cores), ("old",))
        self.assertIsNone(template.reader)
        with self.assertRaisesRegex(ValueError, "cfg"):
            template.with_reader(reader, configuration="b")


if __name__ == "__main__":
    unittest.main()
