"""Archived preparation defaults preserve old general elemental recipes."""

import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import ArrayConstant, TensorPreparation
from lqcdweave.storage.contraction_plans.api import (
    PlanArchiveError, decode_plan_value, encode_plan_value,
)


class TensorPreparationArchiveTests(unittest.TestCase):
    def recipe(self, structure="general"):
        spin = ArrayConstant.from_array(np.arange(16, dtype=np.complex128).reshape(4, 4))
        return TensorPreparation("spin_spatial", ("raw",), (spin,),
                                 True, (1, 0), spatial_structure=structure)

    def test_missing_structure_restores_general_recipe(self):
        original = self.recipe()
        encoded = encode_plan_value(original)
        encoded["value"].pop("spatial_structure")
        self.assertEqual(decode_plan_value(encoded), original)

    def test_explicit_structures_round_trip(self):
        for structure in ("general", "diagonal", "point_diagonal"):
            with self.subTest(structure=structure):
                original = self.recipe(structure)
                self.assertEqual(decode_plan_value(encode_plan_value(original)), original)

    def test_slab_selection_round_trip_and_old_recipe_default(self):
        from dataclasses import replace
        original = replace(self.recipe(), raw_index=2, compute_dtype='complex64')
        self.assertEqual(decode_plan_value(encode_plan_value(original)), original)
        encoded = encode_plan_value(self.recipe())
        encoded['value'].pop('raw_index')
        self.assertIsNone(decode_plan_value(encoded).raw_index)
        for index in (-1, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                replace(original, raw_index=index)

    def test_default_compatibility_does_not_relax_other_fields(self):
        for alteration in ("unknown", "required", "invalid"):
            with self.subTest(alteration=alteration):
                encoded = encode_plan_value(self.recipe())
                if alteration == "unknown":
                    encoded["value"]["future_field"] = True
                elif alteration == "required":
                    encoded["value"].pop("kind")
                else:
                    encoded["value"]["spatial_structure"] = "invalid"
                with self.assertRaises(PlanArchiveError):
                    decode_plan_value(encoded)


if __name__ == "__main__":
    unittest.main()
