"""Sampling equalities require common support, pool and actual point basis."""

from types import SimpleNamespace
import unittest

from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.expansion.sampling_binding import describe_point_equalities


class PointDiagonalBindingTests(unittest.TestCase):
    def description(self, diagonal=True, bindings=None):
        return SimpleNamespace(tensor=TensorAxisMetadata(
            ("point_left", "point_right"), (1, 1), "complex128", None, "point-pair",
            point_diagonal=diagonal, axis_bindings=bindings or {},
        ))

    def equalities(self, descriptions, right_pool="pool"):
        return describe_point_equalities(
            (tuple(descriptions),), ({"left": "p", "right": "p"},),
            {(0, "field"): "pool", (0, "bar"): right_pool},
            {pool: SamplingPoolSpec(pool, 9, 1) for pool in ("pool", right_pool)},
        )

    def test_all_terms_must_declare_support_in_one_actual_pool(self):
        self.assertEqual(self.equalities((self.description(),)), {"pool": [(0, 1)]})
        self.assertEqual(self.equalities((self.description(), self.description(False))), {})
        self.assertEqual(self.equalities((self.description(),), "independent-pool"), {})

    def test_known_basis_ids_and_type_aware_labels_must_agree(self):
        same = {"point_left": ("basis", (1,)), "point_right": ("basis", (1,))}
        self.assertEqual(self.equalities((self.description(bindings=same),)),
                         {"pool": [(0, 1)]})
        for right in (("other-basis", (1,)), ("basis", (True,)), ("basis", (2,))):
            with self.subTest(right=right):
                bindings = {"point_left": ("basis", (1,)), "point_right": right}
                self.assertEqual(self.equalities((self.description(bindings=bindings),)), {})
        self.assertEqual(self.equalities((self.description(bindings={
            "point_left": ("basis", (1,)),
        }),)), {})


if __name__ == "__main__":
    unittest.main()
