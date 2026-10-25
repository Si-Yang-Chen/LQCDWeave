"""Point support survives archives; older tensor headers default to general."""

import unittest

from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.storage.contraction_plans.api import decode_plan_value, encode_plan_value


class PointDiagonalArchiveTests(unittest.TestCase):
    def test_round_trip_and_legacy_default(self):
        metadata = TensorAxisMetadata(
            ("point_left", "color_left", "point_right", "color_right"),
            (2, 3, 2, 3), "complex128", "elemental-A", "component-A",
            point_diagonal=True,
        )
        encoded = encode_plan_value(metadata)
        restored = decode_plan_value(encoded)
        self.assertTrue(restored.point_diagonal)
        self.assertEqual(restored, metadata)
        encoded["value"].pop("point_diagonal")
        encoded["value"].pop("structure")
        self.assertFalse(decode_plan_value(encoded).point_diagonal)


if __name__ == "__main__":
    unittest.main()
