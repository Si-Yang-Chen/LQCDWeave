"""Actual dataset/coordinate identities carried by reader axis metadata."""

import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    TensorAxisMetadata,
    canonical_axis_binding_identity,
    canonical_coordinate_identity,
)


class TensorAxisBindingTests(unittest.TestCase):
    def test_coordinate_encoding_matches_independent_json_for_all_label_types(self):
        import json
        class MisleadingInteger(int):
            def __str__(self):
                return 'not-the-integer-value'
        values = (0, -1, 2**100, -(2**100), True, False, 1., -0., 1e-200,
                  '颜色\\"\n', MisleadingInteger(7), np.int64(-9), np.bool_(True), np.float64(.25))
        for value in values:
            with self.subTest(value=repr(value)):
                native = value.item() if isinstance(value, np.generic) else value
                expected = json.dumps(native, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
                self.assertEqual(canonical_coordinate_identity(value), expected)
        for invalid in (None, 1j, np.inf, np.nan, object()):
            with self.subTest(invalid=repr(invalid)), self.assertRaises((TypeError, ValueError)):
                canonical_coordinate_identity(invalid)

    def test_axis_bindings_are_copied_immutable_and_keep_ordered_labels(self):
        labels = [True, 1]
        bindings = {"mode": ("basis-actual", labels)}
        metadata = TensorAxisMetadata(
            ("mode",), (2,), np.complex128, "data", "selection",
            axis_bindings=bindings,
        )
        labels.reverse()
        bindings["mode"] = ("other", (4, 5))

        self.assertEqual(metadata.axis_bindings["mode"], ("basis-actual", (True, 1)))
        self.assertNotEqual(canonical_coordinate_identity(True), canonical_coordinate_identity(1))
        self.assertNotEqual(canonical_coordinate_identity(1), canonical_coordinate_identity(1.0))
        self.assertEqual(
            canonical_axis_binding_identity(metadata.axis_bindings),
            (("mode", "basis-actual", ("true", "1")),),
        )
        with self.assertRaises(TypeError):
            metadata.axis_bindings["mode"] = ("mutated", (0, 1))

    def test_unbound_metadata_remains_compatible(self):
        metadata = TensorAxisMetadata(
            ("mode",), (1,), np.complex128, None, "legacy-selection",
        )
        self.assertEqual(dict(metadata.axis_bindings), {})

    def test_axis_bindings_validate_axis_dataset_and_extent(self):
        base = dict(
            axes=("mode",), shape=(2,), dtype=np.complex128,
            data_identity="data", selection_identity="selection",
        )
        with self.assertRaisesRegex(ValueError, "absent tensor axis"):
            TensorAxisMetadata(**base, axis_bindings={"point": ("points", (0, 1))})
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            TensorAxisMetadata(**base, axis_bindings={"mode": ("", (0, 1))})
        with self.assertRaisesRegex(ValueError, "labels for extent"):
            TensorAxisMetadata(**base, axis_bindings={"mode": ("basis", (0,))})
        with self.assertRaisesRegex(TypeError, "JSON scalar"):
            TensorAxisMetadata(**base, axis_bindings={"mode": ("basis", (0, None))})
        with self.assertRaisesRegex(TypeError, "JSON scalar"):
            TensorAxisMetadata(**base, axis_bindings={"mode": ("basis", (0, 1 + 2j))})


if __name__ == "__main__":
    unittest.main()
