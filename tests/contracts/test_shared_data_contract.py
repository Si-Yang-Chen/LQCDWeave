"""Version, layout, dtype and identity semantics of the existing pure contracts."""

import unittest

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json, stable_digest
from lqcdweave.contracts.correlator.api import (
    TensorAxisMetadata, TensorDescription, TensorSpec, canonical_axis_binding_identity,
)
from lqcdweave.contracts.time_selection import (
    TimeSelectionRecord, dataset_physical_times, bind_time_selections, normalize_time_selection,
)
from fixtures.data_contract import specification


class SharedDataContractTests(unittest.TestCase):
    def test_definition_schema_and_persistent_dtype_are_exact_not_inferred(self):
        original = specification()
        self.assertEqual(original.axis_names, ("time", "mode"))
        self.assertEqual(original.shape, (2, 3))
        for dtype in ("int32", "int64", "float32", "float64", "complex64", "complex128"):
            with self.subTest(dtype=dtype):
                self.assertEqual(specification(dtype=dtype).document["dtype"], dtype)
        for change in ({"schema": 2}, {"schema": True}, {"dtype": "bool"}, {"dtype": "uint32"}, {"path": "/a/file.npy"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DatasetSpec({**original.document, **change})

    def test_definition_snapshot_and_coordinate_identity_preserve_order_and_type(self):
        original = specification()
        detached = original.document
        detached["axes"][0]["coordinates"].reverse()
        self.assertEqual(original.coordinates("time"), [6, 0])
        self.assertNotEqual(DatasetSpec(detached).definition_id, original.definition_id)
        reordered_keys = dict(reversed(tuple(original.document.items())))
        self.assertEqual(DatasetSpec(reordered_keys).definition_id, original.definition_id)
        self.assertEqual(original.definition_id, stable_digest(original.document))
        self.assertEqual(len({canonical_json(value) for value in (True, 1, 1.0)}), 3)
        self.assertEqual(canonical_json((1, 2)), canonical_json([1, 2]))

    def test_time_record_keeps_physical_extent_and_saved_slot_order(self):
        full = normalize_time_selection(None, Lt=8)
        self.assertEqual(full.times, tuple(range(8)))
        record = TimeSelectionRecord(8, (6, 0))
        self.assertEqual(record.physical_time(0), 6)
        self.assertEqual(record.slot(0), 1)
        with self.assertRaises(KeyError):
            record.slot(7)
        spec = specification()
        mapped = DatasetSpec(bind_time_selections(spec.document, {"time": record}))
        self.assertEqual(mapped.coordinates("time"), [0, 1])
        self.assertEqual(dataset_physical_times(mapped), (6, 0))
        self.assertEqual(spec.coordinates("time"), [6, 0])
        self.assertNotEqual(mapped.definition_id, spec.definition_id)
        with self.assertRaisesRegex(ValueError, "rebind"):
            bind_time_selections(mapped.document, {"time": TimeSelectionRecord(8, (0, 6))})

    def test_tensor_abi_equality_is_not_scientific_identity(self):
        left = TensorSpec((3,), np.complex128, description=TensorDescription(axes=("mode",), data_identity="a"))
        right = TensorSpec((3,), np.complex128, description=TensorDescription(axes=("mode",), data_identity="b"))
        self.assertEqual(left, right)
        self.assertNotEqual(left.data_identity, right.data_identity)
        metadata = TensorAxisMetadata(("mode",), (3,), np.complex128, "data", "selection", axis_bindings={"mode": ("actual-basis", (True, 1, 1.0))})
        identity = canonical_axis_binding_identity(metadata.axis_bindings)
        self.assertEqual(identity, (("mode", "actual-basis", ("true", "1", "1.0")),))
        with self.assertRaisesRegex(ValueError, "labels"):
            TensorAxisMetadata(("mode",), (2,), np.complex128, "data", "selection", axis_bindings={"mode": ("actual-basis", (True, 1, 1.0))})



if __name__ == "__main__":
    unittest.main()
