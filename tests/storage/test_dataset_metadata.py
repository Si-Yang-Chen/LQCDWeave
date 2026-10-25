"""Metadata inspection remains scientific and does not read array payloads."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter


def _spec():
    return DatasetSpec(dict(schema=1, kind="metadata-fixture",
        configuration=dict(ensemble="metadata-ensemble", configuration="metadata-cfg",
                           lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=[5, 2]),
              dict(name="mode", coordinates=[9, 3])], dtype="float64",
        bases={"mode": dict(data_id="actual-basis", indices=[9, 3])},
        definition={"gauge_id": "explicit-gauge"},
        conventions={"normalization": "unscaled"}, parents=[]))


class DatasetMetadataObservationTests(unittest.TestCase):
    def test_complete_metadata_observation_does_not_load_npy_or_infer_unknown_identity(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "dataset"
            with DatasetWriter(directory, _spec(), provenance={"source_id": "unverified"}) as writer:
                writer.write(0, np.asarray([1.0, 2.0]))
                writer.write(1, np.asarray([3.0, 4.0]))
                identity = writer.publish()
            index = DatasetIndex(root / "index")
            index.register(directory, cfg="metadata-cfg")

            with patch.object(np, "load", side_effect=AssertionError("metadata read opened NPY")):
                metadata = index.observe_metadata(identity)
                item, = index.metadata_inventory()

            self.assertEqual(metadata["data_id"], identity)
            self.assertEqual(metadata["source_origin"], {
                "status": "unknown", "reason": "legacy-unstructured-provenance"})
            self.assertEqual(metadata["provenance"], {"source_id": "unverified"})
            self.assertEqual(metadata["spec"].coordinates("time"), [5, 2])
            self.assertEqual(metadata["spec"].document["bases"]["mode"], {
                "data_id": "actual-basis", "indices": [9, 3]})
            self.assertEqual(item["data_id"], identity)
            self.assertEqual(item["source_origin"], metadata["source_origin"])


if __name__ == "__main__":
    unittest.main()
