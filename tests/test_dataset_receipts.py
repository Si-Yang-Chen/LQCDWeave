"""Graph receipts are parsed and matched to current indexed identities."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.runtime.datasets import (
    read_dataset_request_receipt, read_dataset_request_receipts,
    save_dataset_request_receipt,
)


def _spec(times=(0, 1)):
    return DatasetSpec(dict(
        schema=1,
        kind="receipt_fixture_field",
        configuration=dict(ensemble="receipt-fixture", configuration="cfg-1",
                           lattice=[1, 1, 1, len(times)]),
        axes=[dict(name="time", coordinates=list(times)),
              dict(name="component", coordinates=["value"])],
        dtype="float64",
        bases={},
        definition=dict(operation="identity"),
        conventions=dict(normalization="unscaled"),
        parents=[],
    ))


class DatasetReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.spec = _spec()
        self.directory = self.root / "dataset"
        with DatasetWriter(self.directory, self.spec,
                            provenance={"producer": "receipt-fixture"}) as writer:
            for position, time in enumerate(self.spec.coordinates("time")):
                writer.write(position, np.asarray([float(time + 1)]))
            self.identity = writer.publish()
        self.index.register(self.directory)
        self.request = DatasetRequest(self.spec, data_id=self.identity)
        self.receipt = save_dataset_request_receipt(
            self.root / "receipts", {"field": self.request}, self.index,
            {"field": self.identity}, project_id="project-A")

    def test_batch_reader_deduplicates_paths_and_returns_detached_records(self):
        alias = self.root / "receipt-copy.jsonl.gz"
        alias.write_bytes(self.receipt.read_bytes())
        duplicate_spelling = self.receipt.parent / "unused" / ".." / self.receipt.name
        opened = []
        original_open = self.index.open

        def tracked_open(identity):
            reader = original_open(identity)
            opened.append(reader)
            return reader

        with patch.object(self.index, "open", side_effect=tracked_open), \
             patch.object(self.index, "observations",
                          side_effect=AssertionError("receipt reader scanned the index")):
            results = read_dataset_request_receipts(
                (self.receipt, duplicate_spelling, alias), self.index,
                expected_project_id="project-A")

        self.assertEqual(len(results), 2)
        self.assertEqual([result["receipt_path"] for result in results],
                         [str(self.receipt.resolve()), str(alias.resolve())])
        self.assertEqual(results[0]["schema"], 4)
        self.assertEqual(results[0]["project_id"], "project-A")
        self.assertEqual(results[0]["records"][0]["data_id"], self.identity)
        self.assertEqual(results[0]["dataset_observations"][0]["data_id"],
                         self.identity)
        self.assertEqual(len(opened), 1)
        for reader in opened:
            with self.assertRaisesRegex(ValueError, "closed"):
                reader.read(time=0)
        self.assertIsNot(results[0]["resolved_graph"], results[1]["resolved_graph"])

    def test_wrong_project_is_rejected_before_dataset_open(self):
        with patch.object(self.index, "open",
                          side_effect=AssertionError("dataset opened before project check")):
            with self.assertRaisesRegex(ValueError, "project_id"):
                read_dataset_request_receipt(
                    self.receipt, self.index, expected_project_id="other-project")

    def test_receipt_rechecks_referenced_data_identity(self):
        self.index.unregister(self.identity)
        with self.assertRaises((KeyError, FileNotFoundError)):
            read_dataset_request_receipt(self.receipt, self.index,
                                          expected_project_id="project-A")

    def test_structurally_truncated_receipt_is_rejected(self):
        self.receipt.write_bytes(self.receipt.read_bytes()[:-12])
        with self.assertRaises((EOFError, ValueError, OSError)):
            read_dataset_request_receipt(self.receipt, self.index)


if __name__ == "__main__":
    unittest.main()
