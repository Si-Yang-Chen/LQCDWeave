"""Project receipts explicitly bound public dataset retention decisions."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.storage.registration import read_registration
from lqcdweave.runtime.datasets import (
    read_dataset_request_receipt, save_dataset_request_receipt,
)
from lqcdweave.runtime.project_datasets import (
    apply_project_dataset_retention, plan_project_dataset_retention,
)


def _spec():
    return DatasetSpec(dict(schema=1, kind="project-retention-fixture",
        configuration=dict(ensemble="test", configuration="project-retention",
                           lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=[0, 1]),
              dict(name="mode", coordinates=[0])], dtype="float64", bases={},
        definition=dict(producer="project-retention-test"),
        conventions=dict(normalization="unscaled"), parents=[]))


class ProjectDatasetRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.index = DatasetIndex(self.root / "index", managed_cache_root=self.cache)
        self.spec = _spec()
        self.source = self.root / "source"
        with DatasetWriter(self.source, self.spec,
                            provenance={"producer": "fixture"}) as writer:
            for position in writer.missing:
                writer.write(position, np.asarray([position + 1.0]))
            self.identity = writer.publish()
        self.index.register(self.source)
        self.managed = self.cache / "managed-copy"
        self.index.materialize(self.identity, self.managed, retention_managed=True)
        self.request = DatasetRequest(self.spec.document,
                                       data_id=self.identity)

    def test_receipt_publication_stales_plan_and_retains_its_dataset(self):
        plan = plan_project_dataset_retention(self.index,
            [{"data_id": self.identity, "location": self.managed}], (),
            expected_project_id="project-1",
            complete_index_inventory=True, complete_receipt_inventory=True)
        starting_revision = self.index.publication_revision

        receipt_path = save_dataset_request_receipt(self.root / "receipts",
            {"data": self.request}, self.index, {"data": self.identity},
            project_id="project-1")
        self.assertGreater(self.index.publication_revision, starting_revision)
        receipt = read_dataset_request_receipt(receipt_path, self.index,
                                                expected_project_id="project-1")
        after_first_publication = self.index.publication_revision
        same_path = save_dataset_request_receipt(self.root / "receipts",
            {"data": self.request}, self.index, {"data": self.identity},
            project_id="project-1")
        self.assertNotEqual(same_path, receipt_path)
        self.assertGreater(self.index.publication_revision, after_first_publication)

        with self.assertRaisesRegex(ValueError, "stale"):
            apply_project_dataset_retention(self.index, plan)
        receipt_plan = plan_project_dataset_retention(self.index,
            [{"data_id": self.identity, "location": self.managed}],
            [receipt_path], expected_project_id="project-1",
            complete_index_inventory=True, complete_receipt_inventory=True)
        self.assertEqual(receipt_plan["decisions"][0]["action"], "retain")
        self.assertTrue(self.managed.exists())

    def test_explicit_candidate_is_the_only_location_touched(self):
        unrelated = self.root / "unregistered-dataset-looking-directory"
        unrelated.mkdir()
        (unrelated / "user.txt").write_text("preserve", encoding="utf-8")
        managed_note = self.managed / "user.txt"
        managed_note.write_text("preserve", encoding="utf-8")
        other_document = self.spec.document
        other_document["configuration"]["configuration"] = "other-cfg"
        other_spec = DatasetSpec(other_document)
        with DatasetWriter(self.managed, other_spec,
                           provenance={"producer": "other-cfg-fixture"}) as writer:
            for position in writer.missing:
                writer.write(position, np.asarray([position + 10.0]))
            other_identity = writer.publish()
        self.index.register(self.managed, cfg="other-cfg")
        plan = plan_project_dataset_retention(self.index,
            [{"data_id": self.identity, "location": self.managed}], (),
            expected_project_id="project-1",
            complete_index_inventory=True, complete_receipt_inventory=True)
        result = apply_project_dataset_retention(self.index, plan)
        self.assertEqual(result["decisions"][0]["state"], "retired")
        self.assertTrue(self.managed.exists())
        self.assertEqual(set(read_registration(self.managed)), {"other-cfg"})
        self.assertTrue((self.managed / "other-cfg.npy").is_file())
        self.assertTrue(managed_note.is_file())
        self.assertTrue(self.source.exists())
        locations = {item["data_id"]: item["location"]
                     for item in self.index.metadata_inventory()}
        self.assertEqual(locations[self.identity], str(self.source.resolve()))
        self.assertEqual(locations[other_identity], str(self.managed.resolve()))
        self.assertEqual((unrelated / "user.txt").read_text(encoding="utf-8"), "preserve")

    def test_failed_receipt_replace_leaves_no_receipt_and_invalidates_old_plan(self):
        plan = plan_project_dataset_retention(self.index,
            [{"data_id": self.identity, "location": self.managed}], (),
            expected_project_id="project-1",
            complete_index_inventory=True, complete_receipt_inventory=True)
        starting_revision = self.index.publication_revision
        receipt_directory = self.root / "failed-receipt"
        real_replace = os.replace

        def fail_receipt_replace(source, destination):
            if Path(destination).parent == receipt_directory:
                raise OSError("injected receipt atomic-replace failure")
            return real_replace(source, destination)

        with patch("lqcdweave.storage.dataset.os.replace",
                   new=fail_receipt_replace):
            with self.assertRaisesRegex(OSError, "receipt atomic-replace"):
                save_dataset_request_receipt(receipt_directory,
                    {"data": self.request}, self.index, {"data": self.identity},
                    project_id="project-1")
        self.assertGreater(self.index.publication_revision, starting_revision)
        self.assertEqual(list(receipt_directory.glob("graph-*.jsonl.gz")), [])
        self.assertEqual(list(receipt_directory.glob("*.tmp")), [])
        with self.assertRaisesRegex(ValueError, "stale"):
            apply_project_dataset_retention(self.index, plan)
        with self.index.open(self.identity) as reader:
            self.assertEqual(reader.data_id, self.identity)


if __name__ == "__main__":
    unittest.main()
