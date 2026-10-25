"""Storage enforcement of the shared version, precision and identity contract."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.dataset import stable_digest
from lqcdweave.storage.dataset import (
    DatasetReader, DatasetWriter, read_dataset_metadata,
)
from lqcdweave.storage.registration import read_registration, registration_lock
from fixtures.data_contract import specification, publish


class SharedDataContractStorageTests(unittest.TestCase):
    def test_publication_versions_are_separate_from_definition_and_validation(self):
        with TemporaryDirectory() as root:
            directory = Path(root) / "dataset"
            spec = specification()
            identity = publish(directory, spec)
            registration_path = directory / "dataset.json"
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            self.assertEqual(registration["format"], "easy-distillation.dataset")
            self.assertEqual(registration["version"], 1)
            record = read_registration(directory)["cfg"]
            self.assertEqual(record["identity"]["schema"], 1)
            self.assertEqual(record["identity"]["spec"]["schema"], 1)
            self.assertEqual(record["execution"]["schema"], 1)
            self.assertEqual(record["execution"]["scientific_validation"], {"status": "unknown"})
            self.assertEqual(stable_digest(record["identity"]), identity)
            self.assertNotEqual(identity, spec.definition_id)
            self.assertNotIn("blocks", record["identity"])
            self.assertNotIn("storage", record["identity"])
            self.assertNotIn("execution", record["identity"])
            metadata = read_dataset_metadata(directory, cfg="cfg")
            self.assertEqual(metadata["data_id"], identity)
            with registration_lock(directory):
                registration["version"] = 999
                registration_path.write_text(json.dumps(registration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registration"):
                DatasetReader(directory)

    def test_writer_rejects_implicit_precision_and_shape_changes(self):
        with TemporaryDirectory() as root:
            with DatasetWriter(Path(root) / "dataset", specification(), provenance={}) as writer:
                for value in (np.zeros(3, dtype=np.complex64), np.zeros(2, dtype=np.complex128)):
                    with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaisesRegex(ValueError, "shape/dtype"):
                        writer.write(0, value)
                self.assertEqual(writer.missing, (0, 1))



if __name__ == "__main__":
    unittest.main()
