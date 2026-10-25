"""Per-cfg files and their shared, explicit scientific registration."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json, stable_digest
from lqcdweave.storage import dataset as storage
from lqcdweave.storage.correlator_results import save_correlator


def _spec(configuration, *, kind="registration-fixture", modes=(4, 1, 7)):
    return DatasetSpec(dict(schema=1, kind=kind,
        configuration=dict(ensemble="shared-ensemble", configuration=configuration,
                           lattice=[2, 2, 2, 3]),
        axes=[dict(name="time", coordinates=[2, 0]),
              dict(name="mode", coordinates=list(modes))],
        dtype="float64",
        bases={"mode": dict(data_id="actual-eigenvector-dataset", indices=list(modes))},
        definition={"operator": "shared-definition"},
        conventions={"normalization": "unscaled"}, parents=[]))


def _origin(*, known=True):
    names = ("source_system", "dataset_id", "revision", "producer", "producer_revision")
    return dict(schema=1, kind="external_import", origin={name:
        (dict(status="known", value=f"known-{name}") if known else
         dict(status="unknown", reason=f"not recorded: {name}")) for name in names},
        registration_runtime=dict(status="known", value={"source": "fixture"}))


def _publish(directory, spec, values=None, *, provenance=None):
    values = np.arange(np.prod(spec.shape), dtype=np.float64).reshape(spec.shape) if values is None else values
    with storage.DatasetWriter(directory, spec,
            provenance={"producer": "registration-fixture"} if provenance is None else provenance) as writer:
        for index in writer.missing:
            writer.write(index, values[index])
        return writer.publish(), values


class DatasetRegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_registration_recreates_state_directory_removed_by_another_cfg_cleanup(self):
        original_open = Path.open
        removed = False

        def open_after_cleanup(path, *args, **kwargs):
            nonlocal removed
            if path.name == "registration.lock" and not removed:
                path.parent.rmdir()
                removed = True
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", open_after_cleanup):
            with storage.registration_lock(self.root):
                self.assertTrue((self.root / ".state" / "registration.lock").is_file())
        self.assertTrue(removed)
        self.assertFalse((self.root / ".state").exists())

    def test_multi_cfg_registration_deduplicates_public_information_and_index_is_explicit(self):
        shared = self.root / "shared"
        specs = [_spec("cfg-A"), _spec("cfg-B")]
        published = {}
        for spec in specs:
            published[spec.document["configuration"]["configuration"]] = _publish(
                shared, spec)[0]

        disk = json.loads((shared / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(set(disk["configurations"]), {"cfg-A", "cfg-B"})
        self.assertEqual(disk["format"], "easy-distillation.dataset")
        self.assertEqual(disk["common"]["identity"]["spec"]["kind"],
                         "registration-fixture")
        self.assertEqual(disk["common"]["identity"]["provenance"],
                         {"producer": "registration-fixture"})
        self.assertIsNone(disk["common"]["identity"]["spec"]["configuration"]["configuration"])
        self.assertEqual(set(storage.read_registration(shared)), {"cfg-A", "cfg-B"})
        self.assertNotEqual(published["cfg-A"], published["cfg-B"])
        self.assertTrue((shared / "cfg-A.npy").is_file())
        self.assertTrue((shared / "cfg-B.npy").is_file())
        with self.assertRaises(KeyError):
            storage.read_dataset_metadata(shared)

        index = storage.DatasetIndex(self.root / "index")
        index.register(shared, cfg="cfg-A")
        self.assertEqual(index.observe_metadata(published["cfg-A"])["configuration"], "cfg-A")
        with index.open(published["cfg-A"]) as reader:
            self.assertEqual(reader.configuration, "cfg-A")
        with self.assertRaises(KeyError):
            index.open(published["cfg-B"])

        unregistered = self.root / "valid-but-unregistered"
        unregistered_id, _ = _publish(unregistered, _spec("cfg-C"))
        self.assertEqual(index.observations()[0]["data_id"], published["cfg-A"])
        with self.assertRaises(KeyError):
            index.open(unregistered_id)

    def test_metadata_observation_keeps_known_and_unknown_source_origin_without_loading_npy(self):
        spec = _spec("source-cfg", kind="external-source")
        values = np.arange(np.prod(spec.shape), dtype=np.float64).reshape(spec.shape)
        source_file = self.root / "source.npy"
        np.save(source_file, values, allow_pickle=False)
        directory = self.root / "registered-source"
        data_id = storage.register_npy(directory, source_file, spec,
            provenance=_origin(known=True), provenance_mode="external_import")
        index = storage.DatasetIndex(self.root / "index")
        index.register(directory)

        with patch.object(np, "load", side_effect=AssertionError("metadata read opened NPY")):
            metadata = index.observe_metadata(data_id)
            inventory, = index.metadata_inventory()
        self.assertEqual(metadata["source_origin"]["status"], "recorded")
        self.assertEqual(metadata["source_origin"]["origin"]["dataset_id"],
                         {"status": "known", "value": "known-dataset_id"})
        self.assertEqual(inventory["data_id"], data_id)
        self.assertEqual(metadata["spec"].coordinates("mode"), [4, 1, 7])

        unknown_directory = self.root / "unknown-origin"
        unknown_id = storage.register_npy(unknown_directory, source_file,
            _spec("unknown-cfg", kind="unknown-origin"), provenance=_origin(known=False),
            provenance_mode="external_import")
        unknown = storage.read_dataset_metadata(unknown_directory)
        self.assertEqual(unknown["source_origin"]["origin"]["producer"],
            {"status": "unknown", "reason": "not recorded: producer"})
        self.assertEqual(unknown["data_id"], unknown_id)

    def test_scientific_identity_corruption_is_rejected_without_rebinding_its_id(self):
        directory = self.root / "corrupt-declaration"
        data_id, _ = _publish(directory, _spec("cfg-corrupt"))
        records = storage.read_registration(directory)
        records["cfg-corrupt"]["identity"]["spec"]["kind"] = "changed-kind"
        storage.write_registration(directory, records)
        with self.assertRaisesRegex(ValueError, "identity or completion state differs"):
            storage.read_dataset_metadata(directory, cfg="cfg-corrupt")
        corrupted_identity = storage.read_registration(directory)["cfg-corrupt"]["identity"]
        self.assertEqual(storage.read_registration(directory)["cfg-corrupt"]["data_id"], data_id)
        self.assertNotEqual(stable_digest(corrupted_identity), data_id)

    def test_collection_hdf5_keeps_actual_shapes_and_dtypes_in_named_fields(self):
        directory = self.root / "collection"
        left = np.arange(6, dtype=np.float32).reshape(2, 3)
        right = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128)
        data_id = save_correlator(directory, {"left": left, "right": right},
                                  configuration="cfg-fields")
        metadata = storage.read_dataset_metadata(directory, cfg="cfg-fields")
        self.assertEqual(metadata["spec"].axis_names, ())
        self.assertIsNone(metadata["spec"].document["dtype"])
        self.assertEqual(metadata["storage"]["format"], "hdf5")
        spec_fields = metadata["spec"].document["definition"]["fields"]
        left_path = spec_fields[0]["path"]
        right_path = spec_fields[1]["path"]
        self.assertEqual(metadata["fields"][canonical_json(left_path)]["shape"], [2, 3])
        with storage.DatasetReader(directory, cfg="cfg-fields") as reader:
            np.testing.assert_array_equal(reader.read_field(left_path), left)
            np.testing.assert_array_equal(reader.read_field(right_path), right)
            self.assertEqual(reader.fields[canonical_json(right_path)]["dtype"],
                             np.dtype("complex128").str)
        self.assertEqual(storage.read_registration(directory)["cfg-fields"]["data_id"], data_id)


if __name__ == "__main__":
    unittest.main()
