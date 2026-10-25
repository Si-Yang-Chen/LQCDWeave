"""Index copies of one actual ID must agree on their parsed source claim."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
import lqcdweave.storage.dataset as dataset_storage
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetReader, DatasetWriter, read_dataset_metadata, register_npy,
)
from lqcdweave.storage.registration import (
    read_registration, registration_lock, write_registration,
)


def _spec():
    return DatasetSpec(dict(schema=1, kind="source-consensus-fixture", dtype="float64",
        configuration=dict(ensemble="consensus-fixture", configuration="cfg-1",
                           lattice=[1, 1, 1, 1]),
        axes=[dict(name="time", coordinates=[0]),
              dict(name="component", coordinates=[0])],
        bases={}, definition={}, conventions={}, parents=[]))


def _external_provenance(*, runtime):
    values = dict(source_system="archive", dataset_id="consensus/cfg-1",
        revision="r1", producer="solver", producer_revision="solver-v1")
    origin = {name: dict(status="known", value=value)
              for name, value in values.items()}
    return dict(schema=1, kind="external_import", origin=origin,
        registration_runtime=dict(status="known", value=runtime))


class DatasetSourceConsensusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = _spec()
        self.values = np.array([[2.5]], dtype=np.float64)
        self.npy = self.root / "payload.npy"
        np.save(self.npy, self.values, allow_pickle=False)

    def make_legacy_and_external(self):
        # Source provenance and a per-publication realization ID contribute
        # to actual identity. Marker-shaped legacy metadata remains unknown to
        # source-policy consumers.
        provenance = _external_provenance(runtime={"python": "3.13"})
        identity_provenance = {key: provenance[key]
                               for key in ("schema", "kind", "origin")}
        legacy = self.root / "legacy-copy"
        with DatasetWriter(legacy, self.spec, provenance=identity_provenance) as writer:
            writer.write(0, self.values[0])
            legacy_id = writer.publish()
        external = self.root / "external-copy"
        imported_id = register_npy(external, self.npy, self.spec,
            provenance=provenance, provenance_mode="external_import")
        self.assertNotEqual(legacy_id, imported_id)
        with DatasetReader(legacy) as reader:
            self.assertEqual(reader.source_origin["status"], "unknown")
        with DatasetReader(external) as reader:
            self.assertEqual(reader.source_origin["status"], "recorded")
        return legacy_id, legacy, imported_id, external

    def test_distinct_source_claims_have_distinct_actual_ids(self):
        legacy_id, legacy, imported_id, external = self.make_legacy_and_external()
        index = DatasetIndex(self.root / "index")
        self.assertEqual(index.register(legacy), legacy_id)
        self.assertEqual(index.register(external), imported_id)
        self.assertEqual(index.observe_metadata(legacy_id)["source_origin"]["status"], "unknown")
        self.assertEqual(index.observe_metadata(imported_id)["source_origin"]["status"], "recorded")

    def test_missing_copy_is_skipped_but_existing_invalid_copy_fails_closed(self):
        valid = self.root / "valid-copy"
        identity = register_npy(valid, self.npy, self.spec,
            provenance=_external_provenance(runtime={"python": "3.13"}),
            provenance_mode="external_import")
        missing = self.root / "absent-copy"
        index = DatasetIndex(self.root / "index")
        index.register(valid)
        missing.mkdir()
        with registration_lock(missing):
            write_registration(missing, read_registration(valid))
        self.assertEqual(index.register(missing), identity)
        shutil.rmtree(missing)
        # Metadata consensus must not load the underlying NPY payload.
        with patch("numpy.load", side_effect=AssertionError("metadata loaded payload")):
            self.assertEqual(index.observe_metadata(identity)["source_origin"]["status"],
                             "recorded")
        with index.open(identity) as reader:
            self.assertEqual(reader.data_id, identity)
            np.testing.assert_array_equal(reader.read(time=0), self.values[0])
        missing.mkdir()
        with registration_lock(missing):
            invalid = dict(format="easy-distillation.dataset", version=999,
                           common={}, configurations={})
            (missing / "dataset.json").write_text(json.dumps(invalid), encoding="utf-8")
        for operation in (lambda: index.open(identity),
                          lambda: index.open_selected(identity, {"time": [0]}),
                          lambda: index.observe_metadata(identity),
                          lambda: index.observations()):
            with self.subTest(operation=operation), self.assertRaises((ValueError, FileNotFoundError)):
                operation()

    def test_same_origin_with_distinct_registration_runtime_has_distinct_instances(self):
        first = self.root / "external-python-313"
        second = self.root / "external-python-314"
        first_id = register_npy(first, self.npy, self.spec,
            provenance=_external_provenance(runtime={"python": "3.13"}),
            provenance_mode="external_import")
        second_id = register_npy(second, self.npy, self.spec,
            provenance=_external_provenance(runtime={"python": "3.14"}),
            provenance_mode="external_import")
        self.assertNotEqual(first_id, second_id)
        first_record = read_registration(first)["cfg-1"]
        second_record = read_registration(second)["cfg-1"]
        self.assertEqual(first_record["identity"]["provenance"],
                         second_record["identity"]["provenance"])
        self.assertEqual(first_record["execution"]["registration_event"]["runtime"],
                         {"status": "known", "value": {"python": "3.13"}})
        self.assertEqual(second_record["execution"]["registration_event"]["runtime"],
                         {"status": "known", "value": {"python": "3.14"}})
        index = DatasetIndex(self.root / "index")
        self.assertEqual(index.register(first), first_id)
        self.assertEqual(index.register(second), second_id)
        self.assertEqual(read_dataset_metadata(first, cfg="cfg-1")["data_id"], first_id)
        with index.open(first_id) as reader:
            self.assertEqual(reader.source_origin["status"], "recorded")
        metadata = index.observe_metadata(second_id)
        self.assertEqual(metadata["state"], "metadata")
        self.assertEqual(metadata["source_origin"]["status"], "recorded")

    def test_registered_copies_must_agree_on_parsed_source_origin(self):
        first = self.root / "registered-copy-a"
        second = self.root / "registered-copy-b"
        identity = register_npy(first, self.npy, self.spec,
            provenance=_external_provenance(runtime={"python": "3.13"}),
            provenance_mode="external_import")
        second.mkdir()
        with registration_lock(second):
            write_registration(second, read_registration(first))

        index = DatasetIndex(self.root / "index")
        self.assertEqual(index.register(first), identity)
        self.assertEqual(index.register(second), identity)
        read_metadata = dataset_storage.read_dataset_metadata

        def conflicting_origin(directory, *, cfg=None):
            metadata = read_metadata(directory, cfg=cfg)
            if Path(directory).resolve() == second.resolve():
                origin = dict(metadata["source_origin"])
                fields = dict(origin["origin"])
                fields["revision"] = dict(status="known", value="different-revision")
                origin["origin"] = fields
                metadata["source_origin"] = origin
            return metadata

        # Both copies retain the same actual ID and valid registrations. The
        # injected discrepancy isolates the index's parsed-origin consensus.
        with patch.object(dataset_storage, "read_dataset_metadata", conflicting_origin):
            with self.assertRaisesRegex(ValueError, "copies disagree on source origin"):
                index.observe_metadata(identity)


if __name__ == "__main__":
    unittest.main()
