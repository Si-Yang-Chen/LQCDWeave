"""Managed-cache retirement protects readers, other cfg data, and sources."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetReader, DatasetWriter, register_npy,
)
from lqcdweave.storage.registration import read_registration


def _spec(configuration):
    return DatasetSpec(dict(schema=1, kind="managed-cache-fixture",
        configuration=dict(ensemble="managed-ensemble", configuration=configuration,
                           lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=[0, 1]),
              dict(name="component", coordinates=["value"])], dtype="float64",
        bases={}, definition={"role": configuration}, conventions={}, parents=[]))


def _publish(directory, spec):
    values = np.asarray([[1.25], [2.5]], dtype=np.float64)
    with DatasetWriter(directory, spec, provenance={"producer": "retention-fixture"}) as writer:
        for index in writer.missing:
            writer.write(index, values[index])
        return writer.publish(), values


class DatasetRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.index = DatasetIndex(self.root / "index", managed_cache_root=self.cache)

    def test_retirement_removes_only_owned_cfg_and_preserves_sources_and_other_cfg(self):
        source = self.root / "source"
        spec_a, spec_b = _spec("cfg-A"), _spec("cfg-B")
        identity_a, _ = _publish(source, spec_a)
        identity_b, values_b = _publish(source, spec_b)
        self.index.register(source, cfg="cfg-A")
        self.index.register(source, cfg="cfg-B")

        managed = self.cache / "copy"
        self.index.materialize(identity_a, managed, retention_managed=True)
        self.index.materialize(identity_b, managed, retention_managed=True)
        self.assertEqual(self.index.describe_managed_location(identity_a, managed)["state"], "active")
        self.assertEqual(self.index.verify_managed_location(identity_b, managed)["data_id"],
                         identity_b)
        source_payload_a = source / "cfg-A.npy"
        source_payload_b = source / "cfg-B.npy"

        retired = self.index.retire_managed_location(identity_a, managed)
        self.assertEqual(retired["state"], "retired")
        records = read_registration(managed)
        self.assertEqual(set(records), {"cfg-B"})
        self.assertFalse((managed / "cfg-A.npy").exists())
        self.assertTrue((managed / "cfg-B.npy").is_file())
        self.assertTrue(source_payload_a.is_file())
        self.assertTrue(source_payload_b.is_file())
        with self.index.open(identity_b) as reader:
            np.testing.assert_array_equal(reader.read(time=0), values_b[0])
        with self.assertRaisesRegex(ValueError, "not owned and active"):
            self.index.describe_managed_location(identity_a, managed)

    def test_active_reader_blocks_managed_retirement_until_closed(self):
        source = self.root / "reader-source"
        identity, _ = _publish(source, _spec("reader-cfg"))
        self.index.register(source)
        managed = self.cache / "reader-copy"
        self.index.materialize(identity, managed, retention_managed=True)

        reader = DatasetReader(managed, cfg="reader-cfg")
        try:
            with self.assertRaisesRegex(FileExistsError, "active reader"):
                self.index.retire_managed_location(identity, managed)
            self.assertTrue((managed / "reader-cfg.npy").is_file())
        finally:
            reader.close()
        self.assertEqual(self.index.retire_managed_location(identity, managed)["state"], "retired")
        self.assertFalse((managed / "reader-cfg.npy").exists())

    def test_unmanaged_copy_and_external_source_are_never_retired_as_cache_data(self):
        spec = _spec("external-cfg")
        values = np.asarray([[4.0], [6.0]], dtype=np.float64)
        external_file = self.root / "external.npy"
        np.save(external_file, values, allow_pickle=False)
        source = self.root / "external-registration"
        identity = register_npy(source, external_file, spec,
                                provenance={"source": "external-fixture"})
        self.index.register(source)

        unmanaged = self.root / "ordinary-copy"
        self.index.materialize(identity, unmanaged)
        with self.assertRaisesRegex(ValueError, "direct nonsymlink cache-root child"):
            self.index.describe_managed_location(identity, unmanaged)
        with self.assertRaisesRegex(ValueError, "direct nonsymlink cache-root child"):
            self.index.retire_managed_location(identity, unmanaged)
        self.assertTrue((unmanaged / "external-cfg.npy").is_file())

        managed = self.cache / "managed-copy"
        self.index.materialize(identity, managed, retention_managed=True)
        self.index.retire_managed_location(identity, managed)
        self.assertTrue(external_file.is_file())
        self.assertTrue((source / "dataset.json").is_file())
        self.assertTrue((unmanaged / "external-cfg.npy").is_file())

    def test_managed_destination_must_be_a_direct_cache_child(self):
        source = self.root / "nested-source"
        identity, _ = _publish(source, _spec("nested-cfg"))
        self.index.register(source)
        nested = self.cache / "nested" / "copy"
        nested.parent.mkdir()
        with self.assertRaisesRegex(ValueError, "direct nonsymlink cache-root children"):
            self.index.materialize(identity, nested, retention_managed=True)
        self.assertFalse(nested.exists())


if __name__ == "__main__":
    unittest.main()
