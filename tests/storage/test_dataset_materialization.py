"""Materialized cfg files preserve actual IDs and numerical bytes."""

from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import unittest

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetWriter, read_dataset_metadata,
)
from lqcdweave.storage.registration import read_registration


def _spec(configuration):
    return DatasetSpec(dict(schema=1, kind="materialization-fixture",
        configuration=dict(ensemble="copy-ensemble", configuration=configuration,
                           lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=[3, 1]),
              dict(name="mode", coordinates=[8, 2, 5])], dtype="complex128",
        bases={"mode": dict(data_id="actual-basis", indices=[8, 2, 5])},
        definition={"operator": "copy-test"}, conventions={"normalization": "unscaled"},
        parents=[]))


def _values(seed):
    return (np.arange(6, dtype=np.float64).reshape(2, 3) + seed).astype(np.complex128) * (1 + 2j)


def _publish(directory, spec, values):
    with DatasetWriter(directory, spec, provenance={"producer": "copy-fixture"}) as writer:
        for index in writer.missing:
            writer.write(index, values[index])
        return writer.publish()


def _payload(directory, cfg):
    path = Path(read_dataset_metadata(directory, cfg=cfg)["storage"]["path"])
    return path if path.is_absolute() else Path(directory) / path


class DatasetMaterializationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_copy_keeps_actual_id_and_payload_bits_after_source_is_removed(self):
        source = self.root / "source"
        spec = _spec("copy-cfg")
        values = _values(0.125)
        identity = _publish(source, spec, values)
        index = DatasetIndex(self.root / "index")
        index.register(source, cfg="copy-cfg")
        source_bytes = _payload(source, "copy-cfg").read_bytes()

        destination = self.root / "materialized"
        self.assertEqual(index.materialize(identity, destination), identity)
        target = _payload(destination, "copy-cfg")
        self.assertEqual(target.read_bytes(), source_bytes)
        self.assertEqual(index.observe_metadata(identity)["data_id"], identity)
        self.assertEqual(index.materialize(identity, destination), identity)

        shutil.rmtree(source)
        with index.open(identity) as reader:
            self.assertEqual(reader.data_id, identity)
            for time, expected in zip(spec.coordinates("time"), values, strict=True):
                actual = reader.read(time=time)
                self.assertEqual(actual.dtype, np.dtype("complex128"))
                np.testing.assert_array_equal(actual, expected)

    def test_shared_destination_materializes_only_explicit_cfgs(self):
        source = self.root / "multi-source"
        specs = (_spec("copy-A"), _spec("copy-B"))
        values = {spec.document["configuration"]["configuration"]: _values(index + 3)
                  for index, spec in enumerate(specs)}
        identities = {spec.document["configuration"]["configuration"]:
                      _publish(source, spec, values[spec.document["configuration"]["configuration"]])
                      for spec in specs}
        index = DatasetIndex(self.root / "index")
        for cfg, identity in identities.items():
            index.register(source, cfg=cfg)

        destination = self.root / "shared-copy"
        for cfg, identity in identities.items():
            self.assertEqual(index.materialize(identity, destination), identity)
        with index.open(identities["copy-A"]) as reader_a, \
             index.open(identities["copy-B"]) as reader_b:
            np.testing.assert_array_equal(reader_a.read(time=3), values["copy-A"][0])
            np.testing.assert_array_equal(reader_b.read(time=3), values["copy-B"][0])
        self.assertEqual(set(read_registration(destination)), {"copy-A", "copy-B"})


if __name__ == "__main__":
    unittest.main()
