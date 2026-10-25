"""The optional locator index is explicit and resolves registered cfg data."""

from pathlib import Path

import unittest

from test_dataset_index import DatasetWriter, _DatasetIndexFixture, np, spec


class DatasetIndexTests(_DatasetIndexFixture):
    def test_registration_is_explicit_and_selects_the_registered_cfg(self):
        directory, identity = self.publish()
        self.assertEqual(self.index.observations(), ())
        self.assertEqual(self.index.register(directory, cfg="cfg-7"), identity)

        observation, = self.index.observations()
        self.assertEqual(observation["data_id"], identity)
        self.assertEqual(observation["state"], "complete")
        with self.index.open(identity) as reader:
            self.assertEqual(reader.configuration, "cfg-7")
            self.assertEqual(reader.read(time=1).tolist(), [14 + 1j, 11 + 1j, 17 + 1j])

    def test_partial_cfg_cannot_be_indexed_until_published(self):
        partial = self.root / "partial"
        with DatasetWriter(partial, spec(), provenance={"producer": "partial"}) as writer:
            writer.write(0, np.ones(3, dtype=np.complex128))
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.index.register(partial)
        self.assertEqual(self.index.observations(), ())

    def test_index_does_not_discover_an_unregistered_valid_directory(self):
        registered, registered_id = self.publish("registered")
        self.index.register(registered, cfg="cfg-7")
        _, unregistered_id = self.publish("unregistered")

        observations = self.index.observations()
        self.assertEqual([item["data_id"] for item in observations], [registered_id])
        with self.assertRaises(KeyError):
            self.index.open(unregistered_id)


if __name__ == "__main__":
    unittest.main()
