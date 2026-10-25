"""Scientific production and resume behavior for whole-file datasets."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.storage.dataset import (
    DatasetReader, DatasetWriter, inspect_dataset, read_dataset_metadata,
)


def _spec(*, configuration="resume-cfg", dtype="complex128", modes=(7, 2, 5), definition=None):
    return DatasetSpec(dict(schema=1, kind="ordered-basis-fixture",
        configuration=dict(ensemble="resume-ensemble", configuration=configuration,
                           lattice=[2, 2, 2, 2]),
        axes=[dict(name="time", coordinates=[3, 1]),
              dict(name="mode", coordinates=list(modes))],
        dtype=dtype, bases={"mode": dict(data_id="actual-ordered-basis", indices=list(modes))},
        definition={"gauge_id": "gauge-realization"} if definition is None else definition,
        conventions={"normalization": "unscaled"}, parents=[]))


def _row(value, dtype="complex128"):
    return np.asarray([value + 1j * index for index in range(3)], dtype=dtype)


class DatasetLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "dataset"
        self.spec = _spec()
        self.provenance = {"producer": "lifecycle-fixture", "source": "gauge-realization"}

    def test_partial_is_unreadable_and_resume_fills_only_missing_rows(self):
        first_row = _row(30)
        with DatasetWriter(self.directory, self.spec, provenance=self.provenance) as writer:
            self.assertEqual(writer.format, "npy")
            self.assertEqual(writer.missing, (0, 1))
            writer.write(0, first_row)
            self.assertEqual(writer.missing, (1,))

        partial = inspect_dataset(self.directory, cfg="resume-cfg")
        self.assertEqual(partial["state"], "partial")
        self.assertEqual(partial["completed"], (0,))
        with self.assertRaises(FileNotFoundError):
            DatasetReader(self.directory, cfg="resume-cfg")

        with DatasetWriter(self.directory, self.spec, provenance=self.provenance,
                           resume=True) as writer:
            self.assertEqual(writer.missing, (1,))
            with self.assertRaises(FileExistsError):
                writer.write(0, first_row)
            writer.write(1, _row(10))
            data_id = writer.publish()

        self.assertFalse((self.directory / ".state" / "resume-cfg.json").exists())
        self.assertTrue((self.directory / "resume-cfg.npy").is_file())
        metadata = read_dataset_metadata(self.directory, cfg="resume-cfg")
        self.assertEqual(metadata["data_id"], data_id)
        self.assertEqual(metadata["spec"].document["bases"]["mode"],
                         {"data_id": "actual-ordered-basis", "indices": [7, 2, 5]})
        self.assertEqual(metadata["spec"].coordinates("mode"), [7, 2, 5])
        with DatasetReader(self.directory, cfg="resume-cfg") as reader:
            self.assertEqual(reader.read(time=3).dtype, np.dtype("complex128"))
            self.assertEqual(reader.read(time=3).shape, (3,))
            np.testing.assert_array_equal(reader.read(time=3), first_row)
            np.testing.assert_array_equal(reader.read(time=1), _row(10))

        with self.assertRaises(FileExistsError):
            DatasetWriter(self.directory, self.spec, provenance=self.provenance)

    def test_resume_rejects_changed_definition_or_provenance_without_changing_partial(self):
        with DatasetWriter(self.directory, self.spec, provenance=self.provenance) as writer:
            writer.write(0, _row(20))
        state_path = self.directory / ".state" / "resume-cfg.json"
        state_before = state_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "definition or provenance differs"):
            DatasetWriter(self.directory, self.spec,
                          provenance={"producer": "different"}, resume=True)
        changed_spec = _spec(definition={"gauge_id": "different-gauge-realization"})
        with self.assertRaisesRegex(ValueError, "definition or provenance differs"):
            DatasetWriter(self.directory, changed_spec, provenance=self.provenance, resume=True)
        self.assertEqual(state_path.read_bytes(), state_before)
        self.assertEqual(inspect_dataset(self.directory, cfg="resume-cfg")["completed"], (0,))

    def test_recovery_attempt_can_itself_be_interrupted_and_resumed(self):
        with DatasetWriter(self.directory, self.spec, provenance=self.provenance) as writer:
            writer.write(0, _row(30))
        with DatasetWriter(self.directory, self.spec, provenance=self.provenance,
                           resume=True) as writer:
            self.assertEqual(writer.missing, (1,))
        partial = inspect_dataset(self.directory, cfg="resume-cfg")
        attempts = partial["execution"]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts],
                         ["interrupted", "interrupted"])
        self.assertEqual(attempts[1]["recovered_from"], attempts[0]["writer_session_id"])
        with DatasetWriter(self.directory, self.spec, provenance=self.provenance,
                           resume=True) as writer:
            writer.write(1, _row(10))
            writer.publish()
        with DatasetReader(self.directory, cfg="resume-cfg") as reader:
            np.testing.assert_array_equal(reader.read(time=3), _row(30))
            np.testing.assert_array_equal(reader.read(time=1), _row(10))
        attempts = read_dataset_metadata(self.directory, cfg="resume-cfg")["execution"]["attempts"]
        self.assertEqual([attempt["status"] for attempt in attempts],
                         ["interrupted", "interrupted", "complete"])
        self.assertEqual(attempts[2]["recovered_from"], attempts[1]["writer_session_id"])


if __name__ == "__main__":
    unittest.main()
