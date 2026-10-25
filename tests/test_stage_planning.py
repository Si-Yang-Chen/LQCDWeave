"""Read-only stage planning over declared storage coverage."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.runtime.dataset_stage import stage_plan
from lqcdweave.storage.dataset import DatasetWriter, inspect_dataset


def spec(mass=0.4):
    return DatasetSpec(dict(schema=1, kind="reference_block",
        configuration=dict(ensemble="test", configuration="cfg", lattice=[1,1,1,2]),
        axes=[dict(name="time", coordinates=[0,1]),
              dict(name="mode", coordinates=[0,1])],
        dtype="complex128", bases={}, definition=dict(mass=mass),
        conventions={}, parents=[]))


class StagePlanningTests(unittest.TestCase):
    def test_missing_partial_and_complete_are_read_only(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            expected, provenance = spec(), {"source": "test"}
            decision = stage_plan(output, expected, provenance)
            self.assertEqual((decision["action"], decision["missing"]),
                             ("compute", (0,1)))
            self.assertFalse(output.exists())
            with DatasetWriter(output, expected, provenance=provenance) as writer:
                writer.write(0, np.ones(2, dtype=np.complex128))
                active = stage_plan(output, expected, provenance)
                self.assertEqual((active["action"], active["completed"], active["missing"]),
                                 ("compute", (0,), (1,)))
            decision = stage_plan(output, expected, provenance)
            self.assertEqual((decision["completed"], decision["missing"]),
                             ((0,), (1,)))
            with DatasetWriter(output, expected, provenance=provenance, resume=True) as writer:
                self.assertEqual(writer.missing, (1,))
                writer.write(1, 2*np.ones(2, dtype=np.complex128))
                identity = writer.publish()
            decision = stage_plan(output, expected, provenance)
            self.assertEqual((decision["action"], decision["data_id"]),
                             ("reuse", identity))
            with self.assertRaisesRegex(ValueError,"definition"):
                stage_plan(output, spec(mass=0.5), provenance)
            with self.assertRaisesRegex(ValueError,"provenance"):
                stage_plan(output, expected, {"source": "changed"})

    def test_resume_rejects_corrupt_whole_partial_array(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            expected, provenance = spec(), {"source": "test"}
            with DatasetWriter(output, expected, provenance=provenance) as writer:
                writer.write(0, np.ones(2, dtype=np.complex128))
            self.assertEqual(inspect_dataset(output, cfg="cfg")["state"], "partial")
            temporary_array = output / ".state" / "cfg.npy"
            np.save(temporary_array, np.ones(1, dtype=np.complex128), allow_pickle=False)
            self.assertEqual(stage_plan(output, expected, provenance)["action"], "compute")
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                DatasetWriter(output, expected, provenance=provenance, resume=True)
            self.assertEqual(inspect_dataset(output, cfg="cfg")["state"], "partial")
            self.assertFalse((output / "cfg.npy").exists())
            self.assertFalse((output / "cfg.h5").exists())


if __name__ == "__main__":
    unittest.main()
