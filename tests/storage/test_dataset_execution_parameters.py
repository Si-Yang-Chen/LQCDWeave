"""Execution policy is recorded on producer attempts, outside data identity."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, stable_digest
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, read_dataset_metadata


def _spec():
    return DatasetSpec(dict(schema=1, kind="execution-parameter-fixture",
        configuration=dict(ensemble="fixture", configuration="cfg-ep", lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=[0, 1]),
              dict(name="component", coordinates=["value"])], dtype="float64",
        bases={}, definition={"operator": "test"}, conventions={"normalization": "unscaled"},
        parents=[]))


def _row(value):
    return np.asarray([float(value)], dtype=np.float64)


def _context(run_id):
    return dict(run_id=run_id, producer_attempt_id=None, stage="measurement")


class DatasetExecutionParameterTests(unittest.TestCase):
    def test_parameters_are_detached_and_saved_outside_the_scientific_identity(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "parameterized"
            parameters = {"solver": {"iterations": 8, "options": [1, 2, 3]}}
            with DatasetWriter(directory, _spec(), provenance={"producer": "execution-test"},
                               attempt_context=_context("run-1")) as writer:
                writer.bind_producer_attempt("attempt-1", execution_parameters=parameters)
                parameters["solver"]["options"].append(4)
                writer.write(0, _row(1))
                writer.write(1, _row(2))
                data_id = writer.publish()

            metadata = read_dataset_metadata(directory, cfg="cfg-ep")
            attempt, = metadata["execution"]["attempts"]
            self.assertEqual(attempt["execution_parameters"], {
                "solver": {"iterations": 8, "options": [1, 2, 3]}})
            self.assertEqual(metadata["data_id"], data_id)
            self.assertEqual(stable_digest(metadata["record"]["identity"]), data_id)
            self.assertNotIn("execution_parameters", metadata["record"]["identity"])
            with DatasetReader(directory, cfg="cfg-ep") as reader:
                self.assertEqual(reader.execution["attempts"][0]["execution_parameters"],
                                 attempt["execution_parameters"])

    def test_resume_records_each_real_attempt_and_preserves_completed_rows(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "resumed"
            provenance = {"producer": "execution-test"}
            original = _row(14)
            with DatasetWriter(directory, _spec(), provenance=provenance,
                               attempt_context=_context("run-first")) as writer:
                writer.bind_producer_attempt("attempt-first",
                    execution_parameters={"solver": {"iterations": 4}})
                writer.write(0, original)

            with DatasetWriter(directory, _spec(), provenance=provenance, resume=True,
                               attempt_context=_context("run-second")) as writer:
                self.assertEqual(writer.missing, (1,))
                writer.bind_producer_attempt("attempt-second",
                    execution_parameters={"solver": {"iterations": 9}})
                writer.write(1, _row(22))
                data_id = writer.publish()

            metadata = read_dataset_metadata(directory, cfg="cfg-ep")
            first, second = metadata["execution"]["attempts"]
            self.assertEqual([first["status"], second["status"]], ["interrupted", "complete"])
            self.assertEqual(first["execution_parameters"], {"solver": {"iterations": 4}})
            self.assertEqual(second["execution_parameters"], {"solver": {"iterations": 9}})
            with DatasetReader(directory, cfg="cfg-ep") as reader:
                self.assertEqual(reader.data_id, data_id)
                np.testing.assert_array_equal(reader.read(time=0), original)
                np.testing.assert_array_equal(reader.read(time=1), _row(22))


if __name__ == "__main__":
    unittest.main()
