"""Shared scientific data semantics at actual planning/storage handoffs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.planning.dataset import plan_requirement
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter, inspect_dataset
from fixtures.data_contract import specification, publish


class DataContractHandoffTests(unittest.TestCase):
    def test_unknown_or_different_actual_basis_and_dtype_do_not_reuse(self):
        with TemporaryDirectory() as root:
            source = specification()
            directory = Path(root) / "dataset"
            publish(directory, source)
            observed = inspect_dataset(directory)
            self.assertEqual(plan_requirement(source, [observed])["action"], "reuse")
            self.assertEqual(plan_requirement(specification(basis=None), [observed])["action"], "blocked")
            for changed in (specification(basis="other-realization"), specification(dtype="complex64")):
                plan = plan_requirement(changed, [observed])
                self.assertEqual(plan["action"], "compute")
                self.assertIsNone(plan["data_id"])
                self.assertEqual(plan["missing"], (0, 1))

    def test_ordered_selection_is_explicit_and_reports_request_positions(self):
        with TemporaryDirectory() as root:
            directory = Path(root) / "dataset"
            identity = publish(directory, specification())
            observed = inspect_dataset(directory)
            request = specification(times=(0, 6), modes=(7, 4))
            self.assertEqual(plan_requirement(request, [observed])["action"], "compute")
            plan = plan_requirement(request, [observed], subset_axes=("time", "mode"))
            self.assertEqual(plan["data_id"], identity)
            self.assertEqual(plan["completed"], (0, 1))
            self.assertEqual(plan["missing"], ())
            self.assertEqual(plan["selection"], {"mode": [7, 4]})
            with DatasetReader(directory) as reader:
                np.testing.assert_array_equal(reader.read(time=0, select=plan["selection"]), [7 + 1j, 4 + 1j])

    def test_complete_candidates_do_not_stitch_or_promote_partial_coverage(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            first = root / "first"
            second = root / "second"
            publish(first, specification(times=(6,)))
            publish(second, specification(times=(0,)))
            observations = [inspect_dataset(first), inspect_dataset(second)]
            request = specification(times=(0, 6))
            plan = plan_requirement(request, observations, subset_axes=("time",))
            self.assertEqual(plan["action"], "compute")
            self.assertEqual(len(plan["completed"]), 1)
            self.assertEqual(len(plan["missing"]), 1)
            selected = next(value for value in observations if value["data_id"] == plan["data_id"])
            self.assertEqual(request.coordinates("time")[plan["completed"][0]], selected["spec"].coordinates("time")[0])
            partial = root / "partial"
            with DatasetWriter(partial, request, provenance={}) as writer:
                writer.write(0, np.zeros((3,), dtype=np.complex128))
            observed = inspect_dataset(partial)
            self.assertEqual(observed["state"], "partial")
            self.assertEqual(observed["completed"], (0,))
            self.assertEqual(plan_requirement(request, [observed])["completed"], ())
            with self.assertRaises(FileNotFoundError):
                DatasetReader(partial)



if __name__ == "__main__":
    unittest.main()
