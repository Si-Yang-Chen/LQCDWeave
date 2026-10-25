"""Project graph planning uses only explicitly supplied dataset receipts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.runtime.datasets import (
    read_dataset_request_receipt, save_dataset_request_receipt,
)
from lqcdweave.runtime.project_datasets import (
    plan_project_dataset_requests, run_project_dataset_requests,
)


def _spec(*, kind="field", times=(0, 1), parents=(), definition=None):
    return DatasetSpec(dict(
        schema=1,
        kind=kind,
        configuration=dict(ensemble="ensemble-a", configuration="cfg-a",
                           lattice=[1, 1, 1, len(times)]),
        axes=[dict(name="time", coordinates=list(times)),
              dict(name="component", coordinates=["value"])],
        dtype="float64",
        bases={},
        definition=(dict(stage=kind) if definition is None else definition),
        conventions=dict(normalization="unscaled"),
        parents=list(parents),
    ))


def _row(time):
    return np.asarray([float(time + 1)], dtype=np.float64)


class ProjectDatasetCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.receipts = self.root / "receipts"
        self.project_id = "project-a"

    def _publish(self, name, spec, *, offset=0.0):
        directory = self.root / name
        with DatasetWriter(directory, spec,
                            provenance={"producer": "project-coverage-fixture"}) as writer:
            for position, time in enumerate(spec.coordinates("time")):
                writer.write(position, _row(time) + offset)
            identity = writer.publish()
        self.assertEqual(self.index.register(directory), identity)
        return identity, directory

    def _receipt(self, name, spec, identity):
        request = DatasetRequest(spec, data_id=identity)
        return save_dataset_request_receipt(
            self.receipts / name, {"old-name": request}, self.index,
            {"old-name": identity}, project_id=self.project_id)

    def test_query_uses_only_complete_datasets_named_by_receipts(self):
        spec = _spec()
        identity, _ = self._publish("listed-source", spec)
        decoy_id, _ = self._publish("unlisted-decoy", spec, offset=10.0)
        receipt = self._receipt("listed", spec, identity)
        request = DatasetRequest(spec)

        with patch.object(self.index, "observations",
                          side_effect=AssertionError("project query scanned the index")):
            result = plan_project_dataset_requests(
                {"new-name": request}, self.index, (receipt,),
                expected_project_id=self.project_id)

        self.assertEqual(result["project_id"], self.project_id)
        self.assertTrue(result["project_id_verified"])
        self.assertEqual(result["stages"]["new-name"]["action"], "reuse")
        self.assertEqual(result["stages"]["new-name"]["data_id"], identity)
        self.assertNotEqual(identity, decoy_id)

    def test_empty_receipts_offer_no_automatic_candidates(self):
        request = DatasetRequest(_spec())
        result = plan_project_dataset_requests(
            {"new-name": request}, self.index, (),
            expected_project_id=self.project_id)
        self.assertEqual(result["stages"]["new-name"]["action"], "compute")
        self.assertEqual(result["project_id"], self.project_id)
        self.assertFalse(result["project_id_verified"])

    def test_wrong_project_label_is_rejected(self):
        spec = _spec()
        identity, _ = self._publish("source", spec)
        receipt = self._receipt("source", spec, identity)
        with self.assertRaisesRegex(ValueError, "project_id"):
            plan_project_dataset_requests(
                {"new-name": DatasetRequest(spec)}, self.index, (receipt,),
                expected_project_id="other-project")

    def test_run_reuses_receipt_input_and_publishes_a_child(self):
        source_spec = _spec(kind="source")
        source_id, _ = self._publish("source", source_spec)
        receipt = self._receipt("source", source_spec, source_id)

        child_document = _spec(kind="derived").document
        child_document["parents"] = [{"request": "source"}]
        child_document["definition"] = {
            "operation": "scale-by-two",
            "input_roles": {"source": {"request": "source"}},
        }
        requests = {
            "source": DatasetRequest(source_spec),
            "child": DatasetRequest(child_document),
        }
        output = self.root / "child-output"
        calls = []

        def produce(spec, inputs, writer):
            calls.append(tuple(writer.missing))
            for position in writer.missing:
                time = spec.coordinates("time")[position]
                writer.write(position, 2.0 * inputs["source"].read(time=time))

        with patch.object(self.index, "observations",
                          side_effect=AssertionError("project run scanned the index")):
            identities = run_project_dataset_requests(
                requests, self.index,
                {"child": output},
                {"child": {"producer": "project-coverage-test"}},
                {"child": produce},
                (receipt,), expected_project_id=self.project_id)

        self.assertEqual(identities["source"], source_id)
        self.assertEqual(calls, [(0, 1)])
        with self.index.open(identities["child"]) as reader:
            self.assertEqual(reader.spec.document["parents"], [source_id])
            for time in (0, 1):
                np.testing.assert_array_equal(reader.read(time=time), 2.0 * _row(time))

        child_receipt = save_dataset_request_receipt(
            self.receipts / "child", requests, self.index, identities,
            project_id=self.project_id)
        checked = read_dataset_request_receipt(
            child_receipt, self.index, expected_project_id=self.project_id)
        self.assertEqual(checked["project_id"], self.project_id)
        self.assertEqual(checked["resolved_graph"]["bindings"]["source"], source_id)


if __name__ == "__main__":
    unittest.main()