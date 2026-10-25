"""Integer-only compiled archives and explicit offline v4 conversion."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator import api as c
from lqcdweave.contraction.planning.api import compile_plan
from lqcdweave.contraction.execution.api import execute_plan
from lqcdweave.storage.contraction_plans.api import (
    save_plan, load_plan, migrate_plan_bundle, PlanArchiveError,
)


class IntegerBundleTests(unittest.TestCase):
    def bundle(self):
        specs = {"a": c.TensorSpec((2, 3), "float64"), "b": c.TensorSpec((2, 3), "float64")}
        terms = (c.ContractionTerm(("a", "b"), "ij,ij->i", 2),)
        return c.PlanBundle(c.WorkloadSpec(specs, terms),
                            (c.PlanBatch(compile_plan(specs, terms), (0,)),), c.ExecutionBudget())

    def legacy_document(self):
        # Independent old schema fixture: one binary operation with ordered
        # dependencies, no search history or arrays.
        bundle = self.bundle()
        document = {"format": "easy-distillation.contraction-plan", "version": 4,
                    "payload_type": "plan-bundle", "payload": {}}
        payload = document["payload"]
        workload = c.encode_workload(bundle.workload)
        workload["terms"][0]["subscripts"] = "ij,ij->i"
        plan = c.encode_plan(bundle.batches[0].plan)
        plan["operations"][0]["input_subscripts"] = ["ij", "ij"]
        plan["operations"][0]["output_subscript"] = "i"
        plan["terms"][0]["output_subscript"] = "i"
        payload.update(workload=workload, batches=[{"term_indices": [0], "memory_estimate": {}, "plan": plan}],
                       budget=bundle.budget.as_dict(), metadata={})
        payload["signature"] = c.json_digest({"workload": workload, "batches": [{"term_indices": [0], "plan": plan}]})
        return document

    def test_integer_roundtrip_does_not_search_or_render(self):
        bundle = self.bundle()
        with TemporaryDirectory() as directory:
            path = Path(directory)/"plan.json"
            save_plan(bundle, path)
            document = json.loads(path.read_text())
            self.assertEqual(document["version"], 8)
            self.assertEqual(document["payload"]["workload"]["terms"][0]["subscripts"],
                             {"inputs": [[105, 106], [105, 106]], "output": [105]})
            operation = document["payload"]["batches"][0]["plan"]["operations"][0]
            self.assertEqual(operation["input_subscripts"], [[105, 106], [105, 106]])
            with patch("opt_einsum.contract_path", side_effect=AssertionError("search")), \
                 patch("lqcdweave.contraction.execution.executor.einsum_equation", side_effect=AssertionError("string")):
                loaded = load_plan(path, expected_workload=bundle.workload)
                a = np.arange(6.).reshape(2, 3)
                actual = execute_plan(loaded.batches[0].plan, {"a": a, "b": a}).values[0]
            np.testing.assert_array_equal(actual, [10., 100.])
            self.assertIsInstance(loaded.workload.terms[0].subscripts, c.AxisPattern)

    def test_old_bundle_requires_explicit_conversion_and_preserves_cost_and_path(self):
        old = self.legacy_document()
        with self.assertRaisesRegex(PlanArchiveError, "version"):
            load_plan(json.dumps(old))
        with TemporaryDirectory() as directory:
            path = Path(directory)/"new.json"
            with patch("opt_einsum.contract_path", side_effect=AssertionError("search")):
                migrate_plan_bundle(json.dumps(old), path)
                new = load_plan(path)
            self.assertEqual(new.batches[0].plan.operations, self.bundle().batches[0].plan.operations)
            self.assertEqual(new.budget, self.bundle().budget)

    def test_corrupt_legacy_signature_or_integer_ids_fail_before_publish(self):
        old = self.legacy_document()
        old["payload"]["workload"]["terms"][0]["subscripts"] = "ij,ij->j"
        with TemporaryDirectory() as directory:
            path = Path(directory)/"new.json"
            with self.assertRaisesRegex(PlanArchiveError, "signature"):
                migrate_plan_bundle(json.dumps(old), path)
            self.assertFalse(path.exists())
            malformed = self.legacy_document()
            malformed["payload"]["workload"]["metadata"] = False
            with self.assertRaisesRegex(PlanArchiveError, "invalid legacy"):
                migrate_plan_bundle(json.dumps(malformed), path)
            self.assertFalse(path.exists())
            save_plan(self.bundle(), path)
            document = json.loads(path.read_text())
            for invalid in (True, -1, "i", 1.5):
                broken = deepcopy(document)
                broken["payload"]["batches"][0]["plan"]["operations"][0]["input_subscripts"][0][0] = invalid
                with self.subTest(invalid=invalid), self.assertRaises(PlanArchiveError):
                    load_plan(json.dumps(broken))


if __name__ == "__main__":
    unittest.main()
