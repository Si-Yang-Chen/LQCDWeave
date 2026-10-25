import json
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import weakref
from collections import UserDict
from types import MappingProxyType

import numpy as np

from lqcdweave.contraction import compile_plan
from lqcdweave.contracts.correlator import api as contracts
from lqcdweave.storage.contraction_plans.api import (
    ARCHIVE_FORMAT,
    ARCHIVE_VERSION,
    PlanArchiveError,
    decode_plan_value,
    encode_plan_value,
    load_plan,
    save_plan,
)


class PlanArchiveApiTests(unittest.TestCase):
    def setUp(self):
        specs = {
            "input": contracts.TensorSpec(
                (3,), np.dtype("complex128"),
                description=contracts.TensorDescription(
                    axes=("time",),
                    data_identity="ensemble-A",
                    selection_identity="source-times-0-2",
                    role="perambulator",
                    source_bindings=("dataset-A",),
                    transform=("gamma5", "dagger"),
                    semantics={"phase": "periodic", "basis": "eigen-7"},
                    bindings={"configuration": "cfg-003", "source_time": 0},
                ),
            ),
        }
        terms = (contracts.ContractionTerm(("input",), "i->i", 1 + 2j),)
        workload = contracts.WorkloadSpec(
            specs,
            terms,
            input_sources={"input": {"family": "perambulator", "selection": "source-times-0-2"}},
            convention="explicit-v1",
            metadata={"output_recipes": [[0]], "calibration": {"sample_count": 3}},
        )
        plan = compile_plan(specs, terms, optimize="greedy")
        self.bundle = contracts.PlanBundle(
            workload,
            (contracts.PlanBatch(plan, (0,), {"host_bytes": 96}),),
            contracts.ExecutionBudget(host_bytes=4096, workspace_bytes=64),
            {"performance_fingerprint": {"backend": "numpy", "device": "cpu"}},
        )

    def test_v4_plan_bundle_round_trips_paths_bytes_and_json_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            save_plan(self.bundle, path)
            raw = path.read_bytes()
            document = json.loads(raw.decode("utf-8"))
            self.assertEqual(document["format"], ARCHIVE_FORMAT)
            self.assertEqual(document["version"], ARCHIVE_VERSION)
            self.assertEqual(document["payload_type"], "plan-bundle")
            self.assertEqual(set(document), {"format", "version", "payload_type", "payload"})
            self.assertEqual(document["payload"]["signature"], self.bundle.signature)

            from_path = load_plan(path, expected_signature=self.bundle.signature,
                                  expected_workload=self.bundle.workload)
            from_bytes = load_plan(raw, expected_workload=self.bundle.workload)
            from_text = load_plan(raw.decode("utf-8"), expected_workload=self.bundle.workload)
            self.assertEqual(from_path.signature, self.bundle.signature)
            self.assertEqual(from_bytes.signature, self.bundle.signature)
            self.assertEqual(from_text.signature, self.bundle.signature)
            self.assertEqual(from_path.workload.specs["input"].description,
                             self.bundle.workload.specs["input"].description)

    def test_structural_signature_and_payload_types_are_checked(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            save_plan(self.bundle, path)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["payload"]["signature"] = "0" * 64
            with self.assertRaisesRegex(PlanArchiveError, "structural signature"):
                load_plan(json.dumps(document))

    def test_previous_envelope_version_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            save_plan(self.bundle, path)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["version"] = 3
            with self.assertRaisesRegex(PlanArchiveError, "unsupported plan archive version"):
                load_plan(json.dumps(document))

    def test_plan_codec_is_whitelisted_and_keeps_description_identity(self):
        description = self.bundle.workload.specs["input"].description
        decoded = decode_plan_value(encode_plan_value(description))
        self.assertEqual(decoded, description)
        with self.assertRaisesRegex(PlanArchiveError, "unsupported plan value"):
            encode_plan_value(np.arange(3))
        with self.assertRaisesRegex(PlanArchiveError, "unsupported plan value"):
            encode_plan_value(lambda: None)

    def test_repeated_record_types_keep_current_values_and_schema_checks(self):
        first = contracts.TensorSpec((2,), np.dtype("float32"))
        second = contracts.TensorSpec((3, 4), np.dtype("complex128"))
        for expected in (first, second, first):
            self.assertEqual(decode_plan_value(encode_plan_value(expected)), expected)
        # Metadata reuse must not accept new fields or broaden the existing
        # distinction between current and legacy optional-field defaults.
        for alteration in ("extra", "required", "optional"):
            encoded = encode_plan_value(second)
            if alteration == "extra":
                encoded["value"]["unexpected"] = 1
            else:
                encoded["value"].pop("dtype" if alteration == "required" else "description")
            with self.subTest(alteration=alteration):
                with self.assertRaises(PlanArchiveError):
                    decode_plan_value(encoded)
        legacy = encode_plan_value(second)
        legacy["type"] = "TensorSpec"
        legacy["value"].pop("description")
        self.assertEqual(decode_plan_value(legacy), second)

    def test_codec_metadata_does_not_retain_decoded_records(self):
        def decode_once():
            constant = contracts.ArrayConstant.from_array(np.arange(8, dtype=np.complex128))
            decoded = decode_plan_value(encode_plan_value(constant))
            np.testing.assert_array_equal(decoded.array(), constant.array())
            return weakref.ref(decoded)

        previous = decode_once()
        gc.collect()
        self.assertIsNone(previous())

    def test_non_dict_mappings_and_invalid_scalar_or_type_still_use_the_codec(self):
        expected = contracts.TensorSpec((2, 3), np.dtype("complex128"))
        encoded = encode_plan_value(expected)
        encoded["value"] = UserDict(encoded["value"])
        self.assertEqual(decode_plan_value(MappingProxyType(encoded)), expected)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PlanArchiveError, "non-finite"):
                    decode_plan_value(value)
                with self.assertRaisesRegex(PlanArchiveError, "non-finite"):
                    encode_plan_value(value)
        with self.assertRaisesRegex(PlanArchiveError, "unknown plan archive type"):
            decode_plan_value({"type": "os.system", "value": {}})

    def test_literal_children_keep_nesting_limit_and_float_validation(self):
        for container in ("python.tuple", "mapping"):
            for literal in (None, "value", 7, True):
                encoded = literal
                for _ in range(128):
                    encoded = {"type": container, "value": (
                        [encoded] if container == "python.tuple" else [["key", encoded]])}
                with self.subTest(container=container, literal=literal):
                    decode_plan_value(encoded)
                    excessive = {"type": "python.tuple", "value": [encoded]}
                    with self.assertRaisesRegex(PlanArchiveError, "nesting"):
                        decode_plan_value(excessive)
        for value in (float("nan"), float("inf"), -float("inf")):
            record = encode_plan_value(contracts.OutputNode("constant", 1.0))
            record["value"]["value"] = value
            for encoded in (record, {"type": "python.tuple", "value": [value]},
                            {"type": "mapping", "value": [["key", value]]}):
                with self.subTest(value=value), self.assertRaisesRegex(PlanArchiveError, "non-finite"):
                    decode_plan_value(encoded)


if __name__ == "__main__":
    unittest.main()
