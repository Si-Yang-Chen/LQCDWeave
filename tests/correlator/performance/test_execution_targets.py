import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.target import ExecutionTarget
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contraction.performance.model import CostProfile
from lqcdweave.contraction.performance.profile_io import save_cost_profile
from lqcdweave.contraction.performance.targets import (
    load_execution_target,
    register_execution_target,
    resolve_target_profile,
)


def target(name="cpu-test", *, fingerprint="unmeasured", profile=None):
    return ExecutionTarget(
        name=name,
        backend="numpy",
        runtime="cpu",
        memory=TargetHardwareMemory(host_available_bytes=4096),
        hardware_fingerprint=fingerprint,
        profile=profile,
        operations=["contract", "sparse_bridge"],
        software=[["python", "3.12"]],
    )


class ExecutionTargetRecordTests(unittest.TestCase):
    def test_record_normalizes_sequences_and_round_trips_json(self):
        original = target()
        encoded = json.dumps(original.to_dict())
        restored = ExecutionTarget.from_dict(json.loads(encoded))

        self.assertEqual(restored, original)
        self.assertIsInstance(restored.operations, tuple)
        self.assertIsInstance(restored.software, tuple)
        self.assertIsInstance(restored.software[0], tuple)

    def test_record_validates_required_declarations(self):
        with self.assertRaisesRegex(ValueError, "name"):
            target(" ")
        with self.assertRaisesRegex(ValueError, "backend/runtime"):
            ExecutionTarget("bad-runtime", "numpy", "cuda", TargetHardwareMemory())
        with self.assertRaisesRegex(TypeError, "TargetHardwareMemory"):
            ExecutionTarget("bad-memory", "numpy", "cpu", {})


class ExecutionTargetArchiveTests(unittest.TestCase):
    def test_display_name_can_use_unicode_spaces_and_host_prefix(self):
        record = target('csns:ln005 执行设备')
        with tempfile.TemporaryDirectory() as directory:
            saved = register_execution_target(record, directory)
            self.assertEqual(load_execution_target(record.name, [directory]), record)
            self.assertNotIn(':', saved.name)

    def test_register_and_load_target(self):
        with tempfile.TemporaryDirectory() as directory:
            registered = register_execution_target(target(), directory)
            loaded = load_execution_target("cpu-test", [directory])

            self.assertEqual(loaded, target())
            self.assertEqual(registered.name, "cpu-test.json")
            self.assertEqual(json.loads(registered.read_text())['format'],
                             "easy-distillation.execution-target")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_lookup_accepts_aliases_and_deduplicates_identical_documents(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            document = {
                "format": "easy-distillation.execution-target",
                "target": target().to_dict(),
                "aliases": ["cpu-alias"],
            }
            encoded = json.dumps(document)
            Path(first, "cpu-test.json").write_text(encoded, encoding="utf-8")
            Path(second, "cpu-test.json").write_text(encoded, encoding="utf-8")

            self.assertEqual(load_execution_target("cpu-alias", [first]), target())
            self.assertEqual(load_execution_target("cpu-test", [first, second]), target())

    def test_lookup_reports_missing_and_ambiguous_names(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            with self.assertRaisesRegex(ValueError, "missing-target"):
                load_execution_target("missing-target", [first])

            register_execution_target(target(fingerprint="machine-a"), first)
            register_execution_target(target(fingerprint="machine-b"), second)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                load_execution_target("cpu-test", [first, second])

    def test_target_record_never_imports_cupy_or_probes_hardware(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "cupy" or name.startswith("cupy."):
                raise AssertionError("ExecutionTarget must not import CuPy")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
                patch("builtins.__import__", side_effect=guarded_import), \
                patch("platform.node", side_effect=AssertionError("hardware probe")), \
                patch("platform.processor", side_effect=AssertionError("hardware probe")):
            record = target()
            register_execution_target(record, directory)
            loaded = load_execution_target(record.name, [directory])
            self.assertEqual(loaded, record)


class TargetCostProfileTests(unittest.TestCase):
    def test_model_is_found_by_registered_name_in_user_profiles(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'os.environ', {'LQCDWEAVE_PROFILE_DIR': directory}):
            save_cost_profile(CostProfile(version='named'), Path(directory, 'cpu-test.json'))
            self.assertEqual(resolve_target_profile(target())['profile'].version, 'named')

    def test_explicit_profile_precedes_target_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            implicit_path = Path(directory, "implicit.json")
            explicit_path = Path(directory, "explicit.json")
            save_cost_profile(
                CostProfile(version="implicit", hardware_fingerprint="board-a"),
                implicit_path,
            )
            save_cost_profile(
                CostProfile(version="explicit", hardware_fingerprint="board-a"),
                explicit_path,
            )

            resolved = resolve_target_profile(
                target(fingerprint="board-a", profile=str(implicit_path)),
                explicit_path,
            )
            self.assertEqual(resolved["profile"].version, "explicit")
            self.assertEqual(resolved["target"]["name"], "cpu-test")

    def test_known_profile_hardware_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory, "other-machine.json")
            save_cost_profile(
                CostProfile(hardware_fingerprint="board-b"), profile_path
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_target_profile(
                    target(fingerprint="board-a"), profile_path
                )

    def test_default_fallback_is_uncalibrated_and_inherits_target_fingerprint(self):
        resolved = resolve_target_profile(target(fingerprint="declared-board"))

        self.assertEqual(resolved["profile"].hardware_fingerprint, "declared-board")
        self.assertEqual(resolved["calibration_status"], "uncalibrated")
        self.assertEqual(resolved["coverage"]["calibration_status"], "uncalibrated")
        self.assertEqual(resolved["coverage"]["target_operations"],
                         ["contract", "sparse_bridge"])
        self.assertIn("do not by themselves", resolved["coverage"]["description"])

    def test_default_fallback_keeps_unmeasured_fingerprint_unmeasured(self):
        resolved = resolve_target_profile(target())
        self.assertEqual(resolved["profile"].hardware_fingerprint, "unmeasured")
        self.assertEqual(resolved["calibration_status"], "uncalibrated")


if __name__ == "__main__":
    unittest.main()
