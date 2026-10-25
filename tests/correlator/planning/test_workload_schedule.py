"""P's finite saved-workload scheduler and its public memory boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    ExecutionBudget,
    TensorSpec,
    WorkloadSpec,
)
from lqcdweave.contracts.memory import BudgetExceededError
from lqcdweave.contraction.planning.workload_schedule import (
    list_plan_candidates,
    plan_workload,
)
from lqcdweave.storage.contraction_plans.api import (
    ARCHIVE_VERSION,
    load_plan,
    save_plan,
    validate_bundle,
)
from lqcdweave.contraction.contraction_session import ContractionSession


class WorkloadScheduleTests(unittest.TestCase):
    def test_public_signatures_take_target_instead_of_backend(self):
        list_parameters = inspect.signature(list_plan_candidates).parameters
        self.assertEqual(
            tuple(list_parameters),
            ("spec", "budget", "max_candidates", "optimize", "target"),
        )
        self.assertEqual(list_parameters["target"].default, None)
        workload_parameters = inspect.signature(plan_workload).parameters
        self.assertEqual(tuple(workload_parameters)[-1], "target")
        self.assertEqual(workload_parameters["target"].default, None)
        self.assertNotIn("backend", list_parameters)
        self.assertNotIn("backend", workload_parameters)

    @staticmethod
    def _workload():
        arrays = {
            "a": np.arange(1, 7, dtype=np.float64).reshape(2, 3),
            "b": np.arange(2, 14, dtype=np.float64).reshape(3, 4),
            "c": np.arange(3, 11, dtype=np.float64).reshape(4, 2),
            "d": np.arange(4, 19, dtype=np.float64).reshape(3, 5),
            "e": np.arange(5, 15, dtype=np.float64).reshape(2, 5),
        }
        terms = (
            ContractionTerm(("a", "b"), "ij,jk->ik"),
            ContractionTerm(("b", "c"), "ij,jk->ik"),
            ContractionTerm(("a", "d"), "ij,jk->ik"),
            ContractionTerm(("c", "e"), "ij,jk->ik"),
        )
        workload = WorkloadSpec(
            specs={key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()},
            terms=terms,
            input_sources={key: {"family": key} for key in arrays},
            output_groups=((0, 1), (2,), (3,)),
        )
        return workload, arrays

    @staticmethod
    def _single_product_workload():
        specs = {
            "left": TensorSpec((2, 3), np.dtype("float64")),
            "right": TensorSpec((3, 4), np.dtype("float64")),
        }
        term = ContractionTerm(("left", "right"), "ik,kj->ij")
        return WorkloadSpec(specs, (term,), metadata={"assembly_extra_bytes": 13})

    @staticmethod
    def _budget(host_bytes=1 << 20, *, device_bytes=None, workspace_bytes=0,
                host_staging_bytes=0):
        return ExecutionBudget(
            host_bytes=host_bytes,
            device_bytes=device_bytes,
            min_reserve_bytes=0,
            workspace_bytes=workspace_bytes,
            host_staging_bytes=host_staging_bytes,
        )

    def test_candidate_order_output_groups_and_whole_workload_output_reserve(self):
        workload, _ = self._workload()

        candidates = list_plan_candidates(workload, self._budget())

        self.assertEqual([item.metadata["batch_width"] for item in candidates], [3, 1, 2])
        self.assertEqual([len(item.batches) for item in candidates], [1, 3, 2])
        self.assertEqual({item.metadata["optimize"] for item in candidates}, {"greedy"})
        self.assertEqual(
            [item.term_indices for item in candidates[1].batches],
            [(0, 1), (2,), (3,)],
        )
        # Independent shape arithmetic: the four float64 outputs have 64, 48,
        # 80 and 160 bytes, so every batch must reserve the complete 352 bytes.
        for batch in candidates[1].batches:
            self.assertEqual(batch.memory_estimate["output_bytes"], 352)
        self.assertEqual(
            [item.metadata["batch_width"]
             for item in list_plan_candidates(workload, self._budget(), max_candidates=2)],
            [3, 1],
        )
        for candidate in candidates:
            validate_bundle(candidate)

    def test_independent_memory_oracle_and_explicit_target_accounting(self):
        workload = self._single_product_workload()

        # Hand calculation for A(2x3) @ B(3x4), float64: inputs 144 B,
        # loader snapshots 144 B, one 2x4 result/intermediate 64 B, the
        # complete output 64 B, assembly reserve 13 B, and workspace
        # reserve 5 B plus packed operands/result (144+64=208 B).
        # Thus logical peak=642 B; staging adds 7 B to host requirements.
        expected_logical_peak = 144 + 144 + 64 + 64 + 13 + (5 + 208)
        self.assertEqual(expected_logical_peak, 642)

        gpu_budget = self._budget(
            host_bytes=295,
            device_bytes=expected_logical_peak,
            workspace_bytes=5,
            host_staging_bytes=7,
        )
        gpu_candidate = plan_workload(workload, gpu_budget, target="gpu")
        batch = gpu_candidate.batches[0]
        self.assertEqual(batch.memory_estimate["input_bytes"], 144)
        self.assertEqual(batch.memory_estimate["snapshot_bytes"], 144)
        self.assertEqual(batch.memory_estimate["intermediate_bytes"], 64)
        self.assertEqual(batch.memory_estimate["output_bytes"], 64)
        self.assertEqual(batch.memory_estimate["assembly_extra_bytes"], 13)
        self.assertEqual(batch.memory_estimate["workspace_bytes"], 213)
        self.assertEqual(batch.memory_estimate["logical_peak_bytes"], expected_logical_peak)
        self.assertEqual(batch.memory_estimate["host_required_bytes"], 295)

        # target=None with a device cap preserves the old metadata-only GPU
        # inference. A one-byte short device cap is rejected before execution.
        short_gpu_budget = self._budget(
            host_bytes=1000,
            device_bytes=expected_logical_peak - 1,
            workspace_bytes=5,
            host_staging_bytes=7,
        )
        with self.assertRaisesRegex(BudgetExceededError, "device bytes"):
            list_plan_candidates(workload, short_gpu_budget)

        # With no device cap, target=None retains CPU host-peak accounting.
        no_device_cap = self._budget(
            host_bytes=500,
            workspace_bytes=5,
            host_staging_bytes=7,
        )
        with self.assertRaisesRegex(BudgetExceededError, "host bytes"):
            list_plan_candidates(workload, no_device_cap)

        # An explicit CPU target uses the host peak and ignores a logical
        # device cap, matching the old explicit-NumPy accounting behavior.
        cpu_candidate = plan_workload(workload, short_gpu_budget, target="cpu")
        self.assertEqual(cpu_candidate.batches[0].memory_estimate["host_required_bytes"], 649)
        self.assertEqual(cpu_candidate.budget.device_bytes, expected_logical_peak - 1)

    def test_too_small_budget_and_retired_or_invalid_options_keep_clear_errors(self):
        workload, _ = self._workload()

        with self.assertRaisesRegex(BudgetExceededError, "No sampled schedule fits"):
            list_plan_candidates(workload, self._budget(host_bytes=1))
        with self.assertRaisesRegex(ValueError, "optimize='joint'.*retired"):
            plan_workload(workload, self._budget(), optimize="joint")
        with self.assertRaisesRegex(TypeError, "target must be"):
            list_plan_candidates(workload, self._budget(), target=False)
        with self.assertRaisesRegex(ValueError, "target must be"):
            list_plan_candidates(workload, self._budget(), target="accelerator")

    def test_module_has_no_backend_probe_or_runtime_adapter_dependency(self):
        module_path = (
            Path(__file__).parents[3]
            / "src"
            / "lqcdweave"
            / "contraction"
            / "planning"
            / "workload_schedule.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        forbidden = (
            "importlib",
            "cupy",
            "lqcdweave.contraction.budget",
            "lqcdweave.runtime",
            "lqcdweave.adapters",
        )
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(
            [name for name in imported if any(
                name == prefix or name.startswith(prefix + ".") for prefix in forbidden
            )],
            imported,
        )

    def test_saved_bundle_roundtrip_and_session_execution_match_direct_oracle(self):
        workload, arrays = self._workload()
        selected = plan_workload(workload, self._budget())
        self.assertEqual(selected.metadata["batch_width"], 3)
        validate_bundle(selected)

        cfg = 2
        bound = {
            key: value + cfg * (index + 1)
            for index, (key, value) in enumerate(arrays.items())
        }

        def loader(requested_cfg, key, source):
            self.assertEqual(requested_cfg, cfg)
            self.assertEqual(source["family"], key)
            return bound[key]

        expected = (
            bound["a"] @ bound["b"],
            bound["b"] @ bound["c"],
            bound["a"] @ bound["d"],
            bound["c"] @ bound["e"],
        )
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "scheduled-plan.json"
            save_plan(selected, archive)
            restored = load_plan(
                archive,
                expected_signature=selected.signature,
                expected_workload=workload,
            )
            self.assertEqual(restored.signature, selected.signature)
            self.assertEqual(restored.metadata["batch_width"], 3)
            session = ContractionSession.from_plan(
                restored,
                backend="numpy",
                validate_bundle=validate_bundle,
            )
            actual = session.execute_cfg(cfg, loader).values

        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, reference, rtol=0, atol=0)
        self.assertEqual(ARCHIVE_VERSION, 8)


if __name__ == "__main__":
    unittest.main()
