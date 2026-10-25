"""Direct contract tests for X's saved PlanBundle execution service."""

from pathlib import Path
from types import ModuleType
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.memory import BudgetExceededError
from lqcdweave.contraction.execution.api import (
    PlanMismatchError,
    SourceIdentityError,
)
from lqcdweave.contraction.execution import api as x_api
from lqcdweave.contraction.execution import scheduled as x_scheduled
from lqcdweave.contraction.execution.api import (
    BundleExecutionResult,
    BundleExecutionStats,
    PreparedBundle,
    execute_bundle,
    prepare_bundle_execution,
)
from lqcdweave.contraction.planning.api import compile_plan
from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    ExecutionBudget,
    PlanBatch,
    PlanBundle,
    TensorSpec,
    WorkloadSpec,
)
from lqcdweave.storage.contraction_plans.api import (
    load_plan,
    save_plan,
    validate_bundle,
)


class SavedBundleExecutionTests(unittest.TestCase):
    @staticmethod
    def _bundle(*, budget=None):
        specs = {
            key: TensorSpec((2,), np.complex128)
            for key in ("a", "b", "c")
        }
        terms = (
            ContractionTerm(("a", "b"), "i,i->", coefficient=0.5 + 0.25j),
            ContractionTerm(("b", "c"), "i,i->", coefficient=-1 + 0.5j),
            ContractionTerm(("a", "c"), "i,i->", coefficient=0.75 - 0.5j),
        )
        groups = ((0, 1), (2,))
        workload = WorkloadSpec(
            specs=specs,
            terms=terms,
            input_sources={
                key: {"family": "shared-getter-buffer", "key": key}
                for key in specs
            },
            convention="saved-bundle-execution-test-v1",
            output_groups=groups,
        )
        batches = []
        for term_indices in groups:
            selected = tuple(terms[index] for index in term_indices)
            input_keys = tuple(dict.fromkeys(
                key for term in selected for key in term.inputs
            ))
            plan = compile_plan(
                {key: specs[key] for key in input_keys}, selected,
                optimize="greedy",
            )
            batches.append(PlanBatch(plan, term_indices))
        return PlanBundle(
            workload,
            tuple(batches),
            budget or ExecutionBudget(host_bytes=1 << 20, min_reserve_bytes=0),
        )

    @staticmethod
    def _samples():
        return {
            2: {
                "a": np.array([1 + 2j, 3 - 1j], dtype=np.complex128),
                "b": np.array([2 - 3j, -1 + 4j], dtype=np.complex128),
                "c": np.array([5 + 1j, -2 - 2j], dtype=np.complex128),
            },
            5: {
                "a": np.array([-2 + 0.5j, 1 + 3j], dtype=np.complex128),
                "b": np.array([0.25 - 2j, 4 + 0.75j], dtype=np.complex128),
                "c": np.array([3 - 1j, -0.5 + 2j], dtype=np.complex128),
            },
        }

    @staticmethod
    def _reference(samples, cfg):
        values = samples[cfg]
        terms = (
            np.einsum("i,i->", values["a"], values["b"], optimize=False)
            * (0.5 + 0.25j),
            np.einsum("i,i->", values["b"], values["c"], optimize=False)
            * (-1 + 0.5j),
            np.einsum("i,i->", values["a"], values["c"], optimize=False)
            * (0.75 - 0.5j),
        )
        return terms

    class _SharedBufferLoader:
        def __init__(self, samples):
            self.samples = samples
            self.buffer = np.empty(2, dtype=np.complex128)
            self.calls = []

        def source_identity(self, cfg, key, source):
            return source

        def __call__(self, cfg, key, source):
            self.calls.append((cfg, key, source["key"]))
            self.buffer[...] = self.samples[cfg][key]
            return self.buffer

    class _WrongIdentityLoader(_SharedBufferLoader):
        def source_identity(self, cfg, key, source):
            return {"family": "different-source", "key": key}

    def test_public_api_executes_archived_bundle_with_owned_snapshots_and_group_tail(self):
        required = {
            "BundleExecutionResult",
            "BundleExecutionStats",
            "execute_bundle",
            "PreparedBundle",
            "prepare_bundle_execution",
        }
        self.assertTrue(required.issubset(set(x_api.__all__)))

        bundle = self._bundle()
        samples = self._samples()
        loader = self._SharedBufferLoader(samples)
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "saved-bundle.json"
            save_plan(bundle, archive)
            restored = load_plan(archive, expected_signature=bundle.signature)
        validate_bundle(restored)
        self.assertEqual(restored.workload.output_groups, ((0, 1), (2,)))

        prepared = prepare_bundle_execution(restored, backend="numpy")
        self.assertIsInstance(prepared, PreparedBundle)
        with (
            patch(
                "lqcdweave.contraction.execution.scheduled.prepare_bundle_execution",
                side_effect=AssertionError("prepared metadata must be reused"),
            ),
            patch(
                "lqcdweave.contraction.planning.api.compile_plan",
                side_effect=AssertionError("execution must not compile or search a path"),
            ),
        ):
            first = execute_bundle(
                restored, 2, loader, backend="numpy", prepared=prepared,
            )
            second = execute_bundle(
                restored, 5, loader, backend="numpy", prepared=prepared,
            )

        self.assertIsInstance(first, BundleExecutionResult)
        self.assertIsInstance(first.stats, BundleExecutionStats)
        for actual, expected in zip(first.values, self._reference(samples, 2), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        for actual, expected in zip(second.values, self._reference(samples, 5), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        self.assertEqual(
            tuple(tuple(batch["term_indices"]) for batch in first.stats.batches),
            ((0, 1), (2,)),
        )
        self.assertEqual(
            loader.calls,
            [(cfg, key, key) for cfg in (2, 5) for key in ("a", "b", "c")],
        )
        # The second cfg reused the getter's mutable array. Earlier outputs stay
        # unchanged because input snapshots and result arrays own their storage.
        for actual, expected in zip(first.values, self._reference(samples, 2), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_prepared_bundle_and_loader_identity_errors_precede_reads(self):
        bundle = self._bundle()
        prepared = prepare_bundle_execution(bundle, backend="numpy")
        foreign_bundle = self._bundle()
        foreign_loader = self._SharedBufferLoader(self._samples())
        with self.assertRaisesRegex(PlanMismatchError, "different bundle"):
            execute_bundle(
                foreign_bundle, 2, foreign_loader,
                backend="numpy", prepared=prepared,
            )
        self.assertEqual(foreign_loader.calls, [])

        wrong_loader = self._WrongIdentityLoader(self._samples())
        with self.assertRaisesRegex(SourceIdentityError, "does not match"):
            execute_bundle(
                bundle, 2, wrong_loader, backend="numpy", prepared=prepared,
            )
        self.assertEqual(wrong_loader.calls, [])

    def test_shape_and_dtype_errors_are_rejected_before_a_result_is_delivered(self):
        bundle = self._bundle()
        cases = (
            ("shape", np.zeros(3, dtype=np.complex128), "shape"),
            ("dtype", np.zeros(2, dtype=np.float64), "dtype"),
        )
        for label, invalid, message in cases:
            with self.subTest(label=label):
                calls = []

                def loader(cfg, key, source):
                    calls.append((cfg, key))
                    return invalid

                with self.assertRaisesRegex(PlanMismatchError, message):
                    execute_bundle(bundle, 2, loader, backend="numpy")
                self.assertEqual(calls, [(2, "a")])

    def test_actual_small_host_budget_rejects_before_loader_calls(self):
        bundle = self._bundle(
            budget=ExecutionBudget(host_bytes=1, min_reserve_bytes=0),
        )
        loader = self._SharedBufferLoader(self._samples())

        with self.assertRaises(BudgetExceededError):
            execute_bundle(bundle, 2, loader, backend="numpy")

        self.assertEqual(loader.calls, [])

    def test_prepare_normalizes_numpy_names_and_rejects_unsupported_backends(self):
        bundle = self._bundle()
        for backend in (np, "numpy", "NumPy", "NUMPY"):
            with self.subTest(backend=backend):
                prepared = prepare_bundle_execution(bundle, backend=backend)
                self.assertIsInstance(prepared, PreparedBundle)
                self.assertEqual(prepared.kernels, ())

        for backend in ("cuda", object()):
            with self.subTest(backend=backend):
                with patch.object(x_scheduled, "prepare_plan_kernels") as prepare_kernels:
                    with self.assertRaisesRegex(ValueError, "backend must be NumPy or CuPy"):
                        prepare_bundle_execution(bundle, backend=backend)
                prepare_kernels.assert_not_called()

    def test_cupy_module_and_case_insensitive_name_prepare_the_same_kernel_path(self):
        bundle = self._bundle()
        fake_cupy = ModuleType("cupy")
        with (
            patch("importlib.import_module", return_value=fake_cupy) as import_module,
            patch.object(
                x_scheduled,
                "prepare_plan_kernels",
                side_effect=lambda plan: ("mock-kernel", id(plan)),
            ) as prepare_kernels,
        ):
            by_module = prepare_bundle_execution(bundle, backend=fake_cupy)
            by_name = prepare_bundle_execution(bundle, backend="CuPy")

        self.assertEqual(by_module.kernels, by_name.kernels)
        self.assertEqual(len(by_module.kernels), len(bundle.batches))
        expected_plans = tuple(batch.plan for batch in bundle.batches)
        self.assertEqual(len(prepare_kernels.call_args_list), 2 * len(expected_plans))
        for call, expected in zip(prepare_kernels.call_args_list, expected_plans * 2, strict=True):
            self.assertIs(call.args[0], expected)
        import_module.assert_called_once_with("cupy")

    def test_workflow_session_keeps_the_normalized_numpy_module_call(self):
        from lqcdweave.contraction.contraction_session import ContractionSession

        bundle = self._bundle()
        loader = self._SharedBufferLoader(self._samples())
        session = ContractionSession.from_plan(
            bundle, backend=np, validate_bundle=validate_bundle,
        )
        self.assertIs(session.backend, np)
        result = session.execute_cfg(2, loader)
        for actual, expected in zip(result.values, self._reference(self._samples(), 2), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
