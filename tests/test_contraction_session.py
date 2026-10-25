"""Persisted execution sessions consume storage through injected callbacks."""

from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    ExecutionBudget,
    PlanBatch,
    PlanBundle,
    TensorSpec,
    WorkloadSpec,
)
from lqcdweave.contraction.planning.api import compile_plan
from lqcdweave.contraction.contraction_session import ContractionSession
from lqcdweave.storage.contraction_plans.api import (
    load_plan as archive_load_plan,
    save_plan,
    validate_bundle as archive_validate_bundle,
)


class ContractionSessionStorageInjectionTests(unittest.TestCase):
    @staticmethod
    def _bundle():
        workload = WorkloadSpec(
            specs={"input": TensorSpec((2,), np.float64)},
            terms=(ContractionTerm(("input",), "i->i"),),
            input_sources={"input": {"family": "test-vector", "version": 1}},
        )
        plan = compile_plan(workload.specs, workload.terms)
        return PlanBundle(
            workload,
            (PlanBatch(plan, (0,)),),
            ExecutionBudget(host_bytes=1 << 20, min_reserve_bytes=0),
        )

    def test_dynamic_tensor_session_keeps_greedy_execution(self):
        values = np.array([1 + 2j, 3 - 1j, -2 + 0.5j], dtype=np.complex128)
        session = ContractionSession(backend="numpy")
        result = session.tensors(
            {"input": values}, (ContractionTerm(("input",), "i->"),),
        )
        self.assertEqual(session.optimize, "greedy")
        np.testing.assert_allclose(result.values, (np.sum(values),))

    def test_default_backend_is_actual_numpy(self):
        session = ContractionSession()
        self.assertIs(session.backend, np)
        with self.assertRaisesRegex(ValueError, "session backend must be NumPy or CuPy"):
            ContractionSession(backend=type("Impostor", (), {"__name__": "numpy"})())

    def test_joint_policy_is_rejected_before_backend_or_cache_initialization(self):
        with (
            patch("lqcdweave.contraction.contraction_session.ContractionPlanCache") as plan_cache,
            patch("lqcdweave.contraction.contraction_session.importlib.import_module") as import_backend,
        ):
            with self.assertRaisesRegex(ValueError, "has been retired"):
                ContractionSession(optimize="joint", backend="cupy")
        import_backend.assert_not_called()
        plan_cache.assert_not_called()

    def test_archive_is_loaded_and_validated_once_before_repeated_cfg_execution(self):
        bundle = self._bundle()
        counts = {"loads": 0, "validations": 0, "reads": []}

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "session.plan.json"
            save_plan(bundle, archive)

            def load_once(path):
                counts["loads"] += 1
                return archive_load_plan(path)

            def validate_once(value):
                counts["validations"] += 1
                return archive_validate_bundle(value)

            def loader(cfg, key, source):
                counts["reads"].append((cfg, key, source["family"]))
                return np.array([cfg, cfg + 1], dtype=np.float64)

            with patch(
                "lqcdweave.contraction.contraction_session.compile_plan_cached",
                side_effect=AssertionError("persisted execution must not search for a path"),
            ):
                session = ContractionSession.from_plan(
                    archive,
                    load_plan=load_once,
                    validate_bundle=validate_once,
                )
                first = session.execute_cfg(2, loader)
                second = session.execute_cfg(4, loader)

        self.assertEqual(counts["loads"], 1)
        self.assertEqual(counts["validations"], 1)
        self.assertEqual(
            counts["reads"],
            [(2, "input", "test-vector"), (4, "input", "test-vector")],
        )
        np.testing.assert_array_equal(first.values[0], np.array([2.0, 3.0]))
        np.testing.assert_array_equal(second.values[0], np.array([4.0, 5.0]))

    def test_saved_meson_recipes_execute_after_archive_reload_without_replanning(self):
        keys = ("left", "middle", "right")
        base_values = {
            "left": np.array([1 + 2j, 3 - 1j], dtype=np.complex128),
            "middle": np.array([2 - 3j, -1 + 4j], dtype=np.complex128),
            "right": np.array([5 + 1j, -2 - 2j], dtype=np.complex128),
        }
        increments = {
            "left": np.array([0.5 - 0.25j, 1 + 0.5j], dtype=np.complex128),
            "middle": np.array([-1 + 0.5j, 0.25 + 1j], dtype=np.complex128),
            "right": np.array([0.75 + 0.5j, -0.5 + 0.25j], dtype=np.complex128),
        }
        workload = WorkloadSpec(
            specs={key: TensorSpec((2,), np.complex128) for key in keys},
            terms=tuple(ContractionTerm((key,), "i->") for key in keys),
            input_sources={key: {"family": key} for key in keys},
            convention="meson-simple-v1",
            output_groups=((0, 1), (2,)),
            metadata={
                "adapter": "meson-simple-v1",
                "output_recipes": [[0, 1], [2]],
                "assembly_extra_bytes": 96,
            },
        )
        plan = compile_plan(workload.specs, workload.terms)
        bundle = PlanBundle(
            workload,
            (PlanBatch(plan, (0, 1, 2)),),
            ExecutionBudget(host_bytes=1 << 20, min_reserve_bytes=0),
        )
        buffer = np.empty(2, dtype=np.complex128)
        loader_calls = []

        def loader(cfg, key, source):
            loader_calls.append((cfg, key, source["family"]))
            buffer[:] = base_values[key] + cfg * increments[key]
            return buffer

        def expected(cfg):
            sums = {
                key: sum(base_values[key] + cfg * increments[key])
                for key in keys
            }
            return (sums["left"] * sums["middle"], sums["right"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "meson.plan.json"
            save_plan(bundle, archive)
            with patch(
                "lqcdweave.contraction.contraction_session.compile_plan_cached",
                side_effect=AssertionError("persisted execution must not replan"),
            ):
                session = ContractionSession.from_plan(
                    archive,
                    backend="numpy",
                    load_plan=archive_load_plan,
                    validate_bundle=archive_validate_bundle,
                )
                first = session.execute_cfg(1, loader)
                second = session.execute_cfg(3, loader)

        np.testing.assert_allclose(first.values, expected(1), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(second.values, expected(3), rtol=1e-13, atol=1e-13)
        self.assertIsInstance(first.values, tuple)
        self.assertEqual(
            loader_calls,
            [(cfg, key, key) for cfg in (1, 3) for key in keys],
        )

    def test_archive_validator_failure_propagates_to_the_caller(self):
        bundle = self._bundle()

        def reject(_bundle):
            raise ValueError("archive validation sentinel")

        with self.assertRaisesRegex(ValueError, "archive validation sentinel"):
            ContractionSession.from_plan(bundle, validate_bundle=reject)

    def test_absent_runtime_fingerprint_does_not_observe_environment(self):
        with patch("lqcdweave.runtime.environment.runtime_fingerprint") as observe:
            session = ContractionSession.from_plan(
                self._bundle(), validate_bundle=archive_validate_bundle,
            )
        observe.assert_not_called()
        self.assertIsNotNone(session.bundle)

    def test_matching_runtime_fingerprint_is_observed_once_before_preparation(self):
        fingerprint = {"backend": "numpy", "version": "test-version"}
        bundle = replace(self._bundle(), metadata={"performance_fingerprint": fingerprint})
        events = []
        prepared = object()

        def observe(backend):
            self.assertIs(backend, np)
            events.append("observe")
            return dict(fingerprint)

        def prepare(value, *, backend):
            self.assertIs(value, bundle)
            self.assertIs(backend, np)
            events.append("prepare")
            return prepared

        with (
            patch("lqcdweave.runtime.environment.runtime_fingerprint", side_effect=observe) as observer,
            patch("lqcdweave.contraction.execution.api.prepare_bundle_execution", side_effect=prepare),
        ):
            session = ContractionSession.from_plan(bundle, validate_bundle=archive_validate_bundle)
        observer.assert_called_once_with(np)
        self.assertEqual(events, ["observe", "prepare"])
        self.assertIs(session._prepared_bundle, prepared)

    def test_mismatched_and_present_empty_fingerprints_reject_before_preparation(self):
        for saved in ({"backend": "other"}, {}):
            with self.subTest(saved=saved):
                bundle = replace(self._bundle(), metadata={"performance_fingerprint": saved})
                with (
                    patch("lqcdweave.runtime.environment.runtime_fingerprint", return_value={"backend": "numpy"}) as observe,
                    patch("lqcdweave.contraction.execution.api.prepare_bundle_execution") as prepare,
                ):
                    with self.assertRaisesRegex(ValueError, "differs from this runtime"):
                        ContractionSession.from_plan(bundle, validate_bundle=archive_validate_bundle)
                observe.assert_called_once_with(np)
                prepare.assert_not_called()

    def test_runtime_fingerprint_failure_propagates_before_preparation(self):
        bundle = replace(self._bundle(), metadata={"performance_fingerprint": {}})
        failure = RuntimeError("environment observation sentinel")
        with (
            patch("lqcdweave.runtime.environment.runtime_fingerprint", side_effect=failure),
            patch("lqcdweave.contraction.execution.api.prepare_bundle_execution") as prepare,
        ):
            with self.assertRaises(RuntimeError) as caught:
                ContractionSession.from_plan(bundle, validate_bundle=archive_validate_bundle)
        self.assertIs(caught.exception, failure)
        prepare.assert_not_called()

    def test_unknown_frontend_is_rejected_before_session_execution(self):
        base = self._bundle()
        workload = WorkloadSpec(
            specs=base.workload.specs,
            terms=base.workload.terms,
            input_sources=base.workload.input_sources,
            convention="unknown-adapter-v1",
            metadata={"adapter": "unknown-adapter-v1"},
        )
        plan = compile_plan(workload.specs, workload.terms)
        bundle = PlanBundle(
            workload,
            (PlanBatch(plan, (0,)),),
            base.budget,
        )
        loader_calls = []

        with self.assertRaisesRegex(ValueError, "unsupported saved frontend"):
            session = ContractionSession.from_plan(
                bundle,
                validate_bundle=archive_validate_bundle,
            )
            session.execute_cfg(1, lambda *args: loader_calls.append(args))

        self.assertEqual(loader_calls, [])


if __name__ == "__main__":
    unittest.main()
