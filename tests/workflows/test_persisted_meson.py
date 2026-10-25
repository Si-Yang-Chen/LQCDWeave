from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import ExecutionBudget, TensorSpec, validate_frontend_workload
from lqcdweave.contracts.memory import BudgetExceededError, MemoryLimits
from lqcdweave.storage.contraction_plans import api as plan_archive
from lqcdweave.contraction.correlator import (
    execute_persisted_meson_cfg,
    open_persisted_meson_session,
    plan_persisted_meson,
    save_persisted_meson_plan,
)


VECTOR = TensorSpec((2,), np.complex128)
BASE = {
    "p0": np.array([1 + 2j, 3 - 1j], dtype=np.complex128),
    "p1": np.array([2 - 3j, -1 + 4j], dtype=np.complex128),
    "v0": np.array([5 + 1j, -2 - 2j], dtype=np.complex128),
    "v1": np.array([0.5 - 0.25j, 1 + 0.5j], dtype=np.complex128),
}
DELTA = {
    "p0": np.array([0.5 - 0.25j, 1 + 0.5j], dtype=np.complex128),
    "p1": np.array([0.75 + 0.5j, -0.5 + 0.25j], dtype=np.complex128),
    "v0": np.array([-1 + 0.5j, 0.25 + 1j], dtype=np.complex128),
    "v1": np.array([0.25 + 0.75j, -0.5 + 0.25j], dtype=np.complex128),
}


def make_diagram(groups, equations=None, propagator_types=None):
    return SimpleNamespace(
        operands=groups,
        subscripts=equations or ["i,i->"] * len(groups),
        propagator_types=propagator_types,
    )


def diagram_set():
    return (
        make_diagram(
            [([(0, 0, 1)], [0]), ([(0, 0, 1)], [1])],
            propagator_types=[["VSV"], ["VSV"]],
        ),
        make_diagram([([(1, 0, 1)], [0])], propagator_types=[["VSV"]]),
    )


def plan_arguments():
    return dict(
        diagrams=diagram_set(),
        times=(11, 17),
        vertex_specs={(0, 11): VECTOR, (1, 17): VECTOR},
        propagator_specs={(0, 11, 17): VECTOR, (1, 11, 17): VECTOR},
        vertex_sources={0: {"dataset_declaration": "v0"}, 1: {"dataset_declaration": "v1"}},
        propagator_sources={
            0: {"dataset_declaration": "p0"},
            1: {"dataset_declaration": "p1"},
        },
    )


class SharedBufferGetter:
    def __init__(self, label, cfg, shared_buffer, calls):
        self.label = label
        self.cfg = cfg
        self.shared_buffer = shared_buffer
        self.calls = calls

    def get(self, *times):
        self.calls.append((self.cfg, self.label, tuple(times)))
        self.shared_buffer[:] = BASE[self.label] + self.cfg * DELTA[self.label]
        return self.shared_buffer


def getters_for_cfg(cfg, shared_buffer, calls):
    return (
        [SharedBufferGetter("v0", cfg, shared_buffer, calls),
         SharedBufferGetter("v1", cfg, shared_buffer, calls)],
        [SharedBufferGetter("p0", cfg, shared_buffer, calls),
         SharedBufferGetter("p1", cfg, shared_buffer, calls)],
    )


def expected_values(cfg):
    def dot(left, right):
        return np.einsum("i,i->", left, right)

    values = {name: value + cfg * DELTA[name] for name, value in BASE.items()}
    first = dot(values["p0"], values["v0"]) * dot(values["p0"], values["v1"])
    second = dot(values["p1"], values["v0"])
    return (first, second)


class PersistedMesonWorkflowTests(unittest.TestCase):
    def test_archive_reopens_once_and_executes_two_cfgs_without_replanning(self):
        budget = ExecutionBudget(host_bytes=1 << 20, min_reserve_bytes=0)
        calls = []
        shared_buffer = np.empty(2, dtype=np.complex128)
        first_vertices, first_props = getters_for_cfg(2, shared_buffer, calls)
        bundle = plan_persisted_meson(**plan_arguments(), budget=budget, max_candidates=4)

        self.assertEqual(calls, [])  # E/P planning receives only structural declarations.
        self.assertEqual(bundle.budget, budget)
        self.assertEqual(bundle.metadata["optimize"], "greedy")
        self.assertEqual(bundle.workload.output_groups, ((0, 1), (2,)))
        self.assertEqual(bundle.workload.metadata["output_recipes"], ((0, 1), (2,)))
        self.assertEqual(bundle.workload.metadata["assembly_extra_bytes"], 96)

        with TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "persisted-meson.plan.json"
            self.assertEqual(save_persisted_meson_plan(bundle, archive), archive)
            with (
                patch(
                    "lqcdweave.contraction.correlator.plan_workload",
                    side_effect=AssertionError("reopened execution must not search schedules"),
                ) as scheduler_search,
                patch(
                    "lqcdweave.contraction.contraction_session.compile_plan_cached",
                    side_effect=AssertionError("reopened execution must not compile plans"),
                ) as dynamic_compile,
                patch(
                    "lqcdweave.contraction.correlator._load_plan",
                    wraps=plan_archive.load_plan,
                ) as load_spy,
                patch(
                    "lqcdweave.storage.contraction_plans.api._validate_bundle",
                    wraps=plan_archive._validate_bundle,
                ) as validate_spy,
                patch("lqcdweave.contraction.correlator.validate_frontend_workload",
                      wraps=validate_frontend_workload) as workflow_frontend,
                patch("lqcdweave.contraction.contraction_session.validate_frontend_workload",
                      wraps=validate_frontend_workload) as session_frontend,
            ):
                session = open_persisted_meson_session(
                    archive, backend="numpy", expected_workload=bundle.workload,
                )
                first = execute_persisted_meson_cfg(
                    session, 2, vertices=first_vertices, propagators=first_props,
                )
                second_vertices, second_props = getters_for_cfg(5, shared_buffer, calls)
                second = execute_persisted_meson_cfg(
                    session, 5, vertices=second_vertices, propagators=second_props,
                )

            self.assertEqual(load_spy.call_count, 1)
            self.assertEqual(validate_spy.call_count, 1)
            self.assertEqual(workflow_frontend.call_count + session_frontend.call_count, 1)
            self.assertEqual(scheduler_search.call_count, 0)
            self.assertEqual(dynamic_compile.call_count, 0)
            self.assertEqual(session.bundle.budget, budget)
            self.assertEqual(session.bundle.workload.output_groups, ((0, 1), (2,)))
            self.assertEqual(session.bundle.workload.metadata["output_recipes"], ((0, 1), (2,)))
            self.assertEqual(session.bundle.workload.metadata["assembly_extra_bytes"], 96)
            declared = session.bundle.workload.input_sources["propagator:0:11:17"]
            self.assertEqual(declared["source"], {"dataset_declaration": "p0"})
            self.assertNotIn("actual_id_verified", declared)

        np.testing.assert_allclose(first.values, expected_values(2), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(second.values, expected_values(5), rtol=1e-13, atol=1e-13)
        # The first cfg's returned scalars remain stable after the same buffer serves cfg 5.
        np.testing.assert_allclose(first.values, expected_values(2), rtol=1e-13, atol=1e-13)
        self.assertEqual(len(calls), 8)
        self.assertEqual(
            {(cfg, label) for cfg, label, _times in calls},
            {(cfg, label) for cfg in (2, 5) for label in ("p0", "p1", "v0", "v1")},
        )
        self.assertEqual(
            {(label, times) for _cfg, label, times in calls},
            {("p0", (11, 17)), ("p1", (11, 17)), ("v0", (11,)), ("v1", (17,))},
        )

    def test_save_validates_the_full_bundle_only_once(self):
        bundle = plan_persisted_meson(**plan_arguments())
        with TemporaryDirectory() as directory:
            archive = Path(directory) / 'meson.json'
            with patch.object(plan_archive, '_validate_bundle',
                              wraps=plan_archive._validate_bundle) as validate:
                save_persisted_meson_plan(bundle, archive)
            self.assertEqual(validate.call_count, 1)
            self.assertIs(validate.call_args.args[0], bundle)

    def test_changed_python_bundle_and_corrupted_archive_fail_before_preparation(self):
        import json
        from lqcdweave.contraction.contraction_session import ContractionSession
        bundle = plan_persisted_meson(**plan_arguments())
        batch = bundle.batches[0]
        invalid = replace(bundle, batches=(replace(batch,
            plan=replace(batch.plan, optimize='')), *bundle.batches[1:]))
        with TemporaryDirectory() as directory:
            archive = Path(directory) / 'meson.json'
            with self.assertRaisesRegex(ValueError, 'optimize'):
                save_persisted_meson_plan(invalid, archive)
            self.assertFalse(archive.exists())
            with patch('lqcdweave.contraction.execution.api.prepare_bundle_execution') as prepare:
                with self.assertRaisesRegex(ValueError, 'optimize'):
                    ContractionSession.from_plan(invalid, validate_bundle=plan_archive.validate_bundle)
            prepare.assert_not_called()
            save_persisted_meson_plan(bundle, archive)
            open_persisted_meson_session(archive)
            document = json.loads(archive.read_text())
            document['payload']['signature'] = 'corrupted'
            archive.write_text(json.dumps(document))
            with patch('lqcdweave.contraction.execution.api.prepare_bundle_execution') as prepare:
                with self.assertRaisesRegex(ValueError, 'signature mismatch'):
                    open_persisted_meson_session(archive)
            prepare.assert_not_called()

    def test_saved_insufficient_memory_is_rejected_before_getters(self):
        bundle = plan_persisted_meson(**plan_arguments())
        invalid = replace(bundle, budget=ExecutionBudget(host_bytes=1, min_reserve_bytes=0))
        calls = []
        with TemporaryDirectory() as directory:
            archive = Path(directory) / 'too-small.json'
            save_persisted_meson_plan(invalid, archive)
            session = open_persisted_meson_session(archive)
            vertices, propagators = getters_for_cfg(1, np.empty(2, dtype=np.complex128), calls)
            with patch('lqcdweave.contraction.execution.scheduled._fresh_runtime_limits',
                       return_value=MemoryLimits(host_bytes=1, device_bytes=None)):
                with self.assertRaises(BudgetExceededError):
                    execute_persisted_meson_cfg(session, 1, vertices=vertices, propagators=propagators)
        self.assertEqual(calls, [])

    def test_private_handoffs_reject_a_different_bundle_identity(self):
        from lqcdweave.contraction.contraction_session import ContractionSession
        bundle = plan_persisted_meson(**plan_arguments())
        other = replace(bundle)
        with patch('lqcdweave.contraction.execution.api.prepare_bundle_execution') as prepare:
            with self.assertRaisesRegex(ValueError, 'does not match'):
                ContractionSession._from_validated_plan(other, validated_bundle=bundle)
        prepare.assert_not_called()

    def test_saved_recipe_group_mismatch_fails_before_any_getter(self):
        bundle = plan_persisted_meson(**plan_arguments(), budget=ExecutionBudget(
            host_bytes=1 << 20, min_reserve_bytes=0,
        ))
        mismatched_workload = replace(bundle.workload, output_groups=((0,), (1, 2)))
        mismatched_bundle = replace(bundle, workload=mismatched_workload)
        calls = []

        with TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "forged-groups.plan.json"
            # A can store this structurally valid generic plan; W must enforce the Meson recipe contract.
            plan_archive.save_plan(mismatched_bundle, archive)
            with self.assertRaisesRegex(ValueError, "output_groups must match"):
                session = open_persisted_meson_session(archive, backend="numpy")
                vertices, propagators = getters_for_cfg(1, np.empty(2, dtype=np.complex128), calls)
                execute_persisted_meson_cfg(
                    session, 1, vertices=vertices, propagators=propagators,
                )
        self.assertEqual(calls, [])

    def test_tight_budget_propagates_the_existing_scheduler_error(self):
        with self.assertRaises(BudgetExceededError):
            plan_persisted_meson(
                **plan_arguments(),
                budget=ExecutionBudget(host_bytes=1, min_reserve_bytes=0),
            )

    def test_metadata_only_planning_does_not_observe_runtime_memory(self):
        budget = ExecutionBudget(host_bytes=1 << 20, min_reserve_bytes=0)
        with patch(
            "lqcdweave.contraction.correlator._resolve_workload_limits",
            side_effect=AssertionError("metadata-only planning must not probe resources"),
        ):
            bundle = plan_persisted_meson(**plan_arguments(), budget=budget)
        self.assertEqual(bundle.budget, budget)

    def test_explicit_backend_startup_limits_feed_the_pure_plan(self):
        budget = ExecutionBudget(host_bytes=1 << 20, device_bytes=1 << 20,
                                 min_reserve_bytes=0)
        limits = MemoryLimits(host_bytes=1 << 18, device_bytes=1 << 19)
        for backend, target in ((np, "cpu"), ("NumPy", "cpu"), ("CuPy", "gpu")):
            with self.subTest(backend=backend):
                from lqcdweave.contraction.planning.api import plan_workload
                with (
                    patch("lqcdweave.contraction.correlator._resolve_workload_limits",
                          return_value=limits) as observe,
                    patch("lqcdweave.contraction.correlator.plan_workload",
                          wraps=plan_workload) as pure_plan,
                ):
                    bundle = plan_persisted_meson(
                        **plan_arguments(), budget=budget, backend=backend,
                    )
                observe.assert_called_once_with(budget, backend)
                self.assertEqual(pure_plan.call_args.kwargs["target"], target)
                self.assertEqual(bundle.budget.host_bytes, limits.host_bytes)
                self.assertEqual(bundle.budget.device_bytes, limits.device_bytes)
                self.assertEqual(bundle.budget.min_reserve_bytes, budget.min_reserve_bytes)

    def test_invalid_request_is_rejected_before_startup_observation(self):
        for candidates, error in ((True, TypeError), (1.5, TypeError), (0, ValueError), (5, ValueError)):
            with self.subTest(candidates=candidates):
                with patch("lqcdweave.contraction.correlator._resolve_workload_limits") as observe:
                    with self.assertRaises(error):
                        plan_persisted_meson(**plan_arguments(), max_candidates=candidates,
                                             backend="numpy")
                observe.assert_not_called()
        with patch("lqcdweave.contraction.correlator._resolve_workload_limits") as observe:
            with self.assertRaisesRegex(TypeError, "budget must be an ExecutionBudget"):
                plan_persisted_meson(**plan_arguments(), budget={}, backend="numpy")
        observe.assert_not_called()

    def test_exhausted_startup_limits_reject_before_path_planning(self):
        with (
            patch("lqcdweave.contraction.correlator._resolve_workload_limits",
                  return_value=MemoryLimits(host_bytes=0, device_bytes=None)),
            patch("lqcdweave.contraction.correlator.plan_workload") as pure_plan,
        ):
            with self.assertRaisesRegex(BudgetExceededError, "No memory remains"):
                plan_persisted_meson(**plan_arguments(), backend="numpy")
        pure_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
