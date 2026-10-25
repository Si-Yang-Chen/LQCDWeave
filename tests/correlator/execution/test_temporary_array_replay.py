"""Real CPU scratch spill/restore through the public fixed-plan executor."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import gc
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np

from lqcdweave.adapters import NpyBlockBinding, NpySceneBlockLoader
from lqcdweave.contracts.correlator.api import (
    Action,
    ContractionProblem,
    ContractionTerm,
    ResourceTransition,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorSpec,
    validate_executable_resources,
)
from lqcdweave.contraction.execution.api import replay_scene_storage, replay_storage_batches
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState,
    audit_resource_trajectory,
    logical_cost_profile,
    logical_source_problem,
)
from lqcdweave.storage.contraction_plans.api import (
    load_storage_schedule,
    save_storage_schedule,
)
from lqcdweave.storage.dataset import TemporaryNpyArrayStore


class _RecordingScope:
    def __init__(self, inner, store, loader):
        self.inner = inner
        self.store = store
        self.loader = loader
        self.root = inner._root
        self.closed = False
        self.close_calls = 0

    def _live_file_bytes(self):
        return sum(path.stat().st_size for path in self.root.glob("*.npy"))

    def write_array(self, value):
        borrowed = any(np.shares_memory(value, raw) for raw in self.loader.raw_blocks.values())
        self.store.source_refs.append(weakref.ref(value))
        receipt = self.inner.write_array(value)
        file_path = self.root / receipt.filename
        actual_bytes = file_path.stat().st_size
        self.store.writes.append({
            "array_nbytes": int(value.nbytes),
            "actual_file_bytes": actual_bytes,
            "declared_file_bytes": self.store.storage_bytes(value.shape, value.dtype.str),
            "input_shared_with_raw": borrowed,
            "live_file_bytes": self._live_file_bytes(),
            "receipt": receipt,
        })
        return receipt

    def read_array(self, receipt, *, shape, dtype):
        gc.collect()
        self.store.source_released_before_read.append(
            all(reference() is None for reference in self.store.source_refs)
        )
        self.store.raw_owner_active_at_read.append(bool(self.loader.raw_blocks))
        if self.store.fail_restore:
            raise OSError("injected scratch restore read failure")
        if self.store.corrupt_before_read:
            file_path = self.root / receipt.filename
            value = bytearray(file_path.read_bytes())
            value[-1] ^= 1
            file_path.write_bytes(value)
        if self.store.wrong_shape_before_read:
            np.save(self.root / receipt.filename, np.zeros((5,), dtype=np.float64),
                    allow_pickle=False)
        return self.inner.read_array(receipt, shape=shape, dtype=dtype)

    def remove_array(self, receipt):
        self.inner.remove_array(receipt)

    def close(self):
        self.close_calls += 1
        self.inner.close()
        self.closed = True
        self.store.scope_roots_after_close.append(self.root.exists())
        if self.store.fail_scope_close:
            raise RuntimeError("injected scratch scope close failure")


class _RecordingStore(TemporaryNpyArrayStore):
    def __init__(self, directory):
        super().__init__(directory)
        self.loader = None
        self.scopes = []
        self.scope_roots_after_close = []
        self.writes = []
        self.corrupt_before_read = False
        self.wrong_shape_before_read = False
        self.fail_restore = False
        self.fail_scope_close = False
        self.source_refs = []
        self.source_released_before_read = []
        self.raw_owner_active_at_read = []

    def open_scope(self):
        scope = _RecordingScope(super().open_scope(), self, self.loader)
        self.scopes.append(scope)
        return scope


class _TrackingLoader(NpySceneBlockLoader):
    def __init__(self, bindings, materializers, *, temporary_array_store=None):
        super().__init__(bindings, materializers, backend=np,
                         temporary_array_store=temporary_array_store)
        self.raw_blocks = {}
        self.reads = []
        self.closed_blocks = []
        self.close_all_calls = 0
        self.fail_close_all = False
        if temporary_array_store is not None:
            temporary_array_store.loader = self

    def read_block(self, key, block):
        self.reads.append(key)
        value = super().read_block(key, block)
        self.raw_blocks[key] = value
        return value

    def close_block(self, key, value):
        self.closed_blocks.append(key)
        self.raw_blocks.pop(key, None)

    def close_all(self):
        self.close_all_calls += 1
        if self.fail_close_all:
            raise RuntimeError("injected loader close_all failure")


def _profile():
    return CostProfile(
        preparation_call_seconds=7.0,
        preparation_kernel_call_seconds=5.0,
        preparation_flop_seconds=0.5,
        preparation_traffic_seconds_per_byte=0.25,
        kernel_call_seconds_float64=0.75,
        flop_seconds=0.0,
        traffic_seconds_per_byte=0.0,
    )


def _plan_and_files(directory, *, two_inputs=False, disk_limit_bytes=None, profile=None):
    profile = _profile() if profile is None else profile
    values = {"A": np.array([1.0, 2.0, 3.0, 4.0])}
    terms = (ContractionTerm(("A",), "i->"),)
    if two_inputs:
        values["B"] = np.array([4.0, 3.0, 2.0, 1.0])
        terms = (ContractionTerm(("A", "B"), "i,i->"),)
    problem = logical_source_problem(ContractionProblem(
        {key: TensorSpec(value.shape, value.dtype) for key, value in values.items()},
        terms,
        {key: SourceSpec(key, 0) for key in values},
    ))
    model = logical_cost_profile(profile)
    state = ContractionState.initial(problem)
    for key in values:
        state = state.load(key, model, 1_000_000)
    state = state.contract(0, tuple(range(len(values))), model, 1_000_000)

    blocks = {}
    preparations = {}
    bindings = {}
    materializers = {}
    load_actions = []
    for key, value in values.items():
        block_key = "raw_" + key.lower()
        file_path = Path(directory) / f"{key}.npy"
        np.save(file_path, value, allow_pickle=False)
        blocks[block_key] = StorageBlock(
            dataset="fixture:" + key,
            selection="all",
            read_bytes=value.nbytes,
            host_bytes=value.nbytes,
        )
        preparations[key] = SourcePreparation(
            (block_key,),
            workspace_host_bytes=16,
            traffic_bytes=12,
            kernel_calls=1,
            dtype="float64",
        )
        bindings[block_key] = NpyBlockBinding(
            file_path, (slice(None),), tuple(value.shape), value.dtype.str,
        )
        materializers[key] = lambda metadata, arrays, block=block_key: arrays[block]
        load_actions.append(Action("read_block", block_key))
        load_actions.append(next(action for action in state.actions
                                 if action.kind == "load" and action.tensor == key))

    spill_actions = tuple(
        Action("spill_disk", key,
               resource=ResourceTransition("tensor", "host", "disk", values[key].nbytes))
        for key in values
    )
    restore_actions = tuple(
        Action("restore_host", key,
               resource=ResourceTransition("tensor", "disk", "host", values[key].nbytes))
        for key in values
    )
    compute_actions = tuple(action for action in state.actions if action.kind != "load")
    close_actions = tuple(Action("close_block", key) for key in blocks)
    actions = tuple(load_actions) + spill_actions + restore_actions + compute_actions + close_actions
    state = replace(state, actions=actions, seconds=sum(action.seconds for action in actions))
    plan = audit_resource_trajectory(
        state,
        blocks,
        preparations,
        profile=profile,
        memory_limit=1_000_000,
        disk_limit_bytes=disk_limit_bytes,
    )
    return plan, problem, blocks, preparations, bindings, materializers, values, profile


def _loader(plan_data, store=None):
    _, _, _, _, bindings, materializers, _, _ = plan_data
    return _TrackingLoader(bindings, materializers, temporary_array_store=store)


class TemporaryArrayReplayTests(unittest.TestCase):
    def test_npy_store_uses_file_object_array_io_without_iter_chunk_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TemporaryNpyArrayStore(directory)
            scope = store.open_scope()
            values = np.arange(16, dtype=np.float64)
            try:
                with mock.patch("numpy.nditer", side_effect=AssertionError("chunked copy path")):
                    receipt = scope.write_array(values)
                with mock.patch("numpy.load", side_effect=AssertionError("loader staging")), \
                        mock.patch("numpy.fromfile", side_effect=AssertionError("flat staging")):
                    restored = scope.read_array(receipt, shape=values.shape, dtype=values.dtype.str)
                np.testing.assert_array_equal(restored, values)
                self.assertTrue(restored.flags.owndata)
                scope.remove_array(receipt)
            finally:
                scope.close()

    def test_single_cpu_path_uses_real_npy_io_and_reports_file_byte_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _RecordingStore(directory)
            data = _plan_and_files(directory, disk_limit_bytes=store.storage_bytes((4,), "<f8"))
            plan, _, blocks, preparations, _, _, values, profile = data
            loader = _loader(data, store)

            report = replay_scene_storage(plan, loader, backend=np, measure=True)

            self.assertAlmostEqual(report.values[0], 10.0)
            self.assertEqual(loader.reads, ["raw_a"])
            self.assertEqual(loader.closed_blocks, ["raw_a"])
            self.assertEqual(loader.close_all_calls, 1)
            self.assertEqual(len(store.writes), 1)
            write = store.writes[0]
            self.assertEqual(write["array_nbytes"], values["A"].nbytes)
            self.assertFalse(write["input_shared_with_raw"])
            self.assertLessEqual(write["actual_file_bytes"], write["declared_file_bytes"])
            self.assertEqual(write["live_file_bytes"], write["actual_file_bytes"])
            self.assertEqual(store.source_released_before_read, [True])
            self.assertEqual(store.raw_owner_active_at_read, [True])
            self.assertTrue(store.scopes[0].closed)
            self.assertEqual(store.scope_roots_after_close, [False])

            payload_disk_peak = max(point.temporary_disk_bytes for point in plan.memory)
            self.assertEqual(payload_disk_peak, values["A"].nbytes)
            self.assertEqual(report.execution_profile["temporary_file_peak_bytes"],
                             write["declared_file_bytes"])
            self.assertGreater(report.execution_profile["temporary_file_peak_bytes"], payload_disk_peak)

            expected_preparation = sum(prep.seconds(profile) for prep in preparations.values())
            self.assertAlmostEqual(plan.seconds, plan.state.seconds + expected_preparation)
            self.assertAlmostEqual(
                sum(timing.estimated_seconds for timing in report.timings),
                plan.seconds,
            )
            estimated_by_kind = {timing.kind: timing.estimated_seconds for timing in report.timings}
            observed_by_kind = {timing.kind: timing.observed_seconds for timing in report.timings}
            for kind in ("read_block", "spill_disk", "restore_host"):
                self.assertEqual(estimated_by_kind[kind], 0)
                self.assertGreater(observed_by_kind[kind], 0)
            self.assertAlmostEqual(
                estimated_by_kind["load"],
                next(a.seconds for a in plan.state.actions if a.kind == "load")
                + expected_preparation,
            )
            self.assertEqual(plan.memory[2].logical_host_bytes, 32)
            self.assertEqual(plan.memory[2].raw_host_bytes, 32)
            self.assertEqual(plan.memory[2].temporary_disk_bytes, 32)
            self.assertEqual(plan.memory[3].temporary_disk_bytes, 32)
            self.assertEqual(plan.memory[3].logical_host_bytes, 32)
            self.assertEqual(plan.memory[3].host_bytes, 64)
            self.assertEqual(plan.memory[4].temporary_disk_bytes, 0)

    def test_restore_primary_error_wins_and_scope_and_loader_close_once(self):
        with tempfile.TemporaryDirectory() as directory:
            bound = TemporaryNpyArrayStore(directory).storage_bytes((4,), "<f8")
            data = _plan_and_files(directory, disk_limit_bytes=bound)
            plan = data[0]
            store = _RecordingStore(directory)
            store.fail_restore = True
            store.fail_scope_close = True
            loader = _loader(data, store)
            loader.fail_close_all = True

            with self.assertRaisesRegex(OSError, "scratch restore read failure"):
                replay_scene_storage(plan, loader, backend=np)

            self.assertEqual(loader.reads, ["raw_a"])
            self.assertEqual(loader.closed_blocks, ["raw_a"])
            self.assertEqual(loader.close_all_calls, 1)
            self.assertEqual(len(store.scopes), 1)
            self.assertEqual(store.scopes[0].close_calls, 1)
            self.assertTrue(store.scopes[0].closed)
            self.assertEqual(store.scope_roots_after_close, [False])

    def test_spill_rejects_borrowed_noncontiguous_and_aliased_arrays_without_copy(self):
        from lqcdweave.contraction.execution.storage_runtime import _BlockExecution

        with tempfile.TemporaryDirectory() as directory:
            bound = TemporaryNpyArrayStore(directory).storage_bytes((4,), "<f8")
            data = _plan_and_files(directory, disk_limit_bytes=bound)
            plan = data[0]
            spill = next(action for action in plan.state.actions if action.kind == "spill_disk")
            cases = (
                ("borrowed", np.arange(8, dtype=np.float64)[:4], "owned contiguous"),
                ("noncontiguous", np.arange(8, dtype=np.float64)[::2], "owned contiguous"),
            )
            for name, value, message in cases:
                with self.subTest(case=name):
                    store = _RecordingStore(directory)
                    loader = _TrackingLoader({}, {}, temporary_array_store=store)
                    scope = store.open_scope()
                    execution = _BlockExecution(plan, loader, np, temporary_scope=scope)
                    try:
                        with self.assertRaisesRegex(ValueError, message):
                            execution.move_tensor(spill, {"A": value},
                                                  lambda _name, function, **_kw: function())
                    finally:
                        execution.close()
                    self.assertEqual(store.writes, [])
                    self.assertEqual(store.scopes[0].close_calls, 1)
                    self.assertEqual(store.scope_roots_after_close, [False])

            with self.subTest(case="live-alias"):
                store = _RecordingStore(directory)
                loader = _TrackingLoader({}, {}, temporary_array_store=store)
                scope = store.open_scope()
                execution = _BlockExecution(plan, loader, np, temporary_scope=scope)
                owned = np.arange(4, dtype=np.float64)
                try:
                    with self.assertRaisesRegex(ValueError, "live aliases"):
                        execution.move_tensor(spill, {"A": owned, "alias": owned.view()},
                                              lambda _name, function, **_kw: function())
                finally:
                    execution.close()
                self.assertEqual(store.writes, [])
                self.assertEqual(store.scopes[0].close_calls, 1)
                self.assertEqual(store.scope_roots_after_close, [False])

    def test_two_live_files_include_headers_in_pre_io_disk_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = TemporaryNpyArrayStore(directory)
            bound = probe.storage_bytes((4,), "<f8")
            data = _plan_and_files(directory, two_inputs=True, disk_limit_bytes=2 * bound)
            plan, _, _, _, _, _, _, _ = data
            store = _RecordingStore(directory)
            loader = _loader(data, store)

            report = replay_scene_storage(plan, loader, backend=np, measure=True)

            self.assertAlmostEqual(report.values[0], 20.0)
            self.assertEqual(len(store.writes), 2)
            self.assertEqual([entry["actual_file_bytes"] for entry in store.writes], [bound, bound])
            self.assertEqual([entry["live_file_bytes"] for entry in store.writes], [bound, 2 * bound])
            self.assertEqual(store.source_released_before_read, [True, True])
            self.assertEqual(store.raw_owner_active_at_read, [True, True])
            self.assertEqual(report.execution_profile["temporary_file_peak_bytes"], 2 * bound)
            self.assertEqual(max(point.temporary_disk_bytes for point in plan.memory), 64)
            self.assertEqual(store.scope_roots_after_close, [False])

            too_small = replace(plan, disk_limit_bytes=2 * bound - 1)
            limited_store = _RecordingStore(directory)
            limited_loader = _loader(data, limited_store)
            with self.assertRaisesRegex(MemoryError, "scratch-file byte cap"):
                replay_scene_storage(too_small, limited_loader, backend=np)
            self.assertEqual(limited_loader.reads, [])
            self.assertEqual(limited_store.scopes, [])
            self.assertEqual(limited_loader.close_all_calls, 0)

    def test_explicit_capability_backend_batch_and_default_gates_precede_io(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _RecordingStore(directory)
            bound = store.storage_bytes((4,), "<f8")
            data = _plan_and_files(directory, disk_limit_bytes=bound)
            plan = data[0]

            no_store = _loader(data)
            with self.assertRaises(NotImplementedError):
                replay_scene_storage(plan, no_store)
            self.assertEqual(no_store.reads, [])

            loader = _loader(data, store)
            fake_cupy = type("FakeCuPy", (), {"__name__": "cupy"})
            with self.assertRaises(NotImplementedError):
                replay_scene_storage(plan, loader, backend=fake_cupy)
            self.assertEqual(loader.reads, [])
            self.assertEqual(store.scopes, [])

            with self.assertRaises(NotImplementedError):
                validate_executable_resources(plan)
            with self.assertRaises(NotImplementedError):
                replay_storage_batches((plan,), loader, lambda *_: None, prejoined=True)
            self.assertEqual(loader.reads, [])

            uncapped = replace(plan, disk_limit_bytes=None)
            with self.assertRaises(NotImplementedError):
                replay_scene_storage(uncapped, loader)
            self.assertEqual(loader.reads, [])

            reserve_limited = replace(
                plan,
                memory_limit=plan.peak_host_bytes + store.host_reserve_bytes - 1,
                feasible=True,
            )
            with self.assertRaisesRegex(MemoryError, "memory or disk cap"):
                replay_scene_storage(reserve_limited, loader)
            self.assertEqual(loader.reads, [])
            self.assertEqual(store.scopes, [])

    def test_positive_cap_resource_archive_roundtrips_and_executes_with_injected_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _RecordingStore(directory)
            bound = store.storage_bytes((4,), "<f8")
            data = _plan_and_files(directory, disk_limit_bytes=bound)
            plan, problem, blocks, preparations, _, _, _, profile = data
            archive = Path(directory) / "schedule.json"
            save_storage_schedule(
                archive, (plan,), shapes={"A": [4]}, hardware={"target": "cpu"},
            )
            payload = archive.read_text(encoding="utf-8")
            self.assertNotIn("temporary_array_store", payload)
            self.assertNotIn(directory, payload)

            restored, _ = load_storage_schedule(
                archive,
                shapes={"A": [4]},
                hardware={"target": "cpu"},
                expected=problem,
                blocks=blocks,
                preparations=preparations,
                profile=profile,
            )
            self.assertEqual(restored[0].state.actions, plan.state.actions)
            loader = _loader(data, store)
            report = replay_scene_storage(restored[0], loader, measure=True)
            self.assertAlmostEqual(report.values[0], 10.0)
            self.assertEqual(report.execution_profile["temporary_file_peak_bytes"], bound)

    def test_partial_write_shape_and_kernel_failures_clean_scope_and_raw_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            bound = TemporaryNpyArrayStore(directory).storage_bytes((4,), "<f8")
            data = _plan_and_files(directory, disk_limit_bytes=bound)
            plan = data[0]

            for failure in ("partial_write", "wrong_shape", "kernel"):
                with self.subTest(failure=failure):
                    store = _RecordingStore(directory)
                    store.wrong_shape_before_read = failure == "wrong_shape"
                    loader = _loader(data, store)
                    if failure == "partial_write":
                        def fail_after_partial(stream, array, *, allow_pickle=True):
                            stream.write(b"partial")
                            raise OSError("injected NPY write failure")

                        patcher = mock.patch("lqcdweave.storage.dataset.np.save",
                                             side_effect=fail_after_partial)
                    elif failure == "kernel":
                        patcher = mock.patch(
                            "lqcdweave.contraction.execution.scene_runtime.contract_dense",
                            side_effect=RuntimeError("injected contraction failure"),
                        )
                    else:
                        patcher = mock.patch.object(store, "wrong_shape_before_read", True)
                    expected_error = {"partial_write": OSError, "wrong_shape": ValueError,
                                      "kernel": RuntimeError}[failure]
                    with patcher, self.assertRaises(expected_error):
                        replay_scene_storage(plan, loader, backend=np)
                    self.assertEqual(loader.reads, ["raw_a"])
                    self.assertEqual(loader.closed_blocks, ["raw_a"])
                    self.assertEqual(loader.close_all_calls, 1)
                    self.assertEqual(len(store.scopes), 1)
                    self.assertTrue(store.scopes[0].closed)
                    self.assertEqual(store.scope_roots_after_close, [False])


if __name__ == "__main__":
    unittest.main()
