"""P/W freeze storage resources; X consumes and independently checks the ABI."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import test_sparse_point_sink_batch as sparse_fixture
from test_scene_storage_batches import batch, Loader
from lqcdweave.contracts.correlator.api import (
    SceneStoragePlan, SourceReadRequest, TensorAxisMetadata, TensorPreparation,
    describe_coordinates,
)
from lqcdweave.contraction.planning.api import (
    prepare_coordinate_batch_workspace, coordinate_batch_sparse_workspace_bytes,
    build_cfg_plan, audit_cfg_plan, join_storage_batches, solve_storage, plan_scene_storage,
)
from lqcdweave.contraction.execution.api import (
    replay_scene_storage, replay_storage_batches, iter_execute_cfg_plan,
)
from lqcdweave.contraction.execution.domain import BoundContractionExecution
from lqcdweave.contraction.execution.scene_runtime import _coordinate_batch_sparse_workspace_bytes
from lqcdweave.contraction.execution.storage_runtime import validate_coordinate_batch_workspace
from lqcdweave.storage.contraction_plans.api import (
    save_cfg_plan, load_cfg_plan, save_storage_schedule, load_storage_schedule,
)
from lqcdweave.contraction import replay as convenience


def legacy_fixture(*, same_point=False):
    # Reproduce the descriptor delivered before B4 without invoking X preparation.
    with patch('lqcdweave.contraction.planning.storage.prepare_coordinate_batch_workspace',
               side_effect=lambda plan: (plan, 0)):
        return sparse_fixture.SparsePointSinkBatchCatalogTests._four_part_path(1 << 28, same_point=same_point)


def described_cfg_inputs(plan):
    # Metadata-only recipes make the existing synthetic source bounds archivable.
    # This fixture checks resource delivery/roundtrip, not a new physical reader.
    preparations = {key: replace(prep, recipe=TensorPreparation(
        'identity', inputs=(prep.blocks[0],))) for key, prep in plan.preparations.items()}
    requests = {key: SourceReadRequest(key, 'elemental', (), (), TensorAxisMetadata(
        ('byte',), (block.host_bytes,), 'uint8', block.dataset, block.selection))
        for key, block in plan.blocks.items()}
    return replace(plan, preparations=preparations), preparations, requests


class ReplayPreparationBoundaries(unittest.TestCase):
    def test_p_matches_legacy_all_point_envelope_and_preserves_opaque_reserves(self):
        for same_point in (False, True):
            with self.subTest(same_point=same_point):
                _provider, _expanded, path = legacy_fixture(same_point=same_point)
                legacy = path.plan.with_reserves(reserved_host_bytes=123)
                corrections = []
                for index, action in enumerate(legacy.state.actions):
                    if action.kind != 'sparse_bridge' or action.reused:
                        continue
                    expected = _coordinate_batch_sparse_workspace_bytes(legacy.state, action)
                    self.assertEqual(coordinate_batch_sparse_workspace_bytes(legacy.state, action), expected)
                    point = next(point for point in legacy.memory
                                 if point.action_index == index and point.phase == 'sparse_bridge')
                    corrections.append(max(0, expected - point.workspace_host_bytes))
                correction = max(corrections)
                self.assertGreater(correction, 0)
                old_execution_envelope = legacy.with_reserves(
                    reserved_host_bytes=legacy.reserved_host_bytes + correction)
                prepared, added = prepare_coordinate_batch_workspace(legacy)
                self.assertEqual(added, correction)
                self.assertEqual(prepared.reserved_host_bytes, 123)
                self.assertEqual(prepared.peak_host_bytes, old_execution_envelope.peak_host_bytes)
                self.assertEqual(prepared.peak_device_bytes, old_execution_envelope.peak_device_bytes)
                self.assertEqual([p.host_bytes for p in prepared.memory],
                                 [p.host_bytes for p in old_execution_envelope.memory])
                self.assertEqual([p.device_bytes for p in prepared.memory],
                                 [p.device_bytes for p in old_execution_envelope.memory])
                self.assertIs(prepared.state, legacy.state)
                self.assertIs(prepared.events, legacy.events)
                self.assertEqual(prepare_coordinate_batch_workspace(prepared), (prepared, 0))
                self.assertIs(prepare_coordinate_batch_workspace(prepared)[0], prepared)
                for cap in (legacy.peak_host_bytes, prepared.peak_host_bytes - 1,
                            prepared.peak_host_bytes):
                    old_limited = replace(legacy, memory_limit=cap).with_reserves(
                        reserved_host_bytes=123 + correction)
                    new_limited, _ = prepare_coordinate_batch_workspace(replace(legacy, memory_limit=cap))
                    self.assertEqual(new_limited.feasible, old_limited.feasible)
                validate_coordinate_batch_workspace(prepared, np)

    def test_exact_and_dp_reject_final_workspace_overflow_before_io(self):
        provider, expanded, _path = sparse_fixture.SparsePointSinkBatchCatalogTests._four_part_path(1 << 28)
        inputs = expanded.inputs.task_batches[0]
        for optimizer in ('exact', 'dynamic_programming'):
            with self.subTest(optimizer=optimizer):
                options = dict(optimizer=optimizer, deadline_seconds=3, max_states=10000)
                accepted = solve_storage(inputs.problem, inputs.blocks, inputs.preparations,
                                         memory_limit=12064, **options)
                self.assertEqual(accepted.status, 'OPTIMAL')
                self.assertTrue(accepted.plan.feasible)
                self.assertEqual(accepted.plan.peak_host_bytes, 12064)
                self.assertEqual(max(point.host_bytes for point in accepted.plan.memory), 12064)
                validate_coordinate_batch_workspace(accepted.plan, np)
                rejected = solve_storage(inputs.problem, inputs.blocks, inputs.preparations,
                                         memory_limit=12063, **options)
                # A rejected physical goal is not a proof that every logical
                # route is impossible, and must not abort candidate search.
                self.assertEqual(rejected.status, 'UNSOLVED')
                self.assertIsNone(rejected.plan)
                self.assertIsNone(rejected.state)
                self.assertEqual(rejected.stop_reason, 'physical_admission')
                self.assertFalse(rejected.limited)
                self.assertIsNone(rejected.lower_bound_seconds)
                self.assertEqual(rejected.complete_plans, 0)
                self.assertIsNone(rejected.first_solution_seconds)
                self.assertGreater(rejected.statistics['rejected_physical_candidates'], 0)
                # Independently audit this fixed route one byte below its peak;
                # X must still reject an infeasible supplied plan before IO.
                denied = plan_scene_storage(accepted.plan.state, inputs.blocks, inputs.preparations,
                                            memory_limit=12063, profile=accepted.plan.cost_profile)
                self.assertEqual(denied.peak_host_bytes, 12064)
                self.assertFalse(denied.feasible)
                with self.assertRaisesRegex(MemoryError, 'supplied storage plan is infeasible'):
                    replay_scene_storage(denied, expanded.inputs.loader)
        self.assertEqual(provider.slab_reads, {'left': 0, 'right': 0})
        for loader in provider.loaders.values():
            self.assertEqual((loader.reads, loader.materializations, loader.closes), ([], [], []))

    def test_exact_workspace_boundary_archive_replays_without_planning(self):
        for optimizer in ('exact', 'dynamic_programming'):
            with self.subTest(optimizer=optimizer):
                provider, expanded, _path = sparse_fixture.SparsePointSinkBatchCatalogTests._four_part_path(1 << 28)
                inputs = expanded.inputs.task_batches[0]
                chosen = solve_storage(inputs.problem, inputs.blocks, inputs.preparations,
                                       memory_limit=12064, optimizer=optimizer,
                                       deadline_seconds=3, max_states=10000).plan
                shapes = {key: list(spec.shape) for key, spec in inputs.problem.specs.items()}
                hardware = {'backend': 'numpy', 'memory': 12064}
                with TemporaryDirectory() as directory, ExitStack() as stack:
                    filename = Path(directory) / 'sparse-boundary.json.gz'
                    save_storage_schedule(filename, (chosen,), shapes=shapes, hardware=hardware)
                    for module, names in (
                        ('lqcdweave.contraction.planning.api',
                         ('solve_storage', 'plan_scene_storage', 'find_contraction_path',
                          'prepare_coordinate_batch_workspace', 'join_storage_batches')),
                        ('lqcdweave.contraction.planning.storage',
                         ('plan_scene_storage', 'prepare_coordinate_batch_workspace')),
                    ):
                        for name in names:
                            stack.enter_context(patch(module + '.' + name,
                                                      side_effect=AssertionError('archive/X replanned')))
                    restored, _ = load_storage_schedule(filename, shapes=shapes, hardware=hardware,
                        expected=(inputs.problem,), blocks=inputs.blocks, preparations=inputs.preparations)
                    saved = restored[0]
                    self.assertTrue(saved.feasible)
                    self.assertEqual(saved.memory_limit, 12064)
                    self.assertEqual(saved.peak_host_bytes, 12064)
                    self.assertEqual(saved.memory, chosen.memory)
                    self.assertEqual(saved.events, chosen.events)
                    self.assertEqual(saved.state.actions, chosen.state.actions)
                    loader = expanded.inputs.loader
                    with loader.activate(0):
                        values = replay_scene_storage(saved, loader)
                reference = [sum(coefficient * np.einsum('ic,iq,cd,qd->',
                    provider.left_by_time[t][:, 0, :], gamma, provider.records[t]['values'][0],
                    provider.right_by_time[t][:, 0, :], optimize=False)
                    for gamma, coefficient in zip(provider.gammas[t], provider.coefficients[t], strict=True))
                    for t in range(2)]
                np.testing.assert_allclose(values[0], reference, rtol=2e-13, atol=2e-13)
                self.assertEqual(provider.slab_reads, {'left': 1, 'right': 1})
                for lane in provider.loaders.values():
                    self.assertEqual((len(lane.reads), len(lane.materializations), len(lane.closes)), (1, 1, 1))

    def test_saved_p_plan_executes_without_join_preparation_or_descriptor_replacement(self):
        _provider, expanded, path = sparse_fixture.SparsePointSinkBatchCatalogTests._four_part_path(1 << 28)
        saved = path.plan
        memory, events, actions = saved.memory, saved.events, saved.state.actions
        with ExitStack() as stack:
            for name in ('join_storage_batches', 'prepare_coordinate_batch_workspace',
                         'find_contraction_path', 'plan_scene_storage'):
                stack.enter_context(patch('lqcdweave.contraction.planning.api.' + name,
                                          side_effect=AssertionError('planning during X')))
            stack.enter_context(patch.object(SceneStoragePlan, 'with_reserves',
                                            side_effect=AssertionError('reserve during X')))
            result = BoundContractionExecution(expanded, path).execute(backend=np)
        self.assertEqual(len(result.term_values), 2)
        self.assertIs(path.plan, saved)
        self.assertIs(saved.memory, memory)
        self.assertIs(saved.events, events)
        self.assertIs(saved.state.actions, actions)

    def test_strict_x_rejects_unprepared_sparse_plans_before_io(self):
        provider, expanded, path = legacy_fixture()
        original = path.plan
        loader = expanded.inputs.loader
        for execute in (lambda: replay_scene_storage(original, loader),
                        lambda: replay_storage_batches((original,), loader, lambda *_: None)):
            with self.assertRaisesRegex(MemoryError, 'saved coordinate-batched sparse workspace'):
                execute()
        self.assertIs(path.plan, original)
        self.assertEqual(provider.slab_reads, {'left': 0, 'right': 0})
        self.assertTrue(all(not loader.reads for loader in provider.loaders.values()))

    def test_strict_x_rejects_nonprejoined_call_before_read(self):
        loader = Loader()
        with self.assertRaisesRegex(ValueError, 'prejoined=True'):
            replay_storage_batches((batch('a'), batch('a')), loader, lambda *_: None,
                                   prejoined=False)
        self.assertEqual(loader.reads, [])

    def test_w_convenience_joins_and_prepares_before_x_and_preserves_context_identity(self):
        _provider, expanded, path = legacy_fixture()
        original = path.plan
        loader = expanded.inputs.loader
        contexts, results = [], []
        def activate(plan):
            contexts.append(plan)
            return loader.activate(0)
        with patch.object(convenience, 'prepare_coordinate_batch_workspace',
                          wraps=prepare_coordinate_batch_workspace) as prepare:
            convenience.replay_storage_batches((original,), loader,
                lambda index, report: results.append(np.array(report[0], copy=True)),
                prejoined=True, batch_context=activate, io_overlap=False)
        self.assertEqual(prepare.call_count, 1)
        self.assertIs(contexts[0], original)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].shape, (2,))
        # The facade still defaults to the historic independent-plan convenience.
        ordinary_loader, values = Loader(), []
        with patch.object(convenience, 'join_storage_batches', wraps=join_storage_batches) as join:
            convenience.replay_storage_batches((batch('a'), batch('a')), ordinary_loader,
                                               lambda i, report: values.append(report[0]))
        self.assertEqual(join.call_count, 1)
        self.assertEqual(ordinary_loader.reads, ['a'])
        self.assertEqual(len(values), 2)

    def test_cfg_freezes_sparse_workspace_offline_and_archive_preserves_it(self):
        _provider, _expanded, path = legacy_fixture()
        original, preparations, requests = described_cfg_inputs(path.plan)
        points, _ = describe_coordinates()
        with patch('lqcdweave.contraction.execution.scene_runtime._coordinate_batch_sparse_workspace_bytes',
                   side_effect=AssertionError('P imported X')):
            cfg = build_cfg_plan(original, None, points.points, requests, preparations,
                                 original.state.problem.specs, debug=True)
        saved = cfg.templates[0].plan
        validate_coordinate_batch_workspace(saved, np)
        for action in saved.state.actions:
            if action.kind == 'sparse_bridge' and not action.reused:
                self.assertGreaterEqual(cfg.statistics['peak_operation_workspace_bytes'],
                                        _coordinate_batch_sparse_workspace_bytes(saved.state, action))
        with TemporaryDirectory() as directory:
            filename = Path(directory) / 'sparse-cfg.json'
            save_cfg_plan(filename, cfg)
            with patch('lqcdweave.contraction.planning.api.prepare_coordinate_batch_workspace',
                       side_effect=AssertionError('archive replanned')):
                restored = load_cfg_plan(filename)
        self.assertEqual(restored.peak_host_bytes, cfg.peak_host_bytes)
        self.assertEqual(restored.templates[0].plan.memory, saved.memory)
        self.assertEqual(restored.templates[0].plan.reserved_host_bytes, original.reserved_host_bytes)
        self.assertEqual(audit_cfg_plan(restored)['peak_host_bytes'], cfg.peak_host_bytes)
        validate_coordinate_batch_workspace(restored.templates[0].plan, np)
        with self.assertRaisesRegex(MemoryError, 'saved coordinate-batched sparse workspace'):
            validate_coordinate_batch_workspace(original, np)

    def test_cfg_rejects_incomplete_saved_workspace_before_loader_factory(self):
        _provider, _expanded, path = legacy_fixture()
        original, preparations, requests = described_cfg_inputs(path.plan)
        points, _ = describe_coordinates()
        cfg = build_cfg_plan(original, None, points.points, requests, preparations,
                             original.state.problem.specs)
        template = cfg.templates[0]
        memory = tuple(replace(point, workspace_host_bytes=0)
                       if point.phase == 'sparse_bridge' else point
                       for point in template.plan.memory)
        incomplete = replace(template.plan, memory=memory)
        cfg = replace(cfg, templates=(replace(template, plan=incomplete),))
        def unavailable(_cfg):
            self.fail('incomplete cfg reached its loader factory')
        with self.assertRaisesRegex(MemoryError, 'saved coordinate-batched sparse workspace'):
            tuple(iter_execute_cfg_plan(cfg, ('cfg',), unavailable))
        self.assertIs(cfg.templates[0].plan, incomplete)
        self.assertIs(incomplete.memory, memory)


if __name__ == '__main__':
    unittest.main()
