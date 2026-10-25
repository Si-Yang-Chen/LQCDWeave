"""Resident scalar grouping preserves algebra and the conservative memory trace."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, SourcePreparation,
    device_allocation_bytes,
)
from lqcdweave.contracts.correlator.scalar_reduction_group import (
    scalar_reduction_geometry, validate_scalar_reduction_groups,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, plan_scene_storage,
    lower_scalar_reduction_groups,
)


def fixture(count=4, *, upfront=True, dtype=np.float32, shape=(2, 3), chain=False,
            producers_first=False, reload_second=False):
    profile = CostProfile()
    specs = {}
    for index in range(count):
        for name, member_shape in ((f'a{index}', (2, 3) if chain else shape),
                                   (f'b{index}', (3, 2) if chain else shape)):
            specs[name] = TensorSpec(member_shape, dtype)
        if chain:
            specs[f'c{index}'] = TensorSpec((2, 2), dtype)
    terms = tuple(ContractionTerm((f'a{i}', f'b{i}', f'c{i}'), 'ij,jk,ik->') if chain
                  else ContractionTerm((f'a{i}', f'b{i}'), 'ij,ij->' if len(shape) == 2 else 'i,i->')
                  for i in range(count))
    problem = ContractionProblem(specs, terms,
        {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend='cupy')
    state = ContractionState.initial(problem)
    if upfront:
        for key in specs:
            state = state.load(key, profile, 1 << 30)
    if reload_second:
        state = state.release('a1')
    for index in range(count):
        if reload_second and index == 1:
            state = state.load('a1', profile, 1 << 30)
        if not upfront:
            for prefix in ('a', 'b', 'c') if chain else ('a', 'b'):
                state = state.load(f'{prefix}{index}', profile, 1 << 30)
        state = state.contract(index, (0, 1), profile, 1 << 30)
        if chain and not producers_first:
            state = state.contract(index, (0, 1), profile, 1 << 30)
    if chain and producers_first:
        for index in range(count):
            state = state.contract(index, (0, 1), profile, 1 << 30)
    plan = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in specs},
        profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
    return plan, profile


class ScalarReductionGroupingTests(unittest.TestCase):
    def assert_envelope(self, before, after):
        self.assertEqual(len(before.memory), len(after.memory))
        for old, new in zip(before.memory, after.memory):
            extra = 0
            for group in after.scalar_reduction_groups:
                first = group.action_indices[0]
                read = old.phase.startswith('read:')
                if old.action_index == first and old.phase == 'contract':
                    extra += device_allocation_bytes(group.workspace_bytes)
                for member in group.action_indices[1:]:
                    if ((old.action_index > first or old.action_index == first and not read)
                            and (old.action_index < member or old.action_index == member and read)):
                        extra += device_allocation_bytes(
                            before.state.catalog[before.state.actions[member].tensor].dtype.itemsize)
            self.assertEqual(new.scalar_reduction_device_bytes, extra)
            self.assertEqual(replace(new, scalar_reduction_device_bytes=0), old)
        self.assertEqual(after.peak_device_bytes, max(p.device_bytes for p in after.memory))
        validate_scalar_reduction_groups(after)

    def test_ready_snapshot_groups_keep_actions_blocks_costs_and_independent_owners(self):
        for dtype in (np.float32, np.complex64):
            original, _ = fixture(11, dtype=dtype, shape=(513,))
            selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
            self.assertTrue(stats['applied'])
            self.assertEqual(tuple(map(lambda group: len(group.action_indices), selected.scalar_reduction_groups)), (11,))
            self.assertIs(selected.state, original.state)
            self.assertIs(selected.blocks, original.blocks)
            self.assertIs(selected.events, original.events)
            self.assertEqual(selected.seconds, original.seconds)
            self.assertEqual(stats['theoretical_launches_saved'], 20)
            names = []
            for group in selected.scalar_reduction_groups:
                for index in group.action_indices:
                    action = original.state.actions[index]
                    names.append(action.tensor)
                    self.assertEqual(scalar_reduction_geometry(action, original.state.catalog).blocks, 3)
                self.assertEqual(group.workspace_bytes, len(group.action_indices) * 3 * 2 * np.dtype(dtype).itemsize)
            self.assertEqual(len(names), len(set(names)))
            self.assert_envelope(original, selected)
            again, detail = lower_scalar_reduction_groups(selected, target_runtime='cuda')
            self.assertIs(again, selected)
            self.assertEqual(detail['reason'], 'existing_scalar_groups')

    def test_future_loads_and_unfinished_producers_cannot_enter_snapshot(self):
        for kwargs in ({'upfront': False}, {'chain': True, 'producers_first': False}):
            original, _ = fixture(**kwargs)
            selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
            self.assertIs(selected, original)
            self.assertFalse(stats['applied'])
        original, _ = fixture(chain=True, producers_first=True)
        selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
        self.assertEqual(stats['grouped_reductions'], 4)
        self.assert_envelope(original, selected)

    def test_core_and_sparse_boundaries_partition_ready_candidates(self):
        for boundary in ('core', 'sparse_bridge', 'baryon_join'):
            original, _ = fixture(5)
            actions = list(original.state.actions)
            indices = [i for i, action in enumerate(actions) if action.kind == 'contract']
            actions[indices[1]] = (replace(actions[indices[1]], core=SimpleNamespace(inputs=()))
                if boundary == 'core' else replace(actions[indices[1]], kind=boundary))
            original = replace(original, state=replace(original.state, actions=tuple(actions)))
            selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
            self.assertTrue(stats['applied'])
            self.assertEqual(selected.scalar_reduction_groups[0].action_indices, tuple(indices[2:]))
            self.assert_envelope(original, selected)

    def test_released_input_invalidates_queued_candidate_until_reload(self):
        original, _ = fixture(reload_second=True)
        selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
        indices = [i for i, action in enumerate(original.state.actions) if action.kind == 'contract']
        self.assertTrue(stats['applied'])
        self.assertEqual(selected.scalar_reduction_groups[0].action_indices,
                         (indices[0], indices[2], indices[3]))
        self.assert_envelope(original, selected)

    def test_scalar_intermediate_consumed_later_is_not_terminal(self):
        profile = CostProfile()
        specs = {f'{name}{i}': TensorSpec((2, 3) if name != 'c' else (), np.float32)
                 for i in range(3) for name in ('a', 'b', 'c')}
        problem = ContractionProblem(specs,
            tuple(ContractionTerm((f'a{i}', f'b{i}', f'c{i}'), 'ij,ij,->') for i in range(3)),
            {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend='cupy')
        state = ContractionState.initial(problem)
        for key in specs:
            state = state.load(key, profile, 1 << 30)
        for index in range(3):
            state = state.contract(index, (0, 1), profile, 1 << 30)
        for index in range(3):
            state = state.contract(index, (0, 1), profile, 1 << 30)
        original = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in specs},
            profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
        selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
        self.assertIs(selected, original)
        self.assertEqual(stats['excluded']['nonterminal_or_shared_result'], 3)
        self.assertEqual(stats['candidates'], 0)

    def test_dtype_and_geometry_partition_without_changing_blocks(self):
        original, _ = fixture(4)
        actions = list(original.state.actions)
        indices = [i for i, action in enumerate(actions) if action.kind == 'contract']
        catalog = dict(original.state.catalog)
        # A consistently wider source/output dtype is unsupported rather than
        # silently normalized into the supported scalar precision.
        for key in (*actions[indices[1]].operands, actions[indices[1]].tensor):
            catalog[key] = replace(catalog[key], dtype=np.dtype('float64'))
        # An equal-size but transposed equation has a distinct source geometry.
        from lqcdweave.contracts.correlator.api import AxisPattern
        actions[indices[2]] = replace(actions[indices[2]], equation=AxisPattern(((0, 1), (1, 0)), ()))
        key = actions[indices[2]].operands[1]
        catalog[key] = replace(catalog[key], shape=(3, 2))
        original = replace(original, state=replace(original.state, actions=tuple(actions), catalog=catalog))
        selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
        self.assertEqual(selected.scalar_reduction_groups[0].action_indices, (indices[0], indices[3]))
        self.assertEqual(stats['candidates'], 3)
        self.assert_envelope(original, selected)

    def test_budget_shrinks_groups_and_keeps_all_old_scratch(self):
        # Partials must exceed one pool allocation unit; for tiny reductions
        # all widths share a 512-byte partial owner, changing the boundary.
        original, _ = fixture(4, shape=(65536,))
        original = replace(original, memory_limit=original.peak_device_bytes + 4096)
        selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
        self.assertTrue(stats['applied'])
        self.assertEqual([len(g.action_indices) for g in selected.scalar_reduction_groups], [2, 2])
        self.assertGreater(stats['resource_rejections'], 0)
        self.assertLessEqual(selected.peak_device_bytes, original.memory_limit)
        self.assert_envelope(original, selected)
        tight = replace(original, memory_limit=original.peak_device_bytes)
        unchanged, detail = lower_scalar_reduction_groups(tight, target_runtime='cuda')
        self.assertIs(unchanged, tight)
        self.assertFalse(detail['applied'])

    def test_unsupported_domains_contract_failure_and_nonzero_bucket_return_original(self):
        original, _ = fixture()
        for candidate, runtime, width in (
            (original, 'cpu', 8), (original, 'hip', 8),
            (replace(original, explicit_resources=True), 'cuda', 8),
            (replace(original, retention_planned=True), 'cuda', 8),
            (replace(original, feasible=False), 'cuda', 8),
            (original, 'cuda', 1), (original, 'cuda', 33),
            (replace(original, memory=(replace(original.memory[0], scalar_reduction_device_bytes=1), *original.memory[1:])), 'cuda', 8),
        ):
            selected, stats = lower_scalar_reduction_groups(candidate, target_runtime=runtime, max_tasks=width)
            self.assertIs(selected, candidate)
            self.assertFalse(stats['applied'])
        with patch('lqcdweave.contraction.planning.scalar_reduction_groups.validate_scalar_reduction_groups',
                   side_effect=NotImplementedError('extended contract')):
            selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
        self.assertIs(selected, original)
        self.assertEqual(stats['reason'], 'contract_validation_unsupported_or_failed')

    def test_old_cartesian_and_gemm_group_members_are_excluded(self):
        from correlator.planning.test_gemm_groups import fixture as matrix_fixture
        from lqcdweave.contraction.planning.api import lower_independent_dense_batches
        original, profile = matrix_fixture()
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        for plan in (original, grouped):
            selected, stats = lower_scalar_reduction_groups(plan, target_runtime='cuda')
            self.assertIs(selected, plan)
            self.assertEqual(stats['excluded']['existing_batch_member'], 12)

    def test_memory_read_phase_includes_early_output_until_original_contract(self):
        original, _ = fixture(3)
        actions = [i for i, a in enumerate(original.state.actions) if a.kind == 'contract']
        points = []
        for point in original.memory:
            if point.action_index in actions and point.phase == 'contract':
                points.append(replace(point, phase='read:synthetic'))
            points.append(point)
        original = replace(original, memory=tuple(points))
        selected, stats = lower_scalar_reduction_groups(original, target_runtime='cuda')
        self.assertTrue(stats['applied'])
        self.assert_envelope(original, selected)


if __name__ == '__main__':
    unittest.main()
