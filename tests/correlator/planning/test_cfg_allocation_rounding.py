"""Small independent allocation oracles for whole-cfg GPU envelopes (no CUDA)."""
from dataclasses import replace
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, ContractionProblem, ContractionTerm, SourcePreparation,
    SourceSpec, TensorPreparation, TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, plan_scene_storage, build_cfg_plan, audit_cfg_plan,
)
from lqcdweave.contraction.planning import cfg_rounding
from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value


def tiny(*, backend='cupy', repeat=2, duplicate_output=False, workspace=0, cap=None, debug=False):
    specs = {key: TensorSpec((1,), 'float32') for key in ('a', 'b')}
    terms = (ContractionTerm(('a', 'b'), 'i,i->'),)
    problem = ContractionProblem(specs, terms, {key: SourceSpec(key, 0, read_bytes=0) for key in specs}, backend=backend)
    profile = CostProfile()
    state = ContractionState.initial(problem)
    for key in specs:
        state = state.load(key, profile, 1 << 20)
    state = state.contract(0, (0, 1), profile, 1 << 20)
    if duplicate_output:
        state = replace(state, results=(state.results[0],) * 9)
    preps = {key: SourcePreparation((), workspace_device_bytes=workspace if key == 'a' else 0,
        recipe=TensorPreparation('constant', constants=(ArrayConstant.from_array(np.array([index+1], np.float32)),)))
        for index, key in enumerate(specs)}
    selected = plan_scene_storage(state, {}, preps, profile=profile, memory_limit=1 << 20)
    memory = tuple(replace(point, packed_value_cache_device_bytes=0) for point in selected.memory)
    selected = replace(selected, memory=memory, packed_value_cache_device_bytes=0,
                       peak_device_bytes=max(point.device_bytes for point in memory))
    points, _ = describe_coordinates(time_pairs=[{0: 0}] * repeat)
    cfg = build_cfg_plan(selected, None, points.points, {}, preps, specs,
        prefetch_host_bytes=0, prefetch_device_bytes=0,
        target_summary=None if cap is None else {'device_memory_limit': cap}, debug=debug)
    return selected, cfg


class CfgAllocationRoundingTests(unittest.TestCase):
    def test_independent_scalar_owner_and_workspace_oracle_all_reused_coordinate(self):
        imported = 'cupy' in sys.modules
        _, cfg = tiny()
        # Two distinct 4-byte constants; a, b and scalar result at the first
        # contraction; independent permitted a/b/result layout-copy slots.
        expected = (2 + 3 + 3) * 512
        self.assertEqual(cfg.peak_device_bytes, expected)
        self.assertTrue(all(cfg.instances[1].reused.array()))
        self.assertEqual(audit_cfg_plan(cfg)['peak_device_bytes'], expected)
        self.assertEqual(cfg.statistics['constant_device_bytes'], 1024)
        self.assertEqual(cfg.statistics['constant_device_payload_bytes'], 8)
        self.assertEqual('cupy' in sys.modules, imported)

    def test_tight_cap_rejects_before_execution(self):
        with self.assertRaisesRegex(MemoryError, 'resource envelope'):
            tiny(cap=4095)
        self.assertEqual(tiny(cap=4096)[1].peak_device_bytes, 4096)

    def test_output_final_slot_rounds_each_fallback_occurrence(self):
        _, cfg = tiny(duplicate_output=True)
        self.assertEqual(cfg.statistics['native_output_snapshot_device_bytes'], 512)
        self.assertEqual(cfg.statistics['output_snapshot_device_bytes'], 9 * 512)
        # Both constants remain cached; first-coordinate a/b/scalar remain live
        # for the next coordinate while all nine debug/measure copies exist.
        self.assertEqual(cfg.peak_device_bytes, 14 * 512)
        self.assertEqual(audit_cfg_plan(cfg)['peak_device_bytes'], cfg.peak_device_bytes)

    def test_archive_and_independent_audit_reject_understated_legacy_envelope(self):
        _, cfg = tiny()
        restored = decode_plan_value(encode_plan_value(cfg))
        self.assertEqual(audit_cfg_plan(restored)['peak_device_bytes'], cfg.peak_device_bytes)
        stats = {key: value for key, value in cfg.statistics.items()
                 if key not in ('cfg_allocation_rounding_version', 'constant_device_bytes')}
        old = replace(restored, statistics=stats, peak_device_bytes=4095)
        with self.assertRaisesRegex(ValueError, 'device live storage'):
            audit_cfg_plan(old)
        # Merely changing an informational statistic cannot suppress the check.
        altered = replace(cfg, statistics={**cfg.statistics, 'constant_device_bytes': 0}, peak_device_bytes=3584)
        with self.assertRaisesRegex(ValueError, 'device live storage'):
            audit_cfg_plan(altered)
        for name in ('native_output_snapshot_device_bytes', 'output_snapshot_device_bytes'):
            changed = replace(cfg, statistics={**cfg.statistics, name: 4})
            with self.assertRaisesRegex(ValueError, 'snapshot allocation metadata'):
                audit_cfg_plan(changed)

    def test_cpu_has_no_gpu_request_charge_and_same_actions(self):
        gpu, gc = tiny()
        cpu, cc = tiny(backend='numpy')
        self.assertEqual(cpu.state.actions, gpu.state.actions)
        self.assertEqual(cc.peak_device_bytes, 0)
        self.assertEqual(audit_cfg_plan(cc)['peak_device_bytes'], 0)
        self.assertEqual(cc.statistics['cfg_allocation_rounding_version'], 0)
        self.assertEqual(gc.statistics['cfg_allocation_rounding_version'], 1)

    def test_unique_constants_round_individually_and_equal_constants_share(self):
        a = ArrayConstant.from_array(np.array([1], np.float32))
        b = ArrayConstant.from_array(np.array([2], np.float32))
        def prep(c): return SourcePreparation((), recipe=TensorPreparation('constant', constants=(c,)))
        self.assertEqual(cfg_rounding.constant_bytes({'a': prep(a), 'b': prep(b)}, device=True), 1024)
        self.assertEqual(cfg_rounding.constant_bytes({'a': prep(a), 'again': prep(a)}, device=True), 512)

    def test_declared_preparation_workspace_is_retained_and_rounded(self):
        _, cfg = tiny(workspace=4097)
        # First load: two constant owners, one source owner, one 4097-byte
        # declared slot rounded to 4608. No workspace is charged for reuse.
        self.assertEqual(cfg.peak_device_bytes, 2*512 + 512 + 4608)
        self.assertEqual(audit_cfg_plan(cfg)['peak_device_bytes'], cfg.peak_device_bytes)
        # Debug deliberately rebinds preparations dynamically: max the actual
        # declared workspace before adding request padding, exactly as normal.
        self.assertEqual(tiny(workspace=4097, debug=True)[1].peak_device_bytes,
                         cfg.peak_device_bytes)

    def test_profile_fallback_explicit_allocations_fit_recipe_padding_bound(self):
        from lqcdweave.contraction.execution.preparation import _bind_spin_operation

        class Tracked(np.ndarray):
            allocations = 0

            def __array_finalize__(self, parent):
                if parent is not None and self.size and not np.shares_memory(self, parent):
                    type(self).allocations += 1

        for rows in (0, 1, 3):
            for diagonal in (False, True):
                with self.subTest(profile_rows=rows, diagonal=diagonal):
                    values = np.arange(4, dtype=np.float64).reshape(2, 2).astype(np.complex128)
                    constants = [np.eye(4, dtype=np.complex64)]
                    if rows:
                        constants.extend((np.ones((rows, 2), np.complex64),
                                          np.full((rows, 2), 2, np.complex64),
                                          np.arange(1, rows+1, dtype=np.float32)))
                    saved = tuple(ArrayConstant.from_array(value) for value in constants)
                    recipe = TensorPreparation('spin_spatial', ('raw',), saved,
                        conjugate=True, permutation=(1, 0),
                        configuration='cfg' if rows else None,
                        spatial_structure='diagonal' if diagonal else 'general',
                        compute_dtype='complex64')
                    spec = TensorSpec((4, 4, 2) if diagonal else (4, 4, 2, 2), 'complex64')
                    operation = _bind_spin_operation(np, recipe, spec)
                    tracked = tuple(value.view(Tracked) for value in constants)
                    raw = values.view(Tracked)
                    Tracked.allocations = 0
                    result = operation(tracked, {'raw': raw})
                    observed = Tracked.allocations
                    reference = operation(tuple(constants), {'raw': values})
                    np.testing.assert_array_equal(result, reference)
                    bound = cfg_rounding.preparation_padding({'spin': SourcePreparation((), recipe=recipe)})['spin']
                    self.assertGreater(observed, 0)
                    self.assertLessEqual(observed * 511, bound)

    def test_window_reset_keeps_constants_once_and_audits_actual_owners(self):
        from lqcdweave.contraction.planning import cfg_plan
        with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1):
            _, cfg = tiny(repeat=3)
        self.assertTrue(all(not instance.reused.array().any() for instance in cfg.instances))
        self.assertEqual(cfg.statistics['constant_device_bytes'], 1024)
        self.assertEqual(cfg.peak_device_bytes, 4096)
        self.assertEqual(audit_cfg_plan(cfg)['peak_device_bytes'], 4096)

    def test_budget_shrinks_reuse_window_using_rounded_read_and_owner_peak(self):
        from correlator.planning.test_cartesian_cfg_handoff import read_between_children
        from lqcdweave.contraction.planning import cfg_plan
        selected, requests, preparations, specs = read_between_children()
        requests = {key: replace(request, coordinate_arguments=((0, '0', 0, None),))
                    for key, request in requests.items()}
        points, _ = describe_coordinates(time_pairs=[{0: value} for value in (0, 1, 0)])
        bound = cfg_plan._bind_cfg_template(selected, None, requests, preparations, specs,
            output_mode='dataset', prefetch_host_bytes=0, prefetch_device_bytes=0)
        full = bound(points.points, reuse_window=3)
        small = bound(points.points, reuse_window=1)
        self.assertLess(small.peak_device_bytes, full.peak_device_bytes)
        with patch('lqcdweave.contraction.planning.api.find_contraction_path',
                   side_effect=AssertionError('resource reconstruction searched')):
            limited = build_cfg_plan(selected, None, points.points, requests, preparations, specs,
                output_mode='dataset', prefetch_host_bytes=0, prefetch_device_bytes=0,
                target_summary={'device_memory_limit': small.peak_device_bytes}, debug=True)
        self.assertEqual(limited.statistics['reuse_window'], 1)
        self.assertGreater(limited.statistics['resource_attempts'], 1)
        self.assertEqual(limited.peak_device_bytes, small.peak_device_bytes)
        self.assertEqual(limited.templates[0].plan.state.actions, full.templates[0].plan.state.actions)
        self.assertEqual(audit_cfg_plan(limited)['peak_device_bytes'], limited.peak_device_bytes)

    def test_legacy_archive_is_audited_not_upgraded_or_researched(self):
        from lqcdweave.storage.contraction_plans.api import save_cfg_plan, load_cfg_plan
        from lqcdweave.contraction.planning.api import validate_cfg_allocation_envelope
        _, cfg = tiny()
        stats = {key: value for key, value in cfg.statistics.items()
                 if key not in ('cfg_allocation_rounding_version', 'output_snapshot_device_bytes',
                                'native_output_snapshot_device_bytes')}
        legacy = replace(cfg, statistics=stats)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.plan.gz'
            save_cfg_plan(path, legacy)
            with patch('lqcdweave.contraction.planning.api.find_contraction_path',
                       side_effect=AssertionError('archive load searched')):
                restored = load_cfg_plan(path)
            self.assertEqual(restored, legacy)
            self.assertNotIn('cfg_allocation_rounding_version', restored.statistics)
            save_cfg_plan(path, replace(legacy, peak_device_bytes=4095))
            with self.assertRaisesRegex(MemoryError, 'existing selected contraction path'):
                load_cfg_plan(path)
        _, cpu = tiny(backend='numpy')
        cpu = replace(cpu, statistics={})
        self.assertIs(validate_cfg_allocation_envelope(cpu), cpu)


if __name__ == '__main__':
    unittest.main()
