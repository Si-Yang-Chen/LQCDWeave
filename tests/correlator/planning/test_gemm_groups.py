"""Resident independent GEMMs preserve selected algebra and owner bounds."""
from dataclasses import replace
from types import MappingProxyType
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, TensorSpec, SourceSpec, SourcePreparation,
    DenseGemmGroup, dense_batch_memory, validate_executable_resources,
    ArrayConstant, TensorPreparation, describe_coordinates,
    PlannedContraction,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, describe_dense_batch, plan_scene_storage,
    lower_independent_dense_batches, build_cfg_plan, audit_cfg_plan,
)
from lqcdweave.contraction.execution.operations.dense_batch import bind_dense_batch
from lqcdweave.contraction.execution.scene_runtime import bind_scene_execution
from lqcdweave.storage.contraction_plans.api import (
    encode_plan_value, decode_plan_value, save_storage_schedule, load_storage_schedule,
)
from lqcdweave.contraction.planning.gemm_groups import lower_independent_gemm_path
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.contraction.performance.api import predict_plan_cost


def fixture(*, loads_upfront=True, chained=False, count=3):
    profile = CostProfile()
    specs = {**{f'a{g}_{i}': TensorSpec((2+i, 5), np.complex128)
                for g in range(count) for i in range(2)},
             **{f'b{g}_{j}': TensorSpec((5, 1+j), np.complex128) for g in range(count) for j in range(2)}}
    if chained:
        specs.update({f'd{j}': TensorSpec((1+j, 4), np.complex128) for j in range(2)})
    terms = tuple(ContractionTerm((f'a{g}_{i}', f'b{g}_{j}', f'd{j}') if chained
                                  else (f'a{g}_{i}', f'b{g}_{j}'),
                                  'ij,jk,kl->il' if chained else 'ij,jk->ik')
                  for g in range(count) for i in range(2) for j in range(2))
    problem = ContractionProblem(specs, terms, {k: SourceSpec(k, 0, read_bytes=0) for k in specs},
                                 backend='cupy')
    state = ContractionState.initial(problem)
    if loads_upfront:
        for key in specs:
            state = state.load(key, profile, 1 << 30)
    indices = []
    for g in range(count):
        if not loads_upfront:
            for key in (f'a{g}_0', f'a{g}_1', f'b{g}_0', f'b{g}_1'):
                state = state.load(key, profile, 1 << 30)
        group_indices = []
        for i in range(4):
            group_indices.append(len(state.actions))
            state = state.contract(4*g+i, (0, 1), profile, 1 << 30)
        indices.append(tuple(group_indices))
        if chained:
            intermediates = [state.actions[index].tensor for index in group_indices]
            for i, key in enumerate(intermediates):
                state = state.contract(4*g+i, (0, 1), profile, 1 << 30)
                state = state.release(key)
    plan = plan_scene_storage(state, {}, {key: SourcePreparation(()) for key in specs},
                              profile=profile, memory_limit=1 << 30, host_memory_limit=1 << 30)
    batches = tuple(describe_dense_batch(state.actions, group, state.catalog, profile, backend='cupy')
                    for group in indices)
    assert all(batches)
    plan = replace(plan, kernel_batches=batches)
    memory = dense_batch_memory(plan)
    return replace(plan, memory=memory, peak_device_bytes=max(p.device_bytes for p in memory)), profile


class IndependentGemmPlanTests(unittest.TestCase):
    def test_cfg_bound_group_attributes_match_live_properties_and_avoid_reexpansion(self):
        from contextlib import ExitStack
        original, profile = fixture()
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        specs = original.state.problem.specs
        preps = {key: SourcePreparation((), recipe=TensorPreparation('constant', constants=(
            ArrayConstant.from_array(np.full(spec.shape, index+1, dtype=spec.dtype)),)))
            for index, (key, spec) in enumerate(specs.items())}
        points, _ = describe_coordinates(time_pairs=[{0:i} for i in range(4)])
        names = ('action_indices', 'results', 'output_bytes', 'packing_bytes')
        native = cfg_plan._bind_cfg_template(grouped, None, {}, preps, specs)
        live = cfg_plan._bind_cfg_template(grouped, None, {}, preps, specs)
        live.keywords['batch_attributes'] = cfg_plan._bind_batch_attributes(
            live.keywords['remapped_batches'], debug=True)
        collected, records = [], []
        for bound in (native, live):
            calls = dict.fromkeys(names, 0)
            def counted(name, getter):
                def read(batch):
                    calls[name] += 1
                    return getter(batch)
                return property(read)
            with ExitStack() as stack:
                for name in names:
                    stack.enter_context(patch.object(DenseGemmGroup, name,
                        counted(name, getattr(DenseGemmGroup, name).fget)))
                collected.append(bound(points.points, reuse_window=2))
            records.append(calls)
        self.assertEqual(replace(collected[0], statistics={}), replace(collected[1], statistics={}))
        for name in names:
            self.assertLess(records[0][name], records[1][name], name)
        audit_cfg_plan(collected[0])

    def test_cfg_debug_and_extended_groups_keep_property_hooks(self):
        original, profile = fixture()
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        group, = grouped.kernel_batches
        class ExtendedGroup(DenseGemmGroup):
            pass
        extended = ExtendedGroup(**vars(group))
        calls = []
        original_property = DenseGemmGroup.packing_bytes.fget
        def counted(batch):
            calls.append(batch)
            return original_property(batch)
        with patch.object(DenseGemmGroup, 'packing_bytes', property(counted)):
            fixed = cfg_plan._bind_batch_attributes((group,), debug=False)[3]
            debug = cfg_plan._bind_batch_attributes((group,), debug=True)[3]
            fallback = cfg_plan._bind_batch_attributes((extended,), debug=False)[3]
            self.assertEqual(len(calls), 1)
            for _ in range(2):
                self.assertEqual(fixed(group), original_property(group))
            self.assertEqual(len(calls), 1)
            for _ in range(2):
                self.assertEqual(debug(group), original_property(group))
                self.assertEqual(fallback(extended), original_property(extended))
            self.assertEqual(len(calls), 5)

    def test_cfg_partial_reuse_metadata_matches_live_group_expansion(self):
        original, profile = fixture()
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        specs = original.state.problem.specs
        preps = {key: SourcePreparation((), recipe=TensorPreparation('constant', constants=(
            ArrayConstant.from_array(np.ones(spec.shape, dtype=spec.dtype)),))) for key, spec in specs.items()}
        points, _ = describe_coordinates(time_pairs=[{0:0}, {0:1}, {0:0}])
        for window in (1, 2, 3):
            with self.subTest(reuse_window=window):
                bound = cfg_plan._bind_cfg_template(grouped, None, {}, preps, specs)
                live = cfg_plan._bind_cfg_template(grouped, None, {}, preps, specs)
                live.keywords['batch_attributes'] = cfg_plan._bind_batch_attributes(
                    live.keywords['remapped_batches'], debug=True)
                actual = bound(points.points, reuse_window=window)
                expected = live(points.points, reuse_window=window)
                self.assertEqual(replace(actual, statistics={}), replace(expected, statistics={}))
                self.assertEqual(len(actual.instances[0].kernel_batches), 1)
                self.assertEqual(actual.instances[0].kernel_batches[0],
                                 actual.templates[0].plan.kernel_batches[0].batches[0])
                audit_cfg_plan(actual)

    def test_wide_ready_group_preserves_path_views_and_saved_selection(self):
        original, profile = fixture(count=18)
        grouped, stats = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        group, = grouped.kernel_batches
        self.assertEqual(len(group.batches), 18)
        self.assertIs(grouped.state, original.state)
        self.assertTrue(stats['resource_audited'])
        self.assertEqual(group.pointer_bytes, 512)
        validate_executable_resources(grouped)
        self.assertEqual(decode_plan_value(encode_plan_value(grouped)), grouped)
        bound = bind_dense_batch(group, original.state.catalog, np)
        previous = None
        for seed in (451, 452):
            rng = np.random.default_rng(seed)
            arrays = {key: rng.normal(size=spec.shape) + 1j*rng.normal(size=spec.shape)
                      for key, spec in original.state.problem.specs.items()}
            bound(arrays)
            actual = tuple(arrays[key] for key in original.state.results)
            expected = tuple(arrays[t.inputs[0]] @ arrays[t.inputs[1]] for t in original.state.problem.terms)
            for value, reference in zip(actual, expected, strict=True):
                np.testing.assert_allclose(value, reference, rtol=2e-12, atol=2e-12)
            if previous is not None:
                for old, reference in zip(*previous, strict=True):
                    np.testing.assert_allclose(old, reference, rtol=2e-12, atol=2e-12)
                self.assertFalse(np.shares_memory(actual[0], previous[0][0]))
            previous = actual, expected

    def test_fixed_pointer_table_capacity_keeps_overflow_as_an_atomic_batch(self):
        original, profile = fixture(count=22)
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        group, tail = grouped.kernel_batches
        self.assertEqual(len(group.batches), 21)
        self.assertEqual(tail, original.kernel_batches[-1])
        self.assertLessEqual(3 * len(group.batches) * 8, group.pointer_bytes)
        with self.assertRaisesRegex(ValueError, '2–21'):
            replace(group, batches=original.kernel_batches)
        validate_executable_resources(grouped)

    def test_path_handoff_updates_cost_and_archive_restores_without_group_search(self):
        original, profile = fixture()
        prediction = predict_plan_cost((original,), profile, workload={}, provenance={})
        prediction['predicted_seconds'] = original.seconds + 1
        path = PlannedContraction(None, original, None, prediction, 1,
                                  profile=profile, plan_identity=id(original))
        path = lower_independent_gemm_path(path, target_runtime='cuda')
        self.assertEqual(path.plan_identity, id(path.plan))
        self.assertAlmostEqual(path.prediction['total_seconds'], path.plan.seconds)
        self.assertAlmostEqual(path.prediction['predicted_seconds'], path.plan.seconds+1)
        self.assertEqual(path.prediction['kernel_models']['cuda-cublas-gemm-batched-v1']
                         ['calibration_status'], 'estimated-from-dense-v1')
        shapes = {key: list(spec.shape) for key, spec in original.state.problem.specs.items()}
        hardware = {'backend':'cupy', 'memory':1 << 30}
        with TemporaryDirectory() as directory:
            archive = Path(directory)/'group.json'
            save_storage_schedule(archive, (path.plan,), shapes=shapes, hardware=hardware)
            with patch('lqcdweave.contraction.planning.gemm_groups.lower_independent_dense_batches',
                       side_effect=AssertionError('archive cannot regroup')):
                restored, _ = load_storage_schedule(archive, shapes=shapes, hardware=hardware,
                    expected=(original.state.problem,), blocks={}, preparations=original.preparations)
        self.assertEqual(restored[0].kernel_batches, path.plan.kernel_batches)

    def test_owner_survives_early_child_release_and_frozen_dispatch_skips_future_starts(self):
        original, profile = fixture(chained=True)
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        self.assertIsInstance(grouped.kernel_batches[0], DenseGemmGroup)
        validate_executable_resources(grouped)
        rng = np.random.default_rng(407)
        inputs = {key: rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
                  for key, spec in original.state.problem.specs.items()}

        class Execution:
            def begin_action(self, *args): pass
            def end_action(self, *args): pass
            def source(self, key): return original.state.problem.sources[key]
            def materialize(self, key, source): return inputs[key]
            def materializes_owned(self, key): return False

        table = bind_scene_execution(grouped.state, grouped.kernel_batches, np, events=())
        values = table.program.run(Execution(), {})
        for term, value in zip(original.state.problem.terms, values):
            expected = inputs[term.inputs[0]]@inputs[term.inputs[1]]@inputs[term.inputs[2]]
            np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)

    def test_offline_group_keeps_actions_and_roundtrips_selected_owner_contract(self):
        original, profile = fixture()
        grouped, stats = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        self.assertTrue(stats['applied'])
        self.assertIs(grouped.state, original.state)
        self.assertEqual(len(grouped.kernel_batches), 1)
        group = grouped.kernel_batches[0]
        self.assertIsInstance(group, DenseGemmGroup)
        self.assertEqual(group.matrix_shape, (5, 5, 3))
        self.assertEqual(group.output_bytes, sum(b.output_bytes for b in original.kernel_batches)+512)
        self.assertLess(grouped.seconds, original.seconds)
        self.assertEqual(stats['cost_status'], 'estimated-from-dense-v1')
        self.assertEqual(decode_plan_value(encode_plan_value(grouped)), grouped)
        validate_executable_resources(grouped)

    def test_gpu_cap_rejects_group_without_changing_existing_implementation(self):
        original, profile = fixture()
        limited = replace(original, memory_limit=original.peak_device_bytes)
        grouped, stats = lower_independent_dense_batches(limited, profile=profile, target_runtime='cuda')
        self.assertIs(grouped, limited)
        self.assertFalse(stats['applied'])
        self.assertGreater(stats['resource_rejections'], 0)

    def test_future_inputs_and_non_cuda_runtime_do_not_get_grouped(self):
        original, profile = fixture(loads_upfront=False)
        self.assertIs(lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')[0], original)
        original, profile = fixture()
        self.assertIs(lower_independent_dense_batches(original, profile=profile, target_runtime='hip')[0], original)
        self.assertIs(lower_independent_dense_batches(original, profile=profile, target_runtime='cpu')[0], original)

    def test_numerical_reference_uses_fresh_arrays_and_all_child_views(self):
        original, profile = fixture()
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        bound = bind_dense_batch(grouped.kernel_batches[0], grouped.state.catalog, np)
        for seed in (401, 402):
            rng = np.random.default_rng(seed)
            arrays = {key: rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
                      for key, spec in original.state.problem.specs.items()}
            bound(arrays)
            for term, key in zip(original.state.problem.terms, original.state.results):
                np.testing.assert_allclose(arrays[key], arrays[term.inputs[0]]@arrays[term.inputs[1]],
                                           rtol=2e-12, atol=2e-12)

    def test_cfg_preserves_group_and_complete_reuse_without_search(self):
        original, profile = fixture()
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        specs = original.state.problem.specs
        preps = {key: SourcePreparation((), recipe=TensorPreparation('constant', constants=(
            ArrayConstant.from_array(np.full(spec.shape, index+1, dtype=spec.dtype)),)))
            for index, (key, spec) in enumerate(specs.items())}
        points, _ = describe_coordinates(time_pairs=[{0:0, 1:0}, {0:1, 1:1}])
        with patch('lqcdweave.contraction.planning.gemm_groups.lower_independent_dense_batches',
                   side_effect=AssertionError('cfg must preserve selection')):
            cfg = build_cfg_plan(grouped, None, points.points, {}, preps, specs, debug=True)
        self.assertIsInstance(cfg.instances[0].kernel_batches[0], DenseGemmGroup)
        self.assertEqual(cfg.instances[1].kernel_batches, ())
        audit_cfg_plan(cfg)

    def test_cfg_partial_reuse_retains_original_child_batches(self):
        original, profile = fixture()
        grouped, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        specs = original.state.problem.specs
        # Equal immutable inputs share global identities, making later children
        # reused while the first Cartesian child still needs execution.
        preps = {key: SourcePreparation((), recipe=TensorPreparation('constant', constants=(
            ArrayConstant.from_array(np.ones(spec.shape, dtype=spec.dtype)),))) for key, spec in specs.items()}
        points, _ = describe_coordinates()
        cfg = build_cfg_plan(grouped, None, points.points, {}, preps, specs, debug=True)
        self.assertEqual(len(cfg.instances[0].kernel_batches), 1)
        self.assertEqual(cfg.instances[0].kernel_batches[0], cfg.templates[0].plan.kernel_batches[0].batches[0])
        audit_cfg_plan(cfg)


if __name__ == '__main__':
    unittest.main()
