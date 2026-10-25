"""Window-local data bindings reuse only equivalent immutable execution structure."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from correlator.planning import test_cfg_plan as cfg_fixtures
from correlator.execution import test_cfg_runtime as runtime_fixtures
from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, CfgInstance, DenseBatchOperand, DenseBatchResult, DenseGemmGroup,
    TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.execution import cfg_runtime, storage_runtime, scene_runtime
from lqcdweave.contraction.execution.api import execute_cfg_windows
from lqcdweave.contraction.expansion import saved_inputs
from lqcdweave.contraction.planning.api import iter_cfg_plan_windows
from lqcdweave.contraction.correlator_cfg import CompleteCalculation, source_loader_factory


class CfgTableReuseTests(unittest.TestCase):
    def test_batch_kernel_identity_reuse_requires_native_fields_and_keeps_current_numerics(self):
        from correlator.planning.test_gemm_groups import fixture
        from lqcdweave.contraction.planning.api import lower_independent_dense_batches
        from lqcdweave.contraction.execution.diagnostics import execution_debug

        class ExtendedGroup(DenseGemmGroup):
            pass
        class ExtendedOperand(DenseBatchOperand):
            pass
        class ExtendedResult(DenseBatchResult):
            pass
        class CountingKernels(dict):
            batch_gets = 0
            def get(self, key, default=None):
                self.batch_gets += int(key[0] == 'batch')
                return super().get(key, default)

        plan, profile = fixture(chained=True)
        plan, _ = lower_independent_dense_batches(plan, profile=profile, target_runtime='cuda')
        group = next(batch for batch in plan.kernel_batches if type(batch) is DenseGemmGroup)
        first, *children = group.batches
        operand, *left = first.left
        result, *results = first.results
        operand_child = replace(first, left=(ExtendedOperand(**vars(operand)), *left))
        result_child = replace(first, results=(ExtendedResult(**vars(result)), *results))
        variants = ((group, True), (first, True), (ExtendedGroup(**vars(group)), False),
                    (replace(group, batches=(operand_child, *children)), False),
                    (replace(group, batches=(result_child, *children)), False),
                    (replace(group, seconds=np.float64(group.seconds)), False))
        rng = np.random.default_rng(418)
        inputs = {key: rng.normal(size=spec.shape) + 1j*rng.normal(size=spec.shape)
                  for key, spec in plan.state.problem.specs.items()}
        for batch, native in variants:
            for debug in (False, True):
                with self.subTest(batch=type(batch).__name__, native=native, debug=debug):
                    cache, identities = CountingKernels(), {}
                    with execution_debug(debug):
                        first_table = scene_runtime.bind_scene_execution(plan.state, (batch,), np,
                            events=plan.events, shared_residency=True, kernel_cache=cache,
                            _batch_kernels=identities)
                        table = scene_runtime.bind_scene_execution(plan.state, (batch,), np,
                            events=plan.events, shared_residency=True, kernel_cache=cache,
                            _batch_kernels=identities)
                    self.assertEqual(cache.batch_gets, 1 if native and not debug else 2)
                    self.assertTrue(all(table.kernels[key] is kernel
                                        for key, kernel in first_table.kernels.items()))
                    for scale in (1, -2j):
                        current = {key: value * scale for key, value in inputs.items()}
                        execution = SimpleNamespace(source=lambda key: None,
                            materialize=lambda key, source: current[key], materializes_owned=lambda key: False,
                            begin_action=lambda index, timed: None, end_action=lambda index, timed: None)
                        actual = table.program.run(execution, {})
                        expected = tuple(current[term.inputs[0]] @ current[term.inputs[1]] @ current[term.inputs[2]]
                                         for term in plan.state.problem.terms)
                        for value, reference in zip(actual, expected, strict=True):
                            np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)

    def test_group_pattern_reuse_preserves_live_extension_hooks_and_current_arrays(self):
        from correlator.planning.test_gemm_groups import fixture
        from lqcdweave.contraction.planning.api import lower_independent_dense_batches
        from lqcdweave.contraction.execution.diagnostics import execution_debug

        class ExtendedGroup(DenseGemmGroup):
            pass

        scene, profile = fixture(chained=True)
        scene, _ = lower_independent_dense_batches(scene, profile=profile, target_runtime='cuda')
        groups = sum(type(batch) is DenseGemmGroup for batch in scene.kernel_batches)
        self.assertGreater(groups, 0)
        mask = ArrayConstant.from_array(np.zeros(len(scene.state.actions), dtype=bool))
        getter = DenseGemmGroup.action_indices.fget

        def bind(count, debug, extended):
            batches = tuple(ExtendedGroup(**vars(batch)) if extended and type(batch) is DenseGemmGroup
                            else batch for batch in scene.kernel_batches)
            current = replace(scene, kernel_batches=batches)
            template = SimpleNamespace(keys=tuple(scene.state.catalog), plan=current)
            instance = SimpleNamespace(template=0, reused=mask, events=current.events, kernel_batches=batches)
            plan = SimpleNamespace(templates=(template,), instances=(instance,) * count)
            calls = []
            def read(batch):
                calls.append(batch)
                return getter(batch)
            with (execution_debug(debug), patch.object(DenseGemmGroup, 'action_indices', property(read))):
                _, tables = cfg_runtime._bind_cfg_tables(plan, np)
            self.assertEqual(len(tables), 1)
            return len(calls), next(iter(tables.values()))

        # Warm the common numerical-program caches before comparing property
        # hooks. Metadata reuse never suppresses a debug or extension hook.
        for debug, extended in ((False, False), (True, False), (False, True)):
            bind(1, debug, extended)
        rng = np.random.default_rng(912)
        inputs = {key: rng.normal(size=spec.shape) + 1j*rng.normal(size=spec.shape)
                  for key, spec in scene.state.problem.specs.items()}
        for debug, extended in ((False, False), (True, False), (False, True)):
            with self.subTest(debug=debug, extended=extended):
                one, _ = bind(1, debug, extended)
                four, table = bind(4, debug, extended)
                self.assertEqual(four - one, 0 if not debug and not extended else 3 * groups)
                for scale in (1, -2j):
                    current = {key: value * scale for key, value in inputs.items()}
                    execution = SimpleNamespace(source=lambda key: None,
                        materialize=lambda key, source: current[key], materializes_owned=lambda key: False,
                        begin_action=lambda index, timed: None, end_action=lambda index, timed: None)
                    actual = table.program.run(execution, {})
                    expected = tuple(current[term.inputs[0]] @ current[term.inputs[1]] @ current[term.inputs[2]]
                                     for term in scene.state.problem.terms)
                    for value, reference in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)

    def test_multiple_mask_states_preserve_action_fields_identity_and_window_cache(self):
        from lqcdweave.contraction.execution.diagnostics import execution_debug

        class ExtendedAction(Action):
            pass

        class ExtendedKind(str):
            pass

        original = self.windows(1)[0][0]
        template, = original.templates
        actions = template.plan.state.actions
        costs = (0.0, -0.0, 2.5, float('nan'), float('inf'), -3.0)
        native = tuple(replace(action, reused=bool(index % 2),
                               seconds=costs[index % len(costs)])
                       for index, action in enumerate(actions))
        extended = (*native[:-1], ExtendedAction(**vars(native[-1])))
        other_cost = (replace(native[0], seconds=np.float64(2.5)), *native[1:])
        other_kind = (replace(native[0], kind=ExtendedKind(native[0].kind)), *native[1:])
        masks = (np.zeros(len(actions), dtype=bool), np.ones(len(actions), dtype=bool),
                 np.arange(len(actions)) % 2 == 0)
        self.assertGreaterEqual(len(actions), 6)
        for selected in (native, extended, other_cost, other_kind):
            with self.subTest(native=selected is native):
                state = replace(template.plan.state, actions=selected)
                current_template = replace(template,
                    plan=replace(template.plan, state=state, kernel_batches=()))
                plan = replace(original, templates=(current_template,), instances=tuple(
                    replace(original.instances[0], reused=ArrayConstant.from_array(mask),
                            kernel_batches=()) for mask in masks))
                with (execution_debug(True),
                      patch.object(cfg_runtime, 'bind_scene_execution', side_effect=lambda *a, **k: object()) as reference_bind):
                    expected, _ = cfg_runtime._bind_cfg_tables(plan, np)
                self.assertTrue(all(call.kwargs['_action_indices'] is None
                                    for call in reference_bind.call_args_list))
                cache = cfg_runtime._CfgTableCache(np)
                with (patch.object(cfg_runtime, '_native_reuse_flags',
                                   wraps=cfg_runtime._native_reuse_flags) as flags,
                      patch.object(cfg_runtime, 'bind_scene_execution', side_effect=lambda *a, **k: object()) as scene_bind):
                    actual, tables = cfg_runtime._bind_cfg_tables(plan, np, cache)
                    self.assertEqual(flags.call_count, 1)
                    for call in scene_bind.call_args_list:
                        indices = call.kwargs['_action_indices']
                        if selected is native:
                            self.assertEqual(list(indices), [index for index, action
                                in enumerate(call.args[0].actions) if not action.reused])
                        else:
                            self.assertIsNone(indices)
                    self.assertEqual(actual.keys(), expected.keys())
                    for key, current in actual.items():
                        self.assertEqual(replace(current, actions=()), replace(expected[key], actions=()))
                        for value, reference, base in zip(
                                current.actions, expected[key].actions, selected, strict=True):
                            self.assertEqual(replace(value, seconds=0.0), replace(reference, seconds=0.0))
                            self.assertIs(type(value.seconds), type(reference.seconds))
                            self.assertEqual(np.asarray(value.seconds).tobytes(),
                                             np.asarray(reference.seconds).tobytes())
                            self.assertEqual(value is base, reference is base)
                            if not value.reused:
                                self.assertIs(value.seconds, base.seconds)
                            else:
                                self.assertEqual(value.seconds, 0.0)
                    flags.reset_mock()
                    scene_bind.reset_mock()
                    next_states, next_tables = cfg_runtime._bind_cfg_tables(plan, np, cache)
                    flags.assert_not_called()
                    scene_bind.assert_not_called()
                    self.assertTrue(all(next_states[key] is value for key, value in actual.items()))
                    self.assertTrue(all(next_tables[key] is value for key, value in tables.items()))
                    self.assertEqual(cache.compiled_states, len(masks))
                    self.assertEqual(len(cache.entries[0][1]), len(masks))

    def test_multi_mask_native_replay_matches_debug_and_fresh_single_points(self):
        from correlator.execution.test_cfg_windows import CfgWindowExecutionTests
        fixture = CfgWindowExecutionTests()
        fixture.setUp()
        coordinates = ({0: 1, 1: 0}, {0: 1, 1: 0}, {0: 0, 1: 0}, {0: 1, 1: 0},
                       {0: 1, 1: 1}, {0: 0, 1: 0}, {0: 1, 1: 0}, {0: 1, 1: 1})
        points, _ = describe_coordinates(time_pairs=coordinates)
        args = (*fixture.args[:2], points.points, *fixture.args[3:])
        windows = list(iter_cfg_plan_windows(*args, window_size=len(coordinates)))
        self.assertEqual(len(windows), 1)
        self.assertGreaterEqual(len({instance.reused.data for instance in windows[0].instances}), 3)
        actual, checked, fresh = [], [], []
        with patch.object(cfg_runtime, '_native_reuse_flags', wraps=cfg_runtime._native_reuse_flags) as flags:
            execute_cfg_windows(iter(windows), 'a', fixture.factory,
                lambda cfg, index, values: actual.append((index, values)))
            self.assertEqual(flags.call_count, 1)
        with patch.object(cfg_runtime, '_native_reuse_flags', side_effect=AssertionError('debug snapshot')):
            execute_cfg_windows(iter(windows), 'a', fixture.factory,
                lambda cfg, index, values: checked.append((index, values)), debug=True)
        # Independent single-point plans cannot reuse values from another pair.
        for coordinate in coordinates:
            single, _ = describe_coordinates(time_pairs=[coordinate])
            plans = iter_cfg_plan_windows(*fixture.args[:2], single.points, *fixture.args[3:], window_size=1)
            execute_cfg_windows(plans, 'a', fixture.factory,
                lambda cfg, index, values: fresh.append(values) if index is not None else None, debug=True)
        self.assertEqual([index for index, _ in actual], list(range(8)) + [None])
        for (index, values), (other_index, references), singles in zip(
                actual[:-1], checked[:-1], fresh, strict=True):
            self.assertEqual(index, other_index)
            for value, reference, single in zip(values, references, singles, strict=True):
                np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)
                np.testing.assert_allclose(value, single, rtol=3e-12, atol=3e-12)

    def test_extended_or_mutable_instances_and_debug_keep_lookup_with_current_values(self):
        from lqcdweave.contraction.execution.diagnostics import execution_debug

        class ExtendedInstance(CfgInstance):
            pass

        windows, expected = self.windows(2)
        extended = [replace(plan, instances=tuple(ExtendedInstance(**vars(instance))
            for instance in plan.instances)) for plan in windows]
        mutable = [replace(plan, instances=tuple(replace(instance, events=list(instance.events))
            for instance in plan.instances)) for plan in windows]
        for plans, debug, direct in ((windows, False, True), (windows, True, False),
                                      (extended, False, False), (mutable, False, False)):
            with self.subTest(debug=debug, direct=direct, extended=plans is extended):
                cache = cfg_runtime._CfgTableCache(np)
                with execution_debug(debug):
                    for plan in plans:
                        cfg_runtime._bind_cfg_tables(plan, np, cache)
                        self.assertEqual(len(cache.instance_tables), len(plan.instances))
                        self.assertTrue(all((table is not None) == direct
                                            for table in cache.instance_tables))
                actual = []
                factory = source_loader_factory(CompleteCalculation(plans[0], options={'readers': {}}))
                execute_cfg_windows(iter(plans), 'cfg', factory,
                    lambda cfg, index, values: actual.append((index, values)), debug=debug)
                self.assertEqual(actual[-1], (None, None))
                self.assertEqual([index for index, _ in actual[:-1]], [0, 1])
                for index, values in actual[:-1]:
                    for value, reference in zip(values, expected, strict=True):
                        np.testing.assert_allclose(value, reference * (index + 1)**2,
                                                   rtol=2e-13, atol=2e-13)

    def test_equal_dense_programs_keep_distinct_current_input_and_output_bindings(self):
        actions = (Action('contract', 'x', ('a', 'b'), 'ij,jk->ik'),
                   Action('contract', 'y', ('c', 'd'), 'ij,jk->ik'))
        catalog = {key: TensorSpec(shape, 'complex128') for key, shape in (
            ('a', (2, 3)), ('b', (3, 2)), ('c', (2, 3)), ('d', (3, 2)))}
        state = SimpleNamespace(actions=actions, catalog=catalog, results=('x', 'y'))
        with patch.object(scene_runtime, 'bind_dense_action',
                          wraps=scene_runtime.bind_dense_action) as bind:
            tables = scene_runtime.bind_scene_execution(state, (), np, events=())
        self.assertEqual(bind.call_count, 1)
        self.assertIs(tables.kernels[0], tables.kernels[1])
        execution = SimpleNamespace(begin_action=lambda *args: None)
        for scale in (1, -2j):
            arrays = {'a': np.arange(6).reshape(2, 3) * scale + 1j,
                      'b': np.arange(6).reshape(3, 2) - 2j,
                      'c': np.arange(6, 12).reshape(2, 3) - 0.25j,
                      'd': np.arange(6, 12).reshape(3, 2) * scale + 3j}
            expected = tuple(np.einsum('ij,jk->ik', arrays[left], arrays[right], optimize=False)
                             for left, right in (('a', 'b'), ('c', 'd')))
            values = tables.program.run(execution, arrays)
            for value, reference in zip(values, expected, strict=True):
                np.testing.assert_array_equal(value, reference)
            self.assertFalse(np.shares_memory(arrays['x'], arrays['y']))
            self.assertFalse(np.shares_memory(*values))

    def test_dense_programs_separate_input_shape_dtype_and_equation(self):
        actions = (Action('contract', 'x', ('a', 'b'), 'ij,jk->ik'),
                   Action('contract', 'y', ('c', 'd'), 'ij,jk->ik'),
                   Action('contract', 'z', ('e', 'f'), 'ij,jk->ik'),
                   Action('contract', 'w', ('a', 'b'), 'ij,kj->ik'))
        catalog = {key: TensorSpec(shape, dtype) for key, shape, dtype in (
            ('a', (2, 3), 'complex128'), ('b', (3, 3), 'complex128'),
            ('c', (4, 3), 'complex128'), ('d', (3, 3), 'complex128'),
            ('e', (2, 3), 'complex64'), ('f', (3, 3), 'complex64'))}
        state = SimpleNamespace(actions=actions, catalog=catalog)
        with patch.object(scene_runtime, 'bind_dense_action',
                          wraps=scene_runtime.bind_dense_action) as bind:
            tables = scene_runtime.bind_scene_execution(state, (), np)
        self.assertEqual(bind.call_count, 4)
        self.assertEqual(len({id(kernel) for kernel in tables.kernels.values()}), 4)

    def test_changed_reuse_and_event_patterns_share_kernels_with_current_readers(self):
        from correlator.execution.test_cfg_windows import CfgWindowExecutionTests
        fixture = CfgWindowExecutionTests()
        fixture.setUp()
        coordinates = ({0: 1, 1: 0}, {0: 1, 1: 0}, {0: 0, 1: 0}, {0: 1, 1: 0},
                       {0: 1, 1: 1}, {0: 0, 1: 0}, {0: 1, 1: 0}, {0: 1, 1: 1})
        points, _ = describe_coordinates(time_pairs=coordinates)
        args = (*fixture.args[:2], points.points, *fixture.args[3:])
        windows = list(iter_cfg_plan_windows(*args, window_size=2))
        actual, checked = [], []
        with patch.object(scene_runtime, 'bind_dense_action',
                          wraps=scene_runtime.bind_dense_action) as bind:
            report = execute_cfg_windows(iter(windows), 'a', fixture.factory,
                lambda cfg, index, values: actual.append((index, values)))
        execute_cfg_windows(iter(windows), 'a', fixture.factory,
            lambda cfg, index, values: checked.append((index, values)), debug=True)
        self.assertGreater(report['compiled_execution_tables'], 1)
        # Each template contract is bound once despite changing resource events
        # and reuse masks. Debug uses the independent interpreted dispatch.
        contracts = sum(action.kind == 'contract' for action in windows[0].templates[0].plan.state.actions)
        self.assertEqual(bind.call_count, contracts)
        self.assertEqual([index for index, _ in actual], list(range(8)) + [None])
        for (index, values), (other_index, references) in zip(actual[:-1], checked[:-1], strict=True):
            self.assertEqual(index, other_index)
            for value, reference in zip(values, references, strict=True):
                np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)

    def windows(self, count=4):
        selected, preparations, specs = cfg_fixtures.CfgPlanTests.dense_fixture()
        points, _ = describe_coordinates(time_pairs=[{'source': i} for i in range(count)])
        plans = list(iter_cfg_plan_windows(selected, None, points.points, {}, preparations, specs,
                                          window_size=1))
        # Reconstructed (value-equal, identity-distinct) descriptors must work
        # too, as they do when the planning producer runs in another process.
        result = []
        for index, plan in enumerate(plans):
            template, = plan.templates
            batches = tuple(replace(batch) for batch in template.plan.kernel_batches)
            state = replace(template.plan.state, catalog=dict(template.plan.state.catalog))
            template = replace(template, plan=replace(template.plan, state=state,
                                                      kernel_batches=batches))
            preps = {key: replace(prep, recipe=replace(prep.recipe, constants=tuple(
                        ArrayConstant.from_array(value.array() * (index + 1))
                        for value in prep.recipe.constants)))
                     for key, prep in plan.preparations.items()}
            result.append(replace(plan, templates=(template,), preparations=preps,
                instances=tuple(replace(instance, kernel_batches=batches) for instance in plan.instances)))
        arrays = {key: prep.recipe.constants[0].array() for key, prep in preparations.items()}
        expected = tuple(arrays[term.inputs[0]] @ arrays[term.inputs[1]]
                         for term in selected.state.problem.terms)
        return result, expected

    def test_equivalent_windows_bind_once_and_use_current_numeric_constants(self):
        windows, expected = self.windows()
        outputs = []
        factory = source_loader_factory(CompleteCalculation(windows[0], options={'readers': {}}))
        with (patch.object(cfg_runtime, 'bind_scene_execution',
                           wraps=cfg_runtime.bind_scene_execution) as bind,
              patch.object(storage_runtime, 'validate_executable_resources',
                           side_effect=AssertionError('deep resource audit')),
              patch.object(saved_inputs, '_check_preparations',
                           side_effect=AssertionError('deep preparation audit'))):
            report = execute_cfg_windows(iter(windows), 'cfg', factory,
                lambda cfg, index, values: outputs.append((index, values)))
        self.assertEqual(bind.call_count, 1)
        self.assertEqual(report['compiled_template_variants'], 1)
        self.assertEqual(report['compiled_execution_tables'], 1)
        self.assertEqual(report['execution_table_cache_hits'], len(windows) - 1)
        self.assertEqual(outputs[-1], (None, None))
        for index, values in outputs[:-1]:
            for actual, reference in zip(values, expected, strict=True):
                np.testing.assert_allclose(actual, reference * (index + 1)**2,
                                           rtol=2e-13, atol=2e-13)

    def test_debug_keeps_deep_audits_and_same_independent_numerics(self):
        windows, expected = self.windows(2)
        factory = source_loader_factory(CompleteCalculation(windows[0],
            options={'readers': {}, 'debug': True}))
        outputs = []
        with (patch.object(storage_runtime, 'validate_executable_resources',
                           wraps=storage_runtime.validate_executable_resources) as resources,
              patch.object(saved_inputs, '_check_preparations',
                           wraps=saved_inputs._check_preparations) as preparations):
            execute_cfg_windows(iter(windows), 'cfg', factory,
                lambda cfg, index, values: outputs.append((index, values)), debug=True)
        self.assertEqual(resources.call_count, len(windows))
        self.assertEqual(preparations.call_count, len(windows))
        for index, values in outputs[:-1]:
            for actual, reference in zip(values, expected, strict=True):
                np.testing.assert_allclose(actual, reference * (index + 1)**2,
                                           rtol=2e-13, atol=2e-13)

    def test_shape_dtype_action_and_full_batch_changes_invalidate_tables(self):
        original = self.windows(1)[0][0]
        template, = original.templates
        state = template.plan.state
        tensor = state.actions[0].tensor
        spec = state.catalog[tensor]
        variants = []
        for changed in (TensorSpec((*spec.shape[:-1], spec.shape[-1] + 1), spec.dtype),
                        TensorSpec(spec.shape, 'complex64')):
            variants.append(('shape/dtype', replace(template.plan,
                state=replace(state, catalog=dict(state.catalog, **{tensor: changed})))))
        variants.append(('action', replace(template.plan, state=replace(state,
            actions=(replace(state.actions[0], seconds=state.actions[0].seconds + 1), *state.actions[1:])))))
        batch, *rest = template.plan.kernel_batches
        variants.append(('batch layout', replace(template.plan, kernel_batches=(replace(batch,
            packing_bytes=batch.packing_bytes + 16), *rest))))
        for label, variant in variants:
            with self.subTest(variant=label):
                cache = cfg_runtime._CfgTableCache(np)
                changed = replace(original, templates=(replace(template, plan=variant),),
                    instances=tuple(replace(instance, kernel_batches=variant.kernel_batches)
                                    for instance in original.instances))
                with patch.object(cfg_runtime, 'bind_scene_execution', side_effect=lambda *a, **k: object()) as bind:
                    cfg_runtime._bind_cfg_tables(original, np, cache)
                    cfg_runtime._bind_cfg_tables(changed, np, cache)
                self.assertEqual(bind.call_count, 2)
                self.assertIsNot(bind.call_args_list[0].kwargs['kernel_cache'],
                                  bind.call_args_list[1].kwargs['kernel_cache'])
                self.assertEqual(cache.table_hits, 0)
                self.assertEqual(len(cache.entries), 1)
                self.assertEqual(len(cache.entries[0][2]), 1)

    def test_workflow_window_factory_reuses_reader_checks_but_refreshes_headers(self):
        plan, expanded, selected, elemental, propagator = runtime_fixtures.CfgReplayTests().setup_plan()
        points, _ = describe_coordinates(time_pairs=[{0: 1, 1: 0}, {0: 0, 1: 1}])
        windows = list(iter_cfg_plan_windows(selected, plan.templates[0].output, points.points,
            saved_inputs.describe_input_reads(expanded.inputs), expanded.inputs.loader.preparations,
            expanded.inputs.loader.specs, window_size=1))
        factory = source_loader_factory(CompleteCalculation(plan,
            options={'readers': expanded.inputs.loader.input_readers}))
        with (patch.object(saved_inputs, '_check_conventions', wraps=saved_inputs._check_conventions) as check,
              patch.object(elemental, 'describe_block', wraps=elemental.describe_block) as describe_elemental,
              patch.object(propagator, 'describe_block', wraps=propagator.describe_block) as describe_propagator):
            report = execute_cfg_windows(iter(windows), 'cfg', factory, lambda *args: None)
        self.assertEqual(check.call_count, 2)
        described = sum(1 + len(request.aliases) for window in windows for request in window.requests.values())
        self.assertEqual(describe_elemental.call_count + describe_propagator.call_count, described)
        self.assertEqual(report['compiled_execution_tables'], 1)
        self.assertEqual(report['execution_table_cache_hits'], 1)

    def test_source_metadata_snapshot_and_replacement_cannot_leave_stale_tables(self):
        original = self.windows(1)[0][0]
        template, = original.templates
        state = template.plan.state
        key = next(iter(state.problem.sources))
        mutable = {'convention': {'order': [1, 0]}}
        source = replace(state.problem.sources[key], metadata=mutable)
        mutable['convention']['order'][0] = 7
        self.assertEqual(source.metadata['convention']['order'], (1, 0))
        with self.assertRaises(TypeError):
            source.metadata['convention']['order'][0] = 8
        def changed_plan(source):
            problem = replace(state.problem, sources=dict(state.problem.sources, **{key: source}))
            updated = replace(template, plan=replace(template.plan, state=replace(state, problem=problem)))
            return replace(original, templates=(updated,))
        cache = cfg_runtime._CfgTableCache(np)
        with patch.object(cfg_runtime, 'bind_scene_execution', side_effect=lambda *a, **k: object()) as bind:
            cfg_runtime._bind_cfg_tables(changed_plan(source), np, cache)
            cfg_runtime._bind_cfg_tables(changed_plan(replace(source,
                metadata={'convention': {'order': [0, 1]}})), np, cache)
        self.assertEqual(bind.call_count, 2)
        self.assertIsNot(bind.call_args_list[0].kwargs['kernel_cache'],
                          bind.call_args_list[1].kwargs['kernel_cache'])
        self.assertEqual(cache.table_hits, 0)

    def test_cache_is_execution_local_and_rejects_another_backend_object(self):
        window = self.windows(1)[0][0]
        first = cfg_runtime._CfgTableCache(np)
        second = cfg_runtime._CfgTableCache(np)
        with patch.object(cfg_runtime, 'bind_scene_execution', side_effect=lambda *a, **k: object()) as bind:
            cfg_runtime._bind_cfg_tables(window, np, first)
            cfg_runtime._bind_cfg_tables(window, np, second)
        self.assertEqual(bind.call_count, 2)
        with self.assertRaisesRegex(ValueError, 'another backend'):
            cfg_runtime._bind_cfg_tables(window, object(), first)


if __name__ == '__main__':
    unittest.main()
