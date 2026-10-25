"""Fixed calls preserve values, lifecycle order and diagnostic isolation."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution import bound_program, scene_runtime
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.storage_runtime import _BlockExecution
from lqcdweave.contracts.correlator.api import Action, DenseGemmGroup, StorageEvent
from correlator.execution.test_debug_execution import _make_dense_execution
from correlator.planning import test_dense_batches as dense_fixtures


class BoundProgramTests(unittest.TestCase):
    def test_debug_ignores_private_native_action_indices(self):
        _, path, _, _ = _make_dense_execution()
        plan = path.plan
        expected = scene_runtime.bind_scene_execution(plan.state, plan.kernel_batches, np,
            events=plan.events, retained=plan.retained_after)
        with execution_debug(True):
            checked = scene_runtime.bind_scene_execution(plan.state, plan.kernel_batches, np,
                events=plan.events, retained=plan.retained_after, _action_indices=())
        self.assertTrue(expected.kernels)
        self.assertEqual(checked.kernels.keys(), expected.kernels.keys())
        self.assertEqual(checked.program.preparation_keys, expected.program.preparation_keys)
        self.assertEqual(len(checked.program.calls), len(expected.program.calls))

    def test_shared_group_binding_keeps_chained_results_and_intermediate_releases(self):
        from correlator.planning.test_gemm_groups import fixture
        from lqcdweave.contraction.planning.api import lower_independent_dense_batches

        plan, profile = fixture(chained=True)
        plan, _ = lower_independent_dense_batches(plan, profile=profile, target_runtime='cuda')
        self.assertTrue(any(isinstance(batch, DenseGemmGroup) for batch in plan.kernel_batches))
        tables = scene_runtime.bind_scene_execution(plan.state, plan.kernel_batches, np,
            events=plan.events, shared_residency=True)
        for seed in (512, 513):
            rng = np.random.default_rng(seed)
            inputs = {key: rng.normal(size=spec.shape) + 1j*rng.normal(size=spec.shape)
                      for key, spec in plan.state.problem.specs.items()}
            execution = SimpleNamespace(source=lambda key: None,
                materialize=lambda key, source: inputs[key], materializes_owned=lambda key: False,
                begin_action=lambda index, timed: None, end_action=lambda index, timed: None)
            arrays = {}
            actual = tables.program.run(execution, arrays)
            expected = tuple(inputs[term.inputs[0]] @ inputs[term.inputs[1]] @ inputs[term.inputs[2]]
                             for term in plan.state.problem.terms)
            for value, reference in zip(actual, expected, strict=True):
                np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)
            released = {action.tensor for action in plan.state.actions if action.kind == 'release'}
            self.assertTrue(released)
            self.assertFalse(released.intersection(arrays))

    def test_shared_binding_keeps_events_on_reused_actions_and_final_snapshot(self):
        _, path, loader, _ = _make_dense_execution()
        action = next(action for action in path.plan.state.actions if action.kind == 'load')
        key, value = action.tensor, loader.arrays[action.tensor]

        class ExtendedAction(Action):
            pass

        actions = tuple(replace(action, reused=True) for _ in range(6))
        events = (StorageEvent(2, 'read', 'input'), StorageEvent(2, 'release', 'input'),
                  StorageEvent(4, 'release', 'other'), StorageEvent(6, 'release', 'final'))
        for extended, retained, mutable in ((False, (), False), (True, (), False),
                                             (False, (key,), False), (False, (), True)):
            with self.subTest(extended=extended, retained=retained, mutable_events=mutable):
                selected = tuple(ExtendedAction(**vars(a)) for a in actions) if extended else actions
                state = replace(path.plan.state, actions=selected, results=(key,))
                current_events = list(events) if mutable else events
                recorded = []
                def snapshot(array, **kwargs):
                    recorded.append(('copy', key))
                    return np.array(array, **kwargs)
                backend = SimpleNamespace(__name__='numpy', array=snapshot)
                execution = SimpleNamespace(
                    begin_action=lambda index, timed: recorded.append(('before', index)),
                    end_action=lambda index, timed: recorded.append(('after', index)),
                    retain_tensor=lambda name, array: recorded.append(('retain', name)))
                tables = scene_runtime.bind_scene_execution(state, (), backend, events=current_events,
                    retained=retained, shared_residency=True)
                result, = tables.program.run(execution, {key: value})
                expected = [('before', 0)]
                for index in range(6):
                    if index == 2:
                        expected.append(('before', 2))
                    if retained:
                        expected.append(('retain', key))
                    if index in (2, 4):
                        expected.append(('after', index))
                expected.extend((('copy', key), ('after', 6)))
                self.assertEqual(recorded, expected)
                np.testing.assert_array_equal(result, value)
                self.assertFalse(np.shares_memory(result, value))

    def test_builtin_dense_call_skips_numerical_forwarder(self):
        _, path, loader, expected = _make_dense_execution()
        with patch.object(bound_program, '_numerical',
                          side_effect=AssertionError('builtin dense call used _numerical')):
            actual = replay_scene_storage(path.plan, loader, io_overlap=False)
        np.testing.assert_allclose(actual[0], expected, rtol=2e-12, atol=2e-12)

    def test_bound_default_preserves_batch_outputs_without_interpreter_or_empty_hooks(self):
        fixture = dense_fixtures.DenseBatchIntegrationTests()
        plan = fixture.plan().plan
        arrays = fixture.arrays(plan)
        with (patch.object(scene_runtime, '_replay', side_effect=AssertionError('interpreted')),
              patch.object(_BlockExecution, 'end_action', side_effect=AssertionError('empty hook'))):
            actual = replay_scene_storage(plan, dense_fixtures.ArrayLoader(arrays))
        with execution_debug(True):
            audited = replay_scene_storage(plan, dense_fixtures.ArrayLoader(arrays))
        measured = replay_scene_storage(plan, dense_fixtures.ArrayLoader(arrays), measure=True)
        for term, value, checked, sampled in zip(plan.state.problem.terms, actual,
                                                audited, measured.values, strict=True):
            np.testing.assert_allclose(value, arrays[term.inputs[0]] @ arrays[term.inputs[1]])
            np.testing.assert_array_equal(value, checked)
            np.testing.assert_array_equal(value, sampled)
        self.assertEqual(len(measured.timings), len(plan.state.actions))

    def test_bound_program_rebinds_arrays_and_preserves_physical_event_order(self):
        _, path, loader, expected = _make_dense_execution()
        plan = path.plan
        tables = scene_runtime.bind_scene_execution(plan.state, plan.kernel_batches, np,
                                                    events=plan.events, retained=plan.retained_after)
        self.assertEqual(tables.program.preparation_keys, tuple(
            action.tensor for action in plan.state.actions
            if action.kind == 'load' and not action.reused))
        outputs = []
        for scale in (1, 2):
            if scale == 2:
                loader.arrays = {key: value * 2 for key, value in loader.arrays.items()}
            execution = _BlockExecution(plan, loader, np, execution_tables=tables, io_overlap=False)
            read_indices, release_indices = [], []
            begin, end = execution.begin_action, execution.end_action
            def record_begin(index, timed):
                read_indices.append(index)
                return begin(index, timed)
            def record_end(index, timed):
                release_indices.append(index)
                return end(index, timed)
            try:
                with (patch.object(execution, 'begin_action', record_begin),
                      patch.object(execution, 'end_action', record_end)):
                    outputs.append(tables.program.run(execution, {})[0])
                self.assertEqual(read_indices, sorted({0} | {
                    event.action_index for event in plan.events if event.kind == 'read'}))
                self.assertEqual(release_indices, sorted({
                    event.action_index for event in plan.events if event.kind != 'read'}))
                self.assertIsNone(execution.costs)
            finally:
                execution.close()
        np.testing.assert_allclose(outputs[0], expected)
        np.testing.assert_allclose(outputs[1], 4 * expected)

    def test_default_keeps_external_input_specification_check(self):
        _, path, loader, _ = _make_dense_execution()
        loader.arrays['left'] = loader.arrays['left'][:1]
        with self.assertRaisesRegex(ValueError, 'mismatched tensor'):
            replay_scene_storage(path.plan, loader, io_overlap=False)

    def test_shared_reuse_preserves_events_and_ordinary_retained_cache_restoration(self):
        _,path,loader,_=_make_dense_execution()
        original=path.plan.state
        action=next(action for action in original.actions if action.kind=='load')
        key=action.tensor
        state=replace(original,actions=(replace(action,reused=True),),results=(key,))
        value=loader.arrays[key]
        events=(StorageEvent(0,'read','block'),StorageEvent(0,'release','block'),
                StorageEvent(1,'release','final'))
        for shared in (False,True):
            with self.subTest(shared=shared):
                calls=[]
                def cache_value(key):
                    calls.append(('restore',key))
                    return value
                execution=SimpleNamespace(
                    begin_action=lambda index,timed:calls.append(('before',index)),
                    end_action=lambda index,timed:calls.append(('after',index)),
                    has_retained=lambda name:name==key,retained_value=cache_value)
                tables=scene_runtime.bind_scene_execution(state,(),np,events=events,
                                                         shared_residency=shared)
                self.assertEqual(tables.program.preparation_keys, ())
                arrays={key:value} if shared else {}
                result,=tables.program.run(execution,arrays)
                expected=[('before',0)]
                if not shared: expected.append(('restore',key))
                self.assertEqual(calls,[*expected,('after',0),('after',1)])
                np.testing.assert_array_equal(result,value)
                self.assertFalse(np.shares_memory(result,value))


if __name__ == '__main__':
    unittest.main()
