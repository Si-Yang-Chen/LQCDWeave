"""Shared programs consume fresh data and preserve each entry's ownership."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import gc
import weakref
import numpy as np

from lqcdweave.contraction.execution import storage_runtime, scene_runtime, executor
from lqcdweave.contraction.execution.api import ExecutionBindingCache, replay, replay_scene_storage
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.results import binding_cache as outputs
from lqcdweave.contracts.correlator.api import ContractionTerm
from lqcdweave.contraction.correlator import save_calculation_plan, load_calculation_plan
from lqcdweave.contraction.contraction_session import ContractionSession


class CommonExecutionBindingsTests(unittest.TestCase):
    def calculation(self, **options):
        from test_kernel_batching import KernelBatchingTests
        return KernelBatchingTests().native(**options)

    def test_fixed_calculation_reuses_both_tables_and_reads_current_input(self):
        calculation = self.calculation()
        reader = calculation.expanded.bindings.propagator_map['S^q']
        with patch.object(storage_runtime, 'bind_scene_execution', wraps=storage_runtime.bind_scene_execution) as bind, \
             patch.object(outputs, 'bind_output_assembly', wraps=outputs.bind_output_assembly) as assembly:
            first = calculation.run()
            count = bind.call_count
            self.assertGreater(count, 0)
            self.assertEqual(assembly.call_count, 1)
            reader.sectors['vsv'] *= 2
            second = calculation.run()
            self.assertEqual(bind.call_count, count)
            self.assertEqual(assembly.call_count, 1)
            # The meson two-point function has two propagator factors.
            np.testing.assert_allclose(second.value, 4 * first.value, rtol=3e-12, atol=3e-12)
            self.assertFalse(np.shares_memory(first.value, second.value))
            checked = calculation.run(debug=True)
            np.testing.assert_allclose(checked.value, second.value, rtol=3e-12, atol=3e-12)
            self.assertGreater(bind.call_count, count)
            count = bind.call_count
            np.testing.assert_array_equal(first.run().value, second.value)
            self.assertEqual(bind.call_count, count)

    def test_coordinate_batches_reuse_each_selected_program(self):
        calculation = self.calculation(time_axes={0: 1, 1: (0, 1)}, batch=1)
        self.assertGreater(len(calculation.path.batch_paths), 1)
        first = calculation.run()
        with patch.object(storage_runtime, 'bind_scene_execution', side_effect=AssertionError('rebound')):
            second = calculation.run()
        np.testing.assert_array_equal(first.value, second.value)
        np.testing.assert_allclose(calculation.run(debug=True).value, first.value, rtol=3e-12, atol=3e-12)

    def test_replaced_path_archive_and_concurrent_owners(self):
        calculation = self.calculation()
        reference = calculation.run().value.copy()
        with patch.object(storage_runtime, 'bind_scene_execution', side_effect=AssertionError('rebound')):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: calculation.run().value, range(2)))
        for value in results:
            np.testing.assert_array_equal(value, reference)
        self.assertFalse(np.shares_memory(*results))
        new_plan = replace(calculation.plan)
        changed = replace(calculation, path=replace(calculation.path, plan=new_plan, plan_identity=id(new_plan)))
        with patch.object(storage_runtime, 'bind_scene_execution', wraps=storage_runtime.bind_scene_execution) as bind:
            np.testing.assert_array_equal(changed.run().value, reference)
            self.assertGreater(bind.call_count, 0)
        with TemporaryDirectory() as directory:
            archive = Path(directory) / 'fixed.json.gz'
            save_calculation_plan(archive, calculation)
            restored = load_calculation_plan(archive, expected=calculation)
            self.assertIsNone(restored._execution_bindings._binding)
            np.testing.assert_array_equal(restored.run().value, reference)
            with patch.object(storage_runtime, 'bind_scene_execution', side_effect=AssertionError('rebound')):
                np.testing.assert_array_equal(restored.run().value, reference)

    def test_execution_cache_does_not_retain_current_binding_envelope_or_arrays(self):
        from correlator.execution.test_debug_execution import _make_dense_execution
        from lqcdweave.contraction.execution.api import execute_contraction_path
        expanded, path, loader, _ = _make_dense_execution()
        owners = [weakref.ref(loader), *(weakref.ref(array) for array in loader.arrays.values())]
        cache = ExecutionBindingCache()
        execute_contraction_path(expanded, path, execution_cache=cache)
        self.assertIsNotNone(cache._binding)
        del expanded, path, loader
        gc.collect()
        self.assertTrue(all(owner() is None for owner in owners))

    def test_rejected_extension_state_does_not_publish_a_runtime_owner(self):
        from correlator.execution.test_debug_execution import _make_dense_execution
        expanded, path, loader, _ = _make_dense_execution()
        extension = SimpleNamespace(**vars(path.plan.state), done=True, current_reader=loader)
        owner = weakref.ref(loader)
        cache = ExecutionBindingCache()
        replay(extension, lambda key, source: loader.arrays[key], execution_cache=cache)
        self.assertIsNone(cache._binding)
        del expanded, path, extension, loader
        gc.collect()
        self.assertIsNone(owner())

    def test_low_level_storage_and_bare_replay_share_cache_policy(self):
        from correlator.execution.test_debug_execution import _make_dense_execution
        _, path, loader, expected = _make_dense_execution()
        cache = ExecutionBindingCache()
        for run, module in ((lambda: replay_scene_storage(path.plan, loader, execution_cache=cache), storage_runtime),
                            (lambda: replay(path.plan.state, lambda key, source: loader.arrays[key],
                                           execution_cache=cache), scene_runtime)):
            with patch.object(module, 'bind_scene_execution', wraps=module.bind_scene_execution) as bind:
                first, = run()
                calls = bind.call_count
                second, = run()
                self.assertEqual(bind.call_count, calls)
                self.assertGreater(calls, 0)
            np.testing.assert_allclose(first, expected, rtol=3e-12, atol=3e-12)
            np.testing.assert_array_equal(first, second)

    def test_mutable_events_and_failed_binding_never_publish_a_table(self):
        from correlator.execution.test_debug_execution import _make_dense_execution
        _, path, loader, _ = _make_dense_execution()
        cache = ExecutionBindingCache()
        plan = replace(path.plan, events=list(path.plan.events))
        with patch.object(storage_runtime, 'bind_scene_execution', wraps=storage_runtime.bind_scene_execution) as bind:
            for _ in range(2):
                replay_scene_storage(plan, loader, execution_cache=cache)
            self.assertEqual(bind.call_count, 2)
        with patch.object(storage_runtime, 'bind_scene_execution', side_effect=RuntimeError('bind failure')):
            with self.assertRaisesRegex(RuntimeError, 'bind failure'):
                replay_scene_storage(path.plan, loader, execution_cache=cache)
        self.assertIsNone(cache._binding)
        replay_scene_storage(path.plan, loader, execution_cache=cache)
        with patch.object(storage_runtime, 'bind_scene_execution', side_effect=AssertionError('rebound')):
            replay_scene_storage(path.plan, loader, execution_cache=cache)

    def test_external_source_execution_does_not_borrow_bare_state_tables(self):
        from correlator.execution.test_debug_execution import _make_dense_execution
        _, path, loader, expected = _make_dense_execution()
        cache = ExecutionBindingCache()
        replay(path.plan.state, lambda key, source: loader.arrays[key], execution_cache=cache)
        execution = storage_runtime._BlockExecution(path.plan, loader, np)
        execution.execution_tables = None
        try:
            with patch.object(scene_runtime, 'bind_scene_execution', wraps=scene_runtime.bind_scene_execution) as bind:
                value, = replay(path.plan.state, loader, source_execution=execution, execution_cache=cache)
            self.assertEqual(bind.call_count, 1)
            np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-12)
        finally:
            execution.close()

    def test_dynamic_session_normalizes_precision_and_reuses_lifetimes(self):
        session = ContractionSession()
        left = np.arange(6, dtype=np.float64).reshape(2, 3)
        right = (1j + np.arange(12).reshape(3, 4)).astype(np.complex64)
        original = left.copy(), right.copy()
        term = ContractionTerm(('left', 'right'), 'ij,jk->ik')
        with patch.object(executor, '_bind_use_counts', wraps=executor._bind_use_counts) as bind:
            first = session.tensors(dict(left=left, right=right), (term,))
            second = session.tensors(dict(left=2*left, right=right), (term,))
            self.assertEqual(bind.call_count, 1)
            with execution_debug(True):
                checked = session.tensors(dict(left=left, right=right), (term,))
            self.assertEqual(bind.call_count, 2)
            session.clear()
            session.tensors(dict(left=left, right=right), (term,))
            self.assertEqual(bind.call_count, 3)
        self.assertEqual(first.values[0].dtype, np.dtype('complex64'))
        expected = np.einsum('ij,jk->ik', left.astype(np.complex64), right, optimize=False)
        np.testing.assert_array_equal(first.values[0], expected)
        np.testing.assert_array_equal(second.values[0], 2*expected)
        np.testing.assert_array_equal(checked.values[0], expected)
        for current, reference in zip((left, right), original):
            np.testing.assert_array_equal(current, reference)

    def test_native_baryon_core_reuses_tables_and_matches_symbolic_reference(self):
        from test_structured_correlator import (
            ArrayBaryonElemental, BaryonComponent, Operator, OperatorTerm, SparseSpinTensor,
            Baryon, ArrayPropagator, HadronIrrep, Tag, Hadron, Flavor, gen_correlator, calculate, CostProfile,
        )
        rng = np.random.default_rng(181)
        def random(shape): return rng.normal(size=shape) + 1j*rng.normal(size=shape)
        component = BaryonComponent.paths()
        reader = ArrayBaryonElemental({'vvv': random((1, 1, 2, 1, 1, 1))}, (component,), cfg='cfg')
        operator = Operator.from_terms('baryon', (
            OperatorTerm.from_baryon(component, SparseSpinTensor(((0, 1, 2, 1),)), coefficient=.4+.3j),
            OperatorTerm.from_baryon(component, SparseSpinTensor(((3, 2, 1, .5j),)), coefficient=-.2j),
        ))
        vertices = {False: Baryon(reader, operator), True: Baryon(reader, operator, True)}
        propagator = ArrayPropagator({'vsv': random((2, 2, 4, 4, 1, 1))}, time_convention='absolute', cfg='cfg')
        row = HadronIrrep('N', (0, 0, 0), 'G_1', 1, Tag(0, 0))[0]
        axis = [Hadron(row, Flavor('uds'))]
        options = dict(vertex_map=lambda row: vertices[row.dagger],
            propagator_map={'S^q': propagator, 'S^s': propagator}, configuration='cfg', usedNe=1,
            time_axes={0: 1, 1: 0}, memory_limit=1 << 28, profile=CostProfile(), baryon_vertices='joined')
        reference = calculate(gen_correlator([axis.copy(), axis.copy()]), optimize='none', **options).value
        selected = calculate(gen_correlator([axis.copy(), axis.copy()], structured=True), execute=False,
            search=dict(component_workers=1, root_width=1, branch_width=1, deadline_seconds=2), **options)
        self.assertTrue(any(action.kind == 'baryon_join' and action.core is not None
                            for action in selected.plan.state.actions))
        first = selected.run().value
        propagator.sectors['vsv'] *= 2
        with patch.object(storage_runtime, 'bind_scene_execution', side_effect=AssertionError('rebound')):
            second = selected.run().value
        np.testing.assert_allclose(first, reference, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(second, 8*reference, rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
