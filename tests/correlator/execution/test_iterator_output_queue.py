"""Direct/W iterator delivery and actual optional CuPy worker lifecycles.

Protocol doubles prove dependencies and bounded ownership, not GPU timing.
The real-CuPy class runs the same iterator and result worker on an actual device.
"""
from contextlib import contextmanager
from dataclasses import replace
import gc
import threading
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contraction.execution import cfg_runtime
from lqcdweave.contraction.execution.cfg_runtime import (
    execute_cfg_plan, iter_execute_cfg_plan,
)
from lqcdweave.contraction.execution.device import active_device_execution
from lqcdweave.contraction.execution.output_queue import (
    OutputConsumerQueue, active_output_queue,
)
from lqcdweave.contraction.correlator_cfg import (
    iter_memory_calculation, source_loader_factory,
)
from correlator.execution.test_device import _Tracker, _targets, _factory, _assert_tree_allclose
from correlator.execution.test_output_queue import _OutputCupy, _plan


def _calculation(selected, complete, backend, closed, cfgs=('a', 'b'), **options):
    tracker = getattr(backend, 'tracker', None) or _Tracker()
    return replace(complete, options=dict(complete.options,
        configuration=cfgs, backend=backend, readers=None,
        source_factory=_factory(complete, selected, tracker, closed), **options))


def _events(calculation, backend, report, *, transform=None, **options):
    return iter_execute_cfg_plan(calculation.plan, calculation.options['configuration'],
        source_loader_factory(calculation), backend=backend, report=report,
        transform=transform, **options)


def _no_workers(test):
    test.assertFalse(any(t.name.startswith(('contraction-results', 'contraction-io'))
                        for t in threading.enumerate()))
    test.assertIsNone(active_output_queue())
    test.assertIsNone(active_device_execution())


class IteratorOutputQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu_selected, cls.cpu_complete = _plan(_targets()[0])
        cls.gpu_selected, cls.gpu_complete = _plan(_targets()[1])

    def test_direct_results_are_ordered_complete_and_survive_later_cfgs(self):
        cpu_closed = []
        cpu = _calculation(self.cpu_selected, self.cpu_complete, np, cpu_closed)
        expected = list(_events(cpu, np, {}))
        backend = _OutputCupy(); closed = []; report = {}
        gpu = _calculation(self.gpu_selected, self.gpu_complete, backend, closed)
        actual = []
        iterator = _events(gpu, backend, report)
        first = next(iterator)
        snapshot = tuple(value.copy() for value in first[2])
        self.assertIsNone(active_output_queue(), 'private queue leaked into caller')
        self.assertIs(type(first[2]), tuple)
        self.assertTrue(all(type(value) is np.ndarray for value in first[2]))
        actual.append(first)
        actual.extend(iterator)
        self.assertEqual([(cfg, index) for cfg, index, _ in actual],
                         [(cfg, index) for cfg in ('a', 'b') for index in (0, 1, 2, None)])
        self.assertEqual(closed, ['a', 'b'])
        self.assertEqual(report['completed'], ('a', 'b'))
        self.assertEqual(report['output_queue_tasks'], 6)
        self.assertLessEqual(report['output_queue_peak_tasks'], 2)
        self.assertLessEqual(report['device']['peak_pending_outputs'], 2)
        for (_, index, result), (_, _, truth) in zip(actual, expected, strict=True):
            if index is None:
                self.assertIsNone(result)
            else:
                _assert_tree_allclose(result, truth)
        for value, truth in zip(first[2], snapshot, strict=True):
            np.testing.assert_array_equal(value, truth)
        _no_workers(self)

    def test_transform_overlaps_second_replay_but_third_waits_for_delivery(self):
        started = threading.Event(); second_replay = threading.Event()
        replayed = []; transformed = []; delivered = []; queue_ids = []
        original = cfg_runtime.replay
        def replay(*args, **kwargs):
            replayed.append(len(replayed))
            queue_ids.append(id(active_output_queue()))
            if len(replayed) == 2:
                self.assertTrue(started.wait(5), 'first transform did not start')
                second_replay.set()
            if len(replayed) == 3:
                self.assertEqual(delivered, [0], 'third replay preceded first delivery')
            return original(*args, **kwargs)
        def transform(cfg, index, values):
            self.assertIsNone(active_output_queue())
            self.assertIsNone(active_device_execution())
            self.assertTrue(threading.current_thread().name.startswith('contraction-results'))
            transformed.append(index)
            if index == 0:
                started.set()
                self.assertTrue(second_replay.wait(5), 'no replay/transform overlap')
            return values[0].copy()
        backend = _OutputCupy(); closed = []; report = {}
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed, ('a',))
        with patch.object(cfg_runtime, 'replay', replay):
            iterator = _events(calculation, backend, report, transform=transform)
            first = next(iterator)
            self.assertEqual(first[1], 0)
            self.assertEqual(replayed, [0, 1])
            self.assertEqual(len(set(queue_ids)), 1)
            delivered.append(first[1])
            rest = list(iterator)
        self.assertEqual([entry[1] for entry in rest], [1, 2, None])
        self.assertEqual(transformed, [0, 1, 2])
        self.assertEqual(closed, ['a'])
        _no_workers(self)

    def test_lazy_configuration_and_reader_scope_stay_on_iterator_thread(self):
        owner = threading.get_ident(); claims = []; factory_threads = []
        class Configurations:
            def __iter__(self):
                self.assert_owner()
                return self
            def assert_owner(self):
                if threading.get_ident() != owner:
                    raise AssertionError('configuration iterator advanced on worker')
            def __next__(self):
                self.assert_owner()
                if len(claims) == 2:
                    raise StopIteration
                cfg = ('a', 'b')[len(claims)]
                claims.append(cfg)
                return cfg
        backend = _OutputCupy(); closed = []; report = {}
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed,
                                   Configurations())
        factory = source_loader_factory(calculation)
        @contextmanager
        def traced_factory(cfg):
            factory_threads.append(threading.get_ident())
            with factory(cfg) as loader:
                yield loader
            factory_threads.append(threading.get_ident())
        iterator = iter_execute_cfg_plan(calculation.plan, calculation.options['configuration'],
                                        traced_factory, backend=backend, report=report)
        self.assertEqual(claims, [])
        list(iterator)
        self.assertEqual(claims, ['a', 'b'])
        self.assertEqual(factory_threads, [owner] * 4)
        self.assertEqual(closed, ['a', 'b'])
        _no_workers(self)

    def test_early_close_joins_slow_transform_before_readers_and_device(self):
        second_started = threading.Event(); release_second = threading.Event()
        close_started = threading.Event(); callback_finished = threading.Event()
        closed = []; transformed = []; close_observations = []; replays = []
        backend = _OutputCupy()
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed, ('a',))
        factory = source_loader_factory(calculation)
        @contextmanager
        def observed_factory(cfg):
            with factory(cfg) as loader:
                yield loader
            close_observations.append(callback_finished.is_set())
        original = cfg_runtime.replay
        def replay(*args, **kwargs):
            replays.append(1)
            return original(*args, **kwargs)
        def transform(cfg, index, values):
            transformed.append(index)
            if index == 1:
                second_started.set()
                self.assertTrue(release_second.wait(5), 'close did not release worker')
                self.assertFalse(closed, 'reader closed while worker still ran')
                self.assertFalse(backend.tracker.pools[0].freed)
                callback_finished.set()
            return values
        def releaser():
            if close_started.wait(5):
                release_second.set()
        thread = threading.Thread(target=releaser)
        try:
            with patch.object(cfg_runtime, 'replay', replay):
                iterator = iter_execute_cfg_plan(calculation.plan, ('a',), observed_factory,
                                                 backend=backend, transform=transform)
                first = next(iterator)
                self.assertTrue(second_started.wait(5))
                thread.start()
                close_started.set()
                iterator.close()
        finally:
            release_second.set()
            if thread.ident is not None:
                thread.join(5)
        self.assertEqual(len(replays), 2)
        self.assertEqual(transformed, [0, 1])
        self.assertEqual(closed, ['a'])
        self.assertEqual(close_observations, [True])
        self.assertTrue(backend.tracker.pools[0].freed)
        self.assertTrue(all(np.isfinite(value).all() for value in first[2]))
        _no_workers(self)

    def test_transform_failure_closes_all_readers_without_completion(self):
        backend = _OutputCupy(); closed = []; calls = []; delivered = []
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed)
        def transform(cfg, index, values):
            calls.append((cfg, index))
            raise RuntimeError('iterator transform failed')
        with self.assertRaisesRegex(RuntimeError, 'iterator transform failed'):
            for row in _events(calculation, backend, {}, transform=transform):
                delivered.append(row)
        self.assertEqual(calls, [('a', 0)])
        self.assertFalse(any(index is None for _, index, _ in delivered))
        self.assertIn('a', closed)
        self.assertEqual(len(closed), len(set(closed)))
        self.assertTrue(backend.tracker.pools[0].freed)
        _no_workers(self)

    def test_cpu_tight_and_debug_transforms_remain_synchronous_and_single_point_lazy(self):
        for mode in ('cpu', 'tight', 'debug'):
            with self.subTest(mode=mode):
                selected, complete = ((self.cpu_selected, self.cpu_complete) if mode == 'cpu'
                                      else (self.gpu_selected, self.gpu_complete))
                backend = np if mode == 'cpu' else _OutputCupy()
                closed = []; report = {}; threads = []; replayed = []
                calculation = _calculation(selected, complete, backend, closed, ('a',))
                def transform(cfg, index, values):
                    threads.append(threading.get_ident())
                    return values
                options = {'debug': mode == 'debug'}
                if mode == 'tight':
                    options.update(host_memory_limit=complete.plan.peak_host_bytes,
                                   device_memory_limit=complete.plan.peak_device_bytes)
                original = cfg_runtime.replay
                def replay(*args, **kwargs):
                    replayed.append(1)
                    return original(*args, **kwargs)
                with patch.object(cfg_runtime, 'replay', replay):
                    iterator = _events(calculation, backend, report, transform=transform, **options)
                    self.assertEqual(next(iterator)[1], 0)
                    self.assertEqual(len(replayed), 1)
                    self.assertIsNone(active_output_queue())
                    list(iterator)
                self.assertEqual(threads, [threading.get_ident()] * 3)
                self.assertEqual(report['output_queue_tasks'], 0)
                self.assertEqual(report['output_buffer_host_bytes'], 0)
                self.assertEqual(closed, ['a'])
                _no_workers(self)

    def test_yielded_transforms_are_not_retained_after_next_delivery_or_close(self):
        class Result:
            pass
        refs = []
        def transform(cfg, index, values):
            result = Result(); refs.append(weakref.ref(result))
            return result
        backend = _OutputCupy(); closed = []
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed, ('a',))
        iterator = _events(calculation, backend, {}, transform=transform)
        first = next(iterator)
        first_ref = weakref.ref(first[2]); del first
        second = next(iterator)
        gc.collect()
        self.assertIsNone(first_ref(), 'iterator retained previously delivered transform')
        del second
        iterator.close(); gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        _no_workers(self)

    def test_default_host_views_release_after_delivery_and_early_close(self):
        backend = _OutputCupy(); closed = []
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed, ('a',))
        iterator = _events(calculation, backend, {})
        first = next(iterator)
        refs = [weakref.ref(value) for value in first[2]]
        del first
        second = next(iterator)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs),
                        'previous host views remained owned by the iterator')
        refs = [weakref.ref(value) for value in second[2]]
        del second
        iterator.close(); gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        _no_workers(self)

    def test_transform_error_remains_primary_when_reader_cleanup_also_fails(self):
        backend = _OutputCupy(); closed = []
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed, ('a',))
        factory = source_loader_factory(calculation)
        @contextmanager
        def broken_factory(cfg):
            with factory(cfg) as loader:
                try:
                    yield loader
                finally:
                    raise ValueError('reader teardown failed')
        release = threading.Event()
        def transform(cfg, index, values):
            self.assertTrue(release.wait(5))
            raise RuntimeError('primary transform failed')
        original_submit = OutputConsumerQueue.submit
        def submit(queue, cfg, index, values):
            if index == 1:
                # Raise in the wrapper while replay is suspended at its yield.
                # Closing replay with GeneratorExit used to lose cleanup notes.
                release.set()
                with self.assertRaisesRegex(RuntimeError, 'primary transform failed'):
                    queue.pending[0].result(timeout=5)
            return original_submit(queue, cfg, index, values)
        with patch.object(OutputConsumerQueue, 'submit', submit):
            with self.assertRaisesRegex(RuntimeError, 'primary transform failed') as caught:
                list(iter_execute_cfg_plan(calculation.plan, ('a',), broken_factory,
                                           backend=backend, transform=transform))
        self.assertTrue(any('reader teardown failed' in note
                            for note in getattr(caught.exception, '__notes__', ())))
        self.assertEqual(closed, ['a'])
        self.assertTrue(backend.tracker.pools[0].freed)
        _no_workers(self)

    def test_configuration_iteration_failure_closes_claimed_reader_scope(self):
        backend = _OutputCupy(); closed = []
        def configurations():
            yield 'a'
            raise RuntimeError('configuration source failed')
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed,
                                   configurations())
        with self.assertRaisesRegex(RuntimeError, 'configuration source failed'):
            list(_events(calculation, backend, {}))
        self.assertEqual(closed, ['a'])
        self.assertTrue(backend.tracker.pools[0].freed)
        _no_workers(self)

    def test_workflow_early_close_preserves_complete_cfg_and_closes_claimed_sources(self):
        backend = _OutputCupy(); closed = []; report = {}
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed)
        iterator = iter_memory_calculation(calculation, report)
        first = next(iterator)
        self.assertEqual(first[0], 'a')
        iterator.close()
        cpu = _calculation(self.cpu_selected, self.cpu_complete, np, [], ('a',))
        expected = dict(iter_memory_calculation(cpu, {}))
        _assert_tree_allclose(first[1], expected['a'])
        self.assertIn('a', closed)
        self.assertEqual(len(closed), len(set(closed)))
        self.assertTrue(backend.tracker.pools[0].freed)
        _no_workers(self)

    def test_workflow_uses_one_queue_and_bound_assemblers_with_complete_cfg_results(self):
        backend = _OutputCupy(); closed = []; report = {}; assembly_threads = []
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed)
        from lqcdweave.contraction import correlator_cfg
        original = correlator_cfg.bind_point_assemblers
        def bind(calculation):
            assemblers = original(calculation)
            def wrap(assembler):
                def assemble(values):
                    assembly_threads.append(threading.current_thread().name)
                    return assembler(values)
                return assemble
            return tuple(wrap(assembler) for assembler in assemblers)
        with (patch.object(correlator_cfg, 'bind_point_assemblers', side_effect=bind) as binding,
              patch.object(cfg_runtime, 'OutputConsumerQueue', wraps=OutputConsumerQueue) as queue):
            iterator = iter_memory_calculation(calculation, report)
            first = next(iterator)
            self.assertEqual(first[0], 'a')
            reference = first[1]
            rest = list(iterator)
            self.assertEqual([cfg for cfg, _ in rest], ['b'])
            self.assertEqual(queue.call_count, 1)
            self.assertEqual(binding.call_count, 1)
        cpu_closed = []
        cpu = _calculation(self.cpu_selected, self.cpu_complete, np, cpu_closed)
        expected = dict(iter_memory_calculation(cpu, {}))
        _assert_tree_allclose(reference, expected['a'])
        _assert_tree_allclose(rest[0][1], expected['b'])
        self.assertEqual(report['completed'], ('a', 'b'))
        self.assertEqual(closed, ['a', 'b'])
        self.assertEqual(len(assembly_threads), 6)
        self.assertTrue(all(name.startswith('contraction-results') for name in assembly_threads))
        _no_workers(self)

    def test_resume_and_close_restore_callers_current_queue_context(self):
        backend = _OutputCupy(); closed = []
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed, ('a',))
        iterator = _events(calculation, backend, {})
        next(iterator)
        try:
            with OutputConsumerQueue(lambda *row: None) as caller_queue:
                next(iterator)
                self.assertIs(active_output_queue(), caller_queue,
                              'resuming iterator replaced the caller queue context')
                iterator.close()
                self.assertIs(active_output_queue(), caller_queue,
                              'closing iterator replaced the caller queue context')
        finally:
            iterator.close()
        _no_workers(self)

    def test_callback_facade_keeps_one_queue_and_unchanged_completion_contract(self):
        backend = _OutputCupy(); closed = []; rows = []
        calculation = _calculation(self.gpu_selected, self.gpu_complete, backend, closed, ('a',))
        with patch.object(cfg_runtime, 'OutputConsumerQueue', wraps=OutputConsumerQueue) as queue:
            report = execute_cfg_plan(calculation.plan, ('a',), source_loader_factory(calculation),
                                      lambda *row: rows.append(row), backend=backend)
        self.assertEqual(queue.call_count, 1)
        self.assertEqual([row[1] for row in rows], [0, 1, 2, None])
        self.assertEqual(report['completed'], ('a',))
        self.assertEqual(closed, ['a'])
        _no_workers(self)


class RealCuPyIteratorOutputQueueTests(unittest.TestCase):
    """GPU tier: exercise real queue workers, not only manual ticket downloads."""
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no device')
        except Exception as error:
            raise unittest.SkipTest(f'real CuPy iterator queue requires a working device: {error}')
        cls.backend = cupy
        cls.selected, cls.complete = _plan(_targets()[1])
        cls.cpu_selected, cls.cpu_complete = _plan(_targets()[0])

    def test_real_worker_outputs_match_numpy_and_remain_owned_across_cfgs(self):
        closed = []; report = {}; worker_threads = []
        calculation = _calculation(self.selected, self.complete, self.backend, closed)
        def transform(cfg, index, values):
            worker_threads.append(threading.current_thread().name)
            self.assertTrue(all(type(value) is np.ndarray for value in values))
            return values
        actual = list(_events(calculation, self.backend, report, transform=transform))
        expected = list(_events(_calculation(self.cpu_selected, self.cpu_complete, np, []), np, {}))
        for (_, index, values), (_, _, truth) in zip(actual, expected, strict=True):
            if index is not None:
                _assert_tree_allclose(values, truth)
        self.assertEqual(closed, ['a', 'b'])
        self.assertEqual(report['output_queue_tasks'], 6)
        self.assertLessEqual(report['device']['peak_pending_outputs'], 2)
        self.assertTrue(all(name.startswith('contraction-results') for name in worker_threads))
        _no_workers(self)

    def test_real_worker_failure_drains_transfers_before_reader_close(self):
        closed = []; calls = []; devices = []
        calculation = _calculation(self.selected, self.complete, self.backend, closed, ('a',))
        original = cfg_runtime.replay
        def replay(*args, **kwargs):
            devices.append(active_device_execution())
            return original(*args, **kwargs)
        def transform(cfg, index, values):
            calls.append(index)
            raise RuntimeError('real queue transform failed')
        with patch.object(cfg_runtime, 'replay', replay):
            with self.assertRaisesRegex(RuntimeError, 'real queue transform failed'):
                list(_events(calculation, self.backend, {}, transform=transform))
        self.assertEqual(calls, [0])
        self.assertEqual(closed, ['a'])
        self.assertTrue(devices)
        self.assertEqual(devices[0]._downloads, set())
        _no_workers(self)

    def test_real_worker_early_close_joins_slow_consumer_and_preserves_returned_arrays(self):
        entered = threading.Event(); release = threading.Event(); finished = threading.Event()
        closed = []; devices = []; threads = []
        calculation = _calculation(self.selected, self.complete, self.backend, closed, ('a',))
        original = cfg_runtime.replay
        def replay(*args, **kwargs):
            devices.append(active_device_execution())
            return original(*args, **kwargs)
        def transform(cfg, index, values):
            threads.append(threading.current_thread().name)
            if index == 1:
                entered.set()
                self.assertTrue(release.wait(5))
                self.assertFalse(closed)
                finished.set()
            return values
        with patch.object(cfg_runtime, 'replay', replay):
            iterator = _events(calculation, self.backend, {}, transform=transform)
            try:
                first = next(iterator)
                snapshot = tuple(value.copy() for value in first[2])
                self.assertTrue(entered.wait(5))
            finally:
                release.set()
                iterator.close()
        self.assertTrue(finished.is_set())
        self.assertEqual(closed, ['a'])
        self.assertEqual(devices[0]._downloads, set())
        self.assertTrue(all(name.startswith('contraction-results') for name in threads))
        for value, expected in zip(first[2], snapshot, strict=True):
            np.testing.assert_array_equal(value, expected)
        _no_workers(self)


if __name__ == '__main__':
    unittest.main()
