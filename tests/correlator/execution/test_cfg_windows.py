"""Plan delivery overlaps replay while one IO queue owns window-local IDs."""
from contextlib import contextmanager
from dataclasses import replace
import os
import threading
import unittest
from unittest.mock import patch

import numpy as np
from correlator.execution import test_cfg_runtime as fixtures
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.expansion.api import restore_input_loader, describe_input_reads
from lqcdweave.contraction.planning.api import iter_cfg_plan_windows
from lqcdweave.contraction.execution.api import execute_cfg_windows
from lqcdweave.contraction.correlator_stream import plan_cfg_windows, PlanningWorkerError


class CfgWindowExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CfgReplayTests()
        self.plan, expanded, selected, self.elemental, self.propagator = self.fixture.setup_plan()
        points, _ = describe_coordinates(time_pairs=[{0:1, 1:0}, {0:0, 1:1}])
        self.args = (selected, self.plan.templates[0].output, points.points,
                     describe_input_reads(expanded.inputs), expanded.inputs.loader.preparations,
                     expanded.inputs.loader.specs)
        self.records, self.closed = [], []

    @contextmanager
    def factory(self, plan, cfg):
        readers = {}
        for request in plan.requests.values():
            for actual in (request, *request.aliases):
                readers[actual.reader_role] = fixtures.SafeReader(
                    self.propagator if actual.family == 'propagator' else self.elemental,
                    cfg, 1, self.records)
        loader = restore_input_loader(plan.requests, readers, plan.preparations,
                                      plan.specs, configuration=cfg)
        try:
            yield loader
        finally:
            self.closed.append(plan.statistics['coordinate_offset'])

    def test_every_window_validates_saved_sparse_workspace_before_reader_io(self):
        plans = list(iter_cfg_plan_windows(*self.args, window_size=1))
        for rejected in (0, 1):
            with self.subTest(rejected_window=rejected):
                admitted, notified = [], []
                self.closed.clear()
                self.records.clear()
                @contextmanager
                def factory(plan, cfg):
                    admitted.append(plan.statistics['coordinate_offset'])
                    with self.factory(plan, cfg) as loader:
                        yield loader
                calls = []
                def validate(plan, backend):
                    self.assertIs(backend, np)
                    calls.append(plan)
                    if len(calls) == rejected + 1:
                        raise MemoryError('saved sparse workspace is incomplete')
                with patch('lqcdweave.contraction.execution.cfg_windows.'
                           'validate_coordinate_batch_workspace', side_effect=validate):
                    with self.assertRaisesRegex(MemoryError, 'saved sparse workspace'):
                        execute_cfg_windows(iter(plans), 'a', factory,
                            lambda cfg, index, values: notified.append(index))
                self.assertEqual(admitted, list(range(rejected)))
                self.assertEqual(self.closed, list(range(rejected)))
                self.assertNotIn(None, notified)
                if rejected == 0:
                    self.assertEqual(self.records, [])

    def test_private_cpu_producer_and_one_queue_match_complete_execution(self):
        expected = self.fixture.run_plan(capacity=0)[1]
        outputs, completions = {}, []
        def consume(cfg, index, values):
            if index is None:
                completions.append(cfg)
            else:
                outputs[cfg, index] = values
        with plan_cfg_windows(*self.args, window_size=1) as windows:
            worker = windows.process
            self.assertNotEqual(worker.pid, os.getpid())
            with patch('lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor',
                       wraps=__import__('concurrent.futures').futures.ThreadPoolExecutor) as pool:
                report = execute_cfg_windows(windows, 'a', self.factory, consume)
            self.assertEqual(pool.call_count, 1)
        self.assertIsNotNone(worker.poll())
        self.assertEqual(completions, ['a'])
        self.assertEqual(report['completed'], ('a',))
        self.assertEqual(report['planned_windows'], 2)
        delivery = windows.statistics
        self.assertEqual(delivery['planning_received_windows'], 2)
        self.assertGreater(delivery['planning_received_bytes'], 0)
        self.assertGreaterEqual(delivery['planning_receive_seconds'], 0)
        self.assertGreaterEqual(delivery['planning_restore_seconds'], 0)
        self.assertEqual(windows.receiver.templates, ())
        self.assertLessEqual(report['peak_bound_windows'], 2)
        self.assertEqual(self.closed, [0, 1])
        for key, actual in outputs.items():
            for reference, value in zip(expected[key], actual, strict=True):
                np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)

    def test_first_window_executes_before_future_window_is_ready(self):
        plans = list(iter_cfg_plan_windows(*self.args, window_size=1))
        first_consumed = threading.Event()
        class Delayed:
            def __init__(self):
                self.index = 0
                self.closed = False
            def __iter__(self):
                return self
            def __next__(self):
                if self.index == 2:
                    raise StopIteration
                if self.index == 1 and not first_consumed.is_set():
                    raise AssertionError('blocked compute on unfinished future planning')
                value = plans[self.index]
                self.index += 1
                return value
            def next_ready(self):
                return next(self) if first_consumed.is_set() else None
            def close(self):
                self.closed = True
        windows = Delayed()
        indices = []
        def consume(cfg, index, values):
            indices.append(index)
            if index == 0:
                first_consumed.set()
        execute_cfg_windows(windows, 'a', self.factory, consume, prefetch_host_bytes=0)
        self.assertEqual(indices, [0, 1, None])
        self.assertTrue(windows.closed)

    def test_later_planner_failure_never_publishes_cfg(self):
        first = next(iter_cfg_plan_windows(*self.args, window_size=1))
        def windows():
            yield first
            raise RuntimeError('later planning failed')
        notifications = []
        with self.assertRaisesRegex(RuntimeError, 'later planning failed'):
            execute_cfg_windows(windows(), 'a', self.factory,
                                lambda cfg, index, values: notifications.append(index))
        self.assertEqual(notifications, [0])
        self.assertEqual(self.closed, [0])

    def test_later_resource_failure_and_consumer_error_cleanup(self):
        plans = list(iter_cfg_plan_windows(*self.args, window_size=1))
        invalid = replace(plans[1], peak_host_bytes=10**12)
        with self.assertRaises(MemoryError):
            execute_cfg_windows(iter((plans[0], invalid)), 'a', self.factory,
                                lambda *args: None, host_memory_limit=10**9)
        self.assertEqual(self.closed, [0])
        self.closed.clear()
        with plan_cfg_windows(*self.args, window_size=1) as windows:
            worker = windows.process
            def fail(*args):
                raise RuntimeError('consumer failed')
            with self.assertRaisesRegex(RuntimeError, 'consumer failed'):
                execute_cfg_windows(windows, 'a', self.factory, fail)
        self.assertIsNotNone(worker.poll())
        self.assertIn(0, self.closed)

    def test_worker_error_is_reported_and_reaped(self):
        with plan_cfg_windows(*self.args, window_size=0) as windows:
            worker = windows.process
            with self.assertRaisesRegex(PlanningWorkerError, 'window_size'):
                next(windows)
        self.assertIsNotNone(worker.poll())

    def test_cpu_producer_builds_next_window_without_another_consumer_call(self):
        with plan_cfg_windows(*self.args, window_size=1) as windows:
            first = next(windows)
            self.assertEqual(first.statistics['coordinate_offset'], 0)
            # Waiting only observes the already submitted CPU request. No
            # next() call drives binding while the consumer owns window 0.
            kind, following = windows.future.result(timeout=30)
            self.assertEqual(kind, 'window')
            self.assertEqual(following.statistics['coordinate_offset'], 1)
            self.assertIs(windows.next_ready(), following)
        self.assertIsNotNone(windows.process.poll())

    def test_ready_next_window_reads_before_current_output_is_consumed(self):
        plans = list(iter_cfg_plan_windows(*self.args, window_size=1))
        future_read = threading.Event()
        class Ready:
            def __init__(self): self.items = iter(plans)
            def __iter__(self): return self
            def __next__(self): return next(self.items)
            def next_ready(self): return next(self)
        @contextmanager
        def factory(plan, cfg):
            with self.factory(plan, cfg) as loader:
                if plan.statistics['coordinate_offset']:
                    original = loader.prefetch_block
                    def read(*args):
                        result = original(*args)
                        future_read.set()
                        return result
                    loader.prefetch_block = read
                yield loader
        def consume(cfg, index, values):
            if index == 0:
                self.assertTrue(future_read.wait(timeout=5),
                                'next window was not read ahead of current output')
        execute_cfg_windows(Ready(), 'a', factory, consume)


if __name__ == '__main__':
    unittest.main()
