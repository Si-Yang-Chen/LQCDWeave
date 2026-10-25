"""Real worker handshakes for original-plan IO / CPU / compute dependencies."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.execution.read_ahead import ReadAhead
from lqcdweave.contraction.execution.bound_program import BoundProgram, _ReadCall
from lqcdweave.contraction.execution.storage_runtime import _BlockExecution
from correlator.execution.test_read_ahead_device_budget import _Loader, _plan


class StagedLoader(_Loader):
    def __init__(self):
        super().__init__({'A': 1, 'B': 1}, {'A': 1, 'B': 1})
        self._prefetch_stages = self.host, self.prepare, self.close_host
        self.cpu_started, self.release_cpu, self.io_advanced = Event(), Event(), Event()
        self.prepared, self.raw_closed, self.threads = [], [], {}
        self.fail_prepare = False

    def host(self, key, block):
        self.threads['read:' + key] = current_thread().name
        if key == 'B':
            if not self.cpu_started.wait(5):
                raise AssertionError('CPU preparation did not start')
            self.io_advanced.set()
        return 'host:' + key

    def prepare(self, key, block, value):
        self.threads['prepare:' + key] = current_thread().name
        if key == 'A':
            self.cpu_started.set()
            if not self.release_cpu.wait(5):
                raise AssertionError('CPU preparation was not released')
            if self.fail_prepare:
                raise RuntimeError('staging failed')
        self.prepared.append(key)
        return 'uploaded:' + key

    def close_host(self, key, value):
        self.raw_closed.append((key, value))


def make_queue(loader):
    plan = _plan(('A', 'B'))
    return plan, ReadAhead(plan, loader, {'A': 1, 'B': 1}, 2,
                          device_sizes={'A': 1, 'B': 1}, device_capacity=2)


def test_io_advances_while_cpu_prepares_and_original_events_keep_their_budget():
    loader = StagedLoader()
    plan, queue = make_queue(loader)
    try:
        queue.start_after(-1)
        assert loader.io_advanced.wait(5)
        assert not queue.pending['A'].done()
        assert queue.pending['A'].event is plan.events[0]
        assert queue.pending['B'].event is plan.events[1]
        assert (queue.reserved_bytes, queue.reserved_device_bytes) == (2, 2)
        loader.release_cpu.set()
        assert queue.take('A', plan.blocks['A']) == (True, 'uploaded:A')
        assert queue.take('B', plan.blocks['B']) == (True, 'uploaded:B')
        assert loader.prepared == ['A', 'B']
        assert loader.threads['read:B'].startswith('contraction-io')
        assert loader.threads['prepare:A'].startswith('contraction-prepare')
        assert queue.reserved_bytes == queue.reserved_device_bytes == 0
    finally:
        loader.release_cpu.set()
        queue.close()


def test_saved_compute_cursor_waits_for_host_preparation_without_blocking_io():
    loader = StagedLoader()
    plan, queue = make_queue(loader)
    finished = []
    execution = SimpleNamespace(read_ahead=queue, before={0: (plan.events[0],)},
                                _waiting_read=None)
    execution.reads_ready = lambda index: _BlockExecution.reads_ready(execution, index)
    execution.wait_for_reads = lambda: _BlockExecution.wait_for_reads(execution)

    def begin(index, timed):
        assert queue.pending['A'].done()
        assert queue.take('A', plan.blocks['A'])[1] == 'uploaded:A'

    execution.begin_action = begin
    execution.end_action = lambda *args: None
    calls = (_ReadCall(0), lambda context: finished.append('compute'))
    program = BoundProgram(calls, (), np, False, None)
    queue.start_after(-1)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(program.run, execution, {})
            try:
                assert loader.io_advanced.wait(5)
                assert not future.done()
                assert not finished
                assert program.calls is calls
                loader.release_cpu.set()
                assert future.result(5) == ()
                assert finished == ['compute']
            finally:
                loader.release_cpu.set()
    finally:
        queue.close()


def test_close_drains_active_cpu_and_disposes_queued_host_without_uploading_it():
    loader = StagedLoader()
    _, queue = make_queue(loader)
    queue.start_after(-1)
    try:
        assert loader.io_advanced.wait(5)
        with ThreadPoolExecutor(max_workers=1) as closer:
            future = closer.submit(queue.close)
            try:
                with queue._prepare_condition:
                    assert queue._prepare_condition.wait_for(
                        lambda: queue._prepare_stopping, timeout=5)
                assert not future.done()  # A still owns its CPU/staging input.
                loader.release_cpu.set()
                future.result(5)
            finally:
                loader.release_cpu.set()
        assert loader.prepared == ['A']
        assert loader.closed == ['A']
        assert loader.raw_closed == [('B', 'host:B')]
        assert queue.pool is None and queue.prepare_pool is None
        assert queue.reserved_bytes == queue.reserved_device_bytes == 0
    finally:
        loader.release_cpu.set()
        queue.close()


def test_prepare_failure_disposes_host_on_owner_and_releases_admission():
    loader = StagedLoader()
    loader.fail_prepare = True
    plan, queue = make_queue(loader)
    queue.start_after(-1)
    try:
        assert loader.io_advanced.wait(5)
        loader.release_cpu.set()
        with pytest.raises(RuntimeError, match='staging failed'):
            queue.take('A', plan.blocks['A'])
        assert loader.raw_closed == [('A', 'host:A')]
        assert queue.reserved_bytes == queue.reserved_device_bytes == 1
        assert queue.take('B', plan.blocks['B'])[0]
    finally:
        loader.release_cpu.set()
        queue.close()


def test_cpu_reader_keeps_one_worker_and_no_readiness_dispatch():
    loader = _Loader({'A': 1, 'B': 1}, {'A': 0, 'B': 0})
    plan, queue = make_queue(loader)
    try:
        queue.start_after(-1)
        assert not queue.split_preparation
        assert queue.take('A', plan.blocks['A'])[0]
        assert queue.prepare_pool is None
        execution = SimpleNamespace(read_ahead=queue)
        program = BoundProgram((lambda context: None,), (), np, False, None)
        with patch('lqcdweave.contraction.execution.bound_program.drive_plan_cursors',
                   side_effect=AssertionError('unnecessary dispatch')):
            assert program.run(execution, {}) == ()
    finally:
        queue.close()


def test_small_resident_inputs_with_many_blocks_keep_existing_read_worker():
    keys = ('A', 'B', 'C', 'D', 'E')
    sizes = dict.fromkeys(keys, 1)
    loader = _Loader(sizes, sizes)
    loader.input_readers = {'resident': SimpleNamespace(host_resident=True)}
    def unnecessary_stage(*args):
        raise AssertionError('resident inputs need no additional CPU worker')
    loader._prefetch_stages = (unnecessary_stage,) * 3
    plan = _plan(keys)
    for block in plan.blocks.values():
        block.read_bytes = 1
    queue = ReadAhead(plan, loader, sizes, 5, device_sizes=sizes, device_capacity=5)
    try:
        queue.start_after(-1)
        assert not queue.split_preparation
        for key in keys:
            assert queue.take(key, plan.blocks[key])[0]
        assert queue.prepare_pool is None
    finally:
        queue.close()
