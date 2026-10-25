"""Device reservations stay independent from host read-ahead capacity."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from threading import Event, Thread
import unittest
import weakref
from unittest.mock import patch

from lqcdweave.contraction.execution.read_ahead import (
    ReadAhead,
    _PrefixedReader,
    prepare_batch_read_ahead,
    prepare_read_ahead,
)
from lqcdweave.contracts.correlator.storage import StorageEvent


class _RecordingPool(ThreadPoolExecutor):
    def __init__(self, *, max_workers, thread_name_prefix):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self.shutdown_calls = []
        self.submissions = 0

    def submit(self, *args, **kwargs):
        self.submissions += 1
        return super().submit(*args, **kwargs)

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append((wait, cancel_futures))
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


class _Loader:
    prefetch_across_batches = True

    def __init__(self, host_sizes, device_sizes, *, fail=()):
        self.host_sizes = dict(host_sizes)
        self.device_sizes = dict(device_sizes)
        self.fail = set(fail)
        self.prefetched = []
        self.finished = []
        self.closed = []

    def can_prefetch_block(self, key, block):
        return True

    def prefetch_host_bytes(self, key, block):
        return self.host_sizes[key]

    def prefetch_device_bytes(self, key, block):
        return self.device_sizes[key]

    def prefetch_block(self, key, block):
        self.prefetched.append(key)
        if key in self.fail:
            raise RuntimeError(f"prefetch failed for {key}")
        return f"value:{key}"

    def finish_prefetch(self, key, block, value):
        self.finished.append(key)
        return value

    def close_prefetch(self, key, value):
        self.closed.append(key)


def _event(key, index):
    return StorageEvent(action_index=index, kind="read", block=key)


def _plan(keys, *, peak_host=0, peak_device=0, backend="cupy"):
    blocks = {
        key: SimpleNamespace(host_bytes=1, staging_host_bytes=0, device_bytes=1)
        for key in keys
    }
    return SimpleNamespace(
        events=tuple(_event(key, index) for index, key in enumerate(keys)),
        blocks=blocks,
        state=SimpleNamespace(
            problem=SimpleNamespace(backend=backend),
            actions=tuple(() for _ in ()),
        ),
        memory_limit=100,
        host_memory_limit=100,
        peak_host_bytes=peak_host,
        peak_device_bytes=peak_device,
        memory=(SimpleNamespace(device_bytes=peak_device),),
    )


class ReadAheadDeviceBudgetTests(unittest.TestCase):
    def test_idle_worker_does_not_retain_consumed_buffer(self):
        class Buffer:
            pass
        class WeakLoader(_Loader):
            def prefetch_block(self, key, block):
                value = Buffer()
                self.reference = weakref.ref(value)
                return value
        plan = _plan(('A',))
        loader = WeakLoader({'A': 1}, {'A': 0})
        queue = ReadAhead(plan, loader, {'A': 1}, 1, device_capacity=0)
        self.addCleanup(queue.close)
        queue.start_after(-1)
        found, value = queue.take('A', plan.blocks['A'])
        self.assertTrue(found)
        self.assertIs(loader.reference(), value)
        del value
        self.assertIsNone(loader.reference())
        self.assertEqual(queue.reserved_bytes, 0)

    def test_repeated_block_does_not_overwrite_an_unconsumed_ticket(self):
        plan = _plan(('A', 'B'))
        plan.events = (_event('A', 0), _event('A', 2), _event('B', 3))
        loader = _Loader({'A': 0, 'B': 0}, {'A': 0, 'B': 0})
        queue = ReadAhead(plan, loader, {'A': 0, 'B': 0}, 0, device_capacity=0)
        self.addCleanup(queue.close)
        queue.start_after(-1)
        first = queue.pending['A']
        queue.start_after(-1)
        self.assertIs(queue.pending['A'], first)
        self.assertEqual(queue.submitted_reads, 1)
        self.assertEqual(queue.take('A', plan.blocks['A']), (True, 'value:A'))
        queue.start_after(1)
        self.assertEqual(queue.take('A', plan.blocks['A']), (False, None))
        queue.start_after(2)
        self.assertEqual(queue.take('B', plan.blocks['B']), (True, 'value:B'))
        self.assertEqual(loader.prefetched, ['A', 'B'])

    def test_tiny_reads_have_bounded_pending_tickets_and_one_worker_submission(self):
        keys = tuple(str(i) for i in range(400))
        plan = _plan(keys)
        for block in plan.blocks.values():
            block.host_bytes = block.device_bytes = 0
        loader = _Loader(dict.fromkeys(keys, 0), dict.fromkeys(keys, 0))
        queue = ReadAhead(plan, loader, dict.fromkeys(keys, 0), 0, device_capacity=0)
        self.addCleanup(queue.close)
        with patch('lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor', _RecordingPool):
            queue.start_after(-1)
            self.assertLessEqual(len(queue.pending), 128)
            self.assertLess(queue.submitted_reads, len(keys))
            for i, key in enumerate(keys):
                queue.start_after(i - 1)
                self.assertEqual(queue.take(key, plan.blocks[key]), (True, 'value:' + key))
            self.assertEqual(queue.pool.submissions, 1)
        self.assertEqual(loader.prefetched, list(keys))
        self.assertEqual((queue.submitted_reads, queue.completed_reads), (400, 400))

    def test_close_cancels_queued_read_and_joins_running_read_before_cleanup(self):
        started, release = Event(), Event()
        class BlockingLoader(_Loader):
            def prefetch_block(self, key, block):
                if key == 'A':
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError('test read was not released')
                return super().prefetch_block(key, block)
        plan = _plan(('A', 'B'))
        loader = BlockingLoader({'A': 1, 'B': 1}, {'A': 0, 'B': 0})
        queue = ReadAhead(plan, loader, {'A': 1, 'B': 1}, 2, device_capacity=0)
        queue.start_after(-1)
        self.assertTrue(started.wait(2))
        queued = queue.pending['B']
        errors = []
        def close():
            try:
                queue.close()
            except BaseException as exc:
                errors.append(exc)
        closer = Thread(target=close)
        closer.start()
        try:
            with queue._condition:
                self.assertTrue(queue._condition.wait_for(queued.cancelled, timeout=2))
            self.assertEqual(loader.closed, [])
        finally:
            release.set()
            closer.join(5)
            queue.close()
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(loader.prefetched, ['A'])
        self.assertEqual(loader.closed, ['A'])
        self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))

    def test_failed_read_does_not_stop_subsequent_ticket(self):
        plan = _plan(('A', 'B'))
        loader = _Loader({'A': 1, 'B': 1}, {'A': 0, 'B': 0}, fail=('A',))
        queue = ReadAhead(plan, loader, {'A': 1, 'B': 1}, 2, device_capacity=0)
        self.addCleanup(queue.close)
        queue.start_after(-1)
        with self.assertRaisesRegex(RuntimeError, 'prefetch failed for A'):
            queue.take('A', plan.blocks['A'])
        self.assertEqual(queue.take('B', plan.blocks['B']), (True, 'value:B'))
        self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))

    def test_device_limit_blocks_submission_until_take_releases_both_budgets(self):
        plan = _plan(("A", "B"))
        loader = _Loader({"A": 4, "B": 4}, {"A": 6, "B": 6})
        queue = ReadAhead(
            plan, loader, {"A": 4, "B": 4}, 8,
            device_sizes={"A": 6, "B": 6}, device_capacity=6,
        )
        self.addCleanup(queue.close)

        with patch(
            "lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor",
            _RecordingPool,
        ):
            queue.start_after(-1)
            self.assertEqual(set(queue.pending), {"A"})
            self.assertEqual(queue.reserved_bytes, 4)
            self.assertEqual(queue.reserved_device_bytes, 6)
            self.assertEqual(queue.peak_device_bytes, 6)

            found, value = queue.take("A", plan.blocks["A"])
            self.assertTrue(found)
            self.assertEqual(value, "value:A")
            self.assertEqual(queue.reserved_bytes, 0)
            self.assertEqual(queue.reserved_device_bytes, 0)

            queue.start_after(0)
            self.assertEqual(set(queue.pending), {"B"})
            self.assertEqual(queue.reserved_bytes, 4)
            self.assertEqual(queue.reserved_device_bytes, 6)
            self.assertTrue(queue.take("B", plan.blocks["B"])[0])

        self.assertEqual(queue.reserved_bytes, 0)
        self.assertEqual(queue.reserved_device_bytes, 0)
        self.assertEqual(queue.submitted_reads, 2)
        self.assertEqual(queue.completed_reads, 2)

    def test_failed_future_releases_host_and_device_reservations(self):
        plan = _plan(("A",))
        loader = _Loader({"A": 3}, {"A": 5}, fail=("A",))
        queue = ReadAhead(
            plan, loader, {"A": 3}, 3,
            device_sizes={"A": 5}, device_capacity=5,
        )
        self.addCleanup(queue.close)

        with patch(
            "lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor",
            _RecordingPool,
        ):
            queue.start_after(-1)
            self.assertEqual(queue.reserved_bytes, 3)
            self.assertEqual(queue.reserved_device_bytes, 5)
            with self.assertRaisesRegex(RuntimeError, "prefetch failed"):
                queue.take("A", plan.blocks["A"])

        self.assertEqual(queue.reserved_bytes, 0)
        self.assertEqual(queue.reserved_device_bytes, 0)
        self.assertFalse(queue.pending)

    def test_close_releases_both_reservations_and_closes_completed_buffer(self):
        plan = _plan(("A",))
        loader = _Loader({"A": 2}, {"A": 7})
        queue = ReadAhead(
            plan, loader, {"A": 2}, 2,
            device_sizes={"A": 7}, device_capacity=7,
        )

        with patch(
            "lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor",
            _RecordingPool,
        ):
            queue.start_after(-1)
            self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (2, 7))
            queue.pending['A'].result()
            queue.close()

        self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
        self.assertEqual(loader.closed, ["A"])

    def test_extend_prefixes_device_size_queries_and_reservations(self):
        plan = _plan(("A",))
        loader = _Loader({"A": 2}, {"A": 4})
        empty = SimpleNamespace(events=(), blocks={})
        queue = ReadAhead(empty, loader, {}, 2, device_capacity=4)
        self.addCleanup(queue.close)
        queue.extend(
            plan, loader, {"A": 2}, device_sizes={"A": 4}, prefix="cfg-2:",
        )
        owner = queue.owners["cfg-2:A"]
        self.assertEqual(owner.prefetch_device_bytes("cfg-2:A", plan.blocks["A"]), 4)

        with patch(
            "lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor",
            _RecordingPool,
        ):
            queue.start_after(-1)
            self.assertEqual(queue.reserved_device_bytes, 4)
            self.assertTrue(queue.take("cfg-2:A", plan.blocks["A"])[0])

        self.assertEqual(loader.prefetched, ["A"])
        self.assertEqual(loader.finished, ["A"])
        self.assertEqual(queue.reserved_device_bytes, 0)

    def test_prepare_functions_bound_default_and_explicit_device_capacity(self):
        single = _plan(("A",), peak_host=10, peak_device=92)
        loader = _Loader({"A": 8}, {"A": 6})

        adjusted, queue, host_reserve = prepare_read_ahead(single, loader)
        self.assertIs(adjusted, single)
        self.assertEqual(host_reserve, 8)
        self.assertIsNotNone(queue)
        self.assertEqual(queue.device_capacity, 6)
        self.assertEqual(queue.device_sizes, {"A": 6})

        _same, blocked_queue, blocked_reserve = prepare_read_ahead(
            single, loader, device_capacity=5,
        )
        self.assertIsNone(blocked_queue)
        self.assertEqual(blocked_reserve, 0)

        first = _plan(("A",), peak_host=10, peak_device=90)
        second = _plan(("B",), peak_host=20, peak_device=95)
        batch_loader = _Loader(
            {"A": 4, "B": 4}, {"A": 4, "B": 4},
        )
        _plans, batch_queue, _offsets, _reserve = prepare_batch_read_ahead(
            (first, second), batch_loader,
        )
        self.assertIsNotNone(batch_queue)
        self.assertEqual(batch_queue.device_capacity, 5)
        self.assertEqual(batch_queue.device_sizes, {"A": 4, "B": 4})

    def test_cpu_loader_without_device_accounting_keeps_zero_device_cost(self):
        class HostOnlyLoader(_Loader):
            prefetch_device_bytes = None

        plan = _plan(("A",), peak_host=10, backend="numpy")
        loader = HostOnlyLoader({"A": 4}, {})
        _adjusted, queue, reserve = prepare_read_ahead(plan, loader)
        self.addCleanup(queue.close)

        self.assertEqual(reserve, 4)
        self.assertEqual(queue.device_sizes, {"A": 0})
        self.assertEqual(queue.device_capacity, 0)
        with patch(
            "lqcdweave.contraction.execution.read_ahead.ThreadPoolExecutor",
            _RecordingPool,
        ):
            queue.start_after(-1)
            self.assertEqual(queue.reserved_device_bytes, 0)
            self.assertTrue(queue.take("A", plan.blocks["A"])[0])


if __name__ == "__main__":
    unittest.main()
