"""A read completion wakes its consumers without waking unrelated tickets."""
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event, current_thread
import unittest

from correlator.execution.test_read_ahead_device_budget import _Loader, _plan
from lqcdweave.contraction.execution.read_ahead import ReadAhead


class _ConditionProbe(Condition):
    def __init__(self, expected_waiters=1):
        super().__init__()
        self.wait_started = Event()
        self.expected_waiters = expected_waiters
        self.consumer_waits = 0
        self.completion_notifications = 0

    def wait(self, timeout=None):
        if current_thread().name.startswith('ticket-consumer'):
            self.consumer_waits += 1
            if self.consumer_waits == self.expected_waiters:
                self.wait_started.set()
        return super().wait(timeout)

    def notify_all(self):
        self.completion_notifications += 1
        return super().notify_all()


class ReadAheadNotificationTests(unittest.TestCase):
    def test_earlier_completion_does_not_wake_consumer_of_later_ticket(self):
        started_a, release_a, started_b, release_b = (Event() for _ in range(4))

        class Loader(_Loader):
            def prefetch_block(self, key, block):
                started, release = ((started_a, release_a) if key == 'A'
                                    else (started_b, release_b))
                started.set()
                if not release.wait(5):
                    raise RuntimeError('controlled read was not released')
                return super().prefetch_block(key, block)

        plan = _plan(('A', 'B'))
        loader = Loader({'A': 1, 'B': 1}, {'A': 0, 'B': 0})
        queue = ReadAhead(plan, loader, {'A': 1, 'B': 1}, 2, device_capacity=0)
        condition = queue._condition = _ConditionProbe()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='ticket-consumer') as consumers:
            try:
                queue.start_after(-1)
                self.assertTrue(started_a.wait(2))
                future = consumers.submit(queue.take, 'B', plan.blocks['B'])
                self.assertTrue(condition.wait_started.wait(2))
                release_a.set()
                self.assertTrue(started_b.wait(2))
                with condition:
                    self.assertEqual(condition.completion_notifications, 0)
                    self.assertEqual(condition.consumer_waits, 1)
                    self.assertEqual(queue.pending['A'].waiters, 0)
                self.assertFalse(future.done())
                release_b.set()
                self.assertEqual(future.result(2), (True, 'value:B'))
                self.assertEqual(condition.completion_notifications, 1)
                self.assertEqual(queue.take('A', plan.blocks['A']), (True, 'value:A'))
                self.assertEqual((queue.reserved_bytes, queue.reserved_device_bytes), (0, 0))
            finally:
                release_a.set()
                release_b.set()
                queue.close()

    def test_already_completed_ticket_does_not_need_completion_notification(self):
        started_b, release_b = Event(), Event()

        class Loader(_Loader):
            def prefetch_block(self, key, block):
                if key == 'B':
                    started_b.set()
                    if not release_b.wait(5):
                        raise RuntimeError('controlled read was not released')
                return super().prefetch_block(key, block)

        plan = _plan(('A', 'B'))
        queue = ReadAhead(plan, Loader({'A': 1, 'B': 1}, {'A': 0, 'B': 0}),
                          {'A': 1, 'B': 1}, 2, device_capacity=0)
        condition = queue._condition = _ConditionProbe()
        try:
            queue.start_after(-1)
            self.assertTrue(started_b.wait(2))  # A was published before B began.
            self.assertEqual(condition.completion_notifications, 0)
            self.assertEqual(queue.take('A', plan.blocks['A']), (True, 'value:A'))
            self.assertEqual(condition.consumer_waits, 0)
            self.assertEqual(condition.completion_notifications, 0)
        finally:
            release_b.set()
            queue.close()

    def test_failed_read_wakes_all_waiters_and_preserves_first_error(self):
        started, release = Event(), Event()
        first_error = RuntimeError('original read error')

        class Loader(_Loader):
            def prefetch_block(self, key, block):
                started.set()
                if not release.wait(5):
                    raise RuntimeError('controlled read was not released')
                raise first_error

        plan = _plan(('A',))
        queue = ReadAhead(plan, Loader({'A': 1}, {'A': 0}),
                          {'A': 1}, 1, device_capacity=0)
        condition = queue._condition = _ConditionProbe(expected_waiters=2)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ticket-consumer') as consumers:
            try:
                queue.start_after(-1)
                self.assertTrue(started.wait(2))
                ticket = queue.pending['A']
                futures = [consumers.submit(ticket.result) for _ in range(2)]
                self.assertTrue(condition.wait_started.wait(2))
                with condition:
                    self.assertEqual(ticket.waiters, 2)
                release.set()
                for future in futures:
                    self.assertIs(future.exception(2), first_error)
                self.assertEqual(ticket.waiters, 0)
                self.assertEqual(condition.completion_notifications, 1)
                with self.assertRaises(RuntimeError) as raised:
                    queue.take('A', plan.blocks['A'])
                self.assertIs(raised.exception, first_error)
                self.assertEqual(queue.reserved_bytes, 0)
            finally:
                release.set()
                queue.close()


if __name__ == '__main__':
    unittest.main()
