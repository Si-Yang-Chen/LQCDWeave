"""Avoid advancing read-ahead on every compute-only Action."""
import unittest
from dataclasses import replace

import numpy as np

from lqcdweave.contraction.execution.storage_runtime import _BlockExecution
from correlator.execution.test_io_overlap import _ProtocolLoader, _two_dot_plan


class _ReadAheadSpy:
    def __init__(self):
        self.started = []
        self.taken = []

    def take(self, key, block):
        self.taken.append((key, block))
        return False, None

    def start_after(self, index):
        self.started.append(index)


class StorageRuntimeReadAheadKickTests(unittest.TestCase):
    def test_initial_kick_and_read_advance_skip_compute_only_actions(self):
        selected, arrays = _two_dot_plan()
        # The first logical load is deliberately independent of a physical
        # read so the initial kick is exercised before the first read event.
        selected = replace(
            selected,
            events=tuple(event for event in selected.events if event.action_index != 0),
        )
        loader = _ProtocolLoader(arrays, prefetchable=())
        read_ahead = _ReadAheadSpy()
        execution = _BlockExecution(selected, loader, np, task_read_ahead=read_ahead)

        def timed(_name, function, **_metadata):
            return function()

        try:
            execution.begin_action(0, timed)
            self.assertEqual(read_ahead.started, [0])
            self.assertNotIn(0, execution.before)

            execution.begin_action(1, timed)
            self.assertEqual(read_ahead.taken[0][0], "B")
            self.assertEqual(read_ahead.started, [0, 1])

            execution.begin_action(2, timed)
            execution.end_action(2, timed)
            self.assertEqual(read_ahead.started, [0, 1])
            self.assertNotIn(2, execution.before)
            self.assertNotIn(2, execution.after)
        finally:
            execution.close()


if __name__ == "__main__":
    unittest.main()
