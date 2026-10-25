"""Compact integer schedule versus the previous independent dict/list replay."""
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import ArrayConstant, describe_coordinates
from lqcdweave.contraction.execution.api import execute_cfg_plan
from lqcdweave.contraction.execution.bound_program import BoundProgram, _after
from lqcdweave.contraction.execution.cfg_releases import _CompactReleaseTable, bind_tensor_releases
from lqcdweave.contraction.execution.cfg_runtime import _CfgBlockExecution, _NodeView
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.storage_runtime import _BlockExecution
from lqcdweave.contraction.expansion.api import restore_input_loader
from lqcdweave.contraction.planning.api import build_cfg_plan
from . import test_cfg_tensor_releases as release_fixture


def constant(rows, dtype='<i8'):
    return ArrayConstant.from_array(np.asarray(rows, dtype=dtype).reshape(-1, 2))


def previous(record):
    releases = {}
    if record is not None:
        for index, gid in record.array().tolist():
            releases.setdefault(index, []).append(gid)
    return releases


class CompactReleaseTableTests(unittest.TestCase):
    def test_read_only_no_python_row_expansion_and_lookup_order(self):
        record = constant([[0, 9], [2, 7], [2, 3], [8, 11], [8, 15]])
        table = bind_tensor_releases(record, native=True)
        self.assertIs(type(table), _CompactReleaseTable)
        self.assertTrue(table._words.readonly)
        self.assertEqual(table._words.nbytes, 16 * 5)
        with self.assertRaises(TypeError): table._words[0] = 7
        expected = previous(record)
        for index in (0, 1, 2, 2, 3, 8, 10, 0, 2, 8, -1):
            self.assertEqual(tuple(table.get(index, ())), tuple(expected.get(index, ())))
        self.assertIsNone(table.get(123))
        marker = object()
        self.assertIs(table.get(123, marker), marker)

    def test_empty_and_unsupported_keep_original_mapping(self):
        for record in (None, constant([]), constant([[2, 1], [0, 3]]),
                       constant([[1, 3]], '>i8'), constant([[1, 3]], '<f8')):
            result = bind_tensor_releases(record, native=True)
            self.assertIs(type(result), dict)
            self.assertEqual(result, previous(record))
        record = constant([[1, 3]])
        extension = SimpleNamespace(array=record.array)
        self.assertEqual(bind_tensor_releases(extension, native=True), previous(record))
        self.assertIs(type(bind_tensor_releases(record, native=False)), dict)

    def make_execution(self, rows, arrays, *, debug=False, execution_type=_CfgBlockExecution):
        with execution_debug(debug), patch.object(_BlockExecution, '__init__', return_value=None):
            return execution_type(None, None, np, tensor_releases=constant(rows), arrays=arrays)

    def test_storage_event_precedes_tensor_release_and_owner_last_reference(self):
        owner = np.arange(6)
        reference = weakref.ref(owner)
        arrays = {10: owner[:3], 11: owner[3:]}
        del owner
        execution = self.make_execution([[0, 10], [2, 11]], arrays)
        events = []
        def storage_first(execution, index, timed):
            events.append((index, tuple(execution.arrays)))
        with patch.object(_BlockExecution, 'end_action', storage_first):
            execution.end_action(0, None)
            self.assertIsNotNone(reference())
            execution.end_action(1, None)
            execution.end_action(2, None)
        self.assertEqual(events, [(0, (10, 11)), (1, (11,)), (2, (11,))])
        self.assertIsNone(reference())

    def test_duplicate_repeated_and_early_missing_release_errors_match(self):
        cases = [([[1, 2], [1, 2]], [1]), ([[1, 2]], [1, 1]),
                 ([[0, 7]], [0]), ([[1, 2], [3, 2]], [1, 3]),
                 ([[1, 2], [3, 4]], [3, 1])]
        for rows, visits in cases:
            with self.subTest(rows=rows, visits=visits):
                expected = previous(constant(rows)); old = {2: 'a', 4: 'b'}; new = old.copy()
                execution = self.make_execution(rows, new)
                with patch.object(_BlockExecution, 'end_action', return_value=None):
                    for index in visits:
                        error = None
                        try:
                            for gid in expected.get(index, ()): del old[gid]
                        except KeyError as caught: error = caught
                        if error is None: execution.end_action(index, None)
                        else:
                            with self.assertRaises(KeyError) as caught: execution.end_action(index, None)
                            self.assertEqual(caught.exception.args, error.args)
                        self.assertEqual(new, old)

    def test_debug_extension_and_unsorted_fallback(self):
        class Extension(_CfgBlockExecution): pass
        cases = [([[1, 2]], True, _CfgBlockExecution, {}),
                 ([[1, 2]], False, Extension, {}),
                 ([[2, 2], [1, 3]], False, _CfgBlockExecution, {}),
                 ([[1, 2]], False, _CfgBlockExecution, SimpleNamespace())]
        for rows, debug, cls, arrays in cases:
            execution = self.make_execution(rows, arrays, debug=debug, execution_type=cls)
            self.assertIs(type(execution.tensor_releases), dict)
            self.assertEqual(execution.tensor_releases, previous(constant(rows)))

    def test_unsorted_table_replays_identical_release_boundaries(self):
        rows = [[4, 20], [0, 10], [4, 21], [2, 15]]
        old = {key: np.asarray([key]) for key in (10, 15, 20, 21)}
        new = old.copy(); table = previous(constant(rows))
        execution = self.make_execution(rows, new)
        self.assertIs(type(execution.tensor_releases), dict)
        with patch.object(_BlockExecution, 'end_action', return_value=None):
            for index in range(5):
                for gid in table.get(index, ()): del old[gid]
                execution.end_action(index, None)
                self.assertEqual(tuple(new), tuple(old))

    def test_large_schedule_stays_compact_and_repeated_lookup_is_nonconsuming(self):
        ids = np.arange(20000, dtype=np.int64)
        table = bind_tensor_releases(ArrayConstant.from_array(np.column_stack((ids // 10, ids))), native=True)
        self.assertIs(type(table), _CompactReleaseTable)
        self.assertEqual(table._words.nbytes, 20000 * 16)
        arrays = dict.fromkeys(map(int, ids))
        for index in range(2000):
            if index % 101 == 0:
                expected = tuple(range(index * 10, index * 10 + 10))
                self.assertEqual(tuple(table.get(index, ())), expected)
                self.assertEqual(tuple(table.get(index, ())), expected)
            table.release(index, arrays)
        self.assertFalse(arrays)

    def test_terminal_release_follows_output_snapshot(self):
        arrays = {10: np.asarray([2.]), 11: np.asarray([7.])}
        execution = self.make_execution([[1, 10], [3, 11]], arrays)
        table = BoundProgram((partial(_after, index=1),), ('answer',), np, False, 3)
        with patch.object(_BlockExecution, 'end_action', return_value=None):
            outputs = table.run(execution, _NodeView({'answer': 11}, arrays))
        np.testing.assert_array_equal(outputs[0], [7.])
        self.assertEqual(arrays, {})

    def test_event_failure_preserves_pending_tensor_release(self):
        arrays = {2: object()}
        execution = self.make_execution([[1, 2]], arrays)
        with patch.object(_BlockExecution, 'end_action', side_effect=RuntimeError('storage close')):
            with self.assertRaisesRegex(RuntimeError, 'storage close'): execution.end_action(1, None)
        self.assertIn(2, arrays)
        with patch.object(_BlockExecution, 'end_action', return_value=None): execution.end_action(1, None)
        self.assertEqual(arrays, {})

    def test_two_coordinate_public_replay_matches_old_mapping_and_debug(self):
        original, values = release_fixture.CfgTensorReleaseTests().fixture()
        selected = original.templates[0].plan
        points, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 0}])
        cfg = build_cfg_plan(selected, None, points.points, {}, selected.preparations,
                             selected.state.problem.specs, prefetch_host_bytes=0, debug=True)
        self.assertTrue(np.all(cfg.instances[1].reused.array()))
        @contextmanager
        def factory(configuration):
            yield restore_input_loader(cfg.requests, {}, cfg.preparations, cfg.specs,
                                       configuration=configuration)
        results = []; traces = []
        end_action = _CfgBlockExecution.end_action
        for mode in ('compact', 'old', 'debug'):
            trace = []; output = []; executions = []
            def observe(execution, index, timed):
                if not executions or executions[-1] is not execution: executions.append(execution)
                trace.append((id(execution), index, tuple(execution.tensor_releases.get(index, ()))))
                end_action(execution, index, timed)
            def old_binding(record, *, native): return previous(record)
            with patch.object(_CfgBlockExecution, 'end_action', observe):
                if mode == 'old':
                    with patch('lqcdweave.contraction.execution.cfg_runtime.bind_tensor_releases', old_binding):
                        execute_cfg_plan(cfg, ['tiny'], factory, lambda cfg, i, values: output.append(values)
                                         if i is not None else None)
                else:
                    execute_cfg_plan(cfg, ['tiny'], factory, lambda cfg, i, values: output.append(values)
                                     if i is not None else None, debug=mode == 'debug')
            # Interpreter visits all boundaries; native program may skip empty ones.
            # A wholly reused native point with no release/event boundaries
            # need not call end_action at all; its output is still delivered.
            self.assertGreaterEqual(len(executions), 1)
            self.assertLessEqual(len(executions), 2)
            if mode == 'compact':
                self.assertTrue(any(type(execution.tensor_releases) is _CompactReleaseTable for execution in executions))
            else:
                self.assertTrue(all(type(execution.tensor_releases) is dict for execution in executions))
            releases = [(index, gids) for _, index, gids in trace if gids]
            for identity in {item[0] for item in trace}:
                visits = [index for owner, index, _ in trace if owner == identity]
                self.assertEqual(visits, sorted(set(visits)))
            traces.append(releases); results.append(np.asarray(output))
        self.assertEqual(traces[0], traces[1]); self.assertEqual(traces[0], traces[2])
        np.testing.assert_array_equal(results[0], results[1]); np.testing.assert_array_equal(results[0], results[2])
        expected = [np.einsum('ij,jk,ki->', *(values[f'{x}{i}'] for x in 'abc')) for i in range(8)]
        np.testing.assert_allclose(results[0][0], expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(results[0][0], results[0][1])


if __name__ == '__main__': unittest.main()
