"""Saved-plan identity, independent resource progress and completion gates."""
from dataclasses import dataclass
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.plan_cursor import (
    WAITING, drive_plan_cursors, plan_operations,
)
from lqcdweave.contraction.execution import bound_program, scene_runtime
from lqcdweave.contraction.execution.storage_runtime import _BlockExecution
from correlator.execution.test_debug_execution import _make_dense_execution


@dataclass(frozen=True)
class Operation:
    name: str
    resource: str
    dependencies: tuple = ()


class PlanCursorTests(unittest.TestCase):
    def test_blocked_resource_does_not_block_io_and_release_waits_for_completion(self):
        # The saved order puts a consumer before IO on a different resource.
        # No new global sort may be needed to progress the independent queues.
        saved = (
            Operation('compute', 'gpu', ('upload',)),
            Operation('release', 'cpu', ('compute',)),
            Operation('read', 'io'),
            Operation('upload', 'upload', ('read',)),
        )
        completed, pending, submitted, log = set(), [], [], []
        owner = {'input': object()}

        def ready(operation):
            return all(dep in completed for dep in operation.dependencies)

        def submit(resource, operation):
            self.assertIs(operation, next(op for op in saved if op.name == operation.name))
            self.assertEqual(resource, operation.resource)
            self.assertTrue(ready(operation))
            submitted.append(operation.name)
            log.append(('submit', operation.name))
            if operation.name == 'release':
                self.assertIn('compute', completed)
                del owner['input']
                completed.add('release')
            else:
                # Submission alone must not permit a consumer or buffer release.
                self.assertIn('input', owner)
                pending.append(operation)

        def wait():
            self.assertTrue(pending, 'blocked without an outstanding completion')
            operation = pending.pop(0)
            completed.add(operation.name)
            log.append(('complete', operation.name))

        cursors = {resource: plan_operations(saved,
            belongs=lambda op, resource=resource: op.resource == resource, ready=ready)
            for resource in ('gpu', 'cpu', 'io', 'upload')}
        drive_plan_cursors(cursors, submit=submit, wait=wait)
        self.assertEqual(submitted, ['read', 'upload', 'compute', 'release'])
        self.assertEqual(log, [('submit', 'read'), ('complete', 'read'),
            ('submit', 'upload'), ('complete', 'upload'),
            ('submit', 'compute'), ('complete', 'compute'), ('submit', 'release')])
        self.assertFalse(owner)

    def test_readiness_keeps_head_order_and_never_eagerly_copies_steps(self):
        touched = []
        a, b = object(), object()

        def source():
            touched.append('a')
            yield a
            touched.append('b')
            yield b

        admitted = set()
        cursor = plan_operations(source(), ready=lambda op: op in admitted)
        self.assertEqual(touched, [])
        self.assertIs(next(cursor), WAITING)
        admitted.add(b)
        self.assertIs(next(cursor), WAITING)
        self.assertEqual(touched, ['a'])
        admitted.add(a)
        self.assertIs(next(cursor), a)
        self.assertIs(next(cursor), b)
        with self.assertRaises(StopIteration):
            next(cursor)

    def test_submission_failure_closes_cursors_without_submitting_later_work(self):
        closed, submitted = [], []

        def source(name):
            try:
                yield name
                yield 'must-not-submit'
            finally:
                closed.append(name)

        def submit(resource, operation):
            submitted.append(operation)
            if resource == 'b':
                raise RuntimeError('device enqueue failed')

        with self.assertRaisesRegex(RuntimeError, 'device enqueue failed'):
            drive_plan_cursors({'a': source('a'), 'b': source('b')},
                submit=submit, wait=lambda: self.fail('unexpected wait'))
        self.assertEqual(submitted, ['a', 'b'])
        self.assertCountEqual(closed, ['a', 'b'])

    def test_all_submitted_does_not_claim_device_completion(self):
        pending = []
        drive_plan_cursors({'gpu': iter(('kernel',))},
            submit=lambda resource, operation: pending.append(operation),
            wait=lambda: self.fail('completion drain belongs to the owner scope'))
        self.assertEqual(pending, ['kernel'])

    def test_blocked_without_events_reports_owner_error_instead_of_spinning(self):
        def no_events():
            raise RuntimeError('no pending event can satisfy the saved dependency')
        with self.assertRaisesRegex(RuntimeError, 'no pending event'):
            drive_plan_cursors({'cpu': plan_operations((object(),), ready=lambda op: False)},
                submit=lambda *args: self.fail('not ready'), wait=no_events)

    def test_existing_replay_uses_cursor_with_current_data_and_original_lifecycle(self):
        _, path, loader, expected = _make_dense_execution()
        plan = path.plan
        tables = scene_runtime.bind_scene_execution(plan.state, plan.kernel_batches, np,
            events=plan.events, retained=plan.retained_after)
        original_calls = tables.program.calls
        cursor_views = []

        def cursor(calls):
            self.assertIs(calls, original_calls)
            cursor_views.append(calls)
            return plan_operations(calls)

        results = []
        for scale in (1, 2):
            if scale == 2:
                loader.arrays = {key: value * 2 for key, value in loader.arrays.items()}
            execution = _BlockExecution(plan, loader, np, execution_tables=tables, io_overlap=False)
            try:
                with (patch.object(bound_program, 'plan_operations', side_effect=cursor),
                      patch.object(scene_runtime, 'bind_scene_execution',
                                   side_effect=AssertionError('rebound while running'))):
                    results.append(tables.program.run(execution, {})[0])
            finally:
                execution.close()
        self.assertEqual(len(cursor_views), 2)
        self.assertIs(tables.program.calls, original_calls)
        np.testing.assert_allclose(results[0], expected)
        np.testing.assert_allclose(results[1], expected * 4)
        self.assertFalse(np.shares_memory(*results))


if __name__ == '__main__':
    unittest.main()
