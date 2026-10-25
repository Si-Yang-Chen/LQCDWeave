"""Numerical bindings can survive different programs without retaining inputs."""
from types import SimpleNamespace
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from correlator.planning import test_cfg_plan as cfg_fixtures
from lqcdweave.contraction.execution import scene_runtime
from lqcdweave.contracts.correlator.api import StorageEvent


def exercise(test, backend):
    selected, preparations, _ = cfg_fixtures.CfgPlanTests.dense_fixture(backend.__name__)
    state = selected.state
    reference = {key: prep.recipe.constants[0].array() for key, prep in preparations.items()}
    cache, calls, owners = {}, [], []
    contract_index = next(i for i, action in enumerate(state.actions) if action.kind == 'contract')
    # These are protocol events: callbacks record their positions, while each
    # numerical result is checked against a separate NumPy matrix product.
    patterns = ((), (StorageEvent(contract_index, 'release', 'boundary'),),
                (StorageEvent(len(state.actions), 'release', 'final'),))
    with (patch.object(scene_runtime, 'bind_dense_batch', wraps=scene_runtime.bind_dense_batch) as batches,
          patch.object(scene_runtime, 'bind_dense_action', wraps=scene_runtime.bind_dense_action) as actions):
        for scale, events in enumerate(patterns, 1):
            tables = scene_runtime.bind_scene_execution(state, selected.kernel_batches, backend,
                events=events, shared_residency=True, kernel_cache=cache)
            current = {key: backend.array(value * scale) for key, value in reference.items()}
            owners.extend(weakref.ref(value) for value in current.values())
            recorded = []
            execution = SimpleNamespace(
                source=lambda key: None, materialize=lambda key, source: current[key],
                materializes_owned=lambda key: False,
                begin_action=lambda index, timed: recorded.append(('before', index)),
                end_action=lambda index, timed: recorded.append(('after', index)))
            arrays = {}
            values = tables.program.run(execution, arrays)
            for term, value in zip(state.problem.terms, values, strict=True):
                value = backend.asnumpy(value) if backend is not np else value
                expected = reference[term.inputs[0]] @ reference[term.inputs[1]] * scale**2
                np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-12)
            expected_events = [('before', 0)] + [('after', event.action_index) for event in events]
            test.assertEqual(recorded, expected_events)
            calls.append(tables.program)
            del current, arrays, values, value, execution, tables
        test.assertEqual(batches.call_count, len(selected.kernel_batches))
        test.assertEqual(actions.call_count, sum(
            action.kind == 'contract' and i not in {j for batch in selected.kernel_batches
                                                     for j in batch.action_indices}
            for i, action in enumerate(state.actions)))
    test.assertEqual(len({id(program) for program in calls}), len(patterns))
    gc.collect()
    test.assertTrue(all(owner() is None for owner in owners))


class SceneKernelReuseTests(unittest.TestCase):
    def test_event_programs_share_batch_kernels_and_release_current_arrays(self):
        exercise(self, np)


class CuPySceneKernelReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
        except ImportError as error:
            raise unittest.SkipTest('CuPy is unavailable') from error
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise unittest.SkipTest('CUDA device is unavailable')
        cls.backend = cp

    def test_event_programs_share_batch_kernels_and_release_current_arrays(self):
        exercise(self, self.backend)


if __name__ == '__main__':
    unittest.main()
