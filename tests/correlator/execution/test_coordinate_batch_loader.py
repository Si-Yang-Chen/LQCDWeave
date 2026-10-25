"""One task-scoped device loader owns reads retained across coordinate batches.

The CuPy protocol is simulated with NumPy; no GPU is imported or initialized.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from correlator.execution.test_device import _FakeCupy, _Tracker
from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionProblem, ContractionTerm, SourceSpec,
    SourcePreparation, StorageBlock, TensorPreparation, TensorSpec,
)
from lqcdweave.contraction.planning.api import (
    ContractionState, CostProfile, join_storage_batches, plan_scene_storage,
)
from lqcdweave.contraction.execution.device import active_device_execution, device_execution
from lqcdweave.contraction.execution import storage_runtime


class Inputs:
    def __init__(self, arrays, preparations, specs, *, prefetch):
        self.arrays, self.preparations, self.specs = arrays, preparations, specs
        self.prefetch = prefetch
        self.reads, self.prefetches, self.closed = [], [], []

    def read_block(self, key, block):
        self.reads.append(key)
        return self.arrays[key].copy()

    def close_block(self, key, value):
        self.closed.append(key)

    def close_all(self):
        pass

    def can_prefetch_block(self, key, block):
        return self.prefetch

    def prefetch_host_bytes(self, key, block):
        return 2 * block.host_bytes

    def prefetch_block(self, key, block):
        self.prefetches.append(key)
        return self.arrays[key].copy()

    def finish_prefetch(self, key, block, value):
        return value

    def close_prefetch(self, key, value):
        self.closed.append(key)


def fixture(backend='cupy', *, prefetch=False, tail=False):
    profile = CostProfile()
    arrays = {'raw:source': np.arange(1, 6, dtype=np.complex128)}
    arrays.update({f'raw:sink:{i}': np.arange(5, dtype=np.complex128) + (i + 1) * (1 + .2j)
                   for i in range(3 if tail else 4)})
    closure, specs, plans = {}, {}, []
    for key in arrays:
        prepared = 'prepared:' + key
        closure[prepared] = SourcePreparation((key,), recipe=TensorPreparation('identity', (key,),
            compute_dtype='complex128'))
        specs[prepared] = TensorSpec((5,), np.complex128)
    for index in range(2):
        source, sink = (('source', 'sink') if tail else
                        (f'batch:{index}:source', f'batch:{index}:sink'))
        width = 1 if tail and index == 1 else 2
        members = tuple(f'prepared:raw:sink:{i}' for i in range(2 * index, 2 * index + width))
        raw = ('raw:source', *(f'raw:sink:{i}' for i in range(2 * index, 2 * index + width)))
        selected = {source: TensorSpec((5,), np.complex128), sink: TensorSpec((width, 5), np.complex128)}
        sources = {key: SourceSpec(key, 0, read_bytes=0) for key in selected}
        problem = ContractionProblem(selected, (ContractionTerm((sink, source),
            AxisPattern(((0, 1), (1,)), (0,))),), sources, backend=backend)
        preps = {source: closure['prepared:raw:source'],
                 sink: SourcePreparation(raw[1:], recipe=TensorPreparation('stack', members,
                    compute_dtype='complex128'))}
        closure.update(preps)
        specs.update(selected)
        blocks = {key: StorageBlock('fixture', key, arrays[key].nbytes, host_bytes=arrays[key].nbytes,
                  device_bytes=arrays[key].nbytes if backend == 'cupy' else 0) for key in raw}
        state = ContractionState.initial(problem)
        for key in selected:
            state = state.load(key, profile, 1 << 20)
        state = state.contract(0, (0, 1), profile, 1 << 20)
        plans.append(plan_scene_storage(state, blocks, preps, profile=profile,
                                       memory_limit=1 << 20, host_memory_limit=1 << 20))
    plans = join_storage_batches(plans)
    return plans, Inputs(arrays, closure, specs, prefetch=prefetch)


class CoordinateBatchLoaderTests(unittest.TestCase):
    def run_fixture(self, backend_name, prefetch, *, streaming=False, tail=False):
        plans, loader = fixture(backend_name, prefetch=prefetch, tail=tail)
        tracker = _Tracker()
        backend = _FakeCupy(tracker) if backend_name == 'cupy' else np
        values = []

        def consume(index, outputs):
            device = active_device_execution()
            if device is None:
                host = tuple(outputs)
            else:
                tracker.download_depth += 1
                try:
                    host = device.download(outputs)
                finally:
                    tracker.download_depth -= 1
            values.extend(np.array(item, copy=True) for item in host)

        with device_execution(backend, device_memory_limit=1 << 20):
            storage_runtime.replay_storage_batches(iter(plans) if streaming else plans,
                loader, consume, backend=backend, io_overlap=prefetch, streaming=streaming)
        for index, value in enumerate(values):
            width = 1 if tail and index == 1 else 2
            expected = np.array([loader.arrays[f'raw:sink:{i}'] @ loader.arrays['raw:source']
                                 for i in range(2 * index, 2 * index + width)])
            np.testing.assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
        all_reads = loader.reads + loader.prefetches
        self.assertEqual(sorted(all_reads), sorted(loader.arrays))
        self.assertEqual(sorted(loader.closed), sorted(loader.arrays))
        self.assertEqual(all_reads.count('raw:source'), 1)
        if prefetch:
            self.assertTrue(loader.prefetches)

    def test_cupy_two_batches_keep_one_owner_without_prefetch(self):
        self.run_fixture('cupy', False)

    def test_cupy_two_batches_keep_one_owner_with_prefetch(self):
        self.run_fixture('cupy', True)

    def test_short_tail_rebinds_same_named_preparations_without_replacing_raw_owner(self):
        for backend in ('numpy', 'cupy'):
            for prefetch in (False, True):
                with self.subTest(backend=backend, prefetch=prefetch):
                    self.run_fixture(backend, prefetch, tail=True)

    def test_numpy_task_batches_remain_numerically_equivalent(self):
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                self.run_fixture('numpy', prefetch)

    def test_streaming_does_not_receive_a_full_task_binding_marker(self):
        with patch.object(storage_runtime, '_bind_batch_loader', create=True,
                          side_effect=AssertionError('streaming bound a complete task')):
            self.run_fixture('numpy', False, streaming=True)

    def test_legacy_conflicting_names_keep_the_original_per_plan_binding(self):
        for conflict in ('preparation', 'shape'):
            with self.subTest(conflict=conflict):
                first = SimpleNamespace(
                    preparations={'role': SourcePreparation(('raw:first',))},
                    state=SimpleNamespace(catalog={'role': TensorSpec((5,), np.complex128)}))
                second = SimpleNamespace(
                    preparations={'role': SourcePreparation(
                        ('raw:second',) if conflict == 'preparation' else ('raw:first',))},
                    state=SimpleNamespace(catalog={'role': TensorSpec(
                        (3,) if conflict == 'shape' else (5,), np.complex128)}))
                legacy_loader, sentinel = object(), object()
                with patch.object(storage_runtime, '_bind_loader_for_plan', return_value=sentinel) as bind:
                    actual = storage_runtime._bind_batch_loader(legacy_loader, (first, second), np)
                self.assertIs(actual, sentinel)
                bind.assert_called_once_with(legacy_loader, first, np)


if __name__ == '__main__':
    unittest.main()
