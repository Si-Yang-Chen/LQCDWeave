"""Scalar 16/32 CPU source/driver protocols and opt-in real CUDA gates."""
import ctypes
import hashlib
import os
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern, device_allocation_bytes
from lqcdweave.contraction.execution import scalar_reduction_group as module
from lqcdweave.contraction.execution import reduction_group_runtime as runtime
from lqcdweave.contraction.execution.raw_arguments import _BoundDriverArguments
from lqcdweave.contraction.execution.reduction_kernel import _kernel, reduction_sources
from lqcdweave.contraction.execution import bound_program, scene_runtime
from lqcdweave.contraction.execution.binding_cache import ExecutionBindingCache
from lqcdweave.contraction.execution.storage_runtime import replay_scene_storage
from correlator.execution.test_scalar_reduction_group import _SourceBackend, _reference, _allowance
from correlator.execution.test_scalar_group_execution import Backend, ArrayLoader, prepared, execution


def tasks_for(count, dtype, shape=(2, 3, 7, 13)):
    rng = np.random.default_rng(793)
    tasks = []
    for task in range(count):
        pair = []
        for side in range(2):
            value = (rng.normal(size=shape) + (task+1)/16).astype(dtype)
            if np.dtype(dtype).kind == 'c':
                value += (1j*rng.normal(size=shape)).astype(dtype)
            if task % 3 == 1:
                value = value[..., ::-1]
            elif task % 3 == 2:
                value = np.broadcast_to(value[:1], shape)
            pair.append(value if side == 0 else value.transpose(2, 3, 0, 1))
        tasks.append(tuple(pair))
    return tasks


class ScalarGroupWidth32Tests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()
        runtime.FAILED_SUBMISSIONS.clear()

    def test_original_eight_source_bytes_and_capacity_buckets(self):
        # Frozen a0421415 source hashes, not a second copy of the generator.
        expected = {
            'float32': ('bbefb879922614562c7f653260b4d081c37951c1d3ab37c64752fe33876928bc',
                        '389fa5e39515aed12f554cc46bb82ac804ee173ac947983e908502a4f7f1a66e'),
            'complex64': ('1f78eb942c978176a1871036bb3ca4525afff592c26825d61417fcfbee41e3ef',
                          'c43775fad2f4e2acef2c86ec21bf65ab74633a31727afc7c2e630795a5116594'),
        }
        for dtype, hashes in expected.items():
            sources = module.scalar_group_sources(('i', 'i'), ((513,),)*2, dtype)
            self.assertEqual(tuple(hashlib.sha256(s.encode()).hexdigest() for s in sources), hashes)
        backend = _SourceBackend(metadata_only=True)
        for count, capacity in ((2, 8), (8, 8), (9, 16), (16, 16), (17, 32), (32, 32)):
            run = module.bind_scalar_reduction_group('i,i->', ((513,),)*2,
                np.complex64, (8216,)*count, backend, share_small_outputs=True)
            self.assertEqual(run.capacity, capacity)
            self.assertEqual(run.parameter_bytes, ((capacity*4+2)*8, (capacity+2)*8))
            self.assertEqual(run.shared_output_bytes, count*8)
            self.assertEqual(run.output_owner_count, 1)
            run([(np.ones(513, np.complex64),)*2]*count)
        self.assertEqual(len(backend.sources), 6)  # Two kernels per capacity, not per count.
        for rank, count in ((5, 32), (11, 16), (23, 8)):
            labels = 'abcdefghijklmnopqrstuvwxyz'[:rank]
            self.assertIsNone(module.bind_scalar_reduction_group(f'{labels},{labels}->',
                ((513,)+(1,)*(rank-1),)*2, np.complex64, (8216,)*count, backend))
        largest = module.bind_scalar_reduction_group('i,i->', ((262145,),)*2,
            np.complex64, (2*262145*8+8,)*32, backend)
        self.assertEqual((largest.blocks, largest.partial_bytes), (1024, 512*1024))

    def test_16_and_32_partial_final_bytes_independent_tree_and_old_eight(self):
        equation = parse_axis_pattern('abcd,cdab->').normalized()
        for dtype in (np.float32, np.complex64):
            for count in (16, 32):
                tasks = tasks_for(count, dtype)
                shapes = tuple(value.shape for value in tasks[0])
                for allowance in (_allowance(shapes, dtype), np.dtype(dtype).itemsize*3):
                    backend = _SourceBackend()
                    run = module.bind_scalar_reduction_group(equation, shapes, dtype,
                        (allowance,)*count, backend, share_small_outputs=True)
                    outputs = run(tasks)
                    partials = backend.partial_snapshot.copy()
                    self.assertEqual(len(backend.launches), 2)
                    self.assertLessEqual(run.workspace_bytes, run.workspace_allowance)
                    for i, (values, result) in enumerate(zip(tasks, outputs, strict=True)):
                        partial, expected = _reference(*values, equation.inputs, run.blocks)
                        self.assertEqual(partials[i].tobytes(), partial.tobytes())
                        self.assertEqual(result.tobytes(), expected.tobytes())
                    old = module.bind_scalar_reduction_group(equation, shapes, dtype,
                        (allowance,)*8, backend, share_small_outputs=True)
                    for offset in range(0, count, 8):
                        references = old(tasks[offset:offset+8])
                        self.assertEqual(backend.partial_snapshot.tobytes(), partials[offset:offset+8].tobytes())
                        self.assertEqual(b''.join(v.tobytes() for v in references),
                                         b''.join(v.tobytes() for v in outputs[offset:offset+8]))
                    owner = outputs[0].base
                    self.assertEqual([v.ctypes.data-owner.ctypes.data for v in outputs],
                                     list(range(0, count*np.dtype(dtype).itemsize, np.dtype(dtype).itemsize)))
                    ref = weakref.ref(owner)
                    last = outputs[-1]
                    del owner, outputs, result
                    self.assertIsNotNone(ref())
                    del last
                    self.assertIsNone(ref())

    def test_actual_driver_parameter_table_32_refreshes_addresses_and_signed_strides(self):
        backend = _SourceBackend(metadata_only=True)
        calls, bindings = [], []

        def binder(_backend, kernel, grid, block, template, positions, *, pointer_getter):
            fake = SimpleNamespace(kernel=SimpleNamespace(ptr=len(bindings)+1))
            def driver(function, *args):
                address = args[-2]
                table = (ctypes.c_uint64*len(template)).from_address(address)
                calls.append((grid, tuple(ctypes.c_uint64.from_address(p).value for p in table)))
            bound = _BoundDriverArguments(fake, grid, block, object, template, positions,
                pointer_getter, launch=driver, ensure_context=lambda: None,
                current_stream=lambda: SimpleNamespace(ptr=17))
            bindings.append(bound)
            return lambda pointers, result: bound(pointers, SimpleNamespace(data=SimpleNamespace(ptr=result.ctypes.data)))

        tasks = tasks_for(32, np.complex64)
        shapes = tuple(v.shape for v in tasks[0])
        with patch.object(module, 'bind_pointer_arguments', side_effect=binder):
            run = module.bind_scalar_reduction_group('abcd,cdab->', shapes, np.complex64,
                (_allowance(shapes, np.complex64),)*32, backend, share_small_outputs=True)
            old = run(tasks)
            fresh = tasks_for(32, np.complex64)
            new = run(fresh)
        self.assertEqual(len(bindings), 2)
        self.assertEqual(len(calls), 4)
        self.assertEqual(run.parameter_bytes, (2576, 272))
        for round_index, (current, outputs) in enumerate(((tasks, old), (fresh, new))):
            first, finish = calls[2*round_index][1], calls[2*round_index+1][1]
            for i, pair in enumerate(current):
                self.assertEqual(first[i*10:i*10+2], tuple(v.ctypes.data for v in pair))
                self.assertEqual(tuple(ctypes.c_int64(v).value for v in first[i*10+2:i*10+10]),
                                 pair[0].strides + pair[1].strides)
            self.assertEqual(first[-2], run.blocks)
            self.assertEqual(first[-1], finish[0])
            self.assertEqual((finish[-1], *finish[1:32]), tuple(v.ctypes.data for v in outputs))
        self.assertNotEqual(calls[0][1][0], calls[2][1][0])
        bad = list(fresh); bad[-1] = (fresh[-1][0][:-1], fresh[-1][1])
        with self.assertRaisesRegex(ValueError, 'geometry'):
            run(bad)
        bad[-1] = (fresh[-1][0].astype(np.complex128), fresh[-1][1])
        with self.assertRaisesRegex(ValueError, 'geometry'):
            run(bad)
        invalid = SimpleNamespace(shape=shapes[0], dtype=np.dtype('complex64'),
            strides=fresh[-1][0].strides, data=SimpleNamespace(ptr=0))
        bad[-1] = (invalid, fresh[-1][1])
        with self.assertRaisesRegex(ValueError, 'pointer'):
            run(bad)
        self.assertEqual(len(calls), 4)

    def test_32_first_middle_view_failure_and_failed_sync_keep_all_owners(self):
        for fail_at in (1, 17, 32):
            class Broken(np.ndarray):
                calls = 0
                def __getitem__(self, key):
                    type(self).calls += 1
                    if type(self).calls == fail_at:
                        raise RuntimeError('view failed')
                    return super().__getitem__(key)
            backend = _SourceBackend(metadata_only=True)
            original_empty = backend.empty
            backend.empty = lambda shape, dtype: (original_empty(shape, dtype).view(Broken)
                if np.dtype(dtype) == np.dtype('complex64') else original_empty(shape, dtype))
            backend.synchronize.side_effect = RuntimeError('sync failed')
            run = module.bind_scalar_reduction_group('i,i->', ((513,),)*2,
                np.complex64, (8216,)*32, backend, share_small_outputs=True)
            partial = np.empty(run.partial_bytes//16, np.complex128)
            pool = object()
            device = SimpleNamespace(backend=backend, pool=pool, reduction_partial=lambda *_: partial)
            tasks = tuple((np.ones(513, np.complex64),)*2 for _ in range(32))
            with patch('lqcdweave.contraction.execution.device.active_device_execution', return_value=device):
                try:
                    run(tasks)
                except RuntimeError as error:
                    self.assertIn('view failed', str(error))
                    error.__traceback__ = None
                else:
                    self.fail('missing failure')
            held = runtime.FAILED_SUBMISSIONS.pop()
            self.assertIs(held[1][0], tasks[0][0])
            self.assertEqual(len(held[1]), 64)
            self.assertIs(held[2], partial)
            self.assertIs(held[4], pool)
            self.assertEqual(held[5].nbytes, 256)
            self.assertEqual(len(held[3]), fail_at-1)
            self.assertEqual(backend.launches, [])

    def test_32_borrow_miss_submission_failure_and_all_live_subset_envelopes(self):
        backend = _SourceBackend(metadata_only=True)
        run = module.bind_scalar_reduction_group('i,i->', ((513,),)*2,
            np.complex64, (8216,)*32, backend, share_small_outputs=True)
        tasks = tuple((np.ones(513, np.complex64),)*2 for _ in range(32))
        pool = object()
        device = SimpleNamespace(backend=backend, pool=pool, reduction_partial=lambda *_: None)
        with patch('lqcdweave.contraction.execution.device.active_device_execution', return_value=device):
            values = run(tasks)
            self.assertEqual(len(backend.allocations), 2)
            self.assertEqual(run.partial_bytes, 32*3*16)
            self.assertLessEqual(32*1024*16, 8*1024**2)
            for which in (1, 2):
                backend.fail_at = len(backend.launches)+which
                backend.synchronize.side_effect = RuntimeError('sync failed')
                with self.assertRaisesRegex(RuntimeError, 'submission failed'):
                    run(tasks)
                held = runtime.FAILED_SUBMISSIONS.pop()
                self.assertIs(held[4], pool)
                self.assertEqual(len(held[3]), 32)
                self.assertEqual(held[5].nbytes, 256)
                self.assertEqual(held[2].nbytes, run.partial_bytes)
        # Any subset is characterized solely by its number of live scalars.
        # Logical births/read/reused slots select subsets, not a different fee.
        for dtype in (np.float32, np.complex64):
            for count in (16, 32):
                for alive in range(1, count+1):
                    self.assertLessEqual(device_allocation_bytes(count*np.dtype(dtype).itemsize),
                                         alive*device_allocation_bytes(np.dtype(dtype).itemsize))

    def test_public_32_replay_current_values_and_partial_reuse_whole_fallback(self):
        plan, inputs = prepared(count=32, dtype=np.complex64)
        backend, cache = Backend(), ExecutionBindingCache()
        original = replace(plan, scalar_reduction_groups=())
        with patch.object(bound_program, 'bind_output_layout', return_value=None):
            for scale in (1, 3):
                current = {key: value*scale for key, value in inputs.items()}
                actual = replay_scene_storage(plan, ArrayLoader(current, backend), backend=backend,
                    execution_cache=cache, io_overlap=False)
                expected = replay_scene_storage(original, ArrayLoader(current, backend), backend=backend,
                    io_overlap=False)
                self.assertEqual(tuple(v.tobytes() for v in actual), tuple(v.tobytes() for v in expected))
            state = replace(plan.state, actions=tuple(replace(a, reused=True) if a.kind == 'load' else a
                                                     for a in plan.state.actions))
            tables = scene_runtime.bind_scene_execution(state, (), backend, events=(), shared_residency=True,
                scalar_reduction_groups=plan.scalar_reduction_groups)
            self.assertEqual(next(iter(tables.scalar_starts.values())).kernel.task_count, 32)
            binding = {key: i for i, key in enumerate(state.catalog)}
            store = {binding[key]: value for key, value in inputs.items()}
            class Arrays:
                def _dense_call_context(self):return store, binding
                def __getitem__(self, key):return store[binding[key]]
            actual = tables.program.run(execution(replace(plan, state=state), tables, inputs), Arrays())
            actions = list(state.actions)
            index = plan.scalar_reduction_groups[0].action_indices[17]
            actions[index] = replace(actions[index], reused=True)
            partial = replace(state, actions=tuple(actions))
            fallback = scene_runtime.bind_scene_execution(partial, (), backend, events=(), shared_residency=True,
                scalar_reduction_groups=plan.scalar_reduction_groups)
            self.assertEqual(fallback.scalar_groups, ())
            retained = store[binding[actions[index].tensor]]
            repeated = fallback.program.run(execution(replace(plan, state=partial), fallback, inputs), Arrays())
            self.assertIs(store[binding[actions[index].tensor]], retained)
            self.assertEqual(tuple(v.tobytes() for v in actual), tuple(v.tobytes() for v in repeated))


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE') == '1',
                     'explicit CUDA gate; no device probe in CPU suite')
class ScalarGroupWidth32CuPyTests(unittest.TestCase):
    def test_public_32_storage_replay_binds_and_matches_original(self):
        """Exercise both new 16/32 capacities through public storage replay."""
        import cupy as cp
        for count in (16, 32):
            for dtype in (np.float32, np.complex64):
                plan, host = prepared(count=count, dtype=dtype)
                cache = ExecutionBindingCache()
                original = replace(plan, scalar_reduction_groups=())
                previous, held = None, []
                for scale in (1, 3):
                    arrays = {key: cp.asarray(value*scale) for key, value in host.items()}
                    arrays = {key: (value[::-1] if i % 3 == 1 else
                                    cp.broadcast_to(value[:1], value.shape) if i % 3 == 2 else value)
                              for i, (key, value) in enumerate(arrays.items())}
                    held.append(arrays)
                    pointers = tuple(value.data.ptr for value in arrays.values())
                    if previous is not None:self.assertNotEqual(pointers, previous)
                    previous = pointers
                    actual = replay_scene_storage(plan, ArrayLoader(arrays, cp), backend=cp,
                        execution_cache=cache, io_overlap=False)
                    expected = replay_scene_storage(original, ArrayLoader(arrays, cp), backend=cp, io_overlap=False)
                    tables = [entry[2] for entry in cache._binding[2].values()
                              if isinstance(entry[2], scene_runtime.BoundSceneExecution)]
                    self.assertEqual(len(tables), 1)
                    self.assertEqual(len(tables[0].scalar_starts), 1)
                    run = next(iter(tables[0].scalar_starts.values())).kernel
                    self.assertEqual((run.task_count, run.capacity, run.output_owner_count), (count, count, 1))
                    self.assertEqual(run.shared_output_bytes, count*np.dtype(dtype).itemsize)
                    self.assertEqual(run.output_view_method, 'inplace-shape')
                    for left, right in zip(actual, expected, strict=True):
                        self.assertEqual(cp.asnumpy(left).tobytes(), cp.asnumpy(right).tobytes())

    def test_32_members_partial_final_and_current_pointer_bytes(self):
        """Compare original kernels with both new 16/32 capacities bytewise."""
        import cupy as cp
        equation = parse_axis_pattern('abcd,cdab->').normalized()
        for count in (16, 32):
            for dtype in (np.float32, np.complex64):
                host = tasks_for(count, dtype)
                shapes = tuple(v.shape for v in host[0])
                wide = np.dtype('complex128' if np.dtype(dtype).kind == 'c' else 'float64')
                run = module.bind_scalar_reduction_group(equation, shapes, dtype,
                    (_allowance(shapes, dtype),)*count, cp, share_small_outputs=True)
                self.assertEqual(run.capacity, count)
                self.assertEqual(run.output_view_method, 'inplace-shape')
                previous = None
                held = []
                for scale in (1, 3):
                    # Apply layouts on device: cp.asarray would otherwise discard signed/zero strides.
                    tasks = []
                    for i, pair in enumerate(host):
                        values = tuple(cp.asarray(v*scale) for v in pair)
                        if i % 3 == 1:values = tuple(v[..., ::-1] for v in values)
                        if i % 3 == 2:values = tuple(cp.broadcast_to(v[:1], v.shape) for v in values)
                        tasks.append(values)
                    held.append(tasks)
                    if previous is not None:self.assertNotEqual(tasks[0][0].data.ptr, previous)
                    previous = tasks[0][0].data.ptr
                    partials = cp.empty((count, run.blocks), dtype=wide)
                    device = SimpleNamespace(backend=cp, reduction_partial=lambda *_: partials)
                    with patch('lqcdweave.contraction.execution.device.active_device_execution', return_value=device):
                        results = run(tasks)
                    for i, (values, value) in enumerate(zip(tasks, results, strict=True)):
                        first, finish, _ = reduction_sources(equation.inputs, (), shapes,
                            tuple(v.strides for v in values), np.dtype(dtype), run.blocks)
                        original = cp.empty(run.blocks, dtype=wide)
                        expected = cp.empty((), dtype=dtype)
                        _kernel(cp.RawKernel, first, 'ed_precise_partial')((run.blocks,), (256,), (*values, original))
                        _kernel(cp.RawKernel, finish, 'ed_precise_finish')((1,), (256,), (original, expected))
                        self.assertEqual(cp.asnumpy(partials[i]).tobytes(), cp.asnumpy(original).tobytes())
                        self.assertEqual(cp.asnumpy(value).tobytes(), cp.asnumpy(expected).tobytes())
                    owner = results[0].base
                    self.assertTrue(all(v.base is owner for v in results))
                    self.assertEqual([v.data.ptr-owner.data.ptr for v in results],
                        list(range(0, count*np.dtype(dtype).itemsize, np.dtype(dtype).itemsize)))



if __name__ == '__main__':
    unittest.main()
