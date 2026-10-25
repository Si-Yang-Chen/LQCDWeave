"""Small CPU ownership oracles; explicit CUDA gates remain separately opt-in."""
import gc
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contraction.execution import reduction_group_runtime as runtime
from lqcdweave.contraction.execution.matrix_reduction_group import bind_matrix_reduction_group as _matrix
from lqcdweave.contraction.execution.scalar_reduction_group import bind_scalar_reduction_group as _scalar
from lqcdweave.contraction.execution.reduction_kernel import _kernel
from lqcdweave.contraction.execution.operations.dense_batch import _bind_contiguous_concatenation
from lqcdweave.contraction.planning.api import audit_cfg_plan, build_cfg_plan
from lqcdweave.contracts.correlator.api import describe_coordinates, device_allocation_bytes
from correlator.execution.test_matrix_reduction_group import MatrixBackend, tasks_for
from correlator.execution.test_scalar_reduction_group import _SourceBackend


def bind_matrix_reduction_group(*args, **kwargs):
    return _matrix(*args, **kwargs, share_small_outputs=True)


def bind_scalar_reduction_group(*args, **kwargs):
    return _scalar(*args, **kwargs, share_small_outputs=True)


class SharedReductionOutputTests(unittest.TestCase):
    def tearDown(self):
        runtime.FAILED_SUBMISSIONS.clear()
        _kernel.cache_clear()

    def test_unadmitted_internal_binders_and_nonordinary_scenes_stay_independent(self):
        backend = _SourceBackend(metadata_only=True)
        run = _scalar('i,i->', ((513,),)*2, np.float32, (4108,)*4, backend)
        values = run(((np.ones(513, np.float32), np.ones(513, np.float32)),)*4)
        self.assertEqual(run.output_owner_count, 4)
        self.assertEqual(len(backend.allocations), 5)
        self.assertTrue(all(value.base is None for value in values))
        native = SimpleNamespace(kind='contract', core=None, resource=None)
        self.assertTrue(runtime._native_output_rounding(SimpleNamespace(actions=(native,))))
        for action in (SimpleNamespace(kind='sparse_bridge', core=None, resource=None),
                       SimpleNamespace(kind='contract', core=object(), resource=None),
                       SimpleNamespace(kind='load', core=None, resource=object())):
            self.assertFalse(runtime._native_output_rounding(SimpleNamespace(actions=(native, action))))

    def test_private_comparison_limit_is_consumed_only_during_binding(self):
        backend = _SourceBackend(metadata_only=True)
        args = ('i,i->', ((513,),)*2, np.float32, (4108,)*4, backend)
        with patch.object(runtime, '_SHARED_OUTPUT_MAX_BYTES', 0):
            baseline = bind_scalar_reduction_group(*args)
        candidate = bind_scalar_reduction_group(*args)
        tasks = ((np.ones(513, np.float32), np.ones(513, np.float32)),)*4
        self.assertEqual(baseline.output_owner_count, 4)
        self.assertEqual(candidate.output_owner_count, 1)
        # The restored/changed constant never changes an existing bound call.
        with patch.object(runtime, '_SHARED_OUTPUT_MAX_BYTES', 0):
            old, new = baseline(tasks), candidate(tasks)
        self.assertTrue(all(value.base is None for value in old))
        self.assertTrue(all(value.base is new[0].base for value in new))

    def test_native_shape_assignment_skips_reshape_and_keeps_adjacent_borrowing(self):
        class Tracked(np.ndarray):
            slices = 0
            shapes = 0
            reshapes = 0
            def __getitem__(self, key):
                type(self).slices += 1
                return super().__getitem__(key)
            def __setattr__(self, name, value):
                if name == 'shape':
                    type(self).shapes += 1
                return super().__setattr__(name, value)
            def reshape(self, *args, **kwargs):
                type(self).reshapes += 1
                return super().reshape(*args, **kwargs)

        class Backend(MatrixBackend):
            def empty(self, shape, dtype):
                value = super().empty(shape, dtype)
                if np.dtype(dtype) == np.dtype('complex64'):
                    value = value.view(Tracked)
                    Tracked.slices = Tracked.shapes = Tracked.reshapes = 0
                return value

        for native in (False, True):
            backend = Backend(metadata_only=True)
            self.assertFalse(runtime._native_cupy_views(backend, np.ndarray))
            with patch.object(runtime, '_native_cupy_views', return_value=native):
                run = bind_matrix_reduction_group(((1, 37), (3, 37)), np.complex64, (72,)*4, backend)
            results = run(tasks_for(1, 37, 3, np.complex64, 4))
            # NumPy implements its setter differently; count our API operations,
            # not NumPy's internal objects as evidence of CuPy's object count.
            self.assertEqual(Tracked.slices, 4)
            self.assertEqual(Tracked.shapes, 4 if native else 0)
            self.assertEqual(Tracked.reshapes, 0 if native else 4)
            self.assertEqual(run.output_view_method, 'inplace-shape' if native else 'slice-reshape')
            borrowed = _bind_contiguous_concatenation(0, (4, 3))(results, np.dtype('complex64'))
            self.assertIsNotNone(borrowed)
            self.assertTrue(all(np.shares_memory(borrowed, result) for result in results))
            self.assertEqual(borrowed.ctypes.data, results[0].ctypes.data)

    def test_native_first_and_middle_shape_failures_keep_explicit_memory_owners(self):
        for fail_at in (1, 3):
            class BrokenShape(np.ndarray):
                shapes = 0
                def __setattr__(self, name, value):
                    if name == 'shape':
                        type(self).shapes += 1
                        if type(self).shapes == fail_at:
                            raise RuntimeError('shape assignment failed')
                    return super().__setattr__(name, value)

            class Backend(_SourceBackend):
                def empty(self, shape, dtype):
                    value = super().empty(shape, dtype)
                    return value.view(BrokenShape) if np.dtype(dtype) == np.dtype('float32') else value

            backend = Backend(metadata_only=True)
            backend.synchronize.side_effect = RuntimeError('sync failed')
            with patch.object(runtime, '_native_cupy_views', return_value=True):
                run = bind_scalar_reduction_group('i,i->', ((513,),)*2, np.float32, (4108,)*4, backend)
            partial = np.empty(run.partial_bytes//8, np.float64)
            pool = object()
            device = SimpleNamespace(backend=backend, reduction_partial=lambda *_: partial, pool=pool)
            tasks = ((np.ones(513, np.float32), np.ones(513, np.float32)),)*4
            with patch('lqcdweave.contraction.execution.device.active_device_execution', return_value=device):
                try:
                    run(tasks)
                except RuntimeError as error:
                    self.assertIn('shape assignment', str(error))
                    self.assertTrue(error.__notes__)
                    error.__traceback__ = None
                else:
                    self.fail('expected shape assignment failure')
            held = runtime.FAILED_SUBMISSIONS.pop()
            self.assertIs(held[1][0], tasks[0][0])
            self.assertIs(held[2], partial)
            self.assertIs(held[4], pool)
            self.assertEqual(held[5].nbytes, 16)
            self.assertEqual(len(held[3]), fail_at-1)
            self.assertEqual(backend.launches, [])

    def test_inplace_and_reshape_protocols_keep_partial_and_final_bytes(self):
        for dtype in (np.float32, np.complex64):
            for count in (2, 4, 8):
                tasks = tasks_for(3, 37, 2, dtype, count)
                runs = []
                for native in (False, True):
                    backend = MatrixBackend()
                    with patch.object(runtime, '_native_cupy_views', return_value=native):
                        run = bind_matrix_reduction_group(((3, 37), (2, 37)), dtype, (144,)*count, backend)
                    values = run(tasks)
                    runs.append((backend.partial_snapshot.tobytes(), tuple(value.tobytes() for value in values)))
                self.assertEqual(runs[0], runs[1])

    def test_exact_boundary_address_ranges_last_reference_and_above_boundary(self):
        for dtype in (np.float32, np.complex64):
            itemsize = np.dtype(dtype).itemsize
            for count in (2, 4, 8):
                for elements in (512//(count*itemsize), 512//(count*itemsize)+1):
                    backend = MatrixBackend(metadata_only=True)
                    shapes = ((1, 37), (elements, 37))
                    allowance = elements*(itemsize + 16)
                    run = bind_matrix_reduction_group(shapes, dtype, (allowance,)*count, backend)
                    tasks = tasks_for(1, 37, elements, dtype, count)
                    outputs = run(tasks)
                    shared = count*elements*itemsize <= 512
                    self.assertEqual(run.output_owner_count, 1 if shared else count)
                    self.assertEqual(len(backend.allocations), run.output_owner_count+1)
                    self.assertLessEqual(run.workspace_bytes, run.workspace_allowance)
                    self.assertEqual(run.workspace_bytes, run.partial_bytes+count*elements*itemsize)
                    for i, value in enumerate(outputs):
                        self.assertEqual(value.shape, (1, elements))
                        value.fill(i+1)
                        for other in outputs[:i]:
                            self.assertFalse(np.shares_memory(value, other))
                    for i, value in enumerate(outputs):
                        np.testing.assert_array_equal(value, i+1)
                    if shared:
                        owner = outputs[0].base
                        self.assertTrue(all(value.base is owner for value in outputs))
                        self.assertEqual(owner.nbytes, count*elements*itemsize)
                        self.assertEqual([value.ctypes.data-owner.ctypes.data for value in outputs],
                                         list(range(0, owner.nbytes, elements*itemsize)))
                        reference = weakref.ref(owner)
                        last = outputs[-1]
                        del owner, outputs, value, other
                        gc.collect()
                        self.assertIsNotNone(reference())
                        del last
                        self.assertIsNone(reference())
                    else:
                        self.assertTrue(all(value.base is None for value in outputs))

    def test_first_and_middle_view_failures_keep_owner_inputs_partials_and_pool(self):
        for fail_at in (1, 3):
            class BrokenView(np.ndarray):
                calls = 0
                def __getitem__(self, key):
                    type(self).calls += 1
                    if type(self).calls == fail_at:
                        raise RuntimeError('view construction failed')
                    return super().__getitem__(key)

            class Backend(_SourceBackend):
                def empty(self, shape, dtype):
                    value = super().empty(shape, dtype)
                    return value.view(BrokenView) if np.dtype(dtype) == np.dtype('float32') else value

            backend = Backend(metadata_only=True)
            backend.synchronize.side_effect = RuntimeError('sync failed')
            run = bind_scalar_reduction_group('i,i->', ((513,),)*2, np.float32, (4108,)*4, backend)
            partial = np.empty(run.partial_bytes//8, np.float64)
            pool = object()
            device = SimpleNamespace(backend=backend, reduction_partial=lambda *_: partial, pool=pool)
            tasks = tuple((np.ones(513, np.float32), np.ones(513, np.float32)) for _ in range(4))
            with patch('lqcdweave.contraction.execution.device.active_device_execution', return_value=device):
                try:
                    run(tasks)
                except RuntimeError as error:
                    self.assertIn('view construction', str(error))
                    self.assertTrue(error.__notes__)
                    error.__traceback__ = None
                else:
                    self.fail('expected view failure')
            held = runtime.FAILED_SUBMISSIONS.pop()
            self.assertIs(held[2], partial)
            self.assertIs(held[4], pool)
            self.assertIs(held[1][0], tasks[0][0])
            self.assertEqual(len(held[3]), fail_at-1)
            owner = held[5]
            self.assertEqual(owner.nbytes, 16)
            reference = weakref.ref(owner)
            del owner
            gc.collect()
            self.assertIsNotNone(reference())
            self.assertEqual(backend.launches, [])
            self.assertEqual(backend.synchronize.call_count, 1)

    def test_every_live_subset_fits_original_rounded_owner_envelope(self):
        checked = 0
        for count in range(2, 9):
            for itemsize in (4, 8):
                for child_elements in range(1, 512//(count*itemsize)+1):
                    child_bytes = child_elements*itemsize
                    shared = device_allocation_bytes(count*child_bytes)
                    for mask in range(1, 1 << count):
                        # Logical births, read slots, reused coordinates and
                        # releases only choose a live subset. Its previous
                        # per-child owner charge always dominates one block.
                        old = mask.bit_count()*device_allocation_bytes(child_bytes)
                        self.assertLessEqual(shared, old)
                        checked += 1
        self.assertGreater(checked, 10_000)

    def test_cfg_read_and_fully_reused_coordinate_keep_original_budget(self):
        from correlator.planning.test_cartesian_cfg_handoff import read_between_children
        from contracts.test_cartesian_reduction_group import annotated, describe
        base, requests, preparations, specs = read_between_children()
        grouped = annotated(base, (describe(base),))
        self.assertLessEqual(sum(batch.output_bytes for batch in grouped.kernel_batches), 512)
        points, _ = describe_coordinates(time_pairs=[{0: 0}, {0: 0}])
        cfg = build_cfg_plan(grouped, None, points.points, requests, preparations, specs,
                             prefetch_host_bytes=0, prefetch_device_bytes=0, debug=True)
        self.assertTrue(all(cfg.instances[1].reused.array()))
        self.assertEqual(audit_cfg_plan(cfg)['peak_device_bytes'], cfg.peak_device_bytes)
        template = cfg.templates[0].plan
        first, later = template.cartesian_reduction_groups[0].batch_action_indices
        read, = (event for event in cfg.instances[0].events if event.kind == 'read')
        self.assertLess(first[0], read.action_index)
        self.assertLess(read.action_index, later[0])
        self.assertTrue(any(point.cartesian_reduction_device_bytes >= 512
                            for point in template.memory if point.action_index == read.action_index))
        # No P mutation discounts the existing independent children, including
        # the early child at the read and ordinary owners in the reused point.
        self.assertEqual(cfg.statistics['cfg_allocation_rounding_version'], 1)


@unittest.skipUnless(os.environ.get('LQCDWEAVE_CUDA_GATE') == '1',
                     'explicit CUDA gate; CPU suite does not initialize CUDA')
class SharedReductionOutputCuPyTests(unittest.TestCase):
    def test_shared_owner_offsets_and_nonoverlap_scalar_matrix_boundary(self):
        import cupy as cp
        for dtype in (np.float32, np.complex64):
            for count in (2, 4, 8):
                elements = 512//(count*np.dtype(dtype).itemsize)
                for output_elements in (1, elements, elements+1):
                    tasks = tuple((cp.ones((1, 513), dtype), cp.full((output_elements, 513), i+1, dtype))
                                  for i in range(count))
                    allowance = output_elements*(3*16+np.dtype(dtype).itemsize)
                    run = bind_matrix_reduction_group(((1, 513), (output_elements, 513)), dtype,
                                                       (allowance,)*count, cp)
                    values = run(tasks)
                    cp.cuda.get_current_stream().synchronize()
                    owners = {value.data.mem.ptr for value in values}
                    self.assertEqual(len(owners), run.output_owner_count)
                    for i, value in enumerate(values):
                        np.testing.assert_array_equal(cp.asnumpy(value), 513*(i+1))
                    if run.output_owner_count == 1:
                        self.assertEqual(run.output_view_method, 'inplace-shape')
                        self.assertEqual(values[0].data.mem.size, 512)
                        self.assertEqual([value.data.ptr-values[0].data.ptr for value in values],
                                         [i*output_elements*np.dtype(dtype).itemsize for i in range(count)])
                        borrowed = _bind_contiguous_concatenation(0, (count, output_elements))(values, np.dtype(dtype))
                        self.assertIsNotNone(borrowed)
                        self.assertEqual(borrowed.data.ptr, values[0].data.ptr)
                        self.assertIs(borrowed.data.mem, values[0].data.mem)
                    values[0].fill(-1)
                    for i, value in enumerate(values[1:], 1):
                        np.testing.assert_array_equal(cp.asnumpy(value), 513*(i+1))
                    if run.output_owner_count == 1:
                        root = values[0]
                        while root.base is not None:
                            root = root.base
                        reference = weakref.ref(root)
                        survivor = values[-1]
                        del root, values, value, borrowed
                        gc.collect()
                        self.assertIsNotNone(reference())
                        del survivor
                        gc.collect()
                        self.assertIsNone(reference())
                scalar = bind_scalar_reduction_group('i,i->', ((513,),)*2, dtype, (8200,)*count, cp)
                result = scalar(tuple((cp.ones(513, dtype), cp.full(513, i+1, dtype)) for i in range(count)))
                self.assertEqual(len({value.data.mem.ptr for value in result}), 1)
                self.assertEqual(scalar.output_view_method, 'inplace-shape')
                self.assertTrue(all(value.shape == () for value in result))
                for i, value in enumerate(result):
                    np.testing.assert_array_equal(cp.asnumpy(value), 513*(i+1))
