"""Pointer-only packing: dispatch ownership and optional actual CUDA oracles."""
import gc
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

import numpy as np

from lqcdweave.contraction.execution.operations.packing_kernel import (
    bind_packing, _raw_kernel,
)
from lqcdweave.contraction.execution.operations.dense_batch import (
    _BatchSide, _contiguous_concatenation,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug


class RawPackingProtocolTests(unittest.TestCase):
    def tearDown(self):
        _raw_kernel.cache_clear()

    def test_single_identity_side_borrows_each_current_owner_without_caching(self):
        forbidden = Mock(side_effect=AssertionError('a contiguous input needs no copy'))
        side = _BatchSide((('input', None),), (2, 3), 0, False,
                          packer=forbidden, original_packer=forbidden,
                          input_layouts=((0, 1),))
        refs = []
        for scale in (1, 3):
            value = np.arange(6, dtype=np.float64).reshape(2, 3) * scale
            refs.append(weakref.ref(value))
            for debug in (False, True):
                with execution_debug(debug):
                    self.assertIs(side.pack({'input': value}, np, np.dtype('float64')), value)
            del value
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        forbidden.assert_not_called()

    def test_single_identity_side_keeps_layout_promotion_and_shape_boundaries(self):
        def copy(values, destination):
            if values[0].shape != destination.shape:
                raise ValueError('input shape differs from saved layout')
            destination[...] = values[0]
            return destination

        side = _BatchSide((('input', None),), (2, 3), 0, False,
                          packer=copy, original_packer=copy, input_layouts=((0, 1),))
        host = np.arange(6, dtype=np.float64).reshape(2, 3)
        for kind, value in (('negative', host[:, ::-1]),
                            ('Fortran', np.asfortranarray(host)),
                            ('promotion', host.astype(np.float32))):
            with self.subTest(layout=kind):
                result = side.pack({'input': value}, np, np.dtype('float64'))
                self.assertFalse(np.shares_memory(result, value))
                self.assertTrue(result.flags.c_contiguous)
                self.assertEqual(result.dtype, np.dtype('float64'))
                np.testing.assert_array_equal(result, value)
        with self.assertRaisesRegex(ValueError, 'input shape'):
            side.pack({'input': host.reshape(3, 2)}, np, np.dtype('float64'))
        transposed = _BatchSide((('input', (1, 0)),), (3, 2), 0, False,
                               packer=copy, original_packer=copy, input_layouts=((1, 0),))
        value = np.asfortranarray(host)
        result = transposed.pack({'input': value}, np, np.dtype('float64'))
        self.assertTrue(np.shares_memory(result, value))
        np.testing.assert_array_equal(result, host.T)

    def test_pointer_launches_chunking_and_bounded_code_cache(self):
        kernels = []
        def factory(source, name, **kwargs):
            kernel = Mock()
            kernels.append(kernel)
            return kernel
        backend = SimpleNamespace(__name__='cupy', RawKernel=factory,
                                  ElementwiseKernel=Mock())
        arrays = [np.ones((2, 3)) for _ in range(19)]
        run = bind_packing(backend, tuple(a.shape for a in arrays),
                           tuple(a.dtype for a in arrays), 1, np.float64)
        destination = np.empty((2, 57))
        run(arrays, destination)
        self.assertEqual(len(kernels), 1)
        for kernel, size in zip(kernels, (19,)):
            grid, block, arguments = kernel.call_args.args
            self.assertEqual(block, (256,))
            self.assertEqual(len(arguments), size + 1)
            self.assertIs(arguments[-1], destination)
        # Recording mocks retain launch args; the bound implementation must not.
        for kernel in kernels:
            kernel.reset_mock()
        del arguments
        refs = [weakref.ref(a) for a in arrays] + [weakref.ref(destination)]
        del arrays, destination
        gc.collect()
        self.assertTrue(all(r() is None for r in refs))
        for step in range(1, 90):
            values = [np.ones((2, 3 * step))[:, ::step] for _ in range(19)]
            run(values, np.empty((2, 57)))
            for kernel in kernels:
                kernel.reset_mock()
        self.assertLessEqual(_raw_kernel.cache_info().currsize, 64)
        backend.ElementwiseKernel.assert_not_called()

    def test_raw_thirtytwo_and_legacy_eight_input_chunk_boundaries(self):
        class RawCall:
            def __init__(self):
                self.calls = []

            def __call__(self, grid, block, arguments):
                self.calls.append((grid, block, tuple(id(value) for value in arguments)))

        for count in (9, 17, 31, 32, 33, 65):
            with self.subTest(count=count):
                raw_kernels = []

                def raw_factory(source, name, **kwargs):
                    kernel = RawCall()
                    raw_kernels.append(kernel)
                    return kernel

                backend = SimpleNamespace(__name__='cupy', RawKernel=raw_factory,
                                          ElementwiseKernel=Mock())
                arrays = [np.full((2, 3), index, dtype=np.float64)
                          for index in range(count)]
                run = bind_packing(backend, ((2, 3),) * count,
                                   (np.float64,) * count, 1, np.float64)
                target = np.empty((2, 3 * count))
                run(arrays, target)
                raw_chunks = [len(call[2]) - 1 for kernel in raw_kernels
                              for call in kernel.calls]
                self.assertEqual(raw_chunks, [min(32, count - offset)
                                              for offset in range(0, count, 32)])
                self.assertTrue(all((chunk + 1) * 8 < 3072 for chunk in raw_chunks))
                self.assertEqual(sum(len(kernel.calls) for kernel in raw_kernels),
                                 (count + 31) // 32)
                raw_arguments = [call[2] for kernel in raw_kernels for call in kernel.calls]
                expected_arguments = [tuple(id(value) for value in
                    (*arrays[offset:offset + 32], target))
                    for offset in range(0, count, 32)]
                self.assertEqual(raw_arguments, expected_arguments)
                backend.ElementwiseKernel.assert_not_called()

                legacy_calls = []

                class LegacyCall:
                    def __call__(self, *args, size):
                        legacy_calls.append((len(args) - 1, size, id(args[-1])))

                legacy_kernels = []

                def elementwise_factory(*args, **kwargs):
                    kernel = LegacyCall()
                    legacy_kernels.append(kernel)
                    return kernel

                legacy_backend = SimpleNamespace(__name__='cupy', RawKernel=None,
                    ElementwiseKernel=elementwise_factory)
                legacy_run = bind_packing(legacy_backend, ((2, 3),) * count,
                    (np.float64,) * count, 1, np.float64)
                legacy_target = np.empty((2, 3 * count))
                legacy_run(arrays, legacy_target)
                self.assertEqual([call[0] for call in legacy_calls],
                                 [min(8, count - offset)
                                  for offset in range(0, count, 8)])
                self.assertEqual(len(legacy_kernels), (count + 7) // 8)
                self.assertTrue(all(64 + 16 * 2 + chunk * (16 + 16 * 2) < 3072
                                    for chunk, _, _ in legacy_calls))
                self.assertTrue(all(call[2] == id(legacy_target) for call in legacy_calls))

    def test_noncontiguous_destination_uses_existing_elementwise_path(self):
        kernel = Mock()
        backend = SimpleNamespace(__name__='cupy', RawKernel=Mock(),
                                  ElementwiseKernel=Mock(return_value=kernel))
        arrays = [np.ones((2, 3)), np.ones((2, 3))]
        run = bind_packing(backend, ((2, 3),) * 2, (np.float64,) * 2, 1, np.float64)
        target = np.empty((2, 6), order='F')
        self.assertIs(run(arrays, target), target)
        backend.RawKernel.assert_not_called()
        kernel.assert_called_once_with(*arrays, target, size=12)

    def test_original_pointer_packer_maps_permutations_and_singleton_axes(self):
        compiled = []

        def factory(source, name, **kwargs):
            kernel = Mock()
            compiled.append((source, kernel))
            return kernel

        backend = SimpleNamespace(__name__='cupy', RawKernel=factory,
                                  ElementwiseKernel=Mock())
        first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, ::-1, ::-2]
        second = (1j * np.arange(24, dtype=np.float32)).astype('complex64')
        second = second.reshape(2, 3, 4)[:, ::-1, ::-2]

        prepend_layouts = ((None, 1, 2, 0), (None, 1, 2, 0))
        prepend_shapes = ((1, 3, 2, 2),) * 2
        prepend = bind_packing(backend, prepend_shapes, (first.dtype, second.dtype), 1,
                               np.complex128, input_layouts=prepend_layouts)
        self.assertTrue(callable(prepend.pack_original))
        prepend_target = np.empty((1, 6, 2, 2), dtype=np.complex128)
        prepend.pack_original((first, second), prepend_target)
        _, prepend_kernel = compiled[-1]
        prepend_args = prepend_kernel.call_args.args[2]
        self.assertIs(prepend_args[0], first)
        self.assertIs(prepend_args[1], second)
        self.assertIs(prepend_args[-1], prepend_target)
        self.assertIn('-16LL', compiled[-1][0])  # mapped negative byte stride
        self.assertIn('complex<double>', compiled[-1][0])  # write-time promotion

        append_layouts = ((2, 0, 1, None), (2, 0, 1, None))
        append_shapes = ((2, 2, 3, 1),) * 2
        append = bind_packing(backend, append_shapes, (first.dtype, second.dtype), 3,
                              np.complex128, input_layouts=append_layouts)
        append_target = np.empty((2, 2, 3, 2), dtype=np.complex128)
        append.pack_original((first, second), append_target)
        _, append_kernel = compiled[-1]
        append_args = append_kernel.call_args.args[2]
        self.assertIs(append_args[0], first)
        self.assertIs(append_args[1], second)
        self.assertIs(append_args[-1], append_target)
        self.assertIn('96LL', compiled[-1][0])  # mapped original-axis stride

    def test_original_pointer_capability_is_absent_on_legacy_fallbacks(self):
        values = (np.arange(4, dtype=np.float16).reshape(2, 2),) * 2
        logical = tuple(value.T[None, ...] for value in values)
        layout = ((None, 1, 0), (None, 1, 0))
        def elementwise_copy(*args, size):
            destination, inputs = args[-1], args[:-1]
            destination[...] = np.concatenate(inputs, axis=0).astype(destination.dtype)

        elementwise = Mock(side_effect=elementwise_copy)
        missing_raw = SimpleNamespace(__name__='cupy', RawKernel=None,
                                      ElementwiseKernel=Mock(return_value=elementwise))
        run = bind_packing(missing_raw, ((1, 2, 2),) * 2, (np.float16,) * 2, 0,
                           np.float32, input_layouts=layout)
        self.assertFalse(hasattr(run, 'pack_original'))
        target = np.empty((2, 2, 2), dtype=np.float32)
        run(logical, target)
        np.testing.assert_array_equal(target, np.concatenate(logical, axis=0).astype(np.float32))
        elementwise.assert_called_once_with(*logical, target, size=8)

        unsupported_elementwise = Mock(side_effect=elementwise_copy)
        unsupported = SimpleNamespace(__name__='cupy', RawKernel=Mock(),
                                      ElementwiseKernel=Mock(return_value=unsupported_elementwise))
        run = bind_packing(unsupported, ((1, 2, 2),) * 2, (np.float16,) * 2, 0,
                           np.float32, input_layouts=layout)
        self.assertFalse(hasattr(run, 'pack_original'))
        run(logical, target)
        np.testing.assert_array_equal(target, np.concatenate(logical, axis=0).astype(np.float32))
        unsupported.RawKernel.assert_not_called()
        unsupported_elementwise.assert_called_once_with(*logical, target, size=8)

    def test_original_stride_mapping_runs_only_for_new_layouts(self):
        indexed = []

        class Steps(tuple):
            def __getitem__(self, index):
                indexed.append(index)
                return super().__getitem__(index)

        class Metadata:
            def __init__(self, strides):
                self._strides = Steps(strides)
                self.reads = 0

            @property
            def strides(self):
                self.reads += 1
                return self._strides

        compiled = []

        def factory(source, name, **kwargs):
            kernel = Mock()
            compiled.append((source, kernel))
            return kernel

        backend = SimpleNamespace(__name__='cupy', RawKernel=factory, ElementwiseKernel=Mock())
        pack = bind_packing(backend, ((1, 3, 2),) * 2, (np.float64,) * 2, 0, np.float64,
                            input_layouts=((None, 1, 0),) * 2)
        target = np.empty((2, 3, 2))
        first = (Metadata((24, 8)), Metadata((24, 8)))
        pack.pack_original(first, target)
        self.assertEqual(indexed, [1, 0, 1, 0])
        self.assertEqual([value.reads for value in first], [1, 1])
        indexed.clear()
        fresh = (Metadata((24, 8)), Metadata((24, 8)))
        pack.pack_original(fresh, target)
        self.assertEqual(indexed, [])
        self.assertEqual([value.reads for value in fresh], [1, 1])
        self.assertEqual(len(compiled), 1)
        arguments = compiled[0][1].call_args.args[2]
        self.assertIs(arguments[0], fresh[0])
        self.assertIs(arguments[1], fresh[1])
        changed = (Metadata((24, -8)), Metadata((24, 0)))
        pack.pack_original(changed, target)
        self.assertEqual(indexed, [1, 0, 1, 0])
        self.assertEqual(len(compiled), 2)
        # Mixed-stride stacks share source; mapped signed/zero strides now
        # belong to fixed host scalar slots after the 32 input pointers.
        arguments = compiled[1][1].call_args.args[2]
        self.assertIs(arguments[0], changed[0])
        self.assertIs(arguments[1], changed[1])
        self.assertEqual(arguments[32:36], (np.int64(-8), np.int64(24), np.int64(0), np.int64(24)))
        self.assertTrue(all(type(value) is np.int64 for value in arguments[32:36]))
        self.assertIs(arguments[-1], target)

    def test_logical_and_original_calls_separate_equal_stride_keys(self):
        compiled = []

        def factory(source, name, **kwargs):
            kernel = Mock()
            compiled.append((source, kernel))
            return kernel

        backend = SimpleNamespace(__name__='cupy', RawKernel=factory, ElementwiseKernel=Mock())
        pack = bind_packing(backend, ((2, 2),) * 2, (np.float64,) * 2, 0, np.float64,
                            input_layouts=((1, 0),) * 2)
        values = tuple(np.ones((2, 2)) for _ in range(2))
        target = np.empty((4, 2))
        pack(values, target)
        pack.pack_original(values, target)
        self.assertEqual(len(compiled), 2)
        self.assertNotEqual(compiled[0][0], compiled[1][0])
        pack(tuple(value.copy() for value in values), target)
        pack.pack_original(tuple(value.copy() for value in values), target)
        self.assertEqual(len(compiled), 2)
        self.assertEqual([kernel.call_count for _, kernel in compiled], [2, 2])


class RawPackingCupyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f'CUDA unavailable: {exc}') from exc

    def tearDown(self):
        _raw_kernel.cache_clear()

    def test_single_identity_batch_side_current_owner_and_copy_fallbacks(self):
        cp = self.cp
        for dtype in ('float32', 'float64', 'complex64', 'complex128'):
            host = np.arange(6, dtype=np.float64).reshape(2, 3).astype(dtype)
            pack = bind_packing(cp, ((2, 3),), (np.dtype(dtype),), 0, np.dtype(dtype),
                                input_layouts=((0, 1),))
            side = _BatchSide((('input', None),), (2, 3), 0, False,
                              packer=pack, original_packer=pack.pack_original,
                              input_layouts=((0, 1),))
            for scale in (1, 3):
                with self.subTest(dtype=dtype, scale=scale):
                    value = cp.asarray(host * scale)
                    result = side.pack({'input': value}, cp, np.dtype(dtype))
                    self.assertIs(result, value)
                    np.testing.assert_array_equal(cp.asnumpy(result), host * scale)
                    for view, expected in ((value[:, ::-1], (host * scale)[:, ::-1]),
                                           (cp.asfortranarray(value), host * scale)):
                        result = side.pack({'input': view}, cp, np.dtype(dtype))
                        self.assertNotEqual(result.data.ptr, view.data.ptr)
                        np.testing.assert_array_equal(cp.asnumpy(result), expected)

    def test_current_raw_strides_and_call_mode_keep_distinct_numeric_layouts(self):
        cp = self.cp
        run = bind_packing(cp, ((2, 2),) * 2, (np.float64,) * 2, 0, np.float64,
                           input_layouts=((1, 0),) * 2)
        for layout in ('C', 'negative', 'broadcast', 'C'):
            with self.subTest(layout=layout):
                host = tuple(np.arange(4., dtype=np.float64).reshape(2, 2) + offset
                             for offset in (0, 10))
                device = tuple(cp.asarray(value) for value in host)
                if layout == 'negative':
                    host = tuple(value[:, ::-1] for value in host)
                    device = tuple(value[:, ::-1] for value in device)
                elif layout == 'broadcast':
                    host = tuple(np.broadcast_to(value[:1], (2, 2)) for value in host)
                    device = tuple(cp.broadcast_to(value[:1], (2, 2)) for value in device)
                for original in (False, True):
                    target = cp.empty((4, 2), dtype=cp.float64)
                    (run.pack_original if original else run)(device, target)
                    expected = np.concatenate(tuple(value.T if original else value for value in host))
                    np.testing.assert_array_equal(cp.asnumpy(target), expected)

    def test_strides_axes_chunks_empty_and_fresh_inputs_against_numpy(self):
        cp = self.cp
        for axis in range(3):
            for dtype in ('float32', 'float64', 'complex64', 'complex128'):
                with self.subTest(axis=axis, dtype=dtype):
                    base = np.arange(4 * 6 * 8).reshape(4, 6, 8).astype(dtype)
                    if np.dtype(dtype).kind == 'c':
                        base += 1j * base / 7
                    source = cp.asarray(base)
                    values = [source[::-1, ::-2, ::-2] for _ in range(19)]
                    shapes = tuple(a.shape for a in values)
                    run = bind_packing(cp, shapes, (np.dtype(dtype),) * 19, axis, dtype)
                    for scale in (1, 3):
                        source = cp.asarray(base * scale)
                        values = [source[::-1, ::-2, ::-2] for _ in range(19)]
                        shape = list(shapes[0]); shape[axis] *= 19
                        target = cp.empty(shape, dtype=dtype)
                        self.assertIs(run(values, target), target)
                        reference = np.concatenate([base[::-1, ::-2, ::-2] * scale] * 19, axis=axis)
                        np.testing.assert_array_equal(cp.asnumpy(target), reference)
                    # Empty lanes interspersed with nonempty lanes, including chunk boundaries.
                    index = [slice(None)] * 3; index[axis] = slice(0, 0)
                    values = [a[tuple(index)] if i % 3 == 0 else a for i, a in enumerate(values)]
                    shapes = tuple(a.shape for a in values)
                    run = bind_packing(cp, shapes, (np.dtype(dtype),) * 19, axis, dtype)
                    reference = np.concatenate([cp.asnumpy(a) for a in values], axis=axis)
                    target = cp.empty(reference.shape, dtype=dtype)
                    run(values, target)
                    np.testing.assert_array_equal(cp.asnumpy(target), reference)

    def test_thirtytwo_input_chunks_cover_mixed_dtypes_and_layouts(self):
        cp = self.cp

        def inputs_for(count, scale):
            values = []
            for index in range(count):
                dtype = (np.float32, np.float64, np.complex64, np.complex128)[index % 4]
                base = (np.arange(12, dtype=np.float64).reshape(3, 4) + index / 7) * scale
                if np.dtype(dtype).kind == "c":
                    base = base + 1j * (base[::-1] / 11)
                host = base.astype(dtype)
                layout = index % 4
                if layout == 0:
                    host_view = host[::-1, ::-1]
                    device_view = cp.asarray(host)[::-1, ::-1]
                elif layout == 1:
                    host_view = np.asfortranarray(host)
                    device_view = cp.asfortranarray(cp.asarray(host))
                elif layout == 2:
                    host_view = np.broadcast_to(host[:1, :], (3, 4))
                    device_view = cp.broadcast_to(cp.asarray(host[:1, :]), (3, 4))
                else:
                    host_view = host
                    device_view = cp.asarray(host)
                values.append((host_view, device_view))
            return values

        for count in (9, 17, 31, 32, 33, 65):
            with self.subTest(count=count):
                first = inputs_for(count, 1)
                shapes = tuple(device.shape for _, device in first)
                dtypes = tuple(device.dtype for _, device in first)
                run = bind_packing(cp, shapes, dtypes, 1, cp.complex128)
                for scale in (1, 3):
                    values = first if scale == 1 else inputs_for(count, scale)
                    host_values = [host for host, _ in values]
                    device_values = [device for _, device in values]
                    expected = np.concatenate(host_values, axis=1).astype(np.complex128)
                    target = cp.empty(expected.shape, dtype=cp.complex128)
                    self.assertIs(run(device_values, target), target)
                    np.testing.assert_array_equal(cp.asnumpy(target), expected)

    def test_dtype_promotion_fortran_zero_stride_and_noncontiguous_target(self):
        cp = self.cp
        for dtype in ('float32', 'float64', 'complex64', 'complex128'):
            with self.subTest(dtype=dtype):
                left = cp.asfortranarray(cp.arange(12, dtype=dtype).reshape(3, 4))
                right = cp.broadcast_to(cp.arange(3, dtype=cp.int16)[:, None], (3, 4))
                run = bind_packing(cp, (left.shape, right.shape),
                                   (left.dtype, right.dtype), 1, dtype)
                expected = np.concatenate([cp.asnumpy(left), cp.asnumpy(right)], axis=1).astype(dtype)
                for order in ('C', 'F'):
                    target = cp.empty((3, 8), dtype=dtype, order=order)
                    run((left, right), target)
                    np.testing.assert_array_equal(cp.asnumpy(target), expected)
        # Unsupported half source retains the existing implementation.
        values = (cp.ones((2, 2), dtype=cp.float16), cp.ones((2, 2), dtype=cp.float32))
        run = bind_packing(cp, ((2, 2),) * 2, tuple(a.dtype for a in values), 1, cp.float32)
        target = cp.empty((2, 4), dtype=cp.float32)
        run(values, target)
        np.testing.assert_array_equal(cp.asnumpy(target), np.ones((2, 4), dtype=np.float32))

    def test_original_pointer_permutation_stack_axes_negative_strides_and_promotion(self):
        cp = self.cp
        host_first_base = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        host_second_base = (1j * np.arange(24, dtype=np.float32)).astype('complex64').reshape(2, 3, 4)
        host_first = host_first_base[:, ::-1, ::-2]
        host_second = host_second_base[:, ::-1, ::-2]
        values = (cp.asarray(host_first_base)[:, ::-1, ::-2],
                  cp.asarray(host_second_base)[:, ::-1, ::-2])
        for layouts, shapes, axis, target_shape, expected_layout in (
                (((None, 1, 2, 0),) * 2, ((1, 3, 2, 2),) * 2, 1, (1, 6, 2, 2), 'prepend'),
                (((2, 0, 1, None),) * 2, ((2, 2, 3, 1),) * 2, 3, (2, 2, 3, 2), 'append')):
            with self.subTest(layout=expected_layout):
                run = bind_packing(cp, shapes, tuple(value.dtype for value in values), axis,
                                   cp.complex128, input_layouts=layouts)
                expected = np.concatenate([
                    (value.transpose((1, 2, 0))[None, ...] if expected_layout == 'prepend' else
                     value.transpose((2, 0, 1))[..., None])
                    for value in (host_first, host_second)
                ], axis=axis).astype(np.complex128)
                target = cp.empty(target_shape, dtype=cp.complex128)
                run.pack_original(values, target)
                np.testing.assert_array_equal(cp.asnumpy(target), expected)

                # A non-C target keeps the previous view-based Elementwise fallback.
                logical = tuple(value.transpose((1, 2, 0))[None, ...]
                                if expected_layout == 'prepend' else
                                value.transpose((2, 0, 1))[..., None] for value in values)
                fallback_target = cp.empty(target_shape, dtype=cp.complex128, order='F')
                run(logical, fallback_target)
                np.testing.assert_array_equal(cp.asnumpy(fallback_target), expected)

    def test_single_mapped_f_owner_transpose_borrows_logical_view(self):
        cp = self.cp
        owner = cp.asfortranarray(cp.arange(6, dtype=cp.float64).reshape(2, 3))
        actual = _contiguous_concatenation(
            (owner,), 0, (1, 3, 2), cp.float64,
            input_layouts=((None, 1, 0),))
        self.assertTrue(owner.flags.f_contiguous)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(actual.data.ptr, owner.data.ptr)
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(owner).T[None, ...])
