"""Host-slot protocol and actual CUDA checks for deferred raw consumers."""
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.raw_arguments import bind_pointer_arguments
from lqcdweave.contraction.execution.operations.output_kernel import (
    bind_packed_output_copy, _raw_copy_kernel,
)
from lqcdweave.contraction.execution.operations.packing_kernel import (
    bind_packing, bind_gemm_group_packing, _raw_kernel,
)
from lqcdweave.contraction.execution.operations.result_views import (
    _DeferredResultView, _ResultOwner, _array_pointer, bind_result_views,
)
from .test_gemm_group_packing import descriptors
from .test_raw_arguments import _FakeCNumpyArray


def _references(owner, indices):
    layouts = bind_result_views(owner.shape, owner.dtype,
        tuple((index, (lane, slice(None), slice(None)), ())
              for index, lane in enumerate(indices)))
    shared = _ResultOwner(owner)
    return tuple(_DeferredResultView(shared, layout) for _, layout in layouts)


class _HostDeviceArray:
    def __init__(self, value):
        self.value = value
        self.data = SimpleNamespace(ptr=value.ctypes.data)

    def __getattr__(self, name):
        return getattr(self.value, name)


class _SlotRecordingKernel:
    def __init__(self, launches):
        self.calls = []
        self.launches = launches

    def __call__(self, grid, block, arguments):
        record = (grid, block,
            tuple(int(argument.value[0]) for argument in arguments),
            tuple(id(argument) for argument in arguments))
        self.calls.append(record)
        self.launches.append(record)


_SlotRecordingKernel.__name__ = 'RawKernel'
_SlotRecordingKernel.__module__ = 'cupy._core.raw'


class _SlotBackend:
    __name__ = 'cupy'
    cuda = SimpleNamespace(runtime=SimpleNamespace(is_hip=False),
                           function=SimpleNamespace(CNumpyArray=_FakeCNumpyArray))

    def __init__(self):
        self.kernels = []
        self.launches = []

    def RawKernel(self, source, name, **kwargs):
        kernel = _SlotRecordingKernel(self.launches)
        self.kernels.append(kernel)
        return kernel

    def ElementwiseKernel(self, *args, **kwargs):
        raise AssertionError('supported native consumer needs no legacy kernel')

    def empty(self, shape, dtype, order='C'):
        return _HostDeviceArray(np.empty(shape, dtype=dtype, order=order))


class RawConsumerArgumentsCpuTests(unittest.TestCase):
    def tearDown(self):
        _raw_kernel.cache_clear()
        _raw_copy_kernel.cache_clear()

    def test_mixed_current_addresses_fill_directly_and_failure_clears_slots(self):
        backend = _SlotBackend()
        kernel = backend.RawKernel('', '')
        binding = bind_pointer_arguments(backend, kernel, (1,), (256,),
            (None,) * 4, ((0, 0), (1, 1), (2, 2)), pointer_getter=_array_pointer)
        previous = None
        for generation in (2, 17):
            root = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + generation
            reference, = _references(root, (1,))
            array = (np.arange(12, dtype=np.float64) + generation)[::2]
            device = _HostDeviceArray(array[::-1])
            output = backend.empty((3,), np.float64)
            inputs = (reference, array, device) if generation == 2 else (device, reference, array)
            expected = (root[1].ctypes.data, array.ctypes.data, device.value.ctypes.data)
            if generation != 2:
                expected = (expected[2], expected[0], expected[1])
            with (patch.object(_DeferredResultView, 'pointer_address',
                               side_effect=AssertionError('per-input method call')),
                  patch.object(_DeferredResultView, 'array',
                               side_effect=AssertionError('eager view'))):
                binding(inputs, output)
            self.assertEqual(backend.launches[-1][2], (*expected, output.data.ptr))
            self.assertEqual(tuple(binding._local.storage[0]), (0,) * 4)
            if previous is not None:
                self.assertNotEqual(expected[1], previous[0].owner.address + previous[0].layout.offset)
                np.testing.assert_array_equal(previous[0].array(), previous[1][1])
            previous = reference, root

        failure = RuntimeError('current address unavailable')
        class Broken:
            @property
            def data(self):
                raise failure
        submitted = len(backend.launches)
        with self.assertRaises(RuntimeError) as caught:
            binding((reference, Broken(), array), output)
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(backend.launches), submitted)
        self.assertEqual(tuple(binding._local.storage[0]), (0,) * 4)

    def _run_consumer(self, kind):
        backend = _SlotBackend()
        _FakeCNumpyArray.constructed = 0
        count = 6 if kind == 'group' else (257 if kind == 'snapshot' else 33)
        shape = (3, 2) if kind == 'group' else (2, 3)
        indices = (0, 2, 4, 1, 3, 5) if kind == 'group' else tuple(range(1, count + 1))
        if kind == 'concat':
            binding = bind_packing(backend, (shape,) * count, (np.float64,) * count,
                                   1, np.float64)
        elif kind == 'snapshot':
            binding = bind_packed_output_copy(backend, shape, np.float64, count)
        else:
            sides, catalog, matrix_shape = descriptors(count=3)
            binding = bind_gemm_group_packing(backend, sides, catalog, matrix_shape, np.float64)
        previous_ids = None
        for generation in (1, 9):
            launch_start = len(backend.launches)
            owner = _HostDeviceArray(np.arange((count + 2) * np.prod(shape), dtype=np.float64)
                                     .reshape(count + 2, *shape) + generation)
            references = _references(owner, indices)
            input_addresses = tuple(owner.data.ptr + lane * np.prod(shape) * 8 for lane in indices)
            owner_ref, value_ref = weakref.ref(owner), weakref.ref(owner.value)
            with (execution_debug(False), patch.object(_DeferredResultView, 'array',
                    side_effect=AssertionError('native call must use current address')),
                    patch.object(_DeferredResultView, 'pointer_argument',
                    side_effect=AssertionError('native call must not wrap a pointer scalar'))):
                if kind == 'group':
                    keys = tuple(key for side in sides for key, _ in side.operands)
                    node = _NodeView({key: key for key in keys}, dict(zip(keys, references)))
                    destination = binding.pack_buffer(node)
                    del node
                else:
                    target_shape = (count, *shape) if kind == 'snapshot' else (shape[0], count * shape[1])
                    destination = backend.empty(target_shape, np.float64)
                    binding(references, destination)
            chunk = 128 if kind == 'snapshot' else 32
            addresses = []
            ids = []
            for offset, record in zip(range(0, count, chunk),
                                      backend.launches[launch_start:], strict=True):
                grid, block, words, parameter_ids = record
                lanes = min(chunk, count - offset)
                addresses.extend(words[:lanes])
                self.assertEqual(words[-1], destination.data.ptr)
                if kind == 'snapshot':
                    self.assertEqual(words[-2], offset)
                self.assertEqual(block, (256,))
                ids.extend(parameter_ids)
            self.assertEqual(tuple(addresses), input_addresses)
            if previous_ids is not None:
                self.assertEqual(ids, previous_ids)
                self.assertEqual(_FakeCNumpyArray.constructed, len(ids))
            previous_ids = ids
            del owner, references, destination
            gc.collect()
            self.assertIsNone(owner_ref())
            self.assertIsNone(value_ref())

    def test_concat_chunks_reuse_host_slots_without_retaining_current_owners(self):
        self._run_consumer('concat')

    def test_output_chunks_reuse_host_slots_and_keep_destination_lane_offsets(self):
        self._run_consumer('snapshot')

    def test_group_packing_uses_current_noncontiguous_owner_slices(self):
        self._run_consumer('group')

    def test_getter_failure_clears_partial_slots_without_submitting(self):
        backend = _SlotBackend()
        kernel = backend.RawKernel('', '')
        first = _HostDeviceArray(np.arange(6, dtype=np.float64))
        failure = RuntimeError('input getter failed')

        def get_pointer(value):
            if value is failure:
                raise failure
            return value.data.ptr

        binding = bind_pointer_arguments(backend, kernel, (1,), (256,),
            (None, None, None), ((0, 0), (1, 1)), pointer_getter=get_pointer)
        with self.assertRaises(RuntimeError) as caught:
            binding((first, failure), first)
        self.assertIs(caught.exception, failure)
        np.testing.assert_array_equal(binding._local.storage[0], 0)
        self.assertEqual(backend.launches, [])

    def test_wide_slots_keep_fixed_words_and_clear_after_success_or_launch_error(self):
        backend = _SlotBackend()
        kernel = backend.RawKernel('', '')
        constant = _HostDeviceArray(np.arange(3, dtype=np.float64))
        template = (None,) * 8 + (np.int64(-7), constant, np.uint64(2**63 + 1)) + (None,) * 8 + (np.int64(-2), None)
        positions = tuple((slot, index) for index, slot in enumerate((*range(8), *range(11, 19))))
        binding = bind_pointer_arguments(backend, kernel, (1,), (256,), template, positions)
        fixed = {8: (-7) % 2**64, 9: constant.data.ptr, 10: 2**63 + 1, 19: (-2) % 2**64}
        self.assertEqual(len(binding.clear_ranges), 3)
        for generation in (1, 9):
            inputs = tuple(_HostDeviceArray(np.full(2, generation + index, dtype=np.float64)) for index in range(16))
            output = backend.empty((2,), np.float64)
            binding(inputs, output)
            delivered = dict(fixed)
            delivered.update({slot: inputs[index].data.ptr for slot, index in positions})
            delivered[20] = output.data.ptr
            self.assertEqual(backend.launches[-1][2], tuple(delivered[index] for index in range(21)))
            expected = tuple(fixed.get(index, 0) for index in range(21))
            self.assertEqual(tuple(binding._local.storage[0]), expected)
            failure = RuntimeError('wide launch failed')
            with patch.object(type(kernel), '__call__', side_effect=failure):
                with self.assertRaises(RuntimeError) as caught:
                    binding(inputs, output)
            self.assertIs(caught.exception, failure)
            self.assertEqual(tuple(binding._local.storage[0]), expected)

    def test_wide_getter_failure_clears_partial_slots_and_allows_fresh_retry(self):
        backend = _SlotBackend()
        kernel = backend.RawKernel('', '')
        inputs = tuple(_HostDeviceArray(np.full(2, index, dtype=np.float64)) for index in range(16))
        output = backend.empty((2,), np.float64)
        failure = RuntimeError('wide input getter failed')

        def get_pointer(value):
            if value is failure:
                raise failure
            return value.data.ptr

        template = (None,) * 16 + (np.int64(-3), None)
        binding = bind_pointer_arguments(backend, kernel, (1,), (256,), template,
            tuple((index, index) for index in range(16)), pointer_getter=get_pointer)
        with self.assertRaises(RuntimeError) as caught:
            binding((*inputs[:5], failure, *inputs[6:]), output)
        self.assertIs(caught.exception, failure)
        expected = (0,) * 16 + ((-3) % 2**64, 0)
        self.assertEqual(tuple(binding._local.storage[0]), expected)
        self.assertEqual(backend.launches, [])
        binding(inputs, output)
        self.assertEqual(backend.launches[-1][2], tuple(value.data.ptr for value in inputs) + ((-3) % 2**64, output.data.ptr))
        self.assertEqual(tuple(binding._local.storage[0]), expected)


class RawConsumerArgumentsCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceProperties(0)
        except Exception as error:
            raise unittest.SkipTest(f'actual CUDA raw consumer checks require CuPy: {error}')
        if cp.cuda.runtime.is_hip or not callable(getattr(cp.cuda.function, 'CNumpyArray', None)):
            raise unittest.SkipTest('CUDA CNumpyArray argument capability is unavailable')
        cls.cp = cp

    def test_concat_and_snapshot_use_fresh_deferred_addresses_on_current_stream(self):
        cp = self.cp
        count, shape = 33, (3, 4)
        pack = bind_packing(cp, (shape,) * count, (np.complex128,) * count, 1, np.complex128)
        copy = bind_packed_output_copy(cp, shape, np.complex128, count)
        streams = (cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True))
        submitted = []
        for generation, stream in enumerate(streams, 1):
            host = (np.arange((count + 2) * 12).reshape(count + 2, *shape)
                    + generation * 11 + 0.25j * generation).astype(np.complex128)
            with stream:
                owner = cp.asarray(host)
                values = _references(owner, range(1, count + 1))
                concatenated = cp.empty((3, count * 4), dtype=cp.complex128)
                snapshot = cp.empty((count, *shape), dtype=cp.complex128)
                with (execution_debug(False), patch.object(_DeferredResultView, 'array',
                        side_effect=AssertionError('native consumer must stay pointer-only')),
                        patch.object(_DeferredResultView, 'pointer_argument',
                        side_effect=AssertionError('native consumer must avoid pointer scalars'))):
                    pack(values, concatenated)
                    copy(values, snapshot)
                submitted.append((owner, values, concatenated, snapshot, host[1:count + 1]))
        # Observe after both independent streams have received their work.
        for stream in streams:
            stream.synchronize()
        for owner, values, concatenated, snapshot, expected in submitted:
            np.testing.assert_array_equal(cp.asnumpy(concatenated), np.concatenate(expected, axis=1))
            np.testing.assert_array_equal(cp.asnumpy(snapshot), expected)
        calls = next(iter(copy._variants.values()))
        self.assertEqual(len(calls), 3)
        for _kernel, native, start, stop, offset, _grid, _block in calls:
            self.assertIsNotNone(native)
            words = native._local.storage[0]
            self.assertEqual(tuple(map(int, words[:stop - start])),
                             tuple(value.pointer_address() for value in submitted[-1][1][start:stop]))
            self.assertEqual(int(words[-2]), start)
            self.assertEqual(int(words[-1]), submitted[-1][3].data.ptr)
        # Debug retains the original argument delivery with the same numbers.
        with execution_debug(True):
            pack(submitted[-1][1], submitted[-1][2])
            copy(submitted[-1][1], submitted[-1][3])
        cp.cuda.get_current_stream().synchronize()
        np.testing.assert_array_equal(cp.asnumpy(submitted[-1][2]),
                                      np.concatenate(submitted[-1][-1], axis=1))
        np.testing.assert_array_equal(cp.asnumpy(submitted[-1][3]), submitted[-1][-1])

    def test_group_packing_keeps_fresh_nonzero_offsets_and_exact_workspace(self):
        cp = self.cp
        sides, catalog, matrix_shape = descriptors(np.complex128, count=3)
        binding = bind_gemm_group_packing(cp, sides, catalog, matrix_shape, np.complex128)
        keys = tuple(key for side in sides for key, _ in side.operands)
        indices = (1, 3, 5, 2, 4, 6)
        for generation in (2, 7):
            with self.subTest(generation=generation):
                host = (np.arange(8 * 3 * 2).reshape(8, 3, 2) + generation
                        + 1j * generation / 3).astype(np.complex128)
                owner = cp.asarray(host)
                values = _references(owner, indices)
                node = _NodeView({key: key for key in keys}, dict(zip(keys, values)))
                with (execution_debug(False), patch.object(_DeferredResultView, 'array',
                        side_effect=AssertionError('group copy must stay pointer-only')),
                        patch.object(_DeferredResultView, 'pointer_argument',
                        side_effect=AssertionError('group copy must avoid pointer scalars'))):
                    actual = binding.pack_buffer(node)
                cp.cuda.get_current_stream().synchronize()
                expected = np.stack([np.concatenate((host[indices[2 * lane]],
                    host[indices[2 * lane + 1]]), axis=1) for lane in range(3)])
                self.assertEqual(actual.nbytes, 3 * np.prod(matrix_shape) * 16)
                np.testing.assert_array_equal(cp.asnumpy(actual), expected)


if __name__ == '__main__':
    unittest.main()
