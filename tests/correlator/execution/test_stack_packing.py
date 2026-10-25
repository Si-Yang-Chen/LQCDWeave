"""Equal-stack code sharing without a device geometry or pointer allocation."""
import ctypes
import gc
import importlib
import re
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from lqcdweave.contraction.execution.operations.packing_kernel import (
    bind_packing, _raw_kernel,
)


class _StackBackend:
    """Interpret actual emitted address expressions; CUDA remains a separate gate."""
    __name__ = 'cupy'
    ElementwiseKernel = Mock(side_effect=AssertionError('unexpected fallback'))

    def __init__(self):
        self.sources = []
        self.calls = []
        self.fail = False

    def RawKernel(self, source, name, **kwargs):
        self.sources.append(source)
        size = int(re.search(r'if \(q >= (\d+)LL\)', source)[1])
        read = re.search(r'char\*>\(input\) \+ \((.*?)\)\);', source)[1]
        write = re.search(r'y\[(.*?)\] = static_cast', source)[1]
        def expression(text):
            return compile(text.replace('static_cast<long long>(blockIdx.y)', 'lane')
                           .replace('LL', '').replace('/', '//'), '<packing address>', 'eval')
        read, write = expression(read), expression(write)

        def launch(grid, block, args):
            if self.fail:
                raise RuntimeError('launch failed')
            self.calls.append((grid, block, len(args)))
            assert len(args) == 35  # 32 pointers, two signed int64s and result.
            span, offset, destination = args[-3:]
            assert type(span) is np.int64 and type(offset) is np.int64
            assert all(type(p) is np.uint64 and p == 0 for p in args[grid[1]:32])
            for lane in range(grid[1]):
                value = args[lane]
                for q in range(grid[0] * block[0]):
                    if q >= size:
                        continue
                    local = dict(q=q, lane=lane, span=int(span), offset=int(offset))
                    address = eval(read, {'__builtins__': {}}, local)
                    target = eval(write, {'__builtins__': {}}, local)
                    raw = ctypes.string_at(value.ctypes.data + address, value.itemsize)
                    destination.flat[target] = np.frombuffer(raw, dtype=value.dtype)[0]
        return launch


class StackPackingTests(unittest.TestCase):
    def tearDown(self):
        _raw_kernel.cache_clear()

    def test_contiguous_dimensions_share_code_across_ranks(self):
        backend = _StackBackend()
        for shape in ((1, 12), (1, 3, 4), (1, 2, 1, 2, 3)):
            values = tuple(np.arange(12, dtype=np.float32).reshape(shape) + i for i in range(3))
            run = bind_packing(backend, (shape,)*3, (np.float32,)*3, 0, np.float32)
            target = np.empty((3, *shape[1:]), dtype=np.float32)
            run(values, target)
            np.testing.assert_array_equal(target, np.concatenate(values))
        self.assertEqual(len(backend.sources), 1)
        self.assertIn('q * (4LL)', backend.sources[0])

    def test_code_reuses_counts_and_chunk_offsets_with_fresh_strided_inputs(self):
        for axis in (0, 1, 2):
            for dtype in (np.float32, np.complex64):
                backend = _StackBackend()
                for count in (1, 3, 32, 33, 65):
                    for style in ('negative', 'broadcast'):
                        values = []
                        for lane in range(count):
                            value = (np.arange(15).reshape(3, 5) + lane).astype(dtype)
                            if dtype == np.complex64:
                                value += 1j * value / 4
                            values.append(value[::-1, ::-1] if style == 'negative'
                                          else np.broadcast_to(value[:1, :], (3, 5)))
                        logical = tuple(np.expand_dims(v, axis) for v in values)
                        layouts = tuple((*range(axis), None, *range(axis, 2)) for _ in values)
                        run = bind_packing(backend, tuple(v.shape for v in logical),
                            tuple(v.dtype for v in values), axis, np.complex128, input_layouts=layouts)
                        expected = np.concatenate(logical, axis=axis).astype(np.complex128)
                        result = np.empty(expected.shape, dtype=expected.dtype, order='C')
                        run.pack_original(values, result)
                        np.testing.assert_array_equal(result, expected)
                        values = ([np.ascontiguousarray((v * 2)[::-1, ::-1])[::-1, ::-1]
                                   for v in values] if style == 'negative' else
                                  [np.broadcast_to(v[:1] * 2, (3, 5)) for v in values])
                        run.pack_original(values, result)
                        expected = np.concatenate(tuple(np.expand_dims(v, axis) for v in values), axis=axis)
                        np.testing.assert_array_equal(result, expected)
                # Geometry/strides may specialize. The stack count and second/
                # third chunk offset must not create another CUDA source.
                self.assertEqual(len(set(backend.sources)), 2)
                self.assertTrue(all(call[2] * 8 == 280 for call in backend.calls))

    def test_native_abi_keeps_constants_padded_slots_and_current_pointer_positions(self):
        backend = SimpleNamespace(__name__='cupy', ElementwiseKernel=Mock(),
                                  RawKernel=Mock(return_value=Mock()))
        value = np.arange(10, dtype=np.float32).reshape(1, 10)
        values = (value,) * 33
        run = bind_packing(backend, (value.shape,) * 33, (value.dtype,) * 33, 0, np.float32)
        target = np.empty((33, 10), np.float32)
        native = Mock()
        with patch('lqcdweave.contraction.execution.operations.packing_kernel.bind_pointer_arguments',
                   return_value=native) as bind:
            run(values, target)
        self.assertEqual(bind.call_count, 2)
        for index, call in enumerate(bind.call_args_list):
            template, positions = call.args[4:6]
            count = 32 if index == 0 else 1
            self.assertEqual(len(template), 35)
            self.assertEqual(positions, tuple((lane, index*32+lane) for lane in range(count)))
            self.assertTrue(all(v is None for v in template[:count]))
            self.assertTrue(all(type(v) is np.uint64 and v == 0 for v in template[count:32]))
            self.assertEqual(tuple(template[-3:-1]), (330, index*320))
            self.assertIsNone(template[-1])
        self.assertEqual(native.call_count, 2)
        self.assertEqual(backend.RawKernel.call_count, 1)

    def test_binding_keeps_no_input_owner_on_success_or_failure(self):
        backend = _StackBackend()
        run = bind_packing(backend, ((1, 3),)*2, (np.float32,)*2, 0, np.float32)
        for failure in (False, True):
            values = tuple(np.ones((1, 3), np.float32) for _ in range(2))
            target = np.empty((2, 3), np.float32)
            refs = [weakref.ref(v) for v in values] + [weakref.ref(target)]
            backend.fail = failure
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                    run(values, target)
            else:
                run(values, target)
            del values, target
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))


class StackPackingCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as error:
            raise unittest.SkipTest(f'CUDA unavailable: {error}') from error

    def test_cuda_equal_stack_shared_code_chunks_permutations_and_negative_strides(self):
        cp = self.cp
        for axis in (0, 2):
            host = [(np.arange(15).reshape(3, 5) + (1+1j)*i).astype(np.complex64) for i in range(65)]
            logical_shape = (1, 5, 3) if axis == 0 else (5, 3, 1)
            layout = (None, 1, 0) if axis == 0 else (1, 0, None)
            run = bind_packing(cp, (logical_shape,)*65, (np.complex64,)*65, axis,
                               np.complex128, input_layouts=(layout,)*65)
            # The original owners expose a reversed axis; the mapped layout
            # transposes that view in-kernel.
            values = [cp.asarray(value)[:, ::-1] for value in host]
            expected = np.stack([value[:, ::-1].T for value in host], axis=axis)
            target = cp.empty(expected.shape, cp.complex128)
            run.pack_original(values, target)
            np.testing.assert_array_equal(cp.asnumpy(target), expected)


if __name__ == '__main__':
    unittest.main()
