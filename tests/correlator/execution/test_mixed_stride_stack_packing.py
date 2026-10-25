"""Mixed-stride stack code sharing, host ABI ownership and a separate CUDA gate."""
import ctypes
import gc
import importlib
import re
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from lqcdweave.contraction.execution.operations.packing_kernel import bind_packing, _raw_kernel
from lqcdweave.contraction.execution.operations.result_views import (
    _DeferredResultView, _ResultOwner, bind_result_views,
)
from lqcdweave.contraction.execution.raw_arguments import _BoundPointerArguments


class _Slot:
    def __init__(self, value):
        self.value = value


class _Output:
    def __init__(self, value):
        self.value = value
        self.flags = value.flags
        self.data = SimpleNamespace(ptr=value.ctypes.data)


class _Kernel:
    """Execute the actual generated address expressions against host buffers."""
    def __init__(self, source, backend):
        self.source, self.backend = source, backend
        signature = source.split('ed_dense_batch_pack_raw(', 1)[1].split(') {', 1)[0]
        self.names = [argument.strip().split()[-1] for argument in signature.split(',')]
        types = {'float': np.dtype('float32'), 'double': np.dtype('float64'),
                 'complex<float>': np.dtype('complex64'), 'complex<double>': np.dtype('complex128')}
        self.input_dtype = types[re.search(r'const (.*?)\* x0', source)[1]]
        self.output_dtype = types[re.search(r', (.*?)\* y\)', source)[1].split(', ')[-1]]
        self.size = int(re.search(r'if \(q >= (\d+)LL\)', source)[1])
        self.read = self.expression(re.search(r'char\*>\(input\) \+ \((.*?)\)\);', source)[1])
        self.write = self.expression(re.search(r'y\[(.*?)\] = static_cast', source)[1])
        self.cases = {}
        for lane, input_index, body in re.findall(r'case (\d+): input = x(\d+);(.*?)break;', source):
            self.cases[int(lane)] = (int(input_index), re.findall(r'(s\d+) = (t\d+_\d+);', body))

    @staticmethod
    def expression(value):
        return compile(value.replace('static_cast<long long>(blockIdx.y)', 'lane')
                       .replace('LL', '').replace('/', '//'), '<mixed stack address>', 'eval')

    def __call__(self, grid, block, arguments):
        native = isinstance(arguments[0], _Slot)
        arguments = tuple(int(a.value[0]) for a in arguments) if native else arguments
        self.backend.calls.append((grid, block, tuple(int(a) if np.isscalar(a) else a.ctypes.data
                                                     for a in arguments)))
        if self.backend.fail:
            raise RuntimeError('packing launch failed')
        parameters = dict(zip(self.names, arguments, strict=True))
        destination = parameters['y'] if native else parameters['y'].ctypes.data
        for lane in range(grid[1]):
            input_index, assignments = self.cases[lane]
            value = parameters[f'x{input_index}']
            pointer = int(value) if native or np.isscalar(value) else value.ctypes.data
            local = {name: ctypes.c_int64(int(parameters[name])).value
                     for name in ('span', 'offset')}
            local.update((name, ctypes.c_int64(int(parameters[param])).value)
                         for name, param in assignments)
            local['lane'] = lane
            for q in range(self.size):
                local['q'] = q
                address = eval(self.read, {'__builtins__': {}}, local)
                output_index = eval(self.write, {'__builtins__': {}}, local)
                raw = ctypes.string_at(pointer + address, self.input_dtype.itemsize)
                scalar = np.frombuffer(raw, dtype=self.input_dtype)[0]
                converted = np.asarray(scalar, dtype=self.output_dtype).tobytes()
                ctypes.memmove(destination + output_index * self.output_dtype.itemsize,
                               converted, self.output_dtype.itemsize)


class _Backend:
    __name__ = 'cupy'
    ElementwiseKernel = Mock(side_effect=AssertionError('unexpected elementwise fallback'))

    def __init__(self):
        self.sources, self.calls = [], []
        self.fail = False

    def RawKernel(self, source, name, **kwargs):
        self.sources.append(source)
        return _Kernel(source, self)


def _values(count, phase=0, dtype=np.complex64):
    values = []
    for lane in range(count):
        base = (np.arange(15).reshape(3, 5) + 100 * phase + lane).astype(dtype)
        if np.dtype(dtype).kind == 'c':
            base += 1j * base / 8
        style = (lane + phase) % 4
        values.append(base if style == 0 else base[::-1, ::-1] if style == 1 else
                      np.broadcast_to(base[:1], base.shape) if style == 2 else np.asfortranarray(base))
    return tuple(values)


class MixedStrideStackPackingTests(unittest.TestCase):
    def tearDown(self):
        _raw_kernel.cache_clear()

    def test_generated_addresses_permutations_casts_signed_zero_strides_and_chunks(self):
        for axis in (0, 1, 2):
            for dtype in (np.float32, np.complex64):
                backend = _Backend()
                layout = tuple([1, 0][:axis] + [None] + [1, 0][axis:])
                shape = tuple(1 if index is None else (3, 5)[index] for index in layout)
                for count in (2, 31, 32, 33, 65):
                    run = bind_packing(backend, (shape,) * count, (dtype,) * count,
                                       axis, np.complex128, input_layouts=(layout,) * count)
                    for phase in (0, 1, 2):
                        values = _values(count, phase, dtype)
                        expected = np.stack([value.T for value in values], axis=axis)
                        target = np.empty(expected.shape, np.complex128)
                        self.assertIs(run.pack_original(values, target), target)
                        np.testing.assert_array_equal(target, expected)
                mixed_sources = [source for source in backend.sources if 'long long t0_0' in source]
                self.assertEqual(len(mixed_sources), 1)
                self.assertNotRegex(mixed_sources[0], r'[/%%] (?:s\d|t\d|span|offset)')

    def test_same_stride_and_contiguous_paths_keep_priority(self):
        backend = _Backend()
        run = bind_packing(backend, ((1, 3, 5),) * 2, (np.float32,) * 2, 0, np.float32)
        values = tuple(np.arange(15, dtype=np.float32).reshape(1, 3, 5) for _ in range(2))
        target = np.empty((2, 3, 5), np.float32)
        run(values, target)
        self.assertNotIn('long long t0_0', backend.sources[0])
        self.assertIn('q * (4LL)', backend.sources[0])
        run((values[0], values[1][:, :, ::-1]), target)
        self.assertIn('long long t0_0', backend.sources[1])
        np.testing.assert_array_equal(target, np.concatenate((values[0], values[1][:, :, ::-1])))

    def test_four_effective_dimensions_and_original_unit_axes(self):
        backend = _Backend()
        shape = (1, 2, 1, 3, 2, 4)
        run = bind_packing(backend, (shape,) * 4, (np.float64,) * 4, 0, np.float64)
        base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        values = (base, base[:, ::-1, :, :, :, ::-1], np.asfortranarray(base),
                  np.broadcast_to(base[:, :1], shape))
        target = np.empty((4,) + shape[1:], np.float64)
        run(values, target)
        np.testing.assert_array_equal(target, np.concatenate(values))
        self.assertEqual(len(backend.calls[0][2]) * 8, 1304)
        self.assertNotIn('long long s4;', backend.sources[0])

    def test_abi_limit_uses_original_fallback_without_partial_mixed_binding(self):
        sources = []
        backend = SimpleNamespace(__name__='cupy', ElementwiseKernel=Mock(),
            RawKernel=lambda source, *a, **k: sources.append(source) or Mock())
        for rank, expected_mixed in ((10, True), (11, False)):
            shape = (1,) + (2,) * rank
            base = np.zeros(shape, np.float32)
            # C and F layouts disagree at every adjacent non-unit axis;
            # a jointly contiguous high-rank shape is allowed to fold first.
            values = (base, np.asfortranarray(base))
            run = bind_packing(backend, (shape,) * 2, (np.float32,) * 2, 0, np.float32)
            with patch('lqcdweave.contraction.execution.operations.packing_kernel.bind_pointer_arguments', return_value=None) as bind:
                run(values, np.empty((2,) + shape[1:], np.float32))
            self.assertEqual('long long t0_0' in sources[-1], expected_mixed)
            self.assertLessEqual(len(bind.call_args.args[4]) * 8, 3072)
            self.assertEqual(bind.call_count, 1)

    def test_native_abi_refreshes_current_addresses_strides_and_releases_owners(self):
        backend = _Backend()
        bindings = []
        def bind(cp, kernel, grid, block, template, positions, **kwargs):
            result = _BoundPointerArguments(kernel, grid, block, _Slot, template, positions, **kwargs)
            bindings.append(result)
            return result
        run = bind_packing(backend, ((1, 3, 5),) * 34, (np.complex64,) * 34,
                           0, np.complex128, input_layouts=((None, 0, 1),) * 34)
        with patch('lqcdweave.contraction.execution.operations.packing_kernel.bind_pointer_arguments', side_effect=bind):
            for phase in range(3):
                values = _values(34, phase)
                expected = np.stack(values)
                target = _Output(np.empty(expected.shape, np.complex128))
                refs = [weakref.ref(value) for value in values] + [weakref.ref(target), weakref.ref(target.value)]
                backend.fail = phase == 1
                if backend.fail:
                    with self.assertRaisesRegex(RuntimeError, 'packing launch failed'):
                        run.pack_original(values, target)
                else:
                    self.assertIs(run.pack_original(values, target), target)
                    np.testing.assert_array_equal(target.value, expected)
                    last_calls = backend.calls[-2:]
                    for start, call in zip((0, 32), last_calls):
                        count = 32 if start == 0 else 2
                        self.assertEqual(call[2][:count], tuple(v.ctypes.data for v in values[start:start+count]))
                        self.assertTrue(all(v == 0 for v in call[2][count:32]))
                        self.assertEqual(call[2][-3:-1], (510, start * 15))
                for binding in bindings:
                    if hasattr(binding._local, 'storage'):
                        self.assertEqual(tuple(binding._local.storage[0]), tuple(map(int, binding.fixed_words)))
                    self.assertEqual(binding.constant_owners, ())
                    # The first trailing two-input chunk is C/reversed-C:
                    # its two coordinates fold together. Other chunks include
                    # a broadcast or F layout that preserves their boundary.
                    self.assertEqual(len(binding.fixed_words) * 8,
                                     536 if binding is bindings[1] else 792)
                del values, target
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(len(backend.sources), 2)

    def test_common_signed_dimensions_fold_across_singletons_and_stack_axis(self):
        for axis in (0, 2, 4):
            backend = _Backend()
            layout = tuple(list(range(4))[:axis] + [None] + list(range(4))[axis:])
            shape = tuple(1 if i is None else (2, 3, 2, 5)[i] for i in layout)
            count = 65
            run = bind_packing(backend, (shape,) * count, (np.float64,) * count,
                               axis, np.float64, input_layouts=(layout,) * count)
            values = []
            for lane in range(count):
                columns = 6 + lane % 3
                base = np.arange(12 * columns, dtype=np.float64).reshape(2, 3, 2, columns)
                value = base[..., :5]
                if lane % 3 == 1:
                    value = value[::-1, ::-1, ::-1, ::-1]
                elif lane % 3 == 2:
                    value = np.broadcast_to(base[:1, :1, :1, :1], value.shape)
                values.append(value)
            expected = np.stack(values, axis=axis)
            target = np.empty_like(expected)
            run.pack_original(values, target)
            np.testing.assert_array_equal(target, expected)
            self.assertEqual([len(call[2]) * 8 for call in backend.calls], [792, 792, 280])
            self.assertIn('(q / 5LL) * s0 + (q % 5LL) * s1', backend.sources[0])

    def test_one_disagreeing_lane_preserves_all_required_coordinate_boundaries(self):
        backend = _Backend()
        values = []
        for lane in range(32):
            columns = 6 + lane % 3
            base = np.arange(12 * columns, dtype=np.float32).reshape(2, 3, 2, columns)
            value = base[..., :5]
            if lane == 31:
                value = value.transpose(2, 1, 0, 3)
            values.append(value[None])
        shape = (1, 2, 3, 2, 5)
        run = bind_packing(backend, (shape,) * 32, (np.float32,) * 32, 0, np.float32)
        target = np.empty((32, 2, 3, 2, 5), np.float32)
        run(values, target)
        np.testing.assert_array_equal(target, np.concatenate(values))
        self.assertEqual(len(backend.calls[0][2]) * 8, 1304)

    def test_high_rank_common_contiguity_can_fit_bounded_abi_after_folding(self):
        backend = _Backend()
        shape = (1,) + (2,) * 11
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values = (base, base[(slice(None),) + (slice(None, None, -1),) * 11])
        run = bind_packing(backend, (shape,) * 2, (np.float32,) * 2, 0, np.float32)
        target = np.empty((2,) + shape[1:], np.float32)
        run(values, target)
        np.testing.assert_array_equal(target, np.concatenate(values))
        self.assertEqual(len(backend.calls[0][2]) * 8, 536)

    def test_stride_variant_eviction_does_not_recompile_or_keep_arrays(self):
        backend = _Backend()
        run = bind_packing(backend, ((1, 3, 5),) * 2, (np.float32,) * 2, 0, np.float32)
        for skip in (*range(1, 12), 1):
            base = np.arange(15 * skip, dtype=np.float32).reshape(1, 3, 5 * skip)
            values = (base[:, :, ::skip], base[:, ::-1, ::skip])
            target = np.empty((2, 3, 5), np.float32)
            run(values, target)
            np.testing.assert_array_equal(target, np.concatenate(values))
        self.assertEqual(len(backend.sources), 1)

    def test_current_deferred_owners_supply_strided_views_without_materialization(self):
        backend = _Backend()
        run = bind_packing(backend, ((1, 3, 5),) * 2, (np.complex64,) * 2,
                           0, np.complex128, input_layouts=((None, 0, 1),) * 2)
        for phase in range(2):
            values, expected, refs = [], [], []
            for columns in (10, 15):
                owner = (np.arange(3 * columns).reshape(3, columns) + phase * 100).astype(np.complex64)
                layout = bind_result_views(owner.shape, owner.dtype,
                    (('result', (slice(0, 3), slice(1, 6)), ()),))[0][1]
                values.append(_DeferredResultView(_ResultOwner(owner), layout))
                refs.append(weakref.ref(owner))
                expected.append(owner[:, 1:6].copy())
            target = np.empty((2, 3, 5), np.complex128)
            with patch.object(_DeferredResultView, 'array', side_effect=AssertionError('unexpected materialization')):
                run.pack_original(values, target)
            np.testing.assert_array_equal(target, np.stack(expected))
            del owner, values
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(len(backend.sources), 1)


class MixedStrideStackPackingCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as error:
            raise unittest.SkipTest(f'CUDA unavailable: {error}') from error

    def test_cuda_mixed_stack_chunks_permutations_strides_and_fresh_inputs(self):
        cp = self.cp
        for axis in (0, 1, 2):
            layout = tuple([1, 0][:axis] + [None] + [1, 0][axis:])
            shape = tuple(1 if index is None else (3, 5)[index] for index in layout)
            run = bind_packing(cp, (shape,) * 65, (np.complex64,) * 65,
                               axis, np.complex128, input_layouts=(layout,) * 65)
            for phase in (0, 1):
                host = _values(65, phase)
                values = []
                for lane in range(65):
                    base = cp.asarray((np.arange(15).reshape(3, 5) + 100 * phase + lane).astype(np.complex64))
                    base += 1j * base / 8
                    style = (lane + phase) % 4
                    values.append(base if style == 0 else base[::-1, ::-1] if style == 1 else
                                  cp.broadcast_to(base[:1], base.shape) if style == 2 else cp.asfortranarray(base))
                expected = np.stack([v.T for v in host], axis=axis)
                target = cp.empty(expected.shape, cp.complex128)
                run.pack_original(values, target)
                np.testing.assert_array_equal(cp.asnumpy(target), expected)

    def test_cuda_production_shapes_four_effective_strides(self):
        cp = self.cp
        for shape in ((4, 120, 4, 120), (4, 70, 4, 120), (4, 120, 4, 70)):
            host = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(np.complex64)
            host += 1j * host / 8
            expected = np.stack((host, host[::-1, :, :, ::-1], np.asfortranarray(host),
                                 np.broadcast_to(host[:1], shape)))
            base = cp.asarray(host)
            values = (base, base[::-1, :, :, ::-1], cp.asfortranarray(base),
                      cp.broadcast_to(base[:1], shape))
            run = bind_packing(cp, ((1,) + shape,) * 4, (np.complex64,) * 4, 0,
                               np.complex64, input_layouts=((None, 0, 1, 2, 3),) * 4)
            target = cp.empty(expected.shape, cp.complex64)
            run.pack_original(values, target)
            np.testing.assert_array_equal(cp.asnumpy(target), expected)

    def test_cuda_common_signed_contiguity_across_chunks_and_stack_axis(self):
        cp = self.cp
        for axis in (0, 2):
            host, values = [], []
            for lane in range(65):
                columns = 121 + lane % 3
                base = np.arange(4 * 120 * 4 * columns, dtype=np.float32).reshape(4, 120, 4, columns)
                base = base.astype(np.complex64)
                base += 1j * base / 8
                device = cp.asarray(base)
                value, gpu = base[..., :120], device[..., :120]
                if lane % 3 == 1:
                    value, gpu = value[::-1, ::-1, ::-1, ::-1], gpu[::-1, ::-1, ::-1, ::-1]
                elif lane % 3 == 2:
                    value = np.broadcast_to(base[:1, :1, :1, :1], value.shape)
                    gpu = cp.broadcast_to(device[:1, :1, :1, :1], gpu.shape)
                host.append(value)
                values.append(gpu)
            layout = tuple(list(range(4))[:axis] + [None] + list(range(4))[axis:])
            shape = tuple(1 if i is None else (4, 120, 4, 120)[i] for i in layout)
            run = bind_packing(cp, (shape,) * 65, (np.complex64,) * 65,
                               axis, np.complex64, input_layouts=(layout,) * 65)
            expected = np.stack(host, axis=axis)
            target = cp.empty(expected.shape, cp.complex64)
            run.pack_original(values, target)
            np.testing.assert_array_equal(cp.asnumpy(target), expected)


if __name__ == '__main__':
    unittest.main()
