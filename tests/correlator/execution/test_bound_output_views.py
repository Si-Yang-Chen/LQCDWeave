"""Bound output views preserve snapshots while avoiding redundant reshapes."""
import copy
import gc
import pickle
import weakref
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, TensorSpec,
)
from lqcdweave.contraction.execution.operations.dense_batch import (
    bind_dense_batch, contract_dense_batch,
)
from lqcdweave.contraction.execution.operations.output import (
    PackedOutputs, _BoundLayoutPackedOutputs, bind_output_layout,
)
from lqcdweave.contraction.execution.operations.output_kernel import _raw_copy_kernel
from lqcdweave.contraction.execution.bound_program import bind_program
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.planning.api import CostProfile, describe_dense_batch


def _dense_case(kind, backend='numpy'):
    specs = {
        'l0': TensorSpec((4,), np.complex128),
        'l1': TensorSpec((4,), np.complex128),
    }
    if kind == 'vector':
        specs.update(r0=TensorSpec((4, 2), np.complex128),
                     r1=TensorSpec((4, 3), np.complex128))
        pattern = AxisPattern(((0,), (0, 1)), (1,))
    elif kind == 'scalar':
        specs.update(r0=TensorSpec((4,), np.complex128),
                     r1=TensorSpec((4,), np.complex128))
        pattern = AxisPattern(((0,), (0,)), ())
    else:
        raise ValueError(kind)

    actions = []
    for i in range(2):
        for j in range(2):
            left, right = f'l{i}', f'r{j}'
            output = f'{kind}{i}{j}'
            shape = specs[right].shape[1:] if kind == 'vector' else ()
            specs[output] = TensorSpec(shape, np.complex128)
            actions.append(Action('contract', output, (left, right), pattern, seconds=1e9))
    batch = describe_dense_batch(actions, tuple(range(4)), specs, CostProfile(), backend=backend)
    if batch is None:
        raise AssertionError('fixture must lower to one complete dense batch')
    return batch, specs


def _dense_arrays(specs, seed):
    rng = np.random.default_rng(seed)
    arrays = {}
    for key, spec in specs.items():
        if key.startswith(('vector', 'scalar')):
            continue
        values = rng.normal(size=spec.shape)
        if spec.dtype.kind == 'c':
            values = values + 1j * rng.normal(size=spec.shape)
        arrays[key] = values.astype(spec.dtype)
    return arrays


def _dense_reference(kind, left, right):
    if kind == 'vector':
        return np.einsum(left, (0,), right, (0, 1), (1,), optimize=False)
    return np.einsum(left, (0,), right, (0,), (), optimize=False)


class _RawCopyKernel:
    def __init__(self, backend, source):
        self.backend = backend
        self.source = source

    def __call__(self, grid, block, arguments):
        values, offset, destination = arguments[:-2], int(arguments[-2]), arguments[-1]
        self.backend.raw_launches.append((len(values), grid, block))
        for lane, value in enumerate(values):
            destination[offset + lane] = value


class _RawCopyBackend:
    __name__ = 'cupy'

    def __init__(self):
        self.raw_sources = []
        self.raw_launches = []
        self.elementwise_launches = []
        self.allocations = []

    def empty(self, shape, dtype):
        self.allocations.append((tuple(shape), np.dtype(dtype)))
        return np.empty(shape, dtype=dtype)

    def RawKernel(self, source, name, options=()):
        self.raw_sources.append((source, name, options))
        return _RawCopyKernel(self, source)

    def ElementwiseKernel(self, inputs, outputs, operation, name, **kwargs):
        def launch(*arguments, size):
            values, offset, destination = arguments[:-3], int(arguments[-2]), arguments[-1]
            self.elementwise_launches.append(len(values))
            for lane, value in enumerate(values):
                destination[offset + lane] = value
        return launch


def _empty_program_state(results, catalog):
    return SimpleNamespace(actions=(), results=tuple(results), catalog=catalog, problem=None)


def _bind_empty_program(results, catalog, backend):
    return bind_program(
        _empty_program_state(results, catalog), backend, {}, {}, {}, {}, (), (),
        (None, None, None, None),
    )


def _run_dense_case(kind, backend=np):
    batch, specs = _dense_case(kind, 'cupy' if backend is not np else 'numpy')
    host = _dense_arrays(specs, seed=810 + (kind == 'scalar'))
    arrays = host if backend is np else {key: backend.asarray(value) for key, value in host.items()}
    expected = {
        f'{kind}{i}{j}': _dense_reference(kind, host[f'l{i}'], host[f'r{j}'])
        for i in range(2) for j in range(2)
    }
    phases = []

    def timed(name, function, **options):
        phases.append((name, options))
        return function()

    output = contract_dense_batch(batch, arrays, backend=backend, timed=timed,
                                  device=backend is not np)
    if backend is not np:
        backend.cuda.get_current_stream().synchronize()
    for key, value in expected.items():
        actual = arrays[key] if backend is np else backend.asnumpy(arrays[key])
        np.testing.assert_allclose(actual, value, rtol=3e-12, atol=3e-12)
        if backend is np:
            assert np.shares_memory(arrays[key], output)
        else:
            assert output.data.ptr <= arrays[key].data.ptr < output.data.ptr + output.nbytes
    assert [name for name, _ in phases] == [
        'dense_batch_pack', 'dense_batch_pack', 'contract_kernel',
    ]
    assert all(options.get('device', False) is (backend is not np)
               for _, options in phases)
    return batch, specs, arrays, output, expected, phases


class PackedOutputViewTests(unittest.TestCase):
    def test_layout_views_keep_owner_order_snapshot_and_zero_dimensional_arrays(self):
        def make_outputs():
            first = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
            second = (np.arange(12, dtype=np.float32).reshape(2, 2, 3) + 20)
            scalar = np.array(7.5, dtype=np.float64)
            sources = {'a': first, 'b': second, 's': scalar}
            refs = tuple(weakref.ref(value) for value in sources.values())
            layout = bind_output_layout(('b', 's', 'a', 'b'), {
                key: TensorSpec(value.shape, value.dtype) for key, value in sources.items()
            })
            expected = tuple(sources[key].copy() for key in ('b', 's', 'a', 'b'))
            packed = layout.copy(sources, np)
            for value in sources.values():
                value[...] = -1
            return packed, expected, refs

        packed, expected, refs = make_outputs()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertEqual(packed.positions, ((0, 0), (1, 0), (0, 1), (0, 0)))
        self.assertEqual(len(packed.owners), 2)
        for index, (actual, reference) in enumerate(zip(packed, expected, strict=True)):
            self.assertEqual(actual.shape, reference.shape)
            self.assertEqual(actual.dtype, reference.dtype)
            np.testing.assert_array_equal(actual, reference)
            if index == 1:
                self.assertIs(type(actual), np.ndarray)
                self.assertEqual(actual.shape, ())
                self.assertTrue(np.shares_memory(actual, packed.owners[1]))
            else:
                self.assertIs(actual.base, packed.owners[0])
                self.assertTrue(np.shares_memory(actual, packed.owners[0]))

    def test_positions_can_reorder_and_repeat_the_same_owner_lane(self):
        owner = np.empty((2, 3, 4), dtype=np.int64)
        owner[...] = np.arange(24, dtype=np.int64).reshape(owner.shape)
        packed = PackedOutputs((owner,), ((0, 1), (0, 0), (0, 1)))
        self.assertEqual(packed.positions, ((0, 1), (0, 0), (0, 1)))
        for actual, lane in zip(packed, (1, 0, 1), strict=True):
            self.assertIs(actual.base, owner)
            np.testing.assert_array_equal(actual, owner[lane])
            self.assertTrue(np.shares_memory(actual, owner))

    def test_non_scalar_squeeze_compiles_single_index_and_preserves_measurement_phases(self):
        batch, specs, arrays, output, expected, phases = _run_dense_case('vector')
        self.assertIsNone(batch.packing_dtype)
        self.assertEqual([value.shape for value in expected.values()], [(2,), (3,), (2,), (3,)])
        bound = bind_dense_batch(batch, specs, np)
        slices = bound.func.__self__.result_slices
        self.assertTrue(all(isinstance(bounds[0], int) and squeeze == ()
                            for _, bounds, squeeze in slices))
        self.assertTrue(all(isinstance(bounds[1], slice) for _, bounds, _ in slices))
        self.assertEqual([name for name, _ in phases], [
            'dense_batch_pack', 'dense_batch_pack', 'contract_kernel',
        ])
        self.assertEqual(output.shape, (2, 5))

    def test_scalar_squeeze_keeps_slice_path_and_zero_dimensional_array_results(self):
        batch, specs, arrays, output, expected, phases = _run_dense_case('scalar')
        self.assertIsNone(batch.packing_dtype)
        bound = bind_dense_batch(batch, specs, np)
        slices = bound.func.__self__.result_slices
        self.assertTrue(all(all(isinstance(axis, slice) for axis in bounds)
                            and squeeze == (0, 1) for _, bounds, squeeze in slices))
        self.assertEqual(output.shape, (2, 2))
        for key in expected:
            self.assertIs(type(arrays[key]), np.ndarray)
            self.assertEqual(arrays[key].shape, ())
        self.assertEqual([name for name, _ in phases], [
            'dense_batch_pack', 'dense_batch_pack', 'contract_kernel',
        ])


class BoundOutputSnapshotProgramTests(unittest.TestCase):
    def setUp(self):
        _raw_copy_kernel.cache_clear()
        self.backend = _RawCopyBackend()

    def tearDown(self):
        _raw_copy_kernel.cache_clear()

    def test_raw_snapshot_uses_current_strides_bounded_chunks_and_owned_results(self):
        base_values = np.arange(800, dtype=np.float64).reshape(20, 40)
        base = (base_values + 1j * (base_values + 0.5)).astype(np.complex128)
        transposed_values = np.arange(800, dtype=np.float64).reshape(40, 20)
        transposed = (transposed_values + 1j * (transposed_values + 0.25)).astype(np.complex128)
        arrays = {}
        for index in range(259):
            row, column = index % 17, (index * 3) % 36
            region = base[row:row + 3, column:column + 4]
            mode = index % 4
            if mode == 0:
                value = region
            elif mode == 1:
                value = region[::-1]
            elif mode == 2:
                value = region[:, ::-1]
            else:
                row2, column2 = (index * 2) % 36, (index * 3) % 17
                value = transposed[row2:row2 + 4, column2:column2 + 3].T
            arrays[str(index)] = value
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        expected = tuple(np.array(value, copy=True) for value in arrays.values())
        references = tuple(weakref.ref(value) for value in arrays.values())
        layout = bind_output_layout(tuple(arrays), catalog, backend=self.backend, validate=False)

        first = layout.copy(arrays, self.backend)
        self.assertFalse(layout.validate)
        self.assertEqual(len(first.owners), 1)
        self.assertEqual([count for count, _grid, _block in self.backend.raw_launches], [128, 128, 3])
        self.assertTrue(all((count+2)*8 <= 3072 for count, _grid, _block in self.backend.raw_launches))
        self.assertEqual(len(self.backend.elementwise_launches), 0)
        self.assertEqual(len(layout.copy_programs[0]._variants), 1)
        self.assertTrue(any('(-16LL)' in source for source, _name, _options in self.backend.raw_sources))
        self.assertTrue(all('destination_lane_offset=' not in source
                            for source, _name, _options in self.backend.raw_sources))
        self.assertEqual(len(self.backend.raw_sources), 2)
        compiled_kernels = len(self.backend.raw_sources)
        for actual, reference in zip(first, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
        for value in arrays.values():
            value[...] = 0
        for actual, reference in zip(first, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)

        second = layout.copy(arrays, self.backend)
        self.assertEqual([count for count, _grid, _block in self.backend.raw_launches],
                         [128, 128, 3, 128, 128, 3])
        self.assertEqual(len(self.backend.raw_sources), compiled_kernels)
        for actual in second:
            np.testing.assert_array_equal(actual, np.zeros((3, 4), dtype=np.complex128))

        del arrays, region, value
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_raw_lane_width_respects_pointer_offset_and_destination_budget(self):
        from lqcdweave.contraction.execution.operations import output_kernel
        arrays={str(i):np.array(i,dtype=np.int64) for i in range(9)}
        catalog={key:TensorSpec(value.shape,value.dtype) for key,value in arrays.items()}
        with patch.object(output_kernel,'_MAX_ARGUMENT_BYTES',80):
            layout=bind_output_layout(tuple(arrays),catalog,backend=self.backend,validate=False)
        copied=layout.copy(arrays,self.backend)
        self.assertEqual([count for count,_,_ in self.backend.raw_launches],[8,1])
        self.assertTrue(all((count+2)*8<=80 for count,_,_ in self.backend.raw_launches))
        np.testing.assert_array_equal(np.asarray(copied).reshape(-1),np.arange(9))

    def test_stride_variant_cache_is_bounded_and_contains_metadata_only(self):
        base = np.arange(10000, dtype=np.float32).reshape(100, 100)
        catalog = {'x': TensorSpec((2, 3), np.float32)}
        layout = bind_output_layout(('x',), catalog, backend=self.backend, validate=False)
        copier = layout.copy_programs[0]
        for row_step in range(1, 10):
            column_step = 1 + row_step % 4
            value = base[:2 * row_step:row_step, :3 * column_step:column_step]
            self.assertEqual(value.shape, (2, 3))
            packed = layout.copy({'x': value}, self.backend)
            np.testing.assert_array_equal(packed[0], value)
        self.assertEqual(len(copier._variants), 8)
        self.assertEqual(_raw_copy_kernel.cache_info().maxsize, 64)
        for key, calls in copier._variants.items():
            self.assertTrue(all(type(stride) is int for input_strides in key for stride in input_strides))
            for kernel, native, start, stop, offset, grid, block in calls:
                self.assertIsInstance(kernel, _RawCopyKernel)
                self.assertIsNone(native)
                self.assertTrue(all(type(value) is int for value in (start, stop)))
                self.assertIsInstance(offset, np.int64)
                self.assertTrue(all(type(value) is int for value in (*grid, *block)))

    def test_bound_program_checks_internal_shape_dtype_only_in_current_debug_context(self):
        catalog = {'out': TensorSpec((3,), np.float32)}
        with execution_debug(False):
            program = _bind_empty_program(('out',), catalog, self.backend)
        self.assertFalse(program.output_layout.validate)
        self.assertIsNotNone(program.output_layout.copy_programs[0])
        result = program.run(None, {'out': np.array([1, 2, 3], dtype=np.float32)})
        np.testing.assert_array_equal(result[0], [1, 2, 3])

        raw_launches = len(self.backend.raw_launches)
        with execution_debug(True):
            for bad in (np.ones(4, dtype=np.float32), np.ones(3, dtype=np.float64)):
                with self.assertRaisesRegex(ValueError, 'saved output layout'):
                    program.run(None, {'out': bad})
        with execution_debug(True):
            checked = program.run(None, {'out': np.array([4, 5, 6], dtype=np.float32)})
        np.testing.assert_array_equal(checked[0], [4, 5, 6])
        self.assertEqual(len(self.backend.raw_launches), raw_launches)
        self.assertEqual(self.backend.elementwise_launches, [1])

        direct = bind_output_layout(('out',), catalog)
        self.assertTrue(direct.validate)
        with self.assertRaisesRegex(ValueError, 'saved output layout'):
            direct.copy({'out': np.ones(4, dtype=np.float32)}, self.backend)

    def test_native_snapshot_provenance_is_private_and_serialization_drops_it(self):
        arrays = {'a': np.arange(3, dtype=np.float32), 'b': np.arange(3, dtype=np.float32) + 5}
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        layout = bind_output_layout(('b', 'a', 'b'), catalog,
                                    backend=self.backend, validate=False)
        with execution_debug(False):
            packed = layout.copy(arrays, self.backend)
        self.assertIs(type(packed), _BoundLayoutPackedOutputs)
        self.assertIsInstance(packed, tuple)
        self.assertIs(packed.provenance, layout)
        self.assertEqual(packed.positions, ((0, 0), (0, 1), (0, 0)))
        with self.assertRaises(AttributeError):
            packed.provenance = None

        for restored in (copy.copy(packed), copy.deepcopy(packed),
                         pickle.loads(pickle.dumps(packed))):
            self.assertIs(type(restored), PackedOutputs)
            self.assertEqual(restored.positions, packed.positions)
            self.assertEqual(len(restored), len(packed))

        self.assertIs(type(PackedOutputs(packed.owners, packed.positions)), PackedOutputs)
        validated = bind_output_layout(('a', 'b'), catalog,
                                       backend=self.backend, validate=True).copy(arrays, self.backend)
        self.assertIs(type(validated), PackedOutputs)
        numpy_snapshot = bind_output_layout(('a', 'b'), catalog, validate=False).copy(arrays, np)
        self.assertIs(type(numpy_snapshot), PackedOutputs)
        with execution_debug(True):
            debug_snapshot = layout.copy(arrays, self.backend)
        self.assertIs(type(debug_snapshot), PackedOutputs)

    def test_float16_missing_rawkernel_and_debug_keep_legacy_copy_path(self):
        cases = []
        half_backend = _RawCopyBackend()
        cases.append((half_backend, np.float16, False))
        missing_backend = _RawCopyBackend()
        missing_backend.RawKernel = None
        cases.append((missing_backend, np.float32, False))
        debug_backend = _RawCopyBackend()
        cases.append((debug_backend, np.float32, True))
        for backend, dtype, debug in cases:
            with self.subTest(dtype=dtype, debug=debug, raw=callable(backend.RawKernel)):
                catalog = {'x': TensorSpec((2, 3), dtype)}
                layout = bind_output_layout(('x',), catalog, backend=backend, validate=False)
                self.assertEqual(layout.copy_programs[0] is None,
                                 dtype == np.float16 or not callable(backend.RawKernel))
                value = np.arange(6, dtype=dtype).reshape(2, 3)
                with execution_debug(debug):
                    packed = layout.copy({'x': value}, backend)
                np.testing.assert_array_equal(packed[0], value)
                if debug:
                    self.assertEqual(backend.raw_sources, [])
                self.assertEqual(backend.elementwise_launches, [1])


class RealCuPyBoundOutputViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as error:
            raise unittest.SkipTest(f'actual CuPy output oracle requires a working device: {error}')
        cls.cp = cupy

    def test_actual_gpu_oracle_for_multilane_scalar_and_squeezed_batch_views(self):
        cp = self.cp
        host_owner = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        host_scalar = np.array(3.25, dtype=np.float64)
        sources = {
            'a': cp.asarray(host_owner[0]),
            'b': cp.asarray(host_owner[1]),
            's': cp.asarray(host_scalar),
        }
        layout = bind_output_layout(('b', 's', 'a', 'b'), {
            key: TensorSpec(value.shape, value.dtype) for key, value in sources.items()
        })
        packed = layout.copy(sources, cp)
        owner, scalar_owner = packed.owners
        self.assertIsInstance(packed[1], cp.ndarray)
        self.assertEqual(packed[1].shape, ())
        self.assertEqual(packed[0].data.ptr, owner.data.ptr)
        self.assertEqual(packed[2].data.ptr, owner.data.ptr + owner.strides[0])
        self.assertEqual(packed[3].data.ptr, owner.data.ptr)
        self.assertEqual(packed[1].data.ptr, scalar_owner.data.ptr)
        repeated = PackedOutputs(packed.owners, ((0, 0), (1, 0), (0, 0), (0, 0)))
        self.assertEqual(repeated[0].data.ptr, repeated[3].data.ptr)
        for value in sources.values():
            value[...] = 0
        np.testing.assert_array_equal(cp.asnumpy(packed[0]), host_owner[1])
        np.testing.assert_array_equal(cp.asnumpy(packed[1]), host_scalar)
        np.testing.assert_array_equal(cp.asnumpy(packed[2]), host_owner[0])
        np.testing.assert_array_equal(cp.asnumpy(packed[3]), host_owner[1])
        np.testing.assert_array_equal(cp.asnumpy(repeated[0]), host_owner[1])
        np.testing.assert_array_equal(cp.asnumpy(repeated[1]), host_scalar)
        np.testing.assert_array_equal(cp.asnumpy(repeated[2]), host_owner[1])
        np.testing.assert_array_equal(cp.asnumpy(repeated[3]), host_owner[1])

        for kind in ('vector', 'scalar'):
            with self.subTest(kind=kind):
                _batch, _specs, arrays, _output, _expected, phases = _run_dense_case(kind, cp)
                self.assertEqual([name for name, _ in phases], [
                    'dense_batch_pack', 'dense_batch_pack', 'contract_kernel',
                ])
                self.assertTrue(all(options.get('device') for _, options in phases))


class RealCuPyOutputSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as error:
            raise unittest.SkipTest(f'actual CuPy snapshot oracle requires a working device: {error}')
        cls.cp = cupy

    def test_bound_raw_snapshot_copies_strides_scalar_empty_and_repeated_results(self):
        cp = self.cp
        host_base = (np.arange(24).reshape(4, 6) + 1j * np.arange(24, 48).reshape(4, 6)).astype(np.complex128)
        host_transposed = (np.arange(6).reshape(3, 2) + 1j * np.arange(6, 12).reshape(3, 2)).astype(np.complex128)
        host_values = {
            'a': cp.asarray(host_base)[::2, 1:4],
            'b': cp.asarray(host_transposed).T,
            'c': cp.asarray(host_base)[::2, 1:4][::-1, ::-1],
            's': cp.asarray(np.array(3.25, dtype=np.float64)),
            'e': cp.empty((0, 2), dtype=cp.complex64),
        }
        order = ('b', 's', 'a', 'e', 'c', 'b')
        catalog = {key: TensorSpec(tuple(value.shape), value.dtype)
                   for key, value in host_values.items()}
        expected = {key: np.array(cp.asnumpy(value), copy=True)
                    for key, value in host_values.items()}
        program = _bind_empty_program(order, catalog, cp)
        self.assertTrue(all(copier is not None
                            for copier in program.output_layout.copy_programs))
        with patch.object(cp, 'ElementwiseKernel',
                          side_effect=AssertionError('supported snapshots must use RawKernel')):
            packed = program.run(None, host_values)
        self.assertEqual(packed.positions, ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (0, 0)))
        self.assertEqual(len(packed.owners), 3)
        for output, key in zip(packed, order, strict=True):
            self.assertEqual(output.shape, expected[key].shape)
            self.assertEqual(output.dtype, expected[key].dtype)
            np.testing.assert_array_equal(cp.asnumpy(output), expected[key])
        for value in host_values.values():
            value[...] = 0
        for output, key in zip(packed, order, strict=True):
            np.testing.assert_array_equal(cp.asnumpy(output), expected[key])


if __name__ == '__main__':
    unittest.main()
