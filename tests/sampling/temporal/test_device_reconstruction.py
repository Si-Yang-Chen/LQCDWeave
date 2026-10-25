"""Device reduction protocol and original streaming-CPU numerical oracles."""

import gc
from math import prod
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np

from lqcdweave.contracts.correlator.cfg_plan import CfgOutputField
from lqcdweave.sampling.temporal.device_reconstruction import (
    DeviceTimePairAccumulator, estimate_device_accumulator_bytes,
)
from lqcdweave.sampling.temporal.reconstruction import TimePairAccumulator
from lqcdweave.storage._cfg_correlator_output import PackedDtOutputWriter


def _field(path, shape=(2, 3), dtype='complex64', *, dt=True, role=None):
    dtype = np.dtype(dtype)
    return CfgOutputField(path, shape, dtype.str, 0,
                          (prod(shape) * dtype.itemsize + 7) // 8,
                          0 if dt else -1, role)


class _DeviceArray:
    """Small CPU protocol double whose implicit host coercion is forbidden."""
    def __init__(self, value):
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def __array__(self, *args, **kwargs):
        raise AssertionError('device value was coerced to NumPy')

    def __getitem__(self, index):
        return _DeviceArray(self.value[index])

    def __setitem__(self, index, value):
        self.value[index] = value.value if isinstance(value, _DeviceArray) else value


class _DeviceBackend:
    __name__ = 'cupy'
    ndarray = _DeviceArray

    def __init__(self):
        self.allocations = []
        self.adds = 0
        self.divides = 0

    def zeros(self, shape, dtype):
        self.allocations.append((shape, np.dtype(dtype)))
        return _DeviceArray(np.zeros(shape, dtype=dtype))

    def empty(self, shape, dtype):
        self.allocations.append((shape, np.dtype(dtype)))
        return _DeviceArray(np.empty(shape, dtype=dtype))

    def moveaxis(self, value, source, destination):
        return _DeviceArray(np.moveaxis(value.value, source, destination))

    def add(self, left, right, *, out):
        self.adds += 1
        np.add(left.value, right.value, out=out.value)

    def divide(self, left, right, *, out):
        self.divides += 1
        np.divide(left.value, right, out=out.value)

    def asarray(self, *args, **kwargs):
        raise AssertionError('unexpected host-to-device upload')

    def asnumpy(self, *args, **kwargs):
        raise AssertionError('unexpected device-to-host transfer')


class DeviceAccumulatorTests(unittest.TestCase):
    def test_all_native_mean_dtypes_match_original_ordered_cpu_accumulator(self):
        rng = np.random.default_rng(317)
        pairs = ((7, 1), (3, 5), (1, 4), (4, 7), (0, 2), (7, 3))
        dts = (3, 2, 0, 7, 4, 2)
        for dtype in ('float16', 'float32', 'float64', 'complex64', 'complex128'):
            for shape in ((), (2, 3), (0, 2)):
                with self.subTest(dtype=dtype, shape=shape):
                    field = _field(('value',), shape, dtype)
                    cpu = TimePairAccumulator({field.path: (shape, dtype)}, Lt=8)
                    device = DeviceTimePairAccumulator((field,), Lt=8, backend=np)
                    for source, sink in pairs:
                        value = rng.normal(size=shape)
                        if np.dtype(dtype).kind == 'c':
                            value = value + 1j * rng.normal(size=shape)
                        value = np.asarray(value, dtype=dtype)
                        cpu.add(source, sink, ((field.path, value),))
                        device.add(source, sink, {field.path: value})
                    expected = np.empty((*shape, len(dts)), dtype=dtype)
                    cpu.write_means(field.path, dts, expected)
                    actual = device.means(field.path, dts)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(device.counts, cpu.counts)
                    np.testing.assert_array_equal(device.sums[field.path], cpu.sums[field.path])
                    self.assertEqual(actual.dtype, np.dtype(dtype))
                    self.assertTrue(actual.flags.c_contiguous)

    def test_original_packed_writer_oracle_first_seen_and_physical_wraparound(self):
        fields = (_field(('corr',)),
                  _field(('source_loop',), (2,), 'complex128', dt=False, role='source'),
                  _field(('sink_loop',), (), 'int16', dt=False, role='sink'))
        coordinates = tuple({'source': source, 'sink': sink}
                            for source, sink in ((2, 0), (0, 2), (1, 2),
                                                 (0, 1), (2, 1), (1, 0)))
        times = (6, 1, 4)
        plan = SimpleNamespace(output_fields=fields, physical_lt=8, one_point_slots=3,
                               output_layout='dt_packed', time_roles=('sink', 'source'),
                               coordinate_shape=(len(coordinates),))
        binding = SimpleNamespace(physical_times=times, physical_lt=8)
        cpu = PackedDtOutputWriter(plan, 'cfg', '.', binding, coordinates=coordinates)
        device = DeviceTimePairAccumulator(fields, Lt=8, backend=np, one_point_slots=3)
        for index, point in enumerate(coordinates):
            value = {'corr': np.full((2, 3), index + 1j * (index + 1), dtype='complex64'),
                     'source_loop': np.array([index + 2j, index - 3j], dtype='complex128'),
                     'sink_loop': np.array(index + 11, dtype='int16')}
            cpu.consume(index, value)
            device.add(times[point['source']], times[point['sink']],
                       {field.path: value[field.path[0]] for field in fields},
                       slots=point, reject_equal=True)
            # Reused contribution buffers must not remain aliased to either result.
            for array in value.values():
                array[...] = -999
        result = device.finalize(range(1, 8))
        expected = np.empty((2, 3, 7), dtype='complex64')
        cpu.accumulator.write_means(('corr',), range(1, 8), expected)
        np.testing.assert_array_equal(result[('corr',)], expected)
        np.testing.assert_array_equal(device.counts, cpu.counts)
        for field in fields[1:]:
            np.testing.assert_array_equal(result[field.path], np.moveaxis(cpu.loops[field.path], 0, -1))
            np.testing.assert_array_equal(device.seen[field.path], cpu.seen[field.path])

    def test_complex_native_rounding_and_order_are_not_promoted_or_reassociated(self):
        for dtype, large in (('complex64', 2 ** 24), ('complex128', 2 ** 54)):
            field = _field((), (), dtype)
            device = DeviceTimePairAccumulator((field,), Lt=4, backend=np)
            for value in (large + 1j * large, 1 + 1j, -large - 1j * large):
                device.add(0, 1, {(): np.array(value, dtype=dtype)})
            np.testing.assert_array_equal(device.means((), (1,)), np.zeros(1, dtype=dtype))
            self.assertEqual(device.counts[1], 3)

    def test_no_device_coercion_transfers_pair_tensor_or_add_time_allocations(self):
        backend = _DeviceBackend()
        fields = (_field(('corr',), (3, 2)),
                  _field(('loop',), (2,), 'uint8', dt=False, role='source'))
        device = DeviceTimePairAccumulator(fields, Lt=17, backend=backend, one_point_slots=2)
        corr = _DeviceArray(np.arange(6, dtype='complex64').reshape(2, 3).T)
        loop = _DeviceArray(np.array([4, 7], dtype='uint8'))
        self.assertEqual(len(backend.allocations), 2)
        for index in range(100):
            device.add(0, 3, {('corr',): corr, ('loop',): loop}, slots={'source': index % 2})
        self.assertEqual(len(backend.allocations), 2)
        out = device.finalize((3, 1, 3))
        np.testing.assert_array_equal(out[('corr',)].value[..., 0], corr.value)
        self.assertTrue(np.isnan(out[('corr',)].value[..., 1]).all())
        np.testing.assert_array_equal(out[('loop',)].value, np.array([[4, 4], [7, 7]], dtype='uint8'))
        self.assertEqual(backend.adds, 100)
        self.assertEqual(backend.divides, 2)
        self.assertEqual([shape for shape, _ in backend.allocations],
                         [(17, 3, 2), (2, 2), (3, 2, 3), (2, 2)])
        self.assertEqual(estimate_device_accumulator_bytes(fields, 17, 2, (3, 1, 3)),
                         sum(prod(shape) * dtype.itemsize for shape, dtype in backend.allocations))

    def test_outputs_and_inputs_have_independent_ownership_and_close_is_idempotent(self):
        fields = (_field(('corr',), (2,)),
                  _field(('loop',), (2,), dt=False, role='source'))
        device = DeviceTimePairAccumulator(fields, Lt=3, backend=np, one_point_slots=1)
        value = np.array([2 + 1j, 6 - 2j], dtype='complex64')
        borrowed = weakref.ref(value)
        device.add(0, 1, {('corr',): value, ('loop',): value}, slots={'source': 0})
        del value
        gc.collect()
        self.assertIsNone(borrowed())
        out = device.finalize((1,))
        for path, array in out.items():
            resident = device.sums.get(path, device.loops.get(path))
            self.assertFalse(np.shares_memory(array, resident))
            array[...] = 42
        np.testing.assert_array_equal(device.means(('corr',), (1,)), [[2 + 1j], [6 - 2j]])
        np.testing.assert_array_equal(device.one_point(('loop',)), [[2 + 1j], [6 - 2j]])
        resident = weakref.ref(device.sums[('corr',)])
        device.close()
        device.close()
        gc.collect()
        self.assertIsNone(resident())
        for array in out.values():
            self.assertTrue((array == 42).all())
        for operation in (lambda: device.means(('corr',), (1,)),
                          lambda: device.one_point(('loop',)),
                          lambda: device.finalize((1,)),
                          lambda: device.add(0, 1, {})):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                operation()

    def test_missing_dt_and_incomplete_one_point_coverage_are_explicit(self):
        fields = (_field(('corr',), ()),
                  _field(('loop',), (), dt=False, role='source'))
        device = DeviceTimePairAccumulator(fields, Lt=5, backend=np, one_point_slots=2)
        value = np.array(4, dtype='complex64')
        device.add(4, 1, {('corr',): value, ('loop',): value}, slots={'source': 1})
        np.testing.assert_array_equal(device.means(('corr',), (2, 0)), [4, np.nan])
        with self.assertRaisesRegex(ValueError, 'no source pairs'):
            device.means(('corr',), (2, 0), missing='raise')
        for operation in (lambda: device.one_point(('loop',)), lambda: device.finalize((2,))):
            with self.assertRaisesRegex(ValueError, 'incomplete one-point'):
                operation()
        device.add(1, 4, {('corr',): value, ('loop',): value}, slots={'source': 0})
        np.testing.assert_array_equal(device.one_point(('loop',)), [4, 4])

    def test_unsigned_physical_endpoints_do_not_overflow(self):
        field = _field((), (), 'float32')
        device = DeviceTimePairAccumulator((field,), Lt=7, backend=np)
        self.assertEqual(device.add(np.uint64(6), np.uint64(1), {(): np.array(2, dtype='float32')}), 2)

    def test_metadata_rejection_precedes_any_numerical_mutation(self):
        fields = (_field(('corr',), ()),
                  _field(('loop',), (), dt=False, role='source'))
        device = DeviceTimePairAccumulator(fields, Lt=5, backend=np, one_point_slots=2)
        value = np.array(1, dtype='complex64')
        good = {('corr',): value, ('loop',): value}
        bad_calls = (
            lambda: device.add(1, 1, good, slots={'source': 0}, reject_equal=True),
            lambda: device.add(0, 2, good),
            lambda: device.add(0, 2, good, slots={'source': 2}),
            lambda: device.add(-1, 2, good, slots={'source': 0}),
            lambda: device.add(0, 5, good, slots={'source': 0}),
            lambda: device.add(0, 2, {('corr',): value}, slots={'source': 0}),
            lambda: device.add(0, 2, {('corr',): value, ('loop',): np.ones(2, dtype='complex64')}, slots={'source': 0}),
            lambda: device.add(0, 2, {('corr',): value, ('loop',): np.array(1, dtype='complex128')}, slots={'source': 0}),
        )
        for operation in bad_calls:
            with self.assertRaises((ValueError, TypeError)):
                operation()
            self.assertFalse(device.counts.any())
            self.assertFalse(device.sums[('corr',)].any())
            self.assertFalse(device.seen[('loop',)].any())
        with self.assertRaisesRegex(TypeError, 'already reside'):
            device.add(0, 2, {('corr',): 1, ('loop',): value}, slots={'source': 0})

    def test_backend_failure_disables_accumulator_but_retains_buffers_until_close(self):
        backend = _DeviceBackend()
        fields = (_field(('first',), ()), _field(('second',), ()))
        device = DeviceTimePairAccumulator(fields, Lt=4, backend=backend)
        residents = [weakref.ref(value) for value in device.sums.values()]
        value = _DeviceArray(np.array(1, dtype='complex64'))
        original = backend.add
        def fail_second(left, right, *, out):
            if backend.adds:
                raise RuntimeError('injected backend failure')
            original(left, right, out=out)
        backend.add = fail_second
        with self.assertRaisesRegex(RuntimeError, 'injected backend failure'):
            device.add(0, 1, {field.path: value for field in fields})
        self.assertTrue(all(reference() is not None for reference in residents))
        self.assertFalse(device.counts.any())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            device.finalize((1,))
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            device.add(0, 1, {field.path: value for field in fields})
        # The enclosing device scope establishes its stream fence here, before
        # explicitly releasing the unusable accumulator's pending outputs.
        device.close()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in residents))

    def test_unsupported_dtypes_are_rejected_before_backend_allocation(self):
        for dtype in ('>c16', '>f8', 'longdouble', 'clongdouble'):
            for dt in (False, True):
                with self.subTest(dtype=dtype, dt=dt):
                    if np.dtype(dtype) in (np.dtype('float64'), np.dtype('complex128')):
                        continue  # Platforms whose longdouble has no extra precision.
                    field = _field((), (), dtype, dt=dt, role='source')
                    backend = _DeviceBackend()
                    with self.assertRaisesRegex(TypeError, 'unsupported device accumulator dtype'):
                        DeviceTimePairAccumulator((field,), Lt=4, backend=backend, one_point_slots=1)
                    with self.assertRaisesRegex(TypeError, 'unsupported device accumulator dtype'):
                        estimate_device_accumulator_bytes((field,), 4, 1, (1, 2))
                    self.assertEqual(backend.allocations, [])

    def test_constructor_failure_keeps_partial_buffers_in_exception_frame(self):
        backend = _DeviceBackend()
        original = backend.zeros
        residents = []
        def fail_second(shape, dtype):
            if residents:
                raise MemoryError('injected allocation failure')
            value = original(shape, dtype)
            residents.append(weakref.ref(value))
            return value
        backend.zeros = fail_second
        fields = (_field(('first',)), _field(('second',)))
        try:
            DeviceTimePairAccumulator(fields, Lt=4, backend=backend)
        except MemoryError as error:
            self.assertTrue(all(reference() is not None for reference in residents))
            # X can fence its scope while the original exception still owns
            # the failed constructor frame. Clearing it models completed unwind.
            error.__traceback__ = None
        else:
            self.fail('injected allocation failure was not propagated')
        gc.collect()
        self.assertTrue(all(reference() is None for reference in residents))

    def test_estimator_is_pure_and_validates_host_metadata(self):
        fields = (_field(('corr',), (3, 4), 'complex128'),
                  _field(('loop',), (7,), 'float32', dt=False, role='sink'))
        with mock.patch('importlib.import_module', side_effect=AssertionError('backend imported')):
            self.assertEqual(estimate_device_accumulator_bytes(fields, 128, 40, range(1, 64)),
                             3 * 4 * 16 * (128 + 63) + 7 * 4 * (40 + 40))
        for dt in ((-1,), (128,), (1.0,), (True,)):
            with self.assertRaisesRegex(ValueError, 'dt coordinates'):
                estimate_device_accumulator_bytes(fields, 128, 40, dt)
        with self.assertRaisesRegex(ValueError, 'one-point fields require'):
            estimate_device_accumulator_bytes(fields, 128, 0, (1,))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            estimate_device_accumulator_bytes((fields[0], fields[0]), 128, 0, (1,))

    def test_debug_finiteness_does_not_mutate_rejected_contribution(self):
        field = _field((), (), 'float32')
        device = DeviceTimePairAccumulator((field,), Lt=4, backend=np, debug=True)
        with self.assertRaisesRegex(ValueError, 'nonfinite correlator'):
            device.add(0, 1, {(): np.array(np.inf, dtype='float32')})
        self.assertFalse(device.counts.any())
        self.assertFalse(device.sums[()].any())


class DeviceAccumulatorCuPyTests(unittest.TestCase):
    """Optional actual-device checks; CPU doubles do not establish GPU support."""
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest('requires a working CUDA device')
        except ImportError as exc:
            raise unittest.SkipTest('requires CuPy and a working CUDA device') from exc
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy/CUDA is unavailable: {exc}') from exc
        cls.cp = cupy

    def test_native_complex_means_and_first_seen_match_original_cpu(self):
        cp = self.cp
        rng = np.random.default_rng(412)
        for dtype in ('complex64', 'complex128'):
            with self.subTest(dtype=dtype):
                fields = (_field(('corr',), (3, 2), dtype),
                          _field(('loop',), (3, 2), dtype, dt=False, role='source'))
                cpu = TimePairAccumulator({('corr',): ((3, 2), dtype)}, Lt=7)
                loops = {}
                with cp.cuda.Stream(non_blocking=True) as stream:
                    device = DeviceTimePairAccumulator(fields, Lt=7, backend=cp, one_point_slots=3)
                    for index in range(12):
                        source, sink = index % 7, (index * 2 + 3) % 7
                        host = (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))).astype(dtype).T
                        value = cp.asarray(host)
                        cpu.add(source, sink, ((('corr',), host),))
                        loops.setdefault(index % 3, host.copy())
                        device.add(source, sink, {('corr',): value, ('loop',): value},
                                   slots={'source': index % 3})
                        value[...] = -999
                    result = device.finalize((6, 2, 0, 5, 1))
                    stream.synchronize()
                expected = np.empty((3, 2, 5), dtype=dtype)
                cpu.write_means(('corr',), (6, 2, 0, 5, 1), expected)
                tolerance = 2e-6 if dtype == 'complex64' else 2e-15
                np.testing.assert_allclose(cp.asnumpy(result[('corr',)]), expected,
                                           rtol=tolerance, atol=tolerance, equal_nan=True)
                np.testing.assert_array_equal(cp.asnumpy(result[('loop',)]),
                                              np.stack([loops[index] for index in range(3)], axis=-1))
                device.close()
                self.assertIsInstance(result[('corr',)], cp.ndarray)


if __name__ == '__main__':
    unittest.main()
