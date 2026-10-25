"""One owned snapshot per result key preserves every logical output position."""
import gc
import pickle
import unittest
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import OutputNode, TensorSpec
from lqcdweave.contraction.execution.device import device_execution
from lqcdweave.contraction.execution.output_program import bind_device_output_assembly
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.output import bind_output_layout
from lqcdweave.contraction.results.api import bind_output_assembly
from .test_bound_output_views import _RawCopyBackend, _bind_empty_program
from .test_device_output_program import _fixture, _term
from .test_output_owner_snapshot import _bind
from .test_output_queue import _OutputCupy


class OutputSnapshotDedupTests(unittest.TestCase):
    def test_current_keys_copy_once_without_value_or_array_identity_merging(self):
        # All three values agree. Two keys even point to the same object, but
        # only repeated *keys* can share a snapshot lane.
        first = np.arange(6, dtype=np.complex64).reshape(2, 3)[::-1, ::-1]
        arrays = {'a': first, 'b': first.copy(), 'c': first}
        order = ('c', 'a', 'b', 'a', 'c', 'b')
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        layout = bind_output_layout(order, catalog)
        self.assertEqual(layout.groups[0][0], ('c', 'a', 'b'))
        self.assertEqual(layout.positions, ((0, 0), (0, 1), (0, 2), (0, 1), (0, 0), (0, 2)))
        self.assertEqual(layout.nbytes, first.nbytes * 3)
        reference = first.copy()
        snapshot = layout.copy(arrays, np)
        self.assertEqual(snapshot.owners[0].shape, (3, 2, 3))
        for value in snapshot:
            np.testing.assert_array_equal(value, reference)
        self.assertTrue(np.shares_memory(snapshot[1], snapshot[3]))
        self.assertFalse(np.shares_memory(snapshot[0], snapshot[1]))
        self.assertFalse(np.shares_memory(snapshot[1], snapshot[2]))
        first[...] = -7
        rebound = layout.copy(arrays, np)
        np.testing.assert_array_equal(rebound[1], first)
        np.testing.assert_array_equal(rebound[2], reference)
        for value in snapshot:
            np.testing.assert_array_equal(value, reference)

    def test_mixed_dtype_shape_scalar_empty_and_borrowed_input_lifetimes(self):
        arrays = {'v': np.arange(6, dtype=np.float32)[::-2],
                  's': np.array(2 + 3j, dtype=np.complex128),
                  'e': np.empty((0, 2), dtype=np.complex64)}
        order = ('s', 'e', 'v', 's', 'v', 'e')
        expected = tuple(arrays[key].copy() for key in order)
        refs = tuple(weakref.ref(value) for value in arrays.values())
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        program = _bind_empty_program(order, catalog, np)
        snapshot = bind_output_layout(order, catalog).copy(arrays, np)
        ordinary = program.run(None, arrays)
        self.assertEqual(tuple(len(owner) for owner in snapshot.owners), (1, 1, 1))
        self.assertEqual(sum(owner.nbytes for owner in snapshot.owners),
                         sum(value.nbytes for value in arrays.values()))
        del arrays
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        for restored in (ordinary, snapshot, pickle.loads(pickle.dumps(snapshot))):
            for value, reference in zip(restored, expected, strict=True):
                self.assertEqual(value.shape, reference.shape)
                self.assertEqual(value.dtype, reference.dtype)
                self.assertEqual(value.tobytes(), reference.tobytes())
        empty = bind_output_layout((), {})
        self.assertEqual(empty.nbytes, 0)
        self.assertEqual(empty.copy({}, np), ())

    def test_native_copy_launches_scale_with_distinct_keys_not_logical_terms(self):
        backend = _RawCopyBackend()
        arrays = {str(i): np.arange(6, dtype=np.float64).reshape(2, 3)[:, ::-1] + i
                  for i in range(129)}
        distinct = tuple(arrays)
        logical = distinct + distinct[::-1] + distinct
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        layout = bind_output_layout(logical, catalog, backend=backend, validate=False)
        with execution_debug(False):
            snapshot = layout.copy(arrays, backend, owner_snapshot=True)
        self.assertEqual([count for count, _, _ in backend.raw_launches], [128, 1])
        self.assertEqual(snapshot.owners[0].shape, (129, 2, 3))
        for value, key in zip(snapshot, logical, strict=True):
            np.testing.assert_array_equal(value, arrays[key])
        with execution_debug(True):
            checked = layout.copy(arrays, backend)
        for value, key in zip(checked, logical, strict=True):
            np.testing.assert_array_equal(value, arrays[key])

    def test_host_download_preserves_repeated_positions_and_independent_runs(self):
        backend = _OutputCupy()
        original = np.arange(6, dtype=np.complex64).reshape(2, 3)
        arrays = {'a': backend.asarray(original), 'b': backend.asarray(original + 7j)}
        catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
        order = ('b', 'a', 'b', 'a', 'b')
        layout = bind_output_layout(order, catalog)
        with device_execution(backend, device_memory_limit=4096) as device:
            snapshot = layout.copy(arrays, backend)
            first = device.finish_download(device.start_download(snapshot))
            self.assertEqual(sum(owner.nbytes for owner in snapshot.owners), original.nbytes * 2)
            self.assertEqual(device.statistics['output_download_calls'], 1)
            next_snapshot = layout.copy({key: value * 3 for key, value in arrays.items()}, backend)
            second = device.finish_download(device.start_download(next_snapshot))
            self.assertEqual(device.statistics['output_download_calls'], 2)
        for i, key in enumerate(order):
            np.testing.assert_array_equal(first[i], arrays[key])
            np.testing.assert_array_equal(second[i], arrays[key] * 3)
        self.assertTrue(np.shares_memory(first[0], first[2]))
        self.assertFalse(np.shares_memory(first[0], second[0]))

    def test_device_assembly_direct_and_indirect_keep_all_term_positions(self):
        # OutputRecipe term IDs remain unique, while X may compute identical
        # physical result keys for multiple logical terms.
        for count in (3, 100):
            with self.subTest(count=count):
                values = tuple(np.arange(6.).reshape(2, 3) + (i % 2) for i in range(count))
                root = OutputNode('tuple', children=(
                    OutputNode('add', children=(_term(0), _term(2))),
                    OutputNode('diagram_sum', tuple((f't{i}', 1)
                                                   for i in range(count)))))
                recipe, backend, bound, _, _ = _bind(root, values)
                arrays = {'even': values[0], 'odd': values[1]}
                keys = tuple('odd' if i % 2 else 'even' for i in range(count))
                layout = bind_output_layout(keys, {
                    key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()
                }, backend=backend, validate=False)
                reference = bind_output_assembly(recipe)(values)
                for shift in (0, 5):
                    shifted = {key: value + shift for key, value in arrays.items()}
                    snapshot = layout.copy(shifted, backend, owner_snapshot=True)
                    self.assertEqual(snapshot.owners[0].shape, (2, 2, 3))
                    actual = bound(snapshot)
                    expected = bind_output_assembly(recipe)(tuple(value + shift for value in values))
                    for left, right in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-13)
                    if not shift:
                        for left, right in zip(actual, reference, strict=True):
                            np.testing.assert_array_equal(left, right)
                self.assertEqual(len(bound._owner_layout_schemas), 1)

    def test_host_assembly_keeps_complex_coefficients_after_snapshot_dedup(self):
        values = tuple(np.array(i % 2 + 2j, dtype='complex64') for i in range(100))
        root = OutputNode('diagram_sum', tuple((f't{i}', 1 + (i % 3) * 1j)
                                               for i in range(100)))
        recipe, _, _, _, _ = _bind(root, values)
        arrays = {'even': values[0], 'odd': values[1]}
        keys = tuple('odd' if i % 2 else 'even' for i in range(100))
        layout = bind_output_layout(keys, {
            key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()})
        packed = layout.copy(arrays, np)
        actual = bind_output_assembly(recipe)(packed)
        expected = sum((1 + (i % 3) * 1j) * values[i] for i in range(100))
        np.testing.assert_allclose(actual, expected, rtol=2e-6)


class RealCuPyOutputSnapshotDedupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            cupy.cuda.runtime.getDeviceProperties(0)
        except Exception as error:
            raise unittest.SkipTest(f'actual CUDA snapshot dedup test requires a device: {error}')
        cls.cp = cupy

    def test_actual_native_snapshot_and_assembly_keep_repeated_term_coefficients(self):
        cp = self.cp
        for count, shape in ((3, (2, 3)), (100, (2, 3)), (200, ())):
            with self.subTest(count=count, shape=shape):
                base = np.arange(6).reshape(shape) if shape else np.array(2.25)
                host = {key: (base + shift + 2j).astype('complex128')
                        for key, shift in (('even', 0), ('odd', 1))}
                keys = tuple('odd' if i % 2 else 'even' for i in range(count))
                coefficients = tuple(1 + (i % 3) * 1j for i in range(count))
                root = OutputNode('diagram_sum', tuple((f't{i}', coefficient)
                                   for i, coefficient in enumerate(coefficients)))
                values = tuple(host[key] for key in keys)
                recipe, problem = _fixture(root, *values)
                bound = bind_device_output_assembly(recipe, problem, cp)
                layout = bind_output_layout(keys, {
                    key: TensorSpec(value.shape, value.dtype) for key, value in host.items()
                }, backend=cp, validate=False)
                arrays = {key: cp.asarray(value) for key, value in host.items()}
                first = layout.copy(arrays, cp, owner_snapshot=True)
                self.assertEqual(first.owners[0].shape, (2, *shape))
                expected = sum(c * value for c, value in zip(coefficients, values, strict=True))
                np.testing.assert_allclose(cp.asnumpy(bound(first)), expected, rtol=3e-12, atol=3e-12)
                second = layout.copy({key: value + 5 for key, value in arrays.items()},
                                     cp, owner_snapshot=True)
                np.testing.assert_allclose(cp.asnumpy(bound(second)), expected + 5 * sum(coefficients),
                                           rtol=3e-12, atol=3e-12)
                np.testing.assert_allclose(cp.asnumpy(bound(first)), expected, rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
