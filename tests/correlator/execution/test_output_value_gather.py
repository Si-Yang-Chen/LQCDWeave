"""Fixed snapshot keys resolve current owners without per-key raw callbacks."""
import gc
import pickle
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.output import BoundOutputLayout, bind_output_layout
from .test_bound_output_views import _RawCopyBackend


class OutputValueGatherCpuTests(unittest.TestCase):
    def test_native_bulk_snapshot_rebinds_current_arrays_and_preserves_repeated_order(self):
        backend = _RawCopyBackend()
        keys = ('b', 'a', 'b')
        catalog = {key: TensorSpec((2, 3), np.complex128) for key in ('a', 'b')}
        layout = bind_output_layout(keys, catalog, backend=backend, validate=False)
        for offset in (0, 10):
            arrays = {offset + 1: np.arange(6).reshape(2, 3).astype(np.complex128) + offset,
                      offset + 2: np.full((2, 3), 4 - 2j + offset)}
            node = _NodeView({'a': offset + 1, 'b': offset + 2}, arrays)
            with execution_debug(False), patch.object(node, '_raw_output_value',
                    side_effect=AssertionError('per-key raw lookup')):
                packed = layout.copy(node, backend, owner_snapshot=True)
                np.testing.assert_array_equal(packed.owners[0], np.stack(
                    [arrays[offset + 2], arrays[offset + 1]]))
                self.assertEqual(packed.positions, ((0, 0), (0, 1), (0, 0)))
                arrays[offset + 1] = np.full((2, 3), 12 + offset + 3j)
                replacement = layout.copy(node, backend, owner_snapshot=True)
                np.testing.assert_array_equal(replacement.owners[0][1], arrays[offset + 1])
                np.testing.assert_array_equal(packed.owners[0][1],
                    np.arange(6).reshape(2, 3) + offset)
            self.assertFalse(np.shares_memory(replacement.owners[0], arrays[offset + 1]))

    def test_single_empty_and_plain_mapping_reads_keep_order_and_validation(self):
        class Traced(dict):
            def __getitem__(self, key):
                self.reads.append(key)
                return super().__getitem__(key)

        arrays = Traced(a=np.array(3.), b=np.array(5.))
        arrays.reads = []
        catalog = {key: TensorSpec((), np.float64) for key in arrays}
        for keys in (('a',), ('b', 'a', 'b')):
            arrays.reads.clear()
            layout = bind_output_layout(keys, catalog)
            restored = pickle.loads(pickle.dumps(layout))
            values = restored.copy(arrays, np)
            self.assertEqual(arrays.reads, list(dict.fromkeys(keys)))
            self.assertEqual([value.item() for value in values], [float(arrays.get(key)) for key in keys])
        empty = BoundOutputLayout((((), (), np.dtype('float64')),), (), 0)
        self.assertEqual(empty.copy({}, np), ())
        empty_copier = type('EmptyCopy', (), {
            'accepts_deferred_views': True,
            '__call__': lambda self, values, destination: True})()
        empty_raw = BoundOutputLayout(empty.groups, (), 0, False, (empty_copier,))
        self.assertEqual(empty_raw.copy(_NodeView({}, {}), np), ())
        with self.assertRaisesRegex(ValueError, 'saved output layout'):
            layout.copy({'a': np.ones(2), 'b': np.array(5.)}, np)

    def test_missing_native_keys_preserve_first_error_and_legacy_protocol_fallback(self):
        backend = _RawCopyBackend()
        layout = bind_output_layout(('a', 'b'), {
            key: TensorSpec((), np.float64) for key in ('a', 'b')}, backend=backend, validate=False)
        for binding, arrays in (({'a': 1}, {}), ({'a': 1, 'b': 2}, {1: np.array(1.)}),
                                ({'b': 2}, {2: np.array(2.)})):
            node = _NodeView(binding, arrays)
            with self.assertRaises(KeyError) as old:
                tuple(node._raw_output_value(key) for key in ('a', 'b'))
            with self.assertRaises(KeyError) as new:
                layout.copy(node, backend)
            self.assertEqual(new.exception.args, old.exception.args)

        class Legacy:
            def __init__(self): self.reads = []
            def _raw_output_value(self, key):
                self.reads.append(key)
                return np.array(2. if key == 'a' else 4.)

        legacy = Legacy()
        packed = layout.copy(legacy, backend)
        self.assertEqual(legacy.reads, ['a', 'b'])
        np.testing.assert_array_equal(packed.owners[0], [2., 4.])

    def test_debug_uses_original_views_and_layout_does_not_retain_source_owners(self):
        backend = _RawCopyBackend()
        layout = bind_output_layout(('a',), {'a': TensorSpec((2,), np.float64)},
                                    backend=backend, validate=False)
        source = np.array([1., 2.])
        reference = weakref.ref(source)
        node = _NodeView({'a': 1}, {1: source})
        with execution_debug(True), patch.object(node, '_raw_output_values',
                side_effect=AssertionError('debug bulk lookup')):
            result = layout.copy(node, backend)
        del source, node
        gc.collect()
        self.assertIsNone(reference())
        np.testing.assert_array_equal(result[0], [1., 2.])


if __name__ == '__main__':
    unittest.main()
