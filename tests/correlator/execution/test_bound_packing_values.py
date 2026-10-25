"""Fixed input lookup preserves current owners, ordering and debug behavior."""
import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.cfg_runtime import _NodeView
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.dense_batch import _BatchSide
from lqcdweave.contraction.execution.operations.packing_kernel import (
    _bind_packing_values, _packing_values,
)
from lqcdweave.contraction.execution.operations.result_views import bind_result_views


class BoundPackingValuesTests(unittest.TestCase):
    def test_current_mapping_order_duplicates_and_single_inputs(self):
        getter = _bind_packing_values((('b', None), ('a', None), ('b', (0,))))
        single = _bind_packing_values((('a', None),))
        for offset in (0, 10):
            binding = {'a': offset + 1, 'b': offset + 2}
            arrays = {offset + 1: object(), offset + 2: object()}
            node = _NodeView(binding, arrays)
            with execution_debug(False), patch.object(node, '_raw_output_value',
                    side_effect=AssertionError('per-input callback')):
                values = getter(node)
                self.assertEqual(values, (arrays[offset + 2], arrays[offset + 1], arrays[offset + 2]))
                self.assertEqual(single(node), (arrays[offset + 1],))
                replacement = object()
                arrays[offset + 1] = replacement
                self.assertIs(single(node)[0], replacement)

    def test_deferred_owner_stays_lazy_and_debug_materializes_original_views(self):
        owner = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        layouts = bind_result_views(owner.shape, owner.dtype,
            (('a', (0, slice(None), slice(None)), ()),
             ('b', (1, slice(None), slice(None)), ())))
        node = _NodeView({'a': 1, 'b': 2}, {})
        node._publish_result_views(owner, layouts)
        operands = (('b', None), ('a', None))
        getter = _bind_packing_values(operands)
        raw = (node._raw_output_value('b'), node._raw_output_value('a'))
        with execution_debug(False), patch.object(type(raw[0]), 'array',
                side_effect=AssertionError('unnecessary view')):
            self.assertEqual(getter(node), raw)
        with execution_debug(True), patch.object(node, '_raw_output_values',
                side_effect=AssertionError('debug bypassed original lookup')):
            actual = getter(node)
        for value, expected in zip(actual, (owner[1], owner[0]), strict=True):
            np.testing.assert_array_equal(value, expected)
            self.assertTrue(np.shares_memory(value, owner))

    def test_empty_plain_and_legacy_protocol_inputs_keep_their_semantics(self):
        self.assertEqual(_bind_packing_values(())(object()), ())
        getter = _bind_packing_values((('b', None), ('a', None), ('b', None)))
        values = {'a': object(), 'b': object()}
        self.assertEqual(getter(values), (values['b'], values['a'], values['b']))

        class Legacy:
            def __init__(self):
                self.raw, self.normal = [], []

            def _raw_output_value(self, key):
                self.raw.append(key)
                return values[key]

            def __getitem__(self, key):
                self.normal.append(key)
                return values[key]

        legacy = Legacy()
        with execution_debug(False):
            self.assertEqual(getter(legacy), (values['b'], values['a'], values['b']))
        self.assertEqual(legacy.raw, ['b', 'a', 'b'])
        with execution_debug(True):
            getter(legacy)
        self.assertEqual(legacy.normal, ['b', 'a', 'b'])

    def test_missing_local_and_global_keys_keep_original_first_error(self):
        operands = (('a', None), ('b', None))
        getter = _bind_packing_values(operands)
        for binding, arrays in (({'a': 1}, {}), ({'a': 1, 'b': 2}, {1: object()}),
                                ({'b': 2}, {2: object()})):
            node = _NodeView(binding, arrays)
            with execution_debug(False):
                with self.assertRaises(KeyError) as old:
                    _packing_values(node, operands)
                with self.assertRaises(KeyError) as new:
                    getter(node)
            self.assertEqual(new.exception.args, old.exception.args)

    def test_batch_side_uses_bulk_current_owners_and_preserves_numeric_packing(self):
        operands = (('a', None), ('b', None))

        def pack(values, destination):
            np.concatenate(values, axis=0, out=destination)
            return destination

        side = _BatchSide(operands, (4, 3), 0, False, original_packer=pack)
        for offset in (0, 5):
            first = np.arange(6, dtype=np.float64).reshape(2, 3) + offset
            second = np.arange(6, dtype=np.float64).reshape(2, 3) + offset + 10
            node = _NodeView({'a': 1, 'b': 2}, {1: first, 2: second})
            with execution_debug(False), patch.object(node, '_raw_output_value',
                    side_effect=AssertionError('per-input callback')):
                result = side.pack(node, np, np.dtype('float64'))
            np.testing.assert_array_equal(result, np.concatenate((first, second), axis=0))
            self.assertFalse(np.shares_memory(result, first))
            self.assertFalse(np.shares_memory(result, second))

    def test_binding_does_not_retain_previous_instances_or_numeric_owners(self):
        getter = _bind_packing_values((('a', None), ('b', None)))
        refs = []
        for point in (0, 1):
            first, second = np.full((2, 3), point), np.full((2, 3), point + 1)
            node = _NodeView({'a': 1, 'b': 2}, {1: first, 2: second})
            refs.extend((weakref.ref(first), weakref.ref(second), weakref.ref(node)))
            self.assertIs(getter(node)[0], first)
            del first, second, node
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))


if __name__ == '__main__':
    unittest.main()
