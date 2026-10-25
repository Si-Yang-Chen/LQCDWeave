"""Bounded key-table reuse never caches current IDs or numerical owners."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import ArrayConstant, describe_coordinates
from lqcdweave.contraction.execution import cfg_runtime
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.preparation import _NumericNodeBinding
from lqcdweave.contraction.planning.api import build_cfg_plan
from correlator.planning import test_cfg_plan as fixtures


class NodeKeyTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected, preparations, specs = fixtures.CfgPlanTests.dense_fixture()
        points, _ = describe_coordinates(time_pairs=[{0: 0, 1: 0}, {0: 1, 1: 0}])
        cls.plan = build_cfg_plan(selected, None, points.points, {}, preparations, specs)

    def test_every_binding_has_fresh_exact_integer_ids_and_independent_mutation(self):
        keys = ('minimum', 'zero', 'large', 'maximum')
        skeleton = keys, dict.fromkeys(keys)
        first = ArrayConstant.from_array(np.array([-2**63, 0, 2**53+1, 2**63-1], dtype='<i8'))
        second = ArrayConstant.from_array(np.array([4, 3, 2, 1], dtype='<i8'))
        a = cfg_runtime._bind_node_ids(keys, first, skeleton)
        b = cfg_runtime._bind_node_ids(keys, second, skeleton)
        self.assertIs(type(a), _NumericNodeBinding)
        self.assertEqual(list(a.items()), list(zip(keys, map(int, first.array()))))
        self.assertTrue(all(type(value) is int for value in a.values()))
        self.assertEqual(b, dict(zip(keys, (4, 3, 2, 1))))
        a['minimum'] = 99
        del b['zero']
        self.assertEqual(skeleton[1], dict.fromkeys(keys))
        self.assertEqual(cfg_runtime._bind_node_ids(keys, first, skeleton)['minimum'], -2**63)
        short = ArrayConstant.from_array(np.array([1], dtype='<i8'))
        with self.assertRaises(ValueError):
            cfg_runtime._bind_node_ids(keys, short, skeleton)

    def test_debug_and_mutable_external_ids_ignore_key_template(self):
        class Unusable:
            def keys(self): raise AssertionError('used cached key table')
        class ExplicitOnly(np.ndarray):
            def tolist(self): raise AssertionError('debug skipped explicit int conversion')
        keys = ('a', 'b')
        record = ArrayConstant.from_array(np.array([3, 7], dtype='<i8'))
        with patch.object(ArrayConstant, 'array', return_value=record.array().view(ExplicitOnly)), execution_debug(True):
            self.assertEqual(cfg_runtime._bind_node_ids(keys, record, (keys, Unusable())), {'a': 3, 'b': 7})
        values = np.array(['3', '7'], dtype=object)
        external = SimpleNamespace(array=lambda: values)
        self.assertEqual(cfg_runtime._bind_node_ids(keys, external, (keys, Unusable())), {'a': 3, 'b': 7})
        values[0] = '9'
        self.assertEqual(cfg_runtime._bind_node_ids(keys, external, (keys, Unusable()))['a'], 9)
        values[0] = 'invalid'
        with self.assertRaises(ValueError):
            cfg_runtime._bind_node_ids(keys, external, (keys, Unusable()))

    def test_one_bounded_key_table_handles_equal_and_changed_windows(self):
        current = cfg_runtime._node_key_template(self.plan, None)
        self.assertTrue(all(value is None for value in current[1].values()))
        self.assertLessEqual(sys.getsizeof(current[1]), cfg_runtime._NODE_KEY_TEMPLATE_MAX_BYTES)
        template = self.plan.templates[0]
        decoded_keys = tuple(key.encode().decode() for key in template.keys)
        equal = replace(self.plan, templates=(replace(template, keys=decoded_keys),))
        reused = cfg_runtime._node_key_template(equal, current)
        self.assertIs(reused[0], equal.templates[0].keys)
        self.assertIs(reused[1], current[1])
        reordered = replace(equal, templates=(replace(template, keys=tuple(reversed(template.keys))),))
        changed = cfg_runtime._node_key_template(reordered, reused)
        self.assertIsNot(changed[1], reused[1])
        self.assertEqual(tuple(changed[1]), reordered.templates[0].keys)
        # A single retained table is chosen by coordinate frequency, not by
        # accumulating a dictionary per template or per instance.
        mixed = replace(self.plan, templates=(template, reordered.templates[0]),
            instances=(self.plan.instances[0], *[replace(self.plan.instances[1], template=1)]*3))
        selected = cfg_runtime._node_key_template(mixed, current)
        self.assertIs(selected[0], mixed.templates[1].keys)
        for cap in (0, sys.getsizeof(current[1])-1):
            with patch.object(cfg_runtime, '_NODE_KEY_TEMPLATE_MAX_BYTES', cap):
                self.assertIsNone(cfg_runtime._node_key_template(self.plan, None))

    def test_extensions_and_debug_do_not_admit_a_shared_key_table(self):
        with execution_debug(True):
            self.assertIsNone(cfg_runtime._node_key_template(self.plan, None))
        self.assertIsNone(cfg_runtime._node_key_template(SimpleNamespace(), None))
        instance = self.plan.instances[0]
        ids = instance.node_ids.array().copy()
        external = SimpleNamespace(shape=ids.shape, dtype='<i8', array=lambda: ids)
        mutable = replace(self.plan, instances=(replace(instance, node_ids=external), *self.plan.instances[1:]))
        self.assertIsNone(mutable._binding_validation)
        self.assertIsNone(cfg_runtime._node_key_template(mutable, None))
        class ExtensionKey(str): pass
        template = self.plan.templates[0]
        extended = replace(self.plan, templates=(replace(template, keys=tuple(map(ExtensionKey, template.keys))),))
        self.assertIsNone(cfg_runtime._node_key_template(extended, None))

    def test_concurrent_bindings_do_not_share_mutable_ids(self):
        skeleton = cfg_runtime._node_key_template(self.plan, None)
        keys = skeleton[0]
        def bind(offset):
            record = ArrayConstant.from_array(np.arange(len(keys), dtype='<i8') + offset)
            binding = cfg_runtime._bind_node_ids(keys, record, skeleton)
            binding[keys[0]] = -offset
            return binding
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(bind, range(1, 33)))
        self.assertEqual([value[keys[0]] for value in values], list(range(-1, -33, -1)))
        self.assertEqual(len({id(value) for value in values}), 32)
        self.assertTrue(all(value is None for value in skeleton[1].values()))

    def test_public_cfg_output_matches_original_binding_and_warm_cache_reuses_keys(self):
        from test_cfg_execution_binding_cache import ExecutionBindingCacheTests
        calculation, _, _ = ExecutionBindingCacheTests().fixture()
        original = cfg_runtime._bind_node_ids
        with patch.object(cfg_runtime, '_bind_node_ids', wraps=original) as bound:
            first = calculation.run()
        self.assertTrue(all(call.args[2] is not None for call in bound.call_args_list))
        owner = calculation._execution_bindings
        node_keys = owner._binding[-1]
        with patch.object(cfg_runtime, '_node_key_template', side_effect=AssertionError('rebuilt warm keys')):
            second = calculation.run()
        self.assertIs(owner._binding[-1], node_keys)
        with patch.object(cfg_runtime, '_bind_node_ids', side_effect=lambda keys, ids, key_template=None: original(keys, ids)):
            reference = calculation.run()
        np.testing.assert_array_equal(first.value['two_point'], reference.value['two_point'])
        np.testing.assert_array_equal(second.value['two_point'], reference.value['two_point'])
        self.assertTrue(all(value is None for value in node_keys[1].values()))


if __name__ == '__main__': unittest.main()
