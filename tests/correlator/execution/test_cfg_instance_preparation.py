"""High-reuse input identity binding preserves current values and boundaries."""
import gc
import importlib
from types import MappingProxyType
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import SourcePreparation, TensorPreparation, TensorSpec
from lqcdweave.contraction.execution.cfg_runtime import _LocalLoader
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.preparation import bind_prepared_loader
from lqcdweave.contraction.execution.preparation_kernel import FusedSpinPreparation
from correlator.execution.test_cfg_template_preparation import _backend, _global_tables, _spin, _template


class _CountingBinding(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


def _fixture(*, backend=None, frozen=True, leaves=('a', 'b') * 8, role_count=9):
    names = tuple(dict.fromkeys(leaves))
    closure = {name: _spin(np.eye(4), raw='raw-' + name) for name in names}
    specs = {name: TensorSpec((4, 4, 2, 3), np.float64) for name in closure}
    roles = tuple('role-' + str(index) for index in range(role_count))
    for key in roles:
        closure[key] = SourcePreparation(tuple('raw-' + name for name in names),
            recipe=TensorPreparation('stack', leaves))
        specs[key] = TensorSpec((len(leaves), 4, 4, 2, 3), np.float64)
    tables, item_specs, bindings, blocks = {}, {}, [], []
    for point in range(2):
        raw_ids = {name: f'raw-{name}-{point}' for name in names}
        role_ids, preps, global_specs = _global_tables(closure, specs, point, raw_ids)
        tables.update(preps)
        item_specs.update(global_specs)
        bindings.append(_CountingBinding(role_ids))
        blocks.append({raw_ids[name]: np.arange(6).reshape(2, 3) + point * 10 + index
                       for index, name in enumerate(names)})
    loader = bind_prepared_loader(None, MappingProxyType(tables) if frozen else tables,
        MappingProxyType(item_specs) if frozen else item_specs, backend=_backend() if backend is None else backend)
    template = _template(closure, specs)
    return loader, loader.bind_template_preparations(template), roles, bindings, blocks


def _inputs_only(program, blocks, destination=None, *, inputs=None):
    return np.stack(inputs)


class CfgInstancePreparationTests(unittest.TestCase):
    def test_instance_getters_keep_singleton_tuple_and_wide_or_repeated_input_order(self):
        for leaves, read_order in (
                (('a',), ('a',)),
                (('a', 'b'), ('a', 'b')),
                (('c', 'a', 'b', 'c', 'b', 'a'), ('c', 'a', 'b', 'c', 'b', 'a')),
                (tuple('abcdefghi'), tuple('abcdefghi')),
                (('a',) * 16, ('a',)),
                (('a', 'b') * 8, ('a', 'b'))):
            with self.subTest(leaves=leaves):
                loader, bound, keys, nodes, raw_sets = _fixture(leaves=leaves, role_count=64)
                local = _LocalLoader(loader, nodes[0], bound, preparation_keys=keys)
                self.assertIsNotNone(local.instance_preparations)
                blocks = _CountingBinding(raw_sets[0])
                with patch.object(FusedSpinPreparation, 'materialize', _inputs_only):
                    actual = local.materialize(keys[0], None, blocks)
                    expected_keys = [f'raw-{name}-0' for name in read_order]
                    self.assertEqual(blocks.reads, expected_keys)
                    np.testing.assert_array_equal(actual,
                        np.stack([raw_sets[0][f'raw-{name}-0'] for name in leaves]))
                    del blocks[expected_keys[0]]
                    blocks.reads.clear()
                    with self.assertRaises(KeyError) as caught:
                        local.materialize(keys[0], None, blocks)
                    self.assertEqual(caught.exception.args, (expected_keys[0],))
                    self.assertEqual(blocks.reads, expected_keys[:1])

    def test_repeated_roles_read_current_raw_once_and_preserve_distinct_pooled_inputs(self):
        loader, bound, keys, bindings, raw_sets = _fixture()
        self.assertIs(bound.roles['a'][0], bound.roles['b'][0])
        with patch.object(FusedSpinPreparation, 'materialize', _inputs_only):
            for used_keys in (keys[:1], keys):
                for point, (nodes, raw) in enumerate(zip(bindings, raw_sets, strict=True)):
                    with self.subTest(instance_binding=len(used_keys) > 1, point=point):
                        nodes.reads.clear()
                        local = _LocalLoader(loader, nodes, bound, preparation_keys=used_keys)
                        self.assertEqual(local.instance_preparations is not None, len(used_keys) > 1)
                        blocks = _CountingBinding(raw)
                        actual = local.materialize(keys[0], None, blocks)
                        self.assertEqual(blocks.reads, [f'raw-a-{point}', f'raw-b-{point}'])
                        self.assertEqual(nodes.reads, ['a', 'b'])
                        np.testing.assert_array_equal(actual,
                            np.stack([raw[f'raw-a-{point}'], raw[f'raw-b-{point}']] * 8))
                        replacement = np.full((2, 3), 100 + point)
                        blocks[f'raw-a-{point}'] = replacement
                        blocks.reads.clear()
                        current = local.materialize(keys[0], None, blocks)
                        self.assertEqual(blocks.reads, [f'raw-a-{point}', f'raw-b-{point}'])
                        np.testing.assert_array_equal(current,
                            np.stack([replacement, raw[f'raw-b-{point}']] * 8))
                        self.assertFalse(np.shares_memory(current, replacement))

    def test_current_ids_bind_once_order_duplicates_and_replaced_blocks_remain_current(self):
        loader, bound, keys, bindings, raw_sets = _fixture()
        with (patch.object(FusedSpinPreparation, 'materialize', _inputs_only),
              patch.object(loader, 'materialize_template_preparation', side_effect=AssertionError('rebound IDs'))):
            for point, (nodes, raw) in enumerate(zip(bindings, raw_sets, strict=True)):
                local = _LocalLoader(loader, nodes, bound, preparation_keys=keys)
                self.assertIsNotNone(local.instance_preparations)
                self.assertEqual(nodes.reads, ['a', 'b'])
                for key in keys:
                    np.testing.assert_array_equal(local.materialize(key, None, raw),
                        np.stack([raw[f'raw-a-{point}'], raw[f'raw-b-{point}']] * 8))
                raw[f'raw-a-{point}'] = np.full((2, 3), 100 + point)
                np.testing.assert_array_equal(local.materialize(keys[0], None, raw)[0], raw[f'raw-a-{point}'])
                self.assertEqual(nodes.reads, ['a', 'b'])
        self.assertEqual(len(bound.input_layouts), 1)

    def test_instance_and_template_bindings_do_not_retain_raw_owners(self):
        loader, bound, keys, bindings, blocks = _fixture()
        refs = [weakref.ref(value) for raw in blocks for value in raw.values()]
        instances = [loader.bind_instance_preparations(bound, nodes, keys) for nodes in bindings]
        with patch.object(FusedSpinPreparation, 'materialize', _inputs_only):
            for instance, raw in zip(instances, blocks, strict=True):
                value = instance.materialize(keys[0], raw)
                self.assertFalse(any(np.shares_memory(value, item) for item in raw.values()))
        del raw, blocks
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertEqual(len(bound.input_layouts), 1)

    def test_low_reference_count_and_low_reuse_keep_lazy_lookups(self):
        loader, bound, keys, bindings, _ = _fixture()
        self.assertIsNone(loader.bind_instance_preparations(bound, bindings[0], keys[:1]))
        self.assertEqual(bindings[0].reads, [])
        # A large variant with no repeated leaves must not pay eager binding.
        roles = {f'role-{i}': (object(), tuple(f'leaf-{i}-{j}' for j in range(16))) for i in range(8)}
        from dataclasses import replace
        low_reuse = replace(bound, roles=MappingProxyType(roles), input_layouts=type(bound.input_layouts)())
        self.assertIsNone(loader.bind_instance_preparations(low_reuse, {}, tuple(roles)))

    def test_only_actual_load_keys_participate_in_binding(self):
        loader, bound, keys, bindings, _ = _fixture()
        # Unused roles with absent instance identities must not affect binding.
        instance = loader.bind_instance_preparations(bound, bindings[0], keys[:4])
        self.assertIsNotNone(instance)
        self.assertEqual(set(instance.roles), set(keys[:4]))
        self.assertEqual(bindings[0].reads, ['a', 'b'])

    def test_missing_id_falls_back_and_fails_only_when_its_role_is_consumed(self):
        loader, bound, keys, bindings, blocks = _fixture()
        nodes = dict(bindings[0])
        del nodes['b']
        local = _LocalLoader(loader, nodes, bound, preparation_keys=keys)
        self.assertIsNone(local.instance_preparations)
        with patch.object(FusedSpinPreparation, 'materialize', _inputs_only):
            np.testing.assert_array_equal(local.materialize('a', None, blocks[0]),
                                          np.stack([blocks[0]['raw-a-0']]))
            with self.assertRaises(KeyError) as caught:
                local.materialize(keys[0], None, blocks[0])
        self.assertEqual(caught.exception.args, ('b',))

    def test_none_raw_id_and_unknown_global_id_fall_back(self):
        loader, bound, keys, bindings, _ = _fixture()
        for global_id in ('not-present', bindings[0][keys[0]]):
            with self.subTest(global_id=global_id):
                nodes = dict(bindings[0], b=global_id)
                self.assertIsNone(loader.bind_instance_preparations(bound, nodes, keys))

    def test_scope_debug_cpu_and_mutable_loader_boundaries(self):
        loader, bound, keys, bindings, _ = _fixture()
        other, _, _, _, _ = _fixture()
        with self.assertRaisesRegex(ValueError, 'another loader scope'):
            other.bind_instance_preparations(bound, bindings[0], keys)
        with execution_debug(True):
            self.assertIsNone(_LocalLoader(loader, bindings[0], bound, preparation_keys=keys).instance_preparations)
        for options in ({'backend': np}, {'frozen': False}):
            with self.subTest(options=options):
                current, current_bound, _, nodes, _ = _fixture(**options)
                self.assertIsNone(current_bound)
                self.assertIsNone(_LocalLoader(current, nodes[0], current_bound,
                                               preparation_keys=keys).instance_preparations)

    def test_variant_cache_is_bounded_and_preserves_distinct_actual_load_orders(self):
        loader, bound, keys, bindings, _ = _fixture()
        orders = [keys[index:] + keys[:index] for index in range(9)]
        for order in orders:
            self.assertIsNotNone(loader.bind_instance_preparations(bound, bindings[0], order))
        self.assertEqual(len(bound.input_layouts), 8)
        self.assertNotIn(orders[0], bound.input_layouts)
        self.assertIn(orders[-1], bound.input_layouts)


class CuPyCfgInstancePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cp = importlib.import_module('cupy')
            if cls.cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')

    def test_current_instance_binding_matches_numpy_and_debug_on_changed_strides(self):
        cp = self.cp
        loader, bound, keys, bindings, host_sets = _fixture(backend=cp)
        outputs = []
        refs = []
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            for point, (nodes, host) in enumerate(zip(bindings, host_sets, strict=True)):
                blocks = {key: cp.asarray(value, dtype=cp.float64) for key, value in host.items()}
                if point:
                    blocks = {key: value[:, ::-1] for key, value in blocks.items()}
                refs.extend(weakref.ref(value) for value in blocks.values())
                raw_a = host[f'raw-a-{point}']
                raw_b = host[f'raw-b-{point}']
                if point:
                    raw_a, raw_b = raw_a[:, ::-1], raw_b[:, ::-1]
                expected = np.stack([np.einsum('ab,ij->abij', np.eye(4), value)
                                     for value in (raw_a, raw_b)] * 8)
                for used_keys in (keys, keys[:1]):
                    with self.subTest(instance_binding=len(used_keys) > 1, point=point):
                        local = _LocalLoader(loader, nodes, bound, preparation_keys=used_keys)
                        self.assertEqual(local.instance_preparations is not None, len(used_keys) > 1)
                        current = _CountingBinding(blocks)
                        actual = local.materialize(keys[0], None, current)
                        self.assertEqual(current.reads, [f'raw-a-{point}', f'raw-b-{point}'])
                        with execution_debug(True):
                            audited = local.materialize(keys[0], None, blocks)
                        stream.synchronize()
                        np.testing.assert_array_equal(cp.asnumpy(actual), expected)
                        np.testing.assert_array_equal(cp.asnumpy(audited), expected)
                        self.assertFalse(any(cp.shares_memory(actual, value) for value in blocks.values()))
                        outputs.append(actual)
            self.assertNotEqual(outputs[0].data.ptr, outputs[1].data.ptr)
        del blocks, current, audited, actual
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))


if __name__ == '__main__':
    unittest.main()
