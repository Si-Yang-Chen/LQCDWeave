"""Numerical packing ownership, admission and failure isolation."""
from dataclasses import replace
from contextlib import nullcontext
import gc
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

from lqcdweave.contraction.execution.packed_values import (
    PackedValueCache, pack_operand, packed_values_scope,
)
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.planning.packed_values import admit_packed_values
from lqcdweave.contracts.correlator.api import MemoryPoint, SceneStoragePlan, SourcePreparation


def backend(stream=7):
    return SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(ptr=stream)))


class PackedValueTests(unittest.TestCase):
    def test_disabled_nested_execution_scopes_mask_outer_cache_and_restore_it(self):
        cases=(('no execution',None,False),
               ('no cache',SimpleNamespace(packed_value_cache=None),False),
               ('zero capacity',SimpleNamespace(packed_value_cache=PackedValueCache(
                   0,array_type=np.ndarray)),False),
               ('debug',SimpleNamespace(packed_value_cache=PackedValueCache(
                   64,array_type=np.ndarray)),True))
        for label,execution,debug in cases:
            for fail in (False,True):
                with self.subTest(scope=label,exception=fail):
                    self._assert_disabled_nested_scope(
                        lambda:packed_values_scope(execution,backend()),debug,fail)

    def test_disabled_activation_masks_outer_cache_and_restores_it(self):
        for capacity,stream in ((0,7),(64,None)):
            for fail in (False,True):
                with self.subTest(capacity=capacity,stream=stream,exception=fail):
                    inner=PackedValueCache(capacity,array_type=np.ndarray)
                    self._assert_disabled_nested_scope(
                        lambda:inner.activate(backend(stream)),False,fail)
                    self.assertEqual(inner.hits,0)
                    self.assertEqual(inner.bytes,0)

    def _assert_disabled_nested_scope(self,scope,debug,fail):
        outer=PackedValueCache(64,array_type=np.ndarray)
        a=np.arange(4.)
        calls=[]
        def copy():
            calls.append(1)
            return a.copy()
        with outer.activate(backend()):
            expected=pack_operand(a,(),32,copy)
            outcome=self.assertRaisesRegex(ValueError,'abort child') if fail else nullcontext()
            with outcome:
                with execution_debug(debug),scope():
                    first=pack_operand(a,(),32,copy)
                    second=pack_operand(a,(),32,copy)
                    self.assertIsNot(first,expected)
                    self.assertIsNot(second,first)
                    np.testing.assert_array_equal(first,a)
                    np.testing.assert_array_equal(second,a)
                    self.assertEqual(len(calls),3)
                    self.assertEqual(outer.hits,0)
                    if fail:
                        raise ValueError('abort child')
            self.assertIs(pack_operand(a,(),32,copy),expected)
            self.assertEqual(outer.hits,1)
            self.assertEqual(outer.bytes,32)
        self.assertIsNot(pack_operand(a,(),32,copy),expected)

    def test_debug_execution_scope_preserves_original_copies(self):
        cache=PackedValueCache(64,array_type=np.ndarray)
        a=np.arange(4.)
        execution=SimpleNamespace(packed_value_cache=cache)
        with execution_debug(True),packed_values_scope(execution,backend()):
            first=pack_operand(a,(),32,lambda:a.copy())
            second=pack_operand(a,(),32,lambda:a.copy())
        self.assertIsNot(first,second)
        self.assertEqual(cache.hits,0)

    def test_extension_stream_without_stable_identity_keeps_original_path(self):
        cache=PackedValueCache(64,array_type=np.ndarray)
        a=np.arange(4.)
        extension=SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda:object()))
        with cache.activate(extension):
            first=pack_operand(a,(),32,lambda:a.copy())
            second=pack_operand(a,(),32,lambda:a.copy())
        self.assertIsNot(first,second)
        self.assertEqual(cache.hits,0)

    def test_reuse_is_owner_and_recipe_exact_and_matches_independent_product(self):
        cache=PackedValueCache(512,array_type=np.ndarray)
        rng=np.random.default_rng(22)
        left=rng.normal(size=(3,4,2))+1j*rng.normal(size=(3,4,2))
        right=rng.normal(size=(4,2,5))+1j*rng.normal(size=(4,2,5))
        calls=[]
        def packed(value,axes):
            shape=(3,8) if value.shape==(3,4,2) else (8,5)
            def copy():
                calls.append(1)
                return value.transpose(axes).copy().reshape(shape)
            return pack_operand(value,(axes,shape),np.prod(shape).item()*16,copy)
        # The right copy is larger than 512 B and stays on the original path.
        with cache.activate(backend()):
            a=packed(left,(0,1,2)); b=packed(right,(0,1,2))
            np.testing.assert_allclose(a@b,np.einsum('akb,kbc->ac',left,right))
            self.assertIs(packed(left,(0,1,2)),a)
            self.assertIsNot(packed(left.copy(),(0,1,2)),a)
        self.assertEqual(cache.hits,1)
        self.assertLessEqual(cache.peak_bytes,cache.capacity)
        # An execution binding alone cannot make the next run reuse values.
        outside=packed(left,(0,1,2))
        self.assertIsNot(outside,a)

    def test_source_release_invalidates_all_recipes_without_retaining_source_or_base(self):
        cache=PackedValueCache(4096,array_type=np.ndarray)
        base=np.arange(12.).reshape(3,4).copy(); source=base.T
        source_ref,base_ref=weakref.ref(source),weakref.ref(base)
        with cache.activate(backend()):
            for recipe in ('a','b'):
                pack_operand(source,recipe,source.nbytes,lambda:source.copy())
        self.assertEqual(len(cache.entries),2)
        del source,base
        gc.collect()
        self.assertIsNone(source_ref()); self.assertIsNone(base_ref())
        self.assertEqual(cache.source_releases,2)
        self.assertEqual(cache.bytes,0)

    def test_capacity_eviction_oversize_failure_and_stream_change(self):
        cache=PackedValueCache(48,array_type=np.ndarray)
        a=np.arange(4.);b=a+2
        with cache.activate(backend()):
            first=pack_operand(a,(),32,lambda:a.copy())
            pack_operand(b,(),32,lambda:b.copy())
            self.assertEqual(cache.evictions,1)
            self.assertLessEqual(cache.peak_bytes,48)
            def fail(): raise RuntimeError('copy failed')
            with self.assertRaisesRegex(RuntimeError,'copy failed'):
                pack_operand(a,('new',),32,fail)
            self.assertEqual(cache.bytes,0)
            pack_operand(a,(),32,lambda:a.copy())
        with cache.activate(backend(8)):
            changed=pack_operand(a,(),32,lambda:a.copy())
            self.assertIsNot(changed,first)
        cache.clear()
        self.assertEqual(cache.bytes,0)
        self.assertEqual(len(cache.entries),0)
        self.assertGreater(cache.cleared_entries,0)

    def test_nested_scope_failure_does_not_leak_or_replace_outer_owner(self):
        a=np.arange(4.);outer=PackedValueCache(64,array_type=np.ndarray)
        inner=PackedValueCache(64,array_type=np.ndarray)
        with outer.activate(backend()):
            expected=pack_operand(a,(),32,lambda:a.copy())
            with self.assertRaisesRegex(ValueError,'abort'):
                with inner.activate(backend()):
                    self.assertIsNot(pack_operand(a,(),32,lambda:a.copy()),expected)
                    raise ValueError('abort')
            self.assertIs(pack_operand(a,(),32,lambda:a.copy()),expected)
        self.assertIsNot(pack_operand(a,(),32,lambda:a.copy()),expected)


class AdmissionTests(unittest.TestCase):
    def plan(self,spare=1024,backend_name='cupy',opaque=False):
        state=SimpleNamespace(problem=SimpleNamespace(backend=backend_name))
        prep=SourcePreparation((),recipe=None) if opaque else SimpleNamespace(recipe=object())
        point=MemoryPoint(0,'contract',0,100,0,0,0,0)
        return SceneStoragePlan(state,{}, {'a':prep},(),(point,),0,{},0,100,True,
                                100+spare,None)

    def test_p_reserves_only_spare_and_keeps_budgets_and_opaque_behavior(self):
        base=self.plan()
        plan=admit_packed_values(base)
        self.assertEqual(plan.packed_value_cache_device_bytes,256)
        self.assertEqual(plan.peak_device_bytes,356)
        self.assertEqual(plan.memory[0].device_bytes,356)
        self.assertEqual(plan.reserved_device_bytes,base.reserved_device_bytes)
        self.assertIs(admit_packed_values(plan),plan)
        self.assertEqual(admit_packed_values(self.plan(0)).packed_value_cache_device_bytes,0)
        self.assertEqual(admit_packed_values(self.plan(backend_name='numpy')).packed_value_cache_device_bytes,0)
        self.assertEqual(admit_packed_values(self.plan(opaque=True)).packed_value_cache_device_bytes,0)

    def test_external_reserve_can_shrink_or_disable_optimization_without_failing(self):
        plan=admit_packed_values(self.plan())
        reduced=plan.with_reserves(reserved_device_bytes=1000)
        self.assertTrue(reduced.feasible)
        self.assertEqual(reduced.packed_value_cache_device_bytes,24)
        self.assertEqual(reduced.peak_device_bytes,1124)
        disabled=plan.with_reserves(reserved_device_bytes=1024)
        self.assertTrue(disabled.feasible)
        self.assertEqual(disabled.packed_value_cache_device_bytes,0)

    def test_old_archive_absence_restores_zero_capacity(self):
        from lqcdweave.storage.contraction_plans.api import encode_plan_value,decode_plan_value
        # Use a real native complete plan to exercise the registered codec.
        from correlator.planning.test_cfg_plan import CfgPlanTests
        plan,_,_=CfgPlanTests.dense_fixture()
        encoded=encode_plan_value(plan)
        def remove(value):
            if isinstance(value,dict):
                value.pop('packed_value_cache_device_bytes',None)
                for item in value.values(): remove(item)
            elif isinstance(value,list):
                for item in value: remove(item)
        remove(encoded)
        restored=decode_plan_value(encoded)
        self.assertEqual(restored.packed_value_cache_device_bytes,0)
        self.assertTrue(all(p.packed_value_cache_device_bytes==0 for p in restored.memory))

    def test_cfg_envelope_preserves_explicit_cache_separately_from_io_and_output(self):
        from correlator.planning.test_cfg_plan import CfgPlanTests
        from lqcdweave.contracts.correlator.api import describe_coordinates
        from lqcdweave.contraction.planning.api import build_cfg_plan,audit_cfg_plan
        template,preparations,specs=CfgPlanTests.dense_fixture('cupy')
        base=replace(template,preparations=preparations)
        admitted=admit_packed_values(base)
        points,_=describe_coordinates()
        cfg=build_cfg_plan(admitted,None,points.points,{},preparations,specs,debug=True)
        cache=cfg.statistics['packed_value_cache_device_bytes']
        self.assertGreater(cache,0)
        self.assertLessEqual(cfg.peak_device_bytes+cfg.prefetch_device_bytes,
                             cfg.statistics['device_memory_cap'])
        self.assertEqual(audit_cfg_plan(cfg)['peak_device_bytes'],cfg.peak_device_bytes)


if __name__=='__main__': unittest.main()
