"""Repeated complete-cfg calls reuse programs and still consume current data."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

from lqcdweave.contraction.execution import cfg_runtime
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.contraction.correlator import load_calculation_plan,save_calculation_plan


class ExecutionBindingCacheTests(unittest.TestCase):
    def fixture(self):
        from test_complete_cfg_output import CompleteOutputTests
        helper=CompleteOutputTests()
        selected,coordinates,elemental,propagator=helper.selected()
        records=[];closed=[]
        complete=complete_calculation(selected,coordinates=coordinates)
        factory=helper.factory(complete.plan,elemental,propagator,records,closed)
        complete=replace(complete,options=dict(complete.options,configuration='a',source_factory=factory))
        return complete,records,closed

    def test_repeated_handle_and_changed_cfg_reuse_tables_without_reusing_values(self):
        calculation,records,closed=self.fixture()
        with patch.object(cfg_runtime,'bind_scene_execution',wraps=cfg_runtime.bind_scene_execution) as bind:
            first=calculation.run()
            self.assertGreater(bind.call_count,0)
            self.assertGreater(first.report['compiled_execution_tables'],0)
            count=len(records)
            with patch.object(cfg_runtime,'_bind_cfg_tables',side_effect=AssertionError('rebound fixed plan')):
                second=replace(calculation,options=dict(calculation.options,configuration='b')).run()
            self.assertGreater(len(records),count)
            self.assertEqual(first.report['compiled_execution_tables'],bind.call_count)
        self.assertEqual(second.report['compiled_execution_tables'],0)
        self.assertEqual(second.report['compiled_template_variants'],0)
        self.assertEqual(second.report['execution_table_plan_reuses'],1)
        self.assertGreater(second.report['execution_table_cache_hits'],0)
        self.assertEqual(closed,['a','b'])
        np.testing.assert_allclose(second.value['two_point'],16*first.value['two_point'],rtol=3e-12,atol=3e-12)
        # Earlier output remains independently owned after another replay.
        original=first.value['two_point'].copy()
        np.testing.assert_array_equal(calculation.run().value['two_point'],original)
        np.testing.assert_array_equal(first.value['two_point'],original)

    def test_debug_rebinds_and_replaced_plan_invalidates_normal_entry(self):
        calculation,_,_=self.fixture()
        reference=calculation.run().value['two_point']
        with patch.object(cfg_runtime,'_bind_cfg_tables',wraps=cfg_runtime._bind_cfg_tables) as bind:
            for _ in range(2):
                checked=calculation.run(debug=True)
                self.assertEqual(checked.report['execution_table_plan_reuses'],0)
                self.assertGreater(checked.report['compiled_execution_tables'],0)
                np.testing.assert_allclose(checked.value['two_point'],reference,rtol=3e-12,atol=3e-12)
            self.assertEqual(bind.call_count,2)
            self.assertEqual(calculation.run().report['execution_table_plan_reuses'],1)
            self.assertEqual(bind.call_count,2)
            changed=replace(calculation,plan=replace(calculation.plan))
            self.assertEqual(changed.run().report['execution_table_plan_reuses'],0)
            self.assertEqual(bind.call_count,3)

    def test_backend_identity_and_failed_binding_do_not_reuse_incomplete_tables(self):
        calculation,_,_=self.fixture()
        owner=cfg_runtime.CfgExecutionCache()
        def bind(backend):
            return owner._bind(calculation.plan,backend,cfg_runtime._CfgTableCache(backend))
        with patch.object(cfg_runtime,'bind_scene_execution',side_effect=RuntimeError('binding failed')):
            with self.assertRaisesRegex(RuntimeError,'binding failed'):
                bind(np)
        self.assertIsNone(owner._binding)
        first=bind(np)
        self.assertIs(bind(np)[0],first[0])
        class Backend:
            __name__='numpy'
            def __getattr__(self,name):return getattr(np,name)
        other=Backend()
        self.assertIsNot(bind(other)[0],first[0])
        self.assertIs(owner._binding[1],other)
        with execution_debug(True):
            self.assertIsNot(bind(other)[0],owner._binding[2])
        self.assertIs(owner._binding[1],other)

    def test_concurrent_replays_keep_reader_and_array_owners_separate(self):
        calculation,_,closed=self.fixture()
        reference=calculation.run().value['two_point'].copy()
        other=replace(calculation,options=dict(calculation.options,configuration='b'))
        with patch.object(cfg_runtime,'_bind_cfg_tables',side_effect=AssertionError('rebound fixed plan')):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first=pool.submit(calculation.run)
                second=pool.submit(other.run)
                a,b=first.result(),second.result()
        np.testing.assert_array_equal(a.value['two_point'],reference)
        np.testing.assert_allclose(b.value['two_point'],16*reference,rtol=3e-12,atol=3e-12)
        self.assertCountEqual(closed,['a','a','b'])

    def test_mutable_event_descriptions_bind_each_time(self):
        calculation,_,_=self.fixture()
        plan=calculation.plan
        current=plan.instances[0]
        instance=replace(current,events=list(current.events))
        mutable=replace(plan,instances=(instance,*plan.instances[1:]))
        owner=cfg_runtime.CfgExecutionCache()
        with patch.object(cfg_runtime,'_bind_cfg_tables',wraps=cfg_runtime._bind_cfg_tables) as bind:
            owner._bind(mutable,np,cfg_runtime._CfgTableCache(np))
            owner._bind(mutable,np,cfg_runtime._CfgTableCache(np))
        self.assertEqual(bind.call_count,2)
        self.assertIsNone(owner._binding)

    def test_archive_contains_plan_and_new_handle_binds_once(self):
        calculation,_,_=self.fixture()
        reference=calculation.run().value['two_point'].copy()
        with TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json.gz'
            save_calculation_plan(path,calculation)
            restored=load_calculation_plan(path,configuration='a',
                source_factory=calculation.options['source_factory'])
            self.assertIsNone(restored._execution_bindings._binding)
            first=restored.run()
            self.assertGreater(first.report['compiled_execution_tables'],0)
            np.testing.assert_array_equal(first.value['two_point'],reference)
            self.assertEqual(restored.run().report['execution_table_plan_reuses'],1)


if __name__=='__main__':unittest.main()
