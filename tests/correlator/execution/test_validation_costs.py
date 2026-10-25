"""Cheap default admission, optional deep internal coverage diagnostics."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from lqcdweave.contracts.execution import TermValue
from lqcdweave.contraction.execution import domain, storage_runtime
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.planning.api import join_storage_batches, retain_storage_tensors
from correlator.execution.test_debug_execution import _make_dense_execution
from test_scene_storage_batches import batch, Loader


class ExecutionValidationCostTests(unittest.TestCase):
    def test_direct_and_planned_coverage_audit_is_debug_only_once(self):
        for optimize in ('none','anytime'):
            expanded,path,_,expected=_make_dense_execution(optimize=optimize)
            with patch.object(domain,'_validate_term_values',wraps=domain._validate_term_values) as audit:
                normal=domain.execute_contraction_path(expanded,path,debug=False)
                self.assertEqual(audit.call_count,0)
                checked=domain.execute_contraction_path(expanded,path,debug=True)
                self.assertEqual(audit.call_count,1)
            np.testing.assert_allclose(normal.term_values[0].value,expected)
            np.testing.assert_array_equal(normal.term_values[0].value,checked.term_values[0].value)

    def test_cardinality_stays_default_and_wrong_keys_are_debug_diagnostics(self):
        expanded,_,_,_=_make_dense_execution()
        bad=(TermValue('wrong',expanded.coordinates.points[0].key,np.array(1.)),)
        domain._check_term_values(expanded,bad)
        with execution_debug(True), self.assertRaisesRegex(ValueError,'term_id, coordinate_key'):
            domain._check_term_values(expanded,bad)
        with self.assertRaisesRegex(ValueError,'number of term results'):
            domain._check_term_values(expanded,())
        with self.assertRaisesRegex(ValueError,'number of term results'):
            domain._check_term_values(expanded,bad+bad)

    def test_retained_after_scan_is_debug_only_and_before_guard_remains(self):
        plans=retain_storage_tensors(join_storage_batches((batch('a'),batch('a'))))
        # Track only scans of retained-after descriptor tuples: one execution
        # lookup set is necessary; a second postcondition set is debug-only.
        normal_counts=[]
        for debug in (False,True):
            calls=[]
            def tracked_set(value=()):
                if any(value is p.retained_after for p in plans): calls.append(value)
                return set(value)
            values=[]
            with execution_debug(debug), patch.object(storage_runtime,'set',tracked_set,create=True):
                storage_runtime.replay_storage_batches(plans,Loader(),lambda i,r:values.append(r[0]),io_overlap=False)
            self.assertEqual(values,[6.,6.])
            normal_counts.append(len(calls))
        self.assertEqual(normal_counts[1]-normal_counts[0],len(plans))
        loader=Loader()
        with self.assertRaisesRegex(RuntimeError,'schedule is incomplete'):
            storage_runtime.replay_storage_batches(plans[1:],loader,lambda *args:None,io_overlap=False)
        self.assertEqual(loader.reads,[])

    def test_debug_catches_injected_missing_retention_and_cleans_up(self):
        plans=retain_storage_tensors(join_storage_batches((batch('a'),batch('a'))))
        loader=Loader();original=storage_runtime._replay_storage
        def omit_retention(*args,**kwargs):
            result=original(*args,**kwargs)
            args[4].retained.clear()
            return result
        with execution_debug(True), patch.object(storage_runtime,'_replay_storage',omit_retention):
            with self.assertRaisesRegex(RuntimeError,'audited batch plan'):
                storage_runtime.replay_storage_batches(plans,loader,lambda *args:None,io_overlap=False)
        self.assertEqual(loader.reads,loader.closes)
