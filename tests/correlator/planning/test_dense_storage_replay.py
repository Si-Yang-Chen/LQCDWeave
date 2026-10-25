"""Direct dense resource accounting matches independent algebra replay."""
from dataclasses import replace
from unittest.mock import patch
import unittest

from correlator.planning import test_dense_batches
from lqcdweave.contraction.planning.api import CostProfile,plan_scene_storage
from lqcdweave.contraction.planning import storage


class DenseStorageReplayTests(unittest.TestCase):
    def test_complete_memory_trace_matches_debug_for_both_backends(self):
        original=test_dense_batches.DenseBatchIntegrationTests().plan().plan
        for backend in ('numpy','cupy'):
            state=replace(original.state,problem=replace(original.state.problem,backend=backend))
            kwargs=dict(profile=CostProfile(),memory_limit=10**9,host_memory_limit=10**9,
                        reserved_host_bytes=13,reserved_device_bytes=17 if backend=='cupy' else 0)
            reference=plan_scene_storage(state,original.blocks,original.preparations,debug=True,**kwargs)
            with patch.object(storage,'_apply_selected',side_effect=AssertionError('repeated algebra')):
                actual=plan_scene_storage(state,original.blocks,original.preparations,**kwargs)
            self.assertEqual(actual.memory,reference.memory)
            self.assertEqual(actual.events,reference.events)
            self.assertEqual(actual.preparation_costs,reference.preparation_costs)
            self.assertEqual((actual.peak_host_bytes,actual.peak_device_bytes),
                             (reference.peak_host_bytes,reference.peak_device_bytes))

    def test_cost_consistency_is_debug_but_residency_remains_required(self):
        original=test_dense_batches.DenseBatchIntegrationTests().plan().plan
        actions=list(original.state.actions)
        index=next(i for i,a in enumerate(actions) if a.kind=='contract' and not a.reused)
        actions[index]=replace(actions[index],seconds=actions[index].seconds+1)
        state=replace(original.state,actions=tuple(actions))
        kwargs=dict(profile=CostProfile(),memory_limit=10**9)
        plan_scene_storage(state,original.blocks,original.preparations,**kwargs)
        with self.assertRaises(ValueError):
            plan_scene_storage(state,original.blocks,original.preparations,debug=True,**kwargs)
        actions[index]=replace(actions[index],operands=('missing',))
        with self.assertRaisesRegex(ValueError,'operands must be resident'):
            plan_scene_storage(replace(state,actions=tuple(actions)),original.blocks,
                               original.preparations,**kwargs)
