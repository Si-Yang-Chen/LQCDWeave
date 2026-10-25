import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import numpy as np
from test_scene_storage_batches import batch,Loader
from lqcdweave.contraction.planning.api import (
    ContractionState, join_storage_batches, logical_cost_profile, plan_scene_storage,
    retain_storage_tensors,
)
from lqcdweave.contraction.execution.api import replay_storage_batches
from lqcdweave.contraction.performance.api import CostProfile, predict_plan_cost
from lqcdweave.storage.contraction_plans.api import save_storage_schedule,load_storage_schedule

class CountLoader(Loader):
    def __init__(self):super().__init__();self.prepared=[]
    def materialize(self,key,source,blocks):
        self.prepared.append(key)
        return super().materialize(key,source,blocks)

class LogicalRetentionTests(unittest.TestCase):
    def test_retains_inputs_and_results_until_explicit_pressure_eviction(self):
        original=join_storage_batches((batch('a'),batch('b'),batch('a')))
        plans=retain_storage_tensors(original)
        for before,after in zip(original,plans):
            self.assertEqual([(a.kind,a.tensor,a.operands,a.equation) for a in before.state.actions],
                             [(a.kind,a.tensor,a.operands,a.equation) for a in after.state.actions])
        self.assertIn('a',plans[1].retained_before)
        self.assertIn('a',plans[1].retained_after)
        self.assertIn('b',plans[2].retained_after) # No future demand does not trigger automatic release.
        self.assertTrue(next(a for a in plans[2].state.actions if a.kind=='load').reused)
        self.assertTrue(next(a for a in plans[2].state.actions if a.kind=='contract').reused)
        loader=CountLoader();results=[];reports=[]
        with patch('lqcdweave.contraction.execution.scene_runtime.np.einsum',wraps=np.einsum) as contract:
            replay_storage_batches(plans,loader,lambda i,r:(results.append(r.values[0]),reports.append(r)),measure=True,prejoined=True)
        self.assertEqual(results,[6.,6.,6.]);self.assertEqual(contract.call_count,2)
        self.assertEqual(loader.prepared,['a','b']);self.assertEqual(loader.reads,['a','b'])
        self.assertLess(sum(p.seconds for p in plans),sum(p.seconds for p in original))
        self.assertAlmostEqual(sum(t.estimated_seconds for r in reports for t in r.timings),sum(p.seconds for p in plans))
        prediction=predict_plan_cost(plans,CostProfile(),workload={},provenance={})
        self.assertAlmostEqual(prediction['total_seconds'],sum(p.seconds for p in plans))
        self.assertGreater(plans[0].peak_host_bytes,original[0].peak_host_bytes)

    def test_memory_pressure_evicts_explicitly_and_prepares_again_without_rereading(self):
        original=join_storage_batches((batch('a'),batch('b'),batch('a')))
        original=list(original);original[1]=replace(original[1],memory_limit=original[1].peak_host_bytes)
        plans=retain_storage_tensors(original)
        self.assertIn('a',plans[1].retained_releases)
        self.assertEqual(plans[1].retained_before,())
        self.assertEqual(plans[1].retained_after,())
        self.assertTrue(all(p.peak_host_bytes<=p.memory_limit for p in plans))
        loader=CountLoader();values=[]
        replay_storage_batches(plans,loader,lambda i,r:values.append(r[0]),prejoined=True)
        self.assertEqual(values,[6.,6.,6.]);self.assertEqual(loader.prepared,['a','b','a'])
        self.assertEqual(loader.reads,['a','b'])

    def test_fixed_path_failure_never_launches_second_search(self):
        from lqcdweave.contraction.planning.api import solve_storage_batch
        original=batch('a')
        with patch('lqcdweave.contraction.planning.batch.solve_storage',
                   side_effect=AssertionError('unexpected second optimization')):
            result=solve_storage_batch(original.state.problem,original.blocks,original.preparations,
                memory_limit=1,template=original.state,source_map={'a':'a'},allow_research=False)
        self.assertIsNone(result.plan)
        self.assertTrue(result.diagnostic['research_disabled'])

    def test_device_retention_counts_against_device_budget(self):
        base=batch('a');problem=replace(base.state.problem,backend='cupy')
        model=logical_cost_profile(CostProfile())
        # Device ownership is charged in 512-byte allocation units, including
        # raw input, prepared snapshots, scratch and scalar output.
        device_cap=4096
        state=ContractionState.initial(problem).load('a',model,device_cap)
        state=state.contract(0,(0,),model,device_cap).release('a')
        blocks={'a':replace(base.blocks['a'],host_bytes=0,device_bytes=32,transfer_bytes=32)}
        original=plan_scene_storage(state,blocks,base.preparations,profile=CostProfile(),
                                   memory_limit=device_cap,host_memory_limit=256)
        self.assertTrue(original.feasible)
        plans=retain_storage_tensors(join_storage_batches((original,original)))
        self.assertGreater(plans[0].peak_device_bytes,original.peak_device_bytes)
        self.assertTrue(all(p.peak_device_bytes<=device_cap for p in plans))
        self.assertTrue(all(point.retained_host_bytes==0 for p in plans for point in p.memory))
        self.assertTrue(next(a for a in plans[1].state.actions if a.kind=='load').reused)
        undersized=plan_scene_storage(state,blocks,base.preparations,profile=CostProfile(),
                                     memory_limit=256,host_memory_limit=256)
        self.assertFalse(undersized.feasible)
        with self.assertRaisesRegex(MemoryError, 'base schedule exceeds memory limits'):
            retain_storage_tensors((undersized,))

    def test_archive_restores_cache_policy_and_exception_closes_inputs(self):
        plans=retain_storage_tensors(join_storage_batches((batch('a'),batch('a'))))
        shapes={'a':[4]};hardware={'backend':'numpy','memory':10000}
        expected=tuple(p.state.problem for p in plans)
        blocks={k:v for p in plans for k,v in p.blocks.items()}
        preparations={k:v for p in plans for k,v in p.preparations.items()}
        with TemporaryDirectory() as d:
            path=Path(d)/'schedule.gz'
            save_storage_schedule(path,plans,shapes=shapes,hardware=hardware)
            restored,_=load_storage_schedule(path,shapes=shapes,hardware=hardware,
                expected=expected,blocks=blocks,preparations=preparations)
            self.assertEqual(restored[1].retained_before,plans[1].retained_before)
            loader=CountLoader();values=[]
            replay_storage_batches(restored,loader,lambda i,r:values.append(r[0]),prejoined=True)
            self.assertEqual(values,[6.,6.]);self.assertEqual(loader.prepared,['a'])
        loader=CountLoader()
        def fail(i,r):raise RuntimeError('consumer failed')
        with self.assertRaises(RuntimeError):replay_storage_batches(plans,loader,fail,prejoined=True)
        self.assertEqual(loader.reads,loader.closes)

if __name__=='__main__':unittest.main()
