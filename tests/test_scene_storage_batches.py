import unittest
from dataclasses import replace
import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourcePreparation, SourceSpec,
    StorageBlock, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState,
    logical_source_problem, logical_cost_profile,
    plan_scene_storage, join_storage_batches,
)
from lqcdweave.contraction import replay_storage_batches
from lqcdweave.contraction.planning.api import solve_storage_batch, PathSearchExhausted


def batch(key, limit=10000):
    profile = CostProfile()
    problem = ContractionProblem({key:TensorSpec((4,), 'float64')},
        (ContractionTerm((key,), 'i->'),), {key:SourceSpec(key,32)})
    model = logical_cost_profile(profile)
    state = ContractionState.initial(logical_source_problem(problem))
    state = state.load(key,model,10000).contract(0,(0,),model,10000).release(key)
    return plan_scene_storage(state,{key:StorageBlock(key,'all',32,host_bytes=32)},
        {key:SourcePreparation((key,))},profile=profile,memory_limit=limit)


class Loader:
    def __init__(self): self.reads=[];self.closes=[]
    def read_block(self,key,spec):
        self.reads.append(key)
        return np.arange(4,dtype=float)
    def materialize(self,key,source,blocks): return blocks[key]
    def close_block(self,key,value): self.closes.append(key)


class BatchTests(unittest.TestCase):
    def test_gpu_metadata_reserves_enforce_host_and_device_caps_independently(self):
        model=logical_cost_profile(CostProfile())
        problem=replace(batch('a').state.problem,backend='cupy')
        # Raw block, snapshot, scalar output and scratch each require one
        # 512-byte allocation unit. Leave exactly the tested device reserve.
        device_cap=4*512+16
        state=ContractionState.initial(problem).load('a',model,device_cap)
        state=state.contract(0,(0,),model,device_cap).release('a')
        plan=plan_scene_storage(state,{'a':StorageBlock('a','all',32,device_bytes=32,staging_host_bytes=64)},
            {'a':SourcePreparation(('a',))},profile=CostProfile(),memory_limit=device_cap,host_memory_limit=256)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.peak_device_bytes,4*512)
        self.assertIs(plan.with_reserves(),plan)
        adjusted=plan.with_reserves(reserved_host_bytes=8,reserved_device_bytes=16)
        self.assertTrue(adjusted.feasible)
        self.assertEqual(adjusted.peak_host_bytes,plan.peak_host_bytes+8)
        self.assertEqual(adjusted.peak_device_bytes,plan.peak_device_bytes+16)
        self.assertFalse(plan.with_reserves(reserved_host_bytes=256).feasible)
        self.assertFalse(plan.with_reserves(reserved_device_bytes=17).feasible)
        self.assertFalse(plan.with_reserves(reserved_device_bytes=device_cap).feasible)

    def test_measured_reserve_reaudit_preserves_lifetimes_and_rejects_before_io(self):
        joined=join_storage_batches([batch('a',256),batch('b',256),batch('a',256)])
        adjusted=tuple(p.with_reserves(reserved_host_bytes=16) for p in joined)
        for old,new in zip(joined,adjusted):
            self.assertEqual(new.peak_host_bytes,old.peak_host_bytes+16)
            self.assertEqual(new.events,old.events)
            self.assertEqual(new.seconds,old.seconds)
            self.assertIs(new.state,old.state)
        loader=Loader();values=[]
        replay_storage_batches(adjusted,loader,lambda i,r:values.append(r[0]))
        self.assertEqual(values,[6.,6.,6.])
        self.assertEqual(loader.reads,['a','b'])
        excessive=tuple(p.with_reserves(reserved_host_bytes=256) for p in joined)
        loader=Loader()
        with self.assertRaises(MemoryError):
            replay_storage_batches(excessive,loader,lambda *args:None)
        self.assertEqual(loader.reads,[])
        with self.assertRaises(ValueError): joined[0].with_reserves(reserved_host_bytes=-1)
        with self.assertRaises(ValueError): joined[0].with_reserves(reserved_device_bytes=1)

    def test_search_accounts_for_incoming_raw_and_reuses_within_headroom(self):
        original=batch('a')
        blocks=dict(original.blocks,kept=StorageBlock('kept','all',200,host_bytes=200))
        result=solve_storage_batch(original.state.problem,blocks,original.preparations,
            resident_blocks=('kept',),memory_limit=500,template=original.state,source_map={'a':'a'},
            max_states=0)
        self.assertTrue(result.plan.feasible)
        self.assertTrue(result.diagnostic['reused_operation_order'])
        self.assertEqual(result.diagnostic['raw_reservation_host_bytes'],232)
        self.assertEqual(result.plan.blocks['a'].host_bytes,32)
        # The same individual path fits 200 bytes, but its incoming raw data
        # already uses that space. The constrained search must not accept it.
        with self.assertRaises(PathSearchExhausted):
            solve_storage_batch(original.state.problem,blocks,original.preparations,
                resident_blocks=('kept',),memory_limit=200,max_states=100,deadline_seconds=1)

    def test_shadow_search_keeps_read_staging_in_budget(self):
        original=batch('a')
        blocks={'a':replace(original.blocks['a'],staging_host_bytes=600)}
        with self.assertRaises(PathSearchExhausted):
            solve_storage_batch(original.state.problem,blocks,original.preparations,
                memory_limit=500,max_states=100,deadline_seconds=1)

    def test_late_reuse_one_read_and_cost_with_outputs_consumed(self):
        plans=(batch('a'),batch('b'),batch('a'))
        joined=join_storage_batches(plans)
        self.assertEqual(joined[1].peak_host_bytes,plans[1].peak_host_bytes+32)
        self.assertGreater(joined[2].seconds, 0)
        self.assertFalse(hasattr(joined[2], 'read_seconds'))
        self.assertFalse(hasattr(joined[0], 'read_seconds'))
        loader=Loader();results=[];timed=[]
        def consume(index,report):
            results.append(float(report.values[0]))
            timed.extend(t.estimated_seconds for t in report.timings)
            if index==0: self.assertNotIn('a',loader.closes)
        replay_storage_batches(plans,loader,consume,measure=True)
        self.assertEqual(results,[6.,6.,6.])
        self.assertEqual(loader.reads,['a','b'])
        self.assertEqual(loader.closes,['b','a'])
        self.assertAlmostEqual(sum(timed),sum(p.seconds for p in joined))

    def test_shared_peak_rejects_before_read_even_when_individual_plans_fit(self):
        first=batch('a');limit=first.peak_host_bytes
        plans=tuple(batch(k,limit) for k in ('a','b','a'))
        self.assertTrue(all(p.feasible for p in plans))
        self.assertFalse(join_storage_batches(plans)[1].feasible)
        loader=Loader()
        with self.assertRaises(MemoryError): replay_storage_batches(plans,loader,lambda i,r:None)
        self.assertEqual(loader.reads,[])

    def test_failure_closes_retained_blocks(self):
        loader=Loader()
        def fail(index,report): raise RuntimeError('output failed')
        with self.assertRaisesRegex(RuntimeError,'output failed'):
            replay_storage_batches((batch('a'),batch('a')),loader,fail)
        self.assertEqual(loader.closes,['a'])

    def test_different_prefixes_require_global_union_before_joining(self):
        a=batch('a');b=replace(a,blocks={'a':replace(a.blocks['a'],selection='different')})
        with self.assertRaisesRegex(ValueError,'maximum-prefix'):
            join_storage_batches((a,b))


if __name__=='__main__': unittest.main()
