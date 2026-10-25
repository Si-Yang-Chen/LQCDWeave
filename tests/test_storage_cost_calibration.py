import unittest
from dataclasses import replace
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contracts.measurement import ActionTiming,PhaseTiming,FileIOEvent,ReplayReport
from lqcdweave.contraction.performance.api import (
    calibrate_cost_profile, evaluate_cost_profile,
    execution_cost_samples, update_cost_profile,
)
from lqcdweave.contraction.execution.api import replay_scene_storage
from test_scene_storage_batches import batch,Loader


class _CalibrationFixture(unittest.TestCase):
    def report(self):
        timings=[]
        # Independent feature directions identify all four materialization terms.
        for calls,flops,traffic in [(0,0,0),(1,0,0),(0,100,0),(0,0,1000),(2,80,500)]:
            observed=.01+calls*.003+flops*.00002+traffic*.000001
            timings.append(ActionTiming('load',1.,observed,phases=(PhaseTiming('materialize',observed),),
                model_features=dict(storage_preparation=True,kernel_calls=calls,flops=flops,
                                    traffic_bytes=traffic,dtype='complex128')))
        for calls,flops,traffic in [(1,0,0),(1,100,0),(1,0,1000),(2,100,500)]:
            observed=calls*.02+flops*.00003+traffic*.000002
            timings.append(ActionTiming('contract',1.,observed,
                model_features=dict(kernel_calls=calls,flops=flops,traffic_bytes=traffic,dtype='complex128')))
        io=tuple(FileIOEvent(0,'a','sample','npy_whole_read',size,.1+size*.0001) for size in (10,100,1000))
        return ReplayReport((),tuple(timings),None,None,io_events=io)


class CalibrationTests(_CalibrationFixture):
    def test_storage_runtime_keeps_preparation_samples_and_feedback_uses_compute_only(self):
        report=replay_scene_storage(batch('a'),Loader(),measure=True)
        source=next(t for t in report.timings if t.kind=='load')
        self.assertTrue(source.model_features['storage_preparation'])
        samples=execution_cost_samples(report)
        self.assertIn('source_preparation',{sample['category'] for sample in samples})
        updated=update_cost_profile(CostProfile(),report,verified=True,hardware_fingerprint='test')
        self.assertGreater(updated.sample_count,0)
        self.assertFalse(hasattr(updated,'read_call_seconds'))
        delayed_read=replace(report,timings=tuple(
            replace(timing,phases=tuple(
                replace(phase,observed_seconds=phase.observed_seconds+10.)
                if phase.name.startswith('read_block') else phase
                for phase in timing.phases
            )) if timing.kind=='load' else timing
            for timing in report.timings
        ))
        self.assertEqual(updated,update_cost_profile(
            CostProfile(),delayed_read,verified=True,hardware_fingerprint='test'))
        for field in ['preparation_call_seconds','preparation_kernel_call_seconds',
                      'preparation_flop_seconds','preparation_traffic_seconds_per_byte']:
            with self.assertRaises(ValueError):replace(CostProfile(),**{field:-1})


if __name__=='__main__':unittest.main()
