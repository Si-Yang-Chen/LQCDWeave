import unittest
from dataclasses import dataclass,replace
from types import SimpleNamespace
from lqcdweave.adapters.scene_device_budget import SceneDeviceAdmission


@dataclass
class Plan:
    reserved_device_bytes:int=100
    memory_limit:int=1000
    retention_planned:bool=False
    feasible:bool=True
    def with_reserves(self,**kw):return replace(self,**kw)


class Pool:
    def __init__(self):self.used=200;self.reserved=700;self.limit=0;self.trimmed=0
    def used_bytes(self):return self.used
    def total_bytes(self):return self.reserved
    def free_all_blocks(self):self.reserved=self.used;self.trimmed+=1
    def get_limit(self):return self.limit
    def set_limit(self,*,size):self.limit=size


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.pool=Pool();self.external=120
        self.backend=SimpleNamespace(__name__='cupy',get_default_memory_pool=lambda:self.pool,
            cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None),
                runtime=SimpleNamespace(memGetInfo=lambda:(2000-self.pool.reserved-self.external,2000))))

    def test_refreshes_reserve_and_trims_real_idle_allocations(self):
        guard=SceneDeviceAdmission(backend=self.backend,safety_bytes=50)
        p=guard.prepare(Plan());self.assertEqual(p.reserved_device_bytes,170)
        with guard.batch_context(p):
            self.assertEqual(self.pool.reserved,200)
            self.assertEqual(self.pool.limit,0)
        self.assertEqual(self.pool.limit,0)
        guard.observe({'sampled_peak_bytes':{'non_pool':250}})
        next_plan=guard.prepare(Plan())
        self.assertEqual(next_plan.reserved_device_bytes,300)
        with guard.batch_context(next_plan): self.assertEqual(self.pool.limit,0)

    def test_external_growth_is_recorded_and_updates_next_plan(self):
        guard=SceneDeviceAdmission(backend=self.backend,safety_bytes=50)
        plan=guard.prepare(Plan());self.external=200
        with guard.batch_context(plan): pass
        self.assertEqual(guard.last_admission['reserve_excess_bytes'],30)
        self.assertEqual(guard.prepare(Plan()).reserved_device_bytes,250)

    def test_live_allocations_over_target_do_not_reject_execution(self):
        guard=SceneDeviceAdmission(backend=self.backend,safety_bytes=50)
        plan=guard.prepare(Plan());self.pool.used=900;self.pool.reserved=950
        with guard.batch_context(plan): pass
        self.assertEqual(guard.last_admission['device_target_excess_bytes'],20)
        self.assertEqual(self.pool.limit,0)

    def test_backend_allocation_failure_propagates(self):
        guard=SceneDeviceAdmission(backend=self.backend,safety_bytes=50)
        with self.assertRaisesRegex(MemoryError,'backend allocation'):
            with guard.batch_context(guard.prepare(Plan())):
                raise MemoryError('backend allocation')

    def test_restores_limit_after_consumer_exception(self):
        guard=SceneDeviceAdmission(backend=self.backend,safety_bytes=50)
        p=guard.prepare(Plan());self.pool.limit=800
        with self.assertRaises(RuntimeError):
            with guard.batch_context(p):raise RuntimeError('failed consumer')
        self.assertEqual(self.pool.limit,800)


if __name__=='__main__':unittest.main()
