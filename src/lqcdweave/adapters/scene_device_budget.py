"""Scoped CuPy pool cap for an explicitly selected device and storage plan."""
from contextlib import contextmanager
import time


class SceneDeviceAdmission:
    """Refresh fixed-path cache budgets from actual CuPy pool/device usage.

    Call prepare before logical retention, and batch_context after its evictions.
    No path search or physical input eviction is performed here.
    """
    def __init__(self, *, backend, safety_bytes=512 << 20):
        if backend.__name__ != 'cupy': raise ValueError('device admission requires CuPy')
        if type(safety_bytes) is not int or safety_bytes < 0: raise ValueError('invalid safety margin')
        self.backend=backend;self.safety_bytes=safety_bytes;self.non_pool_highwater=0
        self.last_admission={}

    def snapshot(self):
        self.backend.cuda.get_current_stream().synchronize()
        pool=self.backend.get_default_memory_pool()
        used=int(pool.used_bytes());reserved=int(pool.total_bytes())
        free,total=self.backend.cuda.runtime.memGetInfo()
        device=int(total-free)
        return dict(pool_used=used,pool_reserved=reserved,pool_idle=max(0,reserved-used),
                    device_used=device,non_pool=max(0,device-reserved))

    def prepare(self, plan):
        started=time.perf_counter()
        if plan.retention_planned: raise ValueError('refresh budget before cache retention')
        measured=self.snapshot()
        self.non_pool_highwater=max(self.non_pool_highwater,measured['non_pool'])
        reserve=max(plan.reserved_device_bytes,self.non_pool_highwater+self.safety_bytes)
        self.last_admission=dict(before_retention=measured,non_pool_highwater=self.non_pool_highwater,
                                 safety_bytes=self.safety_bytes,reserved_device_bytes=reserve,
                                 prepare_seconds=time.perf_counter()-started)
        return plan.with_reserves(reserved_device_bytes=reserve)

    def observe(self, profile):
        peaks=profile.get('sampled_peak_bytes',{})
        self.non_pool_highwater=max(self.non_pool_highwater,int(peaks.get('non_pool',0)))

    @contextmanager
    def batch_context(self, plan):
        started=time.perf_counter()
        pool=self.backend.get_default_memory_pool()
        before=self.snapshot()
        # Eviction has already removed logical references. Return idle pool
        # allocations so reserved blocks cannot accumulate across time pairs.
        if before['pool_idle']: pool.free_all_blocks()
        after=self.snapshot()
        self.non_pool_highwater=max(self.non_pool_highwater,after['non_pool'])
        available=plan.memory_limit-plan.reserved_device_bytes
        self.last_admission.update(after_eviction=before,after_pool_trim=after,
            planned_pool_budget=available,existing_pool_limit=pool.get_limit(),
            reserve_excess_bytes=max(0,after['non_pool']-plan.reserved_device_bytes),
            pool_budget_excess_bytes=max(0,after['pool_reserved']-available),
            device_target_excess_bytes=max(0,after['device_used']-plan.memory_limit))
        self.last_admission['pool_trim_and_admit_seconds']=time.perf_counter()-started
        # The target constrains planning, not the allocator. Measurements feed
        # the next batch's reserve; genuine backend failures still propagate.
        yield



@contextmanager
def scene_device_budget(plan, *, backend):
    """Limit default-pool allocations; validate declared non-pool reserve first.

    Requires exclusive use of this process's default pool during the context.
    Library allocations outside that pool are NOT capped by CuPy; callers must
    include their bounds in the reserve and validate actual device measurements.
    Restores the previous pool limit on both success and failure.
    """
    if backend.__name__ != "cupy" or plan.state.problem.backend != "cupy":
        raise ValueError("device budget requires a CuPy plan and backend")
    if not plan.feasible:
        raise MemoryError("storage plan is not feasible")
    pool = backend.get_default_memory_pool()
    backend.cuda.get_current_stream().synchronize()
    free, total = backend.cuda.runtime.memGetInfo()
    non_pool = max(0, int(total - free) - int(pool.total_bytes()))
    if non_pool > plan.reserved_device_bytes:
        raise MemoryError("current non-pool device usage exceeds the declared reserve; replan")
    available = plan.memory_limit - plan.reserved_device_bytes
    if available <= 0:
        raise MemoryError("device reserve leaves no tensor budget")
    previous = pool.get_limit()
    limit = min(previous, available) if previous else available
    if pool.used_bytes() > limit:
        raise MemoryError("pre-existing live pool allocations exceed the tensor budget")
    if pool.total_bytes() > limit:
        pool.free_all_blocks()
    pool.set_limit(size=limit)
    try:
        yield
    finally:
        pool.set_limit(size=previous)
