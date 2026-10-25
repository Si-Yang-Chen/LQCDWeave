"""Resource protocol tests; these do not claim real CUDA validation."""
from types import SimpleNamespace
from threading import Thread
from unittest.mock import patch
import numpy as np
import pytest
from correlator.execution.test_device import _Tracker,_Pool,_FakeCupy
from lqcdweave.contraction.execution.run_resources import RunResources
from lqcdweave.contraction.execution.device import device_execution,active_device_execution
from lqcdweave.adapters import hardware_budget as hb
from lqcdweave.runtime import memory
from lqcdweave.adapters.selection import host_arrays_resident
from lqcdweave.contraction.execution import read_ahead as ra

def backend():
    tracker=_Tracker();cp=_FakeCupy(tracker);state={'context':10}
    cp.cuda.driver=SimpleNamespace(ctxGetCurrent=lambda:state['context'])
    class Pool(_Pool):
        def free_bytes(self):return 600 if not self.freed else 0
    cp.cuda.MemoryPool=lambda:Pool(tracker)
    return cp,tracker,state

def test_pool_reuse_scope_cleanup_and_explicit_close():
    cp,tracker,state=backend();owner=RunResources()
    with owner.activate():
        for i in range(2):
            with device_execution(cp,device_memory_limit=1024) as device:
                ticket=device.upload(np.array([i]))
                np.testing.assert_array_equal(device.ready(ticket,wait=True),[i])
            assert active_device_execution() is None
        assert len(tracker.pools)==1 and not tracker.pools[0].freed
        assert owner.reusable_bytes(cp)==600
    owner.close();assert tracker.pools[0].freed
    owner.close()
    with owner.activate(),device_execution(cp,device_memory_limit=1024):pass
    assert len(tracker.pools)==2
    owner.close()

def test_hip_keeps_fresh_scope_pools_without_querying_context():
    cp,tracker,_=backend();owner=RunResources()
    cp.cuda.runtime.is_hip=True
    with patch.object(cp.cuda.driver,'ctxGetCurrent',side_effect=RuntimeError('unsupported')) as context:
        with owner.activate():
            for i in range(2):
                with device_execution(cp,device_memory_limit=1024) as device:
                    ticket=device.upload(np.array([i]))
                    np.testing.assert_array_equal(device.ready(ticket,wait=True),[i])
                assert tracker.pools[-1].freed
                assert owner.reusable_bytes(cp)==0
        assert len(tracker.pools)==2
        owner.close()
        context.assert_not_called()

def test_pool_budget_context_invalidation_and_error_cleanup():
    cp,t,s=backend();owner=RunResources()
    with owner.activate():
        with device_execution(cp,device_memory_limit=1024):pass
        with device_execution(cp,device_memory_limit=512):pass
        assert t.pools[0].freed and t.pools[1].limit==512
        s['context']=11
        with device_execution(cp,device_memory_limit=512):pass
        assert t.pools[1].freed
        with pytest.raises(RuntimeError,match='numerical failure'):
            with device_execution(cp,device_memory_limit=512):
                raise RuntimeError('numerical failure')
        assert t.pools[2].freed and owner.reusable_bytes(cp)==0

def test_busy_owner_does_not_lend_or_credit_pool_to_concurrent_run():
    cp,t,s=backend();owner=RunResources();seen=[]
    with owner.activate(),device_execution(cp,device_memory_limit=1024):
        assert owner.reusable_bytes(cp)==0
        def concurrent():
            with owner.activate(),device_execution(cp,device_memory_limit=1024):
                seen.append(active_device_execution())
        thread=Thread(target=concurrent);thread.start();thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(t.pools)==2 and t.pools[1].freed
    assert seen and not t.pools[0].freed
    owner.close()

def test_fresh_headroom_includes_only_this_idle_pool():
    cp,t,s=backend();owner=RunResources()
    with owner.activate():
        with device_execution(cp,device_memory_limit=1024):pass
        with patch.object(hb,'_effective_local_host_available_bytes',return_value=4096),patch.object(memory,'device_memory_info',side_effect=[(100,2048),(20,2048)]):
            first=hb.resolve_diagram_memory_limits(backend=cp,memory_limit=1024,host_memory_limit=1024)
            second=hb.resolve_diagram_memory_limits(backend=cp,memory_limit=1024,host_memory_limit=1024)
        assert first.device_bytes==700
        assert second.device_bytes==620
    with patch.object(memory,'device_memory_info',return_value=(100,2048)):
        assert memory.resolve_memory_limits(None,cp,host_available=4096).device_free_bytes==100
    owner.close()

def test_resident_detection_rejects_memmap_and_views(tmp_path):
    data=np.arange(8)
    assert host_arrays_resident({'a':data,'b':[data[1:]]})
    mapped=np.memmap(tmp_path/'mapped',mode='w+',shape=(8,),dtype='float64')
    assert not host_arrays_resident(mapped)
    assert not host_arrays_resident(np.asarray(mapped)[1:])
    assert not host_arrays_resident(object())

def test_small_resident_reads_skip_only_the_prefetch_worker():
    loader=SimpleNamespace(input_readers={'p':SimpleNamespace(host_resident=True)})
    for name in ra._METHODS:setattr(loader,name,lambda *a:None)
    plan=SimpleNamespace(blocks={'p':SimpleNamespace(read_bytes=100)},
        host_memory_limit=4096,memory_limit=4096,peak_host_bytes=100,state=SimpleNamespace(problem=SimpleNamespace(backend='numpy')))
    with patch.object(ra,'_read_sizes',return_value=({},{})) as sizes,patch.object(ra,'_resolve_device_capacity',return_value=0):
        assert ra.prepare_read_ahead(plan,loader)==(plan,None,0);sizes.assert_not_called()
        loader.input_readers['p'].host_resident=False
        ra.prepare_read_ahead(plan,loader);assert sizes.call_count==1
        loader.input_readers['p'].host_resident=True
        plan.blocks['p'].read_bytes=9*1024**2
        ra.prepare_read_ahead(plan,loader);assert sizes.call_count==2

