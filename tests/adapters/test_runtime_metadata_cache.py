"""Static topology/hardware caching must not cache live capacity or device identity."""
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import pytest
from lqcdweave.adapters import hardware_budget as hb,execution_target as et
from lqcdweave.contracts.memory import MemoryLimits

def test_cgroup_cache_reads_membership_and_mounts_each_time():
    hb._parse_cgroup_mounts.cache_clear()
    text={'/proc/self/cgroup':'0::/job-a\n','/proc/self/mountinfo':'1 0 0:1 / /cg rw - cgroup2 cgroup rw\n'}
    def read(path,**kwargs):return text[path.as_posix()]
    with patch.object(hb.Path,'read_text',read),patch.object(hb.Path,'resolve',lambda p:p),patch.object(hb.Path,'is_dir',return_value=False):
        first=hb._cgroup_mount_candidates()
        assert first==hb._cgroup_mount_candidates()
        assert hb._parse_cgroup_mounts.cache_info().hits==1
        text['/proc/self/cgroup']='0::/job-b\n'
        second=hb._cgroup_mount_candidates()
        assert first!=second
        text['/proc/self/mountinfo']='1 0 0:1 / /other rw - cgroup2 cgroup rw\n'
        assert second!=hb._cgroup_mount_candidates()

def test_cached_paths_do_not_cache_headroom():
    with patch.object(hb,'_cgroup_mount_candidates',return_value=(('cgroup2',hb.Path('/cg')),)),patch.object(hb,'_read_integer_file',side_effect=[100,20,100,80]):
        assert hb._cgroup_memory_headroom_bytes()==80
        assert hb._cgroup_memory_headroom_bytes()==20

def fake_backend():
    backend=ModuleType('cupy');backend.__version__='1'
    state=dict(device=0,context=10,runtime=12040,driver=12090,queries=0)
    def properties(device):
        state['queries']+=1
        return dict(name=f'gpu-{device}'.encode())
    backend.cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=False,getDevice=lambda:state['device'],
        getDeviceProperties=properties,runtimeGetVersion=lambda:state['runtime'],driverGetVersion=lambda:state['driver']),
        driver=SimpleNamespace(ctxGetCurrent=lambda:state['context']))
    return backend,state

def test_hardware_cache_invalidates_for_device_context_and_versions():
    et._checked_fingerprint.cache_clear()
    backend,state=fake_backend()
    target=et.observe_execution_target(backend,MemoryLimits(None,None))
    et.check_execution_target(target,backend)
    count=state['queries']
    et.check_execution_target(target,backend);assert state['queries']==count
    state['context']=11;et.check_execution_target(target,backend)
    assert state['queries']==count+1
    state['device']=1
    with pytest.raises(ValueError,match='hardware'):et.check_execution_target(target,backend)
    state['device']=0;state['runtime']+=1
    with pytest.raises(ValueError,match='hardware'):et.check_execution_target(target,backend)
    backend.__version__='2'
    with pytest.raises(ValueError,match='software'):et.check_execution_target(target,backend)

def test_hip_observes_current_hardware_without_unsupported_context_query():
    backend,state=fake_backend()
    backend.cuda.runtime.is_hip=True
    with patch.object(backend.cuda.driver,'ctxGetCurrent',side_effect=RuntimeError('unsupported')) as context:
        target=et.observe_execution_target(backend,MemoryLimits(None,None))
        et.check_execution_target(target,backend)
        count=state['queries']
        et.check_execution_target(target,backend)
        assert state['queries']==count+1
        state['device']=1
        with pytest.raises(ValueError,match='hardware'):et.check_execution_target(target,backend)
        context.assert_not_called()

def test_cuda_context_errors_are_not_hidden():
    backend,_=fake_backend()
    target=et.observe_execution_target(backend,MemoryLimits(None,None))
    with patch.object(backend.cuda.driver,'ctxGetCurrent',side_effect=RuntimeError('driver failed')):
        with pytest.raises(RuntimeError,match='driver failed'):et.check_execution_target(target,backend)

