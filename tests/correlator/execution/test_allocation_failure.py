"""Failure-only evidence and physical prefetch/retirement byte accounting."""
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
import pytest

from lqcdweave.contraction.execution.allocation_failure import (
    annotate_allocation_failure, device_buffer_bytes,
)
from lqcdweave.contraction.execution.bound_program import BoundProgram
from lqcdweave.contraction.execution.device import DeviceSourceLoader


class Array:
    def __init__(self, nbytes, memory=None):
        self.nbytes=nbytes
        self.data=SimpleNamespace(mem=memory)
        self.shape=(nbytes//8,)
        self.dtype=np.dtype('complex64')


def context():
    pool=SimpleNamespace(used_bytes=lambda:1024,total_bytes=lambda:1536,
        free_bytes=lambda:512,n_free_blocks=lambda:2,get_limit=lambda:2048)
    memory=SimpleNamespace(ptr=128,size=512)
    value=Array(128,memory)
    action=SimpleNamespace(kind='contract',tensor='result',operands=('left','right'),equation='ij,jk->ik')
    plan=SimpleNamespace(state=SimpleNamespace(actions=(action,)),peak_device_bytes=1536,
        memory_limit=2048,reserved_device_bytes=0,packed_value_cache_device_bytes=0,memory=())
    execution=SimpleNamespace(plan=plan,live={'raw':value},loader=None,
        read_ahead=SimpleNamespace(device_capacity=512,reserved_device_bytes=512))
    device=SimpleNamespace(pool=pool,statistics={'uploaded_bytes':128},_uploads=set(),_retired_inputs=set(),
        reduction_workspace=SimpleNamespace(buffer=None,report=lambda:dict(reduction_workspace_reserved_bytes=0)))
    return device,execution,value


def test_rounding_for_metadata_and_real_owner_extent_for_views():
    loader=DeviceSourceLoader(None,None)
    assert loader.prefetch_device_bytes('raw',SimpleNamespace(device_bytes=513))==1024
    assert loader.prefetch_device_bytes('raw',SimpleNamespace(device_bytes=0))==0
    assert device_buffer_bytes(Array(128,SimpleNamespace(size=4096)))==4096
    assert device_buffer_bytes(Array(513))==1024
    assert device_buffer_bytes(Array(0))==0


def test_failed_bound_call_keeps_primary_error_and_observes_before_cleanup_without_sync():
    device,execution,value=context()
    error=MemoryError('original allocation failure')
    error._size,error._total,error._limit=1024,1536,2048
    def fail(context,*,key):raise error
    program=BoundProgram((partial(fail,key='result'),),(),SimpleNamespace(__name__='cupy'),True,None)
    with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
        with pytest.raises(MemoryError) as caught:
            program.run(execution,{'left':value,'right':value})
    assert caught.value is error
    payload=error.lqcdweave_allocation_failure
    assert payload['operation']['action_index']==0
    assert payload['pool']['used_bytes']==1024
    assert payload['pool']['total_bytes']==1536
    assert payload['read_ahead']['reserved_device_bytes']==512
    assert payload['logical_values']['payload_bytes']==256
    assert payload['visible_runtime_values']['visible_allocation_owner_bytes']==512
    assert payload['raw_values']['visible_allocation_owner_bytes']==512
    assert execution.live['raw'] is value
    assert 'allocation evidence' in error.__notes__[0]
    json.dumps(payload)


def test_normal_calls_do_not_probe_allocator_and_cpu_failure_keeps_original_behavior():
    execution=SimpleNamespace(end_action=lambda *_:None)
    with patch('lqcdweave.contraction.execution.allocation_failure.annotate_allocation_failure',
               side_effect=AssertionError('normal allocator probe')):
        program=BoundProgram((lambda context:None,),(),np,False,None)
        assert program.run(execution,{})==()
        program=BoundProgram((lambda context:None,),(),SimpleNamespace(__name__='cupy'),True,None)
        assert program.run(execution,{})==()
        def fail(context):raise MemoryError('cpu failure')
        program=BoundProgram((fail,),(),np,False,None)
        with pytest.raises(MemoryError,match='cpu failure'):
            program.run(execution,{})


def test_diagnostics_never_replace_primary_and_repeated_layers_do_not_duplicate_notes():
    device,execution,value=context()
    def broken():raise RuntimeError('injected pool observation failure')
    device.pool.used_bytes=broken
    error=MemoryError('primary')
    with patch('lqcdweave.contraction.execution.device.active_device_execution',return_value=device):
        annotate_allocation_failure(error,execution,{'left':value})
        before=len(error.__notes__)
        annotate_allocation_failure(error,execution)
    assert len(error.__notes__)==before
    assert error.lqcdweave_allocation_failure['pool']['used_bytes']['unavailable']
    assert str(error)=='primary'


def test_retired_upload_reservation_uses_allocation_owner_not_slice_payload():
    from correlator.execution.test_device import _FakeCupy,_Tracker
    from lqcdweave.contraction.execution.device import device_execution
    class Event:
        def record(self,*args):pass
        def query(self):return False
        def synchronize(self):pass
    cp=_FakeCupy(_Tracker());cp.ndarray=Array;cp.cuda.Event=Event
    tickets=[]
    def retire(device,ticket):
        tickets.append(ticket)
        return True
    queue=SimpleNamespace(device_capacity=4096,_stopping=False,retire_device_input=retire)
    with device_execution(cp,device_memory_limit=8192) as device:
        value=Array(128,SimpleNamespace(ptr=128,size=4096))
        device.release_input(value,queue)
        assert len(tickets)==1
        assert tickets[0].device_bytes==4096
        assert tickets[0].value is value
