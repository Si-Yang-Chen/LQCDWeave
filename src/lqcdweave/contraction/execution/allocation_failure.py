"""Failure-only accelerator allocation evidence; no normal-path sampling."""
from functools import partial
import heapq
import json
from itertools import chain

from lqcdweave.contracts.correlator.api import device_allocation_bytes


def device_buffer_bytes(value):
    """Count the actual allocation owner, including padding and borrowed views."""
    size = int(getattr(value, 'nbytes', 0))
    memory = getattr(getattr(value, 'data', None), 'mem', None)
    extent = getattr(memory, 'size', None)
    return max(size, extent) if type(extent) is int else device_allocation_bytes(size)


def _call_info(operation, plan):
    result = {}
    while isinstance(operation, partial):
        kwargs = operation.keywords or {}
        if 'index' in kwargs:
            result['action_index'] = kwargs['index']
        if 'key' in kwargs:
            result['tensor'] = str(kwargs['key'])
        operation = kwargs.get('operation', operation.func)
    result['call'] = getattr(operation, '__name__', type(operation).__name__)
    closure = getattr(operation, '__closure__', None)
    if closure:
        values = dict(zip(operation.__code__.co_freevars, closure))
        if 'result' in values:
            result['tensor'] = str(values['result'].cell_contents)
    actions = getattr(getattr(plan, 'state', None), 'actions', ())
    if 'action_index' not in result and 'tensor' in result:
        result['action_index'] = next((index for index,action in enumerate(actions)
            if action.tensor==result['tensor'] and action.kind!='release'), None)
    index = result.get('action_index')
    if type(index) is int and 0<=index<len(actions):
        action = actions[index]
        result.update(kind=action.kind,tensor=action.tensor,operands=list(action.operands),
                      equation=str(action.equation))
    return result


def _owners(items):
    owners, rows = {}, []
    for key,value in items:
        nbytes = getattr(value,'nbytes',None)
        if not isinstance(nbytes,int):continue
        memory = getattr(getattr(value,'data',None),'mem',None)
        if memory is not None:
            identity = getattr(memory,'ptr',None)
            identity = id(memory) if identity is None else identity
            owners[identity] = max(owners.get(identity,0),device_buffer_bytes(value))
        rows.append(dict(key=str(key),payload_bytes=nbytes,shape=list(value.shape),dtype=str(value.dtype)))
    return dict(arrays=len(rows),payload_bytes=sum(row['payload_bytes'] for row in rows),
        visible_allocation_owners=len(owners),visible_allocation_owner_bytes=sum(owners.values()),
        largest=heapq.nlargest(8,rows,key=lambda row:row['payload_bytes']))


def _pool_value(pool, name):
    function = getattr(pool,name,None)
    try:
        return function() if callable(function) else None
    except Exception as error:
        return dict(unavailable=str(error))


def annotate_allocation_failure(error, execution, arrays=None, operation=None):
    """Attach observations before storage/queue cleanup, without retry or sync.

    A kernel's local temporary frames have already unwound. Pool total includes
    cache/split chunks; owner sums describe only visible live arrays and do not
    claim to partition the pool or prove fragmentation by themselves.
    """
    if hasattr(error,'lqcdweave_allocation_failure'):
        return
    try:
        from .device import active_device_execution
        device = active_device_execution()
        if device is None:return
        plan = getattr(execution,'plan',None)
        queue = getattr(execution,'read_ahead',None)
        call = _call_info(operation,plan)
        raw = getattr(execution,'live',{})
        logical = () if arrays is None else tuple(arrays.items())
        tickets = list(device._uploads) + list(device._retired_inputs)
        if queue is not None:
            tickets.extend(getattr(item,'value',None) for item in getattr(queue,'pending',{}).values())
            tickets.extend(getattr(queue,'deferred',{}).values())
        inputs = [(f'input-ticket:{index}',getattr(ticket,'value',None))
                  for index,ticket in enumerate(tickets)]
        prepared = getattr(getattr(execution,'loader',None),'prepared',None)
        constants = [(f'constant:{index}',value) for index,value in
                     enumerate(getattr(prepared,'_constant_pool',{}).values())]
        scratch = [('reduction-workspace',device.reduction_workspace.buffer)]
        payload = dict(
            observation='after failed kernel locals unwind, before X storage/queue cleanup; no synchronization/retry',
            exception=dict(request_bytes=getattr(error,'_size',None),
                allocated_so_far=getattr(error,'_total',None),limit=getattr(error,'_limit',None)),
            operation=call,
            pool={name:_pool_value(device.pool,name) for name in
                  ('used_bytes','total_bytes','free_bytes','n_free_blocks','get_limit')},
            plan=dict(peak_device_bytes=getattr(plan,'peak_device_bytes',None),
                memory_limit=getattr(plan,'memory_limit',None),
                reserved_device_bytes=getattr(plan,'reserved_device_bytes',None),
                packed_value_cache_device_bytes=getattr(plan,'packed_value_cache_device_bytes',None)),
            logical_values=_owners(logical),
            raw_values=_owners(raw.items()),
            visible_runtime_values=_owners(chain(logical,raw.items(),inputs,constants,scratch)),
            read_ahead=(None if queue is None else {name:getattr(queue,name,None) for name in
                ('device_capacity','reserved_device_bytes','peak_device_bytes','retired_input_bytes',
                 'peak_retired_input_bytes','submitted_reads','completed_reads','pending_read_limit')}),
            device_statistics=dict(device.statistics),
            upload_owners=len(device._uploads),retired_input_owners=len(device._retired_inputs),
            input_ticket_errors=[dict(index=index,error=str(ticket.error),kind=type(ticket.error).__name__)
                for index,ticket in enumerate(tickets) if getattr(ticket,'error',None) is not None],
            reduction_workspace=device.reduction_workspace.report())
        index = call.get('action_index')
        payload['planned_points_at_operation'] = [
            dict(phase=point.phase,raw_device_bytes=point.raw_device_bytes,
                logical_device_bytes=point.logical_device_bytes,workspace_device_bytes=point.workspace_device_bytes,
                reserved_device_bytes=point.reserved_device_bytes,
                allocation_rounding_device_bytes=getattr(point,'allocation_rounding_device_bytes',0),
                scalar_reduction_device_bytes=getattr(point,'scalar_reduction_device_bytes',0),
                device_bytes=point.device_bytes)
            for point in getattr(plan,'memory',()) if point.action_index==index]
        error.lqcdweave_allocation_failure = payload
        if callable(getattr(error,'add_note',None)):
            error.add_note('本软件 allocation evidence: '+json.dumps(payload,sort_keys=True))
    except Exception as diagnostic_error:
        if callable(getattr(error,'add_note',None)):
            error.add_note(f'Allocation evidence unavailable: {diagnostic_error!r}')


__all__ = ['device_buffer_bytes','annotate_allocation_failure']
