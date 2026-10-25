"""Continuous cfg replay using the common scene interpreter and read queue."""
from collections import deque
from collections.abc import MutableMapping
from contextlib import ExitStack
from dataclasses import replace
from operator import itemgetter
from types import SimpleNamespace
from time import perf_counter
import numpy as np
import sys
from lqcdweave.contracts.correlator.api import (
    Action, ArrayConstant, CfgInstance, CfgPlan, DenseKernelBatch, DenseGemmGroup, StorageEvent,
)
from lqcdweave.contraction.planning.api import validate_cfg_allocation_envelope
from .read_ahead import ReadAhead
from .storage_runtime import _BlockExecution, validate_coordinate_batch_workspace
from .scene_runtime import replay, bind_scene_execution
from .device import active_device_execution, device_execution, DeviceSourceLoader
from .diagnostics import debug_enabled, execution_debug
from .binding_cache import ExecutionBindingCache
from .preparation import bind_prepared_loader, PreparedSourceLoader, _NumericNodeBinding
from .operations.result_views import _DeferredResultView, _ResultOwner
from .output_queue import active_output_queue, output_queue_reserve, OutputConsumerQueue
from .packed_values import PackedValueCache
from .cfg_releases import _CompactReleaseTable, bind_tensor_releases


def _close_many(callbacks,primary=None):
    errors=[]
    for close in callbacks:
        try: close()
        except BaseException as error: errors.append(error)
    if primary is not None:
        for error in errors:
            if callable(getattr(primary,'add_note',None)):
                primary.add_note(f'cfg cleanup also failed: {error!r}')
    elif errors:
        raise errors[0]


def _close_cfg_events(events, primary=None):
    """Let replay cleanup annotate the actual consumer failure, if any."""
    if primary is None or isinstance(primary, GeneratorExit):
        events.close()
        return
    try:
        events.throw(primary)
    except BaseException as error:
        if error is not primary:
            raise
    finally:
        events.close()


def _bind_node_ids(keys, node_ids, key_template=None):
    """Bind validated cfg IDs without constructing one NumPy scalar per ID."""
    values = node_ids.array()
    native = type(node_ids) is ArrayConstant and not debug_enabled()
    values = values.tolist() if native else map(int, values)
    if native and key_template is not None and key_template[0] is keys:
        # Copy just the preallocated key table, then fill this instance's IDs.
        # No current binding is mutated or shared with another replay/loader.
        binding = _NumericNodeBinding(key_template[1])
        binding.update(zip(keys, values, strict=True))
        return binding
    return _NumericNodeBinding(zip(keys, values, strict=True))


# Dispatch metadata, not numerical storage. Keep one key-only dictionary per
# selected plan/window, regardless of its number of coordinates or cfgs.
_NODE_KEY_TEMPLATE_MAX_BYTES = 8 * 1024 * 1024


def _node_key_template(plan, previous):
    """Reuse one native template's keys, never its IDs, arrays or readers."""
    if type(plan) is not CfgPlan or plan._binding_validation is None or debug_enabled():
        return None
    counts = {}
    for instance in plan.instances:
        counts[instance.template] = counts.get(instance.template, 0) + 1
    keys = plan.templates[max(counts, key=counts.get)].keys
    if type(keys) is not tuple or any(type(key) is not str for key in keys):
        return None
    if previous is not None and (keys is previous[0] or keys == previous[0]):
        return keys, previous[1]
    # A key table needs at least one pointer per entry. Avoid constructing an
    # obviously oversized candidate; use its actual allocation for admission.
    if len(keys) * np.dtype(np.intp).itemsize > _NODE_KEY_TEMPLATE_MAX_BYTES:
        return None
    template = dict.fromkeys(keys)
    if sys.getsizeof(template) > _NODE_KEY_TEMPLATE_MAX_BYTES:
        return None
    return keys, template


def _tensor_release_pattern(instance):
    table = getattr(instance, 'tensor_releases', None)
    return () if table is None else tuple(np.unique(table.array()[:, 0]).tolist())


class _CfgBlockExecution(_BlockExecution):
    """Apply the frozen global final uses after each numerical Action."""
    def __init__(self, *args, tensor_releases=None, arrays=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.arrays = arrays
        self.tensor_releases = bind_tensor_releases(tensor_releases,
            native=type(self) is _CfgBlockExecution and type(arrays) is dict and not debug_enabled())

    def end_action(self, index, timed):
        super().end_action(index, timed)
        if type(self.tensor_releases) is _CompactReleaseTable:
            self.tensor_releases.release(index, self.arrays)
        else:
            for gid in self.tensor_releases.get(index, ()):
                del self.arrays[gid]


class _NodeView(MutableMapping):
    """Local template IDs address the current cfg's global array owners."""
    def __init__(self,binding,arrays): self.binding,self.arrays=binding,arrays
    def __getitem__(self,key):
        gid=self.binding[key]
        value=self.arrays[gid]
        if type(value) is _DeferredResultView:
            value=self.arrays[gid]=value.array()
        return value
    def __setitem__(self,key,value): self.arrays[self.binding[key]]=value
    def __delitem__(self,key): del self.arrays[self.binding[key]]
    def __iter__(self): return (key for key,gid in self.binding.items() if gid in self.arrays)
    def __len__(self): return sum(gid in self.arrays for gid in self.binding.values())

    def _raw_output_value(self,key): return self.arrays[self.binding[key]]

    def _dense_call_context(self):
        if (type(self) is _NodeView and type(self.arrays) is dict
                and type(self.binding) in (dict, _NumericNodeBinding)):
            return self.arrays, self.binding
        return ()

    def _raw_output_values(self,key_getter,single):
        gids=key_getter(self.binding)
        return (self.arrays[gids],) if single else itemgetter(*gids)(self.arrays)

    def _publish_result_views(self,value,layouts):
        # Build all references before publication, as for eager result views.
        owner=_ResultOwner(value)
        results={self.binding[key]:_DeferredResultView(owner,layout) for key,layout in layouts}
        self.arrays.update(results)


class _LocalLoader:
    preparation_bound = True
    def __init__(self,loader,binding,template_preparations=None,*,preparation_keys=None):
        self.loader,self.binding=loader,binding
        self.template_preparations=template_preparations
        bind_instance=getattr(loader,'bind_instance_preparations',None)
        self.instance_preparations=(bind_instance(template_preparations,binding,preparation_keys)
            if preparation_keys is not None and template_preparations is not None
            and callable(bind_instance) and not debug_enabled() else None)
        self.read_seconds=0.0;self.reads=0
        # Host-call wall time only; GPU time here reflects submission/waiting, not kernel time.
        self.source_materialization_call_seconds=0.0;self.source_materialization_calls=0
    def read_block(self,key,block):
        start=perf_counter()
        try: return self.loader.read_block(key,block)
        finally:
            self.read_seconds+=perf_counter()-start;self.reads+=1
    def close_block(self,key,value): return self.loader.close_block(key,value)
    def materializes_owned(self,key):
        return (isinstance(self.loader,(PreparedSourceLoader,DeviceSourceLoader))
                and self.loader.materializes_owned(str(self.binding[key])))
    def materialize(self,key,source,blocks):
        start=perf_counter()
        try:
            if self.template_preparations is not None and not debug_enabled():
                if self.instance_preparations is not None:
                    value=self.instance_preparations.materialize(key,blocks)
                    if value is not None: return value
                else:
                    materialize=getattr(self.loader,'materialize_template_preparation',None)
                    if callable(materialize):
                        value=materialize(self.template_preparations,key,self.binding,blocks)
                        if value is not None: return value
            return self.loader.materialize(str(self.binding[key]),source,blocks)
        finally:
            self.source_materialization_call_seconds+=perf_counter()-start
            self.source_materialization_calls+=1


class _QueueView:
    def __init__(self,owner,prefix): self.owner,self.prefix=owner,prefix
    def take(self,key,block): return self.owner.queue.take(self.prefix+key,block)
    def start_after(self,index): self.owner.advance(index)
    @property
    def split_preparation(self): return self.owner.queue.split_preparation
    def pending_read(self,key): return self.owner.queue.pending_read(self.prefix+key)


class _ContinuousQueue:
    def __init__(self,plan,configurations,loader_factory,capacity,device_capacity,device=None):
        self.plan,self.iterator,self.factory=plan,iter(configurations),loader_factory
        self.device=device
        self.queue=ReadAhead(SimpleNamespace(blocks={},events=()),None,{},capacity,
                             device_sizes={},device_capacity=device_capacity)
        self.claimed=deque(); self.exhausted=False; self.serial=0; self.offset=0
        self.events=tuple(replace(e,action_index=e.action_index+instance.action_offset)
            for instance in plan.instances for e in instance.events if e.kind=='read')
        self.sizes={key:request.prefetch_host_bytes for key,request in plan.requests.items()
                    if request.prefetch_host_bytes is not None}
        self.span=plan.instances[-1].action_offset+len(plan.templates[
            plan.instances[-1].template].plan.state.actions)+1
        self.first_size=self.sizes.get(self.events[0].block) if self.events else None

    def claim(self):
        if self.exhausted: return False
        try: cfg=next(self.iterator)
        except StopIteration: self.exhausted=True; return False
        scope=ExitStack()
        try:
            loader=scope.enter_context(self.factory(cfg))
            if self.device is not None:
                loader=self.device.bind_loader(loader,self.plan.preparations,self.plan.specs)
            else:
                loader=bind_prepared_loader(loader,self.plan.preparations,self.plan.specs)
            prefix=f'{self.serial}:'
            stream=SimpleNamespace(blocks=self.plan.blocks,events=self.events)
            # Eligibility may shrink at execution, but cannot grow a saved bound.
            sizes={key:loader.prefetch_host_bytes(key,self.plan.blocks[key])
                for key in self.sizes if loader.can_prefetch_block(key,self.plan.blocks[key])}
            device_sizes={key:loader.prefetch_device_bytes(key,self.plan.blocks[key])
                          for key in sizes} if self.device is not None else {}
            self.queue.extend(stream,loader,sizes,offset=self.offset,prefix=prefix,
                              device_sizes=device_sizes)
            self.claimed.append((cfg,loader,scope,prefix,self.offset))
            self.serial+=1; self.offset+=self.span
        except BaseException:
            scope.close(); raise
        return True

    def advance(self,index):
        self.queue.start_after(index)
        # Claim only after the previous read stream has been admitted, with
        # room for at least one actual future read. next() stays on this thread.
        while (not self.exhausted and self.first_size is not None
               and self.queue.cursor>=len(self.queue.reads)
               and self.queue.capacity-self.queue.reserved_bytes>=self.first_size
               and (self.device is None or self.queue.device_capacity-
                    self.queue.reserved_device_bytes>=self.plan.blocks[self.events[0].block].device_bytes)):
            if not self.claim(): break
            previous=self.queue.reserved_bytes
            self.queue.start_after(index)
            if self.queue.reserved_bytes==previous: break

    def close(self):
        callbacks=[self.queue.close]
        while self.claimed: callbacks.append(self.claimed.popleft()[2].close)
        _close_many(callbacks)

    def plan_for(self, prefix):
        return self.plan

    def coordinate_offset(self, prefix):
        return 0

    def release_plan(self, prefix):
        pass


def iter_execute_cfg_plan(plan,configurations,loader_factory,*,backend=np,
                          host_memory_limit=None,device_memory_limit=None,
                          prefetch_host_bytes=None,prefetch_device_bytes=None,debug=False,report=None,
                          transform=None,device_transform=None,execution_cache=None):
    """Yield complete ordered host results inside the saved target lifetime.

    With spare host/device budget, two existing output slots overlap replay,
    download and the optional ``transform(cfg, index, host_values)`` callback.
    A transform is called only for points, never for cfg completion. It may run
    on the single result worker; it must not access input readers or CUDA state.
    The caller owns previously yielded values and must close early-stopped
    iterators. CPU, debug and tight-budget execution stay synchronous.
    A CfgExecutionCache reuses only the same immutable plan/backend's callable
    tables across calls; every call still opens current readers and arrays.
    """
    validate_cfg_allocation_envelope(plan)
    if device_transform is not None and plan.backend != 'cupy':
        raise ValueError('device output postprocessing requires a GPU execution target')
    if getattr(backend,'__name__',None) != plan.backend:
        raise ValueError('execution backend differs from saved cfg target')
    if execution_cache is not None and type(execution_cache) is not CfgExecutionCache:
        raise TypeError('execution_cache must be a CfgExecutionCache')
    from lqcdweave.contracts.correlator.precision import require_uniform_contraction_dtype
    for template in plan.templates:
        require_uniform_contraction_dtype(template.plan.state.problem.specs)
        validate_coordinate_batch_workspace(template.plan, backend)
    if device_memory_limit is None and plan.backend == 'cupy':
        device_memory_limit=min(t.plan.memory_limit for t in plan.templates)
    with execution_debug(debug), device_execution(backend,device_memory_limit=device_memory_limit):
        options=dict(host_memory_limit=host_memory_limit,device_memory_limit=device_memory_limit,
            prefetch_host_bytes=prefetch_host_bytes,prefetch_device_bytes=prefetch_device_bytes,report=report,
            device_transform=device_transform,execution_cache=execution_cache)
        if active_output_queue() is not None:
            # Callback/window facades already own the same two-slot queue.
            if transform is not None:
                raise ValueError('a callback consumer already owns result transformation')
            yield from _iter_execute_cfg_plan(plan,configurations,loader_factory,**options)
        else:
            yield from _iter_host_results(plan,configurations,loader_factory,options,transform,report)


def _iter_host_results(plan,configurations,loader_factory,options,transform,report):
    """Bound direct iteration with the same consumer used by callback replay."""
    completed=deque()
    def consume(cfg,index,values):
        if index is not None and transform is not None:
            values=transform(cfg,index,values)
        completed.append((cfg,index,values))
    with OutputConsumerQueue(consume, device_transform=options.get("device_transform")) as output_queue:
        events=_iter_execute_cfg_plan(plan,configurations,loader_factory,**options)
        buffered=0
        try:
            for cfg,index,values in events:
                output_queue.submit(cfg,index,values)
                values=None
                buffered+=1
                if not output_queue.reserve or index is None:
                    output_queue.flush()
                    while completed:
                        item=completed.popleft();buffered-=1
                        with output_queue.suspended():
                            yield item
                        item=None
                elif buffered>=2:
                    if not completed:
                        output_queue.wait_one()
                    item=completed.popleft();buffered-=1
                    with output_queue.suspended():
                        yield item
                    item=None
        finally:
            primary=sys.exc_info()[1]
            _close_many((lambda:_close_cfg_events(events,primary),),primary)
            completed.clear()
            if report is not None:
                report.update(output_queue.report())


def _iter_execute_cfg_plan(plan,configurations,loader_factory,*,host_memory_limit=None,
                          device_memory_limit=None,prefetch_host_bytes=None,
                          prefetch_device_bytes=None,report=None,_owner=None,device_transform=None,
                          execution_cache=None):
    """Execute a lazy cfg iterator with a persistent worker-local IO queue.

    factory(cfg) is an explicit context manager yielding a described SourceLoader.
    consume(cfg,point_index,values) consumes one selected output; point_index=None
    marks cfg completion, so W can publish only a fully consumed cfg.
    CPU wall time is returned separately from the saved modeled costs.
    """
    if host_memory_limit is not None and host_memory_limit<plan.peak_host_bytes:
        raise MemoryError('allocated worker memory is below the saved cfg envelope')
    capacity=plan.prefetch_host_bytes
    if host_memory_limit is not None: capacity=min(capacity,host_memory_limit-plan.peak_host_bytes)
    if prefetch_host_bytes is not None:
        if type(prefetch_host_bytes) is not int or prefetch_host_bytes<0:
            raise ValueError('prefetch capacity must be nonnegative integer bytes')
        capacity=min(capacity,prefetch_host_bytes)
    device_capacity=plan.prefetch_device_bytes
    if device_memory_limit is not None:
        if device_memory_limit<plan.peak_device_bytes:
            raise MemoryError('allocated device memory is below the saved cfg envelope')
        device_capacity=min(device_capacity,device_memory_limit-plan.peak_device_bytes)
    if prefetch_device_bytes is not None:
        if type(prefetch_device_bytes) is not int or prefetch_device_bytes<0:
            raise ValueError('device prefetch capacity must be nonnegative integer bytes')
        device_capacity=min(device_capacity,prefetch_device_bytes)
    device=active_device_execution()
    backend=np if device is None else device.backend
    output_queue=active_output_queue()
    output_reserve=(output_queue_reserve(plan,device,host_memory_limit,device_memory_limit)
                    if output_queue is not None else 0)
    capacity=max(0,capacity-output_reserve)
    device_capacity=max(0,device_capacity-output_reserve)
    workspace = None if device is None else getattr(device, 'reduction_workspace', None)
    if workspace is not None:
        if _owner is None:
            reserve=device.reserve_reduction_workspace(plan.peak_device_bytes,
                limit=device_memory_limit,output_reserve=output_reserve)
            device_capacity=max(0,min(device_capacity,
                device.limit-plan.peak_device_bytes-device.constant_reserve-output_reserve)-reserve)
        else:
            # Streaming windows may admit a larger future peak while their
            # shared queue is already running. Keep the original allocation
            # until that whole trajectory can reserve one common scratch bound.
            workspace.configure(0)
    owner=(_ContinuousQueue(plan,configurations,loader_factory,capacity,device_capacity,device)
           if _owner is None else _owner)
    start=perf_counter(); completed=[]; peak_queue=0; replay_seconds=0.0
    synchronous_reads=0;synchronous_read_seconds=0.0
    source_materialization_calls=0;source_materialization_call_seconds=0.0
    output_download_seconds=0.0
    packed_statistics={}
    # Decode each distinct saved action mask once. Most coordinates differ in
    # data IDs, while their selected action/reuse pattern is identical.
    # Retain one current ID dictionary, plus one bounded key-only template.
    # Historical point IDs stay in the archive's packed tables, not Python dicts.
    preparation_start=perf_counter()
    table_cache=_CfgTableCache(backend)
    states,execution_tables=(_bind_cfg_tables(plan,backend,table_cache) if execution_cache is None
                            else execution_cache._bind(plan,backend,table_cache))
    bound_plan=plan
    preparation_seconds=perf_counter()-preparation_start
    try:
        while owner.claimed or owner.claim():
            cfg,loader,scope,prefix,cfg_offset=owner.claimed[0]
            plan=owner.plan_for(prefix)
            coordinate_offset=owner.coordinate_offset(prefix)
            if output_queue is not None:
                output_queue.configure(device,output_queue_reserve(
                    plan,device,host_memory_limit,device_memory_limit))
            arrays={}
            raw={}
            opened=set()
            execution=None
            finished=False
            template_preparation_bindings={}
            packed_cache=PackedValueCache(
                (plan.statistics.get('packed_value_cache_device_bytes', 0)
                 if device is not None and not debug_enabled()
                 and all(prep.recipe is not None for prep in plan.preparations.values()) else 0),
                array_type=getattr(backend,'ndarray',None))
            try:
                if plan is not bound_plan:
                    preparation_start=perf_counter()
                    states,execution_tables=_bind_cfg_tables(plan,backend,table_cache)
                    preparation_seconds+=perf_counter()-preparation_start
                    bound_plan=plan
                owner.advance(cfg_offset-1)
                for point_index,instance in enumerate(plan.instances):
                    if output_queue is not None:
                        output_queue.acquire()
                    template=plan.templates[instance.template]
                    binding=_bind_node_ids(template.keys,instance.node_ids,table_cache.node_key_template)
                    if instance.template not in template_preparation_bindings:
                        bind_template=getattr(loader,'bind_template_preparations',None)
                        template_preparation_bindings[instance.template]=(
                            bind_template(template) if callable(bind_template) else None)
                    state=states[instance.template,instance.reused.data]
                    local_preps={key:plan.preparations[str(binding[key])]
                                 for key in template.plan.preparations}
                    tables=table_cache.instance_tables[point_index]
                    if tables is None or debug_enabled():
                        tables=execution_tables[((instance.template,instance.reused.data),
                            tuple(batch.action_indices for batch in instance.kernel_batches),
                            tuple((event.action_index,event.kind) for event in instance.events),
                            _tensor_release_pattern(instance))]
                    local_plan=replace(template.plan,state=state,blocks=plan.blocks,
                        preparations=local_preps,events=instance.events,
                        retained_before=(),retained_after=(),retained_releases=(),
                        kernel_batches=instance.kernel_batches,
                        scalar_reduction_groups=tables.scalar_groups,
                        cartesian_reduction_groups=tables.cartesian_groups)
                    local_loader=_LocalLoader(
                        loader,binding,template_preparation_bindings[instance.template],
                        preparation_keys=(None if tables.program is None else tables.program.preparation_keys))
                    execution=_CfgBlockExecution(local_plan,local_loader,backend,shared=(raw,opened),
                        tensor_releases=instance.tensor_releases, arrays=arrays,
                        io_overlap=False,task_read_ahead=_QueueView(owner,prefix),
                        action_offset=cfg_offset+instance.action_offset,
                        owned_preparations=local_loader.materializes_owned,
                        execution_tables=tables,packed_value_cache=packed_cache)
                    replay_start=perf_counter()
                    values=replay(state,local_loader,backend=backend,source_execution=execution,
                                  shared_arrays=_NodeView(binding,arrays),
                                  _owner_snapshot=(device is not None and (device_transform is None
                                      or getattr(device_transform,'_accepts_owner_snapshots',False))))
                    source_materialization_calls+=local_loader.source_materialization_calls
                    source_materialization_call_seconds+=local_loader.source_materialization_call_seconds
                    if device_transform is not None:
                        if device is None:
                            raise ValueError("device output postprocessing requires a GPU execution target")
                        values=device_transform(cfg,coordinate_offset+point_index,values,template)
                    if device is not None and len(values):
                        download_start=perf_counter()
                        try:
                            values=(device.start_download(values)
                                    if output_queue is not None and output_queue.reserve else
                                    device.download(values))
                        finally:
                            output_download_seconds+=perf_counter()-download_start
                    replay_seconds+=perf_counter()-replay_start
                    synchronous_reads+=local_loader.reads
                    synchronous_read_seconds+=local_loader.read_seconds
                    yield cfg,coordinate_offset+point_index,values
                    values=None
                    for gid in instance.releases: del arrays[gid]
                    peak_queue=max(peak_queue,owner.queue.reserved_bytes)
                    execution.close(); execution=None
                if arrays or raw: raise RuntimeError('complete cfg replay left live arrays')
                if output_queue is not None:
                    output_queue.flush()
                completed.append(cfg)
                finished=True
                yield cfg,None,None
            finally:
                primary=sys.exc_info()[1]
                callbacks=[]
                # Always release numerical copies, including consumer/kernel errors.
                packed_cache.clear()
                for key,value in packed_cache.report().items():
                    packed_statistics[key]=(max(packed_statistics.get(key,0),value)
                        if key.endswith(('capacity_bytes','peak_device_bytes'))
                        else packed_statistics.get(key,0)+value)
                if output_queue is not None:
                    callbacks.append(lambda:output_queue.flush(cancel=primary is not None))
                if not finished: callbacks.append(owner.queue.close)
                if execution is not None: callbacks.append(execution.close)
                callbacks.extend(lambda key=key,value=value:loader.close_block(key,value)
                                 for key,value in tuple(raw.items()))
                owner.claimed.popleft()
                callbacks.append(scope.close)
                _close_many(callbacks,primary)
                owner.release_plan(prefix)
                raw.clear(); arrays.clear()
                # The queue owns future reads. This cfg has consumed every
                # saved read before its reader scope is released.
    except BaseException as primary:
        _close_many((owner.close,),primary)
        raise
    else:
        owner.close()
    result={'completed':tuple(completed),'execution_seconds':perf_counter()-start,
            'numerical_replay_seconds':replay_seconds,
            'descriptor_preparation_seconds':preparation_seconds,
            'compiled_template_variants':table_cache.compiled_states,
            'compiled_execution_tables':table_cache.compiled_tables,
            'execution_table_cache_hits':table_cache.table_hits,
            'execution_table_plan_reuses':table_cache.plan_reuses,
            'source_materialization_calls':source_materialization_calls,
            'source_materialization_call_seconds':source_materialization_call_seconds,
            'output_download_seconds':output_download_seconds,
            'prefetch_capacity_bytes':owner.queue.capacity,'sampled_queue_bytes':peak_queue,
            'peak_prefetch_reserved_bytes':owner.queue.peak_reserved_bytes,
            'prefetch_device_capacity_bytes':owner.queue.device_capacity,
            'peak_prefetch_device_bytes':owner.queue.peak_device_bytes,
            'peak_retired_input_bytes':owner.queue.peak_retired_input_bytes,
            'retired_input_count':owner.queue.retired_input_count,
            'prefetched_reads':owner.queue.completed_reads,
            'prefetch_pending_read_limit':owner.queue.peak_pending_read_limit,
            'peak_pending_input_reads':owner.queue.peak_pending_reads,
            'background_read_seconds':owner.queue.read_seconds,
            'prefetch_wait_seconds':owner.queue.wait_seconds,
            'synchronous_reads':synchronous_reads,
            'synchronous_read_seconds':synchronous_read_seconds}
    if device is not None:
        if workspace is not None:
            device.statistics.update(workspace.report())
        result['device']=dict(device.statistics)
    result.update(packed_statistics)
    if report is not None: report.update(result)
    return result


class _CfgTableCache:
    """Execution-local callable views, bounded by consecutive window variants.

    Entries hold immutable template descriptions only, never readers, blocks,
    cfg bindings or numerical arrays. Full template equality also supports
    separately decoded windows; a template index alone is not an identity.
    A new window retains only the program variants it actually uses. Numerical
    kernels remain with an equal template across mask/event changes, bounded
    by its dense equation/spec/layout combinations, core Actions and saved
    batches/children. A changing
    stream cannot accumulate an unbounded historical plan cache.
    """
    def __init__(self,backend):
        self.backend=backend
        self.entries=[]
        self.compiled_states=0
        self.compiled_tables=0
        self.table_hits=0
        self.plan_reuses=0
        self.instance_tables=()
        self.node_key_template=None


class CfgExecutionCache(ExecutionBindingCache):
    """One immutable plan/backend's host dispatch, without runtime input owners.

    CompleteCalculation owns this process-local cache. A replaced plan or backend
    replaces its sole entry; debug and extension plans always bind afresh. The
    per-run cache and all loader, residency, queue and result owners stay local.
    Concurrent runs can share read-only call tables; publication is one complete
    tuple, so concurrent first binds may duplicate work but cannot mix tables.
    """
    def _bind(self,plan,backend,cache):
        with self.factory_scope():
            return self._bind_configured(plan,backend,cache)

    def _bind_configured(self,plan,backend,cache):
        reusable=(type(plan) is CfgPlan and plan._binding_validation is not None
                  and not debug_enabled())
        previous=self._lookup(plan,backend) if reusable else None
        if previous is not None:
            states,tables,instances,entries,node_keys=previous
            cache.instance_tables=instances
            cache.entries=entries
            cache.node_key_template=node_keys
            cache.table_hits=len(tables)
            cache.plan_reuses=1
            return states,tables
        states,tables=_bind_cfg_tables(plan,backend,cache)
        if (reusable and all(table is not None for table in cache.instance_tables)
                and all(type(action) is Action for template in plan.templates
                        for action in template.plan.state.actions)):
            self._publish(plan,backend,states,tables,cache.instance_tables,cache.entries,
                          cache.node_key_template)
        return states,tables


def _native_reuse_flags(actions):
    """Describe native reuse/zero-cost bits once, without retaining Actions."""
    if type(actions) is not tuple:
        return None
    flags=bytearray(len(actions))
    for index,action in enumerate(actions):
        if (type(action) is not Action or type(action.kind) is not str or type(action.reused) is not bool
                or type(action.seconds) not in (int,float,bool)):
            return None
        flags[index]=action.reused | ((action.seconds!=0.0)<<1)
    flags=np.frombuffer(flags,dtype=np.uint8)
    return flags&1!=0,flags&2!=0


def _batch_action_indices(batch,groups):
    """Expand each immutable native GEMM group once per current binding."""
    if groups is None or type(batch) is not DenseGemmGroup:
        return batch.action_indices
    key=id(batch)
    entry=groups.get(key)
    if entry is None:
        entry=groups[key]=(batch,batch.action_indices)
    return entry[1]


def _active_scalar_groups(state, batches, groups):
    """Filter saved groups against this variant's actual publication order.

    Partial dense/GEMM reuse can move an input's physical birth back to its
    logical Action. Scalar members being paid does not alone prove that all
    their inputs are still ready at the saved group start. No group is rebuilt.
    """
    if not groups:
        return ()
    starts = {group.action_indices[0]: group for group in groups
              if all(not state.actions[index].reused for index in group.action_indices)}
    if not starts:
        return ()
    dense = {batch.action_indices[0]: batch for batch in batches}
    live, selected = set(), []
    for index, action in enumerate(state.actions):
        batch = dense.get(index)
        if batch is not None:
            live.update(result.tensor for result in batch.results)
        group = starts.get(index)
        if group is not None and all(key in live for member in group.action_indices
                                     for key in state.actions[member].operands):
            selected.append(group)
            live.update(state.actions[member].tensor for member in group.action_indices)
        if action.kind == 'release':
            live.discard(action.tensor)
        elif action.kind in ('load', 'contract', 'sparse_bridge', 'baryon_join'):
            # A reused value is declared at its saved handoff position only.
            live.add(action.tensor)
    return tuple(selected)


def _active_cartesian_groups(state, batches, groups):
    """Keep whole saved groups only when original actual batches are ready.

    No group-created early result participates in this baseline scan. Partial
    reuse and dependency-driven child removal therefore cannot cascade into a
    falsely ready downstream group. Selection itself belongs to P, never X.
    """
    if not groups:
        return ()
    actual = {batch.action_indices:batch for batch in batches if type(batch) is DenseKernelBatch}
    candidates = {}
    for group in groups:
        if (all(child in actual for child in group.batch_action_indices)
                and all(not state.actions[index].reused for child in group.batch_action_indices for index in child)):
            candidates[group.batch_action_indices[0][0]] = group
    if not candidates:
        return ()
    dense = {batch.action_indices[0]:batch for batch in batches}
    live, selected = set(), []
    for index, action in enumerate(state.actions):
        group = candidates.get(index)
        if group is not None and all(operand.tensor in live for child in group.batch_action_indices
                for side in (actual[child].left,actual[child].right) for operand in side):
            selected.append(group)
        batch = dense.get(index)
        if batch is not None:
            live.update(result.tensor for result in batch.results)
        if action.kind == 'release':
            live.discard(action.tensor)
        elif action.kind in ('load','contract','sparse_bridge','baryon_join'):
            live.add(action.tensor)
    return tuple(selected)


def _bind_cfg_tables(plan,backend,cache=None):
    cache=_CfgTableCache(backend) if cache is None else cache
    if cache.backend is not backend:
        raise ValueError('execution table cache belongs to another backend')
    cache.node_key_template=_node_key_template(plan,cache.node_key_template)
    states={};execution_tables={};entries=[];template_entries={};instance_tables=[]
    direct_instances=not debug_enabled()
    pending_masks={};reuse_flags={};batch_kernels={}
    group_indices={} if direct_instances else None
    if direct_instances and type(plan) is CfgPlan:
        for instance in plan.instances:
            if (type(instance) is CfgInstance and type(instance.reused) is ArrayConstant
                    and type(instance.reused.data) is str):
                pending_masks.setdefault(instance.template,set()).add(instance.reused.data)
    for instance in plan.instances:
        if instance.template not in template_entries:
            template=plan.templates[instance.template]
            # C freezes state/actions/specs and complete batch records.
            # SourceSpec also copies JSON metadata into nested tuples/proxies;
            # callers replace descriptions to change their semantics.
            # Include all captured state, rather than shape alone or a hash
            # that could confuse different sparse/core or layout semantics.
            signature=(template.keys,template.plan.state,template.plan.kernel_batches,
                       template.plan.scalar_reduction_groups,template.plan.cartesian_reduction_groups)
            previous=next((entry for entry in cache.entries if entry[0]==signature),None)
            entry=(signature,{}, {}, {} if previous is None else previous[3])
            entries.append(entry)
            template_entries[instance.template]=(entry,previous)
            batch_kernels[instance.template]={} if direct_instances else None
            pending=pending_masks.get(instance.template,())
            if previous is not None:
                pending=set(pending).difference(previous[1])
            # A single mask keeps the scalar path. Amortize the native flag
            # snapshot only when at least three new variants need it.
            reuse_flags[instance.template]=(_native_reuse_flags(template.plan.state.actions)
                if len(pending)>=3 else None)
        entry,previous=template_entries[instance.template]
        key=(instance.template,instance.reused.data)
        state_key=instance.reused.data
        if key not in states:
            state=None if previous is None else previous[1].get(state_key)
            if state is None:
                original=plan.templates[instance.template].plan.state
                mask=instance.reused.array()
                flags=reuse_flags[instance.template]
                if (flags is not None and type(instance) is CfgInstance
                        and type(instance.reused) is ArrayConstant):
                    # CfgPlan has checked the immutable bool mask's extent.
                    # Only changed records need a Python Action visit.
                    changed=np.flatnonzero((mask!=flags[0]) | (mask&flags[1]))
                    state=original
                    if len(changed):
                        actions=list(original.actions)
                        for index in changed:
                            action=actions[index]
                            reused=bool(mask[index])
                            actions[index]=replace(action,reused=reused,
                                seconds=0.0 if reused else action.seconds)
                        state=replace(original,actions=tuple(actions))
                else:
                    actions=[]; changed=False
                    for action,reused in zip(original.actions,mask,strict=True):
                        reused=bool(reused)
                        if action.reused != reused or (reused and action.seconds != 0.0):
                            action=replace(action,reused=reused,
                                           seconds=0.0 if reused else action.seconds)
                            changed=True
                        actions.append(action)
                    state=replace(original,actions=tuple(actions)) if changed else original
                cache.compiled_states+=1
            states[key]=entry[1][state_key]=state
        event_pattern=tuple((event.action_index,event.kind) for event in instance.events)
        release_pattern=_tensor_release_pattern(instance)
        variant=(key,tuple(_batch_action_indices(batch,group_indices)
                           for batch in instance.kernel_batches),event_pattern,release_pattern)
        if variant not in execution_tables:
            # Global block IDs vary with each window. The bound program only
            # captures event positions/kinds; _BlockExecution owns actual IDs.
            table_key=(state_key,instance.kernel_batches,event_pattern,release_pattern)
            tables=None if previous is None else previous[2].get(table_key)
            if tables is None:
                tables=bind_scene_execution(states[key],instance.kernel_batches,backend,
                                             events=instance.events,shared_residency=True,
                                             producer_batches=plan.templates[instance.template].plan.kernel_batches,
                                             tensor_release_indices=release_pattern,
                                             kernel_cache=entry[3],
                                             scalar_reduction_groups=_active_scalar_groups(
                                                 states[key], instance.kernel_batches,
                                                 plan.templates[instance.template].plan.scalar_reduction_groups),
                                             cartesian_reduction_groups=_active_cartesian_groups(
                                                 states[key], instance.kernel_batches,
                                                 plan.templates[instance.template].plan.cartesian_reduction_groups),
                                             _batch_kernels=batch_kernels[instance.template],
                                             _action_indices=(np.flatnonzero(~instance.reused.array())
                                                 if reuse_flags[instance.template] is not None
                                                 and type(instance) is CfgInstance
                                                 and type(instance.reused) is ArrayConstant else None))
                cache.compiled_tables+=1
            else:
                cache.table_hits+=1
            execution_tables[variant]=entry[2][table_key]=tables
        # C freezes native instances and batch records. Keep one reference per
        # current point, rather than rebuilding the same expanded batch/event
        # lookup at replay. Extensions and debug retain the original lookup.
        instance_tables.append(execution_tables[variant] if direct_instances
            and type(instance) is CfgInstance
            and type(instance.template) is int
            and type(instance.reused) is ArrayConstant and type(instance.reused.data) is str
            and type(instance.events) is tuple
            and all(type(event) is StorageEvent and type(event.action_index) is int
                    and type(event.kind) is str for event in instance.events)
            and all(type(batch) in (DenseKernelBatch,DenseGemmGroup)
                    for batch in instance.kernel_batches) else None)
    cache.entries=entries
    cache.instance_tables=tuple(instance_tables)
    return states,execution_tables


def execute_cfg_plan(plan,configurations,loader_factory,consume,**options):
    """Callback facade over the same persistent lazy replay engine."""
    report={}
    with OutputConsumerQueue(consume, device_transform=options.get("device_transform")) as output_queue:
        events=iter_execute_cfg_plan(plan,configurations,loader_factory,report=report,**options)
        try:
            for cfg,index,values in events:
                output_queue.submit(cfg,index,values)
        finally:
            primary=sys.exc_info()[1]
            _close_many((lambda:_close_cfg_events(events,primary),),primary)
    report.update(output_queue.report())
    return report


__all__=['CfgExecutionCache','execute_cfg_plan','iter_execute_cfg_plan']
