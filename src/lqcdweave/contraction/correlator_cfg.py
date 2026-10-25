"""Complete cfg planning/loading front end; execution binds only reader handles."""
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass,field,replace
from time import perf_counter
import numpy as np
from lqcdweave.contracts.correlator.api import CoordinatePoint,CoordinateSpec,CfgOutputField,CfgPlan
from lqcdweave.contraction.expansion.api import (
    describe_input_reads,describe_input_preparations,restore_input_loader,
    describe_coordinate_reads,
    unknown_reader_aliases,bound_source_identities,
    input_readers,input_sampling_signature,
)
from lqcdweave.contraction.planning.api import build_cfg_plan
from lqcdweave.contraction.execution.api import CfgExecutionCache,execute_cfg_plan,iter_execute_cfg_plan
from lqcdweave.contraction.results.api import describe_output_fields,freeze_output_recipe,OutputAssemblyCache
from lqcdweave.storage.contraction_plans.api import load_cfg_plan,save_cfg_plan
from lqcdweave.contraction.calculation import Calculation


def is_cfg_iterator(configuration):
    return (configuration is not None and not isinstance(configuration,(str,bytes))
            and hasattr(configuration,'__iter__'))


@dataclass(frozen=True)
class CfgSources:
    """Explicit runtime reader binding and physical time coordinates."""
    readers: Mapping
    physical_times: tuple[int,...] = ()
    physical_lt: int | None = None
    provenance: Mapping = field(default_factory=dict)
    parents: tuple[str,...] = ()
    lattice: tuple[int,int,int,int] | None = None
    ensemble: str | None = None


# Compatibility import and constructor; public results have exactly one type.
CompleteCalculation = Calculation


def _run_complete_calculation(self,*,backend=None,measure=False,device_memory_limit=None,debug=None):
    from lqcdweave.contracts.correlator.precision import require_uniform_contraction_dtype
    for template in self.plan.templates:
        require_uniform_contraction_dtype(template.plan.state.problem.specs)
    from lqcdweave.adapters.execution_target import check_execution_target
    from lqcdweave.adapters.hardware_budget import resolve_diagram_memory_limits
    backend=check_execution_target(self.plan.execution_target,backend or self.plan.backend)
    debug=self.options.get('debug',False) if debug is None else debug
    if type(debug) is not bool: raise TypeError('debug must be bool')
    if measure:
        raise ValueError('complete cfg replay reports wall time; calibration uses the ordinary measured route')
    if backend.__name__=='numpy' and device_memory_limit is not None:
        raise ValueError('device_memory_limit requires a GPU backend')
    host_cap=self.options.get('host_memory_limit')
    if host_cap is None: host_cap=_host_cap(self.plan)
    saved_device=min(t.plan.memory_limit for t in self.plan.templates) if self.plan.backend=='cupy' else None
    requested_device=(self.options.get('device_memory_limit')
                      if device_memory_limit is None else device_memory_limit)
    if self.plan.backend=='numpy':
        # The archived calculate(memory_limit=...) spelling remains a host cap on CPU.
        if requested_device is not None: host_cap=min(host_cap,requested_device)
        device_cap=None
    else:
        device_cap=saved_device if requested_device is None else min(saved_device,requested_device)
    limits=resolve_diagram_memory_limits(backend=backend,
        memory_limit=device_cap if self.plan.backend=='cupy' else host_cap,
        host_memory_limit=host_cap,execution_mode='local')
    if limits.host_bytes<self.plan.peak_host_bytes:
        raise MemoryError('worker host memory is below the saved cfg requirement')
    if limits.device_bytes is not None and limits.device_bytes<self.plan.peak_device_bytes:
        raise MemoryError('worker device memory is below the saved cfg requirement')
    calculation=replace(self,options=dict(self.options,host_memory_limit=limits.host_bytes,
        device_memory_limit=limits.device_bytes,backend=backend,debug=debug))
    if self.plan.output_mode!='memory':
        from lqcdweave.contraction.correlator_output import run_dataset_calculation
        return run_dataset_calculation(calculation)
    cfg=self.options.get('configuration')
    if is_cfg_iterator(cfg):
        report={}
        from lqcdweave.contraction.calculation import _CfgValueStream
        return replace(calculation,value=_CfgValueStream(iter_memory_calculation(calculation,report),report),report=report)
    factory=source_loader_factory(calculation)
    assemblers=bind_point_assemblers(calculation)
    if self.plan.output_coordinates:
        results=[None]*len(self.plan.coordinates)
        def consume(configuration,index,values):
            if index is not None:
                for position,value in assemble_instance(self.plan,assemblers,index,values):
                    results[position]=value
    else:
        results=[]
        def consume(configuration,index,values):
            if index is not None:
                results.append(assemblers[self.plan.instances[index].template](values))
    report=execute_cfg_plan(self.plan,(cfg,),factory,consume,**execution_options(calculation))
    value=stack_point_results(results,self.plan.coordinate_shape)
    return replace(calculation,value=value,report=report)



def _host_cap(plan):
    caps=[t.plan.host_memory_limit for t in plan.templates if t.plan.host_memory_limit is not None]
    if plan.backend=='numpy': caps.extend(t.plan.memory_limit for t in plan.templates)
    if not caps: raise ValueError('complete cfg requires a finite host memory budget')
    return min(caps)


def execution_options(calculation):
    return dict(backend=calculation.options.get('backend',np),
        host_memory_limit=calculation.options.get('host_memory_limit'),
        device_memory_limit=calculation.options.get('device_memory_limit'),
        prefetch_host_bytes=calculation.options.get('prefetch_host_bytes'),
        prefetch_device_bytes=calculation.options.get('prefetch_device_bytes'),
        debug=calculation.options.get('debug',False),execution_cache=calculation._execution_bindings)


def bind_point_assemblers(calculation):
    """Reuse only immutable-plan programs; each evaluation owns its values."""
    debug=calculation.options.get('debug',False)
    cache=calculation._output_bindings if type(calculation.plan) is CfgPlan else OutputAssemblyCache()
    return cache.bind(calculation.plan,tuple(template.output for template in calculation.plan.templates),debug=debug)


def assemble_instance(plan,assemblers,index,values):
    """Recover original coordinates from a saved instance or coordinate batch."""
    instance=plan.instances[index]
    assemble=assemblers[instance.template]
    if not instance.point_indices:
        yield index,assemble(values)
        return
    axes=plan.templates[instance.template].output_axes
    from lqcdweave.contraction.correlator_batch_output import _owned
    for lane,position in enumerate(instance.point_indices):
        selected=tuple(value if axis is None else value[(slice(None),)*axis+(lane,)]
                       for value,axis in zip(values,axes,strict=True))
        yield position,_owned(assemble(selected))


def stack_point_results(values,shape):
    first=values[0]
    if not shape: return first
    if isinstance(first,dict):
        return {k:stack_point_results([v[k] for v in values],shape) for k in first}
    if isinstance(first,(list,tuple)):
        items=[stack_point_results([v[i] for v in values],shape) for i in range(len(first))]
        return tuple(items) if isinstance(first,tuple) else items
    if isinstance(first,np.ndarray) and first.dtype.kind=='O':
        result=np.empty(first.shape,dtype=object)
        for index in np.ndindex(first.shape):
            result[index]=stack_point_results([value[index] for value in values],shape)
        return result
    array=np.stack(values,axis=-1)
    return array.reshape((*array.shape[:-1],*shape))


def check_coordinate_sampling(expanded, coordinates, readers=None):
    """A fixed Wick graph cannot silently change pools or inclusion laws."""
    sampled = {key: primary for key, primary in expanded.inputs.read_requests.items()
               if any(request.family == 'elemental' and len(request.arguments) == 4
                      and any(kind in request.arguments[-1]
                      for kind in ('p', 'd')) for request in (primary, *primary.aliases))}
    if not sampled:
        return
    reference = input_readers(expanded.inputs)
    expected = input_sampling_signature(sampled, reference)
    current = reference if readers is None else readers
    for point in coordinates.points:
        requests = {key: request.at(dict(point.values)) for key, request in sampled.items()}
        if input_sampling_signature(requests, current) != expected:
            raise ValueError('coordinate changes sampling pools or inclusion laws; select a compatible template')


def complete_calculation(calculation,*,coordinates=None,output_mode='memory',output=None,
                         configuration=None,source_factory=None,readers=None,physical_lt=None,output_definition=None,
                         time_roles=('0','1'),dt_fields=('two_point','connected','disconnected'),
                         output_layout='pairs_and_dt',dt_coordinates=None,postprocess_backend='cpu',
                         coordinate_batches=None):
    """Complete an already selected path for cfg replay without another search."""
    start=perf_counter()
    from lqcdweave.contraction.correlator_postprocessing import validate_postprocess_backend, reserve_gpu_postprocessing
    target_backend=(calculation.plan.backend if isinstance(calculation.plan,CfgPlan)
                    else calculation.plan.state.problem.backend if calculation.plan is not None else 'numpy')
    validate_postprocess_backend(postprocess_backend,target_backend,output_mode=output_mode)
    if isinstance(calculation.plan,CfgPlan):
        if postprocess_backend=='gpu' or calculation.plan.statistics.get('postprocess_backend')=='gpu':
            raise ValueError('GPU output layout must be completed from the original selected template')
        if output_mode!=calculation.plan.output_mode:
            raise ValueError('complete cfg buffering mode is fixed; select it during initial planning')
        completed=configure_complete_output(calculation,physical_lt=physical_lt,time_roles=time_roles,
            dt_fields=dt_fields,output_definition=output_definition,output=output,
            source_factory=source_factory,readers=readers,configuration=configuration)
        from lqcdweave.contraction.correlator_dt_output import configure_packed_dt
        return configure_packed_dt(completed,output_layout,dt_coordinates)
    if calculation.path.batch_paths:
        raise ValueError('complete a fixed-coordinate template before expanding coordinate batches')
    if calculation.plan is None: raise ValueError('complete cfg delivery requires a selected path')
    expanded=calculation.expanded
    coordinates=expanded.coordinates if coordinates is None else coordinates
    check_coordinate_sampling(expanded, coordinates)
    current_readers=input_readers(expanded.inputs)
    for point in coordinates.points:
        symbols = tuple(axis for axis, _ in point.values)
        if len({str(axis) for axis in symbols}) != len(symbols):
            raise ValueError('complete plans require distinct time keys after archive role encoding')
    preparations,specs=describe_input_preparations(expanded.inputs)
    if len(expanded.coordinates.points)!=1:
        raise ValueError('complete cfg delivery requires a single-coordinate selected template')
    point_coordinates=CoordinateSpec('bound',(),(),(),expanded.coordinates.points,())
    output_recipe=freeze_output_recipe(replace(expanded.output_recipe,coordinates=point_coordinates))
    selected=calculation.plan
    postprocess_statistics={}
    if postprocess_backend=='gpu':
        from lqcdweave.contraction.correlator_window_output import describe_window_output
        layout=describe_window_output(output_recipe,selected.state.problem,preparations,specs,coordinates,
            physical_lt=physical_lt,time_roles=time_roles,dt_fields=dt_fields,
            output_layout=output_layout,dt_coordinates=dt_coordinates)
        selected,postprocess_statistics=reserve_gpu_postprocessing(selected,output_recipe,layout)
    common=dict(output_mode=output_mode, reader_aliases=unknown_reader_aliases(expanded.inputs),
        execution_target=calculation.options.get('execution_target'),
        target_summary=calculation.options.get('execution_target_summary'),
        debug=calculation.options.get('debug',False))
    if coordinate_batches is None:
        plan=build_cfg_plan(selected,output_recipe,coordinates.points,
            describe_input_reads(expanded.inputs),preparations,specs,
            read_metadata=describe_coordinate_reads(expanded.inputs,coordinates.points),**common)
    else:
        if postprocess_backend != 'cpu':
            raise ValueError('coordinate-batched output currently uses CPU assembly')
        from lqcdweave.contraction.planning.api import build_coordinate_cfg_plan
        batches=[]
        for descriptor,child_plan in zip(coordinate_batches.batches,coordinate_batches.plans,strict=True):
            inputs=descriptor.inputs
            batches.append(dict(plan=child_plan,preparations=descriptor.preparation_closure,
                specs=descriptor.specs,requests=descriptor.requests,
                point_indices=inputs.coordinate_batch.point_indices,output_axes=inputs.batch_output_axes))
        plan=build_coordinate_cfg_plan(batches,output_recipe,coordinates,**common)
    fields=[]; words=dt_words=0
    for path,shape,dtype in describe_output_fields(output_recipe,calculation.plan.state.problem):
        count=(int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize+7)//8
        dt_offset=dt_words if physical_lt is not None and path and path[-1] in dt_fields else -1
        fields.append(CfgOutputField(path,shape,dtype,words,count,dt_offset))
        words+=count
        if dt_offset>=0: dt_words+=count
    if physical_lt is not None and (type(physical_lt) is not int or physical_lt<1):
        raise ValueError('physical Lt must be a positive integer')
    # Packing copies and finite checks coexist with the assembled point. dt
    # sums are preallocated for all bins; absent bins have no published payload.
    def packed_shape(size):
        columns=min(size,4096)
        return ((size+columns-1)//columns,columns) if columns else (0,0)
    output_shape,dt_shape=packed_shape(words),packed_shape(dt_words)
    largest_bytes=max((int(np.prod(f.shape,dtype=object))*np.dtype(f.dtype).itemsize for f in fields),default=0)
    # Only floating/complex fields can contain non-finite values. The packed
    # int64 payload needs no finite-value scan or boolean scratch reservation.
    finite_bytes=max((int(np.prod(f.shape,dtype=object)) for f in fields
                      if np.dtype(f.dtype).kind in 'fc'),default=0)
    packing=int(np.prod(output_shape))*8+finite_bytes+largest_bytes
    dt_buffer=sum(int(np.prod(f.shape,dtype=object))*np.dtype(f.dtype).itemsize
                  for f in fields if f.dt_offset>=0)*(physical_lt or 0)
    extra=(packing+dt_buffer+int(np.prod(dt_shape))*8+(physical_lt or 0)*8) if output_mode=='dataset' else 0
    peak=plan.peak_host_bytes+extra
    cap=_host_cap(plan)
    if peak>cap:
        raise MemoryError(f'complete cfg including output buffers requires {peak} host bytes; cap is {cap}')
    from lqcdweave.contracts.correlator.api import json_digest
    from lqcdweave.storage.contraction_plans.api import encode_plan_value
    definition=dict(output_definition or {})
    definition['input_sampling_signature']=input_sampling_signature(plan.requests,current_readers)
    from lqcdweave.contraction.correlator_result_metadata import operator_coordinates
    definition['frontend_operator_coordinates'] = encode_plan_value(operator_coordinates(expanded.correlator))
    definition['frontend_coordinates'] = dict(
        mode=coordinates.mode, axes=[[axis, list(values)] for axis, values in coordinates.axes],
        scan_axes=list(coordinates.scan_axes), scan_shape=list(coordinates.scan_shape),
        points=[list(point.values) for point in coordinates.points],
    )
    definition['task_signature']=json_digest(encode_plan_value(dict(
        output=output_recipe.root,preparations=preparations,specs=specs)))
    plan=replace(plan,output_definition=definition,coordinate_shape=tuple(coordinates.scan_shape),output_fields=tuple(fields),
        output_words=words,dt_words=dt_words,physical_lt=physical_lt,time_roles=tuple(map(str,time_roles)),
        output_storage_shape=output_shape,dt_storage_shape=dt_shape,
        peak_host_bytes=peak,output_host_bytes=plan.output_host_bytes+extra,
        prefetch_host_bytes=min(plan.prefetch_host_bytes,cap-peak),
        statistics=dict(plan.statistics,**postprocess_statistics,peak_host_bytes=peak,
            kernel_batching=calculation.prediction.get('kernel_batching',
                calculation.options.get('kernel_batching',True)),
            output_host_bytes=plan.output_host_bytes+extra,
            prefetch_host_bytes=min(plan.prefetch_host_bytes,cap-peak)))
    options=dict(calculation.options,configuration=(expanded.bindings.configuration
        if configuration is None else configuration),source_factory=source_factory,
        readers=readers,output=output,planning_seconds=calculation.planning_seconds+perf_counter()-start)
    completed=CompleteCalculation(plan,options=options)
    from lqcdweave.contraction.correlator_dt_output import configure_packed_dt
    return configure_packed_dt(completed,output_layout,dt_coordinates)


def configure_complete_output(calculation,*,physical_lt,time_roles,dt_fields,output_definition,
                              output,source_factory,readers,configuration):
    """Offline extension of an already selected cfg's output arrangement."""
    plan=calculation.plan
    if plan.output_mode!='dataset' or physical_lt is None:
        raise ValueError('complete-plan output completion currently requires dataset dt reconstruction')
    if type(physical_lt) is not int or physical_lt<1: raise ValueError('invalid physical Lt')
    fields=[];words=0
    for f in plan.output_fields:
        offset=words if f.path and f.path[-1] in dt_fields else -1
        fields.append(replace(f,dt_offset=offset))
        if offset>=0: words+=f.words
    columns=min(words,4096);shape=((words+columns-1)//columns,columns) if columns else (0,0)
    def dt_bytes(fields,lt,shape):
        return (lt or 0)*(sum(int(np.prod(f.shape,dtype=object))*np.dtype(f.dtype).itemsize
            for f in fields if f.dt_offset>=0)+8)+int(np.prod(shape))*8
    delta=dt_bytes(fields,physical_lt,shape)-dt_bytes(plan.output_fields,plan.physical_lt,plan.dt_storage_shape)
    peak=plan.peak_host_bytes+delta
    cap=_host_cap(plan)
    if peak>cap: raise MemoryError('dt output buffers exceed complete cfg envelope')
    definition=dict(plan.output_definition);definition.update(output_definition or {})
    plan=replace(plan,output_fields=tuple(fields),dt_words=words,dt_storage_shape=shape,
        physical_lt=physical_lt,time_roles=tuple(map(str,time_roles)),output_definition=definition,
        peak_host_bytes=peak,output_host_bytes=plan.output_host_bytes+delta,
        prefetch_host_bytes=min(plan.prefetch_host_bytes,cap-peak),
        statistics=dict(plan.statistics,peak_host_bytes=peak,
            output_host_bytes=plan.output_host_bytes+delta,
            prefetch_host_bytes=min(plan.prefetch_host_bytes,cap-peak)))
    options=dict(calculation.options)
    for key,value in dict(output=output,source_factory=source_factory,readers=readers,configuration=configuration).items():
        if value is not None: options[key]=value
    return replace(calculation,plan=plan,options=options)


def source_loader_factory(calculation):
    """Bind saved cfgs, or (window, cfg), with one weak convention cache.

    The two-argument form is X's existing window-factory protocol; source
    factories still receive only the actual cfg, as on complete replay.
    """
    factory=calculation.options.get('source_factory')
    readers=calculation.options.get('readers')
    if factory is None and readers is None: raise ValueError('complete cfg replay requires reader bindings')
    convention_cache={}  # Execution-factory lifetime; weak reader references only.
    @contextmanager
    def bind(cfg, *window_cfg):
        plan=calculation.plan
        if window_cfg:
            if len(window_cfg)!=1 or not isinstance(cfg,CfgPlan):
                raise TypeError('window source binding requires (CfgPlan, configuration)')
            plan,cfg=cfg,window_cfg[0]
        if factory is None:
            binding=readers if isinstance(readers,CfgSources) else CfgSources(readers)
            loader=restore_input_loader(plan.requests,binding.readers,
                plan.preparations,plan.specs,configuration=cfg,
                reader_aliases=plan.reader_aliases,debug=calculation.options.get('debug',False),
                convention_cache=convention_cache)
            _check_sampling_binding(plan,binding.readers)
            from lqcdweave.contraction.elemental_preparation import prepare_elemental_sources
            prepare_elemental_sources(plan.requests, binding.readers,
                backend=calculation.options.get('backend', np),
                host_limit=calculation.options.get('host_memory_limit'),
                device_limit=calculation.options.get('device_memory_limit'))
            identities,known=bound_source_identities(loader)
            loader.output_binding=replace(binding,parents=identities,provenance=dict(binding.provenance,
                source_identities_known=known))
            yield loader
        else:
            with factory(cfg) as current:
                binding=current if isinstance(current,CfgSources) else CfgSources(current)
                current=binding.readers
                if not isinstance(current,Mapping):
                    raise TypeError('cfg source factory must yield a reader-role mapping')
                loader=restore_input_loader(plan.requests,current,
                    plan.preparations,plan.specs,configuration=cfg,
                    reader_aliases=plan.reader_aliases,debug=calculation.options.get('debug',False),
                    convention_cache=convention_cache)
                _check_sampling_binding(plan,current)
                from lqcdweave.contraction.elemental_preparation import prepare_elemental_sources
                prepare_elemental_sources(plan.requests, current,
                    backend=calculation.options.get('backend', np),
                    host_limit=calculation.options.get('host_memory_limit'),
                    device_limit=calculation.options.get('device_memory_limit'))
                identities,known=bound_source_identities(loader)
                loader.output_binding=replace(binding,parents=identities,provenance=dict(binding.provenance,
                    source_identities_known=known))
                yield loader
    return bind


def _check_sampling_binding(plan,readers):
    expected=plan.output_definition.get('input_sampling_signature')
    if expected is not None and input_sampling_signature(plan.requests,readers)!=expected:
        raise ValueError('saved sampling pools or inclusion laws differ from current readers')


def load_calculation(archive,*,configuration,source_factory=None,readers=None,
                     output=None,host_memory_limit=None,device_memory_limit=None,debug=False):
    start=perf_counter(); plan=load_cfg_plan(archive)
    if (output is None)!=(plan.output_mode=='memory'):
        raise ValueError('execution output mode differs from the saved buffering arrangement')
    return CompleteCalculation(plan,options=dict(configuration=configuration,
        source_factory=source_factory,readers=readers,output=output,
        host_memory_limit=host_memory_limit,device_memory_limit=device_memory_limit,
        execution_target=plan.execution_target,debug=debug,loading_seconds=perf_counter()-start,
        kernel_batching=plan.statistics.get('kernel_batching',True)))


def archive_readers(plan,propagators,vertices):
    """Explicit logical roles, without reconstructing the operator basis."""
    from lqcdweave.contraction.binding.api import RecipeVertex
    if isinstance(vertices,RecipeVertex):
        vertices=vertices.elemental
    hints=plan.output_definition.get('frontend_reader_keys',{})
    if hints and isinstance(vertices,Mapping):
        mapped=dict(vertices)
        for role,key in hints.items():
            if role not in mapped and key in vertices:
                vertex=vertices[key]
                reader=getattr(vertex,'reader',None)
                mapped[role]=reader if reader is not None else getattr(vertex,'elemental',vertex)
        vertices=mapped
    return readers_for_requests(plan.requests, propagators, vertices)


def readers_for_requests(requests, propagators, vertices):
    """Resolve current handles for fixed logical roles, without E or P."""
    result={}
    requests=tuple(r for primary in requests.values() for r in (primary,*primary.aliases))
    elemental_roles=tuple(dict.fromkeys(r.reader_role for r in requests if r.family=='elemental'))
    for request in requests:
        if request.family=='propagator':
            symbol=request.reader_role.removeprefix('propagator:')
            mapping={} if propagators is None else propagators
            matches=[reader for key,reader in mapping.items() if str(key)==symbol or str(key)==request.reader_role]
            if not matches: raise ValueError(f'missing saved reader role {request.reader_role!r}')
            result[request.reader_role]=matches[0]
        elif isinstance(vertices,Mapping):
            result[request.reader_role]=vertices[request.reader_role]
        elif len(elemental_roles)==1 and vertices is not None and not callable(vertices):
            result[request.reader_role]=vertices
        else:
            raise ValueError('archive loading needs elemental readers by saved role, or a cfg source factory')
    return result


def iter_memory_calculation(calculation,report):
    """Yield one (cfg, result); retaining previous results is caller-owned."""
    assemblers=bind_point_assemblers(calculation)
    batched=bool(calculation.plan.output_coordinates)
    if batched:
        def assemble(cfg,index,values):
            return tuple(assemble_instance(calculation.plan,assemblers,index,values))
    else:
        def assemble(cfg,index,values):
            return assemblers[calculation.plan.instances[index].template](values)
    events=iter_execute_cfg_plan(calculation.plan,calculation.options['configuration'],
        source_loader_factory(calculation),report=report,transform=assemble,
        **execution_options(calculation))
    results=[None]*len(calculation.plan.coordinates) if batched else []
    try:
        for cfg,index,values in events:
            if index is None:
                value=stack_point_results(results,calculation.plan.coordinate_shape)
                results=[None]*len(calculation.plan.coordinates) if batched else []
                yield cfg,value
                value=None
            else:
                if batched:
                    for position,value in values:
                        results[position]=value
                else:
                    results.append(values)
    finally:
        events.close()
        results.clear()


__all__=['CompleteCalculation','CfgSources','complete_calculation','load_calculation','source_loader_factory']
