"""Complete a task declaration once, independently of its output destination."""
from dataclasses import replace
from time import perf_counter

from lqcdweave.contracts.correlator.api import describe_coordinates, device_allocator_headroom
from lqcdweave.contraction.expansion.api import (
    input_readers, describe_input_reads, describe_input_preparations,
    describe_coordinate_reads, unknown_reader_aliases,
)
from lqcdweave.contraction.planning.api import plan_coordinate_delivery, supports_coordinate_batches
from .correlator_cfg import complete_calculation


def plan_task(correlator, options):
    """Use one selected template, saved coordinate lifting and cfg closure."""
    from .correlator import calculate
    started=perf_counter()
    options=dict(options)
    declared_vertices=options.get('vertex_map')
    coordinates,grouping=describe_coordinates(time_axes=options.pop('time_axes',None),
        time_pairs=options.pop('time_pairs',None),batch=options.pop('batch','auto'))
    output=options.pop('output',None)
    representative=max(coordinates.points,key=lambda point:len(set(value for _,value in point.values)))
    selected=calculate(correlator,time_axes=dict(representative.values) or None,
                       execute=False,**options)
    # A declaration without explicit coordinates uses E's bound point.
    if not representative.values:
        coordinates=selected.expanded.coordinates
    inputs=selected.inputs
    readers=input_readers(inputs)
    batches=None
    delivery=None
    problem=selected.plan.state.problem
    can_batch=supports_coordinate_batches(selected.plan)
    if coordinates.scan_shape and can_batch:
        preparations,specs=describe_input_preparations(inputs)
        target=selected.options.get('execution_target')
        cap=selected.path.memory_limit
        allocator=device_allocator_headroom(cap,backend=problem.backend,
            device_name=getattr(target,'name',None))
        delivery=plan_coordinate_delivery(selected.plan,selected.expanded.output_recipe,
            coordinates,describe_input_reads(inputs),preparations,specs,batch=grouping,
            read_metadata=describe_coordinate_reads(inputs,coordinates.points),
            reader_aliases=unknown_reader_aliases(inputs),memory_limit=cap,
            host_memory_limit=selected.path.host_memory_limit,
            kernel_batching=selected.options.get('kernel_batching',False),
            target_runtime=getattr(target,'runtime',None),
            allocator_reserve_bytes=allocator['reserve_bytes'],debug=options.get('debug',False))
        batches=delivery.replay
    elif grouping.size not in ('auto',1):
        raise NotImplementedError('this tensor structure currently supports scalar coordinate instances only')
    completed=complete_calculation(selected,coordinates=coordinates,readers=readers,
        output_mode='dataset' if output is not None else 'memory',output=output,
        coordinate_batches=batches)
    if delivery is not None:
        statistics=dict(completed.plan.statistics,coordinate_selection=delivery.statistics)
        if delivery.mode=='scalar':
            statistics['coordinate_batch_sizes']=(1,)*len(coordinates.points)
        completed=replace(completed,plan=replace(completed.plan,statistics=statistics,
            output_definition=dict(completed.plan.output_definition,
                                   coordinate_selection=delivery.statistics)))
    selection=selected.path.prediction.get('baryon_vertex_selection')
    if selection is not None:
        completed=replace(completed,plan=replace(completed.plan,
            statistics=dict(completed.plan.statistics,baryon_vertex_selection=selection)))
    # Named frontend maps retain their reader lookup keys, not their operator
    # definitions. The saved recipes already contain the numerical vertex.
    from collections.abc import Mapping
    if isinstance(declared_vertices,Mapping):
        roles={id(reader):role for role,reader in readers.items()}
        hints={}
        for key,vertex in declared_vertices.items():
            reader=getattr(vertex,'reader',None)
            if reader is None:
                reader=getattr(vertex,'elemental',vertex)
            role=roles.get(id(reader))
            if role is not None and isinstance(key,str):
                hints.setdefault(role,key)
        if hints:
            completed=replace(completed,plan=replace(completed.plan,
                output_definition=dict(completed.plan.output_definition,frontend_reader_keys=hints)))
    return replace(completed,options=dict(completed.options,planning_seconds=perf_counter()-started))
