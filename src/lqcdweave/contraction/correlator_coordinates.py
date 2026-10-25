"""Retain a chosen dense DAG while binding a bounded coordinate scan."""
from dataclasses import replace

from lqcdweave.adapters.execution_target import check_execution_target
from lqcdweave.adapters.hardware_budget import resolve_diagram_memory_limits
from lqcdweave.contracts.correlator.api import (
    PlannedContraction, describe_coordinates, device_allocator_headroom,
)
from lqcdweave.contraction.expansion.api import (
    describe_input_reads, describe_input_preparations, describe_coordinate_reads,
    input_readers, restore_input_loader, unknown_reader_aliases,
)
from lqcdweave.contraction.planning.api import plan_coordinate_batches
from lqcdweave.contraction.calculation import Calculation
from lqcdweave.contraction.correlator_cfg import readers_for_requests, check_coordinate_sampling


def select_coordinate_replay(calculation, *, time_axes=None, time_pairs=None, batch='auto',
                             configuration=None, propagator_map=None, vertex_map=None,
                             backend=None, debug=None, retain_raw_results=False):
    """Prepare ordinary task-batch records, then use the common .run() executor.

    Input reads, preparations, numerical kernels and output assembly remain in
    their existing services. This function never calls symbolic expansion or a
    contraction-path search and never catches numerical allocation failures.
    """
    if calculation.path is None or calculation.plan is None or calculation.path.batch_paths:
        raise ValueError('coordinate replay requires a selected single-coordinate dense template')
    expanded = calculation.expanded
    if len(expanded.coordinates.points) != 1:
        raise ValueError('coordinate replay requires a selected single-coordinate template')
    debug = calculation.options.get('debug', False) if debug is None else debug
    if type(debug) is not bool:
        raise TypeError('debug must be bool')
    if type(retain_raw_results) is not bool:
        raise TypeError('retain_raw_results must be bool')
    if time_axes is None and time_pairs is None:
        time_axes = dict(expanded.coordinates.points[0].values)
    coordinates, grouping = describe_coordinates(time_axes=time_axes, time_pairs=time_pairs, batch=batch)
    target = calculation.options.get('execution_target')
    backend = check_execution_target(target, backend or calculation.plan.state.problem.backend)
    if backend.__name__ != calculation.plan.state.problem.backend:
        raise ValueError('coordinate replay backend differs from the selected path')
    limits = resolve_diagram_memory_limits(backend=backend,
        memory_limit=calculation.path.memory_limit, host_memory_limit=calculation.path.host_memory_limit,
        execution_mode='local')
    requests = describe_input_reads(expanded.inputs)
    if propagator_map is None and vertex_map is None:
        readers = input_readers(expanded.inputs)
    else:
        readers = readers_for_requests(requests, propagator_map, vertex_map)
    cfg = expanded.bindings.configuration if configuration is None else configuration
    check_coordinate_sampling(expanded, coordinates, readers)
    closure, specs = describe_input_preparations(expanded.inputs)
    aliases = unknown_reader_aliases(expanded.inputs)
    cap = limits.device_bytes if backend.__name__ == 'cupy' else limits.host_bytes
    allocator = device_allocator_headroom(cap, backend=backend.__name__,
        device_name=getattr(target, 'name', None))
    # A scalar coordinate result may itself be rebound to other coordinates.
    # Its previous automatic allowance is replaced, never counted as a caller
    # reserve and added again. Repeated .run() uses the saved batches directly.
    template = calculation.plan
    previous = calculation.options.get('coordinate_allocator_headroom', {}).get('reserve_bytes', 0)
    if previous:
        template = template.with_reserves(reserved_device_bytes=template.reserved_device_bytes-previous)
    output_reserve = None
    assemble_coordinates = bool(coordinates.scan_shape) and not retain_raw_results
    if assemble_coordinates:
        from math import prod
        import numpy as np
        from lqcdweave.contraction.results.api import describe_output_layout
        scalar_recipe = expanded.output_recipe
        point_workspace, fields = describe_output_layout(scalar_recipe, template.state.problem)
        field_bytes = sum(prod(shape) * np.dtype(dtype).itemsize
                          for _, shape, dtype in fields)
        caller_reserve = max(0, template.reserved_host_bytes - expanded.inputs.assembly_host_bytes)
        # Only final fields survive coordinates; stacking overlaps their two
        # owners. Raw terms and R scratch survive one batch/point, not the scan.
        output_reserve = caller_reserve + 2 * len(coordinates.points) * field_bytes + point_workspace
    selected = plan_coordinate_batches(template, expanded.output_recipe, coordinates,
        requests, closure, specs, batch=grouping,
        read_metadata=describe_coordinate_reads(expanded.inputs, coordinates.points), reader_aliases=aliases,
        memory_limit=cap, allocator_reserve_bytes=allocator['reserve_bytes'],
        host_memory_limit=limits.host_bytes,
        kernel_batching=calculation.options.get('kernel_batching', False), debug=debug,
        target_runtime=getattr(target, 'runtime', None), output_host_reserve_bytes=output_reserve)
    # Pair scans acquire the explicit pair role inside coordinate planning.
    # Keep parent and child BatchSpec identical, including that normalized role.
    grouping = selected.batches[0].inputs.batch
    all_requests, all_preparations, all_specs, all_blocks = {}, {}, {}, {}
    for descriptor in selected.batches:
        all_requests.update(descriptor.requests)
        all_preparations.update(descriptor.preparation_closure)
        all_specs.update(descriptor.specs)
        all_blocks.update(descriptor.blocks)
    loader = restore_input_loader(all_requests, readers, all_preparations, all_specs,
        configuration=cfg, reader_aliases=aliases, debug=debug)
    children = tuple(replace(descriptor.inputs, loader=loader, blocks=all_blocks,
                             read_requests=all_requests) for descriptor in selected.batches)
    scanned = bool(coordinates.scan_shape)
    inputs = replace(children[0], coordinates=coordinates, coordinate_batch=None,
        task_batches=children if scanned else (), loader=loader, batch=grouping, batch_output_axes=())
    rebound = replace(expanded, inputs=inputs, coordinates=coordinates, batch=grouping,
        bindings=replace(expanded.bindings, configuration=cfg),
        output_recipe=replace(expanded.output_recipe, coordinates=coordinates), identity_token=object())
    raw_term_bytes = sum(prod(template.state.catalog[key].shape) * template.state.catalog[key].dtype.itemsize
                         for key in template.state.results) if assemble_coordinates else 0
    coordinate_statistics = dict(selected.statistics, allocator_headroom=allocator)
    if assemble_coordinates:
        coordinate_statistics['raw_measurement_host_reserve_bytes'] = (
            len(coordinates.points) * (template.reserved_host_bytes + 2 * raw_term_bytes)
            + (selected.width * raw_term_bytes if backend.__name__ == 'cupy' else 0))
    prediction = dict(coordinate_batching=coordinate_statistics,
        peak_host_bytes=selected.statistics['peak_host_bytes'],
        peak_device_bytes=selected.statistics['peak_device_bytes'],
        kernel_batching=calculation.options.get('kernel_batching', False))
    paths = tuple(PlannedContraction(replace(rebound, inputs=child), plan, None, prediction, 0.,
        'planned', plan.memory_limit, plan.host_memory_limit, id(plan), plan.cost_profile,
        plan.reserved_host_bytes, plan.reserved_device_bytes)
        for child, plan in zip(children, selected.plans, strict=True))
    first = paths[0]
    path = replace(first, expanded=rebound, prediction=prediction,
        planning_seconds=selected.statistics['binding_and_storage_seconds'], batch_paths=paths if scanned else ())
    return Calculation(rebound, path, options=dict(calculation.options, execute=False,
        configuration=cfg, batch=batch, coordinate_batching=prediction['coordinate_batching'],
        coordinate_output_assembly=assemble_coordinates,
        coordinate_allocator_headroom=allocator),
        _execution_bindings=calculation._execution_bindings,
        _execution_resources=calculation._execution_resources,
        _output_bindings=calculation._output_bindings)
