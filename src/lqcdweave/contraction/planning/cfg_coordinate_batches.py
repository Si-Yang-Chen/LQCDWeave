"""Close selected coordinate batches in the existing complete cfg records."""
from dataclasses import replace
from time import perf_counter

from lqcdweave.contracts.correlator.api import CoordinatePoint
from .cfg_plan import _bind_cfg_template, audit_cfg_plan, _CfgResourceExceeded
from . import cfg_rounding


def build_coordinate_cfg_plan(batches, output, coordinates, *, output_mode,
                              reader_aliases=(), execution_target=None,
                              target_summary=None, debug=False):
    """Keep each selected DAG and tail; close resources before the next batch.

    Batches carry already-bound physical requests. Empty coordinate bindings
    preserve their original times; only cfg/readers are replaced at replay.
    This adds no second operation graph and performs no contraction search.
    """
    started = perf_counter()
    templates, instances = [], []
    requests, blocks, preparations, specs = {}, {}, {}, {}
    parts = []
    node_offset = action_offset = 0
    for index, batch in enumerate(batches):
        binding = _bind_cfg_template(batch['plan'], output, batch['requests'],
            batch['preparations'], batch['specs'], output_mode=output_mode,
            reader_aliases=reader_aliases, execution_target=execution_target,
            target_summary=target_summary, debug=debug)
        part = binding((CoordinatePoint(str(index), ()),), reuse_window=1,
                       node_offset=node_offset, output_instances=1)
        template, = part.templates
        templates.append(replace(template, output_axes=tuple(batch['output_axes'])))
        instance, = part.instances
        indices = tuple(batch['point_indices'])
        instances.append(replace(instance, template=index, point_indices=indices,
            coordinate=tuple((str(role), value) for role, value in coordinates.points[indices[0]].values),
            action_offset=action_offset))
        for target, source in ((requests, part.requests), (blocks, part.blocks),
                               (preparations, part.preparations), (specs, part.specs)):
            target.update(source)
        node_offset += part.statistics['binding_node_count']
        action_offset += len(template.plan.state.actions) + 1
        parts.append(part)

    first = parts[0]
    statistics = dict(first.statistics)
    sums = ('instances', 'read_blocks', 'compute_nodes', 'prepared_nodes',
            'binding_node_count', 'binding_bytes', 'batch_owner_bytes',
            'compute_seconds', 'preparation_seconds', 'binding_seconds', 'resource_seconds')
    for key in sums:
        statistics[key] = sum(part.statistics[key] for part in parts)
    for key in first.statistics:
        if key.startswith('peak_') or key in ('output_snapshot_device_bytes',
                                             'native_output_snapshot_device_bytes'):
            statistics[key] = max(part.statistics[key] for part in parts)
    constant_host = cfg_rounding.constant_bytes(preparations) if first.backend == 'numpy' else 0
    constant_device = cfg_rounding.constant_bytes(preparations, device=True) if first.backend == 'cupy' else 0
    # Coordinate planning already reserves the entire scan's retained outputs
    # in every child plan. Carry that shared allowance once, not per batch.
    output_bytes = max(part.output_host_bytes for part in parts)
    peak_host = max(part.peak_host_bytes - part.statistics['constant_host_bytes']
                    - part.output_host_bytes for part in parts) + constant_host + output_bytes
    peak_device = max(part.peak_device_bytes - part.statistics['constant_device_bytes']
                     for part in parts) + constant_device
    host_cap, device_cap = statistics['host_memory_cap'], statistics['device_memory_cap']
    for name, peak, cap in (('host', peak_host, host_cap), ('device', peak_device, device_cap)):
        if cap is not None and peak > cap:
            raise _CfgResourceExceeded(f'complete coordinate batches require {peak} {name} bytes; cap is {cap}')
    prefetch_host = min(min(part.prefetch_host_bytes for part in parts), host_cap - peak_host)
    prefetch_device = (min(min(part.prefetch_device_bytes for part in parts), device_cap - peak_device)
                       if device_cap is not None else 0)
    statistics.update(templates=len(templates), output_host_bytes=output_bytes,
        constant_host_bytes=constant_host, constant_device_bytes=constant_device,
        constant_device_payload_bytes=(cfg_rounding.constant_bytes(preparations) if first.backend == 'cupy' else 0),
        peak_host_bytes=peak_host, peak_device_bytes=peak_device,
        prefetch_host_bytes=prefetch_host, prefetch_device_bytes=prefetch_device,
        coordinate_batch_sizes=tuple(len(instance.point_indices) for instance in instances),
        coordinate_count=len(coordinates.points), path_searches=0,
        lifecycle_seconds=perf_counter() - started)
    result = replace(first, templates=tuple(templates), instances=tuple(instances),
        requests=requests, blocks=blocks, preparations=preparations, specs=specs,
        output_coordinates=tuple(tuple((str(role), value) for role, value in point.values)
                                 for point in coordinates.points),
        peak_host_bytes=peak_host, peak_device_bytes=peak_device,
        output_host_bytes=output_bytes, prefetch_host_bytes=prefetch_host,
        prefetch_device_bytes=prefetch_device, statistics=statistics)
    if debug:
        audit_cfg_plan(result)
    return result
