"""Pure sampling compatibility for a fixed graph's coordinate replay.

Pool names may change on rebinding; their equality relations and inclusion
laws may not. These are metadata checks, never a second graph expansion.
"""
from lqcdweave.contracts.dataset import canonical_json
from .saved_inputs import _selection


def input_sampling_signature(requests, readers):
    pools, rows = {}, []

    def pool_number(identity):
        return pools.setdefault(identity, len(pools))

    for primary in requests.values():
        for request in (primary, *primary.aliases):
            if request.family != 'elemental' or len(request.arguments) != 4:
                continue
            reader = readers[request.reader_role]
            component, momentum, time, sector = request.arguments
            if not any(kind in sector for kind in ('p', 'd')):
                continue
            if component.time_offsets != (0, 0):
                raise NotImplementedError('sampled complete replay currently requires spatial equal-time elementals')
            header = reader.describe_block(*request.arguments, **_selection(request))
            endpoints = []
            for side, axis, kind in (('bar', 'left', sector[0]), ('field', 'right', sector[2])):
                if kind == 'p':
                    pool_id = reader.point_pool_id(component, momentum, time, side)
                    spec = reader.sampling_group(pool_id)
                    if spec is None:
                        raise ValueError('complete replay requires actual point sampling metadata')
                    observed = spec.observed_counts
                    law = (spec.spatial_sites, spec.sampled_points, spec.sampling_design,
                           spec.compensation, None if observed is None else
                           tuple(sorted((repr(k), int(v)) for k, v in observed.items())))
                    endpoints.append((axis, 'P', pool_number(('P', pool_id)), law))
                elif kind == 'd':
                    source = reader.dilution_basis(component, momentum, time, sector, side)
                    selection = getattr(reader, 'time_selection', None)
                    physical = time if selection is None else selection.times[time]
                    groups = tuple((pool_number(('D', source.pool_key(physical, i))),
                                    group.n, group.m, group.branch, group.covariance_factor)
                                   for i, group in enumerate(source.groups))
                    endpoints.append((axis, 'D', source.algorithm, groups))
            rows.append((sector, header.structure, header.point_diagonal, tuple(endpoints)))
    return canonical_json(rows)


__all__ = ['input_sampling_signature']
