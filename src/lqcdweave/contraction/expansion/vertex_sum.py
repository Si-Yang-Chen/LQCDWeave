"""Register a whole vertex as one existing source-preparation dependency tree."""
from dataclasses import replace
from math import prod

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant, SourcePreparation, SourceSpec, TensorPreparation, TensorSpec,
)


def register_vertex_sum(members, factors, *, specs, sources, blocks, preparations,
                        tensor_requests, tensor_metadata=None):
    members, factors = tuple(members), tuple(factors)
    weights = tuple(complex(item.operator_term.coefficient).conjugate()
                    if item.dagger else complex(item.operator_term.coefficient)
                    for item in factors)
    if len(members) == 1 and weights == (1,):
        return members[0]
    identity = ('whole-meson', members, weights)
    key = 'whole-meson:' + str(len(specs))
    # Registration is call-local; numerical arrays and reader owners never enter
    # this identity. Repeated appearances of the same vertex reuse its result.
    for previous, request in tensor_requests.items():
        if request == identity:
            return previous
    first = specs[members[0]]
    shape = first.shape
    if any(specs[member].shape != shape for member in members):
        raise ValueError('whole vertex members have different endpoint extents')
    dtype = np.result_type(*(specs[member].dtype for member in members), np.asarray(weights).dtype)
    dependencies = tuple(dict.fromkeys(block for member in members
                                      for block in preparations[member].blocks))
    description = replace(first.description, source_bindings=dependencies,
        transform=(('operation', 'whole-meson-sum'),),
        semantics={**dict(first.description.semantics),
                   **({} if tensor_metadata is None else {
                       'structure': tensor_metadata.structure,
                       'point_diagonal': tensor_metadata.point_diagonal}),
                   'coefficient_owner': 'vertex-preparation',
                   'components': tuple(dict(specs[member].description.semantics) for member in members),
                   'coefficients': weights},
        bindings={**dict(first.description.bindings),
                  'component_bindings': tuple(dict(specs[member].description.bindings) for member in members)})
    specs[key] = TensorSpec(shape, dtype, description=description)
    nbytes = prod(shape) * dtype.itemsize
    region_bytes = sum(blocks[block].read_bytes for block in dependencies)
    sources[key] = SourceSpec(region=key, region_bytes=region_bytes,
        host_region_bytes=sum(blocks[block].host_bytes for block in dependencies),
        snapshot_bytes=nbytes, metadata={'family': 'whole-meson', 'blocks': dependencies})
    member_workspace = max(preparations[member].workspace_host_bytes for member in members)
    device_workspace = max(preparations[member].workspace_device_bytes for member in members)
    device = any(blocks[block].device_bytes for block in dependencies)
    scratch = nbytes if len(members) > 1 else 0
    # Result residency is accounted by SourceSpec. One reusable component buffer
    # coexists with a member's existing preparation workspace.
    preparations[key] = SourcePreparation(dependencies,
        workspace_host_bytes=member_workspace + (0 if device else scratch),
        workspace_device_bytes=device_workspace + (scratch if device else 0),
        flops=prod(shape) * (2 * len(members) - 1),
        traffic_bytes=nbytes * (3 * len(members) - 1),
        kernel_calls=sum(preparations[member].kernel_calls for member in members) + 2 * len(members) - 1,
        dtype=dtype.name,
        recipe=TensorPreparation('linear_combination', members,
                                 (ArrayConstant.from_array(np.asarray(weights)),)))
    tensor_requests[key] = identity
    return key
