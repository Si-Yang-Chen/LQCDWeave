"""High point endpoints share the ordinary V/D/P production and pair IO."""

from lqcdweave.contracts.mode_partition import ModePartition
from lqcdweave.contracts.propagator import PropagatorEndpoint
from dataclasses import replace
import numpy as np
from .products import product_shape


class HighmodeBasis:
    """View of an actual orthonormal parent used only to apply Q=I-U U†.

    No Dirac solves are performed for these parent vectors. The underlying
    handle and its borrowed payload must remain valid during generator.load.
    """

    basis_orthonormal = True
    basis_orthogonal = True

    def __init__(self, basis, partition):
        if not isinstance(partition, ModePartition):
            raise TypeError("partition must be a ModePartition")
        if getattr(basis, "basis_orthonormal", False) is not True:
            raise ValueError("high complement requires an explicitly orthonormal parent basis")
        if getattr(basis, "basis_id", None) != partition.parent_basis_id:
            raise ValueError("high basis identity differs from the mode partition")
        labels = tuple(getattr(basis, "mode_labels", ()))
        positions = {label: position for position, label in enumerate(labels)}
        if len(positions) != len(labels) or any(mode not in positions for mode in partition.projection_modes):
            raise ValueError("high basis lacks the complete low+medium parent coverage")
        self.basis, self.partition = basis, partition
        self.positions = tuple(positions[mode] for mode in partition.projection_modes)
        self.Ne = len(self.positions)
        self.basis_id, self.mode_labels = partition.parent_basis_id, partition.projection_modes
        self.time_selection = getattr(basis, "time_selection", None)

    def load(self, key):
        # Generator copies this view before any other handle read.
        return self.basis.load(key)[:, list(self.positions)]

    @property
    def supports_mode_reads(self):
        return callable(getattr(self.basis, 'read_modes', None))

    def read_modes(self, key, time, start, stop):
        """Borrow one physical-time parent span, retaining projection order."""
        positions = self.positions[start:stop]
        first, last = min(positions), max(positions) + 1
        block = self.basis.read_modes(key, time, first, last)
        if positions == tuple(range(first, last)):
            return block
        return block[[position - first for position in positions]]

    def point_endpoint(self, endpoint):
        metadata = endpoint.metadata
        previous = metadata.get("high_mode_projection")
        if previous is not None and previous != self.partition.document:
            raise ValueError("point endpoint has a different high projection")
        metadata["high_mode_projection"] = self.partition.document
        return PropagatorEndpoint("P", endpoint.data_id, endpoint.indices, metadata=metadata)


def project_point_source(rhs, vectors, *, array_module):
    """Apply the spatial-color complement, independently for every RHS/spin."""
    coefficients = array_module.einsum("izyxc,rzyxsc->rsi", vectors.conj(), rhs)
    rhs -= array_module.einsum("izyxc,rsi->rzyxsc", vectors, coefficients)


def sample_high_points(solution, points, vectors, *, array_module):
    """Return sampled Q solution without modifying the solved full field."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    sampled = solution[:, z, y, x].copy()
    coefficients = array_module.einsum("izyxc,rzyxsc->rsi", vectors.conj(), solution)
    sampled -= array_module.einsum("ipc,rsi->rpsc", vectors[:, z, y, x], coefficients)
    return sampled


def project_highmode_block(block, *, partition, overlaps, auxiliary, array_module=np):
    """Derive any P-containing V/D/P pair from actual full-parent blocks.

    ``overlaps`` maps sink/source to U(point) in (parent_mode,point,color)
    order. ``auxiliary`` has sink=U† S B, source=A† S U, and cross=U† S U
    when both endpoints are P. Unlike native projected point solves, this
    advanced route requires the full-parent propagators to exist already.
    """
    partitions = partition if isinstance(partition, dict) else dict(sink=partition, source=partition)
    sides = tuple(side for side, endpoint in (('sink', block.sink), ('source', block.source))
                  if endpoint.kind == 'P')
    needed = set(sides) | ({'cross'} if len(sides) == 2 else set())
    if not sides or set(overlaps) != set(sides) or set(auxiliary) != needed:
        raise ValueError("high derivation requires exactly the point-side overlaps and auxiliary blocks")
    xp = array_module
    values = xp.asarray(block.values)
    if values.shape != product_shape(block.product, len(block.sink.indices), len(block.source.indices)):
        raise ValueError("raw block shape differs from its endpoints")
    endpoints = dict(sink=block.sink, source=block.source)
    parent_endpoints = {}
    matrices = {}
    for side in sides:
        part = partitions[side]
        if not isinstance(part, ModePartition) or 'high_mode_projection' in endpoints[side].metadata:
            raise ValueError("high derivation requires a partition and raw point endpoints")
        parent_endpoints[side] = PropagatorEndpoint('V', part.parent_basis_id, part.projection_modes)
        matrices[side] = xp.asarray(overlaps[side])
        if matrices[side].shape != (len(part.projection_modes), len(endpoints[side].indices), 3):
            raise ValueError("overlap must cover all low+medium parent modes and ordered points")
    for role, other in auxiliary.items():
        sink = parent_endpoints['sink'] if role in ('sink', 'cross') else block.sink
        source = parent_endpoints['source'] if role in ('source', 'cross') else block.source
        if ((other.sink, other.source) != (sink, source)
                or (other.sink_time, other.source_time) != (block.sink_time, block.source_time)
                or other.values.shape != product_shape(other.product, len(sink.indices), len(source.indices))):
            raise ValueError("auxiliary block must bind the complete actual parent and physical time pair")
    dtype = np.result_type(values.dtype, *(value.dtype for value in matrices.values()),
                           *(other.values.dtype for other in auxiliary.values()))
    if dtype.kind != 'c':
        raise ValueError("high derivation requires complex arrays")
    result = xp.array(values, dtype=dtype, copy=True)
    ns = 3 * len(block.sink.indices) if block.sink.kind == 'P' else len(block.sink.indices)
    nr = 3 * len(block.source.indices) if block.source.kind == 'P' else len(block.source.indices)
    flat = result.reshape(4, 4, ns, nr)
    if 'sink' in sides:
        overlap = matrices['sink'].reshape(len(parent_endpoints['sink'].indices), ns)
        other = xp.asarray(auxiliary['sink'].values, dtype=dtype).reshape(4, 4, len(overlap), nr)
        flat -= overlap.T @ other
    if 'source' in sides:
        overlap = matrices['source'].reshape(len(parent_endpoints['source'].indices), nr).conj()
        other = xp.asarray(auxiliary['source'].values, dtype=dtype).reshape(4, 4, ns, len(overlap))
        flat -= other @ overlap
    if len(sides) == 2:
        left = matrices['sink'].reshape(len(parent_endpoints['sink'].indices), ns)
        right = matrices['source'].reshape(len(parent_endpoints['source'].indices), nr).conj()
        other = xp.asarray(auxiliary['cross'].values, dtype=dtype)
        # Contract one parent index at a time. A direct three-input einsum
        # repeats both parent sums for every pair of point/color indices.
        ml, mr = len(left), len(right)
        if ns * ml * mr + ns * mr * nr <= ml * mr * nr + ns * ml * nr:
            flat += (left.T @ other) @ right
        else:
            flat += left.T @ (other @ right)
    for side in sides:
        endpoint = endpoints[side]
        metadata = endpoint.metadata
        metadata['high_mode_projection'] = partitions[side].document
        endpoints[side] = PropagatorEndpoint('P', endpoint.data_id, endpoint.indices, metadata=metadata)
    return replace(block, values=result, sink=endpoints['sink'], source=endpoints['source'], derived=True)
