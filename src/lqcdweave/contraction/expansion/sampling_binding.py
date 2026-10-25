"""Expansion helpers for explicit Diagram point-pool metadata and scenes."""

from __future__ import annotations

from collections.abc import Mapping

from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contracts.correlator.vertices import uses_temporal_endpoint_order
from lqcdweave.contracts.correlator.api import canonical_axis_binding_identity
from lqcdweave.contracts.elemental_sectors import sector_layout
from .sampling import (
    SamplingGroup,
    SamplingScene,
    expand_sampling_scenes,
)


def describe_point_pool_metadata(factors_by_vertex, states):
    """Collect declared point-pool identities for every selected Meson P leg.

    ``factors_by_vertex`` contains header-only factor descriptions. Baryon
    vertices are passed with both state sides fixed to ``v``; they therefore
    contribute no point legs and need no synthetic pool metadata.
    """
    point_pool_by_leg, pool_specs = {}, {}
    for vertex_index, (descriptors, state) in enumerate(
            zip(factors_by_vertex, states, strict=True)):
        for state_side, graph_endpoint in (("left", "field"), ("right", "bar")):
            if state[state_side] != "p":
                continue
            descriptor_pools = tuple(
                (item, item.point_pool_ids.get(
                    _physical_point_endpoint(item, graph_endpoint)))
                for item in descriptors
            )
            pool_ids = {pool_id for _item, pool_id in descriptor_pools}
            if len(pool_ids) != 1 or None in pool_ids:
                raise ValueError(
                    f"P endpoint {(vertex_index, graph_endpoint)} lacks one explicit pool identity"
                )
            pool_id = next(iter(pool_ids))
            metadata = [item.point_pool_metadata.get(selected_pool_id)
                        for item, selected_pool_id in descriptor_pools]
            if any(not isinstance(item, SamplingPoolSpec) for item in metadata):
                raise ValueError(f"sampling pool metadata is missing for explicit pool {pool_id!r}")
            if any(item != metadata[0] for item in metadata[1:]):
                raise ValueError(f"one current endpoint has inconsistent metadata for pool {pool_id!r}")
            for descriptor in descriptors:
                selected_count = (descriptor.usedNp if descriptor.usedNp is not None
                                  else metadata[0].sampled_points)
                logical_shape = logical_vertex_tail_shape(descriptor)
                point_axis = _logical_point_axis_index(descriptor, graph_endpoint)
                if (point_axis is None or point_axis >= len(logical_shape)
                        or logical_shape[point_axis] != selected_count):
                    raise ValueError(
                        f"P endpoint {(vertex_index, graph_endpoint)} has an inconsistent logical point-axis extent "
                        f"selected points but pool {pool_id!r} declares Np={selected_count}"
                    )
            old = pool_specs.setdefault(pool_id, metadata[0])
            if old != metadata[0]:
                raise ValueError(f"sampling pool {pool_id!r} has inconsistent metadata across endpoints")
            point_pool_by_leg[(vertex_index, graph_endpoint)] = pool_id
    return point_pool_by_leg, pool_specs


def _physical_point_endpoint(description, graph_endpoint):
    """Map a Diagram's graph side to a temporal link's physical endpoint."""
    if uses_temporal_endpoint_order(description.operator_term, description.current_term):
        return "bar" if graph_endpoint == "field" else "field"
    return graph_endpoint


def _logical_point_axis_index(description, endpoint_side):
    """Return the selected P endpoint's logical axis after source daggering."""
    logical_axes = {
        ("v2p", "bar"): 1,
        ("p2v", "field"): 0,
        ("p2p", "field"): 0,
        ("p2p", "bar"): 2,
    }
    return logical_axes.get((sector_layout(description.logical_sector), endpoint_side))


def expand_explicit_point_pools(
    point_pool_by_leg,
    pool_specs,
    *,
    usedNp,
    groups_independent,
    local_weights=None,
    point_equalities=None,
) -> tuple[SamplingScene, ...]:
    """Build SamplingGroups from explicit pool identities and expand scenes."""
    if not isinstance(point_pool_by_leg, Mapping) or not isinstance(pool_specs, Mapping):
        raise TypeError("point-pool legs and metadata must be explicit mappings")
    legs_by_pool = {}
    for (vertex, endpoint), pool_id in point_pool_by_leg.items():
        legs_by_pool.setdefault(pool_id, []).append(f"v{vertex}:{endpoint}")
    groups = []
    for pool_id, legs in legs_by_pool.items():
        try:
            metadata = pool_specs[pool_id]
        except KeyError as exc:
            raise ValueError(f"sampling metadata is missing for pool {pool_id!r}") from exc
        if not isinstance(metadata, SamplingPoolSpec):
            raise TypeError("pool_specs values must be SamplingPoolSpec metadata")
        sample_count = metadata.sampled_points if usedNp is None else usedNp
        if sample_count > metadata.sampled_points:
            raise ValueError(
                f"usedNp={sample_count} exceeds explicitly registered pool {pool_id!r} "
                f"coverage {metadata.sampled_points}"
            )
        groups.append(SamplingGroup(
            metadata.group_id, tuple(legs), metadata.spatial_sites, sample_count,
            metadata.compensation, metadata.observed_counts, metadata.sampling_design,
        ))
    return expand_sampling_scenes(groups, groups_independent=groups_independent,
                                  local_weights=local_weights, point_equalities=point_equalities)


def describe_point_equalities(factors_by_vertex, states, point_pool_by_leg, pool_specs):
    """Identify guaranteed P/P equalities in original per-pool leg order.

    All terms at a vertex must promise support on the same ordered diagonal.
    Independent pools and observed compensation retain their original scenes.
    """
    legs_by_pool = {}
    for leg, pool_id in point_pool_by_leg.items():
        legs_by_pool.setdefault(pool_id, []).append(leg)
    equalities = {}
    for vertex, (descriptions, state) in enumerate(zip(factors_by_vertex, states, strict=True)):
        if state["left"] != "p" or state["right"] != "p":
            continue
        field, bar = (vertex, "field"), (vertex, "bar")
        pool_id = point_pool_by_leg[field]
        if (pool_id != point_pool_by_leg[bar]
                or pool_specs[pool_id].compensation != "expected"):
            continue
        if not all(item.tensor.point_diagonal for item in descriptions):
            continue
        # Reader declarations describe selected index order. If actual basis
        # bindings are present, both endpoint identities and labels must agree.
        if any(_point_basis_identity(item.tensor, "point_left")
               != _point_basis_identity(item.tensor, "point_right")
               for item in descriptions):
            continue
        legs = legs_by_pool[pool_id]
        equalities.setdefault(pool_id, []).append((legs.index(field), legs.index(bar)))
    return equalities


def _point_basis_identity(tensor, axis):
    binding = tensor.axis_bindings.get(axis)
    return (None if binding is None else
            canonical_axis_binding_identity({"point": binding}))


def describe_local_elemental_weights(factors_by_vertex, states, point_pool_by_leg,
                                     pool_specs, *, elemental_weighted):
    """Describe weights already present in each selected current's raw block.

    Positions use exactly the per-pool leg order consumed by the scene builder.
    Recorded sample counts describe production, while usedNp selects the target
    estimator's current sample count in expand_explicit_point_pools.
    """
    legs_by_pool = {}
    for (vertex, endpoint), pool_id in point_pool_by_leg.items():
        legs_by_pool.setdefault(pool_id, []).append((vertex, endpoint))
    local_weights = {}
    for vertex, (descriptions, state) in enumerate(zip(factors_by_vertex, states, strict=True)):
        if "p" not in state.values():
            continue
        marks = {item.elemental_weighted for item in descriptions}
        if marks != {elemental_weighted}:
            raise ValueError("elemental_weighted option differs from the stored elemental weighting")
        if not elemental_weighted:
            continue
        for pool_id, legs in legs_by_pool.items():
            positions = tuple(i for i, (owner, _) in enumerate(legs) if owner == vertex)
            if positions:
                pool = pool_specs[pool_id]
                if pool.compensation != "expected":
                    raise ValueError("elemental_weighted requires expected point compensation")
                local_weights.setdefault(pool_id, []).append((positions, pool.sampled_points))
    return local_weights


def logical_vertex_tail_shape(description):
    """Return the Meson factor tail shape after the existing dagger transpose."""
    permutations = {
        "v2v": (1, 0),
        "v2p": (1, 2, 0),
        "p2v": (2, 0, 1),
        "p2p": (2, 3, 0, 1),
    }
    expected_rank = {"v2v": 2, "v2p": 3, "p2v": 3, "p2p": 4}
    shape = tuple(description.tensor.shape)
    if description.dagger:
        try:
            shape = tuple(shape[index] for index in permutations[sector_layout(description.raw_sector)])
        except KeyError as exc:
            raise ValueError(f"unknown raw Meson sector {description.raw_sector!r}") from exc
    if len(shape) != expected_rank.get(sector_layout(description.logical_sector)):
        raise ValueError(
            f"elemental metadata rank for {description.logical_sector} is {len(shape)} "
            "after dagger orientation"
        )
    for sector, color_axes in (("v2p", (2,)), ("p2v", (1,)), ("p2p", (1, 3))):
        if sector_layout(description.logical_sector) == sector and any(shape[axis] != 3 for axis in color_axes):
            raise ValueError(f"elemental {sector} color axes must have extent three")
    return shape


__all__ = [
    "describe_point_pool_metadata",
    "expand_explicit_point_pools",
    "logical_vertex_tail_shape",
    "describe_local_elemental_weights",
    "describe_point_equalities",
]
