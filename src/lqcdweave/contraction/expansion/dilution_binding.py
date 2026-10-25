"""Actual D groups and same-component summation scenes."""

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product

from lqcdweave.contracts.correlator.api import AxisPattern
from lqcdweave.contracts.dilution import DilutionSource, DILUTION_WEIGHT_SCHEME
from .dilution_weights import dilution_weight_scenes
from .physical_inputs import _vertex_endpoint_axis_name
from lqcdweave.contracts.correlator.vertices import uses_temporal_endpoint_order


def _random(group):
    return group.branch != "exact" and not (group.branch == "qr" and group.m == group.n)


@dataclass(frozen=True)
class DilutionScene:
    factors: tuple
    coefficient: Fraction = Fraction(1)
    equalities: tuple = ()
    pools: tuple = ()


def _physical_time(factor, side):
    if uses_temporal_endpoint_order(factor.operator_term, factor.current_term):
        side = 'bar' if side == 'field' else 'field'
    time = factor.field_time if side == "field" else factor.bar_time
    selection = factor.endpoint_time_selection
    return time if selection is None else selection.times[time]


def expand_dilution_scenes(factors, vertices, *, elemental_weighted, groups_independent=None):
    """Partition flattened D axes into their declared, independent groups.

    One D component label is shared by a vertex endpoint and its adjacent
    propagator endpoint. Same labels are merged without moving either half.
    """
    endpoints, choices, actual_sources = [], [], {}
    for index, factor in enumerate(factors):
        if "d" not in (factor.logical_sector[0], factor.logical_sector[2]):
            continue
        if factor.elemental_weighted != elemental_weighted:
            raise ValueError("elemental_weighted option differs from the stored elemental weighting")
        if elemental_weighted and factor.dilution_weight_scheme != DILUTION_WEIGHT_SCHEME:
            raise ValueError("weighted D elemental requires component_inclusion-v1; regenerate old weighted data")
        for side, source in factor.dilution_endpoints.items():
            if not isinstance(source, DilutionSource):
                raise TypeError("D endpoint requires DilutionSource metadata")
            previous = actual_sources.setdefault(source.data_id, source)
            if previous != source:
                raise ValueError("one actual D dataset has inconsistent dilution source metadata")
            axis, _ = _vertex_endpoint_axis_name(factor, side)
            binding = factor.tensor.axis_bindings.get(axis)
            if binding is None or binding[0] != source.data_id:
                raise ValueError("D elemental axis identity differs from its dilution source")
            indices = tuple(binding[1])
            source.validate_selection(indices)
            selected_groups = tuple(i for i in range(len(source.groups))
                                    if set(source.group_indices(i)).issubset(indices))
            endpoints.append((index, side, source, axis, indices))
            choices.append(selected_groups)
    if not endpoints:
        yield DilutionScene(tuple(factors))
        return
    random_points = any(
        "p" in (factor.logical_sector[0], factor.logical_sector[2])
        and any(pool.sampled_points < pool.spatial_sites
                for pool in factor.point_pool_metadata.values())
        for factor in factors
    )
    if (random_points and any(_random(group) for source in actual_sources.values() for group in source.groups)
            and groups_independent is not True):
        raise ValueError("joint D and P sampling requires groups_independent=True")
    for selected in product(*choices):
        selections = [dict(factor.basis_selections) for factor in factors]
        pools, definitions = {}, {}
        for (vertex, side, source, axis, indices), group_index in zip(endpoints, selected, strict=True):
            group = source.groups[group_index]
            # Match source labels to the current reader's actual ordered axis.
            group_labels = source.group_indices(group_index)
            lookup = {label: position for position, label in enumerate(indices)}
            positions = tuple(lookup[label] for label in group_labels)
            old_positions = factors[vertex].basis_selections.get(axis)
            selections[vertex][axis] = (positions if old_positions is None else
                                       tuple(old_positions[i] for i in positions))
            time = _physical_time(factors[vertex], side)
            pool = source.pool_key(time, group_index)
            if pool is None:
                if _random(group):
                    raise ValueError("random D endpoint requires an explicit physical-time pool identity")
                pool = (source.parent_basis_id, time, "exact", tuple(group.parent_modes))
            signature = group.n, group.m, group.branch, group.covariance_factor, group.vector_labels
            previous = definitions.setdefault(pool, signature)
            if previous != signature:
                raise ValueError("one D pool has inconsistent dilution group metadata")
            pools.setdefault(pool, [group, []])[1].append((vertex, side))
        selected_factors = []
        for vertex, (factor, selection) in enumerate(zip(factors, selections, strict=True)):
            if not selection:
                selected_factors.append(factor)
                continue
            meson = vertices[vertex]
            term = factor.operator_term
            tensor = meson.reader.describe_block(
                term.component, term.momentum, factor.anchor_time % meson.reader.Lt,
                factor.raw_sector, mode_indices=selection, usedNp=factor.usedNp,
            )
            selected_factors.append(replace(factor, tensor=tensor, usedNe=None,
                                            basis_selections=selection))
        scene_options = []
        random_draws = {
            (source.data_id, source.pool_id(_physical_time(factors[vertex], side)))
            for (vertex, side, source, _axis, _indices), group_index in zip(endpoints, selected, strict=True)
            if _random(source.groups[group_index])
        }
        if (len({data_id for data_id, _pool in random_draws}) > 1
                and len({pool for _data_id, pool in random_draws}) > 1
                and groups_independent is not True):
            raise ValueError("different D realizations require groups_independent=True or an explicit shared pool")
        for pool, (group, legs) in pools.items():
            local_entries, equalities = [], []
            for vertex in dict.fromkeys(vertex for vertex, _side in legs):
                positions = tuple(i for i, (owner, _side) in enumerate(legs) if owner == vertex)
                if elemental_weighted:
                    local_entries.append((positions, group.m))
                factor = selected_factors[vertex]
                if len(positions) == 2 and factor.tensor.structure == "diagonal":
                    equalities.append(positions)
            terms = dilution_weight_scenes(group, tuple(f"v{v}:{s}" for v, s in legs),
                local_entries=tuple(local_entries), equalities=tuple(equalities))
            scene_options.append(tuple((tuple(legs), scene.partitions["dilution"], scene.coefficient)
                                       for scene in terms))
        for scenes in product(*scene_options):
            coefficient = Fraction(1)
            equalities = []
            for legs, partition, weight in scenes:
                coefficient *= weight
                if len(set(partition)) < len(partition):
                    equalities.append((legs, partition))
            yield DilutionScene(tuple(selected_factors), coefficient, tuple(equalities),
                                tuple((leg, pool) for pool, (_group, legs) in pools.items() for leg in legs))


def bind_dilution_equalities(network, scene):
    """Merge complete D component labels; keep spin and Wick connectivity."""
    if not scene.equalities:
        return network
    locations = {}
    for index, request in enumerate(network.requests):
        if request.kind == "vertex" and request.sector[0] == "d":
            locations[request.vertex_index, "field"] = (index, 2)
        if request.kind == "vertex" and request.sector[2] == "d":
            locations[request.vertex_index, "bar"] = (index, 4 if request.sector[0] == "p" else 3)
    parent = {}
    def find(label):
        parent.setdefault(label, label)
        if parent[label] != label:
            parent[label] = find(parent[label])
        return parent[label]
    for legs, partition in scene.equalities:
        for block in dict.fromkeys(partition):
            members = tuple(leg for leg, value in zip(legs, partition, strict=True) if value == block)
            if len(members) < 2 or not any(leg in locations for leg in members):
                continue
            if any(leg not in locations for leg in members):
                raise ValueError("equal D components must belong to one contraction network")
            labels = tuple(network.requests[locations[leg][0]].subscripts[locations[leg][1]]
                           for leg in members)
            for label in labels[1:]:
                parent[find(label)] = find(labels[0])
    requests = tuple(replace(request, subscripts=tuple(
        find(label) for label in request.subscripts)) for request in network.requests)
    return replace(network, requests=requests,
                   equation=AxisPattern(tuple(request.subscripts for request in requests), ()))
