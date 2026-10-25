"""Pure lowering from one-in/one-out V/D/P graphs to Einstein networks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from lqcdweave.contracts.correlator.api import AxisPattern

from .sampling import SamplingGroup, SamplingScene, expand_sampling_scenes


@dataclass(frozen=True)
class TensorRequest:
    """One requested tensor and the Einstein labels assigned to its axes."""

    kind: str
    tensor_label: object
    sector: str
    subscripts: tuple[int, ...]
    vertex_index: int | None = None
    source_vertex: int | None = None
    sink_vertex: int | None = None
    endpoint_side: str | None = None


@dataclass(frozen=True)
class ContractionNetwork:
    """One scalar Einstein network after sampled point identities are merged."""

    requests: tuple[TensorRequest, ...]
    equation: AxisPattern
    vertex_indices: tuple[int, ...]
    propagator_edges: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class StateSamplingVariant:
    vertex_states: tuple[dict[str, str], ...]
    scene: SamplingScene


def expand_state_sampling_variants(
    vertex_count,
    current_vertices,
    point_pool_by_leg,
    sampling_pools,
    *,
    groups_independent=None,
    usedNp=None,
):
    """Expand V/P endpoint states and exact scenes over declared pool IDs."""
    current = set(current_vertices)
    if any(type(index) is not int or not 0 <= index < vertex_count for index in current):
        raise ValueError("current vertex index outside diagram")
    options = (({"left": "v", "right": "v"},
                {"left": "v", "right": "p"},
                {"left": "p", "right": "v"},
                {"left": "p", "right": "p"}) if vertex in current
               else ({"left": "v", "right": "v"},)
               for vertex in range(vertex_count))
    for states in product(*options):
        legs_by_pool = {}
        for vertex, state in enumerate(states):
            for state_side, endpoint_side in (("left", "field"), ("right", "bar")):
                if state[state_side] != "p":
                    continue
                leg_id = f"v{vertex}:{endpoint_side}"
                if (vertex, endpoint_side) not in point_pool_by_leg:
                    raise ValueError(f"P endpoint {(vertex, endpoint_side)} has no explicit sampling pool")
                group_id = point_pool_by_leg[vertex, endpoint_side]
                if group_id not in sampling_pools:
                    raise ValueError(f"sampling pool metadata is missing for {group_id!r}")
                legs_by_pool.setdefault(group_id, []).append(leg_id)
        groups = []
        for group_id, legs in legs_by_pool.items():
            template = sampling_pools[group_id]
            if not isinstance(template, SamplingGroup):
                raise TypeError("sampling_pools must contain SamplingGroup metadata")
            sample_count = template.sampled_points if usedNp is None else usedNp
            groups.append(SamplingGroup(
                template.group_id, tuple(legs), template.spatial_sites, sample_count,
                template.compensation, template.observed_counts, template.sampling_design,
            ))
        scenes = expand_sampling_scenes(groups, groups_independent=groups_independent)
        for scene in scenes:
            yield StateSamplingVariant(tuple(dict(state) for state in states), scene)


def lower_state_graph(adjacency_matrix, vertex_states, *, point_labels=None, mode_pools=None):
    """Lower a meson-like graph using explicit V/D/P state and sampling labels.

    Different disconnected fermion components are joined exactly when they
    share an explicit point or D-component class. Baryon slot graphs use the separate dense-V
    expansion path and fail here rather than losing their ordered legs.
    """
    states = tuple(vertex_states)
    point_labels = {} if point_labels is None else dict(point_labels)
    count = len(states)
    if len(adjacency_matrix) != count or any(len(row) != count for row in adjacency_matrix):
        raise ValueError("Diagram adjacency must be square and match vertex states")
    for index, state in enumerate(states):
        if not isinstance(state, dict) or set(state) != {"left", "right"} or any(
            value not in {"v", "d", "p"} for value in state.values()
        ):
            raise ValueError(f"invalid V/D/P endpoint state at vertex {index}")

    edges = []
    for source, row in enumerate(adjacency_matrix):
        for sink, cell in enumerate(row):
            for label in _flatten_edge_labels(cell):
                edges.append((label, source, sink))
    if not edges:
        raise ValueError("Diagram has no propagator edges")
    outgoing, incoming = [0] * count, [0] * count
    for _label, source, sink in edges:
        outgoing[source] += 1
        incoming[sink] += 1
    active = {vertex for _, source, sink in edges for vertex in (source, sink)}
    if any(outgoing[v] != 1 or incoming[v] != 1 for v in active):
        raise NotImplementedError("sector lowering currently requires one-in/one-out meson vertices")

    parent = list(range(count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for _label, source, sink in edges:
        union(source, sink)

    point_owners = {}
    for _label, source, sink in edges:
        for vertex, state_side, endpoint_side in (
            (source, "left", "field"), (sink, "right", "bar")
        ):
            if states[vertex][state_side] != "p":
                continue
            key = (vertex, endpoint_side)
            if key not in point_labels:
                raise ValueError(f"P endpoint {key} has no explicit sampling-scene point label")
            point_class = point_labels[key]
            if point_class in point_owners:
                union(vertex, point_owners[point_class])
            else:
                point_owners[point_class] = vertex

    mode_owners = {}
    for (vertex, _side), pool in (mode_pools or {}).items():
        if pool in mode_owners:
            union(vertex, mode_owners[pool])
        else:
            mode_owners[pool] = vertex

    grouped_edges = {}
    for edge in edges:
        grouped_edges.setdefault(find(edge[1]), []).append(edge)
    networks = []
    for component_edges in sorted(grouped_edges.values(), key=lambda items: min(x[1] for x in items)):
        networks.append(_lower_connected_component(component_edges, states, point_labels))
    return tuple(networks)


def _lower_connected_component(edges, states, point_labels):
    vertices = tuple(sorted({vertex for _, source, sink in edges for vertex in (source, sink)}))
    point_ids = {}
    propagator_requests, propagator_subscripts = [], []
    vertex_sides = {vertex: {} for vertex in vertices}
    for edge_index, (label, source, sink) in enumerate(edges):
        src_spin, snk_spin = 4 * edge_index, 4 * edge_index + 1
        src_mode, snk_mode = 4 * edge_index + 2, 4 * edge_index + 3
        src_color, snk_color = 4 * len(edges) + 2 * edge_index, 4 * len(edges) + 2 * edge_index + 1
        src_state, snk_state = states[source]["left"], states[sink]["right"]

        def endpoint_label(vertex, side, state, color_axis, mode_axis):
            if state in {"v", "d"}:
                return mode_axis, None
            point_class = point_labels[(vertex, side)]
            if point_class not in point_ids:
                point_ids[point_class] = 6 * len(edges) + len(point_ids)
            return point_ids[point_class], color_axis

        src_spatial, src_color_index = endpoint_label(
            source, "field", src_state, src_color, src_mode)
        snk_spatial, snk_color_index = endpoint_label(
            sink, "bar", snk_state, snk_color, snk_mode)
        sector = f"{snk_state}s{src_state}"
        subscript = (snk_spin, src_spin, snk_spatial) + (() if snk_color_index is None else (snk_color_index,)) + (src_spatial,) + (() if src_color_index is None else (src_color_index,))
        propagator_requests.append(TensorRequest(
            "propagator", label, sector, subscript,
            source_vertex=source, sink_vertex=sink,
        ))
        propagator_subscripts.append(subscript)
        vertex_sides[source]["field"] = (src_spin, src_state, src_spatial, src_color_index)
        vertex_sides[sink]["bar"] = (snk_spin, snk_state, snk_spatial, snk_color_index)

    vertex_requests = []
    for vertex in vertices:
        if set(vertex_sides[vertex]) != {"field", "bar"}:
            raise NotImplementedError("meson vertex needs one field-facing and one bar-facing leg")
        field = vertex_sides[vertex]["field"]
        bar = vertex_sides[vertex]["bar"]
        state = states[vertex]
        sector = f"{state['left']}2{state['right']}"
        field_suffix = (field[2],) + (() if field[3] is None else (field[3],))
        bar_suffix = (bar[2],) + (() if bar[3] is None else (bar[3],))
        subscript = (field[0], bar[0]) + field_suffix + bar_suffix
        vertex_requests.append(TensorRequest(
            "vertex", vertex, sector, subscript, vertex_index=vertex,
        ))
    requests = tuple(propagator_requests + vertex_requests)
    equation = AxisPattern(tuple(request.subscripts for request in requests), ())
    return ContractionNetwork(requests, equation, vertices, tuple(edges))


def _flatten_edge_labels(value):
    if isinstance(value, bool):
        raise TypeError("propagator labels cannot be bool")
    if isinstance(value, Integral):
        if int(value) == 0:
            return ()
        if int(value) < 0:
            raise ValueError("propagator labels must be nonnegative")
        return (int(value),)
    if isinstance(value, (list, tuple)):
        return tuple(label for child in value for label in _flatten_edge_labels(child))
    raise ValueError(f"invalid propagator adjacency entry {value!r}")


__all__ = [
    "TensorRequest", "ContractionNetwork", "StateSamplingVariant",
    "expand_state_sampling_variants", "lower_state_graph",
]
