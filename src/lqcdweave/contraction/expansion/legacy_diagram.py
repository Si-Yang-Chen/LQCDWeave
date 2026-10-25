"""Data-free lowering of legacy symbolic Diagram graphs into sector terms.

The endpoint-state axis rules and Einstein label order are copied from
``localized_blending/source/lattice/quark_diagram.py``:
``StateExpandedDiagram._build_subscripts_with_types``. Sampling constraints
are supplied explicitly by the caller; pool identity is never inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from numbers import Integral
from collections.abc import Mapping

from .physical_inputs import PhysicalInputRegistry
from .glueball import glueball_term
from lqcdweave.contraction.binding.api import Glueball
from .elemental_structure import lower_complete_diagonals
from .dilution_binding import expand_dilution_scenes, bind_dilution_equalities
from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS
from .state_graph import (
    ContractionNetwork,
    StateSamplingVariant,
    TensorRequest,
    expand_state_sampling_variants,
    lower_state_graph,
)
from lqcdweave.contracts.correlator.api import (
    AxisPattern, DiagramInputs, TermContribution, ContractionTerm, OutputNode,
    ContractionProblem, SparseBridgeSpec,
)
from lqcdweave.contraction.results.api import collect_diagrams
from lqcdweave.contraction.binding.api import bind_slot_vertices, validate_vertex_time_bindings
from .sampling_binding import (
    describe_point_pool_metadata,
    expand_explicit_point_pools,
    describe_local_elemental_weights,
    describe_point_equalities,
)
from lqcdweave.operators.correlator.api import ModeBlockedDiagram


@dataclass(frozen=True)
class _VertexChoice:
    offsets: tuple[int, int]
    indices: tuple[int, ...]


class LegacyDiagramBinding:
    """Adapt the legacy ``propagator_map``/``vertex_map`` onto native readers.

    This adapter consumes already-bound reader objects. ``describe`` calls only
    their metadata methods; ``read_block`` and vertex materialization are delayed
    until the scheduler replays the selected path.
    """

    def describe(self, correlator, *, bindings, coordinates, batch):
        if coordinates.scan_shape:
            raise NotImplementedError(
                "LegacyDiagramBinding currently lowers one fixed time coordinate; scans remain a separate batch stage"
            )
        if not isinstance(bindings.propagator_map, Mapping):
            raise TypeError("LegacyDiagramBinding requires a propagator_map mapping")
        if bindings.vertex_map is None:
            raise TypeError("LegacyDiagramBinding requires a vertex_map mapping or metadata-only callable")
        if batch.size != "auto" or batch.axis is not None:
            raise NotImplementedError("legacy Diagram binding does not implement execution batching")

        diagrams = collect_diagrams(correlator)
        bindings = replace(bindings, propagator_map=_referenced_propagators(
            ((diagram.diagram.adjacency_matrix, diagram.propagator_list) for diagram in diagrams),
            bindings.propagator_map))
        time_values = _coordinate_map(coordinates)
        registry = PhysicalInputRegistry(bindings, coordinate_values=time_values)
        specs, sources = registry.specs, registry.sources
        blocks, preparations = registry.blocks, registry.preparations
        tensor_requests, batch_bindings = registry.tensor_requests, registry.batch_bindings
        add_sparse_record = registry.add_sparse_record
        register_propagator = registry.register_propagator
        register_dense_vertex = registry.register_dense_vertex
        all_terms, term_ids, diagram_recipes = [], [], {}
        sparse_bridges = []
        selected_vertices, vertex_options, factor_metadata, network_templates = {}, {}, {}, {}

        def describe_vertex(vertex, anchor, sector, indices):
            key = (id(vertex), anchor, sector, indices)
            if key not in factor_metadata:
                factor_metadata[key] = vertex.describe_core_factors(
                    anchor, sector, indices=indices,
                    usedNe=bindings.usedNe, usedNp=bindings.usedNp,
                )
            return factor_metadata[key]
        def graph_key(value):
            return tuple(graph_key(item) for item in value) if isinstance(value, (list, tuple)) else value
        for diagram_index, diagram in enumerate(diagrams):
            count = len(diagram.vertex_list)
            if len(diagram.time_list) != count or len(diagram.diagram.adjacency_matrix) != count:
                raise ValueError("Diagram time, vertex, and adjacency counts disagree")
            if any(len(row) != count for row in diagram.diagram.adjacency_matrix):
                raise ValueError("Diagram adjacency matrix must be square")
            anchors = tuple(_resolve_time(value, time_values) for value in diagram.time_list)
            vertices = tuple(_resolve_vertex(bindings.vertex_map, row) for row in diagram.vertex_list)
            if isinstance(diagram, ModeBlockedDiagram):
                if bindings.usedNe != diagram.mode_extent:
                    raise ValueError("partition endpoint must equal the explicit requested Ne")
                chosen = []
                for vertex, selection in zip(vertices, diagram.mode_selections, strict=True):
                    identity = (id(vertex), selection)
                    if identity not in selected_vertices:
                        selected_vertices[identity] = vertex.with_mode_indices(selection)
                    chosen.append(selected_vertices[identity])
                vertices = tuple(chosen)
            vertices = bind_slot_vertices(vertices, bindings.propagator_map.values())
            _validate_configuration(bindings.configuration, vertices, bindings.propagator_map)
            data_term = glueball_term(registry, vertices, anchors, diagram.time_list)
            if data_term is not None:
                term_id = f"diagram:{diagram_index}:term:{len(all_terms)}"
                all_terms.append(data_term)
                term_ids.append(term_id)
                diagram_recipes[diagram] = OutputNode("term", TermContribution(term_id))
                continue

            for vertex in vertices:
                if id(vertex) not in vertex_options:
                    vertex_options[id(vertex)] = _vertex_choices(vertex), _vertex_state_options(vertex)
            choices = tuple(vertex_options[id(vertex)][0] for vertex in vertices)
            vertex_state_options = tuple(vertex_options[id(vertex)][1] for vertex in vertices)
            topology = graph_key(diagram.diagram.adjacency_matrix)
            variant_outputs = []
            for selected in product(*choices):
                # For each allowed endpoint-state combination, join point legs
                # over every explicitly registered pool before lowering components.
                for states in product(*vertex_state_options):
                    state_factor_options = tuple(
                        describe_vertex(vertex, anchor, _vertex_sector(states[index]),
                                        selected[index].indices)
                        for index, (vertex, anchor) in enumerate(zip(vertices, anchors, strict=True))
                    )
                    pool_by_leg, pool_specs = describe_point_pool_metadata(state_factor_options, states)
                    local_weights = describe_local_elemental_weights(
                        state_factor_options, states, pool_by_leg, pool_specs,
                        elemental_weighted=bindings.elemental_weighted)
                    sampling_scenes = expand_explicit_point_pools(
                        pool_by_leg, pool_specs, usedNp=bindings.usedNp,
                        groups_independent=bindings.groups_independent,
                        local_weights=local_weights,
                        point_equalities=describe_point_equalities(
                            state_factor_options, states, pool_by_leg, pool_specs),
                    )
                    for scene in sampling_scenes:
                        if bindings.elemental_weighted and scene.coefficient == 0:
                            continue
                        labels = {
                            (int(leg.split(":", 1)[0][1:]), leg.split(":", 1)[1]): label
                            for leg, label in scene.point_labels.items()
                        }
                        for original_factors in product(*state_factor_options):
                            for dilution_scene in expand_dilution_scenes(
                                    original_factors, vertices, elemental_weighted=bindings.elemental_weighted,
                                    groups_independent=bindings.groups_independent):
                                selected_factors = dilution_scene.factors
                                mode_classes = {}
                                for group_index, (legs, partition) in enumerate(dilution_scene.equalities):
                                    for leg, block in zip(legs, partition, strict=True):
                                        mode_classes[leg] = (group_index, block)
                                network_key = (topology, tuple((s["left"], s["right"]) for s in states),
                                               tuple(sorted(labels.items())), dilution_scene.equalities)
                                if network_key not in network_templates:
                                    network_templates[network_key] = lower_state_graph(
                                        diagram.diagram.adjacency_matrix, states, point_labels=labels,
                                        mode_pools=mode_classes,
                                    )
                                networks = network_templates[network_key]
                                if not networks:
                                    raise ValueError("legacy Diagram lowering produced no contraction network")
                                selected_networks = tuple(lower_complete_diagonals(
                                    bind_dilution_equalities(network, dilution_scene), selected_factors)
                                    for network in networks)
                                edge_keys = {}
                                for network_index, network in enumerate(selected_networks):
                                    for request_index, request in enumerate(network.requests):
                                        if request.kind != "propagator":
                                            continue
                                        edge_label = request.tensor_label
                                        if edge_label >= len(diagram.propagator_list):
                                            raise ValueError(f"propagator label {edge_label} is outside Diagram.propagator_list")
                                        propagator_symbol = diagram.propagator_list[edge_label]
                                        prop_reader = _lookup_propagator(bindings.propagator_map, propagator_symbol)
                                        source_factor = selected_factors[request.source_vertex]
                                        sink_factor = selected_factors[request.sink_vertex]
                                        edge_keys[(network_index, request_index)] = register_propagator(
                                            prop_reader, propagator_symbol, source_factor,
                                            sink_factor, request.sector,
                                            source_role=diagram.time_list[request.source_vertex],
                                            sink_role=diagram.time_list[request.sink_vertex],
                                        )

                                network_term_ids = []
                                for network_index, network in enumerate(selected_networks):
                                    keys = []
                                    retained_requests = []
                                    sparse_vertices = []
                                    for request_index, request in enumerate(network.requests):
                                        if request.kind == "propagator":
                                            keys.append(edge_keys[(network_index, request_index)])
                                            retained_requests.append(request)
                                        else:
                                            descriptor = selected_factors[request.vertex_index]
                                            meson = vertices[request.vertex_index]
                                            if descriptor.tensor.sparse_layout == "coo":
                                                if request.sector != "p2p":
                                                    raise NotImplementedError(
                                                        "sparse elemental sources currently support only P2P vertices"
                                                    )
                                                metadata = meson.describe_sparse_record(descriptor)
                                                if (descriptor.tensor.point_shape != metadata.point_shape
                                                        or descriptor.tensor.point_diagonal != metadata.point_diagonal
                                                        or descriptor.tensor.structure != metadata.structure
                                                        or descriptor.tensor.nnz_upper != metadata.nnz
                                                        or descriptor.tensor.dtype != metadata.value_dtype
                                                        or descriptor.tensor.data_identity != metadata.data_identity
                                                        or descriptor.tensor.selection_identity != metadata.selection_identity):
                                                    raise ValueError(
                                                        "P2P tensor metadata and sparse record metadata disagree"
                                                    )
                                                sparse_identity = (
                                                    id(meson), descriptor.term_index,
                                                    metadata.data_identity, metadata.selection_identity,
                                                    descriptor.dagger,
                                                )
                                                record_key = add_sparse_record(
                                                    sparse_identity, metadata, meson.reader,
                                                    lambda m=meson, d=descriptor: m.read_factor_block(d),
                                                    lambda raw, m=meson, d=descriptor:
                                                        m.release_factor_block(d, raw),
                                                    lambda raw, m=meson, d=descriptor:
                                                        m.materialize_sparse_bridge_part(d, raw),
                                                    descriptor,
                                                )
                                                sparse_vertices.append((request.vertex_index, request,
                                                                        descriptor, metadata, record_key))
                                                continue
                                            key = register_dense_vertex(meson, descriptor,
                                                time_role=diagram.time_list[request.vertex_index])
                                            keys.append(key)
                                            retained_requests.append(request)

                                    if sparse_vertices:
                                        if len(sparse_vertices) > 2:
                                            raise NotImplementedError(
                                                "direct lowering supports at most two sparse P2P currents per network"
                                            )
                                        current_legs = {
                                            (vertex_index, side)
                                            for vertex_index, _request, _descriptor, _metadata, _record in sparse_vertices
                                            for side in ("field", "bar")
                                        }
                                        point_legs = {
                                            (vertex, side)
                                            for vertex in network.vertex_indices
                                            for side, state_side in (("field", "left"), ("bar", "right"))
                                            if states[vertex][state_side] == "p"
                                        }
                                        if len(sparse_vertices) > 1 and point_legs != current_legs:
                                            raise NotImplementedError(
                                                "sparse P2P bridges require their point legs to be the only "
                                                "sampled legs in that network"
                                            )
                                        term_index = len(all_terms)
                                        bridge_pairs = []
                                        for (vertex_index, _request, descriptor, metadata,
                                             record_key) in sparse_vertices:
                                            outgoing = [
                                                (i, request) for i, request in enumerate(network.requests)
                                                if request.kind == "propagator"
                                                and request.source_vertex == vertex_index
                                            ]
                                            incoming = [
                                                (i, request) for i, request in enumerate(network.requests)
                                                if request.kind == "propagator"
                                                and request.sink_vertex == vertex_index
                                            ]
                                            if len(outgoing) != 1 or len(incoming) != 1:
                                                raise NotImplementedError(
                                                    "sparse P2P bridge requires one incoming and one outgoing propagator"
                                                )
                                            field_position, field_request = outgoing[0]
                                            bar_position, bar_request = incoming[0]
                                            field_key = edge_keys[(network_index, field_position)]
                                            bar_key = edge_keys[(network_index, bar_position)]
                                            field_axes = _propagator_point_endpoint_axes(
                                                field_request.sector, "source",
                                            )
                                            bar_axes = _propagator_point_endpoint_axes(
                                                bar_request.sector, "sink",
                                            )
                                            if field_position < bar_position:
                                                bridge_left, bridge_right = field_key, bar_key
                                                left_axes, right_axes = field_axes, bar_axes
                                                transpose_bridge = False
                                            else:
                                                bridge_left, bridge_right = bar_key, field_key
                                                left_axes, right_axes = bar_axes, field_axes
                                                transpose_bridge = True
                                            raw_point_shape = descriptor.tensor.point_shape
                                            if (raw_point_shape is None
                                                    or len(raw_point_shape) != 2
                                                    or len(descriptor.tensor.shape) != 4
                                                    or descriptor.tensor.shape[1] != 3
                                                    or descriptor.tensor.shape[3] != 3):
                                                raise ValueError(
                                                    "sparse P2P direct bridge requires explicit two-ended point extents "
                                                    "and color extents of three"
                                                )
                                            if tuple(metadata.point_shape) != tuple(raw_point_shape):
                                                raise ValueError(
                                                    "sparse P2P record point extents differ from the elemental view"
                                                )
                                            # The sparse materializer first changes raw storage axes into
                                            # logical field/bar order for a source dagger. The C bridge
                                            # records these exact COO-axis extents; transpose_bridge then
                                            # maps them onto the chosen contraction operand order.
                                            selected_point_shape = (descriptor.tensor.shape[0],
                                                                    descriptor.tensor.shape[2])
                                            record_point_shape = (tuple(reversed(selected_point_shape))
                                                                  if descriptor.dagger
                                                                  else selected_point_shape)
                                            used_np, used_np_right = record_point_shape
                                            bridge_pairs.append((
                                                frozenset((bridge_left, bridge_right)),
                                                bridge_left, bridge_right, record_key,
                                                int(used_np), int(used_np_right),
                                                metadata.nnz, left_axes, right_axes, transpose_bridge,
                                                labels[(vertex_index, "field")]
                                                == labels[(vertex_index, "bar")],
                                            ))
                                        if len({item[0] for item in bridge_pairs}) != 1:
                                            raise NotImplementedError(
                                                "direct two-current cycle currently requires both P2P vertices "
                                                "to bridge the same propagator pair"
                                            )
                                        for (_, bridge_left, bridge_right, record_key, used_np,
                                             used_np_right, nnz_upper, left_axes, right_axes, transpose_bridge,
                                             same_point) in bridge_pairs:
                                            sparse_bridges.append(SparseBridgeSpec(
                                                bridge_left, bridge_right, record_key, used_np,
                                                nnz_upper, left_axes=left_axes, right_axes=right_axes,
                                                same_point=same_point, scene=term_index,
                                                transpose_bridge=transpose_bridge,
                                                used_np_right=used_np_right,
                                            ))
                                    term_equation = AxisPattern(
                                        tuple(request.subscripts for request in retained_requests), ()
                                    )
                                    term_id = f"diagram:{diagram_index}:term:{len(all_terms)}"
                                    all_terms.append(ContractionTerm(tuple(keys), term_equation))
                                    term_ids.append(term_id)
                                    network_term_ids.append(term_id)

                                # Independent fermion components multiply; scenes and
                                # state/current choices add. Put the statistical and
                                # operator coefficients in this explicit recipe once.
                                leaves = [OutputNode("term", TermContribution(term_id))
                                          for term_id in network_term_ids]
                                value = leaves[0] if len(leaves) == 1 else OutputNode("multiply", children=tuple(leaves))
                                # Whole vertices own their complex weights, so
                                # no later multiply converts exact sampling
                                # Fractions to the numeric output scalar domain.
                                coefficient = complex(scene.coefficient * dilution_scene.coefficient)
                                for item in selected_factors:
                                    if hasattr(item, 'factors'):
                                        continue
                                    weight = item.operator_term.coefficient
                                    coefficient *= weight.conjugate() if item.dagger else weight
                                if coefficient != 1:
                                    value = OutputNode("multiply", children=(
                                        OutputNode("constant", coefficient), value,
                                    ))
                                variant_outputs.append(value)

            if not variant_outputs:
                raise ValueError("Diagram expansion produced no state/current/scene terms")
            diagram_recipes[diagram] = (
                variant_outputs[0] if len(variant_outputs) == 1
                else OutputNode("add", children=tuple(variant_outputs))
            )

        if not all_terms:
            raise ValueError("nonempty Diagram set expanded to no ContractionTerms")
        visible = {key for term in all_terms for key in term.inputs}
        problem = ContractionProblem(
            {key: specs[key] for key in visible}, tuple(all_terms),
            {key: source for key, source in sources.items() if key in visible or key.startswith('sparse-record:')},
            sparse_bridges=tuple(sparse_bridges),
            single_read_regions=True,
        )


        batch_topology = (
            tuple((term.subscripts.normalized(), len(term.inputs)) for term in problem.terms),
            tuple((str(diagram), recipe) for diagram, recipe in diagram_recipes.items()),
        )
        return DiagramInputs(
            problem, blocks, {key: preparations[key] for key in problem.sources}, registry.loader(), coordinates, batch,
            tuple(term_ids), diagram_recipes=diagram_recipes,
            tensor_requests={key: value for key, value in tensor_requests.items() if key in visible},
            batch_bindings={key: value for key, value in batch_bindings.items() if key in visible},
            batch_topology=batch_topology,
            read_requests=registry.read_requests,
        )


def _coordinate_map(coordinates):
    if not coordinates.points:
        raise ValueError("Diagram coordinate metadata has no selected point")
    return dict(coordinates.points[0].values)


def _resolve_time(symbol, values):
    if symbol in values:
        value = values[symbol]
    elif isinstance(symbol, Integral) and not isinstance(symbol, bool):
        value = symbol
    else:
        raise ValueError(f"symbolic Diagram time {symbol!r} has no fixed coordinate binding")
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"time binding for {symbol!r} must be a nonnegative integer")
    return int(value)


def _resolve_vertex(vertex_map, row):
    if isinstance(vertex_map, Mapping):
        try:
            vertex = vertex_map[row]
        except KeyError as exc:
            raise KeyError(f"vertex_map has no bound object for {row!r}") from exc
    else:
        vertex = vertex_map(row)
        if vertex is None and callable(getattr(row, "describe_factors", None)):
            vertex = row
    if isinstance(vertex, Glueball):
        return vertex
    if not callable(getattr(vertex, "describe_factors", None)):
        raise TypeError("vertex_map values must expose Meson.describe_factors metadata")
    if not callable(getattr(vertex, "read_factor_block", None)) or not callable(
            getattr(vertex, "materialize_unweighted_factor", None)):
        raise TypeError("bound Meson must expose read_factor_block and unweighted materialization")
    return vertex


def _lookup_propagator(propagator_map, symbol):
    try:
        reader = propagator_map[symbol]
    except KeyError as exc:
        raise KeyError(f"propagator_map has no data reader for {symbol!r}") from exc
    if not callable(getattr(reader, "describe_block", None)) or not callable(
            getattr(reader, "block", None)):
        raise TypeError("propagator_map values must expose describe_block and block")
    return reader


def _referenced_propagators(graphs, mapping):
    """Use actual nonzero adjacency labels, including ordered baryon leg cells."""
    if not isinstance(mapping, Mapping):
        raise TypeError('Diagram binding requires a propagator_map mapping')
    def labels(value):
        if isinstance(value, bool):
            raise TypeError('propagator labels cannot be bool')
        if isinstance(value, Integral):
            if value < 0:
                raise ValueError('propagator labels must be nonnegative')
            return () if value == 0 else (int(value),)
        if isinstance(value, (list, tuple)):
            return tuple(label for child in value for label in labels(child))
        raise ValueError('invalid propagator adjacency entry')
    result = {}
    for adjacency, symbols in graphs:
        for label in labels(adjacency):
            if label >= len(symbols):
                raise ValueError('propagator adjacency label is outside its declared symbol table')
            symbol = symbols[label]
            if symbol not in result:
                result[symbol] = _lookup_propagator(mapping, symbol)
    return result


def _validate_configuration(configuration, vertices, propagator_map):
    readers = [getattr(vertex, "reader", None) for vertex in vertices]
    readers.extend(propagator_map.values())
    validate_vertex_time_bindings(vertices, tuple(propagator_map.values()))
    if configuration is None:
        return
    for reader in readers:
        cfg = getattr(reader, "cfg", None)
        if cfg is not None and cfg != configuration:
            raise ValueError("configuration disagrees with a bound elemental/propagator reader")


def _vertex_choices(vertex):
    operator = getattr(vertex, "operator", None)
    terms = tuple(getattr(operator, "terms", ()))
    if not terms:
        raise ValueError("bound Meson operator has no OperatorTerm definitions")
    probe = getattr(vertex, "term_time_offsets", None)
    grouped = {}
    for index in range(len(terms)):
        offsets = (0, 0) if not callable(probe) else tuple(int(v) for v in probe(index))
        if len(offsets) != 2:
            raise ValueError("Meson.term_time_offsets must return (bar_delta, field_delta)")
        grouped.setdefault(offsets, []).append(index)
    return tuple(_VertexChoice(offsets, tuple(indices)) for offsets, indices in grouped.items())


def _vertex_state_options(vertex):
    """Return explicitly requested endpoint states after checking B capability.

    Current operators retain their established four-sector decomposition.
    Ordinary operators remain V/V by default even when a composite reader owns
    more data. A caller may request a logical sector tuple on a Meson binding.
    """
    operator = getattr(vertex, "operator", None)
    declared = getattr(vertex, "available_sectors", None)
    available = declared() if callable(declared) else None
    state_by_sector = {sector: {"left": sector[0], "right": sector[2]}
                       for sector in ELEMENTAL_SECTORS}
    if available is not None:
        if not isinstance(available, tuple):
            raise TypeError("binding available_sectors must return a tuple or None")
        if not available:
            raise ValueError("binding declares no available elemental sectors")
        if any(not isinstance(sector, str) for sector in available):
            raise TypeError("binding available sectors must be strings")
        if len(set(available)) != len(available):
            raise ValueError("binding available sectors must be unique")
    if available is not None:
        unknown = tuple(sector for sector in available if sector not in state_by_sector)
        if unknown:
            raise ValueError(f"binding declares unsupported elemental sectors: {unknown!r}")

    requested = getattr(vertex, "requested_sectors", None)
    if requested is not None:
        if not isinstance(requested, tuple) or not requested:
            raise TypeError("requested_sectors must be a nonempty tuple or None")
        if any(not isinstance(sector, str) for sector in requested):
            raise TypeError("requested_sectors entries must be strings")
        if len(set(requested)) != len(requested):
            raise ValueError("requested_sectors entries must be unique")
        unknown = tuple(sector for sector in requested if sector not in state_by_sector)
        if unknown:
            raise ValueError(f"requested_sectors contains unsupported sectors: {unknown!r}")

    current = tuple(getattr(operator, "current_terms", ()))
    if current:
        if requested is not None and any("d" in sector for sector in requested):
            kinds = {kind for sector in requested for kind in (sector[0], sector[2])}
            required = tuple(sector for sector in ELEMENTAL_SECTORS
                             if sector[0] in kinds and sector[2] in kinds)
        else:
            required = (ELEMENTAL_SECTORS if available is not None and any("d" in sector for sector in available)
                        else ("v2v", "v2p", "p2v", "p2p"))
        if requested is not None and set(requested) != set(required):
            raise ValueError("current vertices require the complete four-sector decomposition or complete V/D/P endpoint decomposition")
        sectors = required
    else:
        sectors = ("v2v",) if requested is None else requested
    if available is not None:
        missing = tuple(sector for sector in sectors if sector not in available)
        if missing:
            raise ValueError(
                f"requested elemental sectors are not available from the binding: {missing!r}"
            )
    return tuple(state_by_sector[sector] for sector in sectors)


def _vertex_sector(state):
    return f"{state['left']}2{state['right']}"


def _propagator_point_endpoint_axes(sector, endpoint):
    """Return (spin, point, color) axes for one declared P endpoint."""
    if endpoint == "sink" and sector[0] == "p":
        return (0, 2, 3)
    if endpoint == "source" and sector[2] == "p":
        return (1, 4, 5) if sector[0] == "p" else (1, 3, 4)
    raise NotImplementedError(f"sparse P2P bridge needs a point-valued {endpoint} endpoint; propagator sector {sector!r} does not provide one")


__all__ = [
    "ContractionNetwork", "LegacyDiagramBinding", "StateSamplingVariant",
    "TensorRequest", "expand_state_sampling_variants", "lower_state_graph",
]
