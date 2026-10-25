"""Metadata-only binding for dense three-line baryon two-point Diagrams."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
import math
from numbers import Integral

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ArrayConstant,
    AxisPattern,
    BatchSpec,
    BaryonJoinSpec,
    CoordinateSpec,
    ContractionProblem,
    ContractionTerm,
    DiagramInputs,
    OutputNode,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorAxisMetadata,
    TensorDescription,
    TensorSpec,
    TensorPreparation,
    TermContribution,
    canonical_axis_binding_identity,
)
from lqcdweave.contraction.results.api import collect_diagrams
from lqcdweave.contraction.binding.api import (bind_slot_vertices, validate_vertex_time_bindings, map_edge_time_slot)
from lqcdweave.operators.correlator.api import ModeBlockedDiagram
from lqcdweave.contraction.binding.api import (
    Baryon,
    BaryonFactorMetadata,
    Meson,
    operator_term_semantics,
)
from .sampling_binding import logical_vertex_tail_shape
from .legacy_diagram import _referenced_propagators
from .tensor_description import (
    make_tensor_description,
    point_pool_bindings,
    stable_binding_value,
)
from .physical_inputs import PhysicalInputRegistry
from .glueball import glueball_term
from .baryon_factor_network import (RowFactor, entry_row_coefficients, factor_row_coefficients,
                                    merged_coefficients, plan_factor_network, auto_row_lowering)

#: Lowerings of a whole Baryon vertex: one closed join (only when declared),
#: or an opened network whose spin constants are split into two parts, into
#: their nonzero entries or kept whole. Each forced form applies to every
#: family; "auto" chooses per row family.
BARYON_VERTEX_LOWERINGS = ("auto", "dense", "joined", "nonzero", "pair")
from lqcdweave.contraction.binding.api import Glueball


_PROPAGATOR_AXES = {
    "vsv": ("sink_spin", "source_spin", "sink_mode", "source_mode"),
}


@dataclass(frozen=True, slots=True)
class _QuarkEdge:
    propagator: int
    source: int
    sink: int
    source_slot: int
    sink_slot: int


@dataclass(frozen=True, slots=True)
class _VertexRole:
    kind: str
    orientation: str


class BaryonDiagramBinding:
    """Bind dense-V baryon/multi-meson quark graphs to native array readers.

    ``sector_scope="dense-v"`` explicitly requests only VSV propagators and
    V2V meson/current vertices. Baryon vertices use VVV factors. Dense
    ``ContractionTerm`` objects reuse the shared fixed-order executor for
    ``optimize="none"``.
    """

    def __init__(self, *, sector_scope: str, coordinate_slabs=False, vertices="auto"):
        if sector_scope != "dense-v":
            raise ValueError("BaryonDiagramBinding currently supports sector_scope='dense-v'")
        if vertices not in BARYON_VERTEX_LOWERINGS:
            raise ValueError(f"baryon vertices must be one of {BARYON_VERTEX_LOWERINGS}")
        self.coordinate_slabs = coordinate_slabs
        # "joined" keeps the closed whole-join lowering only when declared;
        # every other value opens every request.
        self.vertices = vertices
        self.sector_scope = sector_scope
        # The "auto" form of each opened row family, read by calculate's selection.
        self.row_choices = []

    def describe(
        self,
        correlator,
        *,
        bindings,
        coordinates: CoordinateSpec,
        batch: BatchSpec,
    ) -> DiagramInputs:
        if coordinates.scan_shape:
            raise NotImplementedError("baryon Diagram scans are not wired in this first-stage binding")
        if batch.size != "auto" or batch.axis is not None:
            raise NotImplementedError("baryon Diagram execution batching is not wired")
        if bindings.usedNp is not None or bindings.groups_independent is not None:
            raise NotImplementedError("point sampling metadata is outside dense-v scope")
        if not isinstance(bindings.propagator_map, Mapping):
            raise TypeError("BaryonDiagramBinding requires a propagator_map mapping")
        if bindings.vertex_map is None:
            raise TypeError("BaryonDiagramBinding requires a vertex_map mapping or callable")

        diagrams = collect_diagrams(correlator)
        if not diagrams:
            raise ValueError("BaryonDiagramBinding requires at least one Diagram")
        bindings = replace(bindings, propagator_map=_referenced_propagators(
            ((diagram.diagram.adjacency_matrix, diagram.propagator_list) for diagram in diagrams),
            bindings.propagator_map))
        time_values = _coordinate_map(coordinates)

        specs = {}
        sources = {}
        blocks = {}
        preparations = {}
        tensor_requests = {}
        materializers = {}
        block_readers = {}
        block_metadata = {}
        physical_blocks = {}
        logical_tensors = {}
        spin_entries_cache = {}
        propagator_metadata_cache = {}
        terms = []
        baryon_joins = []
        term_ids = []
        diagram_recipes = {}
        backend_name = None
        selected_vertices = {}
        exchange_spaces = {}
        registry = PhysicalInputRegistry(bindings, coordinate_values=time_values)
        registry.preparations, registry.block_metadata = preparations, block_metadata
        registry.specs, registry.sources, registry.blocks = specs, sources, blocks
        registry.tensor_requests = tensor_requests

        def get_block(metadata, reader, family, read_callback):
            source_identity = metadata.data_identity
            if source_identity is None:
                source_identity = f"runtime-object:{id(reader)}"
            identity = (family, source_identity, metadata.selection_identity)
            block_key = physical_blocks.get(identity)
            raw_bytes = math.prod(metadata.shape) * metadata.dtype.itemsize
            if block_key is None:
                block_key = f"baryon-block:{len(blocks)}"
                blocks[block_key] = StorageBlock(
                    dataset=f"{family}:{source_identity}",
                    selection=metadata.selection_identity,
                    read_bytes=raw_bytes,
                    host_bytes=raw_bytes if backend_name == "numpy" else 0,
                    device_bytes=raw_bytes if backend_name == "cupy" else 0,
                )
                physical_blocks[identity] = block_key
                block_readers[block_key] = read_callback
                block_metadata[block_key] = metadata
            else:
                previous = block_metadata[block_key]
                if (previous.shape != metadata.shape or previous.dtype != metadata.dtype
                        or previous.axes != metadata.axes or blocks[block_key].read_bytes != raw_bytes
                        or canonical_axis_binding_identity(previous.axis_bindings)
                        != canonical_axis_binding_identity(metadata.axis_bindings)):
                    raise ValueError(
                        "shared baryon block identity has inconsistent axes, shape, dtype, or axis bindings"
                    )
            return block_key

        def add_tensor(
            logical_identity,
            metadata,
            reader,
            family,
            read_callback,
            materialize_callback,
            *,
            output_shape=None,
            output_dtype=None,
            workspace_bytes=0,
            output_axes=None,
            transform=(),
            description_semantics=None,
            description_bindings=None,
            read_arguments=None,
            read_selection=None,
            read_coordinates=(),
            read_coordinate_tables=(),
        ):
            key = logical_tensors.get(logical_identity)
            shape = metadata.shape if output_shape is None else tuple(output_shape)
            dtype = metadata.dtype if output_dtype is None else np.dtype(output_dtype)
            if key is not None:
                if specs[key].shape != shape or specs[key].dtype != dtype:
                    raise ValueError("shared baryon logical tensor has inconsistent TensorSpec")
                expected_bindings = specs[key].description.bindings.get("axis_bindings", {})
                if (canonical_axis_binding_identity(expected_bindings)
                        != canonical_axis_binding_identity(metadata.axis_bindings)):
                    raise ValueError(
                        "shared baryon logical tensor has inconsistent axis bindings"
                    )
                return key

            block_key = get_block(metadata, reader, family, read_callback)
            key = f"baryon-tensor:{len(specs)}"
            actual_bindings = {
                "configuration": stable_binding_value(
                    bindings.configuration if bindings.configuration is not None
                    else getattr(reader, "cfg", None)
                ),
            }
            if description_bindings:
                actual_bindings.update(description_bindings)
            description = make_tensor_description(
                metadata,
                role=family,
                source_bindings=(block_key,),
                axes=output_axes,
                transform=transform,
                semantics=description_semantics,
                bindings=actual_bindings,
            )
            spec = TensorSpec(shape, dtype, description=description)
            block = blocks[block_key]
            specs[key] = spec
            sources[key] = SourceSpec(
                region=block_key,
                region_bytes=block.read_bytes,
                host_region_bytes=block.host_bytes,
                snapshot_bytes=math.prod(shape) * dtype.itemsize,
                metadata={"family": family, "block": block_key},
            )
            preparations[key] = SourcePreparation(
                (block_key,),
                workspace_host_bytes=workspace_bytes if backend_name == "numpy" else 0,
                workspace_device_bytes=workspace_bytes if backend_name == "cupy" else 0,
                kernel_calls=1 if workspace_bytes else 0,
                dtype=dtype.name,
                recipe=(TensorPreparation('identity', (block_key,)) if family == 'propagator' else None),
            )
            if read_arguments is not None:
                registry.register_read(key, reader, "propagator" if family == "propagator" else "elemental",
                                       read_arguments, read_selection or {},
                                       role=(f'propagator:{description_semantics["symbol"]}'
                                             if family == 'propagator' else None),
                                       coordinates=read_coordinates,
                                       coordinate_tables=(read_coordinate_tables()
                                           if callable(read_coordinate_tables) else read_coordinate_tables))
                if self.coordinate_slabs:
                    from .baryon_batching import register_baryon_slab
                    register_baryon_slab(registry, key, reader, family, read_arguments,
                        read_selection or {}, materialize_callback, workspace_bytes)
            tensor_requests[key] = logical_identity
            materializers[key] = materialize_callback
            logical_tensors[logical_identity] = key
            return key

        def spin_entries_once(vertex, description):
            # The spin is independent of the numerical Phi/propagator blocks.
            # Scope this small constant to one expansion; Baryon bindings can
            # be rebound for the next task or configuration.
            identity = (id(vertex), id(vertex.reader), id(description.operator_term),
                        description.term_index, vertex.spin_basis, description.dagger,
                        description.field_order, description.endpoint_times)
            cached = spin_entries_cache.get(identity)
            if cached is None:
                entries = vertex.describe_spin_entries(description)
                # Keep the term alive: legacy Operator.terms may create new
                # records, whose id could otherwise be reused within describe.
                spin_entries_cache[identity] = (description.operator_term, entries)
                return entries
            return cached[1]

        def propagator_metadata_once(reader, sink_time, source_time, selection):
            # The same physical propagator header is revisited by many Wick
            # graphs. Keep only its immutable selection metadata in this
            # expansion; endpoint compatibility still checks each graph edge.
            identity = (id(reader), sink_time, source_time,
                        tuple(selection.get("usedNe", ())),
                        tuple(sorted((axis, tuple(indices)) for axis, indices in
                                     selection.get("mode_indices", {}).items())))
            cached = propagator_metadata_cache.get(identity)
            if cached is None:
                metadata = reader.describe_block(sink_time, source_time, "vsv", **selection)
                propagator_metadata_cache[identity] = (reader, metadata)
                return metadata
            return cached[1]

        def add_spin_tensor(vertex, description):
            entries = spin_entries_once(vertex, description)
            identity = (
                "baryon-spin", id(vertex), description.term_index,
                vertex.spin_basis, description.dagger, entries,
            )
            key = logical_tensors.get(identity)
            if key is not None:
                return key
            key = f"baryon-tensor:{len(specs)}"
            nbytes = 64 * np.dtype(np.complex128).itemsize
            tensor_description = TensorDescription(
                axes=("spin_slot0", "spin_slot1", "spin_slot2"),
                role="baryon-spin",
                transform=(
                    ("operation", "three-spin-coefficients"),
                    ("spin_basis", vertex.spin_basis),
                    ("source_adjoint", description.dagger),
                    ("field_order", tuple(description.field_order)),
                ),
                semantics={
                    "entries": entries,
                    "operator_term": operator_term_semantics(description.operator_term),
                    "coefficient_owner": "OutputRecipe",
                },
            )
            specs[key] = TensorSpec((4, 4, 4), np.complex128, description=tensor_description)
            sources[key] = SourceSpec(
                region=f"{key}:spin-constant", region_bytes=0, read_bytes=0,
                snapshot_bytes=nbytes, metadata={"family": "baryon-spin"},
            )
            preparations[key] = SourcePreparation(
                (), workspace_host_bytes=nbytes,
                workspace_device_bytes=nbytes if backend_name == "cupy" else 0,
                kernel_calls=1, dtype="complex128",
                recipe=vertex.describe_spin_preparation(description, entries=entries),
            )
            tensor_requests[key] = identity
            materializers[key] = lambda _, v=vertex, d=description, e=entries: v.materialize_spin_factor(d, entries=e)
            logical_tensors[identity] = key
            return key

        def vertex_tensor_adder(role, vertex):
            def add_vertex_tensor(*args, **kwargs):
                # A symbolic time role follows its coordinate axis on rebinding.
                if not isinstance(role, Integral):
                    kwargs['read_coordinates'] = ((2, str(role), 0, int(vertex.reader.Lt)),)
                return add_tensor(*args, **kwargs)
            return add_vertex_tensor

        def add_meson_vertex(vertex, descriptor, add_vertex_tensor):
            from lqcdweave.contracts.correlator.api import MesonCoreMetadata
            from .vertex_sum import register_vertex_sum
            descriptions = descriptor.factors if isinstance(descriptor, MesonCoreMetadata) else (descriptor,)
            members = tuple(_add_selected_vertex_tensor(add_vertex_tensor, add_spin_tensor, spin_entries_once,
                vertex, item, configuration=bindings.configuration) for item in descriptions)
            if not isinstance(descriptor, MesonCoreMetadata):
                preparations[members[0]] = replace(preparations[members[0]], recipe=
                    vertex.describe_factor_preparation(descriptor, preparations[members[0]].blocks[0]))
                return members[0]
            for member, item in zip(members, descriptions):
                preparations[member] = replace(preparations[member], recipe=
                    vertex.describe_factor_preparation(item, preparations[member].blocks[0],
                                                       compact_structure=False,
                                                       embed_structure=descriptor.mixed_structure))
            return register_vertex_sum(members, descriptions, specs=specs,
                sources=sources, blocks=blocks, preparations=preparations,
                tensor_requests=tensor_requests, tensor_metadata=descriptor.tensor)

        # Every request opens unless "joined" is declared.
        network = (plan_factor_network(diagrams, bindings, time_values)
                   if self.vertices != "joined" else None)
        if network is not None:
            backend_name = _backend_name(tuple(network.rows.values()))
            _validate_configuration(bindings.configuration, tuple(network.rows.values()),
                                    bindings.propagator_map)
            packed_terms, term_nodes, checked = {}, {}, set()
            coefficient_inputs = {}

            def add_constant(identity, values, axes, operation, members):
                key = logical_tensors.get(identity)
                if key is not None:
                    return key
                key = f"baryon-tensor:{len(specs)}"
                specs[key] = TensorSpec(values.shape, np.complex128, description=TensorDescription(
                    axes=axes,
                    role="baryon-spin",
                    transform=(("operation", operation),
                               ("spin_basis", members[0].spin_basis),
                               ("source_adjoint", members[0].dagger)),
                    # The exact values are the identity: equal-shaped families
                    # (e.g. two source vertices) must never share a description.
                    semantics={"rows": len(members), "coefficient_owner": "factor",
                               "entries": tuple((*map(int, index), complex(values[index]))
                                                for index in zip(*np.nonzero(values)))}))
                sources[key] = SourceSpec(region=f"{key}:spin-constant", region_bytes=0, read_bytes=0,
                    snapshot_bytes=values.nbytes, metadata={"family": "baryon-spin"})
                preparations[key] = SourcePreparation((), workspace_host_bytes=values.nbytes,
                    workspace_device_bytes=values.nbytes if backend_name == "cupy" else 0,
                    kernel_calls=1, dtype="complex128",
                    recipe=TensorPreparation("constant", constants=(ArrayConstant.from_array(values),)))
                tensor_requests[key] = identity
                materializers[key] = lambda _, v=values, xp=members[0].xp: xp.asarray(v)
                logical_tensors[identity] = key
                return key

            def add_coefficients(rows, anchor, part):
                # One exact row-stacked constant per part: its weighted spin parts share Phi.
                members = tuple(network.rows[row] for row in rows)
                identity = ("baryon-row-coefficients", tuple(id(v) for v in members), anchor, part)
                cached = coefficient_inputs.get(identity)
                if cached is not None:
                    return cached
                values = np.stack([merged_coefficients(vertex, network.parts[(row, anchor)][part],
                                                       spin_entries_once)
                                   for row, vertex in zip(rows, members)])
                # "auto" applies the per-family rule; calculate also plans the uniform forms.
                choice = auto_row_lowering(values)
                self.row_choices.append(choice)
                lowering = choice if self.vertices == "auto" else self.vertices
                factor = factor_row_coefficients(values, bounded=False) if lowering == "pair" else None
                if lowering == "nonzero":
                    factor = entry_row_coefficients(values)
                    if factor is not None:
                        keys = (add_constant((*identity, "weight"), factor.weight,
                                             ("operator_row", "spin_entry"), "row-spin-entry-weights", members),
                                *(add_constant((*identity, "select", j), select, ("spin_entry", f"spin_slot{j}"),
                                               "spin-entry-selection", members)
                                  for j, select in enumerate(factor.select)))
                        cached = coefficient_inputs[identity] = (factor, keys)
                        return cached
                if factor is None:
                    keys = (add_constant(identity, values, ("operator_row", "spin_slot0", "spin_slot1",
                                                            "spin_slot2"),
                                         "row-merged-spin-coefficients", members),)
                else:
                    a, b = factor.pairs
                    rank = ("factor_rank",) if factor.rank > 1 else ()
                    keys = (add_constant((*identity, "pair"), factor.pair,
                                         (*rank, f"spin_slot{a}", f"spin_slot{b}"),
                                         "row-spin-pair-coefficients", members),)
                    if factor.row is not None:
                        keys += (add_constant((*identity, "row"), factor.row,
                                              ("operator_row", f"spin_slot{factor.slot}", *rank),
                                              "row-spin-slot-coefficients", members),)
                cached = coefficient_inputs[identity] = (factor, keys)
                return cached

            def lower_component(diagram, keys, anchors, component):
                """One ordinary term per choice of one part at each member vertex."""
                families = network.families[component.key]
                choices = tuple(range(len(network.parts[(families[slot][0], anchors[index])]))
                                for slot, index in enumerate(component.vertices))
                result = []
                for choice in product(*choices):
                    vertices = [network.rows[key] for key in keys]
                    descriptions = [None] * len(vertices)
                    vertex_keys = [None] * len(vertices)
                    factors, coefficient = {}, 1
                    for slot, index in enumerate(component.vertices):
                        rows = families[slot]
                        vertex = vertices[index] = network.rows[rows[0]]
                        part = network.parts[(rows[0], anchors[index])][choice[slot]]
                        add_vertex_tensor = vertex_tensor_adder(diagram.time_list[index], vertex)
                        if not isinstance(vertex, Baryon):
                            descriptions[index] = part
                            vertex_keys[index] = add_meson_vertex(vertex, part, add_vertex_tensor)
                            coefficient *= _factor_coefficient(part)
                            continue
                        phi = descriptions[index] = part[0]
                        elemental = _add_vertex_tensor(add_vertex_tensor, vertex, phi, _vertex_workspace(phi),
                                                       configuration=bindings.configuration)
                        preparations[elemental] = replace(preparations[elemental],
                            recipe=vertex.describe_elemental_preparation(phi, preparations[elemental].blocks[0]))
                        factors[slot] = add_coefficients(rows, anchors[index], choice[slot])
                        vertex_keys[index] = (factors[slot][1][0], elemental, None)
                    inputs, equation, joins = _lower_dense_v_component(
                        diagram, tuple(vertices), tuple(descriptions), tuple(vertex_keys),
                        component.edges, bindings, add_tensor, spin_entries_once,
                        propagator_metadata_once, scene=len(terms))
                    inputs, subscripts, output = list(inputs), list(equation.inputs), []
                    first_row = 1 + max(label for axes in subscripts for label in axes)
                    # Each low-rank split or entry split contracts its own axis.
                    first_rank = first_row + len(component.vertices)
                    by_name = {join.name: join for join in joins}
                    for slot, index in enumerate(component.vertices):
                        if slot not in factors:  # a Meson part carries no row axis
                            continue
                        join = by_name[f"baryon-vertex:{index}"]
                        factor, factor_keys = factors[slot]
                        labels = join.spin_labels
                        if factor is None:
                            inputs.append(factor_keys[0])
                            subscripts.append((first_row + slot, *labels))
                            output.append(first_row + slot)
                        elif not isinstance(factor, RowFactor):
                            # Nonzero entries: the entry axis joins the weight and three selections.
                            entry = first_rank + slot
                            inputs.extend(factor_keys)
                            subscripts.append((first_row + slot, entry))
                            subscripts.extend((entry, label) for label in labels)
                            output.append(first_row + slot)
                        else:
                            rank = (first_rank + slot,) if factor.rank > 1 else ()
                            inputs.append(factor_keys[0])
                            subscripts.append((*rank, *(labels[k] for k in factor.pairs)))
                            if factor.row is None:
                                # The row is that slot's spin; R selects and scales it.
                                output.append(labels[factor.slot])
                            else:
                                inputs.append(factor_keys[1])
                                subscripts.append((first_row + slot, labels[factor.slot], *rank))
                                output.append(first_row + slot)
                        inputs.append(join.elemental)
                        subscripts.append(join.mode_labels)
                    term_id = f"baryon-factor-network:{len(terms)}"
                    terms.append(ContractionTerm(tuple(inputs), AxisPattern(tuple(subscripts), tuple(output))))
                    term_ids.append(term_id)
                    term_nodes[term_id] = OutputNode("term", TermContribution(term_id))
                    result.append((term_id, tuple((slot, factors[slot][0]) for slot in sorted(factors)),
                                   coefficient))
                return tuple(result)

            def component_node(packed, selection):
                nodes = []
                for term_id, factors, scale in packed:
                    chosen = []
                    for slot, factor in factors:
                        row = selection[slot]
                        if isinstance(factor, RowFactor) and factor.row is None:
                            scale *= factor.factors[row]
                            row = factor.spins[row]
                        chosen.append(row)
                    node = (OutputNode("select", tuple(chosen), (term_nodes[term_id],)) if factors
                            else term_nodes[term_id])
                    nodes.append(node if scale == 1 else
                                 OutputNode("multiply", children=(OutputNode("constant", scale), node)))
                return nodes[0] if len(nodes) == 1 else OutputNode("add", children=tuple(nodes))

            for diagram, (anchors, keys, components) in zip(diagrams, network.diagrams, strict=True):
                for index, row in enumerate(diagram.vertex_list):
                    item = keys[index], anchors[index]
                    if item in checked or not isinstance(network.rows[keys[index]], Baryon):
                        continue
                    checked.add(item)
                    factors = tuple(description for part in network.parts[item] for description in part)
                    symmetry = getattr(row, "quark_symmetry", ())
                    if symmetry:
                        _validate_exchange_endpoints(symmetry, factors)
                    exchange = getattr(row, "quark_exchange", None)
                    if exchange is not None:
                        _record_exchange_spaces(exchange_spaces, exchange, anchors[index],
                                                network.rows[keys[index]].dagger, factors)
                if components is None:
                    term_id = f"glueball:{len(terms)}"
                    terms.append(glueball_term(registry, (network.rows[keys[0]],), anchors, diagram.time_list,
                                               add_tensor=add_tensor))
                    term_ids.append(term_id)
                    diagram_recipes[diagram] = OutputNode("term", TermContribution(term_id))
                    continue
                nodes = []
                for component in components:
                    packed = packed_terms.get(component.key)
                    if packed is None:
                        packed = packed_terms[component.key] = lower_component(diagram, keys, anchors, component)
                    nodes.append(component_node(packed, component.selection))
                diagram_recipes[diagram] = (nodes[0] if len(nodes) == 1
                                            else OutputNode("multiply", children=tuple(nodes)))

        for diagram_index, diagram in enumerate(diagrams if network is None else ()):
            count = len(diagram.vertex_list)
            if len(diagram.time_list) != count or len(diagram.diagram.adjacency_matrix) != count:
                raise ValueError("Diagram time, vertex, and adjacency counts disagree")
            if any(len(row) != count for row in diagram.diagram.adjacency_matrix):
                raise ValueError("Diagram adjacency matrix must be square")

            vertices = tuple(_resolve_dense_vertex(bindings.vertex_map, row)
                             for row in diagram.vertex_list)
            if isinstance(diagram, ModeBlockedDiagram):
                if bindings.usedNe != diagram.mode_extent:
                    raise ValueError("partition endpoint must equal the explicit requested Ne")
                if any(not isinstance(v, Baryon) for v in vertices):
                    raise NotImplementedError("baryon mode partitions require pure Baryon graphs")
                chosen = []
                for vertex, selection in zip(vertices, diagram.mode_selections, strict=True):
                    identity = (id(vertex), selection)
                    if identity not in selected_vertices:
                        selected_vertices[identity] = vertex.with_mode_indices(selection)
                    chosen.append(selected_vertices[identity])
                vertices = tuple(chosen)
            vertices = bind_slot_vertices(vertices, bindings.propagator_map.values())
            diagram_backend = _backend_name(vertices)
            if backend_name is None:
                backend_name = diagram_backend
            elif backend_name != diagram_backend:
                raise ValueError("all Diagrams in one request must use the same array backend")
            _validate_configuration(bindings.configuration, vertices, bindings.propagator_map)
            anchors = tuple(_resolve_time(value, time_values) for value in diagram.time_list)
            data_term = glueball_term(registry, vertices, anchors, diagram.time_list,
                                      add_tensor=add_tensor)
            if data_term is not None:
                term_id = f"diagram:{diagram_index}:term:{len(terms)}"
                terms.append(data_term)
                term_ids.append(term_id)
                diagram_recipes[diagram] = OutputNode("term", TermContribution(term_id))
                continue
            edges, vertex_roles = _extract_dense_v_graph(
                diagram.diagram.adjacency_matrix, vertices,
            )
            meson_used_ne = (_meson_used_ne(bindings.usedNe)
                             if any(isinstance(vertex, Meson) for vertex in vertices) else bindings.usedNe)

            anchors = tuple(_resolve_time(value, time_values) for value in diagram.time_list)
            factor_groups_by_vertex = []
            for vertex_index, (vertex, role) in enumerate(
                    zip(vertices, vertex_roles, strict=True)):
                if isinstance(vertex, Baryon):
                    expected_dagger = role.orientation == "source"
                    if vertex.dagger != expected_dagger:
                        raise ValueError(
                            f"Baryon vertex {vertex_index} dagger state disagrees "
                            "with its graph orientation"
                        )
                    factors = vertex.describe_core_factors(anchors[vertex_index], usedNe=bindings.usedNe)
                    symmetry = getattr(diagram.vertex_list[vertex_index], "quark_symmetry", ())
                    if symmetry:
                        _validate_exchange_endpoints(symmetry, factors)
                    component = getattr(diagram.vertex_list[vertex_index], "quark_exchange", None)
                    if component is not None:
                        _record_exchange_spaces(exchange_spaces, component, anchors[vertex_index],
                                                vertex.dagger, factors)
                else:
                    if any("d" in sector for sector in (vertex.requested_sectors or ())):
                        raise NotImplementedError("mixed baryon graphs do not yet support D elemental sectors")
                    factors = vertex.describe_core_factors(
                        anchors[vertex_index], "v2v", usedNe=meson_used_ne,
                    )
                if not factors:
                    raise ValueError(f"vertex {vertex_index} operator has no dense-v factors")
                factor_groups_by_vertex.append(
                    _group_factor_descriptions(vertex, tuple(factors))
                )

            components = _components_from_edges(len(vertices), edges)
            diagram_nodes = []
            offset_combinations = tuple(product(*factor_groups_by_vertex))
            for offset_index, factor_groups in enumerate(offset_combinations):
                for combination_index, selected in enumerate(product(*factor_groups)):
                    vertex_keys = []
                    for vertex_index, (vertex, descriptor) in enumerate(zip(vertices, selected, strict=True)):
                        add_vertex_tensor = vertex_tensor_adder(diagram.time_list[vertex_index], vertex)
                        if not isinstance(vertex, Baryon):
                            vertex_keys.append(add_meson_vertex(vertex, descriptor, add_vertex_tensor))
                            continue
                        members = (_add_selected_vertex_tensor(add_vertex_tensor, add_spin_tensor,
                            spin_entries_once, vertex, descriptor, configuration=bindings.configuration),)
                        registered = members[0]
                        core = registered[2] if len(registered) == 3 else None
                        factors = getattr(descriptor, 'factors', (descriptor,))
                        elementals = (tuple(part.inputs[1] for part in core.components)
                                      if core is not None else (registered[1],))
                        for item, elemental in zip(factors, elementals, strict=True):
                            preparations[elemental] = replace(preparations[elemental], recipe=
                                vertex.describe_elemental_preparation(
                                    item, preparations[elemental].blocks[0]))
                        vertex_keys.append(registered)
                    vertex_keys = tuple(vertex_keys)
                    component_nodes = []
                    for component_index, component_edges in enumerate(components):
                        term_inputs, term_equation, component_joins = _lower_dense_v_component(
                            diagram,
                            vertices,
                            selected,
                            vertex_keys,
                            component_edges,
                            bindings,
                            add_tensor,
                            spin_entries_once,
                            propagator_metadata_once,
                            scene=len(terms),
                        )
                        term_id = (
                            f"baryon-diagram:{diagram_index}:offset:{offset_index}:"
                            f"combination:{combination_index}:component:{component_index}"
                        )
                        terms.append(ContractionTerm(term_inputs, term_equation))
                        baryon_joins.extend(component_joins)
                        term_ids.append(term_id)
                        component_nodes.append(OutputNode("term", TermContribution(term_id)))
                    value = (component_nodes[0] if len(component_nodes) == 1
                             else OutputNode("multiply", children=tuple(component_nodes)))
                    coefficient = math.prod(_factor_coefficient(item) for item in selected)
                    if coefficient != 1:
                        value = OutputNode("multiply", children=(
                            OutputNode("constant", coefficient), value,
                        ))
                    diagram_nodes.append(value)
            diagram_recipes[diagram] = (
                diagram_nodes[0] if len(diagram_nodes) == 1
                else OutputNode("add", children=tuple(diagram_nodes))
            )

        if not terms:
            raise ValueError("baryon Diagram expansion produced no ContractionTerms")
        visible = {key for term in terms for key in term.inputs}
        visible.update(key for join in baryon_joins for key in (
            join.core.inputs if join.core is not None else (join.spin, join.elemental)))
        problem = ContractionProblem(
            {key: specs[key] for key in visible},
            tuple(terms),
            {key: sources[key] for key in visible},
            backend=backend_name,
            single_read_regions=True,
            baryon_joins=tuple(baryon_joins),
            factorized_vertices=network is not None,
        )

        class Loader:
            input_readers = registry.reader_bindings
            read_requests = registry.read_requests
            def can_prefetch_block(self, key, block):
                return False

            def read_block(self, key, block):
                try:
                    read = block_readers[key]
                except KeyError as exc:
                    raise ValueError(f"no runtime reader was registered for baryon block {key}") from exc
                value = read()
                metadata = block_metadata[key]
                try:
                    shape, dtype = tuple(value.shape), np.dtype(value.dtype)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise TypeError("baryon block reader must return an ndarray-like value") from exc
                if shape != metadata.shape or dtype != metadata.dtype:
                    raise ValueError("baryon block shape/dtype differs from header metadata")
                if _array_backend_name(value) != backend_name:
                    raise TypeError("baryon block backend differs from the declared array backend")
                return value

            def close_block(self, key, value):
                return None

            def materialize(self, key, source, raw_blocks):
                dependencies = preparations[key].blocks
                if not dependencies:
                    return materializers[key](None)
                if len(dependencies) != 1:
                    raise ValueError("baryon materializer expects one raw block per logical tensor")
                return materializers[key](raw_blocks[dependencies[0]])

        loader = Loader()
        loader.specs, loader.preparations = specs, preparations
        loader._batch_bindings = registry.batch_bindings
        return DiagramInputs(
            problem,
            blocks,
            {key: preparations[key] for key in visible},
            loader,
            coordinates,
            batch,
            tuple(term_ids),
            diagram_recipes=diagram_recipes,
            tensor_requests={key: tensor_requests[key] for key in visible},
            read_requests=registry.read_requests,
            batch_bindings={key: value for key, value in registry.batch_bindings.items() if key in visible},
        )

def _meson_used_ne(used_ne):
    """The one Meson/current mode count of a mixed dense-V request."""
    if isinstance(used_ne, (tuple, list)):
        counts = tuple(used_ne)
        if (len(counts) != 3 or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral) for value in counts)):
            raise ValueError("mixed dense-V usedNe tuples require three integer Baryon slots")
        if len(set(counts)) != 1:
            raise ValueError(
                "mixed dense-V graphs require equal per-leg usedNe when a tuple is supplied"
            )
        return int(counts[0])
    if isinstance(used_ne, (bool, np.bool_)):
        raise TypeError("usedNe cannot be bool")
    return used_ne


def _record_exchange_spaces(spaces, component, anchor, dagger, factors):
    """One multiplet's components must read the same actual endpoint spaces."""
    _validate_exchange_endpoints(component.active, factors)
    family = (component.representation, component.names, component.flavors, anchor, dagger)
    for description in factors:
        metadata = description.tensor
        signature = (metadata.shape, description.endpoint_times,
                     canonical_axis_binding_identity(metadata.axis_bindings),
                     description.mode_indices, description.field_order)
        if family in spaces and spaces[family] != signature:
            raise ValueError("matrix quark exchange components require the same actual endpoint spaces")
        spaces[family] = signature


def _validate_exchange_endpoints(symmetry, factors):
    """A reduced Wick orbit needs compatible actual axes before any reads."""
    for description in factors:
        metadata = description.tensor
        order = description.field_order
        for permutation, _ in symmetry:
            for slot in range(3):
                left, right = order[slot], order[permutation[slot]]
                if left == right:
                    continue
                a, b = metadata.axes[left], metadata.axes[right]
                if (metadata.shape[left] != metadata.shape[right]
                        or description.endpoint_times[left] != description.endpoint_times[right]
                        or metadata.axis_bindings.get(a) != metadata.axis_bindings.get(b)
                        or (description.mode_indices is not None
                            and description.mode_indices[left] != description.mode_indices[right])):
                    raise ValueError(
                        "quark exchange symmetry requires the same actual basis, time and ordered mode selection on exchanged legs"
                    )


def _resolve_dense_vertex(vertex_map, row):
    if isinstance(vertex_map, Mapping):
        try:
            vertex = vertex_map[row]
        except KeyError as exc:
            raise KeyError(f"vertex_map has no bound Baryon/Meson for {row!r}") from exc
    else:
        vertex = vertex_map(row)
        if vertex is None and callable(getattr(row, "describe_factors", None)):
            vertex = row
    if not isinstance(vertex, (Baryon, Meson, Glueball)):
        raise TypeError("dense-V BaryonDiagramBinding requires Baryon or Meson vertex objects")
    return vertex


def _group_factor_descriptions(vertex, descriptions):
    """Group Meson terms that share their actual dense-V endpoints."""
    if isinstance(vertex, Baryon):
        return (tuple(descriptions),)
    groups = {}
    for description in descriptions:
        pools = tuple(sorted(description.point_pool_ids.items()))
        key = (
            description.anchor_time,
            description.bar_time,
            description.field_time,
            description.endpoint_offsets,
            pools,
        )
        groups.setdefault(key, []).append(description)
    return tuple(tuple(items) for items in groups.values())


def _components_from_edges(vertex_count, edges):
    """Partition a dense-V Diagram into independent quark-line components."""
    parent = list(range(vertex_count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for edge in edges:
        union(edge.source, edge.sink)

    grouped = {}
    for edge in edges:
        grouped.setdefault(find(edge.source), []).append(edge)
    return tuple(tuple(items) for items in grouped.values())


def _parse_adjacency_cell(cell, location):
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, (bool, np.bool_)):
        raise TypeError(f"adjacency entry {location} cannot be bool")
    if isinstance(cell, Integral):
        value = int(cell)
        if value < 0:
            raise ValueError(f"adjacency propagator ID at {location} cannot be negative")
        return ("scalar", value)
    if not isinstance(cell, (list, tuple)):
        raise TypeError(f"adjacency entry {location} must be an integer or nested slot list")
    if not cell:
        return ("vector", ())
    if all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
           for value in cell):
        values = tuple(int(value) for value in cell)
        if any(value < 0 for value in values):
            raise ValueError(f"adjacency propagator IDs at {location} cannot be negative")
        return ("vector", values)
    if not all(isinstance(row, (list, tuple)) for row in cell):
        raise ValueError(f"adjacency entry {location} must be a rectangular slot matrix")
    rows = []
    width = None
    for row in cell:
        current = tuple(row)
        if width is None:
            width = len(current)
        if len(current) != width or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in current):
            raise ValueError(f"adjacency slot matrix at {location} must be rectangular integers")
        values = tuple(int(value) for value in current)
        if any(value < 0 for value in values):
            raise ValueError(f"adjacency propagator IDs at {location} cannot be negative")
        rows.append(values)
    return ("matrix", tuple(rows))


def _extract_dense_v_graph(adjacency, vertices):
    """Keep the source/sink quark-slot coordinates while extracting graph edges."""
    count = len(vertices)
    if count < 2:
        raise ValueError("dense-V quark graphs need at least two vertices")
    cells = {}
    outgoing, incoming = [0] * count, [0] * count
    for source, row in enumerate(adjacency):
        for sink, cell in enumerate(row):
            parsed = _parse_adjacency_cell(cell, (source, sink))
            cells[(source, sink)] = parsed
            kind, value = parsed
            if kind == "scalar":
                edge_count = int(value != 0)
            elif kind == "vector":
                edge_count = sum(item != 0 for item in value)
            else:
                edge_count = sum(item != 0 for line in value for item in line)
            outgoing[source] += edge_count
            incoming[sink] += edge_count

    roles = []
    for index, vertex in enumerate(vertices):
        degrees = (outgoing[index], incoming[index])
        if isinstance(vertex, Baryon):
            if degrees == (3, 0):
                roles.append(_VertexRole("baryon", "source"))
            elif degrees == (0, 3):
                roles.append(_VertexRole("baryon", "sink"))
            else:
                raise ValueError(
                    f"Baryon vertex {index} has outgoing/incoming degrees {degrees}; "
                    "Baryons require source (3,0) or sink (0,3) topology"
                )
        elif isinstance(vertex, Meson):
            if degrees != (1, 1):
                raise ValueError(
                    f"Meson vertex {index} has outgoing/incoming degrees {degrees}; "
                    "Meson/current vertices require (1,1) topology"
                )
            roles.append(_VertexRole("meson", "meson"))
        else:
            raise TypeError(f"unsupported dense-V vertex type at index {index}")

    edges = []
    for (source, sink), (kind, value) in cells.items():
        if kind == "scalar":
            if value == 0:
                continue
            if roles[source].kind != "meson" or roles[sink].kind != "meson":
                raise ValueError("scalar adjacency cells are valid only between one-leg Meson vertices")
            edges.append(_QuarkEdge(value, source, sink, 0, 0))
            continue

        source_slots = 3 if roles[source].orientation == "source" else 1
        sink_slots = 3 if roles[sink].orientation == "sink" else 1
        if kind == "vector":
            if not any(value):
                continue
            if len(value) == source_slots and sink_slots == 1:
                pairs = ((source_slot, 0, prop) for source_slot, prop in enumerate(value))
            elif source_slots == 1 and len(value) == sink_slots:
                pairs = ((0, sink_slot, prop) for sink_slot, prop in enumerate(value))
            elif source_slots == 3 and sink_slots == 3 and len(value) == 3:
                # A flat three-edge cell has only one declared order, so its slot
                # pairing is the explicit positional identity convention.
                pairs = ((slot, slot, prop) for slot, prop in enumerate(value))
            else:
                raise ValueError(
                    f"one-dimensional adjacency cell {(source, sink)} cannot encode "
                    f"the {source_slots}x{sink_slots} endpoint slots"
                )
        else:
            if not any(prop for row in value for prop in row):
                continue
            if len(value) != source_slots or any(len(row) != sink_slots for row in value):
                raise ValueError(
                    f"adjacency slot matrix {(source, sink)} has shape "
                    f"{(len(value), len(value[0]) if value else 0)}, expected "
                    f"{(source_slots, sink_slots)}"
                )
            pairs = ((source_slot, sink_slot, prop)
                     for source_slot, row in enumerate(value)
                     for sink_slot, prop in enumerate(row))
        for source_slot, sink_slot, propagator in pairs:
            if propagator:
                edges.append(_QuarkEdge(
                    propagator, source, sink, source_slot, sink_slot,
                ))

    if not edges:
        raise ValueError("dense-V quark graph has no propagator edges")
    for index, role in enumerate(roles):
        outgoing_edges = [edge for edge in edges if edge.source == index]
        incoming_edges = [edge for edge in edges if edge.sink == index]
        if role.orientation == "source":
            slots = tuple(edge.source_slot for edge in outgoing_edges)
            if sorted(slots) != [0, 1, 2]:
                raise ValueError(f"Baryon source vertex {index} must use each quark slot once")
        elif role.orientation == "sink":
            slots = tuple(edge.sink_slot for edge in incoming_edges)
            if sorted(slots) != [0, 1, 2]:
                raise ValueError(f"Baryon sink vertex {index} must use each quark slot once")
        elif (len(outgoing_edges) != 1 or len(incoming_edges) != 1
              or outgoing_edges[0].source_slot != 0 or incoming_edges[0].sink_slot != 0):
            raise ValueError(f"Meson vertex {index} must have one incoming and one outgoing line")

    if not any(role.kind == "baryon" for role in roles):
        raise ValueError("BaryonDiagramBinding requires at least one Baryon vertex per Diagram")
    return tuple(edges), tuple(roles)


def _add_selected_vertex_tensor(add_tensor, add_spin_tensor, spin_entries_once,
                                vertex, description, *, configuration):
    if isinstance(vertex, Baryon):
        from lqcdweave.contracts.correlator.api import BaryonCoreMetadata
        if isinstance(description, BaryonCoreMetadata):
            spins, elementals = [], []
            for component in description.factors:
                elementals.append(_add_vertex_tensor(add_tensor, vertex, component, _vertex_workspace(component),
                                                     configuration=configuration))
                spins.append(add_spin_tensor(vertex, component))
            core = vertex.register_core(description.factors, spins, elementals,
                _spin_entries=tuple(spin_entries_once(vertex, component)
                                    for component in description.factors))
            return spins[0], elementals[0], core
        elemental_key = _add_vertex_tensor(
            add_tensor, vertex, description, _vertex_workspace(description),
            configuration=configuration,
        )
        return add_spin_tensor(vertex, description), elemental_key
    if description.logical_sector != "v2v":
        raise ValueError(f"unsupported Meson/current sector {description.logical_sector!r}")
    metadata = description.tensor
    tail_shape = _logical_meson_tail_shape(description)
    logical_identity = (
        "meson-vertex", id(vertex), description.term_index,
        description.logical_sector, description.dagger,
        metadata.selection_identity,
    )
    workspace = math.prod(metadata.shape) * metadata.dtype.itemsize
    permutations = {
        "v2v": (1, 0), "v2p": (1, 2, 0),
        "p2v": (2, 0, 1), "p2p": (2, 3, 0, 1),
    }
    permutation = (permutations[description.raw_sector] if description.dagger
                   else tuple(range(len(metadata.axes))))
    logical_tail_axes = tuple(metadata.axes[index] for index in permutation)
    operator_term = description.operator_term
    return add_tensor(
        logical_identity,
        metadata,
        vertex.reader,
        "meson-elemental",
        lambda v=vertex, d=description: v.read_factor_block(d),
        lambda raw, v=vertex, d=description: v.materialize_unweighted_factor(d, raw),
        read_arguments=(operator_term.component, operator_term.momentum, description.anchor_time, description.raw_sector),
        read_selection=dict(usedNe=description.usedNe),
        output_shape=(4, 4, *tail_shape),
        output_dtype=np.result_type(metadata.dtype, np.complex128),
        workspace_bytes=workspace,
        output_axes=("spin_field", "spin_bar", *logical_tail_axes),
        transform=(
            ("operation", "gamma-spin-spatial-product"),
            ("dagger", description.dagger),
            ("axis_permutation", permutation),
        ),
        description_semantics={
            "logical_sector": description.logical_sector,
            "raw_sector": description.raw_sector,
            "operator_term": operator_term_semantics(operator_term),
            "coefficient_owner": "OutputRecipe",
        },
        description_bindings={
            "configuration": stable_binding_value(configuration),
            "anchor_time": int(description.anchor_time),
            "bar_time": int(description.bar_time),
            "field_time": int(description.field_time),
            "bar_time_unwrapped": int(description.bar_time_unwrapped),
            "field_time_unwrapped": int(description.field_time_unwrapped),
            "endpoint_offsets": tuple(description.endpoint_offsets),
            "point_pool_ids": dict(description.point_pool_ids),
            "point_pool_metadata": point_pool_bindings(description),
            "usedNe": stable_binding_value(description.usedNe),
            "usedNp": stable_binding_value(description.usedNp),
            "phase_convention": vertex.phase_convention,
            "boundary": vertex.boundary,
        },
    )


def _factor_coefficient(description):
    from lqcdweave.contracts.correlator.api import MesonCoreMetadata
    if isinstance(description, MesonCoreMetadata):
        return 1
    if isinstance(description, BaryonFactorMetadata):
        return description.coefficient
    coefficient = description.operator_term.coefficient
    return coefficient.conjugate() if description.dagger else coefficient


def _logical_meson_tail_shape(description):
    if description.tensor.sparse_layout != "dense":
        raise NotImplementedError("dense-v baryon binding requires dense V2V Meson tensors")
    return logical_vertex_tail_shape(description)


def _endpoint_mode_extent(vertex, description, edge, *, source_end):
    if isinstance(vertex, Baryon):
        slot = edge.source_slot if source_end else edge.sink_slot
        axis = description.field_order[slot]
        return description.tensor.shape[axis]
    shape = _logical_meson_tail_shape(description)
    return shape[0] if source_end else shape[1]


def _endpoint_time(vertex, description, *, source_end):
    if isinstance(vertex, Baryon):
        return description.time
    return description.field_time if source_end else description.bar_time


def _lower_dense_v_component(
    diagram, vertices, descriptions, vertex_keys, edges, bindings, add_tensor,
    spin_entries_once, propagator_metadata_once, *, scene,
):
    """Build one fixed VSV contraction while retaining every endpoint slot."""
    edge_labels = {
        edge: ({"spin": 4 * index, "mode": 4 * index + 1},
               {"spin": 4 * index + 2, "mode": 4 * index + 3})
        for index, edge in enumerate(edges)
    }

    edge_keys = {}
    for edge in edges:
        if edge.propagator >= len(diagram.propagator_list):
            raise ValueError(
                f"propagator label {edge.propagator} is outside Diagram.propagator_list"
            )
        source_vertex, sink_vertex = vertices[edge.source], vertices[edge.sink]
        source_description, sink_description = descriptions[edge.source], descriptions[edge.sink]
        source_time = _endpoint_time(source_vertex, source_description, source_end=True)
        sink_time = _endpoint_time(sink_vertex, sink_description, source_end=False)
        selection = {
            "usedNe": (
                _endpoint_mode_extent(sink_vertex, sink_description, edge, source_end=False),
                _endpoint_mode_extent(source_vertex, source_description, edge, source_end=True),
            ),
        }
        source_modes = getattr(source_description, "mode_indices", None)
        sink_modes = getattr(sink_description, "mode_indices", None)
        if source_modes is not None or sink_modes is not None:
            if not isinstance(source_vertex, Baryon) or not isinstance(sink_vertex, Baryon):
                raise NotImplementedError("selected baryon legs require pure Baryon graphs")
            extents = selection["usedNe"]
            selection = {"mode_indices": {
                "sink_mode": (sink_modes[sink_description.field_order[edge.sink_slot]]
                              if sink_modes is not None else tuple(range(extents[0]))),
                "source_mode": (source_modes[source_description.field_order[edge.source_slot]]
                                if source_modes is not None else tuple(range(extents[1]))),
            }}
        else:
            extents = selection["usedNe"]
        symbol = diagram.propagator_list[edge.propagator]
        reader = _lookup_propagator(bindings.propagator_map, symbol)
        source_record = (getattr(source_vertex.reader, 'time_selection', None)
                         if isinstance(source_vertex, Baryon) else source_description.endpoint_time_selection)
        sink_record = (getattr(sink_vertex.reader, 'time_selection', None)
                       if isinstance(sink_vertex, Baryon) else sink_description.endpoint_time_selection)
        source_time = map_edge_time_slot(source_time, source_record, reader, 'source_time',
            physical_offsets=getattr(source_vertex, '_physical_offset_mapping', False))
        sink_time = map_edge_time_slot(sink_time, sink_record, reader, 'sink_time',
            physical_offsets=getattr(sink_vertex, '_physical_offset_mapping', False))
        metadata = propagator_metadata_once(reader, sink_time, source_time, selection)
        expected_shape = _expected_propagator_shape(
            "vsv", extents[0], extents[1], None, None,
        )
        _validate_propagator_metadata(metadata, "vsv", expected_shape)
        endpoint_checks = {
            "source": _compare_propagator_vertex_endpoint(
                metadata, "source", source_vertex, source_description,
                vertex_index=edge.source, slot=edge.source_slot,
                meson_endpoint="field",
            ),
            "sink": _compare_propagator_vertex_endpoint(
                metadata, "sink", sink_vertex, sink_description,
                vertex_index=edge.sink, slot=edge.sink_slot,
                meson_endpoint="bar",
            ),
        }
        logical_identity = (
            "propagator", id(reader), "vsv", metadata.selection_identity,
            tuple(_endpoint_check_identity(endpoint_checks[name])
                  for name in ("source", "sink")),
        )
        edge_keys[edge] = add_tensor(
            logical_identity,
            metadata,
            reader,
            "propagator",
            lambda r=reader, st=sink_time, rt=source_time, sel=selection:
                r.block(st, rt, "vsv", **sel),
            lambda raw: raw,
            read_arguments=(sink_time, source_time, "vsv"), read_selection=selection,
            read_coordinate_tables=lambda: _edge_coordinate_tables(diagram, edge, vertices, descriptions, reader),
            description_semantics={
                "sector": "vsv",
                "symbol": str(symbol),
                "time_convention": getattr(reader, "time_convention", None),
            },
            description_bindings={
                "configuration": stable_binding_value(bindings.configuration),
                "sink_time": int(sink_time),
                "source_time": int(source_time),
                "usedNe": tuple(int(value) for value in extents),
                "mode_indices": selection.get("mode_indices"),
                "endpoint_axis_bindings": endpoint_checks,
            },
        )

    component_vertices = tuple(sorted({
        vertex for edge in edges for vertex in (edge.source, edge.sink)
    }))
    vertex_subscripts = []
    ordered_vertex_keys = []
    joins = []
    for vertex_index in component_vertices:
        vertex = vertices[vertex_index]
        description = descriptions[vertex_index]
        if isinstance(vertex, Baryon):
            spin_labels = [None, None, None]
            mode_labels = [None, None, None]
            for edge in edges:
                if edge.source == vertex_index:
                    axis = description.field_order[edge.source_slot]
                    spin_labels[axis] = edge_labels[edge][1]["spin"]
                    mode_labels[axis] = edge_labels[edge][1]["mode"]
                if edge.sink == vertex_index:
                    axis = description.field_order[edge.sink_slot]
                    spin_labels[axis] = edge_labels[edge][0]["spin"]
                    mode_labels[axis] = edge_labels[edge][0]["mode"]
            if any(label is None for label in (*spin_labels, *mode_labels)):
                raise ValueError(f"Baryon vertex {vertex_index} has an unbound quark slot")
            spin_key, elemental_key, *cores = vertex_keys[vertex_index]
            entries = spin_entries_once(vertex, description)
            joins.append(BaryonJoinSpec(
                scene, f"baryon-vertex:{vertex_index}", spin_key, elemental_key,
                tuple(spin_labels), tuple(mode_labels),
                sparse_spin=entries,
                core=(cores[0] if cores else vertex.register_core(
                    (description,), (spin_key,), (elemental_key,), weighted=False,
                    _spin_entries=(entries,))),
            ))
        else:
            ordered_vertex_keys.append(vertex_keys[vertex_index])
            outgoing = [edge for edge in edges if edge.source == vertex_index]
            incoming = [edge for edge in edges if edge.sink == vertex_index]
            if len(outgoing) != 1 or len(incoming) != 1:
                raise ValueError(f"Meson vertex {vertex_index} requires one incoming and one outgoing edge")
            field_edge, bar_edge = outgoing[0], incoming[0]
            field_labels, bar_labels = edge_labels[field_edge][1], edge_labels[bar_edge][0]
            vertex_subscripts.append(
                (field_labels["spin"], bar_labels["spin"],
                 field_labels["mode"], bar_labels["mode"])
            )

    propagator_subscripts = tuple(
        (edge_labels[edge][0]["spin"], edge_labels[edge][1]["spin"],
         edge_labels[edge][0]["mode"], edge_labels[edge][1]["mode"])
        for edge in edges
    )
    prop_keys = tuple(edge_keys[edge] for edge in edges)
    equation = AxisPattern((*vertex_subscripts, *propagator_subscripts), ())
    return (*ordered_vertex_keys, *prop_keys), equation, tuple(joins)


def _edge_coordinate_tables(diagram, edge, vertices, descriptions, reader):
    """Freeze declared endpoint time maps, so execution only selects a row."""
    result = []
    for argument, index, source_end, axis in ((0, edge.sink, False, 'sink_time'),
                                             (1, edge.source, True, 'source_time')):
        role = diagram.time_list[index]
        if isinstance(role, Integral):
            continue
        vertex, description = vertices[index], descriptions[index]
        if isinstance(vertex, Baryon):
            record = getattr(vertex.reader, 'time_selection', None)
            offset = 0
        else:
            record = description.endpoint_time_selection
            offset = ((description.field_time_unwrapped if source_end else description.bar_time_unwrapped)
                      - description.anchor_time)
        period = int(vertex.reader.Lt)
        table = tuple(map_edge_time_slot((slot + offset) % period, record, reader, axis,
            physical_offsets=getattr(vertex, '_physical_offset_mapping', False)) for slot in range(period))
        result.append((argument, str(role), table))
    return tuple(result)


def _add_vertex_tensor(add_tensor, vertex, description, workspace_bytes, *, configuration):
    reader = vertex.reader
    metadata = description.tensor
    logical_identity = (
        "baryon-vertex", id(vertex), description.term_index, description.time,
        description.dagger, description.field_order, metadata.selection_identity,
    )
    shape = metadata.shape
    operator_term = description.operator_term
    return add_tensor(
        logical_identity,
        metadata,
        reader,
        "baryon-elemental",
        lambda v=vertex, d=description: v.read_factor_block(d),
        lambda raw, v=vertex, d=description: v.materialize_unweighted_elemental(d, raw),
        read_arguments=(operator_term.component, operator_term.momentum, description.time, "vvv"),
        read_selection=({"usedNe": description.usedNe} if description.mode_indices is None else
                        {"mode_indices": dict(zip(("mode0", "mode1", "mode2"), description.mode_indices))}),
        output_shape=shape,
        output_dtype=np.complex128,
        workspace_bytes=workspace_bytes,
        output_axes=metadata.axes,
        transform=(
            ("operation", "ordered-elemental"),
            ("spin_basis", vertex.spin_basis),
            ("source_adjoint", description.dagger),
            ("field_order", tuple(description.field_order)),
        ),
        description_semantics={
            "sector": "vvv",
            "operator_term": operator_term_semantics(operator_term),
            "coefficient_owner": "OutputRecipe",
        },
        description_bindings={
            "configuration": stable_binding_value(configuration),
            "anchor_time": int(description.time),
            "endpoint_time_offsets": tuple(description.endpoint_time_offsets),
            "endpoint_times": tuple(description.endpoint_times),
            "field_order": tuple(description.field_order),
            "usedNe": stable_binding_value(description.usedNe),
            "phase_convention": vertex.phase_convention,
            "boundary": vertex.boundary,
        },
    )


def _vertex_workspace(description):
    # Source adjoint can overlap the owned spatial copy and its conjugate.
    return (1 + int(description.dagger)) * math.prod(description.tensor.shape) * np.dtype(np.complex128).itemsize


def _lookup_propagator(propagator_map, symbol):
    try:
        reader = propagator_map[symbol]
    except KeyError as exc:
        raise KeyError(f"propagator_map has no data reader for {symbol!r}") from exc
    if not callable(getattr(reader, "describe_block", None)) or not callable(
        getattr(reader, "block", None)
    ):
        raise TypeError("baryon propagator bindings must expose describe_block and block")
    return reader


def _expected_propagator_shape(sector, sink_mode, source_mode, sink_point, source_point):
    if sector == "vsv":
        return (4, 4, sink_mode, source_mode)
    raise ValueError(f"unknown propagator sector {sector!r}")


def _validate_propagator_metadata(metadata, sector, expected_shape):
    if not isinstance(metadata, TensorAxisMetadata):
        raise TypeError("propagator describe_block must return TensorAxisMetadata")
    if metadata.axes != _PROPAGATOR_AXES[sector] or metadata.shape != expected_shape:
        raise ValueError(
            f"propagator {sector} axes/extents disagree with the graph endpoint mapping"
        )
    if metadata.dtype.kind != "c" or metadata.sparse_layout != "dense":
        raise NotImplementedError("dense-v Baryon diagrams require dense complex VSV propagators")


def _compare_propagator_vertex_endpoint(
    propagator_metadata, endpoint, vertex, vertex_description, *,
    vertex_index, slot, meson_endpoint,
):
    """Compare one actual VSV endpoint with its connected raw vertex axis.

    The small identity check follows legacy_diagram._compare_propagator_endpoint,
    _vertex_endpoint_axis_name, and _binding_value_identity. It is kept local
    to the Baryon provider so E does not depend on another provider's private
    helpers or on an additional shared tensor contract.
    """
    if endpoint == "source":
        propagator_axis = "source_mode"
    elif endpoint == "sink":
        propagator_axis = "sink_mode"
    else:
        raise ValueError("VSV endpoints must be source or sink")

    if isinstance(vertex, Baryon):
        if (isinstance(slot, bool) or not isinstance(slot, Integral)
                or not 0 <= int(slot) < len(vertex_description.field_order)):
            raise ValueError("Baryon endpoint requires a valid quark slot")
        raw_axis = int(vertex_description.field_order[int(slot)])
        if not 0 <= raw_axis < 3:
            raise ValueError("Baryon field_order must map to one of its three raw mode axes")
        vertex_axis = f"mode{raw_axis}"
        vertex_kind = "baryon"
        vertex_endpoint = "leg"
    elif isinstance(vertex, Meson):
        if vertex_description.logical_sector != "v2v":
            raise ValueError("dense-v Baryon diagrams require V/V Meson endpoint metadata")
        raw_axes = tuple(vertex_description.tensor.axes)
        if len(raw_axes) != 2:
            raise ValueError("V/V Meson metadata must expose two ordered raw mode axes")
        field_index = 1 if vertex_description.dagger else 0
        bar_index = 0 if vertex_description.dagger else 1
        vertex_axis = raw_axes[field_index if meson_endpoint == "field" else bar_index]
        vertex_kind = "meson"
        vertex_endpoint = meson_endpoint
    else:
        raise TypeError("dense-v endpoint identities require Baryon or Meson vertices")

    propagator_binding = propagator_metadata.axis_bindings.get(propagator_axis)
    vertex_binding = vertex_description.tensor.axis_bindings.get(vertex_axis)
    if propagator_binding is not None and vertex_binding is not None:
        if (_axis_binding_identity(propagator_axis, propagator_binding)
                != _axis_binding_identity(vertex_axis, vertex_binding)):
            raise ValueError(
                f"propagator {endpoint} endpoint actual dataset ID or ordered coordinates "
                f"disagree with vertex {vertex_kind} endpoint: "
                f"{propagator_binding!r} != {vertex_binding!r}"
            )
        status, knowledge = "verified", "both-bound"
    else:
        status = "unknown"
        knowledge = ("both-unbound" if propagator_binding is None and vertex_binding is None
                     else "one-unbound")
    result = {
        "status": status,
        "knowledge": knowledge,
        "propagator_axis": propagator_axis,
        "vertex_axis": vertex_axis,
        "vertex_kind": vertex_kind,
        "vertex_endpoint": vertex_endpoint,
        "propagator": propagator_binding,
        "vertex": vertex_binding,
        "vertex_index": int(vertex_index),
    }
    if vertex_kind == "baryon":
        result["vertex_slot"] = int(slot)
    return result


def _axis_binding_identity(axis, binding):
    if binding is None:
        return None
    # Axis names describe the two different tensors; compare only the shared
    # actual dataset and ordered coordinate identity here.
    return canonical_axis_binding_identity({"endpoint": binding})


def _endpoint_check_identity(check):
    return (
        check["status"], check["knowledge"], check["propagator_axis"],
        check["vertex_axis"], check["vertex_kind"], check["vertex_endpoint"],
        check["vertex_index"], check.get("vertex_slot"),
        _axis_binding_identity(check["propagator_axis"], check["propagator"]),
        _axis_binding_identity(check["vertex_axis"], check["vertex"]),
    )


def _validate_configuration(configuration, vertices, propagator_map):
    readers = [vertex.reader for vertex in vertices]
    readers.extend(propagator_map.values())
    validate_vertex_time_bindings(vertices, tuple(propagator_map.values()))
    if configuration is None:
        return
    for reader in readers:
        cfg = getattr(reader, "cfg", None)
        if cfg is not None and cfg != configuration:
            raise ValueError("configuration disagrees with a bound baryon/propagator reader")


def _backend_name(vertices):
    names = {getattr(vertex.xp, "__name__", None) for vertex in vertices}
    if len(names) != 1:
        raise ValueError("all vertices in one dense-V Diagram must use the same array backend")
    name = next(iter(names))
    if name not in {"numpy", "cupy"}:
        raise NotImplementedError("baryon Diagram binding requires one NumPy or CuPy array module")
    return name


def _array_backend_name(value):
    if isinstance(value, np.ndarray):
        return "numpy"
    if type(value).__module__.split(".", 1)[0] == "cupy":
        return "cupy"
    return None


def _coordinate_map(coordinates):
    if len(coordinates.points) != 1:
        raise NotImplementedError("baryon Diagram binding currently accepts one fixed coordinate")
    return dict(coordinates.points[0].values)


def _resolve_time(symbol, values):
    from numbers import Integral

    if symbol in values:
        value = values[symbol]
    elif isinstance(symbol, Integral) and not isinstance(symbol, (bool, np.bool_)):
        value = symbol
    else:
        raise ValueError(f"symbolic baryon Diagram time {symbol!r} has no fixed coordinate binding")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("baryon Diagram time bindings must be nonnegative integers")
    return int(value)


__all__ = ["BaryonDiagramBinding"]
