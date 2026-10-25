"""Read-only Diagram measurement planning over registered datasets."""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
import sys

import numpy as np
from lqcdweave.operators.correlator.api import Diagram, ModeBlockedDiagram
from sympy import Symbol

from lqcdweave.adapters.baryon_dataset import DatasetBaryonElemental
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contraction.binding.api import (
    Baryon, Meson, operator_term_semantics, validate_slot_compatibility,
)
from lqcdweave.contraction.expansion.api import NativeDiagramBinding, expand_correlator
from lqcdweave.contraction.results.api import collect_diagrams
from lqcdweave.contracts.dataset import (
    DatasetRequest,
    DatasetSpec,
    canonical_json,
)
from lqcdweave.contracts.correlator.api import BatchSpec, DiagramBindings, TensorDescription, describe_coordinates
from lqcdweave.contracts.elemental import BaryonComponent, GaugePath
from lqcdweave.contracts.current import DirectedCurrentBasis
from lqcdweave.operators.single_hadron.api import SpectralProfile
from lqcdweave.planning.dataset import plan_dataset_graph, plan_requirement
from lqcdweave.planning.diagram_measurement import (
    diagram_measurement_document,
    validate_diagram_endpoint_descriptions,
    validate_diagram_input_specs,
)
from lqcdweave.planning.production import vsp_dataset_document
from lqcdweave.runtime.datasets import apply_source_policy


_PROPAGATOR_SECTORS = ("vsv", "psv", "vsp", "psp")
_ELEMENTAL_SECTORS = ("v2v", "v2p", "p2v", "p2p")
_DEFAULT_USED_NE = object()
_UNBOUNDED_METADATA_LIMIT = sys.maxsize


def _jsonable(value):
    """Detach supported finite scientific scalars and small metadata records."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, complex):
        if not np.isfinite(value):
            raise ValueError("Diagram science metadata must be finite")
        return [float(value.real), float(value.imag)]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Diagram science metadata must be finite")
        return value
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Diagram science mappings require string keys")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported Diagram science value {type(value).__name__}")


def _identity_symbol(value):
    return str(value)


def _freeze_used_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be bool")
    if isinstance(value, Integral):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
        return int(value)
    if isinstance(value, (tuple, list)):
        counts = tuple(_freeze_used_count(item, name) for item in value)
        if not counts:
            raise ValueError(f"{name} cannot be empty")
        return counts
    raise TypeError(f"{name} must be a positive integer, an ordered count tuple, or None")


def _frozen_bindings(diagram, bindings):
    if any(isinstance(atom, ModeBlockedDiagram) for atom in collect_diagrams(diagram)):
        raise NotImplementedError("persistent mode-block measurements require a selection-aware science schema")
    if not isinstance(bindings, DiagramBindings):
        raise TypeError("bindings must be an explicit DiagramBindings instance")
    if not isinstance(bindings.propagator_map, Mapping) or not isinstance(bindings.vertex_map, Mapping):
        raise TypeError("persistent diagrams require explicit propagator_map and vertex_map mappings")
    if bindings.configuration is None:
        raise ValueError("persistent diagrams require an explicit configuration identity")

    original_propagators = dict(bindings.propagator_map)
    if not original_propagators or any(not isinstance(value, DatasetPropagator)
                                       for value in original_propagators.values()):
        raise TypeError("propagator_map values must be DatasetPropagator instances")
    vertices = dict(bindings.vertex_map)
    if any(isinstance(vertex, (Meson, Baryon)) and vertex.mode_indices is not None for vertex in vertices.values()):
        raise NotImplementedError("persistent mode-selected vertices require a selection-aware science schema")
    if not vertices or any(
            not ((isinstance(value, Meson) and isinstance(value.elemental, DatasetElemental)) or
                 (isinstance(value, Baryon) and
                  isinstance(value.elemental, DatasetBaryonElemental)))
            for value in vertices.values()):
        raise TypeError(
            "vertex_map values must be Meson/DatasetElemental or "
            "Baryon/DatasetBaryonElemental instances"
        )

    copied_vertices = {}
    clones = {}
    for label, vertex in vertices.items():
        cache_key = id(vertex)
        if cache_key not in clones:
            if vertex.xp is not np:
                raise TypeError("persistent Diagram measurement currently supports NumPy readers only")
            for term in vertex.operator.terms:
                _require_profile_identity(term.profile)
            operator_copy = deepcopy(vertex.operator)
            if isinstance(vertex, Baryon):
                clones[cache_key] = Baryon(
                    vertex.elemental, operator_copy, source=vertex.dagger,
                    phase_convention=vertex.phase_convention,
                    spin_basis=vertex.spin_basis, boundary=vertex.boundary,
                    array_module=vertex.xp,
                )
            else:
                clones[cache_key] = Meson(
                    vertex.elemental,
                    operator_copy,
                    source=vertex.dagger,
                    phase_convention=vertex.phase_convention,
                    boundary=vertex.boundary,
                    array_module=vertex.xp,
                    requested_sectors=vertex.requested_sectors,
                )
        copied_vertices[label] = clones[cache_key]

    diagrams = collect_diagrams(diagram)
    if len(diagrams) != 1 or not isinstance(diagrams[0], Diagram):
        raise TypeError("persistent measurement requires exactly one scalar Diagram")
    source = diagrams[0]
    detached = deepcopy(source)
    propagators = {}
    for original, clone in zip(source.propagator_list, detached.propagator_list, strict=True):
        if original is None:
            continue
        reader = original_propagators.get(original)
        if reader is not None:
            propagators[clone] = reader
    if not propagators:
        raise ValueError("propagator_map has no entry for the Diagram's propagators")
    frozen = DiagramBindings(
        propagators,
        copied_vertices,
        bindings.configuration,
        _freeze_used_count(bindings.usedNe, "usedNe"),
        _freeze_used_count(bindings.usedNp, "usedNp"),
        bindings.groups_independent,
        bindings.elemental_weighted,
    )
    return detached, frozen


def _times_and_symbols(diagram, source_time_symbol, sink_time_symbol, extent):
    if (not isinstance(source_time_symbol, Symbol) or
            not isinstance(sink_time_symbol, Symbol)):
        raise TypeError("source and sink time axes must be explicit SymPy symbols")
    if (source_time_symbol == sink_time_symbol or
            str(source_time_symbol) == str(sink_time_symbol)):
        raise ValueError("source_time_symbol and sink_time_symbol must be distinct")
    if not isinstance(diagram.time_list, (tuple, list)):
        raise TypeError("Diagram.time_list must be an ordered sequence")
    times = tuple(diagram.time_list)
    if source_time_symbol not in times or sink_time_symbol not in times:
        raise ValueError("both source and sink symbols must occur in Diagram.time_list")
    for value in times:
        if value in (source_time_symbol, sink_time_symbol):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            free = getattr(value, "free_symbols", ())
            if free:
                raise ValueError("Diagram contains an unbound time symbol outside the source/sink pair")
            raise TypeError("fixed Diagram times must be integer coordinates or the declared pair symbols")
        if not 0 <= int(value) < extent:
            raise ValueError("fixed Diagram time lies outside the declared coordinate extent")
    return tuple(range(extent))


def _reader_sectors(propagator):
    dataset = propagator.dataset
    if isinstance(dataset, Mapping):
        sectors = dict(dataset)
    else:
        sectors = {propagator.kind: dataset}
    if not sectors or set(sectors) - set(_PROPAGATOR_SECTORS):
        raise ValueError("DatasetPropagator exposes an unsupported product sector")
    return sectors


def _complex_snapshot(value):
    number = complex(value)
    if not np.isfinite(number):
        raise ValueError("Diagram operator coefficients must be finite")
    return [float(number.real), float(number.imag)]


def _require_profile_identity(profile):
    if profile is None:
        return
    if type(profile) is not SpectralProfile or "apply_block" in vars(profile):
        raise ValueError(
            "persistent Diagram measurement requires a canonical profile identity "
            "(no profile or the exact SpectralProfile type)"
        )


def _operator_snapshot(vertex):
    operator = vertex.operator
    terms = []
    for term in operator.terms:
        _require_profile_identity(term.profile)
        semantic = _jsonable(operator_term_semantics(term))
        for optional in ("gamma", "profile", "spin", "current_term"):
            if semantic.get(optional) is None:
                semantic[optional] = "none"
        semantic["coefficient"] = _complex_snapshot(term.coefficient)
        semantic["anchor_time_offset"] = int(term.anchor_time_offset)
        if term.current_term is not None:
            current_record = term.current_term.as_dict()
            if current_record.get("wilson_r") is None:
                current_record["wilson_r"] = "none"
            semantic["current_term"] = _jsonable(current_record)
        try:
            semantic["momentum_coordinate"] = _jsonable(
                vertex.elemental.momenta[int(term.momentum)])
        except (IndexError, TypeError) as exc:
            raise ValueError("operator momentum index has no registered coordinate") from exc
        terms.append(semantic)
    result = dict(
        name=str(getattr(operator, "name", type(operator).__qualname__)),
        terms=terms,
        current_term_ranges=_jsonable(getattr(operator, "current_term_ranges", ())),
        source_dagger=bool(vertex.dagger),
        phase_convention=vertex.phase_convention,
        elemental_boundary=vertex.boundary,
    )
    if isinstance(vertex, Meson) and vertex.requested_sectors is not None:
        result["requested_sectors"] = list(vertex.requested_sectors)
    if isinstance(vertex, Baryon):
        result["spin_basis"] = vertex.spin_basis
    return result


def _component_snapshot(component):
    if not hasattr(component, "key") or not hasattr(component, "time_offsets"):
        raise TypeError("sampling keys must use registered elemental components")
    key = component.key
    if isinstance(component, BaryonComponent):
        if isinstance(key, tuple) and len(key) == 3 and all(
                isinstance(path, GaugePath) for path in key):
            key_record = dict(kind="baryon-path-triple",
                              paths=[list(path.steps) for path in key])
        elif isinstance(key, tuple) and len(key) == 2 and key[0] in ("projected", "distance"):
            key_record = dict(kind=f"baryon-{key[0]}", value=_jsonable(key[1]))
        else:
            raise TypeError("persistent Baryon components require a registered spatial key")
    elif hasattr(key, "steps") and hasattr(key, "displacement"):
        key_record = dict(kind="gauge-path", steps=list(key.steps),
                          displacement=list(key.displacement))
    else:
        key_record = _jsonable(key)
    return dict(key=key_record, time_offsets=list(component.time_offsets))


def _sampling_snapshot(elemental):
    pool_ids = []
    for (component, momentum, time, side), group_id in elemental.sampling_pool_ids.items():
        momentum_coordinate = elemental.momenta[int(momentum)]
        pool_ids.append(dict(
            component=_component_snapshot(component),
            momentum_index=int(momentum),
            momentum_coordinate=_jsonable(momentum_coordinate),
            time=int(time), side=side, pool_id=group_id,
        ))
    pool_ids.sort(key=canonical_json)
    pools = []
    for group_id, group in sorted(elemental.sampling_groups.items()):
        observed = group.observed_counts
        observed_records = [] if observed is None else sorted(
            ((_jsonable(key), int(count)) for key, count in observed.items()),
            key=lambda item: canonical_json(item[0]),
        )
        pools.append(dict(
            group_id=group_id,
            spatial_sites=int(group.spatial_sites),
            sampled_points=int(group.sampled_points),
            sampling_design=group.sampling_design,
            compensation=group.compensation,
            observed_counts_state=("not-provided" if observed is None else "provided"),
            observed_counts=observed_records,
        ))
    return dict(pool_ids=pool_ids, pools=pools)


def _tag_snapshot(tag):
    if (not hasattr(tag, "tag") or not hasattr(tag, "time") or
            isinstance(tag.tag, bool) or type(tag.tag) is not int or
            isinstance(tag.time, bool) or type(tag.time) is not int):
        raise ValueError("persistent propagator tags require explicit integer label and time")
    return dict(tag=tag.tag, time=tag.time)


def _time_entry_snapshot(value):
    if isinstance(value, Symbol):
        return dict(symbol=str(value), assumptions=_jsonable(dict(value.assumptions0)))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("fixed Diagram time entry is not an integer or Symbol")
    return dict(fixed=int(value))


def _crop_prefix(spec, usedNe, usedNp, *, apply_modes, apply_points):
    """Build an exact ordered DatasetSpec view for consumed tail prefixes."""
    document = spec.document
    axes = document["axes"]
    view_axes = []
    mode_axes = [axis["name"] for axis in axes if "mode" in axis["name"]]
    point_axes = [axis["name"] for axis in axes if "point" in axis["name"]]

    def counts(value, names, label):
        if value is None or not names:
            return {}
        values = (value,) * len(names) if isinstance(value, int) else tuple(value)
        if len(values) != len(names):
            raise ValueError(f"{label} must match the selected input view axes")
        result = {}
        for axis, count in zip(names, values, strict=True):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{label} entries must be positive integers")
            coordinates = next(item["coordinates"] for item in axes if item["name"] == axis)
            if count > len(coordinates):
                raise ValueError(f"{label} exceeds registered {axis} coverage")
            result[axis] = coordinates[:count]
        return result

    selected = {}
    if apply_modes:
        selected.update(counts(usedNe, mode_axes, "usedNe"))
    if apply_points:
        selected.update(counts(usedNp, point_axes, "usedNp"))
    for axis in axes:
        name = axis["name"]
        if name in selected:
            axis["coordinates"] = list(selected[name])
            if name in document["bases"]:
                document["bases"][name]["indices"] = list(selected[name])
        if name in mode_axes or name in point_axes:
            view_axes.append(name)
    return DatasetSpec(document), tuple(view_axes)


def _crop_axis_labels(spec, requested):
    """Create an exact ordered view from endpoint labels reported by E."""
    document = spec.document
    selected_axes = []
    for name, labels in requested.items():
        if name not in spec.axis_names or "mode" not in name:
            raise ValueError("E endpoint view may select only registered mode axes")
        labels = tuple(labels)
        available = tuple(spec.coordinates(name))
        if (not labels or len(labels) > len(available) or
                tuple(canonical_json(value) for value in labels) !=
                tuple(canonical_json(value) for value in available[:len(labels)])):
            raise ValueError("E endpoint labels must be a nonempty ordered prefix of the actual propagator axis")
        for axis in document["axes"]:
            if axis["name"] == name:
                axis["coordinates"] = list(labels)
        if name in document["bases"]:
            document["bases"][name]["indices"] = list(labels)
        selected_axes.append(name)
    return DatasetSpec(document), tuple(selected_axes)


def _verified_reader_subset_axes(reader, observed_by_id):
    """Preserve a public DatasetView selection when reopening inputs.

    A Diagram may receive an existing view, such as the temporal 6/7
    channel view of the registered eight-channel directed V/V dataset. The
    graph must reopen that same view against its parent rather than silently
    widening it to the full parent. Only coordinate selections are inherited;
    the scientific definition and all actual parent identities must remain
    identical.
    """
    actual_id = getattr(reader, "data_id", None)
    requested = getattr(reader, "spec", None)
    source_reader = getattr(reader, "reader", reader)
    parent = getattr(source_reader, "spec", None)
    if not isinstance(requested, DatasetSpec) or not isinstance(parent, DatasetSpec):
        return ()
    registered = observed_by_id.get(actual_id)
    if (registered is None or not isinstance(registered.get("spec"), DatasetSpec)
            or canonical_json(parent.document) != canonical_json(registered["spec"].document)):
        raise ValueError("Diagram input view parent differs from the indexed actual dataset")
    requested_document, parent_document = requested.document, parent.document
    if (requested.axis_names != parent.axis_names
            or requested_document["kind"] != parent_document["kind"]
            or requested_document["configuration"] != parent_document["configuration"]
            or requested_document["dtype"] != parent_document["dtype"]
            or requested_document["definition"] != parent_document["definition"]
            or requested_document["conventions"] != parent_document["conventions"]
            or requested_document["parents"] != parent_document["parents"]
            or set(requested_document["bases"]) != set(parent_document["bases"])):
        raise ValueError("Diagram input view may not alter its parent's scientific identity")
    for name in requested_document["bases"]:
        requested_binding = requested_document["bases"][name]
        parent_binding = parent_document["bases"][name]
        if requested_binding["data_id"] != parent_binding["data_id"]:
            raise ValueError("Diagram input view may not change actual basis or point IDs")

    inherited = []
    for name in requested.axis_names:
        parent_coordinates = parent.coordinates(name)
        requested_coordinates = requested.coordinates(name)
        if requested_coordinates == parent_coordinates:
            continue
        if name != "component" and "mode" not in name and "point" not in name:
            raise ValueError(f"Diagram input view may not select the {name!r} axis")
        positions = {canonical_json(value): index
                     for index, value in enumerate(parent_coordinates)}
        try:
            selected_positions = [positions[canonical_json(value)]
                                  for value in requested_coordinates]
        except KeyError as exc:
            raise ValueError(f"Diagram input view selects an unavailable {name!r} coordinate") from exc
        if len(set(selected_positions)) != len(selected_positions):
            raise ValueError(f"Diagram input view repeats a {name!r} coordinate")
        if "mode" in name and selected_positions != list(range(len(selected_positions))):
            raise ValueError("inherited mode views must be ordered prefixes")
        if name == "component":
            if (parent_document["kind"] != "elemental_vv"
                    or parent_document["definition"].get("directed_basis") != DirectedCurrentBasis.SCHEMA
                    or parent_coordinates != [{"path": [index]} for index in range(8)]
                    or requested_coordinates != [{"path": [6]}, {"path": [7]}]):
                raise ValueError("only an exact temporal 6/7 view of registered directed V/V is supported")
        requested_binding = requested_document["bases"].get(name)
        if requested_binding is not None and requested_binding["indices"] != requested_coordinates:
            raise ValueError("Diagram input view basis indices differ from selected coordinates")
        inherited.append(name)
    return tuple(inherited)


def _propagator_mode_requirements(descriptions, symbol, sector, reader):
    """Union exact endpoint prefixes emitted by E for one consumed propagator."""
    available_axes = set(reader.spec.axis_names)
    labels_by_axis = {name: [] for name in available_axes if "mode" in name}
    for description in descriptions:
        if (description.role != "propagator" or
                description.semantics.get("symbol") != _identity_symbol(symbol) or
                description.semantics.get("sector") != sector):
            continue
        checks = description.bindings.get("endpoint_axis_bindings")
        if not isinstance(checks, Mapping):
            raise ValueError("E propagator description omitted its endpoint identities")
        for check in checks.values():
            axis = check.get("propagator_axis")
            binding = check.get("propagator")
            if axis not in labels_by_axis or not isinstance(binding, (tuple, list)) or len(binding) != 2:
                continue
            actual_id, labels = binding
            expected_id = reader.spec.document["bases"].get(axis, {}).get("data_id")
            if actual_id != expected_id:
                raise ValueError("E endpoint actual basis ID differs from the registered propagator view")
            labels_by_axis[axis].append(tuple(labels))
    requested = {}
    for axis, candidates in labels_by_axis.items():
        if not candidates:
            continue
        longest = max(candidates, key=len)
        encoded = tuple(canonical_json(value) for value in longest)
        if any(tuple(canonical_json(value) for value in candidate) != encoded[:len(candidate)]
               for candidate in candidates):
            raise ValueError("E endpoint mode prefixes for one propagator are incompatible")
        requested[axis] = longest
    return requested


def _psv_request_view_for_vsp(parent_spec, vsp_spec):
    """Describe the exact PSV prefix which is the inverse view of a VSP."""
    document = parent_spec.document
    vsp_document = vsp_spec.document
    if document.get("kind") != "psv" or vsp_document.get("kind") != "vsp":
        raise ValueError("gamma5-derived VSP parent and product kinds are incompatible")
    psv_to_vsp_axes = {
        "source_time": "sink_time",
        "sink_time": "source_time",
        "sink_spin": "source_spin",
        "source_spin": "sink_spin",
        "sink_point": "source_point",
        "sink_color": "source_color",
        "source_mode": "sink_mode",
    }
    parent_axes = {axis["name"]: axis for axis in document["axes"]}
    product_axes = {axis["name"]: axis for axis in vsp_document["axes"]}
    if set(parent_axes) != set(psv_to_vsp_axes):
        raise ValueError("gamma5-derived VSP parent has a noncanonical PSV axis set")
    if set(product_axes) != set(psv_to_vsp_axes.values()):
        raise ValueError("gamma5-derived VSP has a noncanonical axis set")

    for psv_axis, vsp_axis in psv_to_vsp_axes.items():
        parent_axes[psv_axis]["coordinates"] = list(product_axes[vsp_axis]["coordinates"])
    for psv_axis in ("sink_point", "source_mode"):
        document["bases"][psv_axis]["indices"] = list(parent_axes[psv_axis]["coordinates"])
    return DatasetSpec(document)


def _preflight_diagram(diagram, bindings, source_symbol, sink_symbol, times):
    descriptions = []
    for source_time in times:
        for sink_time in times:
            expanded = expand_correlator(
                diagram,
                adapter=NativeDiagramBinding(),
                bindings=bindings,
                time_axes={source_symbol: source_time, sink_symbol: sink_time},
                batch="auto",
            )
            if expanded.inputs is None:
                raise ValueError("fixed Diagram did not lower to registered tensor inputs")
            descriptions.extend(
                tensor.description
                for tensor in expanded.inputs.problem.specs.values()
                if isinstance(tensor.description, TensorDescription)
            )
    validate_diagram_endpoint_descriptions(descriptions)
    if bindings.usedNe is not None and not any(
            any("mode" in axis for axis in description.axes)
            for description in descriptions):
        raise ValueError("usedNe is unused by this Diagram and cannot be certified")
    if bindings.usedNp is not None and not any(
            any("point" in axis for axis in description.axes)
            for description in descriptions):
        raise ValueError("usedNp is unused by this Diagram and cannot be certified")
    return tuple(descriptions)


def _normalize_sink_batch(sink_batch, sink_symbol):
    if sink_batch is None:
        return None
    if not isinstance(sink_batch, BatchSpec):
        raise TypeError("sink_batch must be a BatchSpec or None")
    if sink_batch.axis is not None and sink_batch.axis != sink_symbol:
        raise ValueError("sink_batch may only select the explicit sink-time symbol")
    return BatchSpec(sink_batch.size, sink_symbol)


def _preflight_sink_batch(diagram, bindings, source_symbol, sink_symbol,
                          source_times, sink_times, sink_batch):
    """Prove batch compatibility from E metadata before creating an output."""
    for source_time in source_times:
        expected_coordinates, expected_batch = describe_coordinates(
            time_axes={source_symbol: source_time, sink_symbol: sink_times},
            batch=sink_batch,
        )
        expanded = expand_correlator(
            diagram,
            adapter=NativeDiagramBinding(),
            bindings=bindings,
            time_axes={source_symbol: source_time, sink_symbol: sink_times},
            batch=sink_batch,
        )
        if (expanded.coordinates != expected_coordinates or
                expanded.batch != expected_batch):
            raise ValueError(
                "sink batching changed the fixed source or ordered sink coordinates"
            )
        if expanded.inputs is None:
            raise ValueError("fixed Diagram did not lower to registered tensor inputs")


def _build_requests(index, diagram, bindings, source_time_symbol, sink_time_symbol,
                    data_ids=None, sink_batch=None):
    diagram, bindings = _frozen_bindings(diagram, bindings)
    description = collect_diagrams(diagram)
    if len(description) != 1:
        raise TypeError("persistent measurement requires exactly one Diagram")

    raw_propagators = {}
    by_symbol_text = {}
    for symbol in diagram.propagator_list:
        try:
            reader = bindings.propagator_map[symbol]
        except KeyError:
            continue
        by_symbol_text[_identity_symbol(symbol)] = symbol
        raw_propagators[symbol] = (reader, _reader_sectors(reader))
    if len(by_symbol_text) != len(raw_propagators):
        raise ValueError("propagator symbols must have unique stable text identities")

    raw_vertices = {}
    for label in diagram.vertex_list:
        try:
            vertex = bindings.vertex_map[label]
        except KeyError as exc:
            raise KeyError(f"vertex_map has no explicit binding for {label!r}") from exc
        raw_vertices[label] = vertex

    # This legacy fixed two-axis helper has one shared slot coordinate space.
    # General endpoint domains remain with native calculate and E's bindings.
    try:
        time_selection = validate_slot_compatibility([
            *(reader for reader, _ in raw_propagators.values()),
            *(vertex.elemental for vertex in raw_vertices.values()),
        ])
    except ValueError as exc:
        raise ValueError("fixed two-axis Diagram requires identical ordered slot selections; "
                         "use native calculate with explicit endpoint bindings") from exc

    # Obtain only the execution extent needed to enumerate all scalar time
    # pairs. E metadata expansion identifies the exact input sectors below;
    # unused members of a caller's composite reader are not science inputs.
    if not raw_propagators:
        raise ValueError("persistent Diagram requires at least one registered propagator")
    extent = next(iter(raw_propagators.values()))[0].Lt
    times = _times_and_symbols(diagram, source_time_symbol, sink_time_symbol, extent)
    sink_batch = _normalize_sink_batch(sink_batch, sink_time_symbol)
    descriptions = _preflight_diagram(
        diagram, bindings, source_time_symbol, sink_time_symbol, times,
    )
    if sink_batch is not None:
        _preflight_sink_batch(
            diagram, bindings, source_time_symbol, sink_time_symbol,
            times, times, sink_batch,
        )

    consumed_prop_roles = {
        (description.semantics.get("symbol"), description.semantics.get("sector"))
        for description in descriptions if description.role == "propagator"
    }
    consumed_elemental_ids = {
        (description.data_identity,
         description.semantics.get("raw_sector"))
        for description in descriptions
        if description.role in ("elemental", "meson-elemental")
    }
    consumed_baryon_ids = {
        (description.data_identity, description.semantics.get("sector"))
        for description in descriptions if description.role == "baryon-elemental"
    }
    # Sparse P/P records are represented by a bridge source, not a dense
    # TensorDescription. Current Mesons retain their established decomposition;
    # ordinary Mesons may explicitly opt into P/P through requested_sectors.
    for vertex in raw_vertices.values():
        operator = vertex.operator
        explicit_p2p = (isinstance(vertex, Meson)
                        and vertex.requested_sectors is not None
                        and "p2p" in vertex.requested_sectors)
        if explicit_p2p or getattr(operator, "current_terms", ()) or any(
                getattr(term, "current_term", None) is not None
                for term in operator.terms):
            reader = vertex.elemental.datasets.get("p2p")
            if reader is not None:
                consumed_elemental_ids.add((reader.data_id, "p2p"))

    # Validate only the concrete sectors and point coordinates consumed by the
    # Diagram. Endpoint descriptions remain a separate E boundary check.
    used_specs = {}
    for symbol, (_reader, sectors) in raw_propagators.items():
        for sector, reader in sectors.items():
            if (_identity_symbol(symbol), sector) in consumed_prop_roles:
                used_specs[f"propagator-{_identity_symbol(symbol)}-{sector}"] = reader.spec
    for position, (_label, vertex) in enumerate(raw_vertices.items()):
        for sector, reader in vertex.elemental.datasets.items():
            consumed = (consumed_baryon_ids if isinstance(vertex, Baryon)
                        else consumed_elemental_ids)
            if (reader.data_id, sector) in consumed:
                used_specs[f"vertex-{position}-{sector}"] = reader.spec
        if (isinstance(vertex, Meson) and vertex.elemental.point_dataset is not None and
                (vertex.elemental.datasets.get("p2p") is not None) and
                (vertex.elemental.datasets["p2p"].data_id, "p2p")
                in consumed_elemental_ids):
            used_specs[f"vertex-{position}-points"] = vertex.elemental.point_dataset.spec
            right_points = vertex.elemental.point_right_dataset
            if right_points is None:
                raise ValueError("P/P elemental input is missing its actual right point reader")
            used_specs[f"vertex-{position}-points-right"] = right_points.spec
    configuration, validated_times = validate_diagram_input_specs(used_specs)
    if validated_times != times:
        raise ValueError("registered input time extent changed during Diagram preflight")
    if bindings.configuration != configuration["configuration"]:
        raise ValueError("DiagramBindings.configuration differs from registered input configuration")

    # VSP identity includes its declared actual inputs. Gamma5-derived VSPs
    # additionally require a registered PSV parent whose definition reproduces
    # the complete stored record. Independently inverted VSPs remain supported.
    observations = index.observations()
    observed_by_id = {item["data_id"]: item for item in observations}
    for symbol, (_reader, sectors) in raw_propagators.items():
        for sector_name, sector_reader in sectors.items():
            if ((_identity_symbol(symbol), sector_name) in consumed_prop_roles
                    and "high_mode_projection" in sector_reader.spec.document["definition"]):
                from lqcdweave.production.perambulator.highmode_planning import validate_registered_highmode
                stored_highmode = observed_by_id.get(sector_reader.data_id)
                if stored_highmode is None:
                    raise ValueError("high-mode propagator is absent from the verified index")
                validate_registered_highmode(index, stored_highmode["spec"])
        vsp_reader = sectors.get("vsp")
        if vsp_reader is None or (_identity_symbol(symbol), "vsp") not in consumed_prop_roles:
            continue
        stored = observed_by_id.get(vsp_reader.data_id)
        if stored is None:
            raise ValueError("registered VSP actual dataset is absent from the verified index")
        vsp_spec = stored["spec"]
        document = vsp_spec.document
        definition = document["definition"]
        parents = document["parents"]
        derivation = definition.get("derivation")
        if "high_mode_projection" in definition:
            continue  # Every consumed high-mode sector was verified above.
        if derivation == "gamma5_hermiticity":
            if len(parents) != 1 or parents[0] not in observed_by_id:
                raise ValueError("VSP lacks a verified registered gamma5-derived PSV parent")
            parent_observation = observed_by_id[parents[0]]
            parent = parent_observation["spec"]
            parent_view = _psv_request_view_for_vsp(parent, vsp_spec)
            parent_coverage = plan_requirement(
                parent_view, [parent_observation],
                subset_axes=("sink_point", "source_mode"),
            )
            if (parent_coverage["action"] != "reuse"
                    or parent_coverage["data_id"] != parents[0]
                    or parent_coverage["completed"] != tuple(range(parent_view.shape[0]))):
                raise ValueError(
                    "registered VSP actual PSV parent does not cover its declared point/mode view"
                )
            expected_vsp = DatasetSpec(vsp_dataset_document(
                parent_view.document,
                psv_id=parents[0],
                gauge_id=definition.get("gauge_id"),
                basis_id=document["bases"]["sink_mode"]["data_id"],
                points_id=document["bases"]["source_point"]["data_id"],
            ))
            if expected_vsp.definition_id != vsp_spec.definition_id:
                raise ValueError("registered VSP does not match its actual PSV derivation")
        elif derivation is None:
            expected_parents = {
                definition.get("gauge_id"),
                document["bases"]["sink_mode"]["data_id"],
                document["bases"]["source_point"]["data_id"],
            }
            if (None in expected_parents or set(parents) != expected_parents or
                    any(parent_id not in observed_by_id for parent_id in expected_parents)):
                raise ValueError("independent VSP does not bind its registered gauge, basis and point parents")
        else:
            raise ValueError("unsupported VSP derivation declaration")

    requests = {}
    view_specs = {}
    role_nodes = {}
    identity_nodes = {}
    prop_runtime = {}
    vertex_runtime = {}

    def add_reader(role, reader, *, apply_modes=False, apply_points=False,
                   point_labels=None, axis_labels=None, used_ne=_DEFAULT_USED_NE):
        if not isinstance(reader.spec, DatasetSpec):
            raise TypeError("registered Diagram readers must expose a DatasetSpec")
        inherited_subset_axes = _verified_reader_subset_axes(reader, observed_by_id)
        selected_spec = reader.spec
        if point_labels is not None:
            if selected_spec.document["kind"] != "spatial_points" or "point" not in selected_spec.axis_names:
                raise ValueError("explicit point labels require a registered spatial point reader")
            point_labels = list(point_labels)
            if apply_points and bindings.usedNp is not None:
                if bindings.usedNp > len(point_labels):
                    raise ValueError("usedNp exceeds the declared endpoint point labels")
                point_labels = point_labels[:bindings.usedNp]
            stored = {canonical_json(label) for label in selected_spec.coordinates("point")}
            selected = [canonical_json(label) for label in point_labels]
            if (not selected or len(set(selected)) != len(selected) or
                    any(label not in stored for label in selected)):
                raise ValueError("requested actual point labels are absent or repeated")
            document = selected_spec.document
            for axis in document["axes"]:
                if axis["name"] == "point":
                    axis["coordinates"] = list(point_labels)
            if "point" in document["bases"]:
                document["bases"]["point"]["indices"] = list(point_labels)
            selected_spec = DatasetSpec(document)
        if axis_labels is None:
            view, subset_axes = _crop_prefix(
                selected_spec,
                bindings.usedNe if used_ne is _DEFAULT_USED_NE else used_ne,
                bindings.usedNp,
                apply_modes=apply_modes, apply_points=apply_points,
            )
        else:
            if apply_points:
                raise ValueError("an E endpoint mode view cannot also select point axes")
            view, subset_axes = _crop_axis_labels(selected_spec, axis_labels)
        subset_axes = tuple(dict.fromkeys((*inherited_subset_axes, *subset_axes)))
        actual_id = reader.data_id
        if not isinstance(actual_id, str) or not actual_id:
            raise ValueError("registered Diagram readers require actual dataset IDs")
        key = actual_id, view.definition_id
        node = identity_nodes.get(key)
        if node is None:
            node = role
            suffix = 1
            while node in requests:
                suffix += 1
                node = f"{role}-{suffix}"
            requests[node] = DatasetRequest(
                view, subset_axes=subset_axes, data_id=actual_id,
            )
            view_specs[node] = view
            identity_nodes[key] = node
        role_nodes[role] = node
        return node

    has_baryon_vertex = any(isinstance(vertex, Baryon) for vertex in raw_vertices.values())
    has_meson_vertex = any(isinstance(vertex, Meson) for vertex in raw_vertices.values())
    meson_used_ne = bindings.usedNe
    if (has_baryon_vertex and has_meson_vertex and
            isinstance(bindings.usedNe, (tuple, list))):
        counts = tuple(bindings.usedNe)
        if (len(counts) != 3 or any(isinstance(value, (bool, np.bool_)) or
                                    not isinstance(value, Integral) for value in counts)
                or len(set(counts)) != 1):
            raise ValueError("mixed dense-V graphs require equal per-leg usedNe counts")
        meson_used_ne = int(counts[0])
    for position, (symbol, (reader, sectors)) in enumerate(raw_propagators.items()):
        sector_nodes = {}
        for sector in _PROPAGATOR_SECTORS:
            product_reader = sectors.get(sector)
            if product_reader is None or (_identity_symbol(symbol), sector) not in consumed_prop_roles:
                continue
            role = f"propagator-{position}-{sector}"
            spec = product_reader.spec
            mode_axes = any("mode" in name for name in spec.axis_names)
            point_axes = any("point" in name for name in spec.axis_names)
            axis_labels = None
            if has_baryon_vertex and sector == "vsv":
                axis_labels = _propagator_mode_requirements(
                    descriptions, symbol, sector, product_reader,
                )
                if set(axis_labels) != {"sink_mode", "source_mode"}:
                    raise ValueError("dense-V Baryon edges must bind both actual VSV mode endpoints")
            sector_nodes[sector] = add_reader(
                role, product_reader,
                apply_modes=mode_axes and sector != "psp",
                apply_points=point_axes,
                axis_labels=axis_labels,
                used_ne=(meson_used_ne if has_baryon_vertex and has_meson_vertex
                         and sector != "vsv" else _DEFAULT_USED_NE),
            )
        if sector_nodes:
            prop_runtime[symbol] = sector_nodes

    vertex_clones = {}
    for position, (label, vertex) in enumerate(raw_vertices.items()):
        elemental = vertex.elemental
        sector_nodes = {}
        if isinstance(vertex, Baryon):
            reader = elemental.datasets.get("vvv")
            if reader is None or (reader.data_id, "vvv") not in consumed_baryon_ids:
                raise ValueError("Baryon vertex has no consumed registered vvv dataset")
            role = f"vertex-{position}-vvv"
            sector_nodes["vvv"] = add_reader(
                role, reader, apply_modes=True,
            )
            clone_key = id(vertex)
            if clone_key not in vertex_clones:
                vertex_clones[clone_key] = dict(
                    kind="baryon", operator=deepcopy(vertex.operator),
                    source=vertex.dagger, phase_convention=vertex.phase_convention,
                    spin_basis=vertex.spin_basis, boundary=vertex.boundary,
                    array_module=vertex.xp, sector_nodes=sector_nodes,
                )
            vertex_runtime[label] = vertex_clones[clone_key]
            continue
        is_current = (bool(getattr(vertex.operator, "current_terms", ())) or
                      any(getattr(term, "current_term", None) is not None
                          for term in vertex.operator.terms))
        explicitly_requests_p2p = (vertex.requested_sectors is not None
                                   and "p2p" in vertex.requested_sectors)
        for sector in _ELEMENTAL_SECTORS:
            reader = elemental.datasets.get(sector)
            if reader is None:
                continue
            used = ((reader.data_id, sector) in consumed_elemental_ids
                    or ((is_current or explicitly_requests_p2p) and sector == "p2p"))
            if not used:
                continue
            role = f"vertex-{position}-{sector}"
            mode_axes = any("mode" in name for name in reader.spec.axis_names)
            point_axes = any("point" in name for name in reader.spec.axis_names)
            sector_nodes[sector] = add_reader(
                role, reader, apply_modes=mode_axes, apply_points=point_axes,
                used_ne=(meson_used_ne if has_baryon_vertex and has_meson_vertex
                         else _DEFAULT_USED_NE),
            )
        if not sector_nodes:
            raise ValueError("Diagram vertex has no consumed registered elemental sector")
        point_node = point_right_node = None
        if "p2p" in sector_nodes:
            if elemental.point_dataset is None:
                raise ValueError("P/P elemental input is missing its actual point reader")
            if elemental.point_right_dataset is None:
                raise ValueError("P/P elemental input is missing its actual right point reader")
            p2p_spec = elemental.datasets["p2p"].spec
            definition = p2p_spec.document["definition"]
            explicit = definition["point_right_selection"] == "explicit_ordered_labels"
            point_node = add_reader(f"vertex-{position}-points", elemental.point_dataset,
                apply_points=bindings.usedNp is not None,
                point_labels=(p2p_spec.coordinates("point_left") if explicit else None))
            if explicit:
                point_right_node = add_reader(
                    f"vertex-{position}-points-right", elemental.point_right_dataset,
                    apply_points=bindings.usedNp is not None,
                    point_labels=definition["point_right_labels"],
                )
            else:
                point_right_node = point_node
        clone_key = id(vertex)
        if clone_key not in vertex_clones:
            vertex_clones[clone_key] = dict(
                kind="meson",
                operator=deepcopy(vertex.operator),
                source=vertex.dagger,
                phase_convention=vertex.phase_convention,
                boundary=vertex.boundary,
                array_module=vertex.xp,
                sampling_pool_ids=dict(elemental.sampling_pool_ids),
                sampling_groups=dict(elemental.sampling_groups),
                sector_nodes=sector_nodes,
                point_node=point_node,
                point_right_node=point_right_node,
                requested_sectors=vertex.requested_sectors,
            )
        vertex_runtime[label] = vertex_clones[clone_key]

    if not requests:
        raise ValueError("Diagram has no registered input requests")
    planned_configuration, planned_times = validate_diagram_input_specs(view_specs)
    if planned_times != times or planned_configuration != configuration:
        raise ValueError("registered input time extents changed during planning")

    line_snapshot = []
    for symbol, sector_nodes in prop_runtime.items():
        reader, sectors = raw_propagators[symbol]
        flavor = getattr(symbol, "flavor", None)
        if flavor not in {"u", "d", "s", "c", "t", "b"}:
            raise ValueError("persistent propagator line requires a declared quark flavor")
        line_sectors = []
        for sector, node in sector_nodes.items():
            spec = view_specs[node]
            document = spec.document
            line_sectors.append(dict(
                sector=sector,
                input_role=node,
                actual_data_id=requests[node].data_id,
                definition=_jsonable(document["definition"]),
                conventions=_jsonable(document["conventions"]),
                parents=list(document["parents"]),
            ))
        line_snapshot.append(dict(
            symbol=str(symbol),
            symbol_assumptions=_jsonable(dict(getattr(symbol, "assumptions0", {}))),
            flavor=flavor,
            source_tag=_tag_snapshot(getattr(symbol, "source_tag", None)),
            sink_tag=_tag_snapshot(getattr(symbol, "sink_tag", None)),
            sectors=line_sectors,
        ))

    vertex_snapshot = []
    for position, (label, vertex) in enumerate(raw_vertices.items()):
        sectors = vertex_runtime[label]["sector_nodes"]
        sector_records = []
        for sector, node in sectors.items():
            document = view_specs[node].document
            sector_records.append(dict(
                sector=sector,
                input_role=node,
                actual_data_id=requests[node].data_id,
                definition=_jsonable(document["definition"]),
                conventions=_jsonable(document["conventions"]),
                parents=list(document["parents"]),
            ))
        vertex_snapshot.append(dict(
            position=position,
            label=str(label),
            operator=_operator_snapshot(vertex),
            sectors=sector_records,
            sampling=_sampling_snapshot(vertex.elemental),
        ))

    graph_record = dict(
        adjacency=_jsonable(diagram.diagram.adjacency_matrix),
        time_list=[_time_entry_snapshot(value) for value in diagram.time_list],
        vertex_list=[str(value) for value in diagram.vertex_list],
        propagator_list=[str(value) for value in diagram.propagator_list],
    )
    unique_pool_ids = {
        group_id
        for vertex in raw_vertices.values()
        for group_id in vertex.elemental.sampling_pool_ids.values()
    }
    groups_independent = bindings.groups_independent
    if groups_independent is None:
        if len(unique_pool_ids) > 1:
            raise ValueError("multiple sampling pools require an explicit independence declaration")
        groups_independent = "single-explicit-pool" if unique_pool_ids else "not-sampled"
    science = dict(
        graph=graph_record,
        source_time_symbol=str(source_time_symbol),
        sink_time_symbol=str(sink_time_symbol),
        configuration=configuration,
        usedNe=("all-registered" if bindings.usedNe is None else _jsonable(bindings.usedNe)),
        usedNp=("all-registered" if bindings.usedNp is None else bindings.usedNp),
        groups_independent=groups_independent,
        propagator_lines=line_snapshot,
        vertices=vertex_snapshot,
    )
    input_ids = {node: {"request": node} for node in view_specs}
    document = diagram_measurement_document(
        configuration, times, view_specs, input_ids, science,
    )
    requests["measurement"] = DatasetRequest(document)
    graph = plan_dataset_graph(requests, observations, data_ids=data_ids)
    return dict(
        requests=requests,
        graph=graph,
        measurement_spec=graph["stages"]["measurement"]["spec"],
        source_times=tuple(times),
        sink_times=tuple(times),
        time_selection=time_selection,
        sink_batch=sink_batch,
        input_roles=dict(role_nodes),
        _propagators=prop_runtime,
        _vertices=vertex_runtime,
        _diagram=diagram,
        _bindings=bindings,
        _source_symbol=source_time_symbol,
        _sink_symbol=sink_time_symbol,
        _sink_batch=sink_batch,
        _configuration=bindings.configuration,
    )


def plan_indexed_diagram_measurement(index, diagram, *, bindings,
                                    source_time_symbol, sink_time_symbol,
                                    data_ids=None, sink_batch=None,
                                    source_policy="allow_unknown"):
    """Preview a fixed Diagram over all declared source/sink coordinate pairs.

    Unmarked inputs retain full physical times. Explicitly compact inputs must
    share one ordered selection; output coordinates are slots with that mapping.
    Asymmetric/mixed coordinate spaces require native calculate instead.

    Planning expands the existing Diagram adapter for each scalar pair. An
    explicit ``sink_batch`` additionally preflights one ordered sink scan per
    source row through E's existing dense batching adapter. It reads metadata
    only, except that sparse P/P support may read registered point-coordinate
    rows to establish the record layout. Propagator and elemental numerical
    blocks are never requested here.
    """
    result = _build_requests(
        index, diagram, bindings, source_time_symbol, sink_time_symbol,
        data_ids=data_ids, sink_batch=sink_batch,
    )
    result["graph"] = apply_source_policy(
        result["graph"], result["requests"], index, source_policy=source_policy,
    )
    return {key: result[key] for key in (
        "requests", "graph", "measurement_spec", "source_times", "sink_times",
        "input_roles", "sink_batch", "time_selection",
    )}


__all__ = ['plan_indexed_diagram_measurement']
