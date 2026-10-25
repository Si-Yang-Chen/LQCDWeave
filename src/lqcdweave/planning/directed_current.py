"""Compatibility of registered one-link current elementals and VSV inputs."""

from lqcdweave.contracts.current import DirectedCurrentBasis
from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.contracts.elemental import integer


def _known(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"unknown {label}; explicit registration is required")


def _full_times(spec, axis, extent):
    coordinates = spec.coordinates(axis)
    if any(type(t) is not int for t in coordinates) or sorted(coordinates) != list(range(extent)):
        raise ValueError(f"{axis} must cover every physical time exactly once")


def _basis(spec):
    document = spec.document
    bindings = []
    for axis in ("sink_mode", "source_mode"):
        binding = document["bases"].get(axis)
        if binding is None:
            raise ValueError(f"missing {axis} actual basis binding")
        _known(binding["data_id"], "actual basis identity")
        bindings.append(binding)
    if bindings[0] != bindings[1]:
        raise ValueError("current and VSV legs require one identical ordered basis")
    return bindings[0]


def validate_directed_current_datasets(elemental, *vsv_specs):
    """Check physical coordinates and stored identities before format adaptation.

    Input specs are DatasetSpec objects from verified dataset readers. The
    current dataset may permute its first three named axes, but its two mode
    axes must retain the old bar/field order. Every VSV has absolute times.
    """
    if not vsv_specs:
        raise ValueError("at least one registered VSV dataset is required")
    document = elemental.document
    cfg = document["configuration"]
    _known(cfg["ensemble"], "ensemble identity")
    _known(cfg["configuration"], "configuration identity")
    lt = cfg["lattice"][3]
    if document["kind"] != "elemental_vv":
        raise ValueError("directed current requires a registered elemental_vv dataset")
    axes = elemental.axis_names
    if set(axes[:3]) != {"component", "time", "momentum"} or axes[3:] != ("sink_mode", "source_mode"):
        raise ValueError("directed current elemental axes must name component,time,momentum,sink_mode,source_mode")
    if document["dtype"] not in ("complex64", "complex128"):
        raise ValueError("directed current elemental storage must be complex")
    _full_times(elemental, "time", lt)
    components = elemental.coordinates("component")
    expected = [{"path": [direction]} for direction in range(8)]
    if any(component not in components for component in expected):
        raise ValueError("all eight raw directed one-link components are required")
    momenta = elemental.coordinates("momentum")
    if any(not isinstance(momentum, list) or len(momentum) != 3 or
           any(type(value) is not int for value in momentum) for momentum in momenta):
        raise ValueError("momentum coordinates must be integer XYZ triples")
    if document["conventions"].get("phase") != "midpoint":
        raise ValueError("directed current requires the midpoint Fourier phase")
    if document["conventions"].get("normalization") != "unscaled":
        raise ValueError("directed current requires unscaled raw elementals")
    definition = document["definition"]
    boundary = definition.get("path_boundary")
    if boundary not in ("periodic", "open"):
        raise ValueError("explicit periodic or open path boundary is required")
    if definition.get("directed_basis") != DirectedCurrentBasis.SCHEMA:
        raise ValueError("directed basis schema differs from the eight-link v1 basis")
    _known(definition.get("gauge_id"), "elemental gauge identity")
    basis = _basis(elemental)
    if definition["gauge_id"] not in document["parents"] or basis["data_id"] not in document["parents"]:
        raise ValueError("elemental parent dataset references are incomplete")

    for vsv in vsv_specs:
        data = vsv.document
        if data["kind"] != "vsv" or data["configuration"] != cfg:
            raise ValueError("VSV role or configuration differs from the current elemental")
        if vsv.axis_names not in (
            ("sink_time", "source_time", "sink_spin", "source_spin", "sink_mode", "source_mode"),
            ("source_time", "sink_time", "sink_spin", "source_spin", "sink_mode", "source_mode"),
        ):
            raise ValueError("VSV axes must have explicit absolute source and sink times")
        _full_times(vsv, "sink_time", lt)
        _full_times(vsv, "source_time", lt)
        if vsv.coordinates("sink_spin") != [0, 1, 2, 3] or vsv.coordinates("source_spin") != [0, 1, 2, 3]:
            raise ValueError("VSV spin coordinates must be ordered 0,1,2,3")
        if data["dtype"] not in ("complex64", "complex128"):
            raise ValueError("VSV storage must be complex")
        if _basis(vsv) != basis:
            raise ValueError("VSV actual basis identity or ordered modes differ from the current elemental")
        if basis["data_id"] not in data["parents"]:
            raise ValueError("VSV basis parent reference is missing")
        if data["conventions"].get("normalization") != "unscaled":
            raise ValueError("VSV requires unscaled normalization")
        if data["conventions"].get("spin_basis") != "degrand_rossi":
            raise ValueError("VSV requires the DeGrand-Rossi spin basis")
        fermion_boundary = data["conventions"].get("temporal_boundary")
        allowed = ("open",) if boundary == "open" else ("periodic", "antiperiodic")
        if fermion_boundary not in allowed:
            raise ValueError("VSV temporal boundary is incompatible with the elemental path boundary")
        _known(data["definition"].get("gauge_id"), "VSV gauge identity")
        if data["definition"]["gauge_id"] not in data["parents"]:
            raise ValueError("VSV gauge parent reference is missing")

    return dict(configuration=cfg, boundary=boundary, momenta=momenta,
                basis=basis, directions=expected, times=tuple(range(lt)))


def directed_current_pair_views(first, second, vsv, *, usedNe=None):
    """Validate explicit ordered mode prefixes for two currents and one VSV.

    No common size is inferred. Without usedNe the complete stored ordered
    modes must match; IDs and original producer definitions remain unchanged.
    """
    result = []
    count = None if usedNe is None else integer(usedNe, "usedNe", 1)
    for spec in (first, second, vsv):
        document = spec.document
        for name in ("sink_mode", "source_mode"):
            modes = spec.coordinates(name)
            if count is not None:
                if count > len(modes):
                    raise ValueError("usedNe exceeds registered current/VSV coverage")
                modes = modes[:count]
            for axis in document["axes"]:
                if axis["name"] == name:
                    axis["coordinates"] = modes
            if name in document["bases"]:
                document["bases"][name]["indices"] = modes
        result.append(DatasetSpec(document))
    one = validate_directed_current_datasets(result[0], result[2])
    two = validate_directed_current_datasets(result[1], result[2])
    if one["boundary"] != two["boundary"]:
        raise ValueError("paired current path boundaries must match")
    return tuple(result)


def directed_current_pair_times(extent, times, name):
    """Preserve explicit physical anchors; default to every physical time."""
    times = tuple(range(extent)) if times is None else tuple(integer(t, name, 0) for t in times)
    if not times or len(set(times)) != len(times) or any(t >= extent for t in times):
        raise ValueError(f"{name} must be nonempty unique physical times inside the lattice")
    return times


def directed_current_pair_document(first, second, vsv, input_ids, first_definition,
                                   second_definition, *, first_times=None, second_times=None):
    """Describe the existing ordered connected trace and its actual input views.

    Operator definitions are JSON snapshots prepared/validated by the workflow;
    this metadata builder does not construct operators or run contractions.
    Input IDs may be logical references until the shared graph resolves them.
    """
    first, second, vsv = directed_current_pair_views(first, second, vsv)
    ids = tuple(input_ids)
    if len(ids) != 3:
        raise ValueError("three explicit current/current/VSV identities required")
    canonical_json((first_definition, second_definition))
    extent = first.document["configuration"]["lattice"][3]
    first_times = directed_current_pair_times(extent, first_times, "first_times")
    second_times = directed_current_pair_times(extent, second_times, "second_times")
    roles = ("first_current", "second_current", "vsv")
    return dict(schema=1, kind="directed_current_pair", configuration=first.document["configuration"],
        axes=[dict(name="first_anchor_time", coordinates=list(first_times)),
              dict(name="second_anchor_time", coordinates=list(second_times))],
        dtype="complex128", bases={}, parents=list(ids),
        definition=dict(measurement="ordered_connected_directed_current_pair",
            first_operator=first_definition, second_operator=second_definition,
            input_roles=dict(zip(roles, ids)),
            input_views=dict(zip(roles, (spec.document for spec in (first, second, vsv))))),
        conventions=dict(formula="sum_terms bfji,ackl,afki,bcjl->",
            normalization="unscaled", spin_basis="degrand_rossi", time_coordinates="absolute",
            wick_sign="not_applied", flavor_factor="not_applied", source_average="not_applied",
            conjugation="none", projection="complex", interpretation="explicit_linked_operator"))


def directed_current_chain_views(current, incoming_vsv, outgoing_vsv, *, usedNe=None):
    """Validate the registered inputs for one open V/V current chain.

    The current elemental and both VSV roles must name the same actual,
    ordered eigenvector basis. Their gauge variants and the two VSV Dirac
    definitions remain independent metadata; this open tensor chain does not
    assert a conserved-current identity.
    """
    specs = (current, incoming_vsv, outgoing_vsv)
    validate_directed_current_datasets(*specs)
    for role, spec in zip(("incoming VSV", "outgoing VSV"), specs[1:]):
        definition = spec.document["definition"]
        dirac = definition.get("dirac")
        if (not isinstance(dirac, dict) or not dirac or
                not isinstance(dirac.get("action"), str) or not dirac["action"].strip()):
            raise ValueError(f"{role} needs an explicit Dirac definition")
        canonical_json(dirac)
        tolerance = definition.get("solver_tolerance")
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
                not 0 < tolerance < 1):
            raise ValueError(f"{role} needs an explicit solver tolerance between 0 and 1")

    count = None if usedNe is None else integer(usedNe, "usedNe", 1)
    result = []
    for spec in specs:
        document = spec.document
        for name in ("sink_mode", "source_mode"):
            modes = spec.coordinates(name)
            if count is not None:
                if count > len(modes):
                    raise ValueError("usedNe exceeds registered current/VSV coverage")
                modes = modes[:count]
            for axis in document["axes"]:
                if axis["name"] == name:
                    axis["coordinates"] = modes
            document["bases"][name]["indices"] = modes
        result.append(DatasetSpec(document))
    validate_directed_current_datasets(*result)
    return tuple(result)


def directed_current_chain_times(extent, times, name):
    """Preserve explicit absolute-time order; default to every physical time."""
    times = tuple(range(extent)) if times is None else tuple(integer(t, name, 0) for t in times)
    if not times or len(set(times)) != len(times) or any(t >= extent for t in times):
        raise ValueError(f"{name} must be nonempty unique physical times inside the lattice")
    return times


def directed_current_chain_document(current, incoming_vsv, outgoing_vsv, input_ids,
                                    operator_definition, *, source_times=None,
                                    sink_times=None, anchor_times=None):
    """Describe the open tensor returned by the copied V/V single-link kernel.

    The output keeps the kernel's external spin/mode axes and adds named,
    absolute source, sink and current-anchor time axes. Input IDs can remain
    logical DatasetRequest references until the shared graph resolves them.
    """
    current, incoming_vsv, outgoing_vsv = directed_current_chain_views(
        current, incoming_vsv, outgoing_vsv)
    ids = tuple(input_ids)
    if len(ids) != 3:
        raise ValueError("explicit current, incoming VSV and outgoing VSV identities required")
    canonical_json(operator_definition)
    extent = current.document["configuration"]["lattice"][3]
    source_times = directed_current_chain_times(extent, source_times, "source_times")
    sink_times = directed_current_chain_times(extent, sink_times, "sink_times")
    anchor_times = directed_current_chain_times(extent, anchor_times, "anchor_times")
    outgoing_modes = outgoing_vsv.coordinates("sink_mode")
    incoming_modes = incoming_vsv.coordinates("source_mode")
    output_axes = [
        dict(name="source_time", coordinates=list(source_times)),
        dict(name="sink_time", coordinates=list(sink_times)),
        dict(name="anchor_time", coordinates=list(anchor_times)),
        dict(name="external_sink_spin", coordinates=[0, 1, 2, 3]),
        dict(name="external_source_spin", coordinates=[0, 1, 2, 3]),
        dict(name="external_sink_mode", coordinates=list(outgoing_modes)),
        dict(name="external_source_mode", coordinates=list(incoming_modes)),
    ]
    basis_id = current.document["bases"]["sink_mode"]["data_id"]
    roles = ("current", "incoming_vsv", "outgoing_vsv")
    return dict(schema=1, kind="directed_current_chain",
        configuration=current.document["configuration"], axes=output_axes,
        dtype="complex128", bases={
            "external_sink_mode": dict(data_id=basis_id, indices=list(outgoing_modes)),
            "external_source_mode": dict(data_id=basis_id, indices=list(incoming_modes)),
        }, parents=list(ids),
        definition=dict(measurement="open_directed_current_vsv_chain",
            contraction_schema="lattice.current.v2v-termwise-contraction/v1",
            operator=operator_definition,
            time_selection=dict(source=list(source_times), sink=list(sink_times),
                                anchor=list(anchor_times)),
            input_roles=dict(zip(roles, ids)),
            input_views=dict(zip(roles, (spec.document for spec in
                                         (current, incoming_vsv, outgoing_vsv))))),
        conventions=dict(time_coordinates="absolute", spin_basis="degrand_rossi",
            normalization="termwise_input_normalization", projection="complex",
            conjugation="none", wick_sign="not_applied", flavor_factor="not_applied",
            source_average="not_applied", interpretation="open_tensor_chain_no_ward_claim"))
