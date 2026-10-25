"""Registered high-mode validation and read-only planning."""

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.planning.dataset import plan_dataset_graph, plan_requirement
from lqcdweave.planning.highmode import (
    HIGHMODE_INPUTS, highmode_dataset_document, select_highmode_input_views,
    validate_highmode_spec,
)
from lqcdweave.planning.production import vsp_dataset_document
from lqcdweave.runtime.datasets import apply_source_policy


def _registered_view(index, identity, view):
    with index.open(identity) as reader:
        observed = reader.observation()
    subset = tuple(axis for axis in view.axis_names if axis in (
        "mode", "point", "sink_mode", "source_mode", "sink_point", "source_point"))
    result = plan_requirement(view, [observed], subset_axes=subset, data_id=identity)
    if result["action"] != "reuse":
        raise ValueError("registered parent does not cover its declared high-mode view")
    return subset


def _overlap_parents(index, inputs):
    bases, points = {}, {}
    for role, spec in inputs.items():
        if not role.startswith("overlap_"):
            continue
        definition = spec.document["definition"]
        views = definition.get("input_views", {})
        if set(views) != {"basis", "points"}:
            raise ValueError("overlap must retain its actual input views")
        for name, target in (("basis", bases), ("points", points)):
            identity = definition.get(name + "_id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("overlap input requires an actual parent ID")
            view = DatasetSpec(views[name])
            _registered_view(index, identity, view)
            # One actual basis may serve two endpoints with different views.
            target[(role, identity)] = view
    return bases, points


def _validate_raw_parents(index, spec):
    doc = spec.document
    definition = doc["definition"]
    if "high_mode_projection" in definition:
        raise ValueError("high-mode derivation requires raw propagator inputs")
    derivation = definition.get("derivation")
    if derivation is not None:
        if doc["kind"] != "vsp" or derivation != "gamma5_hermiticity" or len(doc["parents"]) != 1:
            raise ValueError("unsupported raw propagator derivation")
        parent_id = doc["parents"][0]
        with index.open(parent_id) as reader:
            parent = reader.spec
        if parent.document["kind"] != "psv":
            raise ValueError("gamma5 VSP requires its actual reverse PSV")
        _validate_raw_parents(index, parent)
        # Adapted from diagram_measurement._psv_request_view_for_vsp: retain
        # the established inverse axis map; this does not apply numerical gamma5.
        mapping = dict(source_time="sink_time", sink_time="source_time",
            sink_spin="source_spin", source_spin="sink_spin",
            sink_point="source_point", sink_color="source_color", source_mode="sink_mode")
        document = parent.document
        if set(parent.axis_names) != set(mapping):
            raise ValueError("reverse PSV has noncanonical axes")
        for axis in document["axes"]:
            axis["coordinates"] = spec.coordinates(mapping[axis["name"]])
            if axis["name"] in document["bases"]:
                document["bases"][axis["name"]]["indices"] = axis["coordinates"]
        view = DatasetSpec(document)
        _registered_view(index, parent_id, view)
        expected = DatasetSpec(vsp_dataset_document(view.document, psv_id=parent_id,
            gauge_id=definition.get("gauge_id"),
            basis_id=doc["bases"]["sink_mode"]["data_id"],
            points_id=doc["bases"]["source_point"]["data_id"]))
        if expected.document != doc:
            raise ValueError("VSP differs from its registered gamma5 derivation")
        return
    gauge_id = definition.get("gauge_id")
    expected = {gauge_id, *(binding["data_id"] for binding in doc["bases"].values())}
    if (not isinstance(gauge_id, str) or not gauge_id
            or set(doc["parents"]) != expected or len(doc["parents"]) != len(expected)):
        raise ValueError("raw propagator parents do not bind actual gauge and endpoints")
    with index.open(gauge_id) as gauge:
        if gauge.spec.document["kind"] != "gauge" or gauge.spec.document["configuration"] != doc["configuration"]:
            raise ValueError("raw propagator gauge has incompatible kind or configuration")
    for axis, binding in doc["bases"].items():
        with index.open(binding["data_id"]) as parent:
            parent_spec = parent.spec
        dimension = "mode" if axis.endswith("mode") else "point"
        allowed_kinds = (("explicit_basis", "laplace_basis") if dimension == "mode"
                         else ("spatial_points",))
        if parent_spec.document["kind"] not in allowed_kinds:
            raise ValueError("raw propagator endpoint kind is incompatible with its axis")
        if (parent_spec.document["configuration"] != doc["configuration"]
                or dimension not in parent_spec.axis_names
                or any(label not in parent_spec.coordinates(dimension) for label in binding["indices"])):
            raise ValueError("raw propagator endpoint lacks registered identity or coverage")


def validate_registered_highmode(index, spec):
    """Verify the complete stored derivation against registered actual parents."""
    inputs, identities = validate_highmode_spec(spec)
    for role, view in inputs.items():
        _registered_view(index, identities[role], view)
        if not role.startswith("overlap_"):
            _validate_raw_parents(index, view)
    _overlap_parents(index, inputs)
    return None


def plan_indexed_highmode_propagators(index, *, products=("PSV",), vsv_id,
        psv_id=None, vsp_id=None, psp_id=None, overlap_sink_id=None,
        overlap_source_id=None, data_ids=None, source_policy="allow_unknown"):
    """Validate registered full-time inputs and preview derived outputs.

    This workflow performs storage verification; the underlying specification
    and view builders are pure. Missing raw sectors are never produced here.
    """
    if isinstance(products, str):
        raise TypeError("products must be a sequence of sector names")
    products = tuple(str(value).lower() for value in products)
    if not products or len(set(products)) != len(products) or set(products) - set(HIGHMODE_INPUTS):
        raise ValueError("products must uniquely name PSV, VSP and/or PSP")
    if source_policy not in ("allow_unknown", "require_known"):
        raise ValueError("invalid source policy")
    identities = dict(vsv=vsv_id, psv=psv_id, vsp=vsp_id, psp=psp_id,
                      overlap_sink=overlap_sink_id, overlap_source=overlap_source_id)
    required = set().union(*(HIGHMODE_INPUTS[product] for product in products))
    if any(not isinstance(identities[role], str) or not identities[role] for role in required):
        raise ValueError("every required high-mode input needs an actual registered ID")
    original = {}
    for role in sorted(required):
        with index.open(identities[role]) as reader:
            original[role] = reader.spec
        if not role.startswith("overlap_"):
            _validate_raw_parents(index, original[role])
    basis_views, point_views = _overlap_parents(index, original)
    requests, output_specs, input_roles = {}, {}, {}
    for product in products:
        roles = HIGHMODE_INPUTS[product]
        selected = select_highmode_input_views(product, {role: original[role] for role in roles})
        # The pure builder is invoked per product. Distinct endpoint views of
        # the same ID are checked independently below rather than conflated.
        bases, points = {}, {}
        for role in roles:
            if not role.startswith("overlap_"):
                continue
            definition = selected[role].document["definition"]
            bid, pid = definition["basis_id"], definition["points_id"]
            bases[(role, bid)] = basis_views[(role, bid)]
            points[(role, pid)] = point_views[(role, pid)]
        ids = {role: identities[role] for role in roles}
        document = highmode_dataset_document(product, selected, ids,
            basis_specs=bases, point_specs=points)
        output_specs[product] = DatasetSpec(document)
        node_names = {}
        for role in roles:
            node = product + "-input-" + role
            node_names[role] = node
            subset = _registered_view(index, ids[role], selected[role])
            requests[node] = DatasetRequest(selected[role], data_id=ids[role],
                                             subset_axes=subset)
        template = output_specs[product].document
        parent_nodes = {}
        for role in roles:
            parent_nodes.setdefault(ids[role], node_names[role])
        template["parents"] = [{"request": parent_nodes[identity]}
                               for identity in sorted(parent_nodes)]
        template["definition"]["input_roles"] = {
            role: {"request": node_names[role]} for role in roles}
        requests[product] = DatasetRequest(template)
        input_roles[product] = node_names
    graph = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    graph = apply_source_policy(graph, requests, index, source_policy=source_policy)
    return dict(requests=requests, graph=graph, output_specs=output_specs,
                input_roles=input_roles)
