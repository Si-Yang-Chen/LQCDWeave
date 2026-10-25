"""Pure specifications for orthonormal endpoint-complement propagators."""

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import dataset_time_selection
from lqcdweave.planning.overlap import overlap_dataset_document


HIGHMODE_SCHEME = "orthonormal_laplace_complement_v1"
HIGHMODE_INPUTS = {
    "psv": ("vsv", "psv", "overlap_sink"),
    "vsp": ("vsv", "vsp", "overlap_source"),
    "psp": ("vsv", "psv", "vsp", "psp", "overlap_sink", "overlap_source"),
}
_TAILS = {
    "vsv": ("sink_mode", "source_mode"),
    "psv": ("sink_point", "sink_color", "source_mode"),
    "vsp": ("sink_mode", "source_point", "source_color"),
    "psp": ("sink_point", "sink_color", "source_point", "source_color"),
}


def _binding(spec, axis):
    binding = spec.document["bases"].get(axis)
    if (not isinstance(binding, dict) or not isinstance(binding.get("data_id"), str)
            or not binding["data_id"] or binding.get("indices") != spec.coordinates(axis)):
        raise ValueError(f"{axis} requires an actual basis ID and exact ordered labels")
    return binding


def _equal_binding(left, left_axis, right, right_axis):
    if _binding(left, left_axis) != _binding(right, right_axis):
        raise ValueError(f"incompatible actual endpoint: {left_axis}/{right_axis}")


def select_highmode_input_views(product, inputs):
    """Select raw sectors by overlap projectors and the output's free endpoints.

    This is metadata selection, including arbitrary ordered mode labels. It
    never reduces or reorders an overlap's mode set: that set defines Q.
    Numerical readers must later apply the returned exact views.
    """
    product = str(product).lower()
    if product not in HIGHMODE_INPUTS or set(inputs) != set(HIGHMODE_INPUTS[product]):
        raise ValueError("high-mode inputs must exactly match the product roles")
    wanted = {}
    for side in ("sink", "source"):
        role = "overlap_" + side
        if role in inputs:
            wanted[side + "_mode"] = _binding(inputs[role], "mode")
            wanted[side + "_point"] = _binding(inputs[role], "point")
        else:
            axis = side + "_mode"
            wanted[axis] = _binding(inputs[product], axis)
    selected = {}
    for role, spec in inputs.items():
        if role not in _TAILS:
            selected[role] = spec
            continue
        doc = spec.document
        for axis in _TAILS[role]:
            if axis.endswith("color"):
                continue
            actual, target = _binding(spec, axis), wanted[axis]
            if actual["data_id"] != target["data_id"]:
                raise ValueError(f"{axis} has a different actual basis identity")
            if any(label not in actual["indices"] for label in target["indices"]):
                raise ValueError(f"{axis} lacks requested ordered coverage")
            doc["bases"][axis] = target
            for item in doc["axes"]:
                if item["name"] == axis:
                    item["coordinates"] = target["indices"]
        selected[role] = DatasetSpec(doc)
    return selected


def highmode_dataset_document(product, inputs, input_ids, *, basis_specs, point_specs):
    """Build one output from exact selected input specs and actual identities.

    Input views are chosen by the caller before this pure validation. Overlap
    mode coordinates define the projector, so they cannot be silently cropped.
    ``basis_specs`` and ``point_specs`` map actual IDs to the registered views
    embedded in the overlaps. Storage verification remains the caller's duty.
    """
    product = str(product).lower()
    if product not in HIGHMODE_INPUTS:
        raise ValueError("high-mode product must be PSV, VSP or PSP")
    roles = HIGHMODE_INPUTS[product]
    if set(inputs) != set(roles) or set(input_ids) != set(roles):
        raise ValueError("high-mode inputs must exactly match the product roles")
    if any(not isinstance(inputs[role], DatasetSpec) for role in roles):
        raise TypeError("high-mode inputs must be DatasetSpec objects")
    if any(not isinstance(input_ids[role], str) or not input_ids[role] for role in roles):
        raise ValueError("high-mode inputs need actual registered identities")
    anchor = inputs["vsv"].document
    cfg = anchor["configuration"]
    if any(cfg[key] is None for key in ("ensemble", "configuration")):
        raise ValueError("high-mode inputs require a known configuration")
    times = list(range(cfg["lattice"][3]))
    time_selections = {axis: dataset_time_selection(inputs["vsv"], axis) for axis in ("source_time", "sink_time")}
    science = {key: anchor["definition"].get(key)
               for key in ("gauge_id", "dirac", "solver_tolerance")}
    tolerance = science["solver_tolerance"]
    if (not isinstance(science["gauge_id"], str) or not science["gauge_id"]
            or not isinstance(science["dirac"], dict) or not science["dirac"]
            or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("explicit gauge, Dirac and positive solver tolerance required")
    conventions = anchor["conventions"]
    if (conventions.get("normalization") != "unscaled"
            or conventions.get("spin_basis") != "degrand_rossi"
            or conventions.get("temporal_boundary") not in ("periodic", "antiperiodic")):
        raise ValueError("unsupported propagator conventions")
    for role in roles:
        spec = inputs[role]
        doc = spec.document
        if doc["configuration"] != cfg or doc["dtype"] not in ("complex64", "complex128"):
            raise ValueError("high-mode inputs differ in configuration or complex dtype")
        if role not in _TAILS:
            continue
        if (doc["kind"] != role or spec.axis_names !=
                ("source_time", "sink_time", "sink_spin", "source_spin") + _TAILS[role]):
            raise ValueError("canonical propagator axes required")
        for axis in ("source_time", "sink_time"):
            expected_selection = time_selections[axis]
            if dataset_time_selection(spec, axis) != expected_selection:
                raise ValueError("raw sectors have incompatible ordered time selections")
            expected_times = times if expected_selection is None else list(range(expected_selection.Nt))
            if spec.coordinates(axis) != expected_times:
                raise ValueError("complete canonical physical time coverage required")
        for axis in ("sink_spin", "source_spin"):
            if spec.coordinates(axis) != list(range(4)):
                raise ValueError("canonical spin coordinates required")
        for axis in _TAILS[role]:
            if axis.endswith("color"):
                if spec.coordinates(axis) != [0, 1, 2]:
                    raise ValueError("canonical color coordinates required")
            else:
                _binding(spec, axis)
        current_conventions = dict(doc["conventions"])
        if role == "vsp" and doc["definition"].get("derivation") == "gamma5_hermiticity":
            if current_conventions.pop("gamma5", None) != "degrand_rossi.gamma(15)":
                raise ValueError("unsupported gamma5 input convention")
        if (any(doc["definition"].get(key) != value for key, value in science.items())
                or current_conventions != conventions
                or "high_mode_projection" in doc["definition"]):
            raise ValueError("raw sectors differ in Dirac/conventions or already contain a projector")
    if "psv" in inputs:
        _equal_binding(inputs["vsv"], "source_mode", inputs["psv"], "source_mode")
    if "vsp" in inputs:
        _equal_binding(inputs["vsv"], "sink_mode", inputs["vsp"], "sink_mode")
    for side, raw in (("sink", "psv"), ("source", "vsp")):
        role = "overlap_" + side
        if role not in inputs:
            continue
        overlap = inputs[role]
        mode = _binding(overlap, "mode")
        point = _binding(overlap, "point")
        basis_key = ((role, mode["data_id"]) if (role, mode["data_id"]) in basis_specs
                     else mode["data_id"])
        point_key = ((role, point["data_id"]) if (role, point["data_id"]) in point_specs
                     else point["data_id"])
        basis, points = basis_specs[basis_key], point_specs[point_key]
        if basis.document["conventions"].get("normalization") != "orthonormal":
            raise ValueError("high-mode projector requires an orthonormal basis")
        _, _, expected, _ = overlap_dataset_document(basis, points,
            basis_id=mode["data_id"], points_id=point["data_id"],
            selection=time_selections[side + "_time"])
        if DatasetSpec(expected).document != overlap.document:
            raise ValueError("overlap differs from its actual basis and point views")
        _equal_binding(overlap, "mode", inputs["vsv"], side + "_mode")
        _equal_binding(overlap, "point", inputs[raw], side + "_point")
        if product == "psp":
            _equal_binding(inputs[raw], side + "_point", inputs["psp"], side + "_point")
    output = inputs[product].document
    output["conventions"] = dict(conventions)
    output["dtype"] = np.dtype(np.result_type(*[
        np.dtype(inputs[role].document["dtype"]) for role in roles])).name
    output["parents"] = sorted(set(input_ids.values()))
    output["definition"] = dict(science, high_mode_projection=dict(
        scheme=HIGHMODE_SCHEME,
        sides=(["sink", "source"] if product == "psp" else
               ["sink" if product == "psv" else "source"]),
        overlap_representation="eta_dagger_xi"),
        input_roles={role: input_ids[role] for role in roles},
        input_views={role: inputs[role].document for role in roles})
    return DatasetSpec(output).document


def validate_highmode_spec(spec, *, allow_output_view=False):
    """Rebuild a derived record from its embedded views, without storage I/O.

    Output views may select existing endpoint labels without changing the
    embedded projector or derivation. Full published records remain exact.
    This proves internal consistency only. Indexed consumers must separately
    verify the declared views against the actual registered parents.
    """
    if not isinstance(spec, DatasetSpec):
        raise TypeError("high-mode validation requires a DatasetSpec")
    if type(allow_output_view) is not bool:
        raise TypeError("allow_output_view must be bool")
    document = spec.document
    definition = document["definition"]
    roles = definition.get("input_roles")
    views = definition.get("input_views")
    if not isinstance(roles, dict) or not isinstance(views, dict):
        raise ValueError("high-mode product lacks input roles and views")
    inputs = {role: DatasetSpec(view) for role, view in views.items()}
    bases, points = {}, {}
    for role, view in inputs.items():
        if not role.startswith("overlap_"):
            continue
        overlap = view.document["definition"]
        embedded = overlap.get("input_views", {})
        if set(embedded) != {"basis", "points"}:
            raise ValueError("high-mode overlap lacks its input views")
        bases[(role, overlap["basis_id"])] = DatasetSpec(embedded["basis"])
        points[(role, overlap["points_id"])] = DatasetSpec(embedded["points"])
    expected = highmode_dataset_document(document["kind"], inputs, roles,
        basis_specs=bases, point_specs=points)
    if allow_output_view:
        for axis in expected["axes"]:
            name = axis["name"]
            if name not in expected["bases"]:
                continue
            labels = spec.coordinates(name)
            if any(label not in axis["coordinates"] for label in labels):
                raise ValueError("high-mode output view requests unavailable endpoint labels")
            axis["coordinates"] = labels
            expected["bases"][name]["indices"] = labels
    if DatasetSpec(expected).document != document:
        raise ValueError("high-mode product does not match its declared derivation")
    return inputs, dict(roles)
