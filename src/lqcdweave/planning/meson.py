"""Pure compatibility checks for the first full-time connected measurement."""

from lqcdweave.contracts.elemental import GaugePath
from lqcdweave.contracts.time_selection import dataset_time_selection, bind_time_selections


VSV_AXES = ("sink_time", "source_time", "sink_spin", "source_spin", "sink_mode", "source_mode")
ELEMENTAL_AXES = ("component", "momentum", "time", "sink_mode", "source_mode")


def _known(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"unknown {label}; explicit registration is required")


def validate_meson_inputs(forward, reverse, sink, source):
    """Require a single actual V basis and all physical source/sink times.

    This deliberately supports one common ordered mode subset, not arbitrary
    mixed endpoints. Dirac definitions may differ between the two quark lines.
    There is no inference from matching shapes, filenames, Ne, or seeds.
    """
    specs = (forward, reverse, sink, source)
    documents = [s.document for s in specs]
    cfg = documents[0]["configuration"]
    _known(cfg["ensemble"], "ensemble identity")
    _known(cfg["configuration"], "configuration identity")
    lt = cfg["lattice"][3]
    selection = dataset_time_selection(forward, "source_time")
    execution_lt = lt if selection is None else selection.Nt
    reference_basis = None
    boundaries = []
    for i, (spec, d) in enumerate(zip(specs, documents)):
        propagator = i < 2
        if d["configuration"] != cfg:
            raise ValueError("configuration or lattice mismatch")
        if d["kind"] != ("vsv" if propagator else "elemental_vv"):
            raise ValueError("wrong dataset kind for meson input")
        supported_axes = ((VSV_AXES, ("source_time", "sink_time", *VSV_AXES[2:])) if propagator else
                          (ELEMENTAL_AXES, ("time", "component", "momentum", "sink_mode", "source_mode")))
        if spec.axis_names not in supported_axes:
            raise ValueError("unsupported axes; register an explicit canonical layout")
        if d["dtype"] not in ("complex64", "complex128"):
            raise ValueError("meson inputs require complex storage")
        for name in (("sink_time", "source_time") if propagator else ("time",)):
            coords = spec.coordinates(name)
            if dataset_time_selection(spec, name) != selection:
                raise ValueError("meson inputs require matching ordered time selections")
            if any(type(t) is not int for t in coords) or sorted(coords) != list(range(execution_lt)):
                raise ValueError("full physical time coverage is required")
        for name in ("sink_mode", "source_mode"):
            basis = d["bases"].get(name)
            if basis is None:
                raise ValueError(f"missing {name} basis binding")
            _known(basis["data_id"], "actual basis identity")
            if reference_basis is None:
                reference_basis = basis
            elif basis != reference_basis:
                raise ValueError("actual basis identity or ordered subset mismatch")
        definition, conventions = d["definition"], d["conventions"]
        _known(definition.get("gauge_id"), "gauge variant identity")
        if conventions.get("normalization") != "unscaled":
            raise ValueError("declare unscaled inputs; no implicit normalization conversion")
        if propagator:
            for name in ("sink_spin", "source_spin"):
                if spec.coordinates(name) != [0, 1, 2, 3]:
                    raise ValueError("spin axes must be ordered 0,1,2,3")
            if conventions.get("spin_basis") != "degrand_rossi":
                raise ValueError("explicit DeGrand-Rossi spin basis required")
            boundary = conventions.get("temporal_boundary")
            if boundary not in ("periodic", "antiperiodic"):
                raise ValueError("declare periodic or antiperiodic fermion boundary")
            boundaries.append(boundary)
            dirac = definition.get("dirac", {})
            _known(dirac.get("action"), "Dirac action")
            if type(dirac.get("mass")) not in (int, float):
                raise ValueError("explicit Dirac mass required")
            tolerance = definition.get("solver_tolerance")
            if type(tolerance) not in (int, float) or not 0 < tolerance < 1:
                raise ValueError("explicit solver tolerance required separately from storage dtype")
        else:
            if conventions.get("phase") not in ("anchor", "midpoint", "none"):
                raise ValueError("explicit elemental phase convention required")
            for p in spec.coordinates("momentum"):
                if not isinstance(p, list) or len(p) != 3 or any(type(x) is not int for x in p):
                    raise ValueError("momentum coordinates must be integer XYZ triples")
            for component in spec.coordinates("component"):
                if not isinstance(component, dict) or set(component) != {"path"}:
                    raise ValueError("this workflow supports explicitly stored raw spatial paths")
                path = GaugePath(tuple(component["path"]))
                if any(step >= 6 for step in path.steps):
                    raise ValueError("time-displaced paths are outside the equal-time workflow")
    if boundaries[0] != boundaries[1]:
        raise ValueError("mixed fermion temporal boundaries are not supported")
    return tuple(range(execution_lt))


def meson_measurement_document(configuration, times, input_specifications, input_ids,
                               sink_definition, source_definition):
    """The shared full-time output definition for concrete inputs or graph templates."""
    roles = ("forward", "reverse", "sink_elemental", "source_elemental")
    input_specifications, input_ids = tuple(input_specifications), tuple(input_ids)
    if len(input_specifications) != 4 or len(input_ids) != 4:
        raise ValueError("four explicit meson input specifications and identities are required")
    selection = dataset_time_selection(input_specifications[0], "source_time")
    if selection is not None:
        times = selection.times
    document = dict(schema=1, kind="meson_connected", configuration=configuration,
        axes=[dict(name="source_time", coordinates=list(times)),
              dict(name="sink_time", coordinates=list(times))],
        dtype="complex128", bases={}, parents=list(input_ids),
        definition=dict(measurement="connected_equal_time_meson",
                        sink_operator=sink_definition, source_operator=source_definition,
                        input_roles=dict(zip(roles, input_ids)),
                        input_views=dict(zip(roles, input_specifications))),
        conventions=dict(formula="-Tr[Phi_sink tau_forward Phi_source tau_reverse]",
                         source="gamma4 Gamma_dagger gamma4; spatial_dagger; conjugate_coefficient",
                         normalization="unscaled", time_coordinates="absolute",
                         spin_basis="degrand_rossi"))

    return document if selection is None else bind_time_selections(document, {"source_time": selection, "sink_time": selection})
