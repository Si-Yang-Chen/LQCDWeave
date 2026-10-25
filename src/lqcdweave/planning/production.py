"""Input compatibility and logical estimates for registered-field production."""

from collections.abc import Mapping
import math
import numbers
from typing import Literal

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import (
    TimeSelectionRecord, normalize_time_selection, dataset_physical_times,
    dataset_time_selection, bind_time_selections,
)
from lqcdweave.contracts.current import (
    CURRENT_DIRECTED_POINT_SCHEMA, DirectedCurrentBasis,
)
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, GaugePath, integer, spatial_projection_rows,
)
from lqcdweave.contracts.memory import MemoryEstimate
from lqcdweave.contracts.dilution import DILUTION_WEIGHT_SCHEME


_PERAMBULATOR_COMPLEX_ITEMSIZE = {"complex64": 8, "complex128": 16}
_PERAMBULATOR_POINT_ITEMSIZE = {"int32": 4, "int64": 8}


def validate_gauge_field(gauge):
    """Require a known, complete, canonical registered gauge field."""
    d = gauge.document
    if dataset_time_selection(gauge) is not None:
        raise ValueError("gauge fields require full physical time axes, not compact slot mappings")
    cfg = d["configuration"]
    if not cfg["ensemble"] or not cfg["configuration"] or d["kind"] != "gauge":
        raise ValueError("registered gauge with known configuration required")
    axes = ("time", "z", "y", "x", "direction", "color_row", "color_column")
    lx,ly,lz,lt = cfg["lattice"]
    if gauge.axis_names != axes or d["dtype"] not in ("complex64", "complex128"):
        raise ValueError("unsupported gauge axes or dtype")
    for name, extent in zip(axes, (lt,lz,ly,lx,4,3,3)):
        coordinates = gauge.coordinates(name)
        if (any(type(c) is not int for c in coordinates)
                or (sorted(coordinates) if name == "time" else coordinates) != list(range(extent))):
            raise ValueError("complete time coverage and canonical gauge coordinates required")
    return tuple(range(lt))


def validate_elemental_fields(gauge, basis, *, selection=None):
    """Check canonical field axes and complete physical time coverage, without I/O."""
    if dataset_time_selection(gauge) is not None:
        raise ValueError("gauge fields require full physical time axes, not compact slot mappings")
    gd, bd = gauge.document, basis.document
    cfg = gd["configuration"]
    if any(not cfg[k] for k in ("ensemble", "configuration")):
        raise ValueError("known ensemble and configuration identities are required")
    if bd["configuration"] != cfg:
        raise ValueError("gauge and basis configuration/lattice mismatch")
    if gd["kind"] != "gauge" or bd["kind"] not in ("explicit_basis", "laplace_basis"):
        raise ValueError("registered gauge and actual vector basis datasets are required")
    lx, ly, lz, lt = cfg["lattice"]
    layouts = ((gauge, ("time", "z", "y", "x", "direction", "color_row", "color_column")),
               (basis, ("time", "mode", "z", "y", "x", "color")))
    extents = dict(time=lt, z=lz, y=ly, x=lx, direction=4,
                   color_row=3, color_column=3, color=3)
    for spec, axes in layouts:
        if spec.axis_names != axes or spec.document["dtype"] not in ("complex64", "complex128"):
            raise ValueError("unsupported field axes or dtype")
        for name in axes:
            coords = spec.coordinates(name)
            if any(type(n) is not int or n < 0 for n in coords):
                raise ValueError("field coordinates must be nonnegative integers")
            if name == "mode":
                continue  # Preserve actual stored mode labels and their order.
            expected = list(range(extents[name]))
            if name == "time" and spec is basis and selection is not None:
                required = normalize_time_selection(selection, Lt=lt)
                available = TimeSelectionRecord(lt, dataset_physical_times(spec))
                available.require_times(required.times)
                continue
            if (sorted(coords) if name == "time" else coords) != expected:
                raise ValueError("complete time coverage and canonical spatial/color order required")
    return tuple(range(lt)) if selection is None else normalize_time_selection(selection, Lt=lt).times


def elemental_dataset_document(gauge, basis, *, gauge_id, basis_id, components,
                                momenta, modes, phase_convention, selection=None,
                                product="v2v", right_basis=None, right_basis_id=None,
                                right_modes=None, elemental_weighted=False):
    """Describe a raw spatial elemental using the existing field contract.

    IDs may be explicit logical references in a DatasetRequest template;
    only resolving them to actual IDs produces a publishable DatasetSpec.
    Callers supply already normalized path/momentum coordinates, not filenames.
    """
    from lqcdweave.contracts.elemental_sectors import sector_kinds
    left, right = sector_kinds(product)
    if "p" in (left, right):
        raise ValueError("vector elemental planner requires V/D endpoints")
    times = validate_elemental_fields(gauge, basis, selection=selection)
    right_basis = basis if right_basis is None else right_basis
    right_basis_id = basis_id if right_basis_id is None else right_basis_id
    validate_elemental_fields(gauge, right_basis, selection=_endpoint_selection(selection, components))
    modes = list(modes)
    right_modes = modes if right_modes is None else list(right_modes)
    for labels, handle in ((modes, basis), (right_modes, right_basis)):
        if not labels or labels != handle.coordinates("mode")[:len(labels)]:
            raise ValueError("elemental modes must be an explicit stored basis prefix")
    def axis(name, values):
        return dict(name=name, coordinates=list(values))
    document = dict(schema=1, kind="elemental_vv", configuration=gauge.document["configuration"],
        axes=[axis("time", times), axis("component", components), axis("momentum", momenta),
              axis("sink_mode", modes), axis("source_mode", right_modes)],
        dtype="complex128", bases=dict(sink_mode=dict(data_id=basis_id, indices=modes),
            source_mode=dict(data_id=right_basis_id, indices=right_modes)),
        definition=dict(gauge_id=gauge_id, path_boundary="periodic"),
        conventions=dict(normalization="unscaled", phase=phase_convention),
        parents=[gauge_id, basis_id] + ([] if right_basis_id == basis_id else [right_basis_id]))
    if product != "v2v":
        document["definition"]["logical_sector"] = product
    _bind_dilution_conventions(document, (("bar", left, basis, modes),
                                        ("field", right, right_basis, right_modes)), elemental_weighted)
    conventions = basis.document["conventions"]
    if (left == right and basis_id == right_basis_id and modes == right_modes
            and (conventions.get("normalization") == "orthonormal" or conventions.get("basis_orthogonal") is True)):
        structures = [dict(component=ci, momentum=mi, structure="diagonal")
                      for ci, component in enumerate(components) if not component["path"]
                      for mi, momentum in enumerate(momenta) if tuple(momentum) == (0, 0, 0)]
        if structures:
            document["conventions"]["elemental_structures"] = structures
    return document if selection is None else bind_time_selections(document, {"time": selection})


def _bind_dilution_conventions(document, endpoints, elemental_weighted):
    """Persist explicit D source records without inventing labels or pools."""
    from lqcdweave.contracts.dilution import DilutionSource
    sources = {}
    for side, kind, basis, labels in endpoints:
        if kind != "d":
            continue
        document["conventions"]["dilution_weight_scheme"] = DILUTION_WEIGHT_SCHEME
        conventions = basis.document["conventions"]
        source = DilutionSource.from_document(conventions.get("dilution_source", conventions.get("dilution_basis")))
        if source is None:
            if elemental_weighted:
                raise ValueError("weighted D elemental registration requires explicit dilution source")
            continue
        source.validate_selection(labels)
        if source.data_id != document["bases"]["sink_mode" if side == "bar" and "sink_mode" in document["bases"]
                else "source_mode" if side == "field" and "source_mode" in document["bases"]
                else "mode_left" if side == "bar" else "mode_right"]["data_id"]:
            raise ValueError("D source identity differs from the bound actual vector dataset")
        sources[side] = source.document
    if sources:
        document["conventions"]["dilution_endpoints"] = sources
    if elemental_weighted:
        document["conventions"]["elemental_weighted"] = True



def _endpoint_selection(selection, components, *, boundary="periodic"):
    """Resolve actual final path endpoints before indexing any compact field."""
    if selection is None:
        return None
    record = normalize_time_selection(selection)
    endpoints = []
    for time in record.times:
        for component in components:
            path = GaugePath.coerce(component["path"])
            lo, hi = path.time_bounds
            if boundary == "open" and (time + lo < 0 or time + hi >= record.Lt):
                continue
            endpoint = (time + path.displacement[3]) % record.Lt
            if endpoint not in endpoints:
                endpoints.append(endpoint)
    # No valid open-boundary endpoint means no field data is consumed. The
    # anchor record remains a conservative input requirement for that case.
    return TimeSelectionRecord(record.Lt, tuple(endpoints or record.times))


def directed_current_dataset_document(gauge, basis, *, gauge_id, basis_id,
                                      momentum_list=((0, 0, 0),), usedNe=None,
                                      boundary="periodic", selection=None):
    """Describe the existing eight-link V/V basis without constructing a producer.

    Physical time is complete; mode labels preserve the actual ordered prefix.
    Gauge and basis IDs may be logical DatasetRequest references until resolved.
    The exact requested input views are part of the persistent definition.
    """
    if selection is not None:
        selection = normalize_time_selection(selection, Lt=gauge.document["configuration"]["lattice"][3])
        endpoints = _endpoint_selection(selection, [dict(path=[i]) for i in range(8)], boundary=boundary)
        required = TimeSelectionRecord(selection.Lt, tuple(dict.fromkeys((*selection.times, *endpoints.times))))
    else:
        required = None
    validate_elemental_fields(gauge, basis, selection=required)
    if boundary not in ("periodic", "open"):
        raise ValueError("boundary must be 'periodic' or 'open'")
    momenta = [list(integer(n, "momentum") for n in p) for p in momentum_list]
    if not momenta or any(len(p) != 3 for p in momenta):
        raise ValueError("nonempty XYZ momenta are required")
    if len({tuple(p) for p in momenta}) != len(momenta):
        raise ValueError("momentum coordinates must be unique")
    modes = basis.coordinates("mode")
    count = len(modes) if usedNe is None else integer(usedNe, "usedNe", 1)
    if count > len(modes):
        raise ValueError("usedNe exceeds registered basis coverage")
    modes = modes[:count]
    view = basis.document
    view["axes"][1]["coordinates"] = modes
    if "mode" in view["bases"]:
        view["bases"]["mode"]["indices"] = modes
    requested = DatasetSpec(view)
    document = elemental_dataset_document(gauge, requested, gauge_id=gauge_id,
        basis_id=basis_id, components=[dict(path=[i]) for i in range(8)],
        momenta=momenta, modes=modes, phase_convention="midpoint", selection=selection)
    document["definition"].update(path_boundary=boundary,
        directed_basis=DirectedCurrentBasis.SCHEMA,
        input_views=dict(gauge=gauge.document, basis=requested.document))
    return requested, document


def validate_point_field(gauge, points, *, selection=None):
    """Require a registered unit-delta point realization at every physical time."""
    times = validate_gauge_field(gauge)
    if selection is not None:
        required = normalize_time_selection(selection, Lt=len(times))
        TimeSelectionRecord(required.Lt, dataset_physical_times(points)).require_times(required.times)
        times = required.times
    document = points.document
    if document["configuration"] != gauge.document["configuration"]:
        raise ValueError("gauge and point configuration/lattice mismatch")
    if (document["kind"] != "spatial_points" or points.axis_names != ("time", "point", "xyz")
            or document["dtype"] not in ("int32", "int64")
            or points.coordinates("xyz") != ["x", "y", "z"]
            or any(type(time) is not int for time in points.coordinates("time"))
            or (selection is None and sorted(points.coordinates("time")) != list(times))):
        raise ValueError("complete canonical spatial point dataset required")
    if any(type(label) is not int or label < 0 for label in points.coordinates("point")):
        raise ValueError("point labels must be explicit nonnegative integer indices")
    conventions = document["conventions"]
    if (conventions.get("representation") != "coordinate_delta" or
            conventions.get("normalization") != "unit_delta"):
        raise ValueError("explicit coordinate_delta/unit_delta point conventions required")
    return times


def point_elemental_dataset_document(gauge, basis, points, *, product, gauge_id,
                                      basis_id, points_id, components, momenta,
                                      modes=(), point_labels=(),
                                      points_right=None, points_right_id=None,
                                      point_right_labels=None,
                                      phase_convention="anchor",
                                      current_schema=None, temporal_path_schema=None,
                                      path_boundary="periodic", selection=None, elemental_weighted=False):
    """Describe one point-ended elemental over registered inputs.

    The point-pair product uses fixed left-point row slots. Its right-point
    structure is reconstructed from the explicitly bound point dataset and
    component path; values are never used to infer sparse support. IDs may be
    logical DatasetRequest references until the request graph resolves them.
    """
    from lqcdweave.contracts.elemental_sectors import sector_layout
    logical_product = product
    product = sector_layout(product)
    components = list(components)
    if selection is not None:
        selection = normalize_time_selection(selection, Lt=gauge.document["configuration"]["lattice"][3])
        endpoints = _endpoint_selection(selection, components, boundary=path_boundary)
    else:
        endpoints = None
    if product != "v2p" or points_right is None:
        validate_point_field(gauge, points, selection=endpoints if product == "v2p" else selection)
    times = tuple(range(gauge.document["configuration"]["lattice"][3])) if selection is None else selection.times
    products = {"v2p", "p2v", "p2p"}
    if product not in products:
        raise ValueError("product must be v2p, p2v or p2p")
    vector_product = product in {"v2p", "p2v"}
    point_current = current_schema is not None
    temporal_path = temporal_path_schema is not None
    if point_current and temporal_path:
        raise ValueError("current_schema and temporal_path_schema are mutually exclusive")
    if temporal_path and temporal_path_schema != TEMPORAL_POINT_PATH_SCHEMA:
        raise ValueError("unsupported temporal point-path schema")
    if temporal_path and product not in {"v2p", "p2v", "p2p"}:
        raise ValueError("temporal point-path schema supports V/P, P/V and P/P")
    if point_current:
        if current_schema != CURRENT_DIRECTED_POINT_SCHEMA:
            raise ValueError("unsupported point-current schema")
        if product not in {"v2p", "p2v", "p2p"}:
            raise ValueError("point-current schema supports V/P, P/V and P/P products")
        if path_boundary not in {"periodic", "open"}:
            raise ValueError("point-current path boundary must be periodic or open")
        if phase_convention != "midpoint":
            raise ValueError("point-current products require midpoint phase")
    elif path_boundary not in (("periodic", "open") if temporal_path else ("periodic",)):
        raise ValueError("ordinary point-elemental products require periodic paths")
    if vector_product:
        if basis is None:
            raise ValueError("V/P elemental products require an actual vector basis")
        validate_elemental_fields(gauge, basis, selection=selection if product == "v2p" else endpoints)
        if not basis_id:
            raise ValueError("V/P elemental products require an actual basis identity")
    elif basis_id is not None:
        raise ValueError("P/P elemental products do not bind a vector basis")
    if phase_convention not in ("anchor", "midpoint", "none"):
        raise ValueError("phase_convention must be anchor, midpoint or none")

    point_labels = list(point_labels)
    if any(type(label) is not int or label < 0 for label in point_labels):
        raise ValueError("point labels must be explicit nonnegative integers")
    if points_right is not None:
        validate_point_field(gauge, points_right, selection=endpoints)
        if points_right_id is None:
            raise ValueError("a separate right point view requires its actual point ID")
    right_points = points if points_right is None else points_right
    if product == "p2p":
        validate_point_field(gauge, right_points, selection=endpoints)
    right_id = points_id if points_right_id is None else points_right_id

    def validate_point_labels(labels, spec, *, prefix=False, name="point"):
        labels = list(labels)
        available = spec.coordinates("point")
        if (not labels or any(type(label) is not int or label < 0 for label in labels)
                or len(set(labels)) != len(labels)
                or any(label not in available for label in labels)):
            raise ValueError(f"{name} labels must be explicit, unique registered point coordinates")
        if prefix and labels != available[:len(labels)]:
            raise ValueError(f"{name} labels must be an explicit stored point prefix")
        return labels

    if product == "v2p":
        # Historically point_labels described this right endpoint. Keep that
        # spelling as the fallback while new callers use point_right_labels.
        selected_right = (point_labels if point_right_labels is None
                          else list(point_right_labels))
        selected_right = validate_point_labels(selected_right, right_points,
            prefix=point_right_labels is None, name="right point")
    else:
        point_labels = validate_point_labels(point_labels, points, prefix=True,
                                             name="left point")
        selected_right = None
        if product == "p2p":
            if point_right_labels is None:
                if right_id == points_id:
                    selected_right = point_labels
                else:
                    selected_right = right_points.coordinates("point")
            else:
                selected_right = point_right_labels
            selected_right = validate_point_labels(selected_right, right_points,
                prefix=False, name="right point")
        elif points_right is not None or points_right_id is not None or point_right_labels is not None:
            raise ValueError("right point inputs are unused by this product")
    modes = list(modes)
    if vector_product:
        if not modes or modes != basis.coordinates("mode")[:len(modes)]:
            raise ValueError("elemental modes must be an explicit stored basis prefix")
        if any(type(label) is not int or label < 0 for label in modes):
            raise ValueError("mode labels must be explicit nonnegative integers")
    elif modes:
        raise ValueError("P/P elemental products do not have mode coordinates")

    components = list(components)
    if not components:
        raise ValueError("at least one ordered path is required")
    normalized_components = []
    for component in components:
        if (not isinstance(component, dict) or set(component) != {"path"}
                or not isinstance(component["path"], (list, tuple))):
            raise ValueError("component coordinates must be explicit path objects")
        path = list(component["path"])
        if point_current:
            if len(path) != 1 or type(path[0]) is not int or path[0] not in (6, 7):
                raise ValueError("point-current components must be one temporal step 6 or 7")
        elif temporal_path:
            if any(type(step) is not int or not 0 <= step < 8 for step in path):
                raise ValueError("temporal point-path components require ordered directions 0..7")
        elif any(type(step) is not int or not 0 <= step < 6 for step in path):
            raise ValueError("point elemental paths must contain only ordered spatial steps 0..5")
        normalized_components.append(dict(path=path))
    if temporal_path and not any(step >= 6 for item in normalized_components
                                 for step in item["path"]):
        raise ValueError("temporal point-path schema requires at least one temporal step")
    if len({tuple(c["path"]) for c in normalized_components}) != len(normalized_components):
        raise ValueError("component paths must be unique")

    normalized_momenta = [list(integer(n, "momentum") for n in p) for p in momenta]
    if (not normalized_momenta or any(len(p) != 3 for p in normalized_momenta)
            or len({tuple(p) for p in normalized_momenta}) != len(normalized_momenta)):
        raise ValueError("unique nonempty integer XYZ momenta are required")

    def valid_reference(value):
        return ((isinstance(value, str) and bool(value)) or
                (isinstance(value, dict) and set(value) == {"request"} and
                 isinstance(value["request"], str) and bool(value["request"])))

    if not valid_reference(gauge_id) or not valid_reference(points_id):
        raise ValueError("actual gauge and point dataset references are required")
    if points_right_id is not None and not valid_reference(points_right_id):
        raise ValueError("actual right point dataset reference is required")
    if vector_product and not valid_reference(basis_id):
        raise ValueError("actual vector basis dataset reference is required")

    def axis(name, values):
        return dict(name=name, coordinates=list(values))

    bases = {}
    parents = [gauge_id]
    if vector_product:
        basis_binding = dict(data_id=basis_id, indices=modes)
        if product == "v2p":
            axes = [axis("time", times), axis("component", normalized_components),
                    axis("momentum", normalized_momenta), axis("mode_left", modes),
                    axis("point_right", selected_right), axis("color_right", range(3))]
            bases = dict(mode_left=basis_binding,
                         point_right=dict(data_id=right_id, indices=selected_right))
        else:
            axes = [axis("time", times), axis("component", normalized_components),
                    axis("momentum", normalized_momenta), axis("point_left", point_labels),
                    axis("color_left", range(3)), axis("mode_right", modes)]
            bases = dict(point_left=dict(data_id=points_id, indices=point_labels),
                         mode_right=basis_binding)
        parents.extend((basis_id, right_id if product == "v2p" else points_id))
        definition = dict(gauge_id=gauge_id, path_boundary=path_boundary)
    else:
        axes = [axis("time", times), axis("component", normalized_components),
                axis("momentum", normalized_momenta), axis("point_left", point_labels),
                axis("color_left", range(3)), axis("color_right", range(3))]
        bases = dict(point_left=dict(data_id=points_id, indices=point_labels))
        for identity in (points_id, right_id):
            if identity not in parents:
                parents.append(identity)
        if right_id == points_id and selected_right == point_labels:
            # Preserve the original same-pool scientific definition exactly.
            definition = dict(gauge_id=gauge_id, path_boundary=path_boundary,
                              point_right_id=points_id,
                              point_right_selection="same_ordered_prefix_as_point_left")
        else:
            definition = dict(gauge_id=gauge_id, path_boundary=path_boundary,
                              point_right_id=right_id,
                              point_right_selection="explicit_ordered_labels",
                              point_right_labels=selected_right)

    document = dict(schema=1,
        kind={"v2p": "elemental_v2p", "p2v": "elemental_p2v",
              "p2p": "elemental_p2p_row_sparse"}[product],
        configuration=gauge.document["configuration"], axes=axes,
        dtype="complex128", bases=bases, definition=definition,
        conventions=dict(normalization="unscaled", phase=phase_convention),
        parents=parents)
    if logical_product != product:
        document["definition"]["logical_sector"] = logical_product
    if vector_product:
        side = "bar" if product == "v2p" else "field"
        kind = logical_product[0] if side == "bar" else logical_product[-1]
        _bind_dilution_conventions(document, ((side, kind, basis, modes),), elemental_weighted)
    elif elemental_weighted:
        document["conventions"]["elemental_weighted"] = True
    if point_current:
        document["definition"]["directed_current_schema"] = current_schema
    if temporal_path:
        document["definition"]["temporal_path_schema"] = temporal_path_schema
    if product == "p2p":
        document["conventions"]["point_pair_layout"] = "left-point-row-slots-v1"
        if right_id == points_id and selected_right == point_labels:
            lattice = gauge.document["configuration"]["lattice"]
            diagonal_components = []
            for row, component in enumerate(normalized_components):
                path = GaugePath.coerce(component["path"])
                displacement = path.displacement
                if (all(offset % extent == 0
                        for offset, extent in zip(displacement[:3], lattice[:3]))
                        and displacement[3] % lattice[3] == 0):
                    diagonal_components.append(row)
            if diagonal_components:
                document["conventions"]["point_diagonal_components"] = diagonal_components
                document["conventions"]["elemental_structures"] = [
                    dict(component=row, momentum=mi,
                         structure="diagonal" if not normalized_components[row]["path"]
                         else "point_diagonal")
                    for row in diagonal_components for mi in range(len(normalized_momenta))]
    return document if selection is None else bind_time_selections(document, {"time": selection})


def propagator_dataset_document(configuration, times, *, product, gauge_id,
                                 dirac, solver_tolerance, temporal_boundary,
                                 basis_id=None, modes=(), points_id=None, point_labels=(),
                                 points_snk_id=None, point_snk_labels=(),
                                 points_src_id=None, point_src_labels=(),
                                 basis_src_id=None, modes_src=None,
                                 basis_snk_id=None, modes_snk=None,
                                 dtype="complex128", spin_basis="degrand_rossi",
                                 normalization="unscaled", source_selection=None, sink_selection=None):
    """Describe the existing single-process VSV/PSV/PSP/VSP array layouts.

    Input field compatibility is validated separately, before building a request.
    IDs may be logical references until DatasetRequest resolves them. Point
    labels name the ordered entries of a real point dataset, whose data records
    the XYZ coordinates at each time; labels and array sizes never infer a set.
    """
    modes, point_labels = list(modes), list(point_labels)
    point_snk_labels = list(point_snk_labels)
    point_src_labels = list(point_src_labels)
    if points_id is not None:
        if product == "VSP":
            raise ValueError("VSP requires an explicit source point binding")
        if (points_snk_id is not None or points_src_id is not None or
                point_snk_labels or point_src_labels):
            raise ValueError("pass points_id or explicit sink/source point bindings")
        points_snk_id = points_id
        point_snk_labels = point_labels
        if product == "PSP":
            points_src_id = points_id
            point_src_labels = point_labels
    if product not in ("VSV", "PSV", "PSP", "VSP"):
        raise ValueError("product must be VSV, PSV, PSP or VSP")
    if dtype not in ("complex64", "complex128"):
        raise ValueError("propagator dtype must be complex64 or complex128")
    if not isinstance(spin_basis, str) or not spin_basis:
        raise ValueError("explicit nonempty spin basis required")
    if normalization != "unscaled":
        raise ValueError("the current propagator kernel supports only unscaled normalization")

    endpoint_vector_args = (basis_src_id is not None or modes_src is not None or
                            basis_snk_id is not None or modes_snk is not None)
    vector_src_id = vector_snk_id = None
    modes_source = modes_sink = None
    legacy_vector_layout = False
    if product == "VSV":
        if basis_id is not None and endpoint_vector_args:
            raise ValueError("pass legacy basis_id/modes or endpoint-specific vector bases")
        if endpoint_vector_args:
            vector_src_id, vector_snk_id = basis_src_id, basis_snk_id
            modes_source = list(modes if modes_src is None else modes_src)
            modes_sink = list(modes if modes_snk is None else modes_snk)
            if vector_src_id is None or vector_snk_id is None:
                raise ValueError("VSV requires actual source and sink vector basis IDs")
            if modes_src is None and not modes or modes_snk is None and not modes:
                raise ValueError("VSV requires explicit source and sink mode coordinates")
            legacy_vector_layout = (vector_src_id == vector_snk_id and
                                    modes_source == modes_sink)
        else:
            vector_src_id = vector_snk_id = basis_id
            modes_source = modes_sink = list(modes)
            legacy_vector_layout = True
        if vector_src_id is None or not modes_source or not modes_sink:
            raise ValueError("actual vector bases and explicit mode coordinates required")
    elif product == "PSV":
        if basis_snk_id is not None or modes_snk is not None:
            raise ValueError("PSV has a source vector basis and a sink point basis only")
        if basis_id is not None and (basis_src_id is not None or modes_src is not None):
            raise ValueError("pass legacy basis_id/modes or endpoint-specific source basis")
        vector_src_id = basis_id if basis_src_id is None else basis_src_id
        modes_source = list(modes if modes_src is None else modes_src)
        if vector_src_id is None or not modes_source:
            raise ValueError("actual source vector basis and explicit mode coordinates required")
        legacy_vector_layout = basis_src_id is None and modes_src is None
    elif product == "VSP":
        if basis_id is not None or basis_src_id is not None or modes_src is not None:
            raise ValueError("VSP has a sink vector basis and a source point basis only")
        vector_snk_id = basis_snk_id
        modes_sink = list(modes if modes_snk is None else modes_snk)
        if vector_snk_id is None or not modes_sink:
            raise ValueError("VSP requires an actual sink vector basis and explicit mode coordinates")
        if modes_src is not None:
            raise ValueError("VSP has no source vector mode axis")
    elif endpoint_vector_args:
        raise ValueError("PSP has no vector basis endpoints")
    if product in ("PSV", "PSP") and (points_snk_id is None or not point_snk_labels):
        raise ValueError("actual sink point realization and ordered point labels required")
    if product in ("PSP", "VSP") and (points_src_id is None or not point_src_labels):
        raise ValueError("actual source point realization and ordered point labels required")
    if product == "VSP" and (points_snk_id is not None or point_snk_labels):
        raise ValueError("VSP has no sink point endpoint")
    if temporal_boundary not in ("periodic", "antiperiodic"):
        raise ValueError("explicit fermion temporal boundary required")
    def axis(name, values):
        return dict(name=name, coordinates=list(values))
    lt = configuration["lattice"][3]
    source_times = times if source_selection is None else normalize_time_selection(source_selection, Lt=lt).times
    sink_times = times if sink_selection is None else normalize_time_selection(sink_selection, Lt=lt).times
    if source_selection is not None or sink_selection is not None:
        source_selection = TimeSelectionRecord(lt, tuple(source_times))
        sink_selection = TimeSelectionRecord(lt, tuple(sink_times))
    axes = [axis("source_time", source_times), axis("sink_time", sink_times),
            axis("sink_spin", range(4)), axis("source_spin", range(4))]
    point_snk = dict(data_id=points_snk_id, indices=list(point_snk_labels))
    point_src = dict(data_id=points_src_id, indices=list(point_src_labels))
    if product == "VSV":
        axes += [axis("sink_mode", modes_sink), axis("source_mode", modes_source)]
        if legacy_vector_layout:
            # Preserve the original default scientific document and identity.
            mode = dict(data_id=vector_src_id, indices=list(modes_source))
            bases = dict(sink_mode=mode, source_mode=mode)
            parents = [vector_src_id, gauge_id]
        else:
            bases = dict(
                sink_mode=dict(data_id=vector_snk_id, indices=list(modes_sink)),
                source_mode=dict(data_id=vector_src_id, indices=list(modes_source)))
            parents = []
            for parent in (vector_snk_id, vector_src_id, gauge_id):
                if parent not in parents:
                    parents.append(parent)
    elif product == "PSV":
        axes += [axis("sink_point", point_snk_labels), axis("sink_color", range(3)),
                 axis("source_mode", modes_source)]
        mode = dict(data_id=vector_src_id, indices=list(modes_source))
        bases = dict(sink_point=point_snk, source_mode=mode)
        parents = [vector_src_id, gauge_id, points_snk_id]
    elif product == "VSP":
        axes += [axis("sink_mode", modes_sink), axis("source_point", point_src_labels),
                 axis("source_color", range(3))]
        bases = dict(
            sink_mode=dict(data_id=vector_snk_id, indices=list(modes_sink)),
            source_point=point_src,
        )
        parents = []
        for parent in (vector_snk_id, gauge_id, points_src_id):
            if parent not in parents:
                parents.append(parent)
    else:
        axes += [axis("sink_point", point_snk_labels), axis("sink_color", range(3)),
                 axis("source_point", point_src_labels), axis("source_color", range(3))]
        bases = dict(sink_point=point_snk, source_point=point_src)
        parents = []
        for parent in (gauge_id, points_snk_id, points_src_id):
            if not any(parent == existing for existing in parents):
                parents.append(parent)
    document = dict(schema=1, kind=product.lower(), configuration=configuration, axes=axes,
        dtype=dtype, bases=bases, parents=parents,
        definition=dict(gauge_id=gauge_id, solver_tolerance=solver_tolerance, dirac=dirac),
        conventions=dict(normalization=normalization, spin_basis=spin_basis,
                         temporal_boundary=temporal_boundary))
    return (document if source_selection is None else bind_time_selections(document,
        {"source_time": source_selection, "sink_time": sink_selection}))


def vsp_dataset_document(psv_document, *, psv_id, gauge_id, basis_id, points_id):
    """Describe VSP derived by gamma5 hermiticity from this Wilson PSV view.

    The input is a fully resolved PSV specification. The derived VSP binds
    the actual requested gauge, basis, point set and PSV parent. Its definition stores the stable
    axis transformation, while ordered coverage stays on the output axes and
    bases so a smaller request can reuse a complete VSP through DatasetView.
    This contract is intentionally limited to the indexed isotropic Wilson
    reference convention; a dataset named ``psv`` alone does not prove the
    gamma5-hermiticity relation.
    """
    try:
        psv_document = DatasetSpec(psv_document).document
    except (TypeError, ValueError) as exc:
        raise ValueError("VSP derivation requires a valid resolved PSV dataset") from exc
    if psv_document.get("kind") != "psv":
        raise ValueError("VSP derivation requires a PSV dataset definition")
    if any(not isinstance(value, str) or not value
           for value in (gauge_id, basis_id, points_id)):
        raise ValueError("VSP derivation requires actual gauge, basis and point IDs")
    if not (isinstance(psv_id, str) and psv_id or
            isinstance(psv_id, dict) and set(psv_id) == {"request"}
            and isinstance(psv_id["request"], str) and psv_id["request"]):
        raise ValueError("VSP derivation requires a PSV dataset ID or request reference")

    expected_axes = (
        "source_time", "sink_time", "sink_spin", "source_spin",
        "sink_point", "sink_color", "source_mode",
    )
    axes = psv_document.get("axes")
    if (not isinstance(axes, list)
            or any(not isinstance(item, dict) or set(item) != {"name", "coordinates"}
                   for item in axes)
            or tuple(item["name"] for item in axes) != expected_axes):
        raise ValueError("VSP derivation requires the canonical PSV axis order")
    coordinates = {item["name"]: list(item["coordinates"]) for item in axes}
    configuration = psv_document.get("configuration")
    if (not isinstance(configuration, dict)
            or not isinstance(configuration.get("ensemble"), str) or not configuration["ensemble"]
            or not isinstance(configuration.get("configuration"), str) or not configuration["configuration"]
            or len(configuration.get("lattice", ())) != 4
            or any(type(length) is not int or length < 1 for length in configuration["lattice"])):
        raise ValueError("VSP derivation requires a known Wilson configuration")
    lt = configuration["lattice"][3]
    if type(lt) is not int or lt < 1 or psv_document.get("dtype") != "complex128":
        raise ValueError("VSP derivation requires complex128 Wilson coverage")
    source_selection = dataset_time_selection(psv_document, "source_time")
    sink_selection = dataset_time_selection(psv_document, "sink_time")
    if (source_selection is None) != (sink_selection is None):
        raise ValueError("VSP derivation requires both endpoint mappings or physical full-time data")
    if source_selection is not None and psv_document["conventions"]["time_coordinates"]["layout"] != "absolute":
        raise ValueError("VSP derivation requires absolute source/sink slots")
    source_times = list(range(lt if source_selection is None else source_selection.Nt))
    sink_times = list(range(lt if sink_selection is None else sink_selection.Nt))
    if (any(type(time) is not int for time in coordinates["source_time"] + coordinates["sink_time"])
            or coordinates["source_time"] != source_times or coordinates["sink_time"] != sink_times):
        raise ValueError("VSP derivation requires complete absolute source/sink times")
    if (coordinates["sink_spin"] != list(range(4))
            or coordinates["source_spin"] != list(range(4))
            or coordinates["sink_color"] != list(range(3))
            or any(type(value) is not int for name in ("sink_spin", "source_spin", "sink_color")
                   for value in coordinates[name])):
        raise ValueError("VSP derivation requires canonical DeGrand-Rossi spin/color axes")

    bases = psv_document.get("bases")
    if not isinstance(bases, dict) or set(bases) != {"sink_point", "source_mode"}:
        raise ValueError("VSP derivation requires PSV point and mode bindings")
    point_binding, mode_binding = bases["sink_point"], bases["source_mode"]
    point_labels, modes = coordinates["sink_point"], coordinates["source_mode"]
    if (not isinstance(point_binding, dict) or not isinstance(mode_binding, dict)
            or point_binding.get("indices") != point_labels
            or mode_binding.get("indices") != modes
            or point_binding.get("data_id") != points_id
            or mode_binding.get("data_id") != basis_id
            or any(type(value) is not int or value < 0 for value in point_labels + modes)
            or not point_labels or not modes):
        raise ValueError("PSV point/mode axes must bind the requested actual inputs")

    definition = psv_document.get("definition")
    conventions = psv_document.get("conventions")
    if not isinstance(definition, dict) or not isinstance(conventions, dict):
        raise ValueError("PSV Dirac definition and conventions are required")
    dirac = definition.get("dirac")
    if (not isinstance(dirac, dict) or set(dirac) != {
            "action", "mass", "r", "lattice_spacing", "clover"}):
        raise ValueError("VSP derivation requires the verified Wilson Dirac definition")
    mass = dirac["mass"]
    tolerance = definition.get("solver_tolerance")
    if (dirac["action"] != "wilson" or isinstance(mass, bool)
            or not isinstance(mass, (int, float)) or not math.isfinite(mass)
            or isinstance(dirac["r"], bool) or not isinstance(dirac["r"], (int, float)) or dirac["r"] != 1
            or isinstance(dirac["lattice_spacing"], bool)
            or not isinstance(dirac["lattice_spacing"], (int, float)) or dirac["lattice_spacing"] != 1
            or isinstance(dirac["clover"], bool)
            or not isinstance(dirac["clover"], (int, float)) or dirac["clover"] != 0
            or isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("VSP derivation supports only isotropic, real-mass, no-clover Wilson")
    boundary = conventions.get("temporal_boundary")
    if (set(conventions) != ({"normalization", "spin_basis", "temporal_boundary"} | ({"time_coordinates"} if source_selection is not None else set()))
            or conventions.get("normalization") != "unscaled"
            or conventions.get("spin_basis") != "degrand_rossi"
            or boundary not in ("periodic", "antiperiodic")):
        raise ValueError("VSP derivation requires unscaled DeGrand-Rossi periodic Wilson PSV")
    psv_gauge = definition.get("gauge_id")
    if (set(definition) != {"gauge_id", "solver_tolerance", "dirac"}
            or psv_gauge != gauge_id):
        raise ValueError("PSV does not bind the requested actual gauge")
    expected_parents = {psv_gauge, mode_binding["data_id"], point_binding["data_id"]}
    if (not isinstance(psv_document.get("parents"), list)
            or set(psv_document["parents"]) != expected_parents):
        raise ValueError("PSV parent IDs do not match its gauge, basis and point bindings")

    input_axis_map = {
        "source_time": "sink_time",
        "sink_time": "source_time",
        "sink_spin": "source_spin",
        "source_spin": "sink_spin",
        "sink_mode": "source_mode",
        "source_point": "sink_point",
        "source_color": "sink_color",
    }
    output_axes = [
        dict(name=name, coordinates=coordinates[input_name])
        for name, input_name in input_axis_map.items()
    ]
    document = dict(
        schema=1,
        kind="vsp",
        configuration=configuration,
        axes=output_axes,
        dtype=psv_document["dtype"],
        bases=dict(
            sink_mode=dict(data_id=basis_id, indices=modes),
            source_point=dict(data_id=points_id, indices=point_labels),
        ),
        definition=dict(
            derivation="gamma5_hermiticity",
            gauge_id=gauge_id,
            solver_tolerance=tolerance,
            dirac=dict(dirac),
            input_product="psv",
            input_axis_map=input_axis_map,
        ),
        conventions=dict(
            normalization="unscaled",
            spin_basis="degrand_rossi",
            temporal_boundary=boundary,
            gamma5="degrand_rossi.gamma(15)",
        ),
        parents=[psv_id],
    )

    if source_selection is not None:
        records = {"source_time": sink_selection, "sink_time": source_selection}
        for axis in document["axes"]:
            if axis["name"] in records:
                axis["coordinates"] = list(records[axis["name"]].times)
        document = bind_time_selections(document, records)
    return document


def elemental_projection_documents(raw, rows, *, raw_id, usedNe=None):
    """Describe raw-path input coverage and its projected spatial derivative.

    This consumes metadata only. The input remains a view of the actual raw
    parent; projected component IDs never attest raw-path coverage. Coefficient
    terms retain their explicit order, including duplicates and zero weights.
    """
    from lqcdweave.contracts.elemental_catalog import catalog_definition
    directory = catalog_definition(rows)
    document = raw.document
    cfg = document["configuration"]
    layouts = (("time", "component", "momentum", "sink_mode", "source_mode"),
               ("component", "momentum", "time", "sink_mode", "source_mode"))
    conventions = document["conventions"]
    gauge_id = document["definition"].get("gauge_id")
    if (document["kind"] != "elemental_vv" or raw.axis_names not in layouts or
            document["dtype"] not in ("complex64", "complex128") or
            any(not cfg[field] for field in ("ensemble", "configuration")) or
            conventions.get("phase") not in ("anchor", "midpoint", "none") or
            conventions.get("normalization") != "unscaled" or
            document["definition"].get("path_boundary") not in ("periodic", "open") or
            not isinstance(gauge_id, str) or not gauge_id or
            set(document["bases"]) != {"sink_mode", "source_mode"}):
        raise ValueError("known raw spatial elemental_vv identity and conventions required")
    times = raw.coordinates("time")
    selection = dataset_time_selection(raw)
    extent = cfg["lattice"][3] if selection is None else selection.Nt
    if any(type(t) is not int for t in times) or sorted(times) != list(range(extent)):
        raise ValueError("complete physical time coverage required")
    for momentum in raw.coordinates("momentum"):
        if len(momentum) != 3 or any(type(value) is not int for value in momentum):
            raise ValueError("explicit integer XYZ momenta required")
    for name in ("sink_mode", "source_mode"):
        if name not in document["bases"] or not document["bases"][name]["data_id"]:
            raise ValueError("actual ordered basis bindings required")
    parsed = spatial_projection_rows(rows)
    required = {path for terms in parsed for path, _ in terms}
    components = []
    for coordinate in raw.coordinates("component"):
        if not isinstance(coordinate, dict) or set(coordinate) != {"path"}:
            raise ValueError("explicit raw path coordinates required")
        path = GaugePath(tuple(coordinate["path"]))
        if any(step >= 6 for step in path.steps) or coordinate != dict(path=list(path.steps)):
            raise ValueError("canonical spatial raw path coordinates required")
        if path in required:
            components.append(coordinate)
            required.remove(path)
    if required:
        raise ValueError("missing raw path coverage for projection")
    for axis in document["axes"]:
        name = axis["name"]
        if name == "component":
            axis["coordinates"] = components
        elif name in ("sink_mode", "source_mode") and usedNe is not None:
            count = integer(usedNe, "usedNe", 1)
            if count > len(axis["coordinates"]):
                raise ValueError("usedNe exceeds raw mode coverage")
            axis["coordinates"] = axis["coordinates"][:count]
            document["bases"][name]["indices"] = axis["coordinates"]
    requested = DatasetSpec(document)
    normalized = [dict(id=position, terms=[dict(path=list(path.steps),
        coefficient=[coefficient.real, coefficient.imag]) for path, coefficient in terms])
        for position, terms in enumerate(parsed)]
    axes = [dict(name="time", coordinates=list(times) if selection is not None else list(range(cfg["lattice"][3]))),
            dict(name="component", coordinates=[dict(projection=i) for i in range(len(parsed))]),
            *[dict(name=name, coordinates=requested.coordinates(name))
              for name in ("momentum", "sink_mode", "source_mode")]]
    projected = dict(schema=1, kind="elemental_projected", configuration=cfg,
        axes=axes, dtype="complex128", bases=document["bases"], parents=[raw_id],
        definition=dict(raw_id=raw_id, method="linear_raw_path_combination", rows=normalized,
                        input_views=dict(raw=requested.document)), conventions=conventions)
    if directory is not None:
        projected["definition"]["catalog"] = directory
    return requested, projected


def estimate_elemental_projection_memory(
    raw_spec: DatasetSpec,
    projected_spec: DatasetSpec,
    *,
    raw_id: str,
    target: Literal["cpu"] = "cpu",
    workspace_reserve_bytes: int = 0,
    host_staging_bytes: int = 0,
) -> tuple[MemoryEstimate, MemoryEstimate]:
    """Estimate one persisted raw-to-projected Elemental row at a time.

    This pure model follows the existing NumPy projection workflow, not the
    fused ``ElementalKernel.calc_path_blocks(..., rows=...)`` kernel. The complete raw
    parent is mapped, while each missing time materializes only the
    selected path stack, the dense coefficient matrix, and one complex128
    projected row. Caller reserves cover opaque Python, NumPy/BLAS, and reader
    staging; the estimate is not an RSS or native-workspace bound.
    """
    if not isinstance(raw_spec, DatasetSpec) or not isinstance(projected_spec, DatasetSpec):
        raise TypeError("raw_spec and projected_spec must be DatasetSpec values")
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError("raw_id must be a nonempty actual dataset ID")
    if target != "cpu":
        raise ValueError("Elemental projection memory model supports target='cpu' only")
    reserves = {}
    for name, value in (
        ("workspace_reserve_bytes", workspace_reserve_bytes),
        ("host_staging_bytes", host_staging_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        reserves[name] = int(value)

    output = projected_spec.document
    definition = output["definition"]
    if (output["kind"] != "elemental_projected" or
            definition.get("raw_id") != raw_id or output["parents"] != [raw_id]):
        raise ValueError("projected spec must bind the requested actual raw parent")
    rows = definition.get("rows", ())
    if definition.get("catalog") is not None:
        from lqcdweave.contracts.elemental_catalog import stored_catalog_table
        rows = stored_catalog_table(definition["catalog"], rows)
    requested, expected = elemental_projection_documents(
        raw_spec, rows, raw_id=raw_id,
        usedNe=len(projected_spec.coordinates("sink_mode")))
    if (DatasetSpec(expected).definition_id != projected_spec.definition_id or
            output["definition"].get("input_views", {}).get("raw") != requested.document):
        raise ValueError("projected spec differs from its actual raw view and recipe")

    dtype_itemsize = {"complex64": 8, "complex128": 16}
    raw_itemsize = dtype_itemsize.get(raw_spec.document["dtype"])
    if raw_itemsize is None:
        raise ValueError("raw Elemental projection supports complex64/complex128 input")
    parent_bytes = math.prod(raw_spec.shape) * raw_itemsize
    input_bytes = (len(requested.coordinates("component")) *
                   len(requested.coordinates("momentum")) *
                   len(requested.coordinates("sink_mode")) *
                   len(requested.coordinates("source_mode")) * raw_itemsize)
    output_shape = projected_spec.shape
    output_bytes = math.prod(output_shape[1:]) * 16
    row_count = len(projected_spec.coordinates("component"))
    path_count = len(requested.coordinates("component"))
    rows = spatial_projection_rows(definition["rows"])
    coefficient_itemsize = 16 if any(
        coefficient.imag != 0 for row in rows for _path, coefficient in row
    ) else 8
    coefficient_bytes = row_count * path_count * coefficient_itemsize
    finite_mask_bytes = output_bytes // 16
    workspace_bytes = (reserves["workspace_reserve_bytes"] +
                       reserves["host_staging_bytes"])
    # These are non-overlapping lifecycle phases. The raw parent remains the
    # registered input throughout; re_combine's contracted result and returned
    # output coexist, while its moveaxis result is a view of the contracted
    # allocation. The workflow releases the input stack after recombine and
    # the projected row after synchronous writer.write.
    phases = (
        MemoryEstimate(input_bytes=parent_bytes,
                       intermediate_bytes=input_bytes + coefficient_bytes + output_bytes,
                       output_bytes=output_bytes,
                       workspace_bytes=workspace_bytes),
        MemoryEstimate(input_bytes=parent_bytes,
                       output_bytes=output_bytes,
                       assembly_extra_bytes=finite_mask_bytes,
                       workspace_bytes=workspace_bytes),
    )
    return max(phases, key=lambda item: item.logical_peak_bytes), MemoryEstimate()


def _perambulator_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _perambulator_array_bytes(shape, itemsize):
    return math.prod(int(dimension) for dimension in shape) * int(itemsize)


def _perambulator_dtype_itemsize(value, name):
    if value in _PERAMBULATOR_COMPLEX_ITEMSIZE:
        return _PERAMBULATOR_COMPLEX_ITEMSIZE[value]
    if value in _PERAMBULATOR_POINT_ITEMSIZE:
        return _PERAMBULATOR_POINT_ITEMSIZE[value]
    raise ValueError(f"unsupported {name} dtype {value!r}")


def estimate_perambulator_memory(
    input_specs: Mapping[str, DatasetSpec],
    configuration: Mapping,
    product: str,
    *,
    gauge_id: str,
    basis_src_id: str | None = None,
    basis_snk_id: str | None = None,
    points_src_id: str | None = None,
    points_snk_id: str | None = None,
    mode_src_count: int = 0,
    mode_snk_count: int = 0,
    point_src_count: int = 0,
    point_snk_count: int = 0,
    same_eigenvector: bool = False,
    dtype: str = "complex128",
    execution_options: Mapping | None = None,
    target: Literal["cpu", "gpu"] = "cpu",
    workspace_reserve_bytes: int = 0,
    host_staging_bytes: int = 0,
    source_selection=None, sink_selection=None,
) -> tuple[MemoryEstimate, MemoryEstimate]:
    """Estimate one perambulator stage's logical host and device peaks.

    The returned pair is ``(host_estimate, device_estimate)`` and consists only
    of the shared C ``MemoryEstimate`` record.  It models the complete parent
    reader mappings, the generator's owned basis/point copies, selected-sink
    output (full time by default), complex128 solver arrays, selected-mode/point temporaries, stream
    scratch/cache, and the synchronous host row and finite mask. It estimates
    one contracted sink per call. Multiple sinks and retained raw fields need
    a task-specific combined estimate; this function does not model them.

    ``workspace_reserve_bytes`` and ``host_staging_bytes`` are caller
    declarations for opaque backend/native workspace and host-side
    reader/adapter/transfer overhead.  They are not measured bounds.  A CPU
    target charges the execution workspace to host; a GPU target charges it to
    both host and device as a conservative workflow-specific model.  The host
    staging declaration is always charged to host.

    The estimates describe a caller-modeled logical peak, not RSS or a complete
    allocator/native-library bound.  Distinct phases are compared by peak;
    their non-overlapping temporary arrays are not summed together.
    """
    if not isinstance(input_specs, Mapping):
        raise TypeError("input_specs must map actual IDs to DatasetSpec values")
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    if target not in ("cpu", "gpu"):
        raise ValueError("target must be 'cpu' or 'gpu'")
    product = str(product).upper()
    if product not in {"VSV", "PSV", "PSP", "VSP"}:
        raise ValueError("product must be VSV, PSV, PSP or VSP")
    for name, value in (("gauge_id", gauge_id), ("basis_src_id", basis_src_id),
                        ("basis_snk_id", basis_snk_id), ("points_src_id", points_src_id),
                        ("points_snk_id", points_snk_id)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} must be a nonempty actual ID or None")
    counts = {
        "mode_src_count": _perambulator_nonnegative_int(mode_src_count, "mode_src_count"),
        "mode_snk_count": _perambulator_nonnegative_int(mode_snk_count, "mode_snk_count"),
        "point_src_count": _perambulator_nonnegative_int(point_src_count, "point_src_count"),
        "point_snk_count": _perambulator_nonnegative_int(point_snk_count, "point_snk_count"),
    }
    workspace_reserve_bytes = _perambulator_nonnegative_int(
        workspace_reserve_bytes, "workspace_reserve_bytes")
    host_staging_bytes = _perambulator_nonnegative_int(
        host_staging_bytes, "host_staging_bytes")
    contract_itemsize = _perambulator_dtype_itemsize(dtype, "output")
    options = {} if execution_options is None else execution_options
    if not isinstance(options, Mapping):
        raise TypeError("execution_options must be a mapping")
    mrhs = options.get("MRHS", False)
    streamed = options.get("stream_eigenvectors", False)
    if any(type(value) is not bool for value in (mrhs, streamed)):
        raise ValueError("MRHS and stream_eigenvectors must be bool")

    def option_integer(name, default, minimum):
        value = options.get(name, default)
        if value is None and name == "rhs_batch_size":
            return None
        if type(value) is not int or value < minimum:
            raise ValueError(f"execution_options.{name} must be an integer >= {minimum}")
        return value

    rhs_batch = option_integer("rhs_batch_size", None, 1)
    sink_ne_block = option_integer("sink_ne_block_size", 32, 1)
    sink_time_block = option_integer("sink_time_block_size", 1, 1)
    sink_cache_limit = option_integer("sink_cache_limit_bytes", 0, 0)
    lattice = configuration.get("lattice")
    if (not isinstance(lattice, (tuple, list)) or len(lattice) != 4 or
            any(type(size) is not int or size < 1 for size in lattice)):
        raise ValueError("configuration.lattice must contain positive (Lx,Ly,Lz,Lt) integers")
    lx, ly, lz, lt = lattice
    source_record = normalize_time_selection(source_selection, Lt=lt)
    sink_record = normalize_time_selection(sink_selection, Lt=lt)
    nsrc, nsnk = source_record.Nt, sink_record.Nt
    if sink_time_block > lt:
        raise ValueError("sink_time_block_size cannot exceed physical Lt")
    if sink_cache_limit and not streamed:
        raise ValueError("sink_cache_limit_bytes requires stream_eigenvectors=True")
    if dtype not in ("complex64", "complex128"):
        raise ValueError("dtype must be complex64 or complex128")
    volume = lx * ly * lz
    if not isinstance(gauge_id, str) or not gauge_id or gauge_id not in input_specs:
        raise ValueError("input_specs must contain the actual gauge dataset")
    for actual_id, spec in input_specs.items():
        if not isinstance(actual_id, str) or not actual_id or not isinstance(spec, DatasetSpec):
            raise TypeError("input_specs must map actual IDs to DatasetSpec values")
    if input_specs[gauge_id].document["kind"] != "gauge":
        raise ValueError("gauge_id must identify a gauge DatasetSpec")
    if input_specs[gauge_id].document["configuration"] != dict(configuration):
        raise ValueError("gauge DatasetSpec configuration differs from configuration")
    validate_gauge_field(input_specs[gauge_id])

    def require_input(actual_id, name, *, point=False):
        if actual_id is None or actual_id not in input_specs:
            raise ValueError(f"{name} requires its actual input DatasetSpec")
        spec = input_specs[actual_id]
        itemsize = _perambulator_dtype_itemsize(
            spec.document["dtype"], name) if point else contract_itemsize
        return spec, itemsize

    source_vectors = product in ("VSV", "PSV")
    sink_vectors = product in ("VSV", "VSP")
    source_points = product in ("PSP", "VSP")
    sink_points = product in ("PSP", "PSV")
    if source_vectors and counts["mode_src_count"] <= 0:
        raise ValueError(f"{product} requires a nonempty source basis")
    if sink_vectors and counts["mode_snk_count"] <= 0:
        raise ValueError(f"{product} requires a nonempty sink basis")
    if source_points and counts["point_src_count"] <= 0:
        raise ValueError(f"{product} requires a nonempty source point set")
    if sink_points and counts["point_snk_count"] <= 0:
        raise ValueError(f"{product} requires a nonempty sink point set")
    if source_vectors != (basis_src_id is not None):
        raise ValueError(f"{product} source basis ID does not match its product")
    if sink_vectors != (basis_snk_id is not None):
        raise ValueError(f"{product} sink basis ID does not match its product")
    if source_points != (points_src_id is not None):
        raise ValueError(f"{product} source point ID does not match its product")
    if sink_points != (points_snk_id is not None):
        raise ValueError(f"{product} sink point ID does not match its product")
    if type(same_eigenvector) is not bool:
        raise TypeError("same_eigenvector must be bool")
    if same_eigenvector and (product != "VSV" or basis_src_id != basis_snk_id or
                             counts["mode_snk_count"] > counts["mode_src_count"]):
        raise ValueError("same_eigenvector requires a shared VSV source-prefix basis")

    src_basis_itemsize = snk_basis_itemsize = None
    if source_vectors:
        src_basis_spec, src_basis_itemsize = require_input(basis_src_id, "source basis")
        validate_elemental_fields(input_specs[gauge_id], src_basis_spec, selection=source_selection)
        if counts["mode_src_count"] > len(src_basis_spec.coordinates("mode")):
            raise ValueError("mode_src_count exceeds registered source basis coverage")
    if sink_vectors:
        snk_basis_spec, snk_basis_itemsize = require_input(basis_snk_id, "sink basis")
        validate_elemental_fields(input_specs[gauge_id], snk_basis_spec, selection=sink_selection)
        if counts["mode_snk_count"] > len(snk_basis_spec.coordinates("mode")):
            raise ValueError("mode_snk_count exceeds registered sink basis coverage")
    src_point_itemsize = snk_point_itemsize = None
    if source_points:
        src_point_spec, src_point_itemsize = require_input(
            points_src_id, "source points", point=True)
        validate_point_field(input_specs[gauge_id], src_point_spec, selection=source_selection)
        if counts["point_src_count"] > len(src_point_spec.coordinates("point")):
            raise ValueError("point_src_count exceeds registered source point coverage")
    if sink_points:
        snk_point_spec, snk_point_itemsize = require_input(
            points_snk_id, "sink points", point=True)
        validate_point_field(input_specs[gauge_id], snk_point_spec, selection=sink_selection)
        if counts["point_snk_count"] > len(snk_point_spec.coordinates("point")):
            raise ValueError("point_snk_count exceeds registered sink point coverage")

    parent_bytes = sum(
        _perambulator_array_bytes(spec.shape,
            _perambulator_dtype_itemsize(spec.document["dtype"], "input"))
        for spec in input_specs.values())
    gauge_device_bytes = _perambulator_array_bytes(
        input_specs[gauge_id].shape,
        _perambulator_dtype_itemsize(input_specs[gauge_id].document["dtype"], "gauge"))
    source_vector_bytes = (nsrc * counts["mode_src_count"] * volume * 3 * contract_itemsize
                           if source_vectors else 0)
    sink_vector_bytes = (nsnk * counts["mode_snk_count"] * volume * 3 * contract_itemsize
                         if sink_vectors else 0)
    source_point_bytes = (nsrc * counts["point_src_count"] * 3 * src_point_itemsize
                          if source_points else 0)
    sink_point_bytes = (nsnk * counts["point_snk_count"] * 3 * snk_point_itemsize
                        if sink_points else 0)

    if product == "VSV":
        output_shape = (nsnk, 4, 4, counts["mode_snk_count"], counts["mode_src_count"])
    elif product == "PSV":
        output_shape = (nsnk, 4, 4, counts["point_snk_count"], 3,
                        counts["mode_src_count"])
    elif product == "PSP":
        output_shape = (nsnk, 4, 4, counts["point_snk_count"], 3,
                        counts["point_src_count"], 3)
    else:
        output_shape = (nsnk, 4, 4, counts["mode_snk_count"],
                        counts["point_src_count"], 3)
    output_bytes = _perambulator_array_bytes(output_shape, contract_itemsize)
    # One producer invocation returns the selected sink-time stack for one source
    # time; a caller script saves that entire stack as one source-time row.
    output_row_bytes = output_bytes
    output_row_elements = output_row_bytes // contract_itemsize

    def batch_size(maximum):
        if not mrhs:
            return 1
        return min(maximum, maximum if rhs_batch is None else rhs_batch)

    capacity = max(batch_size(4) if counts["mode_src_count"] else 1,
                   batch_size(12) if counts["point_src_count"] else 1)
    rhs_solution_bytes = 2 * _perambulator_array_bytes(
        (capacity, lt, lz, ly, lx, 4, 3), 16)
    eigen_solution_bytes = (_perambulator_array_bytes((4, lt, lz, ly, lx, 4, 3), 16)
                            if source_vectors and batch_size(4) < 4 else 0)
    point_index_bytes = 0  # calc selects one physical time by scalar indexing.
    source_scratch_bytes = (volume * 3 * contract_itemsize
                            if streamed and source_vectors else 0)
    sink_block_modes = min(counts["mode_snk_count"], sink_ne_block)
    sink_scratch_bytes = (sink_block_modes * volume * 3 * contract_itemsize
                          if streamed and sink_vectors else 0)
    scratch_bytes = source_scratch_bytes + sink_scratch_bytes
    sink_cache_bytes = (min(sink_cache_limit,
        nsnk * counts["mode_snk_count"] * volume * 3 * contract_itemsize)
        if streamed and sink_vectors else 0)
    base_intermediate = rhs_solution_bytes + eigen_solution_bytes + point_index_bytes + scratch_bytes
    # Actual identical basis objects may share one staged array. The estimate
    # only uses this optimization when explicitly declared, with equal counts
    # and equal time maps; equal data IDs alone do not establish object identity.
    vector_snapshot_bytes = 0 if streamed else source_vector_bytes + sink_vector_bytes
    if (not streamed and same_eigenvector and source_record == sink_record
            and counts["mode_src_count"] == counts["mode_snk_count"]):
        vector_snapshot_bytes -= sink_vector_bytes
    vector_load_extra_bytes = (max(source_vector_bytes, sink_vector_bytes) // contract_itemsize
                               if not streamed else 0)  # finite-value mask

    point_snapshot_bytes = source_point_bytes + sink_point_bytes
    # The gauge payload has a known declared shape. Conservatively count it as
    # resident in device memory for a GPU target, separately from opaque backend
    # workspace, which remains covered only by the caller's reserve.
    device_snapshot_bytes = (vector_snapshot_bytes + point_snapshot_bytes +
                             gauge_device_bytes if target == "gpu" else 0)
    point_validation_bytes = max(
        (counts["point_src_count"] * 3 * src_point_itemsize if source_points else 0),
        (counts["point_snk_count"] * 3 * snk_point_itemsize if sink_points else 0),
    )
    point_host_load_peak = max(
        (2 * source_point_bytes if source_points else 0),
        ((source_point_bytes if source_points else 0) + 2 * sink_point_bytes
         if sink_points else 0),
    )
    if target == "gpu":
        # Host stack plus the copied host point table coexist inside
        # DatasetPointHandle.load; the device copy is accounted separately.
        host_point_load_peak = max(
            (2 * source_point_bytes if source_points else 0),
            (2 * sink_point_bytes if sink_points else 0),
        )
        vector_transfer_peak = max(
            (volume * 3 * contract_itemsize if source_vectors else 0),
            (volume * 3 * contract_itemsize if sink_vectors else 0),
        ) if not streamed else 0
    else:
        host_point_load_peak = point_host_load_peak
        vector_transfer_peak = 0

    # New calc projects one physical time at a time: sparse time ordering
    # does not materialize an Lt-sized solution gather. The previous projected
    # tile can remain live while the next is produced by the iterator.
    projection_rhs = 4 if source_vectors else batch_size(12)
    if sink_vectors:
        tile = 4 * projection_rhs * sink_block_modes * 16
        mode_block = sink_block_modes * volume * 3 * contract_itemsize
        projection_temporary = (max(2*tile, tile + mode_block) if streamed else
                                max(2*tile + mode_block, tile + 2*mode_block))
    else:
        projection_temporary = 2 * projection_rhs * counts["point_snk_count"] * 4 * 3 * 16

    target_workspace = workspace_reserve_bytes
    # The opaque backend reserve is charged to both domains on GPU because
    # backend/native host work may accompany device allocations. CPU work uses
    # the host domain only. Host staging is always additional.
    host_workspace = host_staging_bytes + workspace_reserve_bytes
    if target == "cpu":
        target_workspace = 0

    def make_estimate(*, domain, snapshots=0, intermediate=0, output=0,
                      assembly=0, workspace=0):
        return MemoryEstimate(
            input_bytes=parent_bytes if domain == "host" else 0,
            snapshot_bytes=snapshots,
            intermediate_bytes=intermediate,
            output_bytes=output,
            assembly_extra_bytes=assembly,
            workspace_bytes=workspace,
        )

    def peak(estimates):
        return max(estimates, key=lambda estimate: estimate.logical_peak_bytes)

    host_phases = [
        # Point-value preflight is separate from production and uses borrowed
        # mapped rows plus one explicit unique-coordinate result.
        make_estimate(domain="host", intermediate=point_validation_bytes,
                      workspace=host_staging_bytes),
    ]
    if target == "gpu":
        host_phases.extend((
            make_estimate(domain="host", assembly=host_point_load_peak,
                          workspace=host_staging_bytes),
            make_estimate(domain="host", assembly=vector_transfer_peak,
                          workspace=host_workspace),
            make_estimate(domain="host", workspace=host_workspace),
            make_estimate(domain="host", assembly=(output_row_bytes +
                          output_row_elements),
                          workspace=host_workspace),
        ))
        device_phases = [
            make_estimate(domain="device", output=output_bytes,
                          intermediate=base_intermediate,
                          workspace=target_workspace),
            make_estimate(domain="device", snapshots=device_snapshot_bytes,
                          intermediate=base_intermediate + vector_load_extra_bytes +
                          vector_transfer_peak,
                          output=output_bytes, workspace=target_workspace),
            make_estimate(domain="device", snapshots=device_snapshot_bytes,
                          intermediate=base_intermediate + projection_temporary + sink_cache_bytes,
                          output=output_bytes, workspace=target_workspace),
            make_estimate(domain="device", snapshots=gauge_device_bytes,
                          intermediate=base_intermediate - eigen_solution_bytes + sink_cache_bytes,
                          output=output_bytes, workspace=target_workspace),
        ]
        device_estimate = peak(device_phases)
    else:
        # On CPU all generator allocations share the host domain.  Compare
        # load, solve/project, and row-write peaks instead of summing temporaries
        # from phases that cannot overlap.
        host_phases.extend((
            make_estimate(domain="host", snapshots=vector_snapshot_bytes,
                intermediate=base_intermediate + vector_load_extra_bytes +
                point_host_load_peak + point_validation_bytes,
                output=output_bytes, workspace=host_workspace),
            make_estimate(domain="host", snapshots=vector_snapshot_bytes + point_snapshot_bytes,
                intermediate=base_intermediate + projection_temporary + sink_cache_bytes,
                output=output_bytes, workspace=host_workspace),
            make_estimate(domain="host", snapshots=0,
                intermediate=base_intermediate - eigen_solution_bytes + sink_cache_bytes,
                output=output_bytes,
                assembly=output_row_bytes + output_row_elements,
                workspace=host_workspace),
        ))
        device_estimate = MemoryEstimate()
    return peak(host_phases), device_estimate


def _elemental_phase_cache_peak_bytes(volume, phase_key_count, phase_convention):
    phase_array_bytes = volume * 16
    stable_slots = min(64, phase_key_count)
    old_slots = min(64, max(0, phase_key_count - 1))
    stable_bytes = stable_slots * phase_array_bytes
    # momentum_phase allocates xp.indices before testing the "none" shortcut,
    # so even that shortcut overlaps three int64 coordinate arrays with its
    # complex128 ones result. Other conventions additionally overlap float64
    # coordinate/phase arithmetic and the complex exponential input/output.
    build_bytes = volume * (40 if phase_convention == "none" else 64)
    return max(stable_bytes, old_slots * phase_array_bytes + build_bytes)


def estimate_elemental_memory(
    input_specs: Mapping[str, DatasetSpec],
    output_spec: DatasetSpec,
    *,
    gauge_id: str,
    basis_id: str,
    ne_block_size: int = 32,
    target: Literal["cpu"] = "cpu",
    workspace_reserve_bytes: int = 0,
    host_staging_bytes: int = 0,
) -> tuple[MemoryEstimate, MemoryEstimate]:
    """Estimate one raw spatial V/V Elemental producer's logical host peak.

    This model is restricted to the existing NumPy spatial ``elemental_vv``
    workflow. It counts the complete registered input payloads, the full gauge
    stack overlapping ElementalWorkspace's owned complex128 links, one complete
    complex128 output row, the bounded phase LRU, and mode-tile/link arrays
    whose shapes follow directly from ElementalGenerator. Known phases are
    compared by peak rather than summed over the run. Native BLAS scratch,
    filesystem page residency, and any other opaque workspace are not inferred;
    caller reserves remain explicit and this is not an RSS bound.

    The phase-construction model follows NumPy's 64-bit ``indices`` on the
    supported CPU runtime: three int64 coordinate arrays plus overlapping
    float64 phase arithmetic and complex128 exponential arrays. The path-tile
    allowance follows the live ``right``, ``shifted``, ``transported``, left,
    conjugate, phase-weighted-left, and prior/next pair-result arrays in
    ``calc_path_blocks``/``bilinear``. Matrix-kernel internals stay in the
    caller-declared workspace reserve.
    """
    if not isinstance(input_specs, Mapping):
        raise TypeError("input_specs must map actual IDs to DatasetSpec values")
    if not isinstance(output_spec, DatasetSpec):
        raise TypeError("output_spec must be a DatasetSpec")
    if target != "cpu":
        raise ValueError("raw spatial V/V Elemental memory model supports target='cpu' only")
    if (not isinstance(gauge_id, str) or not gauge_id or
            not isinstance(basis_id, str) or not basis_id or gauge_id == basis_id):
        raise ValueError("gauge_id and basis_id must be distinct nonempty actual IDs")
    if set(input_specs) != {gauge_id, basis_id}:
        raise ValueError("input_specs must contain exactly the actual gauge and basis IDs")
    gauge = input_specs[gauge_id]
    basis = input_specs[basis_id]
    if not isinstance(gauge, DatasetSpec) or not isinstance(basis, DatasetSpec):
        raise TypeError("input_specs values must be DatasetSpec instances")
    selection = dataset_time_selection(output_spec)
    times = validate_elemental_fields(gauge, basis, selection=selection)
    if (output_spec.document["kind"] != "elemental_vv" or
            output_spec.axis_names != (
                "time", "component", "momentum", "sink_mode", "source_mode") or
            output_spec.document["dtype"] != "complex128"):
        raise ValueError("output_spec must be a complex128 raw spatial elemental_vv")
    document = output_spec.document
    if (document["configuration"] != gauge.document["configuration"] or
            document["parents"] != [gauge_id, basis_id] or
            document["definition"].get("gauge_id") != gauge_id or
            document["bases"]["sink_mode"]["data_id"] != basis_id or
            document["bases"]["source_mode"]["data_id"] != basis_id):
        raise ValueError("output_spec does not bind the supplied actual gauge and basis IDs")
    if dataset_physical_times(output_spec) != times:
        raise ValueError("output_spec must cover every physical time in canonical order")
    modes = tuple(output_spec.coordinates("sink_mode"))
    if (not modes or tuple(output_spec.coordinates("source_mode")) != modes or
            tuple(basis.coordinates("mode")[:len(modes)]) != modes or
            document["bases"]["sink_mode"]["indices"] != list(modes) or
            document["bases"]["source_mode"]["indices"] != list(modes)):
        raise ValueError("output modes must be the same explicit stored basis prefix at both ends")
    components = tuple(output_spec.coordinates("component"))
    momenta = tuple(output_spec.coordinates("momentum"))
    if not components or not momenta:
        raise ValueError("raw spatial Elemental requires nonempty component and momentum axes")
    paths = []
    for component in components:
        if not isinstance(component, Mapping) or set(component) != {"path"}:
            raise ValueError("raw elemental components must be explicit path records")
        path = GaugePath.coerce(component["path"])
        if any(step >= 6 for step in path.steps):
            raise ValueError("raw spatial Elemental budget model excludes temporal paths")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("raw elemental path coordinates must be unique")
    normalized_momenta = []
    for momentum in momenta:
        if not isinstance(momentum, (tuple, list)) or len(momentum) != 3:
            raise ValueError("momentum coordinates must be XYZ triples")
        normalized_momenta.append(tuple(integer(value, "momentum") for value in momentum))
    phase_convention = document["conventions"].get("phase")
    if phase_convention not in ("anchor", "midpoint", "none"):
        raise ValueError("output phase convention is unsupported by raw Elemental")
    workspace_reserve_bytes = _perambulator_nonnegative_int(
        workspace_reserve_bytes, "workspace_reserve_bytes")
    host_staging_bytes = _perambulator_nonnegative_int(
        host_staging_bytes, "host_staging_bytes")
    ne_block_size = _perambulator_nonnegative_int(ne_block_size, "ne_block_size")
    if ne_block_size <= 0:
        raise ValueError("ne_block_size must be a positive integer")

    def payload_bytes(spec, name):
        return _perambulator_array_bytes(
            spec.shape, _perambulator_dtype_itemsize(spec.document["dtype"], name))

    cfg = gauge.document["configuration"]
    lx, ly, lz, lt = cfg["lattice"]
    volume = lx * ly * lz
    parent_bytes = payload_bytes(gauge, "gauge") + payload_bytes(basis, "basis")
    gauge_stack_bytes = payload_bytes(gauge, "gauge")
    # ElementalWorkspace.load transposes the complete stack into owned C-order
    # complex128 links before its local gauge stack goes out of scope.
    owned_links_bytes = 4 * lt * lz * ly * lx * 3 * 3 * 16
    ne = len(modes)
    tile_modes = min(ne, ne_block_size)
    vector_tile_bytes = tile_modes * volume * 3 * 16
    pair_result_bytes = tile_modes * tile_modes * 16
    link_matrix_bytes = volume * 3 * 3 * 16
    output_row_bytes = _perambulator_array_bytes(output_spec.shape[1:], 16)
    output_mask_bytes = output_row_bytes // 16

    # ElementalWorkspace.phase retains at most 64 distinct (momentum,
    # displacement, convention) arrays. A miss constructs one more phase
    # while older entries are still live; for non-'none' phases, indices,
    # arithmetic and exp temporaries peak at 64 bytes per lattice site.
    phase_keys = {
        (momentum, path.displacement[:3], phase_convention)
        for momentum in normalized_momenta for path in paths
    }
    phase_bytes = volume * 16
    phase_slots = min(64, len(phase_keys))
    stable_phase_bytes = phase_slots * phase_bytes
    phase_cache_peak = _elemental_phase_cache_peak_bytes(
        volume, len(phase_keys), phase_convention)

    # Six mode-tile arrays are simultaneously reachable in the raw bilinear:
    # right, roll( right ), transported right, left, conj(left), and its
    # phase-weighted product. Pair matrices may overlap while assigning the
    # next result. On a path change, the previous three right-tile arrays live
    # while path_transport holds old/new transport, link and product arrays.
    tile_kernel_peak = 6 * vector_tile_bytes + 2 * pair_result_bytes + link_matrix_bytes
    path_change_peak = 3 * vector_tile_bytes + pair_result_bytes + 4 * link_matrix_bytes
    phase_miss_peak = (3 * vector_tile_bytes + pair_result_bytes + link_matrix_bytes +
                       phase_cache_peak)
    calc_intermediate_peak = max(
        tile_kernel_peak + stable_phase_bytes,
        path_change_peak + stable_phase_bytes,
        phase_miss_peak,
    )
    host_workspace = workspace_reserve_bytes + host_staging_bytes

    def estimate(*, snapshots=0, intermediate=0, output=0, assembly=0):
        return MemoryEstimate(
            input_bytes=parent_bytes,
            snapshot_bytes=snapshots,
            intermediate_bytes=intermediate,
            output_bytes=output,
            assembly_extra_bytes=assembly,
            workspace_bytes=host_workspace,
        )

    host_phases = (
        # Full-time reader materialization overlaps the workspace's owned
        # direction-major gauge links during ElementalWorkspace.load.
        estimate(snapshots=owned_links_bytes, intermediate=gauge_stack_bytes),
        # The named v2v workspace buffer is precisely the output row. Per-phase
        # peaks include resident links and phase LRU but do not sum nonoverlap.
        estimate(snapshots=owned_links_bytes,
                 intermediate=calc_intermediate_peak,
                 output=output_row_bytes),
        estimate(snapshots=owned_links_bytes,
                 intermediate=stable_phase_bytes,
                 output=output_row_bytes,
                 assembly=output_mask_bytes),
    )
    host_estimate = max(host_phases, key=lambda item: item.logical_peak_bytes)
    return host_estimate, MemoryEstimate()


def estimate_directed_current_memory(
    input_specs: Mapping[str, DatasetSpec],
    output_spec: DatasetSpec,
    *,
    gauge_id: str,
    basis_id: str,
    target: Literal["cpu"] = "cpu",
    workspace_reserve_bytes: int = 0,
    host_staging_bytes: int = 0,
) -> tuple[MemoryEstimate, MemoryEstimate]:
    """Estimate the indexed full-time eight-link V/V current CPU lifecycle.

    The model follows the existing NumPy producer, not the separate legacy
    all-time output API. It counts both registered parent payloads, the full
    gauge read stack overlapping the owned complex128 link workspace, the
    selected one-time vector reserve and path-transport/bilinear temporaries,
    the bounded midpoint-phase LRU and a single complex128 output row. Assembly
    includes the finite-value mask. Native
    NumPy/BLAS scratch, mmap residency, allocator fragmentation and other opaque
    state remain caller-declared reserves; this is a logical model, not RSS.
    """
    if not isinstance(input_specs, Mapping):
        raise TypeError("input_specs must map actual IDs to DatasetSpec values")
    if not isinstance(output_spec, DatasetSpec):
        raise TypeError("output_spec must be a DatasetSpec")
    if target != "cpu":
        raise ValueError("directed-current memory model supports target='cpu' only")
    if (not isinstance(gauge_id, str) or not gauge_id or
            not isinstance(basis_id, str) or not basis_id or gauge_id == basis_id):
        raise ValueError("gauge_id and basis_id must be distinct nonempty actual IDs")
    if set(input_specs) != {gauge_id, basis_id}:
        raise ValueError("input_specs must contain exactly the actual gauge and basis IDs")
    gauge, basis = input_specs[gauge_id], input_specs[basis_id]
    if not isinstance(gauge, DatasetSpec) or not isinstance(basis, DatasetSpec):
        raise TypeError("input_specs values must be DatasetSpec instances")
    selection = dataset_time_selection(output_spec)
    times = validate_elemental_fields(gauge, basis, selection=selection)

    expected_axes = (
        "time", "component", "momentum", "sink_mode", "source_mode")
    if (output_spec.document["kind"] != "elemental_vv" or
            output_spec.axis_names != expected_axes or
            output_spec.document["dtype"] != "complex128"):
        raise ValueError("output_spec must be a complex128 eight-link elemental_vv")
    document = output_spec.document
    if (document["configuration"] != gauge.document["configuration"] or
            document["parents"] != [gauge_id, basis_id] or
            document["definition"].get("gauge_id") != gauge_id or
            document["definition"].get("directed_basis") !=
                "lattice.current.directed-one-link-basis/v1"):
        raise ValueError("output_spec does not bind the supplied directed-current inputs")
    if dataset_physical_times(output_spec) != times:
        raise ValueError("directed current must cover every physical time in order")
    components = output_spec.coordinates("component")
    if components != [dict(path=[index]) for index in range(8)]:
        raise ValueError("directed current must retain all eight ordered link directions")
    if (document["conventions"].get("normalization") != "unscaled" or
            document["conventions"].get("phase") != "midpoint" or
            document["definition"].get("path_boundary") not in ("periodic", "open")):
        raise ValueError("directed current output has unsupported boundary or conventions")
    expected_views = {"gauge": gauge.document}
    basis_view = basis.document
    modes = tuple(output_spec.coordinates("sink_mode"))
    mode_axis = next((axis for axis in basis_view["axes"] if axis["name"] == "mode"), None)
    if mode_axis is None:
        raise ValueError("actual basis metadata has no mode axis")
    if (not modes or tuple(output_spec.coordinates("source_mode")) != modes or
            tuple(basis.coordinates("mode")[:len(modes)]) != modes):
        raise ValueError("directed current modes must be an explicit stored ordered prefix")
    mode_axis["coordinates"] = list(modes)
    if "mode" in basis_view["bases"]:
        basis_view["bases"]["mode"]["indices"] = list(modes)
    expected_views["basis"] = basis_view
    if document["definition"].get("input_views") != expected_views:
        raise ValueError("output_spec input views differ from the supplied actual parent metadata")
    for role in ("sink_mode", "source_mode"):
        binding = document["bases"].get(role)
        if (not isinstance(binding, Mapping) or
                binding.get("data_id") != basis_id or
                binding.get("indices") != list(modes)):
            raise ValueError(f"directed current {role} must bind the actual basis prefix")

    momenta = tuple(output_spec.coordinates("momentum"))
    if not momenta:
        raise ValueError("directed current requires a nonempty momentum axis")
    normalized_momenta = []
    for momentum in momenta:
        if not isinstance(momentum, (tuple, list)) or len(momentum) != 3:
            raise ValueError("directed-current momentum coordinates must be XYZ triples")
        normalized_momenta.append(tuple(integer(value, "momentum") for value in momentum))
    if len(set(normalized_momenta)) != len(normalized_momenta):
        raise ValueError("directed-current momentum coordinates must be unique")

    workspace_reserve_bytes = _perambulator_nonnegative_int(
        workspace_reserve_bytes, "workspace_reserve_bytes")
    host_staging_bytes = _perambulator_nonnegative_int(
        host_staging_bytes, "host_staging_bytes")

    def payload_bytes(spec, name):
        return _perambulator_array_bytes(
            spec.shape, _perambulator_dtype_itemsize(spec.document["dtype"], name))

    cfg = gauge.document["configuration"]
    lx, ly, lz, lt = cfg["lattice"]
    volume = lx * ly * lz
    parent_bytes = payload_bytes(gauge, "gauge") + payload_bytes(basis, "basis")
    gauge_stack_bytes = payload_bytes(gauge, "gauge")
    owned_links_bytes = 4 * lt * volume * 3 * 3 * 16
    vector_bytes = len(modes) * volume * 3 * 16
    link_matrix_bytes = volume * 3 * 3 * 16
    pair_bytes = len(modes) * len(modes) * 16
    output_row_bytes = _perambulator_array_bytes(output_spec.shape[1:], 16)
    finite_mask_bytes = output_row_bytes // 16
    spatial_and_temporal_displacements = {
        tuple(vector[:3])
        for _index, _name, vector, _gauge_axis, _dagger
        in DirectedCurrentBasis.DIRECTIONS
    }
    phase_key_count = len(normalized_momenta) * len(spatial_and_temporal_displacements)
    stable_phase_bytes = min(64, phase_key_count) * volume * 16
    phase_cache_peak_bytes = _elemental_phase_cache_peak_bytes(
        volume, phase_key_count, "midpoint")

    # Use all selected modes as a conservative bound for the path kernel's
    # default mode tiles. Transport construction includes a backward-link
    # conjugate and the previous path transport; there is no transport cache.
    calculation_intermediate = max(
        stable_phase_bytes + 6 * vector_bytes + 2 * pair_bytes + link_matrix_bytes,
        stable_phase_bytes + 3 * vector_bytes + pair_bytes + 5 * link_matrix_bytes,
        phase_cache_peak_bytes + 3 * vector_bytes + pair_bytes + link_matrix_bytes,
    )
    host_workspace = workspace_reserve_bytes + host_staging_bytes

    def estimate(*, snapshots=0, intermediate=0, output=0, assembly=0):
        return MemoryEstimate(
            input_bytes=parent_bytes,
            snapshot_bytes=snapshots,
            intermediate_bytes=intermediate,
            output_bytes=output,
            assembly_extra_bytes=assembly,
            workspace_bytes=host_workspace,
        )

    host_phases = (
        # _TimeField materializes the complete source gauge stack before the
        # workspace makes its owned direction-major complex128 link copy.
        estimate(snapshots=owned_links_bytes, intermediate=gauge_stack_bytes),
        # The producer keeps its loaded links and one selected time vector;
        # all row-local transport, bilinear and phase miss buffers are
        # included in the intermediate peak.
        estimate(snapshots=owned_links_bytes + vector_bytes,
                 intermediate=calculation_intermediate,
                 output=output_row_bytes),
        # Synchronous row writing happens after calculation
        # temporaries have been released; the row is the only live output.
        estimate(snapshots=owned_links_bytes,
                 intermediate=stable_phase_bytes,
                 output=output_row_bytes,
                 assembly=finite_mask_bytes),
    )
    return max(host_phases, key=lambda item: item.logical_peak_bytes), MemoryEstimate()


def _point_path_transport_build_bytes(paths, link_matrix_bytes):
    """Conservative live transport arrays for one path construction phase."""
    max_steps = max(len(path.steps) for path in paths)
    has_backward = any(
        GaugePath.SIGNS[step] < 0 for path in paths for step in path.steps)
    # path_transport starts with identity (one matrix), then roll/conjugate and
    # multiply temporaries add at most three more matrices; dagger links need
    # one additional complex conjugate copy. Empty paths retain the identity.
    return max(1, (5 if has_backward else 4) if max_steps else 1) * link_matrix_bytes


def estimate_point_elemental_memory(
    input_specs: Mapping[str, DatasetSpec],
    output_specs: Mapping[str, DatasetSpec],
    *,
    ne_block_size: int = 32,
    target: Literal["cpu"] = "cpu",
    workspace_reserve_bytes: int = 0,
    host_staging_bytes: int = 0,
) -> dict[str, tuple[MemoryEstimate, MemoryEstimate]]:
    """Estimate registered V/P, P/V and rectangular P/P point products.

    Inputs are keyed by deduplicated actual dataset ID. Outputs are already
    resolved product specifications. This pure model follows the current
    NumPy reader and ElementalGenerator lifetimes: full gauge conversion,
    ordered point-table materialization, per-time basis row reads, phase and
    transport arrays, output rows, and P/P sparse blocks retained until dense
    row assembly. It is a logical allocation estimate, not an RSS or native
    workspace bound; the caller must reserve opaque Python, NumPy and BLAS
    staging explicitly.
    """
    if not isinstance(input_specs, Mapping) or not isinstance(output_specs, Mapping):
        raise TypeError("input_specs and output_specs must be mappings")
    if not output_specs or set(output_specs) - {"v2p", "p2v", "p2p"}:
        raise ValueError("output_specs must contain selected v2p, p2v or p2p products")
    if target != "cpu":
        raise ValueError("point Elemental memory model supports target='cpu' only")
    workspace_reserve_bytes = _perambulator_nonnegative_int(
        workspace_reserve_bytes, "workspace_reserve_bytes")
    host_staging_bytes = _perambulator_nonnegative_int(
        host_staging_bytes, "host_staging_bytes")
    ne_block_size = _perambulator_nonnegative_int(ne_block_size, "ne_block_size")
    if ne_block_size <= 0:
        raise ValueError("ne_block_size must be a positive integer")

    if any(not isinstance(identity, str) or not identity or
           not isinstance(spec, DatasetSpec)
           for identity, spec in input_specs.items()):
        raise TypeError("input_specs must map nonempty actual IDs to DatasetSpec values")
    if any(not isinstance(spec, DatasetSpec) for spec in output_specs.values()):
        raise TypeError("output_specs values must be DatasetSpec instances")

    roles = {}
    gauge_id = None
    required_inputs = set()
    point_left_by_id = {}
    point_right_by_id = {}
    for product, output in output_specs.items():
        document = output.document
        expected_kind = {
            "v2p": "elemental_v2p",
            "p2v": "elemental_p2v",
            "p2p": "elemental_p2p_row_sparse",
        }[product]
        expected_axes = {
            "v2p": ("time", "component", "momentum", "mode_left", "point_right", "color_right"),
            "p2v": ("time", "component", "momentum", "point_left", "color_left", "mode_right"),
            "p2p": ("time", "component", "momentum", "point_left", "color_left", "color_right"),
        }[product]
        if document["kind"] != expected_kind or output.axis_names != expected_axes or document["dtype"] != "complex128":
            raise ValueError(f"{product} output spec has an unsupported kind, axes or dtype")
        if product == "p2p" and document["conventions"].get("point_pair_layout") != "left-point-row-slots-v1":
            raise ValueError("P/P output must use the row-slot sparse point layout")
        definition = document["definition"]
        bound_gauge = definition.get("gauge_id")
        if not isinstance(bound_gauge, str) or not bound_gauge:
            raise ValueError(f"{product} output must bind its actual gauge ID")
        if gauge_id is None:
            gauge_id = bound_gauge
        elif gauge_id != bound_gauge:
            raise ValueError("all point Elemental outputs must bind one actual gauge ID")
        required_inputs.add(bound_gauge)
        if bound_gauge not in input_specs:
            raise ValueError(f"missing actual gauge spec {bound_gauge!r}")
        gauge = input_specs[bound_gauge]
        times = validate_gauge_field(gauge)
        selection = dataset_time_selection(output)
        if selection is not None:
            times = selection.times
        if document["configuration"] != gauge.document["configuration"] or dataset_physical_times(output) != times:
            raise ValueError(f"{product} output does not cover the gauge's complete physical time")
        phase = document["conventions"].get("phase")
        if phase not in ("anchor", "midpoint", "none"):
            raise ValueError(f"{product} output has an unsupported phase convention")
        current_schema = definition.get("directed_current_schema")
        temporal_path_schema = definition.get("temporal_path_schema")
        if current_schema is not None and temporal_path_schema is not None:
            raise ValueError("point Elemental output cannot combine current and temporal-path schemas")
        if current_schema is not None and current_schema != CURRENT_DIRECTED_POINT_SCHEMA:
            raise ValueError("point Elemental budget model received an unsupported current schema")
        if (temporal_path_schema is not None and
                temporal_path_schema != TEMPORAL_POINT_PATH_SCHEMA):
            raise ValueError("point Elemental budget model received an unsupported temporal-path schema")
        if temporal_path_schema is not None and product not in {"v2p", "p2v", "p2p"}:
            raise ValueError("temporal point-path budget supports V/P, P/V and P/P")
        path_boundary = definition.get("path_boundary", "periodic")
        if (path_boundary not in {"periodic", "open"}
                or (temporal_path_schema is None and current_schema is None
                    and path_boundary != "periodic")):
            raise ValueError("point Elemental output has an unsupported path boundary")
        paths = []
        for component in output.coordinates("component"):
            if not isinstance(component, Mapping) or set(component) != {"path"}:
                raise ValueError("point Elemental components must be explicit path records")
            path = GaugePath.coerce(component["path"])
            if (current_schema is None and temporal_path_schema is None
                    and any(step >= 6 for step in path.steps)):
                raise ValueError("ordinary point Elemental budget excludes temporal paths")
            if current_schema is not None and (len(path.steps) != 1 or path.steps[0] not in (6, 7)):
                raise ValueError("point-current budget requires the existing one-link 6/7 paths")
            if temporal_path_schema is not None and any(
                    step < 0 or step >= 8 for step in path.steps):
                raise ValueError("temporal point-path budget requires ordered directions 0..7")
            paths.append(path)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("point Elemental components must contain distinct paths")
        if temporal_path_schema is not None and not any(
                step >= 6 for path in paths for step in path.steps):
            raise ValueError("temporal point-path budget requires a temporal path component")
        momenta = tuple(tuple(integer(value, "momentum") for value in momentum)
                        for momentum in output.coordinates("momentum"))
        if not momenta or any(not isinstance(value, (list, tuple)) or len(value) != 3
                              for value in momenta):
            raise ValueError("point Elemental momenta must be nonempty XYZ triples")
        phase_convention = phase

        if product in {"v2p", "p2v"}:
            mode_axis = "mode_left" if product == "v2p" else "mode_right"
            basis_id = document["bases"][mode_axis]["data_id"]
            if not isinstance(basis_id, str) or not basis_id:
                raise ValueError(f"{product} output must bind its actual basis ID")
            required_inputs.add(basis_id)
            if basis_id not in input_specs:
                raise ValueError(f"missing actual basis spec {basis_id!r}")
            basis = input_specs[basis_id]
            endpoint_selection = _endpoint_selection(selection, output.coordinates("component"), boundary=path_boundary)
            validate_elemental_fields(gauge, basis, selection=selection if product == "v2p" else endpoint_selection)
            mode_labels = tuple(output.coordinates(mode_axis))
            if (not mode_labels or tuple(basis.coordinates("mode")[:len(mode_labels)]) != mode_labels or
                    document["bases"][mode_axis]["indices"] != list(mode_labels)):
                raise ValueError(f"{product} mode selection must be the bound stored prefix")
        else:
            basis_id = None
            mode_labels = ()

        if product == "v2p":
            point_id = document["bases"]["point_right"]["data_id"]
            labels = tuple(output.coordinates("point_right"))
            if document["bases"]["point_right"]["indices"] != list(labels):
                raise ValueError("V/P point labels disagree with their actual point binding")
            role = dict(left_id=None, left_labels=(), right_id=point_id,
                        right_labels=labels)
            if not isinstance(point_id, str) or not point_id:
                raise ValueError("V/P output must bind its actual right point ID")
            point_right_by_id.setdefault(point_id, []).extend(labels)
        elif product == "p2v":
            point_id = document["bases"]["point_left"]["data_id"]
            labels = tuple(output.coordinates("point_left"))
            if document["bases"]["point_left"]["indices"] != list(labels):
                raise ValueError("P/V point labels disagree with their actual point binding")
            role = dict(left_id=point_id, left_labels=labels,
                        right_id=None, right_labels=())
            if not isinstance(point_id, str) or not point_id:
                raise ValueError("P/V output must bind its actual left point ID")
            point_left_by_id.setdefault(point_id, []).extend(labels)
        else:
            left_id = document["bases"]["point_left"]["data_id"]
            left_labels = tuple(output.coordinates("point_left"))
            if document["bases"]["point_left"]["indices"] != list(left_labels):
                raise ValueError("P/P left labels disagree with their actual point binding")
            right_id = definition.get("point_right_id")
            if definition.get("point_right_selection") == "same_ordered_prefix_as_point_left":
                right_labels = left_labels
                if right_id != left_id:
                    raise ValueError("same-prefix P/P selection must use one actual point ID")
            elif definition.get("point_right_selection") == "explicit_ordered_labels":
                right_labels = tuple(definition.get("point_right_labels", ()))
            else:
                raise ValueError("P/P output has an unsupported right-point selection")
            if (not isinstance(left_id, str) or not left_id or
                    not isinstance(right_id, str) or not right_id or not right_labels):
                raise ValueError("P/P output must bind two actual point endpoint views")
            role = dict(left_id=left_id, left_labels=left_labels,
                        right_id=right_id, right_labels=right_labels)
            point_left_by_id.setdefault(left_id, []).extend(left_labels)
            point_right_by_id.setdefault(right_id, []).extend(right_labels)
        endpoint_selection = _endpoint_selection(selection, output.coordinates("component"), boundary=path_boundary)
        for side, identity in (("left", role["left_id"]), ("right", role["right_id"])):
            if identity is not None:
                required_inputs.add(identity)
                if identity not in input_specs:
                    raise ValueError(f"missing actual point spec {identity!r}")
                validate_point_field(gauge, input_specs[identity], selection=selection if side == "left" else endpoint_selection)
        roles[product] = dict(
            gauge_id=bound_gauge, basis_id=basis_id, mode_labels=mode_labels,
            paths=tuple(paths), momenta=momenta, phase=phase_convention,
            current_schema=current_schema, temporal_path_schema=temporal_path_schema,
            **role)
        bound_ids = {bound_gauge}
        if basis_id is not None:
            bound_ids.add(basis_id)
        if role["left_id"] is not None:
            bound_ids.add(role["left_id"])
        if role["right_id"] is not None:
            bound_ids.add(role["right_id"])
        if set(document["parents"]) != bound_ids:
            raise ValueError(f"{product} output parent IDs disagree with its declared endpoint bindings")

    if set(input_specs) != required_inputs:
        raise ValueError("input_specs must contain exactly the deduplicated actual output parents")
    # The W request builder passes one ordered union for a same-ID left/right
    # point parent: left labels first, then unseen right labels. Separate IDs
    # each carry the union of the roles actually requested from that ID.
    point_view_labels = {}
    for identity in set(point_left_by_id) | set(point_right_by_id):
        labels = []
        for label in point_left_by_id.get(identity, ()):
            if label not in labels:
                labels.append(label)
        for label in point_right_by_id.get(identity, ()):
            if label not in labels:
                labels.append(label)
        point_view_labels[identity] = tuple(labels)
        if any(label not in input_specs[identity].coordinates("point") for label in labels):
            raise ValueError("output point view is not covered by its registered actual input")

    def payload_bytes(spec, name):
        return _perambulator_array_bytes(
            spec.shape, _perambulator_dtype_itemsize(spec.document["dtype"], name))

    def unique_append(values, items):
        result = list(values)
        for item in items:
            if item not in result:
                result.append(item)
        return result

    workspace = workspace_reserve_bytes + host_staging_bytes
    result = {}
    for product, role in roles.items():
        gauge = input_specs[role["gauge_id"]]
        gauge_doc = gauge.document
        lx, ly, lz, lt = gauge_doc["configuration"]["lattice"]
        volume = lx * ly * lz
        paths, momenta = role["paths"], role["momenta"]
        phase = role["phase"]
        phase_keys = {(momentum, path.displacement[:3], phase) for momentum in momenta for path in paths}
        phase_cache_peak = _elemental_phase_cache_peak_bytes(
            volume, len(phase_keys), phase)
        stable_phase_bytes = min(64, len(phase_keys)) * volume * 16

        parent_ids = {role["gauge_id"]}
        if role["basis_id"] is not None:
            parent_ids.add(role["basis_id"])
        if role["left_id"] is not None:
            parent_ids.add(role["left_id"])
        if role["right_id"] is not None:
            parent_ids.add(role["right_id"])
        parent_bytes = sum(payload_bytes(input_specs[identity], identity) for identity in parent_ids)
        gauge_stack_bytes = payload_bytes(gauge, role["gauge_id"])
        owned_links_bytes = 4 * lt * volume * 9 * 16

        left_count = len(role["left_labels"])
        right_count = len(role["right_labels"])
        point_snapshots = 0
        point_load_phases = []
        for identity, labels in ((role["left_id"], role["left_labels"]),
                                 (role["right_id"], role["right_labels"])):
            if identity is None or not labels:
                continue
            point_spec = input_specs[identity]
            itemsize = _perambulator_dtype_itemsize(point_spec.document["dtype"], identity)
            selected_bytes = point_spec.shape[0] * len(labels) * 3 * itemsize
            point_snapshots += selected_bytes
            view_count = len(point_view_labels[identity])
            view_bytes = point_spec.shape[0] * view_count * 3 * itemsize
            # Dataset views materialize canonical per-time rows. Ordered
            # point selection and ElementalGenerator can each own a copy;
            # retain the conservative overlap bound. For P/P,
            # the left table remains resident while the right handle is loaded.
            ordered_copy = selected_bytes
            transient = max(2 * view_bytes, view_bytes + ordered_copy,
                            2 * ordered_copy)
            if point_load_phases:
                transient += point_snapshots - selected_bytes
            point_load_phases.append(transient)

        validation_bytes = 0
        for identity in {role["left_id"], role["right_id"]} - {None}:
            point_spec = input_specs[identity]
            itemsize = _perambulator_dtype_itemsize(point_spec.document["dtype"], identity)
            row_bytes = len(point_spec.coordinates("point")) * 3 * itemsize
            # One complete physical-time row, np.unique(axis=0)'s returned
            # coordinate row and a conservative row-sized sort/index vector.
            validation_bytes = max(validation_bytes, 2 * row_bytes +
                                   len(point_spec.coordinates("point")) * 8)

        output = output_specs[product]
        output_row_bytes = _perambulator_array_bytes(output.shape[1:], 16)
        output_elements = output_row_bytes // 16
        writer_mask_bytes = output_elements
        modes = len(role["mode_labels"])
        tile_modes = min(modes, ne_block_size)
        vector_row_bytes = 0
        vector_tile_bytes = 0
        vector_source_id = role["basis_id"]
        if vector_source_id is not None:
            basis = input_specs[vector_source_id]
            vector_row_bytes = (len(basis.coordinates("mode")) * volume * 3 *
                               _perambulator_dtype_itemsize(basis.document["dtype"], vector_source_id))
            vector_tile_bytes = tile_modes * volume * 3 * 16

        # A directed link transport starts from a full-volume identity and
        # overlaps old result/link arrays while each path step is multiplied.
        # Five link matrices cover a backward-step conjugate plus the retained
        # preceding link during replacement; path changes can retain the prior
        # transport while the next one is constructed.
        link_matrix_bytes = volume * 9 * 16
        transport_build = _point_path_transport_build_bytes(paths, link_matrix_bytes)
        # path_transport always starts with a full-volume identity, including
        # the empty path; open-boundary invalid rows are covered by the valid
        # rows' peak because admission is for the complete-time workflow.
        transport_live = link_matrix_bytes

        # calc_path_blocks materializes right endpoint bases for every right
        # point sector, including P/P. Updating bases for a later path overlaps
        # the previous views' backing array, subtraction input, and new modulo
        # result. A single path overlaps only the latter two arrays.
        right_translation_peak = 0
        if role["right_id"] is not None:
            right_translation_peak = right_count * 3 * 8 * (3 if len(paths) > 1 else 2)

        kernel_intermediate = 0
        sparse_retained = 0
        sparse_build = 0
        point_writer_coordinates = 0
        if product == "v2p":
            link_gather = right_count * 9 * 16
            tile_point = tile_modes * right_count * 3 * 16
            # TimeField reads one complete physical-time basis row before the
            # selected mode tile is copied. Advanced point gathers/conjugation
            # and the phase-weighted tile may coexist with that row.
            vector_work = (vector_row_bytes + vector_tile_bytes + 3 * tile_point +
                           link_gather + right_count * 16 + right_translation_peak)
            kernel_intermediate = vector_work + phase_cache_peak + transport_live
        elif product == "p2v":
            link_gather = left_count * 9 * 16
            tile_point = tile_modes * left_count * 3 * 16
            # The point gather and einsum result overlap with reader-owned
            # fields. The later phase multiplication overlaps the base and
            # result point tiles; two tile-point arrays cover both stages.
            vector_work = vector_row_bytes + 2 * vector_tile_bytes + 2 * tile_point
            vector_work += link_gather + left_count * 16
            kernel_intermediate = vector_work + phase_cache_peak + transport_live
        else:
            max_pairs = min(left_count, right_count)
            pair_indices = max_pairs * 2 * 8
            sparse_value_bytes = sum(
                min(left_count, right_count) * 9 * 16
                for _path in paths for _momentum in momenta)
            indices_by_path = sum(min(left_count, right_count) * 2 * 8 for _path in paths)
            # point_pairs keeps the original left coordinates, two translated
            # coordinate arrays and COO indices. sparse_point_block then has
            # per-pair XYZ, phase, gathered transport, and returned color
            # matrices live together. The full-volume phase cache and current
            # transport are accounted in their own fields below.
            sparse_build = (pair_indices + 2 * left_count * 3 * 8 +
                            max_pairs * 3 * 8 + max_pairs * (16 + 9 * 16 + 9 * 16))
            sparse_build += right_translation_peak
            sparse_retained = sparse_value_bytes + indices_by_path
            # The writer also reads endpoint coordinates, derives expected
            # pairs, expands all sparse blocks to the fixed left row slots,
            # then allocates the finite mask. Python lookup structures remain
            # covered by the explicit caller workspace reserve.
            # On each later component, the prior translated-left array remains
            # live while addition and modulo produce the next one. The source
            # table, those two new arrays, and the right coordinate table can
            # therefore overlap during support verification.
            point_writer_coordinates = (4 * left_count + right_count) * 3 * 8

        # Consecutive temporal paths can read distinct endpoint times. The
        # previous path's converted mode tile, spatial roll and endpoint
        # gather may remain live while the next row is read and converted.
        # Model that transition separately from the fully resident basis
        # payload already counted in input_bytes.
        temporal_transition = 0
        if (role["temporal_path_schema"] is not None and len(paths) > 1
                and product in {"v2p", "p2v"}):
            endpoint_offsets = {
                path.displacement[3] if product == "p2v" else 0
                for path in paths
            }
            endpoint_time_count = min(lt, len({offset % lt for offset in endpoint_offsets}))
            previous_roll = vector_tile_bytes if product == "p2v" else 0
            next_roll = vector_tile_bytes if product == "p2v" else 0
            previous_gather = tile_modes * left_count * 3 * 16
            next_gather = tile_modes * right_count * 3 * 16
            temporal_transition = (
                vector_tile_bytes + previous_roll + previous_gather +
                vector_row_bytes + vector_tile_bytes + next_roll + next_gather +
                right_translation_peak
            )
            if endpoint_time_count > 1:
                temporal_transition += vector_row_bytes

        # Path transport construction is a distinct phase. A previous path's
        # transport may remain live while path_transport builds its successor;
        # the output row (vector products) or prior sparse blocks (P/P) also
        # remain live during that replacement.
        transport_replacement_peak = transport_build + (
            transport_live if len(paths) > 1 else 0)
        endpoint_replacement_peak = transport_live + right_translation_peak
        path_build_intermediate = (max(transport_replacement_peak,
                                       endpoint_replacement_peak) +
                                   stable_phase_bytes + sparse_retained +
                                   temporal_transition)
        if product == "p2p":
            kernel_intermediate = (sparse_retained + sparse_build +
                                   phase_cache_peak + transport_live)

        host_phases = [
            MemoryEstimate(input_bytes=parent_bytes,
                           intermediate_bytes=gauge_stack_bytes,
                           snapshot_bytes=owned_links_bytes,
                           workspace_bytes=workspace),
            MemoryEstimate(input_bytes=parent_bytes,
                           intermediate_bytes=validation_bytes,
                           workspace_bytes=workspace),
        ]
        for transient in point_load_phases:
            host_phases.append(MemoryEstimate(
                input_bytes=parent_bytes,
                snapshot_bytes=owned_links_bytes,
                intermediate_bytes=transient,
                workspace_bytes=workspace))
        host_phases.append(MemoryEstimate(
            input_bytes=parent_bytes,
            snapshot_bytes=owned_links_bytes + point_snapshots,
            intermediate_bytes=kernel_intermediate,
            output_bytes=output_row_bytes,
            assembly_extra_bytes=writer_mask_bytes,
            workspace_bytes=workspace))
        host_phases.append(MemoryEstimate(
            input_bytes=parent_bytes,
            snapshot_bytes=owned_links_bytes + point_snapshots,
            intermediate_bytes=path_build_intermediate,
            output_bytes=output_row_bytes if product != "p2p" else 0,
            workspace_bytes=workspace))
        if product == "p2p":
            host_phases.append(MemoryEstimate(
                input_bytes=parent_bytes,
                snapshot_bytes=owned_links_bytes + point_snapshots,
                intermediate_bytes=(sparse_retained + phase_cache_peak +
                                    point_writer_coordinates),
                output_bytes=output_row_bytes,
                assembly_extra_bytes=writer_mask_bytes,
                workspace_bytes=workspace))
        host_estimate = max(host_phases, key=lambda item: item.logical_peak_bytes)
        result[product] = (host_estimate, MemoryEstimate())
    return result
