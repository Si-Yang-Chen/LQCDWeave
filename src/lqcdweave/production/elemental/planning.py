"""Elemental input preparation and read-only dataset planning."""

import numpy as np

from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.contracts.time_selection import (
    TimeSelectionRecord, dataset_time_selection, dataset_physical_times,
)
from lqcdweave.contracts.current import (
    CURRENT_DIRECTED_POINT_SCHEMA, validate_directed_point_current_spec,
)
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, GaugePath, integer,
)
from lqcdweave.planning.production import (
    validate_elemental_fields, validate_gauge_field, elemental_dataset_document, elemental_projection_documents,
    directed_current_dataset_document, validate_point_field,
    point_elemental_dataset_document,
)
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator
from lqcdweave.runtime.provenance import local_provenance
from lqcdweave.runtime.datasets import (
    apply_source_policy,
)


def prepare_elemental_stage(gauge_data, basis_data, *, paths=((),),
                            momentum_list=((0, 0, 0),), usedNe=None,
                            phase_convention="anchor", ne_block_size=32, selection=None,
                            array_module=np):
    """Prepare a producer on the explicit array backend and its output contract.

    NumPy remains the default. CuPy results stay on device until the caller
    explicitly transfers them for a host-only writer.
    """
    times = validate_elemental_fields(gauge_data.spec, basis_data.spec, selection=selection)
    cfg = gauge_data.spec.document["configuration"]
    generator = ElementalGenerator(cfg["lattice"], DatasetFieldHandle(gauge_data),
        DatasetFieldHandle(basis_data), usedNe=usedNe, paths=paths,
        path_version="v2", momentum_list=momentum_list,
        phase_convention=phase_convention, ne_block_size=ne_block_size, selection=selection,
        array_module=array_module)
    if any(step >= 6 for path in generator.paths for step in path.steps):
        raise ValueError("production workflow currently requires spatial paths")
    modes = basis_data.spec.coordinates("mode")
    count = len(modes) if usedNe is None else generator.usedNe
    if count > len(modes):
        raise ValueError("usedNe exceeds registered basis coverage")
    modes = modes[:count]
    spec = DatasetSpec(elemental_dataset_document(gauge_data.spec, basis_data.spec,
        gauge_id=gauge_data.data_id, basis_id=basis_data.data_id,
        components=[dict(path=list(p.steps)) for p in generator.paths],
        momenta=generator.momentum_list, modes=modes, phase_convention=phase_convention, selection=selection))
    provenance = local_provenance()
    provenance["production"] = dict(workflow="elemental_spatial_vv", array_module=array_module.__name__,
                                    ne_block_size=generator.ne_block_size)
    return generator, spec, provenance, times, cfg


_POINT_ELEMENTAL_PRODUCTS = ("v2p", "p2v", "p2p")


def _input_time_selection(spec, selection):
    """Validate an explicit sparse input in its own saved physical order."""
    if selection is None:
        return None
    return TimeSelectionRecord(spec.document["configuration"]["lattice"][3],
                               dataset_physical_times(spec))


def _physical_time_coordinate(spec, time):
    selection = dataset_time_selection(spec)
    return time if selection is None else selection.slot(time)


def _validate_registered_point_coordinates(gauge, points, *, selection=None):
    """Check actual point values before a workflow creates any output path."""
    times = validate_point_field(gauge.spec, points.spec,
                                 selection=_input_time_selection(points.spec, selection))
    lattice = gauge.spec.document["configuration"]["lattice"]
    upper = np.asarray(lattice[:3], dtype=np.int64)
    count = len(points.spec.coordinates("point"))
    for time in times:
        coordinates = np.asarray(points.read(time=_physical_time_coordinate(points.spec, time)))
        if coordinates.shape != (count, 3) or coordinates.dtype.kind not in "iu":
            raise ValueError("point reader must return an integer (point,xyz) coordinate table")
        if np.any(coordinates < 0) or np.any(coordinates >= upper):
            raise ValueError("registered point coordinates are outside the spatial lattice")
        if len(np.unique(coordinates, axis=0)) != count:
            raise ValueError("registered points must be unique at every physical time")
    return times


def _point_view_spec(spec, labels):
    document = spec.document
    for axis in document["axes"]:
        if axis["name"] == "point":
            axis["coordinates"] = list(labels)
            break
    else:
        raise ValueError("registered point dataset has no point axis")
    return DatasetSpec(document)


def _selected_point_labels(spec, *, count=None, labels=None, name="point"):
    available = spec.coordinates("point")
    if labels is not None:
        if count is not None:
            raise ValueError(f"{name} count and explicit labels are mutually exclusive")
        selected = list(labels)
        if (not selected or any(type(label) is not int or label < 0 for label in selected)
                or len(set(selected)) != len(selected)
                or any(label not in available for label in selected)):
            raise ValueError(f"{name} labels must be an explicit unique registered coordinate selection")
        return selected
    size = len(available) if count is None else integer(count, name, 1)
    if size > len(available):
        raise ValueError(f"{name} exceeds registered point coverage")
    return list(available[:size])


def _normalize_point_elemental_request_options(
    gauge_id, basis_id, points_id, *, paths, momentum_list, sectors, usedNe, usedNp,
    phase_convention, points_right_id=None, usedNp_right=None,
    point_right_labels=None, current_schema=None, temporal_path_schema=None,
    path_boundary="periodic", selection=None,
):
    if point_right_labels is not None:
        try:
            point_right_labels = tuple(point_right_labels)
        except TypeError as exc:
            raise TypeError("point_right_labels must be an iterable of labels") from exc
    if isinstance(sectors, str):
        raise TypeError("sectors must be an iterable of sector names")
    products = tuple(sectors)
    if (not products or len(set(products)) != len(products)
            or set(products) - set(_POINT_ELEMENTAL_PRODUCTS)):
        raise ValueError("sectors must be distinct selections from v2p, p2v and p2p")
    point_current = current_schema is not None
    temporal_path = temporal_path_schema is not None
    if point_current and temporal_path:
        raise ValueError("current_schema and temporal_path_schema are mutually exclusive")
    if temporal_path and temporal_path_schema != TEMPORAL_POINT_PATH_SCHEMA:
        raise ValueError("unsupported temporal point-path schema")
    if temporal_path and not set(products) <= {"v2p", "p2v", "p2p"}:
        raise ValueError("temporal point-path production supports v2p, p2v and p2p sectors only")
    if point_current:
        if current_schema != CURRENT_DIRECTED_POINT_SCHEMA:
            raise ValueError("unsupported point-current schema")
        if not set(products) <= {"v2p", "p2v", "p2p"}:
            raise ValueError("point-current production supports v2p, p2v and p2p sectors only")
        if path_boundary not in {"periodic", "open"}:
            raise ValueError("point-current path boundary must be periodic or open")
        if phase_convention != "midpoint":
            raise ValueError("point-current products require midpoint phase")
    elif path_boundary not in (("periodic", "open") if temporal_path else ("periodic",)):
        raise ValueError("ordinary point-elemental production requires periodic paths")
    vector_products = set(products) & {"v2p", "p2v"}
    if vector_products and (not isinstance(basis_id, str) or not basis_id):
        raise ValueError("V/P sectors require a registered actual basis ID")
    if not vector_products and (basis_id is not None or usedNe is not None):
        raise ValueError("pure P/P production requires basis_id=None and rejects usedNe")
    if not isinstance(gauge_id, str) or not gauge_id:
        raise ValueError("registered actual gauge ID is required")
    if points_id is not None and (not isinstance(points_id, str) or not points_id):
        raise ValueError("points_id must be an actual point ID or None")
    if points_right_id is not None and (not isinstance(points_right_id, str) or not points_right_id):
        raise ValueError("points_right_id must be an actual point ID or None")
    if usedNp_right is not None and point_right_labels is not None:
        raise ValueError("usedNp_right and point_right_labels are mutually exclusive")
    uses_left = bool(set(products) & {"p2v", "p2p"})
    uses_right = bool(set(products) & {"v2p", "p2p"})
    if uses_left and points_id is None:
        raise ValueError("selected left-point sectors require points_id")
    effective_right_id = points_right_id if points_right_id is not None else points_id
    if uses_right and effective_right_id is None:
        raise ValueError("selected right-point sectors require points_right_id or points_id")
    if not uses_right and (points_right_id is not None or usedNp_right is not None
                           or point_right_labels is not None):
        raise ValueError("right point arguments are unused by the selected sectors")
    if not uses_left and points_id is not None and points_right_id is not None \
            and points_id != points_right_id:
        raise ValueError("points_id is unused when only a different right-point pool is selected")
    if not uses_left and usedNp is not None and points_right_id is not None \
            and points_id != points_right_id:
        raise ValueError("use usedNp_right for a distinct right-only point pool")
    if not uses_left and usedNp is not None \
            and (usedNp_right is not None or point_right_labels is not None):
        raise ValueError("usedNp cannot be combined with a right-specific point selection")
    if phase_convention not in ("anchor", "midpoint", "none"):
        raise ValueError("phase_convention must be anchor, midpoint or none")
    normalized_paths = tuple(GaugePath.coerce(path) for path in paths)
    if not normalized_paths or len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("point elemental production requires distinct paths")
    if point_current:
        if any(len(path.steps) != 1 or path.steps[0] not in (6, 7)
               for path in normalized_paths):
            raise ValueError("point-current production requires one-link temporal paths 6 or 7")
    elif temporal_path:
        if any(any(step < 0 or step >= 8 for step in path.steps)
               for path in normalized_paths):
            raise ValueError("temporal point-path production requires ordered directions 0..7")
        if not any(step >= 6 for path in normalized_paths for step in path.steps):
            raise ValueError("temporal point-path production requires at least one temporal step")
    elif any(step >= 6 for path in normalized_paths for step in path.steps):
        raise ValueError("point elemental production requires spatial paths only")
    components = [dict(path=list(path.steps)) for path in normalized_paths]
    momenta = [list(integer(value, "momentum") for value in momentum)
               for momentum in momentum_list]
    if (not momenta or any(len(momentum) != 3 for momentum in momenta)
            or len({tuple(momentum) for momentum in momenta}) != len(momenta)):
        raise ValueError("unique nonempty integer XYZ momenta are required")

    return dict(
        gauge_id=gauge_id, basis_id=basis_id, points_id=points_id,
        points_right_id=points_right_id, effective_right_id=effective_right_id,
        products=products, point_current=point_current, uses_left=uses_left,
        uses_right=uses_right, vector_products=vector_products,
        normalized_paths=normalized_paths, components=components, momenta=momenta,
        usedNe=usedNe, usedNp=usedNp, usedNp_right=usedNp_right,
        point_right_labels=point_right_labels, phase_convention=phase_convention,
        current_schema=current_schema, temporal_path_schema=temporal_path_schema,
        path_boundary=path_boundary, selection=selection,
    )


def _point_elemental_requests_from_specs(gauge_spec, basis_spec, left_points_spec,
                                        right_points_spec, options):
    """Build point request definitions from detached metadata only.

    This function is shared by planning, ordinary execution, and memory
    preflight. It never opens an index entry or reads payload values.
    """
    gauge_id = options["gauge_id"]
    basis_id = options["basis_id"]
    points_id = options["points_id"]
    effective_right_id = options["effective_right_id"]
    products = options["products"]
    uses_left = options["uses_left"]
    uses_right = options["uses_right"]
    vector_products = options["vector_products"]
    if not isinstance(gauge_spec, DatasetSpec):
        raise TypeError("gauge_spec must be a DatasetSpec")
    if uses_left and not isinstance(left_points_spec, DatasetSpec):
        raise TypeError("left_points_spec must be a DatasetSpec for selected left sectors")
    if uses_right and not isinstance(right_points_spec, DatasetSpec):
        raise TypeError("right_points_spec must be a DatasetSpec for selected right sectors")
    validate_gauge_field(gauge_spec)
    if uses_left:
        validate_point_field(gauge_spec, left_points_spec,
                             selection=_input_time_selection(left_points_spec, options["selection"]))
    if uses_right:
        validate_point_field(gauge_spec, right_points_spec,
                             selection=_input_time_selection(right_points_spec, options["selection"]))

    if uses_left:
        left_labels = _selected_point_labels(left_points_spec, count=options["usedNp"], name="usedNp")
    elif uses_right and (options["points_right_id"] is None or
                         options["points_right_id"] == points_id):
        # Backward-compatible right-only shorthand uses usedNp on the one pool.
        left_labels = _selected_point_labels(
            right_points_spec, count=options["usedNp"], name="usedNp")
    else:
        left_labels = []
    if uses_right:
        right_count = options["usedNp_right"]
        right_selection = options["point_right_labels"]
        if (right_count is None and right_selection is None and uses_left and
                effective_right_id == points_id):
            right_labels = list(left_labels)
        elif (right_count is None and right_selection is None and not uses_left and
              (options["points_right_id"] is None or
               options["points_right_id"] == points_id)):
            right_labels = list(left_labels)
        else:
            right_labels = _selected_point_labels(
                right_points_spec, count=right_count, labels=right_selection,
                name="usedNp_right")
    else:
        right_labels = []

    mode_labels = []
    basis_spec_for_outputs = None
    if vector_products:
        if not isinstance(basis_spec, DatasetSpec):
            raise TypeError("basis_spec must be a DatasetSpec for V/P products")
        validate_elemental_fields(gauge_spec, basis_spec,
                                  selection=_input_time_selection(basis_spec, options["selection"]))
        stored_modes = basis_spec.coordinates("mode")
        mode_count = len(stored_modes) if options["usedNe"] is None else integer(
            options["usedNe"], "usedNe", 1)
        if mode_count > len(stored_modes):
            raise ValueError("usedNe exceeds registered basis coverage")
        mode_labels = list(stored_modes[:mode_count])
        basis_spec_for_outputs = basis_spec

    # A shared actual parent is requested once as the ordered union of both
    # endpoint views. Each output keeps its independent endpoint ordering.
    requests = {"gauge": DatasetRequest(gauge_spec, data_id=gauge_id)}
    if uses_left and uses_right and points_id == effective_right_id:
        input_point_labels = list(left_labels)
        for label in right_labels:
            if label not in input_point_labels:
                input_point_labels.append(label)
        requests["points"] = DatasetRequest(
            _point_view_spec(left_points_spec, input_point_labels),
            subset_axes=("point",), data_id=points_id)
        left_node = right_node = "points"
        left_document_spec = _point_view_spec(left_points_spec, left_labels)
        right_document_spec = _point_view_spec(left_points_spec, right_labels)
    elif uses_left and uses_right:
        requests["points"] = DatasetRequest(
            _point_view_spec(left_points_spec, left_labels),
            subset_axes=("point",), data_id=points_id)
        requests["points_right"] = DatasetRequest(
            _point_view_spec(right_points_spec, right_labels),
            subset_axes=("point",), data_id=effective_right_id)
        left_node, right_node = "points", "points_right"
        left_document_spec = _point_view_spec(left_points_spec, left_labels)
        right_document_spec = _point_view_spec(right_points_spec, right_labels)
    elif uses_left:
        requests["points"] = DatasetRequest(
            _point_view_spec(left_points_spec, left_labels),
            subset_axes=("point",), data_id=points_id)
        left_node, right_node = "points", None
        left_document_spec = _point_view_spec(left_points_spec, left_labels)
        right_document_spec = None
    else:
        requests["points"] = DatasetRequest(
            _point_view_spec(right_points_spec, right_labels),
            subset_axes=("point",), data_id=effective_right_id)
        left_node, right_node = None, "points"
        left_document_spec = None
        right_document_spec = _point_view_spec(right_points_spec, right_labels)

    if vector_products:
        requests["basis"] = DatasetRequest(
            basis_spec_for_outputs, data_id=basis_id, subset_axes=("mode",))

    for product in products:
        product_basis = basis_spec_for_outputs if product in vector_products else None
        product_basis_id = {"request": "basis"} if product in vector_products else None
        product_points = (left_document_spec if left_document_spec is not None
                          else right_document_spec)
        product_points_id = ({"request": left_node} if left_node is not None
                             else {"request": right_node})
        right_ref = ({"request": right_node} if right_node is not None
                     else product_points_id)
        use_right_descriptor = (uses_right and uses_left and
            (effective_right_id != points_id or right_labels != left_labels))
        document = point_elemental_dataset_document(
            gauge_spec, product_basis, product_points, product=product,
            gauge_id={"request": "gauge"}, basis_id=product_basis_id,
            points_id=product_points_id, components=options["components"],
            momenta=options["momenta"],
            modes=mode_labels if product in vector_products else (),
            point_labels=(left_labels if product != "v2p" else right_labels),
            points_right=(right_document_spec if use_right_descriptor
                          and product in {"v2p", "p2p"} else None),
            points_right_id=(right_ref if use_right_descriptor
                             and product in {"v2p", "p2p"} else None),
            point_right_labels=(right_labels if product in {"v2p", "p2p"} else None),
            phase_convention=options["phase_convention"],
            current_schema=options["current_schema"],
            temporal_path_schema=options["temporal_path_schema"],
            path_boundary=options["path_boundary"], selection=options["selection"])
        prefix_axes = {
            "v2p": ("mode_left", "point_right"),
            "p2v": ("point_left", "mode_right"),
            "p2p": ("point_left",),
        }[product]
        requests[product] = DatasetRequest(document, subset_axes=prefix_axes)
    return requests


def _indexed_point_elemental_requests(index, gauge_id, basis_id, points_id, *,
                                      paths, momentum_list, sectors, usedNe, usedNp,
                                      phase_convention, points_right_id=None,
                                      usedNp_right=None, point_right_labels=None,
                                      current_schema=None, temporal_path_schema=None,
                                      path_boundary="periodic", selection=None):
    options = _normalize_point_elemental_request_options(
        gauge_id, basis_id, points_id, paths=paths, momentum_list=momentum_list,
        sectors=sectors, usedNe=usedNe, usedNp=usedNp,
        phase_convention=phase_convention, points_right_id=points_right_id,
        usedNp_right=usedNp_right, point_right_labels=point_right_labels,
        current_schema=current_schema, temporal_path_schema=temporal_path_schema,
        path_boundary=path_boundary, selection=selection)
    return _verified_point_elemental_requests(index, options)


def _verified_point_elemental_requests(index, options):
    """Open and validate point values, then reuse the pure request builder."""
    gauge_id = options["gauge_id"]
    basis_id = options["basis_id"]
    points_id = options["points_id"]
    uses_left, uses_right = options["uses_left"], options["uses_right"]
    effective_right_id = options["effective_right_id"]

    # Keep the existing full verifier and actual-coordinate validation for
    # ordinary planning and no-budget execution.
    with index.open(gauge_id) as gauge:
        left_context = index.open(points_id) if points_id is not None else None
        right_context = (index.open(effective_right_id)
                         if uses_right and effective_right_id != points_id else None)
        try:
            left_points = left_context.__enter__() if left_context is not None else None
            right_points = right_context.__enter__() if right_context is not None else left_points
            if left_points is not None:
                _validate_registered_point_coordinates(gauge, left_points, selection=options["selection"])
            if right_points is not None and right_points is not left_points:
                _validate_registered_point_coordinates(gauge, right_points, selection=options["selection"])
            left_spec = left_points.spec if left_points is not None else None
            right_spec = right_points.spec if right_points is not None else None
            if options["vector_products"]:
                with index.open(basis_id) as basis:
                    validate_elemental_fields(gauge.spec, basis.spec,
                                              selection=_input_time_selection(basis.spec, options["selection"]))
                    basis_spec = basis.spec
            else:
                basis_spec = None
            return _point_elemental_requests_from_specs(
                gauge.spec, basis_spec, left_spec, right_spec, options)
        finally:
            if right_context is not None:
                right_context.__exit__(None, None, None)
            if left_context is not None:
                left_context.__exit__(None, None, None)


def plan_indexed_point_elementals(index, gauge_id, basis_id, points_id, *,
                                  paths=((),), momentum_list=((0, 0, 0),),
                                  sectors=_POINT_ELEMENTAL_PRODUCTS, usedNe=None,
                                  usedNp=None, points_right_id=None,
                                  usedNp_right=None, point_right_labels=None,
                                  phase_convention="anchor",
                                  current_schema=None, temporal_path_schema=None,
                                  path_boundary="periodic",
                                  ne_block_size=32, data_ids=None,
                                  source_policy="allow_unknown", selection=None):
    """Preview registered V/P and P/P elemental products without computing them.

    ``basis_id`` is None for a pure P/P request. Full physical time is the
    default; explicit selection plans compact ordered anchor slots. ``points_id`` identifies the left point pool when a left endpoint
    is consumed and remains the legacy single-pool input otherwise. Optional
    right identity/count/labels select V/P's point endpoint or P/P's independent
    right endpoint; the old omitted-right P/P definition remains unchanged.
    """
    integer(ne_block_size, "ne_block_size", 1)
    requests = _indexed_point_elemental_requests(index, gauge_id, basis_id, points_id,
        paths=paths, momentum_list=momentum_list, sectors=sectors, usedNe=usedNe,
        usedNp=usedNp, phase_convention=phase_convention,
        points_right_id=points_right_id, usedNp_right=usedNp_right,
        point_right_labels=point_right_labels, current_schema=current_schema,
        temporal_path_schema=temporal_path_schema,
        path_boundary=path_boundary, selection=selection)
    preview = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    if current_schema is not None:
        for product in _POINT_ELEMENTAL_PRODUCTS:
            if product not in requests:
                continue
            validate_directed_point_current_spec(preview["stages"][product]["spec"])
    if temporal_path_schema is not None:
        for product in requests:
            if product in {"v2p", "p2v"} and preview["stages"][product]["spec"].document[
                    "definition"].get("temporal_path_schema") != TEMPORAL_POINT_PATH_SCHEMA:
                raise ValueError("planned point-path output lost its temporal schema")
    return apply_source_policy(preview, requests, index, source_policy=source_policy)


def _elemental_projection_requests(index, raw_id, rows, usedNe):
    with index.open(raw_id) as raw:
        return _elemental_projection_requests_from_spec(raw.spec, raw_id, rows, usedNe)


def _elemental_projection_requests_from_spec(raw_spec, raw_id, rows, usedNe):
    requested, projected = elemental_projection_documents(
        raw_spec, rows, raw_id=dict(request="raw"), usedNe=usedNe)
    return dict(raw=DatasetRequest(requested, data_id=raw_id,
                                   subset_axes=("component", "sink_mode", "source_mode")),
                projected=DatasetRequest(projected))


def plan_indexed_elemental_projection(index, raw_id, rows, *, usedNe=None,
                                     data_ids=None, source_policy="allow_unknown"):
    """Plan a finite spatial path combination over one registered raw parent."""
    requests = _elemental_projection_requests(index, raw_id, rows, usedNe)
    preview = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(preview, requests, index, source_policy=source_policy)


def _directed_current_requests(index, gauge_id, basis_id, momentum_list, usedNe, boundary, selection=None):
    with index.open(gauge_id) as gauge, index.open(basis_id) as basis:
        requested, document = directed_current_dataset_document(gauge.spec, basis.spec,
            gauge_id=dict(request="gauge"), basis_id=dict(request="basis"),
            momentum_list=momentum_list, usedNe=usedNe, boundary=boundary, selection=selection)
        return dict(gauge=DatasetRequest(gauge.spec, data_id=gauge_id),
            basis=DatasetRequest(requested, data_id=basis_id, subset_axes=("mode",)),
            current=DatasetRequest(document))


def plan_indexed_directed_current(index, gauge_id, basis_id, *,
                                  momentum_list=((0, 0, 0),), usedNe=None,
                                  boundary="periodic", data_ids=None,
                                  source_policy="allow_unknown", selection=None):
    """Preview eight-link production; default full-time or explicit compact anchors."""
    requests = _directed_current_requests(index, gauge_id, basis_id, momentum_list, usedNe, boundary, selection)
    preview = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(preview, requests, index, source_policy=source_policy)
