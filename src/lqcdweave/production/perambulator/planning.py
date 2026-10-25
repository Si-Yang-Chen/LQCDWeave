"""Read-only perambulator dataset planning."""

from collections.abc import Mapping
import json

import numpy as np

from lqcdweave.contracts.dataset import (
    DatasetRequest,
    DatasetSpec,
    canonical_json,
)
from lqcdweave.contracts.time_selection import normalize_time_selection, dataset_time_selection
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.planning.production import (
    propagator_dataset_document,
    validate_elemental_fields,
    validate_gauge_field,
    validate_point_field,
)
from lqcdweave.runtime.datasets import apply_source_policy


_PRODUCTS = ("VSV", "PSV", "PSP", "VSP")
_REQUIRED_DIRAC_FIELDS = (
    "mass", "xi_0", "nu", "clover_coeff_t", "clover_coeff_r",
)
_DIRAC_SEPARATE_FIELDS = frozenset({
    "solver_tolerance", "temporal_boundary", "t_boundary", "dtype",
    "spin_basis", "normalization",
})
def _json_object(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    try:
        result = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a JSON object")
    return result


def _normalize_dirac_definition(value):
    result = _json_object(value, "dirac_definition")
    if _DIRAC_SEPARATE_FIELDS & result.keys():
        raise ValueError("Dirac values recorded in separate DatasetSpec fields must not be duplicated")
    action = result.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("dirac_definition requires a nonempty action label")
    missing = [name for name in _REQUIRED_DIRAC_FIELDS if name not in result]
    if missing:
        raise ValueError("dirac_definition requires explicit physical fields: " +
                         ", ".join(missing))
    for name in _REQUIRED_DIRAC_FIELDS:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"dirac_definition.{name} must be a finite real number")
        # Store and later pass precisely this normalized floating-point value.
        try:
            result[name] = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"dirac_definition.{name} must be a finite real number") from exc
        if not np.isfinite(result[name]):
            raise ValueError(f"dirac_definition.{name} must be a finite real number")
    if result["xi_0"] <= 0 or result["nu"] <= 0:
        raise ValueError("dirac_definition xi_0 and nu must be positive")
    return json.loads(canonical_json(result))


def _normalize_science_options(dirac_definition, solver_tolerance, temporal_boundary,
                               dtype, spin_basis, normalization):
    dirac = _normalize_dirac_definition(dirac_definition)
    if isinstance(solver_tolerance, bool) or not isinstance(solver_tolerance, (int, float)):
        raise ValueError("solver_tolerance must be a finite real number")
    try:
        solver_tolerance = float(solver_tolerance)
    except (OverflowError, ValueError) as exc:
        raise ValueError("solver_tolerance must be a finite real number") from exc
    if not np.isfinite(solver_tolerance) or not 0 < solver_tolerance < 1:
        raise ValueError("solver_tolerance must be in (0,1)")
    if temporal_boundary not in ("periodic", "antiperiodic"):
        raise ValueError("temporal_boundary must be periodic or antiperiodic")
    if dtype not in ("complex64", "complex128"):
        raise ValueError("dtype must be complex64 or complex128")
    if not isinstance(spin_basis, str) or not spin_basis:
        raise ValueError("spin_basis must be a nonempty string")
    if normalization != "unscaled":
        raise ValueError("the current perambulator kernel supports unscaled normalization only")
    return dict(dirac=dirac, solver_tolerance=solver_tolerance,
                temporal_boundary=temporal_boundary, dtype=dtype,
                spin_basis=spin_basis, normalization=normalization)


def _validate_products(products):
    if isinstance(products, str):
        raise TypeError("products must be an iterable of product names")
    try:
        selected = tuple(products)
    except TypeError as exc:
        raise TypeError("products must be an iterable of product names") from exc
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(_PRODUCTS):
        raise ValueError("products must be distinct selections from VSV, PSV, PSP and VSP")
    return selected


def _optional_id(value, name):
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a nonempty actual dataset ID or None")


def _count(value, name):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")


def _prefix_view(spec, axis_name, count):
    coordinates = spec.coordinates(axis_name)
    if count > len(coordinates):
        raise ValueError(f"requested {axis_name} prefix exceeds the registered dataset")
    selected = list(coordinates[:count])
    document = spec.document
    for axis in document["axes"]:
        if axis["name"] == axis_name:
            axis["coordinates"] = selected
            break
    else:
        raise ValueError(f"registered input has no {axis_name!r} axis")
    if axis_name in document["bases"]:
        document["bases"][axis_name]["indices"] = selected
    return DatasetSpec(document), selected


def _validate_registered_point_coordinates(gauge, points, *, selection=None):
    """Validate point values before output creation.

    This is a local copy of the point-value check in
    ``production.elemental.planning._validate_registered_point_coordinates``. It avoids
    a private cross-workflow call while retaining its lattice/range/uniqueness
    contract.
    """
    times = validate_point_field(gauge.spec, points.spec, selection=selection)
    mapping = dataset_time_selection(points.spec, "time")
    upper = np.asarray(gauge.spec.document["configuration"]["lattice"][:3], dtype=np.int64)
    count = len(points.spec.coordinates("point"))
    for time in times:
        coordinates = np.asarray(points.read(time=time if mapping is None else mapping.slot(time)))
        if coordinates.shape != (count, 3) or coordinates.dtype.kind not in "iu":
            raise ValueError("registered point reader must return an integer (point,xyz) table")
        if np.any(coordinates < 0) or np.any(coordinates >= upper):
            raise ValueError("registered point coordinates are outside the spatial lattice")
        if len(np.unique(coordinates, axis=0)) != count:
            raise ValueError("registered points must be unique at every physical time")
    return times


def _validate_input_arguments(gauge_id, products, *, basis_id, basis_src_id, basis_snk_id,
                              points_id, points_src_id, points_snk_id, Ne, Ne_src, Ne_snk,
                              Np, Np_src, Np_snk):
    if not isinstance(gauge_id, str) or not gauge_id:
        raise ValueError("registered actual gauge ID is required")
    for name, value in (("basis_id", basis_id), ("basis_src_id", basis_src_id),
                        ("basis_snk_id", basis_snk_id), ("points_id", points_id),
                        ("points_src_id", points_src_id), ("points_snk_id", points_snk_id)):
        _optional_id(value, name)
    for name, value in (("Ne", Ne), ("Ne_src", Ne_src), ("Ne_snk", Ne_snk),
                        ("Np", Np), ("Np_src", Np_src), ("Np_snk", Np_snk)):
        _count(value, name)

    vector_source = any(product in ("VSV", "PSV") for product in products)
    vector_sink = any(product in ("VSV", "VSP") for product in products)
    point_sink = any(product in ("PSV", "PSP") for product in products)
    point_source = any(product in ("PSP", "VSP") for product in products)

    if basis_id is not None and (basis_src_id is not None or basis_snk_id is not None):
        raise ValueError("pass basis_id or endpoint-specific vector basis IDs")
    if Ne is not None and (Ne_src is not None or Ne_snk is not None):
        raise ValueError("pass Ne or endpoint-specific mode counts")
    if not vector_source and not vector_sink and any(value is not None for value in
                                 (basis_id, basis_src_id, basis_snk_id, Ne, Ne_src, Ne_snk)):
        raise ValueError("vector basis arguments are unused by the selected products")
    if not vector_source and (basis_src_id is not None or Ne_src is not None):
        raise ValueError("source vector arguments are unused by the selected products")
    if not vector_sink and (basis_snk_id is not None or Ne_snk is not None):
        raise ValueError("sink vector arguments are unused by the selected products")
    source_basis_id = basis_src_id if basis_src_id is not None else basis_id
    sink_basis_id = basis_snk_id if basis_snk_id is not None else basis_id
    if vector_source and source_basis_id is None:
        raise ValueError("VSV/PSV requires an actual source vector basis ID")
    if vector_sink and sink_basis_id is None:
        raise ValueError("VSV/VSP requires an actual sink vector basis ID")
    if not vector_sink and (basis_snk_id is not None or Ne_snk is not None):
        raise ValueError("sink vector basis arguments are unused without VSV")

    if points_id is not None and (points_src_id is not None or points_snk_id is not None):
        raise ValueError("pass points_id or endpoint-specific point IDs")
    if Np is not None and (Np_src is not None or Np_snk is not None):
        raise ValueError("pass Np or endpoint-specific point counts")
    if not point_sink and not point_source and any(value is not None for value in
            (points_id, points_src_id, points_snk_id, Np, Np_src, Np_snk)):
        raise ValueError("point arguments are unused by the selected products")
    sink_points_id = points_snk_id if points_snk_id is not None else points_id
    source_points_id = points_src_id if points_src_id is not None else points_id
    if point_sink and sink_points_id is None:
        raise ValueError("PSV/PSP requires an actual sink point ID")
    if point_source and source_points_id is None:
        raise ValueError("PSP/VSP requires an actual source point ID")
    if not point_source and (points_src_id is not None or Np_src is not None):
        raise ValueError("source point arguments are unused without PSP")
    return dict(source_basis_id=source_basis_id if vector_source else None,
                sink_basis_id=sink_basis_id if vector_sink else None,
                sink_points_id=sink_points_id if point_sink else None,
                source_points_id=source_points_id if point_source else None)


def _indexed_requests(index, gauge_id, products, science, *, basis_id, basis_src_id,
                      basis_snk_id, points_id, points_src_id, points_snk_id,
                      Ne, Ne_src, Ne_snk, Np, Np_src, Np_snk,
                      source_selection=None, sink_selection=None):
    resolved_ids = _validate_input_arguments(
        gauge_id, products, basis_id=basis_id, basis_src_id=basis_src_id,
        basis_snk_id=basis_snk_id, points_id=points_id, points_src_id=points_src_id,
        points_snk_id=points_snk_id, Ne=Ne, Ne_src=Ne_src, Ne_snk=Ne_snk,
        Np=Np, Np_src=Np_src, Np_snk=Np_snk)
    vector_source_id = resolved_ids["source_basis_id"]
    vector_sink_id = resolved_ids["sink_basis_id"]
    point_sink_id = resolved_ids["sink_points_id"]
    point_source_id = resolved_ids["source_points_id"]
    needs_vector_source = vector_source_id is not None
    needs_vector_sink = vector_sink_id is not None
    needs_point_sink = point_sink_id is not None
    needs_point_source = point_source_id is not None

    with index.open(gauge_id) as gauge:
        times = validate_gauge_field(gauge.spec)
        configuration = gauge.spec.document["configuration"]
        source_record = normalize_time_selection(source_selection, Lt=configuration['lattice'][3])
        sink_record = normalize_time_selection(sink_selection, Lt=configuration['lattice'][3])
        input_readers = {gauge_id: gauge}

        def reader(identity):
            if identity not in input_readers:
                input_readers[identity] = index.open(identity)
                input_readers[identity].__enter__()
            return input_readers[identity]

        try:
            basis_specs = {}
            for identity in dict.fromkeys((vector_source_id, vector_sink_id)):
                if identity is None:
                    continue
                basis = reader(identity)
                if identity == vector_source_id:
                    validate_elemental_fields(gauge.spec, basis.spec, selection=source_record)
                if identity == vector_sink_id:
                    validate_elemental_fields(gauge.spec, basis.spec, selection=sink_record)
                basis_specs[identity] = basis.spec

            point_specs = {}
            point_readers = {}
            for identity in dict.fromkeys((point_sink_id, point_source_id)):
                if identity is None:
                    continue
                points = reader(identity)
                point_specs[identity] = points.spec
                point_readers[identity] = points
            for identity, points in point_readers.items():
                required_times = tuple(dict.fromkeys(
                    (source_record.times if identity == point_source_id else ()) +
                    (sink_record.times if identity == point_sink_id else ())))
                required = type(source_record)(source_record.Lt, required_times)
                _validate_registered_point_coordinates(gauge, points, selection=required)

            source_modes = None
            sink_modes = None
            source_mode_count = sink_mode_count = None
            if needs_vector_source:
                source_available = basis_specs[vector_source_id].coordinates("mode")
                source_mode_count = (len(source_available) if Ne is None and Ne_src is None
                                     else (Ne if Ne is not None else Ne_src))
                if source_mode_count is None or source_mode_count > len(source_available):
                    raise ValueError("Ne_src exceeds registered vector basis coverage")
                _, source_modes = _prefix_view(basis_specs[vector_source_id],
                                               "mode", source_mode_count)
            if needs_vector_sink:
                sink_available = basis_specs[vector_sink_id].coordinates("mode")
                sink_mode_count = (len(sink_available) if Ne is None and Ne_snk is None
                                   else (Ne if Ne is not None else Ne_snk))
                if sink_mode_count is None or sink_mode_count > len(sink_available):
                    raise ValueError("Ne_snk exceeds registered vector basis coverage")
                _, sink_modes = _prefix_view(basis_specs[vector_sink_id],
                                             "mode", sink_mode_count)

            sink_point_labels = None
            source_point_labels = None
            sink_point_count = source_point_count = None
            if needs_point_sink:
                available = point_specs[point_sink_id].coordinates("point")
                sink_point_count = (len(available) if Np is None and Np_snk is None
                                    else (Np if Np is not None else Np_snk))
                if sink_point_count is None or sink_point_count > len(available):
                    raise ValueError("Np_snk exceeds registered sink point coverage")
                _, sink_point_labels = _prefix_view(point_specs[point_sink_id],
                                                   "point", sink_point_count)
            if needs_point_source:
                available = point_specs[point_source_id].coordinates("point")
                source_point_count = (len(available) if Np is None and Np_src is None
                                      else (Np if Np is not None else Np_src))
                if source_point_count is None or source_point_count > len(available):
                    raise ValueError("Np_src exceeds registered source point coverage")
                _, source_point_labels = _prefix_view(point_specs[point_source_id],
                                                      "point", source_point_count)

            basis_nodes = {}
            basis_identities = [identity for identity in
                                dict.fromkeys((vector_source_id, vector_sink_id))
                                if identity is not None]
            if len(basis_identities) == 1:
                basis_nodes[basis_identities[0]] = "basis"
            else:
                if vector_source_id is not None:
                    basis_nodes.setdefault(vector_source_id, "basis_src")
                if vector_sink_id is not None:
                    basis_nodes.setdefault(vector_sink_id, "basis_snk")

            point_nodes = {}
            point_identities = [identity for identity in
                                dict.fromkeys((point_sink_id, point_source_id))
                                if identity is not None]
            if len(point_identities) == 1:
                point_nodes[point_identities[0]] = "points"
            else:
                if point_sink_id is not None:
                    point_nodes.setdefault(point_sink_id, "points_snk")
                if point_source_id is not None:
                    point_nodes.setdefault(point_source_id, "points_src")

            requests = {"gauge": DatasetRequest(gauge.spec, data_id=gauge_id)}
            for identity, node in basis_nodes.items():
                count = max(value for role_id, value in
                            ((vector_source_id, source_mode_count),
                             (vector_sink_id, sink_mode_count))
                            if role_id == identity and value is not None)
                view, _ = _prefix_view(basis_specs[identity], "mode", count)
                requests[node] = DatasetRequest(view, subset_axes=("mode",), data_id=identity)
            for identity, node in point_nodes.items():
                count = max(value for role_id, value in
                            ((point_sink_id, sink_point_count),
                             (point_source_id, source_point_count))
                            if role_id == identity and value is not None)
                view, _ = _prefix_view(point_specs[identity], "point", count)
                requests[node] = DatasetRequest(view, subset_axes=("point",), data_id=identity)

            source_basis_node = (basis_nodes[vector_source_id]
                                 if vector_source_id is not None else None)
            sink_basis_node = (basis_nodes[vector_sink_id]
                               if vector_sink_id is not None else None)
            sink_points_node = (point_nodes[point_sink_id]
                                if point_sink_id is not None else None)
            source_points_node = (point_nodes[point_source_id]
                                  if point_source_id is not None else None)

            common = dict(
                gauge_id=dict(request="gauge"),
                dirac=science["dirac"],
                solver_tolerance=science["solver_tolerance"],
                temporal_boundary=science["temporal_boundary"],
                dtype=science["dtype"],
                spin_basis=science["spin_basis"],
                normalization=science["normalization"],
                source_selection=source_selection, sink_selection=sink_selection,
            )
            stages = {}
            for product in products:
                name = product.lower()
                if product == "VSV":
                    if (vector_source_id == vector_sink_id and
                            source_modes == sink_modes):
                        template = propagator_dataset_document(
                            configuration, times, product=product,
                            basis_id=dict(request=source_basis_node), modes=source_modes,
                            **common)
                    else:
                        template = propagator_dataset_document(
                            configuration, times, product=product,
                            basis_src_id=dict(request=source_basis_node), modes_src=source_modes,
                            basis_snk_id=dict(request=sink_basis_node), modes_snk=sink_modes,
                            **common)
                    request = DatasetRequest(template,
                        subset_axes=("source_time", "sink_mode", "source_mode"))
                    stages[name] = dict(product=product,
                        basis_src_node=source_basis_node, basis_snk_node=sink_basis_node,
                        basis_src_id=vector_source_id, basis_snk_id=vector_sink_id,
                        mode_src_count=source_mode_count, mode_snk_count=sink_mode_count,
                        same_eigenvector=(
                            vector_source_id == vector_sink_id and
                            len(sink_modes) <= len(source_modes) and
                            source_modes[:len(sink_modes)] == sink_modes))
                elif product == "PSV":
                    template = propagator_dataset_document(
                        configuration, times, product=product,
                        basis_id=dict(request=source_basis_node), modes=source_modes,
                        points_snk_id=dict(request=sink_points_node),
                        point_snk_labels=sink_point_labels, **common)
                    request = DatasetRequest(template,
                        subset_axes=("source_time", "sink_point", "source_mode"))
                    stages[name] = dict(product=product,
                        basis_src_node=source_basis_node,
                        basis_src_id=vector_source_id,
                        point_snk_node=sink_points_node,
                        points_snk_id=point_sink_id,
                        mode_src_count=source_mode_count,
                        point_snk_count=sink_point_count)
                elif product == "VSP":
                    template = propagator_dataset_document(
                        configuration, times, product=product,
                        basis_snk_id=dict(request=sink_basis_node), modes_snk=sink_modes,
                        points_src_id=dict(request=source_points_node),
                        point_src_labels=source_point_labels, **common)
                    request = DatasetRequest(template,
                        subset_axes=("source_time", "sink_mode", "source_point"))
                    stages[name] = dict(product=product,
                        basis_snk_node=sink_basis_node,
                        basis_snk_id=vector_sink_id,
                        point_src_node=source_points_node,
                        points_src_id=point_source_id,
                        mode_snk_count=sink_mode_count,
                        point_src_count=source_point_count,
                        same_eigenvector=False)
                else:
                    template = propagator_dataset_document(
                        configuration, times, product=product,
                        points_snk_id=dict(request=sink_points_node),
                        point_snk_labels=sink_point_labels,
                        points_src_id=dict(request=source_points_node),
                        point_src_labels=source_point_labels, **common)
                    request = DatasetRequest(template,
                        subset_axes=("source_time", "sink_point", "source_point"))
                    stages[name] = dict(product=product,
                        point_snk_node=sink_points_node, point_src_node=source_points_node,
                        points_snk_id=point_sink_id, points_src_id=point_source_id,
                        point_snk_count=sink_point_count,
                        point_src_count=source_point_count)
                requests[name] = request
            input_specs = {identity: opened.spec
                           for identity, opened in input_readers.items()}
            return requests, times, configuration, stages, input_specs
        finally:
            for identity, opened in input_readers.items():
                if identity != gauge_id:
                    opened.close()


def _preview(requests, index, data_ids, source_policy):
    graph = plan_dataset_graph(requests, index.observations(), data_ids=data_ids)
    return apply_source_policy(graph, requests, index, source_policy=source_policy)


def plan_indexed_perambulators(index, gauge_id, *, products=("VSV",),
                               dirac_definition, solver_tolerance,
                               temporal_boundary, basis_id=None, basis_src_id=None,
                               basis_snk_id=None, points_id=None, points_src_id=None,
                               points_snk_id=None, Ne=None, Ne_src=None, Ne_snk=None,
                               Np=None, Np_src=None, Np_snk=None,
                               dtype="complex128", spin_basis="degrand_rossi",
                               normalization="unscaled", data_ids=None,
                               source_policy="allow_unknown",
                               source_selection=None, sink_selection=None):
    """Plan full-time or explicitly selected endpoint datasets from actual inputs."""
    products = _validate_products(products)
    science = _normalize_science_options(dirac_definition, solver_tolerance,
        temporal_boundary, dtype, spin_basis, normalization)
    requests, _, _, _, _ = _indexed_requests(index, gauge_id, products, science,
        basis_id=basis_id, basis_src_id=basis_src_id, basis_snk_id=basis_snk_id,
        points_id=points_id, points_src_id=points_src_id, points_snk_id=points_snk_id,
        Ne=Ne, Ne_src=Ne_src, Ne_snk=Ne_snk, Np=Np, Np_src=Np_src, Np_snk=Np_snk,
        source_selection=source_selection, sink_selection=sink_selection)
    return _preview(requests, index, data_ids, source_policy)


__all__ = [
    "plan_indexed_perambulators",
]
