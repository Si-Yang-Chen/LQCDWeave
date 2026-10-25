"""Pure specifications for realized point/eigenvector overlap maps."""

from collections.abc import Mapping

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import (TimeSelectionRecord, normalize_time_selection, dataset_physical_times, bind_time_selections)
from lqcdweave.contracts.elemental import integer


def _reference(value, name):
    if isinstance(value, str) and value:
        return value
    role = {"basis_id": "basis", "points_id": "points"}[name]
    if (isinstance(value, Mapping) and set(value) == {"request"}
            and value["request"] == role):
        return dict(value)
    raise ValueError(f"{name} must be an actual ID or the exact {role!r} role request")


def _prefix(spec, axis, count):
    labels = spec.coordinates(axis)
    if any(type(label) is not int or label < 0 for label in labels):
        raise ValueError(f"{axis} labels must be nonnegative ordered integers")
    size = len(labels) if count is None else integer(count, axis + " count", 1)
    if size > len(labels):
        raise ValueError(f"{axis} count exceeds registered coverage")
    selected = labels[:size]
    document = spec.document
    for item in document["axes"]:
        if item["name"] == axis:
            item["coordinates"] = selected
    if axis in document["bases"]:
        document["bases"][axis]["indices"] = selected
    return DatasetSpec(document)


def overlap_dataset_document(basis, points, *, basis_id, points_id,
                              usedNe=None, usedNp=None, selection=None):
    """Return selected input specs, an output template and physical times.

    Only metadata is checked here. XYZ payload bounds/uniqueness and finite
    selected vectors are checked by the workflow before output creation.
    The map is eta-dagger-xi; it need not be an orthogonal projector. Logical
    references must be resolved by DatasetRequest before publishing a spec.
    """
    if not isinstance(basis, DatasetSpec) or not isinstance(points, DatasetSpec):
        raise TypeError("basis and points must be DatasetSpec objects")
    if basis.document["kind"] not in ("explicit_basis", "laplace_basis"):
        raise ValueError("overlap requires an explicit_basis or laplace_basis dataset")
    cfg = basis.document["configuration"]
    if (cfg != points.document["configuration"]
            or any(cfg[key] is None for key in ("ensemble", "configuration"))):
        raise ValueError("matching known configuration and lattice identities required")
    lx, ly, lz, lt = cfg["lattice"]
    if (basis.axis_names != ("time", "mode", "z", "y", "x", "color")
            or basis.document["dtype"] not in ("complex64", "complex128")):
        raise ValueError("canonical complex vector basis axes required")
    for axis, extent in (("z", lz), ("y", ly), ("x", lx), ("color", 3)):
        if basis.coordinates(axis) != list(range(extent)):
            raise ValueError("canonical spatial and color coordinates required")
    point_document = points.document
    if (point_document["kind"] != "spatial_points"
            or points.axis_names != ("time", "point", "xyz")
            or point_document["dtype"] not in ("int32", "int64")
            or points.coordinates("xyz") != ["x", "y", "z"]
            or point_document["conventions"].get("representation") != "coordinate_delta"
            or point_document["conventions"].get("normalization") != "unit_delta"):
        raise ValueError("canonical integer coordinate_delta/unit_delta spatial_points required")
    required = TimeSelectionRecord(lt, tuple(range(lt))) if selection is None else normalize_time_selection(selection, Lt=lt)
    for spec in (basis, points):
        times = dataset_physical_times(spec)
        if selection is None and sorted(times) != list(range(lt)):
            raise ValueError("complete physical-time input coverage required")
        TimeSelectionRecord(lt, times).require_times(required.times)
    basis_view = _prefix(basis, "mode", usedNe)
    points_view = _prefix(points, "point", usedNp)
    vector_ref = _reference(basis_id, "basis_id")
    point_ref = _reference(points_id, "points_id")
    modes = basis_view.coordinates("mode")
    labels = points_view.coordinates("point")
    document = dict(schema=1, kind="overlap_vp", configuration=cfg,
        axes=[dict(name="time", coordinates=list(required.times)),
              dict(name="mode", coordinates=modes),
              dict(name="point", coordinates=labels),
              dict(name="color", coordinates=list(range(3)))],
        dtype=basis.document["dtype"],
        bases=dict(mode=dict(data_id=vector_ref, indices=modes),
                   point=dict(data_id=point_ref, indices=labels)),
        definition=dict(basis_id=vector_ref, points_id=point_ref,
                        input_views=dict(basis=basis_view.document,
                                         points=points_view.document)),
        conventions=dict(representation="eta_dagger_xi", normalization="unscaled",
                         vector_conjugation="none"),
        parents=[vector_ref, point_ref])
    if selection is not None:
        document = bind_time_selections(document, {"time": selection})
    return basis_view, points_view, document, required.times


__all__ = ["overlap_dataset_document"]
