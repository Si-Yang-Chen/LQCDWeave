"""Explicit binding, coordinate, batch, and input-directory contracts."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
import itertools
import numbers
import operator
from types import MappingProxyType

import numpy as np

from ..dataset import canonical_json
from ..elemental import normalize_elemental_structure


@dataclass(frozen=True)
class DiagramBindings:
    propagator_map: object = None
    vertex_map: object = None
    configuration: object = None
    usedNe: object = None
    usedNp: object = None
    groups_independent: bool | None = None
    elemental_weighted: bool = False

    def __post_init__(self):
        if self.propagator_map is not None:
            if not isinstance(self.propagator_map, Mapping):
                raise TypeError("propagator_map must be a mapping")
            object.__setattr__(self, "propagator_map",
                               MappingProxyType(dict(self.propagator_map)))
        if self.vertex_map is not None:
            if isinstance(self.vertex_map, Mapping):
                object.__setattr__(self, "vertex_map", MappingProxyType(dict(self.vertex_map)))
            elif not callable(self.vertex_map):
                raise TypeError("vertex_map must be a mapping or callable")
        if self.usedNp is not None and (
                isinstance(self.usedNp, (bool, np.bool_))
                or not isinstance(self.usedNp, numbers.Integral)
                or int(self.usedNp) <= 0):
            raise ValueError("usedNp must be a positive integer or None")
        if self.groups_independent is not None and not isinstance(self.groups_independent, bool):
            raise TypeError("groups_independent must be bool or None")
        if type(self.elemental_weighted) is not bool:
            raise TypeError("elemental_weighted must be bool")
        if self.usedNp is not None:
            object.__setattr__(self, "usedNp", int(self.usedNp))


@dataclass(frozen=True)
class TensorAxisMetadata:
    """Reader-owned axes/shape/dtype identity; no tensor is materialized.

    point_diagonal promises zero support outside equal left/right ordered
    point indices. It says nothing about color, spin, or mode diagonality.
    """

    axes: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: object
    data_identity: str | None
    selection_identity: str
    sparse_layout: str = "dense"
    point_shape: tuple[int, int] | None = None
    nnz_upper: int | None = None
    point_pool_ids: Mapping[str, str] = field(default_factory=dict)
    point_pool_ids_by_batch: tuple[Mapping[str, str], ...] | None = None
    axis_bindings: Mapping[str, tuple[str, tuple[object, ...]]] = field(default_factory=dict)
    point_diagonal: bool = False
    structure: str = "general"

    def __post_init__(self):
        axes = tuple(self.axes)
        shape = tuple(self.shape)
        if (len(axes) != len(shape) or len(axes) != len(set(axes))
                or any(not isinstance(axis, str) or not axis for axis in axes)):
            raise ValueError("tensor metadata requires unique named axes matching shape rank")
        if any(isinstance(size, bool) or not isinstance(size, numbers.Integral) or size < 0
               for size in shape):
            raise ValueError("tensor metadata dimensions must be nonnegative integers")
        dtype = np.dtype(self.dtype)
        if type(self.point_diagonal) is not bool:
            raise TypeError("point_diagonal must be bool")
        structure = normalize_elemental_structure(self.structure, self.point_diagonal)
        extents = dict(zip(axes, shape, strict=True))
        point_axes = {"point_left", "point_right"}
        color_axes = {"color_left", "color_right"}
        if structure == "point_diagonal" and (
                not point_axes.issubset(extents)
                or extents["point_left"] != extents["point_right"]):
            raise ValueError("point-diagonal tensors require equal left/right point extents")
        if structure == "diagonal":
            mode_pairs = (("mode_left", "mode_right"),
                          ("sink_mode", "source_mode"))
            mode_diagonal = any(
                left in extents and right in extents and extents[left] == extents[right]
                for left, right in mode_pairs
            )
            point_color_diagonal = (
                point_axes.issubset(extents) and color_axes.issubset(extents)
                and extents["point_left"] == extents["point_right"]
                and extents["color_left"] == extents["color_right"]
            )
            if not mode_diagonal and not point_color_diagonal:
                raise ValueError(
                    "diagonal tensors require equal left/right mode axes or equal point and color axes"
                )
        if dtype.kind not in "biufc":
            raise TypeError("tensor metadata dtype must be numeric")
        if self.data_identity is not None and (not isinstance(self.data_identity, str) or not self.data_identity):
            raise ValueError("data_identity must be a nonempty string or None")
        if not isinstance(self.selection_identity, str) or not self.selection_identity:
            raise ValueError("selection_identity must be a nonempty string")
        if self.sparse_layout not in {"dense", "coo"}:
            raise ValueError("sparse_layout must be dense or coo")
        if self.sparse_layout == "coo":
            if self.point_shape is None or len(self.point_shape) != 2:
                raise ValueError("COO metadata requires two point-axis extents")
            if self.nnz_upper is None or self.nnz_upper < 0:
                raise ValueError("COO metadata requires a nonnegative nnz_upper")
        pools = dict(self.point_pool_ids)
        if set(pools) - {"point_left", "point_right"} or any(
                not isinstance(pool, str) or not pool for pool in pools.values()):
            raise ValueError("point_pool_ids must map point_left/right axes to explicit pool IDs")
        if any(axis not in axes for axis in pools):
            raise ValueError("sampling pool metadata names a point axis absent from this tensor")
        pools_by_batch = self.point_pool_ids_by_batch
        if pools_by_batch is not None:
            if not axes or axes[0] != "coordinate_batch":
                raise ValueError("per-batch pool metadata requires a leading coordinate_batch axis")
            if len(shape) == 0 or len(pools_by_batch) != shape[0]:
                raise ValueError("one point-pool mapping is required for each coordinate_batch entry")
            normalized = []
            for item in pools_by_batch:
                if not isinstance(item, Mapping):
                    raise TypeError("point_pool_ids_by_batch entries must be mappings")
                current = dict(item)
                if set(current) - {"point_left", "point_right"} or any(
                        not isinstance(pool, str) or not pool for pool in current.values()):
                    raise ValueError("per-batch point pools must map point_left/right to nonempty IDs")
                if any(axis not in axes for axis in current):
                    raise ValueError("per-batch pool metadata names an absent point axis")
                normalized.append(MappingProxyType(current))
            for axis, pool in pools.items():
                if any(item.get(axis) != pool for item in normalized):
                    raise ValueError("common point-pool IDs must agree with every batch entry")
            pools_by_batch = tuple(normalized)
        if not isinstance(self.axis_bindings, Mapping):
            raise TypeError("axis_bindings must map named axes to actual dataset IDs and coordinates")
        bindings = {}
        for axis, entry in self.axis_bindings.items():
            if not isinstance(axis, str) or not axis:
                raise TypeError("axis binding names must be nonempty strings")
            if axis not in axes:
                raise ValueError(f"axis binding {axis!r} names an absent tensor axis")
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise TypeError("axis binding values must be (data_id, ordered_coordinates) pairs")
            data_id, coordinates = entry
            if not isinstance(data_id, str) or not data_id:
                raise ValueError("axis binding dataset IDs must be nonempty strings")
            if not isinstance(coordinates, (tuple, list)):
                raise TypeError("axis binding coordinates must be an ordered tuple")
            coordinates = tuple(_normalize_axis_coordinate(value) for value in coordinates)
            if len(coordinates) != shape[axes.index(axis)]:
                raise ValueError(
                    f"axis binding {axis!r} has {len(coordinates)} labels for extent "
                    f"{shape[axes.index(axis)]}"
                )
            bindings[axis] = (data_id, coordinates)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shape", tuple(int(v) for v in shape))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(
            self, "point_diagonal",
            (structure in {"diagonal", "point_diagonal"}
             and point_axes.issubset(extents)
             and extents["point_left"] == extents["point_right"]),
        )
        if self.point_shape is not None:
            object.__setattr__(self, "point_shape", tuple(int(v) for v in self.point_shape))
        object.__setattr__(self, "point_pool_ids", MappingProxyType(pools))
        if pools_by_batch is not None:
            object.__setattr__(self, "point_pool_ids_by_batch", pools_by_batch)
        object.__setattr__(self, "axis_bindings", MappingProxyType(bindings))


def _normalize_axis_coordinate(value):
    """Return one immutable scalar coordinate without erasing its value type."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        canonical_json(value)
        return value
    raise TypeError("axis binding coordinates must be known JSON scalar labels")


def canonical_coordinate_identity(value):
    """Return a type-aware identity for one immutable axis coordinate label."""
    value = _normalize_axis_coordinate(value)
    # Native mode/point indices dominate metadata construction. Their decimal
    # text is exactly JSON's integer encoding; bool and int subclasses retain
    # the original encoder so type distinctions and custom protocols survive.
    if type(value) is int:
        return str(value)
    return canonical_json(value)


def canonical_axis_binding_identity(axis_bindings):
    """Return ordered axis bindings with bool/int and numeric types distinct."""
    if not isinstance(axis_bindings, Mapping):
        raise TypeError("axis_bindings must be a mapping")
    normalized = []
    for axis, entry in axis_bindings.items():
        if not isinstance(axis, str) or not axis:
            raise TypeError("axis binding names must be nonempty strings")
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError("axis binding values must be (data_id, ordered_coordinates) pairs")
        data_id, coordinates = entry
        if not isinstance(data_id, str) or not data_id:
            raise ValueError("axis binding dataset IDs must be nonempty strings")
        if not isinstance(coordinates, (tuple, list)):
            raise TypeError("axis binding coordinates must be an ordered tuple")
        normalized.append((
            axis,
            data_id,
            tuple(canonical_coordinate_identity(value) for value in coordinates),
        ))
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True)
class CoordinatePoint:
    key: str
    values: tuple[tuple[object, int], ...]


@dataclass(frozen=True)
class CoordinateSpec:
    mode: str
    axes: tuple[tuple[object, tuple[int, ...]], ...]
    fixed_axes: tuple[object, ...]
    scan_axes: tuple[object, ...]
    points: tuple[CoordinatePoint, ...]
    scan_shape: tuple[int, ...]

    def __post_init__(self):
        if self.mode not in {"bound", "axes", "pairs"}:
            raise ValueError("coordinate mode must be bound, axes, or pairs")
        if self.mode == "bound" and (self.scan_shape or self.fixed_axes or self.scan_axes):
            raise ValueError("bound coordinates have no scan shape")
        if self.mode == "bound" and (self.axes or len(self.points) != 1):
            raise ValueError("bound coordinates contain exactly one adapter-selected point")
        if len(self.scan_axes) != len(self.scan_shape):
            raise ValueError("one symbolic scan axis is required per scan dimension")
        if self.mode == "axes":
            symbols = tuple(axis for axis, _ in self.axes)
            if (len(symbols) != len(set(symbols))
                    or set(self.fixed_axes) & set(self.scan_axes)
                    or set(self.fixed_axes) | set(self.scan_axes) != set(symbols)):
                raise ValueError("fixed and scanned axes must partition the declared time_axes")
            if any(not values for _, values in self.axes):
                raise ValueError("declared time axes cannot be empty")
            if len(self.points) != int(np.prod(self.scan_shape, dtype=int)):
                raise ValueError("coordinate point count disagrees with scan_shape")
        elif self.mode == "pairs" and (self.axes or self.fixed_axes or self.scan_axes != ("pair",)):
            raise ValueError("time_pairs must describe one ordered pair axis")
        if self.mode == "pairs" and self.scan_shape != (len(self.points),):
            raise ValueError("time_pairs shape must match the requested pair count")
        if any(tuple(key for key, _ in point.values) != tuple(axis for axis, _ in self.axes)
               for point in self.points if self.mode == "axes"):
            raise ValueError("coordinate point axes do not match time_axes")
        if any(type(size) is not int or size < 0 for size in self.scan_shape):
            raise ValueError("coordinate scan dimensions must be nonnegative integers")


@dataclass(frozen=True)
class BatchSpec:
    size: int | str = "auto"
    axis: object = None

    def __post_init__(self):
        if self.size not in ("auto", "full") and (type(self.size) is not int or self.size <= 0):
            raise ValueError("batch must be 'auto', 'full', or a positive integer")
        if self.axis is not None:
            try:
                hash(self.axis)
            except TypeError as exc:
                raise TypeError("batch axis must be a hashable time-axis key") from exc
            if isinstance(self.axis, (bool, np.bool_)):
                raise TypeError("batch axis cannot be bool")


@dataclass(frozen=True)
class CoordinateBatch:
    """One ordered, unpadded chunk of scan coordinates.

    ``batch_size`` is the resolved chunk size capped to this outer-coordinate
    group; ``actual_size`` records the real number of points, including a
    shorter tail. No padding is represented by this object.
    ``point_indices`` always refer to the original ``CoordinateSpec.points``
    order, so grouping by one axis never changes result ordering.
    """

    batch_index: int
    axis: object | None
    outer_values: tuple[tuple[object, int], ...]
    point_indices: tuple[int, ...]
    points: tuple[CoordinatePoint, ...]
    batch_size: int

    def __post_init__(self):
        if type(self.batch_index) is not int or self.batch_index < 0:
            raise ValueError("coordinate batch index must be a nonnegative integer")
        if self.axis is not None:
            try:
                hash(self.axis)
            except TypeError as exc:
                raise TypeError("coordinate batch axis must be hashable or None") from exc
        outer_values = tuple(self.outer_values)
        if any(not isinstance(item, tuple) or len(item) != 2 for item in outer_values):
            raise TypeError("outer_values must contain (axis, integer-coordinate) pairs")
        indices = tuple(self.point_indices)
        points = tuple(self.points)
        if not indices or len(indices) != len(points):
            raise ValueError("coordinate batch needs one point for each original point index")
        if (any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices) or tuple(sorted(indices)) != indices):
            raise ValueError("coordinate batch point indices must be unique, nonnegative, and ordered")
        if any(not isinstance(point, CoordinatePoint) for point in points):
            raise TypeError("coordinate batch points must be CoordinatePoint objects")
        if len({point.key for point in points}) != len(points):
            raise ValueError("coordinate batch point keys must be unique")
        if type(self.batch_size) is not int or self.batch_size <= 0 or len(points) > self.batch_size:
            raise ValueError("batch_size must be positive and at least the actual batch length")
        object.__setattr__(self, "outer_values", outer_values)
        object.__setattr__(self, "point_indices", indices)
        object.__setattr__(self, "points", points)

    @property
    def actual_size(self):
        return len(self.points)

    @property
    def point_keys(self):
        return tuple(point.key for point in self.points)

    @property
    def is_tail(self):
        return self.actual_size < self.batch_size


@dataclass(frozen=True)
class PhysicalSlabBinding:
    """Runtime-only request for a physical slab, shared by logical tensors."""

    identity: Hashable
    describe_slab: Callable[[tuple[int, ...]], TensorAxisMetadata] = field(
        compare=False, repr=False,
    )
    read_slab: Callable[[tuple[int, ...]], object] = field(compare=False, repr=False)
    read_workspace_bytes: Callable[[TensorAxisMetadata], int] | None = field(
        default=None, compare=False, repr=False,
    )
    read_slab_in_background: Callable[[tuple[int, ...]], object] | None = field(
        default=None, compare=False, repr=False,
    )

    def __post_init__(self):
        if self.identity is None:
            raise ValueError("physical slab identity cannot be None")
        try:
            hash(self.identity)
        except TypeError as exc:
            raise TypeError("physical slab identity must be hashable") from exc
        if not callable(self.describe_slab) or not callable(self.read_slab):
            raise TypeError("physical slab binding requires describe_slab and read_slab callbacks")
        if self.read_workspace_bytes is not None and not callable(self.read_workspace_bytes):
            raise TypeError("read_workspace_bytes must be callable or None")
        if self.read_slab_in_background is not None and not callable(self.read_slab_in_background):
            raise TypeError("read_slab_in_background must be callable or None")


@dataclass(frozen=True)
class BatchTensorBinding:
    """Logical tensor materialization separated from its physical slab read.

    One fixed-coordinate adapter returns one lane. A batch adapter combines
    compatible bindings into ordered ``physical_slabs``, ``times`` and
    ``lane_materializers`` tuples. Several logical bindings may reference the
    same PhysicalSlabBinding.identity and therefore share one physical read.
    Callbacks are runtime-only and are never part of a saved contraction plan.
    """

    logical_role: Hashable
    physical_slabs: tuple[PhysicalSlabBinding, ...]
    times: tuple[int, ...]
    lane_materializers: tuple[Callable[[object], object], ...] = field(
        compare=False, repr=False,
    )
    batch_dependent: bool = True
    topology_key: Hashable | None = None
    workspace_host_bytes: int = 0
    workspace_device_bytes: int = 0

    def __post_init__(self):
        if self.logical_role is None:
            raise ValueError("logical_role must have a stable explicit identity")
        for name, value in (("logical_role", self.logical_role),
                            ("topology_key", self.topology_key)):
            if value is not None:
                try:
                    hash(value)
                except TypeError as exc:
                    raise TypeError(f"{name} must be hashable or None") from exc
        physical_slabs = tuple(self.physical_slabs)
        times = tuple(self.times)
        materializers = tuple(self.lane_materializers)
        if not physical_slabs or len(physical_slabs) != len(times) or len(times) != len(materializers):
            raise ValueError("one physical slab, time coordinate, and materializer are required per lane")
        if any(not isinstance(item, PhysicalSlabBinding) for item in physical_slabs):
            raise TypeError("physical_slabs must contain PhysicalSlabBinding objects")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
               or int(value) < 0 for value in times):
            raise ValueError("slab time coordinates must be nonnegative integers")
        if any(not callable(item) for item in materializers):
            raise TypeError("lane_materializers must be callable")
        if type(self.batch_dependent) is not bool:
            raise TypeError("batch_dependent must be bool")
        if any(type(value) is not int or value < 0 for value in
               (self.workspace_host_bytes, self.workspace_device_bytes)):
            raise ValueError("batch materialization workspace must be nonnegative integer bytes")
        object.__setattr__(self, "physical_slabs", physical_slabs)
        object.__setattr__(self, "times", tuple(int(value) for value in times))
        object.__setattr__(self, "lane_materializers", materializers)


def group_coordinate_batches(coordinates, batch, *, resolved_size=None):
    """Group a task's ordered scan points into true-size coordinate batches.

    A named ``BatchSpec.axis`` batches that scan axis and groups by all other
    coordinate values. ``time_pairs`` use their single ``pair`` axis. With no
    selected axis, all scan points form one flattened coordinate axis. For
    ``size='auto'``, ``resolved_size`` may supply a planner-selected chunk size;
    otherwise all points in each group are returned as one batch. No padding is
    introduced, and the original coordinate indices and keys are preserved.
    """
    if not isinstance(coordinates, CoordinateSpec):
        raise TypeError("coordinates must be a CoordinateSpec")
    if not isinstance(batch, BatchSpec):
        raise TypeError("batch must be a BatchSpec")
    if resolved_size is not None and (
            isinstance(resolved_size, (bool, np.bool_))
            or not isinstance(resolved_size, numbers.Integral)
            or int(resolved_size) <= 0):
        raise ValueError("resolved_size must be a positive integer or None")

    if resolved_size is not None:
        if batch.size == "full":
            raise ValueError("resolved_size cannot override batch.size='full'")
        if batch.size != "auto":
            if int(resolved_size) != batch.size:
                raise ValueError("resolved_size cannot override an explicit batch size")

    axis = batch.axis
    if coordinates.mode == "pairs":
        if axis not in (None, "pair", 0):
            raise ValueError("time_pairs can only be batched along the 'pair' axis")
        axis = "pair"
    elif coordinates.mode == "axes":
        if axis is None and len(coordinates.scan_axes) == 1:
            axis = coordinates.scan_axes[0]
        if axis is not None and axis not in coordinates.scan_axes:
            raise ValueError(f"batch axis {axis!r} is not a scanned coordinate axis")
        if (axis is None and len(coordinates.scan_axes) > 1
                and type(batch.size) is int):
            raise ValueError("batch size with multiple scan axes requires an explicit axis")
    elif coordinates.mode != "bound":
        raise ValueError(f"unsupported coordinate mode {coordinates.mode!r}")

    indexed_points = tuple(enumerate(coordinates.points))
    if not indexed_points:
        raise ValueError("CoordinateSpec must contain at least one point")
    point_keys = tuple(point.key for _, point in indexed_points)
    if any(not isinstance(key, str) or not key for key in point_keys) or len(set(point_keys)) != len(point_keys):
        raise ValueError("CoordinateSpec point keys must be unique nonempty strings")

    grouped = {}
    if axis is None or coordinates.mode in {"bound", "pairs"}:
        grouped[()] = indexed_points
    else:
        for entry in indexed_points:
            _, point = entry
            values = dict(point.values)
            if axis not in values:
                raise ValueError(f"coordinate point {point.key!r} omits batch axis {axis!r}")
            outer = tuple((key, value) for key, value in point.values if key != axis)
            grouped.setdefault(outer, []).append(entry)

    largest_group = max(len(entries) for entries in grouped.values())
    if batch.size == "full":
        chunk_size = largest_group
    elif type(batch.size) is int:
        chunk_size = batch.size
    elif resolved_size is not None:
        chunk_size = int(resolved_size)
    else:
        chunk_size = largest_group

    result = []
    for outer_values, entries in grouped.items():
        group_batch_size = min(chunk_size, len(entries))
        for start in range(0, len(entries), group_batch_size):
            chunk = entries[start:start + group_batch_size]
            result.append(CoordinateBatch(
                len(result), axis, outer_values,
                tuple(index for index, _ in chunk),
                tuple(point for _, point in chunk), group_batch_size,
            ))
    return tuple(result)


def _coordinate_time_key(value, *, name):
    # Preserve the existing advanced Diagram/adapter contract: keys are
    # arbitrary hashable symbolic objects, never coerced to strings or ints.
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable") from exc
    return value


def describe_coordinates(*, time_axes=None, time_pairs=None, batch="auto"):
    """Normalize the mutually exclusive full-axis or explicit-pair request."""
    if time_axes is not None and time_pairs is not None:
        raise ValueError("time_axes and time_pairs are mutually exclusive")
    batch_spec = _batch(batch)
    if time_axes is None and time_pairs is None:
        if batch_spec.size != "auto" or batch_spec.axis is not None:
            raise ValueError("batch has no effect when no scan coordinates are supplied")
        return CoordinateSpec("bound", (), (), (), (CoordinatePoint("bound", ()),), ()), batch_spec

    if time_axes is not None:
        if not isinstance(time_axes, Mapping) or not time_axes:
            raise TypeError("time_axes must be a nonempty ordered mapping")
        axes, fixed_axes, scan_axes = [], [], []
        for symbol, values in time_axes.items():
            symbol = _coordinate_time_key(symbol, name="time_axes key")
            if _is_time(values):
                sequence = (_time(values),)
                fixed_axes.append(symbol)
            elif isinstance(values, (bool, np.bool_)):
                raise TypeError("time axis scalar must be an integer")
            elif isinstance(values, (str, bytes)):
                raise TypeError("time axis coordinates must be an integer sequence")
            else:
                try:
                    sequence = tuple(_time(value) for value in values)
                except TypeError as exc:
                    raise TypeError("time axis must be an integer or integer sequence") from exc
                if not sequence:
                    raise ValueError("time axes cannot be empty")
                scan_axes.append(symbol)
            axes.append((symbol, sequence))
        scan_sequences = [sequence for (symbol, sequence) in axes if symbol in scan_axes]
        points = tuple(
            CoordinatePoint(str(index), tuple((axis[0], value) for axis, value in zip(axes, values, strict=True)))
            for index, values in enumerate(_axes_points(axes, fixed_axes, scan_axes))
        )
        coordinates = CoordinateSpec("axes", tuple(axes), tuple(fixed_axes), tuple(scan_axes),
                                     points, tuple(len(values) for values in scan_sequences))
        batch_spec = _bind_batch_axis(batch_spec, coordinates)
        return coordinates, batch_spec

    if isinstance(time_pairs, (str, bytes)):
        raise TypeError("time_pairs must be a sequence of time mappings")
    pairs = tuple(time_pairs)
    if not pairs:
        raise ValueError("time_pairs cannot be empty")
    points = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or not pair:
            raise TypeError("each time_pair must be a nonempty mapping")
        values = tuple((_coordinate_time_key(symbol, name="time_pairs key"), _time(value))
                       for symbol, value in pair.items())
        if index and tuple(key for key, _ in values) != tuple(key for key, _ in points[0].values):
            raise ValueError("all time_pairs must use the same ordered symbolic axes")
        points.append(CoordinatePoint(str(index), values))
    coordinates = CoordinateSpec("pairs", (), (), ("pair",), tuple(points), (len(points),))
    batch_spec = _bind_batch_axis(batch_spec, coordinates)
    return coordinates, batch_spec


def _batch(value):
    if isinstance(value, BatchSpec):
        return value
    if isinstance(value, Mapping):
        unknown = set(value) - {"axis", "size"}
        if unknown:
            raise ValueError(f"unknown batch options: {sorted(unknown)}")
        return BatchSpec(value.get("size", "auto"), value.get("axis"))
    return BatchSpec(value)


def _bind_batch_axis(batch, coordinates):
    if not coordinates.scan_axes:
        if batch.size != "auto" or batch.axis is not None:
            raise ValueError("batch has no effect when all time_axes are fixed")
        return batch
    axis = batch.axis
    if axis is None:
        if batch.size in ("auto", "full"):
            return batch
        if len(coordinates.scan_axes) == 1:
            return BatchSpec(batch.size, coordinates.scan_axes[-1] if batch.size != "auto" else None)
        raise ValueError("batch size with multiple scan axes requires an explicit axis")
    if coordinates.mode == "pairs" and axis in ("pair", 0):
        return BatchSpec(batch.size, "pair")
    if axis not in coordinates.scan_axes:
        # Integer positional axis is also accepted, unless it names a fixed axis.
        if (isinstance(axis, int) and not isinstance(axis, bool)
                and 0 <= axis < len(coordinates.axes)
                and coordinates.axes[axis][0] in coordinates.scan_axes):
            axis = coordinates.axes[axis][0]
        else:
            raise ValueError(f"batch axis {axis!r} is not a scanned time axis")
    return BatchSpec(batch.size, axis)


def _axes_points(axes, fixed_axes, scan_axes):
    scan_values = [values for symbol, values in axes if symbol in scan_axes]
    for scanned in itertools.product(*scan_values):
        iterator = iter(scanned)
        yield tuple(values[0] if symbol in fixed_axes else next(iterator)
                    for symbol, values in axes)


def _is_time(value):
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        operator.index(value)
        return True
    except TypeError:
        return False


@dataclass(frozen=True)
class TermContribution:
    term_id: str
    coefficient: object = 1

    def __post_init__(self):
        if not isinstance(self.term_id, str) or not self.term_id:
            raise ValueError("term contribution needs a nonempty term_id")
        if (not isinstance(self.coefficient, (numbers.Number, np.number))
                and not bool(getattr(self.coefficient, "is_number", False))):
            raise TypeError("term contribution coefficient must be numeric")


@dataclass(frozen=True)
class DiagramInputs:
    """Unique tensor catalog, all Terms, physical requests, and explicit output map."""

    problem: object
    blocks: object
    preparations: object
    loader: object
    coordinates: CoordinateSpec
    batch: BatchSpec
    term_ids: tuple[str, ...]
    diagram_terms: Mapping = field(default_factory=dict)
    diagram_recipes: Mapping = field(default_factory=dict)
    tensor_requests: Mapping = field(default_factory=dict)
    assembly_host_bytes: int = 0
    assembly_device_bytes: int = 0
    coordinate_batch: CoordinateBatch | None = None
    batch_bindings: Mapping[str, BatchTensorBinding] = field(default_factory=dict)
    batch_topology: Hashable | None = None
    batch_output_axes: tuple[int | None, ...] = ()
    task_batches: tuple["DiagramInputs", ...] = field(default=(), repr=False, compare=False)
    batch_candidates: Mapping[int, tuple["DiagramInputs", ...]] = field(
        default_factory=dict, repr=False, compare=False,
    )
    matrix_kernels: tuple = ()
    read_requests: Mapping = field(default_factory=dict)

    def __post_init__(self):
        from .preparation import SourceReadRequest
        if not isinstance(self.read_requests, Mapping) or any(
                key not in self.blocks or not isinstance(value, SourceReadRequest)
                for key, value in self.read_requests.items()):
            raise TypeError("read_requests must map physical block IDs to SourceReadRequest")
        object.__setattr__(self, 'read_requests', MappingProxyType(dict(self.read_requests)))
        if not isinstance(self.coordinates, CoordinateSpec) or not isinstance(self.batch, BatchSpec):
            raise TypeError("DiagramInputs must declare the coordinates and batch it describes")
        term_ids = tuple(self.term_ids)
        if any(not isinstance(value, str) or not value for value in term_ids):
            raise ValueError("term_ids must be nonempty strings")
        if len(term_ids) != len(set(term_ids)):
            raise ValueError("term_ids must be unique")
        if self.problem is not None and len(term_ids) != len(self.problem.terms):
            raise ValueError("one term_id is required for each ContractionTerm")
        from .matrix_kernel import MatrixKernelSpec
        kernels = tuple(self.matrix_kernels)
        if kernels and (len(kernels) != len(term_ids)
                        or any(not isinstance(item, MatrixKernelSpec) for item in kernels)):
            raise ValueError('matrix_kernels must contain one MatrixKernelSpec per Term')
        object.__setattr__(self, 'matrix_kernels', kernels)
        if not isinstance(self.diagram_terms, Mapping):
            raise TypeError("diagram_terms must map unique Diagram objects to contributions")
        for diagram, contributions in self.diagram_terms.items():
            try:
                hash(diagram)
            except TypeError as exc:
                raise TypeError("diagram_terms keys must be hashable Diagram objects") from exc
            if not isinstance(contributions, tuple) or not all(
                isinstance(item, TermContribution) for item in contributions
            ):
                raise TypeError("diagram_terms values must be tuples of TermContribution")
            if any(item.term_id not in term_ids for item in contributions):
                raise ValueError("diagram contribution references an unknown term_id")
        if not isinstance(self.diagram_recipes, Mapping):
            raise TypeError("diagram_recipes must map Diagram objects to OutputNode trees")
        if set(self.diagram_terms) & set(self.diagram_recipes):
            raise ValueError("a Diagram cannot have both diagram_terms and diagram_recipes")
        for diagram in self.diagram_recipes:
            try:
                hash(diagram)
            except TypeError as exc:
                raise TypeError("diagram_recipes keys must be hashable Diagram objects") from exc
        if not isinstance(self.tensor_requests, Mapping):
            raise TypeError("tensor_requests must map logical tensor IDs to typed requests")
        if self.problem is not None and set(self.tensor_requests) - set(self.problem.specs):
            raise ValueError("tensor_requests contains unknown logical tensor IDs")
        if self.coordinate_batch is not None:
            if not isinstance(self.coordinate_batch, CoordinateBatch):
                raise TypeError("coordinate_batch must be a CoordinateBatch or None")
            for index, point in zip(self.coordinate_batch.point_indices,
                                    self.coordinate_batch.points, strict=True):
                if index >= len(self.coordinates.points) or self.coordinates.points[index] != point:
                    raise ValueError("CoordinateBatch points must match their original CoordinateSpec indices")
        if not isinstance(self.batch_bindings, Mapping):
            raise TypeError("batch_bindings must map logical tensor keys to BatchTensorBinding")
        for key, binding in self.batch_bindings.items():
            if not isinstance(key, str) or not key or not isinstance(binding, BatchTensorBinding):
                raise TypeError("batch_bindings entries require tensor keys and BatchTensorBinding values")
            if self.problem is not None and key not in self.problem.specs:
                raise ValueError(f"batch binding references unknown TensorSpec {key!r}")
            if self.coordinate_batch is None:
                if self.coordinates.scan_shape or len(self.coordinates.points) != 1 or len(binding.times) != 1:
                    raise ValueError("batch tensor bindings without CoordinateBatch require one fixed coordinate")
            elif binding.batch_dependent and len(binding.times) != self.coordinate_batch.actual_size:
                raise ValueError("batch-dependent tensor bindings need one materializer per coordinate")
        output_axes = tuple(self.batch_output_axes)
        if output_axes and len(output_axes) != len(term_ids):
            raise ValueError("batch_output_axes must have one entry per ContractionTerm")
        if any(axis is not None and (type(axis) is not int or axis < 0) for axis in output_axes):
            raise ValueError("batch output axes must be nonnegative integers or None")
        if output_axes and any(axis is not None for axis in output_axes) and self.coordinate_batch is None:
            raise ValueError("batch outputs require an explicit CoordinateBatch")
        if self.batch_topology is not None:
            try:
                hash(self.batch_topology)
            except TypeError as exc:
                raise TypeError("batch_topology must be hashable or None") from exc
        task_batches = tuple(self.task_batches)
        batch_candidates = dict(self.batch_candidates)
        if not isinstance(self.batch_candidates, Mapping):
            raise TypeError("batch_candidates must map positive widths to task batch tuples")
        if task_batches:
            if not self.coordinates.scan_shape:
                raise ValueError("task_batches require a scanned CoordinateSpec")
            if self.coordinate_batch is not None or self.batch_bindings or output_axes:
                raise ValueError("task-level inputs cannot also describe one compute batch")
            if any(not isinstance(item, DiagramInputs) or item.task_batches for item in task_batches):
                raise TypeError("task_batches must contain flat DiagramInputs batch entries")
            if any(item.coordinate_batch is None for item in task_batches):
                raise ValueError("every task batch must declare its CoordinateBatch")
            if any(item.coordinates != self.coordinates or item.batch != self.batch
                   or item.term_ids != term_ids for item in task_batches):
                raise ValueError("task batch coordinates, BatchSpec, and term IDs must match the task")
            if any(item.loader is not self.loader for item in task_batches):
                raise ValueError("all task batches must share one loader for full-task slab reuse")
            all_indices = [index for item in task_batches
                           for index in item.coordinate_batch.point_indices]
            if sorted(all_indices) != list(range(len(self.coordinates.points))) or len(set(all_indices)) != len(all_indices):
                raise ValueError("task batches must cover every task coordinate exactly once")
            if any(dict(item.blocks) != dict(self.blocks) for item in task_batches):
                raise ValueError("task batches must share the full-task physical block catalog")
        marked_records = tuple(
            key for key, source in (self.problem.sources.items() if self.problem is not None else ())
            if isinstance(source.metadata, Mapping)
            and source.metadata.get("family") == "p2p-coordinate-batch"
        )
        if marked_records:
            if self.coordinate_batch is not None:
                expected_keys = self.coordinate_batch.point_keys
                for key in marked_records:
                    if tuple(self.problem.sources[key].metadata["coordinate_keys"]) != expected_keys:
                        raise ValueError(
                            f"coordinate-batched sparse source {key!r} does not match CoordinateBatch order"
                        )
            elif task_batches:
                first_keys = task_batches[0].coordinate_batch.point_keys
                for key in marked_records:
                    if tuple(self.problem.sources[key].metadata["coordinate_keys"]) != first_keys:
                        raise ValueError(
                            f"task-level sparse source {key!r} must describe the first CoordinateBatch"
                        )
            else:
                raise ValueError(
                    "coordinate-batched sparse sources require an explicit CoordinateBatch"
                )
        if task_batches:
            for item in task_batches:
                child_records = {
                    key for key, source in item.problem.sources.items()
                    if isinstance(source.metadata, Mapping)
                    and source.metadata.get("family") == "p2p-coordinate-batch"
                }
                if child_records != set(marked_records):
                    raise ValueError("task batches must expose the same sparse bridge record set")
                for key in child_records:
                    if tuple(item.problem.sources[key].metadata["coordinate_keys"]) != item.coordinate_batch.point_keys:
                        raise ValueError(
                            f"task batch sparse source {key!r} does not match its CoordinateBatch order"
                        )
        if batch_candidates:
            if self.batch.size != "auto" or not self.coordinates.scan_shape:
                raise ValueError("batch_candidates are valid only for scanned batch='auto' tasks")
            if any(type(width) is not int or width <= 0 for width in batch_candidates):
                raise ValueError("batch candidate widths must be positive integers")
            if not task_batches or not any(
                    len(candidate) == len(task_batches)
                    and all(left is right for left, right in zip(candidate, task_batches, strict=True))
                    for candidate in batch_candidates.values()):
                raise ValueError("task_batches must be one of the declared batch_candidates")
            for width, candidate in batch_candidates.items():
                candidate = tuple(candidate)
                if not candidate or any(not isinstance(item, DiagramInputs)
                                        or item.task_batches or item.coordinate_batch is None
                                        for item in candidate):
                    raise TypeError(f"batch candidate {width} must contain flat compute batches")
                if any(item.coordinates != self.coordinates or item.batch != self.batch
                       or item.term_ids != term_ids for item in candidate):
                    raise ValueError(f"batch candidate {width} changes coordinates, BatchSpec, or terms")
                if any(item.loader is not candidate[0].loader for item in candidate):
                    raise ValueError(f"batch candidate {width} must share one task-scoped loader")
                candidate_indices = [index for item in candidate
                                     for index in item.coordinate_batch.point_indices]
                if (sorted(candidate_indices) != list(range(len(self.coordinates.points)))
                        or len(set(candidate_indices)) != len(candidate_indices)):
                    raise ValueError(f"batch candidate {width} does not cover the task coordinates once")
                if any(dict(item.blocks) != dict(self.blocks) for item in candidate):
                    raise ValueError(f"batch candidate {width} differs from the full-task block catalog")
                batch_candidates[width] = candidate
        object.__setattr__(self, "task_batches", task_batches)
        object.__setattr__(self, "batch_candidates", MappingProxyType(batch_candidates))
        object.__setattr__(self, "term_ids", term_ids)
        for name in ("blocks", "preparations", "diagram_terms", "diagram_recipes", "tensor_requests"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{name} must be a mapping")
            copied = dict(mapping)
            if name == "diagram_terms":
                copied = {key: tuple(value) for key, value in copied.items()}
            object.__setattr__(self, name, MappingProxyType(copied))
        object.__setattr__(self, "batch_bindings", MappingProxyType(dict(self.batch_bindings)))
        object.__setattr__(self, "batch_output_axes", output_axes)
        if any(type(n) is not int or n < 0 for n in
               (self.assembly_host_bytes, self.assembly_device_bytes)):
            raise ValueError("assembly workspace must be nonnegative integer bytes")


class DiagramData:
    """Structural adapter interface; describe must not call numeric getters."""

    def describe(self, correlator, *, bindings: DiagramBindings,
                 coordinates: CoordinateSpec, batch: BatchSpec) -> DiagramInputs:
        raise NotImplementedError


def _time(value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("time coordinate must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError("time coordinate must be an integer") from exc


__all__ = [
    "BatchSpec", "BatchTensorBinding", "CoordinateBatch", "CoordinatePoint", "CoordinateSpec",
    "DiagramBindings", "TensorAxisMetadata", "DiagramData", "DiagramInputs",
    "PhysicalSlabBinding", "TermContribution", "describe_coordinates",
    "group_coordinate_batches", "canonical_coordinate_identity",
    "canonical_axis_binding_identity",
]
