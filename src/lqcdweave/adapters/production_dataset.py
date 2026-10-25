"""Borrow registered time-major fields through production's load/key API."""

from numbers import Integral
import numpy as np

from lqcdweave.contracts.time_selection import dataset_time_selection


def _expand_selection(key, shape):
    """Normalize NumPy-style basic and integer-array indexing for lazy reads."""
    key = key if isinstance(key, tuple) else (key,)
    if sum(item is Ellipsis for item in key) > 1:
        raise IndexError("an index can only have a single ellipsis")
    consumed = sum(item is not None and item is not Ellipsis for item in key)
    if consumed > len(shape):
        raise IndexError("too many indices for dataset field")
    if any(item is Ellipsis for item in key):
        index = next(i for i, item in enumerate(key) if item is Ellipsis)
        key = (*key[:index], *(slice(None) for _ in range(len(shape) - consumed)), *key[index + 1:])
    else:
        key = (*key, *(slice(None) for _ in range(len(shape) - consumed)))

    selections = []
    transformed = []
    axis = 0
    for item in key:
        if item is None:
            transformed.append(None)
            continue
        extent = shape[axis]
        axis += 1
        if isinstance(item, slice):
            positions = tuple(range(*item.indices(extent)))
            selections.append((positions, "slice"))
            transformed.append(slice(None))
            continue
        if isinstance(item, (Integral, np.integer)) and not isinstance(item, (bool, np.bool_)):
            position = int(item)
            if position < 0:
                position += extent
            if not 0 <= position < extent:
                raise IndexError("dataset field index is outside its axis")
            selections.append(((position,), "scalar"))
            transformed.append(0)
            continue
        try:
            values = np.asarray(item)
        except Exception as exc:
            raise TypeError("lazy dataset fields accept slices or integer index arrays") from exc
        if values.ndim != 1:
            raise IndexError("lazy dataset field index arrays must be one-dimensional")
        if values.dtype.kind == "b":
            if len(values) != extent:
                raise IndexError("boolean index length differs from its dataset axis")
            requested = np.flatnonzero(values).astype(np.int64, copy=False)
        elif values.dtype.kind in "iu":
            requested = values.astype(np.int64, copy=True)
            requested[requested < 0] += extent
            if np.any((requested < 0) | (requested >= extent)):
                raise IndexError("dataset field index is outside its axis")
        else:
            raise TypeError("lazy dataset field index arrays must contain integers or booleans")
        positions = []
        local = {}
        mapped = []
        for raw in requested:
            position = int(raw)
            if position not in local:
                local[position] = len(positions)
                positions.append(position)
            mapped.append(local[position])
        selections.append((tuple(positions), "array"))
        transformed.append(np.asarray(mapped, dtype=np.intp))
    if axis != len(shape):
        raise IndexError("lazy dataset field index does not cover its axes")
    return tuple(selections), tuple(transformed)


def _read_indexed_dataset(dataset, key, logical_axes):
    """Read only selected DatasetReader coordinates, preserving axis indexing."""
    spec = dataset.spec
    logical_axes = tuple(logical_axes)
    if set(logical_axes) != set(spec.axis_names) or len(logical_axes) != len(spec.axis_names):
        raise ValueError("lazy field axes must match the registered dataset axes")
    if spec.axis_names[0] != "time" or "time" not in logical_axes:
        raise ValueError("lazy production fields require physical time as the stored first axis")
    shape = tuple(len(spec.coordinates(axis)) for axis in logical_axes)
    selections, transformed = _expand_selection(key, shape)
    time_axis = logical_axes.index("time")
    time_positions = selections[time_axis][0]
    selected_axes = {
        name: positions
        for name, (positions, _kind) in zip(logical_axes, selections, strict=True)
        if name != "time"
    }
    compact_shape = tuple(len(positions) for positions, _ in selections)
    dtype = np.dtype(spec.document["dtype"])
    if not all(compact_shape):
        compact = np.empty(compact_shape, dtype=dtype)
    else:
        # Field handles index physical time labels in ascending order, even
        # when the dataset stores its first axis in another declared order.
        requested_times = tuple(sorted(spec.coordinates("time")))
        rows = []
        for position in time_positions:
            read_selection = {}
            for axis, positions in selected_axes.items():
                labels = spec.coordinates(axis)
                if tuple(positions) != tuple(range(len(labels))):
                    read_selection[axis] = [labels[index] for index in positions]
            value = dataset.read(time=requested_times[position], select=read_selection)
            stored_axes = tuple(axis for axis in spec.axis_names if axis != "time")
            requested_axes = tuple(axis for axis in logical_axes if axis != "time")
            if stored_axes != requested_axes:
                value = np.transpose(value, tuple(stored_axes.index(axis) for axis in requested_axes))
            rows.append(value)
        if time_positions:
            compact = np.stack(rows, axis=time_axis)
        else:
            compact = np.empty(compact_shape, dtype=dtype)
    return compact[transformed]


class DatasetFieldHandle:
    """A non-owning handle; its DatasetReader must outlive every calculation.

    The first axis must be complete physical time. Spatial/mode semantics are
    validated by the workflow's planner. Integer indexing reads a physical
    time, even when storage is reordered. Slices materialize selected times.
    No persistent identity is inferred from shape or filenames.
    """

    def __init__(self, dataset):
        spec = dataset.spec
        lt = spec.document["configuration"]["lattice"][3]
        times = spec.coordinates("time")
        self.time_selection = dataset_time_selection(spec)
        if spec.document["kind"] == "gauge" and self.time_selection is not None:
            raise ValueError("gauge fields require full physical time axes, not compact slot mappings")
        if (spec.axis_names[0] != "time" or any(type(t) is not int for t in times)
                or (self.time_selection is None and sorted(times) != list(range(lt)))):
            raise ValueError("complete first-axis physical time coverage required")
        self.dataset = dataset
        self.Ne = len(spec.coordinates("mode")) if "mode" in spec.axis_names else None
        self.basis_id = dataset.data_id if self.Ne is not None else None
        self.mode_labels = tuple(spec.coordinates('mode')) if self.Ne is not None else ()
        conventions = spec.document["conventions"]
        # Representation is known even when an imported D dataset has no
        # stochastic provenance. Unknown provenance must not turn D into V.
        self.vector_kind = 'D' if conventions.get('representation') == 'dilution' else 'V'
        self.dilution_source = None
        if conventions.get('representation') == 'dilution':
            from lqcdweave.contracts.dilution import DilutionBasis
            source_document = conventions.get('dilution_source')
            if source_document is not None:
                self.dilution_source = DilutionBasis.from_document(source_document)
                if self.dilution_source.data_id != dataset.data_id:
                    raise ValueError('D field provenance differs from its actual dataset ID')
            else:
                description = getattr(dataset, 'provenance', {}).get('generator')
                if isinstance(description, dict) and description.get('parent_basis_id'):
                    self.dilution_source = DilutionBasis.from_description(
                        description, data_id=dataset.data_id)
            if self.dilution_source is not None and any(
                    type(index) is not int or not 0 <= index < len(self.dilution_source.vector_labels)
                    for index in self.mode_labels):
                raise ValueError('D field mode labels differ from its declared flattened vector order')
        self.basis_orthogonal = (self.Ne is not None and
            (conventions.get("normalization") == "orthonormal"
             or conventions.get("basis_orthogonal") is True))
        self.basis_orthonormal = (self.Ne is not None and
                                conventions.get("normalization") == "orthonormal")
        self.gauge_id = dataset.data_id if spec.document["kind"] == "gauge" else None

    def load(self, key):
        if key != self.dataset.spec.document["configuration"]["configuration"]:
            raise ValueError("configuration key differs from registered field")
        return _TimeField(self.dataset)

    def read_modes(self, key, time, start, stop):
        """Borrow one physical-time mode slice until the reader is closed."""
        if self.Ne is None or self.dataset.spec.axis_names != (
                "time", "mode", "z", "y", "x", "color"):
            raise ValueError("registered field is not a canonical mode basis")
        if key != self.dataset.spec.document["configuration"]["configuration"]:
            raise ValueError("configuration key differs from registered field")
        if self.time_selection is not None:
            time = self.time_selection.slot(time)
        if (type(time) is not int or not 0 <= time < self.dataset.spec.shape[0]
                or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.Ne):
            raise ValueError("invalid physical time or mode block")
        return self.dataset.read(time=time, select={"mode": slice(start, stop)})


class DatasetPointHandle:
    """Adapt explicit registered time-major XYZ points to the existing producer.

    The dataset/view is borrowed and must outlive load. load returns a lazy
    time-first array view; indexing it reads only the selected time and point
    coordinates. Numerical production validates consumed rows. This coordinate
    view does not establish a bound on the producer's other inputs.
    Intrinsic point representation is checked here; comparing configuration
    identity with the gauge input remains the caller's planning responsibility.
    """

    def __init__(self, dataset):
        spec = dataset.spec
        document = spec.document
        lt = document["configuration"]["lattice"][3]
        self.time_selection = dataset_time_selection(spec)
        if (document["kind"] != "spatial_points" or
                document["conventions"].get("representation") != "coordinate_delta" or
                document["conventions"].get("normalization") != "unit_delta" or
                spec.axis_names != ("time", "point", "xyz") or
                spec.coordinates("xyz") != ["x", "y", "z"] or
                any(type(time) is not int for time in spec.coordinates("time")) or
                (self.time_selection is None and sorted(spec.coordinates("time")) != list(range(lt))) or
                any(type(label) is not int or label < 0 for label in spec.coordinates("point")) or
                document["dtype"] not in ("int32", "int64")):
            raise ValueError("canonical spatial_points coordinate_delta/unit_delta integer XYZ table required")
        self.dataset = dataset
        self.Np = len(spec.coordinates("point"))
        self.basis_id = dataset.data_id

    def load(self, key):
        if key != self.dataset.spec.document["configuration"]["configuration"]:
            raise ValueError("configuration key differs from registered points")
        return _IndexedDatasetField(self.dataset, ("time", "point", "xyz"))


    def read_points(self, key, time, start, stop):
        """Read a physical endpoint through its explicit compact storage map."""
        if key != self.dataset.spec.document["configuration"]["configuration"]:
            raise ValueError("configuration key differs from registered points")
        if self.time_selection is not None:
            time = self.time_selection.slot(time)
        if (type(time) is not int or not 0 <= time < self.dataset.spec.shape[0]
                or type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.Np):
            raise ValueError("invalid physical time or point block")
        return self.dataset.read(time=time, select={"point": slice(start, stop)})


class _TimeField:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.spec.shape
        self.dtype = np.dtype(dataset.spec.document["dtype"])

    def __getitem__(self, key):
        return _read_indexed_dataset(self.dataset, key, self.dataset.spec.axis_names)

    def __array__(self, dtype=None, copy=None):
        value = self[...]
        if dtype is not None:
            value = value.astype(dtype, copy=False)
        if copy:
            value = np.array(value, copy=True)
        return value


class _IndexedDatasetField(_TimeField):
    def __init__(self, dataset, axes):
        self.dataset = dataset
        self.shape = tuple(len(dataset.spec.coordinates(axis)) for axis in axes)
        self.dtype = np.dtype(dataset.spec.document["dtype"])
        self._axes = tuple(axes)

    def __getitem__(self, key):
        return _read_indexed_dataset(self.dataset, key, self._axes)
