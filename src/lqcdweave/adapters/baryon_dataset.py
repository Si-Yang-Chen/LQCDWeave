"""Read published three-vector baryon elementals through borrowed dataset views.

This adapter translates the explicit ``elemental_baryon_vvv`` storage schema
to the existing Baryon reader protocol. It never owns or closes the supplied
reader, and it does not reorder or conjugate its three mode axes.
"""

from numbers import Integral
from types import MappingProxyType

import numpy as np

from lqcdweave.contracts.time_selection import dataset_time_selection, dataset_physical_times

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.elemental import BaryonComponent, GaugePath
from lqcdweave.contracts.elemental_catalog import stored_catalog_components
from .selection import resolve_selections


_AXES = ("time", "component", "momentum", "mode0", "mode1", "mode2")
_PRODUCT_ENCODINGS = {
    "raw_paths": "ordered_path_triples_v1",
    "projected": "projected_row_ids_v1",
    "distance": "distance_triples_v1",
}
_ROLE_NAMES = ("gauge", "basis_leg0", "basis_leg1", "basis_leg2")


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty actual identity")
    return value


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _time_sequence(values, extent):
    if isinstance(values, (bool, np.bool_)):
        raise TypeError("baryon times must be integer coordinates")
    if isinstance(values, Integral):
        values = (values,)
    if isinstance(values, (str, bytes)):
        raise TypeError("baryon times must be integer coordinates")
    try:
        times = tuple(values)
    except TypeError as exc:
        raise TypeError("baryon times must be an integer or iterable of coordinates") from exc
    if not times or any(isinstance(time, (bool, np.bool_))
                        or not isinstance(time, Integral)
                        or not 0 <= int(time) < extent for time in times):
        raise ValueError("baryon slab times must be nonempty physical coordinates")
    return tuple(int(time) for time in times)


def _path_triple(value, name):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three ordered paths")
    paths = []
    for leg, steps in enumerate(value):
        if not isinstance(steps, list):
            raise ValueError(f"{name}[{leg}] must be an ordered step list")
        parsed = tuple(_integer(step, f"{name}[{leg}] step") for step in steps)
        if any(step > 5 for step in parsed):
            raise ValueError("baryon spatial path directions must lie in 0..5")
        paths.append(GaugePath(parsed))
    return tuple(paths)


def _decode_component(value, product):
    if not isinstance(value, dict):
        raise ValueError("baryon component coordinates must be tagged mappings")
    if product == "raw_paths":
        if set(value) != {"encoding", "paths"} or value["encoding"] != "ordered_path_triple_v1":
            raise ValueError("raw baryon components require ordered_path_triple_v1 tags")
        return BaryonComponent(_path_triple(value["paths"], "component paths"))
    if product == "projected":
        if set(value) != {"encoding", "id"} or value["encoding"] != "projected_row_id_v1":
            raise ValueError("projected baryon components require projected_row_id_v1 tags")
        return BaryonComponent(("projected", _integer(value["id"], "projected row ID")))
    if product == "distance":
        if set(value) != {"encoding", "legs"} or value["encoding"] != "distance_triple_v1":
            raise ValueError("distance baryon components require distance_triple_v1 tags")
        raw = value["legs"]
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("distance component must contain three ordered leg lengths")
        return BaryonComponent(("distance", tuple(
            _integer(item, f"distance leg {index}") for index, item in enumerate(raw)
        )))
    raise ValueError(f"unsupported baryon product {product!r}")


def _validate_projected_rows(definition, components):
    rows = definition.get("projected_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("projected baryon datasets require nonempty projected_rows")
    expected_ids = list(range(len(rows)))
    row_ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "terms"}:
            raise ValueError("each projected row requires exactly id and terms")
        row_id = _integer(row["id"], "projected row ID")
        row_ids.append(row_id)
        terms = row["terms"]
        if not isinstance(terms, list) or not terms:
            raise ValueError("projected rows require a nonempty ordered term list")
        for term in terms:
            if not isinstance(term, dict) or set(term) != {"paths", "coefficient"}:
                raise ValueError("projected terms require exactly paths and coefficient")
            _path_triple(term["paths"], "projected term paths")
            coefficient = term["coefficient"]
            if (not isinstance(coefficient, list) or len(coefficient) != 2 or
                    any(isinstance(part, bool) or not isinstance(part, (int, float)) or
                        not np.isfinite(part) for part in coefficient)):
                raise ValueError("projected coefficients must be finite [real, imag] pairs")
    if row_ids != expected_ids:
        raise ValueError("projected row IDs must be consecutive from zero")
    if tuple(component.key for component in components) != tuple(
            ("projected", row_id) for row_id in expected_ids):
        raise ValueError("projected component coordinates must preserve their declared row order")


def _validate_view_documents(spec, definition):
    roles = definition.get("input_roles")
    views = definition.get("input_views")
    if not isinstance(roles, dict) or set(roles) != set(_ROLE_NAMES):
        raise ValueError("baryon dataset must bind gauge and all three ordered basis roles")
    if not isinstance(views, dict) or set(views) != set(_ROLE_NAMES):
        raise ValueError("baryon dataset must preserve all four exact input views")
    for role in _ROLE_NAMES:
        _nonempty_string(roles[role], f"input role {role}")
        view = DatasetSpec(views[role])
        if view.document["configuration"] != spec.document["configuration"]:
            raise ValueError(f"baryon {role} view configuration differs from its output")
        if role == "gauge":
            if view.document["kind"] != "gauge" or roles[role] != definition["gauge_id"]:
                raise ValueError("baryon gauge role must preserve its actual gauge view and ID")
            axes = ("time", "z", "y", "x", "direction", "color_row", "color_column")
            lx, ly, lz, lt = view.document["configuration"]["lattice"]
            extents = dict(time=lt, z=lz, y=ly, x=lx, direction=4,
                           color_row=3, color_column=3)
            _validate_input_field_layout(view, axes, extents, "gauge")
            continue
        if view.document["kind"] not in ("explicit_basis", "laplace_basis"):
            raise ValueError(f"baryon {role} view must be an explicit or Laplace basis")
        axes = ("time", "mode", "z", "y", "x", "color")
        lx, ly, lz, lt = view.document["configuration"]["lattice"]
        selection = dataset_time_selection(view, 'time')
        extents = dict(time=lt if selection is None else selection.Nt, z=lz, y=ly, x=lx, color=3)
        if not set(dataset_physical_times(spec, 'time')).issubset(dataset_physical_times(view, 'time')):
            raise ValueError('baryon basis lacks selected output physical times')
        _validate_input_field_layout(view, axes, extents, role)


def _validate_input_field_layout(spec, axes, extents, role):
    """Retain the producer's canonical field layout in nested role views.

    Physical time is addressed by coordinate, so full time permutations are
    valid. Spatial and color coordinates keep the producer's canonical order.
    """
    if (spec.axis_names != axes or
            spec.document["dtype"] not in ("complex64", "complex128")):
        raise ValueError(f"baryon {role} view must preserve canonical field axes and dtype")
    for name in axes:
        coordinates = spec.coordinates(name)
        if any(type(value) is not int or value < 0 for value in coordinates):
            raise ValueError(f"baryon {role} field coordinates must be nonnegative integers")
        if name == "mode":
            continue
        expected = list(range(extents[name]))
        observed = sorted(coordinates) if name == "time" else coordinates
        if observed != expected:
            raise ValueError(
                f"baryon {role} view must cover full time and canonical spatial/color coordinates"
            )


def _view_specification(dataset):
    spec = getattr(dataset, "spec", None)
    if not isinstance(spec, DatasetSpec):
        raise TypeError("DatasetBaryonElemental requires a registered DatasetReader or DatasetView")
    owner = getattr(dataset, "reader", None)
    stored_spec = getattr(owner, "spec", spec)
    if not isinstance(stored_spec, DatasetSpec):
        raise TypeError("DatasetView must retain its registered parent DatasetSpec")
    stored = stored_spec.document
    view = spec.document
    for name in ("schema", "kind", "configuration", "dtype", "definition", "conventions", "parents"):
        if canonical_json(view[name]) != canonical_json(stored[name]):
            raise ValueError("baryon DatasetView changed its published scientific definition")
    if spec.axis_names != stored_spec.axis_names:
        raise ValueError("baryon DatasetView changed its published axis names or order")
    for axis in ("time", "component", "momentum"):
        if tuple(spec.coordinates(axis)) != tuple(stored_spec.coordinates(axis)):
            raise ValueError(f"baryon DatasetView changed the published {axis} coordinates")
    for index in range(3):
        axis = f"mode{index}"
        selected = tuple(spec.coordinates(axis))
        available = tuple(stored_spec.coordinates(axis))
        if (not selected or len(selected) > len(available) or
                tuple(canonical_json(value) for value in selected) !=
                tuple(canonical_json(value) for value in available[:len(selected)])):
            raise ValueError(f"baryon DatasetView must retain an ordered {axis} prefix")
        view_binding = view["bases"].get(axis)
        stored_binding = stored["bases"].get(axis)
        if (not isinstance(view_binding, dict) or not isinstance(stored_binding, dict)
                or view_binding.get("data_id") != stored_binding.get("data_id")
                or view_binding.get("indices") != list(selected)):
            raise ValueError(f"baryon DatasetView changed its actual {axis} basis binding")
    return spec, stored_spec


def _parse_dataset(dataset):
    spec, stored_spec = _view_specification(dataset)
    document = stored_spec.document
    if document["kind"] != "elemental_baryon_vvv":
        raise ValueError("baryon dataset requires kind 'elemental_baryon_vvv'")
    if spec.axis_names != _AXES or document["dtype"] != "complex128":
        raise ValueError("baryon dataset axes and dtype must be canonical complex128 vvv")
    configuration = document["configuration"]
    selection = dataset_time_selection(spec, 'time')
    lt = configuration["lattice"][3] if selection is None else selection.Nt
    if spec.coordinates("time") != list(range(lt)):
        raise ValueError("baryon dataset must cover every physical time in order")
    definition = document["definition"]
    product = definition.get("product")
    encoding = _PRODUCT_ENCODINGS.get(product)
    if encoding is None or definition.get("component_encoding") != encoding:
        raise ValueError("baryon product and component encoding are missing or inconsistent")
    parents = document["parents"]
    if not parents or _nonempty_string(definition.get("gauge_id"), "gauge_id") != parents[0]:
        raise ValueError("baryon parent list must begin with its actual gauge ID")
    if definition.get("gauge_purpose") != "elemental_path_transport":
        raise ValueError("baryon gauge purpose must be elemental_path_transport")
    conventions = document["conventions"]
    if (conventions.get("normalization") != "unscaled" or
            conventions.get("phase") not in ("anchor", "none") or
            conventions.get("path_boundary") not in ("periodic", "open") or
            conventions.get("color_contraction") != "epsilon" or
            conventions.get("vector_conjugation") != "none"):
        raise ValueError("baryon dataset conventions do not match the published vvv contract")
    _validate_view_documents(spec, definition)
    if set(document["bases"]) != {"mode0", "mode1", "mode2"}:
        raise ValueError("baryon dataset must bind each of its three ordered mode axes")
    expected_parents = [definition["gauge_id"]]
    for index in range(3):
        axis = f"mode{index}"
        binding = document["bases"][axis]
        actual_id = _nonempty_string(binding["data_id"], f"{axis} actual basis ID")
        if definition["input_roles"][f"basis_leg{index}"] != actual_id:
            raise ValueError(f"{axis} basis does not match its actual input role")
        expected_parents.extend([actual_id] if actual_id not in expected_parents else [])
    if document["parents"] != expected_parents:
        raise ValueError("baryon parents must preserve gauge and first-use ordered basis identities")
    for index in range(3):
        view = DatasetSpec(definition["input_views"][f"basis_leg{index}"])
        binding = document["bases"][f"mode{index}"]
        if view.coordinates("mode") != binding["indices"]:
            raise ValueError(f"mode{index} output labels differ from its exact basis-leg view")

    raw_coordinates = tuple(spec.coordinates("component"))
    components = tuple(_decode_component(value, product) for value in raw_coordinates)
    if product == "projected":
        _validate_projected_rows(definition, components)
        if "catalog" in definition:
            components = stored_catalog_components(
                definition["catalog"], definition["projected_rows"], rank=3)
    elif product == "distance":
        if definition.get("distance_rule") != "independent_six_straight_paths_per_nonzero_leg":
            raise ValueError("distance baryon datasets require the declared six-direction rule")
        distances = tuple(component.key[1] for component in components)
        if len(set(distances)) != len(distances):
            raise ValueError("distance triples must be unique")
    else:
        if "projected_rows" in definition or "distance_rule" in definition:
            raise ValueError("raw path datasets cannot carry projected or distance recipes")

    raw_momenta = tuple(spec.coordinates("momentum"))
    momenta = []
    for item in raw_momenta:
        if (not isinstance(item, list) or len(item) != 3 or
                any(type(value) is not int for value in item)):
            raise ValueError("baryon momentum labels must be ordered integer XYZ triples")
        momenta.append(tuple(item))
    if len(set(momenta)) != len(momenta):
        raise ValueError("baryon momentum labels must be unique")
    bindings = {}
    for index in range(3):
        axis = f"mode{index}"
        record = document["bases"][axis]
        bindings[axis] = (record["data_id"], tuple(record["indices"]))
    view_document = spec.document
    mode_bindings = {
        f"mode{index}": (
            view_document["bases"][f"mode{index}"]["data_id"],
            tuple(view_document["bases"][f"mode{index}"]["indices"]),
        )
        for index in range(3)
    }
    return dict(
        dataset=dataset, spec=spec, stored_spec=stored_spec,
        document=document, definition=definition,
        configuration=configuration, product=product, phase=conventions["phase"],
        boundary=conventions["path_boundary"], Lt=lt, time_selection=selection,
        components=components, component_coordinates=raw_coordinates,
        momenta=tuple(momenta), momentum_coordinates=raw_momenta,
        mode_bindings=MappingProxyType(mode_bindings),
    )


class DatasetBaryonElemental:
    """Borrow one registered ``elemental_baryon_vvv`` dataset as a B reader."""

    background_read_safe = True

    def read_slab_in_background(self, *args, **kwargs):
        """Read selected host blocks without mutating adapter or reader state."""
        return self.read_slab(*args, **kwargs)

    def __init__(self, dataset):
        record = _parse_dataset(dataset)
        self.datasets = MappingProxyType({"vvv": dataset})
        self.dataset = dataset
        self.spec = record["spec"]
        self.cfg = record["configuration"]["configuration"]
        self.ensemble = record["configuration"]["ensemble"]
        self.lattice = tuple(record["configuration"]["lattice"])
        self.Lt = self.Nt = record["Lt"]
        self.physical_Lt = self.lattice[3]
        self.time_selection = record['time_selection']
        self.time_selections = {} if self.time_selection is None else {'time': self.time_selection}
        self.time_selection_identity = (None if self.time_selection is None else
                                       (('time', self.physical_Lt, self.time_selection.times),))
        self.phase_convention = record["phase"]
        self.boundary = record["boundary"]
        self.product = record["product"]
        self.components = record["components"]
        self.momenta = record["momenta"]
        self.data_identity = dataset.data_id
        self.basis_bindings = record["mode_bindings"]
        self.sampling_pool_ids = {}
        self.sampling_groups = {}
        self._record = record

    def describe_block(self, component, momentum, t, sector="vvv", *, usedNe=None,
                       mode_indices=None):
        if sector != "vvv":
            raise ValueError("baryon dataset reader supports only the vvv sector")
        record = self._record
        if component not in self.components:
            raise KeyError("baryon component is absent from the registered dataset")
        if isinstance(momentum, (bool, np.bool_)) or not isinstance(momentum, Integral):
            raise ValueError("baryon momentum must be an integer coordinate index")
        momentum = int(momentum)
        if not 0 <= momentum < len(self.momenta):
            raise KeyError("baryon momentum index is outside the registered dataset")
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, Integral):
            raise ValueError("baryon time must be an integer physical coordinate")
        t = int(t)
        if not 0 <= t < self.Lt:
            raise KeyError("baryon time is outside the registered dataset")
        mode_axes = ("mode0", "mode1", "mode2")
        raw_shape = tuple(len(self.spec.coordinates(axis)) for axis in mode_axes)
        if usedNe is None:
            counts = None
        elif isinstance(usedNe, (tuple, list)):
            counts = tuple(usedNe)
            if len(counts) != 3:
                raise ValueError("baryon usedNe must be a positive scalar or three ordered counts")
        else:
            counts = (usedNe,) * 3
        if counts is not None:
            for index, count in enumerate(counts):
                count = _integer(count, f"usedNe[{index}]", minimum=1)
                if count > raw_shape[index]:
                    raise ValueError(f"usedNe[{index}] exceeds registered {mode_axes[index]} coverage")
        selections = resolve_selections(
            mode_axes, raw_shape, usedNe=counts, mode_indices=mode_indices,
        )
        selected_labels = {}
        selected_shape = []
        for index, axis in enumerate(mode_axes):
            selection = selections.get(axis)
            coordinates = tuple(self.spec.coordinates(axis))
            if selection is None:
                positions = tuple(range(raw_shape[index]))
            elif isinstance(selection, slice):
                positions = tuple(range(*selection.indices(raw_shape[index])))
            else:
                positions = tuple(selection)
            if not positions:
                raise ValueError(f"{axis} selection cannot be empty")
            selected_labels[axis] = tuple(coordinates[position] for position in positions)
            selected_shape.append(len(positions))
        component_index = self.components.index(component)
        component_coordinate = record["component_coordinates"][component_index]
        momentum_coordinate = record["momentum_coordinates"][momentum]
        axis_bindings = {
            axis: (record["mode_bindings"][axis][0], selected_labels[axis])
            for axis in mode_axes
        }
        identity = (
            "dataset-baryon-elemental-vvv", self.data_identity,
            component_coordinate, momentum_coordinate, t,
            tuple((axis, selected_labels[axis]) for axis in mode_axes),
        )
        return TensorAxisMetadata(
            mode_axes, tuple(selected_shape), np.complex128, self.data_identity,
            repr(identity), axis_bindings=axis_bindings,
        )

    def block(self, component, momentum, t, sector="vvv", *, usedNe=None,
              mode_indices=None):
        metadata = self.describe_block(
            component, momentum, t, sector, usedNe=usedNe,
            mode_indices=mode_indices,
        )
        component_index = self.components.index(component)
        component_coordinate = self._record["component_coordinates"][component_index]
        momentum_coordinate = self._record["momentum_coordinates"][int(momentum)]
        select = {}
        for axis in ("mode0", "mode1", "mode2"):
            actual = self.spec.coordinates(axis)
            labels = metadata.axis_bindings[axis][1]
            if labels == tuple(actual):
                continue
            positions = tuple(actual.index(label) for label in labels)
            if positions == tuple(range(len(positions))):
                select[axis] = slice(0, len(positions))
            else:
                select[axis] = labels
        value = self.dataset.read(
            time=int(t), component=component_coordinate,
            momentum=momentum_coordinate, select=select,
        )
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError("baryon block shape/dtype differs from its published metadata")
        return value

    def describe_slab(self, component, momentum, times_selected, sector="vvv", *,
                      usedNe=None, mode_indices=None):
        times = _time_sequence(times_selected, self.Lt)
        block = self.describe_block(component, momentum, times[0], sector,
                                    usedNe=usedNe, mode_indices=mode_indices)
        for time in times[1:]:
            other = self.describe_block(component, momentum, time, sector,
                                        usedNe=usedNe, mode_indices=mode_indices)
            if (other.shape != block.shape or other.dtype != block.dtype
                    or other.axis_bindings != block.axis_bindings):
                raise ValueError("baryon slab selections differ across times")
        identity = ("dataset-baryon-elemental-slab", block.data_identity,
                    block.selection_identity, times)
        return TensorAxisMetadata(
            ("coordinate_batch", *block.axes), (len(times), *block.shape),
            block.dtype, block.data_identity, repr(identity),
            axis_bindings=block.axis_bindings,
        )

    def read_slab(self, component, momentum, times_selected, sector="vvv", *,
                  usedNe=None, mode_indices=None):
        metadata = self.describe_slab(
            component, momentum, times_selected, sector,
            usedNe=usedNe, mode_indices=mode_indices,
        )
        times = _time_sequence(times_selected, self.Lt)
        value = np.empty(metadata.shape, dtype=metadata.dtype)
        for index, time in enumerate(times):
            value[index] = self.block(
                component, momentum, int(time), sector,
                usedNe=usedNe, mode_indices=mode_indices,
            )
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError("baryon slab shape/dtype differs from its metadata")
        return value

    def read_slab_workspace_bytes(self, metadata, *, usedNe=None, mode_indices=None):
        if (not isinstance(metadata, TensorAxisMetadata) or not metadata.axes
                or metadata.axes[0] != "coordinate_batch"):
            raise TypeError("baryon workspace requires coordinate_batch metadata")
        raw_shape = tuple(len(self.spec.coordinates(axis))
                          for axis in ("mode0", "mode1", "mode2"))
        selections = resolve_selections(
            ("mode0", "mode1", "mode2"), raw_shape,
            usedNe=usedNe, mode_indices=mode_indices,
        )
        gathered = sum(not isinstance(value, slice) for value in selections.values())
        return gathered * int(np.prod(raw_shape)) * metadata.dtype.itemsize


__all__ = ["DatasetBaryonElemental"]
