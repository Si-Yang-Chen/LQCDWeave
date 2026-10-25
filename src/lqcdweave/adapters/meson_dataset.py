"""Adapt published meson datasets to the native Diagram reader contracts.

These providers borrow an already-open ``DatasetReader``. Storage owns its
maps; returned views remain valid only while that reader remains open. Dense
block descriptions use ``DatasetSpec`` metadata;
P/P elemental descriptions additionally read only the requested physical-time
coordinates from the bound point dataset to reconstruct sparse support, never
the elemental values.
"""

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from types import MappingProxyType

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json, stable_digest
from lqcdweave.contracts.time_selection import dataset_time_selection, dataset_physical_times, normalize_time_selection
from lqcdweave.contracts.current import (
    CURRENT_DIRECTED_POINT_SCHEMA, DirectedCurrentBasis,
    validate_directed_point_current_spec,
)
from lqcdweave.planning.highmode import validate_highmode_spec
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, ElementalComponent, GaugePath,
    SparsePointRecordMetadata, integer, normalize_elemental_structure,
    spatial_projection_rows,
)
from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.elemental_sectors import ELEMENTAL_SECTORS, sector_layout, sector_kinds
from lqcdweave.contracts.dilution import DILUTION_WEIGHT_SCHEME, DilutionBasis
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from .selection import resolve_selections


_PROPAGATOR_TAILS = {
    "vsv": ("sink_spin", "source_spin", "sink_mode", "source_mode"),
    "psv": ("sink_spin", "source_spin", "sink_point", "sink_color", "source_mode"),
    "vsp": ("sink_spin", "source_spin", "sink_mode", "source_point", "source_color"),
    "psp": ("sink_spin", "source_spin", "sink_point", "sink_color",
            "source_point", "source_color"),
}

_ELEMENTAL_TAILS = {
    "v2p": ("mode_left", "point_right", "color_right"),
    "p2v": ("point_left", "color_left", "mode_right"),
    "p2p": ("point_left", "color_left", "point_right", "color_right"),
}
_ELEMENTAL_STORED_TAILS = {
    "v2p": _ELEMENTAL_TAILS["v2p"],
    "p2v": _ELEMENTAL_TAILS["p2v"],
    "p2p": ("point_left", "color_left", "color_right"),
}


def _shape_after_selection(names, shape, selections):
    return tuple(
        len(range(*selections[name].indices(size)))
        if isinstance(selections.get(name), slice)
        else len(selections[name]) if name in selections else size
        for name, size in zip(names, shape, strict=True)
    )


def _elemental_mode_indices(sector, names, indices):
    if sector != 'v2v' or not isinstance(indices, dict):
        return indices
    aliases = dict(zip(('mode_left', 'mode_right'), names))
    normalized = {aliases.get(name, name): value for name, value in indices.items()}
    if len(normalized) != len(indices):
        raise ValueError('mode_indices repeats an elemental axis through an alias')
    return normalized


def _selection_identity(names, selections):
    return tuple((name, repr(selections[name])) for name in names if name in selections)


def _dataset_select(spec, selections):
    """Translate adapter-relative axis indices to DatasetReader selectors."""
    selected = {}
    for name, value in selections.items():
        if isinstance(value, slice):
            selected[name] = value
        else:
            coordinates = spec.coordinates(name)
            selected[name] = tuple(coordinates[index] for index in value)
    return selected


def _selected_axis_coordinates(spec, axis, selection):
    coordinates = tuple(spec.coordinates(axis))
    if selection is None:
        return coordinates
    if isinstance(selection, slice):
        positions = range(*selection.indices(len(coordinates)))
    else:
        positions = selection
    return tuple(coordinates[index] for index in positions)


def _point_diagonal_components(dataset, spec, component_coordinates, components):
    """Resolve stored component-row declarations through an optional view."""
    rows = spec.document["conventions"].get("point_diagonal_components", [])
    if not isinstance(rows, list):
        raise TypeError("point_diagonal_components must be a list of integer row indices")
    if not rows:
        return ()
    source_spec = getattr(getattr(dataset, "reader", None), "spec", spec)
    if "component" not in source_spec.axis_names:
        raise ValueError("point_diagonal_components requires a stored component axis")
    source_coordinates = tuple(source_spec.coordinates("component"))
    if (any(type(row) is not int for row in rows)
            or len(set(rows)) != len(rows)
            or any(not 0 <= row < len(source_coordinates) for row in rows)):
        raise ValueError("point_diagonal_components must contain unique in-range integer row indices")
    selected_components = {
        canonical_json(coordinate): component
        for coordinate, component in zip(component_coordinates, components, strict=True)
    }
    return tuple(
        selected_components[canonical_json(source_coordinates[row])]
        for row in rows
        if canonical_json(source_coordinates[row]) in selected_components
    )


def _elemental_structure_declarations(
        dataset, spec, sector, component_coordinates, components,
        momentum_coordinates, point_bindings):
    """Translate stored row/momentum slots through an optional DatasetView."""
    rows = spec.document["conventions"].get("elemental_structures", [])
    if not isinstance(rows, list):
        raise TypeError("elemental_structures must be a list of component/momentum declarations")
    if not rows:
        return MappingProxyType({})
    source_spec = getattr(getattr(dataset, "reader", None), "spec", spec)
    if "component" not in source_spec.axis_names or "momentum" not in source_spec.axis_names:
        raise ValueError("elemental_structures requires stored component and momentum axes")
    source_components = tuple(source_spec.coordinates("component"))
    source_momenta = tuple(source_spec.coordinates("momentum"))
    selected_components = {
        canonical_json(coordinate): index
        for index, coordinate in enumerate(component_coordinates)
    }
    selected_momenta = {
        canonical_json(coordinate): index
        for index, coordinate in enumerate(momentum_coordinates)
    }
    names = tuple(name for name in source_spec.axis_names
                  if name not in {"component", "momentum", "time"})
    if sector == "p2p":
        definition = source_spec.document["definition"]
        if definition["point_right_selection"] == "same_ordered_prefix_as_point_left":
            source_right = tuple(source_spec.document["bases"]["point_left"]["indices"])
        else:
            source_right = tuple(definition["point_right_labels"])
        source_shape = (len(source_spec.coordinates("point_left")), 3,
                        len(source_right), 3)
        logical_names = _ELEMENTAL_TAILS["p2p"]
    else:
        source_shape = tuple(len(source_spec.coordinates(name)) for name in names)
        logical_names = names
    seen = set()
    translated = {}
    for item in rows:
        if not isinstance(item, dict) or set(item) != {"component", "momentum", "structure"}:
            raise ValueError("elemental_structures entries require component, momentum, and structure")
        component_slot, momentum_slot = item["component"], item["momentum"]
        if (type(component_slot) is not int or not 0 <= component_slot < len(source_components)
                or type(momentum_slot) is not int or not 0 <= momentum_slot < len(source_momenta)):
            raise ValueError("elemental_structures slots must be in-range stored integer indices")
        stored_key = (component_slot, momentum_slot)
        if stored_key in seen:
            raise ValueError("elemental_structures cannot repeat a stored component/momentum pair")
        seen.add(stored_key)
        structure = normalize_elemental_structure(item["structure"])
        _validate_dataset_elemental_structure(
            logical_names, source_shape, structure,
        )
        component_key = canonical_json(source_components[component_slot])
        momentum_key = canonical_json(source_momenta[momentum_slot])
        if component_key in selected_components and momentum_key in selected_momenta:
            translated[(selected_components[component_key], selected_momenta[momentum_key])] = structure
    return MappingProxyType(translated)


def _validate_dataset_elemental_structure(names, shape, structure):
    if structure == "general":
        return
    extents = dict(zip(names, shape, strict=True))
    if structure == "point_diagonal":
        if ("point_left" not in extents or "point_right" not in extents
                or extents["point_left"] != extents["point_right"]):
            raise ValueError("point_diagonal requires equal left/right point extents")
        return
    mode_diagonal = (
        ("mode_left" in extents and "mode_right" in extents
         and extents["mode_left"] == extents["mode_right"])
        or ("sink_mode" in extents and "source_mode" in extents
            and extents["sink_mode"] == extents["source_mode"])
    )
    point_color_diagonal = (
        all(name in extents for name in
            ("point_left", "point_right", "color_left", "color_right"))
        and extents["point_left"] == extents["point_right"]
        and extents["color_left"] == extents["color_right"]
    )
    if not mode_diagonal and not point_color_diagonal:
        raise ValueError("diagonal requires equal left/right mode axes or equal point and color axes")


def _selected_dataset_structure(structure, names, raw_shape, selections, axis_bindings):
    """Apply the requested ordered axes and actual endpoint bindings."""
    extents = dict(zip(names, raw_shape, strict=True))

    def positions(axis):
        selection = selections.get(axis)
        if selection is None:
            return tuple(range(extents[axis]))
        if isinstance(selection, slice):
            return tuple(range(*selection.indices(extents[axis])))
        return tuple(selection)

    def same(left, right):
        left_binding, right_binding = axis_bindings.get(left), axis_bindings.get(right)
        if left_binding is None or right_binding is None:
            bindings_equal = left_binding is right_binding
        else:
            bindings_equal = (
                left_binding[0] == right_binding[0]
                and tuple(canonical_json(value) for value in left_binding[1])
                == tuple(canonical_json(value) for value in right_binding[1])
            )
        return positions(left) == positions(right) and bindings_equal

    if structure == "general":
        return structure
    if structure == "point_diagonal":
        if ("point_left" not in extents or "point_right" not in extents
                or not same("point_left", "point_right")):
            return "general"
        return structure
    mode_pairs = (("mode_left", "mode_right"),
                  ("sink_mode", "source_mode"))
    pair = next((pair for pair in mode_pairs
                 if pair[0] in extents and pair[1] in extents), None)
    if pair is not None:
        return structure if same(*pair) else "general"
    point_color_axes = ("point_left", "point_right", "color_left", "color_right")
    if all(axis in extents for axis in point_color_axes):
        return structure if same("point_left", "point_right") else "general"
    return "general"


def _selected_axis_bindings(spec, bindings, selections, axes):
    result = {}
    for axis in axes:
        actual_id, _coordinates = bindings[axis]
        result[axis] = (
            actual_id,
            _selected_axis_coordinates(spec, axis, selections.get(axis)),
        )
    return result


def _time_sequence(values, extent, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain integer time coordinates")
    if isinstance(values, Integral) and not isinstance(values, (bool, np.bool_)):
        values = (values,)
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or an iterable of integers") from exc
    if not raw:
        raise ValueError(f"{name} cannot be empty")
    result = tuple(integer(value, name, 0) for value in raw)
    if any(value >= extent for value in result):
        raise ValueError(f"{name} must lie in [0, {extent})")
    return result


def _require_coordinate(spec, axis, value):
    try:
        return spec.coordinates(axis).index(value)
    except ValueError as exc:
        raise KeyError(f"missing {axis} coordinate {value}") from exc


def _basis_bindings(spec, mode_axes):
    document = spec.document
    result = {}
    for name in mode_axes:
        binding = document["bases"].get(name)
        if binding is None or not binding["data_id"]:
            raise ValueError(f"{name} requires a known actual basis binding")
        # DatasetSpec validates that basis indices are the ordered axis coordinates.
        result[name] = (binding["data_id"], tuple(binding["indices"]))
    return result


def _validate_directed_vv_component_view(dataset, spec):
    """Require a verified 6/7 view over the registered eight-link V/V source."""
    actual_id = getattr(dataset, "data_id", None)
    requested = getattr(dataset, "spec", None)
    reader = getattr(dataset, "reader", dataset)
    parent_spec = getattr(reader, "spec", None)
    if (not isinstance(requested, DatasetSpec) or
            requested.definition_id != spec.definition_id or
            not isinstance(parent_spec, DatasetSpec)):
        raise ValueError("directed V/V view differs from its registered dataset reader")
    parent = DatasetSpec(parent_spec.document)
    parent_document = parent.document
    view_document = spec.document
    parent_definition = parent_document["definition"]
    definition = view_document["definition"]
    parent_axes = {
        name: tuple(parent.coordinates(name)) for name in parent.axis_names
    }
    if (parent_document["kind"] != "elemental_vv"
            or parent.axis_names != spec.axis_names
            or parent_document["configuration"] != view_document["configuration"]
            or parent_document["dtype"] != view_document["dtype"]
            or parent_document["conventions"] != view_document["conventions"]
            or parent_document["parents"] != view_document["parents"]
            or parent_definition != definition
            or parent_definition.get("directed_basis") != DirectedCurrentBasis.SCHEMA
            or parent_document["conventions"].get("phase") != "midpoint"
            or parent_definition.get("path_boundary") not in {"periodic", "open"}
            or definition.get("path_boundary") not in {"periodic", "open"}
            or parent.coordinates("component") != [
                {"path": [index]} for index in range(8)
            ]
            or spec.coordinates("component") != [{"path": [6]}, {"path": [7]}]
            or set(parent_document["bases"]) != {"sink_mode", "source_mode"}
            or set(view_document["bases"]) != {"sink_mode", "source_mode"}):
        raise ValueError("directed V/V view must select ordered temporal links from its full registered source")
    if (parent_definition.get("gauge_id") not in parent_document["parents"]
            or not isinstance(parent_definition.get("gauge_id"), str)
            or not parent_definition["gauge_id"]):
        raise ValueError("directed V/V parent must retain its actual gauge identity")
    parent_basis_ids = set()
    for role in ("sink_mode", "source_mode"):
        parent_binding = parent_document["bases"][role]
        view_binding = view_document["bases"][role]
        parent_id = parent_binding["data_id"]
        if (not isinstance(parent_id, str) or not parent_id
                or view_binding["data_id"] != parent_id
                or tuple(view_binding["indices"]) != tuple(spec.coordinates(role))
                or tuple(spec.coordinates(role)) != tuple(parent_axes[role][:len(spec.coordinates(role))])):
            raise ValueError("directed V/V view must preserve ordered actual mode prefixes")
        parent_basis_ids.add(parent_id)
    expected_parents = {parent_definition["gauge_id"], *parent_basis_ids}
    if (len(parent_document["parents"]) != len(expected_parents)
            or set(parent_document["parents"]) != expected_parents):
        raise ValueError("directed V/V parent identities must match its gauge and actual bases")
    for axis in parent.axis_names:
        if axis in ("component", "sink_mode", "source_mode"):
            continue
        if spec.coordinates(axis) != parent.coordinates(axis):
            raise ValueError("directed V/V view may only select temporal components and mode prefixes")
    return parent


class DatasetElemental:
    """Read published V/V and point-ended elemental datasets.

    Pass one legacy V/V dataset to preserve its original surface, or a mapping
    of explicit sector names (``v2v``, ``v2p``, ``p2v``, ``p2p``) to verified
    readers/views. Point-ended sectors retain actual basis/point IDs and label
    order. P/P row-slot storage is exposed as sparse COO records; it is never
    expanded to a dense point-pair array. A returned dense block can borrow a
    read-only mapping; gathered blocks and slabs own storage. Keep all readers
    open until borrowed blocks have been consumed.
    """

    background_read_safe = True

    def read_slab_in_background(self, *args, **kwargs):
        """Read host data without changing the caller-owned reader lifetime."""
        return self.read_slab(*args, **kwargs)

    def __init__(self, dataset, *, point_dataset=None, point_right_dataset=None,
                 sampling_pool_ids=None, sampling_groups=None, sampling_selection=None,
                 catalog=None, elemental_structures=None):
        self._composite = isinstance(dataset, Mapping) and not hasattr(dataset, "spec")
        self._explicit_catalog = None
        if catalog is not None:
            from lqcdweave.contracts.elemental_catalog import catalog_definition
            if self._composite or dataset.spec.document['kind'] != 'elemental_projected':
                raise ValueError('explicit catalog binding requires one projected elemental dataset')
            self._explicit_catalog = catalog_definition(catalog)
            if self._explicit_catalog is None:
                raise TypeError('catalog must be a versioned elemental catalog table')
        readers = (dict(dataset) if self._composite else {
            dataset.spec.document['definition'].get('logical_sector', 'v2v'): dataset,
        })
        if not readers or set(readers) - set(ELEMENTAL_SECTORS):
            raise ValueError("elemental readers must name known nonempty sectors")
        if any(not hasattr(reader, "spec") or not isinstance(reader.data_id, str)
               or not reader.data_id for reader in readers.values()):
            raise TypeError("elemental inputs must be registered DatasetReader or DatasetView objects")
        self.datasets = MappingProxyType(readers)
        marks = tuple(reader.spec.document["conventions"].get("elemental_weighted", False)
                      for reader in readers.values())
        if any(type(mark) is not bool for mark in marks) or len(set(marks)) != 1:
            raise ValueError("elemental sectors must declare one boolean elemental_weighted convention")
        self.elemental_weighted = marks[0]
        self.dataset = next(iter(readers.values())) if len(readers) == 1 else self.datasets
        if "p2p" in readers and point_dataset is None:
            raise ValueError("p2p requires its actual left point_dataset reader")
        if "p2p" not in readers and (point_dataset is not None or point_right_dataset is not None):
            raise ValueError("point dataset readers are accepted only with a p2p sector")
        self.point_dataset = point_dataset
        self.point_right_dataset = point_right_dataset
        self._records = {name: self._parse_sector(name, reader, catalog=self._explicit_catalog)
                         for name, reader in readers.items()}
        # Explicit scientific declarations for legacy V/V data. The selected
        # metadata still checks actual endpoint bindings and records structure
        # in its selection identity; published DatasetSpecs remain immutable.
        if elemental_structures is not None:
            if not isinstance(elemental_structures, Mapping):
                raise TypeError('elemental_structures must map (sector, component, momentum) to structure')
            for (sector, component, momentum), structure in elemental_structures.items():
                if sector not in self._records:
                    raise ValueError('structure declaration names an unavailable sector')
                record = self._records[sector]
                if record['layout'] != 'v2v':
                    raise NotImplementedError('explicit legacy structure declarations currently require V/V')
                index = record['components'].index(component)
                momentum = integer(momentum, 'structure momentum', 0)
                if momentum >= len(record['momenta']):
                    raise ValueError('structure declaration momentum is outside coverage')
                structure = normalize_elemental_structure(structure)
                names = record['tail_names']
                shape = tuple(len(record['spec'].coordinates(name)) for name in names)
                _validate_dataset_elemental_structure(names, shape, structure)
                declarations = dict(record['elemental_structures'])
                key = index, momentum
                if declarations.get(key, 'general') not in ('general', structure):
                    raise ValueError('explicit structure conflicts with the stored declaration')
                declarations[key] = structure
                record['elemental_structures'] = MappingProxyType(declarations)
        schemes = {record["dilution_weight_scheme"] for record in self._records.values()
                   if record["dilution_weight_scheme"] is not None}
        if len(schemes) > 1:
            raise ValueError("elemental sectors declare different dilution weight schemes")
        self.dilution_weight_scheme = next(iter(schemes), None)
        self.dilution_endpoints = MappingProxyType({
            (sector, side): source
            for sector, record in self._records.items()
            for side, source in record['dilution_endpoints'].items()
        })
        if "p2p" in self._records:
            point_diagonal_components = list(
                self._records["p2p"]["point_diagonal_components"]
            )
            p2p_structures = self._records["p2p"]["elemental_structures"]
            for (component_index, _momentum), structure in p2p_structures.items():
                if (structure in {"diagonal", "point_diagonal"}
                        and self._records["p2p"]["components"][component_index]
                        not in point_diagonal_components):
                    point_diagonal_components.append(
                        self._records["p2p"]["components"][component_index]
                    )
            self.point_diagonal_components = tuple(point_diagonal_components)
        else:
            self.point_diagonal_components = ()
        self.elemental_structures = MappingProxyType({
            (sector, record["components"][component_index], momentum): structure
            for sector, record in self._records.items()
            for (component_index, momentum), structure in record["elemental_structures"].items()
        })
        first = next(iter(self._records.values()))
        self.spec = first["spec"] if len(readers) == 1 else MappingProxyType(
            {name: record["spec"] for name, record in self._records.items()})
        self.sectors = MappingProxyType(readers)
        self.phase_convention = first["phase"]
        declared_boundaries = {
            record["path_boundary"] for record in self._records.values()
            if record["path_boundary"] is not None
        }
        if len(declared_boundaries) > 1:
            raise ValueError("elemental sectors declare different path boundaries")
        self._path_boundary = (next(iter(declared_boundaries))
                               if declared_boundaries else None)
        current_schemas = {
            record["current_schema"] for record in self._records.values()
            if record["current_schema"] is not None
        }
        if len(current_schemas) > 1:
            raise ValueError("elemental sectors declare different directed-current schemas")
        self.current_schema = (next(iter(current_schemas))
                               if current_schemas else None)
        self.cfg = first["configuration"]["configuration"]
        self.ensemble = first["configuration"]["ensemble"]
        self.lattice = tuple(first["configuration"]["lattice"])
        self.time_selection = first['time_selection']
        self.time_selections = {} if self.time_selection is None else {'time': self.time_selection}
        self.time_selection_identity = (None if self.time_selection is None else
                                       (('time', self.time_selection.Lt, self.time_selection.times),))
        self.physical_Lt = self.lattice[3]
        self.Lt = self.Nt = first['Lt']
        if sampling_selection is not None and self.time_selection is None:
            raise ValueError('sampling_selection requires an explicit compact time selection')
        self.sampling_time_selection = (self.time_selection if sampling_selection is None else
                                        normalize_time_selection(sampling_selection, Lt=self.physical_Lt))
        if sampling_selection is not None:
            self.time_selection_identity += (('sampling_time', self.physical_Lt, self.sampling_time_selection.times),)
        self.components = first["components"]
        self.momenta = first["momentum_coordinates"]
        if len(readers) == 1:
            self.basis_bindings = first["basis_bindings"]
            self.point_bindings = first["point_bindings"]
        else:
            self.basis_bindings = MappingProxyType({
                name: record["basis_bindings"] for name, record in self._records.items()
            })
            self.point_bindings = MappingProxyType({
                name: record["point_bindings"] for name, record in self._records.items()
            })
        if len(readers) == 1:
            self.data_identity = next(iter(readers.values())).data_id
        else:
            self.data_identity = stable_digest({
                "kind": "dataset-elemental-sector-set",
                "sectors": [[name, readers[name].data_id] for name in sorted(readers)],
            })
        self.sampling_pool_ids = ({} if sampling_pool_ids is None
                                  else dict(sampling_pool_ids))
        self.sampling_groups = ({} if sampling_groups is None
                                else dict(sampling_groups))
        for group_id, group in self.sampling_groups.items():
            if not isinstance(group, SamplingPoolSpec) or group.group_id != group_id:
                raise TypeError("sampling_groups must map explicit IDs to SamplingPoolSpec")
        for key, group_id in self.sampling_pool_ids.items():
            if (not isinstance(key, tuple) or len(key) != 4
                    or key[-1] not in ("bar", "field")
                    or not isinstance(group_id, str) or not group_id):
                raise ValueError("sampling_pool_ids keys must be (component,momentum,time,bar|field)")
            component, momentum, time, _ = key
            if (component not in self.components or isinstance(momentum, bool)
                    or not isinstance(momentum, Integral) or not 0 <= momentum < len(self.momenta)
                    or isinstance(time, bool) or not isinstance(time, Integral)
                    or not 0 <= time < (self.Lt if self.sampling_time_selection is None else self.sampling_time_selection.Nt)):
                raise ValueError("sampling_pool_ids contain an unknown component, momentum, or time")
            if group_id not in self.sampling_groups:
                raise ValueError(f"sampling pool {group_id!r} has no registered SamplingPoolSpec")
        self._validate_sector_set()
        if "p2p" in readers:
            record = self._records["p2p"]
            left_id = record["point_bindings"]["point_left"][0]
            right_id = record["point_bindings"]["point_right"][0]
            if self.point_right_dataset is None:
                if right_id != left_id:
                    raise ValueError(
                        "p2p requires its actual right point_dataset reader when endpoint IDs differ"
                    )
                self.point_right_dataset = point_dataset
            self._validate_point_datasets()

    @property
    def dilution_identity(self):
        if not self.dilution_endpoints:
            return None
        return canonical_json({
            "weight_scheme": self.dilution_weight_scheme,
            "endpoints": [[sector, side, source.document]
                          for (sector, side), source in sorted(self.dilution_endpoints.items())],
        })

    @property
    def path_boundary(self):
        """Declared common path boundary, or None for legacy unknown readers."""
        return self._path_boundary

    def available_sectors(self):
        """Return the explicitly bound raw elemental sectors in this reader set.

        This is a metadata-only declaration for B/E planning. It neither opens
        a payload nor implies that absent sectors can be reconstructed.
        """
        return tuple(self.datasets)

    def elemental_structure(self, component, momentum, sector):
        """Return the producer declaration translated to this reader's slots."""
        record = self._records[sector]
        component_index = self.components.index(component)
        momentum = integer(momentum, "momentum", 0)
        if momentum >= len(record["momentum_coordinates"]):
            raise KeyError("missing registered elemental momentum")
        structure = record["elemental_structures"].get(
            (component_index, momentum), "general",
        )
        return normalize_elemental_structure(
            structure,
            sector == "p2p" and component in record["point_diagonal_components"],
        )

    def dilution_basis(self, component, momentum, time, sector, side):
        if sector not in self._records or component not in self.components:
            raise KeyError('missing registered elemental sector or component')
        if side not in ('bar', 'field'):
            raise ValueError('dilution side must be bar or field')
        return self.dilution_endpoints.get((sector, side))

    @staticmethod
    def _parse_sector(sector, dataset, *, catalog=None):
        logical_sector = sector
        sector = sector_layout(sector)
        spec = dataset.spec
        document = spec.document
        declared_sector = document['definition'].get('logical_sector', sector)
        if declared_sector != logical_sector:
            raise ValueError('elemental dataset logical sector differs from its actual endpoint kinds')
        configuration = document["configuration"]
        time_selection = dataset_time_selection(spec, 'time')
        lt = configuration["lattice"][3] if time_selection is None else time_selection.Nt
        phase = document["conventions"].get("phase")
        if phase not in ("midpoint", "anchor", "none"):
            raise ValueError("elemental dataset must declare midpoint, anchor, or none phase")
        names = spec.axis_names
        required = {"component", "momentum", "time"}
        if not required.issubset(names):
            raise ValueError("elemental dataset requires component, momentum, and time axes")
        tail_names = tuple(name for name in names if name not in required)
        basis_bindings, point_bindings = {}, {}
        gauge_id = document["definition"].get("gauge_id")
        current_schema = document["definition"].get("directed_current_schema")
        temporal_path_schema = document["definition"].get("temporal_path_schema")
        path_boundary = document["definition"].get("path_boundary")
        projected = document["kind"] == "elemental_projected"
        direct_projection = (projected and document["definition"].get("method")
                             == "direct_calc_projected")
        if projected:
            definition = document["definition"]
            if direct_projection:
                hashes = definition.get("input_sha256", {})
                transform = definition.get("gauge_transform")
                if (logical_sector != "v2v" or "rows" not in definition
                        or not isinstance(hashes, dict)
                        or not {"gauge", "eigenvector"}.issubset(hashes)
                        or any(not isinstance(value, str) or len(value) != 64
                               or any(c not in "0123456789abcdef" for c in value)
                               for value in hashes.values())
                        or definition.get("gauge_source_sha256") != hashes.get("gauge")
                        or not isinstance(transform, dict)
                        or set(transform) != {"steps", "rho", "dimensions"}
                        or type(transform["steps"]) is not int or transform["steps"] < 0
                        or not isinstance(transform["rho"], (int, float))
                        or not np.isfinite(transform["rho"])
                        or transform["dimensions"] != "spatial"
                        or path_boundary not in ("periodic", "open")
                        or not document["bases"]
                        or any(not binding["data_id"]
                               or binding["data_id"] not in document["parents"]
                               for binding in document["bases"].values())
                        or "raw_id" in definition or "raw" in definition.get("input_views", {})):
                    raise ValueError("direct projected elemental requires actual gauge/basis inputs and transform")
            elif (sector != "v2v" or definition.get("method") != "linear_raw_path_combination"
                    or "rows" not in definition or not definition.get("raw_id")
                    or document["parents"] != [definition["raw_id"]]):
                raise ValueError("projected elemental requires its explicit rows and raw parent")
            if not direct_projection:
                raw_document = definition.get("input_views", {}).get("raw")
                if not isinstance(raw_document, dict):
                    raise ValueError("projected elemental requires its explicit raw input view")
                raw_spec = DatasetSpec(raw_document)
                if (raw_document["kind"] != "elemental_vv"
                        or raw_document["configuration"] != configuration
                        or raw_document["bases"] != document["bases"]
                        or raw_document["conventions"] != document["conventions"]
                        or raw_spec.coordinates("momentum") != spec.coordinates("momentum")):
                    raise ValueError("projected elemental identity differs from its raw input view")
                gauge_id = raw_document["definition"].get("gauge_id")
                path_boundary = raw_document["definition"].get("path_boundary")
                if not gauge_id or path_boundary not in ("periodic", "open"):
                    raise ValueError("projected elemental needs the actual gauge and path boundary")
        if sector == "v2v":
            if (len(tail_names) != 2 or any("mode" not in name for name in tail_names)
                    or names[0] not in required):
                raise ValueError("raw V/V dataset requires two named mode axes and a coordinate first axis")
            basis_bindings = _basis_bindings(spec, tail_names)
            if document["dtype"] not in ("complex64", "complex128", "float32", "float64"):
                raise ValueError("V/V elemental dtype must be floating or complex")
            if current_schema is not None:
                raise ValueError("point-current schema applies only to point-ended products")
            if temporal_path_schema is not None:
                raise ValueError("temporal point-path schema applies only to V/P and P/V products")
            if path_boundary is not None and path_boundary not in ("periodic", "open"):
                raise ValueError("V/V elemental path boundary must be periodic or open")
            if document["definition"].get("directed_basis") == DirectedCurrentBasis.SCHEMA:
                if path_boundary not in ("periodic", "open") or phase != "midpoint":
                    raise ValueError("directed V/V paths require an explicit boundary and midpoint phase")
                _validate_directed_vv_component_view(dataset, spec)
        else:
            expected_kind = {"v2p": "elemental_v2p", "p2v": "elemental_p2v",
                             "p2p": "elemental_p2p_row_sparse"}[sector]
            if document["kind"] != expected_kind:
                raise ValueError(f"{sector} reader requires kind {expected_kind!r}")
            expected_names = ("time", "component", "momentum",
                              *_ELEMENTAL_STORED_TAILS[sector])
            if names != expected_names:
                raise ValueError(f"{sector} dataset axes do not match its published contract")
            if document["dtype"] != "complex128":
                raise ValueError("point-ended elemental datasets require complex128")
            times = spec.coordinates("time")
            if times != list(range(lt)):
                raise ValueError("point-ended elemental time axis must contain every physical time in order")
            definitions, conventions = document["definition"], document["conventions"]
            path_boundary = definitions.get("path_boundary", "periodic")
            if current_schema is not None and temporal_path_schema is not None:
                raise ValueError("point elemental cannot combine current and temporal-path schemas")
            if current_schema is not None:
                if current_schema != CURRENT_DIRECTED_POINT_SCHEMA:
                    raise ValueError("unsupported directed point-current schema")
                validate_directed_point_current_spec(spec)
            if temporal_path_schema is not None:
                if temporal_path_schema != TEMPORAL_POINT_PATH_SCHEMA:
                    raise ValueError("unsupported temporal point-path schema")
                if sector not in {"v2p", "p2v", "p2p"}:
                    raise ValueError("temporal point-path schema supports V/P, P/V and P/P")
            if (path_boundary not in ("periodic", "open")
                    or (current_schema is None and temporal_path_schema is None
                        and path_boundary != "periodic")
                    or not isinstance(gauge_id, str) or not gauge_id
                    or conventions.get("normalization") != "unscaled"):
                raise ValueError("point-ended datasets require an actual gauge, supported path boundary, and unscaled convention")
            if sector == "p2p":
                selection = definitions.get("point_right_selection")
                if (not isinstance(definitions.get("point_right_id"), str)
                        or not definitions["point_right_id"]
                        or selection not in ("same_ordered_prefix_as_point_left",
                                             "explicit_ordered_labels")
                        or conventions.get("point_pair_layout") != "left-point-row-slots-v1"):
                    raise ValueError("p2p row slots require an explicit right endpoint contract")
                if selection == "explicit_ordered_labels":
                    labels = definitions.get("point_right_labels")
                    if (not isinstance(labels, list) or not labels
                            or any(type(label) is not int or label < 0 for label in labels)
                            or len(set(labels)) != len(labels)):
                        raise ValueError("explicit p2p right labels must be unique ordered nonnegative integers")
                elif "point_right_labels" in definitions:
                    raise ValueError("same-prefix p2p definitions must retain their original identity")
                expected_bases = {"point_left"}
                axes_for_basis = ("point_left",)
            else:
                expected_bases = ({"mode_left", "point_right"} if sector == "v2p"
                                  else {"point_left", "mode_right"})
                axes_for_basis = tuple(name for name in expected_names if name in expected_bases)
            if set(document["bases"]) != expected_bases:
                raise ValueError(f"{sector} basis bindings must match its actual endpoint axes")
            basis_bindings = _basis_bindings(spec, tuple(
                name for name in axes_for_basis if "mode" in name))
            for name in expected_bases:
                binding = document["bases"][name]
                if not isinstance(binding["data_id"], str) or not binding["data_id"]:
                    raise ValueError(f"{name} requires an actual basis/point dataset ID")
                if name.startswith("point"):
                    point_bindings[name] = (binding["data_id"], tuple(binding["indices"]))
            if sector == "p2p":
                left_id, left_labels = point_bindings["point_left"]
                right_id = definitions["point_right_id"]
                if definitions["point_right_selection"] == "same_ordered_prefix_as_point_left":
                    if right_id != left_id:
                        raise ValueError("same-prefix p2p right endpoint must reuse its actual left point ID")
                    right_labels = left_labels
                else:
                    right_labels = tuple(definitions["point_right_labels"])
                point_bindings["point_right"] = (right_id, right_labels)
            point_ids = {value[0] for value in point_bindings.values()}
            expected_parents = {gauge_id, *(value[0] for value in basis_bindings.values()), *point_ids}
            if (len(document["parents"]) != len(expected_parents)
                    or set(document["parents"]) != expected_parents):
                raise ValueError(f"{sector} parents do not bind its actual gauge, basis and point IDs")
            if sector == "p2p":
                if basis_bindings:
                    raise ValueError("p2p has no Laplace/vector basis dependency")
            colors = ("color_right",) if sector == "v2p" else (
                ("color_left",) if sector == "p2v" else ("color_left", "color_right"))
            for name in colors:
                if spec.coordinates(name) != [0, 1, 2]:
                    raise ValueError(f"{name} must explicitly order the three colors")
        times = spec.coordinates("time")
        if any(type(value) is not int or not 0 <= value < lt for value in times):
            raise ValueError("elemental time coordinates must be physical integers within Lt")
        raw_components = spec.coordinates("component")
        directed_vv = (
            sector == "v2v"
            and document["definition"].get("directed_basis") == DirectedCurrentBasis.SCHEMA
        )
        temporal_current = current_schema == CURRENT_DIRECTED_POINT_SCHEMA
        temporal_path = temporal_path_schema == TEMPORAL_POINT_PATH_SCHEMA
        supported_step_limit = 8 if temporal_current or temporal_path or directed_vv else 6
        if projected:
            rows = spatial_projection_rows(document["definition"]["rows"])
            if raw_components != [dict(projection=i) for i in range(len(rows))]:
                raise ValueError("projected component coordinates must match explicit ordered rows")
            if not direct_projection:
                available = {GaugePath(tuple(item["path"])) for item in
                             raw_spec.coordinates("component") if isinstance(item, dict) and set(item) == {"path"}}
                if any(path not in available for row in rows for path, _ in row):
                    raise ValueError("projected row requests a missing raw path")
            from lqcdweave.contracts.elemental_catalog import stored_catalog_components
            selection = document["definition"].get("catalog")
            if catalog is not None:
                if selection is not None and selection != catalog:
                    raise ValueError('explicit catalog differs from the registered catalog')
                selection = catalog
            components = stored_catalog_components(selection, document["definition"]["rows"])
        else:
            if any(not isinstance(value, dict) or set(value) != {"path"}
                   or not isinstance(value["path"], list)
                   or any(type(step) is not int or not 0 <= step < supported_step_limit
                          for step in value["path"])
                   for value in raw_components):
                raise ValueError("DatasetElemental requires explicit supported raw path components")
            if (temporal_current or directed_vv) and any(
                    len(value["path"]) != 1 or value["path"][0] not in (6, 7)
                    for value in raw_components):
                raise ValueError("directed current components must be one temporal step 6 or 7")
            if temporal_path and (not raw_components or not any(
                    step >= 6 for component in raw_components for step in component["path"])):
                raise ValueError("temporal point-path datasets require a temporal path component")
            components = tuple(ElementalComponent.path(GaugePath(tuple(value["path"])))
                               for value in raw_components)
        momenta = tuple(tuple(value) for value in spec.coordinates("momentum"))
        if any(len(momentum) != 3 or any(type(value) is not int for value in momentum)
               for momentum in momenta):
            raise ValueError("elemental momentum coordinates must be ordered integer XYZ triples")
        point_diagonal_components = _point_diagonal_components(
            dataset, spec, raw_components, components,
        )
        elemental_structures = _elemental_structure_declarations(
            dataset, spec, sector, raw_components, components,
            tuple(spec.coordinates("momentum")), point_bindings,
        )
        if document["dtype"] in ("complex64", "complex128", "float32", "float64"):
            pass
        else:
            raise ValueError("elemental dataset dtype must be floating or complex")
        dilution_endpoints = {}
        declared_dilution = document['conventions'].get('dilution_endpoints', {})
        if not isinstance(declared_dilution, dict) or set(declared_dilution) - {'bar', 'field'}:
            raise ValueError('dilution_endpoints must bind explicit bar/field provenance')
        for side, kind in zip(('bar', 'field'), sector_kinds(logical_sector), strict=True):
            source = DilutionBasis.from_document(declared_dilution.get(side))
            if source is None:
                if kind == 'd':
                    raise ValueError('D elemental endpoints require actual dilution provenance')
                continue
            if kind != 'd':
                raise ValueError('dilution provenance requires a logical D endpoint')
            left = side == 'bar'
            axis = next((name for name in basis_bindings
                         if name in (('mode_left', 'sink_mode') if left
                                     else ('mode_right', 'source_mode'))), None)
            if axis is None or basis_bindings[axis][0] != source.data_id:
                raise ValueError('D provenance differs from its actual basis binding')
            labels = basis_bindings[axis][1]
            if any(type(label) is not int or not 0 <= label < len(source.vector_labels)
                   for label in labels):
                raise ValueError('D axis labels must address the declared flattened vector order')
            dilution_endpoints[side] = source
        has_dilution = "d" in sector_kinds(logical_sector)
        weight_scheme = document["conventions"].get("dilution_weight_scheme")
        if has_dilution:
            if weight_scheme is None:
                if document["conventions"].get("elemental_weighted", False):
                    raise ValueError(
                        "weighted D elementals require an explicit dilution_weight_scheme"
                    )
                weight_scheme = DILUTION_WEIGHT_SCHEME
            elif weight_scheme != DILUTION_WEIGHT_SCHEME:
                raise ValueError("unsupported D elemental dilution_weight_scheme")
        return dict(sector=logical_sector, layout=sector, dataset=dataset, spec=spec, document=document,
                    configuration=configuration, phase=phase, Lt=lt, time_selection=time_selection,
                    tail_names=tail_names, basis_bindings=basis_bindings,
                    point_bindings=point_bindings, gauge_id=gauge_id,
                    components=components, component_coordinates=tuple(raw_components),
                    point_diagonal_components=point_diagonal_components,
                    dilution_endpoints=dilution_endpoints,
                    dilution_weight_scheme=weight_scheme if has_dilution else None,
                    elemental_structures=elemental_structures,
                    current_schema=current_schema, path_boundary=path_boundary,
                    temporal_path_schema=temporal_path_schema,
                    directed_vv=directed_vv,
                    momenta=momenta,
                    momentum_coordinates=tuple(spec.coordinates("momentum")),
                    times=tuple(times))

    def _validate_sector_set(self):
        records = tuple(self._records.values())
        reference = records[0]
        if any(record["configuration"] != reference["configuration"]
               or record["phase"] != reference["phase"]
               or record["components"] != reference["components"]
               or record["momenta"] != reference["momenta"]
               or record["Lt"] != reference["Lt"]
               or record["time_selection"] != reference["time_selection"] for record in records[1:]):
            raise ValueError("elemental sectors must share configuration, phase, path and momentum definitions")

        # The directed eight-link V/V dataset has its own basis schema and
        # intentionally carries neither point-ended product schema. It may
        # anchor compatible point-ended sectors when the common path,
        # boundary, momentum, configuration and endpoint checks below agree.
        # Point-ended products themselves must still use one identical schema;
        # otherwise one composite reader would mix distinct raw contracts.
        point_records = tuple(record for record in records if record['layout'] != 'v2v')
        point_schemas = {
            (record["current_schema"], record["temporal_path_schema"])
            for record in point_records
        }
        if len(point_schemas) > 1:
            raise ValueError(
                "point-ended elemental sectors must share one current or temporal-path schema"
            )
        if point_schemas:
            point_schema = next(iter(point_schemas))
            if any(
                    record['layout'] == 'v2v' and not record["directed_vv"]
                    and (record["current_schema"], record["temporal_path_schema"])
                    != point_schema
                    for record in records):
                raise ValueError(
                    "unmarked V/V sectors cannot inherit a point-ended schema"
                )
        elif any(
                (record["current_schema"], record["temporal_path_schema"])
                != (reference["current_schema"], reference["temporal_path_schema"])
                for record in records[1:]):
            raise ValueError("elemental sectors must share their product schema")

        if len(records) > 1:
            expected_times = set(range(reference["Lt"]))
            if any(set(record["times"]) != expected_times for record in records):
                raise ValueError("composite elemental sectors must each cover every physical time")
            gauge_ids = {record["gauge_id"] for record in records}
            if len(gauge_ids) != 1 or None in gauge_ids:
                raise ValueError("composite elemental sectors must bind one explicit actual gauge ID")
            roles = {}

            def register(role, binding):
                prior = roles.get(role)
                if prior is None:
                    roles[role] = binding
                    return
                left_id, left_labels = prior
                right_id, right_labels = binding
                prefix = (left_labels[:len(right_labels)] == right_labels
                          or right_labels[:len(left_labels)] == left_labels)
                if left_id != right_id or not prefix:
                    raise ValueError(f"composite elemental {role} axes bind different actual bases or labels")

            for sector, record in self._records.items():
                bindings = record["basis_bindings"]
                points = record["point_bindings"]
                layout = record['layout']
                left_kind, right_kind = sector_kinds(sector)
                if layout == "v2v":
                    axes = record["tail_names"]
                    role_axes = {
                        "mode_left": next((name for name in axes
                                           if name in ("mode_left", "sink_mode")), None),
                        "mode_right": next((name for name in axes
                                            if name in ("mode_right", "source_mode")), None),
                    }
                    if any(name is None for name in role_axes.values()):
                        raise ValueError("composite V/V mode axes must name their left/right or sink/source roles")
                    register((left_kind, 'left'), bindings[role_axes["mode_left"]])
                    register((right_kind, 'right'), bindings[role_axes["mode_right"]])
                elif layout == "v2p":
                    register((left_kind, 'left'), bindings["mode_left"])
                    register("point_right", points["point_right"])
                elif layout == "p2v":
                    register("point_left", points["point_left"])
                    register((right_kind, 'right'), bindings["mode_right"])
                elif sector == "p2p":
                    left = points["point_left"]
                    right = points["point_right"]
                    register("point_left", left)
                    register("point_right", right)

    def _validate_point_datasets(self):
        record = self._records["p2p"]
        for endpoint, reader in (("left", self.point_dataset),
                                  ("right", self.point_right_dataset)):
            data_id, labels = record["point_bindings"][f"point_{endpoint}"]
            spec = getattr(reader, "spec", None)
            if spec is None or getattr(reader, "data_id", None) != data_id:
                raise ValueError(
                    f"p2p requires the actual registered point dataset for its {endpoint} endpoint"
                )
            point_times = tuple(spec.coordinates("time")) if "time" in spec.axis_names else ()
            point_selection = dataset_time_selection(spec, 'time')
            point_extent = self.physical_Lt if point_selection is None else point_selection.Nt
            if (self.time_selection is None) != (point_selection is None):
                raise ValueError('p2p point and elemental coordinate domains differ')
            if self.time_selection is not None:
                needed = set()
                for anchor in self.time_selection.times:
                    for component in self.components:
                        lo, hi = component.key.time_bounds
                        if record['path_boundary'] == 'open' and (anchor+lo < 0 or anchor+hi >= self.physical_Lt):
                            continue
                        offset = 0 if endpoint == 'left' else component.key.displacement[3]
                        needed.add((anchor+offset) % self.physical_Lt)
                point_selection.require_times(needed)
            if (spec.axis_names != ("time", "point", "xyz")
                    or spec.document["kind"] != "spatial_points"
                    or any(type(value) is not int or not 0 <= value < point_extent
                           for value in point_times)
                    or set(point_times) != set(range(point_extent))
                    or spec.coordinates("xyz") != ["x", "y", "z"]
                    or spec.document["dtype"] not in ("int32", "int64")
                    or spec.document["configuration"] != record["configuration"]):
                raise ValueError(
                    f"p2p {endpoint} point dataset must provide matching full-time ordered XYZ coordinates"
                )
            point_conventions = spec.document["conventions"]
            if (point_conventions.get("representation") != "coordinate_delta"
                    or point_conventions.get("normalization") != "unit_delta"):
                raise ValueError(
                    f"p2p {endpoint} point dataset must declare coordinate_delta/unit_delta XYZ data"
                )
            available = tuple(spec.coordinates("point"))
            if (not all(label in available for label in labels)
                    and tuple(labels[:len(available)]) != available):
                raise ValueError(
                    f"p2p {endpoint} point dataset does not cover its declared ordered labels"
                )

    def _raw_shape(self, sector):
        record = self._records[sector]
        if sector == "p2p":
            left_count = len(record["point_bindings"]["point_left"][1])
            right_count = len(record["point_bindings"]["point_right"][1])
            return (left_count, 3, right_count, 3)
        return tuple(record["spec"].shape[record["spec"].axis_names.index(name)]
                     for name in record["tail_names"])

    @staticmethod
    def _right_point_labels(record):
        return record["point_bindings"]["point_right"][1]

    def _request(self, component, momentum, time, sector, *, usedNe=None,
                 mode_indices=None, usedNp=None, point_indices=None):
        if sector not in self._records or component not in self.components:
            raise KeyError(f"missing registered elemental component or {sector} sector")
        record = self._records[sector]
        momentum = integer(momentum, "momentum", 0)
        if momentum >= len(self.momenta):
            raise KeyError("missing registered elemental momentum")
        time = integer(time, "time", 0)
        if time >= self.Lt:
            raise KeyError("elemental time is outside dataset coverage")
        component_index = self.components.index(component)
        component_coordinate = record["component_coordinates"][component_index]
        momentum_coordinate = record["momentum_coordinates"][momentum]
        _require_coordinate(record["spec"], "component", component_coordinate)
        _require_coordinate(record["spec"], "momentum", momentum_coordinate)
        _require_coordinate(record["spec"], "time", time)
        if sector == "p2p":
            names = _ELEMENTAL_TAILS[sector]
            raw_shape = self._raw_shape(sector)
        else:
            names = record["tail_names"]
            raw_shape = self._raw_shape(sector)
        mode_indices = _elemental_mode_indices(sector, names, mode_indices)
        selections = resolve_selections(names, raw_shape, usedNe=usedNe,
                                        mode_indices=mode_indices, usedNp=usedNp,
                                        point_indices=point_indices)
        shape = _shape_after_selection(names, raw_shape, selections)
        basis_bindings = record["basis_bindings"]
        identity_bindings = dict(basis_bindings)
        identity_bindings.update(record["point_bindings"])
        axis_bindings = _selected_axis_bindings(
            record["spec"], identity_bindings, selections,
            tuple(name for name in names
                  if name in identity_bindings and name in record["spec"].axis_names),
        )
        if sector == "p2p":
            right_id, stored_right_labels = record["point_bindings"]["point_right"]
            axis_bindings["point_right"] = (
                right_id,
                self._selected_labels(stored_right_labels, selections.get("point_right")),
            )
        declared_structure = self.elemental_structure(component, momentum, sector)
        structure = _selected_dataset_structure(
            declared_structure, names, raw_shape, selections, axis_bindings,
        )
        if structure != "general":
            left_pool = self.point_pool_id(component, momentum, time, "bar")
            right_pool = self.point_pool_id(component, momentum, time, "field")
            if (left_pool is not None and right_pool is not None
                    and left_pool != right_pool):
                structure = "general"
        identity = (
            f"dataset-elemental-{sector}", self.datasets[sector].data_id,
            component_coordinate, momentum_coordinate, time, names,
            tuple((name, binding) for name, binding in basis_bindings.items()),
            tuple((name, binding) for name, binding in record["point_bindings"].items()),
            _selection_identity(names, selections),
        )
        if declared_structure != "general":
            identity += (("elemental-structure-v1", declared_structure, structure),)
        data_identity = self.datasets[sector].data_id
        if sector == "p2p":
            point_shape = (
                _shape_after_selection(
                    ("point_left", "color_left", "point_right", "color_right"),
                    raw_shape, selections,
                )[0],
                _shape_after_selection(
                    ("point_left", "color_left", "point_right", "color_right"),
                    raw_shape, selections,
                )[2],
            )
            rows, columns = self._point_pairs(
                component, time,
                self._selected_labels(record["spec"].coordinates("point_left"),
                                      selections.get("point_left")),
                self._selected_labels(self._right_point_labels(record),
                                      selections.get("point_right")),
            )
            metadata = TensorAxisMetadata(
                names, shape, record["document"]["dtype"], data_identity,
                repr(identity), sparse_layout="coo", point_shape=point_shape,
                nnz_upper=len(rows), axis_bindings=axis_bindings,
                structure=structure,
            )
        else:
            metadata = TensorAxisMetadata(
                names, shape, record["document"]["dtype"], data_identity,
                repr(identity), axis_bindings=axis_bindings, structure=structure,
            )
        return record, component_coordinate, momentum_coordinate, selections, metadata

    @staticmethod
    def _selected_positions(extent, selection):
        if selection is None:
            return tuple(range(extent))
        if isinstance(selection, slice):
            return tuple(range(*selection.indices(extent)))
        return tuple(selection)

    @classmethod
    def _selected_labels(cls, labels, selection):
        positions = cls._selected_positions(len(labels), selection)
        return tuple(labels[index] for index in positions)

    def _point_coordinates(self, reader, time, labels):
        if reader is None:
            raise ValueError("p2p point-coordinate reader was not bound")
        selection = dataset_time_selection(reader.spec, 'time')
        slot = int(time) if selection is None else selection.slot(time)
        value = reader.read(time=slot, select={"point": tuple(labels)})
        coordinates = np.asarray(value)
        if (coordinates.shape != (len(labels), 3) or coordinates.dtype.kind not in "iu"
                or np.any(coordinates < 0)
                or np.any(coordinates >= np.asarray(self.lattice[:3]))
                or len(np.unique(coordinates, axis=0)) != len(coordinates)):
            raise ValueError("p2p point coordinates must be unique integer XYZ labels inside the lattice")
        return coordinates

    def _point_pairs(self, component, time, left_labels, right_labels):
        record = self._records["p2p"]
        boundary = record["path_boundary"]
        time = int(time) if self.time_selection is None else self.time_selection.physical_time(time)
        endpoint_unwrapped = time + component.key.displacement[3]
        if boundary == "open":
            lo, hi = component.key.time_bounds
            if int(time) + lo < 0 or int(time) + hi >= self.physical_Lt:
                return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            endpoint = endpoint_unwrapped
        else:
            endpoint = endpoint_unwrapped % self.physical_Lt
        left = self._point_coordinates(self.point_dataset, time, left_labels)
        right = self._point_coordinates(self.point_right_dataset, endpoint, right_labels)
        lookup = {tuple(coordinate): index for index, coordinate in enumerate(right)}
        displacement = np.asarray(component.key.displacement[:3], dtype=np.int64)
        extent = np.asarray(self.lattice[:3], dtype=np.int64)
        rows, columns = [], []
        for row, coordinate in enumerate(left):
            target = tuple((coordinate + displacement) % extent)
            column = lookup.get(target)
            if column is not None:
                rows.append(row)
                columns.append(column)
        return np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)

    def _read_p2p(self, component_coordinate, momentum_coordinate, time,
                  selections, *, component):
        spec = self._records["p2p"]["spec"]
        stored_labels = tuple(spec.coordinates("point_left"))
        left_labels = self._selected_labels(stored_labels, selections.get("point_left"))
        right_labels = self._selected_labels(
            self._right_point_labels(self._records["p2p"]),
            selections.get("point_right"),
        )
        rows, columns = self._point_pairs(component, time, left_labels, right_labels)
        row_values = self.datasets["p2p"].read(
            time=int(time), select={
                "component": (component_coordinate,),
                "momentum": (momentum_coordinate,),
                "point_left": left_labels,
            },
        )[0, 0]
        if tuple(row_values.shape) != (len(left_labels), 3, 3):
            raise ValueError("p2p row-slot values differ from the selected left point labels")
        indices = np.column_stack((rows, columns)).astype(np.int64, copy=False)
        values = np.asarray(row_values[rows])
        values.setflags(write=False)
        return MappingProxyType({"type": "sparse", "indices": indices, "values": values})

    def describe_block(self, component, momentum, t, sector, *, usedNe=None,
                       mode_indices=None, usedNp=None, point_indices=None):
        """Describe one selected raw factor from registered headers/point coordinates."""
        return self._request(component, momentum, t, sector, usedNe=usedNe,
                             mode_indices=mode_indices, usedNp=usedNp,
                             point_indices=point_indices)[4]

    def block(self, component, momentum, t, sector, *, usedNe=None,
              mode_indices=None, usedNp=None, point_indices=None):
        """Read one selected factor, borrowing basic dense slices when possible."""
        record, component_coordinate, momentum_coordinate, selections, metadata = self._request(
            component, momentum, t, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        if sector == "p2p":
            return self._read_p2p(component_coordinate, momentum_coordinate, int(t),
                                  selections, component=component)
        value = record["dataset"].read(
            select=_dataset_select(record["spec"], selections),
            time=int(t), component=component_coordinate,
            momentum=momentum_coordinate,
        )
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError(f"{sector} block shape/dtype differs from its metadata")
        return value

    def describe_sparse_record(self, component, momentum, t):
        """Describe one P/P sparse block after reconstructing support from actual XYZ data."""
        record, component_coordinate, momentum_coordinate, _, metadata = self._request(
            component, momentum, t, "p2p")
        return SparsePointRecordMetadata(
            "sparse", metadata.point_shape, metadata.nnz_upper,
            np.dtype(record["document"]["dtype"]), np.dtype("int64"),
            metadata.nnz_upper * (2 * np.dtype("int64").itemsize
                                  + 9 * np.dtype(record["document"]["dtype"]).itemsize),
            self.datasets["p2p"].data_id, metadata.selection_identity,
            point_diagonal=metadata.point_diagonal, structure=metadata.structure,
        )

    def describe_selected_sparse_record(self, component, momentum, t, *, usedNp,
                                        point_indices=None):
        """Describe a selected P/P COO record for consumers that pass point prefixes."""
        record, _, _, _, metadata = self._request(
            component, momentum, t, "p2p", usedNp=usedNp,
            point_indices=point_indices,
        )
        return SparsePointRecordMetadata(
            "sparse", metadata.point_shape, metadata.nnz_upper,
            np.dtype(record["document"]["dtype"]), np.dtype("int64"),
            metadata.nnz_upper * (2 * np.dtype("int64").itemsize
                                  + 9 * np.dtype(record["document"]["dtype"]).itemsize),
            self.datasets["p2p"].data_id, metadata.selection_identity,
            point_diagonal=metadata.point_diagonal, structure=metadata.structure,
        )

    def describe_slab(self, component, momentum, times_selected, sector, *,
                      usedNe=None, mode_indices=None, usedNp=None,
                      point_indices=None):
        """Describe an ordered physical-time slab without loading block values."""
        if sector == "p2p":
            raise NotImplementedError("P/P row-slot datasets are consumed as sparse per-time blocks")
        selected_times = _time_sequence(times_selected, self.Lt, "times_selected")
        first = self._request(component, momentum, selected_times[0], sector,
                              usedNe=usedNe, mode_indices=mode_indices,
                              usedNp=usedNp, point_indices=point_indices)
        record, component_coordinate, momentum_coordinate, selections, block_metadata = first
        for time in selected_times[1:]:
            _require_coordinate(record["spec"], "time", time)
        identity = (
            f"dataset-elemental-{sector}-slab", self.datasets[sector].data_id,
            component_coordinate, momentum_coordinate, selected_times,
            record["tail_names"],
            tuple((name, record["basis_bindings"][name])
                  for name in record["basis_bindings"]),
            _selection_identity(record["tail_names"], selections),
        )
        declared_structure = self.elemental_structure(component, momentum, sector)
        if declared_structure != "general":
            identity += (("elemental-structure-v1", declared_structure,
                          block_metadata.structure),)
        return TensorAxisMetadata(
            ("coordinate_batch", *record["tail_names"]),
            (len(selected_times), *block_metadata.shape), block_metadata.dtype,
            self.datasets[sector].data_id, repr(identity),
            axis_bindings=block_metadata.axis_bindings,
            structure=block_metadata.structure,
        )

    def read_slab(self, component, momentum, times_selected, sector, *,
                  usedNe=None, mode_indices=None, usedNp=None,
                  point_indices=None):
        """Read the selected physical times in caller order into owned storage."""
        if sector == "p2p":
            raise NotImplementedError("P/P row-slot datasets are consumed as sparse per-time blocks")
        metadata = self.describe_slab(
            component, momentum, times_selected, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices,
        )
        selected_times = _time_sequence(times_selected, self.Lt, "times_selected")
        record = self._records[sector]
        if (record["spec"].axis_names[0] == "time"
                or len(set(selected_times)) != len(selected_times)):
            value = np.empty(metadata.shape, dtype=metadata.dtype)
            for index, time in enumerate(selected_times):
                lane = self.block(component, momentum, time, sector, usedNe=usedNe,
                                  mode_indices=mode_indices, usedNp=usedNp,
                                  point_indices=point_indices)
                value[index] = lane
                del lane
        else:
            component_coordinate = record["component_coordinates"][self.components.index(component)]
            momentum_coordinate = record["momentum_coordinates"][integer(momentum, "momentum", 0)]
            selections = resolve_selections(
                record["tail_names"], self._raw_shape(sector), usedNe=usedNe,
                mode_indices=_elemental_mode_indices(sector, record["tail_names"], mode_indices),
                usedNp=usedNp, point_indices=point_indices,
            )
            combined = _dataset_select(record["spec"], selections)
            combined["time"] = selected_times
            value = record["dataset"].read(
                select=combined, component=component_coordinate,
                momentum=momentum_coordinate,
            )
            raw_axes = tuple(name for name in record["spec"].axis_names
                             if name not in {"component", "momentum"})
            expected_axes = ("time", *record["tail_names"])
            if raw_axes != expected_axes:
                value = np.transpose(value, tuple(raw_axes.index(name) for name in expected_axes))
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError(f"{sector} slab shape/dtype differs from its metadata")
        return value

    def read_slab_workspace_bytes(self, metadata, *, usedNe=None, mode_indices=None,
                                  usedNp=None, point_indices=None):
        """Return a conservative host-workspace bound for dense elemental slabs."""
        if not isinstance(metadata, TensorAxisMetadata) or not metadata.axes or metadata.axes[0] != "coordinate_batch":
            raise TypeError("slab workspace requires coordinate_batch metadata")
        sector = next((name for name, reader in self.datasets.items()
                       if reader.data_id == metadata.data_identity), None)
        if sector is None or sector == "p2p":
            raise ValueError("slab metadata does not identify a dense elemental sector")
        shape = self._raw_shape(sector)
        if all(value is None for value in (usedNe, mode_indices, usedNp, point_indices)):
            return 2 * int(np.prod(shape)) * metadata.dtype.itemsize
        record = self._records[sector]
        selections = resolve_selections(record["tail_names"], shape,
                                        usedNe=usedNe, mode_indices=_elemental_mode_indices(
                                            sector, record["tail_names"], mode_indices),
                                        usedNp=usedNp, point_indices=point_indices)
        gathered = sum(not isinstance(value, slice) for value in selections.values())
        if gathered:
            return gathered * int(np.prod(shape)) * metadata.dtype.itemsize
        if record["spec"].axis_names[0] == "time":
            return int(np.prod(metadata.shape[1:])) * metadata.dtype.itemsize
        return 0

    def point_pool_id(self, component, momentum, t, side):
        """Return only an explicitly caller-bound sampling-pool ID."""
        if side not in ("bar", "field"):
            raise ValueError("point pool side must be bar or field")
        return self.sampling_pool_ids.get((component, int(momentum), int(t), side))

    def sampling_group(self, group_id):
        return self.sampling_groups.get(group_id)


class DatasetPropagator:
    """Read published propagator sectors through the ArrayPropagator contract.

    Datasets use explicit ``source_time`` and either ``sink_time`` (absolute)
    or ``relative_sink_time`` axes. The first dataset axis must be one of
    these time axes because DatasetReader deliberately reads one first-axis
    coordinate at a time. ``block`` returns a borrowed view when the selected
    axes are basic slices; gathered selections own their result. Point sectors
    require actual point-basis IDs and ordered point labels on each point axis,
    and explicitly supported spin, temporal-boundary, and normalization
    conventions. Actual point identities are not sampling-pool identities.
    ``read_slab`` returns an owned array. The caller owns and closes the reader.
    """

    background_read_safe = True

    def read_slab_in_background(self, *args, **kwargs):
        """Read host data without changing the caller-owned reader lifetime."""
        return self.read_slab(*args, **kwargs)

    def __init__(self, dataset):
        self._composite = isinstance(dataset, Mapping) and not hasattr(dataset, "spec")
        if self._composite:
            supplied = dict(dataset)
            extras = set(supplied) - set(_PROPAGATOR_TAILS) - {"basis", "gauge", "points"}
            readers = {sector: reader for sector, reader in supplied.items()
                       if sector in _PROPAGATOR_TAILS}
            if extras or not readers:
                raise ValueError("propagator readers must name known nonempty sectors")
            if any(not hasattr(reader, "spec") or not isinstance(reader.data_id, str)
                   or not reader.data_id for reader in supplied.values()):
                raise TypeError("propagator inputs must be registered DatasetReader or DatasetView objects")
            children = {sector: DatasetPropagator(reader)
                        for sector, reader in readers.items()}
            self._sector_readers = MappingProxyType(children)
            self._sector_identities = MappingProxyType({
                sector: stable_digest({"role": sector, "data_id": reader.data_identity})
                for sector, reader in children.items()
            })
            self.dataset = MappingProxyType(readers)
            self.spec = MappingProxyType({sector: reader.spec
                                          for sector, reader in children.items()})
            self.kind = "composite"
            self.cfg = next(iter(children.values())).cfg
            self.ensemble = next(iter(children.values())).ensemble
            self.lattice = next(iter(children.values())).lattice
            first_child = next(iter(children.values()))
            self.Lt = self.Nt = first_child.Lt
            self.physical_Lt = self.lattice[3]
            self.time_selection = first_child.time_selection
            self.time_selections = first_child.time_selections
            self.time_selection_identity = first_child.time_selection_identity
            if any(child.time_selection_identity != self.time_selection_identity for child in children.values()):
                raise ValueError('composite propagator ordered time selections differ')
            self.time_convention = None
            self.source_time_axis = MappingProxyType({
                sector: reader.source_time_axis for sector, reader in children.items()
            })
            self.sink_time_axis = MappingProxyType({
                sector: reader.sink_time_axis for sector, reader in children.items()
            })
            self.basis_bindings = MappingProxyType({
                sector: reader.basis_bindings for sector, reader in children.items()
            })
            self.point_bindings = MappingProxyType({
                sector: reader.point_bindings for sector, reader in children.items()
            })
            self.data_identity = stable_digest({
                "kind": "dataset-propagator-sector-set",
                "sectors": [[sector, children[sector].data_identity]
                            for sector in sorted(children)],
            })
            self._validate_sector_set()
            return
        self._sector_readers = None
        self._sector_identities = None
        self.dataset = dataset
        self.spec = dataset.spec
        d = self.spec.document
        kind = d["kind"]
        if "high_mode_projection" in d["definition"]:
            validate_highmode_spec(self.spec, allow_output_view=True)
        if kind not in _PROPAGATOR_TAILS:
            raise ValueError("DatasetPropagator requires a VSV, PSV, VSP, or PSP dataset")
        names = self.spec.axis_names
        source_axis = "source_time"
        sink_axis = ("relative_sink_time" if "relative_sink_time" in names
                     else "sink_time")
        if source_axis not in names or sink_axis not in names:
            raise ValueError(f"{kind.upper()} dataset requires source_time and sink_time axes")
        tail_names = _PROPAGATOR_TAILS[kind]
        if set(names) != {source_axis, sink_axis, *tail_names} or len(names) != 2 + len(tail_names):
            raise ValueError(f"{kind.upper()} dataset has missing, extra, or duplicate semantic axes")
        if names[0] not in (source_axis, sink_axis):
            raise ValueError("DatasetReader first axis must be source_time or sink_time")
        for name in ("sink_spin", "source_spin"):
            spins = self.spec.coordinates(name)
            if len(spins) != 4 or any(type(value) is not int for value in spins) or spins != [0, 1, 2, 3]:
                raise ValueError("propagator spin axes must be ordered 0,1,2,3")
        for name in (axis for axis in tail_names if axis.endswith("color")):
            colors = self.spec.coordinates(name)
            if len(colors) != 3 or any(type(value) is not int for value in colors) or colors != [0, 1, 2]:
                raise ValueError(f"{kind.upper()} color axes must be ordered 0,1,2")
        lt = d["configuration"]["lattice"][3]
        for name in (source_axis, sink_axis):
            coordinates = self.spec.coordinates(name)
            if any(type(value) is not int or not 0 <= value < lt for value in coordinates):
                raise ValueError(f"{name} coordinates must be integers within Lt")
        basis_axes = tuple(name for name in tail_names
                           if name.endswith("_mode") or name.endswith("_point"))
        if kind != "vsv" and set(d["bases"]) != set(basis_axes):
            raise ValueError(f"{kind.upper()} basis bindings must match its mode and point axes")
        self.basis_bindings = _basis_bindings(self.spec, basis_axes)
        self.point_bindings = {
            name: self.basis_bindings[name]
            for name in basis_axes if name.endswith("_point")
        }
        declared = d["conventions"].get("time_convention")
        axis_convention = "relative" if sink_axis == "relative_sink_time" else None
        if declared is not None and declared not in ("absolute", "relative"):
            raise ValueError("propagator time_convention must be absolute or relative")
        if axis_convention is not None and declared not in (None, "relative"):
            raise ValueError("relative_sink_time axis conflicts with declared absolute time convention")
        self.time_convention = axis_convention or declared or "absolute"
        self.kind = kind
        if kind != "vsv":
            conventions = d["conventions"]
            if conventions.get("spin_basis") != "degrand_rossi":
                raise ValueError("point propagator sectors require the degrand_rossi spin basis")
            if conventions.get("temporal_boundary") not in ("periodic", "antiperiodic"):
                raise ValueError("point propagator sectors require an explicit supported temporal boundary")
            if conventions.get("normalization") != "unscaled":
                raise ValueError("point propagator sectors require the explicit unscaled normalization")
            definition = d["definition"]
            gauge_id = definition.get("gauge_id") if isinstance(definition, dict) else None
            if not isinstance(gauge_id, str) or not gauge_id:
                raise ValueError("point propagator sectors require an actual gauge ID")
            self.gauge_id = gauge_id
            self.spin_basis = conventions["spin_basis"]
            self.temporal_boundary = conventions["temporal_boundary"]
            self.normalization = conventions["normalization"]
        self.source_time_axis, self.sink_time_axis = source_axis, sink_axis
        self.cfg = d["configuration"]["configuration"]
        self.ensemble = d["configuration"]["ensemble"]
        self.lattice = tuple(d["configuration"]["lattice"])
        self.physical_Lt = self.lattice[3]
        source_selection = dataset_time_selection(self.spec, source_axis)
        sink_selection = dataset_time_selection(self.spec, sink_axis)
        if (source_selection is None) != (sink_selection is None):
            raise ValueError('slot propagators require both endpoint selection records')
        self.time_selections = ({} if source_selection is None else
                                {'source_time': source_selection, 'sink_time': sink_selection})
        self.time_selection = source_selection if source_selection == sink_selection else None
        self.time_selection_identity = (None if not self.time_selections else
            tuple((axis, record.Lt, record.times) for axis, record in sorted(self.time_selections.items())))
        self.Lt = self.Nt = (self.physical_Lt if not self.time_selections else
                             max(source_selection.Nt, sink_selection.Nt))
        if source_selection is not None:
            layout = d['conventions']['time_coordinates']['layout']
            if (layout == 'cyclic_difference') != (self.time_convention == 'relative'):
                raise ValueError('slot layout differs from propagator time convention')
            if self.time_convention == 'relative' and source_selection != sink_selection:
                raise ValueError('relative slot propagator requires equal ordered selections')
        self.data_identity = dataset.data_id
        self._tail_names = tail_names

    @staticmethod
    def _prefix_compatible(first, second):
        return (first[:len(second)] == second
                or second[:len(first)] == first)

    @staticmethod
    def _point_projection_signature(reader, side):
        """Describe the effective point endpoint, excluding output views.

        Point labels identify the external delta basis. For a projected sector,
        the nested overlap's complete mode set and actual vector basis define
        the additional Q acting on that basis.
        """
        definition = reader.spec.document["definition"]
        projection = definition.get("high_mode_projection")
        if projection is None:
            return ("raw_delta",)
        if not isinstance(projection, dict):
            raise ValueError("high-mode point endpoint has an invalid projector definition")
        if side not in projection.get("sides", ()):
            return ("raw_delta",)
        views = definition.get("input_views")
        role = "overlap_" + side
        if not isinstance(views, dict) or role not in views:
            raise ValueError(f"high-mode {side} point endpoint lacks its overlap input view")
        overlap = DatasetSpec(views[role])
        basis_id, mode_labels = _basis_bindings(overlap, ("mode",))["mode"]
        # A permutation of an orthonormal basis leaves V V† unchanged.
        complete_mode_set = tuple(sorted(canonical_json(label) for label in mode_labels))
        return ("high_mode_q", projection["scheme"], basis_id, complete_mode_set)

    def _validate_sector_set(self):
        children = self._sector_readers
        records = {sector: reader.spec.document for sector, reader in children.items()}
        first = next(iter(records.values()))
        configuration = first["configuration"]
        definition = first["definition"]
        conventions = first["conventions"]
        gauge_id = definition.get("gauge_id")
        if not isinstance(gauge_id, str) or not gauge_id:
            raise ValueError("composite propagators require an explicit actual gauge ID")
        dirac = definition.get("dirac")
        tolerance = definition.get("solver_tolerance")
        if (not isinstance(dirac, dict) or not dirac
                or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("composite propagators require explicit Dirac and positive solver definitions")
        if (conventions.get("spin_basis") != "degrand_rossi"
                or conventions.get("temporal_boundary") not in ("periodic", "antiperiodic")
                or conventions.get("normalization") != "unscaled"):
            raise ValueError("composite propagators require supported spin, boundary, and normalization conventions")
        for sector, document in records.items():
            current = document["definition"]
            current_conventions = document["conventions"]
            if (document["configuration"] != configuration
                    or current.get("gauge_id") != gauge_id
                    or current.get("dirac") != definition.get("dirac")
                    or current.get("solver_tolerance") != definition.get("solver_tolerance")
                    or current_conventions.get("spin_basis") != conventions.get("spin_basis")
                    or current_conventions.get("temporal_boundary") != conventions.get("temporal_boundary")
                    or current_conventions.get("normalization") != conventions.get("normalization")):
                raise ValueError(f"composite {sector} sector differs in cfg, gauge, Dirac, spin, or boundary definition")
            expected_parents = {gauge_id, *(binding[0]
                for binding in self._sector_readers[sector].basis_bindings.values())}
            parents = document["parents"]
            if "high_mode_projection" in current:
                validate_highmode_spec(children[sector].spec, allow_output_view=True)
            elif sector == "vsp" and current.get("derivation") == "gamma5_hermiticity":
                # The established VSP dataset is derived from its actual PSV
                # parent. With distinct endpoints this is the reverse PSV,
                # rather than the forward PSV in this composite.
                if len(parents) != 1:
                    raise ValueError("derived VSP must name its actual PSV parent")
                if "psv" in children:
                    forward = children["psv"].basis_bindings
                    reverse = children["vsp"].basis_bindings
                    matching_endpoints = all(
                        forward[left][0] == reverse[right][0]
                        and self._prefix_compatible(forward[left][1], reverse[right][1])
                        for left, right in (("sink_point", "source_point"),
                                            ("source_mode", "sink_mode"))
                    )
                    if matching_endpoints and parents[0] != children["psv"].data_identity:
                        raise ValueError("derived VSP must name its actual PSV parent")
            elif (len(parents) != len(set(parents))
                    or set(parents) != expected_parents):
                raise ValueError(f"composite {sector} parents do not bind its actual endpoint bases and gauge")

        roles = {}

        def register(role, binding, sector, projection_signature=None):
            prior = roles.get(role)
            if prior is None:
                roles[role] = (binding, sector, projection_signature)
                return
            prior_binding, _, prior_projection = prior
            same_id = prior_binding[0] == binding[0]
            same_labels = self._prefix_compatible(prior_binding[1], binding[1])
            if not same_id or not same_labels:
                raise ValueError(f"composite propagator {role} endpoints have different actual bases or labels")
            if role.endswith("_point") and prior_projection != projection_signature:
                raise ValueError(
                    f"composite propagator {role} endpoints use different raw or high-mode projections")

        for sector, reader in children.items():
            bindings = reader.basis_bindings
            if sector == "vsv":
                register("sink_mode", bindings["sink_mode"], sector)
                register("source_mode", bindings["source_mode"], sector)
            elif sector == "psv":
                register("sink_point", reader.point_bindings["sink_point"], sector,
                         self._point_projection_signature(reader, "sink"))
                register("source_mode", bindings["source_mode"], sector)
            elif sector == "vsp":
                register("sink_mode", bindings["sink_mode"], sector)
                register("source_point", reader.point_bindings["source_point"], sector,
                         self._point_projection_signature(reader, "source"))
            elif sector == "psp":
                register("sink_point", reader.point_bindings["sink_point"], sector,
                         self._point_projection_signature(reader, "sink"))
                register("source_point", reader.point_bindings["source_point"], sector,
                         self._point_projection_signature(reader, "source"))

    def _require_sector(self, sector):
        if self._composite:
            if sector not in self._sector_readers:
                raise KeyError(f"composite propagator has no {sector!r} sector")
            return
        if sector != self.kind:
            raise KeyError(f"dataset kind {self.kind!r} does not provide sector {sector!r}")

    def _times(self, sink_time, source_time):
        sink_time = integer(sink_time, "sink_time", 0)
        source_time = integer(source_time, "source_time", 0)
        if sink_time >= self.Lt or source_time >= self.Lt:
            raise KeyError("propagator time is outside dataset coverage")
        stored_sink = ((sink_time - source_time) % self.Lt
                       if self.time_convention == "relative" else sink_time)
        _require_coordinate(self.spec, self.source_time_axis, source_time)
        _require_coordinate(self.spec, self.sink_time_axis, stored_sink)
        return sink_time, source_time, stored_sink

    def _describe(self, sink_time, source_time, sector, *, usedNe=None,
                  mode_indices=None, usedNp=None, point_indices=None):
        self._require_sector(sector)
        sink_time, source_time, stored_sink = self._times(sink_time, source_time)
        raw_shape = tuple(self.spec.shape[self.spec.axis_names.index(name)]
                          for name in self._tail_names)
        selections = resolve_selections(
            self._tail_names, raw_shape, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        shape = _shape_after_selection(self._tail_names, raw_shape, selections)
        axis_bindings = _selected_axis_bindings(
            self.spec, self.basis_bindings, selections,
            tuple(name for name in self._tail_names if name in self.basis_bindings),
        )
        identity = (
            f"dataset-{sector}-block", self.data_identity, sink_time, source_time,
            stored_sink, self.time_convention,
            tuple((name, self.basis_bindings[name]) for name in self.basis_bindings),
            tuple((name, getattr(self, name, None))
                  for name in ("spin_basis", "temporal_boundary", "normalization", "gauge_id")),
            _selection_identity(self._tail_names, selections),
        )
        return TensorAxisMetadata(
            self._tail_names, shape, self.spec.document["dtype"],
            self.data_identity, repr(identity), axis_bindings=axis_bindings,
        ), selections, sink_time, source_time, stored_sink

    def describe_block(self, sink_time, source_time, sector="vsv", *,
                       usedNe=None, mode_indices=None, usedNp=None,
                       point_indices=None):
        """Return axis and identity metadata without reading dataset values."""
        if self._composite:
            child = self._sector_readers.get(sector)
            if child is None:
                raise KeyError(f"composite propagator has no {sector!r} sector")
            metadata = child.describe_block(
                sink_time, source_time, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
            return replace(metadata, data_identity=self._sector_identities[sector])
        return self._describe(
            sink_time, source_time, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )[0]

    def block(self, sink_time, source_time, sector="vsv", *, usedNe=None,
              mode_indices=None, usedNp=None, point_indices=None):
        if self._composite:
            child = self._sector_readers.get(sector)
            if child is None:
                raise KeyError(f"composite propagator has no {sector!r} sector")
            return child.block(
                sink_time, source_time, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
        metadata, selections, sink_time, source_time, stored_sink = self._describe(
            sink_time, source_time, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        value = self.dataset.read(
            select=_dataset_select(self.spec, selections),
            **{self.source_time_axis: source_time,
               self.sink_time_axis: stored_sink},
        )
        raw_axes = tuple(name for name in self.spec.axis_names
                         if name not in {self.source_time_axis, self.sink_time_axis})
        if raw_axes != self._tail_names:
            value = np.transpose(value, tuple(raw_axes.index(name) for name in self._tail_names))
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError(f"{self.kind.upper()} block shape/dtype differs from its metadata")
        return value

    def describe_slab(self, sink_times, source_time, sector="vsv", *,
                      usedNe=None, mode_indices=None, usedNp=None,
                      point_indices=None):
        """Describe requested physical sink times in their caller order."""
        if self._composite:
            child = self._sector_readers.get(sector)
            if child is None:
                raise KeyError(f"composite propagator has no {sector!r} sector")
            metadata = child.describe_slab(
                sink_times, source_time, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
            return replace(metadata, data_identity=self._sector_identities[sector])
        self._require_sector(sector)
        source_time = integer(source_time, "source_time", 0)
        if source_time >= self.Lt:
            raise KeyError("source time is outside dataset coverage")
        sinks = _time_sequence(sink_times, self.Lt, "sink_times")
        block_metadata, selections, _, _, _ = self._describe(
            sinks[0], source_time, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        for sink in sinks[1:]:
            self._times(sink, source_time)
        stored_sinks = tuple(
            (sink - source_time) % self.Lt if self.time_convention == "relative" else sink
            for sink in sinks
        )
        identity = (
            f"dataset-{sector}-slab", self.data_identity, source_time,
            sinks, stored_sinks, self.time_convention,
            tuple((name, self.basis_bindings[name]) for name in self.basis_bindings),
            tuple((name, getattr(self, name, None))
                  for name in ("spin_basis", "temporal_boundary", "normalization", "gauge_id")),
            _selection_identity(self._tail_names, selections),
        )
        return TensorAxisMetadata(
            ("coordinate_batch", *self._tail_names),
            (len(sinks), *block_metadata.shape), block_metadata.dtype,
            self.data_identity, repr(identity),
            axis_bindings=block_metadata.axis_bindings,
        )

    def read_slab(self, sink_times, source_time, sector="vsv", *,
                  usedNe=None, mode_indices=None, usedNp=None,
                  point_indices=None):
        """Read only the requested physical sink times and selected axes.

        Source-first datasets issue one coordinate selection for the complete
        ordered sink slab. Sink-first datasets issue one read per requested
        sink, because DatasetReader deliberately binds one first-axis value.
        Both paths return owned storage in caller time order.
        """
        if self._composite:
            child = self._sector_readers.get(sector)
            if child is None:
                raise KeyError(f"composite propagator has no {sector!r} sector")
            return child.read_slab(
                sink_times, source_time, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
        metadata = self.describe_slab(
            sink_times, source_time, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        source_time = integer(source_time, "source_time", 0)
        sinks = _time_sequence(sink_times, self.Lt, "sink_times")
        selections = resolve_selections(
            self._tail_names,
            tuple(self.spec.shape[self.spec.axis_names.index(name)]
                  for name in self._tail_names),
            usedNe=usedNe, mode_indices=mode_indices,
            usedNp=usedNp, point_indices=point_indices,
        )
        gathered_axes = any(not isinstance(value, slice) for value in selections.values())
        if (self.spec.axis_names[0] != self.source_time_axis
                or len(set(sinks)) != len(sinks)
                or gathered_axes):
            value = np.empty(metadata.shape, dtype=metadata.dtype)
            for index, sink in enumerate(sinks):
                lane = self.block(sink, source_time, sector, usedNe=usedNe,
                                  mode_indices=mode_indices, usedNp=usedNp,
                                  point_indices=point_indices)
                value[index] = lane
                del lane
        else:
            selections = resolve_selections(
                self._tail_names,
                tuple(self.spec.shape[self.spec.axis_names.index(name)]
                      for name in self._tail_names),
                usedNe=usedNe, mode_indices=mode_indices,
                usedNp=usedNp, point_indices=point_indices,
            )
            stored_sinks = tuple(
                (sink - source_time) % self.Lt if self.time_convention == "relative" else sink
                for sink in sinks
            )
            combined = _dataset_select(self.spec, selections)
            combined[self.sink_time_axis] = stored_sinks
            value = self.dataset.read(
                select=combined,
                **{self.source_time_axis: source_time},
            )
            raw_axes = tuple(name for name in self.spec.axis_names
                             if name != self.source_time_axis)
            expected_axes = (self.sink_time_axis, *self._tail_names)
            if raw_axes != expected_axes:
                value = np.transpose(value, tuple(raw_axes.index(name) for name in expected_axes))
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError(f"{self.kind.upper()} slab shape/dtype differs from its metadata")
        return value

    def read_slab_workspace_bytes(self, metadata, *, usedNe=None, mode_indices=None,
                                  usedNp=None, point_indices=None):
        """Conservative extra peak for lane staging and exact-index gathers.

        Bindings should pass the same selection arguments used for
        ``read_slab``. Without them, a two-gather upper bound is returned
        because metadata alone does not identify the selection sequence.
        """
        if self._composite:
            for sector, identity in self._sector_identities.items():
                if metadata.data_identity == identity:
                    child = self._sector_readers[sector]
                    return child.read_slab_workspace_bytes(
                        replace(metadata, data_identity=child.data_identity),
                        usedNe=usedNe, mode_indices=mode_indices,
                        usedNp=usedNp, point_indices=point_indices,
                    )
            raise ValueError("slab metadata does not belong to this composite propagator")
        if not isinstance(metadata, TensorAxisMetadata) or not metadata.axes or metadata.axes[0] != "coordinate_batch":
            raise TypeError("slab workspace requires coordinate_batch metadata")
        raw_shape = tuple(self.spec.shape[self.spec.axis_names.index(name)]
                          for name in self._tail_names)
        if all(value is None for value in (usedNe, mode_indices, usedNp, point_indices)):
            return 2 * int(np.prod(raw_shape)) * metadata.dtype.itemsize
        selections = resolve_selections(
            self._tail_names, raw_shape, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        gathered = sum(not isinstance(value, slice) for value in selections.values())
        if gathered:
            # Exact gathers may each hold a full raw lane at once.
            return gathered * int(np.prod(raw_shape)) * metadata.dtype.itemsize
        if self.spec.axis_names[0] != self.source_time_axis:
            # Sink-first storage requires one first-axis read per sink lane.
            # Reserve one selected lane alongside the preallocated output slab.
            return int(np.prod(metadata.shape[1:])) * metadata.dtype.itemsize
        return 0

    def iter_blocks(self, sink_times, source_times, sector="vsv", *,
                    usedNe=None, mode_indices=None, usedNp=None,
                    point_indices=None):
        if self._composite:
            child = self._sector_readers.get(sector)
            if child is None:
                raise KeyError(f"composite propagator has no {sector!r} sector")
            yield from child.iter_blocks(
                sink_times, source_times, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
            return
        sinks = _time_sequence(sink_times, self.Lt, "sink_times")
        sources = _time_sequence(source_times, self.Lt, "source_times")
        for source in sources:
            for sink in sinks:
                yield sink, source, self.block(
                    sink, source, sector, usedNe=usedNe,
                    mode_indices=mode_indices, usedNp=usedNp,
                    point_indices=point_indices,
                )

    def read_times(self, sink_times, source_times, sector="vsv", *,
                   usedNe=None, mode_indices=None, usedNp=None,
                   point_indices=None):
        if self._composite:
            child = self._sector_readers.get(sector)
            if child is None:
                raise KeyError(f"composite propagator has no {sector!r} sector")
            return child.read_times(
                sink_times, source_times, sector, usedNe=usedNe,
                mode_indices=mode_indices, usedNp=usedNp,
                point_indices=point_indices,
            )
        sinks = _time_sequence(sink_times, self.Lt, "sink_times")
        sources = _time_sequence(source_times, self.Lt, "source_times")
        return np.stack([
            np.stack([
                self.block(sink, source, sector, usedNe=usedNe,
                           mode_indices=mode_indices, usedNp=usedNp,
                           point_indices=point_indices)
                for sink in sinks
            ])
            for source in sources
        ])
