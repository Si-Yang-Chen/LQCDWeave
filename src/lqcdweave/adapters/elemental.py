"""Explicit array/NPY access to already produced elemental components."""

import numpy as np
from collections.abc import Mapping
from pathlib import Path
from dataclasses import replace

from lqcdweave.contracts.elemental import (
    ElementalComponent, BaryonComponent, GaugePath, integer,
    normalize_elemental_structure,
)
from lqcdweave.contracts.spatial_sampling import SpatialPointSet, SamplingPoolSpec
from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.time_selection import normalize_time_selection
from lqcdweave.contracts.elemental_sectors import (
    ELEMENTAL_SECTORS, elemental_tail_axes, sector_kinds,
)
from lqcdweave.contracts.dilution import DILUTION_WEIGHT_SCHEME
from .selection import select_axes, select_slab_axes, selected_points, times
from .selection import resolve_selections, host_arrays_resident


def _time_sequence(values, extent, name):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain integer time coordinates")
    try:
        raw = (values,) if isinstance(values, (int, np.integer)) else tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or an iterable of integers") from exc
    if not raw:
        raise ValueError(f"{name} cannot be empty")
    normalized = tuple(integer(value, name, 0) for value in raw)
    if any(value >= extent for value in normalized):
        raise ValueError(f"{name} must lie in [0, {extent})")
    return normalized


def _selection_positions(extent, selection):
    if selection is None:
        return tuple(range(extent))
    if isinstance(selection, slice):
        return tuple(range(*selection.indices(extent)))
    return tuple(selection)


def _endpoint_declarations(sectors, tails, basis_endpoints, dilution_endpoints):
    """Bind explicit actual endpoints without inspecting numerical arrays."""
    from lqcdweave.contracts.dilution import DilutionBasis
    bindings = {} if basis_endpoints is None else dict(basis_endpoints)
    dilution = {} if dilution_endpoints is None else dict(dilution_endpoints)
    for mapping in (bindings, dilution):
        for key in mapping:
            if (not isinstance(key, tuple) or len(key) != 2
                    or key[0] not in sectors or key[1] not in ('bar', 'field')):
                raise ValueError('endpoint declarations require (sector, bar|field) keys')
    for key, source in dilution.items():
        sector, side = key
        if not isinstance(source, DilutionBasis):
            raise TypeError('dilution_endpoints must contain DilutionBasis records')
        if sector not in ELEMENTAL_SECTORS or sector_kinds(sector)[side == 'field'] != 'd':
            raise ValueError('dilution provenance must bind an actual D endpoint')
        bindings.setdefault(key, (source.data_id, tuple(range(len(source.vector_labels)))))
        if bindings[key][0] != source.data_id:
            raise ValueError('D endpoint binding differs from its actual dilution data_id')
    normalized = {}
    for key, binding in bindings.items():
        sector, side = key
        if sector not in ELEMENTAL_SECTORS:
            raise ValueError('bilinear endpoint declarations require a meson sector')
        kind = sector_kinds(sector)[side == 'field']
        axis = ('point_' if kind == 'p' else 'mode_') + ('right' if side == 'field' else 'left')
        if (not isinstance(binding, (tuple, list)) or len(binding) != 2
                or not isinstance(binding[0], str) or not binding[0]):
            raise ValueError('basis_endpoints values require (actual_id, ordered_labels)')
        labels = tuple(binding[1])
        extent = sectors[sector].shape[3 + tails[sector].index(axis)]
        if len(labels) != extent:
            raise ValueError('basis endpoint labels differ from the numerical axis extent')
        if key in dilution and len(labels) != len(dilution[key].vector_labels):
            raise ValueError('D provenance differs from its actual vector-axis extent')
        normalized[key] = (binding[0], labels)
    return normalized, dilution


def _selected_endpoint_bindings(sector, names, raw_shape, selections, endpoints):
    result = {}
    if sector not in ELEMENTAL_SECTORS:
        return result
    for side, kind in zip(('bar', 'field'), sector_kinds(sector), strict=True):
        binding = endpoints.get((sector, side))
        if binding is None:
            continue
        axis = ('point_' if kind == 'p' else 'mode_') + ('right' if side == 'field' else 'left')
        positions = _selection_positions(raw_shape[names.index(axis)], selections.get(axis))
        result[axis] = (binding[0], tuple(binding[1][position] for position in positions))
    return result


def _validate_elemental_structure(sector, shape, structure):
    """Check that a producer declaration is meaningful for a sector header."""
    names = ArrayElemental.sector_tail_axes[sector]
    extents = dict(zip(names, shape, strict=True))
    if structure == "point_diagonal":
        if ("point_left" not in extents or "point_right" not in extents
                or extents["point_left"] != extents["point_right"]):
            raise ValueError("point_diagonal requires equal left/right point extents")
    elif structure == "diagonal":
        mode_pairs = (("mode_left", "mode_right"),
                      ("sink_mode", "source_mode"))
        mode_diagonal = any(
            left in extents and right in extents and extents[left] == extents[right]
            for left, right in mode_pairs
        )
        point_color_diagonal = (
            all(name in extents for name in
                ("point_left", "point_right", "color_left", "color_right"))
            and extents["point_left"] == extents["point_right"]
            and extents["color_left"] == extents["color_right"]
        )
        if not mode_diagonal and not point_color_diagonal:
            raise ValueError("diagonal requires equal left/right mode axes or equal point and color axes")


def _selected_elemental_structure(structure, names, raw_shape, selections):
    """Drop a diagonal proof when the requested ordered basis indices differ."""
    extents = dict(zip(names, raw_shape, strict=True))
    if structure == "general":
        return structure
    if structure == "point_diagonal":
        if ("point_left" not in extents or "point_right" not in extents
                or _selection_positions(extents["point_left"], selections.get("point_left"))
                != _selection_positions(extents["point_right"], selections.get("point_right"))):
            return "general"
        return structure
    mode_pairs = (("mode_left", "mode_right"),
                  ("sink_mode", "source_mode"))
    pair = next((item for item in mode_pairs
                 if item[0] in extents and item[1] in extents), None)
    if pair is not None:
        left, right = pair
        if (_selection_positions(extents[left], selections.get(left))
                != _selection_positions(extents[right], selections.get(right))):
            return "general"
        return structure
    point_pair = ("point_left", "point_right")
    color_pair = ("color_left", "color_right")
    if all(name in extents for name in (*point_pair, *color_pair)):
        for left, right in (point_pair, color_pair):
            if (_selection_positions(extents[left], selections.get(left))
                    != _selection_positions(extents[right], selections.get(right))):
                return "general"
        return structure
    return "general"


class ArrayElemental:
    """View dense sectors in canonical (component,momentum,time,...) order.

    Arrays, including read-only NPY memory maps, are not copied. Component
    definitions are explicit; projected row IDs never imply raw-path coverage.
    All supplied sectors must use the SAME declared phase convention.
    With ``selection``, time indices are compact slots in its ordered record.
    ``Lt`` is the reader execution extent Nt; physical_Lt is metadata only.
    """

    @property
    def dilution_identity(self):
        from lqcdweave.contracts.dataset import canonical_json
        if not self.dilution_endpoints:
            return None
        return canonical_json({
            "weight_scheme": self.dilution_weight_scheme,
            "endpoints": [[sector, side, source.document]
                          for (sector, side), source in sorted(self.dilution_endpoints.items())],
        })

    @property
    def background_read_safe(self):
        """The caller keeps the host reader open and its input arrays stable."""
        return all(isinstance(value, np.ndarray) for value in self.sectors.values())

    @property
    def host_resident(self):
        return type(self) is ArrayElemental and host_arrays_resident(self.sectors)

    def read_slab_in_background(self, *args, **kwargs):
        if not self.background_read_safe:
            raise TypeError("background elemental reads require host NumPy arrays")
        return self.read_slab(*args, **kwargs)

    component_type = ElementalComponent
    sector_tail_axes = {sector: elemental_tail_axes(sector) for sector in ELEMENTAL_SECTORS}
    sector_ranks = {sector: 3 + len(axes) for sector, axes in sector_tail_axes.items()}

    def __init__(self, sectors, components, *, phase_convention, cfg=None,
                 data_identity=None, sampling_pool_ids=None, sampling_groups=None,
                 selection=None, sampling_selection=None, momentum_list=None,
                 elemental_weighted=False, point_diagonal_components=(),
                 elemental_structures=None, basis_endpoints=None,
                 dilution_endpoints=None):
        if type(elemental_weighted) is not bool:
            raise TypeError("elemental_weighted must be bool")
        self.elemental_weighted = elemental_weighted
        if phase_convention not in ("midpoint", "anchor", "none"):
            raise ValueError("declare midpoint, anchor, or none phase")
        self.phase_convention, self.cfg = phase_convention, cfg
        self.data_identity = data_identity
        self.sampling_pool_ids = None if sampling_pool_ids is None else dict(sampling_pool_ids)
        self.sampling_groups = {} if sampling_groups is None else dict(sampling_groups)
        for group_id, group in self.sampling_groups.items():
            if not isinstance(group, SamplingPoolSpec) or group.group_id != group_id:
                raise TypeError("sampling_groups must map explicit IDs to SamplingPoolSpec")
        if self.sampling_pool_ids is not None:
            for key, group_id in self.sampling_pool_ids.items():
                if (not isinstance(key, tuple) or len(key) != 4 or
                        key[-1] not in ("bar", "field") or
                        not isinstance(group_id, str) or not group_id):
                    raise ValueError("sampling_pool_ids keys must be (component,momentum,time,bar|field)")
        self._maps, self._closed = [], False
        self.components = tuple(components)
        if not self.components or any(not isinstance(c, self.component_type) for c in self.components):
            raise ValueError(f"explicit {self.component_type.__name__} definitions are required")
        self._index = {c: i for i, c in enumerate(self.components)}
        if len(self._index) != len(self.components):
            raise ValueError("duplicate component definitions")
        if not isinstance(point_diagonal_components, tuple):
            raise TypeError("point_diagonal_components must be an immutable tuple")
        if any(not isinstance(component, self.component_type)
               for component in point_diagonal_components):
            raise TypeError(f"point_diagonal_components must contain {self.component_type.__name__} values")
        if len(set(point_diagonal_components)) != len(point_diagonal_components):
            raise ValueError("point_diagonal_components must not contain duplicates")
        if not set(point_diagonal_components).issubset(self._index):
            raise ValueError("point_diagonal_components must be a subset of components")
        self.point_diagonal_components = point_diagonal_components
        self._point_diagonal_component_set = frozenset(point_diagonal_components)
        self.sectors = dict(sectors)
        if not self.sectors or set(self.sectors) - self.sector_ranks.keys():
            raise ValueError("known elemental sectors are required")
        leading = None
        for sector, a in self.sectors.items():
            rank = self.sector_ranks[sector]
            if a.ndim != rank or a.shape[0] != len(self.components) or a.dtype.kind not in 'fc':
                raise ValueError(f"invalid {sector} shape/dtype")
            if min(a.shape) < 1:
                raise ValueError("empty elemental axes")
            if any(a.shape[3 + index] != 3 for index, name in enumerate(
                    self.sector_tail_axes[sector]) if name.startswith("color")):
                raise ValueError("point color axes must have size three")
            if leading is not None and a.shape[:3] != leading:
                raise ValueError("sector component/momentum/time coverage differs")
            leading = a.shape[:3]
        self.num_momentum, self.Lt = leading[1:]
        self.basis_endpoints, self.dilution_endpoints = _endpoint_declarations(
            self.sectors, self.sector_tail_axes, basis_endpoints, dilution_endpoints,
        )
        has_dilution = bool(self.dilution_endpoints) or any(
            sector in ELEMENTAL_SECTORS and "d" in sector_kinds(sector)
            for sector in self.sectors
        )
        self.dilution_weight_scheme = DILUTION_WEIGHT_SCHEME if has_dilution else None
        self.momenta = None if momentum_list is None else tuple(
            tuple(integer(value, "momentum coordinate") for value in momentum)
            for momentum in momentum_list)
        if self.momenta is not None and (
                len(self.momenta) != self.num_momentum or
                any(len(momentum) != 3 for momentum in self.momenta) or
                len(set(self.momenta)) != len(self.momenta)):
            raise ValueError("momentum_list must match the stored distinct xyz momentum axis")
        if elemental_structures is None:
            elemental_structures = {}
        if not isinstance(elemental_structures, Mapping):
            raise TypeError("elemental_structures must map (sector, component, momentum-slot) keys")
        normalized_structures = {}
        for key, value in elemental_structures.items():
            if not isinstance(key, tuple) or len(key) != 3:
                raise TypeError("elemental_structures keys must be (sector, component, momentum-slot)")
            sector, component, momentum_slot = key
            if sector not in self.sectors:
                raise ValueError("elemental_structures names an absent sector")
            if component not in self._index:
                raise ValueError("elemental_structures names an unknown component")
            momentum_slot = integer(momentum_slot, "elemental structure momentum slot", 0)
            if momentum_slot >= self.num_momentum:
                raise ValueError("elemental structure momentum slot is outside stored coverage")
            structure = normalize_elemental_structure(value)
            if structure != "general":
                _validate_elemental_structure(
                    sector, tuple(self.sectors[sector].shape[3:]), structure,
                )
            normalized_structures[(sector, component, momentum_slot)] = structure
        self.elemental_structures = normalized_structures
        self.Nt = self.Lt
        self.time_selection = None if selection is None else normalize_time_selection(selection)
        if self.time_selection is not None and self.time_selection.Nt != self.Nt:
            raise ValueError("selection Nt differs from elemental time extent")
        self.time_selections = ({} if self.time_selection is None else {"time": self.time_selection})
        self.physical_Lt = self.Lt if self.time_selection is None else self.time_selection.Lt
        self.time_selection_identity = (None if self.time_selection is None else
                                        (("time", self.physical_Lt, self.time_selection.times),))
        if sampling_selection is not None and self.time_selection is None:
            raise ValueError('sampling_selection requires an explicit compact time selection')
        self.sampling_time_selection = (self.time_selection if sampling_selection is None else
                                        normalize_time_selection(sampling_selection, Lt=self.physical_Lt))
        if self.sampling_time_selection is not None and sampling_selection is not None:
            self.time_selection_identity += (('sampling_time', self.physical_Lt, self.sampling_time_selection.times),)

    @classmethod
    def from_npy(cls, files, components, *, axis_order, phase_convention, cfg=None,
                 data_identity=None, sampling_pool_ids=None, sampling_groups=None,
                 selection=None, sampling_selection=None, momentum_list=None,
                 elemental_weighted=False, point_diagonal_components=(),
                 elemental_structures=None, basis_endpoints=None,
                 dilution_endpoints=None):
        """Open .npy files without pickle; axis_order describes first three axes."""
        if tuple(sorted(axis_order)) != ("component", "momentum", "time"):
            raise ValueError("axis_order must permute component,momentum,time")
        arrays, maps = {}, []
        try:
            for sector, path in files.items():
                a = np.load(path, mmap_mode="r", allow_pickle=False)
                maps.append(a)
                axes = tuple(axis_order.index(k) for k in ("component", "momentum", "time"))
                arrays[sector] = a.transpose((*axes, *range(3, a.ndim)))
            result = cls(arrays, components, phase_convention=phase_convention, cfg=cfg,
                         data_identity=data_identity, sampling_pool_ids=sampling_pool_ids,
                         sampling_groups=sampling_groups, selection=selection, sampling_selection=sampling_selection, momentum_list=momentum_list,
                         elemental_weighted=elemental_weighted,
                         point_diagonal_components=point_diagonal_components,
                         elemental_structures=elemental_structures,
                         basis_endpoints=basis_endpoints,
                         dilution_endpoints=dilution_endpoints)
            result._maps = maps
            return result
        except Exception:
            for a in maps:
                a._mmap.close()
            raise

    @classmethod
    def from_binary(cls, files, components, *, shapes, dtypes, axis_order,
                    phase_convention, offsets=None, cfg=None, data_identity=None,
                    sampling_pool_ids=None, sampling_groups=None, selection=None, sampling_selection=None, momentum_list=None,
                    elemental_weighted=False, point_diagonal_components=(),
                    elemental_structures=None, basis_endpoints=None,
                    dilution_endpoints=None):
        """Open headerless C-order sectors with explicit shape and byte order."""
        if tuple(sorted(axis_order)) != ("component", "momentum", "time"):
            raise ValueError("axis_order must permute component,momentum,time")
        arrays, maps = {}, []
        try:
            axes = tuple(axis_order.index(k) for k in ("component", "momentum", "time"))
            for sector, path in files.items():
                shape, dtype = tuple(shapes[sector]), np.dtype(dtypes[sector])
                offset = 0 if offsets is None else offsets.get(sector, 0)
                if (dtype.hasobject or type(offset) is not int or offset < 0 or
                        Path(path).stat().st_size != offset + int(np.prod(shape)) * dtype.itemsize):
                    raise ValueError(f"invalid {sector} binary shape/dtype/offset or file size")
                array = np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset)
                maps.append(array)
                arrays[sector] = array.transpose((*axes, *range(3, array.ndim)))
            result = cls(arrays, components, phase_convention=phase_convention, cfg=cfg,
                         data_identity=data_identity, sampling_pool_ids=sampling_pool_ids,
                         sampling_groups=sampling_groups, selection=selection, sampling_selection=sampling_selection, momentum_list=momentum_list,
                         elemental_weighted=elemental_weighted,
                         point_diagonal_components=point_diagonal_components,
                         elemental_structures=elemental_structures,
                         basis_endpoints=basis_endpoints,
                         dilution_endpoints=dilution_endpoints)
            result._maps = maps
            return result
        except Exception:
            for array in maps:
                array._mmap.close()
            raise

    def close(self):
        """Close owned memory maps. Borrowed views must no longer be used."""
        self.sectors = {}
        for a in self._maps:
            a._mmap.close()
        self._maps, self._closed = [], True

    def _declared_structure(self, component, momentum, sector):
        structure = self.elemental_structures.get(
            (sector, component, int(momentum)), "general",
        )
        legacy_point_diagonal = (
            sector == "p2p" and component in self._point_diagonal_component_set
        )
        return normalize_elemental_structure(structure, legacy_point_diagonal)

    def elemental_structure(self, component, momentum, sector):
        """Expose the normalized producer declaration for composed readers."""
        if component not in self._index:
            raise KeyError("unknown elemental component")
        momentum = integer(momentum, "momentum", 0)
        if momentum >= self.num_momentum:
            raise KeyError("unknown elemental momentum")
        if sector not in self.sectors:
            return "general"
        return self._declared_structure(component, momentum, sector)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def block(self, component, momentum, t, sector, *, usedNe=None, mode_indices=None,
              usedNp=None, point_indices=None):
        if self._closed:
            raise ValueError("elemental reader is closed")
        momentum, t = integer(momentum, "momentum", 0), integer(t, "time", 0)
        if sector not in self.sectors or component not in self._index or momentum >= self.num_momentum or t >= self.Lt:
            raise KeyError(f"missing elemental coverage: {component}, momentum={momentum}, time={t}, sector={sector}")
        value = self.sectors[sector][self._index[component], momentum, t]
        return select_axes(value, self.sector_tail_axes[sector],
                           usedNe=usedNe, mode_indices=mode_indices,
                           usedNp=usedNp, point_indices=point_indices)

    def describe_block(self, component, momentum, t, sector, *, usedNe=None,
                       mode_indices=None, usedNp=None, point_indices=None):
        """Describe one selected block from array headers/shapes only."""
        if self._closed:
            raise ValueError("elemental reader is closed")
        momentum, t = integer(momentum, "momentum", 0), integer(t, "time", 0)
        if (sector not in self.sectors or component not in self._index
                or momentum >= self.num_momentum or t >= self.Lt):
            raise KeyError(f"missing elemental coverage: {component}, momentum={momentum}, time={t}, sector={sector}")
        names = self.sector_tail_axes[sector]
        raw_shape = tuple(self.sectors[sector].shape[3:])
        selections = resolve_selections(names, raw_shape, usedNe=usedNe,
                                        mode_indices=mode_indices, usedNp=usedNp,
                                        point_indices=point_indices)
        shape = tuple(
            len(range(*selections[name].indices(size))) if isinstance(selections.get(name), slice)
            else len(selections[name]) if name in selections else size
            for name, size in zip(names, raw_shape, strict=True)
        )
        selection = (self.time_selection_identity, repr(component), momentum, t, sector,
                     repr(usedNe), repr(mode_indices), repr(usedNp), repr(point_indices))
        declared_structure = self._declared_structure(component, momentum, sector)
        structure = _selected_elemental_structure(
            declared_structure, names, raw_shape, selections,
        )
        axis_bindings = _selected_endpoint_bindings(
            sector, names, raw_shape, selections, self.basis_endpoints,
        )
        if (structure == 'diagonal' and 'mode_left' in axis_bindings
                and 'mode_right' in axis_bindings
                and axis_bindings['mode_left'] != axis_bindings['mode_right']):
            structure = 'general'
        point_pool_ids = {}
        for axis, side in (("point_left", "bar"), ("point_right", "field")):
            if axis in names:
                pool = self.point_pool_id(component, momentum, t, side)
                if pool is not None:
                    point_pool_ids[axis] = pool
        if (structure != "general"
                and point_pool_ids.get("point_left") is not None
                and point_pool_ids.get("point_right") is not None
                and point_pool_ids["point_left"] != point_pool_ids["point_right"]):
            structure = "general"
        if declared_structure != "general":
            selection += ("elemental-structure-v1", declared_structure, structure)
        if self.elemental_weighted:
            selection += ("single-current-point-weighted-v1",)
        if axis_bindings:
            selection += (tuple(axis_bindings.items()),)
        return TensorAxisMetadata(names, shape, self.sectors[sector].dtype,
                                  self.data_identity, repr(selection),
                                  point_pool_ids=point_pool_ids,
                                  axis_bindings=axis_bindings,
                                  structure=structure)

    def describe_slab(self, component, momentum, times_selected, sector, *, usedNe=None,
                      mode_indices=None, usedNp=None, point_indices=None):
        """Describe one ordered multi-time slab using headers and pool metadata only."""
        if self._closed:
            raise ValueError("elemental reader is closed")
        momentum = integer(momentum, "momentum", 0)
        selected_times = _time_sequence(times_selected, self.Lt, "times_selected")
        if (sector not in self.sectors or component not in self._index
                or momentum >= self.num_momentum):
            raise KeyError(f"missing elemental coverage: {component}, momentum={momentum}, sector={sector}")
        names = self.sector_tail_axes[sector]
        raw_shape = tuple(self.sectors[sector].shape[3:])
        selections = resolve_selections(names, raw_shape, usedNe=usedNe,
                                        mode_indices=mode_indices, usedNp=usedNp,
                                        point_indices=point_indices)
        selected_shape = tuple(
            len(range(*selections[name].indices(size))) if isinstance(selections.get(name), slice)
            else len(selections[name]) if name in selections else size
            for name, size in zip(names, raw_shape, strict=True)
        )
        point_pool_ids_by_batch = []
        for time in selected_times:
            pools = {}
            for axis, side in (("point_left", "bar"), ("point_right", "field")):
                if axis in names:
                    pool_id = self.point_pool_id(component, momentum, time, side)
                    if pool_id is not None:
                        pools[axis] = pool_id
            point_pool_ids_by_batch.append(pools)
        common_pools = {}
        if point_pool_ids_by_batch:
            for axis, pool_id in point_pool_ids_by_batch[0].items():
                if all(item.get(axis) == pool_id for item in point_pool_ids_by_batch):
                    common_pools[axis] = pool_id
        selection = (self.time_selection_identity, repr(component), momentum, selected_times, sector,
                     repr(usedNe), repr(mode_indices), repr(usedNp), repr(point_indices))
        declared_structure = self._declared_structure(component, momentum, sector)
        structure = _selected_elemental_structure(
            declared_structure, names, raw_shape, selections,
        )
        axis_bindings = _selected_endpoint_bindings(
            sector, names, raw_shape, selections, self.basis_endpoints,
        )
        if (structure == 'diagonal' and 'mode_left' in axis_bindings
                and 'mode_right' in axis_bindings
                and axis_bindings['mode_left'] != axis_bindings['mode_right']):
            structure = 'general'
        if (structure != "general" and any(
                item.get("point_left") is not None
                and item.get("point_right") is not None
                and item["point_left"] != item["point_right"]
                for item in point_pool_ids_by_batch)):
            structure = "general"
        if declared_structure != "general":
            selection += ("elemental-structure-v1", declared_structure, structure)
        if self.elemental_weighted:
            selection += ("single-current-point-weighted-v1",)
        if axis_bindings:
            selection += (tuple(axis_bindings.items()),)
        return TensorAxisMetadata(
            ("coordinate_batch", *names),
            (len(selected_times), *selected_shape),
            self.sectors[sector].dtype, self.data_identity, repr(selection),
            point_pool_ids=common_pools,
            point_pool_ids_by_batch=tuple(point_pool_ids_by_batch),
            axis_bindings=axis_bindings,
            structure=structure,
        )

    def _selected_point_diagonal(self, component, sector, raw_shape, *,
                                 usedNp=None, point_indices=None):
        """Return only an explicit P/P declaration that survives point selection."""
        if sector != "p2p" or component not in self._point_diagonal_component_set:
            return False
        points = selected_points(
            self.sector_tail_axes[sector], raw_shape,
            usedNp=usedNp, point_indices=point_indices,
        )
        return points["point_left"] == points["point_right"]

    def read_slab(self, component, momentum, times_selected, sector, *, usedNe=None,
                  mode_indices=None, usedNp=None, point_indices=None):
        """Read an ordered slab in one slice/gather, without repeated ``block`` calls.

        Contiguous increasing times return a borrowed view. Unsorted, repeated,
        or gapped times use one gather and return owned storage. The caller must
        keep this reader open while using a borrowed view.
        """
        metadata = self.describe_slab(
            component, momentum, times_selected, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices,
        )
        selected_times = _time_sequence(times_selected, self.Lt, "times_selected")
        momentum = integer(momentum, "momentum", 0)
        source = self.sectors[sector][self._index[component], momentum]
        value = select_slab_axes(
            source, metadata.axes, selected_times, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp,
            point_indices=point_indices,
        )
        if tuple(value.shape) != metadata.shape or np.dtype(value.dtype) != metadata.dtype:
            raise ValueError("elemental slab shape/dtype differs from its metadata")
        return value

    def dilution_basis(self, component, momentum, time, sector, side):
        """Return explicit D provenance; physical time never defines pool identity."""
        if sector not in self.sectors or component not in self._index:
            raise KeyError('missing elemental sector or component')
        if side not in ('bar', 'field'):
            raise ValueError('dilution side must be bar or field')
        return self.dilution_endpoints.get((sector, side))

    def point_pool_id(self, component, momentum, time, side):
        """Return explicitly configured point-pool identity; never infer by time."""
        if side not in ("bar", "field"):
            raise ValueError("point pool side must be bar or field")
        if self.sampling_pool_ids is None:
            return None
        return self.sampling_pool_ids.get((component, int(momentum), int(time), side))

    def sampling_group(self, group_id):
        return self.sampling_groups.get(group_id)

    def iter_time_blocks(self, component, momentum, times_selected, sector, *,
                         usedNe=None, mode_indices=None, usedNp=None, point_indices=None):
        """Yield selected physical times one block at a time."""
        for t in times(times_selected, self.Lt, "times"):
            yield t, self.block(component, momentum, t, sector,
                                usedNe=usedNe, mode_indices=mode_indices,
                                usedNp=usedNp, point_indices=point_indices)

    def read_times(self, component, momentum, times_selected, sector, *,
                   usedNe=None, mode_indices=None, usedNp=None, point_indices=None):
        """Materialize only requested times in the caller's order."""
        selected = times(times_selected, self.Lt, "times")
        values = [self.block(component, momentum, t, sector,
                             usedNe=usedNe, mode_indices=mode_indices,
                             usedNp=usedNp, point_indices=point_indices) for t in selected]
        if type(values[0]).__module__.split(".", 1)[0] == "cupy":
            import cupy
            return cupy.stack(values)
        return np.stack(values)


class LegacyGaugeLinkElemental(ArrayElemental):
    """Old momentum-free spatial gauge-link V/P blocks, phased on demand.

    Raw arrays use (component,time,...) with no momentum axis. The supplied
    point realization and path order determine the Fourier phase at consumption.
    """

    # Inputs are immutable host arrays/point coordinates. The phase path has a
    # conservative per-request temporary-byte descriptor below.
    background_read_safe = True

    def read_slab_in_background(self, *args, **kwargs):
        if not self.background_read_safe:
            raise TypeError("background legacy gauge-link reads require stable host inputs")
        return self.read_slab(*args, **kwargs)

    def __init__(self, sectors, components, *, points, momentum_list,
                 phase_convention="midpoint", cfg=None):
        if phase_convention not in ("midpoint", "anchor"):
            raise ValueError("gauge-link phase must be midpoint or anchor")
        components = tuple(components)
        if any(not isinstance(c, ElementalComponent) or
               not isinstance(c.key, GaugePath) or
               any(step >= 6 for step in c.key.steps) or
               any(abs(a - b) == 3 for a, b in zip(c.key.steps, c.key.steps[1:]))
               for c in components):
            raise ValueError("legacy gauge-link components must be spatial GaugePath components")
        if not isinstance(points, SpatialPointSet):
            raise ValueError("an explicit SpatialPointSet is required")
        momenta = tuple(tuple(integer(x, "momentum component") for x in p)
                        for p in momentum_list)
        if not momenta or any(len(p) != 3 for p in momenta) or len(set(momenta)) != len(momenta):
            raise ValueError("momentum_list must contain distinct xyz triples")
        raw = dict(sectors)
        if not raw or set(raw) - {"v2p", "p2v"} or any(a.ndim != 5 for a in raw.values()):
            raise ValueError("legacy gauge-link input requires five-axis v2p or p2v")
        super().__init__({name: value[:, None, ...] for name, value in raw.items()},
                         components, phase_convention=phase_convention, cfg=cfg,
                         sampling_pool_ids={}, sampling_groups={})
        if points.times != tuple(range(self.Lt)) or points.lattice_size[3] != self.Lt:
            raise ValueError("point coordinates must cover the same physical times")
        for sector, value in self.sectors.items():
            extent = value.shape[-2] if sector == "v2p" else value.shape[-3]
            if extent != points.num_points:
                raise ValueError("point axis differs from the ordered point realization")
        self.points, self.momentum_list = points, momenta
        self.num_momentum = len(momenta)

    @classmethod
    def from_npy(cls, files, components, *, points, momentum_list,
                 axis_order=("time", "component"), phase_convention="midpoint", cfg=None):
        if tuple(sorted(axis_order)) != ("component", "time"):
            raise ValueError("legacy axis_order must permute time,component")
        arrays, maps = {}, []
        try:
            for sector, path in files.items():
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                maps.append(array)
                if array.ndim != 5:
                    raise ValueError("legacy gauge-link NPY must have five axes")
                axes = (axis_order.index("component"), axis_order.index("time"))
                arrays[sector] = array.transpose((*axes, *range(2, array.ndim)))
            result = cls(arrays, components, points=points, momentum_list=momentum_list,
                         phase_convention=phase_convention, cfg=cfg)
            result._maps = maps
            return result
        except Exception:
            for array in maps:
                array._mmap.close()
            raise

    def block(self, component, momentum, t, sector, *, usedNe=None, mode_indices=None,
              usedNp=None, point_indices=None):
        momentum = integer(momentum, "momentum", 0)
        if momentum >= self.num_momentum:
            raise KeyError(f"missing gauge-link momentum index {momentum}")
        value = super().block(component, 0, t, sector, usedNe=usedNe,
                              mode_indices=mode_indices, usedNp=usedNp,
                              point_indices=point_indices)
        point_axis = "point_right" if sector == "v2p" else "point_left"
        selected = selected_points((point_axis,), (self.points.num_points,),
                                   usedNp=usedNp, point_indices=point_indices)[point_axis]
        xyz = self.points.coordinates[t].take(selected, axis=0).astype(np.float64)
        displacement = np.asarray(component.key.displacement[:3], dtype=np.float64)
        origin = xyz - displacement if sector == "v2p" else xyz
        if self.phase_convention == "midpoint":
            origin = origin + displacement / 2
        angle = np.sum(np.asarray(self.momentum_list[momentum]) * origin /
                       np.asarray(self.points.lattice_size[:3]), axis=1)
        phase = np.exp(2j * np.pi * angle)
        return (value * phase[None, :, None] if sector == "v2p" else
                value * phase[:, None, None])

    def describe_slab(self, component, momentum, times_selected, sector, *, usedNe=None,
                      mode_indices=None, usedNp=None, point_indices=None):
        momentum = integer(momentum, "momentum", 0)
        if momentum >= self.num_momentum:
            raise KeyError(f"missing gauge-link momentum index {momentum}")
        metadata = super().describe_slab(
            component, 0, times_selected, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices,
        )
        return replace(metadata, selection_identity=repr((
            "legacy-gauge-link-slab", metadata.selection_identity, momentum,
            self.phase_convention,
        )))

    def read_slab(self, component, momentum, times_selected, sector, *, usedNe=None,
                  mode_indices=None, usedNp=None, point_indices=None):
        """Read one raw time slab, then apply its momentum phase vectorially."""
        metadata = self.describe_slab(
            component, momentum, times_selected, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices,
        )
        selected_times = _time_sequence(times_selected, self.Lt, "times_selected")
        momentum = integer(momentum, "momentum", 0)
        value = super().read_slab(
            component, 0, selected_times, sector, usedNe=usedNe,
            mode_indices=mode_indices, usedNp=usedNp, point_indices=point_indices,
        )
        point_axis = "point_right" if sector == "v2p" else "point_left"
        selected = selected_points((point_axis,), (self.points.num_points,),
                                   usedNp=usedNp, point_indices=point_indices)[point_axis]
        xyz = self.points.coordinates[np.asarray(selected_times)].take(selected, axis=1).astype(np.float64)
        displacement = np.asarray(component.key.displacement[:3], dtype=np.float64)
        origin = xyz - displacement if sector == "v2p" else xyz
        if self.phase_convention == "midpoint":
            origin = origin + displacement / 2
        angle = np.einsum(
            "d,bpd->bp", np.asarray(self.momentum_list[momentum], dtype=np.float64),
            origin / np.asarray(self.points.lattice_size[:3], dtype=np.float64),
        )
        phase = np.exp(2j * np.pi * angle)
        result = (value * phase[:, None, :, None] if sector == "v2p" else
                  value * phase[:, :, None, None])
        if tuple(result.shape) != metadata.shape or np.dtype(result.dtype) != metadata.dtype:
            raise ValueError("legacy gauge-link slab shape/dtype differs from its metadata")
        return result

    def read_slab_workspace_bytes(self, metadata, **_selection):
        """Bound phase temporaries for a host-only slab read.

        The returned slab and raw gather are accounted for by the caller's
        block bound. This covers the selected int64 XYZ copy, float64 XYZ and
        origin arrays, float64 phase angle, and complex128 phase array.
        """
        if (not isinstance(metadata, TensorAxisMetadata) or not metadata.axes
                or metadata.axes[0] != "coordinate_batch"):
            raise TypeError("legacy gauge-link workspace requires coordinate_batch metadata")
        point_axis = "point_right" if "point_right" in metadata.axes else "point_left"
        if point_axis not in metadata.axes:
            raise ValueError("legacy gauge-link slab lacks its point axis")
        point_count = metadata.shape[metadata.axes.index(point_axis)]
        time_count = metadata.shape[0]
        if type(point_count) is not int or type(time_count) is not int:
            raise TypeError("legacy gauge-link slab shape must contain integer extents")
        return time_count * point_count * 80


class ArrayBaryonElemental(ArrayElemental):
    """Three-mode data: (component,momentum,time,mode0,mode1,mode2)."""

    component_type = BaryonComponent
    sector_ranks = {"vvv": 6}
    sector_tail_axes = {"vvv": ("mode0", "mode1", "mode2")}

    def __init__(self, sectors, components, *, phase_convention="anchor", cfg=None,
                 data_identity=None, sampling_pool_ids=None, sampling_groups=None,
                 selection=None, sampling_selection=None, momentum_list=None,
                 elemental_weighted=False, point_diagonal_components=(),
                 elemental_structures=None, basis_endpoints=None,
                 dilution_endpoints=None):
        if phase_convention not in ("anchor", "none"):
            raise ValueError("baryon phase must be anchor or none")
        if elemental_structures is not None:
            if not isinstance(elemental_structures, Mapping):
                raise TypeError("elemental_structures must map (sector, component, momentum-slot) keys")
            if any(normalize_elemental_structure(structure) != "general"
                   for structure in elemental_structures.values()):
                raise ValueError("three-mode baryon elementals do not support non-general structures")
        super().__init__(sectors, components, phase_convention=phase_convention, cfg=cfg,
                         data_identity=data_identity, sampling_pool_ids=sampling_pool_ids,
                         sampling_groups=sampling_groups, selection=selection, sampling_selection=sampling_selection, momentum_list=momentum_list,
                         elemental_weighted=elemental_weighted,
                         point_diagonal_components=point_diagonal_components,
                         elemental_structures=elemental_structures,
                         basis_endpoints=basis_endpoints,
                         dilution_endpoints=dilution_endpoints)
