"""Factorized single-hadron vertices assembled from stored elemental blocks."""

from dataclasses import dataclass, fields, replace
from collections.abc import Mapping
from copy import copy

import numpy as np

from lqcdweave.contracts.elemental import (
    BaryonComponent, SparsePointBlock, SparsePointRecordMetadata, integer,
)
from lqcdweave.contracts.elemental import GaugePath
from lqcdweave.operators.single_hadron.api import gamma, spin_basis_matrix
from lqcdweave.contracts.correlator.api import (
    TensorAxisMetadata,
    BaryonFactorMetadata,
    BaryonCoreMetadata,
    VertexFactorMetadata,
    MesonCoreMetadata,
    TensorSpec, TensorCoreSpec, CoreComponent,
)


from lqcdweave.contracts.elemental_sectors import (
    ELEMENTAL_SECTORS, sector_layout, reverse_sector, sector_kinds,
)

from .time_coordinates import validate_slot_operator
from lqcdweave.contracts.correlator.vertices import uses_temporal_endpoint_order


_ELEMENTAL_SECTORS = ELEMENTAL_SECTORS


def _check_momentum_binding(reader, operator):
    for term in operator.terms:
        requested = getattr(term, 'momentum_vector', None)
        if requested is None:
            continue
        momenta = getattr(reader, 'momenta', None)
        if momenta is None:
            raise ValueError('table-bound operator requires explicit reader momentum metadata')
        if not 0 <= term.momentum < len(momenta) or tuple(momenta[term.momentum]) != tuple(requested):
            raise ValueError('operator momentum vector differs from the stored momentum order')



@dataclass
class VertexFactor:
    coefficient: complex
    spin: object
    spatial: object
    endpoint_times: tuple
    sector: str

    def dense(self, *, array_module=np):
        xp = array_module
        spatial = self.spatial.to_dense(array_module=xp) if isinstance(self.spatial, SparsePointBlock) else self.spatial
        return self.coefficient * xp.einsum("ab,...->ab...", self.spin, spatial)


class Meson:
    """Consume an ElementalReader and an Operator, without production inputs.

    source=True applies gamma4 Gamma^dagger gamma4 and swaps/conjugates spatial
    endpoints. Mixed sectors are read from the opposite sector in that case.
    phase_convention must match the stored data; it is never silently repaired.
    """

    def __init__(self, elemental, operator, source=False, *, phase_convention="midpoint",
                 boundary="periodic", array_module=np, requested_sectors=None):
        if boundary not in ("periodic", "open"):
            raise ValueError("boundary must be periodic or open")
        if requested_sectors is not None:
            if not isinstance(requested_sectors, tuple) or not requested_sectors:
                raise TypeError("requested_sectors must be a nonempty tuple or None")
            if any(not isinstance(sector, str) for sector in requested_sectors):
                raise TypeError("requested_sectors entries must be strings")
            if len(set(requested_sectors)) != len(requested_sectors):
                raise ValueError("requested_sectors entries must be unique")
            if any(sector not in _ELEMENTAL_SECTORS for sector in requested_sectors):
                raise ValueError("requested_sectors contains an unknown elemental sector")
        self.elemental, self.operator, self.dagger = elemental, operator, bool(source)
        self.phase_convention, self.boundary, self.xp = phase_convention, boundary, array_module
        self.requested_sectors = requested_sectors
        self.reader = elemental if callable(getattr(elemental, "block", None)) else None
        self.usedNe = self.usedNp = None
        self.key = getattr(elemental, "cfg", None)
        self.mode_indices = None

    def with_mode_indices(self, logical_selections):
        """Return a metadata-only V/V view; dagger swaps logical to raw slots."""
        pair = tuple(tuple(s) for s in logical_selections)
        if len(pair) != 2 or any(not s or len(set(s)) != len(s) or
                               any(type(i) is not int or i < 0 for i in s) for s in pair):
            raise ValueError("two nonempty ordered mode selections are required")
        if self.mode_indices is not None:
            raise ValueError("a Meson cannot be selected twice")
        if self.requested_sectors not in (None, ("v2v",)):
            raise NotImplementedError("mode partitioning currently supports dense V/V mesons")
        if any(term.profile is not None for term in self.operator.terms):
            raise NotImplementedError("mode partitioning with spectral profiles is not supported")
        selected = copy(self)
        selected.requested_sectors = ("v2v",)
        selected.mode_indices = pair[::-1] if self.dagger else pair
        return selected

    def with_reader(self, reader, *, configuration=None):
        """Bind an independent view to an explicit reader without loading data."""
        selected = copy(self)
        selected.elemental = selected.reader = reader
        for name in ('_time_slot_tables', '_time_link_tables', '_time_pool_tables',
                     '_physical_offset_mapping', 'endpoint_time_selection'):
            selected.__dict__.pop(name, None)
        selected.key = getattr(reader, "cfg", None) if configuration is None else configuration
        if selected.key is None and reader is self.reader:
            selected.key = self.key
        if getattr(reader, "cfg", None) not in (None, selected.key):
            raise ValueError("reader cfg differs from requested configuration")
        selected._check_reader()
        return selected

    def is_structurally_zero(self, t=0):
        """Prove disjoint V/V mode support from declared diagonal metadata.

        This is a header-only proof, never a numerical zero test. Missing
        structure declarations and overlapping/reordered selections are kept.
        """
        if self.mode_indices is None or not set(self.mode_indices[0]).isdisjoint(self.mode_indices[1]):
            return False
        if self.reader is None:
            return False
        # Validate the actual selections/endpoints before removing anything.
        descriptions = self.describe_factors(t)
        for description in descriptions:
            term = description.operator_term
            if term.coefficient == 0:
                continue
            full = self.reader.describe_block(
                term.component, term.momentum, description.anchor_time % self.reader.Lt,
                description.raw_sector,
            )
            if full.structure != 'diagonal':
                return False
        return True

    def load(self, key, usedNe=None, usedNp=None):
        if callable(getattr(self.elemental, "load", None)):
            loaded = self.elemental.load(key)
            self.reader = loaded if callable(getattr(loaded, "block", None)) else self.elemental
        elif getattr(self.elemental, "cfg", None) not in (None, key):
            raise ValueError("reader cfg differs from requested cfg")
        self.key = key
        self.usedNe = None if usedNe is None else integer(usedNe, "usedNe", 1)
        self.usedNp = None if usedNp is None else integer(usedNp, "usedNp", 1)
        self._check_reader()
        return self

    def _check_reader(self):
        _check_momentum_binding(self.reader, self.operator)
        if self.reader is None or not callable(getattr(self.reader, "block", None)):
            raise ValueError("load an ElementalReader first")
        if self.reader.phase_convention != self.phase_convention:
            raise ValueError("elemental and operator phase conventions differ")

    def available_sectors(self):
        """Return declared logical sectors, or ``None`` for legacy readers.

        The reader capability is metadata-only. Source daggering swaps the two
        mixed sectors just as :meth:`describe_factors` does; it does not infer
        a missing sector from payload shape or a getter.
        """
        if self.reader is None or not callable(getattr(self.reader, "block", None)):
            raise ValueError("load an ElementalReader first")
        describe = getattr(self.reader, "available_sectors", None)
        if not callable(describe):
            return None
        raw_sectors = describe()
        if not isinstance(raw_sectors, tuple):
            raise TypeError("reader available_sectors must return a tuple")
        if not raw_sectors:
            raise ValueError("reader declares no available elemental sectors")
        if any(not isinstance(sector, str) for sector in raw_sectors):
            raise TypeError("reader available sectors must be strings")
        if len(set(raw_sectors)) != len(raw_sectors):
            raise ValueError("reader available sectors must be unique")
        if any(sector not in _ELEMENTAL_SECTORS for sector in raw_sectors):
            raise ValueError("reader declares an unsupported elemental sector")
        if not self.dagger:
            return raw_sectors
        return tuple(reverse_sector(sector) for sector in raw_sectors)

    def term_time_offsets(self, index):
        term = self.operator.terms[index]
        offsets = tuple(term.anchor_time_offset + t for t in term.component.time_offsets)
        return offsets[::-1] if self.dagger else offsets

    def describe_core_factors(self, t, sector="v2v", indices=None, *, usedNe=None, usedNp=None):
        """Group only components whose actual ordered endpoint bindings agree.

        COO and stochastic-D preparation retain their existing specialized
        lowering until their statistical component contracts are migrated.
        """
        requested = range(len(self.operator.terms)) if indices is None else tuple(indices)
        # Keep original term indices for slot/profile tables. Only exact zeros
        # are absent; small nonzero coefficients remain physical components.
        active = tuple(index for index in requested
                       if self.operator.terms[integer(index, "operator term index", 0)].coefficient != 0)
        groups = {}
        for item in self.describe_factors(t, sector, active, usedNe=usedNe, usedNp=usedNp):
            if item.tensor.sparse_layout != 'dense' or item.dilution_endpoints:
                key = ('compatibility', item.term_index)
            else:
                key = (item.logical_sector, item.raw_sector, item.bar_time, item.field_time,
                       item.bar_time_unwrapped, item.field_time_unwrapped, item.endpoint_offsets,
                       item.dagger, item.tensor.shape, item.tensor.axes,
                       repr(dict(item.tensor.axis_bindings)), repr(dict(item.point_pool_ids)),
                       repr(dict(item.point_pool_metadata)), repr(item.mode_indices),
                       repr(dict(item.basis_selections)), item.coordinate_slot_maps[1:],
                       uses_temporal_endpoint_order(item.operator_term, item.current_term))
            groups.setdefault(key, []).append(item)
        result = []
        for key, factors in groups.items():
            if key[0] == 'compatibility' or len(factors) == 1:
                result.extend(factors)
            else:
                values = {field.name: getattr(factors[0], field.name)
                          for field in fields(VertexFactorMetadata)}
                structures = {self._component_structure(item) for item in factors}
                mixed = len(structures) > 1
                if mixed:
                    # A sum only retains support guaranteed by every member.
                    structure = ('point_diagonal' if structures <= {'diagonal', 'point_diagonal'}
                                 and 'p' in factors[0].logical_sector else 'general')
                    values['tensor'] = replace(values['tensor'], structure=structure,
                                               point_diagonal=structure != 'general')
                result.append(MesonCoreMetadata(**values, factors=tuple(factors), mixed_structure=mixed))
        return tuple(result)

    def _component_structure(self, description):
        if description.mode_indices is None or description.tensor.structure != 'general':
            return description.tensor.structure
        # Different ordered selections can turn a raw diagonal into an
        # off-diagonal matching in the selected view. Read headers only.
        term = description.operator_term
        return self.reader.describe_block(term.component, term.momentum,
            description.anchor_time % self.reader.Lt, description.raw_sector).structure

    def _embedding_support(self, description, permutation):
        structure = self._component_structure(description)
        if structure == 'general':
            return ()
        axes = description.tensor.axes
        pairs = ((0, 1),) if sector_layout(description.raw_sector) == 'v2v' else (
            ((0, 2), (1, 3)) if structure == 'diagonal' else ((0, 2),))
        selected = (dict(zip(('mode_left', 'mode_right'), description.mode_indices))
                    if description.mode_indices is not None else {})
        order = permutation or tuple(range(len(axes)))
        support = []
        for left_axis, right_axis in pairs:
            bindings = description.tensor.axis_bindings
            left_binding, right_binding = bindings.get(axes[left_axis]), bindings.get(axes[right_axis])
            if left_binding is not None and right_binding is not None:
                if left_binding[0] != right_binding[0]:
                    raise ValueError('diagonal embedding requires the same actual endpoint basis')
                left, right = left_binding[1], right_binding[1]
            else:
                left = selected.get(axes[left_axis], tuple(range(description.tensor.shape[left_axis])))
                right = selected.get(axes[right_axis], tuple(range(description.tensor.shape[right_axis])))
            positions = {label: index for index, label in enumerate(right)}
            matches = tuple((i, positions[label]) for i, label in enumerate(left) if label in positions)
            # A singleton matching is already the full Cartesian support. Do
            # not mask it or disable the established scalar spin compression.
            if len(matches) == (description.tensor.shape[left_axis]
                                * description.tensor.shape[right_axis]):
                continue
            support.append((order.index(left_axis), order.index(right_axis),
                            tuple(i for i, _ in matches), tuple(j for _, j in matches)))
        return tuple(support)

    def describe_factors(self, t, sector="v2v", indices=None, *, usedNe=None, usedNp=None):
        """Return term/axis metadata without calling ``reader.block`` or ``load``."""
        _check_momentum_binding(self.reader, self.operator)
        time_tables = getattr(self, '_time_slot_tables', None)
        validate_slot_operator(self.reader, self.operator, time_tables)
        t = integer(t, "time", 0)
        if sector not in ELEMENTAL_SECTORS:
            raise ValueError("unknown Meson sector")
        reader = self.reader
        if reader is None:
            raise ValueError("an elemental reader must be bound before Diagram metadata description")
        if reader.phase_convention != self.phase_convention:
            raise ValueError("elemental and operator phase conventions differ")
        declared_boundary = getattr(reader, "path_boundary", None)
        if declared_boundary is not None:
            if declared_boundary not in ("periodic", "open"):
                raise ValueError("elemental reader declares an unsupported path boundary")
            if declared_boundary != self.boundary:
                raise ValueError("Meson boundary differs from registered elemental path boundary")
        if t >= reader.Lt:
            raise ValueError("time is outside elemental coverage")
        requested = range(len(self.operator.terms)) if indices is None else tuple(indices)
        current_terms = tuple(getattr(self.operator, "current_terms", ()))
        current_ranges = tuple(getattr(self.operator, "current_term_ranges", ()))
        term_to_current = {
            index: current_index
            for current_index, group in enumerate(current_ranges)
            for index in group
        }
        result = []
        for index in requested:
            term = self.operator.terms[integer(index, "operator term index", 0)]
            coordinate_slot_maps = ()
            if time_tables is not None:
                coordinate_slot_maps = time_tables[index]
                anchor, bar_time, field_time = (table[t] for table in coordinate_slot_maps)
                if any(value is None for value in (anchor, bar_time, field_time)):
                    raise KeyError('physical anchor or endpoint is missing from the declared time selection')
                anchor_unwrapped = anchor
                bar_time_unwrapped, field_time_unwrapped = bar_time, field_time
            else:
                anchor_unwrapped = t + term.anchor_time_offset
                bar_offset, field_offset = self.term_time_offsets(index)
                bar_time_unwrapped = t + bar_offset
                field_time_unwrapped = t + field_offset
                if self.boundary == "open" and any(
                        endpoint < 0 or endpoint >= reader.Lt
                        for endpoint in (anchor_unwrapped, bar_time_unwrapped, field_time_unwrapped)):
                    raise ValueError("operator endpoint/anchor lies outside open temporal domain")
                anchor = anchor_unwrapped % reader.Lt
                bar_time = bar_time_unwrapped if self.boundary == "open" else bar_time_unwrapped % reader.Lt
                field_time = field_time_unwrapped if self.boundary == "open" else field_time_unwrapped % reader.Lt
            raw_sector = reverse_sector(sector) if self.dagger else sector
            ne = self.usedNe if usedNe is None else usedNe
            if isinstance(ne, Mapping) and "d" not in (raw_sector[0], raw_sector[2]):
                ne = ne.get("V")
            np_count = self.usedNp if usedNp is None else usedNp
            mode_selection = {}
            if "d" in (raw_sector[0], raw_sector[2]) and ne is not None:
                counts = ne if isinstance(ne, Mapping) else {"V": ne}
                exact = {axis: tuple(range(counts[kind.upper()]))
                         for axis, kind in zip(("mode_left", "mode_right"), sector_kinds(raw_sector))
                         if kind != "p" and counts.get(kind.upper()) is not None}
                if exact:
                    mode_selection["mode_indices"] = exact
                ne = None
            if self.mode_indices is not None:
                if raw_sector != "v2v":
                    raise NotImplementedError("selected mode vertices require V/V")
                if ne is not None and any(i >= ne for side in self.mode_indices for i in side):
                    raise ValueError("mode partition exceeds requested Ne coverage")
                ne = None
                mode_selection["mode_indices"] = dict(zip(("mode_left", "mode_right"), self.mode_indices))
            mode_sectors = {s for s in ELEMENTAL_SECTORS if "v" in s or "d" in s}
            point_sectors = {s for s in ELEMENTAL_SECTORS if "p" in s}
            tensor = reader.describe_block(term.component, term.momentum, anchor, raw_sector,
                                           usedNe=ne if raw_sector in mode_sectors else None,
                                           usedNp=np_count if raw_sector in point_sectors else None,
                                           **mode_selection)
            if not isinstance(tensor, TensorAxisMetadata):
                raise TypeError("reader describe_block must return TensorAxisMetadata")
            current_index = term_to_current.get(index)
            current_term = getattr(term, "current_term", None)
            if current_term is None and current_index is not None:
                current_term = current_terms[current_index]
            if current_term is not None:
                bar_endpoint = tuple(current_term.bar_offset)
                field_endpoint = tuple(current_term.field_offset)
                endpoint_offsets = ((field_endpoint, bar_endpoint) if self.dagger
                                    else (bar_endpoint, field_endpoint))
            else:
                bar_endpoint = (0, 0, 0, 0)
                component_key = term.component.key
                field_endpoint = (component_key.displacement if isinstance(component_key, GaugePath)
                                  else (0, 0, 0, term.component.time_offsets[1]))
                endpoint_offsets = ((tuple(field_endpoint), bar_endpoint) if self.dagger
                                    else (bar_endpoint, tuple(field_endpoint)))
            point_pool_ids = {}
            point_pool_metadata = {}
            for side, endpoint in (("bar", bar_time), ("field", field_time)):
                raw_side = ({"bar": "field", "field": "bar"}[side]
                            if self.dagger else side)
                if time_tables is not None:
                    graph_side = ({'bar': 'field', 'field': 'bar'}[side]
                                  if uses_temporal_endpoint_order(term, current_term) else side)
                    point_side = (graph_side == 'bar' and sector[2] == "p") or (graph_side == 'field' and sector[0] == "p")
                    if not point_side:
                        continue
                    pool_time = self._time_pool_tables[index][0 if side == 'bar' else 1][t]
                    if pool_time is None:
                        raise KeyError('sampling endpoint is missing from the declared sampling_selection')
                else:
                    pool_time = endpoint % reader.Lt
                pool_id = reader.point_pool_id(term.component, term.momentum, pool_time, raw_side)
                if pool_id is not None:
                    point_pool_ids[side] = pool_id
                    pool = reader.sampling_group(pool_id)
                    if pool is not None:
                        point_pool_metadata[pool_id] = pool
            link_origin_time = None
            if current_term is not None:
                if time_tables is not None:
                    table = self._time_link_tables[index]
                    link_origin_time = None if table is None else table[t]
                    if table is not None and link_origin_time is None:
                        raise KeyError('physical current link origin is missing from endpoint selection')
                else:
                    link_origin_time = t + current_term.link_origin_offset[3]
            result.append(VertexFactorMetadata(
                index, term, sector, raw_sector, anchor_unwrapped,
                bar_time, field_time, bar_time_unwrapped, field_time_unwrapped,
                endpoint_offsets,
                self.dagger, current_term, current_index, link_origin_time,
                tensor, point_pool_ids, point_pool_metadata, ne, np_count,
                mode_indices=self.mode_indices, coordinate_slot_maps=coordinate_slot_maps,
                endpoint_time_selection=getattr(self, "endpoint_time_selection", None),
                physical_offset_mapping=getattr(self, "_physical_offset_mapping", False),
                elemental_weighted=getattr(reader, "elemental_weighted", False),
                dilution_endpoints=self._dilution_endpoints(
                    term, raw_sector, sector, bar_time, field_time),
                dilution_weight_scheme=getattr(reader, "dilution_weight_scheme", None),
                basis_selections=(mode_selection.get("mode_indices", {})
                                  if "d" in (raw_sector[0], raw_sector[2]) else {}),
            ))
            register = getattr(reader, "register_factor", None)
            if callable(register):
                register(result[-1])
        return tuple(result)

    def _dilution_endpoints(self, term, raw_sector, logical_sector, bar_time, field_time):
        endpoints = {}
        if uses_temporal_endpoint_order(term, getattr(term, 'current_term', None)):
            bar_time, field_time = field_time, bar_time
        for side, kind, time in (("field", logical_sector[0], field_time),
                                 ("bar", logical_sector[2], bar_time)):
            if kind != "d":
                continue
            describe = getattr(self.reader, "dilution_basis", None)
            if not callable(describe):
                raise ValueError("D elemental requires actual dilution source metadata")
            raw_side = (side if self.dagger else
                        {"field": "bar", "bar": "field"}[side])
            source = describe(term.component, term.momentum, time, raw_sector, raw_side)
            if source is None:
                raise ValueError("D elemental requires actual dilution source metadata")
            endpoints[side] = source
        return endpoints

    def read_factor_block(self, description):
        """Read the exact block described earlier; this is only called by runtime."""
        term = description.operator_term
        return self.reader.block(
            term.component, term.momentum, description.anchor_time % self.reader.Lt,
            description.raw_sector,
            usedNe=description.usedNe if any(k != "p" for k in sector_kinds(description.raw_sector)) else None,
            usedNp=description.usedNp if "p" in sector_kinds(description.raw_sector) else None,
            **({"mode_indices": dict(description.basis_selections)}
               if description.basis_selections else
               {"mode_indices": dict(zip(("mode_left", "mode_right"), description.mode_indices))}
               if description.mode_indices is not None else {}),
        )

    def release_factor_block(self, description, raw_block):
        """Release a borrowed sparse record when its execution snapshot is done."""
        if description.raw_sector != "p2p":
            return
        release = getattr(self.reader, "close_sparse_record", None)
        if callable(release):
            term = description.operator_term
            release(term.component, term.momentum,
                    description.anchor_time % self.reader.Lt, raw_block)

    def describe_sparse_record(self, description):
        """Describe one raw sparse P2P record without reading its arrays."""
        if description.logical_sector != "p2p" or description.raw_sector != "p2p":
            raise ValueError("sparse record metadata requires a P2P vertex")
        describe = getattr(self.reader, "describe_sparse_record", None)
        if not callable(describe):
            raise NotImplementedError(
                "P2P metadata declares COO but the bound elemental reader has no sparse record provider"
            )
        term = description.operator_term
        args = (term.component, term.momentum, description.anchor_time % self.reader.Lt)
        selected_describe = getattr(self.reader, "describe_selected_sparse_record", None)
        if description.usedNp is not None and callable(selected_describe):
            metadata = selected_describe(*args, usedNp=description.usedNp)
        else:
            # Legacy providers retain full raw records for the sparse kernel's
            # prefix cut. Physically filtered providers describe the exact cut.
            metadata = describe(*args)
        if not isinstance(metadata, SparsePointRecordMetadata):
            raise TypeError("sparse record provider must return SparsePointRecordMetadata")
        return metadata

    def materialize_unweighted_factor(self, description, raw_block, *, compact_structure=False):
        """Build a dense spin×spatial tensor without applying operator weight.

        Sparse P2P blocks stay sparse and require the dedicated sparse direct
        operation; they are never densified here.
        """
        xp = self.xp
        if isinstance(raw_block, SparsePointBlock) or isinstance(raw_block, Mapping):
            raise NotImplementedError("sparse P2P vertex materialization uses the sparse contraction path")
        term = description.operator_term
        value = raw_block
        if term.profile is not None:
            if description.raw_sector != "v2v":
                raise ValueError("spectral profile requires v2v")
            anchor = description.anchor_time % self.reader.Lt
            original_times = tuple(
                (anchor + offset) % self.reader.Lt
                for offset in term.component.time_offsets
            )
            value = term.profile.apply_block(value, times=original_times,
                                             cfg=self.key, array_module=xp)
        spin = gamma(term.gamma, array_module=xp)
        if self.dagger:
            g5 = gamma(8, array_module=xp)
            spin = g5 @ spin.conj().T @ g5
            permutation = {"v2v": (1, 0), "v2p": (1, 2, 0),
                           "p2v": (2, 0, 1), "p2p": (2, 3, 0, 1)}[sector_layout(description.raw_sector)]
            value = value.conj().transpose(permutation)
        if compact_structure and description.tensor.structure == "diagonal":
            if sector_layout(description.logical_sector) == "v2v":
                value = xp.diagonal(value)
            elif description.logical_sector == "p2p":
                size = value.shape[0] * value.shape[1]
                value = xp.diagonal(value.reshape(size, size)).reshape(value.shape[:2])
        return xp.einsum("ab,...->ab...", spin, value, optimize=False)

    def describe_factor_preparation(self, description, block_id, *, compact_structure=False,
                                    embed_structure=False):
        """Describe the same dense preparation without retaining this binding.

        Spin and selected spectral profile rows are numerical constants;
        elemental data is neither read nor embedded in the description.
        """
        from lqcdweave.contracts.correlator.api import ArrayConstant, TensorPreparation

        term = description.operator_term
        spin = gamma(term.gamma, array_module=np)
        permutation = ()
        if description.dagger:
            g5 = gamma(8, array_module=np)
            spin = g5 @ spin.conj().T @ g5
            permutation = {"v2v": (1, 0), "v2p": (1, 2, 0),
                           "p2v": (2, 0, 1), "p2p": (2, 3, 0, 1)}[sector_layout(description.raw_sector)]
        constants = [ArrayConstant.from_array(spin)]
        cfg = None
        coordinate_constants = ()
        if term.profile is not None:
            if description.raw_sector != "v2v":
                raise ValueError("spectral profile requires v2v")
            profile = term.profile
            if str(self.key) != profile.cfg:
                raise ValueError("profile cfg differs from elemental cfg")
            times = tuple((description.anchor_time + offset) % self.reader.Lt
                          for offset in term.component.time_offsets)
            left, right = profile.left[:, times[0]], profile.right[:, times[1]]
            shape = description.tensor.shape
            if len(shape) != 2 or shape[0] > left.shape[1] or shape[1] > right.shape[1]:
                raise ValueError("profile does not cover the requested block")
            constants += [ArrayConstant.from_array(left[:, :shape[0]]),
                          ArrayConstant.from_array(right[:, :shape[1]]),
                          ArrayConstant.from_array(profile.coefficients)]
            # Elemental read argument 2 is its actual anchor slot. Store only
            # immutable profile definition rows; no production input is read.
            coordinate_constants = tuple((position, 2, tuple(
                (ArrayConstant.from_array(values[:, (anchor + offset) % self.reader.Lt, :extent])
                 if (anchor + offset) % self.reader.Lt < values.shape[1] else None)
                for anchor in range(self.reader.Lt)))
                for position, values, offset, extent in (
                    (1, profile.left, term.component.time_offsets[0], shape[0]),
                    (2, profile.right, term.component.time_offsets[1], shape[1])))
            cfg = profile.cfg
        return TensorPreparation("spin_spatial", (block_id,), tuple(constants),
                                 description.dagger, permutation, cfg,
                                 spatial_structure=("diagonal" if compact_structure and description.tensor.structure == "diagonal"
                                                    else "general"),
                                 coordinate_constants=coordinate_constants,
                                 spatial_support=(self._embedding_support(description, permutation)
                                                  if embed_structure else ()))

    def materialize_sparse_bridge_part(self, description, raw_record):
        """Bind one raw COO record and spin matrix without operator weighting.

        The operator coefficient remains in the Diagram output recipe. Source
        daggering conjugate-transposes the sparse record here, while the
        recipe conjugates the scalar coefficient once.
        """
        if description.logical_sector != "p2p" or description.raw_sector != "p2p":
            raise ValueError("sparse bridge materialization requires a P2P vertex")
        if description.dagger != self.dagger:
            raise ValueError("P2P metadata dagger state differs from its Meson")
        if description.operator_term.profile is not None:
            raise ValueError("spectral profile is supported only for dense V2V factors")
        if not isinstance(raw_record, Mapping):
            raise TypeError("sparse P2P record must be an identity/COO mapping")
        spin = gamma(description.operator_term.gamma, array_module=self.xp)
        record = raw_record
        if self.dagger:
            g5 = gamma(8, array_module=self.xp)
            spin = g5 @ spin.conj().T @ g5
            if raw_record.get("type") == "sparse":
                record = {
                    "type": "sparse",
                    "indices": raw_record["indices"][:, ::-1].copy(),
                    "values": raw_record["values"].conj().swapaxes(-1, -2).copy(),
                }
            elif raw_record.get("type") != "identity":
                raise ValueError("sparse P2P record must be identity or COO")
        return ((spin, 1.0, record),)

    def is_point_split(self):
        return any(self.term_time_offsets(i) != (0, 0) for i in range(len(self.operator.terms)))

    def factors(self, t, sector="v2v", indices=None):
        """Yield owned factors; different endpoint times remain distinct."""
        validate_slot_operator(self.reader, self.operator)
        self._check_reader()
        xp = self.xp
        t = integer(t, "time", 0)
        if t >= self.reader.Lt or sector not in ELEMENTAL_SECTORS:
            raise ValueError("invalid time or sector")
        raw = reverse_sector(sector) if self.dagger else sector
        if indices is None:
            indices = range(len(self.operator.terms))
        for index in indices:
            term = self.operator.terms[index]
            anchor = t + term.anchor_time_offset
            times = tuple(t + v for v in self.term_time_offsets(index))
            if self.boundary == "open" and any(v < 0 or v >= self.reader.Lt for v in (anchor, *times)):
                raise ValueError("operator endpoint/anchor lies outside open temporal domain")
            anchor %= self.reader.Lt
            times = tuple(v % self.reader.Lt for v in times)
            value = self.reader.block(term.component, term.momentum, anchor, raw)
            if isinstance(value, SparsePointBlock):
                if raw != "p2p":
                    raise ValueError("sparse point pairs require p2p sector")
                value = value.copy()
                if self.usedNp is not None:
                    if self.usedNp > min(value.point_shape):
                        raise ValueError("usedNp exceeds stored coverage")
                    valid = (value.rows < self.usedNp) & (value.columns < self.usedNp)
                    value = SparsePointBlock(value.rows[valid], value.columns[valid], value.values[valid],
                                              (self.usedNp, self.usedNp))
            else:
                axes = {"v2v": ("v", "v"), "v2p": ("v", "p", "c"),
                        "p2v": ("p", "c", "v"), "p2p": ("p", "c", "p", "c")}[sector_layout(raw)]
                slices = []
                for axis, kind in enumerate(axes):
                    count = self.usedNe if kind == "v" else self.usedNp if kind == "p" else None
                    if count is not None and count > value.shape[axis]:
                        raise ValueError("requested basis count exceeds stored coverage")
                    slices.append(slice(None, count))
                value = xp.array(value[tuple(slices)], dtype=complex, copy=True)
            if term.profile is not None:
                if raw != "v2v":
                    raise ValueError("spectral profile requires v2v")
                # Profiles are applied in original endpoint order before dagger.
                original_times = tuple((anchor + v) % self.reader.Lt for v in term.component.time_offsets)
                value = term.profile.apply_block(value, times=original_times, cfg=self.key, array_module=xp)
            spin = gamma(term.gamma, array_module=xp)
            c = term.coefficient
            if self.dagger:
                spin = gamma(8, array_module=xp) @ spin.conj().T @ gamma(8, array_module=xp)
                if isinstance(value, SparsePointBlock):
                    value = SparsePointBlock(value.columns, value.rows, value.values.conj().swapaxes(-1, -2), value.point_shape[::-1])
                else:
                    permutation = {"v2v": (1, 0), "v2p": (1, 2, 0), "p2v": (2, 0, 1), "p2p": (2, 3, 0, 1)}[sector_layout(raw)]
                    value = value.conj().transpose(permutation)
                c = c.conjugate()
            yield VertexFactor(c, spin, value, times, sector)

    def get_terms(self, sector, t, indices):
        result, endpoints = None, None
        for factor in self.factors(t, sector, indices):
            if endpoints is not None and factor.endpoint_times != endpoints:
                raise ValueError("cannot sum different endpoint times; consume factors separately")
            endpoints = factor.endpoint_times
            value = factor.dense(array_module=self.xp)
            result = value if result is None else result + value
        if result is None:
            raise ValueError("no operator terms selected")
        return result

    def get(self, t):
        return self.get_terms("v2v", t, range(len(self.operator.terms)))

    def get_v2p(self, t):
        return self.get_terms("v2p", t, range(len(self.operator.terms)))

    def get_p2v(self, t):
        return self.get_terms("p2v", t, range(len(self.operator.terms)))

    def get_p2p(self, t):
        return self.get_terms("p2p", t, range(len(self.operator.terms)))


@dataclass
class BaryonVertexFactor:
    """Owned spatial block and immutable sparse spin coefficients.

    Axis slots retain their quark identity at both ends. field_order specifies
    actual field ordering, so a later Wick layer can determine its own signs.
    """

    coefficient: complex
    spin: object
    spatial: object
    endpoint_times: tuple
    field_order: tuple
    flavors: tuple = ()
    sector: str = "vvv"

    def dense(self, *, array_module=np):
        xp = array_module
        return self.coefficient * xp.einsum("abc,ijk->abcijk", self.spin.dense(array_module=xp), self.spatial)


def _baryon_spin_array(entries):
    """Dense constant shared by B's convenience and neutral preparation APIs."""
    value = np.zeros((4, 4, 4), dtype=np.complex128)
    for a, b, c, coefficient in entries:
        value[a, b, c] = coefficient
    return value


class Baryon:
    """Read three-mode elementals and assemble sparse single-baryon vertices.

    spin_basis explicitly identifies the coefficients returned to consumers.
    source=True is the adjoint with reverse field order, retaining axis slots;
    no Wick signs or propagator contractions are performed here.
    """

    def __init__(self, elemental, operator, source=False, *, phase_convention="anchor",
                 spin_basis="degrand_rossi", array_module=np, boundary="periodic"):
        operator_terms = getattr(operator, "terms", None)
        if operator_terms is None:
            raise TypeError("Baryon binding requires an Operator with ordered terms")
        operator_terms = tuple(operator_terms)
        if not operator_terms:
            raise ValueError("Baryon Operator must contain at least one term")
        for index, term in enumerate(operator_terms):
            if not isinstance(getattr(term, "component", None), BaryonComponent):
                raise TypeError(f"Operator term {index} does not contain a BaryonComponent")
            if getattr(term, "spin", None) is None or not callable(getattr(term.spin, "to_basis", None)):
                raise TypeError(f"Operator term {index} requires a three-index spin tensor")
            if getattr(term, "profile", None) is not None:
                raise ValueError("baryon spectral profiles are not supported")
        spin_basis_matrix(spin_basis)
        if phase_convention not in ("anchor", "none"):
            raise ValueError("baryon phase must be anchor or none")
        if boundary not in ("periodic", "open"):
            raise ValueError("boundary must be periodic or open")
        self.elemental, self.operator, self.dagger = elemental, operator, bool(source)
        self.phase_convention, self.spin_basis, self.xp = phase_convention, spin_basis, array_module
        self.boundary = boundary
        self.reader = elemental if callable(getattr(elemental, "block", None)) else None
        self.key, self.usedNe = getattr(elemental, "cfg", None), None
        self.cores = {}
        self.mode_indices = None

    def with_mode_indices(self, selections):
        """Select original spatial slots, retaining source field-order mapping."""
        selections = tuple(tuple(s) for s in selections)
        if len(selections) != 3 or any(
                not s or len(set(s)) != len(s)
                or any(type(i) is not int or i < 0 for i in s) for s in selections):
            raise ValueError("three nonempty selections of distinct nonnegative positions required")
        if self.mode_indices is not None:
            raise ValueError("Baryon is already mode-selected")
        selected = copy(self)
        selected.mode_indices = selections
        selected.cores = {}
        return selected

    def with_reader(self, reader, *, configuration=None):
        """Bind an independent view; registered cores belong to this reader only."""
        selected = copy(self)
        selected.elemental = selected.reader = reader
        selected.key = getattr(reader, "cfg", None) if configuration is None else configuration
        if selected.key is None and reader is self.reader:
            selected.key = self.key
        selected.cores = {}
        selected._check_reader()
        return selected

    def _check_reader(self):
        _check_momentum_binding(self.reader, self.operator)
        validate_slot_operator(self.reader, self.operator)
        if self.reader is None or not callable(getattr(self.reader, "block", None)):
            raise ValueError("load a baryon elemental reader first")
        if self.reader.phase_convention != self.phase_convention:
            raise ValueError("elemental and vertex phase conventions differ")
        if getattr(self.reader, "cfg", None) not in (None, self.key):
            raise ValueError("reader cfg differs from requested cfg")

    def load(self, key, usedNe=None):
        if callable(getattr(self.elemental, "load", None)):
            loaded = self.elemental.load(key)
            self.reader = loaded if callable(getattr(loaded, "block", None)) else self.elemental
        self.key = key
        if usedNe is None:
            self.usedNe = None
        elif isinstance(usedNe, (int, np.integer)):
            self.usedNe = (integer(usedNe, "usedNe", 1),) * 3
        else:
            self.usedNe = tuple(integer(n, "usedNe", 1) for n in usedNe)
            if len(self.usedNe) != 3:
                raise ValueError("usedNe requires three counts")
        self._check_reader()
        return self

    def describe_factors(self, t, sector="vvv", indices=None, *, usedNe=None):
        """Describe baryon terms from reader headers without reading a block."""
        self._check_reader()
        t = integer(t, "time", 0)
        if sector != "vvv":
            raise ValueError("baryon frontend currently supports vvv only")
        if t < 0:
            raise ValueError("time must be nonnegative")
        requested = range(len(self.operator.terms)) if indices is None else tuple(indices)
        counts = self.usedNe if usedNe is None else usedNe
        if counts is not None:
            if isinstance(counts, (int, np.integer)) and not isinstance(counts, (bool, np.bool_)):
                counts = (integer(counts, "usedNe", 1),) * 3
            else:
                counts = tuple(integer(value, "usedNe", 1) for value in counts)
                if len(counts) != 3:
                    raise ValueError("baryon usedNe requires three mode counts")
        result = []
        for raw_index in requested:
            index = integer(raw_index, "operator term index", 0)
            if index >= len(self.operator.terms):
                raise IndexError("baryon operator term index is outside the operator")
            term = self.operator.terms[index]
            anchor_unwrapped = t + integer(
                getattr(term, "anchor_time_offset", 0), "anchor_time_offset",
            )
            endpoint_offsets = tuple(getattr(term, "endpoint_time_offsets", (0, 0, 0)))
            if (len(endpoint_offsets) != 3
                    or any(type(value) is not int for value in endpoint_offsets)):
                raise ValueError("baryon endpoint_time_offsets must contain three integer offsets")
            if any(endpoint_offsets):
                raise NotImplementedError(
                    "the bound Baryon reader exposes equal-time three-leg elementals only"
                )
            if self.boundary == "open":
                if anchor_unwrapped < 0 or anchor_unwrapped >= self.reader.Lt:
                    raise ValueError("baryon operator anchor lies outside open temporal coverage")
                anchor = anchor_unwrapped
            else:
                anchor = anchor_unwrapped % self.reader.Lt
            selection = {"usedNe": counts}
            if self.mode_indices is not None:
                if counts is not None and any(max(s) >= n for s, n in zip(self.mode_indices, counts)):
                    raise ValueError("requested Ne does not cover selected baryon positions")
                selection = {"mode_indices": dict(zip(("mode0", "mode1", "mode2"), self.mode_indices))}
            tensor = self.reader.describe_block(
                term.component, term.momentum, anchor, sector, **selection,
            )
            if not isinstance(tensor, TensorAxisMetadata):
                raise TypeError("baryon reader describe_block must return TensorAxisMetadata")
            if tensor.axes != ("mode0", "mode1", "mode2") or len(tensor.shape) != 3:
                raise ValueError("baryon elemental axes must be ordered mode0, mode1, mode2")
            if min(tensor.shape) < 1:
                raise ValueError("baryon elemental mode axes must be nonempty")
            result.append(BaryonFactorMetadata(
                id(self), index, term, anchor, tensor, counts, self.dagger,
                (2, 1, 0) if self.dagger else (0, 1, 2),
                term.coefficient.conjugate() if self.dagger else term.coefficient,
                endpoint_offsets, (anchor, anchor, anchor), self.mode_indices,
            ))
        return tuple(result)

    def read_factor_block(self, description):
        """Read the raw three-mode elemental block deferred by describe_factors."""
        self._validate_factor_description(description)
        term = description.operator_term
        selection = {"usedNe": description.usedNe}
        if description.mode_indices is not None:
            selection = {"mode_indices": dict(zip(("mode0", "mode1", "mode2"), description.mode_indices))}
        return self.reader.block(
            term.component, term.momentum, description.time, "vvv",
            **selection,
        )

    def materialize_unweighted_factor(self, description, raw_block):
        """Make one rank-six spin×mode vertex, leaving its coefficient outside."""
        spatial = self.materialize_unweighted_elemental(description, raw_block)
        spin_array = self.materialize_spin_factor(description)
        return self.xp.einsum("abc,ijk->abcijk", spin_array, spatial, optimize=False)

    def describe_spin_entries(self, description):
        """Return exact immutable three-leg coefficients in the bound spin basis.

        Source adjoint and basis conversion remain operator conventions owned
        by B. The caller receives only neutral scalar entries, never the O type.
        No elemental or propagator block is read here.
        """
        self._validate_factor_description(description)
        spin = description.operator_term.spin.to_basis(self.spin_basis)
        if self.dagger:
            spin = spin.source()
        return tuple(tuple(entry) for entry in spin.entries)

    def register_core(self, descriptions, spin_keys, elemental_keys, *, weighted=True,
                      _spin_entries=None):
        """Own a sum of spin×elemental components with identical leg bindings.

        Expansion supplies the registered leaf names; this method reads no data.
        The planner consumes the resulting neutral descriptor, never this owner.
        Expansion may pass B's already described spin entries for this call.
        """
        descriptions = tuple(descriptions)
        spin_keys, elemental_keys = tuple(spin_keys), tuple(elemental_keys)
        if not descriptions or len(descriptions) != len(spin_keys) or len(descriptions) != len(elemental_keys):
            raise ValueError("one spin and elemental identity is required per core component")
        if _spin_entries is not None:
            _spin_entries = tuple(_spin_entries)
            if len(_spin_entries) != len(descriptions):
                raise ValueError("one described spin entry set is required per core component")
        binding = None
        components = []
        for index, (description, spin_key, elemental_key) in enumerate(
                zip(descriptions, spin_keys, elemental_keys, strict=True)):
            self._validate_factor_description(description)
            signature = self._core_binding(description)
            if binding is not None and signature != binding:
                raise ValueError("core components must use identical ordered propagator bindings")
            binding = signature
            components.append(CoreComponent((spin_key, elemental_key),
                description.coefficient if weighted else 1,
                self.describe_spin_entries(description) if _spin_entries is None
                else _spin_entries[index]))
        core = TensorCoreSpec("baryon", ("spin0", "spin1", "spin2", "laplace_vector0",
            "laplace_vector1", "laplace_vector2"),
            TensorSpec((4, 4, 4, *descriptions[0].tensor.shape), np.complex128), tuple(components))
        return self.cores.setdefault(core.identity, core)

    @staticmethod
    def _core_binding(description):
        # OperatorTerm leaves flavours to the diagram's explicit leg bindings.
        flavors = (tuple(getattr(description.operator_term, "flavors", ())),
                   description.operator_term.component.flavors)
        return (description.endpoint_times, description.field_order, description.mode_indices,
                flavors, description.tensor.shape,
                tuple(sorted(description.tensor.axis_bindings.items())), description.dagger)

    def describe_core_factors(self, t, *, usedNe=None):
        """Describe complete operator cores grouped by their actual leg bindings."""
        groups = {}
        for description in self.describe_factors(t, usedNe=usedNe):
            groups.setdefault(self._core_binding(description), []).append(description)
        result = []
        for factors in groups.values():
            values = {item.name: getattr(factors[0], item.name) for item in fields(BaryonFactorMetadata)}
            values["coefficient"] = 1
            result.append(BaryonCoreMetadata(**values, factors=tuple(factors)))
        return tuple(result)

    def materialize_spin_factor(self, description, *, entries=None):
        """Construct dense fallback from coefficients already described by B.

        Supplying the immutable description avoids repeating basis/adjoint
        conversion during execution. Omitting it preserves the convenience API.
        """
        self._validate_factor_description(description)
        if entries is None:
            entries = self.describe_spin_entries(description)
        return self.xp.asarray(_baryon_spin_array(entries))

    def describe_spin_preparation(self, description, *, entries=None):
        """Freeze the same B-owned spin convention as a neutral constant."""
        from lqcdweave.contracts.correlator.api import ArrayConstant, TensorPreparation
        entries = self.describe_spin_entries(description) if entries is None else entries
        return TensorPreparation('constant', constants=(ArrayConstant.from_array(
            _baryon_spin_array(entries)),))

    def describe_elemental_preparation(self, description, block_id):
        """Describe the existing ordered Phi cast/source conjugate, without IO.

        Explicit computation dtype owns the result even for an identity sink.
        X can then adopt it after completion instead of snapshotting it twice.
        """
        from lqcdweave.contracts.correlator.api import TensorPreparation
        self._validate_factor_description(description)
        return TensorPreparation('identity', (block_id,), conjugate=description.dagger,
                                 compute_dtype='complex128')

    def materialize_unweighted_elemental(self, description, raw_block):
        """Own the ordered three-mode block and apply source spatial adjoint."""
        self._validate_factor_description(description)
        actual_shape = tuple(getattr(raw_block, "shape", ()))
        actual_dtype = np.dtype(getattr(raw_block, "dtype", object))
        if actual_shape != description.tensor.shape or actual_dtype != description.tensor.dtype:
            raise ValueError("baryon elemental block differs from its header metadata")
        spatial = self.xp.array(raw_block, dtype=complex, copy=True)
        if self.dagger:
            spatial = spatial.conj()
        return spatial

    def _validate_factor_description(self, description):
        self._check_reader()
        if (not isinstance(description, BaryonFactorMetadata)
                or description.owner_id != id(self)
                or description.dagger != self.dagger):
            raise TypeError("factor description belongs to a different Baryon binding")
        if (description.term_index >= len(self.operator.terms)
                or self.operator.terms[description.term_index] is not description.operator_term):
            raise ValueError("baryon factor description no longer matches its operator term")

    def factors(self, t, sector="vvv", indices=None):
        self._check_reader()
        t = integer(t, "time", 0)
        if sector != "vvv":
            raise ValueError("baryon frontend currently supports vvv only")
        cache = {}
        for index in range(len(self.operator.terms)) if indices is None else indices:
            term = self.operator.terms[integer(index, "term index", 0)]
            offsets = tuple(getattr(term, "endpoint_time_offsets", (0, 0, 0)))
            if len(offsets) != 3 or any(type(value) is not int for value in offsets):
                raise ValueError("baryon endpoint_time_offsets must contain three integer offsets")
            if any(offsets):
                raise NotImplementedError(
                    "the bound Baryon reader exposes equal-time three-leg elementals only"
                )
            anchor_unwrapped = t + integer(
                getattr(term, "anchor_time_offset", 0), "anchor_time_offset",
            )
            if self.boundary == "open":
                if anchor_unwrapped < 0 or anchor_unwrapped >= self.reader.Lt:
                    raise ValueError("baryon operator anchor lies outside open temporal coverage")
                anchor = anchor_unwrapped
            else:
                anchor = anchor_unwrapped % self.reader.Lt
            key = (term.component, term.momentum, anchor)
            # Reuse adjacent identical spatial components; bound cache to one block.
            if key not in cache:
                selection = {} if self.usedNe is None else {"usedNe": self.usedNe}
                if self.mode_indices is not None:
                    if self.usedNe is not None and any(max(s) >= n for s, n in zip(self.mode_indices, self.usedNe)):
                        raise ValueError("requested Ne does not cover selected baryon positions")
                    selection = {"mode_indices": dict(zip(("mode0", "mode1", "mode2"), self.mode_indices))}
                raw = self.reader.block(term.component, term.momentum, anchor, sector, **selection)
                if raw.ndim != 3 or min(raw.shape) < 1:
                    raise ValueError("baryon elemental requires three nonempty mode axes")
                counts = raw.shape if self.mode_indices is not None or self.usedNe is None else self.usedNe
                if any(n > size for n, size in zip(counts, raw.shape)):
                    raise ValueError("usedNe exceeds stored coverage")
                spatial = self.xp.array(raw[tuple(slice(n) for n in counts)], dtype=complex, copy=True)
                if self.dagger:
                    spatial = spatial.conj()
                cache = {key: spatial}
            spin = term.spin.to_basis(self.spin_basis)
            coefficient = term.coefficient
            if self.dagger:
                spin, coefficient = spin.source(), coefficient.conjugate()
            yield BaryonVertexFactor(
                coefficient, spin, cache[key], (anchor, anchor, anchor),
                (2, 1, 0) if self.dagger else (0, 1, 2),
                tuple(getattr(term, "flavors", ())),
            )

    def get_terms(self, sector, t, indices):
        result = None
        for factor in self.factors(t, sector, indices):
            value = factor.dense(array_module=self.xp)
            result = value if result is None else result + value
        if result is None:
            raise ValueError("no operator terms selected")
        return result

    def get(self, t):
        """Explicitly materialize (4,4,4,N0,N1,N2); prefer factors for large data."""
        return self.get_terms("vvv", t, range(len(self.operator.terms)))


class Current(Meson):
    """Current vertex reusing Meson factors and Operator endpoint metadata."""


__all__ = [
    "Meson", "Baryon", "Current", "VertexFactor", "BaryonVertexFactor",
    "VertexFactorMetadata", "BaryonFactorMetadata",
]
