"""Header-only vertex descriptions shared by binding and expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from .inputs import TensorAxisMetadata
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.contracts.elemental import GaugePath


def uses_temporal_endpoint_order(term, current_term=None):
    """Graph field/bar axes of a temporal link are physical bar/field axes.

    This is shared by propagator times and P/D pool assignment. The raw
    elemental is left(bar, t)^dagger U_path right(field, t + displacement).
    Source daggering is already represented by the factor's swapped times.
    """
    if getattr(current_term, "temporal_point_split", False):
        return True
    component = getattr(term, "component", None)
    key = getattr(component, "key", None)
    return isinstance(key, GaugePath) and key.displacement[3] != 0


@dataclass(frozen=True)
class VertexFactorMetadata:
    """Metadata for one unweighted Meson operator term.

    This record contains identities, coordinates and tensor metadata only.
    It never owns an elemental array and ``describe_factors`` must not load one.
    The Operator coefficient is kept separately for the output recipe.
    """

    term_index: int
    operator_term: object
    logical_sector: str
    raw_sector: str
    anchor_time: int
    bar_time: int
    field_time: int
    bar_time_unwrapped: int
    field_time_unwrapped: int
    endpoint_offsets: tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
    dagger: bool
    current_term: object
    current_term_index: int | None
    link_origin_time: int | None
    tensor: TensorAxisMetadata
    point_pool_ids: object
    point_pool_metadata: object
    usedNe: object
    usedNp: object
    coefficient_state: str = "unweighted_tensor; OperatorTerm.coefficient belongs to OutputRecipe"
    mode_indices: object = None
    coordinate_slot_maps: tuple[tuple[int | None, ...], ...] = ()
    endpoint_time_selection: TimeSelectionRecord | None = None
    physical_offset_mapping: bool = False
    elemental_weighted: bool = False
    dilution_endpoints: object = field(default_factory=dict)
    basis_selections: object = field(default_factory=dict)
    dilution_weight_scheme: str | None = None

    def __post_init__(self):
        if not isinstance(self.term_index, int) or self.term_index < 0:
            raise ValueError("term_index must be a nonnegative integer")
        if not isinstance(self.tensor, TensorAxisMetadata):
            raise TypeError("tensor must be TensorAxisMetadata")
        if not isinstance(self.logical_sector, str) or not self.logical_sector:
            raise ValueError("logical_sector must be nonempty")
        if not isinstance(self.raw_sector, str) or not self.raw_sector:
            raise ValueError("raw_sector must be nonempty")
        if type(self.dagger) is not bool:
            raise TypeError("dagger must be bool")
        if type(self.elemental_weighted) is not bool:
            raise TypeError("elemental_weighted must be bool")
        tables = tuple(tuple(values) for values in self.coordinate_slot_maps)
        if tables and (len(tables) != 3 or not tables[0]
                       or len({len(values) for values in tables}) != 1
                       or any(value is not None and (type(value) is not int or value < 0)
                              for values in tables for value in values)):
            raise ValueError('vertex coordinate maps require anchor/bar/field slot tables')
        object.__setattr__(self, 'coordinate_slot_maps', tables)
        if type(self.physical_offset_mapping) is not bool:
            raise TypeError('physical_offset_mapping must be bool')
        if tables and not isinstance(self.endpoint_time_selection, TimeSelectionRecord):
            raise TypeError('vertex slot tables require their ordered endpoint selection')
        object.__setattr__(self, "point_pool_ids", MappingProxyType(dict(self.point_pool_ids)))
        object.__setattr__(self, "point_pool_metadata", MappingProxyType(dict(self.point_pool_metadata)))
        object.__setattr__(self, "dilution_endpoints", MappingProxyType(dict(self.dilution_endpoints)))
        object.__setattr__(self, "basis_selections", MappingProxyType({
            axis: tuple(indices) for axis, indices in self.basis_selections.items()
        }))


@dataclass(frozen=True)
class MesonCoreMetadata(VertexFactorMetadata):
    """Ordered components with identical actual endpoint bindings.

    The inherited representative describes connectivity only. Each member
    retains its own reader selection, spin, profile and operator coefficient.
    """
    factors: tuple[VertexFactorMetadata, ...] = field(default=(), kw_only=True)
    mixed_structure: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'factors', tuple(self.factors))
        if not self.factors:
            raise ValueError('meson core requires component descriptions')


@dataclass(frozen=True)
class BaryonFactorMetadata:
    """Header-only description of one unweighted, ordered three-mode vertex."""

    owner_id: int
    term_index: int
    operator_term: object
    time: int
    tensor: TensorAxisMetadata
    usedNe: tuple[int, int, int] | None
    dagger: bool
    field_order: tuple[int, int, int]
    coefficient: complex
    endpoint_time_offsets: tuple[int, int, int] = (0, 0, 0)
    endpoint_times: tuple[int, int, int] = ()
    mode_indices: tuple | None = None

    def __post_init__(self):
        if type(self.owner_id) is not int or self.owner_id < 0:
            raise ValueError("owner_id must be a nonnegative integer")
        if type(self.term_index) is not int or self.term_index < 0:
            raise ValueError("term_index must be a nonnegative integer")
        if type(self.time) is not int or self.time < 0:
            raise ValueError("time must be a nonnegative integer")
        if not isinstance(self.tensor, TensorAxisMetadata):
            raise TypeError("tensor must be TensorAxisMetadata")
        field_order = tuple(self.field_order)
        if len(field_order) != 3 or set(field_order) != {0, 1, 2}:
            raise ValueError("field_order must be a permutation of (0, 1, 2)")
        offsets = tuple(self.endpoint_time_offsets)
        if len(offsets) != 3 or any(type(value) is not int for value in offsets):
            raise ValueError("endpoint_time_offsets must contain three integer offsets")
        endpoint_times = tuple(self.endpoint_times)
        if not endpoint_times:
            endpoint_times = (self.time,) * 3
        if len(endpoint_times) != 3 or any(type(value) is not int or value < 0 for value in endpoint_times):
            raise ValueError("endpoint_times must contain three nonnegative physical times")
        if type(self.dagger) is not bool:
            raise TypeError("dagger must be bool")
        if self.usedNe is not None:
            used_ne = tuple(self.usedNe)
            if len(used_ne) != 3 or any(type(value) is not int or value <= 0 for value in used_ne):
                raise ValueError("usedNe must be three positive integer mode counts or None")
            object.__setattr__(self, "usedNe", used_ne)
        if self.mode_indices is not None:
            selections = tuple(tuple(s) for s in self.mode_indices)
            if len(selections) != 3 or any(
                    not s or len(set(s)) != len(s)
                    or any(type(i) is not int or i < 0 for i in s) for s in selections):
                raise ValueError("mode_indices must select three ordered mode axes")
            if self.tensor.shape != tuple(len(s) for s in selections):
                raise ValueError("selected baryon dimensions differ from mode positions")
            object.__setattr__(self, "mode_indices", selections)
        object.__setattr__(self, "field_order", field_order)
        object.__setattr__(self, "endpoint_time_offsets", offsets)
        object.__setattr__(self, "endpoint_times", endpoint_times)


@dataclass(frozen=True)
class BaryonCoreMetadata(BaryonFactorMetadata):
    """Binding-owned compatible operator components, before source registration."""
    factors: tuple[BaryonFactorMetadata, ...] = field(default=(), kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        factors = tuple(self.factors)
        if not factors or any(not isinstance(item, BaryonFactorMetadata) for item in factors):
            raise ValueError("baryon core metadata requires factor descriptions")
        object.__setattr__(self, "factors", factors)


__all__ = ["VertexFactorMetadata", "MesonCoreMetadata", "BaryonFactorMetadata", "BaryonCoreMetadata"]
