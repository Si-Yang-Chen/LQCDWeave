"""Actual low/medium parent spaces defining a deterministic high complement."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class ModePartition:
    """Disjoint ordered low modes and complete medium parent groups.

    Labels belong to one actual parent basis, never the flattened D axis.
    The high complement removes all ``projection_modes`` at each time.
    """

    parent_basis_id: str
    low_modes: tuple
    medium_groups: tuple

    def __post_init__(self):
        if not isinstance(self.parent_basis_id, str) or not self.parent_basis_id:
            raise ValueError("mode partition requires an actual parent basis ID")
        low = tuple(self.low_modes)
        groups = tuple(tuple(group) for group in self.medium_groups)
        modes = low + tuple(mode for group in groups for mode in group)
        if (not modes or any(not group for group in groups)
                or any(isinstance(mode, bool) or not isinstance(mode, Integral) or mode < 0
                       for mode in modes) or len(set(modes)) != len(modes)):
            raise ValueError("low and medium parent modes must be distinct nonnegative labels")
        object.__setattr__(self, "low_modes", tuple(int(mode) for mode in low))
        object.__setattr__(self, "medium_groups", tuple(tuple(int(mode) for mode in group)
                                                       for group in groups))

    @property
    def projection_modes(self):
        return self.low_modes + tuple(mode for group in self.medium_groups for mode in group)

    @property
    def document(self):
        return dict(scheme="low-medium-complement-v1", parent_basis_id=self.parent_basis_id,
                    low_modes=list(self.low_modes), medium_groups=[list(g) for g in self.medium_groups])

    @classmethod
    def from_document(cls, document):
        document = dict(document)
        if document.pop("scheme", None) != "low-medium-complement-v1":
            raise ValueError("unsupported three-sector partition scheme")
        return cls(**document)

    def validate_vector(self, endpoint, dilution_source=None):
        if endpoint.kind == "V":
            if endpoint.data_id != self.parent_basis_id or endpoint.indices != self.low_modes:
                raise ValueError("V endpoint must cover the partition's complete ordered low modes")
        elif endpoint.kind == "D":
            if (dilution_source is None or dilution_source.data_id != endpoint.data_id
                    or dilution_source.parent_basis_id != self.parent_basis_id
                    or tuple(g.parent_modes for g in dilution_source.groups) != self.medium_groups
                    or endpoint.indices != dilution_source.ordered_indices):
                raise ValueError("D endpoint must cover the partition's complete medium groups")
