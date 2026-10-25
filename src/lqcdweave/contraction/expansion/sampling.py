"""Joint point-sampling scene expansion for explicitly declared pools.

This module only describes sampling scenes and their exact rational
coefficients.  It does not infer a pool from time, offsets, point counts, seeds,
low-mode dimensions, or coordinate overlap, and it does not read sampled data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from numbers import Integral
from types import MappingProxyType
from typing import TypeAlias

from lqcdweave.contracts.sampling_weights import (
    CompensationScheme,
    blocks_of,
    expected_weight,
    labelled_set_partitions,
    observed_weight,
    scene_coefficients,
)


SUPPORTED_SAMPLING_DESIGN = "simple_random_without_replacement"
ObservedCountKey: TypeAlias = int | tuple[int, ...]


def _freeze_observed_counts(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("observed_counts must map partitions or distinct-point counts to counts")
    copied = {}
    for key, count in value.items():
        if isinstance(key, bool):
            raise TypeError("observed-count keys must be a partition tuple or integer k")
        if isinstance(key, Integral):
            key = int(key)
        elif isinstance(key, (tuple, list)):
            key = tuple(key)
            if any(isinstance(item, bool) or not isinstance(item, Integral) for item in key):
                raise TypeError("partition keys must contain integer block labels")
            key = tuple(int(item) for item in key)
        else:
            raise TypeError("observed-count keys must be a partition tuple or integer k")
        if isinstance(count, bool):
            raise TypeError("observed counts must be integer ordered-tuple counts")
        if isinstance(count, Integral):
            count = int(count)
        elif isinstance(count, Fraction) and count.denominator == 1:
            count = count.numerator
        else:
            raise TypeError("observed counts must be integer ordered-tuple counts")
        copied[key] = count
    return MappingProxyType(copied)


@dataclass(frozen=True)
class SamplingGroup:
    """One declared point pool and the P-end labels sampled from that pool.

    ``group_id`` is the pool identity.  All listed legs use this single random
    point set, even when their physical endpoints occur at different times.
    Distinct group identities are not assumed independent: callers must declare
    joint independence across all nonempty groups to :func:`expand_sampling_scenes`.

    The supported non-empty design is fixed-size simple random sampling without
    replacement over ``spatial_sites`` sites.  ``sampled_points`` is the current
    selected count for this expansion, not a larger backing block's capacity.
    ``observed_counts`` values count ordered distinct tuples in that current
    selection, keyed by an RGS partition tuple or by its number of blocks.
    """

    group_id: str
    legs: tuple[str, ...]
    spatial_sites: int
    sampled_points: int
    compensation: str = CompensationScheme.EXPECTED
    observed_counts: Mapping[ObservedCountKey, int] | None = None
    sampling_design: str = SUPPORTED_SAMPLING_DESIGN

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("sampling group needs an explicit nonempty identity")
        if isinstance(self.legs, str):
            raise TypeError("legs must be an iterable of leg labels, not one string")
        try:
            legs = tuple(self.legs)
        except TypeError as exc:
            raise TypeError("legs must be an iterable of leg labels") from exc
        if any(not isinstance(leg, str) or not leg for leg in legs):
            raise ValueError("sampling legs must have nonempty string labels")
        if len(set(legs)) != len(legs):
            raise ValueError("sampling legs must be unique within a group")
        object.__setattr__(self, "legs", legs)

        sizes = (self.spatial_sites, self.sampled_points)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in sizes):
            raise TypeError("spatial_sites and sampled_points must be integers")
        spatial_sites, sampled_points = (int(value) for value in sizes)
        if spatial_sites < 0 or sampled_points < 0:
            raise ValueError("spatial_sites and sampled_points must be non-negative")
        if sampled_points > spatial_sites:
            raise ValueError("sampled_points cannot exceed spatial_sites")
        if legs and (spatial_sites == 0 or sampled_points == 0):
            raise ValueError("a nonempty sampling group needs positive site and sample counts")
        object.__setattr__(self, "spatial_sites", spatial_sites)
        object.__setattr__(self, "sampled_points", sampled_points)

        if self.compensation not in (CompensationScheme.EXPECTED, CompensationScheme.OBSERVED):
            raise ValueError("compensation must be 'expected' or 'observed'")
        if self.sampling_design != SUPPORTED_SAMPLING_DESIGN:
            raise NotImplementedError(
                f"sampling design {self.sampling_design!r} is unsupported; this module "
                f"implements only {SUPPORTED_SAMPLING_DESIGN!r}"
            )
        object.__setattr__(self, "observed_counts", _freeze_observed_counts(self.observed_counts))


@dataclass(frozen=True)
class SamplingScene:
    """A symbolic free-sum scene and its exact rational coefficient.

    ``partitions`` identifies legs that share a summation label within each pool.
    It does not require different blocks to have different coordinate values:
    the contraction freely sums each block label, so distinct blocks can still
    coincide in an actual assignment.  The zeta-inverted coefficient corrects
    for those finer coincidence patterns.
    """

    point_labels: Mapping[str, str]
    partitions: Mapping[str, tuple[int, ...]]
    coefficient: Fraction

    def __post_init__(self):
        labels = dict(self.point_labels)
        if any(not isinstance(leg, str) or not isinstance(label, str)
               for leg, label in labels.items()):
            raise TypeError("scene point labels must map string leg IDs to string summation labels")
        partitions = {}
        for group_id, partition in self.partitions.items():
            if not isinstance(group_id, str):
                raise TypeError("scene partition keys must be string group IDs")
            frozen_partition = tuple(partition)
            if any(isinstance(block, bool) or not isinstance(block, Integral) or block < 0
                   for block in frozen_partition):
                raise TypeError("scene partitions must contain non-negative integer block labels")
            partitions[group_id] = tuple(int(block) for block in frozen_partition)
        object.__setattr__(self, "point_labels", MappingProxyType(labels))
        object.__setattr__(self, "partitions", MappingProxyType(partitions))
        object.__setattr__(self, "coefficient", Fraction(self.coefficient))


def _group_coefficients(group: SamplingGroup):
    r = len(group.legs)
    partitions = labelled_set_partitions(r)
    if r == 0:
        return partitions, (Fraction(1),)
    if group.compensation == CompensationScheme.EXPECTED:
        coefficients = scene_coefficients(r, group.spatial_sites, group.sampled_points)
        return partitions, tuple(coefficients)

    observed_counts = group.observed_counts
    if observed_counts is None:
        raise ValueError("observed compensation requires counts from the current selected point set")

    valid_keys = set(partitions) | set(range(1, r + 1)) | {0}
    unexpected_keys = set(observed_counts).difference(valid_keys)
    if unexpected_keys:
        raise ValueError(f"observed_counts contains keys outside this group's partitions: {unexpected_keys!r}")

    rhs = []
    for partition in partitions:
        k = len(blocks_of(partition))
        if k == 0:
            rhs.append(Fraction(1))
            continue
        if partition in observed_counts:
            observed = observed_counts[partition]
        elif k in observed_counts:
            observed = observed_counts[k]
        else:
            raise ValueError(
                f"missing current-selection ordered distinct count for group "
                f"{group.group_id!r}, partition={partition}, k={k}"
            )
        weight = observed_weight(group.spatial_sites, group.sampled_points, k, observed)
        current_selection_max = 1
        for step in range(k):
            current_selection_max *= group.sampled_points - step
        if observed > current_selection_max:
            raise ValueError(
                f"observed count {observed} exceeds the {current_selection_max} ordered "
                f"distinct {k}-tuples in the current selection of "
                f"Np={group.sampled_points} points for group {group.group_id!r}"
            )
        rhs.append(weight)
    coefficients = scene_coefficients(r, group.spatial_sites, group.sampled_points, weights=rhs)
    return partitions, tuple(coefficients)


def _freeze_local_weights(local_weights, groups):
    """Validate and freeze per-pool elemental compensation metadata."""
    if local_weights is None:
        return {}
    if not isinstance(local_weights, Mapping):
        raise TypeError("local_weights must map pool IDs to local leg selections")

    groups_by_id = {group.group_id: group for group in groups}
    normalized = {}
    for pool_id, raw_entries in local_weights.items():
        if not isinstance(pool_id, str):
            raise TypeError("local-weight pool IDs must be strings")
        if pool_id not in groups_by_id:
            raise ValueError(f"local_weights contains unknown pool ID {pool_id!r}")
        try:
            entries = tuple(raw_entries)
        except TypeError as exc:
            raise TypeError(
                f"local_weights[{pool_id!r}] must be an iterable of (positions, original_n) entries"
            ) from exc

        group = groups_by_id[pool_id]
        used_positions = set()
        frozen_entries = []
        for entry in entries:
            try:
                positions, original_n = entry
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "each local-weight entry must contain positions and original sampled_points"
                ) from exc
            try:
                positions = tuple(positions)
            except TypeError as exc:
                raise TypeError("local-weight positions must be an iterable of leg indices") from exc
            if not positions:
                raise ValueError("local-weight positions must be nonempty")
            if any(isinstance(position, bool) or not isinstance(position, Integral)
                   for position in positions):
                raise TypeError("local-weight positions must be integer leg indices")
            positions = tuple(int(position) for position in positions)
            if len(set(positions)) != len(positions):
                raise ValueError("local-weight positions must be unique within each entry")
            if any(position < 0 or position >= len(group.legs) for position in positions):
                raise ValueError(
                    f"local-weight positions for pool {pool_id!r} must be in [0, {len(group.legs)})"
                )
            overlap = used_positions.intersection(positions)
            if overlap:
                raise ValueError(
                    f"local-weight entries for pool {pool_id!r} overlap at positions {sorted(overlap)!r}"
                )
            used_positions.update(positions)

            if isinstance(original_n, bool) or not isinstance(original_n, Integral):
                raise TypeError("original sampled_points in local_weights must be an integer")
            original_n = int(original_n)
            if original_n <= 0 or original_n > group.spatial_sites:
                raise ValueError(
                    f"original sampled_points for pool {pool_id!r} must be positive and no larger "
                    f"than spatial_sites={group.spatial_sites}"
                )
            frozen_entries.append((positions, original_n))
        normalized[pool_id] = tuple(frozen_entries)
    return normalized


def _freeze_point_equalities(point_equalities, groups):
    """Validate per-pool equalities between original leg positions."""
    if point_equalities is None:
        return {}
    if not isinstance(point_equalities, Mapping):
        raise TypeError("point_equalities must map pool IDs to position pairs")

    groups_by_id = {group.group_id: group for group in groups}
    normalized = {}
    for pool_id, raw_pairs in point_equalities.items():
        if not isinstance(pool_id, str):
            raise TypeError("point-equality pool IDs must be strings")
        if pool_id not in groups_by_id:
            raise ValueError(f"point_equalities contains unknown pool ID {pool_id!r}")
        try:
            pairs = tuple(raw_pairs)
        except TypeError as exc:
            raise TypeError(
                f"point_equalities[{pool_id!r}] must be an iterable of position pairs"
            ) from exc

        group = groups_by_id[pool_id]
        frozen_pairs = []
        for pair in pairs:
            try:
                positions = tuple(pair)
            except TypeError as exc:
                raise TypeError("each point equality must contain two leg indices") from exc
            if len(positions) != 2:
                raise ValueError("each point equality must contain exactly two leg indices")
            if any(isinstance(position, bool) or not isinstance(position, Integral)
                   for position in positions):
                raise TypeError("point-equality positions must be integer leg indices")
            positions = tuple(int(position) for position in positions)
            if any(position < 0 or position >= len(group.legs) for position in positions):
                raise ValueError(
                    f"point-equality positions for pool {pool_id!r} must be in "
                    f"[0, {len(group.legs)})"
                )
            frozen_pairs.append(positions)
        normalized[pool_id] = tuple(frozen_pairs)
    return normalized


def _quotient_group(group, equalities):
    """Merge declared expected-sampling positions, preserving first appearance."""
    if not equalities or group.compensation != CompensationScheme.EXPECTED:
        return group, tuple(range(len(group.legs)))

    parent = list(range(len(group.legs)))

    def find(position):
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    for left, right in equalities:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    root_to_quotient = {}
    representatives = []
    original_to_quotient = []
    for position in range(len(group.legs)):
        root = find(position)
        quotient_position = root_to_quotient.get(root)
        if quotient_position is None:
            quotient_position = len(representatives)
            root_to_quotient[root] = quotient_position
            representatives.append(position)
        original_to_quotient.append(quotient_position)

    if len(representatives) == len(group.legs):
        return group, tuple(original_to_quotient)

    quotient = SamplingGroup(
        group.group_id,
        tuple(group.legs[position] for position in representatives),
        group.spatial_sites,
        group.sampled_points,
        compensation=group.compensation,
        observed_counts=group.observed_counts,
        sampling_design=group.sampling_design,
    )
    return quotient, tuple(original_to_quotient)


def _quotient_local_entries(local_entries, original_to_quotient):
    """Map already-validated local entries onto quotient positions."""
    return tuple(
        (tuple(dict.fromkeys(original_to_quotient[position] for position in positions)), original_n)
        for positions, original_n in local_entries
    )


def _residual_group_coefficients(group, partitions, local_entries):
    """Remove local elemental weights from the desired group coefficients."""
    if group.compensation != CompensationScheme.EXPECTED:
        raise ValueError("elemental local weighting only supports expected compensation")

    rhs = []
    for partition in partitions:
        residual = expected_weight(
            group.spatial_sites,
            group.sampled_points,
            len(set(partition)),
        )
        for positions, original_n in local_entries:
            local_distinct = len({partition[position] for position in positions})
            residual /= expected_weight(group.spatial_sites, original_n, local_distinct)
        rhs.append(residual)
    return tuple(scene_coefficients(
        len(group.legs),
        group.spatial_sites,
        group.sampled_points,
        weights=rhs,
    ))


def expand_sampling_scenes(
    groups,
    *,
    groups_independent: bool | None = None,
    local_weights=None,
    point_equalities=None,
) -> tuple[SamplingScene, ...]:
    """Enumerate the Cartesian product of scenes for explicit sampling pools.

    ``SamplingGroup.legs`` is the explicit leg-to-pool mapping.  When multiple
    groups are present, callers must set ``groups_independent=True`` to declare
    all nonempty groups jointly independent. Pairwise independence alone is
    insufficient for multiplying inclusion probabilities across three or more
    groups. A correlated or otherwise jointly designed group set is unsupported
    and raises; no relationship is inferred
    from times, offsets, sample sizes, seeds, or matching coordinates.

    The returned ``point_labels`` are symbolic summation labels namespaced by
    pool identity.  They express equalities within a pool partition only; they
    do not assert coordinate inequality between different blocks or pools.
    ``point_equalities`` declares known point-index identities as pairs of
    positions in each original group's ``legs`` tuple. Expected-compensation
    groups are quotiented by these identities before scene enumeration and the
    resulting scenes are lifted back to the original legs. Observed-compensation
    groups ignore these equalities and retain their original scene expansion.
    Empty input, an empty group, and an all-low-mode term each contribute one
    scene with coefficient one.
    """
    try:
        groups = tuple(groups)
    except TypeError as exc:
        raise TypeError("groups must be an iterable of SamplingGroup objects") from exc
    if any(not isinstance(group, SamplingGroup) for group in groups):
        raise TypeError("groups must contain SamplingGroup objects")

    ids = [group.group_id for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError("sampling group IDs must be unique")
    all_legs = [leg for group in groups for leg in group.legs]
    if len(all_legs) != len(set(all_legs)):
        raise ValueError("a P leg may belong to only one sampling group")

    if groups_independent is not None and not isinstance(groups_independent, bool):
        raise TypeError("groups_independent must be True, False, or None")
    sampled_group_count = sum(bool(group.legs) for group in groups)
    if sampled_group_count > 1:
        if groups_independent is None:
            raise ValueError("multiple nonempty sampling groups require an explicit independence declaration")
        if not groups_independent:
            raise NotImplementedError(
                "correlated or jointly designed sampling groups need their joint inclusion "
                "probabilities; only explicitly jointly independent groups are supported"
            )
    elif groups_independent is False:
        raise ValueError("groups_independent=False is only meaningful for multiple groups")

    frozen_local_weights = _freeze_local_weights(local_weights, groups)
    frozen_point_equalities = _freeze_point_equalities(point_equalities, groups)

    choices = []
    for group in groups:
        quotient_group, original_to_quotient = _quotient_group(
            group, frozen_point_equalities.get(group.group_id, ())
        )
        local_entries = _quotient_local_entries(
            frozen_local_weights.get(group.group_id, ()), original_to_quotient
        )
        if local_entries:
            partitions = labelled_set_partitions(len(quotient_group.legs))
            coefficients = _residual_group_coefficients(
                quotient_group, partitions, local_entries
            )
        else:
            partitions, coefficients = _group_coefficients(quotient_group)
        choices.append((group, original_to_quotient, partitions, coefficients))

    scenes = []
    for selected in product(*(range(len(parts)) for _, _, parts, _ in choices)):
        point_labels = {}
        partition_map = {}
        coefficient = Fraction(1)
        for (group, original_to_quotient, partitions, coefficients), index in zip(
            choices, selected, strict=True
        ):
            quotient_partition = partitions[index]
            partition = tuple(
                quotient_partition[position] for position in original_to_quotient
            )
            partition_map[group.group_id] = partition
            coefficient *= coefficients[index]
            for leg, block in zip(group.legs, partition, strict=True):
                point_labels[leg] = f"pool:{group.group_id}:point:{block}"
        scenes.append(SamplingScene(point_labels, partition_map, coefficient))
    return tuple(scenes)


__all__ = [
    "SamplingGroup",
    "SamplingScene",
    "SUPPORTED_SAMPLING_DESIGN",
    "expand_sampling_scenes",
]
