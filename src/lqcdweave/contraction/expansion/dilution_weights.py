"""D-component inclusion weights and equality scenes for explicit groups.

QR columns are exchangeable components of a complete orthonormal rotated
parent basis. Keeping m of n columns includes an ordered tuple containing r
distinct columns with probability (m)_r/(n)_r. The graph sums actual D-column
labels: equality of those labels changes its weight. This is not an inverse
of a Haar-projector tensor moment and never permutes physical graph halves.

The free-sum/equality and local-preweight residual rules are exactly the
existing point-sampling partition algebra, with n parent columns and m saved
D columns instead of spatial sites and sampled points.
"""

from fractions import Fraction
from numbers import Integral

from lqcdweave.contracts.dilution import DilutionGroup
from lqcdweave.contracts.sampling_weights import expected_weight
from .sampling import SamplingGroup, SamplingScene, expand_sampling_scenes


def _group(group):
    if not isinstance(group, DilutionGroup):
        raise TypeError("group must be an explicit DilutionGroup")


def _exact(group):
    return group.branch == "exact" or group.m == group.n


def dilution_component_weight(group, distinct_labels):
    """Return exact H_r for one actual assignment's distinct D labels.

    A diagonal D/D block has r=1 and weight n/m; an off-diagonal same-group
    block has r=2 and weight n(n-1)/(m(m-1)). Independent groups factor their
    own H_r. Legacy unnormalized U(1) has first covariance one and does not
    have the QR column-inclusion law for higher moments.
    """
    _group(group)
    if (isinstance(distinct_labels, bool) or not isinstance(distinct_labels, Integral)
            or distinct_labels < 0):
        raise ValueError("distinct_labels must be a nonnegative integer")
    distinct_labels = int(distinct_labels)
    if _exact(group) or distinct_labels == 0:
        return Fraction(1)
    if group.branch == "u1":
        if distinct_labels > 1:
            raise ValueError("legacy u1 higher-order component inclusion is unsupported")
        return Fraction(1)
    return expected_weight(group.n, group.m, distinct_labels)


def dilution_weight_scenes(group, legs, *, local_entries=(), equalities=(), pool_id="dilution"):
    """Return exact free-sum SamplingScene records for one actual D pool.

    ``legs`` are unique string IDs for the graph's D-component occurrences.
    ``local_entries`` contains ``(positions, original_m)`` per already weighted
    elemental's endpoints from this pool; their H_local_distinct factor is
    removed using actual equality patterns, rather than dividing by (n/m)^k.
    ``equalities`` contains known equal-position pairs, e.g. a proven diagonal
    D/D current. These are quotiented before checking inclusion support, so
    k>m is allowed when the effective number of independent labels is <=m.

    Exact groups yield one unrestricted identity scene without Bell-number
    enumeration or any order limit. The graph's known diagonal support can
    still lower its axes separately. Legacy U(1) permits one occurrence only;
    equalities do not establish a missing higher moment of that phase vector.
    """
    _group(group)
    sampling = SamplingGroup(pool_id, legs, group.n, group.m)
    if group.branch == "u1" and len(sampling.legs) > 1:
        raise ValueError("legacy u1 same-pool higher-order inclusion is unsupported")
    if _exact(group) or group.branch == "u1":
        partition = tuple(range(len(sampling.legs)))
        return (SamplingScene(
            {leg: f"pool:{pool_id}:component:{index}"
             for index, leg in enumerate(sampling.legs)},
            {pool_id: partition}, Fraction(1)),)
    scenes = expand_sampling_scenes(
        (sampling,), local_weights={pool_id: tuple(local_entries)},
        point_equalities={pool_id: tuple(equalities)},
    )
    return tuple(scene for scene in scenes if scene.coefficient)


__all__ = ["dilution_component_weight", "dilution_weight_scenes"]
