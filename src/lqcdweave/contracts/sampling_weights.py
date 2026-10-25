"""Horvitz-Thompson coefficients for point-coincidence scenes.

A scene's contraction merges subscripts, so it performs a **free sum** over its
blocks -- assignments that put two blocks on the same sampled coordinate are
counted.  The estimator needs the blocks mutually distinct.  So the free sum of a
scene counts every tuple whose exact coincidence pattern is *finer* than that
scene, and the weight that makes the sum unbiased is not the naive
``w(k) = C(M,k)/C(Np,k)``.

Write ``Free_sigma`` for a scene's free sum and ``C_pi`` for the sum over tuples
whose exact pattern is ``pi``.  A tuple with pattern ``pi`` is counted by
``Free_sigma`` exactly when ``sigma`` is coarser than ``pi``, so

    E[Free_sigma] = sum over pi finer than sigma of C_pi / w_E(|pi|)

and therefore

    sum_sigma c(sigma) E[Free_sigma] = sum_pi C_pi / w_E(|pi|)
                                        * sum over sigma finer than pi of c(sigma).

Unbiasedness for every ``pi`` therefore requires

    sum over sigma finer than pi of c(sigma) = w_E(|pi|),

i.e. the zeta inversion ``Z^T c = w`` on the refinement partial order, with
``Z[i][j] = 1`` iff scene ``i`` is finer than scene ``j``.

Two consequences worth stating because they are easy to get wrong:

* At ``r = 1`` the equation collapses to ``c = w_E(1) = M/Np``, which is what the
  old code used, so single-point groups keep their arithmetic exactly.
* ``w(0)`` for an empty group is 1 by definition (one empty tuple, always drawn).

Coefficients are rational and, in the ``w_E`` scheme, often integral.  They are
computed and stored as :class:`fractions.Fraction`: the float residual of the
solve already reaches ``3e-12`` at ``r = 4`` on production-sized ``M``, which is
far too large to bake into a lookup table.
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "labelled_set_partitions",
    "blocks_of",
    "is_finer",
    "scene_coefficients",
    "expected_weight",
    "observed_weight",
    "CompensationScheme",
]


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_population(M, Np) -> tuple[int, int]:
    M = _nonnegative_integer(M, "M")
    Np = _nonnegative_integer(Np, "Np")
    if M == 0 or Np == 0:
        raise ValueError("M and Np must be positive when a point is sampled")
    if Np > M:
        raise ValueError("Np cannot exceed M")
    return M, Np


def labelled_set_partitions(r: int) -> List[Tuple[int, ...]]:
    """Every set partition of ``r`` labelled positions, as restricted growth strings.

    Position ``i`` belongs to block ``blocks[i]``, with blocks numbered by first
    appearance so each partition appears exactly once.  The count is the Bell
    number: 1, 2, 5, 15 for ``r = 1..4``.

    This is the enumeration the Horvitz-Thompson sum runs over.  Integer
    partitions collapse relabellings: at ``r = 3`` the integer partition ``[2, 1]``
    stands for the three distinct scenes ``(0,0,1)``, ``(0,1,0)`` and ``(0,1,1)``.
    They share a weight but impose different coincidence patterns, so treating
    them as one term undercounts the sum.
    """
    r = _nonnegative_integer(r, "r")
    partitions: List[Tuple[int, ...]] = []

    def extend(blocks: List[int]) -> None:
        if len(blocks) == r:
            partitions.append(tuple(blocks))
            return
        limit = (max(blocks) + 2) if blocks else 1
        for block in range(limit):
            extend(blocks + [block])

    extend([])
    return partitions


def blocks_of(partition: Sequence[int]) -> List[Tuple[int, ...]]:
    """Group a restricted-growth partition into its blocks, in first-appearance order."""
    grouped: Dict[int, List[int]] = {}
    for position, block in enumerate(partition):
        grouped.setdefault(block, []).append(position)
    return [tuple(positions) for positions in grouped.values()]


def is_finer(fine: Sequence[Sequence[int]], coarse: Sequence[Sequence[int]]) -> bool:
    """Whether every block of ``fine`` sits inside one block of ``coarse``."""
    return all(
        any(set(block) <= set(other) for other in coarse) for block in fine
    )


def expected_weight(M: int, Np: int, k: int) -> Fraction:
    """``w_E(k) = C(M,k)/C(Np,k)``, the expected number of k-tuples a draw contains.

    An empty group needs no coordinate and contributes nothing: there is exactly
    one empty tuple and it is always drawn, so the weight is 1.
    """
    k = _nonnegative_integer(k, "k")
    if k == 0:
        return Fraction(1)
    M, Np = _positive_population(M, Np)
    if k > min(M, Np):
        raise ValueError(
            f"a scene needing {k} distinct coordinates is not estimable from a "
            f"sample of {Np} out of {M}: its inclusion probability is zero"
        )
    numerator = Fraction(1)
    denominator = Fraction(1)
    for step in range(k):
        numerator *= M - step
        denominator *= Np - step
    return numerator / denominator


class CompensationScheme:
    """How the right-hand side of ``Z^T c = w`` is derived (01 SSF6).

    The two schemes differ *only* in the denominator, so they share the scene
    structure and the solver; a scheme is configuration, not an expansion axis.

    ``EXPECTED`` counts how many k-tuples a draw is expected to contain, and is
    design-unbiased for a fixed gauge field (the Horvitz-Thompson result).
    ``OBSERVED`` uses the ordered distinct tuples reported for the current selected
    point set.  This ratio is generally biased.  Supplying observed counts does not
    make this module support a non-uniform or correlated sampling design; those
    designs need their actual joint inclusion probabilities.
    """

    EXPECTED = "expected"
    OBSERVED = "observed"


def observed_weight(M, Np, k, observed_count):
    """The scheme-O right-hand side for one scene: ``C(M,k) / observed_count``.

    This is the observed-frequency counterpart of :func:`expected_weight`, which is
    ``C(M,k) / C(Np,k)``: same numerator, and the denominator counts k-tuples that
    actually occur rather than the number expected.  Feeding the expected count back
    in therefore reproduces scheme E exactly -- a consistency property between the two
    schemes, not evidence that either is right.

    Note the denominator counts ORDERED tuples of distinct coordinates, matching the
    falling factorials in :func:`expected_weight`.  A binomial coefficient is the
    unordered count and would not reproduce scheme E.

    ``observed_count`` is the number of ORDERED distinct k-tuples reported by the
    caller.  This low-level function leaves provenance and selection-bound checks
    to its caller; :class:`SamplingGroup` expansion checks that counts declared as
    belonging to its current selection do not exceed ``(Np)_k``.

    A zero count is an error, not a small weight: the term did not enter the sample,
    so its inclusion probability has no estimate and the weight is undefined.
    Returning 0 would silently drop the term and change the target, so this raises.
    """
    k = _nonnegative_integer(k, "k")
    if k == 0:
        return Fraction(1)
    M, Np = _positive_population(M, Np)
    if k > min(M, Np):
        raise ValueError(
            f"a scene needing {k} distinct coordinates is not estimable from a "
            f"sample of {Np} out of {M}: its inclusion probability is zero"
        )
    if isinstance(observed_count, bool):
        raise TypeError("observed_count must be an integer ordered-tuple count")
    if isinstance(observed_count, Integral):
        count = int(observed_count)
    elif isinstance(observed_count, Fraction) and observed_count.denominator == 1:
        # Keep compatibility with callers that form exact falling factorials as
        # Fraction values, while still rejecting fractional tuple counts.
        count = observed_count.numerator
    else:
        raise TypeError("observed_count must be an integer ordered-tuple count")
    if count <= 0:
        raise ValueError(
            f"observed count for k={k} distinct coordinates is {observed_count}: "
            "the term did not enter the sample, so its weight is undefined rather "
            "than zero (F6.4)"
        )
    possible = Fraction(1)
    for step in range(k):
        possible *= M - step
    return possible / count



def scene_coefficients(
    r: int,
    M: int,
    Np: int,
    *,
    weights: Sequence[Fraction] = None,
) -> List[Fraction]:
    """Solve ``Z^T c = w`` for one sampling group with ``r`` point ends.

    Args:
        r: number of point ends in the group (its point-coincidence positions).
        M: number of lattice points, ``L**3``.
        Np: number of sampled points.
        weights: optional right-hand side, one entry per labelled set partition in
            :func:`labelled_set_partitions` order.  Defaults to the expected
            weights ``w_E`` (compensation scheme E).  Passing observed-count
            weights selects scheme O; the scene structure and this solver are
            unchanged either way.

    Returns:
        One :class:`fractions.Fraction` coefficient per labelled set partition, in
        the same order.
    """
    r = _nonnegative_integer(r, "r")
    partitions = labelled_set_partitions(r)
    if r == 0:
        return [Fraction(1)]

    M, Np = _positive_population(M, Np)
    blocks = [blocks_of(partition) for partition in partitions]
    count = len(partitions)

    if weights is None:
        rhs = [expected_weight(M, Np, len(block)) for block in blocks]
    else:
        rhs = [Fraction(w) for w in weights]
        if len(rhs) != count:
            raise ValueError(
                f"weights must have one entry per labelled set partition "
                f"({count} for r={r}), got {len(rhs)}"
            )

    # Z[i][j] = 1 iff partition i is finer than partition j.  The refinement order
    # is a partial order, so the system is triangular after a topological sort;
    # solve it that way in exact rational arithmetic rather than inverting a
    # matrix of floats.
    zeta = [
        [1 if is_finer(blocks[i], blocks[j]) else 0 for j in range(count)]
        for i in range(count)
    ]

    # Column j of Z^T is row j of Z; solve for c from the finest partitions down.
    # Order by the number of blocks: a finer partition has more blocks, and
    # "finer" always implies "at least as many blocks".
    order = sorted(range(count), key=lambda index: -len(blocks[index]))
    coefficients: List[Fraction] = [Fraction(0)] * count
    for i in order:
        # Z^T c = w at row i reads: sum over j of Z[j][i] c(j) = w(i).
        # Z[j][i] = 1 iff j is finer than i, i.e. c(i) plus the already-solved
        # strictly-finer terms.
        total = Fraction(0)
        for j in range(count):
            if j != i and zeta[j][i]:
                total += coefficients[j]
        coefficients[i] = rhs[i] - total

    return coefficients
