"""Replaceable geometry objectives; no correlator measurements or I/O."""

from collections.abc import Callable, Iterable
from functools import lru_cache
from math import exp, isfinite, sqrt
from numbers import Integral

from lqcdweave.contracts.time_selection import TimeSelection


def exponential_pair_q(
    *, ell: float, differences: Iterable[int]
) -> Callable[[TimeSelection], float]:
    """Build the reference Q objective (smaller is better).

    Q² is the mean paired-source kernel covariance divided by full-source
    covariance. K(t,u)=exp(-circular_distance(t,u)/ell), including the diagonal.
    Kernel scale and requested differences are explicit research choices.
    """
    if not isfinite(ell) or ell <= 0:
        raise ValueError("ell must be finite and positive")
    differences = tuple(differences)
    if not differences or any(
        isinstance(d, bool) or not isinstance(d, Integral) for d in differences
    ):
        raise ValueError("differences must be a nonempty sequence of integers")

    @lru_cache(maxsize=128)
    def kernel(Lt: int, loop: int):
        full = tuple(exp(-min(d, Lt - d) / ell) for d in range(Lt))
        repeats = Lt // loop
        reduced = tuple(
            sum(full[d + r * loop] for r in range(repeats)) / repeats
            for d in range(loop)
        )
        return reduced, sum(full) / Lt

    def objective(selection: TimeSelection) -> float:
        loop, offsets = selection.loop, selection.offsets
        chosen = set(offsets)
        reduced, baseline = kernel(selection.Lt, loop)
        values = {}
        for d in {d % loop for d in differences}:
            sources = [a for a in offsets if (a + d) % loop in chosen]
            if not sources:
                raise ValueError(f"no paired sources for time difference {d}")
            values[d] = sum(
                reduced[(a - b) % loop] for a in sources for b in sources
            ) / len(sources) ** 2
        return sqrt(sum(values[d % loop] for d in differences) / len(differences) / baseline)

    return objective
