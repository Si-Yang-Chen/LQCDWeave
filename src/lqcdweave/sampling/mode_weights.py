"""Mode-pair inclusion weights for explicit without-replacement dilution."""

from numbers import Integral
import numpy as np

from lqcdweave.contracts.elemental import integer


def dilution_pair_weights(total_counts, used_counts):
    """Return diagonal/pair inverse inclusion probabilities, applied once.

    Independent blocks are sampled without replacement. A nontrivial block
    requires at least two sampled modes to estimate its distinct-mode pairs.
    This is not a separable spectral profile.
    """
    total = tuple(integer(v, "total count", 1) for v in total_counts)
    used = (used_counts,) * len(total) if isinstance(used_counts, Integral) else tuple(used_counts)
    used = tuple(integer(v, "used count", 1) for v in used)
    if not total or len(total) != len(used) or any(m > n or (m == 1 and n > 1) for n,m in zip(total,used)):
        raise ValueError("invalid dilution counts or unsampled distinct-mode pairs")
    ratio = np.concatenate([np.full(m,n/m) for n,m in zip(total,used)])
    weights = np.outer(ratio,ratio)
    start = 0
    for n,m in zip(total,used):
        block = weights[start:start+m,start:start+m]
        block.fill(1 if m == 1 else n*(n-1)/(m*(m-1)))
        np.fill_diagonal(block,n/m)
        start += m
    return weights
