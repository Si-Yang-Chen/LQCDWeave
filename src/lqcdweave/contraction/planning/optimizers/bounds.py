"""Certified remaining-cost callbacks, distinct from heuristic ranking."""
from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class LowerBound:
    """The provider must prove admissibility for this exact, fixed domain.

    Binding by identity prevents accidental reuse after replacing a model or
    operation set. Runtime checks cannot prove admissibility of user callbacks.
    """
    domain: object
    estimate: object
    description: str

    def evaluate(self, space, state):
        if self.domain is not space:
            raise ValueError("lower bound belongs to a different search domain")
        value = self.estimate(state)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError("lower bound must be finite and nonnegative")
        return value


def remaining(bound, space, state):
    return 0.0 if bound is None else bound.evaluate(space, state)
