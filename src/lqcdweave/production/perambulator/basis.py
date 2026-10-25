"""Actual source/sink bases; representation belongs to the data, not calc flags."""

from dataclasses import dataclass

from lqcdweave.contracts.propagator import PropagatorEndpoint


@dataclass(frozen=True)
class VectorBasis:
    """An ordered V or D handle and an optional prefix count.

    A handle's endpoint, registered vector_kind or dilution_source identifies D. Ordinary Laplace
    handles identify V. A custom D handle must supply its actual endpoint.
    No normalization or estimator weight is applied to either representation.
    """

    handle: object
    count: int | None = None
    endpoint: PropagatorEndpoint | None = None


@dataclass(frozen=True)
class PointBasis:
    """Ordered point deltas, optionally projected by a complete high complement."""

    handle: object
    count: int | None = None
    highmode_basis: object = None
    endpoint: PropagatorEndpoint | None = None
