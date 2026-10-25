"""Public API for spatial sampling and ordered time-first coordinates."""

from lqcdweave.contracts.spatial_sampling import (
    SamplingPoolSpec, SpatialPointSet, SpatialSamplingDesign,
)

from .points import sample_spatial_points

__all__ = [
    "SamplingPoolSpec",
    "SpatialPointSet",
    "SpatialSamplingDesign",
    "sample_spatial_points",
]
