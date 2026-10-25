"""Compatibility exports for sampling-weight calculations.

The pure mathematical implementation lives in :mod:`lqcdweave.contracts`;
this module remains available for existing expansion callers.
"""

from lqcdweave.contracts.sampling_weights import (
    CompensationScheme,
    blocks_of,
    expected_weight,
    is_finer,
    labelled_set_partitions,
    observed_weight,
    scene_coefficients,
)

__all__ = [
    "labelled_set_partitions",
    "blocks_of",
    "is_finer",
    "scene_coefficients",
    "expected_weight",
    "observed_weight",
    "CompensationScheme",
]
