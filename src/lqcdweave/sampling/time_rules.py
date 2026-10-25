"""Compatibility imports for temporal selection criteria."""

from lqcdweave.sampling.temporal.api import (
    Criterion,
    covers_all_differences,
    covers_differences,
    is_legal,
)

__all__ = ["Criterion", "covers_all_differences", "covers_differences", "is_legal"]
