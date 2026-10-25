"""Compatibility imports for :mod:`lqcdweave.sampling.temporal.api`."""

from lqcdweave.contracts.time_selection import TimeSelection, positive_integer
from lqcdweave.sampling.temporal.api import (
    DEFAULT_CRITERIA,
    SearchLimitExceeded,
    find_min_loop,
    iter_legal_schemes,
    select_best,
    select_t,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "SearchLimitExceeded",
    "TimeSelection",
    "find_min_loop",
    "iter_legal_schemes",
    "positive_integer",
    "select_best",
    "select_t",
]
