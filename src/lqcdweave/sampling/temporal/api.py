"""Public interface for periodic temporal selection and two-point reconstruction.

Time selections are the shared contract object. Searching and ranking, criteria,
objectives, and reconstruction are implemented in the sibling modules.
"""

from lqcdweave.contracts.time_selection import TimeSelection, TimeSelectionRecord

from .criteria import Criterion, covers_all_differences, covers_differences, is_legal
from .objective import exponential_pair_q
from .reconstruction import (
    TwoPointResult,
    TimePairAccumulator,
    periodic_time_difference,
    pair_time_differences,
    count_time_pairs,
    mean_time_pairs,
    reconstruct_two_point,
    reconstruct_two_point_from_coordinates,
)
from .device_reconstruction import (DeviceTimePairAccumulator, estimate_device_accumulator_bytes,
                                    describe_device_accumulator_allocations)
from .search import (
    DEFAULT_CRITERIA,
    SearchLimitExceeded,
    find_min_loop,
    iter_legal_schemes,
    select_best,
    select_t,
    realize_time_selection,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "Criterion",
    "SearchLimitExceeded",
    "TimeSelection",
    "TimeSelectionRecord",
    "TwoPointResult",
    "TimePairAccumulator", "DeviceTimePairAccumulator", "estimate_device_accumulator_bytes",
    "describe_device_accumulator_allocations",
    "periodic_time_difference",
    "pair_time_differences",
    "count_time_pairs",
    "mean_time_pairs",
    "covers_all_differences",
    "covers_differences",
    "exponential_pair_q",
    "find_min_loop",
    "is_legal",
    "iter_legal_schemes",
    "reconstruct_two_point",
    "reconstruct_two_point_from_coordinates",
    "select_best",
    "select_t",
    "realize_time_selection",
]
