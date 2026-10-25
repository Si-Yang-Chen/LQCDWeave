"""Independent boolean legality criteria for periodic time selections."""

from collections.abc import Callable, Iterable
from numbers import Integral

from lqcdweave.contracts.time_selection import TimeSelection

Criterion = Callable[[TimeSelection], bool]


def covers_all_differences(selection: TimeSelection) -> bool:
    """Every periodic time difference has at least one source/sink pair."""
    offsets, loop = selection.offsets, selection.loop
    return len({(b - a) % loop for a in offsets for b in offsets}) == loop


def covers_differences(differences: Iterable[int]) -> Criterion:
    """Build a criterion for an explicitly requested set of time differences."""
    differences = tuple(differences)
    if any(isinstance(d, bool) or not isinstance(d, Integral) for d in differences):
        raise ValueError("time differences must be integers")

    def criterion(selection: TimeSelection) -> bool:
        covered = {(b - a) % selection.loop for a in selection.offsets for b in selection.offsets}
        return all(d % selection.loop in covered for d in differences)

    return criterion


def is_legal(selection: TimeSelection, criteria: Iterable[Criterion]) -> bool:
    """Combine independent criteria with AND; no criteria accepts every selection."""
    for criterion in criteria:
        result = criterion(selection)
        if not isinstance(result, bool):
            raise TypeError("a legality criterion must return bool")
        if not result:
            return False
    return True
