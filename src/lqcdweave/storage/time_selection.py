"""Save and read actual ordered temporal coordinates in the existing NPY format.

These small records do not publish a DatasetSpec or imply production coverage.
The physical lattice extent is bound explicitly on read, never inferred from
the file name or the largest selected coordinate.
"""

from pathlib import Path

import numpy as np

from lqcdweave.contracts.time_selection import (
    TimeSelection, TimeSelectionRecord, positive_integer,
)


def save_time_selection(path, selection: TimeSelection | TimeSelectionRecord) -> None:
    """Create a new integer NPY time table without overwriting an existing file."""
    if not isinstance(selection, (TimeSelection, TimeSelectionRecord)):
        raise TypeError("selection must be a TimeSelection or TimeSelectionRecord")
    times = np.asarray(selection.times, dtype=np.int64)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.save(stream, times, allow_pickle=False)


def load_time_selection(path, *, Lt: int) -> TimeSelectionRecord:
    """Read a one-dimensional integer NPY and retain its exact slot order."""
    Lt = positive_integer(Lt, "Lt")
    with Path(path).open("rb") as stream:
        times = np.lib.format.read_array(stream, allow_pickle=False)
    if times.ndim != 1 or times.dtype.kind not in "iu":
        raise ValueError("saved times must be a one-dimensional integer NPY array")
    return TimeSelectionRecord(Lt, tuple(times))


__all__ = ["save_time_selection", "load_time_selection"]
