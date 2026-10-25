"""Read fixed cfg final-use indices without expanding per-row Python objects."""
import numpy as np

from lqcdweave.contracts.correlator.api import ArrayConstant


class _CompactReleaseTable:
    """A current point's read-only integer schedule, never numerical owners.

    Native replay visits end_action in increasing order (including its final
    output boundary); get also preserves repeated/backwards diagnostic reads.
    Only the last range is cached, so retaining a point does not create an
    action-sized dict or one Python list/int pair per release row.
    """
    __slots__ = ('_words', '_rows', '_index', '_stop', '_result', '_release_index', '_release_position')

    def __init__(self, rows):
        self._words = memoryview(rows).cast('B').cast('q')
        self._rows = len(rows)
        self._index, self._stop, self._result = None, 0, ()
        self._release_index, self._release_position = None, 0

    def release(self, index, arrays):
        if self._release_index is not None and index <= self._release_index:
            # Repeated end_action must still repeat deletions (and raise on
            # a missing ID), not silently treat a consumed cursor as success.
            for gid in self.get(index, ()):
                del arrays[gid]
            return
        self._release_index = index
        words, stop, position = self._words, 2 * self._rows, self._release_position
        while position < stop and words[position] < index:
            position += 2
        while position < stop and words[position] == index:
            del arrays[words[position + 1]]
            position += 2
        self._release_position = position

    def get(self, index, default=None):
        if index == self._index:
            return self._result if self._result else default
        words, size = self._words, self._rows
        start = self._stop
        if self._index is not None and index < self._index:
            # Extensions/observers may revisit an earlier boundary. Locate it
            # without consuming or dropping any release, as the old dict did.
            start, stop = 0, size
            while start < stop:
                middle = (start + stop) // 2
                if words[2 * middle] < index:
                    start = middle + 1
                else:
                    stop = middle
        else:
            while start < size and words[2 * start] < index:
                start += 1
        stop = start
        while stop < size and words[2 * stop] == index:
            stop += 1
        self._index, self._stop = index, stop
        self._result = words[2 * start + 1:2 * stop:2] if stop > start else ()
        return self._result if self._result else default


def bind_tensor_releases(record, *, native):
    if record is None:
        return {}
    rows = record.array()
    if (native and type(record) is ArrayConstant and record.dtype == np.dtype('int64').str
            and len(record.shape) == 2 and record.shape[1] == 2
            and not np.any(rows[1:, 0] < rows[:-1, 0])):
        # Exact ArrayConstant.array() is C-contiguous and backed by immutable
        # bytes. Native int64 admission is required for memoryview's q format.
        return _CompactReleaseTable(rows) if len(rows) else {}
    releases = {}
    for index, gid in rows.tolist():
        releases.setdefault(index, []).append(gid)
    return releases
