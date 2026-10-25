"""Search-local operation identities and compact candidate records.

Descriptors are immutable operations, never successor snapshots. Top-k records
may additionally own a parent-relative delta supplied by the domain. Python's
dictionary checks equality after hashing, so a hash collision cannot merge
distinct operations. IDs are local to one search and never enter saved plans.
"""
from dataclasses import dataclass, field


def order_ties(space, rows, checkpoint):
    """Order (score, ordinal, operation, cost) rows without changing scores."""
    rows.sort(key=lambda row: (row[0], row[1]))
    tie_break = getattr(space, "tie_break", None)
    if tie_break is None:
        return rows
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end][0] == rows[start][0]:
            end += 1
        if end - start > 1:
            decorated = []
            for row in rows[start:end]:
                checkpoint()
                decorated.append((-tie_break(row[2], checkpoint), row[1], row))
            rows[start:end] = [row for _, _, row in sorted(decorated)]
        start = end
    return rows


class OperationPool:
    def __init__(self):
        self._ids = {}
        self._operations = []

    def intern(self, operation):
        found = self._ids.get(operation)
        if found is not None:
            return found
        identity = len(self._operations)
        self._ids[operation] = identity
        self._operations.append(operation)
        return identity

    def __getitem__(self, identity):
        return self._operations[identity]

    def __len__(self):
        return len(self._operations)


@dataclass(frozen=True, slots=True)
class RankedChoice:
    operation_id: int
    score: float
    ordinal: int
    incremental_seconds: float
    delta: object = field(default=None, compare=False, repr=False)
