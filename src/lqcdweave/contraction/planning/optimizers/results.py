"""Unique search result records; execution plans remain owned by contracts."""
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from lqcdweave.contracts.correlator.api import SceneStoragePlan

class CandidateMemoryExceeded(MemoryError):
    """A complete candidate fails the final physical memory admission."""


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class SearchResult(Generic[StateT]):
    status: str
    state: StateT | None
    explored: int
    limited: bool = False
    stop_reason: str = field(default="completed", kw_only=True)
    lower_bound_seconds: float | None = field(default=None, kw_only=True)
    statistics: dict = field(default_factory=dict, kw_only=True)


@dataclass(frozen=True)
class StorageSearchResult:
    status: str
    plan: SceneStoragePlan | None
    explored: int
    limited: bool
    attempts: int
    complete_plans: int
    first_solution_seconds: float | None
    rejected_memory: int
    diagnostic: dict
    stop_reason: str = "completed"
    lower_bound_seconds: float | None = None
    statistics: dict = field(default_factory=dict)

    @property
    def state(self):
        return None if self.plan is None else self.plan.state

    @property
    def absolute_gap_seconds(self):
        if self.plan is None or self.lower_bound_seconds is None:
            return None
        return max(0.0, self.plan.seconds - self.lower_bound_seconds)
