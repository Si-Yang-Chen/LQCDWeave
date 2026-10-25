"""Metadata search contracts, independent of costs, operation kinds and engines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Protocol, TypeVar

StateT = TypeVar("StateT")


class SearchSpace(Protocol[StateT]):
    """Finite graph with nonnegative costs; keys preserve every future choice.

    Feasibility covers the entire history. Finalization is cost-free; charged
    finalization must be an edge. Enumeration must cooperate with deadlines.
    """
    def initial_state(self) -> StateT: ...
    def key(self, state: StateT) -> Hashable: ...
    def successors(self, state: StateT) -> Iterable[StateT]: ...
    def cost(self, state: StateT) -> float: ...
    def is_feasible(self, state: StateT) -> bool: ...
    def finish(self, state: StateT) -> StateT | None: ...


class LayeredSearchSpace(SearchSpace[StateT], Protocol[StateT]):
    """Nonnegative integer layers never increase; every goal is in layer zero."""
    def layer(self, state: StateT) -> int: ...


class CompleteStepSpace(Protocol[StateT]):
    """DFS domain with temporary evaluated edges and compact retained deltas.

    Each yielded edge has operation/state/actions. The operation is hashable
    and describes replay from its source state, including any context needed
    to select the computation. Domains may freeze parent-relative changes for
    top-k edges and apply those without replay; other domains materialize the
    operation from its parent. Engines do not inspect operation kinds or costs.
    Equal complete keys have the same replay descriptors and future incremental
    costs under the session's fixed domain model. Default incremental rankings
    may therefore be reused across arrivals; custom rankings are recomputed.
    """
    def initial_state(self) -> StateT: ...
    def key(self, state: StateT) -> Hashable: ...
    def iter_successors(self, state: StateT, checkpoint: Callable[[], None]) -> Iterable: ...
    def materialize(self, state: StateT, operation: Hashable, checkpoint: Callable[[], None]): ...
    def cost(self, state: StateT) -> float: ...
    def is_feasible(self, state: StateT) -> bool: ...
    def finish(self, state: StateT) -> StateT | None: ...


@dataclass(frozen=True)
class SearchView:
    upper_bound: float = float("inf")
    state: object = None
    plan: object = None
    revision: int = 0


@dataclass(frozen=True)
class SearchEvent:
    kind: str
    candidate: object = None
    rejected: int = 0
    duplicate: int = 0
    dropped: int = 0


