"""Resumable exact graph search over an injected operation domain."""
from __future__ import annotations

import math
from numbers import Real
import time

from .protocols import SearchSpace, LayeredSearchSpace, SearchEvent
from .frontier import PriorityFrontier
from .bounds import remaining
from .budget import DEFAULT_PLANNING_SECONDS


def _cost(space, state, previous=0.0):
    value = space.cost(state)
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < previous):
        raise ValueError("exact search requires finite costs and nonnegative increments")
    return value


def _layer(space, state):
    value = space.layer(state)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("search layers must be nonnegative integers")
    return value


def _finish(space, state, cost):
    final = space.finish(state)
    if final is not None:
        if _cost(space, final) != cost:
            raise ValueError("charged finalization must be an explicit search operation")
        if space.is_feasible(final):
            return final
    return None


class ExactSearch:
    """A* with reopening, or layered DP; zero lower bound gives Dijkstra.

    An active expansion remains in the global bound until every outgoing edge
    is evaluated. Layers affect ordering only, never the coverage certificate.
    """
    exact = True

    def __init__(self, *, layered=False, lower_bound=None):
        self.layered, self.bound = layered, lower_bound
        self.name = "dynamic_programming" if layered else "exact"
        self.complete = False
        self.frontier = PriorityFrontier()
        self.distance, self.layers = {}, {}
        self.active_bound = math.inf
        self.started = False

    @property
    def lower_bound(self):
        if not self.started:
            return 0.0
        return min(self.active_bound, self.frontier.lower_bound)

    def steps(self, space, view):
        start = space.initial_state()
        if _cost(space, start) != 0:
            raise ValueError("initial search cost must be zero")
        if not space.is_feasible(start):
            self.complete = True
            return
        def insert(state):
            key, cost = space.key(state), _cost(space, state)
            level = _layer(space, state) if self.layered else 0
            if self.layered and self.layers.setdefault(key, level) != level:
                raise ValueError("equivalent states must have the same search layer")
            if cost >= self.distance.get(key, math.inf):
                return False
            self.distance[key] = cost
            bound = cost + remaining(self.bound, space, state)
            priority = (-level, cost) if self.layered else (bound, cost)
            self.frontier.offer(key, state, priority, bound)
            return True
        insert(start)
        self.started = True
        while True:
            if self.frontier.lower_bound >= view().upper_bound and math.isfinite(view().upper_bound):
                self.complete = True
                return
            state = self.frontier.pop()
            if state is None:
                self.complete = True
                return
            cost = _cost(space, state)
            self.active_bound = cost + remaining(self.bound, space, state)
            level = _layer(space, state) if self.layered else 0
            final = _finish(space, state, cost)
            if final is not None:
                if self.layered and (level != 0 or _layer(space, final) != 0):
                    raise ValueError("layered search requires every goal in layer zero")
                yield SearchEvent("candidate", final)
                self.active_bound = math.inf
                continue
            if self.active_bound >= view().upper_bound:
                self.active_bound = math.inf
                continue
            yield SearchEvent("expand")
            for candidate in space.successors(state):
                candidate_cost = _cost(space, candidate, cost)
                if not space.is_feasible(candidate):
                    yield SearchEvent("transition", rejected=1)
                    continue
                if self.layered and _layer(space, candidate) > level:
                    raise ValueError("operation increased the search layer; use shortest-path search")
                final = _finish(space, candidate, candidate_cost)
                if final is not None:
                    if self.layered and _layer(space, final) != 0:
                        raise ValueError("layered search requires every goal in layer zero")
                    yield SearchEvent("transition", final)
                else:
                    duplicate = not insert(candidate)
                    yield SearchEvent("transition", duplicate=int(duplicate))
            self.active_bound = math.inf


def solve_shortest_path(space: SearchSpace, *, max_states=1_000_000,
                        deadline_seconds=DEFAULT_PLANNING_SECONDS, clock=time.monotonic, **options):
    from .session import SearchSession
    return SearchSession(space, ExactSearch(lower_bound=options.pop("lower_bound", None)),
                         max_states=max_states, deadline_seconds=deadline_seconds,
                         clock=clock, **options).run()


def solve_layered_search(space: LayeredSearchSpace, *, max_states=1_000_000,
                         deadline_seconds=DEFAULT_PLANNING_SECONDS, clock=time.monotonic, **options):
    from .session import SearchSession
    return SearchSession(space, ExactSearch(layered=True, lower_bound=options.pop("lower_bound", None)),
                         max_states=max_states, deadline_seconds=deadline_seconds,
                         clock=clock, **options).run()


__all__ = ["SearchSpace", "LayeredSearchSpace", "ExactSearch", "solve_shortest_path", "solve_layered_search"]
