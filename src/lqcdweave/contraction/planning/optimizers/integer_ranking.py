"""Incremental ranking of the current workspace's locally scored candidates.

Heap entries own only calculation IDs, scores and insertion-order counters.
Changed occurrence groups and availability invalidate scores; historical DFS
nodes never own a ranking index. Global scorers retain the scan reference.
"""
from dataclasses import dataclass
import heapq
import math
import time

from .integer_scoring import IntegerScorer, validate_score
from .integer_space import IntegerOperation
from .choices import order_ties


def supports_incremental_ranking(scorer):
    return scorer is None or (type(scorer) is IntegerScorer and scorer.local)


@dataclass(frozen=True, slots=True)
class RankedCandidates:
    choices: tuple
    candidate_count: int
    evaluations: int
    scoring_seconds: float


class CandidateRanking:
    """Bounded lazy heap with exactly the full scan's score/order tie breaking."""

    def __init__(self, space, name):
        self.space, self.name = space, name
        self.heap, self.entries = [], {}
        self.dirty = set(space.groups)
        self.revision = 0

    def _score(self, scorer, operation, cost, checkpoint, profile_scoring):
        started = time.perf_counter() if profile_scoring else None
        score = cost if scorer is None else scorer(self.space, operation, cost, checkpoint)
        seconds = 0.0 if started is None else time.perf_counter() - started
        validate_score(score)
        return score, seconds

    def _refresh(self, scorer, checkpoint, profile_scoring):
        evaluations, seconds = 0, 0.0
        for identity in sorted(self.dirty, key=lambda i: self.space.group_ordinals.get(i, -1)):
            checkpoint()
            if identity not in self.space.groups or identity in self.space.terminal_families:
                self.entries.pop(identity, None)
            else:
                calculation = self.space.registry.calculations[identity]
                reused = calculation.result in self.space.available
                cost = 0.0 if reused else calculation.seconds
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("graph increments must be finite and nonnegative")
                operation = IntegerOperation(identity, calculation.result, reused, cost)
                score, elapsed = self._score(scorer, operation, cost, checkpoint, profile_scoring)
                seconds += elapsed
                evaluations += 1
                self.revision += 1
                ordinal = self.space.group_ordinals[identity]
                self.entries[identity] = (self.revision, operation, cost, score, ordinal)
                heapq.heappush(self.heap, (score, ordinal, self.revision, identity))
            self.dirty.discard(identity)
        # Repeated DFS undo may otherwise retain stale scores indefinitely.
        # Publish compaction only after the complete budgeted rebuild succeeds.
        if len(self.heap) > max(64, 2 * len(self.entries)):
            heap = []
            for identity, (revision, _, _, score, ordinal) in self.entries.items():
                checkpoint()
                heap.append((score, ordinal, revision, identity))
            heapq.heapify(heap)
            self.heap = heap
            self.space.rank_heap_compactions += 1
        return evaluations, seconds

    def select(self, width, scorer, checkpoint, profile_scoring, *, include_ties=True):
        evaluations, seconds = self._refresh(scorer, checkpoint, profile_scoring)
        self.space.incremental_rankings += 1
        count = len(self.entries)
        if not count:
            # Keep output-layout completion and its pricing in the existing
            # domain helper, including the independent-family deferral policy.
            choices = tuple(self.space.search_candidates(checkpoint))
            for operation, cost in choices:
                _, elapsed = self._score(scorer, operation, cost, checkpoint, profile_scoring)
                seconds += elapsed
                evaluations += 1
            return RankedCandidates(choices, len(choices), evaluations, seconds)
        self.space.enumerations += 1
        self.space.terminal_deferred += len(self.space.terminal_families)
        selected, popped, boundary = [], [], None
        try:
            while self.heap:
                checkpoint()
                item = heapq.heappop(self.heap)
                _, _, revision, identity = item
                entry = self.entries.get(identity)
                if entry is None or entry[0] != revision:
                    continue
                popped.append(item)
                if len(selected) >= width and item[0] != boundary:
                    break
                selected.append((entry[3], entry[4], entry[1], entry[2]))
                if len(selected) == width:
                    boundary = item[0]
        finally:
            for item in popped:
                heapq.heappush(self.heap, item)
        selected = order_ties(self.space, selected, checkpoint)
        if not include_ties:
            selected = selected[:width]
        return RankedCandidates(tuple((op, cost) for _, _, op, cost in selected),
                                count, evaluations, seconds)
