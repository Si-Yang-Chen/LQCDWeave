"""Exact priority queue with a separate lower-bound heap."""
import heapq
import itertools
import math


class PriorityFrontier:
    def __init__(self):
        self.order, self.bounds = [], []
        self.pending = {}
        self.serial = itertools.count()

    def offer(self, key, state, priority, bound):
        serial = next(self.serial)
        self.pending[key] = (serial, state)
        heapq.heappush(self.order, (priority, serial, key))
        heapq.heappush(self.bounds, (bound, serial, key))

    def _clean(self, heap):
        while heap and self.pending.get(heap[0][2], (None,))[0] != heap[0][1]:
            heapq.heappop(heap)

    @property
    def lower_bound(self):
        self._clean(self.bounds)
        return self.bounds[0][0] if self.bounds else math.inf

    def pop(self):
        self._clean(self.order)
        if not self.order:
            return None
        _, _, key = heapq.heappop(self.order)
        return self.pending.pop(key)[1]
