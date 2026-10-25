"""Cooperative planning budgets, separate from execution memory constraints."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import math
import sys
import time
from types import GeneratorType
from functools import lru_cache


DEFAULT_PLANNING_SECONDS = 3600.0

# The former fixed walk cadence. It is now the floor of the adaptive schedule,
# so a graph that stops growing is still re-measured at the old rate and the
# old behaviour stays reachable through MemorySampling(mode="fixed").
DEFAULT_MEMORY_SAMPLE_INTERVAL = 256

# Retained-object visits one search state may amortize on accounting. The
# adaptive schedule makes the total walk cost O(objects_per_state * states).
DEFAULT_MEMORY_OBJECTS_PER_STATE = 16

# When the measured growth per state is positive, the next walk is predicted
# where the size reaches the memory limit itself. One walk then lands at the
# crossing instead of a whole amortized interval past it. The divisor is the
# safety margin: 1 aims at the limit, and a larger value aims at a fraction of
# the remaining headroom, which costs more walks near the limit.
MEMORY_LIMIT_PREDICTION_DIVISOR = 1


_LEAF_TYPES = frozenset((str, bytes, int, float, bool, complex, type(None), range))
_CONTAINER_TYPES = frozenset((tuple, list, set, frozenset))


@lru_cache(maxsize=128)
def _dataclass_names(cls):
    names = tuple(f.name for f in fields(cls))
    return names, frozenset(names)


class BudgetExhausted(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _retained_walk(roots, seen, total, added=None, known=None):
    """Visit every retained object once; returns (bytes, visited objects).

    The seen set collects the identities visited by this walk. The known set is
    an index of identities another total already accounts for; it is only read,
    so a session-owned static index can be reused by every sample without
    copying it. The added list collects identities this walk newly claimed, for
    callers that charge an index for them.
    """
    pending = list(roots)
    visited = 0
    while pending:
        obj = pending.pop()
        identity = id(obj)
        if identity in seen or (known is not None and identity in known):
            continue
        seen.add(identity)
        visited += 1
        if added is not None:
            added.append(identity)
        total += sys.getsizeof(obj)
        kind = type(obj)
        # Builtin leaves have no retained children. Keep identity accounting
        # and sizeof unchanged while avoiding several protocol/type probes.
        if kind in _LEAF_TYPES:
            continue
        if kind in _CONTAINER_TYPES:
            pending.extend(obj)
        elif isinstance(obj, Mapping):
            pending.extend(obj.keys())
            pending.extend(obj.values())
        elif isinstance(obj, (tuple, list, set, frozenset)):
            pending.extend(obj)
        elif isinstance(obj, GeneratorType):
            if obj.gi_frame is not None:
                pending.append(obj.gi_frame.f_locals)
            if obj.gi_yieldfrom is not None:
                pending.append(obj.gi_yieldfrom)
        elif is_dataclass(obj) and not isinstance(obj, type):
            names, known_fields = _dataclass_names(type(obj))
            pending.extend(getattr(obj, name) for name in names)
            # cached_property values are outside dataclass fields and must
            # count toward the same search memory budget as other metadata.
            attrs = getattr(obj, "__dict__", None)
            if attrs is not None:
                pending.extend(value for name, value in attrs.items() if name not in known_fields)
        elif hasattr(obj, "__dict__") and not isinstance(obj, type) and not callable(obj):
            pending.append(vars(obj))
    return total, visited


def retained_walk(roots, seen, total, *, known=None):
    """One complete retained-object walk over roots, continuing a seen set."""
    return _retained_walk(roots, seen, total, known=known)


class StaticMemory:
    """One session's unchanged owned metadata, with its index also budgeted.

    Callers identify roots whose reachable objects remain unchanged throughout
    the session. Anchors keep them alive, so cached IDs cannot be recycled.
    Mutable search workspaces must never be included here.
    """
    __slots__ = ('ids', 'total', 'anchors', 'id_bytes')

    def __init__(self):
        self.ids, self.anchors = set(), []
        self.total = self.id_bytes = 0

    def include(self, roots):
        roots = tuple(roots)
        self.anchors.extend(roots)
        added = []
        self.total = _retained_walk(roots, self.ids, self.total, added)[0]
        self.id_bytes += sum(sys.getsizeof(identity) for identity in added)

    @property
    def overhead(self):
        return (sys.getsizeof(self) + sys.getsizeof(self.ids) + self.id_bytes
                + sys.getsizeof(self.anchors) + sys.getsizeof(self.total)
                + sys.getsizeof(self.id_bytes))


def retained_measurement(roots, *, static=None):
    """Retained bytes and visited object count for one complete walk.

    This is not process RSS or allocator accounting. Shared objects count once.
    An optional session-owned static graph is counted once, including its
    index, and is excluded from the dynamic walk without copying that index, so
    one sample costs O(dynamic objects) instead of O(all objects).
    """
    if static is None:
        return _retained_walk(roots, set(), 0)
    total, objects = _retained_walk(roots, set(), static.total, known=static.ids)
    return total + static.overhead, objects


def retained_size(roots, *, static=None):
    """Estimate retained Python objects including suspended generator locals.

    Sampling occurs at walk boundaries; one walk may overshoot the limit.
    """
    return retained_measurement(roots, static=static)[0]


class MemorySampling:
    """Adaptive schedule for complete retained-object walks.

    One walk visits every retained object, so it costs O(S) for a live graph of
    S objects, and S grows with the number of retained search states. Walking
    every fixed number of steps therefore costs O(S * steps) in total, which is
    quadratic in the state count. Here the next walk is scheduled
    S / objects_per_state expanded states after the previous measurement, so
    each walk is amortized over at least that many states and the whole search
    pays O(objects_per_state * states) object visits. The expansion count is the
    scheduler clock because retained objects grow with retained states, and it
    is the count the previous fixed cadence was tuned against.

    Two bounds hold for the scheduled walk.

    * Consecutive walks are at most max(minimum_interval, S / objects_per_state)
      states apart, for S the object count of the last walk. The sampled peak
      can therefore trail the true peak by at most that many states of
      retention, and the walk that reports a breach runs at most that many
      states after the true crossing.
    * When the measured growth per state is positive and the limit is finite,
      the next walk is instead predicted where the size reaches the limit.
      One walk then lands at the crossing, and the interval shortens as the
      remaining headroom shrinks. minimum_interval floors the prediction, so
      the final approach keeps the former fixed cadence: once the prediction
      falls below it, the peak cannot grow past the limit by more than
      minimum_interval states of retention before the next walk reports it.
      A growth rate that accelerates by a factor m after a prediction makes
      the walk land (m - 1) times the assumed headroom past the limit.

    mode="fixed" restores the former cadence exactly: one complete walk per
    minimum_interval expanded states, for debug comparisons.
    """
    __slots__ = ("mode", "minimum_interval", "objects_per_state", "limit",
                 "interval", "deadline", "walks", "visited", "seconds",
                 "max_lag", "last_bytes", "last_expansion", "growth_per_expansion")

    def __init__(self, *, limit=None, minimum_interval=DEFAULT_MEMORY_SAMPLE_INTERVAL,
                 objects_per_state=DEFAULT_MEMORY_OBJECTS_PER_STATE, mode="adaptive"):
        if mode not in ("adaptive", "fixed"):
            raise ValueError("memory sampling mode must be 'adaptive' or 'fixed'")
        self.mode = mode
        self.minimum_interval = integer(minimum_interval, "minimum_interval", 1)
        self.objects_per_state = integer(objects_per_state, "objects_per_state", 1)
        self.limit = None if limit is None else integer(limit, "limit", 1)
        self.interval = self.minimum_interval
        self.deadline = 0
        self.walks = self.visited = 0
        self.seconds = 0.0
        self.max_lag = 0
        self.last_bytes = self.growth_per_expansion = 0
        self.last_expansion = None

    def due(self, expansions):
        """Whether a non-forced walk request has to run at this expansion count."""
        return self.walks == 0 or expansions >= self.deadline

    def recorded(self, expansions, size, objects, seconds):
        """Attribute one completed walk and schedule the next one."""
        if self.last_expansion is not None and expansions > self.last_expansion:
            self.growth_per_expansion = max(0, size - self.last_bytes) // (expansions - self.last_expansion)
        self.max_lag = max(self.max_lag, expansions - self.last_expansion
                           if self.last_expansion is not None else 0)
        self.last_bytes, self.last_expansion = size, expansions
        self.walks += 1
        self.visited += objects
        self.seconds += seconds
        if self.mode == "fixed":
            # The former cadence: one complete walk per minimum_interval
            # retained states, for debug comparisons with the adaptive default.
            self.interval = self.minimum_interval
        else:
            # One walk is amortized over at least this many states.
            cap = max(self.minimum_interval, objects // self.objects_per_state)
            predicted = cap
            if self.limit is not None and self.limit > size and self.growth_per_expansion > 0:
                predicted = ((self.limit - size) // MEMORY_LIMIT_PREDICTION_DIVISOR) // self.growth_per_expansion
            self.interval = max(self.minimum_interval, min(cap, predicted))
        self.deadline = expansions + self.interval

    def statistics(self, expansions):
        return dict(mode=self.mode, walks=self.walks, expansions=expansions,
                    visited_objects=self.visited, seconds=self.seconds,
                    interval=self.interval, max_lag=self.max_lag,
                    minimum_interval=self.minimum_interval,
                    objects_per_state=self.objects_per_state,
                    growth_bytes_per_expansion=self.growth_per_expansion)


class SearchBudget:
    def __init__(self, *, deadline_seconds=DEFAULT_PLANNING_SECONDS, max_states=1_000_000,
                 max_transitions=None, search_memory_limit=None, clock=time.monotonic,
                 memory_interval=DEFAULT_MEMORY_SAMPLE_INTERVAL,
                 memory_objects_per_state=DEFAULT_MEMORY_OBJECTS_PER_STATE,
                 memory_sampling="adaptive"):
        if (isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, (int, float))
                or not math.isfinite(deadline_seconds) or deadline_seconds < 0):
            raise ValueError("deadline_seconds must be finite and nonnegative")
        if memory_sampling not in ("adaptive", "fixed"):
            raise ValueError("memory_sampling must be 'adaptive' or 'fixed'")
        self.max_states = integer(max_states, "max_states")
        self.max_transitions = None if max_transitions is None else integer(max_transitions, "max_transitions")
        self.memory_limit = None if search_memory_limit is None else integer(search_memory_limit, "search_memory_limit", 1)
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + deadline_seconds
        self.expanded = self.transitions = self.peak_memory = 0
        # Sampling is disabled entirely without a limit, so the schedule exists
        # exactly when a limit has to be enforced.
        self.memory_sampling = (None if self.memory_limit is None else
            MemorySampling(limit=self.memory_limit, minimum_interval=memory_interval,
                           objects_per_state=memory_objects_per_state, mode=memory_sampling))

    def check(self, *, transition=False):
        if self.clock() >= self.deadline:
            raise BudgetExhausted("deadline")
        if self.max_transitions is not None and self.transitions >= self.max_transitions:
            raise BudgetExhausted("max_transitions")
        if transition:
            self.transitions += 1

    def consume(self, event):
        if event.kind == "expand":
            if self.expanded >= self.max_states:
                raise BudgetExhausted("max_states")
            self.expanded += 1
        elif event.kind == "transition":
            self.transitions += 1

    def memory_due(self):
        """Whether a non-forced retained-size walk is due at this state count."""
        sampling = self.memory_sampling
        return False if sampling is None else sampling.due(self.expanded)

    def sample_memory(self, roots, *, static=None):
        if self.memory_limit is None:
            return
        sampling = self.memory_sampling
        started = None if sampling is None else time.perf_counter()
        size, objects = retained_measurement(roots, static=static)
        if sampling is not None:
            sampling.recorded(self.expanded, size, objects, time.perf_counter() - started)
        self.peak_memory = max(self.peak_memory, size)
        if size > self.memory_limit:
            raise BudgetExhausted("search_memory_limit")

    @property
    def sampled_peak_memory(self):
        """No observation is reported when search memory sampling is disabled."""
        return None if self.memory_limit is None else self.peak_memory

    @property
    def memory_sampling_statistics(self):
        """Walk schedule observations, or None when sampling is disabled."""
        sampling = self.memory_sampling
        return None if sampling is None else sampling.statistics(self.expanded)

    def slice(self, quantum):
        for _ in range(integer(quantum, "quantum", 1)):
            self.check()
            yield None
