"""Ranking features for the integer workspace; never change execution costs.

Element counts describe expressions. Storage delta estimates the live logical
footprint under last-use release, not the retained-cache policy or physical
host/device peaks. Logical sparse cores have no dense storage allocation.
"""
from collections import Counter
import math


SCORERS = (
    "incremental_time", "element_reduction", "shared_savings", "storage_delta",
    "time_per_term", "time_per_removed_element", "sharing_lookahead", "sharing_count",
)
LOCAL_SCORERS = frozenset(("incremental_time", "shared_savings", "time_per_term",
                           "element_reduction", "time_per_removed_element", "sharing_count"))


def normalize_scorer(name):
    """Accept one named policy or an ordered primary/secondary pair."""
    if isinstance(name, (tuple, list)):
        if len(name) != 2 or any(not isinstance(item, str) or item not in SCORERS for item in name):
            raise ValueError("scorer pair must contain exactly two known scorer names")
        return tuple(name)
    if not isinstance(name, str) or name not in SCORERS:
        raise ValueError(f"unknown scorer {name!r}; choose from {', '.join(SCORERS)}")
    return name


def validate_score(score):
    """Keep scalar and lexicographic scores separate from execution costs."""
    if isinstance(score, tuple):
        finite = len(score) == 2 and all(math.isfinite(value) for value in score)
    else:
        finite = math.isfinite(score)
    if not finite:
        raise ValueError("ranking score must be finite")


def reverse_score(score):
    """Reverse every priority for the bounded worst-first candidate heap."""
    return tuple(-value for value in score) if isinstance(score, tuple) else -score


class IntegerScorer:
    """Callable ranking policy with expansion-local facts and immutable sizes."""

    def __init__(self, name):
        self.name = normalize_scorer(name)
        self.scorers = tuple(IntegerScorer(item) for item in self.name) if isinstance(self.name, tuple) else ()
        self.local = all(scorer.local for scorer in self.scorers) if self.scorers else self.name in LOCAL_SCORERS
        self.sizes = {}
        self.begin()

    def begin(self):
        self.references = None
        self.features = {}
        self.opportunities = None
        self.masks = {}
        for scorer in self.scorers:
            scorer.begin()

    def size(self, space, identity):
        if identity not in self.sizes:
            definition = space.registry.data[identity]
            spec = definition.spec
            elements = 0 if spec is None else math.prod(spec.shape)
            nbytes = 0 if spec is None or definition.core is not None else elements * spec.dtype.itemsize
            source = space.problem.sources.get(definition.name)
            if source is not None and source.snapshot_bytes == 0:
                nbytes = 0  # Borrowed views never release their backing allocation.
            self.sizes[identity] = elements, nbytes
        return self.sizes[identity]

    def _features(self, space, operation, checkpoint):
        identity = operation.calculation
        if identity not in self.features:
            occurrences = space.groups[identity]
            occurrence = next(iter(occurrences.values()))
            term = space.terms[occurrence.scene]
            consumed = occurrence.positions + occurrence.cores
            inputs = sum(self.size(space, term.factors[position])[0] for position in consumed)
            output, nbytes = self.size(space, operation.result)
            self.features[identity] = occurrences, (output - inputs) * space.occurrence_count(identity), nbytes
        return self.features[identity]

    def _storage_delta(self, space, operation, checkpoint):
        occurrences, _, nbytes = self._features(space, operation, checkpoint)
        if self.references is None:
            self.references = Counter()
            for term in space.terms:
                checkpoint()
                self.references.update(term.factors)  # Includes pinned completed outputs.
        removed = Counter()
        for occurrence in occurrences.values():
            checkpoint()
            term = space.terms[occurrence.scene]
            removed.update(term.factors[position] for position in occurrence.positions + occurrence.cores)
        released = sum(self.size(space, data)[1] for data, count in removed.items()
                       if data != operation.result and count == self.references[data])
        return (0 if operation.reused else nbytes) - released

    def _sharing_lookahead(self, space, operation, cost, checkpoint):
        if self.opportunities is None:
            entries = []
            for identity, occurrences in space.groups.items():
                checkpoint()
                mask = 0
                for scene in occurrences:
                    mask |= 1 << scene
                self.masks[identity] = mask
                calculation = space.registry.calculations[identity]
                seconds = 0.0 if calculation.result in space.available else calculation.seconds
                gain = seconds * (space.occurrence_count(identity) - 1)
                if gain > 0:
                    entries.append((gain, seconds, mask, calculation.result))
            self.opportunities = sorted(entries, key=lambda entry: entry[0], reverse=True)
        unchanged = ~self.masks[operation.calculation]
        remaining = 0.0
        for upper, seconds, mask, result in self.opportunities:
            checkpoint()
            if upper <= remaining:
                break
            if result == operation.result:
                continue
            # Only unmodified Terms are known to retain this exact operation.
            # Do not count competing opportunities together or predict new ones.
            gain = seconds * max(0, space.mask_count(mask & unchanged) - 1)
            remaining = max(remaining, gain)
        return -(cost * (space.occurrence_count(operation.calculation) - 1) + remaining)

    def _protected_expansion(self, space, operation):
        """Recognize products which grow a protected sum without reducing it.

        Sharing an expensive broadcast product is not a saving when its large
        allocation prevents the ensuing wide reduction. Keep that legal route
        available, but rank reductions and non-expanding sharing ahead of it.
        This uses the ordinary integer occurrence, independently of its source.
        """
        occurrence = next(iter(space.groups[operation.calculation].values()))
        barriers = space._sum_barriers(occurrence.scene)
        if not barriers:
            return False
        term = space.terms[occurrence.scene]
        pattern = space.registry.patterns[term.pattern]
        selected = occurrence.positions + occurrence.cores
        axes = {a for position in selected for a in pattern.inputs[position]}
        if axes != set(occurrence.output) or not any(axes.intersection(trigger) for trigger, _ in barriers):
            return False
        output = self.size(space, operation.result)[0]
        return output > max(self.size(space, term.factors[p])[0] for p in selected)

    def __call__(self, space, operation, cost, checkpoint):
        if self.scorers:
            return tuple(scorer(space, operation, cost, checkpoint) for scorer in self.scorers)
        if self.name == "incremental_time":
            return cost
        k = space.occurrence_count(operation.calculation)
        if self.name == "sharing_count":
            return -k
        if self.name == "time_per_term":
            return cost / k
        if self.name == "shared_savings":
            if self._protected_expansion(space, operation):
                return cost
            return -cost * (k - 1)
        if self.name == "sharing_lookahead":
            return self._sharing_lookahead(space, operation, cost, checkpoint)
        if self.name == "storage_delta":
            return self._storage_delta(space, operation, checkpoint)
        _, reduction, _ = self._features(space, operation, checkpoint)
        if self.name == "element_reduction":
            return reduction
        return cost / max(1, -reduction)
