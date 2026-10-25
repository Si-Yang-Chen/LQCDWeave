"""Hash equal subexpressions, retaining complete keys to resolve collisions.

These keys are local metadata lookups, never persisted scientific identities.
The matching rules mirror the independent scan in state.py. Unary occurrences
are indexed even in longer Terms because a reduction can be shared into them.
"""
from dataclasses import dataclass, field
from lqcdweave.contracts.correlator.api import AxisPattern
from itertools import chain, combinations

from lqcdweave.contracts.correlator.api import cycle_intermediate_labels


@dataclass(frozen=True, slots=True)
class ContractionKey:
    values: tuple
    _hash: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash(self.values))

    def __hash__(self):
        return self._hash


class ContractionIndex:
    """Map a complete ordered computation to the first occurrence per Term."""
    def __init__(self, layouts, checkpoint=lambda: None):
        self.layouts = layouts
        self.checkpoint = checkpoint
        self.groups, self.term_keys = {}, {}
        self.built = set()

    @staticmethod
    def key(layout):
        from .state import _bridge_identity
        return ContractionKey((layout[0], layout[1],
                               _bridge_identity(layout[5]), _bridge_identity(layout[6])))

    def add(self, index, checkpoint=lambda: None):
        """Refresh only operand groups previously requested by this path."""
        term = self.layouts.state.pending[index]
        if term is None or not self.built:
            return
        positions = ((i,) for i in range(len(term.factors)))
        pairs = chain(positions, combinations(range(len(term.factors)), 2))
        # Publish together so interrupted metadata work leaves no partial group.
        entries = list(self._entries(index, pairs, checkpoint, only_built=True))
        self._register(entries)

    def _entries(self, index, pairs, checkpoint, *, only_built=False):
        from .state import _bridge_identity, _bridge_in_operand_order
        layouts = self.layouts
        term = layouts.state.pending[index]
        if term is None:
            return
        records = layouts.bridges.scoped(index)
        joins = layouts.contexts[index][2]
        for pair in pairs:
            checkpoint()
            selected = tuple(term.factors[i] for i in pair)
            if only_built and selected not in self.built:
                continue
            primary = next((b for b in records if len(pair) == 2 and
                            (b.left, b.right) == (selected[0].tensor, selected[1].tensor)), None)
            secondary = next((b for b in records if b is not primary and len(pair) == 2 and
                              {b.left, b.right} == {selected[0].tensor, selected[1].tensor}), None)
            # A reversed protected pair has no legal layout: its matching scan
            # sees a secondary bridge without a primary and rejects sharing.
            # Do not let this unrelated pair prevent indexing valid choices.
            if primary is None and secondary is not None:
                continue
            bridge, cycle = _bridge_in_operand_order(primary), _bridge_in_operand_order(secondary)
            needed = layouts.needed(index, selected)
            external = (cycle_intermediate_labels(
                AxisPattern(tuple(f.labels for f in selected), ()), bridge, cycle,
                needed_labels=tuple(sorted(needed)))[1]
                if cycle is not None else tuple(c for f in selected for c in f.labels))
            labels = (term.output if len(term.factors) == len(pair) and not joins else
                      tuple(dict.fromkeys(c for c in external if c in needed)))
            key = ContractionKey((selected, labels, _bridge_identity(bridge), _bridge_identity(cycle)))
            yield key, index, pair

    def _register(self, entries):
        for key, index, pair in entries:
            self.groups.setdefault(key, {}).setdefault(index, pair)
            self.term_keys.setdefault(index, set()).add(key)

    def remove(self, index):
        for key in self.term_keys.pop(index, ()):
            group = self.groups[key]
            del group[index]
            if not group:
                del self.groups[key]

    def matching(self, layout):
        selected = layout[0]
        if selected not in self.built:
            entries = []
            for index, pairs in self.layouts.factors.matching_terms(selected):
                self.checkpoint()
                entries.extend(self._entries(index, pairs, self.checkpoint))
            # An interrupted build publishes neither rows nor a completed mark.
            self._register(entries)
            self.built.add(selected)
        # Sorting keeps the authoritative stable Term order after local edits.
        yield from sorted(self.groups.get(self.key(layout), {}).items())
