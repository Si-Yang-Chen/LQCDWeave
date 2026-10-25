"""Temporary occurrence and algebra indexes for one immutable search node.

No successor states or costs are cached. Factor equality includes both the
tensor identity and its labels; ordered repeated occurrences remain distinct.
"""
from collections import Counter, OrderedDict
from dataclasses import fields

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, SparseBridgeSpec, Factor, PendingTerm,
)


_JOIN_FIELDS = tuple(f.name for f in fields(BaryonJoinSpec) if f.name != "scene")
_BRIDGE_FIELDS = tuple(f.name for f in fields(SparseBridgeSpec) if f.name != "scene")


def join_identity(join):
    return tuple(getattr(join, name) for name in _JOIN_FIELDS)


def bridge_identity(bridge):
    return None if bridge is None else tuple(getattr(bridge, name) for name in _BRIDGE_FIELDS)


def reuse_indexes(pending):
    """Amortize setup only when there are enough choices and shared operands.

    This is a lookup policy, never candidate truncation. A long single Term
    benefits from direct lookup; duplicate/shared Terms benefit from indexing.
    """
    choices, occurrences, unique = 0, 0, set()
    for term in pending:
        if term is not None:
            size = len(term.factors)
            choices += max(1, size * (size - 1) // 2)
            occurrences += size
            unique.update(term.factors)
    return choices >= 12 and occurrences - len(unique) >= 8


class BridgeIndex:
    """Scoped bridge lookup preserving the problem's original record order."""
    def __init__(self, bridges):
        self.global_records, self.scenes = [], {}
        self.scoped_cache = {}
        for ordinal, bridge in enumerate(bridges):
            entry = ordinal, bridge
            if bridge.scene is None:
                self.global_records.append(entry)
            else:
                self.scenes.setdefault(bridge.scene, []).append(entry)

    def scoped(self, scene):
        found = self.scoped_cache.get(scene)
        if found is None:
            found = tuple(b for _, b in sorted(self.global_records + self.scenes.get(scene, [])))
            self.scoped_cache[scene] = found
        return found


class FactorIndex:
    def __init__(self, pending, checkpoint=lambda: None):
        self.pending = pending
        self.occurrences = {}
        for index, term in enumerate(pending):
            checkpoint()
            if term is None:
                continue
            for position, factor in enumerate(term.factors):
                checkpoint()
                self.occurrences.setdefault(factor, {}).setdefault(index, []).append(position)

    @staticmethod
    def _positions(left, right):
        if right is None:
            for position in left:
                yield (position,)
        else:
            for i in left:
                for j in right:
                    if i < j:
                        yield (i, j)

    def matching_terms(self, selected):
        if len(selected) not in (1, 2):
            raise ValueError("matching requires a unary or binary contraction")
        first = self.occurrences.get(selected[0], {})
        second = self.occurrences.get(selected[1], {}) if len(selected) == 2 else None
        # Insertion order follows pending Term order, including repeated factors.
        for index, left in first.items():
            right = second.get(index) if second is not None else None
            if second is None or (right is not None and left[0] < right[-1]):
                yield index, self._positions(left, right)


class LayoutIndex:
    """Reuse pure layouts under equal output constraints and bridge context."""
    capacity = 256

    def __init__(self, state, checkpoint=lambda: None, *, reuse=True, bridges=None):
        self.state = state
        self.checkpoint = checkpoint
        self.contractions = None
        self.factors = FactorIndex(state.pending, checkpoint) if reuse else None
        self.scene_joins = {}
        for join in state.problem.baryon_joins:
            checkpoint()
            if (join.scene, join.name) not in state.completed_joins:
                self.scene_joins.setdefault(join.scene, []).append(join)
        self.contexts, self.labels = {}, {}
        self.layouts = OrderedDict()
        self.hits = self.misses = 0
        self.previous_context = None
        self.changed_rows = frozenset()
        self.bridges = bridges or BridgeIndex(state.problem.sparse_bridges)
        if not reuse:
            return
        for index, term in enumerate(state.pending):
            checkpoint()
            if term is None:
                continue
            self._context(index, term)

    def _context(self, index, term):
        counts = Counter()
        for factor in term.factors:
            labels = self.labels.setdefault(factor, frozenset(factor.labels))
            counts.update(labels)
        joins = self.scene_joins.get(index, ())
        fixed = frozenset(term.output + tuple(c for j in joins for c in j.spin_labels + j.mode_labels))
        tensors = {factor.tensor for factor in term.factors}
        active = tuple(b for b in self.bridges.scoped(index)
                       if b.left in tensors and b.right in tensors)
        self.contexts[index] = counts, fixed, bool(joins), active

    def advance(self, state, changed):
        """Retarget an exclusively owned index using all changed Term rows."""
        if state.problem is not self.state.problem:
            raise ValueError("layout index belongs to a different problem")
        changed = set(changed)
        changed.update(scene for scene, _ in state.completed_joins ^ self.state.completed_joins)
        old_pending = self.state.pending
        if self.contractions is not None:
            for index in changed:
                self.contractions.remove(index)
        for index in changed:
            self.checkpoint()
            old = old_pending[index]
            if old is not None:
                for factor in set(old.factors):
                    occurrences = self.factors.occurrences[factor]
                    del occurrences[index]
                    if not occurrences:
                        del self.factors.occurrences[factor]
        self.state = state
        self.factors.pending = state.pending
        self.scene_joins = {}
        for join in state.problem.baryon_joins:
            if (join.scene, join.name) not in state.completed_joins:
                self.scene_joins.setdefault(join.scene, []).append(join)
        for index in changed:
            self.checkpoint()
            self.contexts.pop(index, None)
            term = state.pending[index]
            if term is not None:
                for position, factor in enumerate(term.factors):
                    self.factors.occurrences.setdefault(factor, {}).setdefault(index, []).append(position)
                self._context(index, term)
                if self.contractions is not None:
                    self.contractions.add(index, self.checkpoint)

    def retarget(self, state):
        """Find every changed expression/geometry/join row before advancing."""
        old = self.state
        changed = set()
        for i, (before, after) in enumerate(zip(old.pending, state.pending)):
            self.checkpoint()
            if before is not after:
                changed.add(i)
        revised = state.catalog.changed_keys_since(old.catalog)
        if revised:
            changed.update(i for factor, rows in self.factors.occurrences.items()
                           if factor.tensor in revised for i in rows)
        changed.update(scene for scene, _ in old.completed_joins ^ state.completed_joins)
        self.advance(state, changed)
        self.previous_context = old.pending, old.catalog, old.completed_joins
        self.changed_rows = frozenset(changed)

    def needed(self, index, selected):
        counts, fixed, _, _ = self.contexts[index]
        left = self.labels[selected[0]]
        right = self.labels[selected[1]] if len(selected) == 2 else frozenset()
        return frozenset(label for label in left | right
                         if label in fixed or counts[label] > (label in left) + (label in right))

    def substitutions(self, layout):
        """Batch ordinary replacements under each Term's actual constraints.

        Reuse immutable updates only after output and complete bridge identities
        agree. The public atomic transition keeps an independent scan route.
        """
        from .contraction_index import ContractionIndex
        if self.contractions is None:
            self.contractions = ContractionIndex(self, self.checkpoint)
        _, labels, _, _, _, _, _, _, key, _ = layout
        replacement = Factor(key, labels)
        templates = {}
        for index, positions in self.contractions.matching(layout):
            term = self.state.pending[index]
            joins = self.contexts[index][2]
            template_key = term, positions, joins
            cached = templates.get(template_key)
            if cached is None:
                remaining = tuple(f for i, f in enumerate(term.factors) if i not in positions)
                updated = PendingTerm(remaining + (replacement,), term.output)
                done = len(updated.factors) == 1 and labels == term.output and not joins
                cached = templates[template_key] = (None if done else updated, key if done else None)
            yield index, cached

    def layout(self, index, pair):
        if self.factors is None:
            return self.state._contract_layout(index, pair, scene_joins=self.scene_joins)
        term = self.state.pending[index]
        if term is None:
            raise ValueError("scene result is already complete")
        if (len(term.factors) == 1 and pair != (0,)) or (
                len(term.factors) != 1 and
                (len(pair) != 2 or pair[0] >= pair[1] or pair[1] >= len(term.factors))):
            # Preserve the authoritative validator's errors for malformed calls.
            return self.state._contract_layout(index, pair, scene_joins=self.scene_joins)
        selected = tuple(term.factors[position] for position in pair)
        _, _, joins_pending, active = self.contexts[index]
        needed = self.needed(index, selected)
        terminal = term.output if len(term.factors) == len(pair) and not joins_pending else None
        key = selected, needed, terminal, joins_pending, active, tuple(
            self.state.catalog[f.tensor] for f in selected)
        if self.state.problem.terms[index].sum_barriers:
            key += (self.state.problem.terms[index].sum_barriers,)
        cached = self.layouts.get(key)
        if cached is not None:
            self.hits += 1
            valid, value = cached
            self.layouts.move_to_end(key)
        else:
            self.misses += 1
            try:
                value = self.state._contract_layout(index, pair, scene_joins=self.scene_joins,
                                                    active_bridges=active)
                valid = True
            except ValueError as error:
                valid, value = False, str(error)
            self.layouts[key] = valid, value
            if len(self.layouts) > self.capacity:
                self.layouts.popitem(last=False)
        if not valid:
            raise ValueError(value)
        # Direct-path edits replace the active join table. Pure layouts remain
        # reusable, but the shared-effect builder needs the current table.
        return value[:-1] + (self.scene_joins,)
