"""Conservative proof of independent residual Term components."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from .state import ContractionState


def _independent_groups(state, active, ancestry):
    """Group terms unless an exact present or future shared operation is possible.

    A common current factor is opaque. Two common factors can contract together;
    with one common factor, ancestry overlap can still produce the same future
    intermediate. Sparse bridges and baryon joins remain joint while active
    because they consume implicit inputs.
    """
    active = tuple(sorted(active))
    if (any(b.scene is None or b.scene in active for b in state.problem.sparse_bridges)
            or any(j.scene in active and (j.scene, j.name) not in state.completed_joins
                   for j in state.problem.baryon_joins)):
        return (active,)
    factors = {i: Counter(f.tensor for f in state.pending[i].factors) for i in active}
    links = {i: set() for i in active}

    def leaves(keys):
        return frozenset().union(*(ancestry[k] for k in keys))

    for i, j in combinations(active, 2):
        left, right = factors[i], factors[j]
        common = left & right
        a, b, shared = leaves(left - common), leaves(right - common), leaves(common)
        coupled = (sum(common.values()) >= 2 or bool(a & b or a & shared or b & shared)
                   or bool(common) and (sum(left.values()) == 1 or sum(right.values()) == 1))
        if coupled:
            links[i].add(j)
            links[j].add(i)
    groups, remaining = [], set(active)
    while remaining:
        component, queue = set(), [min(remaining)]
        while queue:
            item = queue.pop()
            if item in component:
                continue
            component.add(item)
            queue.extend(links[item] - component)
        remaining.difference_update(component)
        groups.append(tuple(sorted(component)))
    return tuple(groups)


def split_independent_terms(state: ContractionState) -> tuple[tuple[int, ...], ...]:
    """Return original residual term indices with no future shared contraction.

    The function returns indices into ``state.problem.terms``; it does not
    clone or rewrite the residual state. Physical source requirements therefore
    remain in the original problem and must still be planned together.
    """
    if not isinstance(state, ContractionState):
        raise TypeError("state must be a ContractionState")
    active = tuple(index for index, term in enumerate(state.pending) if term is not None)
    if len(active) < 2:
        return (active,) if active else ()

    ancestry = {key: frozenset((key,)) for key in state.problem.sources}
    for action in state.actions:
        if action.kind not in ("contract", "sparse_bridge", "baryon_join"):
            continue
        leaves = frozenset().union(*(ancestry.get(key, frozenset((key,)))
                                     for key in action.operands))
        ancestry[action.tensor] = leaves

    current = {factor.tensor for index in active for factor in state.pending[index].factors}
    # Unknown ancestry means the proof is incomplete. Keep those terms joint
    # rather than treating an unseen intermediate as an independent leaf.
    if any(key not in ancestry for key in current):
        return (active,)
    return _independent_groups(state, active, ancestry)


__all__ = ["split_independent_terms"]
