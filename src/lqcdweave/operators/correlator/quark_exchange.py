"""Exact scalar exchange characters for ordered three-quark coefficients."""

from collections.abc import Mapping
from numbers import Integral
from lqcdweave.contracts.quark_exchange import quark_permutation


_IDENTITY = (0, 1, 2)


def normalize_quark_symmetry(value):
    """Close signed generators; reject inconsistent scalar declarations."""
    if value is None:
        return ()
    if isinstance(value, str):
        if value not in ("symmetric", "antisymmetric"):
            raise ValueError("quark_symmetry must be symmetric, antisymmetric, or a permutation/sign mapping")
        sign = 1 if value == "symmetric" else -1
        value = {(1, 0, 2): sign, (0, 2, 1): sign}
    if not isinstance(value, Mapping):
        raise TypeError("quark_symmetry must be a permutation/sign mapping")
    generators = []
    for permutation, sign in value.items():
        permutation = quark_permutation(permutation)
        if isinstance(sign, bool) or not isinstance(sign, Integral) or sign not in (-1, 1):
            raise ValueError("quark exchange characters must be integer +1 or -1")
        generators.append((tuple(map(int, permutation)), int(sign)))
    group = {_IDENTITY: 1}
    pending = [_IDENTITY]
    while pending:
        current = pending.pop()
        for generator, sign in generators:
            composed = tuple(generator[current[i]] for i in range(3))
            character = group[current] * sign
            if composed in group:
                if group[composed] != character:
                    raise ValueError("inconsistent quark exchange characters under group composition")
            else:
                group[composed] = character
                pending.append(composed)
    return tuple(sorted(group.items())) if len(group) > 1 else ()


def adjoint_quark_symmetry(group):
    """Adjoint fields have reversed order bar(q2), bar(q1), bar(q0)."""
    return {tuple(2 - permutation[2 - i] for i in range(3)): sign
            for permutation, sign in group}


def quark_symmetry_name(group):
    if len(group) == 6:
        return "antisymmetric" if any(sign == -1 for _, sign in group) else "symmetric"
    return repr(tuple((permutation, sign) for permutation, sign in group if permutation != _IDENTITY))


def flavor_preserving_symmetry(group, flavors):
    return tuple((permutation, sign) for permutation, sign in group
                 if tuple(flavors[i] for i in permutation) == tuple(flavors))


def reduce_exchange_terms(terms, symmetries, times):
    """Collect graph orbits before general symbolic simplification.

    Wick signs are already in each term. Only the declared coefficient
    character multiplies them here; a fermionic sign is never applied twice.
    The memo is local to this flavor word and retains no bound arrays.
    """
    from sympy import Add, Mul, S
    from .base_types import Tag
    from .flavor_structure import Propagator

    actions = tuple((vertex, permutation, sign)
                    for vertex, group in symmetries
                    for permutation, sign in group if permutation != _IDENTITY)
    if not actions:
        return terms
    memo = {}

    def canonical(edges):
        if edges not in memo:
            visited = {edges: 1}
            pending = [edges]
            zero = False
            while pending:
                current = pending.pop()
                for vertex, permutation, sign in actions:
                    def move(slot):
                        return vertex * 3 + permutation[slot % 3] if slot // 3 == vertex else slot
                    transformed = tuple(sorted((flavor, move(source), move(sink))
                                               for flavor, source, sink in current))
                    phase = visited[current] * sign
                    if transformed in visited:
                        zero |= visited[transformed] != phase
                    else:
                        visited[transformed] = phase
                        pending.append(transformed)
            representative = min(visited)
            for member, phase in visited.items():
                memo[member] = representative, (0 if zero else phase * visited[representative])
        return memo[edges]

    collected = {}
    for term in terms:
        edges, coefficient = [], S.One
        for factor in Mul.make_args(term):
            if isinstance(factor, Propagator):
                edges.append((factor.flavor, factor.source_tag.tag, factor.sink_tag.tag))
            else:
                coefficient *= factor
        representative, phase = canonical(tuple(sorted(edges)))
        collected[representative] = collected.get(representative, S.Zero) + coefficient * phase
    result = []
    for edges, coefficient in collected.items():
        if coefficient != 0:
            result.append(coefficient * Mul(*(Propagator(
                flavor, Tag(source, times[source // 3]), Tag(sink, times[sink // 3]),
            ) for flavor, source, sink in edges)))
    return result
