"""Pure exact three-slot representations shared by coefficients and Wick graphs.

No hadron symbols, graph reduction, readers or numerical execution live here.
SymPy provides immutable exact small matrices when this optional module is used.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import sympy as sp


_ID = (0, 1, 2)


def quark_permutation(value):
    value = tuple(value)
    if (len(value) != 3 or any(isinstance(i, bool) or not isinstance(i, Integral)
                              for i in value) or set(value) != {0, 1, 2}):
        raise ValueError("quark exchange permutations must contain 0, 1, 2 exactly once")
    return tuple(map(int, value))


def _compose(left, right):
    return tuple(left[right[i]] for i in range(3))


def _exact(matrix):
    return sp.ImmutableMatrix(matrix).applyfunc(lambda value: sp.cancel(sp.expand(value)))


@dataclass(frozen=True, init=False)
class QuarkExchange:
    """Exact exchange matrices: T_a(x[p]) = sum_b D(p)[b,a] T_b(x).

    ``mixed`` uses the ordered (MS, MA) basis, symmetric/antisymmetric
    under exchange of slots 0 and 1. Custom generators can describe a
    subgroup, a changed basis, or a reducible coefficient multiplet.
    """

    elements: tuple

    def __init__(self, generators="mixed"):
        if isinstance(generators, QuarkExchange):
            object.__setattr__(self, "elements", generators.elements)
            return
        if isinstance(generators, str):
            mixed = {(1, 0, 2): sp.diag(1, -1), (0, 2, 1): sp.Matrix([
                [-sp.Rational(1, 2), sp.sqrt(3)/2], [sp.sqrt(3)/2, sp.Rational(1, 2)]])}
            if generators == "mixed":
                generators = mixed
            elif generators in ("symmetric", "antisymmetric"):
                sign = 1 if generators == "symmetric" else -1
                generators = {p: sp.ImmutableMatrix([[sign]]) for p in mixed}
            elif generators == "all":
                generators = {p: sp.diag(1, -1, matrix) for p, matrix in mixed.items()}
            else:
                raise ValueError("quark exchange requires symmetric, antisymmetric, mixed, all, or exact matrices")
        if not isinstance(generators, Mapping) or not generators:
            raise TypeError("matrix quark exchange requires nonempty generator matrices")
        checked, dimension = [], None
        for p, value in generators.items():
            # Share the three-slot permutation boundary, without scalar parity.
            quark_permutation(p)
            p, matrix = tuple(map(int, p)), _exact(value)
            if matrix.rows == 0 or matrix.rows != matrix.cols:
                raise ValueError("exchange matrices must be nonempty and square")
            if dimension is not None and dimension != matrix.rows:
                raise ValueError("exchange matrices must have the same dimension")
            if any(not x.is_number or x.is_finite is not True or x.has(sp.Float) for x in matrix):
                raise ValueError("exchange matrices require finite exact numeric entries")
            dimension = matrix.rows
            checked.append((p, matrix))
        group = {_ID: sp.ImmutableMatrix.eye(dimension)}
        pending = [_ID]
        while pending:
            p = pending.pop()
            for q, matrix in checked:
                pq, value = _compose(q, p), _exact(matrix * group[p])
                if pq in group:
                    if any(x != 0 and sp.simplify(x) != 0 for x in value - group[pq]):
                        raise ValueError("inconsistent exchange matrices under group composition")
                else:
                    group[pq] = value
                    pending.append(pq)
        object.__setattr__(self, "elements", tuple(sorted(group.items())))

    @property
    def dimension(self):
        return self.elements[0][1].rows

    @property
    def matrices(self):
        from types import MappingProxyType
        return MappingProxyType(dict(self.elements))

    def tensor_product(self, other):
        """Exchange both factors together; require the same declared subgroup."""
        other = QuarkExchange(other)
        if self.matrices.keys() != other.matrices.keys():
            raise ValueError("tensor product requires the same exchange subgroup")
        return QuarkExchange({p: sp.kronecker_product(m, other.matrices[p])
                              for p, m in self.elements})

    @classmethod
    def pair(cls, symmetry="all", *, slots=(0, 1)):
        """S2 on two of a baryon's three slots; all orders the S,A components."""
        from numbers import Integral
        slots = tuple(slots)
        if (len(slots) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral)
                                   or i not in range(3) for i in slots) or slots[0] == slots[1]):
            raise ValueError("pair slots must select two distinct baryon slots")
        p = list(_ID)
        p[slots[0]], p[slots[1]] = p[slots[1]], p[slots[0]]
        if symmetry == "all":
            matrix = sp.diag(1, -1)
        elif symmetry in ("symmetric", "antisymmetric"):
            matrix = sp.ImmutableMatrix([[1 if symmetry == "symmetric" else -1]])
        else:
            raise ValueError("pair symmetry must be symmetric, antisymmetric or all")
        return cls({tuple(p): matrix})

    @lru_cache(maxsize=64)
    def adjoint(self):
        return QuarkExchange({tuple(2-p[2-i] for i in range(3)): matrix.conjugate()
                              for p, matrix in self.elements})


