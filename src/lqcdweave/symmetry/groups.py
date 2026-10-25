"""Shared cubic groups and exact operator row bases for both clients.

Named little groups use the canonical symbolic representations in this package's
hardcoded_rep through named.canonical_irreps. The regular-representation method
below is retained only for generic stabilizers not covered by those exact tables.
It originated in V2 P1a:

1. conjugacy classes of the regular permutation representation,
2. class operators ``S_i = sum_{g in C_i} R(g)``,
3. diagonalisation of a fixed random combination ``M = sum a_i S_i`` and
   clustering of the eigenvectors by their ``omega_i = v^d S_i v``,
4. dimension from ``d = sqrt(N / sum_i |omega_i|^2 / |C_i|)`` and class
   characters ``chi(C_i) = d omega_i / |C_i|``,
5. isotypic projectors ``E_rho = (d/N) sum_g conj(chi(g)) R(g)`` with
   idempotency/unitarity assertions,
6. an orthonormal basis ``Q`` of each isotypic subspace,
7. restricted representation ``A_g = Q^d R(g) Q`` (d^2 x d^2),
8. a basis of its commutant via the null space of ``X -> A_g X - X A_g``,
9. a generic minimal projector inside the commutant,
10. the irrep matrices ``rho(g) = U^d R(g) U`` from an orthonormal basis of
    the orbit ``{A_g w}``,
11. self-checks: unitarity, homomorphism, ``sum_g |chi(g)|^2 = N`` and
    character agreement with step 4.

The generic construction is deterministic: the random combinations use
``np.random.default_rng(0)`` and bases are built by SVD/`eigh`.

P1b increment (docs/v2-briefs/p1b.md):

* :func:`double_cover` now builds the *physical* double group from the
  2-cocycle of the SU(2) lifts (``(i, s) * (j, t) = (mul[i, j], s t eps[i, j])``)
  instead of the direct product ``G x Z2``.  Its ``elements`` are
  ``(3x3 spatial matrix, sign)`` pairs (type change versus P1a, documented in
  ``outputs/v2-p1b-report.md``); the irreps then automatically contain the
  spin (double-valued) representations ``G1``/``G2``.
* :func:`subduce` projects a direct sum of ``D^ell (x) D^S`` partial-wave
  blocks onto one irrep, reusing the migrated general kernels
  ``lattice_scattering.symmetry.rotations.partial_wave_rotation`` and
  ``lattice_scattering.symmetry.subduction.row_subduction`` (runtime import is
  permitted by ``docs/v2-interfaces.md`` §12 after the P1b contract update).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import permutations, product

import numpy as np

__all__ = ["Group", "Irrep", "double_cover", "little_group", "oh_group", "subduce"]

_TOL_OMEGA = 1e-8
_TOL_PROJECTOR = 1e-8
_TOL_UNITARY = 1e-9
_TOL_HOMOMORPHISM = 1e-8
_TOL_CHARACTER = 1e-7

# minimal-projector search (step 9 of the algorithm above): the commutant of an
# isotypic component is ``End(M) (x) 1``, so a single eigenvector of a generic
# Hermitian combination only spans a minimal left ideal when its eigenvalue is
# simple in ``M``.  Degenerate eigenvalues (multiplicity ``k * dimension``) are
# therefore grouped by a *relative* gap and either used as a block (``k == 1``)
# or searched recursively in the block's span (``k > 1``, bounded by
# ``_PROJECTOR_MAX_DEPTH``).
_TOL_EIGENGAP = 1e-8
_TOL_SINGULAR = 1e-7
_PROJECTOR_ATTEMPTS = 12
_PROJECTOR_MAX_DEPTH = 4
_SEED = 0

_INVERSION = -np.eye(3, dtype=int)
_IDENTITY = np.eye(3, dtype=int)
_RZ90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
_RX180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=int)
_SIGMA_V = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=int)


# --------------------------------------------------------------------------
# data containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Irrep:
    label: str
    dimension: int
    matrices: tuple[np.ndarray, ...]
    characters: np.ndarray

    def __post_init__(self):
        matrices = tuple(np.array(m, dtype=complex, copy=True) for m in self.matrices)
        characters = np.array(self.characters, dtype=complex, copy=True)
        for array in (*matrices, characters):
            array.setflags(write=False)
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "characters", characters)

    def character(self, index: int) -> complex:
        """Character on the element with the given index in ``Group.elements``."""
        return complex(self.characters[index])


@dataclass(frozen=True)
class Group:
    name: str
    elements: tuple[np.ndarray, ...]
    lifts: tuple[np.ndarray, ...] | None
    order: int
    parity_good: bool
    irreps: tuple[Irrep, ...]
    classes: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        def readonly(value):
            result = np.array(value, copy=True)
            result.setflags(write=False)
            return result
        elements = tuple((readonly(e[0]), e[1]) if isinstance(e, tuple) else readonly(e)
                         for e in self.elements)
        object.__setattr__(self, "elements", elements)
        if self.lifts is not None:
            object.__setattr__(self, "lifts", tuple(readonly(q) for q in self.lifts))

    @property
    def convention(self):
        from .api import CONVENTION
        return CONVENTION

    @cached_property
    def element_names(self):
        """Canonical named operations in exactly Group.elements order."""
        from .named import element_names
        doubled = isinstance(self.elements[0], tuple)
        lifts = _lifts(tuple(e[0] for e in self.elements[::2])) if doubled else self.lifts
        return tuple(element_names(self.elements, lifts, doubled))

    @cached_property
    def multiplication_table(self):
        from .api import compose
        from .named import named_operations
        names = self.element_names
        indices = {name: i for i, name in enumerate(names)}
        if not isinstance(self.elements[0], tuple):
            spatial = {_key(e): i for i, e in enumerate(self.elements)}
            indices = {name: spatial[_key(matrix)] for name, (matrix, _) in named_operations().items()
                       if _key(matrix) in spatial}
        result = np.array([[indices[compose(a, b)] for b in names] for a in names])
        result.setflags(write=False)
        return result

    def multiply(self, left, right):
        """Product of two indices in Group.elements order."""
        return int(self.multiplication_table[left, right])

    def inverse(self, index):
        return int(_inverses(self.multiplication_table)[index])

    def matrices(self, label, *, backend="numpy"):
        """One irrep in group order, as numeric arrays or exact SymPy matrices."""
        irrep = self.irrep(label)
        if backend == "numpy":
            return irrep.matrices
        if backend != "sympy":
            raise ValueError("backend must be numpy or sympy")
        from .api import irrep_matrices
        direction = _GROUP_DIRECTIONS.get(id(self))
        if direction is None:
            raise NotImplementedError("exact view requires a canonical little_group")
        values = irrep_matrices(direction, irrep.label)
        return tuple(values[name] for name in self.element_names)

    def subduction(self, representations, label, *, backend="numpy", **budgets):
        """Shared row/copy convention for a carrier ordered as Group.elements."""
        from .subduction import row_subduction
        return row_subduction(representations, self.matrices(label, backend=backend), backend=backend, **budgets)

    def irrep(self, label: str) -> Irrep:
        label = label.replace("_", "")
        for candidate in self.irreps:
            if candidate.label == label:
                return candidate
        raise KeyError(f"group {self.name!r} has no irrep {label!r}")

    def character_table(self) -> np.ndarray:
        return np.array([irrep.characters for irrep in self.irreps], dtype=complex)


# --------------------------------------------------------------------------
# basic algebra of the permutation representation
# --------------------------------------------------------------------------
def _key(matrix) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(matrix, dtype=int).ravel())


def _matrix_key(matrix) -> tuple[int, ...]:
    """Key of a bare 3x3 spatial operation (double-group elements are pairs)."""
    return tuple(int(v) for v in np.asarray(matrix, dtype=int).ravel())


def _mul_table(elements) -> np.ndarray:
    index = {_key(element): i for i, element in enumerate(elements)}
    n = len(elements)
    mul = np.zeros((n, n), dtype=int)
    for i, left in enumerate(elements):
        for j, right in enumerate(elements):
            mul[i, j] = index[_key(left @ right)]
    return mul


def _identity_index(mul) -> int:
    n = mul.shape[0]
    for i in range(n):
        if np.array_equal(mul[i], np.arange(n)):
            return i
    raise ValueError("elements do not contain the identity")


def _inverses(mul) -> np.ndarray:
    one = _identity_index(mul)
    n = mul.shape[0]
    inverse = np.empty(n, dtype=int)
    for i in range(n):
        inverse[i] = int(np.argmax(mul[i] == one))
    return inverse


def _conjugacy_classes(mul):
    inverse = _inverses(mul)
    n = mul.shape[0]
    seen = np.zeros(n, dtype=bool)
    classes = []
    for i in range(n):
        if seen[i]:
            continue
        members = np.unique(mul[mul[:, i], inverse])
        seen[members] = True
        classes.append(tuple(int(m) for m in members))
    classes.sort(key=lambda cls: (len(cls), cls[0]))
    return tuple(classes)


def _permutation_matrices(mul):
    """``R(g)[mul[i, j], j] = 1``: the left regular representation on C[G].

    Acting on the basis vector ``e_{g_i}`` (unit entry at row ``i``) this gives
    ``R(g) e_{g_i} = e_{g g_i}``, so ``R(g) R(h) = R(gh)`` as required below.
    """
    n = mul.shape[0]
    matrices = []
    for i in range(n):
        matrix = np.zeros((n, n), dtype=complex)
        matrix[mul[i], np.arange(n)] = 1.0
        matrices.append(matrix)
    for i in range(n):
        for j in range(n):
            product_matrix = matrices[i] @ matrices[j]
            if not np.allclose(product_matrix, matrices[int(mul[i, j])], atol=_TOL_HOMOMORPHISM):
                raise RuntimeError("regular permutation representation is not a homomorphism")
    return tuple(matrices)


# --------------------------------------------------------------------------
# the general finite-group irrep algorithm
# --------------------------------------------------------------------------
def _omega_vectors(classes, operators):
    r = len(classes)
    rng = np.random.default_rng(0)
    coefficients = rng.standard_normal(r) + 1j * rng.standard_normal(r)
    combination = np.zeros_like(operators[0])
    for coefficient, operator in zip(coefficients, operators):
        combination = combination + coefficient * operator
    _, vectors = np.linalg.eig(combination)
    omegas = []
    for column in vectors.T:
        vector = column / np.linalg.norm(column)
        omegas.append(np.array([vector.conj() @ (op @ vector) for op in operators]))
    return omegas


def _cluster_omegas(omegas):
    clusters = []
    for omega in omegas:
        for cluster in clusters:
            if np.allclose(omega, cluster[0], atol=_TOL_OMEGA, rtol=_TOL_OMEGA):
                cluster[1].append(omega)
                break
        else:
            clusters.append((omega, [omega]))
    return [(representative, len(members), members) for representative, members in clusters]


def _null_space(matrix, tol=1e-9):
    """Orthonormal basis of the (right) null space of ``matrix``.

    The stacked commutator map has ``n * d^4`` rows but only ``d^4`` columns, so
    the null space is taken from the Gram matrix ``matrix^H matrix``: both give
    the same kernel, the second one is a small Hermitian eigenproblem.
    """
    if matrix.shape[0] > matrix.shape[1]:
        gram = np.conj(matrix.T) @ matrix
        gram = 0.5 * (gram + np.conj(gram.T))
        scale = max(float(np.max(np.abs(np.diag(gram)))), 1.0)
        values, vectors = np.linalg.eigh(gram)
        return vectors[:, values <= tol * scale]
    _, singular, vh = np.linalg.svd(matrix)
    rank = int(np.sum(singular > tol * max(matrix.shape) * max(singular[0], 1.0)))
    return vh[rank:].conj().T


def _restrict_commutant(commutant, subspace):
    """Restrict a commutant basis (columns of ``commutant``) to ``subspace``.

    ``subspace`` has orthonormal columns, so ``subspace^H X subspace`` is the
    restriction of the commutant element ``X`` (which still commutes with the
    restricted representation because ``subspace`` is an invariant subspace).
    """
    size = subspace.shape[1]
    columns = []
    for column in commutant.T:
        element = column.reshape((size, size), order="F")
        restricted = subspace.conj().T @ element @ subspace
        columns.append(restricted.reshape(-1, order="F"))
    return np.column_stack(columns)


def _minimal_projector_seed(basis, restricted, commutant, dimension, rng, depth=0):
    """Seed vector inside one minimal left ideal of the isotypic component.

    ``basis`` are the orthonormal columns of the ambient isotypic basis
    (``n x size``) and ``restricted`` is the representation on the current
    ``size``-dimensional block.  Returns a vector of length ``size`` whose
    group orbit spans exactly ``dimension`` dimensions, or ``None`` if every
    attempt failed (the caller raises; no silently wrong value is returned).
    """
    size = restricted[0].shape[0]
    if depth > _PROJECTOR_MAX_DEPTH:
        return None
    for _ in range(_PROJECTOR_ATTEMPTS):
        coefficients = rng.standard_normal(commutant.shape[1]) + 1j * rng.standard_normal(
            commutant.shape[1]
        )
        combination = np.zeros((size, size), dtype=complex)
        for coefficient, column in zip(coefficients, commutant.T):
            element = column.reshape((size, size), order="F")
            # ``c X + conj(c) X^H`` is Hermitian for every complex ``c`` and
            # spans the whole Hermitian part of the commutant over the reals:
            # ``c = a + i b`` gives ``a (X + X^H) + b i (X - X^H)``.  Using the
            # bare ``X + X^H`` with a real ``c`` is *not* generic - for the
            # two-dimensional irreps of ``O_h^D`` the numerical null-space basis
            # is skew (up to a scalar), so that combination collapses to a
            # multiple of the identity and no eigenvalue cluster is resolvable.
            combination = combination + coefficient * element + np.conj(coefficient) * element.conj().T
        if not np.allclose(combination, combination.conj().T, atol=_TOL_PROJECTOR):
            raise RuntimeError("commutant combination is not Hermitian")
        values, vectors = np.linalg.eigh(combination)
        tolerance = _TOL_EIGENGAP * max(1.0, float(np.max(np.abs(values))))
        blocks = []
        start = 0
        for index in range(1, size + 1):
            if index == size or values[index] - values[index - 1] > tolerance:
                blocks.append(vectors[:, start:index])
                start = index
        for block in blocks:
            rank = block.shape[1]
            if rank < dimension or rank % dimension:
                continue
            if rank == dimension:
                projector = block @ block.conj().T
                _, singular, _ = np.linalg.svd(basis @ projector, full_matrices=False)
                if singular[dimension - 1] <= _TOL_SINGULAR * max(singular[0], 1.0):
                    continue
                direction = rng.standard_normal(rank) + 1j * rng.standard_normal(rank)
                return block @ direction
            inner_restricted = [
                block.conj().T @ matrix @ block for matrix in restricted
            ]
            inner_commutant = _restrict_commutant(commutant, block)
            inner = _minimal_projector_seed(
                block, inner_restricted, inner_commutant, dimension, rng, depth + 1
            )
            if inner is not None:
                return block @ inner
    return None


def _core_irreps(mul, seed=_SEED):
    """All irreps of the group with multiplication table ``mul``.

    ``seed`` fixes the random Hermitian combination of the commutant used to
    pick the minimal projector, so callers (and tests) can reproduce a run.

    Returns ``(classes, entries)`` where each entry is
    ``(dimension, characters, matrices, q_basis)``.
    """
    n = mul.shape[0]
    classes = _conjugacy_classes(mul)
    permutations = _permutation_matrices(mul)
    operators = []
    for members in classes:
        operator = np.zeros((n, n), dtype=complex)
        for g in members:
            operator = operator + permutations[g]
        operators.append(operator)

    clusters = _cluster_omegas(_omega_vectors(classes, operators))
    sizes = np.array([len(members) for members in classes], dtype=float)

    entries = []
    for representative, count, _ in clusters:
        denominator = float(np.sum(np.abs(representative) ** 2 / sizes))
        squared = n / denominator
        dimension = int(round(np.sqrt(squared)))
        if dimension < 1 or abs(np.sqrt(squared) - dimension) > 1e-6:
            raise RuntimeError(
                f"non-integer irrep dimension {np.sqrt(squared)!r} for group of order {n}"
            )
        if count != dimension**2:
            raise RuntimeError(
                f"isotypic component of size {count} contradicts dimension {dimension}"
            )
        class_characters = dimension * representative / sizes
        characters = np.empty(n, dtype=complex)
        for members, value in zip(classes, class_characters):
            characters[list(members)] = value

        projector = np.zeros((n, n), dtype=complex)
        for g in range(n):
            projector = projector + np.conj(characters[g]) * permutations[g]
        projector = projector * (dimension / n)
        if not np.allclose(projector.conj().T, projector, atol=_TOL_PROJECTOR):
            raise RuntimeError("isotypic projector is not Hermitian")
        if not np.allclose(projector @ projector, projector, atol=_TOL_PROJECTOR):
            raise RuntimeError("isotypic projector is not idempotent")
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        support = eigenvalues > 0.5
        if int(np.sum(support)) != dimension**2:
            raise RuntimeError("isotypic projector has the wrong rank")
        basis = eigenvectors[:, support]

        restricted = [basis.conj().T @ permutations[g] @ basis for g in range(n)]
        d2 = dimension**2
        rows = []
        for matrix in restricted:
            rows.append(np.kron(np.eye(d2), matrix) - np.kron(matrix.T, np.eye(d2)))
        commutant = _null_space(np.vstack(rows))
        if commutant.shape[1] != d2:
            raise RuntimeError(
                f"commutant dimension {commutant.shape[1]} != {d2} for dimension {dimension}"
            )

        rng = np.random.default_rng(seed)
        seed_vector = _minimal_projector_seed(
            basis, restricted, commutant, dimension, rng
        )
        if seed_vector is None:
            raise RuntimeError(
                f"no minimal projector for dimension {dimension} in the group of order "
                f"{n} after {_PROJECTOR_ATTEMPTS} attempts up to recursion depth "
                f"{_PROJECTOR_MAX_DEPTH}"
            )
        orbit = np.column_stack([matrix @ seed_vector for matrix in restricted])
        u, singular, _ = np.linalg.svd(basis @ orbit, full_matrices=False)
        rank = int(np.sum(singular > _TOL_SINGULAR * max(singular[0], 1.0)))
        if rank != dimension:
            raise RuntimeError(
                f"orbit of the minimal projector spans {rank} != {dimension} dimensions"
            )
        unitary = u[:, :dimension]

        matrices = []
        for g in range(n):
            matrices.append(unitary.conj().T @ permutations[g] @ unitary)
        entries.append((dimension, characters, tuple(matrices), unitary))

        for g, matrix in enumerate(matrices):
            if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension), atol=_TOL_UNITARY):
                raise RuntimeError("constructed irrep matrix is not unitary")
            if abs(np.trace(matrix) - characters[g]) > _TOL_CHARACTER:
                raise RuntimeError("constructed irrep character disagrees with class character")
        total = float(sum(abs(np.trace(matrix)) ** 2 for matrix in matrices))
        if abs(total - n) > 1e-7:
            raise RuntimeError(f"sum_g |chi(g)|^2 = {total} != {n}")
        for g in range(n):
            for h in range(n):
                product_matrix = matrices[g] @ matrices[h]
                target = matrices[int(mul[g, h])]
                if not np.allclose(product_matrix, target, atol=_TOL_HOMOMORPHISM):
                    raise RuntimeError("constructed irrep matrices are not a homomorphism")

    entries.sort(key=lambda entry: (entry[0], _character_key(entry[1])))
    return classes, entries


def _character_key(characters):
    return tuple(np.round(np.concatenate([characters.real, characters.imag]), 9))


# --------------------------------------------------------------------------
# cubic operations and quaternion lifts
# --------------------------------------------------------------------------
def _cubic_operations():
    """All 48 3x3 signed permutation matrices (the order of an ``O_h`` group).

    Proper operations (``det = +1``) come first. Their first entries follow the
    lexicographic order of the permutation (rows 0,1,2), i.e. the column index
    sequence ``(0, 1, 2), (0, 2, 1), (1, 0, 2), ...``, exactly the order of the
    legacy ``proper_cubic_rotations``; improper operations follow in the same
    sign order, negated.
    """
    operations = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            matrix = np.zeros((3, 3), dtype=int)
            for column, row in enumerate(perm):
                matrix[row, column] = signs[column]
            operations.append(matrix)
    proper = [m for m in operations if _det(m) > 0]
    improper = [m for m in operations if _det(m) < 0]
    return tuple(proper) + tuple(improper)


def _det(matrix) -> int:
    return int(round(float(np.linalg.det(np.asarray(matrix, dtype=float)))))


def _quaternion_from_matrix(rotation):
    """Unit quaternion ``(w, x, y, z)`` of a proper rotation matrix."""
    m = np.asarray(rotation, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quaternion = np.array(q, dtype=float)
    return quaternion / np.linalg.norm(quaternion)


from .rotations import quaternion_rotation


def _lifts(elements):
    """SU(2) lifts: ``quaternion_rotation(q) == element`` for proper operations
    and ``== -element`` for improper ones (the inversion factor is implicit in
    the determinant, as in the legacy ``double_c4v`` convention)."""
    lifts = []
    for element in elements:
        proper = element if _det(element) > 0 else -element
        lifts.append(_quaternion_from_matrix(np.asarray(proper, dtype=float)))
    return tuple(lifts)


# --------------------------------------------------------------------------
# Mulliken labelling
# --------------------------------------------------------------------------
def _label(group_name, elements, entries):
    """Deterministic Mulliken labels for the single-valued irreps."""
    labels = [None] * len(entries)

    def chi(entry, index):
        return complex(entry[1][index])

    def find(matrix):
        key = _key(matrix)
        for i, element in enumerate(elements):
            if _key(element) == key:
                return i
        raise KeyError(f"label reference operation is not in {group_name}")

    if group_name == "Oh":
        inversion = find(_INVERSION)
        c4 = find(_RZ90)
        for k, (dimension, characters, _, _) in enumerate(entries):
            if dimension == 1:
                base = "A1" if chi(entries[k], c4).real > 0 else "A2"
                suffix = "g" if chi(entries[k], inversion).real > 0 else "u"
                labels[k] = base + suffix
            elif dimension == 2:
                labels[k] = "Eg" if chi(entries[k], inversion).real > 0 else "Eu"
            elif dimension == 3:
                base = "T1" if chi(entries[k], c4).real > 0 else "T2"
                suffix = "g" if chi(entries[k], inversion).real > 0 else "u"
                labels[k] = base + suffix
            else:
                labels[k] = f"{dimension}D{k}"
        return labels

    if group_name == "C4v":
        c4 = _find_any(elements, _RZ90)
        if c4 is None:
            # little group of a different C4 axis: use the 4-fold generator
            c4 = _first_of_order(elements, 4)
        sigma = _find_any(elements, _SIGMA_V)
        if sigma is None:
            sigma = _first_mirror(elements)
        for k, (dimension, characters, _, _) in enumerate(entries):
            if dimension == 2:
                labels[k] = "E"
            else:
                a = chi(entries[k], c4).real
                b = chi(entries[k], sigma).real
                labels[k] = {(1, 1): "A1", (1, -1): "A2", (-1, 1): "B1", (-1, -1): "B2"}[
                    (1 if a > 0 else -1, 1 if b > 0 else -1)
                ]
        return labels

    if group_name == "C3v":
        c3 = _first_of_order(elements, 3)
        sigma = _first_improper(elements)
        for k, (dimension, characters, _, _) in enumerate(entries):
            if dimension == 2:
                labels[k] = "E"
            else:
                pair = (
                    1 if chi(entries[k], c3).real > 0 else -1,
                    1 if chi(entries[k], sigma).real > 0 else -1,
                )
                labels[k] = {(1, 1): "A1", (1, -1): "A2"}[pair]
        return labels

    if group_name == "C2v":
        c2 = _first_of_order(elements, 2)
        sigma = _first_improper(elements)
        for k, (dimension, characters, _, _) in enumerate(entries):
            a = chi(entries[k], c2).real
            b = chi(entries[k], sigma).real
            labels[k] = {(1, 1): "A1", (1, -1): "A2", (-1, 1): "B1", (-1, -1): "B2"}[
                (1 if a > 0 else -1, 1 if b > 0 else -1)
            ]
        return labels

    if group_name == "C2":
        c2 = _first_of_order(elements, 2)
        for k, (dimension, characters, _, _) in enumerate(entries):
            labels[k] = "A" if chi(entries[k], c2).real > 0 else "B"
        return labels

    if group_name == "Cs":
        sigma = _first_improper(elements)
        for k, (dimension, characters, _, _) in enumerate(entries):
            labels[k] = "A'" if chi(entries[k], sigma).real > 0 else 'A"'
        return labels

    if group_name == "C3":
        return ["A"] if len(entries) == 1 else [f"E{k}" for k in range(len(entries))]

    if group_name == "C1":
        return ["A"]

    # unknown name: deterministic fallback keyed by dimension and characters
    return [f"{dimension}D{k}" for k, (dimension, _, _, _) in enumerate(entries)]


def _find_any(elements, matrix):
    key = _key(matrix)
    for i, element in enumerate(elements):
        if _key(element) == key:
            return i
    return None


def _first_improper(elements):
    for i, element in enumerate(elements):
        if _det(element) < 0:
            return i
    raise RuntimeError("group has no improper operation")


def _first_mirror(elements):
    """Deterministic reference mirror: improper order-2 operation with the
    lexicographically smallest matrix (identity read row-major)."""
    mirrors = [
        (i, _key(elements[i]))
        for i in range(len(elements))
        if _det(elements[i]) < 0 and np.array_equal(elements[i] @ elements[i], np.eye(3, dtype=int))
    ]
    if not mirrors:
        return _first_improper(elements)
    return min(mirrors, key=lambda pair: pair[1])[0]


def _first_of_order(elements, order):
    """Index of the first element whose exact order is ``order`` (identity skipped)."""
    identity = np.eye(3, dtype=int)
    for i, element in enumerate(elements):
        if np.array_equal(element, identity):
            continue
        current = np.array(element)
        for exponent in range(2, order + 1):
            current = current @ element
            if np.array_equal(current, identity):
                if exponent == order:
                    return i
                break
    raise RuntimeError(f"no element of order {order}")


def _uniquify(labels):
    seen = {}
    out = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            out.append(f"{label}#{seen[label]}")
        else:
            seen[label] = 0
            out.append(label)
    return out


# --------------------------------------------------------------------------
# group assembly
# --------------------------------------------------------------------------
def _build(name, elements, lifts, parity_good, elements_for_labels=None, classified=None, momentum=None):
    mul = _mul_table(elements)
    from .named import canonical_irreps
    named_entries = canonical_irreps(elements, momentum=(0,0,0) if momentum is None else momentum)
    if named_entries is not None:
        irreps = tuple(sorted((Irrep(label, matrices[0].shape[0], matrices,
                                     np.array([np.trace(matrix) for matrix in matrices]))
                               for label, matrices in named_entries), key=lambda r: (r.dimension,r.label)))
        return Group(name, tuple(elements), lifts, len(elements), parity_good, irreps, _conjugacy_classes(mul))
    classes, entries = _core_irreps(mul)
    named = elements if elements_for_labels is None else elements_for_labels
    label_name = name if classified is None else classified
    labels = _uniquify(_label(label_name, named, entries))
    irreps = tuple(
        Irrep(label=label, dimension=dimension, matrices=matrices, characters=characters)
        for label, (dimension, characters, matrices, _) in zip(labels, entries)
    )
    irreps = tuple(sorted(irreps, key=lambda irrep: (irrep.dimension, irrep.label)))
    return Group(
        name=name,
        elements=tuple(elements),
        lifts=lifts,
        order=len(elements),
        parity_good=parity_good,
        irreps=irreps,
        classes=classes,
    )


_OH_CACHE: dict[object, Group] = {}
_GROUP_DIRECTIONS: dict[int, tuple[int, int, int]] = {}


def oh_group() -> Group:
    if "oh" not in _OH_CACHE:
        elements = _cubic_operations()
        _OH_CACHE["oh"] = _build("Oh", elements, _lifts(elements), True)
        _GROUP_DIRECTIONS[id(_OH_CACHE["oh"])] = (0,0,0)
    return _OH_CACHE["oh"]


def _little_group_name(elements):
    """Group name from the order and the improper content, exactly as the
    P1a brief prescribes.  Returns ``None`` for orders outside that table;
    :func:`little_group` then uses ``f"order{n}"`` and the report lists it."""
    order = len(elements)
    has_improper = any(_det(element) < 0 for element in elements)
    if order == 48:
        return "Oh"
    if order == 8:
        return "C4v"
    if order == 6:
        return "C3v"
    if order == 4:
        return "C2v"
    if order == 3:
        return "C3"
    if order == 2:
        return "Cs" if has_improper else "C2"
    if order == 1:
        return "C1"
    return None


def little_group(d) -> Group:
    from ._operations import _momentum
    direction = np.asarray(_momentum(d), dtype=int)
    if direction.shape != (3,):
        raise ValueError("little_group expects a 3-vector d")
    key = ("little", tuple(int(v) for v in direction))
    if key not in _OH_CACHE:
        parent = oh_group()
        elements = tuple(g for g in parent.elements if np.array_equal(g @ direction, direction))
        classified = _little_group_name(elements)
        name = classified if classified is not None else f"order{len(elements)}"
        _OH_CACHE[key] = _build(
            name,
            elements,
            _lifts(elements),
            bool(np.all(direction == 0)),
            classified=classified,
            momentum=tuple(int(v) for v in direction),
        )
        _GROUP_DIRECTIONS[id(_OH_CACHE[key])] = tuple(int(v) for v in direction)
    return _OH_CACHE[key]


def _quaternion_product(a, b):
    """Hamilton product of two ``(w, x, y, z)`` quaternions (P1b cocycle)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _cocycle(group, mul_parent):
    """``eps[i, j] = +-1`` with ``q_i q_j = eps[i, j] q_{mul[i, j]}``."""
    n = group.order
    eps = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            product = _quaternion_product(group.lifts[i], group.lifts[j])
            target = group.lifts[int(mul_parent[i, j])]
            if np.allclose(product, target, atol=1e-9):
                eps[i, j] = 1
            elif np.allclose(product, -target, atol=1e-9):
                eps[i, j] = -1
            else:
                raise RuntimeError(
                    f"lifts of {group.name!r} are not closed under quaternion "
                    f"multiplication: q_{i} q_{j} is neither q_{int(mul_parent[i, j])} "
                    "nor its negative"
                )
    return eps


def _double_multiplication(mul_parent, eps):
    n = mul_parent.shape[0]
    size = 2 * n
    mul = np.zeros((size, size), dtype=int)
    for i in range(n):
        for j in range(n):
            spatial = int(mul_parent[i, j])
            for s in (1, -1):
                for t in (1, -1):
                    left = 2 * i + (0 if s == 1 else 1)
                    right = 2 * j + (0 if t == 1 else 1)
                    sign = s * t * int(eps[i, j])
                    mul[left, right] = 2 * spatial + (0 if sign == 1 else 1)
    return mul


def _double_cover_labels(group, entries):
    """Mulliken labels for the double group (single valued keep the parent
    label; double valued labelled by ``chi(r90) = +-sqrt(2)`` -> ``G1``/``G2``)."""
    n = group.order
    identity = _find_any(group.elements, _IDENTITY)
    center = 2 * identity + 1  # element (identity, -1)
    c4 = None
    for i in range(n):
        if _matrix_key(group.elements[i]) == _matrix_key(_RZ90):
            c4 = 2 * i  # (r90, +1) in the double-group element order
            break

    parent_characters = {
        irrep.label: np.array([complex(np.trace(irrep.matrices[i])) for i in range(n)])
        for irrep in group.irreps
    }
    labels = [None] * len(entries)
    used: set[str] = set()
    root_two = np.sqrt(2.0)
    for k, (dimension, characters, _, _) in enumerate(entries):
        if abs(complex(characters[center]).real - dimension) < 1e-7:
            values = np.array([characters[2 * i] for i in range(n)])
            best, best_deviation = None, np.inf
            for label, reference in parent_characters.items():
                if label in used:
                    continue
                deviation = float(np.max(np.abs(values - reference)))
                if deviation < best_deviation:
                    best, best_deviation = label, deviation
            if best is None or best_deviation >= 1e-6:
                raise RuntimeError(
                    "single-valued irrep of the double cover does not reproduce a "
                    f"parent character row (best deviation {best_deviation})"
                )
            labels[k] = best
            used.add(best)

    counter = 0
    for k, (dimension, characters, _, _) in enumerate(entries):
        if labels[k] is not None:
            continue
        counter += 1
        value = None if c4 is None else float(complex(characters[c4]).real)
        if value is not None and abs(value - root_two) < 1e-7 and "G1" not in used:
            labels[k] = "G1"
        elif value is not None and abs(value + root_two) < 1e-7 and "G2" not in used:
            labels[k] = "G2"
        else:
            while f"G{counter}" in used:
                counter += 1
            labels[k] = f"G{counter}"
        used.add(labels[k])

    if any(label is None for label in labels) or len(set(labels)) != len(labels):
        raise RuntimeError("double-cover labelling did not produce unique labels")
    return labels


_DOUBLE_CACHE: dict[tuple, Group] = {}
#: Memo of ``subduce`` results, keyed by the full immutable request.  Keeps the
#: (fixed by ordered projected columns) row bases cached between identical
#: calls, which the root scan relies on for reproducibility.
_SUBDUCE_CACHE: dict[tuple, dict] = {}


def double_cover(group: Group) -> Group:
    """Physical double group of ``group`` (2-cocycle of the SU(2) lifts).

    Elements are ``(3x3 spatial matrix, sign)`` pairs ordered
    ``(g_0, +1), (g_0, -1), (g_1, +1), ...``; the element ``(g, s)`` stands for
    the SU(2) lift ``s * q_g``.  ``Group.elements`` therefore changes *type*
    relative to P1a (documented in ``outputs/v2-p1b-report.md``), while
    ``Group.irrep``/``Group.character_table`` keep their semantics.  The
    returned group has ``lifts=None`` because the lift data is absorbed in the
    group law itself.

    Raises
    ------
    ValueError
        If ``group`` carries no SU(2) lifts (``group.lifts is None``).
    """
    parent = group
    if parent.lifts is None:
        raise ValueError(
            f"double_cover needs SU(2) lifts, but group {parent.name!r} has lifts=None"
        )
    n = parent.order
    if len(parent.lifts) != n:
        raise ValueError("double_cover needs one lift per group element")

    signature = (
        parent.name,
        n,
        _GROUP_DIRECTIONS.get(id(parent)),
        tuple(_matrix_key(element) for element in parent.elements),
        tuple(tuple(np.round(np.asarray(lift, dtype=float), 9)) for lift in parent.lifts),
    )
    cached = _DOUBLE_CACHE.get(signature)
    if cached is not None:
        return cached

    mul_parent = _mul_table(parent.elements)
    eps = _cocycle(parent, mul_parent)
    mul = _double_multiplication(mul_parent, eps)
    elements = tuple(
        (parent.elements[i], sign) for i in range(n) for sign in (1, -1)
    )

    from .named import canonical_irreps
    # The common fixed direction can be recovered from the spatial stabilizer.
    # Keep it explicit for every little_group-created object, including rotated axes.
    momentum = _GROUP_DIRECTIONS.get(id(parent))
    named_entries = None if momentum is None else canonical_irreps(elements, momentum=momentum, doubled=True, lifts=parent.lifts)
    if named_entries is not None:
        classes = _conjugacy_classes(mul)
        irreps = tuple(sorted((Irrep(label, matrices[0].shape[0], matrices,
                                    np.array([np.trace(matrix) for matrix in matrices]))
                              for label, matrices in named_entries), key=lambda r: (r.dimension,r.label)))
    else:
        classes, entries = _core_irreps(mul)
        labels = _double_cover_labels(parent, entries)
        irreps = tuple(
        sorted(
            (
                Irrep(label=label, dimension=dimension, matrices=matrices, characters=characters)
                for label, (dimension, characters, matrices, _) in zip(labels, entries)
            ),
            key=lambda irrep: (irrep.dimension, irrep.label),
        )
        )
    cover = Group(
        name=parent.name + "^D",
        elements=elements,
        lifts=None,
        order=2 * n,
        parity_good=parent.parity_good,
        irreps=irreps,
        classes=classes,
    )
    _DOUBLE_CACHE[signature] = cover
    if momentum is not None:
        _GROUP_DIRECTIONS[id(cover)] = momentum
    return cover


# --------------------------------------------------------------------------
# subduction of partial-wave sectors onto one irrep
# --------------------------------------------------------------------------
def subduce(sectors, group, irrep, *, intrinsic_parity=1, max_dimension=256) -> dict:
    """Subduce ``D^ell (x) D^S`` blocks of ``sectors`` onto ``irrep``.

    ``sectors`` is any sequence of objects with ``ell`` and ``twice_S``
    attributes (e.g. :class:`lattice_scattering.v2.core.jls.Sector`); the
    ``channel_index`` is ignored and duplicates ``(ell, twice_S)`` are dropped
    while keeping the first occurrence.  ``group`` may be a single-valued
    group or a group returned by :func:`double_cover`.

    For every element ``(m, s)`` the spatial part is used as
    ``proper = m if det m > 0 else -m`` with ``inverted = det m < 0`` and the
    SU(2) lift ``q = group.lifts[i]`` (or ``_quaternion_from_matrix(proper)``
    when the group has no lifts, i.e. for double groups).  The block is
    ``partial_wave_rotation(q, twice_S, (ell,), inverted=inverted,
    intrinsic_parity=intrinsic_parity)``; for a double-group element with
    ``s = -1`` it is additionally multiplied by ``(-1)**twice_S``, i.e.
    ``D^S(-q) = (-1)^{2S} D^S(q)`` while the orbital block only depends on
    ``q mod +-1``.  The total ``U(g)`` is the direct sum over the deduplicated
    sectors in input order.

    The carrier is ``group.irrep(irrep).matrices`` (aligned with
    ``group.elements``); ``row_subduction`` then supplies
    ``multiplicity/basis/row_bases/projector/rank``.

    Ordered projected carrier columns fix the row/copy basis independently of
    eigensolver phases. Results are memoised by convention, sectors, group table,
    irrep, parity and budget to avoid repeating the projection in root scans.
    """
    ordered = []
    seen = set()
    for sector in sectors:
        key = (int(sector.ell), int(sector.twice_S))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    if not ordered:
        raise ValueError("subduce requires at least one (ell, twice_S) sector")

    target = group.irrep(irrep)
    order = group.order
    doubled = bool(group.elements) and isinstance(group.elements[0], tuple)
    if not doubled and any(key[1] % 2 for key in ordered):
        raise ValueError(
            "half-integer twice_S requires the double cover: wrap the group with "
            f"double_cover(...) (got single-valued group {group.name!r})"
        )

    # The cache key must identify the group by its *content*: two distinct
    # little groups can share a name and an order (e.g. non-equivalent C2v
    # stabilisers of different momentum directions), so keying on the name
    # alone would alias their row bases.
    cache_key = (
        group.convention,
        tuple(ordered),
        group.name,
        order,
        tuple(
            (
                np.asarray(element[0] if doubled else element, dtype=int).tobytes(),
                int(element[1]) if doubled else 1,
            )
            for element in group.elements
        ),
        str(irrep),
        int(intrinsic_parity),
        int(max_dimension),
    )
    stored = _SUBDUCE_CACHE.get(cache_key)
    if stored is not None:
        return stored

    from .rotations import partial_wave_rotation

    block_size = {pair: (2 * pair[0] + 1) * (pair[1] + 1) for pair in ordered}
    dimension = sum(block_size.values())
    representations = []
    for index, element in enumerate(group.elements):
        if doubled:
            matrix, sign = element
            sign = int(sign)
        else:
            matrix, sign = element, 1
        matrix = np.asarray(matrix, dtype=int)
        determinant = _det(matrix)
        inverted = determinant < 0
        proper = matrix if determinant > 0 else -matrix
        if group.lifts is not None:
            quaternion = np.asarray(group.lifts[index], dtype=float)
        else:
            quaternion = _quaternion_from_matrix(np.asarray(proper, dtype=float))
        total = np.zeros((dimension, dimension), dtype=complex)
        offset = 0
        for ell, twice_S in ordered:
            block = partial_wave_rotation(
                quaternion,
                twice_S,
                (ell,),
                inverted=inverted,
                intrinsic_parity=intrinsic_parity,
                max_dimension=max_dimension,
            )
            if doubled and sign == -1:
                # D^S(-q) = (-1)^{2S} D^S(q); the orbital factor is q -> -q
                # invariant, so the whole spin block picks up this sign.
                block = block * ((-1) ** twice_S)
            size = block_size[(ell, twice_S)]
            total[offset : offset + size, offset : offset + size] = block
            offset += size
        representations.append(total)

    result = group.subduction(representations, irrep, max_dimension=max_dimension)
    result.update(irrep=irrep, dimension=target.dimension)
    _SUBDUCE_CACHE[cache_key] = result
    return result
