"""Coefficient-only double-cubic coupling in the actual three-quark spin basis.

The matrix-unit algorithm is supplied by the bundled lattice-symmetry module;
no independent group tables or Clebsch--Gordan conventions are maintained here.
"""

from functools import lru_cache
import numpy as np

from lqcdweave.contracts.elemental import BaryonComponent, integer
from .operator import InsertionTerm


class SpinName:
    """Normalized DR three-quark spin rows and their inversion eigenvalue."""

    G1G, G1U, HG, HU = "G1g", "G1u", "Hg", "Hu"


def _symmetry():
    try:
        from lqcdweave.symmetry import irrep_matrices, row_subduction
    except ModuleNotFoundError as exc:
        if exc.name not in {"sympy", "scipy", "lazy_object_proxy"}:
            raise
        raise ModuleNotFoundError(
            f"Missing optional dependency {exc.name!r}. Automatic baryon irrep "
            "coupling requires Python 3.11+ and the [correlator] extra. "
            "Run python -m pip install -e '.[correlator]'. "
            "See docs/modules/operators/baryon_irrep_coupling.md.",
            name=exc.name,
        ) from exc
    return irrep_matrices, row_subduction


def spin_rows(name):
    from .baryon import baryon_local_spin_rows
    from .baryon_spin import BaryonSpinOperator
    if isinstance(name, BaryonSpinOperator):
        return name.rows
    if name not in (SpinName.G1G, SpinName.G1U, SpinName.HG, SpinName.HU):
        raise ValueError("spin must be G1g, G1u, Hg or Hu")
    return baryon_local_spin_rows(name[:-1], name[-1])


@lru_cache(maxsize=28)
def spin_carrier(name):
    """Proper rotations on coefficient columns, not a guessed irrep row basis.

    A coefficient tensor transforms by S* on each leg when the quark field
    basis changes by S. Physical DR generators are -i gamma_j gamma_k / 2;
    the standard G1 matrices supply the named SU(2) lift, not the DR row basis.
    """
    matrices, _ = _symmetry()
    from .baryon_spin import BaryonSpinOperator
    if isinstance(name, BaryonSpinOperator):
        from types import MappingProxyType
        result = {g:np.asarray(m,complex) for g,m in matrices((0,0,0),name.irrep).items()}
        for matrix in result.values():
            matrix.setflags(write=False)
        return MappingProxyType(result)
    from .gamma import gamma
    pauli = np.asarray([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    dr_pauli = [-1j * gamma(j) @ gamma(k) for j, k in ((2, 4), (4, 1), (1, 2))]
    rows = spin_rows(name)
    columns = np.column_stack([row.dense().reshape(-1) for row in rows])
    result = {}
    for element, fundamental in matrices((0, 0, 0), "G1").items():
        s = np.asarray(fundamental, dtype=complex)
        dr = np.trace(s) / 2 * np.eye(4, dtype=complex)
        for sigma, physical in zip(pauli, dr_pauli):
            dr += np.trace(sigma @ s) / 2 * physical
        action = np.kron(np.kron(dr.conj(), dr.conj()), dr.conj())
        induced = columns.conj().T @ action @ columns
        if not np.allclose(action @ columns, columns @ induced, atol=1e-12, rtol=0):
            raise ArithmeticError("local spin rows are not closed under DR rotations")
        induced.setflags(write=False)
        result[element] = induced
    from types import MappingProxyType
    return MappingProxyType(result)


def coupled_baryon_rows(spin, spatial_rows, spatial_irrep, projection, *,
                        copy_index=0, spatial_matrices=None):
    """Couple supplied spatial rows to all rows of one target occurrence.

    Spatial terms are (coefficient, BaryonComponent), in canonical cubic row
    order unless matrices for that actual row basis are explicitly supplied.
    Spatial parity is carried by the final g/u suffix; there is no baryon C.
    """
    matrices, subduction = _symmetry()
    spins = spin_rows(spin)
    from .baryon_spin import BaryonSpinOperator
    spin_name = (spin.irrep + ('g' if spin.parity == 1 else 'u')
                 if isinstance(spin,BaryonSpinOperator) else spin)
    if not isinstance(spatial_irrep, str) or spatial_irrep[-1:] not in ("g", "u"):
        raise ValueError("spatial irrep requires g/u parity, e.g. T1u")
    right = spatial_irrep[:-1].replace("_", "")
    dimensions = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}
    if right not in dimensions:
        raise ValueError("spatial irrep must be A1, A2, E, T1 or T2 with g/u parity")
    target = projection.replace("_", "")
    parity = (1 if spin_name[-1] == "g" else -1) * (1 if spatial_irrep[-1] == "g" else -1)
    if target[-1:] in ("g", "u"):
        requested = 1 if target[-1] == "g" else -1
        if requested != parity:
            raise ValueError("target parity differs from spin times spatial parity")
        target = target[:-1]
    if target not in ("G1", "G2", "H"):
        raise ValueError("baryon target must be G1, G2 or H")
    spatial_rows = tuple(tuple(row) for row in spatial_rows)
    if len(spatial_rows) != dimensions[right] or any(not row for row in spatial_rows):
        raise ValueError("spatial row count differs from irrep dimension or a row is empty")
    for row in spatial_rows:
        for coefficient, component in row:
            if not isinstance(component, BaryonComponent):
                raise TypeError("baryon spatial terms require BaryonComponent keys")
            if not np.isfinite(complex(coefficient)):
                raise ValueError("spatial coefficients must be finite")
    left = spin_carrier(spin)
    canonical = matrices((0, 0, 0), right)
    supplied = canonical if spatial_matrices is None else spatial_matrices
    if set(supplied) != set(left):
        raise ValueError("spatial matrices must cover the same named proper double-cubic operations")
    numeric = {g: np.asarray(supplied[g], dtype=complex) for g in left}
    if any(a.shape != (dimensions[right], dimensions[right]) for a in numeric.values()):
        raise ValueError("spatial matrix dimensions differ from row count")
    # Validate the supplied representation, including its irrep assignment.
    if spatial_matrices is not None:
        from lqcdweave.symmetry import compose
        if any(not np.allclose(numeric[a] @ numeric[b], numeric[compose(a, b)],
                               atol=1e-10, rtol=0) for a in left for b in left):
            raise ValueError("spatial matrices violate the group multiplication law")
        if any(not np.isclose(np.trace(numeric[g]), complex(canonical[g].trace()),
                              atol=1e-10, rtol=0) for g in left):
            raise ValueError("spatial matrices have the wrong irrep characters")
    product = [np.kron(left[g], numeric[g]) for g in left]
    target_matrices = matrices((0, 0, 0), target)
    reduction = subduction(product, [np.asarray(target_matrices[g], complex) for g in left])
    index = integer(copy_index, "copy_index", 0)
    if index >= reduction["multiplicity"]:
        raise ValueError(f"{spin_name[:-1]} x {right} has {reduction['multiplicity']} copies of {target}")
    coefficients = np.column_stack([basis[:, index] for basis in reduction["row_bases"]])
    coefficients.setflags(write=False)
    rows = []
    for column in coefficients.T:
        terms = []
        for si, spin_row in enumerate(spins):
            for ri, spatial in enumerate(spatial_rows):
                weight = column[si * len(spatial_rows) + ri]
                # Only remove numerical zeros of the group projector, never
                # threshold the user's spatial or spin coefficients.
                if abs(weight) > 1e-12:
                    terms.extend(InsertionTerm(component, spin=spin_row,
                                               coefficient=weight * coefficient)
                                 for coefficient, component in spatial)
        rows.append(tuple(terms))
    return tuple(rows), coefficients, reduction["multiplicity"], parity, target
