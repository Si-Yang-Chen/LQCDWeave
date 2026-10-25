"""Complete sparse three-spin insertion rows for single baryons."""

from dataclasses import dataclass, field
import numpy as np

from lqcdweave.contracts.elemental import BaryonComponent, integer
from .operator import Operator
from .gamma import gamma


def spin_basis_matrix(name):
    if name == "degrand_rossi":
        return np.eye(4)
    if name == "dirac_pauli":
        # q_DP = U q_DR; both two-component blocks carry the same rotations.
        eye = np.eye(2)
        return np.block([[eye, eye], [eye, -eye]]) / np.sqrt(2)
    raise ValueError("spin basis must be degrand_rossi or dirac_pauli")





@dataclass(frozen=True, init=False)
class SparseSpinTensor:
    """Immutable nonzero entries (alpha,beta,gamma,value), with zero-based indices."""

    entries: tuple
    basis: str

    def __init__(self, entries, basis="degrand_rossi"):
        spin_basis_matrix(basis)
        merged = {}
        for a, b, c, value in entries:
            key = tuple(integer(i, "spin index", 0) for i in (a, b, c))
            value = complex(value)
            if max(key) > 3 or not np.isfinite(value):
                raise ValueError("spin indices must be 0..3 and coefficients finite")
            merged[key] = merged.get(key, 0) + value
        if any(not np.isfinite(v) for v in merged.values()):
            raise ValueError("nonfinite combined spin coefficient")
        object.__setattr__(self, "entries", tuple((*k, v) for k, v in sorted(merged.items()) if v != 0))
        object.__setattr__(self, "basis", basis)

    def dense(self, *, array_module=np):
        value = np.zeros((4, 4, 4), complex)
        for a, b, c, weight in self.entries:
            value[a, b, c] = weight
        return array_module.asarray(value)

    @classmethod
    def from_dense(cls, value, basis="degrand_rossi"):
        value = np.asarray(value)
        if value.shape != (4, 4, 4):
            raise ValueError("spin tensor must have shape (4,4,4)")
        return cls(((*k, value[k]) for k in np.ndindex(value.shape)), basis)

    def to_basis(self, basis):
        if basis == self.basis:
            return self
        # Coefficients transform contragrediently to q_target = U q_input.
        matrix = spin_basis_matrix(self.basis) @ spin_basis_matrix(basis).conj().T
        value = np.einsum("abc,ai,bj,ck->ijk", self.dense(), matrix, matrix, matrix)
        return self.from_dense(value, basis)

    def source(self):
        """Conjugate coefficients and apply gamma4 on each leg; retain slot labels.

        Adjoint fields are ordered bar(q2) bar(q1) bar(q0). No Wick permutation
        sign is included; slot 0 still refers to the original quark 0.
        """
        u = spin_basis_matrix(self.basis)
        g4 = u @ gamma(8) @ u.conj().T
        if self.basis == "dirac_pauli":
            g4 = np.diag([1, 1, -1, -1])
        value = np.einsum("abc,ia,jb,kc->ijk", self.dense().conj(), g4, g4, g4)
        return self.from_dense(value, self.basis)


def baryon_local_spin_rows(irrep, parity):
    """Return the normalized DR G1/H spin rows with their parity partners.

    Rows follow the existing high-to-low total-spin ordering.  The returned
    tensors are independent of momentum, spatial components, and flavor.
    """
    if parity not in ("g", "u"):
        raise ValueError("parity must be g or u")
    if irrep == "G1":
        rows = [[(0, 1, s, 1/2), (1, 0, s, -1/2)] for s in (0, 1)]
    elif irrep == "H":
        lo, hi = 0, 1
        rows = [[(lo, lo, lo, 1/np.sqrt(2))],
                [(lo, lo, hi, 1/np.sqrt(6)), (lo, hi, lo, 1/np.sqrt(6)), (hi, lo, lo, 1/np.sqrt(6))],
                [(lo, hi, hi, 1/np.sqrt(6)), (hi, lo, hi, 1/np.sqrt(6)), (hi, hi, lo, 1/np.sqrt(6))],
                [(hi, hi, hi, 1/np.sqrt(2))]]
    else:
        raise ValueError("local spin supports G1 and H; G2 requires spatial structure")
    eta = 1 if parity == "g" else -1
    paired_rows = [[entry for a, b, c, value in row
                    for entry in ((a, b, c, value),
                                  (a+2, b+2, c+2, eta*value))]
                   for row in rows]
    return tuple(SparseSpinTensor(row, "degrand_rossi") for row in paired_rows)


@dataclass(frozen=True)
class BaryonOperatorTerm:
    """Compatibility input record accepted by :class:`Operator`."""

    spin: SparseSpinTensor
    component: BaryonComponent
    coefficient: complex = 1
    flavors: tuple = ("q", "q", "q")
    momentum: int = 0
    endpoint_time_offsets: tuple[int, int, int] = field(default=(0, 0, 0), kw_only=True)

    def __post_init__(self):
        if not isinstance(self.spin, SparseSpinTensor) or not isinstance(self.component, BaryonComponent):
            raise ValueError("sparse spin and baryon component required")
        flavors = tuple(str(f) for f in self.flavors)
        if len(flavors) != 3 or any(not f for f in flavors):
            raise ValueError("three quark flavor labels are required")
        object.__setattr__(self, "flavors", flavors)
        offsets = tuple(integer(value, f"endpoint_time_offsets[{index}]")
                        for index, value in enumerate(self.endpoint_time_offsets))
        if len(offsets) != 3:
            raise ValueError("baryon terms require three ordered endpoint offsets")
        object.__setattr__(self, "endpoint_time_offsets", offsets)
        value = complex(self.coefficient)
        if not np.isfinite(value):
            raise ValueError("coefficient must be finite")
        object.__setattr__(self, "coefficient", value)
        integer(self.momentum, "momentum", 0)


class BaryonOperator:
    """Deprecated adapter; canonical storage and validation live in Operator."""

    def __new__(cls, name, insertion_rows, coefficients):
        return Operator(name, insertion_rows, coefficients)

    @classmethod
    def from_terms(cls, name, terms):
        return Operator.from_terms(name, terms)
