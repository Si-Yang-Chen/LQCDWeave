"""Public exact operations over the shared canonical cubic representations.

Clients own their carrier objects (hadrons, paths, partial waves); this module
owns group conventions. Named dictionaries are exact views, not another basis.
"""
from functools import lru_cache
from types import MappingProxyType

import sympy as sp

CONVENTION = "cubic-canonical-v1/projected-columns-v1"


def _momentum(momentum):
    import numpy as np
    try:
        values = tuple(momentum)
        if len(values) != 3 or any(isinstance(x, (bool, np.bool_)) or int(x) != x for x in values):
            raise ValueError
        return tuple(int(x) for x in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("momentum must contain three finite integers") from error


def canonical_label(name, parity=None):
    from .gen_hardcoded_rep import _canonical_name
    name, parity = _canonical_name(name, parity)
    if parity not in (None, -1, 1):
        raise ValueError("parity must be +1 or -1")
    return name.replace("_", "") + ("g" if parity == 1 else "u" if parity == -1 else "")


def reference_frame(momentum):
    """Return (reference momentum, named rotation reference -> momentum).

    Only the direction is reduced; callers retain the physical momentum scale.
    """
    from .gen_hardcoded_rep import momentunSymplify
    from .hardcoded_rep import refRotateDict
    direction = ",".join(str(int(x)) for x in momentunSymplify(_momentum(momentum)))
    for reference, star in refRotateDict.items():
        if direction in star:
            return tuple(map(int, reference.split(","))), star[direction]
    raise NotImplementedError(f"no exact reference frame for {tuple(momentum)}")


@lru_cache(maxsize=512)
def _irrep_matrices(momentum, label, reference):
    from .gen_hardcoded_rep import genLittleGroupIrrep
    values = genLittleGroupIrrep(list(momentum), label, p_ref_irrep=reference)
    return MappingProxyType({name: sp.ImmutableMatrix(matrix) for name, matrix in values.items()})


def irrep_matrices(momentum, label, parity=None, *, reference=False):
    """Read-only exact matrices keyed by named operation.

    At rest an omitted parity denotes the proper rotation subgroup. Supply a
    parity-resolved label for the full group. Numeric clients use Group.irrep.
    """
    return _irrep_matrices(_momentum(momentum), canonical_label(label, parity), reference)


def compose(left, right):
    from .hardcoded_rep import OhD_mul
    return OhD_mul(left, right)


def inverse(element):
    from .hardcoded_rep import OhD_inv
    return OhD_inv(element)


def wigner_rotation(momentum, element):
    from .gen_hardcoded_rep import wignerRotate
    return wignerRotate(_momentum(momentum), element)


def stabilizer(momentum):
    """Named exact Cartesian operations fixing the supplied momentum."""
    p = sp.Matrix(_momentum(momentum))
    return MappingProxyType({name: matrix for name, matrix in irrep_matrices((0, 0, 0), "T1u").items()
                             if matrix * p == p})


def row_transfer(momentum, label, row, parity=None):
    """Coefficients (weight, operation) of the matrix unit P[row, 0]."""
    matrices = irrep_matrices(momentum, label, parity)
    dimension = next(iter(matrices.values())).rows
    if not 0 <= row < dimension:
        raise IndexError("irrep row out of range")
    factor = sp.Rational(dimension, len(matrices))
    return tuple((factor * matrix[row, 0].conjugate(), name)
                 for name, matrix in matrices.items() if matrix[row, 0] != 0)


def project_row(value, momentum, label, row, transform, parity=None):
    """Apply a shared matrix unit to a client carrier through transform(value, g)."""
    return sum((weight * transform(value, name)
                for weight, name in row_transfer(momentum, label, row, parity)), sp.S.Zero)


def restrict_irrep(momentum, source, target, parity=None):
    """Exact row/copy subduction of a rest irrep, in the actual momentum frame.

    Return the same result schema and projected-column convention as
    row_subduction; no legacy reduction or row-connection table is consulted.
    """
    from .subduction import row_subduction
    carrier = irrep_matrices((0, 0, 0), source, parity)
    target_matrices = irrep_matrices(momentum, target)
    names = tuple(target_matrices)
    return row_subduction([carrier[g] for g in names], list(target_matrices.values()), backend="sympy")
