"""Numerical views of the single canonical exact operator representations."""
from functools import lru_cache

import numpy as np


def _array(matrix):
    return np.asarray(matrix, dtype=complex)


@lru_cache(maxsize=1)
def named_operations():
    from .gen_hardcoded_rep import genIrrepOhD
    from .hardcoded_rep import Fermion_rep
    spatial = genIrrepOhD("T_1", -1)
    return {name: (np.rint(_array(matrix).real).astype(int), _array(Fermion_rep[name])[:2,:2])
            for name, matrix in spatial.items()}


def element_names(elements, lifts=None, doubled=False):
    """Name Cartesian operations and their physical SU(2) lifts exactly once."""
    from .rotations import spin_rotation
    operations = named_operations()
    names = []
    for i, element in enumerate(elements):
        matrix, sign = element if doubled else (element, 1)
        matches = [(name, spin) for name, (rotation, spin) in operations.items() if np.array_equal(matrix, rotation)]
        if doubled:
            # Exact operator table uses spinor order (+1/2,-1/2); spin_rotation
            # explicitly uses (-1/2,+1/2). This is a basis-order definition,
            # not a fitted intertwiner between two independently built irreps.
            expected = sign * spin_rotation(lifts[i // 2], 1)[::-1,::-1]
            matches = [(name, spin) for name, spin in matches if np.allclose(spin, expected, atol=1e-9, rtol=0)]
            if len(matches) != 1:
                raise ValueError("named operation has no unique physical SU(2) lift")
        if not matches:
            raise ValueError("operation is not cubic")
        names.append(matches[0][0])
    return names


def canonical_irreps(elements, *, momentum=(0,0,0), doubled=False, lifts=None):
    """Return (canonical label, exact numeric matrices), or None for generic groups.

    Underscores are lexical aliases only: T_1u and T1u have identical row bases.
    For known little groups the matrices originate solely in hardcoded_rep.
    """
    from .gen_hardcoded_rep import momentunSymplify
    from .api import irrep_matrices
    from .hardcoded_rep import OD_irreps, little_group_irreps, refRotateDict
    p = list(momentum)
    if not any(p):
        specifications = [(name, parity) for name in OD_irreps for parity in (1,-1)]
    else:
        simplified = ",".join(str(int(x)) for x in momentunSymplify(p))
        reference = next((key for key, star in refRotateDict.items() if simplified in star), None)
        if reference is None:
            return None
        specifications = [(name, None) for name in little_group_irreps[reference]]
    names = element_names(elements, lifts, doubled)
    results = []
    for name, parity in specifications:
        matrices = irrep_matrices(p, name, parity)
        dimension = matrices["iden"].rows
        central = _array(matrices["riden"])
        if not doubled and not np.allclose(central, np.eye(dimension), atol=1e-12):
            continue
        label = name.replace("_", "") + ("g" if parity == 1 else "u" if parity == -1 else "")
        values = tuple(_array(matrices[key]) for key in names)
        for value in values:
            value.setflags(write=False)
        results.append((label, values))
    if sum(matrices[0].shape[0]**2 for _, matrices in results) != len(elements):
        raise ValueError("canonical irreps do not exhaust the supplied group")
    return results
