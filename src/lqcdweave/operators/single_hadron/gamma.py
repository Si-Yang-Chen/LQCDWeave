"""Euclidean DeGrand--Rossi Gamma matrices, indexed by Clifford bitmask."""

import numpy as np

from lqcdweave.contracts.elemental import integer


_GENERATORS = np.asarray([
    [[0,0,0,1j], [0,0,1j,0], [0,-1j,0,0], [-1j,0,0,0]],
    [[0,0,0,-1], [0,0,1,0], [0,1,0,0], [-1,0,0,0]],
    [[0,0,1j,0], [0,0,0,-1j], [-1j,0,0,0], [0,1j,0,0]],
    [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]],
], dtype=complex)


def gamma(index, *, array_module=np):
    index = integer(index, "gamma index", 0)
    if index > 15:
        raise ValueError("gamma index must be 0..15")
    result = np.eye(4, dtype=complex)
    for bit in range(4):
        if index & (1 << bit):
            result = result @ _GENERATORS[bit]
    return array_module.asarray(result)
