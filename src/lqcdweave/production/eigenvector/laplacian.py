"""Gauge-covariant spatial Laplacian operator.

This module contains an array-only operator. Gauge loading, native backend
objects, and file formats belong to the adapter boundary; gauge transforms
reuse the shared production gauge kernels.
"""

import numpy as np


def spatial_laplacian(field, links, xp=np):
    """Apply ``6V - sum_i(U_i V_{+i} + U_i^dagger_{-i} V_{-i})``.

    ``links`` has shape ``(direction,Lz,Ly,Lx,color,color)`` and ``field``
    has shape ``(Lz,Ly,Lx,3)`` or ``(Lz,Ly,Lx,3,rhs)``. The optional RHS
    axis is retained by the operation.
    """
    volume_shape = tuple(links.shape[1:4])
    if tuple(links.shape) != (3, *volume_shape, 3, 3):
        raise ValueError("spatial links must have shape (3,Lz,Ly,Lx,3,3)")
    if tuple(field.shape[:4]) != (*volume_shape, 3) or field.ndim not in (4, 5):
        raise ValueError("field must have shape (Lz,Ly,Lx,3[,rhs])")
    vector = field[..., None] if field.ndim == 4 else field
    result = 6 * vector
    for mu, axis in enumerate((2, 1, 0)):
        forward = xp.einsum("...ab,...br->...ar", links[mu], xp.roll(vector, -1, axis=axis))
        dagger = links[mu].conj().swapaxes(-1, -2)
        backward = xp.roll(xp.einsum("...ab,...br->...ar", dagger, vector), 1, axis=axis)
        result = result - forward - backward
    return result[..., 0] if field.ndim == 4 else result


__all__ = ["spatial_laplacian"]
