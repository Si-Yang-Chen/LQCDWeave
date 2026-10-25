"""Ordered covariant transport, including temporal paths."""

import numpy as np

from lqcdweave.contracts.elemental import GaugePath, integer


def path_transport(links, path, t, boundary="periodic", *, array_module=np):
    """Return U_path(x) on one anchor slice, or None outside open time bounds.

    links has shape (4,Lt,Lz,Ly,Lx,3,3). Spatial directions are periodic.
    """
    xp = array_module
    path = GaugePath.coerce(path)
    t = integer(t, "anchor time", 0)
    if boundary not in ("periodic", "open") or t >= links.shape[1]:
        raise ValueError("invalid time or boundary")
    lo, hi = path.time_bounds
    if boundary == "open" and (t + lo < 0 or t + hi >= links.shape[1]):
        return None
    offset = [0] * 4
    result = xp.broadcast_to(xp.eye(3, dtype=links.dtype), links.shape[2:]).copy()
    for step in path.steps:
        axis, sign = path.AXES[step], path.SIGNS[step]
        if sign < 0:
            offset[axis] -= 1
        link = xp.roll(links[axis, (t + offset[3]) % links.shape[1]],
                       tuple(-d for d in offset[:3]), axis=(2, 1, 0))
        if sign < 0:
            link = link.conj().swapaxes(-1, -2)
        result = result @ link
        if sign > 0:
            offset[axis] += 1
    return result


def apply_transport(transport, vectors, displacement, *, array_module=np):
    xp = array_module
    dx, dy, dz = displacement[:3]
    shifted = xp.roll(vectors, (-dz, -dy, -dx), axis=(1, 2, 3))
    return xp.einsum("zyxab,ezyxb->ezyxa", transport, shifted, optimize=True)


def momentum_phase(latt_size, momentum, displacement=(0, 0, 0), convention="midpoint", *, array_module=np):
    xp = array_module
    lx, ly, lz, _ = latt_size
    z, y, x = xp.indices((lz, ly, lx))
    if convention == "none":
        return xp.ones((lz, ly, lx), dtype=complex)
    if convention not in ("midpoint", "anchor"):
        raise ValueError("phase convention must be midpoint, anchor, or none")
    half = displacement[:3] if convention == "midpoint" else (0, 0, 0)
    arg = sum(p * (coord + d / 2) / n
              for p, coord, d, n in zip(momentum, (x, y, z), half, (lx, ly, lz)))
    return xp.exp(2j * xp.pi * arg)
