"""Small contraction primitives; point pairs remain sparse."""

import numpy as np

from lqcdweave.contracts.elemental import SparsePointBlock, integer


def re_combine(array, insertion_list, axis, out=None, *, array_module=np):
    """Combine a stored component axis with finite complex (coefficient,index) rows.

    Accumulates a row at a time; out must not alias array. No global autotuning
    state or dense coefficient matrix is required.
    """
    xp = array_module
    axis = integer(axis, "axis")
    if not -array.ndim <= axis < array.ndim:
        raise ValueError("axis outside array rank")
    axis %= array.ndim
    rows = []
    needs_complex=False
    for row in insertion_list:
        entries = []
        for coefficient, index in row:
            c, index = complex(coefficient), integer(index, "component index", 0)
            if not np.isfinite(c) or index >= array.shape[axis]:
                raise ValueError("nonfinite coefficient or missing component")
            needs_complex |= c.imag!=0
            entries.append((c, index))
        rows.append(entries)
    shape = list(array.shape)
    shape[axis] = len(rows)
    coefficient_dtype=np.complex128 if needs_complex else np.float64
    result_dtype=xp.result_type(array.dtype,coefficient_dtype)
    if out is None:
        out = xp.empty(shape, dtype=result_dtype)
    elif tuple(out.shape) != tuple(shape) or not xp.can_cast(result_dtype,out.dtype,casting="same_kind") or xp.may_share_memory(out, array):
        raise ValueError("out must have the result shape, compatible dtype, and not alias input")
    coefficients_host=np.zeros((len(rows),array.shape[axis]),dtype=np.complex128 if needs_complex else np.float64)
    for i,row in enumerate(rows):
        for c,index in row:
            coefficients_host[i,index]+=c if needs_complex else c.real
    coefficients=xp.asarray(coefficients_host)
    contracted=xp.tensordot(coefficients,array,axes=([1],[axis]))
    result=xp.moveaxis(contracted,0,axis)
    out[...] = result
    return out


def bilinear(left, right, phase=None, *, out=None, array_module=np):
    xp=array_module
    weighted=(left.conj() if phase is None else phase[None,...,None]*left.conj()).reshape(left.shape[0],-1)
    value=weighted @ right.reshape(right.shape[0],-1).T
    if out is None:
        return value
    out[...] = value
    return out


def point_pairs(left, right, displacement, spatial_size):
    """Match ordered unique XYZ coordinates without an Np x Np mask."""
    endpoint = (left + np.asarray(displacement[:3])) % np.asarray(spatial_size)
    lookup = {tuple(p): i for i, p in enumerate(right)}
    rows, cols = [], []
    for i, p in enumerate(endpoint):
        if tuple(p) in lookup:
            rows.append(i)
            cols.append(lookup[tuple(p)])
    return np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)


def sparse_point_block(transport, phase, points_left, points_right, displacement, spatial_size,
                       *, array_module=np, pair_indices=None):
    xp = array_module
    rows, cols = point_pairs(points_left, points_right, displacement, spatial_size) if pair_indices is None else pair_indices
    if transport is None:
        rows, cols = rows[:0], cols[:0]
    xyz = xp.asarray(points_left[rows])
    if len(rows):
        x, y, z = xyz.T
        values = phase[z, y, x, None, None] * transport[z, y, x]
    else:
        values = xp.empty((0, 3, 3), complex)
    return SparsePointBlock(xp.asarray(rows), xp.asarray(cols), values,
                            (len(points_left), len(points_right)))


def generate_baryon_elemental(vectors, momentum, lattice_shape, *, phase_sign=1,
                              block_size=256, backend=np, out=None):
    """Color epsilon contraction of three mode fields; no spin/flavor factors.

    Each field has shape (Ne,Lz,Ly,Lx,3); the three mode counts may differ.
    Spatial blocks bound the contraction temporary.
    """
    from .transport import momentum_phase
    xp = backend
    if len(vectors) != 3 or phase_sign not in (-1, 1):
        raise ValueError("three fields and phase_sign +/-1 are required")
    sizes = tuple(integer(n, "extent", 1) for n in lattice_shape)
    if len(sizes) != 3:
        raise ValueError("lattice_shape must be (Lx,Ly,Lz)")
    block_size = integer(block_size, "block_size", 1)
    arrays = [xp.asarray(v) for v in vectors]
    if any(v.ndim != 5 or v.shape[1:] != (*sizes[::-1], 3) for v in arrays):
        raise ValueError("vectors must have shape (Ne,Lz,Ly,Lx,3)")
    momentum = tuple(integer(n, "momentum") for n in momentum)
    if len(momentum) != 3:
        raise ValueError("momentum must have three components")
    phase = momentum_phase((*sizes, 1), momentum, array_module=xp).reshape(-1)
    if phase_sign == -1:
        phase = phase.conj()
    return trilinear(arrays, phase, block_size=block_size, out=out, array_module=xp)


def trilinear(vectors, phase, *, block_size=256, out=None, array_module=np):
    """Epsilon contraction without conjugation, bounded in the spatial index.

    Caller tiles all three mode axes. No volume x N0 x N1 x N2 array is built.
    out is overwritten and must not alias inputs.
    """
    xp = array_module
    block_size = integer(block_size, "block_size", 1)
    if len(vectors) != 3:
        raise ValueError("three vector groups required")
    arrays = [xp.asarray(v) for v in vectors]
    if any(v.ndim != 5 or min(v.shape) < 1 or v.shape[-1] != 3 for v in arrays):
        raise ValueError("vectors must have nonempty (Ne,Lz,Ly,Lx,3) shape")
    if any(v.shape[1:] != arrays[0].shape[1:] for v in arrays):
        raise ValueError("spatial shapes differ")
    phase = xp.asarray(phase).reshape(-1)
    arrays = [v.reshape(v.shape[0], -1, 3) for v in arrays]
    if phase.size != arrays[0].shape[1]:
        raise ValueError("phase volume differs")
    shape = tuple(v.shape[0] for v in arrays)
    if out is None:
        out = xp.empty(shape, dtype=complex)
    elif (out.shape != shape or out.dtype.kind != 'c'
          or any(xp.may_share_memory(out, v) for v in (*arrays, phase))):
        raise ValueError("out requires complex result shape and no input alias")
    result = out
    result.fill(0)
    permutations = ((0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1),
                    (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1))
    for start in range(0, len(phase), block_size):
        s = slice(start, start + block_size)
        for a, b, c, sign in permutations:
            result += sign * xp.einsum("x,ix,jx,kx->ijk", phase[s], arrays[0][:, s, a],
                                       arrays[1][:, s, b], arrays[2][:, s, c], optimize=True)
    return result
