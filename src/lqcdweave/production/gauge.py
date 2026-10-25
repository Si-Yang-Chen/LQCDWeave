"""Explicit spatial gauge-field preparation with bounded matrix batches."""

from functools import lru_cache
from importlib.resources import files
import numpy as np

from lqcdweave.contracts.elemental import integer


def project_su3(links, *, matrix_batch_size=4096, array_module=np, inplace=False):
    """Return a unitary, determinant-one projection of (...,3,3) matrices."""
    xp = array_module
    batch = integer(matrix_batch_size, "matrix_batch_size", 1)
    if links.shape[-2:] != (3,3):
        raise ValueError("links must end in (3,3)")
    result = links if inplace and links.dtype==xp.dtype(complex) and links.flags.c_contiguous else xp.array(links,dtype=complex,copy=True,order="C")
    flat = result.reshape(-1,3,3)
    for start in range(0,len(flat),batch):
        block=flat[start:start+batch]
        if xp is not np:
            identity=xp.eye(3,dtype=block.dtype)
            unitary_error=xp.max(xp.abs(block@block.conj().swapaxes(-1,-2)-identity))
            determinant_error=xp.max(xp.abs(xp.linalg.det(block)-1))
            if float(unitary_error.item())<=1e-12 and float(determinant_error.item())<=1e-12:
                continue
        u, _, vh = xp.linalg.svd(block)
        q = u @ vh
        q *= xp.exp(-1j*xp.angle(xp.linalg.det(q))/3)[...,None,None]
        flat[start:start+batch] = q
    return result


@lru_cache(maxsize=4)
def _cupy_stout_kernel(module_name):
    import cupy
    source=files("lqcdweave.production.elemental").joinpath("stout_smear.cu").read_text(encoding="utf-8")
    module=cupy.RawModule(code=source,options=("--std=c++11",),name_expressions=("stout_smear<double>",))
    return module.get_function("stout_smear<double>")


def stout_smear(links, nstep, rho, *, matrix_batch_size=4096, array_module=np, inplace=False):
    """Spatial staples only; links layout (4,Lt,Lz,Ly,Lx,3,3).

    Return an owned field, preserving temporal links. Hermitian exponentials
    use bounded batches rather than full-lattice eigensolver workspaces.
    """
    xp = array_module
    nstep = integer(nstep, "nstep", 0)
    batch = integer(matrix_batch_size, "matrix_batch_size", 1)
    if not np.isreal(rho) or not np.isfinite(rho) or links.ndim != 7 or links.shape[0] != 4 or links.shape[-2:] != (3,3):
        raise ValueError("finite real rho and (4,Lt,Lz,Ly,Lx,3,3) links required")
    result = links if inplace and links.dtype==xp.dtype(complex) else xp.array(links,dtype=complex,copy=True,order="C")
    if xp is not np and nstep:
        Lt,Lz,Ly,Lx=result.shape[1:5]
        # Keep the original one-thread-per-site/direction GPU kernel. It avoids
        # launching batched 3x3 eigensolvers for every spatial link.
        kernel=_cupy_stout_kernel(getattr(xp,"__name__","cupy"))
        spatial=xp.ascontiguousarray(result[:3])
        aliases_spatial=(spatial.data.ptr==result[:3].data.ptr)
        for _ in range(nstep):
            source=spatial.copy()
            kernel((Lx*Ly*Lz,3,1),(Lt,1,1),(spatial,source,float(rho),Lx,Ly,Lz,Lt))
        if not aliases_spatial:
            result[:3]=spatial
        return result
    for _ in range(nstep):
        previous = result[:3].copy()
        dagger = previous.conj().swapaxes(-1,-2)
        for mu in range(3):
            staple = xp.zeros_like(previous[mu])
            for nu in range(3):
                if mu == nu:
                    continue
                staple += previous[nu] @ xp.roll(previous[mu],-1,axis=3-nu) @ xp.roll(dagger[nu],-1,axis=3-mu)
                staple += (xp.roll(dagger[nu],1,axis=3-nu) @ xp.roll(previous[mu],1,axis=3-nu)
                            @ xp.roll(xp.roll(previous[nu],1,axis=3-nu),-1,axis=3-mu))
            omega = rho * staple @ dagger[mu]
            q = (omega.conj().swapaxes(-1,-2)-omega)*.5j
            trace = xp.trace(q,axis1=-2,axis2=-1)/3
            q -= trace[...,None,None]*xp.eye(3)
            qflat = q.reshape(-1,3,3)
            uflat = previous[mu].reshape(-1,3,3)
            output = result[mu].reshape(-1,3,3)
            for start in range(0,len(qflat),batch):
                s = slice(start,start+batch)
                values, vectors = xp.linalg.eigh(qflat[s])
                expq = (vectors*xp.exp(1j*values)[...,None,:]) @ vectors.conj().swapaxes(-1,-2)
                output[s] = expq @ uflat[s]
    return result
