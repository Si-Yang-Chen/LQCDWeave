"""Canonical spin rotations with an explicit SU(2) quaternion lift."""
from numbers import Integral
import numpy as np
from .angular_momentum import _spin,_generators,spherical_vector_basis


def _quaternion(value):
    if np.iscomplexobj(value):raise ValueError('real unit quaternion required')
    q=np.asarray(value,dtype=float)
    if q.shape!=(4,) or not np.all(np.isfinite(q)) or not np.isclose(np.linalg.norm(q),1,atol=1e-12,rtol=0):
        raise ValueError('unit quaternion (w,x,y,z) required')
    return q/np.linalg.norm(q)


def quaternion_rotation(quaternion):
    """Proper active Cartesian rotation; q and -q have the same SO(3) image."""
    w,x,y,z=_quaternion(quaternion)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def spin_rotation(quaternion,twice_spin,*,max_dimension=256):
    """exp(-i theta n.J), m=-S..S; the lift sign is never discarded."""
    spin=_spin(twice_spin)
    if isinstance(max_dimension,bool) or not isinstance(max_dimension,Integral) or max_dimension<spin+1:
        raise ValueError('spin rotation exceeds dimension budget')
    q=_quaternion(quaternion);radius=np.linalg.norm(q[1:])
    if radius==0:
        return ((-1)**spin if q[0]<0 else 1)*np.eye(spin+1,dtype=complex)
    angle=2*np.arctan2(radius,q[0]);axis=q[1:]/radius
    generator=sum(component*matrix for component,matrix in zip(axis,_generators(spin)))
    values,vectors=np.linalg.eigh(generator)
    return (vectors*np.exp(-1j*angle*values))@vectors.conj().T


def sp_rotation(quaternion,twice_spin,*,inverted=False,intrinsic_parity=1,max_dimension=256):
    """S/P spherical orbital tensor canonical spin for R or inversion*R.

    Intrinsic parity is the two-particle parity product. Inversion contributes
    (-1)^ell to orbit and intrinsic_parity to the channel. The quaternion
    specifies the proper-rotation lift even for an improper spatial action.
    """
    spin=_spin(twice_spin)
    if isinstance(max_dimension,bool) or not isinstance(max_dimension,Integral) or max_dimension<4*(spin+1):
        raise ValueError('S/P rotation exceeds dimension budget')
    if type(inverted)!=bool or isinstance(intrinsic_parity,bool) or intrinsic_parity not in (-1,1):
        raise ValueError('boolean inversion and intrinsic parity +/-1 required')
    r=quaternion_rotation(quaternion)
    orbital=np.zeros((4,4),dtype=complex);orbital[0,0]=1
    u=spherical_vector_basis()
    orbital[1:,1:]=u.conj().T@((-r) if inverted else r)@u
    return (intrinsic_parity if inverted else 1)*np.kron(orbital,spin_rotation(quaternion,spin,max_dimension=max_dimension))


def partial_wave_rotation(quaternion, twice_spin, partial_waves, *, inverted=False,
                          intrinsic_parity=1, max_dimension=256):
    """Direct sum of D^ell tensor D^S in the explicitly supplied ell order.

    Each block uses m_ell=-ell..ell then m_S=-S..S. No helicity, channel
    or exchange projection is implied; this is a representation only.
    """
    spin=_spin(twice_spin)
    waves=tuple(partial_waves)
    if not waves or any(isinstance(l,bool) or not isinstance(l,Integral) or l<0 for l in waves) or len(set(waves))!=len(waves):
        raise ValueError('distinct nonnegative integer partial waves required')
    dimension=sum(2*l+1 for l in waves)*(spin+1)
    if isinstance(max_dimension,bool) or not isinstance(max_dimension,Integral) or max_dimension<dimension:
        raise ValueError('partial-wave rotation exceeds dimension budget')
    if type(inverted)!=bool or isinstance(intrinsic_parity,bool) or intrinsic_parity not in (-1,1):
        raise ValueError('boolean inversion and intrinsic parity +/-1 required')
    spin_matrix=spin_rotation(quaternion,spin,max_dimension=max_dimension)
    result=np.zeros((dimension,dimension),dtype=complex)
    offset=0
    for ell in waves:
        block=np.kron(spin_rotation(quaternion,2*ell,max_dimension=max_dimension),spin_matrix)
        if inverted: block*=intrinsic_parity*(-1)**ell
        size=len(block);result[offset:offset+size,offset:offset+size]=block;offset+=size
    return result
