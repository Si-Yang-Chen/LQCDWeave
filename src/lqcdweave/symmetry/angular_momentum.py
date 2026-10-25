"""Total-J projectors in a spherical product basis, without CG phase choices."""
from numbers import Integral
import numpy as np


def spherical_vector_basis():
    """Cartesian columns for spherical m=-1,0,+1."""
    return np.array([[1,0,-1],[-1j,0,-1j],[0,np.sqrt(2),0]])/np.sqrt(2)


def _spin(value):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
        raise ValueError('twice-spin must be a nonnegative integer')
    return int(value)


def _generators(twice_spin):
    j=twice_spin/2
    m=np.arange(twice_spin+1,dtype=float)-j
    raising=np.zeros((len(m),len(m)),dtype=complex)
    for index,magnetic in enumerate(m[:-1]):
        raising[index+1,index]=np.sqrt((j-magnetic)*(j+magnetic+1))
    lowering=raising.conj().T
    return (raising+lowering)/2,(raising-lowering)/(2j),np.diag(m)


def total_j_projectors(twice_first,twice_second,*,max_dimension=256):
    """Map 2J to orthogonal projectors for j1 tensor j2.

    Basis order: m1 increasing from -j1 to +j1, then m2 increasing.
    Inputs use twice-spin integers to represent half-integers exactly.
    Spectral projectors of (J1+J2)^2 avoid degenerate eigenvector phases.
    No little-group subduction, helicity rotation or exchange restriction.
    Allocation/eigensolve is rejected before work if dimension exceeds budget.
    """
    a=_spin(twice_first);b=_spin(twice_second)
    dimension=(a+1)*(b+1)
    if isinstance(max_dimension,bool) or not isinstance(max_dimension,Integral) or max_dimension<1:
        raise ValueError('positive integer dimension budget required')
    if dimension>max_dimension:
        raise ValueError(f'angular momentum dimension {dimension} exceeds budget')
    first=_generators(a);second=_generators(b)
    j1=a/2;j2=b/2
    casimir=(j1*(j1+1)+j2*(j2+1))*np.eye(dimension,dtype=complex)
    for left,right in zip(first,second):
        casimir+=2*np.kron(left,right)
    values,vectors=np.linalg.eigh(casimir)
    projectors={}
    assigned=np.zeros(dimension,dtype=bool)
    for twice_j in range(abs(a-b),a+b+1,2):
        j=twice_j/2
        mask=np.abs(values-j*(j+1))<=1e-10*max(1.,j*(j+1))
        if np.any(mask&assigned) or np.count_nonzero(mask)!=twice_j+1:
            raise ArithmeticError('total-J eigenspaces not numerically resolved')
        assigned|=mask
        basis=vectors[:,mask]
        projector=basis@basis.conj().T
        projector.setflags(write=False)
        projectors[twice_j]=projector
    if not np.all(assigned):
        raise ArithmeticError('unassigned total-J eigenvalues')
    return projectors


def coupled_j_basis(twice_first,twice_second,twice_j,*,max_dimension=256):
    """Product-basis columns |(j1 j2) J M>, with M ascending.

    Condon--Shortley phase: the highest-M column has positive coefficient
    at the largest admissible m1. Remaining columns follow total J-minus.
    Unlike a projector this fixes relative phases for inter-sector mixing.
    """
    a=_spin(twice_first);b=_spin(twice_second);total=_spin(twice_j)
    if total not in range(abs(a-b),a+b+1,2):raise ValueError('J violates angular momentum addition')
    projector=total_j_projectors(a,b,max_dimension=max_dimension)[total]
    magnetic=np.array([m1+m2 for m1 in range(-a,a+1,2) for m2 in range(-b,b+1,2)])
    indices=np.flatnonzero(magnetic==total)
    pivot=int(indices[-1])
    highest=projector[:,pivot].copy();highest[magnetic!=total]=0
    norm=np.linalg.norm(highest)
    if norm<1e-12:raise ArithmeticError('highest-weight phase is numerically unresolved')
    highest/=norm
    highest*=np.conj(highest[pivot])/abs(highest[pivot])
    first=_generators(a);second=_generators(b)
    lowering=np.kron(first[0]-1j*first[1],np.eye(b+1))+np.kron(np.eye(a+1),second[0]-1j*second[1])
    basis=np.zeros(((a+1)*(b+1),total+1),complex);basis[:,-1]=highest
    j=total/2
    for column in range(total,0,-1):
        m=-j+column
        basis[:,column-1]=lowering@basis[:,column]/np.sqrt((j+m)*(j-m+1))
    if not np.allclose(basis.conj().T@basis,np.eye(total+1),atol=1e-10,rtol=1e-12):raise ArithmeticError('coupled basis lost orthonormality')
    basis.setflags(write=False)
    return basis
