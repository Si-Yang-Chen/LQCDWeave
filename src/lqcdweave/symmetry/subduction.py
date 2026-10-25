"""Character projection for a supplied, consistently ordered finite group."""
from numbers import Integral
import numpy as np


def _projected_columns(projector, *, exact=False):
    """Ordered Gram-Schmidt of P e_0, P e_1, ... fixes copy order and phase.

    This rule uses the carrier's ordered coordinates, never eigenvectors of a
    degenerate eigenspace. Two passes limit numeric loss of orthogonality.
    """
    import sympy as sp
    vectors = []
    for index in range(projector.shape[1]):
        vector = projector[:, index]
        for _ in range(2):
            for previous in vectors:
                inner = (previous.conjugate().T * vector)[0] if exact else np.vdot(previous, vector)
                vector = vector - previous * inner
        norm2 = sp.simplify((vector.conjugate().T * vector)[0]) if exact else np.vdot(vector, vector).real
        is_zero = (norm2 == 0) if exact else (norm2 < 1e-20)
        if is_zero:
            continue
        vector = vector / (sp.sqrt(norm2) if exact else np.sqrt(norm2))
        vectors.append(vector.applyfunc(sp.simplify) if exact else vector)
    if exact:
        return sp.ImmutableMatrix.hstack(*vectors) if vectors else sp.ImmutableMatrix.zeros(projector.rows, 0)
    return np.column_stack(vectors) if vectors else np.zeros((projector.shape[0], 0), complex)


def _exact_row_subduction(representations, carrier, max_dimension, max_group_order):
    import sympy as sp
    from .api import CONVENTION
    matrices = tuple(sp.ImmutableMatrix(u) for u in representations)
    carrier = tuple(sp.ImmutableMatrix(d) for d in carrier)
    if not matrices or len(matrices) != len(carrier) or len(matrices) > max_group_order:
        raise ValueError("matching nonempty group element lists within budget required")
    n, d = matrices[0].rows, carrier[0].rows
    if not 0 < max(n, d) <= max_dimension or min(n, d) < 1:
        raise ValueError("representation dimensions exceed budget or are empty")
    def zero(matrix):
        return all(sp.simplify(x) == 0 for x in matrix)
    for values, size in ((matrices, n), (carrier, d)):
        if any(u.shape != (size, size) or not zero(u.H * u - sp.eye(size)) for u in values):
            raise ValueError("unitary square representation matrices required")
    chars = [sp.trace(a) for a in carrier]
    if sp.simplify(sum(c.conjugate() * c for c in chars) / len(chars)) != 1:
        raise ValueError("characters fail irreducible normalization")
    factor = sp.Rational(d, len(matrices))
    units = tuple(sum((factor * a[row, 0].conjugate() * u for u, a in zip(matrices, carrier)),
                      sp.zeros(n)) for row in range(d))
    p00 = units[0].applyfunc(sp.simplify)
    if not zero(p00.H - p00) or not zero(p00 * p00 - p00):
        raise ValueError("row-zero matrix unit is not an orthogonal projector")
    initial = _projected_columns(p00, exact=True)
    copies = initial.cols
    rows = tuple(sp.ImmutableMatrix((unit * initial).applyfunc(sp.simplify)) for unit in units)
    basis = sp.ImmutableMatrix.hstack(*rows)
    projector = sum((factor * sp.trace(a).conjugate() * u for u, a in zip(matrices, carrier)), sp.zeros(n))
    if not zero(basis.H * basis - sp.eye(d * copies)) or not zero(basis * basis.H - projector):
        raise ValueError("row bases do not resolve the isotypic subspace")
    for u, a in zip(matrices, carrier):
        if not zero(u * basis - basis * sp.kronecker_product(a, sp.eye(copies))):
            raise ValueError("row bases do not intertwine the supplied representations")
    return dict(projector=sp.ImmutableMatrix(projector.applyfunc(sp.simplify)), basis=basis,
                rank=d * copies, multiplicity=copies, irrep_dimension=d, row_bases=rows,
                basis_labels=tuple(dict(row=r, copy_index=c) for r in range(d) for c in range(copies)),
                convention=CONVENTION)


def isotypic_subspace(representations,characters,*,irrep_dimension,max_dimension=256,max_group_order=128):
    """Return projector/basis for all copies of one supplied irrep.

    Caller supplies a unitary group representation and irreducible character
    in identical element order. This checks numerical consistency, not the
    group multiplication law or correctness of a physical irrep assignment.
    Columns span all rows and copies together; no row/occurrence labels are
    inferred from eigenvector order. Missing irreps return an empty basis.
    """
    for value in (irrep_dimension,max_dimension,max_group_order):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
            raise ValueError('positive integer dimensions and budgets required')
    if not 0<len(representations)<=max_group_order:
        raise ValueError('group order exceeds budget or is empty')
    matrices=np.asarray(representations,dtype=complex)
    if matrices.ndim!=3 or matrices.shape[1]!=matrices.shape[2] or not 0<matrices.shape[1]<=max_dimension or not np.all(np.isfinite(matrices)):
        raise ValueError('finite square representation matrices within dimension budget required')
    order,dimension,_=matrices.shape
    chars=np.asarray(characters,dtype=complex)
    if chars.shape!=(order,) or not np.all(np.isfinite(chars)):
        raise ValueError('one finite character per group element required')
    tolerance=1e-10
    if not np.isclose(np.vdot(chars,chars).real/order,1,rtol=0,atol=tolerance) or np.any(abs(chars)>irrep_dimension+tolerance):
        raise ValueError('characters fail irreducible normalization/dimension checks')
    identity=np.eye(dimension)
    if any(not np.allclose(u.conj().T@u,identity,rtol=0,atol=tolerance) for u in matrices):
        raise ValueError('unitary representation matrices required')
    projector=irrep_dimension/order*np.einsum('g,gij->ij',chars.conj(),matrices)
    if not np.allclose(projector,projector.conj().T,rtol=0,atol=tolerance) or not np.allclose(projector@projector,projector,rtol=0,atol=tolerance):
        raise ValueError('character sum is not an orthogonal projector')
    if any(not np.allclose(projector@u,u@projector,rtol=0,atol=tolerance) for u in matrices):
        raise ValueError('projector is not invariant under the supplied representation')
    values,vectors=np.linalg.eigh((projector+projector.conj().T)/2)
    rank=int(np.count_nonzero(values>.5))
    if rank%irrep_dimension:
        raise ValueError('projected rank is not a multiple of irrep dimension')
    basis=_projected_columns(projector)
    if basis.shape[1] != rank:
        raise ValueError('projected-column rank differs from projector rank')
    projector.setflags(write=False);basis.setflags(write=False)
    from .api import CONVENTION
    return {'convention':CONVENTION,'projector':projector,'basis':basis,'rank':rank,'multiplicity':rank//irrep_dimension,
            'irrep_dimension':int(irrep_dimension),'basis_labels':None}


def row_subduction(representations,irrep_matrices,*,max_dimension=256,max_group_order=128,backend="numpy"):
    """Aligned row/copy bases from P_r0 = d/G sum conjugate(D_r0(g)) U(g).

    Copy indices are numerical basis indices, not inferred J/ell/S labels.
    Ordered projected carrier columns fix copy order and phase for both exact
    and numeric backends. Caller supplies the same ordered carrier basis.
    """
    for value in (max_dimension,max_group_order):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
            raise ValueError('positive integer dimensions and budgets required')
    if backend == "sympy":
        return _exact_row_subduction(representations,irrep_matrices,max_dimension,max_group_order)
    if backend != "numpy":
        raise ValueError("backend must be numpy or sympy")
    if not 0<len(irrep_matrices)<=max_group_order:
        raise ValueError('irrep group order exceeds budget or is empty')
    carrier=np.asarray(irrep_matrices,dtype=complex)
    if carrier.ndim!=3 or carrier.shape[1]!=carrier.shape[2] or not carrier.shape[1] or not np.all(np.isfinite(carrier)):
        raise ValueError('finite square irrep matrices required')
    dimension=carrier.shape[1]
    if dimension>max_dimension:
        raise ValueError('irrep dimension exceeds budget')
    if any(not np.allclose(d.conj().T@d,np.eye(dimension),rtol=0,atol=1e-10) for d in carrier):
        raise ValueError('unitary irrep matrices required')
    result=isotypic_subspace(representations,np.trace(carrier,axis1=1,axis2=2),
                            irrep_dimension=dimension,max_dimension=max_dimension,max_group_order=max_group_order)
    matrices=np.asarray(representations,dtype=complex)
    if len(carrier)!=len(matrices):
        raise ValueError('group element counts differ')
    factor=dimension/len(matrices)
    p00=factor*np.einsum('g,gij->ij',carrier[:,0,0].conj(),matrices)
    if not np.allclose(p00,p00.conj().T,rtol=0,atol=1e-10) or not np.allclose(p00@p00,p00,rtol=0,atol=1e-10):
        raise ValueError('row-zero matrix unit is not an orthogonal projector')
    initial=_projected_columns(p00)
    copies=result['multiplicity']
    if initial.shape[1]!=copies:
        raise ValueError('row-zero multiplicity differs from character projection')
    rows=[]
    for row in range(dimension):
        unit=factor*np.einsum('g,gij->ij',carrier[:,row,0].conj(),matrices)
        rows.append(unit@initial)
    basis=np.concatenate(rows,axis=1)
    if not np.allclose(basis.conj().T@basis,np.eye(dimension*copies),rtol=0,atol=1e-10) or not np.allclose(basis@basis.conj().T,result['projector'],rtol=0,atol=1e-10):
        raise ValueError('row bases do not resolve the isotypic subspace')
    for u,d in zip(matrices,carrier):
        if not np.allclose(u@basis,basis@np.kron(d,np.eye(copies)),rtol=0,atol=1e-10):
            raise ValueError('row bases do not intertwine the supplied representations')
    for row in rows:row.setflags(write=False)
    basis.setflags(write=False)
    from .api import CONVENTION
    result.update(convention=CONVENTION,basis=basis,row_bases=tuple(rows),basis_labels=tuple(
        {'row':row,'copy_index':copy} for row in range(dimension) for copy in range(copies)))
    return result
