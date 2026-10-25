"""Small-lattice isotropic Wilson operator and CPU reference solves."""

import numpy as np

from lqcdweave.contracts.elemental import integer

# Copied from operators/single_hadron/gamma.py: fixed DeGrand--Rossi basis.
# Local adapter maintenance; no operator-module implementation dependency.
_WILSON_GENERATORS = np.asarray([
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
            result = result @ _WILSON_GENERATORS[bit]
    return array_module.asarray(result)

from .dense_dirac import DenseDiracBackend


def wilson_matrix(gauge, mass, *, t_boundary=-1, max_dimension=2048):
    """Build D=(m+4)I - 1/2 sum_mu [(1-gamma_mu)U_mu T+ + (1+gamma_mu)U_mu^-dag T-].

    a=r=1, Euclidean DeGrand--Rossi spin basis, periodic space; the temporal
    fermion sign is applied on boundary-crossing hops only. No clover term,
    anisotropy, kappa rescaling or even/odd preconditioning. Rows/columns use
    C-flattened (t,z,y,x,spin,color). Intended only for bounded CPU references.
    """
    limit = integer(max_dimension, "max_dimension", 1)
    if isinstance(mass, (bool, np.bool_)) or not np.isreal(mass) or not np.isfinite(mass):
        raise ValueError("finite real lattice mass required")
    if isinstance(t_boundary, (bool, np.bool_)) or t_boundary not in (-1, 1):
        raise ValueError("t_boundary must be +1 or -1")
    if len(gauge.shape) != 7 or tuple(gauge.shape[-3:]) != (4, 3, 3) or min(gauge.shape[:4]) < 1:
        raise ValueError("gauge requires (Lt,Lz,Ly,Lx,4,3,3)")
    sites = tuple(gauge.shape[:4])
    dimension = int(np.prod(sites)) * 12
    if dimension > limit:
        raise ValueError("Wilson matrix exceeds CPU reference max_dimension")
    source_dtype = np.dtype(gauge.dtype)
    links = np.asarray(gauge[:], dtype=np.complex128)
    if not np.isfinite(links).all():
        raise ValueError("finite gauge links required")
    su3_tolerance = 5e-6 if source_dtype.kind == "c" and source_dtype.itemsize == 8 else 1e-10
    if (np.max(np.abs(links @ links.conj().swapaxes(-1,-2)-np.eye(3))) > su3_tolerance
            or np.max(np.abs(np.linalg.det(links)-1)) > su3_tolerance):
        raise ValueError("Wilson reference requires SU(3) gauge links")
    matrix = (float(mass)+4) * np.eye(dimension, dtype=np.complex128)
    spin = np.eye(4)
    for point in np.ndindex(sites):
        row = int(np.ravel_multi_index(point, sites))*12
        for mu, axis in enumerate((3,2,1,0)):
            ahead, behind = list(point), list(point)
            ahead[axis] = (ahead[axis]+1) % sites[axis]
            behind[axis] = (behind[axis]-1) % sites[axis]
            forward_sign = t_boundary if mu == 3 and point[0] == sites[0]-1 else 1
            backward_sign = t_boundary if mu == 3 and point[0] == 0 else 1
            for neighbor, coefficient, spin_factor, color in (
                (ahead, forward_sign, spin-gamma(1<<mu), links[(*point,mu)]),
                (behind, backward_sign, spin+gamma(1<<mu), links[(*behind,mu)].conj().T),
            ):
                column = int(np.ravel_multi_index(tuple(neighbor), sites))*12
                matrix[row:row+12,column:column+12] -= .5*coefficient*np.kron(spin_factor,color)
    return matrix


class DenseWilsonBackend(DenseDiracBackend):
    """Build the physical Wilson matrix from links, then use checked dense solves."""

    gamma5_hermitian = True
    gamma5 = gamma(15)
    spin_basis = 'degrand_rossi'

    def load(self, gauge, parameters):
        self._matrix = None
        self.last_relative_residuals = None
        if (parameters["xi_0"] != 1 or parameters["nu"] != 1
                or parameters["clover_coeff_t"] != 0 or parameters["clover_coeff_r"] != 0
                or parameters["multigrid"] not in (None, False)):
            raise NotImplementedError("reference backend supports isotropic unimproved Wilson only")
        lx,ly,lz,lt = parameters["latt_size"]
        if tuple(gauge.shape) != (lt,lz,ly,lx,4,3,3):
            raise ValueError("gauge layout differs from requested lattice")
        matrix = wilson_matrix(gauge, parameters["mass"], t_boundary=parameters["t_boundary"],
                               max_dimension=self.max_dimension)
        super().load(matrix, parameters)
