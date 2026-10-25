"""Explicit small-matrix CPU backend, intended for numerical verification."""

import math
from numbers import Integral

import numpy as np


class DenseDiracBackend:
    """Use a complete matrix supplied by gauge_field.load(key).

    The matrix is indexed by C-flattened (t,z,y,x,spin,color). It already
    contains the complete operator: parameters are recorded, never used to
    synthesize a physical Dirac matrix. No stout or multigrid is implemented.
    A backend instance must not be shared by live generators.
    """

    array_module = np
    stream_ordered_solve = True
    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)

    def __init__(self, *, max_dimension=2048):
        if isinstance(max_dimension, bool) or not isinstance(max_dimension, Integral) or max_dimension < 1:
            raise ValueError("max_dimension must be a positive integer")
        self.max_dimension = int(max_dimension)
        self._matrix = None
        self.field_shape = None
        self.last_relative_residuals = None

    def load(self, gauge, parameters):
        self._matrix = None
        self.last_relative_residuals = None
        lx, ly, lz, lt = parameters['latt_size']
        shape = (lt, lz, ly, lx, 4, 3)
        n = math.prod(shape)
        if n > self.max_dimension:
            raise ValueError("matrix exceeds CPU reference max_dimension")
        if parameters['multigrid'] not in (None, False):
            raise NotImplementedError("DenseDiracBackend does not implement multigrid")
        matrix = np.asarray(gauge)
        if matrix.shape != (n, n) or matrix.dtype.kind not in 'fc' or not np.isfinite(matrix).all():
            raise ValueError(f"gauge payload must be a finite explicit matrix of shape {(n, n)}")
        self._matrix = np.array(matrix, dtype=np.complex128, copy=True)
        self.field_shape = shape

    def solve_into(self, rhs, solution, *, tol, maxiter):
        self.last_relative_residuals = None
        if self._matrix is None:
            raise RuntimeError("load an operator before solving")
        if not np.isfinite(tol) or not 0 < tol < 1:
            raise ValueError("tol must be in (0,1)")
        if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        # This is a direct solve; maxiter does not control its work.
        if rhs.ndim != 7 or rhs.shape[1:] != self.field_shape or rhs.shape[0] < 1:
            raise ValueError("rhs has the wrong field shape")
        if (solution.shape != rhs.shape or solution.dtype.kind != 'c'
                or rhs.dtype.kind not in 'fc' or not np.isfinite(rhs).all()):
            raise ValueError("invalid right-hand side or solution buffer")
        if np.shares_memory(rhs, solution):
            raise ValueError("rhs and solution buffers must not overlap")
        columns = rhs.reshape(len(rhs), -1).T
        values = np.linalg.solve(self._matrix, columns)
        with np.errstate(over='ignore', invalid='ignore'):
            solution[...] = values.T.reshape(rhs.shape)
        if not np.isfinite(solution).all():
            raise RuntimeError("non-finite solution")
        # Validate the actual stored solution, not just the temporary solve.
        stored = solution.reshape(len(rhs), -1).T
        error = np.linalg.norm(self._matrix @ stored - columns, axis=0)
        norms = np.linalg.norm(columns, axis=0)
        relative = error / np.where(norms == 0, 1, norms)
        self.last_relative_residuals = tuple(float(x) for x in relative)
        if not np.isfinite(relative).all() or np.any(relative > tol):
            raise RuntimeError(f"CPU solve residual exceeds tol={tol}")
