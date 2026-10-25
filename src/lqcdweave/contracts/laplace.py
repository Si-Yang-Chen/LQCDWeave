"""Backend-neutral boundary for single-process Laplace production."""

from typing import Protocol


class LaplaceBackend(Protocol):
    """One backend instance belongs to one generator, not concurrent owners.

    load owns external format/native-object conversion and returns owned complex
    links (4,Lt,Lz,Ly,Lx,3,3). No native object escapes this interface.
    invalidate discards cached representations after a core-side transform or
    failure. stout_smear returns owned links without modifying its array input.
    solve receives a physical time t on the original lattice, even when
    the caller selects sparse output slots. Links always retain physical Lt.
    solve returns vectors (Ne,Lz,Ly,Lx,3) and eigenvalues (Ne,).
    The backend must honor the requested convergence tolerance or raise on
    failure. For ordinary solves the generator normalizes mode order but does
    not independently impose a runtime residual-quality gate; that gate is distinct
    from tests that compare eigenpair residuals on reference inputs.
    Device initialization and MPI decomposition are outside this contract.

    Optional extension capability (not required for ordinary solves):
    solve_complement(links, latt_size, t, count, tol, prefix) returns the
    lowest count orthonormal eigenvectors in the supplied prefix's orthogonal
    complement, with their eigenvalues. prefix is (Nold,Lz,Ly,Lx,3); it must
    not be changed. The generator checks residuals, orthogonality and prefix
    spectral ordering against its actual loaded spatial operator. Backends
    without this method reject extension, with no solver fallback.
    """

    def load(self, handle, key, latt_size): ...
    def invalidate(self) -> None: ...
    def stout_smear(self, links, latt_size, nstep, rho): ...
    def solve(self, links, latt_size, t, count, tol, polynomial_degree, lambda_cut): ...
