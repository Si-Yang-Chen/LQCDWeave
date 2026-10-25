"""Resolve named Laplace backends for the production generator."""

import numpy as np


def resolve_laplace_backend(backend, array_module=None):
    """Resolve an explicit name or preserve a caller-owned solver instance.

    This only wires adapters; it never initializes runtime or loads gauge data.
    """
    if backend is None:
        raise ValueError("backend must be an explicit name or a LaplaceBackend instance")
    if not isinstance(backend, str):
        return (np if array_module is None else array_module), backend
    if backend not in ("numpy", "cupy", "quda"):
        raise ValueError("backend must be 'numpy', 'cupy' or 'quda'")
    expected = "numpy" if backend == "numpy" else "cupy"
    if array_module is not None and getattr(array_module, "__name__", None) != expected:
        raise ValueError("array_module conflicts with the named backend")
    xp, solver = np, None
    if backend != "numpy":
        import cupy as xp
    if backend == "quda":
        from .laplace_quda import QudaLaplaceBackend
        solver = QudaLaplaceBackend()
    return xp, solver
