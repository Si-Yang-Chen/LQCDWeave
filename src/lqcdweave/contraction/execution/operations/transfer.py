"""Copy an already-resolved array into the caller-selected backend."""

from functools import partial
import numpy as np


def owns_numpy_result(value, backend):
    """Only native NumPy ownership proves a redundant snapshot unnecessary."""
    return backend is np and type(value) is np.ndarray and value.flags.owndata


def copy_to_backend(value, backend, *, phase, timed, device=False):
    """Make the explicit owned copy used for snapshots and transfers."""
    return timed(
        phase,
        partial(backend.array, value, copy=True),
        nbytes=int(value.nbytes),
        device=device,
    )


__all__ = ["copy_to_backend"]
