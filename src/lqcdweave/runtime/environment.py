"""Hardware/software identity for a measured scheduling choice."""
from __future__ import annotations

import platform
import os

import numpy as np
import opt_einsum


def runtime_fingerprint(backend):
    """Return portable JSON metadata; no configuration data is included."""
    identity = {
        "python": platform.python_version(), "system": platform.system(),
        "machine": platform.machine(), "processor": platform.processor(),
        "numpy": np.__version__, "opt_einsum": opt_einsum.__version__,
        "backend": backend.__name__,
        "thread_settings": {
            key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUPY_ACCELERATORS")
        },
    }
    if backend.__name__ == "cupy":
        device = backend.cuda.runtime.getDeviceProperties(backend.cuda.runtime.getDevice())
        name = device["name"]
        identity.update({
            "cupy": backend.__version__,
            "cuda_runtime": backend.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": backend.cuda.runtime.driverGetVersion(),
            "gpu_name": name.decode() if isinstance(name, bytes) else str(name),
            "gpu_total_bytes": int(device["totalGlobalMem"]),
        })
    return identity


__all__ = ["runtime_fingerprint"]
