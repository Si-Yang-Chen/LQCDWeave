"""Normalize already-read array bindings before compiling a dynamic task."""
from dataclasses import replace

from lqcdweave.contracts.correlator.precision import minimum_contraction_dtype


def normalize_array_precision(specs, bindings):
    """Use the common minimum precision rule on the current array backend.

    Uniform inputs are borrowed unchanged. Converted owners are call-local;
    reader data and structural caches are never updated with their values.
    """
    dtype = minimum_contraction_dtype(spec.dtype for spec in specs.values())
    if dtype is None or all(spec.dtype == dtype for spec in specs.values()):
        return specs, bindings
    return ({key: replace(spec, dtype=dtype) for key, spec in specs.items()},
            {key: value if value.dtype == dtype else value.astype(dtype, copy=False)
             for key, value in bindings.items()})
