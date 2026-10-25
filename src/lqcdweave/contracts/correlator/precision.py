"""One floating computation dtype for a complete contraction task."""
import numpy as np

_SUPPORTED = frozenset(map(np.dtype, ('float32', 'float64', 'complex64', 'complex128')))


def declared_contraction_dtype(value):
    """Resolve an explicit task declaration without inspecting or reading data."""
    if value is None:
        return None
    dtype = np.dtype(value)
    if dtype not in _SUPPORTED:
        raise TypeError('contraction dtype must be float32/64 or complex64/128')
    return dtype


def minimum_contraction_dtype(dtypes):
    """Use the least precise floating component, preserving complex values.

    Integer index arrays are not contraction inputs and must not be supplied.
    Integer/bool tensor constants do not lower floating precision.
    """
    values = tuple(np.dtype(value) for value in dtypes)
    floating = tuple(value for value in values if value.kind in 'fc')
    if not floating:
        return None
    if any(value not in _SUPPORTED for value in floating):
        raise TypeError('contractions support float32/64 and complex64/128 inputs')
    component_bytes = min(value.itemsize // (2 if value.kind == 'c' else 1) for value in floating)
    complex_values = any(value.kind == 'c' for value in floating)
    return np.dtype(('complex' if complex_values else 'float')
                    + str(component_bytes * (16 if complex_values else 8)))


def require_uniform_contraction_dtype(specs):
    """Reject mixed precision before numerical reads or kernel binding."""
    dtypes = {np.dtype(spec.dtype) for spec in specs.values()}
    if len(dtypes) > 1:
        raise ValueError('mixed-precision contraction plans are unsupported; '
                         'normalize all inputs before planning')
    return next(iter(dtypes), None)
