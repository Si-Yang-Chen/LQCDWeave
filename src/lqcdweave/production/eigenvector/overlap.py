"""Indexed point/eigenvector overlap gathers.

The stored value is ``M[t, i, p, a] = xi[t, i, z, y, x, a]`` for the
registered point coordinate ``(x, y, z)``.  This is the old ``eta† xi``
convention: the gather does not conjugate, normalize, scale, or phase-adjust
the eigenvector values.
"""

from numbers import Integral

import numpy as np


_COMPLEX_DTYPES = (np.dtype("complex64"), np.dtype("complex128"))


def _prefix_count(value, extent, name):
    if value is None:
        return extent
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer or None")
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    if count > extent:
        raise ValueError(f"{name} exceeds the available extent {extent}")
    return count


def _dtype(value, name):
    try:
        return np.dtype(value.dtype)
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{name} must be an array with a supported dtype") from exc


def _require_array_module(array_module):
    required = ("all", "empty", "isfinite", "unique", "array")
    if any(not callable(getattr(array_module, name, None)) for name in required):
        raise TypeError("array_module must provide NumPy-compatible array operations")


def _check_vectors(vectors, expected_rank, name):
    shape = getattr(vectors, "shape", None)
    if shape is None or len(shape) != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}")
    dtype = _dtype(vectors, name)
    if dtype not in _COMPLEX_DTYPES:
        raise TypeError(f"{name} must have complex64 or complex128 dtype")
    if any(isinstance(size, bool) or int(size) <= 0 for size in shape):
        raise ValueError(f"{name} extents must be positive")
    if int(shape[-1]) != 3:
        raise ValueError(f"{name} must have exactly three color components")
    return tuple(int(size) for size in shape), dtype


def _check_points(points, expected_rank, name):
    shape = getattr(points, "shape", None)
    if shape is None or len(shape) != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}")
    dtype = _dtype(points, name)
    if dtype.kind not in "iu":
        raise TypeError(f"{name} coordinates must use an integer dtype, not bool")
    if any(isinstance(size, bool) or int(size) <= 0 for size in shape):
        raise ValueError(f"{name} extents must be positive")
    if int(shape[-1]) != 3:
        raise ValueError(f"{name} coordinates must be ordered XYZ triples")
    return tuple(int(size) for size in shape)


def _selected_coordinates(points, point_count, extents, array_module, name):
    selected = points[:point_count]
    if not bool(array_module.all(selected >= 0)):
        raise ValueError(f"{name} coordinates must be non-negative")
    for axis, extent in enumerate(extents):
        if not bool(array_module.all(selected[..., axis] < extent)):
            raise ValueError(f"{name} coordinates are outside the spatial lattice")
    if int(array_module.unique(selected, axis=0).shape[0]) != point_count:
        raise ValueError(f"{name} coordinates must be unique")
    return selected


def _gather_row(eigenvectors, points, array_module):
    """The single numeric gather kernel shared by row and full-time calls."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    values = eigenvectors[:, z, y, x, :]
    return array_module.array(values, copy=True)


def _gather_checked_coordinates(eigenvectors, points, array_module):
    """Reuse accepted coordinates while checking the selected numerical input."""
    if not bool(array_module.all(array_module.isfinite(eigenvectors))):
        raise ValueError("selected eigenvectors must contain only finite values")
    return _gather_row(eigenvectors, points, array_module)


def calc_overlap_row(
    eigenvectors,
    points,
    usedNe=None,
    usedNp=None,
    *,
    array_module=np,
):
    """Gather one physical-time row into owned ``(Ne, Np, 3)`` storage.

    ``eigenvectors`` has axes ``(mode, z, y, x, color)`` and ``points`` has
    axes ``(point, xyz)`` with each coordinate stored as ``(x, y, z)``. Only
    the requested mode and point prefixes are validated and read.
    """
    _require_array_module(array_module)
    vector_shape, _dtype_value = _check_vectors(eigenvectors, 5, "eigenvectors")
    point_shape = _check_points(points, 2, "points")
    ne = _prefix_count(usedNe, vector_shape[0], "usedNe")
    np_count = _prefix_count(usedNp, point_shape[0], "usedNp")
    selected_points = _selected_coordinates(
        points, np_count, vector_shape[1:4][::-1], array_module, "points",
    )
    selected_vectors = eigenvectors[:ne]
    return _gather_checked_coordinates(selected_vectors, selected_points, array_module)


def get_overlap_matrix(
    eigenvector_data,
    point_data,
    usedNe=None,
    usedNp=None,
    *,
    array_module=np,
):
    """Gather every physical-time overlap row into owned ``(Lt, Ne, Np, 3)``.

    ``eigenvector_data`` has axes ``(time, mode, z, y, x, color)`` and
    ``point_data`` has axes ``(time, point, xyz)``. The point order and mode
    order are retained exactly. This routine does not conjugate or rescale the
    values and does not infer a sampling rule from the point count.
    """
    _require_array_module(array_module)
    vector_shape, vector_dtype = _check_vectors(
        eigenvector_data, 6, "eigenvector_data",
    )
    point_shape = _check_points(point_data, 3, "point_data")
    lt, mode_extent, lz, ly, lx, _color = vector_shape
    point_lt, point_extent, _xyz = point_shape
    if point_lt != lt:
        raise ValueError("eigenvector and point time extents must agree")
    ne = _prefix_count(usedNe, mode_extent, "usedNe")
    np_count = _prefix_count(usedNp, point_extent, "usedNp")

    # Validate each requested physical-time coordinate row. Coordinates may
    # move with time; uniqueness is required within a row, not across rows.
    for time in range(lt):
        _selected_coordinates(
            point_data[time, :np_count, :], np_count, (lx, ly, lz),
            array_module, "point_data",
        )

    result = array_module.empty((lt, ne, np_count, 3), dtype=vector_dtype)
    for time in range(lt):
        result[time] = _gather_checked_coordinates(
            eigenvector_data[time, :ne], point_data[time, :np_count, :],
            array_module,
        )
    return result


__all__ = ["get_overlap_matrix", "calc_overlap_row"]
