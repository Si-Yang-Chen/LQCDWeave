"""Explicit time, mode and point selections for read-only array adapters."""

from numbers import Integral

import numpy as np
from collections.abc import Mapping
import mmap


def host_arrays_resident(values):
    """Recognize host buffers conservatively, excluding file-backed views."""
    if isinstance(values, Mapping):
        return bool(values) and all(host_arrays_resident(v) for v in values.values())
    if isinstance(values, (tuple, list)):
        return bool(values) and all(host_arrays_resident(v) for v in values)
    if not isinstance(values, np.ndarray):
        return False
    owner = values
    while isinstance(owner, (np.ndarray, memoryview)):
        if isinstance(owner, np.memmap):
            return False
        owner = owner.base if isinstance(owner, np.ndarray) else owner.obj
    return not isinstance(owner, mmap.mmap)


def times(values, extent, name):
    if isinstance(values, Integral) and not isinstance(values, (bool, np.bool_)):
        values = (values,)
    try:
        selected = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain physical time indices") from exc
    if (not selected or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, Integral) or not 0 <= v < extent for v in selected)):
        raise ValueError(f"{name} must contain physical times in [0, {extent})")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{name} must not repeat a time")
    return tuple(int(v) for v in selected)


def _choices(names, shape, kind, used, exact):
    axes = [name for name in names if kind in name]
    if used is not None and exact is not None:
        raise ValueError(f"choose used{'Ne' if kind == 'mode' else 'Np'} or {kind}_indices")
    if used is not None:
        counts = used if isinstance(used, (tuple, list)) else (used,) * len(axes)
        if not axes or len(counts) != len(axes):
            raise ValueError(f"used{'Ne' if kind == 'mode' else 'Np'} must match the {kind} axes")
        exact = dict(zip(axes, (slice(0, count) for count in counts)))
    else:
        exact = {} if exact is None else exact
    if not isinstance(exact, dict) or set(exact) - set(axes):
        raise ValueError(f"{kind}_indices must name available {kind} axes")
    result = {}
    for name, selection in exact.items():
        extent = shape[names.index(name)]
        if isinstance(selection, slice):
            start, stop, step = selection.start, selection.stop, selection.step
            if (any(v is not None and (isinstance(v, (bool, np.bool_)) or
                    not isinstance(v, Integral)) for v in (start, stop, step)) or
                    (start is not None and start < 0) or
                    (stop is not None and (stop < 0 or stop > extent)) or
                    (step is not None and step < 1)):
                raise ValueError(f"invalid {name} slice")
            if not range(*selection.indices(extent)):
                raise ValueError(f"empty {name} selection")
            result[name] = selection
        else:
            try:
                indices = tuple(selection)
            except TypeError as exc:
                raise ValueError(f"{name} needs a slice or index sequence") from exc
            if (not indices or any(isinstance(i, (bool, np.bool_)) or
                    not isinstance(i, Integral) or not 0 <= i < extent for i in indices) or
                    len(set(indices)) != len(indices)):
                raise ValueError(f"invalid {name} indices")
            result[name] = tuple(int(i) for i in indices)
    return result


def resolve_selections(names, shape, *, usedNe=None, mode_indices=None,
                       usedNp=None, point_indices=None):
    selectors = _choices(names, shape, "mode", usedNe, mode_indices)
    selectors.update(_choices(names, shape, "point", usedNp, point_indices))
    return selectors


def select_axes(block, names, *, usedNe=None, mode_indices=None,
                usedNp=None, point_indices=None):
    """Apply basic slices first, then gather the smallest fractions first."""
    selectors = resolve_selections(names, block.shape, usedNe=usedNe,
                                   mode_indices=mode_indices, usedNp=usedNp,
                                   point_indices=point_indices)
    if not selectors:
        return block
    readonly = not getattr(block.flags, "writeable", True)
    result = block
    for name, selection in selectors.items():
        if isinstance(selection, slice):
            axis = names.index(name)
            result = result[(slice(None),) * axis + (selection,)]
    gathers = [(name, selection) for name, selection in selectors.items()
               if not isinstance(selection, slice)]
    gathers.sort(key=lambda item: len(item[1]) / result.shape[names.index(item[0])])
    for name, selection in gathers:
        result = result.take(selection, axis=names.index(name))
    if readonly and getattr(result.flags, "writeable", True):
        result.setflags(write=False)
    return result


def _consecutive_slice(indices):
    if indices and all(value == indices[0] + offset
                       for offset, value in enumerate(indices)):
        return slice(indices[0], indices[0] + len(indices))
    return None


def _slab_array_module(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy
        return cupy
    return np


def select_slab_axes(source, names, time_indices, *, usedNe=None,
                     mode_indices=None, usedNp=None, point_indices=None):
    """Apply time and exact axis gathers once after all basic prefix slices.

    ``source`` has leading time/batch axis described by ``names[0]``. Contiguous
    time selections and contiguous exact-index selections become basic slices.
    Prefix selections such as ``usedNe``/``usedNp`` are also sliced first; all
    remaining time/mode/point gathers are combined into one Cartesian advanced
    index so their temporary never contains unselected large axes.
    """
    names = tuple(names)
    if (not names or names[0] != "coordinate_batch"
            or len(names) != source.ndim
            or len(names) != len(set(names))):
        raise ValueError("slab axes must uniquely describe a leading coordinate_batch dimension")
    time_indices = tuple(time_indices)
    if not time_indices:
        raise ValueError("time_indices cannot be empty")
    selections = resolve_selections(
        names[1:], tuple(source.shape[1:]), usedNe=usedNe,
        mode_indices=mode_indices, usedNp=usedNp,
        point_indices=point_indices,
    )
    basic = [slice(None)] * source.ndim
    gathered = {}
    time_slice = _consecutive_slice(time_indices)
    if time_slice is None:
        gathered[0] = time_indices
    else:
        basic[0] = time_slice
    for name, selection in selections.items():
        axis = names.index(name)
        if isinstance(selection, slice):
            basic[axis] = selection
        else:
            contiguous = _consecutive_slice(selection)
            if contiguous is None:
                gathered[axis] = tuple(selection)
            else:
                basic[axis] = contiguous

    value = source[tuple(basic)]
    if not gathered:
        return value
    xp = _slab_array_module(value)
    gather_axes = tuple(sorted(gathered))
    other_axes = tuple(axis for axis in range(value.ndim) if axis not in gathered)
    moved = xp.moveaxis(value, gather_axes, tuple(range(len(gather_axes))))
    index_arrays = tuple(xp.asarray(gathered[axis], dtype=xp.int64) for axis in gather_axes)
    grid = xp.ix_(*index_arrays)
    gathered_value = moved[(*grid, *(slice(None) for _ in other_axes))]
    output_order = (*gather_axes, *other_axes)
    restore = tuple(output_order.index(axis) for axis in range(value.ndim))
    return xp.transpose(gathered_value, restore)


def selected_points(names, shape, *, usedNp=None, point_indices=None):
    """Return original point IDs in requested output order for each point axis."""
    selectors = _choices(names, shape, "point", usedNp, point_indices)
    result = {}
    for name in names:
        if "point" not in name:
            continue
        original = tuple(range(shape[names.index(name)]))
        selection = selectors.get(name)
        result[name] = original if selection is None else original[selection] if isinstance(selection, slice) else selection
    return result


