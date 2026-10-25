"""Assemble periodic two-point estimates using explicit physical coordinates."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from lqcdweave.contracts.time_selection import (
    TimeSelection, TimeSelectionRecord, positive_integer, normalize_time_selection,
)


@dataclass(frozen=True)
class TwoPointResult:
    """Source-averaged values and the number of available pairs at each dt."""

    values: np.ndarray
    counts: np.ndarray
    dt_axis: int


def _axis(axis: int, ndim: int, name: str) -> int:
    if isinstance(axis, bool) or not isinstance(axis, Integral) or not -ndim <= axis < ndim:
        raise ValueError(f"{name} must identify an existing data axis")
    return int(axis) % ndim


def _coordinates(values, Lt: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in 'iu' or np.any(array < 0) or np.any(array >= Lt):
        raise ValueError(f"{name} must contain integer coordinates in [0, Lt)")
    return array.astype(np.intp, copy=False)


def periodic_time_difference(source_time, sink_time, *, Lt):
    """Map explicit physical endpoint coordinates to a periodic difference.

    Inputs may be scalars or broadcast-compatible arrays. Coordinate validation
    belongs to the selection/binding frontend; no sorting or slot inference occurs.
    """
    return (sink_time - source_time) % Lt


def pair_time_differences(rows, columns, *, Lt, source_times=None,
                          sink_times=None, time_layout='absolute'):
    """Return a source × other physical-difference table, without value data.

    With explicit ordered endpoint tables, rows/columns are their stored slots.
    cyclic_difference requires identical endpoint tables, as checked here. With
    no tables, absolute coordinates or shared physical relative differences are
    selected by time_layout. Geometry arrays never stack numerical result fields.
    """
    Lt = positive_integer(Lt, 'Lt')
    rows, columns = np.asarray(rows), np.asarray(columns)
    if rows.ndim != 1 or columns.ndim != 1:
        raise ValueError('pair coordinates must be one-dimensional lists')
    if source_times is not None:
        sources = _coordinates(source_times, Lt, 'source_times')
        sinks = sources if sink_times is None else _coordinates(sink_times, Lt, 'sink_times')
        if sources.ndim != 1 or sinks.ndim != 1 or not sources.size or not sinks.size:
            raise ValueError('endpoint tables must be nonempty coordinate lists')
        rows = _coordinates(rows, len(sources), 'source slots')
        columns = _coordinates(columns, len(sinks), 'other slots')
        if time_layout == 'cyclic_difference':
            if not np.array_equal(sources, sinks):
                raise ValueError('cyclic_difference requires identical ordered endpoint selections')
            endpoints = sinks[(rows[:, None] + columns[None, :]) % len(sinks)]
        elif time_layout == 'absolute':
            endpoints = sinks[columns][None, :]
        else:
            raise ValueError('slot coordinates require absolute or cyclic_difference layout')
        return periodic_time_difference(sources[rows][:, None], endpoints, Lt=Lt)
    if sink_times is not None:
        raise ValueError('sink_times requires source_times')
    rows = _coordinates(rows, Lt, 'source coordinates')
    columns = _coordinates(columns, Lt, 'other coordinates')
    if time_layout == 'relative':
        return np.broadcast_to(columns, (len(rows), len(columns)))
    if time_layout != 'absolute':
        raise ValueError('physical coordinates require absolute or relative layout')
    return periodic_time_difference(rows[:, None], columns[None, :], Lt=Lt)


def count_time_pairs(differences, *, Lt, valid=None):
    """Count explicitly available pairs per physical dt; values are not read."""
    Lt = positive_integer(Lt, 'Lt')
    differences = _coordinates(differences, Lt, 'differences')
    if valid is not None:
        valid = np.asarray(valid)
        if valid.dtype.kind != 'b' or valid.shape != differences.shape:
            raise ValueError('valid pair mask must be boolean and match geometry')
    return _count_time_pairs_checked(differences, Lt=Lt, valid=valid)


def _count_time_pairs_checked(differences, *, Lt, valid):
    """Count geometry and mask accepted by the current public entry point."""
    entries = differences.reshape(-1) if valid is None else differences[valid]
    return np.bincount(entries, minlength=Lt)


def mean_time_pairs(data, differences, *, Lt, source_axis, other_axis,
                    valid=None, missing='raise', dtype=None, sum_dtype=None):
    """Reduce a supplied pair geometry, retaining relative spectator-axis order.

    Defaults retain the array reconstruction's float64/complex128 promotion.
    Storage explicitly supplies its original floating/complex output dtype and
    promoted sum_dtype. Missing='nan' is an explicit opt-in, never zero filling.
    """
    Lt = positive_integer(Lt, 'Lt')
    array = np.asarray(data)
    if array.dtype.kind not in 'biufc':
        raise ValueError('data must be a real or complex numeric array')
    source_axis = _axis(source_axis, array.ndim, 'source_axis')
    other_axis = _axis(other_axis, array.ndim, 'other_axis')
    if source_axis == other_axis:
        raise ValueError('source_axis and other_axis must be different')
    if missing not in ('raise', 'nan'):
        raise ValueError("missing must be 'raise' or 'nan'")
    differences = _coordinates(differences, Lt, 'differences')
    if differences.shape != (array.shape[source_axis], array.shape[other_axis]):
        raise ValueError('pair geometry differs from numerical time axes')
    if valid is None:
        valid = np.ones(differences.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != differences.shape:
            raise ValueError('valid pair mask differs from geometry')
    return _mean_time_pairs_checked(
        array, differences, Lt=Lt, source_axis=source_axis, other_axis=other_axis,
        valid=valid, missing=missing, dtype=dtype, sum_dtype=sum_dtype,
    )


def _mean_time_pairs_checked(array, differences, *, Lt, source_axis, other_axis,
                             valid, missing, dtype=None, sum_dtype=None):
    """Reduce already checked axes, pair geometry and selection in this call."""
    counts = _count_time_pairs_checked(differences, Lt=Lt, valid=valid)
    uncovered = np.flatnonzero(counts == 0)
    if uncovered.size and missing == 'raise':
        raise ValueError(f'no source pairs for dt={uncovered.tolist()}')
    arranged = np.moveaxis(array, (source_axis, other_axis), (-2, -1))
    dtype = np.result_type(array.dtype, np.float64) if dtype is None else np.dtype(dtype)
    sum_dtype = dtype if sum_dtype is None else np.dtype(sum_dtype)
    values = np.full(arranged.shape[:-2] + (Lt,), np.nan, dtype=dtype)
    for dt in np.flatnonzero(counts):
        rows, columns = np.nonzero(valid & (differences == dt))
        values[..., dt] = np.sum(arranged[..., rows, columns], axis=-1, dtype=sum_dtype) / counts[dt]
    output_axis = other_axis - int(source_axis < other_axis)
    return TwoPointResult(np.moveaxis(values, -1, output_axis), counts, output_axis)


class TimePairAccumulator:
    """Streaming native-dtype per-field sums and shared measured-pair counts.

    fields maps arbitrary field keys to (shape, dtype). Only Lt-sized output
    buffers remain resident; no source × sink numerical matrix is retained.
    Input contributions are consumed synchronously and are never retained.
    """
    def __init__(self, fields, *, Lt):
        self.Lt = positive_integer(Lt, 'Lt')
        self.counts = np.zeros(self.Lt, dtype=np.int64)
        self.sums = {key: np.zeros((self.Lt, *shape), dtype=dtype)
                     for key, (shape, dtype) in fields.items()}

    def add(self, source_time, sink_time, values, *, reject_equal=False):
        """Consume one pair; values is an iterable of (field key, array)."""
        dt = periodic_time_difference(source_time, sink_time, Lt=self.Lt)
        if reject_equal and dt == 0:
            raise ValueError('packed nonzero dt layout received an equal-time pair')
        self.counts[dt] += 1
        for key, value in values:
            self.sums[key][dt] += value
        return dt

    def mean(self, key, dt, *, inplace=False):
        """Return one native-dtype covered mean; optionally reuse its sum buffer."""
        if not self.counts[dt]:
            raise ValueError(f'no source pairs for dt={dt}')
        value = self.sums[key][dt:dt+1].reshape(self.sums[key].shape[1:])
        if not inplace:
            value = np.array(value, copy=True)
        value /= self.counts[dt]
        return value

    def write_means(self, key, coordinates, out, *, missing='nan'):
        """Write trailing-dt native means into caller-owned output storage."""
        if missing not in ('raise', 'nan'):
            raise ValueError("missing must be 'raise' or 'nan'")
        for column, dt in enumerate(coordinates):
            if self.counts[dt]:
                np.divide(np.expand_dims(self.sums[key][dt], -1), self.counts[dt],
                          out=out[..., column:column+1])
            elif missing == 'nan':
                out[..., column] = np.nan
            else:
                raise ValueError(f'no source pairs for dt={dt}')
        return out


def reconstruct_two_point(
    data,
    *,
    Lt: int,
    selected_times,
    tsrc_axis: int,
    dt_axis: int,
    missing: str = 'raise',
) -> TwoPointResult:
    """Reconstruct sparse-index two-point data, e.g. (40, 40) -> (128,).

    Source index i and index difference j identify sink (i+j) % Nt.
    ``selected_times`` must preserve the exact order used in the calculation.
    Both time axes must have length Nt; all spectator axes are preserved.
    """
    Lt = positive_integer(Lt, 'Lt')
    times = _coordinates(selected_times, Lt, 'selected_times')
    if times.ndim != 1 or not times.size or len(np.unique(times)) != times.size:
        raise ValueError('selected_times must be a nonempty unique coordinate list')
    indices = np.arange(times.size)
    differences = pair_time_differences(indices, indices, Lt=Lt,
        source_times=times, sink_times=times, time_layout='cyclic_difference')
    return _reconstruct_from_physical_coordinates(
        data, Lt=Lt, tsrc_axis=tsrc_axis, dt_axis=dt_axis,
        tsrc_times=times, dt_values=differences, missing=missing,
    )


def reconstruct_two_point_from_coordinates(
    data,
    selection: TimeSelection | TimeSelectionRecord | None = None,
    *,
    tsrc_axis: int = 0,
    dt_axis: int = 1,
    missing: str = 'raise',
    Lt: int | None = None,
    tsrc_times=None,
    dt_values=None,
    sink_times=None,
    time_layout: str = 'absolute',
    sink_selection: TimeSelection | TimeSelectionRecord | None = None,
) -> TwoPointResult:
    """Average selected source/sink slots into physical periodic differences.

    The selection path defaults to absolute source-slot × sink-slot data.
    Legacy source-slot × cyclic-difference-slot data requires the explicit
    ``time_layout='cyclic_difference'`` declaration. A distinct sink_selection
    supports asymmetric absolute coverage. No physical coordinate conversion
    is required in the calling script. The historic ``dt_axis`` parameter names
    the second stored time axis, which becomes physical dt in the output.

    Explicit physical-coordinate calls omit both selections and retain their
    previous semantics. Mixing coordinate domains is rejected.
    """
    if time_layout not in ('absolute', 'cyclic_difference'):
        raise ValueError("time_layout must be 'absolute' or 'cyclic_difference'")
    if selection is not None:
        selection = normalize_time_selection(selection)
        if any(value is not None for value in (Lt, tsrc_times, dt_values, sink_times)):
            raise ValueError('selection cannot be combined with explicit coordinates')
        sink = selection if sink_selection is None else normalize_time_selection(sink_selection, Lt=selection.Lt)
        if time_layout == 'cyclic_difference':
            if sink != selection:
                raise ValueError('cyclic_difference requires identical ordered source and sink selections')
            return reconstruct_two_point(
                data, Lt=selection.Lt, selected_times=selection.times,
                tsrc_axis=tsrc_axis, dt_axis=dt_axis, missing=missing,
            )
        sources, sinks = np.asarray(selection.times), np.asarray(sink.times)
        differences = pair_time_differences(sources, sinks, Lt=selection.Lt)
        return _reconstruct_from_physical_coordinates(
            data, Lt=selection.Lt, tsrc_axis=tsrc_axis, dt_axis=dt_axis,
            tsrc_times=sources, dt_values=differences, missing=missing,
        )
    if sink_selection is not None or time_layout != 'absolute':
        raise ValueError('slot layout or sink_selection requires a source selection')
    if Lt is None or tsrc_times is None or dt_values is None:
        raise TypeError('provide selection or explicit Lt, tsrc_times and dt_values')
    return _reconstruct_from_physical_coordinates(
        data, Lt=Lt, tsrc_axis=tsrc_axis, dt_axis=dt_axis,
        tsrc_times=tsrc_times, dt_values=dt_values, sink_times=sink_times,
        missing=missing,
    )


def _reconstruct_from_physical_coordinates(
    data,
    *,
    Lt: int,
    tsrc_axis: int,
    dt_axis: int,
    tsrc_times,
    dt_values,
    sink_times=None,
    missing: str = 'raise',
) -> TwoPointResult:
    """Average available source pairs into the full physical dt=0..Lt-1 axis.

    ``dt_values`` are PHYSICAL time differences: either a shared (Ndt,) array
    or (Nsrc, Ndt) coordinates in source/dt order, independent of data axes.
    ``tsrc_times`` gives actual source coordinates in stored order. Optional
    ``sink_times`` restricts contributions to pairs ending on those time slices.

    Only the source axis is reduced. Other axes retain their relative order;
    the dt axis expands to Lt. Counts are one-dimensional because all spectator
    entries must share the same time geometry. Missing separations raise by
    default; ``missing='nan'`` keeps them explicitly unavailable.
    """
    Lt = positive_integer(Lt, 'Lt')
    array = np.asarray(data)
    if array.dtype.kind not in 'iufc':
        raise ValueError('data must be a real or complex numeric array')
    source_axis = _axis(tsrc_axis, array.ndim, 'tsrc_axis')
    separation_axis = _axis(dt_axis, array.ndim, 'dt_axis')
    if source_axis == separation_axis:
        raise ValueError('tsrc_axis and dt_axis must be different')
    if missing not in ('raise', 'nan'):
        raise ValueError("missing must be 'raise' or 'nan'")
    sources = _coordinates(tsrc_times, Lt, 'tsrc_times')
    nsrc, ndt = array.shape[source_axis], array.shape[separation_axis]
    if sources.shape != (nsrc,) or not nsrc or len(np.unique(sources)) != nsrc:
        raise ValueError('tsrc_times must contain one unique coordinate per source')
    differences = _coordinates(dt_values, Lt, 'dt_values')
    if differences.shape == (ndt,):
        differences = np.broadcast_to(differences, (nsrc, ndt))
    elif differences.shape != (nsrc, ndt):
        raise ValueError('dt_values must have shape (Ndt,) or (Nsrc, Ndt)')
    if not ndt or any(len(np.unique(row)) != ndt for row in differences):
        raise ValueError('each source must have nonempty, unique dt coordinates')
    valid = np.ones((nsrc, ndt), dtype=bool)
    if sink_times is not None:
        sinks = _coordinates(sink_times, Lt, 'sink_times')
        if sinks.ndim != 1 or not sinks.size or len(np.unique(sinks)) != sinks.size:
            raise ValueError('sink_times must be a nonempty unique coordinate list')
        valid &= np.isin((sources[:, None] + differences) % Lt, sinks)
    return _mean_time_pairs_checked(
        array, differences, Lt=Lt, source_axis=source_axis,
        other_axis=separation_axis, valid=valid, missing=missing,
    )
