"""Sparse P2P current bridge without a dense point-to-point vertex tensor.

The inputs have standardized endpoint layout
``left[..., spin_left, point_left, color_left]`` and
``right[spin_right, point_right, color_right, ...]``.  The source-side and
sink-side adapters own any transposes needed to reach this layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

import numpy as np
from lqcdweave.contracts.correlator.api import (
    AxisPattern,
    bridge_intermediate_labels,
    bridge_point_equalities_covered,
    bridge_point_equivalence,
    cycle_external_labels,
    cycle_requires_generic_join,
    cycle_point_equivalence,
    sparse_cycle_endpoint_axes,
    parse_axis_pattern,
    parse_sparse_equation as _parse_cycle_equation,
)
from lqcdweave.contraction.execution.indices import einsum_arguments

from .sparse_join import (
    _coordinate_batch_equation,
    _coordinate_batch_payload,
    _coordinate_validate_dimensions,
    _contract_sparse_p2p_bridge_join_batch,
    contract_sparse_p2p_bridge_join,
    contract_sparse_p2p_join,
    contract_sparse_p2p_cycle_join_batch,
    coordinate_base_axes,
)


_DOUBLE_ATOMIC_PREAMBLE = """
__device__ double p2p_atomic_add(double* address, double value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    unsigned long long* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long prior = *bits, expected;
    do {
        expected = prior;
        prior = atomicCAS(bits, expected,
            __double_as_longlong(value + __longlong_as_double(expected)));
    } while (prior != expected);
    return __longlong_as_double(prior);
#else
    return atomicAdd(address, value);
#endif
}
"""


def _einsum(backend, pattern, operands, **kwargs):
    return backend.einsum(*einsum_arguments(pattern, operands), **kwargs)


_BRIDGE_RETAINED_PATTERN = AxisPattern(
    ((0, 1, 2, 3), (1, 4), (2, 3, 5), (4, 2, 5, 6)),
    (0, 6, 2),
)
_BRIDGE_CLOSED_PATTERN = AxisPattern(
    ((0, 1, 2, 3), (1, 4), (2, 3, 5), (4, 2, 5, 6)),
    (0, 6),
)
_BATCH_BRIDGE_PATTERN = AxisPattern(
    ((0, 1, 2, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5, 6)),
    (0, 1, 6),
)
_CLOSE_BRIDGE_PATTERN = AxisPattern(
    ((0, 1, 2, 3, 4, 5), (2, 4), (0, 3, 5)),
    (1,),
)


def _require_coordinate_batch_complex128_envelope(dtype, label):
    """Keep batch-only numeric temporaries within E's complex128 byte model."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "biufc":
        raise TypeError(f"coordinate-batched {label} must use a numeric dtype")
    promoted = np.result_type(dtype, np.dtype(np.complex128))
    if promoted.kind != "c" or promoted.itemsize > np.dtype(np.complex128).itemsize:
        raise TypeError(
            f"coordinate-batched {label} exceeds the complex128 result envelope"
        )


@lru_cache(maxsize=1)
def _fused_retained_kernel(kernel_type: Any):
    """Gather, dense spin/color contribution and COO accumulation in one launch.

    Ns=4 is fixed, but Nc is supplied at runtime. Input types are independent
    CuPy kernel parameters; loads promote to complex<double> before arithmetic,
    preserving the existing complex128 output/atomic accumulation. CArray
    indices handle strides; each original COO row is an independent summand.
    No gathered endpoints, converted input arrays or contribution tensor exist.
    """
    return kernel_type(
        "raw L left, raw R right, raw G gamma, "
        "raw B blocks, raw int64 left_points, raw int64 right_points, "
        "int64 row_count, int64 outer_b, int64 points_left, int64 points_right, "
        "int64 point_size, int64 point_stride, int64 nc, complex128 coefficient",
        "raw complex128 result",
        """
        const long long k = i % row_count;
        const long long outer = i / row_count;
        const long long a = outer / outer_b;
        const long long b = outer % outer_b;
        const long long p = left_points[k], q = right_points[k];
        complex<double> total(0.0, 0.0);
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            for (long long d = 0; d < nc; ++d) {
                complex<double> spin_right(0.0, 0.0);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const long long gamma_index[2] = {s, t};
                    const long long right_index[4] = {t, q, d, b};
                    spin_right += complex<double>(gamma[gamma_index])
                        * complex<double>(right[right_index]);
                }
                complex<double> color_left(0.0, 0.0);
                for (long long c = 0; c < nc; ++c) {
                    // Direct coordinates avoid decomposing a flattened
                    // index by runtime extents, while retaining all strides.
                    const long long left_index[4] = {a, s, p, c};
                    const long long block_index[3] = {k, c, d};
                    color_left += complex<double>(left[left_index])
                        * complex<double>(blocks[block_index]);
                }
                total += color_left * spin_right;
            }
        }
        const long long offset = outer * point_size + p * point_stride;
        const complex<double> value = coefficient * total;
        double* components = reinterpret_cast<double*>(&result[offset]);
        p2p_atomic_add(&components[0], value.real());
        p2p_atomic_add(&components[1], value.imag());
        """,
        "lqcdweave_fused_retained_p2p",
        preamble=_DOUBLE_ATOMIC_PREAMBLE,
    )


@lru_cache(maxsize=1)
def _retained_accumulator(kernel_type: Any):
    """One GPU launch for COO accumulation, using existing input/output storage.

    Each row contributes separately, including duplicate rows.  The result is
    complex128, so its real and imaginary components use double atomic adds.
    CuPy handles the contribution strides and broadcasts the existing point
    indices; no contiguous copy or auxiliary device array is required.
    Cache the constructor rather than a transient execution backend wrapper;
    CuPy's kernel object maintains its own separate cache for each device.
    """
    return kernel_type(
        "T contribution, int64 point, int64 row_count, "
        "int64 point_size, int64 point_stride",
        "raw complex128 result",
        """
        const long long offset = (i / row_count) * point_size + point * point_stride;
        double* components = reinterpret_cast<double*>(&result[offset]);
        const complex<double> value(contribution);
        p2p_atomic_add(&components[0], value.real());
        p2p_atomic_add(&components[1], value.imag());
        """,
        "lqcdweave_retained_p2p_add",
        preamble=_DOUBLE_ATOMIC_PREAMBLE,
    )


def _point_extents(used_np: int, used_np_right: int | None = None) -> tuple[int, int]:
    """Normalize legacy square and new rectangular COO extent arguments."""
    right = used_np if used_np_right is None else used_np_right
    result = []
    for name, value in (("used_np", used_np), ("used_np_right", right)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a positive integer")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
        result.append(value)
    return tuple(result)


def _legacy_pairs(record: Mapping[str, Any], used_np: int, nc: int,
                  used_np_right: int | None = None):
    used_np, used_np_right = _point_extents(used_np, used_np_right)
    kind = record.get("type")
    if kind == "identity":
        if used_np != used_np_right:
            raise ValueError("identity P2P records require equal point extents; use explicit COO for rectangles")
        points = np.arange(used_np, dtype=np.int64)
        return np.column_stack((points, points)), np.broadcast_to(np.eye(nc, dtype=np.complex128), (used_np, nc, nc))
    if kind != "sparse":
        raise ValueError("legacy P2P record must be identity or sparse")
    indices = np.asarray(record["indices"])
    values = np.asarray(record["values"])
    if indices.dtype.kind not in "iu":
        raise TypeError("legacy P2P COO indices must be integer arrays")
    if values.dtype.kind not in "biufc":
        raise TypeError("legacy P2P COO values must be numeric arrays")
    indices = indices.astype(np.int64, copy=False)
    if indices.ndim != 2 or indices.shape[1] != 2 or values.shape != (len(indices), nc, nc):
        raise ValueError("legacy P2P COO shape is invalid")
    # Match the existing dense indexed-addition semantics exactly.
    valid = (indices[:, 0] < used_np) & (indices[:, 1] < used_np_right)
    indices = indices[valid].copy()
    values = values[valid]
    indices[indices[:, 0] < 0, 0] += used_np
    indices[indices[:, 1] < 0, 1] += used_np_right
    if np.any(indices[:, 0] < 0) or np.any(indices[:, 1] < 0):
        raise IndexError("P2P sparse point index is outside the usedNp block")
    return indices, values


def _matched_pairs(record: Mapping[str, Any], used_np: int, nc: int,
                   used_np_right: int | None = None):
    used_np, used_np_right = _point_extents(used_np, used_np_right)
    indices = np.asarray(record["indices"])
    values = np.asarray(record["values"])
    if indices.dtype.kind not in "iu":
        raise TypeError("matched P2P indices must be integer arrays")
    if values.dtype.kind not in "biufc":
        raise TypeError("matched P2P values must be numeric arrays")
    indices = indices.astype(np.int64, copy=False)
    if indices.shape != (used_np,) or values.shape != (used_np, nc, nc):
        raise ValueError("matched P2P record shape is invalid")
    if np.any(indices < -1) or np.any(indices >= used_np_right):
        raise IndexError("matched P2P right-point index is outside the usedNp block")
    left = np.flatnonzero(indices >= 0)
    return np.column_stack((left, indices[left])), values[left]


def sparse_pairs(record: Mapping[str, Any], used_np: int, nc: int = 3,
                 *, used_np_right: int | None = None):
    """Normalize either on-disk sparse contract without dense materialization."""
    used_np, used_np_right = _point_extents(used_np, used_np_right)
    if not isinstance(record, Mapping):
        raise TypeError("sparse bridge record must be a mapping")
    if record.get("layout") == "matched":
        return _matched_pairs(record, used_np, nc, used_np_right)
    return _legacy_pairs(record, used_np, nc, used_np_right)


def contract_p2p_bridge(
    left: Any,
    right: Any,
    parts: Iterable[tuple[Any, complex, Mapping[str, Any]]],
    *,
    used_np: int,
    used_np_right: int | None = None,
    backend: Any = np,
    left_axes: tuple[int, int, int] = (-3, -2, -1),
    right_axes: tuple[int, int, int] = (0, 1, 2),
    same_point: bool = False,
    transpose_bridge: bool = False,
    equation: AxisPattern | str | None = None,
):
    """Sum sparse P2P records between two propagator factors.

    Each part supplies a spin matrix, scalar coefficient, and a legacy or
    matched sparse record.  Duplicate COO pairs contribute independently.
    The equation may retain either point coordinate independently; this does
    not add an equality between unrelated endpoint points.
    """
    record_left_count, record_right_count = _point_extents(used_np, used_np_right)
    if equation is not None:
        equation = parse_axis_pattern(equation)
        left_labels, right_labels, equation_output = _parse_cycle_equation(equation)
        equation = AxisPattern((left_labels, right_labels), equation_output)
    if equation is not None and not bridge_point_equalities_covered(
            equation, left_axes, right_axes, same_point=same_point):
        point_equivalence = bridge_point_equivalence(
            equation, left_axes, right_axes, same_point=same_point,
        )
        return contract_sparse_p2p_bridge_join(
            left, right, parts, used_np=record_left_count,
            used_np_right=record_right_count,
            left_axes=left_axes, right_axes=right_axes,
            same_point=same_point, transpose_bridge=transpose_bridge,
            equation=equation, point_equivalence=point_equivalence,
            output_labels=equation.output, backend=backend,
        )
    if left.ndim < 3 or right.ndim < 3:
        raise ValueError("bridge operands need spin, point and color endpoint axes")
    left = backend.moveaxis(left, left_axes, (-3, -2, -1))
    right = backend.moveaxis(right, right_axes, (0, 1, 2))
    nc = int(left.shape[-1])
    bridge_left_count, bridge_right_count = (
        (record_right_count, record_left_count)
        if transpose_bridge else (record_left_count, record_right_count)
    )
    if (left.shape[-2] != bridge_left_count
            or right.shape[1:3] != (bridge_right_count, nc)
            or (same_point and record_left_count != record_right_count)):
        raise ValueError("bridge point/color endpoint dimensions do not match")
    spin_left = int(left.shape[-3])
    spin_right = int(right.shape[0])
    left_external = tuple(left.shape[:-3])
    right_external = tuple(right.shape[3:])
    a = left.reshape((-1, spin_left, bridge_left_count, nc))
    b = right.reshape((spin_right, bridge_right_count, nc, -1))
    retained = ()
    raw_labels = None
    output_labels = None
    left_point_label = right_point_label = None
    if equation is not None:
        _, raw_labels, output_labels = bridge_intermediate_labels(
            equation, left_axes, right_axes, same_point=same_point
        )
        left_labels, right_labels, _ = _parse_cycle_equation(equation)
        left_point_label = left_labels[left_axes[1] % len(left_labels)]
        right_point_label = right_labels[right_axes[1] % len(right_labels)]
        shared_point = same_point or left_point_label == right_point_label
        retained = tuple(
            dict.fromkeys(
                label
                for label in (left_point_label, right_point_label)
                if label in output_labels
            )
        )
    point_extent_by_label = {}
    if left_point_label is not None:
        point_extent_by_label[left_point_label] = bridge_left_count
        prior = point_extent_by_label.setdefault(right_point_label, bridge_right_count)
        if prior != bridge_right_count:
            raise ValueError("one sparse bridge point label has inconsistent endpoint extents")
    retained_shape = tuple(point_extent_by_label[label] for label in retained)
    result_shape = (a.shape[0], b.shape[-1]) + retained_shape
    result = backend.zeros(result_shape, dtype=backend.result_type(left, right, np.complex128))
    for gamma, coefficient, record in parts:
        gamma = backend.asarray(gamma)
        if transpose_bridge:
            gamma = gamma.T
        if gamma.shape != (spin_left, spin_right):
            raise ValueError("P2P spin matrix shape does not match propagator endpoints")
        pairs, values = sparse_pairs(
            record, record_left_count, nc, used_np_right=record_right_count,
        )
        if same_point or (left_point_label is not None and left_point_label == right_point_label):
            diagonal = pairs[:, 0] == pairs[:, 1]
            pairs, values = pairs[diagonal], values[diagonal]
        if transpose_bridge:
            pairs = pairs[:, ::-1]
            values = values.transpose(0, 2, 1)
        if not len(pairs):
            continue
        left_points = backend.asarray(pairs[:, 0])
        right_points = backend.asarray(pairs[:, 1])
        blocks = backend.asarray(values)
        gpu_retained = (retained and shared_point
                        and getattr(backend, "__name__", None) == "cupy"
                        and not backend.cuda.runtime.is_hip)
        if gpu_retained:
            point_size = int(np.prod(retained_shape))
            point_stride = sum(
                int(np.prod(retained_shape[axis + 1:]))
                for axis in range(len(retained_shape))
            )
            if (spin_left == spin_right == 4 and nc > 0
                    and all(array.dtype in (np.dtype(np.float32), np.dtype(np.float64),
                                            np.dtype(np.complex64), np.dtype(np.complex128))
                            for array in (a, b, gamma, blocks))):
                _fused_retained_kernel(backend.ElementwiseKernel)(
                    a, b, gamma, blocks, left_points, right_points, len(pairs),
                    b.shape[-1], bridge_left_count, bridge_right_count,
                    point_size, point_stride, nc, coefficient, result,
                    size=a.shape[0] * b.shape[-1] * len(pairs),
                )
                continue
        left_values = backend.take(a, left_points, axis=2)
        right_values = backend.take(b, right_points, axis=1)
        if retained:
            contributions = coefficient * _einsum(
                backend, _BRIDGE_RETAINED_PATTERN,
                (left_values, gamma, blocks, right_values), optimize=True,
            )
            if gpu_retained:
                # Retained coordinates share one point equality.  If both
                # labels are exported, only the diagonal receives values.
                _retained_accumulator(backend.ElementwiseKernel)(
                    contributions, left_points, len(pairs), point_size,
                    point_stride, result,
                )
            else:
                for pair_index, (left_point, right_point) in enumerate(pairs):
                    coordinates = tuple(
                        int(left_point) if label == left_point_label else int(right_point)
                        for label in retained
                    )
                    result[(slice(None), slice(None), *coordinates)] += contributions[:, :, pair_index]
        else:
            result += coefficient * _einsum(
                backend, _BRIDGE_CLOSED_PATTERN,
                (left_values, gamma, blocks, right_values), optimize=True,
            )
    result = result.reshape(left_external + right_external + retained_shape)
    if equation is not None:
        if raw_labels != output_labels:
            projection = AxisPattern((raw_labels,), output_labels)
            result = _einsum(backend, projection, (result,), optimize=False)
    return result


def contract_p2p_bridge_batch(
    left: Any,
    right: Any,
    batch_record: Mapping[str, Any],
    *,
    bridge: Any,
    equation: AxisPattern | str,
    backend: Any = np,
):
    """Execute one lane-aware COO bridge action with bounded row chunks.

    The leading axis is a real coordinate-batch axis on both propagators. Each
    sparse row carries its lane through the gather, so left and right endpoint
    values can only meet within that lane. COO rows are never expanded into a
    dense point-pair tensor.
    """
    if getattr(backend, "__name__", None) != "numpy":
        raise NotImplementedError("coordinate-batched sparse P2P currently supports NumPy only")
    if not isinstance(batch_record, Mapping) or batch_record.get("layout") != "coordinate_batch_coo":
        raise TypeError("coordinate-batched bridge needs the declared COO batch payload")
    point_keys = tuple(batch_record.get("point_keys", ()))
    lanes = tuple(batch_record.get("lanes", ()))
    if not point_keys or len(point_keys) != len(lanes):
        raise ValueError("coordinate-batched bridge payload has inconsistent lane keys and records")
    if tuple(left.shape) == () or tuple(right.shape) == () or left.shape[0] != len(lanes) or right.shape[0] != len(lanes):
        raise ValueError("coordinate-batched propagators need the declared leading lane extent")
    endpoint_dtype = np.result_type(
        np.dtype(left.dtype), np.dtype(right.dtype), np.dtype(np.complex128),
    )
    if endpoint_dtype.kind != "c" or endpoint_dtype.itemsize > np.dtype(np.complex128).itemsize:
        raise TypeError("coordinate-batched endpoints exceed the complex128 result envelope")
    (pattern, lane_label, base_left_labels, base_right_labels,
     base_output_labels, _) = _coordinate_batch_equation(equation)
    _coordinate_validate_dimensions(pattern.inputs, (left, right))

    def unbatch_axis(axis):
        return axis if axis < 0 else axis - 1

    base_left_axes = tuple(unbatch_axis(axis) for axis in bridge.left_axes)
    base_right_axes = tuple(unbatch_axis(axis) for axis in bridge.right_axes)
    if (any(axis % left.ndim == 0 for axis in bridge.left_axes)
            or any(axis % right.ndim == 0 for axis in bridge.right_axes)):
        raise ValueError("coordinate-batched sparse bridge axes cannot select the lane axis")
    base_pattern = AxisPattern((base_left_labels, base_right_labels), base_output_labels)
    point_left_label = base_left_labels[base_left_axes[1] % len(base_left_labels)]
    point_right_label = base_right_labels[base_right_axes[1] % len(base_right_labels)]
    retain_point_label = (
        point_left_label in base_output_labels or point_right_label in base_output_labels
    )
    generic_point_equalities = not bridge_point_equalities_covered(
        base_pattern, base_left_axes, base_right_axes,
        same_point=bridge.same_point,
    )
    raw_base_labels = None
    if not retain_point_label and not generic_point_equalities:
        raw_base_labels, _, _ = bridge_intermediate_labels(
            base_pattern, base_left_axes, base_right_axes,
            same_point=bridge.same_point,
        )

    moved_left = backend.moveaxis(left, bridge.left_axes, (-3, -2, -1))
    moved_right = backend.moveaxis(right, bridge.right_axes, (1, 2, 3))
    lane_count = len(lanes)
    if moved_left.shape[0] != lane_count or moved_right.shape[0] != lane_count:
        raise ValueError("sparse endpoint movement lost the coordinate-batch axis")
    left_external_shape = tuple(moved_left.shape[1:-3])
    right_external_shape = tuple(moved_right.shape[4:])
    spin_left, point_left, color_left = tuple(moved_left.shape[-3:])
    spin_right, point_right, color_right = tuple(moved_right.shape[1:4])
    record_left, record_right = _point_extents(bridge.used_np, bridge.used_np_right)
    bridge_left, bridge_right = (
        (record_right, record_left) if bridge.transpose_bridge
        else (record_left, record_right)
    )
    if (point_left != bridge_left or (point_right, color_right) != (bridge_right, color_left)
            or color_left != color_right
            or (bridge.same_point and record_left != record_right)):
        raise ValueError("coordinate-batched sparse endpoints disagree with bridge extents")
    if (raw_base_labels is not None
            and len(raw_base_labels) != len(left_external_shape) + len(right_external_shape)):
        raise ValueError("coordinate-batched bridge external labels do not match endpoint shapes")
    checked_parts = []
    for parts in lanes:
        if not isinstance(parts, (tuple, list)):
            raise TypeError("each coordinate lane must contain an ordered part tuple")
        checked_lane = []
        for part in parts:
            if not isinstance(part, (tuple, list)) or len(part) != 3:
                raise ValueError("coordinate P2P parts must be (gamma, coefficient, record) triples")
            gamma, coefficient, record = part
            gamma = np.asarray(gamma)
            raw_gamma_shape = ((spin_right, spin_left) if bridge.transpose_bridge
                               else (spin_left, spin_right))
            if gamma.shape != raw_gamma_shape or gamma.dtype.kind not in "biufc":
                raise ValueError("coordinate P2P gamma does not match endpoint spin extents")
            _require_coordinate_batch_complex128_envelope(gamma.dtype, "gamma")
            if not np.all(np.isfinite(gamma)):
                raise ValueError("coordinate P2P gamma must be finite")
            coefficient_array = np.asarray(coefficient)
            if coefficient_array.shape != () or coefficient_array.dtype.kind not in "biufc":
                raise ValueError("coordinate P2P coefficient must be a numeric scalar")
            _require_coordinate_batch_complex128_envelope(coefficient_array.dtype, "coefficient")
            if not bool(np.isfinite(coefficient_array)):
                raise ValueError("coordinate P2P coefficient must be finite")
            if not isinstance(record, Mapping) or record.get("layout") != "normalized-coo":
                raise TypeError("coordinate P2P record must be normalized COO")
            indices = np.asarray(record["indices"])
            values = np.asarray(record["values"])
            if (indices.dtype != np.dtype("int64") or indices.ndim != 2
                    or indices.shape[1] != 2
                    or values.shape != (len(indices), color_left, color_right)
                    or values.dtype.kind not in "biufc"):
                raise ValueError("normalized coordinate P2P COO arrays have invalid shape or dtype")
            _require_coordinate_batch_complex128_envelope(values.dtype, "COO values")
            if (len(indices) and (np.any(indices[:, 0] < 0)
                                  or np.any(indices[:, 0] >= record_left)
                                  or np.any(indices[:, 1] < 0)
                                  or np.any(indices[:, 1] >= record_right))):
                raise IndexError("normalized coordinate P2P index is outside endpoint extent")
            checked_lane.append((gamma, coefficient_array.item(), indices, values))
        checked_parts.append(tuple(checked_lane))

    if retain_point_label or generic_point_equalities:
        return _contract_sparse_p2p_bridge_join_batch(
            left, right, None, bridge=bridge, equation=pattern,
            backend=backend, checked_payload=(point_keys, tuple(checked_parts)),
        )

    # All lane records are checked before allocating the output.  This keeps a
    # malformed tail lane from triggering reshape copies or a result buffer.
    left = moved_left.reshape((lane_count, -1, spin_left, point_left, color_left))
    right = moved_right.reshape((lane_count, spin_right, point_right, color_right, -1))
    left_external = left.shape[1]
    right_external = right.shape[-1]
    result_dtype = backend.result_type(left, right, np.complex128)
    result = backend.zeros(
        (lane_count, left_external, right_external),
        dtype=result_dtype,
    )
    chunk_limit = 64
    row_lanes, row_left, row_right = [], [], []
    row_gamma, row_coefficients, row_values = [], [], []

    def flush_rows():
        if not row_lanes:
            return
        lane_indices = np.asarray(row_lanes, dtype=np.int64)
        left_indices = np.asarray(row_left, dtype=np.int64)
        right_indices = np.asarray(row_right, dtype=np.int64)
        gamma_values = np.stack(row_gamma)
        coefficients = np.asarray(row_coefficients)
        block_values = np.stack(row_values)
        left_values = left[lane_indices, :, :, left_indices, :]
        right_values = right[lane_indices, :, right_indices, :, :]
        contributions = _einsum(
            backend, _BATCH_BRIDGE_PATTERN,
            (left_values, gamma_values, block_values, right_values),
            optimize=False, dtype=result_dtype,
        )
        contributions *= coefficients[:, None, None]
        for index, lane_index in enumerate(row_lanes):
            result[lane_index] += contributions[index]
        row_lanes.clear()
        row_left.clear()
        row_right.clear()
        row_gamma.clear()
        row_coefficients.clear()
        row_values.clear()

    for lane_index, parts in enumerate(checked_parts):
        for gamma, coefficient, indices, values in parts:
            for offset in range(0, len(indices), chunk_limit):
                pair_chunk = indices[offset:offset + chunk_limit]
                value_chunk = values[offset:offset + chunk_limit]
                if bridge.same_point or point_left_label == point_right_label:
                    diagonal = pair_chunk[:, 0] == pair_chunk[:, 1]
                    pair_chunk = pair_chunk[diagonal]
                    value_chunk = value_chunk[diagonal]
                if bridge.transpose_bridge:
                    pair_chunk = pair_chunk[:, ::-1]
                    value_chunk = value_chunk.transpose(0, 2, 1)
                for pair, block in zip(pair_chunk, value_chunk, strict=True):
                    row_lanes.append(lane_index)
                    row_left.append(int(pair[0]))
                    row_right.append(int(pair[1]))
                    row_gamma.append(gamma.T if bridge.transpose_bridge else gamma)
                    row_coefficients.append(coefficient)
                    row_values.append(block)
                    if len(row_lanes) == chunk_limit:
                        flush_rows()
    flush_rows()
    raw_shape = (lane_count, *left_external_shape, *right_external_shape)
    result = result.reshape(raw_shape)
    raw_labels = (lane_label, *raw_base_labels)
    if raw_labels != pattern.output:
        projection = AxisPattern((raw_labels,), pattern.output)
        result = _einsum(backend, projection, (result,), optimize=False)
    return result


def contract_p2p_cycle_batch(
    left: Any,
    right: Any,
    first_batch_record: Mapping[str, Any],
    second_batch_record: Mapping[str, Any],
    *,
    first_bridge: Any,
    second_bridge: Any,
    equation: AxisPattern | str,
    backend: Any = np,
):
    """Contract two lane-aligned sparse bridges with bounded packed COO rows.

    Both complete records are validated before the result is allocated. The
    NumPy path joins the two sparse COO row streams within each lane and packs
    up to 64 matching pairs into each numeric contraction; it never forms a
    lane-by-lane Cartesian product or a dense rank-six vertex.
    """
    if getattr(backend, "__name__", None) != "numpy":
        raise NotImplementedError("coordinate-batched sparse P2P currently supports NumPy only")
    if (not isinstance(first_batch_record, Mapping)
            or not isinstance(second_batch_record, Mapping)
            or first_batch_record.get("layout") != "coordinate_batch_coo"
            or second_batch_record.get("layout") != "coordinate_batch_coo"):
        raise TypeError("coordinate-batched cycle needs two declared COO batch payloads")
    point_keys = tuple(first_batch_record.get("point_keys", ()))
    second_keys = tuple(second_batch_record.get("point_keys", ()))
    first_lanes = tuple(first_batch_record.get("lanes", ()))
    second_lanes = tuple(second_batch_record.get("lanes", ()))
    if (not point_keys or point_keys != second_keys
            or len(point_keys) != len(first_lanes)
            or len(point_keys) != len(second_lanes)):
        raise ValueError("coordinate cycle records must have the same nonempty ordered lane keys")
    if (left.ndim == 0 or right.ndim == 0
            or left.shape[0] != len(point_keys) or right.shape[0] != len(point_keys)):
        raise ValueError("coordinate-batched cycle endpoints need the declared leading lane extent")
    endpoint_dtype = np.result_type(left.dtype, right.dtype, np.complex128)
    if endpoint_dtype.kind != "c" or endpoint_dtype.itemsize > np.dtype(np.complex128).itemsize:
        raise TypeError("coordinate-batched endpoints exceed the complex128 result envelope")
    pattern, lane_label, left_labels, right_labels, _, _ = _coordinate_batch_equation(equation)
    if (left.ndim != len(left_labels) + 1 or right.ndim != len(right_labels) + 1):
        raise ValueError("coordinate cycle equation ranks do not match endpoint arrays")
    _coordinate_validate_dimensions(pattern.inputs, (left, right))
    cycle_external_labels(pattern, first_bridge, second_bridge)
    raw_endpoint_axes = sparse_cycle_endpoint_axes(first_bridge, second_bridge)
    endpoint_axes = {
        name: (slot, coordinate_base_axes(axes, (left, right)[slot].ndim))
        for name, (slot, axes) in raw_endpoint_axes.items()
    }

    def validate_bridge_payload(batch_record, bridge, left_name, right_name):
        left_slot, left_axes = endpoint_axes[left_name]
        right_slot, right_axes = endpoint_axes[right_name]
        left_operand, right_operand = (left, right)[left_slot], (left, right)[right_slot]
        # Batch axis has already been removed from the axis tuple; add one to
        # address each full lane-prefixed operand without copying it.
        left_spin = left_operand.shape[left_axes[0] + 1]
        right_spin = right_operand.shape[right_axes[0] + 1]
        left_color = left_operand.shape[left_axes[2] + 1]
        right_color = right_operand.shape[right_axes[2] + 1]
        return _coordinate_batch_payload(
            batch_record, bridge, spin_left=left_spin, spin_right=right_spin,
            color_left=left_color, color_right=right_color,
        )

    validated_keys1, checked_first = validate_bridge_payload(
        first_batch_record, first_bridge, "first_left", "first_right",
    )
    validated_keys2, checked_second = validate_bridge_payload(
        second_batch_record, second_bridge, "second_left", "second_right",
    )
    if validated_keys1 != point_keys or validated_keys2 != point_keys:
        raise ValueError("coordinate cycle lane records do not match their key order")
    return contract_sparse_p2p_cycle_join_batch(
        left, right, checked_first, checked_second,
        first_bridge=first_bridge, second_bridge=second_bridge,
        equation=pattern, point_keys=point_keys, backend=backend,
    )


def _close_p2p_bridge(
    tensor: Any,
    parts: Iterable[tuple[Any, complex, Mapping[str, Any]]],
    *,
    used_np: int,
    used_np_right: int | None = None,
    left_axes: tuple[int, int, int],
    right_axes: tuple[int, int, int],
    same_point: bool,
    transpose_bridge: bool,
    backend: Any,
):
    """Apply a second sparse current to two endpoint triples of one tensor."""
    axes = tuple(axis % tensor.ndim for axis in (*left_axes, *right_axes))
    if len(set(axes)) != 6:
        raise ValueError("the second P2P bridge needs six distinct resident axes")
    moved = backend.moveaxis(tensor, axes, tuple(range(tensor.ndim - 6, tensor.ndim)))
    outer = tuple(moved.shape[:-6])
    spin_left, points_left, nc_left, spin_right, points_right, nc_right = moved.shape[-6:]
    record_left_count, record_right_count = _point_extents(used_np, used_np_right)
    bridge_left_count, bridge_right_count = (
        (record_right_count, record_left_count)
        if transpose_bridge else (record_left_count, record_right_count)
    )
    if (points_left != bridge_left_count or points_right != bridge_right_count
            or nc_left != nc_right
            or (same_point and record_left_count != record_right_count)):
        raise ValueError("the second P2P endpoint dimensions do not match")
    flat = moved.reshape((-1, spin_left, bridge_left_count, nc_left,
                          spin_right, bridge_right_count, nc_right))
    result = backend.zeros((flat.shape[0],), dtype=backend.result_type(tensor, np.complex128))
    for gamma, coefficient, record in parts:
        gamma = backend.asarray(gamma)
        if transpose_bridge:
            gamma = gamma.T
        if gamma.shape != (spin_left, spin_right):
            raise ValueError("the second P2P spin matrix shape does not match")
        pairs, values = sparse_pairs(
            record, record_left_count, nc_left,
            used_np_right=record_right_count,
        )
        if same_point:
            diagonal = pairs[:, 0] == pairs[:, 1]
            pairs, values = pairs[diagonal], values[diagonal]
        if transpose_bridge:
            pairs = pairs[:, ::-1]
            values = values.transpose(0, 2, 1)
        if not len(pairs):
            continue
        left_points = backend.asarray(pairs[:, 0])
        right_points = backend.asarray(pairs[:, 1])
        selected = flat[:, :, left_points, :, :, right_points, :]
        result += coefficient * _einsum(
            backend, _CLOSE_BRIDGE_PATTERN,
            (selected, gamma, backend.asarray(values)), optimize=True,
        )
    return result.reshape(outer)


def contract_p2p_cycle(
    left: Any,
    right: Any,
    first_parts: Iterable[tuple[Any, complex, Mapping[str, Any]]],
    second_parts: Iterable[tuple[Any, complex, Mapping[str, Any]]],
    *,
    first_bridge: Any,
    second_bridge: Any,
    equation: AxisPattern | str,
    backend: Any = np,
):
    """Contract two sparse currents joining the same two propagators.

    The first current leaves the opposite endpoint triples open. The second
    current closes them without constructing either dense rank-six vertex.
    """
    equation = parse_axis_pattern(equation)
    left_labels, right_labels, output_labels = _parse_cycle_equation(equation)
    equation = AxisPattern((left_labels, right_labels), output_labels)
    if cycle_requires_generic_join(equation, first_bridge, second_bridge):
        point_equivalence = cycle_point_equivalence(
            equation, first_bridge, second_bridge,
        )
        return contract_sparse_p2p_join(
            left, right, first_parts, second_parts,
            first_bridge=first_bridge, second_bridge=second_bridge,
            equation=equation, point_equivalence=point_equivalence,
            output_labels=output_labels, backend=backend,
        )
    _, external, output = cycle_external_labels(equation, first_bridge, second_bridge)
    # Sparse endpoint contraction does not execute the input einsum labels.
    # Enforce a repeated label's diagonal before eliminating either endpoint;
    # otherwise e.g. abedef silently sums independent point axes 2 and 4.
    # Keep the original rank so both endpoint triples remain addressable.
    masked = []
    input_labels = equation.inputs
    for array, labels in zip((left, right), input_labels, strict=True):
        for label in dict.fromkeys(labels):
            axes = [i for i, current in enumerate(labels) if current == label]
            for axis in axes[1:]:
                a, b = [1]*array.ndim, [1]*array.ndim
                a[axes[0]], b[axis] = array.shape[axes[0]], array.shape[axis]
                diagonal = (backend.arange(array.shape[axes[0]]).reshape(a)
                            == backend.arange(array.shape[axis]).reshape(b))
                array = array * diagonal
        masked.append(array)
    left, right = masked
    first_arrays = (left, right)
    raw = contract_p2p_bridge(
        left,
        right,
        first_parts,
        used_np=first_bridge.used_np,
        used_np_right=first_bridge.used_np_right,
        backend=backend,
        left_axes=first_bridge.left_axes,
        right_axes=first_bridge.right_axes,
        same_point=first_bridge.same_point,
        transpose_bridge=first_bridge.transpose_bridge,
    )
    positions = {}
    for slot, axes in enumerate((first_bridge.left_axes, first_bridge.right_axes)):
        rank = first_arrays[slot].ndim
        consumed = {axis % rank for axis in axes}
        for axis in range(rank):
            if axis not in consumed:
                positions[slot, axis] = len(positions)

    def remaining_axes(slot, axes):
        rank = first_arrays[slot].ndim
        try:
            return tuple(positions[slot, axis % rank] for axis in axes)
        except KeyError as exc:
            raise ValueError("two P2P currents cannot consume the same endpoint axis") from exc

    if (second_bridge.left, second_bridge.right) == (first_bridge.left, first_bridge.right):
        second_slots = (0, 1)
    elif first_bridge.left != first_bridge.right and (
        second_bridge.left, second_bridge.right
    ) == (first_bridge.right, first_bridge.left):
        second_slots = (1, 0)
    else:
        # When both factors share one identity, the bridge's left/right axes
        # still name two distinct operand slots in the equation.
        second_slots = (0, 1)

    closed = _close_p2p_bridge(
        raw,
        second_parts,
        used_np=second_bridge.used_np,
        used_np_right=second_bridge.used_np_right,
        left_axes=remaining_axes(second_slots[0], second_bridge.left_axes),
        right_axes=remaining_axes(second_slots[1], second_bridge.right_axes),
        same_point=second_bridge.same_point,
        transpose_bridge=second_bridge.transpose_bridge,
        backend=backend,
    )
    if external != output:
        projection = AxisPattern((external,), output)
        closed = _einsum(backend, projection, (closed,), optimize=False)
    return closed


__all__ = [
    "bridge_intermediate_labels",
    "cycle_external_labels",
    "contract_p2p_bridge",
    "contract_p2p_cycle",
    "sparse_pairs",
]
