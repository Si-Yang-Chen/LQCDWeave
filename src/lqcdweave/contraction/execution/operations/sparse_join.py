"""Memory-bounded COO join for point equalities crossing two P2P currents.

The legacy two-current cycle kernel handles equalities that are local to one
propagator or to one current record. This operation handles the remaining
explicit point-equivalence classes by streaming one bounded chunk of the
second COO record and joining it to rows from the first record. It never builds
an ``nnz_first × nnz_second`` array or a dense P2P vertex.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from numbers import Integral

import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern,
    SPARSE_ENDPOINT_NAMES as _ENDPOINTS,
    DEFAULT_JOIN_INDEX_BLOCK_SIZE,
    ESTIMATED_JOIN_INDEX_BYTES_PER_ROW,
    JOIN_COST_STATUS,
    bridge_point_equivalence,
    cycle_point_equivalence,
    parse_sparse_equation as _parse_cycle_equation,
    parse_axis_pattern,
    axis_ids,
    sparse_cycle_endpoint_axes as _endpoint_axes,
)

from ..indices import einsum_arguments




def _record_extents(used_np, used_np_right=None):
    right = used_np if used_np_right is None else used_np_right
    result = []
    for name, value in (("used_np", used_np), ("used_np_right", right)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        result.append(int(value))
    return tuple(result)


def _record_rows(record, used_np, used_np_right=None):
    """Yield normalized (left point, right point, color matrix) rows lazily."""
    used_np, used_np_right = _record_extents(used_np, used_np_right)
    if not isinstance(record, Mapping):
        raise TypeError("sparse join record must be an identity, COO, or matched mapping")
    if record.get("type") == "identity":
        if used_np != used_np_right:
            raise ValueError("identity P2P records require equal point extents; use explicit COO for rectangles")
        identity = np.eye(3, dtype=np.complex128)
        for point in range(used_np):
            yield point, point, identity
        return
    if record.get("layout") == "matched":
        if (type(record["indices"]).__module__.split(".", 1)[0] != "numpy"
                or type(record["values"]).__module__.split(".", 1)[0] != "numpy"):
            raise NotImplementedError(
                "cross-current matched join currently supports NumPy records only"
            )
        indices, values = np.asarray(record["indices"]), record["values"]
        if indices.shape != (used_np,) or tuple(values.shape) != (used_np, 3, 3):
            raise ValueError("matched P2P record shape is invalid")
        for left, right in enumerate(indices):
            right = int(right)
            if right == -1:
                continue
            if right < 0 or right >= used_np_right:
                raise IndexError("matched P2P right-point index is outside usedNp")
            yield left, right, values[left]
        return
    if record.get("type") != "sparse":
        raise ValueError("sparse join record must be identity or COO")
    if (type(record["indices"]).__module__.split(".", 1)[0] != "numpy"
            or type(record["values"]).__module__.split(".", 1)[0] != "numpy"):
        raise NotImplementedError(
            "cross-current COO join currently supports NumPy records only; "
            "device arrays are never copied to host"
        )
    indices, values = np.asarray(record["indices"]), record["values"]
    if indices.ndim != 2 or indices.shape[1] != 2 or tuple(values.shape) != (len(indices), 3, 3):
        raise ValueError("legacy P2P COO shape is invalid")
    if indices.dtype.kind not in "iu":
        raise TypeError("legacy P2P COO indices must be integers")
    for index, pair in enumerate(indices):
        left, right = int(pair[0]), int(pair[1])
        # Match legacy prefix selection: discard positive out-of-prefix pairs,
        # then apply NumPy negative-index normalization to retained rows.
        if left >= used_np or right >= used_np_right:
            continue
        if left < 0:
            left += used_np
        if right < 0:
            right += used_np_right
        if left < 0 or right < 0:
            raise IndexError("P2P sparse point index is outside the usedNp block")
        yield left, right, values[index]


def _class_coordinates(names, coordinates, point_equivalence):
    values = {}
    for name, coordinate in zip(names, coordinates, strict=True):
        equivalence = point_equivalence[name]
        previous = values.get(equivalence)
        if previous is not None and previous != coordinate:
            return None
        values[equivalence] = coordinate
    return values


def _point_character_values(point_equivalence, point_label_by_endpoint, class_values):
    values = {}
    for name, label in point_label_by_endpoint.items():
        coordinate = class_values[point_equivalence[name]]
        previous = values.get(label)
        if previous is not None and previous != coordinate:
            raise ValueError("one point label maps to inconsistent COO coordinates")
        values[label] = coordinate
    return values


def _selected_operand(array, labels, point_values):
    fixed = {axis: point_values[label] for axis, label in enumerate(labels)
             if label in point_values}
    for axis, coordinate in fixed.items():
        if coordinate >= array.shape[axis]:
            return None, ()
    selector = tuple(fixed.get(axis, slice(None)) for axis in range(array.ndim))
    selected = array[selector]
    retained_labels = tuple(label for axis, label in enumerate(labels) if axis not in fixed)
    return selected, retained_labels


def _output_shape(output_labels, output_point_labels, input_labels, arrays):
    dimensions = {}
    for labels, array in zip(input_labels, arrays, strict=True):
        for label, size in zip(labels, array.shape, strict=True):
            previous = dimensions.setdefault(label, int(size))
            if previous != size:
                raise ValueError(f"einsum label {label!r} has inconsistent dimensions")
    shape = []
    for label in output_labels:
        if label in output_point_labels:
            if label not in dimensions:
                raise ValueError(f"sparse join output point label {label!r} has no operand extent")
            shape.append(dimensions[label])
        elif label in dimensions:
            shape.append(dimensions[label])
        else:
            raise ValueError(f"sparse join output label {label!r} is not an operand axis")
    return tuple(shape)


_COORDINATE_JOIN_CHUNK = 64


def _coordinate_batch_equation(equation):
    left_labels, right_labels, output_labels = _parse_cycle_equation(equation)
    pattern = AxisPattern((left_labels, right_labels), output_labels)
    lane_label = left_labels[0] if left_labels else None
    lane_output_axis = output_labels.index(lane_label) if lane_label in output_labels else -1
    if (not left_labels or not right_labels or not output_labels
            or left_labels[0] != right_labels[0]
            or output_labels.count(lane_label) != 1
            or left_labels.count(left_labels[0]) != 1
            or right_labels.count(right_labels[0]) != 1):
        raise ValueError("coordinate-batched sparse equation must carry one shared lane label")
    return (pattern, lane_label, left_labels[1:], right_labels[1:],
            tuple(label for label in output_labels if label != lane_label), lane_output_axis)


def _coordinate_validate_dimensions(input_labels, arrays):
    dimensions = {}
    for labels, array in zip(input_labels, arrays, strict=True):
        if len(labels) != array.ndim:
            raise ValueError("coordinate sparse equation rank does not match its operand")
        for label, extent in zip(labels, array.shape, strict=True):
            previous = dimensions.setdefault(label, int(extent))
            if previous != extent:
                raise ValueError(f"coordinate sparse label {label!r} has inconsistent dimensions")


def coordinate_base_axes(axes, batched_ndim):
    """Map axes on a lane-prefixed operand to its unbatched operand."""
    result = []
    for axis in axes:
        if isinstance(axis, bool) or not isinstance(axis, Integral):
            raise TypeError("coordinate sparse axes must be integers")
        axis = int(axis)
        if axis < -batched_ndim or axis >= batched_ndim:
            raise ValueError("coordinate sparse axis is outside the lane-prefixed operand")
        normalized = axis % batched_ndim
        if normalized == 0:
            raise ValueError("coordinate sparse bridge axes cannot select the lane axis")
        result.append(normalized - 1)
    return tuple(result)


def _coordinate_batch_payload(batch_record, bridge, *, spin_left, spin_right,
                              color_left, color_right):
    """Validate one complete normalized lane mapping without allocating outputs."""
    if not isinstance(batch_record, Mapping) or batch_record.get("layout") != "coordinate_batch_coo":
        raise TypeError("coordinate sparse input needs the declared lane COO payload")
    point_keys = tuple(batch_record.get("point_keys", ()))
    lanes = tuple(batch_record.get("lanes", ()))
    if not point_keys or len(point_keys) != len(lanes):
        raise ValueError("coordinate sparse input has inconsistent ordered lanes")
    used_left, used_right = _record_extents(bridge.used_np, bridge.used_np_right)
    checked_lanes = []
    for lane_parts in lanes:
        if not isinstance(lane_parts, (tuple, list)):
            raise TypeError("each coordinate sparse lane must contain ordered parts")
        checked_parts = []
        for part in lane_parts:
            if not isinstance(part, (tuple, list)) or len(part) != 3:
                raise ValueError("coordinate sparse parts must be (gamma, coefficient, COO) triples")
            gamma, coefficient, record = part
            gamma = np.asarray(gamma)
            expected_gamma = ((spin_right, spin_left) if bridge.transpose_bridge
                              else (spin_left, spin_right))
            if (gamma.shape != expected_gamma or gamma.dtype.kind not in "biufc"
                    or not np.all(np.isfinite(gamma))):
                raise ValueError("coordinate sparse gamma shape or values are invalid")
            if np.result_type(gamma.dtype, np.dtype(np.complex128)).itemsize > 16:
                raise TypeError("coordinate sparse gamma exceeds the complex128 result envelope")
            coefficient_array = np.asarray(coefficient)
            if (coefficient_array.shape != () or coefficient_array.dtype.kind not in "biufc"
                    or not bool(np.isfinite(coefficient_array))):
                raise ValueError("coordinate sparse coefficient must be a finite numeric scalar")
            if np.result_type(coefficient_array.dtype, np.dtype(np.complex128)).itemsize > 16:
                raise TypeError("coordinate sparse coefficient exceeds the complex128 result envelope")
            if not isinstance(record, Mapping) or record.get("layout") != "normalized-coo":
                raise TypeError("coordinate sparse parts require normalized COO records")
            indices = np.asarray(record["indices"])
            values = np.asarray(record["values"])
            if (indices.dtype != np.dtype("int64") or indices.ndim != 2
                    or indices.shape[1] != 2
                    or values.shape != (len(indices), color_left, color_right)
                    or values.dtype.kind not in "biufc"):
                raise ValueError("normalized coordinate COO arrays have invalid shape or dtype")
            if np.result_type(values.dtype, np.dtype(np.complex128)).itemsize > 16:
                raise TypeError("coordinate sparse values exceed the complex128 result envelope")
            if (len(indices) and (
                    np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= used_left)
                    or np.any(indices[:, 1] < 0) or np.any(indices[:, 1] >= used_right))):
                raise IndexError("normalized coordinate COO index is outside its bridge extent")
            checked_parts.append((gamma, coefficient_array.item(), indices, values))
        checked_lanes.append(tuple(checked_parts))
    return point_keys, tuple(checked_lanes)


def _coordinate_gather(array, labels, lane_rows, point_rows):
    """Gather row-aligned lane/point selections as one advanced-index owner."""
    fixed_axes = tuple(axis for axis, label in enumerate(labels) if label in point_rows)
    if fixed_axes:
        moved = np.moveaxis(array, (0, *tuple(axis + 1 for axis in fixed_axes)),
                            tuple(range(len(fixed_axes) + 1)))
        selectors = (lane_rows, *(point_rows[labels[axis]] for axis in fixed_axes))
        gathered = moved[selectors]
    else:
        gathered = array[lane_rows]
    retained_labels = tuple(label for axis, label in enumerate(labels)
                            if axis not in fixed_axes)
    return gathered, retained_labels


def _flush_coordinate_bridge_rows(rows, *, left, right, left_labels, right_labels,
                                  gamma_labels, color_labels, output_labels,
                                  output_point_labels, result, result_dtype):
    if not rows:
        return
    lane_rows = np.asarray([row[0] for row in rows], dtype=np.int64)
    point_labels = tuple(dict.fromkeys(label for label in rows[0][1]))
    point_rows = {
        label: np.asarray([row[1][label] for row in rows], dtype=np.int64)
        for label in point_labels
    }
    gamma_values = np.stack([row[2] for row in rows])
    coefficients = np.asarray([row[3] for row in rows], dtype=result_dtype)
    block_values = np.stack([row[4] for row in rows])
    selected_left, selected_left_labels = _coordinate_gather(
        left, left_labels, lane_rows, point_rows,
    )
    selected_right, selected_right_labels = _coordinate_gather(
        right, right_labels, lane_rows, point_rows,
    )
    reduced_output = tuple(label for label in output_labels
                           if label not in output_point_labels)
    lane_axis = max(
        (label for labels in (selected_left_labels, selected_right_labels,
                              gamma_labels, color_labels, reduced_output)
         for label in labels),
        default=-1,
    ) + 1
    pattern = AxisPattern(
        tuple((lane_axis, *labels) for labels in (
            selected_left_labels, selected_right_labels, gamma_labels, color_labels,
        )),
        (lane_axis, *reduced_output),
    )
    operands = (selected_left, selected_right, gamma_values, block_values)
    contributions = np.einsum(
        *einsum_arguments(pattern, operands), optimize=False, dtype=result_dtype,
    )
    contributions *= coefficients.reshape((len(rows),) + (1,) * len(reduced_output))
    targets = tuple(
        np.asarray([row[1][label] for row in rows], dtype=np.int64)
        if label in output_point_labels else slice(None)
        for label in output_labels
    )
    np.add.at(result, (lane_rows, *targets), contributions)


def contract_sparse_p2p_bridge_join_batch(
    left, right, batch_record, *, bridge, equation: AxisPattern | str, backend=np,
):
    """Batched generic point-equivalence bridge, packed across lanes and rows."""
    return _contract_sparse_p2p_bridge_join_batch(
        left, right, batch_record, bridge=bridge, equation=equation, backend=backend,
    )


def _contract_sparse_p2p_bridge_join_batch(
    left, right, batch_record, *, bridge, equation, backend=np, checked_payload=None,
):
    """Internal join; checked_payload reuses this call's already validated lanes.

    Only the bridge kernel may supply checked_payload after its complete COO
    preflight. Standalone callers enter through the validating public wrapper.
    """
    module = getattr(backend, "_module", backend)
    if getattr(module, "__name__", None) != "numpy":
        raise NotImplementedError("coordinate-batched generic sparse joins require NumPy")
    pattern, lane_label, left_labels, right_labels, output_labels, lane_output_axis = _coordinate_batch_equation(equation)
    if (left.ndim != len(left_labels) + 1 or right.ndim != len(right_labels) + 1
            or left.shape[0] == 0 or left.shape[0] != right.shape[0]):
        raise ValueError("coordinate sparse equation ranks or lane extents disagree")
    _coordinate_validate_dimensions(
        ((lane_label, *left_labels), (lane_label, *right_labels)), (left, right),
    )
    left_axes = coordinate_base_axes(bridge.left_axes, left.ndim)
    right_axes = coordinate_base_axes(bridge.right_axes, right.ndim)
    if len(left_axes) != 3 or len(right_axes) != 3:
        raise ValueError("coordinate sparse bridge axes need spin, point and color slots")
    if len(set(left_axes)) != 3 or len(set(right_axes)) != 3:
        raise ValueError("coordinate sparse bridge axes must be distinct")
    static_left, static_right = left[0], right[0]
    record_left, record_right = _record_extents(bridge.used_np, bridge.used_np_right)
    expected_left, expected_right = ((record_right, record_left)
                                     if bridge.transpose_bridge
                                     else (record_left, record_right))
    if (static_left.shape[left_axes[1]] != expected_left
            or static_right.shape[right_axes[1]] != expected_right
            or static_left.shape[left_axes[2]] != static_right.shape[right_axes[2]]
            or (bridge.same_point and record_left != record_right)):
        raise ValueError("coordinate sparse endpoints disagree with bridge extents")
    spin_left, spin_right = static_left.shape[left_axes[0]], static_right.shape[right_axes[0]]
    color_left, color_right = static_left.shape[left_axes[2]], static_right.shape[right_axes[2]]
    if checked_payload is None:
        point_keys, lanes = _coordinate_batch_payload(
            batch_record, bridge, spin_left=spin_left, spin_right=spin_right,
            color_left=color_left, color_right=color_right,
        )
    else:
        point_keys, lanes = checked_payload
    if len(point_keys) != left.shape[0]:
        raise ValueError("coordinate sparse lane count disagrees with endpoint arrays")

    base_pattern = AxisPattern((left_labels, right_labels), output_labels)
    point_equivalence = bridge_point_equivalence(
        base_pattern, left_axes, right_axes,
        same_point=bridge.same_point,
    )
    point_label_by_endpoint = {
        "left": left_labels[left_axes[1]],
        "right": right_labels[right_axes[1]],
    }
    point_label_to_class = {}
    for name, label in point_label_by_endpoint.items():
        previous = point_label_to_class.setdefault(label, point_equivalence[name])
        if previous != point_equivalence[name]:
            raise ValueError("equal coordinate point labels have inconsistent bridge classes")
    output_point_labels = set(output_labels) & set(point_label_to_class)
    output_shape = _output_shape(
        output_labels, output_point_labels, (left_labels, right_labels),
        (static_left, static_right),
    )
    gamma_labels = (left_labels[left_axes[0]], right_labels[right_axes[0]])
    color_labels = (left_labels[left_axes[2]], right_labels[right_axes[2]])
    result_dtype = np.result_type(left.dtype, right.dtype, np.complex128)
    result = np.zeros((len(point_keys), *output_shape), dtype=result_dtype)
    rows = []

    def flush():
        _flush_coordinate_bridge_rows(
            rows, left=left, right=right, left_labels=left_labels,
            right_labels=right_labels, gamma_labels=gamma_labels,
            color_labels=color_labels, output_labels=output_labels,
            output_point_labels=output_point_labels, result=result,
            result_dtype=result_dtype,
        )
        rows.clear()

    for lane_index, parts in enumerate(lanes):
        for gamma, coefficient, indices, values in parts:
            oriented_gamma = gamma.T if bridge.transpose_bridge else gamma
            for row_index, pair in enumerate(indices):
                point_left, point_right = int(pair[0]), int(pair[1])
                block = values[row_index]
                if bridge.transpose_bridge:
                    point_left, point_right = point_right, point_left
                    block = block.T
                if (point_equivalence["left"] == point_equivalence["right"]
                        and point_left != point_right):
                    continue
                point_values = {
                    point_label_by_endpoint["left"]: point_left,
                    point_label_by_endpoint["right"]: point_right,
                }
                rows.append((lane_index, point_values, oriented_gamma, coefficient, block))
                if len(rows) == _COORDINATE_JOIN_CHUNK:
                    flush()
    flush()
    return np.moveaxis(result, 0, lane_output_axis)


def _flush_coordinate_cycle_rows(rows, *, left, right, left_labels, right_labels,
                                 endpoint_labels, output_labels, output_point_labels,
                                 gamma_labels, color_labels, result, result_dtype):
    if not rows:
        return
    lane_rows = np.asarray([row[0] for row in rows], dtype=np.int64)
    point_labels = tuple(dict.fromkeys(endpoint_labels.values()))
    point_rows = {
        label: np.asarray([row[1][label] for row in rows], dtype=np.int64)
        for label in point_labels
    }
    gamma1 = np.stack([row[2] for row in rows])
    coefficients = np.asarray([row[3] for row in rows], dtype=result_dtype)
    block1 = np.stack([row[4] for row in rows])
    gamma2 = np.stack([row[5] for row in rows])
    block2 = np.stack([row[6] for row in rows])
    selected_left, selected_left_labels = _coordinate_gather(
        left, left_labels, lane_rows, point_rows,
    )
    selected_right, selected_right_labels = _coordinate_gather(
        right, right_labels, lane_rows, point_rows,
    )
    reduced_output = tuple(label for label in output_labels
                           if label not in output_point_labels)
    input_labels = (
        selected_left_labels, selected_right_labels,
        gamma_labels[0], color_labels[0], gamma_labels[1], color_labels[1],
    )
    lane_axis = max(
        (label for labels in (*input_labels, reduced_output) for label in labels),
        default=-1,
    ) + 1
    pattern = AxisPattern(
        tuple((lane_axis, *labels) for labels in input_labels),
        (lane_axis, *reduced_output),
    )
    operands = (selected_left, selected_right, gamma1, block1, gamma2, block2)
    contributions = np.einsum(
        *einsum_arguments(pattern, operands), optimize=False, dtype=result_dtype,
    )
    contributions *= coefficients.reshape((len(rows),) + (1,) * len(reduced_output))
    targets = tuple(
        np.asarray([row[1][label] for row in rows], dtype=np.int64)
        if label in output_point_labels else slice(None)
        for label in output_labels
    )
    np.add.at(result, (lane_rows, *targets), contributions)


def contract_sparse_p2p_cycle_join_batch(
    left, right, first_lanes, second_lanes, *, first_bridge, second_bridge,
    equation: AxisPattern | str, point_keys, backend=np,
):
    """Join generic cycle rows across lanes in bounded packed einsum flushes.

    Secondary row buckets and the pending numeric pack are independently
    capped at 64 entries. The lane number is part of every bucket key; only
    matched row pairs enter the batched gathers and einsum.
    """
    module = getattr(backend, "_module", backend)
    if getattr(module, "__name__", None) != "numpy":
        raise NotImplementedError("coordinate-batched generic cycle joins require NumPy")
    lane_count = len(point_keys)
    pattern, lane_label, left_labels, right_labels, output_labels, lane_output_axis = _coordinate_batch_equation(equation)
    if (not lane_count or left.shape[0] != lane_count or right.shape[0] != lane_count
            or left.ndim != len(left_labels) + 1 or right.ndim != len(right_labels) + 1
            or len(first_lanes) != lane_count or len(second_lanes) != lane_count):
        raise ValueError("coordinate cycle lane axes or equations disagree")
    _coordinate_validate_dimensions(
        ((lane_label, *left_labels), (lane_label, *right_labels)), (left, right),
    )
    raw_endpoint_axes = _endpoint_axes(first_bridge, second_bridge)
    endpoint_axes = {
        name: (slot, coordinate_base_axes(axes, (left, right)[slot].ndim))
        for name, (slot, axes) in raw_endpoint_axes.items()
    }
    operands = (left[0], right[0])
    labels_by_slot = (left_labels, right_labels)
    point_label_by_endpoint, extent_by_endpoint = {}, {}
    occupied = [set(), set()]
    for name, (slot, axes) in endpoint_axes.items():
        if len(axes) != 3:
            raise ValueError("coordinate cycle endpoint needs spin, point and color axes")
        normalized = tuple(axis % operands[slot].ndim for axis in axes)
        if len(set(normalized)) != 3 or occupied[slot].intersection(normalized):
            raise ValueError("coordinate cycle bridge axes must be distinct and disjoint")
        occupied[slot].update(normalized)
        point_label_by_endpoint[name] = labels_by_slot[slot][normalized[1]]
        extent_by_endpoint[name] = operands[slot].shape[normalized[1]]
        if operands[slot].shape[normalized[2]] != 3:
            raise ValueError("coordinate cycle sparse records require Nc=3")
    for bridge, first_name, second_name in (
            (first_bridge, "first_left", "first_right"),
            (second_bridge, "second_left", "second_right")):
        record_shape = (bridge.used_np, bridge.used_np_right)
        expected = record_shape[::-1] if bridge.transpose_bridge else record_shape
        if (extent_by_endpoint[first_name] != expected[0]
                or extent_by_endpoint[second_name] != expected[1]
                or (bridge.same_point and record_shape[0] != record_shape[1])):
            raise ValueError("coordinate cycle endpoint extent disagrees with its bridge")
    equivalence = cycle_point_equivalence(pattern, first_bridge, second_bridge)
    class_extents = {}
    for name, point_class in equivalence.items():
        extent = extent_by_endpoint[name]
        if class_extents.setdefault(point_class, extent) != extent:
            raise ValueError("coordinate cycle equality class has inconsistent extents")
    point_label_to_class = {}
    for name, label in point_label_by_endpoint.items():
        if point_label_to_class.setdefault(label, equivalence[name]) != equivalence[name]:
            raise ValueError("equal coordinate point labels have inconsistent cycle classes")
    classes_first = tuple(dict.fromkeys(equivalence[name] for name in ("first_left", "first_right")))
    classes_second = tuple(dict.fromkeys(equivalence[name] for name in ("second_left", "second_right")))
    shared_classes = tuple(value for value in classes_first if value in classes_second)
    output_point_labels = set(output_labels) & set(point_label_to_class)
    output_shape = _output_shape(
        output_labels, output_point_labels, labels_by_slot, operands,
    )
    gamma_labels, color_labels = [], []
    for left_name, right_name in (("first_left", "first_right"),
                                  ("second_left", "second_right")):
        left_slot, left_axes = endpoint_axes[left_name]
        right_slot, right_axes = endpoint_axes[right_name]
        gamma_labels.append((
            labels_by_slot[left_slot][left_axes[0] % len(labels_by_slot[left_slot])],
            labels_by_slot[right_slot][right_axes[0] % len(labels_by_slot[right_slot])],
        ))
        color_labels.append((
            labels_by_slot[left_slot][left_axes[2] % len(labels_by_slot[left_slot])],
            labels_by_slot[right_slot][right_axes[2] % len(labels_by_slot[right_slot])],
        ))
    result_dtype = np.result_type(left.dtype, right.dtype, np.complex128)
    result = np.zeros((lane_count, *output_shape), dtype=result_dtype)
    rows = []

    def flush():
        _flush_coordinate_cycle_rows(
            rows, left=left, right=right, left_labels=left_labels,
            right_labels=right_labels, endpoint_labels=point_label_by_endpoint,
            output_labels=output_labels, output_point_labels=output_point_labels,
            gamma_labels=gamma_labels, color_labels=color_labels,
            result=result, result_dtype=result_dtype,
        )
        rows.clear()

    endpoint_names_first = ("first_left", "first_right")
    endpoint_names_second = ("second_left", "second_right")
    for lane_index, (parts1, parts2) in enumerate(zip(first_lanes, second_lanes, strict=True)):
        for gamma1, coefficient1, indices1, values1 in parts1:
            oriented_gamma1 = gamma1.T if first_bridge.transpose_bridge else gamma1
            for gamma2, coefficient2, indices2, values2 in parts2:
                oriented_gamma2 = gamma2.T if second_bridge.transpose_bridge else gamma2
                for offset in range(0, len(indices2), _COORDINATE_JOIN_CHUNK):
                    end = min(offset + _COORDINATE_JOIN_CHUNK, len(indices2))
                    bucket = {}
                    for row2 in range(offset, end):
                        p_left, p_right = map(int, indices2[row2])
                        block2 = values2[row2]
                        if second_bridge.transpose_bridge:
                            p_left, p_right = p_right, p_left
                            block2 = block2.T
                        second_coordinates = _class_coordinates(
                            endpoint_names_second, (p_left, p_right), equivalence,
                        )
                        if second_coordinates is None:
                            continue
                        key = (lane_index, *(second_coordinates[item] for item in shared_classes))
                        bucket.setdefault(key, []).append((
                            p_left, p_right, oriented_gamma2, coefficient2, block2,
                        ))
                    for row1, pair1 in enumerate(indices1):
                        p_left1, p_right1 = map(int, pair1)
                        block1 = values1[row1]
                        if first_bridge.transpose_bridge:
                            p_left1, p_right1 = p_right1, p_left1
                            block1 = block1.T
                        first_coordinates = _class_coordinates(
                            endpoint_names_first, (p_left1, p_right1), equivalence,
                        )
                        if first_coordinates is None:
                            continue
                        key = (lane_index, *(first_coordinates[item] for item in shared_classes))
                        for p_left2, p_right2, gamma2_row, coefficient2_row, block2_row in bucket.get(key, ()):
                            class_values = dict(first_coordinates)
                            class_values.update(_class_coordinates(
                                endpoint_names_second, (p_left2, p_right2), equivalence,
                            ))
                            point_values = _point_character_values(
                                equivalence, point_label_by_endpoint, class_values,
                            )
                            rows.append((
                                lane_index, point_values, oriented_gamma1,
                                coefficient1 * coefficient2_row, block1,
                                gamma2_row, block2_row,
                            ))
                            if len(rows) == _COORDINATE_JOIN_CHUNK:
                                flush()
    flush()
    return np.moveaxis(result, 0, lane_output_axis)


def contract_sparse_p2p_join(
    left,
    right,
    first_parts,
    second_parts,
    *,
    first_bridge,
    second_bridge,
    equation: AxisPattern | str,
    point_equivalence: Mapping[str, object],
    output_labels: tuple[int, ...] | str,
    backend=np,
    join_index_block_size: int = DEFAULT_JOIN_INDEX_BLOCK_SIZE,
):
    """Close two COO P2P vertices while enforcing caller-supplied point classes.

    The secondary-COO bucket contains at most ``join_index_block_size`` rows.
    The record-pair loop is streamed and invokes one explicit, no-search einsum
    per matching pair. Temporary join storage is therefore independent of
    ``nnz_first * nnz_second``; only the selected contraction result and this
    bounded index chunk are live in addition to source tensors.

    ``point_equivalence`` maps the four endpoint names returned by
    :func:`cycle_point_equivalence` to explicit equality-class identities.
    ``output_labels`` is the caller's requested einsum output subscript; point
    labels consumed by COO rows are restored by sparse indexed accumulation.
    """
    if isinstance(join_index_block_size, (bool, np.bool_)) or not isinstance(
            join_index_block_size, Integral) or int(join_index_block_size) <= 0:
        raise ValueError("join_index_block_size must be a positive integer")
    module = getattr(backend, "_module", backend)
    if getattr(module, "__name__", None) not in (None, "numpy"):
        raise NotImplementedError("streaming cross-current COO join is CPU-only")
    join_index_block_size = int(join_index_block_size)
    equation = parse_axis_pattern(equation)
    left_labels, right_labels, equation_output = _parse_cycle_equation(equation)
    output_labels = axis_ids(output_labels)
    if output_labels != equation_output:
        raise ValueError("explicit output_labels must match the contraction equation")
    equivalence = dict(point_equivalence)
    if set(equivalence) != set(_ENDPOINTS):
        raise ValueError("point_equivalence must name all four bridged endpoint axes")
    for name, value in equivalence.items():
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"point equivalence class for {name} must be hashable") from exc

    operands = (left, right)
    operand_labels = (left_labels, right_labels)
    endpoint_axes = _endpoint_axes(first_bridge, second_bridge)
    point_label_by_endpoint = {}
    extent_by_endpoint = {}
    occupied_axes = [set(), set()]
    for name, (slot, triple) in endpoint_axes.items():
        if len(triple) != 3:
            raise ValueError("sparse bridge endpoint axes must have spin, point, and color slots")
        normalized = tuple(axis % operands[slot].ndim for axis in triple)
        if len(set(normalized)) != 3 or occupied_axes[slot].intersection(normalized):
            raise ValueError("cycle bridge axes must be distinct and disjoint per operand")
        occupied_axes[slot].update(normalized)
        point_label_by_endpoint[name] = operand_labels[slot][normalized[1]]
        extent_by_endpoint[name] = int(operands[slot].shape[normalized[1]])
        if operands[slot].shape[normalized[2]] != 3:
            raise ValueError("sparse join color dimensions disagree with Nc=3")

    for bridge, left_name, right_name in (
        (first_bridge, "first_left", "first_right"),
        (second_bridge, "second_left", "second_right"),
    ):
        record_shape = bridge.point_shape
        input_shape = (record_shape[::-1] if bridge.transpose_bridge else record_shape)
        if (extent_by_endpoint[left_name] != input_shape[0]
                or extent_by_endpoint[right_name] != input_shape[1]
                or (bridge.same_point and record_shape[0] != record_shape[1])):
            raise ValueError("sparse join endpoint dimensions disagree with its COO point shape")

    class_extents = {}
    for name, point_class in equivalence.items():
        extent = extent_by_endpoint[name]
        previous = class_extents.setdefault(point_class, extent)
        if previous != extent:
            raise ValueError("one sparse join point-equivalence class has inconsistent extents")

    point_axis_to_class = {}
    for name, label in point_label_by_endpoint.items():
        old = point_axis_to_class.setdefault(label, equivalence[name])
        if old != equivalence[name]:
            raise ValueError("equal point labels must share one explicit point-equivalence class")
    class_names_first, class_names_second = _ENDPOINTS[:2], _ENDPOINTS[2:]
    classes_first = tuple(dict.fromkeys(equivalence[name] for name in class_names_first))
    classes_second = tuple(dict.fromkeys(equivalence[name] for name in class_names_second))
    shared_classes = tuple(value for value in classes_first if value in classes_second)
    output_point_labels = {
        label for label, point_class in point_axis_to_class.items()
        if label in output_labels
    }
    output_shape = _output_shape(
        output_labels, output_point_labels, operand_labels, operands,
    )
    result = backend.zeros(
        output_shape,
        dtype=backend.result_type(left, right, np.complex128),
    )
    reduced_output = tuple(label for label in output_labels if label not in output_point_labels)
    first_parts, second_parts = tuple(first_parts), tuple(second_parts)

    for gamma1, coefficient1, record1 in tuple(first_parts):
        gamma1 = backend.asarray(gamma1)
        if first_bridge.transpose_bridge:
            gamma1 = gamma1.T
        first_left_slot, first_left_axes = endpoint_axes["first_left"]
        first_right_slot, first_right_axes = endpoint_axes["first_right"]
        first_gamma_labels = (
            operand_labels[first_left_slot][first_left_axes[0] % len(operand_labels[first_left_slot])],
            operand_labels[first_right_slot][first_right_axes[0] % len(operand_labels[first_right_slot])],
        )
        first_color_labels = (
            operand_labels[first_left_slot][first_left_axes[2] % len(operand_labels[first_left_slot])],
            operand_labels[first_right_slot][first_right_axes[2] % len(operand_labels[first_right_slot])],
        )
        for gamma2, coefficient2, record2 in second_parts:
            gamma2 = backend.asarray(gamma2)
            if second_bridge.transpose_bridge:
                gamma2 = gamma2.T
            second_left_slot, second_left_axes = endpoint_axes["second_left"]
            second_right_slot, second_right_axes = endpoint_axes["second_right"]
            second_gamma_labels = (
                operand_labels[second_left_slot][second_left_axes[0] % len(operand_labels[second_left_slot])],
                operand_labels[second_right_slot][second_right_axes[0] % len(operand_labels[second_right_slot])],
            )
            second_color_labels = (
                operand_labels[second_left_slot][second_left_axes[2] % len(operand_labels[second_left_slot])],
                operand_labels[second_right_slot][second_right_axes[2] % len(operand_labels[second_right_slot])],
            )
            coefficient = coefficient1 * coefficient2

            second_iter = _record_rows(
                record2, second_bridge.used_np, second_bridge.used_np_right,
            )
            while True:
                chunk = tuple(islice(second_iter, join_index_block_size))
                if not chunk:
                    break
                bucket = {}
                for point_left, point_right, values in chunk:
                    if second_bridge.transpose_bridge:
                        point_left, point_right = point_right, point_left
                    class_values = _class_coordinates(
                        class_names_second, (point_left, point_right), equivalence,
                    )
                    if class_values is None:
                        continue
                    key = tuple(class_values[item] for item in shared_classes)
                    bucket.setdefault(key, []).append((point_left, point_right, values))

                # Re-scan the first COO stream per bounded secondary block;
                # duplicate rows remain separate summands.
                for first_left_point, first_right_point, first_record_values in _record_rows(
                        record1, first_bridge.used_np, first_bridge.used_np_right):
                    if first_bridge.transpose_bridge:
                        first_left_point, first_right_point = first_right_point, first_left_point
                    first_class_values = _class_coordinates(
                        class_names_first, (first_left_point, first_right_point), equivalence,
                    )
                    if first_class_values is None:
                        continue
                    key = tuple(first_class_values[item] for item in shared_classes)
                    first_values = backend.asarray(first_record_values)
                    if first_bridge.transpose_bridge:
                        first_values = first_values.T
                    for second_left_point, second_right_point, second_values in bucket.get(key, ()):
                        class_values = dict(first_class_values)
                        class_values.update(_class_coordinates(
                            class_names_second, (second_left_point, second_right_point), equivalence,
                        ))
                        point_char_values = _point_character_values(
                            equivalence, point_label_by_endpoint, class_values,
                        )
                        selected_left, selected_left_labels = _selected_operand(
                            left, left_labels, point_char_values,
                        )
                        if selected_left is None:
                            continue
                        selected_right, selected_right_labels = _selected_operand(
                            right, right_labels, point_char_values,
                        )
                        if selected_right is None:
                            continue
                        second_values = backend.asarray(second_values)
                        if second_bridge.transpose_bridge:
                            second_values = second_values.T
                        pattern = AxisPattern(
                            (selected_left_labels, selected_right_labels,
                             first_gamma_labels, first_color_labels,
                             second_gamma_labels, second_color_labels),
                            reduced_output,
                        )
                        operands = (selected_left, selected_right, gamma1, first_values,
                                    gamma2, second_values)
                        value = backend.einsum(
                            *einsum_arguments(pattern, operands), optimize=False,
                        )
                        if not output_point_labels:
                            result[...] += coefficient * value
                        else:
                            target = tuple(
                                point_char_values[label]
                                if label in output_point_labels else slice(None)
                                for label in output_labels
                            )
                            result[target] += coefficient * value
    return result


def contract_sparse_p2p_bridge_join(
    left,
    right,
    parts,
    *,
    used_np: int,
    used_np_right: int | None = None,
    left_axes: tuple[int, int, int],
    right_axes: tuple[int, int, int],
    same_point: bool,
    transpose_bridge: bool,
    equation: AxisPattern | str,
    point_equivalence: Mapping[str, object],
    output_labels: tuple[int, ...] | str,
    backend=np,
):
    """Apply one COO bridge while retaining any explicitly related point axis.

    The bridge row is streamed directly into a small einsum result. Any input
    axes carrying either explicit endpoint point label are indexed at that
    COO coordinate, and requested output point labels are restored by indexed
    accumulation. The operation never forms a point-pair tensor.
    """
    module = getattr(backend, "_module", backend)
    if getattr(module, "__name__", None) not in (None, "numpy"):
        raise NotImplementedError("external-point COO bridges currently use NumPy records only")
    equation = parse_axis_pattern(equation)
    left_labels, right_labels, equation_output = _parse_cycle_equation(equation)
    output_labels = axis_ids(output_labels)
    if output_labels != equation_output:
        raise ValueError("explicit output_labels must match the contraction equation")
    if set(point_equivalence) != {"left", "right"}:
        raise ValueError("bridge point_equivalence must name left and right endpoint classes")
    try:
        hash(point_equivalence["left"])
        hash(point_equivalence["right"])
    except TypeError as exc:
        raise TypeError("bridge point-equivalence labels must be hashable") from exc
    record_left_count, record_right_count = _record_extents(used_np, used_np_right)
    left_axes, right_axes = tuple(left_axes), tuple(right_axes)
    if len(left_axes) != 3 or len(right_axes) != 3:
        raise ValueError("sparse bridge endpoint axes need spin, point and color slots")
    left_axes = tuple(axis % left.ndim for axis in left_axes)
    right_axes = tuple(axis % right.ndim for axis in right_axes)
    input_left_count, input_right_count = (
        (record_right_count, record_left_count)
        if transpose_bridge else (record_left_count, record_right_count)
    )
    if (len(set(left_axes)) != 3 or len(set(right_axes)) != 3
            or left.shape[left_axes[1]] != input_left_count
            or right.shape[right_axes[1]] != input_right_count
            or left.shape[left_axes[2]] != right.shape[right_axes[2]]
            or (same_point and record_left_count != record_right_count)):
        raise ValueError("sparse join endpoint shapes disagree")
    point_labels = {
        "left": left_labels[left_axes[1]],
        "right": right_labels[right_axes[1]],
    }
    for label in output_labels:
        if label not in set(left_labels + right_labels) | set(point_labels.values()):
            raise ValueError(f"sparse bridge output label {label!r} is not an input/output point axis")
    point_axis_to_class = {}
    for name, label in point_labels.items():
        previous = point_axis_to_class.setdefault(label, point_equivalence[name])
        if previous != point_equivalence[name]:
            raise ValueError("equal endpoint point labels must use the same equivalence class")
    output_point_labels = set(output_labels) & set(point_axis_to_class)
    dimensions = {}
    for labels, array in zip((left_labels, right_labels), (left, right), strict=True):
        for label, size in zip(labels, array.shape, strict=True):
            old = dimensions.setdefault(label, int(size))
            if old != size:
                raise ValueError(f"einsum label {label!r} has inconsistent dimensions")
    output_shape = tuple(dimensions[label] for label in output_labels)
    result = backend.zeros(output_shape, dtype=backend.result_type(left, right, np.complex128))
    reduced_output = tuple(label for label in output_labels if label not in output_point_labels)
    gamma_labels = (left_labels[left_axes[0]], right_labels[right_axes[0]])
    color_labels = (left_labels[left_axes[2]], right_labels[right_axes[2]])
    point_equivalence = dict(point_equivalence)
    for gamma, coefficient, record in tuple(parts):
        gamma = backend.asarray(gamma)
        if transpose_bridge:
            gamma = gamma.T
        for point_left, point_right, values in _record_rows(
                record, record_left_count, record_right_count):
            if transpose_bridge:
                point_left, point_right = point_right, point_left
            if (point_equivalence["left"] == point_equivalence["right"]
                    and point_left != point_right):
                continue
            point_class_values = {
                point_equivalence["left"]: point_left,
                point_equivalence["right"]: point_right,
            }
            point_character_values = {
                label: point_class_values[point_equivalence[name]]
                for name, label in point_labels.items()
            }
            selected_left, selected_left_labels = _selected_operand(
                left, left_labels, point_character_values,
            )
            if selected_left is None:
                continue
            selected_right, selected_right_labels = _selected_operand(
                right, right_labels, point_character_values,
            )
            if selected_right is None:
                continue
            values = backend.asarray(values)
            if transpose_bridge:
                values = values.T
            pattern = AxisPattern(
                (selected_left_labels, selected_right_labels, gamma_labels, color_labels),
                reduced_output,
            )
            operands = (selected_left, selected_right, gamma, values)
            contribution = backend.einsum(
                *einsum_arguments(pattern, operands), optimize=False,
            )
            if not output_point_labels:
                result[...] += coefficient * contribution
            else:
                target = tuple(
                    point_character_values[label]
                    if label in output_point_labels else slice(None)
                    for label in output_labels
                )
                result[target] += coefficient * contribution
    return result


__all__ = [
    "DEFAULT_JOIN_INDEX_BLOCK_SIZE", "ESTIMATED_JOIN_INDEX_BYTES_PER_ROW",
    "JOIN_COST_STATUS",
    "contract_sparse_p2p_bridge_join", "contract_sparse_p2p_join",
]
