"""Pure coordinate-batched sparse workspace descriptions and frozen admission.

The allocation formulas and bridge lookup were copied verbatim from execution/
scene_runtime.py and operations/sparse_bridge.py during architecture audit B4.
They describe the existing NumPy kernel ABI, without importing or executing X.
Keep dual-estimate tests against X when that ABI changes.
"""
from collections.abc import Mapping
from dataclasses import replace
import math
import numpy as np
from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, SparseBridgeSpec, bridge_intermediate_labels,
    sparse_cycle_endpoint_axes,
)


def _nbytes(spec):
    return math.prod(spec.shape) * spec.dtype.itemsize


def _operand_order(bridge: SparseBridgeSpec) -> SparseBridgeSpec:
    if not bridge.reverse_operands:
        return bridge
    return replace(
        bridge,
        left_axes=bridge.right_axes,
        right_axes=bridge.left_axes,
        transpose_bridge=not bridge.transpose_bridge,
        reverse_operands=False,
    )


def _action_bridges(state, action: Action):
    """Resolve bridge metadata for one already-planned sparse action."""
    if action.kind != "sparse_bridge" or len(action.operands) not in (3, 4):
        raise ValueError("sparse bridge action must carry two legs and one or two records")
    try:
        first_record = next(
            item
            for item in state.problem.sparse_bridges
            if item.scene in (None, action.term_index)
            and (item.left, item.right, item.record) == action.operands[:3]
            and item.same_point == action.same_point
        )
    except StopIteration as exc:
        raise ValueError("sparse bridge action has no matching delivered bridge record") from exc
    first = _operand_order(first_record)
    action_right_count = action.used_np if action.used_np_right is None else action.used_np_right
    if (
        first.left_axes != action.left_axes
        or first.right_axes != action.right_axes
        or first.transpose_bridge != action.transpose_bridge
    ):
        raise ValueError("sparse bridge action endpoint order does not match its term")
    if first.point_shape != (action.used_np, action_right_count):
        raise ValueError("sparse bridge action point shape does not match its record")
    if len(action.operands) == 3:
        return first, None
    try:
        second_record = next(
            item
            for item in state.problem.sparse_bridges
            if item is not first_record
            and item.scene in (None, action.term_index)
            and item.record == action.operands[3]
            and {item.left, item.right} == {first.left, first.right}
        )
    except StopIteration as exc:
        raise ValueError("sparse cycle action has no matching second bridge record") from exc
    second = _operand_order(second_record)
    return first, second


def coordinate_batch_sparse_workspace_bytes(state, action):
    """Describe the NumPy sparse kernel host workspace for one marked NumPy sparse action.

    This is an allocation envelope from the checked C metadata and TensorSpecs,
    not a hardware/RSS bound.  The planner's owner and staging fields account
    for raw blocks, normalized source snapshots and logical endpoint/result
    owners; this function models additional kernel copies and scratch.
    """
    bridge, cycle_bridge = _action_bridges(state, action)
    source = state.problem.sources[bridge.record]
    metadata = source.metadata
    if (not isinstance(metadata, Mapping)
            or metadata.get("family") != "p2p-coordinate-batch"):
        return None
    if state.problem.backend != "numpy":
        raise NotImplementedError(
            "coordinate-batched sparse workspace is budgeted only for NumPy"
        )

    if cycle_bridge is not None:
        second_metadata = state.problem.sources[cycle_bridge.record].metadata
        if (not isinstance(second_metadata, Mapping)
                or second_metadata.get("family") != "p2p-coordinate-batch"):
            raise ValueError("coordinate-batched cycles require two marked sparse records")
        return _coordinate_batch_cycle_workspace_bytes(
            state, action, bridge, cycle_bridge, metadata, second_metadata,
        )

    lane_bounds = tuple(metadata.get("lane_nnz_upper", ()))
    lane_sources = tuple(metadata.get("lane_sources", ()))
    coordinate_keys = tuple(metadata.get("coordinate_keys", ()))
    part_count = metadata.get("part_count")
    if (type(part_count) is not int or part_count <= 0
            or not coordinate_keys or len(lane_bounds) != len(coordinate_keys)
            or len(lane_sources) != len(coordinate_keys)
            or any(type(bound) is not int or bound < 0 for bound in lane_bounds)
            or bridge.nnz_upper != sum(lane_bounds)
            or bridge.part_count != part_count):
        raise ValueError("coordinate-batched workspace metadata disagrees with its bridge")

    raw_value_dtypes = []
    raw_value_sizes = []
    for lane in lane_sources:
        raw = lane.get("metadata") if isinstance(lane, Mapping) else None
        if not isinstance(raw, Mapping):
            raise ValueError("coordinate-batched workspace lacks lane raw metadata")
        kind = raw.get("kind")
        raw_dtype = np.dtype(raw["value_dtype"])
        raw_value_dtypes.append(
            np.dtype(np.complex128) if kind == "identity" else raw_dtype
        )
        raw_value_sizes.append(
            max(raw_dtype.itemsize, np.dtype(np.complex128).itemsize)
            if kind == "identity" else raw_dtype.itemsize
        )
    stacked_value_itemsize = np.result_type(*raw_value_dtypes).itemsize if raw_value_dtypes else (
        np.dtype(np.complex128).itemsize
    )
    filtered_value_itemsize = max(raw_value_sizes, default=np.dtype(np.complex128).itemsize)

    left_spec = state.catalog[bridge.left]
    right_spec = state.catalog[bridge.right]
    if (left_spec.shape[0] != len(coordinate_keys)
            or right_spec.shape[0] != len(coordinate_keys)):
        raise ValueError("coordinate-batched TensorSpecs do not match their lane keys")
    # The batch kernel moves lane to axis 0 before moving the three bridge
    # axes.  It is not an external contraction axis: count only the remaining
    # physical axes here, then include lane explicitly where the kernel owns a
    # batch result.
    bridge_left_axes = {axis % len(left_spec.shape) for axis in bridge.left_axes}
    bridge_right_axes = {axis % len(right_spec.shape) for axis in bridge.right_axes}
    if 0 in bridge_left_axes or 0 in bridge_right_axes:
        raise ValueError("coordinate-batched bridge axes cannot include the lane axis")
    left_axes = {0} | bridge_left_axes
    right_axes = {0} | bridge_right_axes
    left_external = math.prod(
        size for axis, size in enumerate(left_spec.shape) if axis not in left_axes
    )
    right_external = math.prod(
        size for axis, size in enumerate(right_spec.shape) if axis not in right_axes
    )
    spin_left = left_spec.shape[bridge.left_axes[0] % len(left_spec.shape)]
    spin_right = right_spec.shape[bridge.right_axes[0] % len(right_spec.shape)]
    color_left = left_spec.shape[bridge.left_axes[2] % len(left_spec.shape)]
    color_right = right_spec.shape[bridge.right_axes[2] % len(right_spec.shape)]
    if color_left != color_right:
        raise ValueError("coordinate-batched endpoint colors disagree")

    c_rows = min(64, part_count * sum(lane_bounds))
    left_labels, right_labels = action.equation.inputs
    output_labels = action.equation.output
    if (not left_labels or not right_labels or left_labels[0] != right_labels[0]
            or output_labels.count(left_labels[0]) != 1):
        raise ValueError("coordinate-batched action does not expose one lane axis")
    lane_label = left_labels[0]

    def unbatch_axis(axis):
        return axis if axis < 0 else axis - 1

    base_left_labels = left_labels[1:]
    base_right_labels = right_labels[1:]
    base_output_labels = tuple(label for label in output_labels if label != lane_label)
    base_left_axes = tuple(unbatch_axis(axis) for axis in bridge.left_axes)
    base_right_axes = tuple(unbatch_axis(axis) for axis in bridge.right_axes)
    base_pattern = AxisPattern((base_left_labels, base_right_labels), base_output_labels)
    _, raw_base_labels, _ = bridge_intermediate_labels(
        base_pattern, base_left_axes, base_right_axes,
        same_point=bridge.same_point,
    )
    raw_labels = (lane_label, *raw_base_labels)
    point_left = base_left_labels[base_left_axes[1] % len(base_left_labels)]
    point_right = base_right_labels[base_right_axes[1] % len(base_right_labels)]
    filter_points = bridge.same_point or point_left == point_right
    f_rows = min(64, max(lane_bounds, default=0)) if filter_points else 0
    if c_rows and f_rows:
        filtered_base_rows = c_rows + 2 * f_rows - 2
        filtered = (
            filtered_base_rows * (
                2 * np.dtype("int64").itemsize
                + color_left * color_right * filtered_value_itemsize
            )
            + 2 * f_rows
        )
    else:
        filtered = 0

    reshape_copies = _nbytes(left_spec) + _nbytes(right_spec)
    gathered = c_rows * (
        left_external * spin_left * color_left * left_spec.dtype.itemsize
        + spin_right * color_right * right_external * right_spec.dtype.itemsize
    )
    packed = c_rows * (
        3 * np.dtype("int64").itemsize
        + spin_left * spin_right * np.dtype(np.complex128).itemsize
        + np.dtype(np.complex128).itemsize
        + color_left * color_right * stacked_value_itemsize
    )
    contributions = c_rows * left_external * right_external * np.dtype(np.complex128).itemsize
    projected_raw = (
        len(coordinate_keys) * left_external * right_external
        * np.dtype(np.complex128).itemsize
        if (raw_labels != output_labels
            and point_left not in base_output_labels
            and point_right not in base_output_labels) else 0
    )

    # Validation predicates are covered by the existing P ownership/staging
    # points: raw COO index masks are smaller than their registered raw rows,
    # normalized-index masks are smaller than each owned int64 pair row, and
    # the gamma finiteness mask is smaller than the per-part complex128 gamma
    # snapshot.  Do not add these a second time to the X-only scratch reserve.
    compute_workspace = (
        reshape_copies + gathered + packed + filtered + contributions + projected_raw
    )
    return compute_workspace


def _coordinate_batch_cycle_workspace_bytes(
    state, action, first_bridge, second_bridge, first_metadata, second_metadata,
):
    """Known ndarray scratch for the lane-keyed two-record pair-join kernel.

    Python buckets and row-reference queues are caller-declared opaque staging;
    this count covers the known packed NumPy arrays beyond P's resident source
    snapshots and logical result owner.
    """
    if state.problem.backend != "numpy":
        raise NotImplementedError(
            "coordinate-batched sparse workspace is budgeted only for NumPy"
        )

    def read_contract(bridge, metadata):
        bounds = tuple(metadata.get("lane_nnz_upper", ()))
        lanes = tuple(metadata.get("lane_sources", ()))
        keys = tuple(metadata.get("coordinate_keys", ()))
        part_count = metadata.get("part_count")
        if (type(part_count) is not int or part_count <= 0 or not keys
                or len(bounds) != len(keys) or len(lanes) != len(keys)
                or any(type(bound) is not int or bound < 0 for bound in bounds)
                or bridge.nnz_upper != sum(bounds)
                or bridge.part_count != part_count):
            raise ValueError("coordinate-batched cycle metadata disagrees with its bridge")
        value_dtypes = []
        for lane in lanes:
            raw = lane.get("metadata") if isinstance(lane, Mapping) else None
            if not isinstance(raw, Mapping):
                raise ValueError("coordinate-batched cycle lacks lane raw metadata")
            raw_dtype = np.dtype(raw["value_dtype"])
            value_dtypes.append(
                np.dtype(np.complex128) if raw.get("kind") == "identity" else raw_dtype
            )
        stacked_itemsize = (
            np.result_type(*value_dtypes).itemsize if value_dtypes
            else np.dtype(np.complex128).itemsize
        )
        return keys, bounds, lanes, part_count, stacked_itemsize

    first = read_contract(first_bridge, first_metadata)
    second = read_contract(second_bridge, second_metadata)
    keys, first_bounds, _, first_parts, first_values_itemsize = first
    second_keys, second_bounds, _, second_parts, second_values_itemsize = second
    if keys != second_keys:
        raise ValueError("coordinate-batched cycle records disagree on ordered lane keys")
    lane_count = len(keys)

    left_spec = state.catalog[first_bridge.left]
    right_spec = state.catalog[first_bridge.right]
    if (left_spec.shape[0] != lane_count or right_spec.shape[0] != lane_count
            or np.result_type(left_spec.dtype, right_spec.dtype, np.complex128).itemsize > 16):
        raise ValueError("coordinate-batched cycle TensorSpecs disagree with lane metadata")
    full_left_labels, full_right_labels = action.equation.inputs
    output_labels = action.equation.output
    if (not full_left_labels or not full_right_labels
            or full_left_labels[0] != full_right_labels[0]
            or output_labels.count(full_left_labels[0]) != 1):
        raise ValueError("coordinate-batched cycle equation has no unique output lane axis")
    lane_label = full_left_labels[0]
    base_labels = (full_left_labels[1:], full_right_labels[1:])
    specs = (left_spec, right_spec)
    endpoint_axes = sparse_cycle_endpoint_axes(first_bridge, second_bridge)
    point_labels = {}
    endpoint_dimensions = {}
    occupied = [set(), set()]
    for name, (slot, axes) in endpoint_axes.items():
        spec = specs[slot]
        if len(axes) != 3:
            raise ValueError("coordinate-batched cycle endpoints need three axes")
        normalized = tuple(axis % len(spec.shape) for axis in axes)
        base_axes = tuple(axis - 1 for axis in normalized)
        if (any(axis == 0 for axis in normalized)
                or len(set(normalized)) != 3
                or occupied[slot].intersection(normalized)):
            raise ValueError("coordinate-batched cycle endpoint axes overlap the lane or each other")
        occupied[slot].update(normalized)
        labels = base_labels[slot]
        point_labels[name] = labels[base_axes[1]]
        endpoint_dimensions[name] = tuple(spec.shape[axis] for axis in normalized)
        if endpoint_dimensions[name][2] != 3:
            raise ValueError("coordinate-batched cycle kernel requires Nc=3")

    for bridge, left_name, right_name in (
        (first_bridge, "first_left", "first_right"),
        (second_bridge, "second_left", "second_right"),
    ):
        left_spin, left_points, left_color = endpoint_dimensions[left_name]
        right_spin, right_points, right_color = endpoint_dimensions[right_name]
        expected_points = (bridge.used_np, bridge.used_np_right)
        if bridge.transpose_bridge:
            expected_points = expected_points[::-1]
        if (left_points != expected_points[0] or right_points != expected_points[1]
                or left_color != right_color
                or (bridge.same_point and bridge.used_np != bridge.used_np_right)):
            raise ValueError("coordinate-batched cycle endpoint extents disagree with bridge")

    unique_point_labels = tuple(dict.fromkeys(point_labels.values()))
    output_point_labels = tuple(
        label for label in unique_point_labels if label in output_labels
    )
    free_left = math.prod(
        size for axis, size in enumerate(left_spec.shape[1:])
        if base_labels[0][axis] not in unique_point_labels
    )
    free_right = math.prod(
        size for axis, size in enumerate(right_spec.shape[1:])
        if base_labels[1][axis] not in unique_point_labels
    )
    output_labels_without_lane = tuple(label for label in output_labels if label != lane_label)
    dimensions = {}
    for labels, spec in zip(base_labels, specs, strict=True):
        for label, size in zip(labels, spec.shape[1:], strict=True):
            previous = dimensions.setdefault(label, int(size))
            if previous != size:
                raise ValueError("coordinate-batched cycle equation dimensions disagree")
    reduced_output_elements = math.prod(
        dimensions[label] for label in output_labels_without_lane
        if label not in output_point_labels
    )

    pair_upper = sum(
        first_parts * left_bound * second_parts * right_bound
        for left_bound, right_bound in zip(first_bounds, second_bounds, strict=True)
    )
    rows = min(64, pair_upper)
    selectors = rows * 8 * (1 + len(unique_point_labels) + len(output_point_labels))
    gamma_packs = rows * 16 * sum(
        endpoint_dimensions[left_name][0] * endpoint_dimensions[right_name][0]
        for left_name, right_name in (
            ("first_left", "first_right"),
            ("second_left", "second_right"),
        )
    )
    coefficient_pack = rows * 16
    value_packs = rows * sum(
        endpoint_dimensions[left_name][2] * endpoint_dimensions[right_name][2] * itemsize
        for left_name, right_name, itemsize in (
            ("first_left", "first_right", first_values_itemsize),
            ("second_left", "second_right", second_values_itemsize),
        )
    )
    gathers = rows * (
        free_left * left_spec.dtype.itemsize + free_right * right_spec.dtype.itemsize
    )
    contributions = rows * reduced_output_elements * 16
    return selectors + gamma_packs + coefficient_pack + value_packs + gathers + contributions



def _workspace_correction(plan):
    problem = plan.state.problem
    marked = {
        key for key, source in problem.sources.items()
        if isinstance(source.metadata, Mapping)
        and source.metadata.get("family") == "p2p-coordinate-batch"
    }
    if not marked:
        return 0
    points_by_action = {}
    for point in plan.memory:
        if point.phase == "sparse_bridge":
            points_by_action.setdefault(point.action_index, []).append(point)
    seen = set()
    correction = 0
    executed_marked = False
    for index, action in enumerate(plan.state.actions):
        if action.kind != "sparse_bridge":
            continue
        bridges = tuple(
            bridge for bridge in _action_bridges(plan.state, action)
            if bridge is not None
        )
        marked_records = tuple(
            bridge.record for bridge in bridges
            if (isinstance(problem.sources[bridge.record].metadata, Mapping)
                and problem.sources[bridge.record].metadata.get("family")
                == "p2p-coordinate-batch")
        )
        if not marked_records:
            continue
        if len(marked_records) != len(bridges):
            raise ValueError("coordinate-batched sparse actions cannot mix marked and scalar records")
        seen.update(marked_records)
        if action.reused:
            continue
        executed_marked = True
        if problem.backend != "numpy":
            raise NotImplementedError(
                "coordinate-batched sparse workspace admission currently requires NumPy"
            )
        workspace = coordinate_batch_sparse_workspace_bytes(plan.state, action)
        if workspace is None:
            raise ValueError("marked coordinate-batch action lost its source contract")
        points = points_by_action.get(index, ())
        if len(points) != 1:
            raise ValueError("audited plan has no unique sparse action memory point")
        correction = max(
            correction,
            max(0, workspace - points[0].workspace_host_bytes),
        )
    if seen != marked:
        raise ValueError("every coordinate-batched source must have one executed sparse action")
    if not executed_marked:
        return 0

    return correction


def prepare_coordinate_batch_workspace(plan):
    """Freeze known sparse workspace before saving or executing a storage plan.

    Return ``(plan, added_host_bytes)``. The copied legacy correction is charged
    uniformly to every memory point, preserving its exact all-point envelope.
    Charge it as known workspace, leaving caller-declared opaque reserves intact.
    The result is idempotent, and unchanged plans retain object identity.
    """
    correction = _workspace_correction(plan)
    if not correction:
        return plan, 0
    memory = tuple(replace(point, workspace_host_bytes=point.workspace_host_bytes + correction)
                   for point in plan.memory)
    host = max(point.host_bytes for point in memory)
    device = max(point.device_bytes for point in memory)
    primary = device if plan.state.problem.backend == "cupy" else host
    feasible = (primary <= plan.memory_limit
                and (plan.host_memory_limit is None or host <= plan.host_memory_limit)
                and (plan.disk_limit_bytes is None or plan.peak_disk_bytes <= plan.disk_limit_bytes))
    return replace(plan, memory=memory, peak_host_bytes=host, peak_device_bytes=device,
                   feasible=feasible), correction


__all__ = ["coordinate_batch_sparse_workspace_bytes", "prepare_coordinate_batch_workspace"]
