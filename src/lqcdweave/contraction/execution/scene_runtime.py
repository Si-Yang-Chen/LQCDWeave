"""Replay audited contraction actions and collect per-action timing feedback."""

from __future__ import annotations
from lqcdweave.contracts.correlator.api import validate_executable_resources

from .diagnostics import debug_enabled
from .binding_cache import bind_cached, native_state_binding
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
import math
from typing import Any, Callable

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action,
    AxisPattern,
    MatrixKernelSpec,
    FactorizedWideReduction,
    SourceSpec,
    StateDescriptor,
    SourceLoader,
    bridge_intermediate_labels,
    cycle_external_labels,
    sparse_cycle_endpoint_axes,
    DEFAULT_JOIN_INDEX_BLOCK_SIZE,
    dense_batch_leaves, DenseGemmGroup, DenseKernelBatch, DenseBatchOperand, DenseBatchResult,
    ESTIMATED_JOIN_INDEX_BYTES_PER_ROW,
    JOIN_COST_STATUS, StorageEvent,
)
from lqcdweave.contracts.measurement import (
    ActionTiming,
    FileIOEvent,
    PhaseTiming,
    ReplayReport,
)
from lqcdweave.contraction.performance.api import ExecutionMeasurement, CostProfile
from lqcdweave.contraction.planning.api import estimate_matrix_action

from .operations.baryon_join import contract_baryon_join
from .operations.contract import contract_dense, bind_dense_action
from .factorized_reduction import validate_factorized_action
from .operations.dense_batch import bind_dense_batch, contract_dense_batch
from .operations.load import load_retained_source, load_source
from .operations.materialize import materialize_source
from .operations.output import copy_output
from .operations.release import release_tensor
from .operations.sparse_bridge import action_bridges as _action_bridges
from .operations.sparse_bridge import contract_sparse_bridge, validate_sparse_bridge_parts
from .operations.transfer import copy_to_backend
from .bound_program import bind_program
from .scalar_group_execution import bind_scalar_groups
from .cartesian_group_execution import bind_cartesian_groups


def _saved_kernel_path_flops(pattern, operand_shapes, path):
    """Count a saved integer path, mirroring planning.state._join_path_cost."""
    labels, output = pattern.inputs, pattern.output
    dimensions = {
        label: size for term, shape in zip(labels, operand_shapes, strict=True)
        for label, size in zip(term, shape, strict=True)
    }
    pending = [set(term) for term in labels]
    flops = 0
    for step in path:
        if (not step or len(set(step)) != len(step)
                or any(type(index) is not int or index < 0 or index >= len(pending)
                       for index in step)):
            raise ValueError("saved contraction kernel path is invalid")
        selected = set().union(*(pending[index] for index in step))
        remaining = [term for index, term in enumerate(pending) if index not in step]
        needed = set(output).union(*remaining)
        retained = selected & needed
        factor = max(1, len(step) - 1) + bool(selected - retained)
        flops += factor * math.prod(dimensions[label] for label in selected)
        pending = remaining + [retained]
    if len(pending) != 1 or pending[0] != set(output):
        raise ValueError("saved contraction kernel path is incomplete")
    return flops


def _nbytes(spec):
    return math.prod(spec.shape) * spec.dtype.itemsize


def _batch_features(state, batch, *, first):
    if not first:
        return dict(kernel_calls=0, flops=0, traffic_bytes=0,
                    dtype=state.catalog[batch.results[0].tensor].dtype.name,
                    operation_domain="cartesian_batch_member")
    if isinstance(batch, DenseGemmGroup):
        children = [_batch_features(state, child, first=True) for child in batch.batches]
        return dict(kernel_calls=2 + sum(child['kernel_calls']-1 for child in children),
            kernel_calls_kind='estimate',
            flops=sum(child['flops'] for child in children),
            traffic_bytes=sum(child['traffic_bytes'] for child in children)+48*len(children),
            dtype=batch.packing_dtype, operation_domain='independent_gemm_group',
            packing_bytes=batch.packing_bytes, logical_members=len(batch.action_indices),
            calibration_eligible=False, cost_status=batch.cost_status)
    operand = batch.left[0]
    spec = state.catalog[operand.tensor]
    physical_free_rank = batch.left_free_rank - int(batch.stacked_sides[0])
    reduction = math.prod(spec.shape[i] for i in operand.permutation[physical_free_rank:])
    dtype = state.catalog[batch.results[0].tensor].dtype
    packed = batch.packing_dtype is not None
    return dict(kernel_calls=1 + sum(not v for v in batch.prepared_sides) if packed
                else 1 + int(len(batch.left)>1) + int(len(batch.right)>1),
        kernel_calls_kind='estimate',
        flops=batch.output_bytes//dtype.itemsize * reduction,
        traffic_bytes=sum(_nbytes(state.catalog[o.tensor]) for o in (*batch.left,*batch.right))
                      + batch.output_bytes + 2*batch.packing_bytes,
        dtype=dtype.name,operation_domain="cartesian_batch",packing_bytes=batch.packing_bytes,
        logical_members=len(batch.action_indices),calibration_eligible=False)


def _scalar_group_features(state, group, *, first):
    action = state.actions[group.action_indices[0]]
    spec = state.catalog[action.operands[0]]
    count = len(group.action_indices)
    return dict(kernel_calls=2 if first else 0,
        flops=count * math.prod(spec.shape) if first else 0,
        traffic_bytes=(sum(_nbytes(state.catalog[key]) for index in group.action_indices
                           for key in state.actions[index].operands)
                       + count*spec.dtype.itemsize + 2*group.workspace_bytes) if first else 0,
        dtype=spec.dtype.name, logical_members=count,
        operation_domain='scalar_reduction_group' if first else 'scalar_reduction_group_member',
        scalar_reduction_workspace_bytes=group.workspace_bytes if first else 0,
        calibration_eligible=False)


def _cartesian_group_features(state, bound, *, first):
    if not first:
        return dict(kernel_calls=0, reduction_kernel_calls=0, flops=0, traffic_bytes=0,
                    dtype=bound.layouts[0].dtype.name, calibration_eligible=False,
                    operation_domain='cartesian_reduction_group_member')
    children = [_batch_features(state, batch, first=True) for batch in bound.batches]
    covered, packers = set(), []
    for axis, combined in bound.packing:
        covered.update((child, axis) for child in combined.members)
        packers.append(combined.pack_original)
    for child, layout in enumerate(bound.layouts):
        for axis, side in enumerate(layout.sides):
            if not side.prepared and (child, axis) not in covered:
                packers.append(side.original_packer if side.original_packer is not None else side.packer)
    counts = tuple(getattr(packer, 'c_contiguous_kernel_calls_upper_bound', None) for packer in packers)
    known = all(count is not None for count in counts)
    packing = sum(1 if count is None else count for count in counts)
    result = dict(kernel_calls=2+packing, reduction_kernel_calls=2,
        kernel_calls_kind='upper_bound_including_packing' if known else 'estimate_including_packing',
        flops=sum(child['flops'] for child in children),
        traffic_bytes=sum(child['traffic_bytes'] for child in children)+2*bound.kernel.partial_bytes,
        dtype=bound.layouts[0].dtype.name, logical_members=sum(len(b.action_indices) for b in bound.batches),
        cartesian_batches=len(bound.batches), packing_bytes=sum(b.packing_bytes for b in bound.batches),
        cartesian_reduction_workspace_bytes=bound.kernel.partial_bytes,
        operation_domain='cartesian_reduction_group', calibration_eligible=False)
    result['packing_kernel_calls_upper_bound' if known else 'packing_kernel_calls_estimate'] = packing
    return result


def _has_coordinate_batch_sources(problem):
    return any(
        isinstance(source.metadata, Mapping)
        and source.metadata.get("family") == "p2p-coordinate-batch"
        for source in problem.sources.values()
    )


def _coordinate_batch_sparse_workspace_bytes(state, action):
    """Return X's known host workspace for one marked NumPy sparse action.

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

def _model_features(
    state: StateDescriptor,
    action: Action,
    *,
    first_region: bool | None = None,
    source_descriptions=None,
):
    """Record exactly the structural predictors used by CostProfile."""
    sources = state.problem.sources if source_descriptions is None else source_descriptions
    if action.kind == "load":
        source = sources[action.tensor]
        return {
            "first_region": bool(first_region),
            "read_kind": "p2p"
            if isinstance(source.metadata, Mapping) and source.metadata.get("family") == "p2p"
            else "array",
            "read_bytes": (source.region_bytes if source.read_bytes is None else source.read_bytes)
            if first_region
            else 0,
            "transfer_bytes": source.transfer_bytes if first_region else 0,
            "snapshot_bytes": (source.snapshot_bytes if source.snapshot_bytes is not None else _nbytes(state.catalog[action.tensor])),
        }
    if action.kind not in ("contract", "sparse_bridge", "baryon_join"):
        return None
    from lqcdweave.contracts.correlator.meson_core import MesonWholeKernel
    if type(action.core) is MesonWholeKernel:
        from ..performance.api import estimate_meson_whole
        flops, traffic, _ = estimate_meson_whole(action.core, state.catalog,
                                                state.catalog[action.tensor].shape)
        # Fixed numerical stages, not a measured CUDA launch count: packing
        # and multi-tile wide reductions can issue additional launches.
        stages = (2 if action.core.kind == 'loop' else
                  4 if action.core.representation == 'unit-spin'
                       and state.catalog[action.operands[0]].dtype.name == 'complex64' else 5)
        return dict(kernel_calls=0 if action.reused else stages, kernel_calls_kind='estimate',
            dtype=state.catalog[action.operands[0]].dtype.name,
            flops=flops, traffic_bytes=traffic,
            operation_domain='whole_meson', core_materialized=False)
    if type(action.core) is FactorizedWideReduction:
        gemm=action.core.gram_strategy=='gemm'
        return dict(kernel_calls=0 if action.reused else action.core.kernel_calls if gemm else 2,
            dtype='complex128' if gemm else action.core.dtype,
            flops=action.core.flops, traffic_bytes=action.core.traffic_bytes,
            operation_domain='factorized_wide_gemm' if gemm else 'factorized_wide_reduction',
            factorized_partial_bytes=action.core.partial_bytes, final_cast_only=True)
    if isinstance(action.core,MatrixKernelSpec):
        estimate = estimate_matrix_action(action.core,profile=CostProfile())
        return {'kernel_calls':0 if action.reused else sum(s.kernel_calls for s in estimate.stages),
                'dtype':action.core.dtype.name,
                'flops':sum(s.flops for s in estimate.stages),
                'traffic_bytes':sum(s.traffic_bytes for s in estimate.stages)}
    kernel_calls = 1
    if action.kind == "baryon_join":
        if action.core is not None:
            core = action.core
            core_labels, *legs_labels = action.equation.inputs
            legs = action.operands[len(core.inputs):]
            flops = traffic = calls = 0
            elements = math.prod(state.catalog[action.tensor].shape)
            size = _nbytes(state.catalog[action.tensor])
            for index, component in enumerate(core.components):
                specs = [state.catalog[key] for key in component.inputs + legs]
                from lqcdweave.contracts.correlator.cores import baryon_component_pattern
                component_pattern = baryon_component_pattern(
                    core_labels, legs_labels, action.equation.output, len(specs[1].shape))
                if action.kernel_description is not None:
                    from lqcdweave.contraction.performance.api import estimate_vertex_kernel
                    flops += estimate_vertex_kernel(action.kernel_description, component_pattern, specs)[0]
                else:
                    flops += _saved_kernel_path_flops(
                        component_pattern, tuple(s.shape for s in specs), action.kernel_path,
                    )
                part_dtype = np.result_type(*(s.dtype for s in specs))
                part_size = elements * part_dtype.itemsize
                traffic += sum(_nbytes(s) for s in specs) + 2 * part_size + size
                calls += 2
                if component.coefficient != 1:
                    flops += elements
                    traffic += 2 * size
                    calls += 1
                if index:
                    flops += elements
                    traffic += 3 * size
                    calls += 1
            return dict(kernel_calls=0 if action.reused else calls, dtype=state.catalog[action.tensor].dtype.name,
                        flops=flops, traffic_bytes=traffic, core_components=len(core.components),
                        core_materialized=False)
        specs = [state.catalog[key] for key in action.operands]
        if action.kernel_description is not None:
            from lqcdweave.contraction.performance.api import estimate_vertex_kernel
            flops = estimate_vertex_kernel(action.kernel_description, action.equation, specs)[0]
        else:
            flops = _saved_kernel_path_flops(
                action.equation, tuple(spec.shape for spec in specs), action.kernel_path,
            )
        traffic = sum(_nbytes(spec) for spec in specs) + _nbytes(state.catalog[action.tensor])
    else:
        selected = action.operands[:2]
        labels, output = action.equation.inputs, action.equation.output
        dimensions = {}
        for key, term_labels in zip(selected, labels, strict=True):
            for label, size in zip(term_labels, state.catalog[key].shape, strict=True):
                dimensions.setdefault(label, size)
        shape = state.catalog[action.tensor].shape
        input_axes = set().union(*labels)
        reduction = math.prod(dimensions[label] for label in input_axes - set(output))
        flops = math.prod(shape) * max(reduction, 1) * max(len(selected) - 1, 1)
        traffic = sum(_nbytes(state.catalog[key]) for key in selected) + _nbytes(state.catalog[action.tensor])
        if action.kind == "sparse_bridge":
            bridge, cycle_bridge = _action_bridges(state, action)
            left_spec = state.catalog[bridge.left]
            right_spec = state.catalog[bridge.right]
            endpoint_work = (
                left_spec.shape[bridge.left_axes[0]]
                * right_spec.shape[bridge.right_axes[0]]
                * left_spec.shape[bridge.left_axes[2]]
                * right_spec.shape[bridge.right_axes[2]]
            )
            if cycle_bridge is None and action.join_id == "bounded_coo_join":
                pair_bound = bridge.nnz_upper * bridge.part_count
                kernel_calls = pair_bound
                flops = pair_bound * 2 * endpoint_work
                traffic += (
                    sources[bridge.record].region_bytes
                    + pair_bound * _nbytes(state.catalog[action.tensor])
                )
                features = {
                    "sparse_operation": action.join_id,
                    "cost_status": JOIN_COST_STATUS,
                    "join_index_block_size": 1,
                    "estimated_join_workspace_bytes": 2 * _nbytes(state.catalog[action.tensor]),
                    "pair_work_upper_bound": pair_bound,
                }
            elif cycle_bridge is None:
                _, raw_labels, projected_output = bridge_intermediate_labels(
                    action.equation,
                    bridge.left_axes,
                    bridge.right_axes,
                    same_point=bridge.same_point,
                )
                raw_elements = math.prod(dimensions[label] for label in raw_labels)
                projected = raw_labels != projected_output
                kernel_calls = bridge.part_count + int(projected)
                flops = max(bridge.nnz_upper, 1) * raw_elements * endpoint_work
                if projected:
                    flops += raw_elements
                    traffic += 2 * raw_elements * state.catalog[action.tensor].dtype.itemsize
                traffic += sources[bridge.record].region_bytes
            elif action.join_id == "bounded_coo_join":
                first_external, final_external, _ = cycle_external_labels(
                    action.equation, bridge, cycle_bridge,
                )
                final_elements = math.prod(dimensions[label] for label in final_external)
                second_left = state.catalog[cycle_bridge.left]
                second_right = state.catalog[cycle_bridge.right]
                second_endpoint_work = (
                    second_left.shape[cycle_bridge.left_axes[0]]
                    * second_right.shape[cycle_bridge.right_axes[0]]
                    * second_left.shape[cycle_bridge.left_axes[2]]
                    * second_right.shape[cycle_bridge.right_axes[2]]
                )
                pair_bound = (
                    bridge.nnz_upper * cycle_bridge.nnz_upper
                    * bridge.part_count * cycle_bridge.part_count
                )
                join_index_bytes = (
                    min(cycle_bridge.nnz_upper, DEFAULT_JOIN_INDEX_BLOCK_SIZE)
                    * ESTIMATED_JOIN_INDEX_BYTES_PER_ROW
                )
                kernel_calls = pair_bound
                flops = pair_bound * 2 * (endpoint_work + second_endpoint_work)
                traffic += (
                    sources[bridge.record].region_bytes
                    + sources[cycle_bridge.record].region_bytes
                    + join_index_bytes
                    + pair_bound * _nbytes(state.catalog[action.tensor])
                )
                features = {
                    "sparse_operation": action.join_id,
                    "cost_status": JOIN_COST_STATUS,
                    "join_index_block_size": DEFAULT_JOIN_INDEX_BLOCK_SIZE,
                    "estimated_join_workspace_bytes": join_index_bytes,
                    "pair_work_upper_bound": pair_bound,
                }
            else:
                first_external, final_external, projected_output = cycle_external_labels(
                    action.equation, bridge, cycle_bridge
                )
                raw_elements = math.prod(dimensions[label] for label in first_external)
                final_elements = math.prod(dimensions[label] for label in final_external)
                projected = final_external != projected_output
                second_left = state.catalog[cycle_bridge.left]
                second_right = state.catalog[cycle_bridge.right]
                second_endpoint_work = (
                    second_left.shape[cycle_bridge.left_axes[0]]
                    * second_right.shape[cycle_bridge.right_axes[0]]
                    * second_left.shape[cycle_bridge.left_axes[2]]
                    * second_right.shape[cycle_bridge.right_axes[2]]
                )
                flops = (
                    max(bridge.nnz_upper, 1) * raw_elements * endpoint_work
                    + max(cycle_bridge.nnz_upper, 1) * final_elements * second_endpoint_work
                    + (final_elements if projected else 0)
                )
                traffic += (
                    2 * raw_elements * state.catalog[action.tensor].dtype.itemsize
                    + sources[bridge.record].region_bytes
                    + sources[cycle_bridge.record].region_bytes
                )
                kernel_calls = bridge.part_count + cycle_bridge.part_count + int(projected)
                for key, input_labels in zip(selected, labels, strict=True):
                    repeated = len(input_labels) - len(set(input_labels))
                    if repeated:
                        flops += repeated * math.prod(state.catalog[key].shape)
                        traffic += repeated * 3 * _nbytes(state.catalog[key])
                        kernel_calls += repeated * 2
                features = {"sparse_operation": "sparse_cycle_legacy"}
    result = {
        "kernel_calls": 0 if action.reused else kernel_calls,
        "dtype": state.catalog[action.tensor].dtype.name,
        "flops": flops,
        "traffic_bytes": traffic,
    }
    if action.kind == "sparse_bridge":
        result.update(features if "features" in locals() else {})
        batch_workspace = _coordinate_batch_sparse_workspace_bytes(state, action)
        if batch_workspace is not None:
            result["coordinate_batch_known_workspace_host_bytes"] = batch_workspace
            first_bridge, cycle_bridge = _action_bridges(state, action)
            if cycle_bridge is not None:
                first_metadata = state.problem.sources[first_bridge.record].metadata
                second_metadata = state.problem.sources[cycle_bridge.record].metadata
                pair_upper = sum(
                    first_metadata["part_count"] * first_bound
                    * second_metadata["part_count"] * second_bound
                    for first_bound, second_bound in zip(
                        first_metadata["lane_nnz_upper"],
                        second_metadata["lane_nnz_upper"],
                        strict=True,
                    )
                )
                pair_flushes = math.ceil(pair_upper / 64)
                result.pop("join_index_block_size", None)
                result.pop("estimated_join_workspace_bytes", None)
                result.pop("pair_work_upper_bound", None)
                result.update({
                    "sparse_operation": "coordinate_batch_lane_pair_join",
                    "cost_status": JOIN_COST_STATUS,
                    "coordinate_batch_join_chunk_rows": 64,
                    "coordinate_batch_pair_work_upper_bound": pair_upper,
                    "coordinate_batch_numeric_flush_upper_bound": pair_flushes,
                })
                result["kernel_calls"] = pair_flushes
    return result



@dataclass(frozen=True)
class BoundSceneExecution:
    starts: dict
    members: dict
    layouts: dict
    kernels: dict
    program: object = None
    scalar_starts: dict = field(default_factory=dict)
    scalar_members: dict = field(default_factory=dict)
    scalar_groups: tuple = ()
    cartesian_starts: dict = field(default_factory=dict)
    cartesian_members: dict = field(default_factory=dict)
    cartesian_groups: tuple = ()


def _native_batch_description(batch):
    """Prove the fields used by a native batch key cannot change in place."""
    if type(batch) is DenseGemmGroup:
        return (type(batch.seconds) in (int,float) and type(batch.cost_status) is str
                and all(_native_batch_description(child) for child in batch.batches))
    if type(batch) is not DenseKernelBatch:
        return False
    return (type(batch.seconds) in (int,float)
            and all(type(operand) is DenseBatchOperand and type(operand.tensor) is str
                    for side in (batch.left,batch.right) for operand in side)
            and all(type(result) is DenseBatchResult and type(result.tensor) is str
                    for result in batch.results))


def _bind_batch_kernel(batch,catalog,backend,kernel_cache,identities):
    entry=None if identities is None else identities.get(id(batch))
    if entry is not None and entry[1] is not None:
        return entry[1]
    key=('batch',batch)
    kernel=kernel_cache.get(key)
    if kernel is None:
        kernel=kernel_cache[key]=bind_dense_batch(batch,catalog,backend)
    if identities is not None and entry is None:
        identities[id(batch)]=(batch,kernel if _native_batch_description(batch) else None)
    return kernel


def bind_scene_execution(state, kernel_batches, backend, *, events=None, retained=(),
                         shared_residency=False, kernel_cache=None, _batch_kernels=None, _action_indices=None,
                         tensor_release_indices=(), producer_batches=None, scalar_reduction_groups=(),
                         cartesian_reduction_groups=()):
    """Prepare fixed kernel dispatch and layout tables once per saved variant."""
    starts = {b.action_indices[0]: b for b in kernel_batches}
    members = {i: b for b in kernel_batches for i in b.action_indices}
    # A cfg instance may omit a partially reused batch. Its producers still
    # honor the template layout recorded in their shared-owner identities.
    layouts = {side[0].tensor: side[0].permutation for b in dense_batch_leaves(
                   kernel_batches if producer_batches is None else producer_batches)
               for side, prepared in zip((b.left, b.right), getattr(b, 'prepared_sides', (False, False)))
               if prepared}
    # The cfg owner supplies this cache only after matching the complete
    # immutable template and backend. Reuse/event masks change the program,
    # not a numerical program's equation/specs or a saved batch's layout.
    # Plain dense programs contain no input/result names. Equal ordered axes,
    # shapes, dtypes and producer layout can therefore share the callable;
    # each Action still binds its own current array keys in bind_program.
    # Native factorized-wide cores also contain no Action names. They share
    # only by the complete immutable core after each Action's contract check.
    # Other cores keep per-Action bindings; batches keep their full records.
    # No numerical arrays enter this template-local cache.
    if kernel_cache is None:
        kernel_cache = {}
    if debug_enabled():
        _batch_kernels = None
        _action_indices = None
    kernels = {}
    active_actions = ([] if shared_residency and not retained and not debug_enabled()
                      and type(events) is tuple
                      and all(type(event) is StorageEvent and type(event.action_index) is int
                              for event in events) else None)
    indexed_actions = (enumerate(state.actions) if _action_indices is None else
                       ((int(i), state.actions[i]) for i in _action_indices))
    for i, action in indexed_actions:
        if active_actions is not None:
            if type(action) is not Action:
                active_actions = None
            elif not action.reused and (action.kind != 'contract' or i not in members or i in starts):
                active_actions.append(i)
        if action.kind != 'contract' or action.reused or i in members:
            continue
        layout = layouts.get(action.tensor)
        if (type(action) is Action and type(action.core) is FactorizedWideReduction
                and not debug_enabled()):
            # A hit must not bypass the current Action's input/output/equation
            # contract. The callable uses current values in the declared order.
            validate_factorized_action(action, state.catalog)
            if layout is not None:
                raise ValueError('thin wide outputs cannot borrow a batched producer layout')
            from .binding_cache import factorized_gram_tiling_enabled
            key = ('factorized', action.core, id(backend), factorized_gram_tiling_enabled())
        else:
            key = (('dense', action.equation, tuple(
                (state.catalog[operand].shape, state.catalog[operand].dtype)
                for operand in action.operands), layout, action.wide_accumulation) if action.core is None else
                ('action', i, layout))
        kernel = kernel_cache.get(key)
        if kernel is None:
            kernel = kernel_cache[key] = bind_dense_action(action, state.catalog, backend, layout)
        kernels[i] = kernel
    for index, batch in starts.items():
        kernels[index] = _bind_batch_kernel(batch,state.catalog,backend,kernel_cache,_batch_kernels)
    scalar_starts, scalar_members = bind_scalar_groups(state, scalar_reduction_groups, backend,
                                                      kernel_cache, members)
    cartesian_starts, cartesian_members = bind_cartesian_groups(state, cartesian_reduction_groups,
        backend, kernel_cache, starts, kernels, scalar_members)
    program = (None if events is None else bind_program(
        state, backend, starts, members, layouts, kernels, events, retained,
        (contract_dense, contract_dense_batch, contract_sparse_bridge, contract_baryon_join),
        shared_residency=shared_residency, _active_actions=active_actions,
        tensor_release_indices=tensor_release_indices, scalar_starts=scalar_starts,
        scalar_members=scalar_members, cartesian_starts=cartesian_starts,
        cartesian_members=cartesian_members))
    return BoundSceneExecution(starts, members, layouts, kernels, program,
                               scalar_starts, scalar_members,
                               tuple(bound.record for bound in scalar_starts.values()),
                               cartesian_starts, cartesian_members,
                               tuple(bound.record for bound in cartesian_starts.values()))


def replay(state: StateDescriptor, loader: SourceLoader, *, backend=np, measure: bool = False,
           source_execution=None, shared_arrays=None, _owner_snapshot=False, execution_cache=None):
    from .packed_values import packed_values_scope
    with packed_values_scope(source_execution, backend):
        return _replay_dispatch(state, loader, backend=backend, measure=measure,
            source_execution=source_execution, shared_arrays=shared_arrays,
            _owner_snapshot=_owner_snapshot, execution_cache=execution_cache)


def _replay_dispatch(state: StateDescriptor, loader: SourceLoader, *, backend=np, measure: bool = False,
           source_execution=None, shared_arrays=None, _owner_snapshot=False, execution_cache=None):
    """Replay; measure=True enables phase timing, hardware and memory sampling.

    With measurement disabled no profiling thread or device query is created.
    The enabled path synchronizes measured phases and reports synchronous
    calibration timings; it changes the overlap of naturally asynchronous work.
    """
    if execution_cache is not None:
        with execution_cache.scope(state, backend):
            return replay(state, loader, backend=backend, measure=measure,
                source_execution=source_execution, shared_arrays=shared_arrays,
                _owner_snapshot=_owner_snapshot)
    if getattr(backend, '__name__', None) not in {'numpy', 'cupy'}:
        raise NotImplementedError('replay requires a NumPy or CuPy backend')
    from lqcdweave.contracts.correlator.precision import require_uniform_contraction_dtype
    require_uniform_contraction_dtype(state.problem.specs)
    if source_execution is None and _has_coordinate_batch_sources(state.problem):
        raise NotImplementedError(
            "coordinate-batched sparse execution requires an audited SceneStoragePlan"
        )
    if source_execution is None:
        validate_executable_resources(state)
    elif debug_enabled():
        source_execution.validate_state(state)
    if not measure and not debug_enabled() and source_execution is not None and shared_arrays is None:
        _prepare_reduction_workspace(source_execution)
    if not measure:
        if source_execution is not None and not debug_enabled():
            tables = getattr(source_execution, 'execution_tables', None)
            program = None if tables is None else tables.program
            if program is not None:
                arrays = {} if shared_arrays is None else shared_arrays
                return program.run(source_execution, arrays, owner_snapshot=_owner_snapshot)
        return _replay(state, loader, backend=backend, source_execution=source_execution,
                       shared_arrays=shared_arrays)
    with ExecutionMeasurement(backend) as measurement:
        report = _replay(state, loader, backend=backend, measure=True,
                         source_execution=source_execution, measurement=measurement,
                         shared_arrays=shared_arrays)
    return replace(report, execution_profile=measurement.result)


def _prepare_reduction_workspace(execution):
    """Admit ordinary scene scratch before its first queued read or Action."""
    from .device import active_device_execution
    from .output_queue import active_output_queue
    device = active_device_execution()
    reserve_workspace = None if device is None else getattr(device, 'reserve_reduction_workspace', None)
    if not callable(reserve_workspace):
        return
    plan = getattr(execution, 'plan', None)
    if (plan is None or not getattr(execution, 'owns_read_ahead', False)
            or not getattr(execution, 'owns_storage', False)
            or getattr(execution, 'read_ahead_started', True)
            or plan.retained_before or plan.retained_after or plan.explicit_resources):
        return
    output = active_output_queue()
    output_reserve = 0 if output is None else output.reserve
    reserve = reserve_workspace(plan.peak_device_bytes,
        limit=plan.memory_limit, output_reserve=output_reserve)
    queue = execution.read_ahead
    if queue is not None:
        available = max(0, min(device.limit, plan.memory_limit)-plan.peak_device_bytes
                        -device.constant_reserve-output_reserve-reserve)
        queue.device_capacity = min(queue.device_capacity, available)


def _replay(state: StateDescriptor, loader: SourceLoader, *, backend=np, measure: bool = False,
            source_execution=None, measurement=None, shared_arrays=None):
    """Replay one cfg with a separately timed record for every I/O and kernel.

    A load_region/view loader may make its actual file read in either method.
    Both calls and the following host/device snapshot are timed separately.
    CuPy phases synchronize before their wall-clock interval ends; kernel and
    transfer phases also record a CUDA event duration. Timings are synchronous
    calibration observations, not natural asynchronous throughput.
    """
    if not state.done:
        raise ValueError("cannot replay an incomplete state")
    if measure and measurement is None:
        raise ValueError("measured replay requires an ExecutionMeasurement collector")
    arrays: dict[str, Any] = {} if shared_arrays is None else shared_arrays
    kernel_batches = () if source_execution is None else source_execution.plan.kernel_batches
    tables = None if source_execution is None else getattr(source_execution, 'execution_tables', None)
    scalar_groups = (() if source_execution is None else
                     getattr(source_execution.plan, 'scalar_reduction_groups', ()))
    cartesian_groups = (() if source_execution is None else
                        getattr(source_execution.plan, 'cartesian_reduction_groups', ()))
    if tables is not None and debug_enabled() and (tables.scalar_groups or tables.cartesian_groups):
        tables = None  # Debug binds the ordinary per-member reference route.
    if tables is None:
        # An external source execution can supply different batches for the
        # same state. Only bare replay has the complete key in state alone.
        tables = (bind_scene_execution(state, kernel_batches, backend,
                    scalar_reduction_groups=scalar_groups,
                    cartesian_reduction_groups=cartesian_groups) if source_execution is not None else
            bind_cached('scene', state, backend,
            lambda: bind_scene_execution(state, kernel_batches, backend),
            lambda: native_state_binding(state)))
    batch_starts, member_batches = tables.starts, tables.members
    producer_layouts, kernels = tables.layouts, tables.kernels
    batch_members = member_batches
    regions: dict[str, Any] = {}
    opened_regions: set[str] = set()
    timings = []
    output_timings = []
    io_events = []
    device_peak = None
    host_peak = None
    is_gpu = getattr(backend, "__name__", "") == "cupy"
    process = None
    if measure:
        if is_gpu:
            backend.cuda.get_current_stream().synchronize()
        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            pass
    phases: list[PhaseTiming] = []
    current_action_index: int | None = None

    def timed(name: str, fn, *, nbytes: int | None = None, device: bool = False):
        if not measure:
            return fn()
        value, phase = measurement.measure_phase(
            name,
            fn,
            nbytes=nbytes,
            device=device,
            parent_action_index=current_action_index,
        )
        phases.append(phase)
        return value

    for action_index, action in enumerate(state.actions):
        current_action_index = action_index
        phases = []
        first_region = None
        if source_execution is not None:
            source_execution.begin_action(action_index, timed)
        if action.reused and shared_arrays is not None and action.tensor in arrays:
            pass
        elif action.reused and source_execution is not None and source_execution.has_retained(action.tensor):
            load_retained_source(source_execution, action.tensor, arrays, timed=timed)
        elif action.kind == "load":
            if action.reused:
                raise RuntimeError('retained source missing from execution cache')
            source = (state.problem.sources[action.tensor] if source_execution is None
                      else source_execution.source(action.tensor))
            first_region = source.region not in regions
            if first_region and state.problem.single_read_regions and source.region in opened_regions:
                raise RuntimeError("archived plan reads a backing region more than once")
            if source_execution is not None:
                value = materialize_source(
                    source_execution, action.tensor, source, timed=timed, device=is_gpu,
                )
            else:
                value = load_source(loader, action.tensor, source, regions, timed=timed)
            if first_region:
                opened_regions.add(source.region)
            if action.tensor in state.catalog:
                spec = state.catalog[action.tensor]
                if tuple(value.shape) != spec.shape or np.dtype(value.dtype) != spec.dtype:
                    raise ValueError(f"loader returned a mismatched tensor for {action.tensor}")
                if source_execution is not None and source_execution.materializes_owned(action.tensor):
                    arrays[action.tensor] = timed('snapshot', lambda: value, nbytes=0, device=is_gpu)
                else:
                    arrays[action.tensor] = copy_to_backend(
                        value, backend, phase="snapshot", timed=timed, device=is_gpu,
                    )
            else:
                value = validate_sparse_bridge_parts(state.problem, action.tensor, value)
                arrays[action.tensor] = value
        elif action.kind == "contract":
            if action_index in tables.cartesian_starts:
                tables.cartesian_starts[action_index].execute(arrays, timed=timed, device=is_gpu)
            elif action_index in tables.cartesian_members:
                if action.tensor not in arrays:
                    raise RuntimeError('saved Cartesian reduction group did not produce its member result')
            elif action_index in tables.scalar_starts:
                tables.scalar_starts[action_index].execute(arrays, timed=timed, device=is_gpu)
            elif action_index in tables.scalar_members:
                if action.tensor not in arrays:
                    raise RuntimeError('saved scalar group did not produce its member result')
            elif action_index in batch_starts:
                batch = batch_starts[action_index]
                contract_dense_batch(batch, arrays, backend=backend, timed=timed, device=is_gpu,
                                     bound_kernel=kernels[action_index])
            elif action_index in batch_members:
                if action.tensor not in arrays:
                    raise RuntimeError("saved batch did not produce its member result")
            else:
                contract_dense(action, arrays, backend, timed=timed, device=is_gpu,
                               output_layout=producer_layouts.get(action.tensor),
                               bound_kernel=kernels.get(action_index))
        elif action.kind == "sparse_bridge":
            contract_sparse_bridge(state, action, arrays, backend, timed=timed, device=is_gpu)
        elif action.kind == "baryon_join":
            contract_baryon_join(action, arrays, backend, timed=timed, device=is_gpu)
        elif action.kind in ("read_block", "close_block"):
            if source_execution is None:
                raise NotImplementedError("raw lifecycle actions require an audited storage execution")
            # The matching StorageEvent was applied by begin_action/end_action.
            pass
        elif action.kind in ("spill_disk", "restore_host", "discard"):
            if source_execution is None:
                raise NotImplementedError("resource actions require an audited storage execution")
            source_execution.move_tensor(action, arrays, timed)
        elif action.kind == "release":
            sources = state.problem.sources if source_execution is None else source_execution.sources
            release_tensor(action, arrays, regions, sources, timed=timed)
        else:
            raise ValueError(f"unsupported archived action kind {action.kind!r}")
        if source_execution is not None and action.kind in ('load','contract','sparse_bridge','baryon_join'):
            if source_execution.retains(action.tensor):
                if action.kind=='sparse_bridge' and not action.reused:
                    arrays[action.tensor] = copy_to_backend(
                        arrays[action.tensor], backend, phase="retention_snapshot",
                        timed=timed, device=is_gpu,
                    )
                source_execution.retain_tensor(action.tensor,arrays[action.tensor])
        # Do not let loop-local temporaries keep released backing arrays alive.
        value = None
        if source_execution is not None:
            source_execution.end_action(action_index, timed)
        if measure:
            if hasattr(loader, "drain_io_events"):
                for event in loader.drain_io_events():
                    io_events.append(
                        FileIOEvent(
                            action_index,
                            action.tensor,
                            str(event["path"]),
                            str(event["operation"]),
                            int(event["bytes"]),
                            float(event["observed_seconds"]),
                        )
                    )
            timings.append(
                ActionTiming(
                    action.kind,
                    action.seconds if source_execution is None else source_execution.action_seconds(action_index),
                    sum(phase.observed_seconds for phase in phases),
                    action_index,
                    action.tensor,
                    tuple(phases),
                    action.operands,
                    ((state.problem.sources[action.tensor] if source_execution is None
                      else source_execution.source(action.tensor)).region
                     if action.kind == "load" else None),
                    _cartesian_group_features(state, tables.cartesian_members[action_index],
                                              first=action_index in tables.cartesian_starts)
                    if action_index in tables.cartesian_members else
                    _scalar_group_features(state, tables.scalar_members[action_index],
                                           first=action_index in tables.scalar_starts)
                    if action_index in tables.scalar_members else
                    _batch_features(state, member_batches[action_index], first=action_index in batch_starts)
                    if action_index in batch_members else _model_features(
                        state, action, first_region=first_region,
                        source_descriptions=(None if source_execution is None
                                             else source_execution.sources),
                    )
                    if source_execution is None or action.kind != "load"
                    else source_execution.model_features(action.tensor),
                )
            )
            if is_gpu:
                allocated = int(backend.get_default_memory_pool().total_bytes())
                device_peak = max(device_peak or 0, allocated)
            if process is not None:
                host_peak = max(host_peak or 0, int(process.memory_info().rss))
    values = []
    for index, key in enumerate(state.results):
        current_action_index = len(state.actions) + index
        phases = []
        values.append(copy_output(key, arrays, backend, timed=timed, device=is_gpu))
        if measure:
            output_timings.append(
                ActionTiming(
                    "output_copy",
                    0.0,
                    sum(phase.observed_seconds for phase in phases),
                    len(state.actions) + index,
                    key,
                    tuple(phases),
                )
            )
            if is_gpu:
                device_peak = max(device_peak or 0, int(backend.get_default_memory_pool().total_bytes()))
            if process is not None:
                host_peak = max(host_peak or 0, int(process.memory_info().rss))
    if source_execution is not None:
        current_action_index = None
        source_execution.end_action(len(state.actions), timed)
    if measure:
        return ReplayReport(
            tuple(values),
            tuple(timings),
            device_peak,
            host_peak,
            tuple(output_timings),
            tuple(io_events),
        )
    return tuple(values)
