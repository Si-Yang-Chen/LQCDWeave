"""Execute sparse bridge actions using their delivered C records."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action,
    SparseBridgeSpec,
)

from .sparse_bridge_kernel import (
    _require_coordinate_batch_complex128_envelope,
    contract_p2p_bridge,
    contract_p2p_bridge_batch,
    contract_p2p_cycle,
)
from .sparse_bridge_kernel import sparse_pairs


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


def action_bridges(state, action: Action):
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


def contract_sparse_bridge(state, action, arrays, backend, *, timed, device=False):
    """Execute one planned sparse bridge/cycle without path search."""
    if action.reused:
        timed("reuse", lambda: None)
        return
    first_bridge, second_bridge = action_bridges(state, action)
    action_bridges_tuple = tuple(
        bridge for bridge in (first_bridge, second_bridge) if bridge is not None
    )
    marked = tuple(
        isinstance(state.problem.sources[bridge.record].metadata, Mapping)
        and state.problem.sources[bridge.record].metadata.get("family")
        == "p2p-coordinate-batch"
        for bridge in action_bridges_tuple
    )
    if any(marked):
        if not all(marked):
            raise ValueError("coordinate-batched sparse actions cannot mix marked and scalar records")
        if (state.problem.backend != "numpy"
                or getattr(backend, "__name__", None) != "numpy"):
            raise NotImplementedError("coordinate-batched sparse P2P currently supports NumPy only")
        if second_bridge is None:
            kernel = partial(
                contract_p2p_bridge_batch,
                arrays[first_bridge.left],
                arrays[first_bridge.right],
                arrays[first_bridge.record],
                bridge=first_bridge,
                equation=action.equation,
                backend=backend,
            )
        else:
            # Import lazily so ordinary scalar execution keeps the same module
            # initialization path while the batch cycle implementation lands.
            from .sparse_bridge_kernel import contract_p2p_cycle_batch

            kernel = partial(
                contract_p2p_cycle_batch,
                arrays[first_bridge.left],
                arrays[first_bridge.right],
                arrays[first_bridge.record],
                arrays[second_bridge.record],
                first_bridge=first_bridge,
                second_bridge=second_bridge,
                equation=action.equation,
                backend=backend,
            )
    elif second_bridge is None:
        kernel = partial(
            contract_p2p_bridge,
            arrays[first_bridge.left],
            arrays[first_bridge.right],
            arrays[first_bridge.record],
            used_np=action.used_np,
            used_np_right=action.used_np_right,
            backend=backend,
            left_axes=action.left_axes,
            right_axes=action.right_axes,
            same_point=action.same_point,
            transpose_bridge=action.transpose_bridge,
            equation=action.equation,
        )
    else:
        kernel = partial(
            contract_p2p_cycle,
            arrays[first_bridge.left],
            arrays[first_bridge.right],
            arrays[first_bridge.record],
            arrays[second_bridge.record],
            first_bridge=first_bridge,
            second_bridge=second_bridge,
            equation=action.equation,
            backend=backend,
        )
    arrays[action.tensor] = timed("sparse_bridge_kernel", kernel, device=device)


def validate_sparse_bridge_parts(problem, key, value):
    """Check a materialized sparse source against its delivered bridge bounds."""
    bridges = tuple(item for item in problem.sparse_bridges if item.record == key)
    if not bridges:
        return value
    source = problem.sources[key]
    if (isinstance(source.metadata, Mapping)
            and source.metadata.get("family") == "p2p-coordinate-batch"):
        return _validate_coordinate_batch_parts(problem, key, value, bridges, source.metadata)
    if isinstance(value, Mapping):
        raise TypeError(f"sparse source {key!r} must materialize bridge part triples")
    try:
        parts = tuple(value)
    except TypeError as exc:
        raise TypeError(f"sparse source {key!r} must materialize bridge part triples") from exc
    if any(not isinstance(part, (tuple, list)) or len(part) != 3 for part in parts):
        raise ValueError(f"sparse source {key!r} parts must be (spin, coefficient, record) triples")
    for bridge in bridges:
        if len(parts) > bridge.part_count:
            raise ValueError(
                f"sparse source {key!r} has {len(parts)} parts; descriptor allows {bridge.part_count}"
            )
        left = problem.specs[bridge.left]
        right = problem.specs[bridge.right]
        colors = (left.shape[bridge.left_axes[2]], right.shape[bridge.right_axes[2]])
        if colors[0] != colors[1]:
            raise ValueError("sparse bridge color extents disagree with TensorSpec")
        for _, _, record in parts:
            pairs, _ = sparse_pairs(
                record, bridge.used_np, colors[0],
                used_np_right=bridge.used_np_right,
            )
            if len(pairs) > bridge.nnz_upper:
                raise ValueError(
                    f"sparse source {key!r} has {len(pairs)} selected rows; "
                    f"descriptor nnz_upper is {bridge.nnz_upper}"
                )
    return parts


def _validate_coordinate_batch_parts(problem, key, value, bridges, metadata):
    if len(bridges) != 1:
        raise NotImplementedError("coordinate-batched records require exactly one sparse bridge")
    bridge = bridges[0]
    if not isinstance(value, Mapping) or value.get("layout") != "coordinate_batch_coo":
        raise TypeError(f"sparse source {key!r} must materialize the declared lane COO mapping")
    point_keys = tuple(metadata["coordinate_keys"])
    lane_bounds = tuple(metadata["lane_nnz_upper"])
    lane_sources = tuple(metadata["lane_sources"])
    lanes = value.get("lanes")
    if (tuple(value.get("point_keys", ())) != point_keys
            or not isinstance(lanes, (tuple, list))
            or len(lanes) != len(point_keys)
            or len(lane_sources) != len(point_keys)):
        raise ValueError("coordinate sparse materialization differs from its ordered lane metadata")
    if len(lane_bounds) != len(point_keys):
        raise ValueError("coordinate sparse bounds do not match lane metadata")
    def axis_size(spec, axis):
        return spec.shape[axis % len(spec.shape)]

    left, right = problem.specs[bridge.left], problem.specs[bridge.right]
    left_spin = axis_size(left, bridge.left_axes[0])
    right_spin = axis_size(right, bridge.right_axes[0])
    left_color = axis_size(left, bridge.left_axes[2])
    right_color = axis_size(right, bridge.right_axes[2])
    if left_color != right_color:
        raise ValueError("coordinate sparse endpoint color extents disagree")
    expected_point_shape = tuple(metadata["point_shape"])
    part_count = int(metadata["part_count"])
    normalized_lanes = []
    for lane_index, (parts, bound, lane_source) in enumerate(
            zip(lanes, lane_bounds, lane_sources, strict=True)):
        raw_metadata = lane_source.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"coordinate sparse lane {lane_index} has no raw record metadata")
        if raw_metadata.get("backend") != "numpy":
            raise NotImplementedError(
                "coordinate-batched sparse bridge currently supports NumPy records only"
            )
        raw_shape = tuple(raw_metadata["point_shape"])
        materialized_shape = tuple(lane_source.get("materialized_point_shape", ()))
        if materialized_shape != expected_point_shape:
            raise ValueError(
                f"coordinate sparse lane {lane_index} changes materialized point extents"
            )
        raw_nnz = int(raw_metadata["nnz"])
        raw_value_dtype = np.dtype(raw_metadata["value_dtype"])
        raw_index_dtype = np.dtype(raw_metadata["index_dtype"])
        raw_kind = raw_metadata["kind"]
        if not isinstance(parts, (tuple, list)):
            raise TypeError(f"coordinate sparse lane {lane_index} must contain ordered parts")
        if len(parts) > part_count:
            raise ValueError(f"coordinate sparse lane {lane_index} exceeds part_count")
        normalized_parts = []
        lane_nnz = 0
        for part in parts:
            if not isinstance(part, (tuple, list)) or len(part) != 3:
                raise ValueError("coordinate sparse parts must be (gamma, coefficient, record) triples")
            gamma, coefficient, record = part
            gamma_array = np.asarray(gamma)
            gamma_shape = ((right_spin, left_spin) if bridge.transpose_bridge
                           else (left_spin, right_spin))
            if (gamma_array.shape != gamma_shape
                    or gamma_array.dtype.kind not in "biufc"):
                raise ValueError("coordinate sparse gamma shape/dtype differs from endpoint metadata")
            _require_coordinate_batch_complex128_envelope(gamma_array.dtype, "gamma")
            if not np.all(np.isfinite(gamma_array)):
                raise ValueError("coordinate sparse gamma must be finite")
            coefficient_array = np.asarray(coefficient)
            if coefficient_array.shape != () or coefficient_array.dtype.kind not in "biufc":
                raise ValueError("coordinate sparse coefficient must be a numeric scalar")
            _require_coordinate_batch_complex128_envelope(
                coefficient_array.dtype, "coefficient",
            )
            if not bool(np.isfinite(coefficient_array)):
                raise ValueError("coordinate sparse coefficient must be finite")
            if not isinstance(record, Mapping) or record.get("type") != raw_kind:
                raise ValueError("coordinate sparse raw record kind differs from metadata")
            if raw_kind == "identity":
                if set(record) != {"type"} or raw_nnz != min(raw_shape):
                    raise ValueError("coordinate sparse identity record differs from its metadata")
            else:
                if set(record) != {"type", "indices", "values"}:
                    raise ValueError("coordinate sparse COO record has unexpected fields")
                indices = record["indices"]
                values = record["values"]
                if (not isinstance(indices, np.ndarray) or not isinstance(values, np.ndarray)
                        or indices.shape != (raw_nnz, 2)
                        or values.shape != (raw_nnz, left_color, right_color)
                        or np.dtype(indices.dtype) != raw_index_dtype
                        or np.dtype(values.dtype) != raw_value_dtype
                        or int(indices.nbytes + values.nbytes) != int(raw_metadata["record_bytes"])):
                    raise ValueError("coordinate sparse COO buffers differ from their metadata")
                _require_coordinate_batch_complex128_envelope(
                    values.dtype, "raw COO values",
                )
                if raw_nnz:
                    if (np.any(indices[:, 0] < -materialized_shape[0])
                            or np.any(indices[:, 0] >= materialized_shape[0])
                            or np.any(indices[:, 1] < -materialized_shape[1])
                            or np.any(indices[:, 1] >= materialized_shape[1])):
                        raise IndexError("coordinate sparse COO index exceeds its declared point shape")
            pairs, values = sparse_pairs(
                record, bridge.used_np, left_color,
                used_np_right=bridge.used_np_right,
            )
            if len(pairs) > bound:
                raise ValueError(
                    f"coordinate sparse lane {lane_index} part has {len(pairs)} selected COO rows; "
                    f"per-part descriptor bound is {bound}"
                )
            lane_nnz += len(pairs)
            normalized_parts.append((
                gamma_array, coefficient_array.item(),
                {"layout": "normalized-coo", "indices": pairs, "values": values},
            ))
        if lane_nnz > part_count * bound:
            raise ValueError(
                f"coordinate sparse lane {lane_index} has {lane_nnz} selected COO rows across "
                f"{part_count} parts; per-part descriptor bound is {bound}"
            )
        normalized_lanes.append(tuple(normalized_parts))
    total_nnz = sum(len(part[2]["indices"])
                    for lane in normalized_lanes for part in lane)
    if total_nnz > part_count * bridge.nnz_upper:
        raise ValueError("coordinate sparse total COO rows exceed the aggregate bridge bound")
    return {
        "layout": "coordinate_batch_coo",
        "point_keys": point_keys,
        "lanes": tuple(normalized_lanes),
    }


__all__ = ["action_bridges", "contract_sparse_bridge", "validate_sparse_bridge_parts"]
