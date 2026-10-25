"""Independent references for retaining one or both sparse endpoint axes."""

from types import SimpleNamespace
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    parse_axis_pattern,
    SparseBridgeSpec,
    bridge_external_labels,
    bridge_intermediate_labels,
)
from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import (
    contract_p2p_bridge,
    contract_p2p_bridge_batch,
)
from lqcdweave.contraction.execution.scene_runtime import (
    _coordinate_batch_sparse_workspace_bytes,
)


def _rectangular_case(seed=86420, *, left_points=4, right_points=3):
    rng = np.random.default_rng(seed)

    def complex_values(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    left = complex_values((2, 4, left_points, 3))[:, :, ::-1, :]
    right = complex_values((4, right_points, 3, 5))[..., ::-1]
    gamma = complex_values((4, 4))
    pairs = np.asarray(((0, 2), (3, 0), (1, 1), (0, 2)), dtype=np.int64)
    values = complex_values((len(pairs), 3, 3))
    record = {"type": "sparse", "indices": pairs, "values": values}
    return left, right, gamma, record


def _dense_endpoint_reference(
    left,
    right,
    gamma,
    record,
    *,
    output_labels,
    coefficient=0.375 - 0.625j,
    transpose=False,
    same_point=False,
):
    """Directly sum endpoint slices, then reorder explicit retained axes."""
    endpoint_labels = ("p", "q")
    retained = tuple(label for label in endpoint_labels if label in output_labels)
    raw_labels = "ab" + "".join(retained)
    dimension = {
        "a": left.shape[0], "b": right.shape[-1],
        "p": left.shape[2], "q": right.shape[1],
    }
    raw = np.zeros(tuple(dimension[label] for label in raw_labels), dtype=np.complex128)
    for (record_left, record_right), block in zip(
            record["indices"], record["values"], strict=True):
        if (same_point or "p" == "q") and record_left != record_right:
            continue
        left_point, right_point = (
            (record_right, record_left) if transpose
            else (record_left, record_right)
        )
        effective_gamma = gamma.T if transpose else gamma
        effective_block = block.T if transpose else block
        contribution = coefficient * np.einsum(
            "aic,ij,cd,jdb->ab",
            left[:, :, left_point, :], effective_gamma, effective_block,
            right[:, right_point, :, :], optimize=False,
        )
        coordinates = {"a": slice(None), "b": slice(None),
                       "p": left_point, "q": right_point}
        raw[tuple(coordinates[label] for label in raw_labels)] += contribution
    return np.transpose(raw, tuple(raw_labels.index(label) for label in output_labels))


def _normalized_part(indices, values, gamma, coefficient):
    return gamma, coefficient, {
        "layout": "normalized-coo",
        "indices": np.asarray(indices, dtype=np.int64),
        "values": np.asarray(values),
    }


def _coordinate_record(lanes):
    return {
        "layout": "coordinate_batch_coo",
        "point_keys": tuple(f"coordinate-{index}" for index in range(len(lanes))),
        "lanes": tuple(tuple(parts) for parts in lanes),
    }


class SparseEndpointPointRetentionTests(unittest.TestCase):
    def test_contract_helpers_retain_requested_endpoint_labels_without_equating_them(self):
        axes = ((-3, -2, -1), (0, 1, 2))
        self.assertEqual(
            bridge_external_labels("aipc,jqdb->qab", *axes, same_point=False),
            ((97, 98), (113, 97, 98)),
        )
        self.assertEqual(
            bridge_intermediate_labels("aipc,jqdb->qab", *axes, same_point=False),
            ((97, 98), (97, 98, 113), (113, 97, 98)),
        )
        self.assertEqual(
            bridge_intermediate_labels("aipc,jqdb->pqab", *axes, same_point=False),
            ((97, 98), (97, 98, 112, 113), (112, 113, 97, 98)),
        )
        self.assertEqual(
            bridge_intermediate_labels("aipc,jqdb->ab", *axes, same_point=False),
            ((97, 98), (97, 98), (97, 98)),
        )
        with self.assertRaisesRegex(ValueError, "output axis IDs must occur in an input"):
            bridge_external_labels("aipc,jqdb->z", *axes, same_point=False)

    def test_distinct_rectangular_endpoints_retain_left_right_and_both_axes(self):
        left, right, gamma, record = _rectangular_case()
        coefficient = 0.375 - 0.625j
        axes = ((-3, -2, -1), (0, 1, 2))
        for output_labels in ("pab", "qab", "pqab", "aqpb"):
            with self.subTest(output_labels=output_labels):
                expected = _dense_endpoint_reference(
                    left, right, gamma, record,
                    output_labels=output_labels, coefficient=coefficient,
                )
                actual = contract_p2p_bridge(
                    left, right, ((gamma, coefficient, record),),
                    used_np=4, used_np_right=3, left_axes=axes[0],
                    right_axes=axes[1], same_point=False,
                    equation=f"aipc,jqdb->{output_labels}",
                )
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, np.dtype(np.complex128))
                np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def test_transposed_record_maps_endpoint_coordinates_and_nonsymmetric_blocks(self):
        left, right, gamma, record = _rectangular_case(seed=1203)
        # A daggered bridge consumes the record's right endpoint on the
        # equation's left tensor and the record's left endpoint on the right.
        left_dagger = _rectangular_case(seed=1204, left_points=3, right_points=4)[0]
        right_dagger = _rectangular_case(seed=1205, left_points=3, right_points=4)[1]
        expected = _dense_endpoint_reference(
            left_dagger, right_dagger, gamma, record,
            output_labels="pab", transpose=True,
        )
        actual = contract_p2p_bridge(
            left_dagger, right_dagger, ((gamma, 0.375 - 0.625j, record),),
            used_np=4, used_np_right=3, left_axes=(-3, -2, -1),
            right_axes=(0, 1, 2), same_point=False,
            transpose_bridge=True, equation=parse_axis_pattern('aipc,jqdb->pab'),
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def test_diagonal_constraints_and_default_closed_result_are_unchanged(self):
        square_left, square_right, gamma, record = _rectangular_case(
            seed=543, left_points=4, right_points=4,
        )
        diagonal = _dense_endpoint_reference(
            square_left, square_right, gamma, record,
            output_labels="pqab", same_point=True,
        )
        actual_diagonal = contract_p2p_bridge(
            square_left, square_right, ((gamma, 0.375 - 0.625j, record),),
            used_np=4, left_axes=(-3, -2, -1), right_axes=(0, 1, 2),
            same_point=True, equation=parse_axis_pattern('aipc,jqdb->pqab'),
        )
        np.testing.assert_allclose(actual_diagonal, diagonal, rtol=3e-13, atol=3e-13)

        expected_closed = _dense_endpoint_reference(
            square_left, square_right, gamma, record, output_labels="ab",
        )
        actual_closed = contract_p2p_bridge(
            square_left, square_right, ((gamma, 0.375 - 0.625j, record),),
            used_np=4, left_axes=(-3, -2, -1), right_axes=(0, 1, 2),
        )
        np.testing.assert_allclose(actual_closed, expected_closed, rtol=3e-13, atol=3e-13)

    def test_empty_independent_record_returns_zero_on_requested_point_axis(self):
        left, right, gamma, _ = _rectangular_case()
        empty = {"type": "sparse", "indices": np.empty((0, 2), dtype=np.int64),
                 "values": np.empty((0, 3, 3), dtype=np.complex128)}
        actual = contract_p2p_bridge(
            left, right, ((gamma, 1, empty),), used_np=4, used_np_right=3,
            same_point=False, equation=parse_axis_pattern('aipc,jqdb->qab'),
        )
        self.assertEqual(actual.shape, (3, 2, 5))
        np.testing.assert_array_equal(actual, np.zeros_like(actual))

    def test_coordinate_batch_join_retains_independent_endpoint_output(self):
        rng = np.random.default_rng(33018)

        def random(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        left = random((2, 2, 4, 4, 3))
        right = random((2, 4, 3, 3, 5))
        pairs_by_lane = (
            np.asarray(((0, 2), (3, 0), (0, 2)), dtype=np.int64),
            np.asarray(((1, 1), (2, 0), (3, 2)), dtype=np.int64),
        )
        lanes, expected_lanes = [], []
        for lane, pairs in enumerate(pairs_by_lane):
            gamma = random((4, 4))
            values = random((len(pairs), 3, 3))
            coefficient = 0.25 + 0.5j * (lane + 1)
            lanes.append((_normalized_part(pairs, values, gamma, coefficient),))
            record = {"indices": pairs, "values": values}
            expected_lanes.append(_dense_endpoint_reference(
                left[lane], right[lane], gamma, record,
                output_labels="qab", coefficient=coefficient,
            ))
        bridge = SparseBridgeSpec(
            "left", "right", "record", 4, sum(map(len, pairs_by_lane)),
            left_axes=(-3, -2, -1), right_axes=(1, 2, 3),
            same_point=False, used_np_right=3,
        )
        actual = contract_p2p_bridge_batch(
            left, right, _coordinate_record(lanes), bridge=bridge,
            equation=parse_axis_pattern('xaipc,xjqdb->xqab'),
        )
        expected = np.stack(expected_lanes)
        np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13)

    def test_coordinate_batch_workspace_uses_p_owned_result_and_no_raw_projection(self):
        # The retained-output batch route calls the generic join, which allocates
        # the final axis order directly.  P owns that result; X must not add a
        # second raw-label projection buffer with an omitted endpoint extent.
        bridge = SparseBridgeSpec(
            "left", "right", "record", 4, 4,
            left_axes=(-3, -2, -1), right_axes=(1, 2, 3),
            same_point=False, scene=0, part_count=1, used_np_right=3,
        )
        lane_sources = tuple({"metadata": {"kind": "sparse", "value_dtype": "complex128"}}
                             for _ in range(2))
        source = SimpleNamespace(metadata={
            "family": "p2p-coordinate-batch",
            "lane_nnz_upper": (2, 2),
            "lane_sources": lane_sources,
            "coordinate_keys": ("coordinate-0", "coordinate-1"),
            "part_count": 1,
        })
        left_spec = SimpleNamespace(shape=(2, 2, 4, 4, 3), dtype=np.dtype("complex128"))
        right_spec = SimpleNamespace(shape=(2, 4, 3, 3, 5), dtype=np.dtype("complex128"))
        state = SimpleNamespace(
            problem=SimpleNamespace(
                sparse_bridges=(bridge,), sources={"record": source}, backend="numpy",
            ),
            catalog={"left": left_spec, "right": right_spec},
        )
        action = SimpleNamespace(
            kind="sparse_bridge", operands=("left", "right", "record"),
            term_index=0, same_point=False, used_np=4, used_np_right=3,
            left_axes=(-3, -2, -1), right_axes=(1, 2, 3),
            transpose_bridge=False, equation=parse_axis_pattern('xaipc,xjqdb->xq'),
        )
        observed = _coordinate_batch_sparse_workspace_bytes(state, action)

        rows = 4
        reshape = (np.prod(left_spec.shape) + np.prod(right_spec.shape)) * 16
        gathered = rows * (2 * 4 * 3 * 16 + 4 * 3 * 5 * 16)
        packed = rows * (3 * 8 + 4 * 4 * 16 + 16 + 3 * 3 * 16)
        contributions = rows * 2 * 5 * 16
        self.assertEqual(observed, reshape + gathered + packed + contributions)

    def test_real_strict_coordinate_batch_plan_audits_retained_point_shape(self):
        # Reuse the real C task-batch fixture rather than a synthetic state:
        # it supplies two coordinate lanes, four sparse parts per lane, and
        # the same marked source metadata consumed by X admission.
        import sys
        from dataclasses import replace

        sys.path.insert(0, "tests")
        import test_sparse_point_sink_batch as batch_fixture
        from lqcdweave.contracts.correlator.expanded import ExpandedCorrelator
        from lqcdweave.contraction.execution.api import audit_execution_memory
        from lqcdweave.contraction.planning.api import find_contraction_path

        _provider, expanded, _old_path = (
            batch_fixture.SparsePointSinkBatchCatalogTests._four_part_path(
                1 << 28, same_point=False,
            )
        )
        inputs = expanded.inputs
        child = inputs.task_batches[0]
        retained_problem = replace(
            child.problem,
            terms=(replace(child.problem.terms[0], subscripts="aspc,aqrd->ar"),),
        )
        retained_child = replace(child, problem=retained_problem)
        retained_inputs = replace(
            inputs, problem=retained_problem, task_batches=(retained_child,),
        )
        retained_expanded = replace(expanded, inputs=retained_inputs)

        def plan_with_cap(cap):
            return find_contraction_path(
                retained_expanded, memory_limit=cap, optimize="none",
            )

        planned = plan_with_cap(1 << 28)
        child_path, = planned.batch_paths
        state = child_path.plan.state
        action = next(item for item in state.actions if item.kind == "sparse_bridge")
        self.assertEqual(action.equation, parse_axis_pattern('aspc,aqrd->ar'))
        self.assertEqual(state.catalog[action.tensor].shape, (2, 1))

        # Independently sum the known X owners for this real shape.  The
        # explicit right endpoint point label is in the final output, so no
        # raw-label projection allocation is part of the workspace.
        c_rows = min(64, 4 * (1 + 1))
        input_copy = 2 * (2 * 4 * 1 * 3 * 16)
        gathered = c_rows * (4 * 3 * 16 + 4 * 3 * 16)
        packed = c_rows * (3 * 8 + 4 * 4 * 16 + 16 + 3 * 3 * 16)
        contributions = c_rows * 16
        expected_known_workspace = input_copy + gathered + packed + contributions
        self.assertEqual(expected_known_workspace, 7488)

        audit = audit_execution_memory(planned, backend=np)
        batch_record, = audit["batches"]
        point, = (
            item for item in child_path.plan.memory
            if item.phase == "sparse_bridge"
        )
        self.assertEqual(
            batch_record["x_workspace_correction_host_bytes"],
            max(0, expected_known_workspace - point.workspace_host_bytes),
        )
        self.assertEqual(
            audit["peak_host_bytes"], batch_record["corrected_peak_host_bytes"],
        )
        self.assertEqual(
            batch_record["corrected_peak_host_bytes"],
            batch_record["plan_peak_host_bytes"]
            + batch_record["x_workspace_correction_host_bytes"],
        )

        exact_cap = audit["peak_host_bytes"]
        exact = audit_execution_memory(plan_with_cap(exact_cap), backend=np)
        self.assertTrue(exact["feasible"])
        self.assertEqual(exact["peak_host_bytes"], exact_cap)
        # P now freezes the known workspace and rejects before X admission.
        with self.assertRaisesRegex(MemoryError, "audited physical storage budget"):
            plan_with_cap(exact_cap - 1)


if __name__ == "__main__":
    unittest.main()
