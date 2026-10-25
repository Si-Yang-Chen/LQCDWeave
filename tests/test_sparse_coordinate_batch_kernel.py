"""Independent numeric checks for lane-packed sparse coordinate kernels."""

from __future__ import annotations

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import SparseBridgeSpec
from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import (
    contract_p2p_bridge_batch,
    contract_p2p_cycle_batch,
)
from lqcdweave.contraction.execution.operations.sparse_join import (
    contract_sparse_p2p_bridge_join_batch,
)


def _part(indices, values, gamma, coefficient):
    return (gamma, coefficient, {
        "layout": "normalized-coo",
        "indices": np.asarray(indices, dtype=np.int64),
        "values": np.asarray(values),
    })


def _batch(lanes):
    return {
        "layout": "coordinate_batch_coo",
        "point_keys": tuple(f"lane-{index}" for index in range(len(lanes))),
        "lanes": tuple(tuple(parts) for parts in lanes),
    }


class SparseCoordinateBatchKernelTests(TestCase):
    def test_single_generic_repeated_endpoint_label_and_negative_axes(self):
        rng = np.random.default_rng(20261004)
        left = rng.normal(size=(2, 2, 3, 2, 3)) + 1j * rng.normal(size=(2, 2, 3, 2, 3))
        right = rng.normal(size=(2, 2, 4, 2)) + 1j * rng.normal(size=(2, 2, 4, 2))
        lanes = []
        expected = np.zeros((2, 3), dtype=np.complex128)
        for lane in range(2):
            gamma = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            indices = np.asarray(((0, 1), (2, 3), (1, 0)), dtype=np.int64)
            values = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
            coefficient = complex(0.25 + lane, -0.125 * (lane + 1))
            lanes.append((_part(indices, values, gamma, coefficient),))
            for row, (point_left, point_right) in enumerate(indices):
                expected[lane, point_left] += coefficient * np.einsum(
                    "ic,ij,cd,jd->",
                    left[lane, :, point_left, :, point_left], gamma,
                    values[row], right[lane, :, point_right, :],
                    optimize=False,
                )

        bridge = SparseBridgeSpec(
            "left", "right", "record", 3, 3,
            left_axes=(-4, -3, -2), right_axes=(1, 2, 3),
            used_np_right=4,
        )
        with patch("lqcdweave.contraction.execution.operations.sparse_bridge_kernel.contract_p2p_bridge",
                   side_effect=AssertionError("batch kernel dispatched scalar bridge")), patch(
                       "lqcdweave.contraction.execution.operations.sparse_bridge_kernel.np.any",
                       wraps=np.any,
                   ) as bounds:
            actual = contract_p2p_bridge_batch(
                left, right, _batch(lanes), bridge=bridge,
                equation="xspcp,xqrd->xp",
            )
        self.assertEqual(bounds.call_count, 8)  # Four bounds, once per lane.
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_public_join_and_bridge_validate_coo_once_and_match_scalar_sum(self):
        left = np.arange(6, dtype=np.float64).reshape(2, 1, 3, 1) + 1
        right = np.arange(8, dtype=np.float64).reshape(2, 1, 4, 1) + 2
        indices = np.asarray(((2, 3), (0, 1), (2, 3)), dtype=np.int64)
        lanes = [(_part(indices, np.asarray([1, 2, 3], dtype=dtype).reshape(3, 1, 1),
                        np.ones((1, 1), dtype=dtype), coefficient),)
                 for dtype, coefficient in ((np.float64, 0.5), (np.complex64, 1 + 2j))]
        bridge = SparseBridgeSpec(
            "left", "right", "record", 3, 6,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3), used_np_right=4,
        )
        expected = np.zeros((2, 3), dtype=np.complex128)
        for lane, parts in enumerate(lanes):
            gamma, coefficient, record = parts[0]
            for row, (p, q) in enumerate(indices):
                expected[lane, p] += (left[lane, 0, p, 0] * right[lane, 0, q, 0]
                                      * gamma[0, 0] * coefficient * record['values'][row, 0, 0])
        for call in (contract_p2p_bridge_batch, contract_sparse_p2p_bridge_join_batch):
            with self.subTest(entry=call.__name__), patch(
                "lqcdweave.contraction.execution.operations.sparse_join.np.any",
                wraps=np.any,
            ) as bounds:
                actual = call(left, right, _batch(lanes), bridge=bridge,
                              equation="xspc,xqrd->xp")
            self.assertEqual(bounds.call_count, 8)
            np.testing.assert_array_equal(actual, expected)

    def test_public_join_and_bridge_reject_last_lane_bounds_before_output(self):
        left = np.ones((2, 1, 3, 1))
        right = np.ones((2, 1, 4, 1))
        gamma = np.ones((1, 1))
        values = np.ones((1, 1, 1))
        valid = _part(((0, 1),), values, gamma, 1)
        bridge = SparseBridgeSpec(
            "left", "right", "record", 3, 2,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3), used_np_right=4,
        )
        for call in (contract_p2p_bridge_batch, contract_sparse_p2p_bridge_join_batch):
            for pair in ((-1, 0), (3, 0), (0, -1), (0, 4)):
                invalid = _part((pair,), values, gamma, 1)
                with self.subTest(entry=call.__name__, pair=pair), patch(
                    "lqcdweave.contraction.execution.operations.sparse_join.np.zeros",
                    side_effect=AssertionError("allocated before validating the last lane"),
                ), self.assertRaisesRegex(IndexError, "outside"):
                    call(left, right, _batch(((valid,), (invalid,))), bridge=bridge,
                         equation="xspc,xqrd->xp")

    def test_single_shared_point_label_is_retained_and_diagonal(self):
        rng = np.random.default_rng(712)
        left = rng.normal(size=(2, 2, 3, 2, 3)) + 1j * rng.normal(size=(2, 2, 3, 2, 3))
        right = rng.normal(size=(2, 2, 3, 2)) + 1j * rng.normal(size=(2, 2, 3, 2))
        lanes = []
        expected = np.zeros((2, 3), dtype=np.complex128)
        for lane in range(2):
            gamma = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            indices = np.asarray(((0, 0), (0, 1), (1, 1), (2, 2)), dtype=np.int64)
            values = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
            coefficient = 0.5 - 0.25j * lane
            lanes.append((_part(indices, values, gamma, coefficient),))
            for row, (point_left, point_right) in enumerate(indices):
                if point_left != point_right:
                    continue
                expected[lane, point_left] += coefficient * np.einsum(
                    "ic,ij,cd,jd->",
                    left[lane, :, point_left, :, point_left], gamma,
                    values[row], right[lane, :, point_right, :],
                    optimize=False,
                )
        bridge = SparseBridgeSpec(
            "left", "right", "record", 3, 4,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
        )
        actual = contract_p2p_bridge_batch(
            left, right, _batch(lanes), bridge=bridge,
            equation="xspcp,xqpd->xp",
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_single_scatter_interleaves_point_and_free_output_axes(self):
        rng = np.random.default_rng(44)
        left = rng.normal(size=(2, 2, 3, 2)) + 1j * rng.normal(size=(2, 2, 3, 2))
        right = rng.normal(size=(2, 2, 4, 2, 5)) + 1j * rng.normal(size=(2, 2, 4, 2, 5))
        lanes = []
        expected = np.zeros((5, 3, 2), dtype=np.complex128)
        for lane in range(2):
            gamma = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            indices = np.asarray(((0, 2), (1, 1), (0, 2)), dtype=np.int64)
            values = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
            coefficient = 0.2 + (lane + 1) * 0.3j
            lanes.append((_part(indices, values, gamma, coefficient),))
            for row, (point_left, point_right) in enumerate(indices):
                expected[:, point_left, lane] += coefficient * np.einsum(
                    "ic,ij,cd,jde->e",
                    left[lane, :, point_left, :], gamma, values[row],
                    right[lane, :, point_right, :, :], optimize=False,
                )
        bridge = SparseBridgeSpec(
            "left", "right", "record", 3, 3,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3), used_np_right=4,
        )
        actual = contract_p2p_bridge_batch(
            left, right, _batch(lanes), bridge=bridge,
            equation="xspc,xqrdt->tpx",
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def _cycle_fixture(self, *, repeated_left_point=True, rows=3, seed=91):
        rng = np.random.default_rng(seed)
        left = rng.normal(size=(2, 2, 3, 3, 2, 3, 3)) + 1j * rng.normal(
            size=(2, 2, 3, 3, 2, 3, 3),
        )
        right = rng.normal(size=(2, 2, 3, 3, 2, 3, 3)) + 1j * rng.normal(
            size=(2, 2, 3, 3, 2, 3, 3),
        )
        first_lanes, second_lanes = [], []
        for lane in range(2):
            first_indices = np.asarray(
                [((row + lane) % 3, (2 * row + 1) % 3) for row in range(rows)],
                dtype=np.int64,
            )
            second_indices = np.asarray(
                [((row + lane) % 3, (row + 2 * lane) % 3) for row in range(rows)],
                dtype=np.int64,
            )
            first_values = rng.normal(size=(rows, 3, 3)) + 1j * rng.normal(size=(rows, 3, 3))
            second_values = rng.normal(size=(rows, 3, 3)) + 1j * rng.normal(size=(rows, 3, 3))
            gamma1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            gamma2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            first_lanes.append((_part(
                first_indices, first_values, gamma1, 0.75 + 0.125j * lane,
            ),))
            second_lanes.append((_part(
                second_indices, second_values, gamma2, -0.25 + 0.5j * (lane + 1),
            ),))
        first_bridge = SparseBridgeSpec(
            "left", "right", "first", 3, 2 * rows,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
        )
        second_bridge = SparseBridgeSpec(
            "left", "right", "second", 3, 2 * rows,
            left_axes=(4, 5, 6), right_axes=(4, 5, 6),
        )
        equation = ("xapcbpd,xeqfgsh->x" if repeated_left_point
                    else "xapcbrd,xeqfgsh->x")
        return left, right, first_lanes, second_lanes, first_bridge, second_bridge, equation

    @staticmethod
    def _cycle_oracle(left, right, first_lanes, second_lanes, *, repeated_left_point):
        output = np.zeros(left.shape[0], dtype=np.complex128)
        for lane in range(left.shape[0]):
            for gamma1, coefficient1, record1 in first_lanes[lane]:
                for row1, (point1_left, point1_right) in enumerate(record1["indices"]):
                    for gamma2, coefficient2, record2 in second_lanes[lane]:
                        for row2, (point2_left, point2_right) in enumerate(record2["indices"]):
                            if repeated_left_point and point1_left != point2_left:
                                continue
                            left_selected = left[
                                lane, :, point1_left, :, :, point2_left, :,
                            ]
                            right_selected = right[
                                lane, :, point1_right, :, :, point2_right, :,
                            ]
                            contribution = coefficient1 * coefficient2 * np.einsum(
                                "acbd,efgh,ae,cf,bg,dh->",
                                left_selected, right_selected,
                                gamma1, record1["values"][row1],
                                gamma2, record2["values"][row2],
                                optimize=False,
                            )
                            output[lane] += contribution
        return output

    def test_cycle_generic_packs_matching_rows_with_lane_keyed_buckets(self):
        (left, right, first_lanes, second_lanes, first_bridge,
         second_bridge, equation) = self._cycle_fixture(repeated_left_point=True)
        first_batch, second_batch = _batch(first_lanes), _batch(second_lanes)
        expected = self._cycle_oracle(
            left, right, first_lanes, second_lanes, repeated_left_point=True,
        )
        with patch("lqcdweave.contraction.execution.operations.sparse_bridge_kernel.contract_p2p_cycle",
                   side_effect=AssertionError("batch kernel dispatched scalar cycle")):
            actual = contract_p2p_cycle_batch(
                left, right, first_batch, second_batch,
                first_bridge=first_bridge, second_bridge=second_bridge,
                equation=equation,
            )
        np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13)

        cross_lane = np.asarray([
            self._cycle_oracle(
                left[index:index + 1], right[1 - index:2 - index],
                [first_lanes[index]], [second_lanes[index]], repeated_left_point=True,
            )[0]
            for index in range(2)
        ])
        self.assertGreater(float(np.max(np.abs(expected - cross_lane))), 1e-8)

    def test_cycle_mixed_lane_gamma_and_blocks_use_numpy_promotion(self):
        (left, right, first_lanes, second_lanes, first_bridge,
         second_bridge, equation) = self._cycle_fixture(
            repeated_left_point=True, rows=2, seed=910,
        )
        left = left.astype(np.complex64)
        right = right.astype(np.complex64)

        def mixed_parts(lanes, lane_index):
            result = []
            for gamma, coefficient, record in lanes[lane_index]:
                if lane_index == 0:
                    mixed_gamma = gamma.real.astype(np.float64)
                    mixed_values = record["values"].real.astype(np.float64)
                    mixed_coefficient = np.float64(np.real(coefficient))
                else:
                    mixed_gamma = gamma.astype(np.complex64)
                    mixed_values = record["values"].astype(np.complex64)
                    mixed_coefficient = np.complex64(coefficient)
                result.append(_part(
                    record["indices"], mixed_values, mixed_gamma,
                    mixed_coefficient,
                ))
            return tuple(result)

        first_lanes = [mixed_parts(first_lanes, lane) for lane in range(2)]
        second_lanes = [mixed_parts(second_lanes, lane) for lane in range(2)]
        expected = self._cycle_oracle(
            left, right, first_lanes, second_lanes, repeated_left_point=True,
        )
        actual = contract_p2p_cycle_batch(
            left, right, _batch(first_lanes), _batch(second_lanes),
            first_bridge=first_bridge, second_bridge=second_bridge,
            equation=equation,
        )
        self.assertEqual(actual.dtype, np.dtype(np.complex128))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_cycle_moves_lane_to_nonleading_output_position(self):
        (left0, right, first_lanes, second_lanes, first_bridge,
         second_bridge, _equation) = self._cycle_fixture(
            repeated_left_point=True, rows=2, seed=582,
        )
        left = np.repeat(left0[..., None], 5, axis=-1)
        expected = np.zeros((5, 2), dtype=np.complex128)
        for lane in range(2):
            for gamma1, coefficient1, record1 in first_lanes[lane]:
                for row1, (point1_left, _point1_right) in enumerate(record1["indices"]):
                    for gamma2, coefficient2, record2 in second_lanes[lane]:
                        for row2, (point2_left, point2_right) in enumerate(record2["indices"]):
                            if point1_left != point2_left:
                                continue
                            right_selected = right[
                                lane, :, record1["indices"][row1, 1], :,
                                :, point2_right, :,
                            ]
                            for free_axis in range(5):
                                left_selected = left[
                                    lane, :, point1_left, :, :, point2_left, :, free_axis,
                                ]
                                expected[free_axis, lane] += coefficient1 * coefficient2 * np.einsum(
                                    "acbd,efgh,ae,cf,bg,dh->",
                                    left_selected, right_selected,
                                    gamma1, record1["values"][row1],
                                    gamma2, record2["values"][row2],
                                    optimize=False,
                                )
        actual = contract_p2p_cycle_batch(
            left, right, _batch(first_lanes), _batch(second_lanes),
            first_bridge=first_bridge, second_bridge=second_bridge,
            equation="xapcbpdk,xeqfgsh->kx",
        )
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13)

    def test_cycle_without_cross_bridge_equality_keeps_duplicate_pair_summands(self):
        (left, right, first_lanes, second_lanes, first_bridge,
         second_bridge, equation) = self._cycle_fixture(
            repeated_left_point=False, rows=2, seed=103,
        )
        # Duplicate the first sparse row: COO rows are summands, not a set.
        duplicated = []
        for lane_parts in first_lanes:
            gamma, coefficient, record = lane_parts[0]
            duplicated.append((gamma, coefficient, {
                "layout": "normalized-coo",
                "indices": np.concatenate((record["indices"], record["indices"][:1]), axis=0),
                "values": np.concatenate((record["values"], record["values"][:1]), axis=0),
            }))
        first_lanes = [(part,) for part in duplicated]
        first_bridge = replace(first_bridge, nnz_upper=2 * (len(first_lanes[0][0][2]["indices"])))
        expected = self._cycle_oracle(
            left, right, first_lanes, second_lanes, repeated_left_point=False,
        )
        actual = contract_p2p_cycle_batch(
            left, right, _batch(first_lanes), _batch(second_lanes),
            first_bridge=first_bridge, second_bridge=second_bridge,
            equation=equation,
        )
        np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13)

    def test_cycle_packs_tail_after_64_rows_and_accepts_empty_lane(self):
        rng = np.random.default_rng(198)
        left_storage = rng.normal(size=(2, 2, 3, 3, 2, 3, 6)) + 1j * rng.normal(
            size=(2, 2, 3, 3, 2, 3, 6),
        )
        right_storage = rng.normal(size=(2, 2, 3, 3, 2, 3, 6)) + 1j * rng.normal(
            size=(2, 2, 3, 3, 2, 3, 6),
        )
        left, right = left_storage[..., ::2], right_storage[..., ::2]
        self.assertFalse(left.flags.c_contiguous)
        self.assertFalse(right.flags.c_contiguous)
        gamma1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        gamma2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        first_indices = np.column_stack((np.zeros(65, dtype=np.int64),
                                         np.arange(65, dtype=np.int64) % 3))
        first_values = rng.normal(size=(65, 3, 3)) + 1j * rng.normal(size=(65, 3, 3))
        first_part = _part(first_indices, first_values, gamma1, 0.75 - 0.125j)
        second_part = _part(((0, 2),), rng.normal(size=(1, 3, 3))
                            + 1j * rng.normal(size=(1, 3, 3)), gamma2, -0.25 + 0.5j)
        first_lanes = [(first_part,), ()]
        second_lanes = [(second_part,), ()]
        first_bridge = SparseBridgeSpec(
            "left", "right", "first", 3, 65,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
        )
        second_bridge = SparseBridgeSpec(
            "left", "right", "second", 3, 1,
            left_axes=(4, 5, 6), right_axes=(4, 5, 6),
        )
        equation = "xapcbpd,xeqfgsh->x"
        expected = np.zeros(2, dtype=np.complex128)
        for row1, (point1_left, point1_right) in enumerate(first_indices):
            left_selected = left[0, :, point1_left, :, :, 0, :]
            right_selected = right[0, :, point1_right, :, :, 2, :]
            expected[0] += (0.75 - 0.125j) * (-0.25 + 0.5j) * np.einsum(
                "acbd,efgh,ae,cf,bg,dh->",
                left_selected, right_selected,
                gamma1, first_values[row1],
                gamma2, second_part[2]["values"][0],
                optimize=False,
            )
        actual = contract_p2p_cycle_batch(
            left, right, _batch(first_lanes), _batch(second_lanes),
            first_bridge=first_bridge, second_bridge=second_bridge,
            equation=equation,
        )
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13)
        self.assertEqual(actual[1], 0)

    def test_cycle_multiple_parts_keep_lane_local_cartesian_sum(self):
        (left, right, first_lanes, second_lanes, first_bridge,
         second_bridge, equation) = self._cycle_fixture(
            repeated_left_point=True, rows=2, seed=414,
        )
        def add_part(lane_parts, scale, coefficient):
            gamma, _, record = lane_parts[0]
            return lane_parts + ((gamma * scale, coefficient, {
                "layout": "normalized-coo",
                "indices": record["indices"].copy(),
                "values": record["values"] * (scale + 0.25j),
            }),)

        first_lanes = [add_part(parts, 0.5 + 0.25j, 0.125 - 0.5j) for parts in first_lanes]
        second_lanes = [add_part(parts, -0.75 + 0.125j, 0.25 + 0.75j) for parts in second_lanes]
        first_bridge = replace(first_bridge, nnz_upper=8, part_count=2)
        second_bridge = replace(second_bridge, nnz_upper=8, part_count=2)
        expected = self._cycle_oracle(
            left, right, first_lanes, second_lanes, repeated_left_point=True,
        )
        actual = contract_p2p_cycle_batch(
            left, right, _batch(first_lanes), _batch(second_lanes),
            first_bridge=first_bridge, second_bridge=second_bridge,
            equation=equation,
        )
        np.testing.assert_allclose(actual, expected, rtol=8e-13, atol=8e-13)

    def test_cycle_two_lane_scalar_sentinels(self):
        left = np.zeros((2, 1, 1, 3, 1, 1, 3), dtype=np.complex128)
        right = np.zeros_like(left)
        left[0, 0, 0, 0, 0, 0, 0] = 2
        right[0, 0, 0, 0, 0, 0, 0] = 8
        left[1, 0, 0, 0, 0, 0, 0] = 6
        right[1, 0, 0, 0, 0, 0, 0] = 7
        gamma = np.ones((1, 1), dtype=np.complex128)
        first_lanes, second_lanes = [], []
        for first_value, second_value in ((11, 17), (13, 19)):
            first_values = np.zeros((1, 3, 3), dtype=np.complex128)
            second_values = np.zeros((1, 3, 3), dtype=np.complex128)
            first_values[0, 0, 0] = first_value
            second_values[0, 0, 0] = second_value
            first_lanes.append((_part(((0, 0),), first_values, gamma, 1),))
            second_lanes.append((_part(((0, 0),), second_values, gamma, 1),))
        first_bridge = SparseBridgeSpec(
            "left", "right", "first", 1, 2,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
        )
        second_bridge = SparseBridgeSpec(
            "left", "right", "second", 1, 2,
            left_axes=(4, 5, 6), right_axes=(4, 5, 6),
        )
        actual = contract_p2p_cycle_batch(
            left, right, _batch(first_lanes), _batch(second_lanes),
            first_bridge=first_bridge, second_bridge=second_bridge,
            equation="xapcbrd,xeqfgsh->x",
        )
        np.testing.assert_array_equal(actual, np.asarray((2992, 10374), dtype=np.complex128))

    def test_single_transposed_bridge_uses_record_oriented_dagger_data(self):
        rng = np.random.default_rng(805)
        left = rng.normal(size=(2, 2, 3, 2)).astype(np.complex64)
        right = rng.normal(size=(2, 3, 2, 2)).astype(np.complex64)
        lanes = []
        expected = np.zeros((2, 3), dtype=np.complex128)
        indices = np.asarray(((0, 2), (1, 0), (0, 1)), dtype=np.int64)
        for lane in range(2):
            gamma = (rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))).astype(np.complex64)
            values = (rng.normal(size=(3, 2, 2))
                      + 1j * rng.normal(size=(3, 2, 2))).astype(np.complex64)
            coefficient = np.complex64(0.75 - 0.125j * lane)
            lanes.append((_part(indices, values, gamma, coefficient),))
            for row, (record_left, record_right) in enumerate(indices):
                expected[lane, record_right] += coefficient * np.einsum(
                    "ic,ij,cd,jd->",
                    left[lane, :, record_right, :], gamma.T,
                    values[row].T, right[lane, :, record_left, :],
                    optimize=False,
                )
        bridge = SparseBridgeSpec(
            "left", "right", "record", 2, 3,
            left_axes=(1, 2, 3), right_axes=(1, 2, 3),
            transpose_bridge=True, used_np_right=3,
        )
        actual = contract_p2p_bridge_batch(
            left, right, _batch(lanes), bridge=bridge,
            equation="xspc,xqrd->xp",
        )
        self.assertEqual(actual.dtype, np.dtype(np.complex128))
        np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=5e-7)

    def test_single_mixed_lane_pack_promotes_float64_with_complex64(self):
        rng = np.random.default_rng(519)
        left = (rng.normal(size=(2, 2, 3, 2, 3))
                + 1j * rng.normal(size=(2, 2, 3, 2, 3))).astype(np.complex64)
        right = (rng.normal(size=(2, 2, 3, 2))
                 + 1j * rng.normal(size=(2, 2, 3, 2))).astype(np.complex64)
        lanes = []
        expected = np.zeros(2, dtype=np.complex128)
        indices = np.asarray(((0, 1), (2, 0)), dtype=np.int64)
        for lane in range(2):
            if lane == 0:
                gamma = rng.normal(size=(2, 2)).astype(np.float64)
                values = rng.normal(size=(2, 2, 2)).astype(np.float64)
                coefficient = np.float64(0.75)
            else:
                gamma = (rng.normal(size=(2, 2))
                         + 1j * rng.normal(size=(2, 2))).astype(np.complex64)
                values = (rng.normal(size=(2, 2, 2))
                          + 1j * rng.normal(size=(2, 2, 2))).astype(np.complex64)
                coefficient = np.complex64(0.25 - 0.5j)
            lanes.append((_part(indices, values, gamma, coefficient),))
            for row, (point_left, point_right) in enumerate(indices):
                expected[lane] += coefficient * np.einsum(
                    "ic,ij,cd,jd->",
                    left[lane, :, point_left, :, point_left], gamma,
                    values[row], right[lane, :, point_right, :],
                    optimize=False,
                )
        bridge = SparseBridgeSpec(
            "left", "right", "record", 3, 4,
            left_axes=(-4, -3, -2), right_axes=(1, 2, 3),
        )
        actual = contract_p2p_bridge_batch(
            left, right, _batch(lanes), bridge=bridge,
            equation="xspcp,xqrd->x",
        )
        self.assertEqual(actual.dtype, np.dtype(np.complex128))
        np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=5e-7)

    def test_single_late_invalid_lane_and_repeated_extent_fail_before_allocation(self):
        rng = np.random.default_rng(67)
        left = rng.normal(size=(2, 2, 3, 2, 3)) + 1j * rng.normal(size=(2, 2, 3, 2, 3))
        right = rng.normal(size=(2, 2, 4, 2)) + 1j * rng.normal(size=(2, 2, 4, 2))
        gamma = np.ones((2, 2), dtype=np.complex128)
        values = np.ones((1, 2, 2), dtype=np.complex128)
        valid = _part(((0, 1),), values, gamma, 1)
        malformed = _part(((0, 1),), np.ones((1, 2, 3)), gamma, 1)
        batch_record = _batch(((valid,), (malformed,)))
        bridge = SparseBridgeSpec(
            "left", "right", "record", 3, 2,
            left_axes=(-4, -3, -2), right_axes=(1, 2, 3),
            used_np_right=4,
        )
        with patch("lqcdweave.contraction.execution.operations.sparse_join.np.zeros",
                   side_effect=AssertionError("allocated before validation")):
            with self.assertRaisesRegex(ValueError, "invalid shape or dtype"):
                contract_p2p_bridge_batch(
                    left, right, batch_record, bridge=bridge,
                    equation="xspcp,xqrd->xp",
                )

        mismatched_left = rng.normal(size=(2, 2, 3, 2, 2)) + 1j * rng.normal(
            size=(2, 2, 3, 2, 2),
        )
        with patch("lqcdweave.contraction.execution.operations.sparse_join.np.zeros",
                   side_effect=AssertionError("allocated before dimension validation")):
            with self.assertRaisesRegex(ValueError, "label 112 has inconsistent dimensions"):
                contract_p2p_bridge_batch(
                    mismatched_left, right, _batch(((valid,), (valid,)),),
                    bridge=bridge, equation="xspcp,xqrd->xp",
                )
        lane_axis_bridge = replace(bridge, left_axes=(0, 2, 3))
        with self.assertRaisesRegex(ValueError, "cannot select the lane axis"):
            contract_p2p_bridge_batch(
                left, right, _batch(((valid,), (valid,))),
                bridge=lane_axis_bridge, equation="xspcp,xqrd->xp",
            )

    def test_late_invalid_cycle_payload_fails_before_result_allocation(self):
        (left, right, first_lanes, second_lanes, first_bridge,
         second_bridge, equation) = self._cycle_fixture()
        gamma, coefficient, record = second_lanes[-1][0]
        bad_record = dict(record, values=np.ones((len(record["indices"]), 2, 3), dtype=np.complex128))
        bad_second = list(second_lanes)
        bad_second[-1] = ((gamma, coefficient, bad_record),)
        with patch("lqcdweave.contraction.execution.operations.sparse_join.np.zeros",
                   side_effect=AssertionError("allocated before validation")):
            with self.assertRaisesRegex(ValueError, "invalid shape or dtype"):
                contract_p2p_cycle_batch(
                    left, right, _batch(first_lanes), _batch(bad_second),
                    first_bridge=first_bridge, second_bridge=second_bridge,
                    equation=equation,
                )
        mismatched_keys = _batch(second_lanes)
        mismatched_keys["point_keys"] = ("wrong-a", "wrong-b")
        with patch("lqcdweave.contraction.execution.operations.sparse_join.np.zeros",
                   side_effect=AssertionError("allocated before lane-key validation")):
            with self.assertRaisesRegex(ValueError, "same nonempty ordered lane keys"):
                contract_p2p_cycle_batch(
                    left, right, _batch(first_lanes), mismatched_keys,
                    first_bridge=first_bridge, second_bridge=second_bridge,
                    equation=equation,
                )
