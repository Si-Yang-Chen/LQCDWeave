"""Independent references for a two-record cycle retaining an endpoint point."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    parse_axis_pattern,
    SparseBridgeSpec,
    cycle_external_labels,
    cycle_intermediate_labels,
    cycle_point_equalities_covered,
    cycle_requires_generic_join,
)
from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import (
    contract_p2p_cycle,
    contract_p2p_cycle_batch,
)


_KERNEL = "lqcdweave.contraction.execution.operations.sparse_bridge_kernel"
_PAIR_EQUATION = "BAobna,DCqdpc->q"


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _record(indices, values):
    return {
        "type": "sparse",
        "indices": np.asarray(indices, dtype=np.int64),
        "values": np.asarray(values),
    }


def _parts(rng, *, indices, coefficient, transpose=False, multiple=False):
    count = len(indices)
    values = _complex(rng, (count, 3, 3))
    gamma = _complex(rng, (4, 4))
    result = [(gamma, coefficient, _record(indices, values))]
    if multiple:
        values2 = _complex(rng, (count, 3, 3))
        gamma2 = _complex(rng, (4, 4))
        result.append((gamma2, -0.375 + 0.625j, _record(indices[::-1], values2)))
    return tuple(result)


def _scalar_bridges(first_nnz=4, second_nnz=4):
    return (
        SparseBridgeSpec(
            "tensor:29", "tensor:23", "record:current-a", 3, first_nnz,
            left_axes=(1, 4, 5), right_axes=(0, 2, 3),
        ),
        SparseBridgeSpec(
            "tensor:29", "tensor:23", "record:current-b", 3, second_nnz,
            left_axes=(0, 2, 3), right_axes=(1, 4, 5), transpose_bridge=True,
        ),
    )


def _pair_oracle(left, right, first_parts, second_parts, *, output_labels="q"):
    """Explicitly slice physical spin/color/point axes for every COO pair."""
    if output_labels not in ("", "q"):
        raise AssertionError("this oracle intentionally covers only closed and q outputs")
    result = (np.zeros((right.shape[2],), dtype=np.complex128)
              if output_labels == "q" else np.asarray(0j, dtype=np.complex128))
    for gamma1, coefficient1, record1 in first_parts:
        for (first_left, first_right), block1 in zip(
                record1["indices"], record1["values"], strict=True):
            for gamma2, coefficient2, record2 in second_parts:
                for (raw_left2, raw_right2), raw_block2 in zip(
                        record2["indices"], record2["values"], strict=True):
                    # current-b is daggered: raw bar/field coordinates and
                    # the spin/color factors enter the operand axes reversed.
                    left_point2, right_point2 = int(raw_right2), int(raw_left2)
                    block2 = raw_block2.T
                    gamma2_oriented = gamma2.T
                    output_point = int(first_right)
                    left_slab = left[
                        :, :, left_point2, :, int(first_left), :,
                    ]
                    right_slab = right[
                        :, :, output_point, :, right_point2, :,
                    ]
                    value = np.einsum(
                        "BAba,DCdc,AD,ad,BC,bc->",
                        left_slab, right_slab, gamma1, block1,
                        gamma2_oriented, block2,
                        optimize=False,
                    )
                    contribution = coefficient1 * coefficient2 * value
                    if output_labels == "q":
                        result[output_point] += contribution
                    else:
                        result[...] += contribution
    return result


def _normalized_part(indices, values, gamma, coefficient):
    return gamma, coefficient, {
        "layout": "normalized-coo",
        "indices": np.asarray(indices, dtype=np.int64),
        "values": np.asarray(values),
    }


def _batch_payload(lanes):
    return {
        "layout": "coordinate_batch_coo",
        "point_keys": tuple(f"lane-{index}" for index in range(len(lanes))),
        "lanes": tuple(tuple(parts) for parts in lanes),
    }


def _coordinate_bridges(first_lanes, second_lanes):
    first_nnz = sum(len(part[2]["indices"]) for lane in first_lanes for part in lane)
    second_nnz = sum(len(part[2]["indices"]) for lane in second_lanes for part in lane)
    return (
        SparseBridgeSpec(
            "tensor:29", "tensor:23", "record:current-a", 3, first_nnz,
            left_axes=(2, 5, 6), right_axes=(1, 3, 4),
        ),
        SparseBridgeSpec(
            "tensor:29", "tensor:23", "record:current-b", 3, second_nnz,
            left_axes=(1, 3, 4), right_axes=(2, 5, 6), transpose_bridge=True,
        ),
    )


class SparseCycleRetainedPointTests(TestCase):
    def test_cycle_contract_helpers_keep_only_needed_point_incidence(self):
        first, second = _scalar_bridges()
        closed = "BAobna,DCqdpc->"
        opened = _PAIR_EQUATION

        self.assertTrue(cycle_point_equalities_covered(closed, first, second))
        self.assertFalse(cycle_requires_generic_join(closed, first, second))
        self.assertTrue(cycle_requires_generic_join(opened, first, second))
        self.assertEqual(
            cycle_external_labels(closed, first, second),
            ((66, 111, 98, 67, 112, 99), (), ()),
        )
        self.assertEqual(
            cycle_external_labels(opened, first, second),
            ((66, 111, 98, 67, 112, 99), (113,), (113,)),
        )
        self.assertEqual(
            cycle_intermediate_labels(
                closed, first, second, needed_labels=(113, 97, 66),
            ),
            ((66, 111, 98, 67, 112, 99), (113,), ()),
        )
        self.assertEqual(
            cycle_intermediate_labels(
                opened, first, second, needed_labels="qa",
            ),
            ((66, 111, 98, 67, 112, 99), (113,), (113,)),
        )
        with self.assertRaisesRegex(TypeError, "ordered tuple"):
            cycle_intermediate_labels(
                closed, first, second, needed_labels={"q"},
            )
        with self.assertRaisesRegex(ValueError, "output axis IDs must occur in an input"):
            cycle_external_labels("BAobna,DCqdpc->z", first, second)

    def test_daggered_two_record_cycle_retains_q_with_complex_dense_reference(self):
        rng = np.random.default_rng(20261010)
        left = _complex(rng, (4, 4, 3, 3, 3, 3))
        right = _complex(rng, (4, 4, 3, 3, 3, 3))
        first_indices = np.asarray(((0, 1), (2, 0), (0, 1)), dtype=np.int64)
        second_indices = np.asarray(((1, 2), (0, 1), (1, 2)), dtype=np.int64)
        first_parts = _parts(
            rng, indices=first_indices, coefficient=0.375 - 0.625j, multiple=True,
        )
        second_parts = _parts(
            rng, indices=second_indices, coefficient=-0.25 + 0.5j, multiple=True,
        )
        first, second = _scalar_bridges(
            sum(len(record[2]["indices"]) for record in first_parts),
            sum(len(record[2]["indices"]) for record in second_parts),
        )
        expected = _pair_oracle(left, right, first_parts, second_parts)

        with patch(
            f"{_KERNEL}.contract_sparse_p2p_join",
            wraps=__import__(
                "lqcdweave.contraction.execution.operations.sparse_bridge_kernel",
                fromlist=["contract_sparse_p2p_join"],
            ).contract_sparse_p2p_join,
        ) as generic_join:
            actual = contract_p2p_cycle(
                left, right, first_parts, second_parts,
                first_bridge=first, second_bridge=second, equation=_PAIR_EQUATION,
            )
        self.assertEqual(actual.shape, (3,))
        self.assertEqual(actual.dtype, np.dtype(np.complex128))
        np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13)
        generic_join.assert_called_once()

    def test_closed_covered_cycle_keeps_legacy_two_stage_route(self):
        rng = np.random.default_rng(20261011)
        left = _complex(rng, (4, 4, 3, 3, 3, 3))
        right = _complex(rng, (4, 4, 3, 3, 3, 3))
        first_indices = np.asarray(((0, 1), (2, 0)), dtype=np.int64)
        second_indices = np.asarray(((1, 2), (0, 1)), dtype=np.int64)
        first_parts = _parts(rng, indices=first_indices, coefficient=0.5 + 0.25j)
        second_parts = _parts(rng, indices=second_indices, coefficient=0.125 - 0.75j)
        first, second = _scalar_bridges(2, 2)
        expected = _pair_oracle(left, right, first_parts, second_parts, output_labels="")

        with patch(f"{_KERNEL}.contract_sparse_p2p_join",
                   side_effect=AssertionError("closed cycle changed to generic route")):
            actual = contract_p2p_cycle(
                left, right, first_parts, second_parts,
                first_bridge=first, second_bridge=second,
                equation=parse_axis_pattern('BAobna,DCqdpc->'),
            )
        np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13)

    def test_coordinate_batch_retained_cycle_scatter_is_lane_local(self):
        rng = np.random.default_rng(20261012)
        left = _complex(rng, (2, 4, 4, 3, 3, 3, 3))
        right = _complex(rng, (2, 4, 4, 3, 3, 3, 3))
        first_lanes, second_lanes = [], []
        expected = []
        for lane in range(2):
            first_indices = np.asarray(
                ((lane, 1), (2, lane), (lane, 1)), dtype=np.int64,
            )
            second_indices = np.asarray(
                ((1, 2 - lane), (lane, 1), (1, 2 - lane)), dtype=np.int64,
            )
            first_parts = _parts(
                rng, indices=first_indices,
                coefficient=0.375 + 0.125j * lane,
            )
            second_parts = _parts(
                rng, indices=second_indices,
                coefficient=-0.25 + 0.5j * lane,
            )
            first_lanes.append(tuple(
                _normalized_part(record[2]["indices"], record[2]["values"],
                                 record[0], record[1])
                for record in first_parts
            ))
            second_lanes.append(tuple(
                _normalized_part(record[2]["indices"], record[2]["values"],
                                 record[0], record[1])
                for record in second_parts
            ))
            expected.append(_pair_oracle(
                left[lane], right[lane], first_parts, second_parts,
            ))
        first, second = _coordinate_bridges(first_lanes, second_lanes)
        actual = contract_p2p_cycle_batch(
            left, right, _batch_payload(first_lanes), _batch_payload(second_lanes),
            first_bridge=first, second_bridge=second,
            equation=parse_axis_pattern('xBAobna,xDCqdpc->xq'),
        )
        expected = np.stack(expected)
        self.assertEqual(actual.shape, (2, 3))
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13)

        cross_lane = np.stack((
            _pair_oracle(left[0], right[0], first_lanes[0], second_lanes[1]),
            _pair_oracle(left[1], right[1], first_lanes[1], second_lanes[0]),
        ))
        self.assertGreater(float(np.max(np.abs(expected - cross_lane))), 1e-8)

    def test_empty_cycle_record_returns_zero_on_retained_point_axis(self):
        rng = np.random.default_rng(20261013)
        left = _complex(rng, (4, 4, 3, 3, 3, 3))
        right = _complex(rng, (4, 4, 3, 3, 3, 3))
        empty = _record(np.empty((0, 2), dtype=np.int64),
                        np.empty((0, 3, 3), dtype=np.complex128))
        nonempty = _record(((0, 2),), _complex(rng, (1, 3, 3)))
        first_parts = ((_complex(rng, (4, 4)), 1 + 0.25j, empty),)
        second_parts = ((_complex(rng, (4, 4)), -0.5j, nonempty),)
        first, second = _scalar_bridges(0, 1)
        actual = contract_p2p_cycle(
            left, right, first_parts, second_parts,
            first_bridge=first, second_bridge=second, equation=_PAIR_EQUATION,
        )
        self.assertEqual(actual.shape, (3,))
        np.testing.assert_array_equal(actual, np.zeros((3,), dtype=np.complex128))

    def test_planned_shared_cycle_retains_point_for_residual_factor_and_replays(self):
        from lqcdweave.contracts.correlator.api import (
            ContractionProblem, ContractionTerm, SourcePreparation, SourceSpec,
            StorageBlock, TensorSpec,
        )
        from lqcdweave.contraction.performance.api import CostProfile
        from lqcdweave.contraction.planning.api import (
            ContractionState, plan_scene_storage, logical_source_problem, logical_cost_profile,
        )
        from lqcdweave.contraction.execution.api import replay_scene_storage
        from lqcdweave.storage.contraction_plans.api import (
            decode_plan_value, encode_plan_value,
        )

        rng = np.random.default_rng(20261018)
        left = _complex(rng, (4, 4, 3, 3, 3, 3))
        right = _complex(rng, left.shape)
        parts1 = _parts(rng, indices=((0, 1), (2, 0), (0, 1)), coefficient=.3 + .2j)
        parts2 = _parts(rng, indices=((1, 2), (0, 1)), coefficient=-.4j)
        first, second = _scalar_bridges(3, 2)
        values = {
            first.left: left, first.right: right,
            first.record: parts1, second.record: parts2,
            "weights": _complex(rng, (3,)), "other-weights": _complex(rng, (3,)),
        }
        specs = {key: TensorSpec(value.shape, value.dtype)
                 for key, value in values.items() if isinstance(value, np.ndarray)}
        sources = {key: SourceSpec(key, value.nbytes if isinstance(value, np.ndarray) else 4096)
                   for key, value in values.items()}
        problem = ContractionProblem(
            specs,
            tuple(ContractionTerm((first.left, first.right, weights),
                                  "BAobna,DCqdpc,q->")
                  for weights in ("weights", "other-weights")),
            sources, sparse_bridges=(first, second),
        )
        profile, cap = CostProfile(), 1 << 28
        model = logical_cost_profile(profile)
        state = ContractionState.initial(logical_source_problem(problem))
        for key in sources:
            state = state.load(key, model, cap)
        from lqcdweave.contraction.planning.logical_index import LayoutIndex
        layouts = LayoutIndex(state)
        substitutions = dict(layouts.substitutions(state._contract_layout(0, (0, 1))))
        self.assertEqual(set(substitutions), {0, 1})
        for updated, _result in substitutions.values():
            self.assertEqual(updated.factors[-1].labels, (ord("q"),))
        # Sharing replaces the same cycle in both scenes. Its q axis must be
        # retained although both final outputs are scalar.
        state = state.contract(0, (0, 1), model, cap)
        action = state.actions[-1]
        self.assertEqual(action.join_id, "bounded_coo_join")
        self.assertEqual(action.equation.output, (ord("q"),))
        self.assertEqual(state.catalog[action.tensor].shape, (3,))
        self.assertEqual(state.pending[0].factors[-1], state.pending[1].factors[-1])
        state = state.contract(0, (0, 1), model, cap)
        state = state.contract(1, (0, 1), model, cap)
        plan = plan_scene_storage(
            state, {key: StorageBlock(key, "full", spec.region_bytes,
                                     host_bytes=spec.region_bytes)
                    for key, spec in sources.items()},
            {key: SourcePreparation((key,)) for key in sources},
            profile=profile, memory_limit=cap,
        )
        self.assertTrue(plan.feasible)

        class Loader:
            def read_block(self, key, _spec):
                return values[key]

            def materialize(self, key, _source, blocks):
                return blocks[key]

        # The same chosen route and encoded route consume fresh numerical
        # inputs on each replay, without invoking expansion or path search.
        restored = decode_plan_value(encode_plan_value(plan))
        for selected in (plan, restored):
            expected_cycle = _pair_oracle(values[first.left], values[first.right],
                                          parts1, parts2)
            expected = [expected_cycle @ values[key]
                        for key in ("weights", "other-weights")]
            np.testing.assert_allclose(replay_scene_storage(selected, Loader()), expected,
                                       rtol=5e-13, atol=5e-12)
            values[first.left] = -1.25 * values[first.left]


if __name__ == "__main__":
    import unittest

    unittest.main()
