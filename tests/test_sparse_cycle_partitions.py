"""Independent dense oracles for all two-current P2P point partitions."""

import unittest
from dataclasses import replace

import numpy as np

from lqcdweave.contraction.expansion.legacy_diagram import lower_state_graph
from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern,
    ContractionProblem,
    ContractionTerm,
    SourceSpec,
    SparseBridgeSpec,
    TensorSpec,
    bridge_point_equalities_covered,
    bridge_point_equivalence,
    cycle_point_equalities_covered,
    cycle_point_equivalence,
)
from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import (
    contract_p2p_bridge,
    contract_p2p_cycle,
)
from lqcdweave.contraction.execution.operations.sparse_join import contract_sparse_p2p_join
from lqcdweave.contraction.execution.scene_runtime import _model_features
from lqcdweave.contraction.expansion.sampling_coefficients import labelled_set_partitions
from lqcdweave.contraction.execution.indices import einsum_arguments
from lqcdweave.contraction.performance.api import (
    CostProfile,
    execution_cost_samples,
    update_cost_profile,
)
from lqcdweave.contraction.planning.api import ContractionState, rebind_scene_path
from lqcdweave.storage.contraction_plans.api import decode_plan_value, encode_plan_value
from lqcdweave.contracts.measurement import ActionTiming, ReplayReport


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _scene_labels(partition):
    # Ordered point legs are A=v0.field, B=v1.bar, C=v0.bar, D=v1.field.
    return {
        (0, "field"): f"p{partition[0]}",
        (1, "bar"): f"p{partition[1]}",
        (0, "bar"): f"p{partition[2]}",
        (1, "field"): f"p{partition[3]}",
    }


def _bridges(partition, n, *, first_transpose=False, second_transpose=False):
    return (
        SparseBridgeSpec(
            "p0", "p1", "r0", n, n * n,
            left_axes=(1, 4, 5), right_axes=(0, 2, 3),
            same_point=partition[0] == partition[2],
            transpose_bridge=first_transpose,
        ),
        SparseBridgeSpec(
            "p1", "p0", "r1", n, n * n,
            left_axes=(1, 4, 5), right_axes=(0, 2, 3),
            same_point=partition[3] == partition[1],
            transpose_bridge=second_transpose,
        ),
    )


def _record_dense(record, n):
    result = np.zeros((n, 3, n, 3), dtype=np.complex128)
    if record.get("type") == "identity":
        for point in range(n):
            result[point, :, point, :] = np.eye(3)
        return result
    for (left, right), values in zip(record["indices"], record["values"], strict=True):
        if left < n and right < n:
            result[left, :, right, :] += values
    return result


def _dense_oracle(partition, left, right, gamma0, gamma1, record0, record1,
                  n, *, first_transpose=False, second_transpose=False):
    network = lower_state_graph(
        [[0, 1], [2, 0]],
        ({"left": "p", "right": "p"}, {"left": "p", "right": "p"}),
        point_labels=_scene_labels(partition),
    )[0]
    raw0, raw1 = _record_dense(record0, n), _record_dense(record1, n)
    if first_transpose:
        gamma0 = gamma0.T
        raw0 = raw0.transpose(2, 3, 0, 1)
    if second_transpose:
        gamma1 = gamma1.T
        raw1 = raw1.transpose(2, 3, 0, 1)
    vertex0 = np.einsum("AD,naod->ADnaod", gamma0, raw0, optimize=False)
    vertex1 = np.einsum("CB,naob->CBnaob", gamma1, raw1, optimize=False)
    operands = []
    for request in network.requests:
        if request.kind == "propagator":
            operands.append(left if request.tensor_label == 1 else right)
        elif request.vertex_index == 0:
            operands.append(vertex0)
        else:
            operands.append(vertex1)
    return np.einsum(*einsum_arguments(network.equation, operands), optimize=False), network


class _SparseCyclePartitionFixture(unittest.TestCase):
    def _planned_cycle(self, partition, *, backend="numpy"):
        n = 3
        network = lower_state_graph(
            [[0, 1], [2, 0]],
            ({"left": "p", "right": "p"}, {"left": "p", "right": "p"}),
            point_labels=_scene_labels(partition),
        )[0]
        prop_requests = [request for request in network.requests if request.kind == "propagator"]
        equation = AxisPattern(
            (prop_requests[0].subscripts, prop_requests[1].subscripts), (),
        )
        extent = (4, 4, n, 3, n, 3)
        tensor_bytes = int(np.prod(extent)) * np.dtype("complex128").itemsize
        index_bytes = n * n * (2 * 8 + 9 * 16)
        specs = {"p0": TensorSpec(extent, "complex128"),
                 "p1": TensorSpec(extent, "complex128")}
        sources = {
            key: SourceSpec(key, tensor_bytes, host_region_bytes=tensor_bytes,
                            snapshot_bytes=tensor_bytes)
            for key in specs
        }
        for record in ("r0", "r1"):
            sources[record] = SourceSpec(
                record, index_bytes, host_region_bytes=index_bytes,
                metadata={"family": "p2p-record", "backend": "numpy"},
            )
        first, second = _bridges(partition, n)
        problem = ContractionProblem(
            specs, (ContractionTerm(("p0", "p1"), equation),), sources,
            sparse_bridges=(first, second), backend=backend,
        )
        profile = CostProfile()
        state = ContractionState.initial(problem)
        for key in problem.sources:
            state = state.load(key, profile, 1 << 50)
        return state, profile, first, second


class SparseCyclePartitionTests(_SparseCyclePartitionFixture):
    def test_join_action_is_tagged_costed_and_survives_rebind_and_archive(self):
        state, profile, _first, _second = self._planned_cycle((0, 1, 2, 0))
        planned = state.contract(0, (0, 1), profile, 1 << 50)
        action = planned.actions[-1]
        self.assertEqual(action.join_id, "bounded_coo_join")
        features = _model_features(planned, action)
        self.assertEqual(features["sparse_operation"], "bounded_coo_join")
        self.assertEqual(features["cost_status"], "conservative_upper_bound_unmeasured")
        self.assertEqual(features["join_index_block_size"], 64)
        self.assertGreater(features["estimated_join_workspace_bytes"], 0)
        self.assertEqual(features["pair_work_upper_bound"], 3**4)

        roundtrip = decode_plan_value(encode_plan_value(action))
        self.assertEqual(roundtrip.join_id, "bounded_coo_join")
        rebinding = {key: key for key in planned.problem.sources}
        rebound = rebind_scene_path(
            planned, planned.problem, rebinding, profile=profile, memory_limit=1 << 50,
        )
        self.assertEqual(rebound.actions[-1].join_id, "bounded_coo_join")
        damaged = replace(
            planned,
            actions=planned.actions[:-1] + (replace(action, join_id=""),),
        )
        with self.assertRaisesRegex(ValueError, "contraction semantics"):
            rebind_scene_path(
                damaged, planned.problem, rebinding, profile=profile, memory_limit=1 << 50,
            )

    def test_non_numpy_join_is_rejected_during_state_planning_before_reads(self):
        state, profile, _first, _second = self._planned_cycle(
            (0, 1, 2, 0), backend="cupy",
        )
        with self.assertRaisesRegex(NotImplementedError, "CPU bounded COO backend"):
            state.contract(0, (0, 1), profile, 1 << 50)

    def test_unmeasured_join_timings_do_not_fit_legacy_cycle_costs(self):
        join_timing = ActionTiming(
            "sparse_bridge", 1.0, 1.1, operands=("p0", "p1", "r0", "r1"),
            model_features={
                "sparse_operation": "bounded_coo_join",
                "join_id": "bounded_coo_join",
                "cost_status": "conservative_upper_bound_unmeasured",
                "kernel_calls": 81, "dtype": "complex128",
                "flops": 80_000, "traffic_bytes": 4_096,
            },
        )
        cycle_timing = ActionTiming(
            "sparse_bridge", 1.0, 0.9, operands=("p0", "p1", "r0", "r1"),
            model_features={
                "kernel_calls": 2, "dtype": "complex128",
                "flops": 4_000, "traffic_bytes": 2_048,
            },
        )
        report = ReplayReport((), (join_timing, cycle_timing), None, None)
        samples = execution_cost_samples(report)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["category"], "sparse_contraction")
        profile = CostProfile()
        fitted = update_cost_profile(
            profile, report, verified=True, hardware_fingerprint="oracle-cpu",
        )
        self.assertEqual(fitted.sample_count, 1)

        join_only = ReplayReport((), (join_timing,), None, None)
        with self.assertRaisesRegex(ValueError, "bounded_coo_join timings are explicitly unmeasured"):
            update_cost_profile(
                profile, join_only, verified=True, hardware_fingerprint="oracle-cpu",
            )


if __name__ == "__main__":
    unittest.main()
