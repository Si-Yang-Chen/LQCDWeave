"""Checks for splitting complete dense grids along several varying free axes."""
from dataclasses import replace
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import math

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action, AxisPattern, ContractionProblem, ContractionTerm, DenseBatchOperand, SourcePreparation,
    SourceSpec, TensorSpec, validate_executable_resources,
)
from lqcdweave.contraction.execution.operations.dense_batch import contract_dense_batch
from lqcdweave.contraction.planning.api import (
    CostProfile, find_dense_batches, lower_dense_batches, plan_scene_storage, solve_storage,
)
from lqcdweave.contraction.planning import dense_batches


def _grid(left_free_shapes, right_free_shapes, *, pairs=None, seconds=1.0, dtype=np.float64, inner=2):
    """Make independent paid actions for one explicitly contracted Cartesian grid."""
    left_rank = len(left_free_shapes[0])
    right_rank = len(right_free_shapes[0])
    left_axes = tuple(range(left_rank + 1))
    right_free_axes = tuple(range(left_rank + 1, left_rank + 1 + right_rank))
    right_axes = (left_rank, *right_free_axes)
    pattern = AxisPattern((left_axes, right_axes), tuple(range(left_rank)) + right_free_axes)
    catalog = {}
    for i, shape in enumerate(left_free_shapes):
        catalog[f"left-{i}"] = TensorSpec((*shape, inner), dtype)
    for j, shape in enumerate(right_free_shapes):
        catalog[f"right-{j}"] = TensorSpec((inner, *shape), dtype)

    if pairs is None:
        pairs = tuple((i, j) for i in range(len(left_free_shapes))
                      for j in range(len(right_free_shapes)))
    actions, outputs = [], []
    for edge, (i, j) in enumerate(pairs):
        output = f"out-{edge}"
        shape = (*left_free_shapes[i], *right_free_shapes[j])
        catalog[output] = TensorSpec(shape, dtype)
        outputs.append(output)
        actions.append(Action("contract", output, (f"left-{i}", f"right-{j}"),
                              pattern, seconds=seconds))
    state = SimpleNamespace(actions=tuple(actions), catalog=catalog, results=tuple(outputs),
                            problem=SimpleNamespace(backend="numpy"))
    return state, pattern


def _producer_tradeoff_state(*, backend="cupy", extra_margin_scale=0.25):
    """Build an upstream multi-axis grid feeding a direct one-by-two batch."""
    upstream, _ = _grid(((2,),), ((3, 4), (3, 5), (7, 4), (7, 5)),
                        seconds=100.0, dtype=np.float64)
    upstream_actions = tuple(replace(action, tensor=f"x{i}")
                             for i, action in enumerate(upstream.actions))
    catalog = {key: value for key, value in upstream.catalog.items()
               if not key.startswith("out-")}
    catalog.update({f"x{i}": upstream.catalog[f"out-{i}"] for i in range(4)})

    downstream, _ = _grid(((2, 3),), ((5,), (6,)), seconds=100.0,
                          dtype=np.float64, inner=4)
    downstream_actions = tuple(replace(
        action, tensor=f"y{i}",
        operands=("x0", f"consumer-{i}"),
    ) for i, action in enumerate(downstream.actions))
    catalog.update({f"consumer-{i}": downstream.catalog[f"right-{i}"] for i in range(2)})
    catalog.update({f"y{i}": downstream.catalog[f"out-{i}"] for i in range(2)})
    actions = (*upstream_actions, *downstream_actions)
    state = SimpleNamespace(actions=actions, catalog=catalog, results=("y0", "y1"),
                            problem=SimpleNamespace(backend=backend))

    profile = CostProfile(traffic_seconds_per_byte=2**-10, flop_seconds=0.0,
                          kernel_call_seconds_float64=2**-16)
    entries = tuple((index, state.actions[index],
                     dense_batches._descriptor(state.actions[index], catalog)) for index in range(4))
    groups = dense_batches._subgrid_action_groups(entries)
    raw_extras = tuple(dense_batches.describe_dense_batch(
        state.actions, group, catalog, profile, backend=backend) for group in groups)
    raw_downstream = dense_batches.describe_dense_batch(
        state.actions, (4, 5), catalog, profile, backend=backend)
    prepared_downstream = dense_batches.describe_dense_batch(
        state.actions, (4, 5), catalog, profile, backend=backend,
        prepared_inputs={"x0": (0, 1, 2)})
    if not raw_extras or any(batch is None for batch in raw_extras):
        raise AssertionError("tradeoff fixture must produce profitable upstream subgrids")
    if raw_downstream is None or prepared_downstream is None:
        raise AssertionError("tradeoff fixture must produce the downstream baseline batch")
    producer_saving = raw_downstream.seconds - prepared_downstream.seconds
    if producer_saving <= 0:
        raise AssertionError("tradeoff fixture must have a positive producer-layout saving")

    changed = list(actions)
    margin = producer_saving * extra_margin_scale
    for group, batch in zip(groups, raw_extras):
        per_action = (batch.seconds + margin) / len(group)
        for index in group:
            changed[index] = replace(changed[index], seconds=per_action)
    state.actions = tuple(changed)
    return state, profile, groups, raw_extras, producer_saving


def _random_inputs(state, seed=19):
    rng = np.random.default_rng(seed)
    arrays = {}
    for key, spec in state.catalog.items():
        if key.startswith(("left-", "right-")):
            value = rng.normal(size=spec.shape)
            if spec.dtype.kind == 'c':
                value = value + 1j * rng.normal(size=spec.shape)
            arrays[key] = value.astype(spec.dtype)
    return arrays


def _assert_independent_einsum(test, state, pattern, arrays):
    for action in state.actions:
        expected = np.einsum(arrays[action.operands[0]], pattern.inputs[0],
                             arrays[action.operands[1]], pattern.inputs[1], pattern.output,
                             optimize=False)
        np.testing.assert_allclose(arrays[action.tensor], expected, rtol=2e-12, atol=2e-12)


class DenseBatchSubgridTests(unittest.TestCase):
    def test_one_side_multiple_varying_axes_splits_stably_and_executes_on_numpy(self):
        state, pattern = _grid(((2, 3), (2, 4), (5, 3), (5, 4)), ((6,), (7,)))
        batches = find_dense_batches(state, CostProfile())

        self.assertEqual(tuple(batch.action_indices for batch in batches),
                         ((0, 1, 2, 3), (4, 5, 6, 7)))
        members = tuple(index for batch in batches for index in batch.action_indices)
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(set(members), set(range(len(state.actions))))

        arrays = _random_inputs(state)
        for batch in batches:
            contract_dense_batch(batch, arrays, backend=np)
        _assert_independent_einsum(self, state, pattern, arrays)

    def test_both_sides_multiple_varying_axes_form_complete_subgrids(self):
        state, pattern = _grid(
            ((2, 3), (2, 4), (5, 3), (5, 4)),
            ((6, 2), (6, 3), (8, 2), (8, 3)),
            dtype=np.complex128,
        )
        batches = find_dense_batches(state, CostProfile())

        self.assertEqual(tuple(batch.action_indices for batch in batches),
                         ((0, 1, 4, 5), (2, 3, 6, 7),
                          (8, 9, 12, 13), (10, 11, 14, 15)))
        members = tuple(index for batch in batches for index in batch.action_indices)
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(set(members), set(range(len(state.actions))))

        arrays = _random_inputs(state, seed=23)
        for batch in batches:
            contract_dense_batch(batch, arrays, backend=np)
        _assert_independent_einsum(self, state, pattern, arrays)

    def test_incomplete_duplicate_zero_sized_and_unprofitable_grids_stay_unbatched(self):
        left = ((2, 3), (2, 4), (5, 3), (5, 4))
        right = ((6,), (7,))
        full_pairs = tuple((i, j) for i in range(4) for j in range(2))

        incomplete = full_pairs[:-1]
        state, _ = _grid(left, right, pairs=incomplete)
        self.assertEqual(find_dense_batches(state, CostProfile()), ())

        duplicate = (*full_pairs[:-1], full_pairs[0])
        state, _ = _grid(left, right, pairs=duplicate)
        self.assertEqual(find_dense_batches(state, CostProfile()), ())

        zero_sized_left = ((2, 3), (2, 4), (5, 3), (5, 0))
        state, _ = _grid(zero_sized_left, right)
        self.assertEqual(find_dense_batches(state, CostProfile()), ())

        state, _ = _grid(left, right, seconds=0.0)
        self.assertEqual(find_dense_batches(state, CostProfile()), ())

    def test_lowering_a_subgrid_plan_runs_existing_resource_audit(self):
        left_shapes = ((2, 3), (2, 4), (5, 3), (5, 4))
        right_shapes = ((6,), (7,))
        state, pattern = _grid(left_shapes, right_shapes)
        specs = {key: value for key, value in state.catalog.items()
                 if key.startswith(("left-", "right-"))}
        terms = tuple(ContractionTerm(action.operands, pattern) for action in state.actions)
        sources = {key: SourceSpec(key, 0, read_bytes=0) for key in specs}
        problem = ContractionProblem(specs, terms, sources, backend="numpy")
        preparations = {key: SourcePreparation(()) for key in specs}
        profile = CostProfile()
        result = solve_storage(problem, {}, preparations, optimizer="anytime", profile=profile,
                               memory_limit=1 << 24, deadline_seconds=0.5, max_states=10000,
                               root_width=1, branch_width=1, component_workers=1)
        self.assertIsNotNone(result.plan)
        plain = plan_scene_storage(result.plan.state, result.plan.blocks, result.plan.preparations,
                                   profile=profile, memory_limit=1 << 24)

        lowered, report = lower_dense_batches(plain, profile=profile)

        self.assertTrue(report["applied"])
        self.assertEqual(len(lowered.kernel_batches), 2)
        self.assertTrue(report["resource_audited"])
        validate_executable_resources(lowered)

    def _find_with_eligibility_calls(self, state, profile):
        calls = Counter()
        original = dense_batches.dense_batch_producer_eligible

        def count_call(operand, actions, catalog):
            calls[operand] += 1
            return original(operand, actions, catalog)

        with patch.object(dense_batches, "dense_batch_producer_eligible", side_effect=count_call):
            batches = find_dense_batches(state, profile)
        return batches, calls

    def test_cupy_keeps_baseline_when_extra_local_profit_loses_more_producer_savings(self):
        state, profile, groups, extra_batches, producer_saving = _producer_tradeoff_state(
            extra_margin_scale=0.25)
        extra_delta = math.fsum(
            batch.seconds - math.fsum(state.actions[index].seconds for index in batch.action_indices)
            for batch in extra_batches)
        self.assertLess(extra_delta, 0.0)
        self.assertGreater(producer_saving + extra_delta, 0.0)

        batches, calls = self._find_with_eligibility_calls(state, profile)

        self.assertEqual(tuple(batch.action_indices for batch in batches), ((4, 5),))
        self.assertEqual(batches[0].prepared_sides, (True, False))
        self.assertEqual(calls, Counter({
            DenseBatchOperand("x0", (0, 1, 2)): 1,
            DenseBatchOperand("left-0", (0, 1)): 1,
        }))

    def test_cupy_adds_subgrids_when_the_combined_cost_improves(self):
        state, profile, groups, extra_batches, producer_saving = _producer_tradeoff_state(
            extra_margin_scale=2.0)
        extra_delta = math.fsum(
            batch.seconds - math.fsum(state.actions[index].seconds for index in batch.action_indices)
            for batch in extra_batches)
        self.assertLess(producer_saving + extra_delta, 0.0)

        batches, calls = self._find_with_eligibility_calls(state, profile)

        self.assertEqual(tuple(batch.action_indices for batch in batches),
                         (*groups, (4, 5)))
        downstream = next(batch for batch in batches if batch.action_indices == (4, 5))
        self.assertEqual(downstream.prepared_sides, (False, False))
        self.assertEqual(calls, Counter({
            DenseBatchOperand("x0", (0, 1, 2)): 1,
            DenseBatchOperand("left-0", (0, 1)): 1,
        }))

    def test_cupy_keeps_baseline_on_equal_combined_cost(self):
        state, profile, _, extra_batches, producer_saving = _producer_tradeoff_state(
            extra_margin_scale=0.5)
        extra_delta = math.fsum(
            batch.seconds - math.fsum(state.actions[index].seconds for index in batch.action_indices)
            for batch in extra_batches)
        self.assertEqual(producer_saving + extra_delta, 0.0)
        batches = find_dense_batches(state, profile)
        self.assertEqual(tuple(batch.action_indices for batch in batches), ((4, 5),))
        self.assertEqual(batches[0].prepared_sides, (True, False))

    def test_numpy_keeps_direct_batches_and_subgrids_without_producer_checks(self):
        state, profile, groups, _, _ = _producer_tradeoff_state(
            backend="numpy", extra_margin_scale=0.25)
        with patch.object(dense_batches, "dense_batch_producer_eligible",
                          side_effect=AssertionError("NumPy has no producer-layout eligibility pass")):
            batches = find_dense_batches(state, profile)

        self.assertEqual(tuple(batch.action_indices for batch in batches), (*groups, (4, 5)))


class DenseBatchSubgridCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')
        cls.backend = cupy

    def test_actual_cupy_mixed_dtype_subgrids_match_independent_einsum(self):
        cp = self.backend
        state, pattern = _grid(((2, 3), (2, 4), (5, 3), (5, 4)),
                              ((6, 2), (6, 3), (8, 2), (8, 3)), dtype=np.complex128)
        state.problem.backend = 'cupy'
        for key, spec in tuple(state.catalog.items()):
            if key.startswith('left-'):
                state.catalog[key] = TensorSpec(spec.shape, np.complex64)
        batches = find_dense_batches(state, CostProfile())
        self.assertEqual(len(batches), 4)
        host = _random_inputs(state, seed=49)
        arrays = {key: cp.asarray(value) for key, value in host.items()}
        for batch in batches:
            contract_dense_batch(batch, arrays, backend=cp)
        actual = {**host, **{key: cp.asnumpy(arrays[key]) for key in state.results}}
        _assert_independent_einsum(self, state, pattern, actual)


if __name__ == "__main__":
    unittest.main()
