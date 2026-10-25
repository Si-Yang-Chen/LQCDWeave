from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import (
    Action,
    AxisPattern,
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    TensorSpec,
)
from lqcdweave.contracts.correlator.dense_batch import validate_dense_batches
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.execution.operations.dense_batch import contract_dense_batch
from lqcdweave.contraction.performance.api import CostProfile, predict_plan_cost
from lqcdweave.contraction.planning.api import (
    describe_dense_batch,
    find_dense_batches,
    lower_dense_batches,
    plan_scene_storage,
    solve_storage,
)
from lqcdweave.storage.contraction_plans.api import (
    decode_plan_value,
    encode_plan_value,
    load_storage_schedule,
    save_storage_schedule,
)


class ArrayLoader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []

    def read_block(self, *args):
        raise AssertionError("these fixtures have no physical reads")

    def materialize(self, key, *args):
        self.reads.append(key)
        return self.arrays[key]

    def close_all(self):
        pass


class ReductionBatchTests(unittest.TestCase):
    def _grid(self, left_shapes, right_shapes, equation, output_shape, *,
              left_dtype=np.complex128, right_dtype=np.complex128,
              output_dtype=np.complex128, left_dtypes=None, right_dtypes=None):
        specs = {}
        left_keys = []
        right_keys = []
        left_dtypes = tuple(left_dtypes) if left_dtypes is not None else (left_dtype,) * len(left_shapes)
        right_dtypes = tuple(right_dtypes) if right_dtypes is not None else (right_dtype,) * len(right_shapes)
        if len(left_dtypes) != len(left_shapes) or len(right_dtypes) != len(right_shapes):
            raise ValueError("each grid operand needs exactly one dtype")
        for index, shape in enumerate(left_shapes):
            key = f"left_{index}"
            left_keys.append(key)
            specs[key] = TensorSpec(shape, left_dtypes[index])
        for index, shape in enumerate(right_shapes):
            key = f"right_{index}"
            right_keys.append(key)
            specs[key] = TensorSpec(shape, right_dtypes[index])

        pattern = AxisPattern(*equation)
        actions = []
        for i, left in enumerate(left_keys):
            for j, right in enumerate(right_keys):
                key = f"out_{i}_{j}"
                shape = output_shape(left_shapes[i], right_shapes[j])
                specs[key] = TensorSpec(shape, output_dtype)
                actions.append(Action("contract", key, (left, right), pattern, seconds=10.0))
        return actions, specs, tuple(left_keys), tuple(right_keys)

    @staticmethod
    def _random_inputs(specs, *, seed=80, reverse_right=False):
        rng = np.random.default_rng(seed)
        arrays = {}
        for key, spec in specs.items():
            if not key.startswith(("left_", "right_")):
                continue
            value = (rng.standard_normal(spec.shape) + 1j * rng.standard_normal(spec.shape)).astype(spec.dtype)
            if reverse_right and key.startswith("right_") and value.ndim:
                value = value[..., ::-1]
            arrays[key] = value
        return arrays

    @staticmethod
    def _expected(actions, arrays):
        expected = {}
        for action in actions:
            left, right = action.operands
            axes_left, axes_right = action.equation.inputs
            expected[action.tensor] = np.einsum(
                arrays[left], axes_left, arrays[right], axes_right,
                action.equation.output, optimize=False,
            )
        return expected

    def test_left_side_without_free_axis_batches_uniform_dtype_reverse_strides(self):
        actions, specs, left_keys, right_keys = self._grid(
            ((2, 3), (2, 3)),
            ((4, 3, 2), (4, 3, 2), (4, 3, 2)),
            (((0, 1), (2, 1, 0)), (2,)),
            lambda _left, right: (right[0],),
            left_dtypes=(np.complex128, np.complex128),
            right_dtypes=(np.complex128, np.complex128, np.complex128),
        )
        batch = describe_dense_batch(actions, tuple(range(6)), specs, CostProfile())
        self.assertIsNotNone(batch)
        self.assertEqual(batch.stacked_sides, (True, False))
        self.assertEqual(batch.packing_dtype, "complex128")
        self.assertEqual(batch.packing_bytes, 2 * 2 * 3 * 16 + 3 * 4 * 3 * 2 * 16)
        self.assertEqual((batch.left_free_rank, batch.reduction_rank, batch.right_free_rank), (1, 2, 1))
        self.assertEqual(tuple(result.squeeze_axes for result in batch.results), ((0,),) * 6)

        arrays = self._random_inputs(specs, reverse_right=True)
        originals = {key: value.copy() for key, value in arrays.items()}
        expected = self._expected(actions, arrays)
        with patch.object(np, "einsum", side_effect=AssertionError("batch must lower to one GEMM")), \
             patch.object(np, "matmul", wraps=np.matmul) as multiply:
            owner = contract_dense_batch(batch, arrays)
        self.assertEqual(multiply.call_count, 1)
        self.assertEqual(owner.shape, (2, 12))
        for action in actions:
            actual = arrays[action.tensor]
            self.assertEqual(actual.shape, specs[action.tensor].shape)
            self.assertEqual(actual.dtype, np.dtype(np.complex128))
            self.assertFalse(any(np.shares_memory(actual, arrays[key]) for key in (*left_keys, *right_keys)))
            np.testing.assert_allclose(actual, expected[action.tensor], rtol=2e-12, atol=2e-12)
        for key, original in originals.items():
            np.testing.assert_array_equal(arrays[key], original)

        survivor = arrays["out_1_2"].copy()
        del arrays["out_0_0"]
        del arrays["out_0_1"]
        np.testing.assert_array_equal(arrays["out_1_2"], survivor)

    def test_right_side_without_free_axis_preserves_logical_output_shape(self):
        actions, specs, _left_keys, right_keys = self._grid(
            ((2, 3, 2), (3, 3, 2)),
            ((2, 3), (2, 3), (2, 3)),
            (((0, 1, 2), (2, 1)), (0,)),
            lambda left, _right: (left[0],),
            left_dtype=np.complex128,
            right_dtype=np.complex128,
            output_dtype=np.complex128,
        )
        batch = describe_dense_batch(actions, tuple(range(6)), specs, CostProfile())
        self.assertIsNotNone(batch)
        self.assertEqual(batch.stacked_sides, (False, True))
        self.assertEqual(tuple(result.squeeze_axes for result in batch.results), ((1,),) * 6)

        arrays = self._random_inputs(specs, seed=81, reverse_right=True)
        expected = self._expected(actions, arrays)
        with patch.object(np, "einsum", side_effect=AssertionError("batch must lower to one GEMM")), \
             patch.object(np, "matmul", wraps=np.matmul) as multiply:
            contract_dense_batch(batch, arrays)
        self.assertEqual(multiply.call_count, 1)
        for action in actions:
            self.assertEqual(arrays[action.tensor].shape, specs[action.tensor].shape)
            self.assertEqual(arrays[action.tensor].dtype, np.dtype(np.complex128))
            np.testing.assert_allclose(arrays[action.tensor], expected[action.tensor], rtol=2e-12, atol=2e-12)
        self.assertTrue(any(value.strides[-1] < 0 for key, value in arrays.items()
                            if key in right_keys and value.ndim))

    def test_both_sides_without_free_axis_produce_scalars(self):
        actions, specs, _left_keys, _right_keys = self._grid(
            ((2, 3), (2, 3)),
            ((3, 2), (3, 2), (3, 2)),
            (((0, 1), (1, 0)), ()),
            lambda _left, _right: (),
        )
        batch = describe_dense_batch(actions, tuple(range(6)), specs, CostProfile())
        self.assertIsNotNone(batch)
        self.assertEqual(batch.stacked_sides, (True, True))
        self.assertEqual(tuple(result.squeeze_axes for result in batch.results), ((0, 1),) * 6)

        arrays = self._random_inputs(specs, seed=82)
        expected = self._expected(actions, arrays)
        with patch.object(np, "einsum", side_effect=AssertionError("batch must lower to one GEMM")), \
             patch.object(np, "matmul", wraps=np.matmul) as multiply:
            contract_dense_batch(batch, arrays)
        self.assertEqual(multiply.call_count, 1)
        for action in actions:
            self.assertEqual(arrays[action.tensor].shape, ())
            np.testing.assert_allclose(arrays[action.tensor], expected[action.tensor], rtol=2e-12, atol=2e-12)

    def test_equal_shape_disconnected_edges_and_missing_cartesian_pair_are_rejected(self):
        disconnected, disconnected_specs, _, _ = self._grid(
            ((2, 3), (2, 3)),
            ((3, 2), (3, 2)),
            (((0, 1), (1, 0)), ()),
            lambda _left, _right: (),
        )
        state = SimpleNamespace(
            actions=(disconnected[0], disconnected[3]),
            catalog=disconnected_specs,
            results=(disconnected[0].tensor, disconnected[3].tensor),
            problem=SimpleNamespace(backend="numpy"),
        )
        self.assertEqual(find_dense_batches(state, CostProfile()), ())

        complete, specs, _, _ = self._grid(
            ((2, 3), (2, 3)),
            ((4, 3, 2), (4, 3, 2), (4, 3, 2)),
            (((0, 1), (2, 1, 0)), (2,)),
            lambda _left, right: (right[0],),
        )
        self.assertIsNone(describe_dense_batch(complete[:-1], tuple(range(5)), specs, CostProfile()))

    def test_physical_size_one_axes_are_not_virtual_stack_axes(self):
        actions, specs, _left_keys, _right_keys = self._grid(
            ((1, 3), (2, 3)),
            ((1, 3), (2, 3), (3, 3)),
            (((0, 1), (2, 1)), (0, 2)),
            lambda left, right: (left[0], right[0]),
            left_dtype=np.complex128,
            right_dtype=np.complex128,
            output_dtype=np.complex128,
        )
        batch = describe_dense_batch(actions, tuple(range(6)), specs, CostProfile())
        self.assertIsNotNone(batch)
        self.assertEqual(batch.stacked_sides, (False, False))
        self.assertEqual(tuple(result.squeeze_axes for result in batch.results), ((),) * 6)
        self.assertEqual(specs["out_0_0"].shape, (1, 1))

        arrays = self._random_inputs(specs, seed=83)
        expected = self._expected(actions, arrays)
        contract_dense_batch(batch, arrays)
        self.assertEqual(arrays["out_0_0"].shape, (1, 1))
        for action in actions:
            np.testing.assert_allclose(arrays[action.tensor], expected[action.tensor], rtol=2e-12, atol=2e-12)

    def test_legacy_batch_and_result_records_restore_new_fields_to_defaults(self):
        actions, specs, _, _ = self._grid(
            ((2, 3), (3, 3)),
            ((3, 4), (3, 5)),
            (((0, 1), (1, 2)), (0, 2)),
            lambda left, right: (left[0], right[1]),
            left_dtype=np.complex128,
            right_dtype=np.complex128,
            output_dtype=np.complex128,
        )
        batch = describe_dense_batch(actions, tuple(range(4)), specs, CostProfile())
        self.assertIsNotNone(batch)
        self.assertEqual(batch.stacked_sides, (False, False))
        encoded = encode_plan_value(batch)
        encoded["value"].pop("stacked_sides")
        for result in encoded["value"]["results"]["value"]:
            result["value"].pop("squeeze_axes")
        restored = decode_plan_value(encoded)
        self.assertEqual(restored.stacked_sides, (False, False))
        self.assertEqual(tuple(result.squeeze_axes for result in restored.results), ((),) * 4)
        self.assertEqual(restored, batch)

    def test_prepared_side_reuses_uniform_dtype_before_numpy_replay(self):
        actions, specs, _, _ = self._grid(
            ((2, 3),),
            ((4, 3, 2), (4, 3, 2)),
            (((0, 1), (2, 1, 0)), (2,)),
            lambda _left, right: (right[0],),
            left_dtype=np.complex128,
            right_dtype=np.complex128,
            output_dtype=np.complex128,
        )
        requested_layout = {"left_0": (0, 1)}
        batch = describe_dense_batch(
            actions, (0, 1), specs, CostProfile(), backend="cupy",
            prepared_inputs=requested_layout,
        )
        self.assertIsNotNone(batch)
        self.assertEqual(batch.prepared_sides, (True, False))
        self.assertEqual(batch.packing_dtype, "complex128")
        self.assertEqual(batch.packing_bytes, 2 * 4 * 3 * 2 * 16)

        arrays = self._random_inputs(specs, seed=86)
        expected = self._expected(actions, arrays)
        contract_dense_batch(batch, arrays, backend=np)
        for action in actions:
            np.testing.assert_allclose(arrays[action.tensor], expected[action.tensor], rtol=2e-12, atol=2e-12)

    def _storage_problem(self):
        specs = {
            "left_0": TensorSpec((2, 3), np.complex128),
            "left_1": TensorSpec((2, 3), np.complex128),
            "right_0": TensorSpec((4, 3, 2), np.complex128),
            "right_1": TensorSpec((4, 3, 2), np.complex128),
            "right_2": TensorSpec((4, 3, 2), np.complex128),
        }
        terms = tuple(
            ContractionTerm((f"left_{i}", f"right_{j}"), "ab,cba->c")
            for i in range(2) for j in range(3)
        )
        sources = {key: SourceSpec(key, 0, read_bytes=0) for key in specs}
        problem = ContractionProblem(specs, terms, sources, backend="numpy")
        profile = CostProfile(
            kernel_call_seconds=1.0,
            kernel_call_seconds_complex64=1.0,
            kernel_call_seconds_complex128=1.0,
            flop_seconds=0.0,
            traffic_seconds_per_byte=0.0,
            snapshot_call_seconds=0.0,
            snapshot_seconds_per_byte=0.0,
        )
        preparations = {key: SourcePreparation(()) for key in specs}
        result = solve_storage(
            problem,
            {},
            preparations,
            optimizer="anytime",
            kernel_batching=True,
            profile=profile,
            memory_limit=1 << 30,
            deadline_seconds=0.5,
            max_states=10000,
            root_width=1,
            branch_width=1,
            component_workers=1,
        )
        return problem, profile, result.plan

    def test_storage_cap_and_full_scene_archive_preserve_stacked_batch(self):
        problem, profile, planned = self._storage_problem()
        self.assertTrue(planned.kernel_batches)
        self.assertEqual(planned.kernel_batches[0].stacked_sides, (True, False))
        validate_dense_batches(planned)

        plain = plan_scene_storage(
            planned.state,
            planned.blocks,
            planned.preparations,
            profile=profile,
            memory_limit=1 << 30,
        )
        candidate, lowering = lower_dense_batches(plain, profile=profile)
        self.assertTrue(lowering["resource_audited"])
        self.assertEqual(candidate.kernel_batches[0].stacked_sides, (True, False))
        self.assertLessEqual(plain.peak_host_bytes, candidate.peak_host_bytes)
        self.assertGreater(candidate.peak_host_bytes, plain.peak_host_bytes)

        limited = replace(plain, memory_limit=candidate.peak_host_bytes - 1)
        rejected, rejection = lower_dense_batches(limited, profile=profile)
        self.assertFalse(rejected.kernel_batches)
        self.assertEqual(rejection["reason"], "storage_limit")
        self.assertGreater(rejection["candidate_peak_host_bytes"], limited.memory_limit)

        shapes = {key: list(spec.shape) for key, spec in problem.specs.items()}
        hardware = {"backend": "numpy", "memory": candidate.memory_limit}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stacked-reduction-plan.json"
            save_storage_schedule(path, (candidate,), shapes=shapes, hardware=hardware)
            restored, _ = load_storage_schedule(
                path,
                shapes=shapes,
                hardware=hardware,
                expected=(problem,),
                blocks={},
                preparations=candidate.preparations,
            )
        restored_plan = restored[0]
        self.assertEqual(restored_plan.kernel_batches, candidate.kernel_batches)
        self.assertEqual(restored_plan.kernel_batches[0].stacked_sides, (True, False))
        self.assertEqual(restored_plan.kernel_batches[0].results[0].squeeze_axes, (0,))
        validate_dense_batches(restored_plan)
        before = predict_plan_cost((candidate,), profile, workload={}, provenance={})
        after = predict_plan_cost((restored_plan,), profile, workload={}, provenance={})
        self.assertEqual(after, before)

        rng = np.random.default_rng(84)
        arrays = {
            key: (rng.standard_normal(spec.shape) + 1j * rng.standard_normal(spec.shape)).astype(spec.dtype)
            for key, spec in problem.specs.items()
        }
        values = replay_scene_storage(restored_plan, ArrayLoader(arrays))
        for term, value in zip(problem.terms, values):
            expected = np.einsum("ab,cba->c", arrays[term.inputs[0]], arrays[term.inputs[1]],
                                 optimize=False)
            np.testing.assert_allclose(value, expected, rtol=2e-12, atol=2e-12)


class ReductionBatchCupyTests(unittest.TestCase):
    def test_actual_cupy_oracle_for_high_rank_uniform_dtype_reduction(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("CuPy is installed but no CUDA device is visible")
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"CuPy CUDA unavailable: {exc}")

        actions, specs, left_keys, right_keys = ReductionBatchTests()._grid(
            ((4, 4, 16, 16), (4, 4, 16, 16)),
            ((9, 4, 4, 16, 16), (9, 4, 4, 16, 16), (9, 4, 4, 16, 16)),
            (((0, 1, 2, 3), (4, 1, 0, 3, 2)), (4,)),
            lambda _left, right: (right[0],),
            left_dtypes=(np.complex128, np.complex128),
            right_dtypes=(np.complex128, np.complex128, np.complex128),
        )
        batch = describe_dense_batch(actions, tuple(range(6)), specs, CostProfile(), backend="cupy")
        self.assertIsNotNone(batch)
        self.assertEqual(batch.stacked_sides, (True, False))
        self.assertEqual(batch.packing_dtype, "complex128")
        self.assertEqual(batch.packing_bytes, 2 * 4 * 4 * 16 * 16 * 16 + 3 * 9 * 4 * 4 * 16 * 16 * 16)

        host = ReductionBatchTests()._random_inputs(specs, seed=85)
        expected = ReductionBatchTests()._expected(actions, host)
        arrays = {key: cp.asarray(value) for key, value in host.items()}
        contract_dense_batch(batch, arrays, backend=cp)
        cp.cuda.get_current_stream().synchronize()
        for action in actions:
            actual = cp.asnumpy(arrays[action.tensor])
            self.assertEqual(actual.shape, specs[action.tensor].shape)
            self.assertEqual(actual.dtype, np.dtype(np.complex128))
            np.testing.assert_allclose(actual, expected[action.tensor], rtol=2e-12, atol=2e-12)
        self.assertEqual(arrays[left_keys[0]].dtype, np.dtype(np.complex128))
        self.assertEqual(arrays[left_keys[1]].dtype, np.dtype(np.complex128))
        self.assertTrue(all(arrays[key].dtype == np.dtype(np.complex128) for key in right_keys))


if __name__ == "__main__":
    unittest.main()
