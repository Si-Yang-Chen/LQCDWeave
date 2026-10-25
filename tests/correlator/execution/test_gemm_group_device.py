"""Small real-CUDA checks for the fixed independent dense GEMM group."""
import gc
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import contextmanager

import numpy as np

from lqcdweave.contracts.correlator.api import (
    DenseBatchOperand,
    DenseBatchResult,
    DenseGemmGroup,
    DenseKernelBatch,
    TensorSpec,
)
from lqcdweave.contraction.execution.operations.gemm_group import bind_dense_gemm_group
from lqcdweave.contraction.execution.diagnostics import execution_debug


cp = None


def _assert_close(actual, expected, dtype):
    dtype = np.dtype(dtype)
    rtol, atol = (2e-12, 2e-12) if dtype in (np.dtype(np.float64), np.dtype(np.complex128)) else (3e-5, 3e-6)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def _group(dtype, *, mixed=False):
    dtype = np.dtype(dtype)
    catalog = {}
    batches = []
    for lane in range(3):
        left = f"left_{lane}"
        right_a, right_b = f"right_{lane}_a", f"right_{lane}_b"
        out_a, out_b = f"out_{lane}_a", f"out_{lane}_b"
        catalog[left] = TensorSpec((2, 3), dtype)
        catalog[right_a] = TensorSpec((3, 2), np.float32 if mixed and lane == 0 else dtype)
        catalog[right_b] = TensorSpec((3, 2), dtype)
        catalog[out_a] = TensorSpec((2, 2), dtype)
        catalog[out_b] = TensorSpec((2, 2), dtype)
        batches.append(DenseKernelBatch(
            action_indices=(lane * 2, lane * 2 + 1),
            left=(DenseBatchOperand(left, (0, 1)),),
            right=(DenseBatchOperand(right_a, (0, 1)), DenseBatchOperand(right_b, (0, 1))),
            left_free_rank=1,
            reduction_rank=1,
            right_free_rank=1,
            left_axis=0,
            right_axis=1,
            results=(
                DenseBatchResult(out_a, ((0, 2), (0, 2))),
                DenseBatchResult(out_b, ((0, 2), (2, 4))),
            ),
            seconds=0.1,
            packing_bytes=(2 * 3 + 3 * 2 + 3 * 2) * dtype.itemsize,
            output_bytes=2 * 4 * dtype.itemsize,
            packing_dtype=dtype.name,
        ))
    return DenseGemmGroup(tuple(batches), (2, 3, 4), 0.1), catalog


def _multiaxis_group(dtype):
    dtype = np.dtype(dtype)
    catalog = {}
    batches = []
    for lane in range(3):
        left = f"multi_left_{lane}"
        right_a, right_b = f"multi_right_{lane}_a", f"multi_right_{lane}_b"
        out_a, out_b = f"multi_out_{lane}_a", f"multi_out_{lane}_b"
        catalog[left] = TensorSpec((2, 3, 4), dtype)
        catalog[right_a] = TensorSpec((4, 5, 1), dtype)
        catalog[right_b] = TensorSpec((4, 5, 1), dtype)
        catalog[out_a] = TensorSpec((2, 3, 5, 1), dtype)
        catalog[out_b] = TensorSpec((2, 3, 5, 1), dtype)
        batches.append(DenseKernelBatch(
            action_indices=(lane * 2, lane * 2 + 1),
            left=(DenseBatchOperand(left, (0, 1, 2)),),
            right=(DenseBatchOperand(right_a, (0, 1, 2)), DenseBatchOperand(right_b, (0, 1, 2))),
            left_free_rank=2,
            reduction_rank=1,
            right_free_rank=2,
            left_axis=0,
            right_axis=2,
            results=(
                DenseBatchResult(out_a, ((0, 2), (0, 3), (0, 5), (0, 1))),
                DenseBatchResult(out_b, ((0, 2), (0, 3), (0, 5), (1, 2))),
            ),
            seconds=0.1,
            packing_bytes=(24 + 20 + 20) * dtype.itemsize,
            output_bytes=60 * dtype.itemsize,
            packing_dtype=dtype.name,
        ))
    return DenseGemmGroup(tuple(batches), (6, 4, 10), 0.3), catalog


def _fresh_multiaxis_inputs(group, dtype, seed=90211):
    rng = np.random.default_rng(seed)
    arrays, expected = {}, {}
    for lane, batch in enumerate(group.batches):
        left = rng.normal(size=(2, 3, 4)).astype(dtype)
        right_a = rng.normal(size=(4, 5, 1)).astype(dtype)
        right_b = rng.normal(size=(4, 5, 1)).astype(dtype)
        if np.dtype(dtype).kind == "c":
            left = left + (1j * rng.normal(size=left.shape)).astype(dtype)
            right_a = right_a + (1j * rng.normal(size=right_a.shape)).astype(dtype)
            right_b = right_b + (1j * rng.normal(size=right_b.shape)).astype(dtype)
        arrays[batch.left[0].tensor] = cp.asarray(np.ascontiguousarray(left))
        arrays[batch.right[0].tensor] = cp.asarray(np.ascontiguousarray(right_a))
        arrays[batch.right[1].tensor] = cp.asarray(np.ascontiguousarray(right_b))
        product = left.reshape(6, 4) @ np.concatenate((right_a, right_b), axis=2).reshape(4, 10)
        expected[lane] = product.reshape(2, 3, 5, 2)
    return arrays, expected


def _fresh_inputs(group, dtype, *, mixed=False, negative_stride=False, seed=90210):
    rng = np.random.default_rng(seed)
    arrays, expected = {}, {}
    for lane, batch in enumerate(group.batches):
        left = (rng.normal(size=(2, 3))).astype(dtype)
        right_a_dtype = np.float32 if mixed and lane == 0 else dtype
        right_a = rng.normal(size=(3, 2)).astype(right_a_dtype)
        right_b = rng.normal(size=(3, 2)).astype(dtype)
        if np.dtype(dtype).kind == "c":
            left = left + (1j * rng.normal(size=left.shape)).astype(dtype)
            right_a = right_a + (1j * rng.normal(size=right_a.shape)).astype(right_a_dtype)
            right_b = right_b + (1j * rng.normal(size=right_b.shape)).astype(dtype)
        device_left = cp.asarray(left)
        device_right_a = cp.asarray(right_a)
        if negative_stride and lane == 0:
            left, right_a = left[::-1, :], right_a[:, ::-1]
            device_left, device_right_a = device_left[::-1, :], device_right_a[:, ::-1]
        arrays[batch.left[0].tensor] = device_left
        arrays[batch.right[0].tensor] = device_right_a
        arrays[batch.right[1].tensor] = cp.asarray(right_b)
        expected[lane] = left @ np.concatenate((right_a, right_b), axis=1)
    return arrays, expected


def _pointer_mask_group(dtype, *, lanes=3):
    dtype = np.dtype(dtype)
    catalog = {}
    batches = []
    output_bounds = (
        ((0, 2), (0, 3)),
        ((0, 2), (3, 6)),
        ((2, 4), (0, 3)),
        ((2, 4), (3, 6)),
    )
    for lane in range(lanes):
        left_a, left_b = f"pointer_left_{lane}_a", f"pointer_left_{lane}_b"
        right_a, right_b = f"pointer_right_{lane}_a", f"pointer_right_{lane}_b"
        outputs = tuple(f"pointer_out_{lane}_{index}" for index in range(4))
        catalog[left_a] = TensorSpec((2, 3), dtype)
        catalog[left_b] = TensorSpec((2, 3), dtype)
        catalog[right_a] = TensorSpec((3, 3), dtype)
        catalog[right_b] = TensorSpec((3, 3), dtype)
        for key in outputs:
            catalog[key] = TensorSpec((2, 3), dtype)
        batches.append(DenseKernelBatch(
            action_indices=tuple(lane * 4 + index for index in range(4)),
            left=(DenseBatchOperand(left_a, (0, 1)), DenseBatchOperand(left_b, (0, 1))),
            right=(DenseBatchOperand(right_a, (0, 1)), DenseBatchOperand(right_b, (0, 1))),
            left_free_rank=1,
            reduction_rank=1,
            right_free_rank=1,
            left_axis=0,
            right_axis=1,
            results=tuple(DenseBatchResult(key, bounds)
                          for key, bounds in zip(outputs, output_bounds, strict=True)),
            seconds=0.1,
            packing_bytes=30 * dtype.itemsize,
            output_bytes=24 * dtype.itemsize,
            packing_dtype=dtype.name,
        ))
    return DenseGemmGroup(tuple(batches), (4, 3, 6), 0.1 * lanes), catalog


def _pointer_mask_inputs(group, dtype, *, mask, seed, negative_stride=False):
    dtype = np.dtype(dtype)
    if negative_stride and mask != 3:
        raise ValueError("negative-stride inputs are reserved for the grouped/grouped case")
    rng = np.random.default_rng(seed)

    def random_values(shape):
        values = rng.normal(size=shape)
        if dtype.kind == "c":
            values = values + 1j * rng.normal(size=shape)
        return values.astype(dtype)

    arrays, expected = {}, {}
    for lane, batch in enumerate(group.batches):
        left_grouped = bool(mask & 1)
        right_grouped = bool(mask & 2)
        if left_grouped:
            left_host = [random_values((2, 3)) for _ in range(2)]
            left_device = [cp.asarray(value) for value in left_host]
            if negative_stride:
                left_host = [value[::-1, :] for value in left_host]
                left_device = [value[::-1, :] for value in left_device]
        else:
            owner = random_values((4, 3))
            device_owner = cp.asarray(owner)
            left_host = [owner[:2, :], owner[2:, :]]
            left_device = [device_owner[:2, :], device_owner[2:, :]]

        if right_grouped:
            right_host = [random_values((3, 3)) for _ in range(2)]
            right_device = [cp.asarray(value) for value in right_host]
            if negative_stride:
                right_host = [value[:, ::-1] for value in right_host]
                right_device = [value[:, ::-1] for value in right_device]
        else:
            owner = random_values((3, 6))
            device_owner = cp.asarray(owner)
            right_host = [owner[:, :3], owner[:, 3:]]
            right_device = [device_owner[:, :3], device_owner[:, 3:]]

        for operand, value in zip(batch.left, left_device, strict=True):
            arrays[operand.tensor] = value
        for operand, value in zip(batch.right, right_device, strict=True):
            arrays[operand.tensor] = value
        left = np.concatenate(left_host, axis=0)
        right = np.concatenate(right_host, axis=1)
        expected[lane] = left @ right
    return arrays, expected


class CuPyDenseGemmGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global cp
        try:
            import cupy as cupy
        except Exception as error:
            raise unittest.SkipTest(f"CuPy is unavailable; GEMM group GPU checks skipped ({error})") from error
        try:
            is_hip = getattr(cupy.cuda.runtime, "is_hip", False)
            if callable(is_hip):
                is_hip = is_hip()
            if is_hip:
                raise unittest.SkipTest("CuPy HIP is unsupported; GEMM group GPU checks skipped")
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest("CUDA device unavailable; GEMM group GPU checks skipped")
            cupy.cuda.Device()
        except unittest.SkipTest:
            raise
        except Exception as error:
            raise unittest.SkipTest(f"CUDA device unavailable; GEMM group GPU checks skipped ({error})") from error
        cp = cupy

    def test_odd_output_sizes_match_reference_with_aligned_table_prefix(self):
        for scalar in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=np.dtype(scalar)):
                dtype = np.dtype(scalar)
                original, catalog = _group(dtype)
                batches = []
                arrays, expected = {}, []
                rng = np.random.default_rng(27183)
                for batch in original.batches:
                    shapes = ((3, 5), (5, 2), (5, 3))
                    keys = (batch.left[0].tensor, batch.right[0].tensor, batch.right[1].tensor)
                    hosts = []
                    for key, shape in zip(keys, shapes, strict=True):
                        value = rng.normal(size=shape)
                        if dtype.kind == 'c':
                            value = value + 1j*rng.normal(size=shape)
                        value = value.astype(dtype)
                        hosts.append(value)
                        catalog[key] = TensorSpec(shape, dtype)
                        arrays[key] = cp.asarray(value)
                    result_a, result_b = batch.results
                    catalog[result_a.tensor] = TensorSpec((3, 2), dtype)
                    catalog[result_b.tensor] = TensorSpec((3, 3), dtype)
                    batches.append(replace(batch, results=(
                        replace(result_a, bounds=((0, 3), (0, 2))),
                        replace(result_b, bounds=((0, 3), (2, 5)))),
                        packing_bytes=40*dtype.itemsize, output_bytes=15*dtype.itemsize))
                    expected.append(hosts[0] @ np.concatenate(hosts[1:], axis=1))
                group = DenseGemmGroup(tuple(batches), (3, 5, 5), 0.1)
                cube = bind_dense_gemm_group(group, catalog, cp)(arrays, device=True)
                _assert_close(cp.asnumpy(cube), np.stack(expected), dtype)
                self.assertEqual(cube.data.ptr % dtype.alignment, 0)
                for lane, batch in enumerate(batches):
                    _assert_close(cp.asnumpy(arrays[batch.results[0].tensor]), expected[lane][:, :2], dtype)
                    _assert_close(cp.asnumpy(arrays[batch.results[1].tensor]), expected[lane][:, 2:], dtype)

    def test_supported_dtypes_fresh_inputs_and_independent_numpy_reference(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=np.dtype(dtype)):
                group, catalog = _group(dtype)
                bound = bind_dense_gemm_group(group, catalog, cp)
                for seed in (90210, 90211):
                    arrays, expected = _fresh_inputs(group, dtype, seed=seed)
                    cube = bound(arrays, device=True)
                    _assert_close(cp.asnumpy(cube), np.stack(tuple(expected.values())), dtype)
                    for lane, batch in enumerate(group.batches):
                        expected_a, expected_b = np.split(expected[lane], (2,), axis=1)
                        _assert_close(cp.asnumpy(arrays[batch.results[0].tensor]), expected_a, dtype)
                        _assert_close(cp.asnumpy(arrays[batch.results[1].tensor]), expected_b, dtype)
                    with execution_debug(True):
                        checked = bound(arrays, device=True)
                    np.testing.assert_array_equal(cp.asnumpy(checked), cp.asnumpy(cube))

                with self.subTest(dtype=np.dtype(dtype), lanes=21):
                    wide, wide_catalog = _pointer_mask_group(dtype, lanes=21)
                    wide_bound = bind_dense_gemm_group(wide, wide_catalog, cp)
                    with cp.cuda.Stream(non_blocking=True) as stream:
                        for seed in (90410, 90411):
                            arrays, expected = _pointer_mask_inputs(wide, dtype, mask=3,
                                seed=seed, negative_stride=True)
                            cube = wide_bound(arrays, device=True)
                            _assert_close(cp.asnumpy(cube), np.stack(tuple(expected.values())), dtype)
                            for lane, batch in enumerate(wide.batches):
                                for result in batch.results:
                                    (r0, r1), (c0, c1) = result.bounds
                                    _assert_close(cp.asnumpy(arrays[result.tensor]),
                                        expected[lane][r0:r1, c0:c1], dtype)
                            with execution_debug(True):
                                checked = wide_bound(arrays, device=True)
                            _assert_close(cp.asnumpy(checked), cp.asnumpy(cube), dtype)

    def test_saved_cfg_group_replay_and_reuse_without_planning(self):
        from correlator.planning.test_gemm_groups import fixture
        from lqcdweave.contraction.planning.api import lower_independent_dense_batches, build_cfg_plan
        from lqcdweave.contracts.correlator.api import (
            ArrayConstant, SourcePreparation, TensorPreparation, describe_coordinates)
        from lqcdweave.contraction.execution.preparation import bind_prepared_loader
        from lqcdweave.contraction.execution.api import execute_cfg_plan
        from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value
        from dataclasses import replace

        original, profile = fixture()
        original = replace(original, memory_limit=1 << 20, host_memory_limit=1 << 20)
        selected, _ = lower_independent_dense_batches(original, profile=profile, target_runtime='cuda')
        rng = np.random.default_rng(8031)
        values = {key: rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
                  for key, spec in selected.state.problem.specs.items()}
        preps = {key: SourcePreparation((), recipe=TensorPreparation('constant', constants=(
            ArrayConstant.from_array(value),))) for key, value in values.items()}
        points, _ = describe_coordinates(time_pairs=[{0:0,1:0}, {0:1,1:1}])
        cfg = decode_plan_value(encode_plan_value(build_cfg_plan(selected, None, points.points,
            {}, preps, selected.state.problem.specs, prefetch_host_bytes=0, prefetch_device_bytes=0)))
        self.assertIsInstance(cfg.instances[0].kernel_batches[0], DenseGemmGroup)
        self.assertFalse(cfg.instances[1].kernel_batches)

        @contextmanager
        def loader_factory(_):
            yield bind_prepared_loader(None, cfg.preparations, cfg.specs, backend=cp)

        expected = tuple(values[term.inputs[0]]@values[term.inputs[1]] for term in original.state.problem.terms)
        pool, old_limit = cp.get_default_memory_pool(), cp.get_default_memory_pool().get_limit()
        try:
            for debug in (False, True):
                actual = []
                with patch('lqcdweave.contraction.planning.gemm_groups.lower_independent_dense_batches',
                           side_effect=AssertionError('saved cfg cannot regroup')):
                    execute_cfg_plan(cfg, ('2000',), loader_factory,
                        lambda _cfg, index, result: actual.append(result) if index is not None else None,
                        backend=cp, debug=debug)
                self.assertEqual(len(actual), 2)
                for point in actual:
                    for observed, reference in zip(point, expected, strict=True):
                        np.testing.assert_allclose(observed, reference, rtol=2e-12, atol=2e-12)
        finally:
            pool.set_limit(size=old_limit)

    def test_timed_and_unmeasured_calls_keep_host_scalars_and_numeric_results(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=np.dtype(dtype)):
                group, catalog = _group(dtype)
                scalar_refs = []
                original_asarray = np.asarray

                def record_scalar(value, *args, **kwargs):
                    result = original_asarray(value, *args, **kwargs)
                    if np.isscalar(value) and result.shape == ():
                        scalar_refs.append(weakref.ref(result))
                    return result

                # Isolate this group's scalar owners. Patching np.asarray
                # itself also intercepts scalars bound by child fallbacks.
                group_numpy = SimpleNamespace(**vars(np))
                group_numpy.asarray = record_scalar
                with patch('lqcdweave.contraction.execution.operations.gemm_group.np',
                           group_numpy):
                    bound = bind_dense_gemm_group(group, catalog, cp)
                self.assertEqual(len(scalar_refs), 2)
                gc.collect()
                self.assertTrue(all(reference() is not None for reference in scalar_refs))

                phases = []

                def timed(name, operation, *, device):
                    phases.append((name, device))
                    return operation()

                for seed, timer in ((91401, None), (91402, timed)):
                    arrays, expected = _fresh_inputs(group, dtype, seed=seed)
                    cube = bound(arrays, timed=timer, device=True)
                    _assert_close(cp.asnumpy(cube), np.stack(tuple(expected.values())), dtype)
                    for lane, batch in enumerate(group.batches):
                        for result, reference in zip(batch.results,
                                                     np.split(expected[lane], (2,), axis=1), strict=True):
                            _assert_close(cp.asnumpy(arrays[result.tensor]), reference, dtype)
                self.assertEqual(phases, [('dense_batch_pack', True),
                                          ('pointer_setup', True), ('contract_kernel', True)])
                del bound
                gc.collect()
                self.assertTrue(all(reference() is None for reference in scalar_refs))

    def test_packer_converts_negative_stride_and_mixed_input_dtypes(self):
        group, catalog = _group(np.float64, mixed=True)
        arrays, expected = _fresh_inputs(
            group, np.float64, mixed=True, negative_stride=True)
        actual = bind_dense_gemm_group(group, catalog, cp)(arrays, device=True)
        _assert_close(cp.asnumpy(actual), np.stack(tuple(expected.values())), np.float64)

    def test_result_view_keeps_shared_allocation_until_last_view(self):
        group, catalog = _group(np.complex64)
        arrays, expected = _fresh_inputs(group, np.complex64)
        original_empty = cp.empty
        allocations = []
        pool = cp.cuda.MemoryPool()

        def record_empty(shape, *args, **kwargs):
            value = original_empty(shape, *args, **kwargs)
            if value.dtype == np.dtype(np.uint8):
                allocations.append((tuple(shape), value.data.ptr))
            return value

        with cp.cuda.using_allocator(pool.malloc), patch.object(cp, "empty", side_effect=record_empty):
            cube = bind_dense_gemm_group(group, catalog, cp)(arrays, device=True)
        self.assertEqual(allocations, [((512+cube.nbytes,), cube.data.ptr-512)])
        kept_key = group.batches[0].results[0].tensor
        kept_view = arrays[kept_key]
        for batch in group.batches:
            for result in batch.results:
                if result.tensor != kept_key:
                    del arrays[result.tensor]
        del cube
        gc.collect()
        self.assertGreater(pool.used_bytes(), 0)
        _assert_close(cp.asnumpy(kept_view), np.split(expected[0], (2,), axis=1)[0], np.complex64)
        del arrays[kept_key], kept_view
        gc.collect()
        self.assertEqual(pool.used_bytes(), 0)
        pool.free_all_blocks()

    def test_multiaxis_result_views_share_allocation_until_last_view(self):
        group, catalog = _multiaxis_group(np.complex64)
        arrays, expected = _fresh_multiaxis_inputs(group, np.complex64)
        original_empty = cp.empty
        allocations = []
        pool = cp.cuda.MemoryPool()

        def record_empty(shape, *args, **kwargs):
            value = original_empty(shape, *args, **kwargs)
            if value.dtype == np.dtype(np.uint8):
                allocations.append((tuple(shape), value.data.ptr))
            return value

        with cp.cuda.using_allocator(pool.malloc), patch.object(cp, "empty", side_effect=record_empty):
            cube = bind_dense_gemm_group(group, catalog, cp)(arrays, device=True)
        self.assertEqual(tuple(cube.shape), (3, 6, 10))
        self.assertEqual(allocations, [((512+cube.nbytes,), cube.data.ptr-512)])
        for lane, batch in enumerate(group.batches):
            left_view = arrays[batch.results[0].tensor]
            right_view = arrays[batch.results[1].tensor]
            self.assertEqual(tuple(left_view.shape), (2, 3, 5, 1))
            self.assertEqual(tuple(right_view.shape), (2, 3, 5, 1))
            _assert_close(cp.asnumpy(left_view), expected[lane][..., :1], np.complex64)
            _assert_close(cp.asnumpy(right_view), expected[lane][..., 1:], np.complex64)

        del left_view, right_view
        kept_key = group.batches[0].results[0].tensor
        kept_view = arrays[kept_key]
        for batch in group.batches:
            for result in batch.results:
                if result.tensor != kept_key:
                    del arrays[result.tensor]
        del cube
        gc.collect()
        self.assertGreater(pool.used_bytes(), 0)
        _assert_close(cp.asnumpy(kept_view), expected[0][..., :1], np.complex64)
        del arrays[kept_key], kept_view
        gc.collect()
        self.assertEqual(pool.used_bytes(), 0)
        pool.free_all_blocks()

    def test_pointer_setup_uses_independent_side_masks_and_fresh_inputs(self):
        masks = (3, 2, 1, 0, 3)
        cases = [(dtype, 3) for dtype in
                 (np.float32, np.float64, np.complex64, np.complex128)]
        cases.append((np.complex128, 8))
        cases.append((np.complex128, 21))
        original_rawkernel = cp.RawKernel

        for case_index, (dtype, lanes) in enumerate(cases):
            group, catalog = _pointer_mask_group(dtype, lanes=lanes)
            pointer_calls = []
            active_mask = None

            def recording_rawkernel(code, name, *args, **kwargs):
                kernel = original_rawkernel(code, name, *args, **kwargs)
                if name != "ed_gemm_group_pointer_setup":
                    return kernel

                def invoke(*kernel_args, **kernel_kwargs):
                    if active_mask is None:
                        raise AssertionError("pointer setup ran without an active packing mask")
                    kernel_values = kernel_args[2]
                    shapes = tuple(tuple(value.shape) for value in kernel_values)
                    pointer_calls.append((name, active_mask, len(kernel_values), shapes))
                    return kernel(*kernel_args, **kernel_kwargs)

                return invoke

            with patch.object(cp, "RawKernel", new=recording_rawkernel):
                bound = bind_dense_gemm_group(group, catalog, cp)
                for run_index, mask in enumerate(masks):
                    active_mask = mask
                    negative_stride = mask == 3 and run_index == len(masks) - 1
                    with self.subTest(dtype=np.dtype(dtype), lanes=lanes, mask=mask,
                                      run=run_index, negative_stride=negative_stride):
                        arrays, expected = _pointer_mask_inputs(
                            group, dtype, mask=mask,
                            seed=94000 + case_index * 100 + run_index,
                            negative_stride=negative_stride,
                        )
                        calls_before = len(pointer_calls)
                        cube = bound(arrays, device=True)
                        self.assertEqual(tuple(cube.shape), (lanes, 4, 6))
                        references = np.stack(tuple(expected.values()))
                        _assert_close(cp.asnumpy(cube), references, dtype)
                        for lane, batch in enumerate(group.batches):
                            for result, bounds in zip(
                                    batch.results,
                                    (((0, 2), (0, 3)), ((0, 2), (3, 6)),
                                     ((2, 4), (0, 3)), ((2, 4), (3, 6))),
                                    strict=True):
                                row_start, row_stop = bounds[0]
                                column_start, column_stop = bounds[1]
                                reference = expected[lane][row_start:row_stop,
                                                           column_start:column_stop]
                                self.assertEqual(tuple(arrays[result.tensor].shape), (2, 3))
                                _assert_close(cp.asnumpy(arrays[result.tensor]), reference, dtype)

                        self.assertEqual(len(pointer_calls), calls_before + 1)
                        _, _, argument_count, argument_shapes = pointer_calls[-1]
                        if mask == 3:
                            expected_argument_count = 4
                        elif mask in (1, 2):
                            expected_argument_count = lanes + 3
                        else:
                            expected_argument_count = 2 * lanes + 2
                        self.assertEqual(argument_count, expected_argument_count)

                        expected_shapes = []
                        if mask & 2:
                            expected_shapes.append((lanes, 3, 6))
                        else:
                            expected_shapes.extend([(3, 6)] * lanes)
                        if mask & 1:
                            expected_shapes.append((lanes, 4, 3))
                        else:
                            expected_shapes.extend([(4, 3)] * lanes)
                        expected_shapes.extend(((64,), (lanes, 4, 6)))
                        self.assertEqual(argument_shapes, tuple(expected_shapes))

            self.assertEqual(tuple(call[0] for call in pointer_calls),
                             ("ed_gemm_group_pointer_setup",) * len(masks))
            self.assertEqual(tuple(call[1] for call in pointer_calls), masks)


if __name__ == "__main__":
    unittest.main()
