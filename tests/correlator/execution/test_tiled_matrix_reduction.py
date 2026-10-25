"""Fixed output tiles preserve split ownership and the existing wide tree."""
import gc
import unittest
import weakref
from unittest.mock import Mock

import numpy as np

from lqcdweave.contracts.correlator.api import parse_axis_pattern
from lqcdweave.contraction.execution.reduction_kernel import (
    _kernel, bind_raw_reduction, matrix_reduction_sources, tiled_matrix_reduction_source,
)
from correlator.execution.test_raw_precise_reduction import _SourceBackend


class _ObservedBackend(_SourceBackend):
    def __init__(self):
        super().__init__()
        self.allocations = []
        self.geometry = []
        self.current_refs = []
        self.partials = None

    def empty(self, shape, dtype):
        value = np.empty(shape, dtype=dtype)
        self.allocations.append(value.nbytes)
        return value

    def RawKernel(self, source, name, **kwargs):
        invoke = super().RawKernel(source, name, **kwargs)

        def call(grid, block, args):
            self.geometry.append((name, grid, block))
            self.current_refs = [weakref.ref(v) for v in args if isinstance(v, np.ndarray)]
            if name == 'ed_precise_matrix_finish':
                self.partials = args[0].copy()
            return invoke(grid, block, args)
        return call


def _inputs(m, k, n, dtype):
    rng = np.random.default_rng(723)
    left = rng.normal(size=(m, k)).astype(dtype)
    right = rng.normal(size=(n, k)).astype(dtype)
    if np.dtype(dtype).kind == 'c':
        left = left + 0.25j*left[:, ::-1]
        right = right - 0.5j*right[:, ::-1]
    return left[::-1, ::-1], right[::-1, ::-1]


class TiledMatrixReductionTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()

    def test_split_coverage_signed_strides_edges_budget_and_current_inputs(self):
        equation = parse_axis_pattern('ik,jk->ij')
        for dtype in (np.float32, np.complex64):
            narrow = np.dtype(dtype)
            wide = np.dtype('complex128' if narrow.kind == 'c' else 'float64')
            for m, k, n in ((1, 513, 1), (1, 777, 5), (5, 777, 1), (3, 513, 5), (2, 513, 2)):
                for blocks in (1, 3):
                    with self.subTest(dtype=dtype, shape=(m, k, n), blocks=blocks):
                        a, b = _inputs(m, k, n, dtype)
                        backend = _ObservedBackend()
                        allowance = m*n*(blocks*wide.itemsize+narrow.itemsize)
                        run = bind_raw_reduction(equation, (a.shape, b.shape), narrow, backend, allowance)
                        self.assertEqual(run.raw_output_tile, (2, 2) if m*n>1 else (1, 1))
                        result = run(a, b)
                        np.testing.assert_allclose(result, np.einsum('ik,jk->ij', a, b, dtype=wide).astype(dtype),
                                                   rtol=1e-6, atol=1e-6)
                        self.assertEqual(sum(backend.allocations), run.workspace_bytes)
                        self.assertEqual(run.workspace_bytes, allowance)
                        expected_grid = (blocks, (n+1)//2, (m+1)//2) if m*n>1 else (blocks, n, m)
                        self.assertEqual(backend.geometry[0][1:], (expected_grid, (256,)))
                        self.assertEqual(backend.geometry[1][1:], ((m*n,), (256,)))
                        # Each saved partial must own exactly its previous K
                        # subsequence, including budget-forced multiple passes.
                        for split in range(blocks):
                            take = np.arange(k)[np.arange(k)//256 % blocks == split]
                            reference = np.einsum('ik,jk->ij', a[:, take], b[:, take], dtype=wide)
                            np.testing.assert_allclose(backend.partials[:, split].reshape(m, n), reference,
                                                       rtol=1e-12, atol=1e-12)
                        saved = result.copy()
                        np.testing.assert_allclose(run(2*a, b), 2*saved, rtol=1e-6, atol=1e-6)
                        np.testing.assert_array_equal(result, saved)

    def test_source_retains_per_output_tree_and_one_unconditional_block_barrier(self):
        for dtype in ('float32', 'complex64'):
            old, _ = matrix_reduction_sources(dtype)
            source = tiled_matrix_reduction_source(dtype)
            self.assertEqual(source.count('__syncthreads();'), 1)
            self.assertEqual(source.count('for (int step = 16;'), 4)
            self.assertEqual(source.count('for (int step = 4;'), 4)
            self.assertIn('i = split * 256LL + t; i < k; i += blocks * 256LL', source)
            self.assertIn('i = b * 256LL + t; i < k; i += blocks * 256LL', old)
            self.assertNotIn('atomic', source)
            self.assertNotIn('return;', source)

    def test_wide_products_survive_cancellation_in_all_four_tile_lanes(self):
        for dtype in (np.float32, np.complex64):
            a = np.tile(np.array([128.125, 128.125, -128.125, -128.125], dtype), (3, 256))
            a[:, 0] = np.nextafter(np.float32(128.125), np.float32(np.inf))
            b = np.full((5, 1024), 1.125, dtype=dtype)
            if dtype is np.complex64:
                b += .25j
            wide = np.complex128 if dtype is np.complex64 else np.float64
            reference = np.einsum('ik,jk->ij', a, b, dtype=wide).astype(dtype)
            self.assertTrue(np.all(reference != 0))
            run = bind_raw_reduction(parse_axis_pattern('ik,jk->ij'),
                (a.shape, b.shape), a.dtype, _ObservedBackend(), 2000)
            np.testing.assert_array_equal(run(a, b), reference)

    def test_failed_submission_keeps_owners_until_fence_and_grid_limit_keeps_fallback(self):
        for fail_at in (1, 2):
            backend = _ObservedBackend()
            backend.fail_at = fail_at
            backend.synchronize = Mock(side_effect=lambda: self.assertTrue(
                all(ref() is not None for ref in backend.current_refs)))
            a, b = _inputs(3, 513, 5, np.complex64)
            run = bind_raw_reduction(parse_axis_pattern('ik,jk->ij'),
                (a.shape, b.shape), a.dtype, backend, 10000)
            with self.assertRaisesRegex(RuntimeError, 'submission failed'):
                run(a, b)
            backend.synchronize.assert_called_once()
            del a, b
            gc.collect()
            self.assertTrue(all(ref() is None for ref in backend.current_refs))
        backend = _ObservedBackend()
        for m, expected in ((65535, 'matrix_runtime_geometry'), (65536, 'static_geometry')):
            run = bind_raw_reduction(parse_axis_pattern('ik,jk->ij'),
                ((m, 2), (1, 2)), np.dtype('float32'), backend, 1<<24)
            self.assertEqual(run.raw_source_family, expected)
        self.assertEqual(backend.allocations, [])


class TiledMatrixReductionCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')
        cls.cp = cp

    def test_cuda_partials_and_results_match_original_tree_bitwise(self):
        cp = self.cp
        options = ('--std=c++17', '--fmad=false')
        for dtype in (np.float32, np.complex64):
            wide = np.dtype('complex128' if np.dtype(dtype).kind == 'c' else 'float64')
            first, finish = matrix_reduction_sources(np.dtype(dtype).name)
            old = cp.RawKernel(first, 'ed_precise_matrix_partial', options=options)
            tiled = cp.RawKernel(tiled_matrix_reduction_source(np.dtype(dtype).name),
                                 'ed_precise_matrix_tile_partial', options=options)
            end = cp.RawKernel(finish, 'ed_precise_matrix_finish', options=options)
            for m, k, n in ((1, 513, 5), (5, 513, 1), (3, 1025, 5), (2, 134400, 4), (3, 230400, 3)):
                host_a, host_b = _inputs(m, k, n, dtype)
                if k == 1025:
                    host_a[:, :-1] = np.tile(np.array([128.125, 128.125, -128.125, -128.125], dtype), 256)
                    host_a[:, -1] = 0
                    host_a[:, 0] = np.nextafter(np.float32(128.125), np.float32(np.inf))
                    host_b.fill(1.125 + .25j if dtype is np.complex64 else 1.125)
                # Retain actual negative strides on device as well.
                a = cp.asarray(host_a[:, ::-1])[:, ::-1]
                b = cp.asarray(host_b[:, ::-1])[:, ::-1]
                for blocks in (1, min(1024, (k+255)//256)):
                    old_partial = cp.empty((m*n, blocks), dtype=wide)
                    # Guard cells detect writes past an odd output tile.
                    storage = cp.full(m*n*blocks+2, 19, dtype=wide)
                    new_partial = storage[1:-1].reshape(m*n, blocks)
                    strides = (a.strides[0], a.strides[1], b.strides[1], b.strides[0])
                    old_args = tuple(np.int64(v) for v in (k, n, blocks, *strides))
                    new_args = tuple(np.int64(v) for v in (k, m, n, blocks, *strides))
                    old((blocks, n, m), (256,), (a, b, *old_args, old_partial))
                    tiled((blocks, (n+1)//2, (m+1)//2), (256,), (a, b, *new_args, new_partial))
                    np.testing.assert_array_equal(cp.asnumpy(new_partial).view(np.uint8),
                                                  cp.asnumpy(old_partial).view(np.uint8))
                    np.testing.assert_array_equal(cp.asnumpy(storage[[0, -1]]), [19, 19])
                    outputs = [cp.empty((m, n), dtype=dtype) for _ in range(2)]
                    for partials, output in zip((old_partial, new_partial), outputs):
                        end((m*n,), (256,), (partials, np.int64(blocks), output))
                    np.testing.assert_array_equal(cp.asnumpy(outputs[0]).view(np.uint8),
                                                  cp.asnumpy(outputs[1]).view(np.uint8))
                    reference = np.einsum('ik,jk->ij', host_a, host_b, dtype=wide).astype(dtype)
                    np.testing.assert_allclose(cp.asnumpy(outputs[1]), reference, rtol=2e-6, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
