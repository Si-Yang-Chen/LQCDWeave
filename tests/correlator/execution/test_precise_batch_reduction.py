"""Batching cannot undo wide accumulation of cancelling terminal reductions."""
from dataclasses import replace
from unittest.mock import Mock
import unittest

import numpy as np

from lqcdweave.contracts.correlator.api import Action, AxisPattern, TensorSpec
from lqcdweave.contracts.correlator.gemm_group import DenseGemmGroup
from lqcdweave.contraction.execution.batch_reduction import bind_precise_batch
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.dense_batch import bind_batch_layout
from lqcdweave.contraction.execution.operations.gemm_group import bind_dense_gemm_group
from lqcdweave.contraction.execution.reduction_kernel import _kernel
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import describe_dense_batch
from correlator.execution.test_raw_precise_reduction import _SourceBackend


def _fixture(prefix='', *, k=800, m=2, n=3, dtype=np.complex64, offset=0, thin=False):
    x = np.tile(np.array([128.125, 128.125, -128.125, -128.125], np.float32), k//4).astype(dtype)
    x[0] = np.nextafter(np.float32(128.125), np.float32(np.inf))
    left = {f'{prefix}a{i}': (i+1)*x for i in range(m)}
    right = {f'{prefix}b{j}': np.full((2,k) if thin else (k,), (j+1)*1.125, dtype=dtype)
             for j in range(n)}
    arrays = dict(left, **right)
    catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
    equation = AxisPattern(((0,), (1,0) if thin else (0,)), (1,) if thin else ())
    actions = []
    for a in left:
        for b in right:
            key = f'{a}_{b}'
            catalog[key] = TensorSpec((2,) if thin else (), dtype)
            actions.append(Action('contract', key, (a,b), equation, seconds=10))
    batch = describe_dense_batch(actions, tuple(range(m*n)), catalog, CostProfile(), backend='cupy')
    if offset:
        batch = replace(batch, action_indices=tuple(i+offset for i in batch.action_indices))
    return batch, catalog, arrays, actions


def _check(test, arrays, actions):
    for action in actions:
        a,b = (arrays[key] for key in action.operands)
        wide = np.complex128 if a.dtype.kind == 'c' else np.float64
        actual = arrays[action.tensor]
        reference = np.einsum(a, action.equation.inputs[0], b, action.equation.inputs[1],
                              action.equation.output, dtype=wide).astype(a.dtype)
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-9)
        test.assertEqual(actual.dtype, a.dtype)


class PreciseBatchReductionTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()

    def test_cpu_scalar_and_thin_batches_preserve_wide_accumulation_and_new_inputs(self):
        for dtype in (np.float32, np.complex64):
            for thin in (False, True):
                batch, catalog, arrays, actions = _fixture(dtype=dtype, thin=thin)
                layout = bind_batch_layout(batch, catalog, np)
                self.assertIsNotNone(layout.precise_product)
                self.assertIsNone(layout.pointer_product)
                fn = layout.precise_product
                self.assertLessEqual(fn.workspace_bytes, fn.workspace_allowance)
                layout.run(arrays, backend=np)
                _check(self, arrays, actions)
                saved = {a.tensor: arrays[a.tensor].copy() for a in actions}
                arrays['a0'] *= 2
                layout.run(arrays, backend=np)
                _check(self, arrays, actions)
                np.testing.assert_array_equal(arrays[actions[0].tensor], 2*saved[actions[0].tensor])

    def test_gpu_source_route_uses_two_launches_with_original_member_allowance(self):
        batch, catalog, arrays, _ = _fixture()
        left = np.stack([arrays[o.tensor] for o in batch.left])
        right = np.stack([arrays[o.tensor] for o in batch.right], axis=1)
        backend = _SourceBackend()
        run = bind_precise_batch(batch, catalog, (left.shape, right.shape), backend)
        self.assertTrue(run.raw_reduction)
        self.assertEqual(run.workspace_allowance, (800*2+1)*8)
        self.assertLessEqual(run.workspace_bytes, run.workspace_allowance)
        expected = (left.astype(np.complex128) @ right.astype(np.complex128)).astype(np.complex64)
        np.testing.assert_array_equal(run(left,right), expected)
        self.assertEqual(backend.launches, 2)
        with execution_debug(True):
            oracle = bind_precise_batch(batch, catalog, (left.shape, right.shape), backend)
        self.assertFalse(getattr(oracle, 'raw_reduction', False))
        # The protocol backend only needs NumPy numerical operations here.
        backend.matmul = np.matmul
        np.testing.assert_array_equal(oracle(left,right), expected)

    def test_input_only_catalog_preserves_member_allowance_and_wide_results(self):
        for thin in (False, True):
            with self.subTest(thin=thin):
                batch, catalog, arrays, actions = _fixture(thin=thin)
                reference = bind_batch_layout(batch, catalog, np).precise_product
                layout = bind_batch_layout(batch, arrays, np)
                self.assertEqual(layout.precise_product.workspace_allowance,
                                 reference.workspace_allowance)
                self.assertLessEqual(layout.precise_product.workspace_bytes,
                                     reference.workspace_allowance)
                layout.run(arrays, backend=np)
                _check(self, arrays, actions)

    def test_large_output_tiny_workspace_and_legacy_groups_remain_correct(self):
        batch, catalog, arrays, actions = _fixture(k=32, m=16, n=16)
        layout = bind_batch_layout(batch, catalog, np)
        self.assertLessEqual(layout.precise_product.workspace_bytes,
                             layout.precise_product.workspace_allowance)
        layout.run(arrays, backend=np)
        _check(self, arrays, actions)
        first, one, a, act_a = _fixture('x')
        second, two, b, act_b = _fixture('y', offset=6)
        group = DenseGemmGroup((first, second), (2,800,3), 1.)
        arrays = dict(a, **b)
        fn = bind_dense_gemm_group(group, dict(one, **two), np)
        fn(arrays)
        _check(self, arrays, (*act_a,*act_b))
        timed = Mock(side_effect=lambda label, call, **kwargs: call())
        fn(arrays, timed=timed)
        self.assertEqual(timed.call_count, 2)


class PreciseBatchReductionCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')
        cls.cp = cp

    def test_cuda_scalar_batch_matches_wide_independent_reference_and_debug(self):
        cp = self.cp
        for thin in (False, True):
            batch, catalog, host, actions = _fixture(thin=thin)
            arrays = {key: cp.asarray(value) for key, value in host.items()}
            layout = bind_batch_layout(batch, catalog, cp)
            self.assertTrue(layout.precise_product.raw_reduction)
            layout.run(arrays, backend=cp)
            _check(self, {key: cp.asnumpy(value) for key,value in arrays.items()}, actions)
            with execution_debug(True):
                oracle = bind_batch_layout(batch, catalog, cp)
                oracle.run(arrays, backend=cp)
            _check(self, {key: cp.asnumpy(value) for key,value in arrays.items()}, actions)


if __name__ == '__main__':
    unittest.main()
