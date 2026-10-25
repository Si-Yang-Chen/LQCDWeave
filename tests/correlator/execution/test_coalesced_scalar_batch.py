"""Scalar RHS row packing preserves the saved Cartesian output and budget."""
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import Action, AxisPattern, TensorSpec
from lqcdweave.contracts.correlator.gemm_group import DenseGemmGroup
from lqcdweave.contraction.execution.diagnostics import execution_debug
from lqcdweave.contraction.execution.operations.dense_batch import bind_batch_layout
from lqcdweave.contraction.execution.operations.gemm_group import _bind_layouts, bind_dense_gemm_group
from lqcdweave.contraction.execution.reduction_kernel import _kernel
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import describe_dense_batch
from correlator.execution.test_precise_batch_reduction import _fixture, _check
from correlator.execution.test_raw_precise_reduction import _SourceBackend


class _Backend(_SourceBackend):
    """NumPy packing plus the existing generated-load reduction interpreter."""
    matmul = staticmethod(np.matmul)
    concatenate = staticmethod(np.concatenate)
    array = staticmethod(np.array)
    zeros = staticmethod(np.zeros)

    def __init__(self):
        super().__init__()
        self.allocations = []

    def empty(self, shape, dtype, **kwargs):
        value = np.empty(shape, dtype=dtype, **kwargs)
        self.allocations.append(value.nbytes)
        return value


def _multiaxis(dtype=np.complex64, *, n=3, prepared=False):
    rng = np.random.default_rng(402)
    arrays = {}
    for i in range(2):
        a = rng.normal(size=(16, 50)).astype(dtype)
        arrays[f'a{i}'] = (a + 0.5j*a if np.dtype(dtype).kind == 'c' else a)[:, ::-1]
    for j in range(n):
        b = rng.normal(size=(16, 50)).astype(dtype)
        b = b - 0.25j*b if np.dtype(dtype).kind == 'c' else b
        arrays[f'b{j}'] = b.T if prepared else b.T[::-1]
    catalog = {key: TensorSpec(value.shape, value.dtype) for key, value in arrays.items()}
    equation = AxisPattern(((0, 1), (1, 0)), ())
    actions = []
    for i in range(2):
        for j in range(n):
            key = f'c{i}_{j}'
            catalog[key] = TensorSpec((), dtype)
            actions.append(Action('contract', key, (f'a{i}', f'b{j}'), equation, seconds=10))
    batch = describe_dense_batch(actions, tuple(range(2*n)), catalog, CostProfile(),
        backend='cupy', prepared_inputs={'b0': (1, 0)} if prepared else None)
    return batch, catalog, arrays, actions


class CoalescedScalarBatchTests(unittest.TestCase):
    def tearDown(self):
        _kernel.cache_clear()

    def test_multiaxis_current_inputs_output_owners_and_exact_allocation_envelope(self):
        for dtype in (np.float32, np.complex64):
            batch, catalog, arrays, actions = _multiaxis(dtype)
            backend = _Backend()
            # Record the real packing API geometry. Its CPU fallback is enough
            # for this protocol check; actual CUDA packing has a separate gate.
            from lqcdweave.contraction.execution.operations import dense_batch
            with patch.object(dense_batch, 'bind_packing', wraps=dense_batch.bind_packing) as packing:
                layout = bind_batch_layout(batch, catalog, backend)
            rhs = layout.sides[1]
            self.assertEqual(rhs.shape, (3, 16, 50))
            self.assertEqual(rhs.axis, 0)
            self.assertEqual(rhs.input_layouts, ((None, 1, 0),)*3)
            self.assertEqual(packing.call_args.args[1], [(1, 16, 50)]*3)
            self.assertEqual(packing.call_args.args[3], 0)
            self.assertEqual(layout.matrix_shapes, ((2, 800), (3, 800)))
            canonical = bind_batch_layout(batch, catalog, backend, coalesce_reductions=False)
            self.assertEqual(layout.precise_product.workspace_bytes, canonical.precise_product.workspace_bytes)
            self.assertEqual(layout.deferred_layouts, canonical.deferred_layouts)
            owner = layout.run(arrays, backend=backend)
            _check(self, arrays, actions)
            self.assertEqual(sum(backend.allocations[:2]), batch.packing_bytes)
            self.assertEqual(sum(backend.allocations[2:]), layout.precise_product.workspace_bytes)
            self.assertEqual(owner.nbytes, batch.output_bytes)
            for action in actions:
                self.assertTrue(np.shares_memory(arrays[action.tensor], owner))
            previous = {a.tensor: arrays[a.tensor].copy() for a in actions}
            arrays['a0'] *= 2
            layout.run(arrays, backend=backend)
            _check(self, arrays, actions)
            for a in actions[:3]:
                np.testing.assert_allclose(arrays[a.tensor], 2*previous[a.tensor])
            np.testing.assert_array_equal(owner, np.array(list(previous.values())).reshape(2, 3))

    def test_prepared_single_rhs_keeps_original_producer_permutation_and_borrow(self):
        batch, catalog, arrays, actions = _multiaxis(n=1, prepared=True)
        self.assertEqual(batch.prepared_sides, (False, True))
        backend = _Backend()
        layout = bind_batch_layout(batch, catalog, backend)
        packed = layout.sides[1].pack(arrays, backend, np.dtype('complex64'))
        self.assertEqual(packed.shape, (1, 16, 50))
        self.assertTrue(packed.flags.c_contiguous)
        self.assertTrue(np.shares_memory(packed, arrays['b0']))
        self.assertEqual(backend.allocations, [])
        self.assertEqual(batch.right[0].permutation, (1, 0))
        layout.run(arrays, backend=backend)
        _check(self, arrays, actions)
        self.assertEqual(backend.allocations[0], batch.packing_bytes)

    def test_cpu_debug_missing_raw_and_small_workspace_retain_whole_original_layout(self):
        for mode in ('cpu', 'debug', 'missing_raw', 'workspace', 'thin'):
            batch, catalog, arrays, actions = _fixture(
                k=32 if mode == 'workspace' else 800,
                m=16 if mode == 'workspace' else 2,
                n=16 if mode == 'workspace' else 3, thin=mode == 'thin')
            backend = np if mode == 'cpu' else _Backend()
            if mode == 'missing_raw':
                backend.RawKernel = None
            with self.subTest(mode=mode), execution_debug(mode == 'debug'):
                layout = bind_batch_layout(batch, catalog, backend)
                self.assertFalse(layout.sides[1].prepend)
                self.assertEqual(layout.sides[1].axis, batch.right_axis)
                self.assertEqual(layout.matrix_shapes[1][0], layout.matrix_shapes[0][1])
                layout.run(arrays, backend=backend)
                _check(self, arrays, actions)

    def test_legacy_groups_keep_canonical_input_shapes_and_wide_numerics(self):
        first, one, a, acts_a = _fixture('x')
        second, two, b, acts_b = _fixture('y', offset=6)
        group = DenseGemmGroup((first, second), (2, 800, 3), 1.)
        catalog, arrays = dict(one, **two), dict(a, **b)
        backend = _Backend()
        _, layouts, _, _ = _bind_layouts(group, catalog, backend)
        self.assertTrue(all(layout.matrix_shapes == ((2, 800), (800, 3)) for layout in layouts))
        self.assertTrue(all(not layout.sides[1].prepend for layout in layouts))
        bind_dense_gemm_group(group, catalog, backend)(arrays)
        _check(self, arrays, (*acts_a, *acts_b))


class CoalescedScalarBatchCuPyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device')
        except Exception as exc:
            raise unittest.SkipTest(f'CuPy device unavailable: {exc}')
        cls.cp = cp

    def test_cuda_native_packing_chunk_boundary_prepared_rhs_and_canonical_reference(self):
        cp = self.cp
        for n, prepared in ((33, False), (1, True)):
            batch, catalog, host, actions = _multiaxis(n=n, prepared=prepared)
            # Preserve the actual producer's desired contiguous permutation.
            arrays = {key: cp.asarray(value.T).T if key.startswith('b') and prepared
                      else cp.asarray(value) for key, value in host.items()}
            layout = bind_batch_layout(batch, catalog, cp)
            self.assertEqual(layout.matrix_shapes[1], (n, 800))
            self.assertTrue(layout.precise_product.raw_reduction)
            self.assertIsNotNone(layout.sides[0].original_packer)
            if prepared:
                borrowed = layout.sides[1].pack(arrays, cp, np.dtype('complex64'))
                self.assertEqual(borrowed.data.ptr, arrays['b0'].data.ptr)
            layout.run(arrays, backend=cp)
            actual = {key: cp.asnumpy(value) for key, value in arrays.items()}
            _check(self, actual, actions)
            canonical = bind_batch_layout(batch, catalog, cp, coalesce_reductions=False)
            canonical.run(arrays, backend=cp)
            for action in actions:
                np.testing.assert_allclose(cp.asnumpy(arrays[action.tensor]), actual[action.tensor],
                                           rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
