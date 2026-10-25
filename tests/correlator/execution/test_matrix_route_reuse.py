"""Layout routes are reused without reusing operands or numerical policy."""
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np

from lqcdweave.contraction.execution import lowering
from lqcdweave.contraction.execution.diagnostics import execution_debug
from correlator.execution.test_dense_atomic_kernel import _inputs, _loop_reference
from correlator.execution.test_direct_matrix_packing import (
    _layouts, _protocol_packer, _recipe, _SHAPES, _PACKING,
)
from correlator.execution.test_pointer_matrix_product import _Array, _Backend


class MatrixRouteReuseTests(unittest.TestCase):
    def setUp(self):
        self.packing = patch(_PACKING, side_effect=_protocol_packer)
        self.packing.start()
        self.addCleanup(self.packing.stop)
        self.products = []
        original = lowering.bind_pointer_gemm

        def bind(*args, **kwargs):
            product = original(*args, **kwargs)
            if product is not None:
                product.bind_layout = Mock(wraps=product.bind_layout)
                self.products.append(product)
            return product
        self.binding = patch.object(lowering, 'bind_pointer_gemm', side_effect=bind)
        self.binding.start()
        self.addCleanup(self.binding.stop)
        self.backend = _Backend()
        self.backend.pack_calls, self.backend.pack_failure = [], None

    def test_each_layout_classified_once_and_values_always_current(self):
        run = lowering.bind_matmul(_recipe(np.complex128), self.backend)
        direct = self.products[1]  # Lowered fallback is bound before original tensor geometry.
        refs = []
        for scale in (1, 3):
            host = tuple(scale * v for v in _inputs(_SHAPES, np.complex128))
            layouts = _layouts(np, *host)
            layouts += ((host[0], layouts[1][1]),)  # Both directly usable matrices.
            for values in layouts:
                arrays = tuple(map(_Array, values))
                refs.extend(map(weakref.ref, arrays))
                expected = _loop_reference('akb,bkc->ac', values)
                for _ in range(3):
                    np.testing.assert_allclose(run(*arrays).host, expected, rtol=1e-12, atol=1e-12)
                del arrays
        self.assertEqual(direct.bind_layout.call_count, 5)
        self.assertEqual(len(self.backend.calls), 30)
        self.assertEqual(self.backend.synchronizations, 0)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_cached_route_keeps_current_modes_debug_and_pointer_policy(self):
        run = lowering.bind_matmul(_recipe(np.complex128), self.backend)
        host = _inputs(_SHAPES, np.complex128)
        arrays = tuple(map(_Array, host))
        expected = _loop_reference('akb,bkc->ac', host)
        for debug, compute, math in ((False, 0, 0), (False, 1, 0), (False, 0, 1),
                                     (True, 0, 0), (False, 0, 0)):
            self.backend.compute_mode, self.backend.math_mode = compute, math
            before = len(self.backend.calls)
            with execution_debug(debug):
                result = run(*arrays)
            np.testing.assert_allclose(result.host, expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(len(self.backend.calls) - before, int(not (debug or compute or math)))
        self.assertEqual(self.products[1].bind_layout.call_count, 1)
        self.backend.pointer_mode = 1
        with self.assertRaisesRegex(RuntimeError, 'host pointer mode'):
            run(*arrays)

    def test_current_shape_and_dtype_do_not_inherit_a_cached_route(self):
        run = lowering.bind_matmul(_recipe(np.complex128), self.backend)
        host = _inputs(_SHAPES, np.complex128)
        run(*map(_Array, host))
        self.backend.pack_calls.clear()
        with self.assertRaises(ValueError):
            run(_Array(host[0][:1]), _Array(host[1]))
        changed = host[0].real.astype(np.float32), host[1]
        result = run(*map(_Array, changed))
        np.testing.assert_allclose(result.host, _loop_reference('akb,bkc->ac', changed),
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(self.backend.pack_calls, [])

    def test_many_stride_variants_remain_bounded_and_reclassify_safely(self):
        run = lowering.bind_matmul(_recipe(np.complex128), self.backend)
        host = _inputs(_SHAPES, np.complex128)
        for gap in range(1, 38):
            storage = np.empty((2, 4, 3 * gap), dtype=np.complex128)
            left = storage[..., ::gap]
            left[...] = host[0]
            result = run(_Array(left), _Array(host[1]))
            np.testing.assert_allclose(result.host, _loop_reference('akb,bkc->ac', host),
                                       rtol=1e-12, atol=1e-12)
        # The initial variant was evicted by the bounded structural cache.
        before = self.products[1].bind_layout.call_count
        run(*map(_Array, host))
        self.assertEqual(self.products[1].bind_layout.call_count, before + 1)

    def test_packed_copy_reused_only_inside_explicit_numerical_scope(self):
        from lqcdweave.contraction.execution.packed_values import PackedValueCache

        cache = PackedValueCache(4096, array_type=_Array)
        self.backend.cuda.get_current_stream = lambda: SimpleNamespace(
            ptr=7, synchronize=self.backend.synchronize)
        run = lowering.bind_matmul(_recipe(np.complex128), self.backend)
        transposed = lowering.bind_matmul(_recipe(np.complex128, 'akb,bkc->ca'), self.backend)
        host = _inputs(_SHAPES, np.complex128)
        arrays = tuple(map(_Array, host))
        expected = _loop_reference('akb,bkc->ac', host)
        with cache.activate(self.backend):
            np.testing.assert_allclose(run(*arrays).host, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(run(*arrays).host, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(transposed(*arrays).host, expected.T, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(self.backend.pack_calls), 1)
        self.assertEqual(cache.hits, 2)
        self.assertEqual(cache.bytes, 12 * 5 * 16)
        # A structural route cache cannot enable numerical reuse by itself.
        run(*arrays)
        self.assertEqual(len(self.backend.pack_calls), 2)
        # Distinct source objects never inherit a copy, even with equal values.
        with cache.activate(self.backend):
            run(*map(_Array, host))
        self.assertEqual(len(self.backend.pack_calls), 3)
        cache.clear()


if __name__ == '__main__':
    unittest.main()
