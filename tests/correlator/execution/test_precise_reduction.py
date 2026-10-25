"""Strongly cancelling traces retain the declared dtype without naive sums."""
from types import SimpleNamespace
import unittest
import numpy as np
from lqcdweave.contracts.correlator.api import Action, AxisPattern, TensorSpec
from lqcdweave.contraction.execution.reduction import bind_precise_reduction
from lqcdweave.contraction.execution.operations.contract import bind_dense_action, _contract_value
from lqcdweave.contraction.execution.lowering import lower_binary_matmul, bind_matmul, execute_matmul


class PreciseReductionTests(unittest.TestCase):
    def fixture(self):
        n = 50
        p = np.zeros((4, 4, n, n), dtype=np.complex64)
        for spin in range(4):
            p[spin, spin] = np.eye(n, dtype=np.complex64) * np.float32(128.125)
        p[0, 0, 0, 0] = np.nextafter(np.float32(128.125), np.float32(np.inf))
        gamma = np.diag([1, 1, -1, -1]).astype(np.complex64)
        vertices = np.stack([gamma[:, :, None, None] * np.eye(n),
                             -gamma[:, :, None, None] * np.eye(n),
                             1j * gamma[:, :, None, None] * np.eye(n)]).astype(np.complex64)
        equation = AxisPattern(((0, 1, 2, 3), (4, 1, 0, 3, 2)), (4,))
        return equation, p, vertices

    def test_cancelling_trace_matches_double_reference_in_all_dense_entries(self):
        equation, p, vertices = self.fixture()
        reference = np.einsum(p, equation.inputs[0], vertices, equation.inputs[1],
                              equation.output, dtype=np.complex128, optimize=False)
        old = np.einsum(p, equation.inputs[0], vertices, equation.inputs[1], equation.output,
                        optimize=False)
        self.assertGreater(np.max(abs(old-reference)), 1e-6)
        fn = bind_precise_reduction(equation, (p.shape, vertices.shape), (p.dtype, vertices.dtype), np)
        self.assertIsNotNone(fn)
        self.assertLessEqual(fn.workspace_bytes, fn.workspace_allowance)
        self.assertGreater(fn.tile_count, 1)
        catalog = {'p': TensorSpec(p.shape, p.dtype), 'v': TensorSpec(vertices.shape, vertices.dtype)}
        action = Action('contract', 'loop', ('p', 'v'), equation)
        bound = bind_dense_action(action, catalog, np)
        lowering = lower_binary_matmul(equation, (p.shape, vertices.shape), (p.dtype, vertices.dtype))
        calls = [lambda: fn(p, vertices), lambda: bound((p, vertices))[0],
                 lambda: _contract_value(equation, (p, vertices), np, False)[0],
                 lambda: bind_matmul(lowering, np)(p, vertices),
                 lambda: execute_matmul(lowering, p, vertices, np)]
        for run in calls:
            actual = run()
            self.assertEqual(actual.dtype, np.dtype('complex64'))
            np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-12)
        first = fn(p, vertices).copy()
        second = fn(2*p, vertices)
        np.testing.assert_array_equal(second, 2*first)
        np.testing.assert_array_equal(fn(p, vertices), first)

    def test_diagonal_slicing_real_dtype_and_offline_backend_share_rule(self):
        a = np.zeros((4, 4, 50, 50), dtype=np.float32)
        for i in range(4):
            a[i, i] = (1 if i < 2 else -1) * np.eye(50, dtype=np.float32) * 128.125
        a[0, 0, 0, 0] = np.nextafter(np.float32(128.125), np.float32(np.inf))
        equation = AxisPattern(((0, 0, 1, 1),), ())
        reference = np.einsum('iijj->', a, dtype=np.float64)
        for backend in (np, SimpleNamespace(__name__='cupy', einsum=np.einsum)):
            fn = bind_precise_reduction(equation, (a.shape,), (a.dtype,), backend)
            self.assertIsNotNone(fn)
            self.assertLessEqual(fn.workspace_bytes, fn.workspace_allowance)
            self.assertEqual(fn(a).dtype, np.dtype('float32'))
            self.assertEqual(fn(a), reference)

    def test_matrix_products_double_and_tiny_allowances_keep_existing_routes(self):
        self.assertIsNone(bind_precise_reduction('ij,jk->ik', ((20, 30), (30, 40)),
                                                ('complex64',)*2, np))
        self.assertIsNone(bind_precise_reduction('i,i->', ((128,),)*2, ('complex128',)*2, np))
        self.assertIsNone(bind_precise_reduction('i->', ((2,),), ('float32',), np))


if __name__ == '__main__':
    unittest.main()
