"""Path choice, sharing and numeric semantics of integer compiled DAGs."""
import unittest
from unittest.mock import patch

import numpy as np
import opt_einsum

from lqcdweave.contracts.correlator.api import AxisPattern, ContractionTerm, TensorSpec
from lqcdweave.contraction.planning.api import compile_plan
from lqcdweave.contraction.execution.api import execute_plan
from lqcdweave.contraction.execution.lowering import lower_binary_matmul, execute_matmul


class IntegerCompilerTests(unittest.TestCase):
    def test_aligned_axis_is_retained_and_other_axis_summed(self):
        a = np.arange(6., dtype=np.float64).reshape(2, 3)
        b = np.arange(10., 16.).reshape(2, 3)
        specs = {"a": TensorSpec(a.shape, a.dtype), "b": TensorSpec(b.shape, b.dtype)}
        pattern = AxisPattern(((10000, 90), (10000, 90)), (10000,))
        with patch.object(opt_einsum, "contract_path", side_effect=AssertionError("string path wrapper")):
            plan = compile_plan(specs, [ContractionTerm(("a", "b"), pattern)])
        with patch("lqcdweave.contraction.execution.executor.einsum_equation", side_effect=AssertionError("rendered string")):
            actual = execute_plan(plan, {"a": a, "b": b}).values[0]
        expected = np.array([sum(a[i, j]*b[i, j] for j in range(3)) for i in range(2)])
        np.testing.assert_array_equal(actual, expected)
        lowering = lower_binary_matmul(plan.operations[0].equation, (a.shape, b.shape), (a.dtype, b.dtype))
        self.assertIsNotNone(lowering)
        np.testing.assert_array_equal(execute_matmul(lowering, b, a, np), expected)

    def test_diagonal_hyperedge_scalar_and_output_permutation(self):
        a = np.arange(9., dtype=np.float64).reshape(3, 3)
        x = np.array([2., 3., 5.])
        specs = {"a": TensorSpec(a.shape, a.dtype), "x": TensorSpec(x.shape, x.dtype)}
        terms = [
            ContractionTerm(("a",), AxisPattern(((500, 500),), ())),
            ContractionTerm(("x", "x", "x"), AxisPattern(((9,),)*3, ())),
            ContractionTerm(("a",), AxisPattern(((40, 70),), (70, 40))),
        ]
        plan = compile_plan(specs, terms)
        values = execute_plan(plan, {"a": a, "x": x}).values
        self.assertEqual(values[0], sum(a[i, i] for i in range(3)))
        self.assertEqual(values[1], sum(x[i]**3 for i in range(3)))
        np.testing.assert_array_equal(values[2], a.T)

    def test_dummy_renaming_shares_nodes_but_changes_no_axis_relation(self):
        specs = {"a": TensorSpec((2, 3), "float64"), "b": TensorSpec((3, 4), "float64")}
        terms = [ContractionTerm(("a", "b"), pattern) for pattern in (
            "ij,jk->ik", AxisPattern(((100, 200), (200, 300)), (100, 300)))]
        plan = compile_plan(specs, terms)
        self.assertEqual(plan.terms[0].output, plan.terms[1].output)
        self.assertEqual(plan.unique_operations, 1)
        self.assertEqual(plan.operation_use_counts, (2,))
        self.assertIsInstance(plan.operations[0].equation, AxisPattern)

    def test_uses_same_path_choices_as_previous_wrapper(self):
        specs = {"a": TensorSpec((2, 3), "float64"), "b": TensorSpec((3, 4), "float64"),
                 "c": TensorSpec((4, 5), "float64")}
        external_equation = "ij,jk,kl->il"
        term = ContractionTerm(tuple(specs), external_equation)
        for option in ("greedy", "optimal", "dynamic-programming", "auto", [(0, 1), (0, 1)]):
            with self.subTest(option=option):
                old_path, _ = opt_einsum.contract_path(external_equation, *(s.shape for s in specs.values()), shapes=True, optimize=option)
                direct = compile_plan(specs, [term], optimize=option)
                explicit = compile_plan(specs, [term], optimize=old_path)
                self.assertEqual(direct.operations, explicit.operations)

    def test_metadata_not_limited_to_52_symbols(self):
        axes = tuple(range(1000, 1060))
        plan = compile_plan({"a": TensorSpec((1,)*60, "float64")},
                            [ContractionTerm(("a",), AxisPattern((axes,), axes))])
        self.assertEqual(plan.terms[0].output_subscript, axes)


if __name__ == "__main__":
    unittest.main()
