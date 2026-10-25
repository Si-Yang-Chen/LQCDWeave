"""Lazy convenience iteration delegates all numerical work to calc."""

import unittest
from unittest.mock import Mock

from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator


class LaplaceIterationTests(unittest.TestCase):
    def setUp(self):
        self.generator = LaplaceEigenvectorGenerator((2, 2, 2, 3), Mock(), 2, 1e-9,
                                                     backend="numpy")

    def test_all_times_are_lazy_and_each_result_is_forwarded(self):
        vectors, values = object(), object()
        self.generator.calc = Mock(return_value=(vectors, values))
        slices = self.generator.iter_slices()
        self.generator.calc.assert_not_called()
        self.assertEqual(next(slices), (0, vectors, values))
        self.assertEqual(self.generator.calc.call_count, 1)
        self.assertEqual([t for t, _, _ in slices], [1, 2])
        self.assertEqual([call.args[0] for call in self.generator.calc.call_args_list], [0, 1, 2])

    def test_explicit_time_order_and_options_are_preserved(self):
        self.generator.calc = Mock(return_value=(None, None))
        rows = self.generator.iter_slices([2, 0], apply_renorm_phase=False,
                                          polynomial_degree=8, lambda_cut=1.0)
        self.assertEqual([row[0] for row in rows], [2, 0])
        self.generator.calc.assert_called_with(0, apply_renorm_phase=False,
                                                polynomial_degree=8, lambda_cut=1.0)

    def test_not_loaded_and_solver_failure_propagate(self):
        with self.assertRaises(RuntimeError):
            next(self.generator.iter_slices())
        self.generator.calc = Mock(side_effect=[(None, None), RuntimeError("solver failed")])
        slices = self.generator.iter_slices()
        next(slices)
        with self.assertRaisesRegex(RuntimeError, "solver failed"):
            next(slices)
        self.assertEqual(self.generator.calc.call_count, 2)
