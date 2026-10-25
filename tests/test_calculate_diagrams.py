import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.vsv_diagram import VSVData
from lqcdweave.contraction import TensorSpec
from lqcdweave.contraction.planning.api import PathSearchExhausted
from lqcdweave.contraction.vsv_graph import evaluate_vsv_matrix
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import calculate


class CalculateDiagramsTests(unittest.TestCase):
    def case(self):
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ["a", "b"], [None, "q"])
        matrix = np.array([[diagram, 2j * diagram], [diagram ** 2, -diagram]], dtype=object)
        rng = np.random.default_rng(18)
        tensors = [rng.normal(size=(4, 4, 1, 1)) + 1j * rng.normal(size=(4, 4, 1, 1)) for _ in range(4)]
        calls = []
        def vertex(row, time):
            calls.append((row, time))
            return tensors[0 if row == "a" else 1]
        def prop(tag, sink, source):
            calls.append((tag, sink, source))
            return tensors[2 if source == 0 else 3]
        data = VSVData({0: 0, 1: 1}, lambda *a: TensorSpec((4, 4, 1, 1), "complex128"),
                       lambda *a: TensorSpec((4, 4, 1, 1), "complex128"),
                       vertex, prop, "synthetic-test")
        return matrix, data, calls

    def test_plan_only_then_replay_without_search_or_duplicate_reads(self):
        matrix, data, calls = self.case()
        result = calculate(matrix, data=data, memory_limit=1 << 20,
                           optimize="anytime", execute=False)
        self.assertEqual(calls, [])
        self.assertIsNone(result.values)
        self.assertTrue(result.plan.feasible)
        self.assertEqual(len(result.plan.state.problem.terms), 1)
        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError):
            evaluated = result.run(measure=True)
        self.assertEqual(len(calls), 4)
        self.assertIsNotNone(evaluated.report)
        reference = evaluate_vsv_matrix(matrix, data.vertex_getter, data.propagator_getter, data.time_map)
        np.testing.assert_allclose(evaluated.values, reference, rtol=1e-12)

    def test_optimizer_and_storage_selection_preserve_result(self):
        matrix, data, calls = self.case()
        reference = evaluate_vsv_matrix(matrix, data.vertex_getter, data.propagator_getter, data.time_map)
        for optimize, storage in (("anytime", "lifetime"), ("anytime", "bounded")):
            with self.subTest(optimize=optimize):
                result = calculate(matrix, data=data, memory_limit=1 << 20,
                                   optimize=optimize, storage=storage,
                                   search={"deadline_seconds": 2, "max_states": 1000})
                np.testing.assert_allclose(result.values, reference, rtol=1e-12)

    def test_rejection_before_reads_and_actual_shape_check(self):
        matrix, data, calls = self.case()
        with self.assertRaises(PathSearchExhausted):
            calculate(matrix, data=data, memory_limit=1)
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            calculate(matrix, data=data, memory_limit=10000, storage="unknown")
        self.assertEqual(calls, [])
        from dataclasses import replace
        wrong = replace(data, vertex_getter=lambda *a: np.ones((4, 4, 2, 2), dtype=complex))
        result = calculate(matrix, data=wrong, memory_limit=1 << 20, optimize="anytime", execute=False)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            result.run()


if __name__ == "__main__":
    unittest.main()
