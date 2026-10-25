"""Temporary (slow, about 70 s on CSNS CPU) case moved from ``tests/correlator/execution/test_device_output_program.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from correlator.execution.test_device_output_program import GeneratedOutputCodeTests
from correlator.execution.test_device_output_program import ContractionTerm, OutputNode, OutputRecipe, SimpleNamespace, TensorSpec, _run_host_generated, bind_output_assembly, describe_coordinates, np  # noqa: F401


class TemporaryGeneratedOutputCodeTests(GeneratedOutputCodeTests):
    def test_23056_term_segmented_ordered_sum_matches_cpu_recipe(self):
        count = 23056
        coordinates, _ = describe_coordinates()
        ids = tuple(f't{i}' for i in range(count))
        recipe = OutputRecipe(OutputNode('diagram_sum', tuple((key, (-1) ** i * (0.5 + 0.25j))
                              for i, key in enumerate(ids))), ids, coordinates)
        term = ContractionTerm(('input',), 'a->a')
        problem = SimpleNamespace(terms=(term,) * count, specs={'input': TensorSpec((2,), 'complex128')})
        rng = np.random.default_rng(23056)
        values = tuple(row for row in (rng.normal(size=(count, 2)) + 1j * rng.normal(size=(count, 2))))
        actual = _run_host_generated(recipe, problem, values, indirect=True, optimization='-O0')
        expected = bind_output_assembly(recipe)(values)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
        # Cancellation at stage boundaries cannot be reassociated into chunks.
        independent = np.zeros(2, dtype='complex128')
        for i, value in enumerate(values):
            independent = independent + ((-1) ** i * (0.5 + 0.25j)) * value
        np.testing.assert_allclose(actual, independent, rtol=3e-13, atol=3e-13)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryGeneratedOutputCodeTests("test_23056_term_segmented_ordered_sum_matches_cpu_recipe")])
