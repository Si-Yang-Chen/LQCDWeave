"""Selected physical Laplace solves on bounded synthetic CPU gauges."""

import unittest
from unittest.mock import Mock

import numpy as np

from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from tests.fixtures.laplace_reference import GaugeHandle, dense_laplacian, random_gauge, scipy


class SparseLaplaceTests(unittest.TestCase):
    def test_selection_preserves_wrap_order_and_is_lazy(self):
        selection = TimeSelectionRecord(4, (3, 0, 2))
        generator = LaplaceEigenvectorGenerator(
            (2, 2, 2, 4), Mock(), 2, 1e-10, selection=selection, backend="numpy")
        generator.calc = Mock(return_value=(None, None))
        rows = generator.iter_slices()
        generator.calc.assert_not_called()
        self.assertEqual([row[0] for row in rows], [3, 0, 2])
        self.assertEqual([call.args[0] for call in generator.calc.call_args_list], [3, 0, 2])
        self.assertEqual((generator.Lt, generator.Nt), (4, 3))
        self.assertEqual(generator.describe()["time_selection"], selection.document)
        self.assertEqual([row[0] for row in generator.iter_slices(
            selection=TimeSelectionRecord(4, (1, 0)))], [1, 0])
        with self.assertRaises(ValueError):
            list(generator.iter_slices([0], selection=selection))
        with self.assertRaises(ValueError):
            LaplaceEigenvectorGenerator((2, 2, 2, 4), Mock(), 2, 1e-10,
                                        selection=TimeSelectionRecord(3, (0,)),
                                        backend="numpy")

    @unittest.skipIf(scipy is None, "SciPy is required")
    def test_sparse_solutions_match_independent_fulltime_eigensystems(self):
        size = (2, 2, 2, 4)
        gauge = random_gauge(size)
        selection = TimeSelectionRecord(4, (3, 0, 2))
        generator = LaplaceEigenvectorGenerator(
            size, GaugeHandle(gauge), 2, 1e-11, selection=selection,
            backend="numpy").load("synthetic")
        # Independent complete-time dense operators are built before selection.
        references = [np.linalg.eigh(dense_laplacian(
            gauge[t].transpose(3, 0, 1, 2, 4, 5)[:3])) for t in range(size[3])]
        seen = []
        for t, vectors, values in generator.iter_slices():
            seen.append(t)
            expected_values, expected_vectors = references[t]
            actual = vectors.reshape(2, -1).T
            expected = expected_vectors[:, :2]
            np.testing.assert_allclose(values, expected_values[:2], atol=2e-10)
            np.testing.assert_allclose(actual @ actual.conj().T,
                                       expected @ expected.conj().T, atol=2e-9)
        self.assertEqual(seen, list(selection.times))
