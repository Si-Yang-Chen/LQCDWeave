"""Published point-elemental datasets through the Diagram reader contract."""

from pathlib import Path
import json
import unittest

import numpy as np

import test_elemental_point_production as production_tests
from production.elemental.test_elemental import point_basis, reference_matrix
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.correlator.api import TensorAxisMetadata
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.storage.dataset import DatasetReader
from lqcdweave.runtime.datasets import DatasetView


class _DatasetPointElementalAdapterFixture(unittest.TestCase):
    def setUp(self):
        helper = production_tests.PointElementalProductionTests()
        helper.setUp()
        self.helper = helper
        self.addCleanup(helper.doCleanups)
        self.products = helper.run_products("adapter-products")
        self.readers = {
            sector: helper.index.open(identity)
            for sector, identity in self.products["datasets"].items()
        }
        for reader in self.readers.values():
            self.addCleanup(reader.close)
        self.points = helper.index.open(helper.points_id)
        self.addCleanup(self.points.close)
        self.reader = DatasetElemental(self.readers, point_dataset=self.points)
    def _expected_point_sparse(self, component, momentum, time, points_count):
        helper = self.helper
        path = component.key.steps
        displacement = np.zeros(3, dtype=np.int64)
        axes, signs = (0, 1, 2, 0, 1, 2), (1, 1, 1, -1, -1, -1)
        for step in path:
            displacement[axes[step]] += signs[step]
        coords = helper.points[time, :points_count]
        lookup = {tuple(xyz): index for index, xyz in enumerate(coords)}
        matrix = reference_matrix(helper.links, path, time,
                                 helper.momenta[momentum], phase="anchor")
        dense_points = point_basis(coords, helper.sizes)
        expected = dense_points.conj().T @ matrix @ dense_points
        pairs = []
        for left, xyz in enumerate(coords):
            target = tuple((xyz + displacement) % np.asarray(helper.sizes[:3]))
            right = lookup.get(target)
            if right is not None:
                pairs.append(((left, right), expected[left * 3:(left + 1) * 3,
                                                       right * 3:(right + 1) * 3]))
        return pairs
    def _expected_point_sparse_selection(self, component, momentum, time,
                                         left_positions, right_positions):
        helper = self.helper
        path = component.key.steps
        displacement = np.zeros(3, dtype=np.int64)
        axes, signs = (0, 1, 2, 0, 1, 2), (1, 1, 1, -1, -1, -1)
        for step in path:
            displacement[axes[step]] += signs[step]
        labels = tuple(self.readers["p2p"].spec.coordinates("point_left"))
        coords = helper.points[time, :len(labels)]
        matrix = reference_matrix(helper.links, path, time,
                                 helper.momenta[momentum], phase="anchor")
        dense_points = point_basis(coords, helper.sizes)
        expected = dense_points.conj().T @ matrix @ dense_points
        right_position = {position: selected
                          for selected, position in enumerate(right_positions)}
        right_by_coordinate = {
            tuple(coords[position]): position for position in right_positions
        }
        pairs = []
        for selected_left, left in enumerate(left_positions):
            target = tuple((coords[left] + displacement) % np.asarray(helper.sizes[:3]))
            right = right_by_coordinate.get(target)
            if right is not None:
                pairs.append(((selected_left, right_position[right]),
                              expected[left * 3:(left + 1) * 3,
                                       right * 3:(right + 1) * 3]))
        return pairs


class DatasetPointElementalAdapterTests(_DatasetPointElementalAdapterFixture):
    def test_real_published_mixed_sectors_match_independent_full_site_oracle(self):
        component = self.reader.components[1]
        time, momentum = 2, 1
        self.helper.assert_independent_reference(
            self.products, sectors=("v2p", "p2v", "p2p"),
            modes=3, points=3, phase="anchor",
        )
        raw_v2p = self.readers["v2p"].read(time=time)[1, momentum]
        raw_p2v = self.readers["p2v"].read(time=time)[1, momentum]
        np.testing.assert_array_equal(self.reader.block(component, momentum, time, "v2p"), raw_v2p)
        np.testing.assert_array_equal(self.reader.block(component, momentum, time, "p2v"), raw_p2v)

        sparse = self.reader.block(component, momentum, time, "p2p")
        expected_pairs = self._expected_point_sparse(component, momentum, time, 3)
        np.testing.assert_array_equal(sparse["indices"],
                                      np.asarray([pair for pair, _ in expected_pairs], dtype=np.int64))
        np.testing.assert_allclose(sparse["values"],
                                   np.asarray([value for _, value in expected_pairs]),
                                   atol=2e-12, rtol=2e-12)
        metadata = self.reader.describe_block(component, momentum, time, "p2p")
        self.assertIsInstance(metadata, TensorAxisMetadata)
        self.assertEqual(metadata.axes, ("point_left", "color_left", "point_right", "color_right"))
        self.assertEqual(metadata.sparse_layout, "coo")
        self.assertEqual(metadata.point_shape, (3, 3))
        self.assertEqual(metadata.nnz_upper, len(expected_pairs))

        operator = Operator.from_terms("published", [OperatorTerm(0, component, momentum, 1)])
        meson = Meson(self.reader, operator, phase_convention="anchor")
        for sector, expected in (("v2p", raw_v2p), ("p2v", raw_p2v)):
            description = meson.describe_factors(time, sector)[0]
            self.assertEqual(description.tensor.axes, self.reader.describe_block(
                component, momentum, time, sector).axes)
            self.assertEqual(dict(description.tensor.axis_bindings), dict(
                self.reader.describe_block(component, momentum, time, sector).axis_bindings,
            ))
            np.testing.assert_array_equal(meson.read_factor_block(description), expected)


if __name__ == "__main__":
    unittest.main()
