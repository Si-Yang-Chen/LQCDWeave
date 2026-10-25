"""Producer support proofs and n=1 weights against full spatial operators."""

import unittest
from math import prod

import numpy as np

from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.spatial_sampling import SpatialSamplingDesign
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.sampling.spatial.api import sample_spatial_points
from production.elemental.test_elemental import fixture, point_basis, reference_matrix


class PointDiagonalProductionTests(unittest.TestCase):
    def test_structural_proof_local_closed_displaced_and_temporal_paths(self):
        sizes, gauge, vectors, _ = fixture()
        paths = ((), (0, 3), (0,) * sizes[0], (0,), (6,))
        components = tuple(ElementalComponent.path(path) for path in paths)
        table = [dict(component=c, path=p, coefficient=1)
                 for c, p in zip(components, paths)]
        for policy in ("same", "independent"):
            points = sample_spatial_points(SpatialSamplingDesign(sizes, 3, policy), seed=25)
            generator = ElementalGenerator(gauge, vectors, ((0, 0, 0),), table,
                                            points=points, blocks=("p2p",))
            expected = components[:3] + (components[4:] if policy == "same" else ())
            self.assertEqual(generator.point_diagonal_components, expected)
            for t in range(sizes[3]):
                blocks = generator.calc_path_blocks(t)["p2p"]
                for component, row in zip(components, blocks):
                    block = row[0]
                    self.assertEqual(block.point_diagonal, component in expected)
                    self.assertEqual(block.copy().point_diagonal, block.point_diagonal)
                    if block.point_diagonal:
                        np.testing.assert_array_equal(block.rows, block.columns)

    def test_one_sample_weighted_closed_path_against_full_site_oracle(self):
        sizes, gauge, vectors, _ = fixture()
        points = sample_spatial_points(SpatialSamplingDesign(sizes, 1, "same"), seed=7)
        path, coefficient = (0, 3), 0.3 + 0.7j
        component = ElementalComponent.path(path)
        generator = ElementalGenerator(
            gauge, vectors, ((1, -1, 0),),
            [dict(component=component, path=path, coefficient=coefficient)],
            points=points, blocks=("p2p",), elemental_weighted=True,
        )
        for t in (0, 2, 0):
            block = generator.calc_path_blocks(t)["p2p"][0][0]
            basis = point_basis(points.coordinates[t], sizes)
            full = coefficient * reference_matrix(gauge, path, t, (1, -1, 0))
            expected = (basis.conj().T @ full @ basis) * prod(sizes[:3])
            self.assertTrue(block.point_diagonal)
            np.testing.assert_allclose(block.to_dense().reshape(expected.shape), expected,
                                       rtol=1e-12, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
