"""Single-current production against independent full-site matrices."""

from math import prod
import unittest
import numpy as np

from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.contracts.spatial_sampling import SpatialPointSet, SpatialSamplingDesign
from lqcdweave.production.elemental.api import ElementalGenerator
from lqcdweave.sampling.spatial.api import sample_spatial_points
from production.elemental.test_elemental import fixture, point_basis, reference_matrix


class WeightedElementalProductionTests(unittest.TestCase):
    def test_all_sectors_spatial_and_temporal_paths_against_site_oracle(self):
        sizes, gauge, vectors, _ = fixture()
        M, n = prod(sizes[:3]), 3
        paths = ((), (0,), (6,), (7,))
        momenta = ((0, 0, 0), (1, -1, 0))
        components = tuple(ElementalComponent.path(p) for p in paths)
        table = [dict(component=c, path=p, coefficient=0.3 + 0.7j)
                 for c, p in zip(components, paths)]
        for policy in ("same", "independent"):
            points = sample_spatial_points(SpatialSamplingDesign(sizes, n, policy), seed=25)
            generator = ElementalGenerator(
                gauge, vectors, momenta, table, points=points,
                blocks=("v2v", "v2p", "p2v", "p2p"),
                elemental_weighted=True, ne_block_size=1,
            )
            for t in (0, 2, 0):  # Repeated calculation must not accumulate weights.
                actual = generator.calc_path_blocks(t)
                for ri, path in enumerate(paths):
                    dt = components[ri].time_offsets[1]
                    endpoint = (t + dt) % sizes[3]
                    left = {"v": vectors[t].reshape(3, -1).T,
                            "p": point_basis(points.coordinates[t], sizes)}
                    right = {"v": vectors[endpoint].reshape(3, -1).T,
                             "p": point_basis(points.coordinates[endpoint], sizes)}
                    for mi, momentum in enumerate(momenta):
                        matrix = (0.3 + 0.7j) * reference_matrix(gauge, path, t, momentum)
                        for sector in actual:
                            expected = left[sector[0]].conj().T @ matrix @ right[sector[-1]]
                            if sector in ("v2p", "p2v"):
                                expected *= M / n
                            elif sector == "p2p":
                                if points.point_pool_id(t) == points.point_pool_id(endpoint):
                                    weight = np.full((n, n), M * (M - 1) / (n * (n - 1)))
                                    np.fill_diagonal(weight, M / n)
                                else:
                                    weight = np.full((n, n), (M / n) ** 2)
                                expected *= np.repeat(np.repeat(weight, 3, axis=0), 3, axis=1)
                            value = actual[sector][ri][mi]
                            if sector == "p2p":
                                value = value.to_dense()
                            np.testing.assert_allclose(value.reshape(expected.shape), expected,
                                                       rtol=1e-12, atol=1e-11)

    def test_weighted_callback_and_explicit_pool_provenance(self):
        sizes, gauge, vectors, _ = fixture()
        points = sample_spatial_points(SpatialSamplingDesign(sizes, 3, "same"), seed=2)
        component = ElementalComponent.path(())
        table = [dict(component=component, path=(), coefficient=1)]
        raw = ElementalGenerator(gauge, vectors, ((0, 0, 0),), table, points=points,
                                blocks=("v2p",)).calc_path_blocks(0)["v2p"].copy()
        delivered = []
        generator = ElementalGenerator(
            gauge, vectors, ((0, 0, 0),), table, points=points,
            blocks=("v2p",), elemental_weighted=True,
            write=lambda ri, mi, sector, slices, value: delivered.append(value.copy()),
        )
        self.assertIsNone(generator.calc_path_blocks(0))
        np.testing.assert_allclose(delivered[0], raw[0, 0] * 4)
        # Unbound owned point handles retain the existing automatic array load.
        via_handle = ElementalGenerator(
            gauge, vectors, ((0, 0, 0),), table, points=SpatialPointHandle(points),
            blocks=("v2p",), elemental_weighted=True,
        ).calc_path_blocks(0)["v2p"]
        np.testing.assert_allclose(via_handle, raw * 4)
        self.assertEqual(generator.sampling_pool_ids[component, 0, 0, "bar"], points.pool_ids[0])
        self.assertEqual(generator.sampling_groups, points.sampling_groups)
        again = sample_spatial_points(points.design, seed=2)
        np.testing.assert_array_equal(points.coordinates, again.coordinates)
        self.assertNotEqual(points.pool_ids, again.pool_ids)
        restored = SpatialPointSet(sizes, points.times, points.coordinates, points.design,
                                   pool_ids=points.pool_ids)
        self.assertEqual(restored.sampling_groups, points.sampling_groups)

    def test_unknown_sampling_and_boolean_options_are_rejected(self):
        sizes, gauge, vectors, p = fixture()
        table = [dict(component="local", path=(), coefficient=1)]
        with self.assertRaisesRegex(ValueError, "explicit sampling pool"):
            ElementalGenerator(gauge, vectors, ((0, 0, 0),), table,
                               points=p, blocks=("v2p",),
                               elemental_weighted=True)
        with self.assertRaises(TypeError):
            ElementalGenerator(gauge, vectors, ((0, 0, 0),), table, elemental_weighted=1)
        # V/V remains exact and requires no point sampling declaration.
        ElementalGenerator(gauge, vectors, ((0, 0, 0),), table,
                           elemental_weighted=True).calc_path_blocks(0)


if __name__ == "__main__":
    unittest.main()
