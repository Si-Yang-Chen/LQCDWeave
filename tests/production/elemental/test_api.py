"""Independent public-boundary checks for elemental production."""

import unittest

import numpy as np

from lqcdweave.contracts.spatial_sampling import SpatialPointSet
from lqcdweave.production.elemental import (
    BaryonElementalGenerator,
    ElementalGenerator,
    ElementalWorkspace,
    generate_baryon_elemental,
    re_combine,
)
from lqcdweave.production.elemental import api
from lqcdweave.production.elemental.mode_weights import dilution_pair_weights


class _Handle:
    def __init__(self, values):
        self.values = values

    def load(self, key):
        return self.values


class ElementalPublicApiTests(unittest.TestCase):
    def test_package_exports_are_the_api_facade(self):
        exported = (
            ElementalGenerator,
            BaryonElementalGenerator,
            ElementalWorkspace,
            generate_baryon_elemental,
            re_combine,
        )
        for value in exported:
            self.assertIs(getattr(api, value.__name__), value)
        self.assertEqual(set(api.__all__), {value.__name__ for value in exported})

    def test_generators_have_one_public_calculation_method(self):
        for generator_type in (ElementalGenerator, BaryonElementalGenerator):
            methods = {name for name in vars(generator_type)
                       if name.startswith("calc") and callable(getattr(generator_type, name))}
            self.assertEqual(methods, {"calc_path_blocks", "calc_derivative"} if generator_type is ElementalGenerator else {"calc_path_blocks"})

    def test_contract_point_set_produces_owned_sparse_coo_blocks(self):
        sizes = (2, 1, 1, 2)
        lx, ly, lz, lt = sizes
        gauge = np.zeros((lt, lz, ly, lx, 4, 3, 3), dtype=np.complex128)
        gauge[...] = np.eye(3)
        coordinates = np.array([
            [[0, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 0, 0]],
        ], dtype=np.int64)
        point_set = SpatialPointSet(sizes, (0, 1), coordinates)
        generator = ElementalGenerator(
            gauge, None, ((0, 0, 0),),
            [{"component": "local", "path": (), "coefficient": 1}],
            points=point_set, blocks=("p2p",),
        )

        gauge.fill(0)
        result = generator.calc_path_blocks(0)
        block = result["p2p"][0][0]
        self.assertEqual(block.point_shape, (2, 2))
        np.testing.assert_array_equal(block.rows, [0, 1])
        np.testing.assert_array_equal(block.columns, [0, 1])
        np.testing.assert_allclose(block.values, np.broadcast_to(np.eye(3), (2, 3, 3)))
        dense = block.to_dense()
        self.assertEqual(dense.shape, (2, 3, 2, 3))
        np.testing.assert_allclose(dense[0, :, 0, :], np.eye(3))
        np.testing.assert_allclose(dense[1, :, 1, :], np.eye(3))

    def test_local_dilution_helper_keeps_pair_inclusion_weights(self):
        np.testing.assert_array_equal(
            dilution_pair_weights([4], [2]),
            np.array([[2.0, 6.0], [6.0, 2.0]]),
        )
        with self.assertRaises(ValueError):
            dilution_pair_weights([4], [1])


if __name__ == "__main__":
    unittest.main()
