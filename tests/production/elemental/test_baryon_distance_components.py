"""Module tests split from test_qdp_production_blocks.py; shared numerical fixtures stay unchanged."""

from test_qdp_production_blocks import (
    BaryonElementalGenerator,
    _QDPProductionBlockFixture,
)


class QDPProductionBlockTests(_QDPProductionBlockFixture):
    def test_distance_components_keep_explicit_flavors(self):
        component, = BaryonElementalGenerator.distance_components(
            [(0,1,2)], flavors=("u","d","s"))
        self.assertEqual(component.flavors, ("u","d","s"))
        default, = BaryonElementalGenerator.distance_components([(0,0,0)])
        self.assertEqual(default.flavors, ("q","q","q"))
