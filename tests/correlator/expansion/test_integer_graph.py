"""Direct graph connectivity numbering has no alphabet capacity constraint."""
import unittest
from unittest.mock import patch
from lqcdweave.contracts.correlator.api import AxisPattern
from lqcdweave.contraction.expansion.state_graph import lower_state_graph

class IntegerGraphTests(unittest.TestCase):
    def test_connected_graph_metadata_has_no_alphabet_limit(self):
        count = 16
        adjacency = [[0] * count for _ in range(count)]
        for source in range(count):
            adjacency[source][(source + 1) % count] = 1
        with patch('opt_einsum.get_symbol', side_effect=AssertionError('generated labels')):
            network, = lower_state_graph(adjacency, [{'left': 'v', 'right': 'v'}] * count)
        self.assertIsInstance(network.equation, AxisPattern)
        axes = {axis for part in network.equation.inputs for axis in part}
        self.assertEqual(len(axes), 4 * count)
        self.assertTrue(all(type(axis) is int for axis in axes))
