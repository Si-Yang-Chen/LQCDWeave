"""Repeated selected factors use call-local descriptions, never payload reads."""
import unittest
from unittest.mock import patch

from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.expansion.api import NativeDiagramBinding, expand_correlator
from lqcdweave.contraction.expansion.state_graph import lower_state_graph
from lqcdweave.contraction.expansion import physical_inputs
from lqcdweave.contracts.correlator.api import DiagramBindings, OutputNode
from lqcdweave.operators.correlator.api import split_eigenmodes
from lqcdweave.storage.contraction_plans.api import decode_plan_value, encode_plan_value
from test_mode_block_correlator import fixture


class MetadataReuseTests(unittest.TestCase):
    def test_256_terms_describe_each_selected_factor_and_edge_once(self):
        diagram, _, vertices, elemental, propagator = fixture(ne=4)
        blocked = split_eigenmodes(diagram, partitions=(0, 1, 2, 3, 4))
        bindings = DiagramBindings({'q': propagator}, vertices, 'cfg', usedNe=4)
        describe = Meson.describe_factors
        with patch.object(Meson, 'describe_factors', autospec=True, side_effect=describe) as factors, \
             patch.object(elemental, 'describe_block', wraps=elemental.describe_block) as elements, \
             patch.object(propagator, 'describe_block', wraps=propagator.describe_block) as edges, \
             patch('lqcdweave.contraction.expansion.legacy_diagram.lower_state_graph',
                   wraps=lower_state_graph) as networks, \
             patch.object(physical_inputs, 'logical_vertex_tail_shape',
                          wraps=physical_inputs.logical_vertex_tail_shape) as tails, \
             patch.object(physical_inputs, '_logical_vertex_endpoint_bindings',
                          wraps=physical_inputs._logical_vertex_endpoint_bindings) as endpoints, \
             patch.object(elemental, 'block', side_effect=AssertionError('metadata read elemental')), \
             patch.object(propagator, 'block', side_effect=AssertionError('metadata read propagator')):
            expanded = expand_correlator(blocked, adapter=NativeDiagramBinding(), bindings=bindings)
        self.assertEqual(len(expanded.terms), 256)
        self.assertEqual(factors.call_count, 32)
        self.assertEqual(elements.call_count, 32)
        self.assertEqual(edges.call_count, 32)
        self.assertEqual(networks.call_count, 1)
        self.assertEqual(tails.call_count, 32)
        self.assertEqual(endpoints.call_count, 32)
        # Full immutable recipes remain available; no repr or digest replaces
        # their equality check or persistence through the existing codec.
        topology = expanded.inputs.batch_topology
        self.assertTrue(all(isinstance(node, OutputNode) for _, node in topology[1]))
        self.assertEqual(decode_plan_value(encode_plan_value(topology)), topology)
        hash(topology)

    def test_descriptions_are_fresh_for_each_expand_call(self):
        diagram, _, vertices, elemental, propagator = fixture(ne=4)
        bindings = DiagramBindings({'q': propagator}, vertices, 'cfg', usedNe=4)
        adapter = NativeDiagramBinding()
        with patch.object(elemental, 'describe_block', wraps=elemental.describe_block) as e, \
             patch.object(propagator, 'describe_block', wraps=propagator.describe_block) as p:
            first = expand_correlator(diagram, adapter=adapter, bindings=bindings)
            counts = e.call_count, p.call_count
            second = expand_correlator(diagram, adapter=adapter, bindings=bindings)
        self.assertEqual((e.call_count, p.call_count), tuple(2*n for n in counts))
        self.assertEqual(first.problem, second.problem)
        self.assertIsNot(first.inputs.loader, second.inputs.loader)


if __name__ == '__main__':
    unittest.main()
