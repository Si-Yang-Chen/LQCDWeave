"""Row output selection keeps coordinate axes and shares only internal assembly."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from lqcdweave.contracts.correlator.api import (
    OutputNode, OutputRecipe, TermContribution, describe_coordinates,
)
from lqcdweave.contracts.execution import TermValue
from lqcdweave.contraction.results.api import assemble_correlator
import lqcdweave.contraction.results.core as core


class SelectedRowTests(unittest.TestCase):
    def test_selects_result_axis_before_scan_and_assembles_shared_row_once(self):
        coordinates, _ = describe_coordinates(time_axes={'t':(2,0)})
        row = OutputNode('term', TermContribution('row'))
        root = OutputNode('array', children=tuple(OutputNode('select',(i,),(row,)) for i in (2,0)), shape=(2,))
        expanded = SimpleNamespace(output_recipe=OutputRecipe(root,('row',),coordinates),coordinates=coordinates)
        terms = tuple(TermValue('row',point.key,np.array([1,2,3])+10*j)
                      for j,point in enumerate(coordinates.points))
        with patch.object(core,'_assemble_coordinate_values',wraps=core._assemble_coordinate_values) as assembly:
            actual = assemble_correlator(expanded,terms)
        np.testing.assert_array_equal(actual, [[3,13],[1,11]])
        self.assertEqual(assembly.call_count,1)
        self.assertFalse(np.shares_memory(actual,terms[0].value))

    def test_malformed_and_unknown_selection_nodes_fail(self):
        coordinates,_ = describe_coordinates()
        row = OutputNode('term',TermContribution('row'))
        for node in (OutputNode('select',(-1,),(row,)),OutputNode('select',(0,),()),
                     OutputNode('select',(),(row,)),OutputNode('unknown',children=(row,))):
            with self.subTest(node=node), self.assertRaises(ValueError):
                core._validate_output_node(node,{'row'})


if __name__ == '__main__':
    unittest.main()
