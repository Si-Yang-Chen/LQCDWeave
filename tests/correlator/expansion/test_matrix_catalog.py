"""Matrix declarations preserve science without enumerating projected cells."""
from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch
import numpy as np
import sympy as sp

from test_structured_correlator import fixture, task
from lqcdweave.contracts.correlator.api import DiagramBindings, MatrixLinearAxis
from lqcdweave.contraction.expansion.api import describe_meson_matrix
from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, Tag, gen_correlator,
    iter_structured_wick_entries, structured_wick_templates,
    remove_disconneted_diagram, split_meson_mode_pairs,
)
from lqcdweave.contraction.correlator import calculate


class MatrixCatalogTests(unittest.TestCase):
    def setUp(self):
        self.bind, self.elemental, self.propagator = fixture()
        self.bindings = DiagramBindings(vertex_map=self.bind, usedNe=3, configuration='cfg')

    def describe(self, value):
        return describe_meson_matrix(value, bindings=self.bindings)

    def test_no_cell_expansion_selections_or_numeric_reads(self):
        full = split_meson_mode_pairs(task(structured=True, flavor='I0'), partitions=[0,1,2,3])
        with patch('lqcdweave.contraction.expansion.structured._Compiler.task',
                   side_effect=AssertionError('cell expansion')), \
             patch('lqcdweave.operators.correlator.structured.iter_structured_wick_entries',
                   side_effect=AssertionError('cell Wick enumeration')), \
             patch.object(self.elemental, 'block', side_effect=AssertionError('numeric read')), \
             patch.object(self.elemental, 'read_slab', side_effect=AssertionError('numeric read')):
            described = self.describe(full)
        catalog = described.catalog
        self.assertEqual(len(catalog.groups), 1)
        self.assertEqual(len(catalog.axes), 2)
        self.assertEqual(catalog.groups[0].partitions, (0,1,2,3))
        self.assertEqual(catalog.output.kind, 'matrix')
        self.assertEqual(catalog.output.shape, (2,2,6,6))
        self.assertEqual(catalog.output.children, ())
        self.assertEqual(len(catalog.components), 4)
        self.assertTrue(all(vertex.mode_indices is None for vertex in described.component_bindings))
        self.assertEqual([len(axis.coefficients) for axis in catalog.axes], [3,3])

    def test_component_and_axis_reuse_across_output_groups(self):
        full = task(structured=True, flavor='I0')
        connected = remove_disconneted_diagram(full, [r'S^q_\mathrm{local}'])
        result = self.describe({'full':full, 'connected':connected, 'difference':full-connected, 'copy':full})
        self.assertEqual(len(result.catalog.groups), 2)
        self.assertEqual(len(result.catalog.axes), 2)
        self.assertEqual(len(result.catalog.components), 4)
        self.assertEqual(result.catalog.groups[0].axes, result.catalog.groups[1].axes)
        self.assertIs(result.catalog.output.children[0], result.catalog.output.children[3])

    def test_wick_templates_reconstruct_existing_entries_including_mixed_flavors(self):
        rows = [HadronIrrep(name,(0,0,0),'A_1',-1,Tag(0,0))[0] for name in ('z','a')]
        axis = [Hadron(rows[0],Flavor('ud')), Hadron(rows[1],Flavor('uu')+Flavor('dd'))]
        request = gen_correlator([axis.copy(),axis.copy()], [0,1], [False,True],
                                 structured=True,degenerate=True)
        # Use existing graph replacement semantics to compare the complete
        # flavour matrix, including rectangles with zero Wick contribution.
        templates = tuple(structured_wick_templates(request))
        for index, expected_entries in iter_structured_wick_entries(request):
            expected = sum(coefficient * sp.Mul(*diagrams) for coefficient,diagrams in expected_entries)
            replacement = tuple(request.axes[a][i][0] for a,i in enumerate(index))
            actual = 0
            for positions, entries in templates:
                if all(i in selected for i, selected in zip(index, positions)):
                    for coefficient, diagrams in entries:
                        factors = [Diagram(d.diagram,d.time_list,
                                           [replacement[int(slot)] for slot in d.vertex_list],d.propagator_list)
                                   for d in diagrams]
                        actual += coefficient * sp.Mul(*factors)
            self.assertEqual(sp.expand(actual - expected), 0)

    def test_axis_complex_coefficients_and_source_dagger(self):
        described = self.describe(task(structured=True))
        left,right = described.catalog.axes
        np.testing.assert_allclose(left.coefficients, [.3+.7j,-.4j,1-.2j])
        np.testing.assert_allclose(right.coefficients, [.3-.7j,.4j,1+.2j])
        np.testing.assert_array_equal(left.offsets,[0,2,3])
        self.assertFalse(left.coefficients.flags.writeable)
        self.assertTrue(all(c.factor.operator_term.coefficient == 1 for c in described.catalog.components))

    def test_linear_position_weights_and_cancellation(self):
        a = HadronIrrep('a',(0,0,0),'A_1',-1,Tag(0,0))[0]
        z = HadronIrrep('z',(0,0,0),'A_1',-1,Tag(0,0))[0]
        axis = [Hadron((1+sp.I)*a + sp.sqrt(2)*z, Flavor('ud'))]
        request = gen_correlator([axis.copy(),axis.copy()],structured=True)
        result = self.describe(request)
        for side,expr in enumerate((request.axes[0][0][0],request.axes[1][0][0])):
            expected = {}
            from lqcdweave.contraction.binding.api import structured_vertex_components
            registry={}
            for summand in sp.Add.make_args(expr.expand()):
                # Numeric non-real factors are also part of the coefficient.
                row = next(iter(summand.atoms(type(a))))
                coefficient = complex(summand / row)
                for vertex,weight in structured_vertex_components(self.bind(row),registry=registry):
                    key=vertex.operator.terms[0].gamma
                    expected[key]=expected.get(key,0)+coefficient*weight
            actual={}
            selected=result.catalog.axes[side]
            for component,weight in zip(selected.component_ids,selected.coefficients):
                key=result.catalog.components[int(component)].factor.operator_term.gamma
                actual[key]=actual.get(key,0)+weight
            self.assertEqual(set(actual),set(expected))
            for key in expected:
                self.assertAlmostEqual(actual[key],expected[key])

    def test_numerical_catalog_recipes_match_existing_unprojected_requests(self):
        # Execute only in this independent small reference: catalog construction
        # itself never invokes the old numerical path or its component products.
        from lqcdweave.operators.correlator.symbolic_graph import QuarkDiagram
        for flavor in ('charged','I0','I1'):
            described=self.describe(task(structured=True,flavor=flavor))
            catalog=described.catalog
            group=catalog.groups[0]
            axes=[catalog.axes[i] for i in group.axes]
            expressions=np.empty((2,2),dtype=object)
            vertices={('component',i):v for i,v in enumerate(described.component_bindings)}
            for index in np.ndindex(expressions.shape):
                result=0
                choices=[tuple(zip(axis.component_ids[axis.offsets[i]:axis.offsets[i+1]],
                                   axis.coefficients[axis.offsets[i]:axis.offsets[i+1]]))
                         for axis,i in zip(axes,index)]
                for block in group.wick_blocks:
                    if not all(i in selected for i,selected in zip(index,block.axis_positions)):
                        continue
                    for selected in product(*choices):
                        for monomial in block.products:
                            factors=[]
                            for network_id in monomial.networks:
                                network=catalog.networks[network_id]
                                adjacency=[[list(item) if isinstance(item,tuple) else item for item in row]
                                           for row in network.adjacency]
                                factors.append(Diagram(QuarkDiagram(adjacency), list(network.time_labels),
                                    [('component',int(selected[int(slot)][0])) for slot in network.vertex_slots],
                                    list(network.propagators)))
                            result += monomial.coefficient * np.prod([v[1] for v in selected]) * sp.Mul(*factors)
                expressions[index]=result
            options=dict(propagator_map={'S^q':self.propagator,r'S^q_\mathrm{local}':self.propagator},
                         configuration='cfg',usedNe=3,optimize='none',memory_limit=1<<30)
            actual=calculate(expressions,vertex_map=vertices,**options).value
            expected=calculate(task(flavor=flavor),vertex_map=self.bind,**options).value
            np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)

    def test_partitions_rejected_and_representative_times_bound_without_mutation(self):
        request=task(structured=True)
        with self.assertRaises(ValueError):
            self.describe(replace(request,partitions=(0,1,4)))
        result=describe_meson_matrix(request,bindings=self.bindings,representative_times={0:1,1:0})
        self.assertEqual(result.representative_times,((0,1),(1,0)))
        self.assertEqual([component.factor.anchor_time for component in result.catalog.components],[1,1,0,0])
        self.assertTrue(all(vertex.usedNe is None for vertex in result.component_bindings))

    def test_single_loop_uses_its_insertion_label_not_axis_position(self):
        rows=[Hadron(HadronIrrep('a',(0,0,0),'A_1',-1,Tag(0,0))[0],Flavor('uu'))]
        request=gen_correlator([rows],[1],[True],structured=True,degenerate=True)
        result=describe_meson_matrix(request,bindings=self.bindings,representative_times={1:0})
        self.assertEqual(len(result.catalog.axes),1)
        self.assertEqual(result.catalog.components[0].time_label,1)
        self.assertEqual(result.catalog.components[0].factor.anchor_time,0)
        self.assertEqual(result.catalog.networks[0].time_labels,(1,))

    def test_cycles_and_invalid_coefficient_tables_rejected(self):
        cycle=[]
        cycle.append(cycle)
        with self.assertRaises(ValueError):
            self.describe(cycle)
        with self.assertRaises(ValueError):
            MatrixLinearAxis(('a',),[0,2],[0],[1])
        with self.assertRaises(ValueError):
            MatrixLinearAxis(('a',),[0,1],[0],[float('nan')])


if __name__ == '__main__':
    unittest.main()
