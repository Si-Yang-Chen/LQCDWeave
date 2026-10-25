"""Matrix block shapes and placement preserve trailing coordinate axes."""
from types import SimpleNamespace
import unittest
import numpy as np
from lqcdweave.contracts.correlator.api import (
    OutputNode,OutputRecipe,TermContribution,describe_coordinates,ContractionTerm,TensorSpec,
)
from lqcdweave.contracts.execution import TermValue
from lqcdweave.contraction.results.api import (assemble_correlator,estimate_output_retention_bytes,
                                                     bind_output_assembly)
from lqcdweave.contraction.results.core import _validate_output_node


class MatrixOutputTests(unittest.TestCase):
    def test_scatter_sum_matches_independent_dense_addition_and_owns_results(self):
        from unittest.mock import patch
        from dataclasses import replace
        positions = (((2,0),(3,1)), ((0,1),(1,2)), ((),(3,)))
        shape = (3,4,2)
        for dtypes in (('complex64',)*3, ('float32','float64','float32'), ('bool',)*3):
            for scanned in (False,True):
                with self.subTest(dtypes=dtypes,scanned=scanned):
                    coords,_ = describe_coordinates(time_axes={'t':(2,0)} if scanned else None)
                    blocks = tuple(OutputNode('scatter',p,(OutputNode('term',TermContribution(f'b{i}')),),
                                              shape=shape) for i,p in enumerate(positions))
                    recipe = OutputRecipe(OutputNode('add',children=blocks), ('b0','b1','b2'), coords)
                    # Explicit dtype is the production path after normalization.
                    if dtypes[0]=='complex64': recipe=replace(recipe,dtype='complex64')
                    arrays=[]; values=[]; expected=[]
                    for t,point in enumerate(coords.points):
                        full=[]; current=[]
                        for i,(p,dtype) in enumerate(zip(positions,dtypes)):
                            compact=(np.arange(len(p[0])*len(p[1])*2).reshape(len(p[0]),len(p[1]),2)+t-i).astype(dtype)
                            if dtype=='complex64': compact=compact*(1+2j)
                            arrays.append(compact.copy()); current.append(compact)
                            values.append(TermValue(f'b{i}',point.key,compact))
                            dense=np.zeros(shape,dtype=compact.dtype)
                            dense[np.ix_(*p)]=compact
                            full.append(dense)
                        expected.append(sum(full))
                    expected=np.stack(expected,axis=-1) if scanned else expected[0]
                    actual=assemble_correlator(SimpleNamespace(output_recipe=recipe,coordinates=coords),values)
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(actual.dtype,expected.dtype)
                    if not scanned:
                        bound=bind_output_assembly(recipe)
                        with patch('lqcdweave.contraction.results.matrix_output.np.zeros',wraps=np.zeros) as zeros:
                            first=bound(tuple(v.value for v in values))
                        self.assertEqual(sum(tuple(call.args[0])==shape for call in zeros.call_args_list),1)
                        np.testing.assert_array_equal(first,expected)
                        first[...] = 99
                        np.testing.assert_array_equal(bound(tuple(v.value for v in values)),expected)
                    for before,value in zip(arrays,values):
                        np.testing.assert_array_equal(value.value,before)

    def test_scatter_sum_retention_scales_with_compact_blocks(self):
        coords,_=describe_coordinates()
        count=40
        blocks=tuple(OutputNode('scatter',((i,),tuple(range(8))),
                     (OutputNode('term',TermContribution(f't{i}')),),shape=(count,8)) for i in range(count))
        recipe=OutputRecipe(OutputNode('add',children=blocks),tuple(f't{i}' for i in range(count)),coords)
        problem=SimpleNamespace(terms=tuple(ContractionTerm((f'x{i}',),'ij->ij') for i in range(count)),
            specs={f'x{i}':TensorSpec((1,8),'complex128') for i in range(count)})
        reserved=estimate_output_retention_bytes(recipe,problem,(),coords)
        payload=count*8*16
        self.assertGreaterEqual(reserved,2*payload)
        self.assertLess(reserved,8*payload)

    def test_scatter_sum_preserves_signed_zero_and_nonfinite_addition(self):
        from lqcdweave.contraction.results.matrix_output import evaluate_scatter_sum
        arrays=(np.array([complex(-0.,-0.),complex(np.inf,0),complex(np.nan,0),1e-30],np.complex64),
                np.array([complex(-0.,-0.),complex(-np.inf,0),complex(np.nan,0),-1e-30],np.complex64))
        blocks=tuple(OutputNode('scatter',((0,1,2,3),),
                     (OutputNode('constant_array',i),),shape=(5,)) for i in range(2))
        dense=[]
        for values in arrays:
            value=np.zeros(5,np.complex64); value[:4]=values; dense.append(value)
        with np.errstate(invalid='ignore'):
            expected=sum(dense)
            actual=evaluate_scatter_sum(blocks,lambda node:arrays[node.value],())
        np.testing.assert_array_equal(actual,expected)
        finite=np.isfinite(expected.view(np.float32))
        np.testing.assert_array_equal(actual.view(np.uint32)[finite],expected.view(np.uint32)[finite])

    def test_loop_outer_product_and_pc_axes_at_two_coordinates(self):
        coords,_ = describe_coordinates(time_axes={'t':(2,0)})
        left = OutputNode('reshape',children=(OutputNode('term',TermContribution('l')),),shape=(2,1,3,1))
        right = OutputNode('reshape',children=(OutputNode('term',TermContribution('r')),),shape=(1,4,1,3))
        root = OutputNode('multiply',children=(left,right))
        recipe = OutputRecipe(root,('l','r'),coords)
        inputs,expected = [],[]
        for i,point in enumerate(coords.points):
            l = np.arange(6).reshape(2,3)*(1+1j)+i
            r = np.arange(12).reshape(4,3)*(1-2j)+3*i
            inputs += [TermValue('l',point.key,l),TermValue('r',point.key,r)]
            expected.append(l[:,None,:,None]*r[None,:,None,:])
        actual = assemble_correlator(SimpleNamespace(output_recipe=recipe,coordinates=coords),inputs)
        np.testing.assert_equal(actual,np.stack(expected,axis=-1))
        problem = SimpleNamespace(terms=(ContractionTerm(('l',),'ip->ip'),ContractionTerm(('r',),'jq->jq')),
            specs={'l':TensorSpec((2,3),'complex128'),'r':TensorSpec((4,3),'complex128')})
        self.assertGreaterEqual(estimate_output_retention_bytes(recipe,problem,(),coords),actual.nbytes)

    def test_scatter_noncontiguous_positions_and_empty_axes(self):
        for positions in (((2,0),(3,1)),((),(3,1))):
            coords,_ = describe_coordinates(time_axes={'t':(1,0)})
            block_shape = (len(positions[0]),len(positions[1]),2)
            leaf = OutputNode('term',TermContribution('b'))
            root = OutputNode('scatter',positions,(leaf,),shape=(3,4,2))
            recipe = OutputRecipe(root,('b',),coords)
            values = [np.arange(np.prod(block_shape)).reshape(block_shape)+i for i in range(2)]
            actual = assemble_correlator(SimpleNamespace(output_recipe=recipe,coordinates=coords),
                tuple(TermValue('b',point.key,value) for point,value in zip(coords.points,values,strict=True)))
            expected = np.zeros((3,4,2,2),dtype=int)
            for i,row in enumerate(positions[0]):
                for j,col in enumerate(positions[1]):
                    for t in range(2):expected[row,col,:,t]=values[t][i,j,:]
            np.testing.assert_equal(actual,expected)
            problem = SimpleNamespace(terms=(ContractionTerm(('b',),'ijp->ijp'),),
                specs={'b':TensorSpec(block_shape,'int64')})
            self.assertGreaterEqual(estimate_output_retention_bytes(recipe,problem,(),coords),actual.nbytes)

    def test_zero_matrix_and_invalid_shapes(self):
        coords,_ = describe_coordinates()
        zeros = OutputNode('zeros','complex128',shape=(3,4,2,2))
        np.testing.assert_equal(assemble_correlator(SimpleNamespace(coordinates=coords,
            output_recipe=OutputRecipe(zeros,(),coords)),()),np.zeros((3,4,2,2),dtype=complex))
        leaf = OutputNode('term',TermContribution('x'))
        for node in (OutputNode('zeros','object',shape=(2,)),
                     OutputNode('reshape',children=(leaf,),shape=(-1,)),
                     OutputNode('scatter',((1,1),),(leaf,),shape=(2,)),
                     OutputNode('scatter',((2,),),(leaf,),shape=(2,)),
                     OutputNode('transpose',(0,0),(leaf,),shape=(2,3))):
            with self.assertRaises(ValueError):_validate_output_node(node,{'x'})
        bad = OutputNode('reshape',children=(leaf,),shape=(3,))
        with self.assertRaises(ValueError):
            assemble_correlator(SimpleNamespace(coordinates=coords,output_recipe=OutputRecipe(bad,('x',),coords)),
                (TermValue('x','bound',np.ones((2,))),))

    def test_pc_transpose_preserves_trailing_scan_axis(self):
        coords,_=describe_coordinates(time_axes={'t':(1,0)})
        leaf=OutputNode('term',TermContribution('b'))
        root=OutputNode('transpose',(2,3,0,1),(leaf,),shape=(2,4,3,5))
        recipe=OutputRecipe(root,('b',),coords)
        values=[np.arange(120).reshape(3,5,2,4)+i for i in range(2)]
        actual=assemble_correlator(SimpleNamespace(output_recipe=recipe,coordinates=coords),
            tuple(TermValue('b',point.key,value) for point,value in zip(coords.points,values,strict=True)))
        np.testing.assert_array_equal(actual,np.stack(values,axis=-1).transpose(2,3,0,1,4))
        problem=SimpleNamespace(terms=(ContractionTerm(('b',),'pqij->pqij'),),
            specs={'b':TensorSpec((3,5,2,4),'int64')})
        self.assertGreaterEqual(estimate_output_retention_bytes(recipe,problem,(),coords),actual.nbytes)


if __name__=='__main__':unittest.main()
