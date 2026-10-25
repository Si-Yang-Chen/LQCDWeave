"""Independent dense four-factor reference for shared K/H and PC blocks."""
import unittest
import numpy as np

from lqcdweave.contraction.execution.api import contract_meson_matrix_pc, contract_meson_loop_pc
from lqcdweave.contracts.correlator.api import parse_axis_pattern


def projected_vertices(vertices,boundaries):
    if boundaries is None:
        return (vertices,)
    slices=[slice(a,b) for a,b in zip(boundaries,boundaries[1:])]
    result=[]
    for a in range(len(slices)):
        for b in range(a,len(slices)):
            value=np.zeros_like(vertices)
            value[:,:,:,slices[a],slices[b]]=vertices[:,:,:,slices[a],slices[b]]
            if a!=b:
                value[:,:,:,slices[b],slices[a]]=vertices[:,:,:,slices[b],slices[a]]
                value/=np.sqrt(2)
            result.append(value)
    return tuple(result)


class MatrixKernelTests(unittest.TestCase):
    def test_selected_rectangles_have_four_independent_mode_widths(self):
        rng=np.random.default_rng(115)
        def random(shape): return rng.normal(size=shape)+1j*rng.normal(size=shape)
        p0,p1=random((4,4,1,2)),random((4,4,3,4))
        left,right=random((2,4,4,2,3)),random((3,4,4,4,1))
        lw,rw=random((2,2)),random((4,3))
        expected=np.einsum('BAON,DCQP,cADNQ,dCBPO,ic,jd->ij',
            p0,p1,left,right,lw,rw,optimize=True)
        actual=contract_meson_matrix_pc(p0,p1,left,right,lw,rw,mode_block=True,row_block=1)
        np.testing.assert_allclose(actual,expected,rtol=4e-12,atol=5e-11)
        vertices=random((2,4,4,2,1))
        expected=np.einsum('BAON,cABNO,ic->i',p0,vertices,lw,optimize=True)
        actual=contract_meson_loop_pc(p0,vertices,lw,mode_block=True)
        np.testing.assert_allclose(actual,expected,rtol=4e-12,atol=5e-11)

    def fixture(self,ne):
        rng=np.random.default_rng(1834+ne)
        random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
        p0,p1=(random((4,4,ne,ne)) for _ in range(2))
        left,right=random((3,4,4,ne,ne)),random((2,4,4,ne,ne))
        lw,rw=random((2,3)),random((3,2))
        return p0,p1,left,right,lw,rw

    def test_complex_rectangular_operator_axes_and_four_unequal_blocks(self):
        p0,p1,left,right,lw,rw=self.fixture(7)
        left_op=np.einsum('ic,cADNQ->iADNQ',lw,left)
        right_op=np.einsum('jc,cCBPO->jCBPO',rw,right)
        for boundaries in (None,(0,1,3,4,7)):
            lp,rp=projected_vertices(left_op,boundaries),projected_vertices(right_op,boundaries)
            expected=np.empty((2,3,len(lp),len(rp)),dtype=np.complex128)
            for a,v in enumerate(lp):
                for b,w in enumerate(rp):
                    expected[:,:,a,b]=np.einsum('BAON,DCQP,iADNQ,jCBPO->ij',p0,p1,v,w,optimize=True)
            if boundaries is None:expected=expected[:,:,0,0]
            for width in (1,2,5):
                actual=contract_meson_matrix_pc(p0,p1,left,right,lw,rw,partitions=boundaries,row_block=width)
                np.testing.assert_allclose(actual,expected,rtol=4e-12,atol=5e-10)
                self.assertFalse(np.shares_memory(actual,left))

    def test_loop_vector_pc_and_complex_coefficients(self):
        p0,_,vertices,_,weights,_=self.fixture(7)
        weighted=np.einsum('ic,cABNO->iABNO',weights,vertices)
        for boundaries in (None,(0,1,3,4,7)):
            expected=np.stack([np.einsum('BAON,iABNO->i',p0,value,optimize=True)
                               for value in projected_vertices(weighted,boundaries)],axis=-1)
            if boundaries is None:expected=expected[:,0]
            actual=contract_meson_loop_pc(p0,vertices,weights,partitions=boundaries)
            np.testing.assert_allclose(actual,expected,rtol=4e-12,atol=5e-11)

    def test_shared_products_not_repeated_per_operator_pair(self):
        from unittest.mock import patch
        import lqcdweave.contraction.execution.matrix_kernels as kernels
        values=self.fixture(3)
        with patch.object(kernels,'_binary',wraps=kernels._binary) as calls:
            contract_meson_matrix_pc(*values,partitions=(0,1,3),row_block=1)
        product = parse_axis_pattern('DCQP,cCBPO->cDBQO').normalized()
        dot = parse_axis_pattern('iBDOQ,jDBQO->ij').normalized()
        product_calls = [call for call in calls.call_args_list if call.args[0] == product]
        dot_calls = [call for call in calls.call_args_list if call.args[0] == dot]
        self.assertEqual(len(product_calls),8)
        self.assertEqual(len(dot_calls),32)
        # The shared product pattern serves both sides: four right products
        # keyed by the two right component blocks, and four left products
        # keyed by the three left component blocks.
        self.assertEqual(sum(call.args[2].shape[0] == 2 for call in product_calls),4)
        self.assertEqual(sum(call.args[2].shape[0] == 3 for call in product_calls),4)

    def test_invalid_shapes_and_partitions(self):
        values=self.fixture(3)
        for kwargs in (dict(row_block=0),dict(partitions=(0,1,4)),dict(partitions=(0,2,1,3))):
            with self.assertRaises(ValueError):contract_meson_matrix_pc(*values,**kwargs)
        with self.assertRaises(ValueError):
            contract_meson_matrix_pc(*values[:4],np.ones((2,4)),values[-1])

    def test_zero_component_axis_and_inputs_not_modified(self):
        values=self.fixture(3)
        snapshots=tuple(value.copy() for value in values)
        contract_meson_matrix_pc(*values,partitions=(0,1,3))
        for actual,expected in zip(values,snapshots):
            np.testing.assert_array_equal(actual,expected)
        p0,p1,left,right,lw,rw=values
        result=contract_meson_matrix_pc(p0,p1,left[:0],right,lw[:,:0],rw,partitions=(0,1,3))
        self.assertEqual(result.shape,(2,3,3,3))
        np.testing.assert_array_equal(result,0)


if __name__=='__main__':unittest.main()
