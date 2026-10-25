"""A live Fortran matrix can feed GEMM without a packing snapshot."""
import unittest
from unittest.mock import patch
import numpy as np
from lqcdweave.contracts.correlator.api import Action,TensorSpec
from lqcdweave.contraction.planning.api import CostProfile,describe_dense_batch
from lqcdweave.contraction.execution.operations.dense_batch import bind_batch_layout


class MatrixBatchBorrowingTests(unittest.TestCase):
    def test_fortran_single_side_is_borrowed_and_results_are_independent(self):
        rng=np.random.default_rng(9102)
        arrays={'a0':rng.normal(size=(2,4)),'a1':rng.normal(size=(3,4)),
                'b':np.asfortranarray(rng.normal(size=(4,5)))}
        specs={k:TensorSpec(a.shape,a.dtype) for k,a in arrays.items()}
        actions=[]
        expected={}
        for i in range(2):
            key=f'out{i}'
            expected[key]=np.einsum('ik,kj->ij',arrays[f'a{i}'],arrays['b'])
            specs[key]=TensorSpec(expected[key].shape,expected[key].dtype)
            actions.append(Action('contract',key,(f'a{i}','b'),'ik,kj->ij',seconds=1))
        batch=describe_dense_batch(actions,(0,1),specs,CostProfile())
        layout=bind_batch_layout(batch,specs,np)
        with patch.object(np,'array',side_effect=AssertionError('unnecessary snapshot')):
            borrowed=layout.sides[1].pack(arrays,np,layout.dtype)
        self.assertIs(borrowed,arrays['b'])
        with patch.object(np,'einsum',side_effect=AssertionError('matrix used reference kernel')):
            layout.run(arrays,backend=np)
        for key,value in expected.items():
            np.testing.assert_allclose(arrays[key],value,rtol=2e-12,atol=2e-12)
            self.assertFalse(np.shares_memory(arrays[key],borrowed))
        promoted=layout.sides[1].pack(arrays,np,np.dtype('complex128'))
        self.assertFalse(np.shares_memory(promoted,borrowed))
        self.assertEqual(promoted.dtype,np.complex128)
        grouped=bind_batch_layout(batch,specs,np,allow_fortran=False)
        packed=grouped.sides[1].pack(arrays,np,np.dtype('float64'))
        self.assertTrue(packed.flags.c_contiguous)
        self.assertFalse(np.shares_memory(packed,borrowed))
