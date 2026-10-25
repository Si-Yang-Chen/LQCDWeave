import unittest
import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, SparseBridgeSpec, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import ContractionState
from lqcdweave.contraction.execution.scene_runtime import _model_features
from lqcdweave.contraction.execution.operations.sparse_bridge_kernel import contract_p2p_cycle


class SparseCycleDiagonalTests(unittest.TestCase):
    def test_profile_predictors_match_planned_diagonal_cycle_cost(self):
        spec = TensorSpec((4,4,3,3,3,3), 'complex128')
        first = SparseBridgeSpec('same','same','r1',3,5,left_axes=(1,4,5),right_axes=(0,2,3),same_point=True)
        second = SparseBridgeSpec('same','same','r2',3,5,left_axes=(0,2,3),right_axes=(1,4,5),same_point=True)
        problem = ContractionProblem({'same':spec},
            (ContractionTerm(('same','same'),'abedef,ghejel->'),),
            {'same':SourceSpec('same',int(np.prod(spec.shape))*16),
             'r1':SourceSpec('r1',800), 'r2':SourceSpec('r2',800)},
            sparse_bridges=(first,second))
        profile = CostProfile()
        state = ContractionState.initial(problem)
        for key in problem.sources:
            state = state.load(key,profile,1<<30)
        result = state.contract(0,(0,1),profile,1<<30)
        action = result.actions[-1]
        features = _model_features(result,action)
        predicted = profile.contract_seconds(features['flops'],features['traffic_bytes'],
            kind='sparse_bridge',dtype=spec.dtype,kernel_calls=features['kernel_calls'],sparse_records=2)
        self.assertAlmostEqual(predicted,action.seconds,places=14)

    def test_repeated_point_axes_keep_one_point_across_both_currents(self):
        rng=np.random.default_rng(928)
        def random(shape): return rng.normal(size=shape)+1j*rng.normal(size=shape)
        left,right=random((4,4,3,3,3,3)),random((4,4,3,3,3,3))
        original_left,original_right=left.copy(),right.copy()
        spin1,spin2=random((4,4)),random((4,4))
        indices=np.array([[0,0],[1,1],[1,1],[2,2],[0,2]])
        values1,values2=random((5,3,3)),random((5,3,3))
        diagonal1,diagonal2=np.zeros((3,3,3),complex),np.zeros((3,3,3),complex)
        for (a,b),v1,v2 in zip(indices,values1,values2):
            if a==b:
                diagonal1[a]+=v1;diagonal2[a]+=v2
        expected=np.einsum('abpdpf,ghpjpl,bg,pfj,ah,pdl->',left,right,spin1,diagonal1,spin2,diagonal2)
        first=SparseBridgeSpec('same','same','r1',3,5,left_axes=(1,4,5),right_axes=(0,2,3),same_point=True)
        second=SparseBridgeSpec('same','same','r2',3,5,left_axes=(0,2,3),right_axes=(1,4,5),same_point=True)
        actual=contract_p2p_cycle(left,right,((spin1,1,{'type':'sparse','indices':indices,'values':values1}),),
            ((spin2,1,{'type':'sparse','indices':indices,'values':values2}),),
            first_bridge=first,second_bridge=second,equation='abedef,ghejel->')
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
        np.testing.assert_array_equal(left,original_left)
        np.testing.assert_array_equal(right,original_right)


if __name__=='__main__': unittest.main()
