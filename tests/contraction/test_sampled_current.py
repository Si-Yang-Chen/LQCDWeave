"""Finite production legs: literal trace and exhaustive joint subset averages."""
from itertools import combinations, product
import unittest
import numpy as np
from lqcdweave.contraction.sampled_current import current_pair_contributions, PairSamplingPlan
from lqcdweave.operators.single_hadron.gamma import gamma


def tau_from_left(left):
    k,_,_,n=left.shape; tau=np.empty((4,4,n,k),complex); g5=gamma(15)
    for i,j in product(range(k),range(n)):
        tau[:,:,j,i]=g5@left[i,:,:,j].conj().T@g5
    return tau


def literal(left,sink,source,spin,coefficient=1):
    k,_,_,n=left.shape
    x=left.transpose(1,0,2,3).reshape(4*k,4*n)
    b=np.einsum('swmn,wt->smtn',sink,gamma(15)).reshape(4*n,4*n)
    a=np.kron(complex(coefficient).conjugate()*gamma(15)@gamma(8)@gamma(spin).conj().T@gamma(8),source.conj().T)
    return -np.trace(b@x.conj().T@a@x)


class SampledCurrentTests(unittest.TestCase):
    def test_subset_average_of_four_saved_columns_keeps_large_parent_population(self):
        rng=np.random.default_rng(106);pairs=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
        groups=[dict(indices=list(range(4)),labels=list(range(4)),population=106)]
        baseline=PairSamplingPlan([np.ones(4,bool)],groups).sum(pairs)[0]
        for m in (2,3):
            selections=[]
            for cols in combinations(range(4),m):
                s=np.zeros(4,bool);s[list(cols)]=True;selections.append(s)
            np.testing.assert_allclose(PairSamplingPlan(selections,groups).sum(pairs).mean(),baseline,rtol=3e-14,atol=2e-11)

    def test_every_nearly_complete_D_subset_average(self):
        rng=np.random.default_rng(901)
        for n,m in ((3,2),(4,3),(5,4),(5,2),(5,3)):
            pairs=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
            selected=[]
            for indices in combinations(range(n),m):
                s=np.zeros(n,bool);s[list(indices)]=True;selected.append(s)
            plan=PairSamplingPlan(selected,[dict(indices=list(range(n)),labels=list(range(n)),population=n)])
            np.testing.assert_allclose(plan.sum(pairs).mean(),pairs.sum(),rtol=3e-14,atol=2e-13)

    def test_pair_sum_matches_independent_full_spin_space_trace(self):
        rng=np.random.default_rng(512)
        def z(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
        left=z((7,4,4,2));sink=z((4,4,2,2));source=z((7,7))
        values=current_pair_contributions(tau_from_left(left),sink,3,source,.4-.3j)
        np.testing.assert_allclose(values.sum(),literal(left,sink,source,3,.4-.3j),rtol=2e-14,atol=2e-10)
        # An arbitrary finite restriction cannot read any missing external leg.
        chosen=np.array([0,3,6]);restricted=current_pair_contributions(tau_from_left(left[chosen]),sink,3,source[np.ix_(chosen,chosen)],.4-.3j)
        np.testing.assert_allclose(restricted,values[np.ix_(chosen,chosen)],rtol=2e-14,atol=2e-11)

    def test_joint_D_3_choose_2_and_full_color_points_3_choose_2_average(self):
        rng=np.random.default_rng(213)
        def z(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
        basis=np.linalg.qr(z((9,9)))[0].T
        v,d=basis[:1],basis[1:4];q=np.eye(9)-basis[:4].T@basis[:4].conj()
        raw=np.concatenate((v,d,np.eye(9)))
        physical=np.concatenate((v,d,q.T))
        x=z((9,4,4,2));left=np.einsum('kq,qrsn->krsn',physical.conj(),x)
        source=z((9,9));sink=z((4,4,2,2))
        pairs=current_pair_contributions(tau_from_left(left),sink,2,raw.conj()@source@raw.T)
        selections=[]
        for dc,pc in product(combinations(range(3),2),combinations(range(3),2)):
            s=np.zeros(13,bool);s[0]=True;s[1+np.asarray(dc)]=True
            for p in pc:s[4+3*p:7+3*p]=True
            selections.append(s)
        groups=[dict(indices=[1,2,3],labels=[0,1,2],population=3),
                dict(indices=list(range(4,13)),labels=np.repeat(range(3),3),population=3)]
        estimates=PairSamplingPlan(selections,groups).sum(pairs)
        np.testing.assert_allclose(estimates.mean(),literal(x,sink,source,2),rtol=3e-14,atol=2e-10)
        # Demonstrate that the same-point color off-diagonal shares H_1.
        self.assertGreater(np.linalg.norm(pairs[4:7,4:7]-np.diag(np.diag(pairs[4:7,4:7]))),0)

    def test_isolated_selected_points_do_not_invent_a_link_neighbor(self):
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.production.elemental.api import ElementalGenerator
        rng=np.random.default_rng(7);tau=rng.normal(size=(4,4,2,6))+0j
        gauge=np.broadcast_to(np.eye(3,dtype=complex),(1,1,1,4,4,3,3)).copy()
        table=[dict(component=ElementalComponent(0),path=(0,),coefficient=1)]
        # Feed the actual point coordinates to EL: no full acted field is available.
        points=np.array([[[0,0,0],[2,0,0]]])
        blocks=ElementalGenerator(gauge,None,operator_table=table,points=points,momentum_dict={0:(0,0,0)},
            blocks=('p2p',),phase_convention='anchor').calc_path_blocks(0)
        spatial=blocks['p2p'][0][0].to_dense().reshape(6,6)
        result=current_pair_contributions(tau,np.ones((4,4,2,2)),1,spatial)
        np.testing.assert_array_equal(result,0)
        neighboring=np.array([[[0,0,0],[1,0,0]]])
        linked=ElementalGenerator(gauge,None,operator_table=table,points=neighboring,momentum_dict={0:(0,0,0)},
            blocks=('p2p',),phase_convention='anchor').calc_path_blocks(0)['p2p'][0][0].to_dense()
        self.assertGreater(np.linalg.norm(linked),0)

    def test_missing_second_component_is_not_silently_first_order(self):
        with self.assertRaisesRegex(ValueError,'not estimable'):
            PairSamplingPlan([[True,False]], [dict(indices=[0,1],labels=[0,1],population=3)])


if __name__=='__main__':unittest.main()
