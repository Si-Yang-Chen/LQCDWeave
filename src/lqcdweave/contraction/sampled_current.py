"""Connected one-current trace with two finite, already produced legs.

Unlike external_current, neither leg is completed by O acting on a full field.
This kernel retains both current component indices for production subset scans.
"""
import numpy as np
from opt_einsum import contract
from lqcdweave.operators.single_hadron.gamma import gamma


def current_pair_contributions(tau, sink_vertex, source_gamma, source_spatial,
                               coefficient=1, *, array_module=np, backend='numpy'):
    """Unweighted connected contributions (i,j), from finite V† S xi only.

    tau: (4,4,Nmeson,Ncurrent); source_spatial: xi† O xi before dagger;
    sink_vertex: (4,4,Nmeson,Nmeson). No sampling, volume or flavor weights.
    Both spatial current indices remain restricted to the supplied tau axis.
    """
    xp = array_module
    if tau.ndim != 4 or tau.shape[:2] != (4,4):
        raise ValueError('finite propagators require (4,4,meson,current) axes')
    n, k = tau.shape[-2:]
    if sink_vertex.shape != (4,4,n,n) or source_spatial.shape != (k,k):
        raise ValueError('vertex dimensions differ from actual finite propagator axes')
    g4, g5 = gamma(8,array_module=xp), gamma(15,array_module=xp)
    left = xp.einsum('ru,kuvn,vs->krsn',g5,tau.transpose(3,1,0,2).conj(),g5,optimize=True)
    spin = complex(coefficient).conjugate()*g5@g4@gamma(source_gamma,array_module=xp).conj().T@g4
    middle = contract('swmn,wt->stmn',sink_vertex,g5,optimize='greedy',backend=backend)
    # Reuse a transformed left leg; no per-(i,j) full spin/mode temporary.
    a = contract('stmn,irtn->irsm',middle,left.conj(),optimize='greedy',backend=backend)
    b = contract('rv,jvsm->jrsm',spin,left,optimize='greedy',backend=backend)
    result = -(a.reshape(k,-1) @ b.reshape(k,-1).T)
    return result*source_spatial.conj().T


class PairSamplingPlan:
    """Exact component-inclusion weights for a fixed finite-subset catalog.

    Groups have disjoint axis indices, a parent population, and equality labels.
    Three colors at one point share one label. Different groups must be
    independently sampled; cases never combine alternative realizations.
    """
    def __init__(self, selections, groups):
        from lqcdweave.contracts.sampling_weights import expected_weight
        selected=np.asarray(selections,dtype=bool)
        if selected.ndim != 2:
            raise ValueError('selections require (case,current_component) axes')
        linear=selected.astype(float); occupied=set(); corrections=[]
        for group in groups:
            indices=np.asarray(group['indices'],dtype=int)
            labels=np.asarray(group['labels'])
            if len(indices)!=len(labels) or len(set(indices)&occupied):
                raise ValueError('sampling groups require disjoint axes and equality labels')
            occupied.update(indices)
            patterns,inverse=np.unique(selected[:,indices],axis=0,return_inverse=True)
            for pi,pattern in enumerate(patterns):
                positions=np.flatnonzero(pattern)
                if not len(positions):continue
                cases=np.flatnonzero(inverse==pi); same=labels[positions,None]==labels[None,positions]
                count=len(np.unique(labels[positions])); population=group['population']
                one=float(expected_weight(population,count,1))
                # No distinct-label term exists when the whole population is 1.
                two=one if population==1 else float(expected_weight(population,count,2))
                chosen=indices[positions]
                linear[np.ix_(cases,chosen)]=one
                corrections.append((cases,chosen,same,two-one*one,one-two))
        self.linear=linear
        self.corrections=tuple(corrections)
        self.components=selected.shape[1]

    def sum(self,pairs,*,array_module=np):
        xp=array_module
        if pairs.shape!=(self.components,self.components):
            raise ValueError('pair contribution axes differ from selection plan')
        linear=xp.asarray(self.linear)
        result=xp.sum((linear@pairs)*linear,axis=1)
        for cases,chosen,same,free,equal in self.corrections:
            indices=xp.asarray(chosen); block=pairs[xp.ix_(indices,indices)]
            correction=free*block.sum()+equal*xp.sum(block*xp.asarray(same))
            result[xp.asarray(cases)]+=correction
        return result
