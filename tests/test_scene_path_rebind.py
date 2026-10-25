import unittest
from dataclasses import replace
import numpy as np
from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, SourceSpec, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import ContractionState, rebind_scene_path
from lqcdweave.contraction.execution.api import replay
from test_scene_basis_batch import _bridge_problem


def renamed(problem, names):
    return replace(problem,specs={names[k]:v for k,v in problem.specs.items()},
        sources={names[k]:replace(v,region=names[k]) for k,v in problem.sources.items()},
        terms=tuple(replace(t,inputs=tuple(names[k] for k in t.inputs)) for t in problem.terms),
        sparse_bridges=tuple(replace(b,left=names[b.left],right=names[b.right],record=names[b.record])
            for b in problem.sparse_bridges),
        baryon_joins=tuple(replace(j,spin=names[j.spin],elemental=names[j.elemental]) for j in problem.baryon_joins))


class RebindTests(unittest.TestCase):
    def test_baryon_join_preserves_kernel_order_and_values(self):
        specs={k:TensorSpec((2,2),'float64') for k in 'pqr'}
        specs.update({k:TensorSpec((2,2,2),'float64') for k in 'se'})
        join=BaryonJoinSpec(0,'sink','s','e','ijk','abc')
        problem=ContractionProblem(specs,(ContractionTerm(('p','q','r'),'ia,jb,kc->'),),
            {k:SourceSpec(k,int(np.prod(v.shape))*8) for k,v in specs.items()},baryon_joins=(join,))
        profile=CostProfile();old=ContractionState.initial(problem)
        for key in problem.sources: old=old.load(key,profile,1<<20)
        old=old.contract_join(join,profile,1<<20)
        if not old.done: old=old.contract(0,(0,),profile,1<<20)
        names={k:'new:'+k for k in specs}
        new=rebind_scene_path(old,renamed(problem,names),names,profile=profile,memory_limit=1<<20)
        self.assertEqual([a.kernel_path for a in old.actions],[a.kernel_path for a in new.actions])
        rng=np.random.default_rng(127)
        arrays={names[k]:rng.normal(size=v.shape) for k,v in specs.items()}
        expected=np.einsum('ijk,abc,ia,jb,kc->',*(arrays[names[k]] for k in 'sepqr'))
        np.testing.assert_allclose(replay(new,lambda k,s:arrays[k])[0],expected)

    def test_shared_results_renamed_and_recosted(self):
        problem=ContractionProblem({k:TensorSpec((2,2),'float64') for k in 'abc'},
            tuple(ContractionTerm(pair,'ij,jk->ik') for pair in [('a','b'),('a','b'),('a','c')]),
            {k:SourceSpec(k,32) for k in 'abc'})
        old=ContractionState.initial(problem)
        for k in 'abc': old=old.load(k,CostProfile(),100000)
        old=old.contract(0,(0,1),CostProfile(),100000).contract(2,(0,1),CostProfile(),100000).release('a')
        names={'a':'z','b':'y','c':'x'}
        new=rebind_scene_path(old,renamed(problem,names),names,
            profile=CostProfile(),memory_limit=100000)
        self.assertEqual(new.results[0],new.results[1])
        self.assertAlmostEqual(new.seconds,old.seconds)
        arrays={'z':np.arange(4.).reshape(2,2),'y':np.eye(2),'x':np.full((2,2),2.)}
        values=replay(new,lambda k,s:arrays[k])
        for result,expected in zip(values,(arrays['z']@arrays['y'],arrays['z']@arrays['y'],arrays['z']@arrays['x'])):
            np.testing.assert_array_equal(result,expected)
        with self.assertRaisesRegex(ValueError,'bijection'):
            rebind_scene_path(old,renamed(problem,names),{'a':'z'},profile=CostProfile(),memory_limit=100000)
        changed=replace(renamed(problem,names),terms=(ContractionTerm(('z','y'),'ij,jk->ki'),)*3)
        with self.assertRaisesRegex(ValueError,'algebra'):
            rebind_scene_path(old,changed,names,profile=CostProfile(),memory_limit=100000)

    def test_sparse_bounds_recompute_cost_and_memory(self):
        problem=_bridge_problem(2,3);profile=CostProfile()
        old=ContractionState.initial(problem)
        for k in problem.sources: old=old.load(k,profile,1<<30)
        old=old.contract(0,(0,1),profile,1<<30)
        names={k:'new:'+k for k in problem.sources}
        new_problem=renamed(problem,names)
        new_problem=replace(new_problem,sparse_bridges=(replace(new_problem.sparse_bridges[0],nnz_upper=3000),))
        new=rebind_scene_path(old,new_problem,names,profile=profile,memory_limit=1<<30)
        self.assertGreater(new.seconds,old.seconds)
        self.assertGreater(new.peak_bytes,old.peak_bytes)
        with self.assertRaises(MemoryError):
            rebind_scene_path(old,new_problem,names,profile=profile,memory_limit=old.peak_bytes)


if __name__=='__main__': unittest.main()
