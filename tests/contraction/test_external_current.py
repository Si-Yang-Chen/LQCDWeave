import itertools
import unittest
import numpy as np

from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.contraction.external_current import (
    project_fields, sample_fields, apply_source_current, leg_contributions,
)


class ExternalCurrentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(311)
        self.rng = rng
        self.lx, self.n = 4, 2
        self.x = rng.normal(size=(4,4,1,1,4,3,2)) + 1j*rng.normal(size=(4,4,1,1,4,3,2))
        self.u = np.linalg.qr(rng.normal(size=(12,12))+1j*rng.normal(size=(12,12)))[0].T.reshape(12,1,1,4,3)
        self.links = np.broadcast_to(np.eye(3,dtype=complex),(4,1,1,1,4,3,3)).copy()
        for p in range(4):
            self.links[0,0,0,0,p] = np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
        self.component = ElementalComponent(0)
        self.op = Operator.from_terms('test-current', [OperatorTerm(1,self.component,0,coefficient=.7+.2j)])
        self.rows = {self.component: ((GaugePath((0,)),.3-.4j),(GaugePath(()),.2+.5j))}
        self.points = np.array([[i,0,0] for i in range(4)])
        self.sink = rng.normal(size=(4,4,2,2))+1j*rng.normal(size=(4,4,2,2))

    def explicit_source(self):
        spatial = np.zeros((12,12),complex)
        for x in range(4):
            phase = np.exp(2j*np.pi*x/4)
            spatial[x*3:x*3+3,((x+1)%4)*3:((x+1)%4)*3+3] += (.3-.4j)*phase*self.links[0,0,0,0,x]
            spatial[x*3:x*3+3,x*3:x*3+3] += (.2+.5j)*phase*np.eye(3)
        spin = gamma(15)@gamma(8)@gamma(1).conj().T@gamma(8)
        return (.7+.2j).conjugate()*np.kron(spin,spatial.conj().T)

    def test_complex_dagger_momentum_and_link_match_independent_full_matrix(self):
        y = apply_source_current(self.x,self.op,self.rows,self.links,((1,0,0),))
        x = self.x.transpose(0,2,3,4,5,1,6).reshape(48,8)
        expected = (self.explicit_source()@x).reshape(4,1,1,4,3,4,2).transpose(0,5,1,2,3,4,6)
        np.testing.assert_allclose(y,expected,rtol=2e-14,atol=2e-14)
        left = sample_fields(self.x,self.points).reshape(12,4,4,2)
        right = sample_fields(y,self.points).reshape(12,4,4,2)
        density = leg_contributions(left,right,self.sink)
        b = np.einsum('swmn,wt->smtn',self.sink,gamma(15)).reshape(8,8)
        reference = -np.trace(b@x.conj().T@self.explicit_source()@x)
        np.testing.assert_allclose(density.sum(),reference,rtol=2e-13,atol=2e-12)

    def test_three_sector_completion_and_two_three_four_column_subset_averages(self):
        y = apply_source_current(self.x,self.op,self.rows,self.links,((1,0,0),))
        v = leg_contributions(project_fields(self.x,self.u[:2]),project_fields(y,self.u[:2]),self.sink)
        d = leg_contributions(project_fields(self.x,self.u[2:10]),project_fields(y,self.u[2:10]),self.sink)
        hp = sample_fields(self.x,self.points,parent_vectors=self.u[:10]).reshape(12,4,4,2)
        p = leg_contributions(hp,sample_fields(y,self.points).reshape(12,4,4,2),self.sink).reshape(4,3).sum(1)
        exact = leg_contributions(sample_fields(self.x,self.points).reshape(12,4,4,2),
                                  sample_fields(y,self.points).reshape(12,4,4,2),self.sink).sum()
        for m in (2,3,4):
            medium = sum(np.mean([4/m*group[list(indices)].sum()
                                 for indices in itertools.combinations(range(4),m)])
                         for group in (d[:4],d[4:]))
            for count in (1,2,3,4):
                high = np.mean([4/count*p[list(indices)].sum()
                                for indices in itertools.combinations(range(4),count)])
                np.testing.assert_allclose(v.sum()+medium+high,exact,rtol=2e-13,atol=2e-12)

    def test_a_single_anchor_includes_its_unsampled_link_neighbor(self):
        y = apply_source_current(self.x,self.op,self.rows,self.links,((1,0,0),))
        one = self.points[:1]
        left = sample_fields(self.x,one).reshape(3,4,4,2)
        right = sample_fields(y,one).reshape(3,4,4,2)
        self.assertGreater(np.linalg.norm(right),0)
        full = leg_contributions(sample_fields(self.x,self.points).reshape(12,4,4,2),
                                 sample_fields(y,self.points).reshape(12,4,4,2),self.sink).reshape(4,3).sum(1)
        np.testing.assert_allclose(leg_contributions(left,right,self.sink).sum(),full[0])


if __name__ == '__main__':
    unittest.main()
