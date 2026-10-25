"""Module tests split from test_single_hadron.py; shared numerical fixtures stay unchanged."""

from test_single_hadron import (
    ArrayElemental,
    ElementalComponent,
    Meson,
    Operator,
    OperatorPath,
    OperatorTerm,
    SparsePointBlock,
    SpectralProfile,
    _SingleHadronFixture,
    gamma,
    np,
)


class SingleHadronTests(_SingleHadronFixture):
    def test_existing_data_complex_coefficients_and_source_dagger(self):
        op = Operator.from_terms("meson", [OperatorTerm(5, self.components[0], 1, .2+.7j),
                                           OperatorTerm(15, self.components[1], 0, -.4+.3j)])
        for source in (False,True):
            vertex = Meson(self.data, op, source).load("cfg")
            for sector in self.sectors:
                raw = {"v2p":"p2v", "p2v":"v2p"}.get(sector,sector) if source else sector
                expected = 0
                for term in op.terms:
                    index = self.components.index(term.component)
                    spatial = self.sectors[raw][index,term.momentum,1]
                    spin, coeff = gamma(term.gamma), term.coefficient
                    if source:
                        perm = {"v2v":(1,0), "v2p":(1,2,0), "p2v":(2,0,1), "p2p":(2,3,0,1)}[raw]
                        spatial = spatial.conj().transpose(perm)
                        spin = gamma(8) @ spin.conj().T @ gamma(8)
                        coeff = coeff.conjugate()
                    expected += coeff*np.multiply.outer(spin,spatial)
                actual = vertex.get_terms(sector,1,range(2))
                np.testing.assert_allclose(actual,expected)

    def test_temporal_terms_keep_endpoints_and_anchor(self):
        components = (ElementalComponent.path((6,)), ElementalComponent.path(()))
        data = ArrayElemental({"v2v":self.sectors["v2v"]}, components, phase_convention="midpoint")
        op = OperatorPath("temporal", [(1,(6,),0,1,-1), (2,(),1,1,0)])
        sink = Meson(data,op)
        factors = list(sink.factors(0))
        self.assertEqual(factors[0].endpoint_times,(2,0))
        self.assertEqual(factors[1].endpoint_times,(0,0))
        np.testing.assert_array_equal(factors[0].spatial,self.sectors["v2v"][0,0,2])
        with self.assertRaisesRegex(ValueError,"different endpoint"):
            sink.get(0)
        source = Meson(data,op,True)
        self.assertEqual(list(source.factors(0))[0].endpoint_times,(0,2))
        with self.assertRaisesRegex(ValueError,"open temporal"):
            list(Meson(data,op,boundary="open").factors(0))

    def test_profile_and_reader_scratch_ownership(self):
        weights = np.arange(6).reshape(3,2)/5 + 1j
        profile = SpectralProfile(weights,weights*.7j,cfg="cfg")
        op = Operator.from_terms("profile",[OperatorTerm(0,self.components[0],0,profile=profile)])
        actual = Meson(self.data,op).get(1)
        raw = self.sectors["v2v"][0,0,1]
        expected = weights[1].conj()[:,None]*raw*(weights[1]*.7j)[None,:]
        np.testing.assert_allclose(actual,np.multiply.outer(np.eye(4),expected))
        class ScratchReader:
            Lt,phase_convention = 3,"midpoint"
            scratch = np.zeros((2,2),complex)
            def block(self,component,momentum,t,sector):
                self.scratch.fill(momentum+1j)
                return self.scratch
        op = Operator.from_terms("scratch",[OperatorTerm(0,self.components[0],0),OperatorTerm(0,self.components[0],1)])
        factors = list(Meson(ScratchReader(),op).factors(0))
        np.testing.assert_array_equal(factors[0].spatial,1j)
        np.testing.assert_array_equal(factors[1].spatial,1+1j)

    def test_sparse_point_source_dagger_without_dense_storage(self):
        raw = SparsePointBlock(np.array([0,2]),np.array([1,0]),
                               np.arange(18).reshape(2,3,3)*(1+.7j),(3,3))
        class Reader:
            Lt,phase_convention = 3,"none"
            def block(self,*args):
                return raw
        op = Operator.from_terms("p",[OperatorTerm(2,self.components[0],0,.3+.1j)])
        vertex = Meson(Reader(),op,True,phase_convention="none")
        factor = list(vertex.factors(1,"p2p"))[0]
        self.assertIsInstance(factor.spatial,SparsePointBlock)
        expected = raw.to_dense().conj().transpose(2,3,0,1)
        np.testing.assert_array_equal(factor.spatial.to_dense(),expected)
