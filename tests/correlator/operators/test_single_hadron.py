"""Module tests split from test_single_hadron.py; shared numerical fixtures stay unchanged."""

from test_single_hadron import (
    ConservedVectorCurrent,
    DerivativeName,
    GammaName,
    Insertion,
    InsertionGaugeLink,
    InsertionRowMom,
    LocalAxialCurrent,
    LocalVectorCurrent,
    Operator,
    OperatorDisplacement,
    OperatorPath,
    ProjectionName,
    _SingleHadronFixture,
    gamma,
    np,
)


class SingleHadronTests(_SingleHadronFixture):
    def test_frontend_gamma_sign_and_multi_gamma_distance(self):
        entry = InsertionRowMom([5,[[1,0]],13,[[2,0]]],0)
        op = Operator("cyclic",[entry],[1j])
        self.assertEqual([t.coefficient for t in op.terms],[-1j,-2j])
        disp = OperatorDisplacement("d",[entry],[1],[2])
        self.assertEqual([t.component.key for t in disp.terms],[2,2])
        # Explicit bitmask terms do not receive the cyclic naming conversion.
        path = OperatorPath("mask",[(5,(),0,1j)])
        self.assertEqual(path.terms[0].coefficient,1j)

    def test_common_insertion_coupling(self):
        momenta = {10:"0 0 0",20:"1 0 0"}
        insertion = Insertion(GammaName.RHO,DerivativeName.NABLA,ProjectionName.A1,momenta)
        op = Operator("dot",[insertion[0](1,0,0)],[1])
        self.assertEqual([(t.gamma,t.component.key,t.momentum) for t in op.terms],[(1,1,1),(2,2,1),(4,3,1)])
        supplied = {"T1u-":[[[[1,1]],[[1,2]],[[1,3]]]]}
        gauge = InsertionGaugeLink(GammaName.RHO,"T1u-",0,ProjectionName.A1,momenta,supplied)
        self.assertEqual(gauge.rows,insertion.rows)

    def test_gamma_clifford(self):
        for i,j in np.ndindex(4,4):
            a,b = gamma(1<<i),gamma(1<<j)
            np.testing.assert_array_equal(a@b+b@a,2*np.eye(4) if i==j else np.zeros((4,4)))

    def test_current_recipes_spin_factors_and_temporal_offsets(self):
        for d in range(4):
            local = LocalVectorCurrent().operator(d)
            axial = LocalAxialCurrent().operator(d)
            np.testing.assert_array_equal(gamma(local.terms[0].gamma),gamma(1<<d))
            np.testing.assert_array_equal(axial.terms[0].coefficient*gamma(axial.terms[0].gamma),gamma(1<<d)@gamma(15))
            current = ConservedVectorCurrent(wilson_r=.7).operator(d)
            fwd,bwd = current.terms[:2],current.terms[2:]
            np.testing.assert_allclose(sum(t.coefficient*gamma(t.gamma) for t in fwd),-.5*(.7*np.eye(4)-gamma(1<<d)))
            np.testing.assert_allclose(sum(t.coefficient*gamma(t.gamma) for t in bwd),.5*(.7*np.eye(4)+gamma(1<<d)))
            self.assertEqual(fwd[0].component.time_offsets,(0,1 if d==3 else 0))
            self.assertEqual(bwd[0].anchor_time_offset,1 if d==3 else 0)
