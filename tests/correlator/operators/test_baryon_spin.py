"""Module tests split from test_baryon_vertex.py; shared numerical fixtures stay unchanged."""

from test_baryon_vertex import (
    Insertion,
    gamma,
    _BaryonFixture,
    np,
)


class BaryonTests(_BaryonFixture):
    def test_h_middle_rows_and_parity(self):
        for parity in ("g", "u"):
            ins = Insertion.local_spin("H", parity, {0: (0, 0, 0)})
            self.assertEqual([len(row[0].spin.entries) for row in ins.rows], [2, 6, 6, 2])
            # Total spin lowering connects the complete four-row multiplet.
            lowering = np.zeros((4, 4)); lowering[1, 0] = lowering[3, 2] = 1
            for i, weight in enumerate((np.sqrt(3), 2, np.sqrt(3))):
                s = ins.rows[i][0].spin.dense()
                actual = (np.einsum("ia,abc->ibc", lowering, s)
                          + np.einsum("jb,abc->ajc", lowering, s)
                          + np.einsum("kc,abc->abk", lowering, s))
                np.testing.assert_allclose(actual, weight*ins.rows[i+1][0].spin.dense())
            for row in ins.rows:
                s = row[0].spin.dense()
                p = gamma(8)
                np.testing.assert_allclose(np.einsum("ia,jb,kc,abc->ijk", p,p,p,s), (1 if parity == "g" else -1)*s)

    def test_basis_change_preserves_multilinear_form(self):
        spin = Insertion.local_spin("G1", "g", {0: (0,0,0)}).rows[0][0].spin
        eye = np.eye(2); u = np.block([[eye,eye],[eye,-eye]])/np.sqrt(2)
        q = self.rng.normal(size=(3,4)) + 1j*self.rng.normal(size=(3,4))
        converted = spin.to_basis("dirac_pauli")
        expected = np.einsum("abc,a,b,c", spin.dense(), *q)
        actual = np.einsum("abc,a,b,c", converted.dense(), u@q[0],u@q[1],u@q[2])
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(converted.to_basis("degrand_rossi").dense(), spin.dense(), atol=1e-15)
        np.testing.assert_allclose(spin.source().to_basis("dirac_pauli").dense(), converted.source().dense(), atol=1e-15)

    def test_dr_parity_rows_match_declared_coefficients_and_spin(self):
        # One-based coordinates from the user-specified operator definition.
        definitions = {
            "G1": (2, (((1,2,1,1), (2,1,1,-1), (3,4,3,1), (4,3,3,-1)),
                       ((1,2,2,1), (2,1,2,-1), (3,4,4,1), (4,3,4,-1)))),
            "H": (None, (((1,1,1,1), (3,3,3,1)),
                         ((1,1,2,1), (1,2,1,1), (2,1,1,1), (3,3,4,1), (3,4,3,1), (4,3,3,1)),
                         ((1,2,2,1), (2,1,2,1), (2,2,1,1), (3,4,4,1), (4,3,4,1), (4,4,3,1)),
                         ((2,2,2,1), (4,4,4,1)))),
        }
        # Three spin-1/2 generators on the two DR blocks; S^2 identifies
        # the actual multiplet rather than relying only on the irrep name.
        pauli = (np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.diag([1,-1]))
        generators = [np.kron(np.eye(2), p)/2 for p in pauli]
        def total(matrix, tensor):
            return (np.einsum('ia,abc->ibc', matrix, tensor)
                    + np.einsum('jb,abc->ajc', matrix, tensor)
                    + np.einsum('kc,abc->abk', matrix, tensor))
        for irrep, (normalization, rows) in definitions.items():
            for parity, eta in (("g", 1), ("u", -1)):
                insertion = Insertion.local_spin(irrep, parity, {0:(0,0,0)})
                for index, (row, terms) in enumerate(zip(insertion.rows, rows)):
                    with self.subTest(irrep=irrep, parity=parity, row=index):
                        spin = row[0].spin
                        self.assertEqual(spin.basis, 'degrand_rossi')
                        expected = np.zeros((4,4,4), complex)
                        norm = normalization or np.sqrt(len(terms))
                        for a,b,c,value in terms:
                            expected[a-1,b-1,c-1] = value*(eta if a>2 else 1)/norm
                        np.testing.assert_array_equal(spin.dense(), expected)
                        self.assertAlmostEqual(float(np.vdot(expected, expected).real), 1)
                        np.testing.assert_array_equal(spin.source().dense(), eta*expected)
                        j = .5 if irrep == 'G1' else 1.5
                        squared = sum(total(s, total(s, expected)) for s in generators)
                        np.testing.assert_allclose(squared, j*(j+1)*expected, atol=1e-15)
                        np.testing.assert_allclose(total(generators[2], expected),
                                                   (j-index)*expected, atol=1e-15)
