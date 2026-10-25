"""Module tests split from test_baryon_vertex.py; shared numerical fixtures stay unchanged."""

from test_baryon_vertex import (
    Baryon,
    Operator,
    OperatorTerm,
    SparseSpinTensor,
    _BaryonFixture,
    gamma,
    np,
    patch,
)


class BaryonTests(_BaryonFixture):
    def test_core_grouping_is_owned_header_only_and_keeps_bindings(self):
        spin = SparseSpinTensor(((0, 1, 2, 1j),))
        terms = tuple(OperatorTerm.from_baryon(self.component, spin, coefficient=c,
                     anchor_time_offset=offset) for c, offset in ((2, 0), (-1j, 0), (3, 1)))
        vertex = Baryon(self.reader, Operator.from_terms("sum", terms))
        with patch.object(self.reader, "block", side_effect=AssertionError("core read data")):
            groups = vertex.describe_core_factors(0)
            self.assertEqual(tuple(len(g.factors) for g in groups), (2, 1))
            core = vertex.register_core(groups[0].factors, ("S1", "S2"), ("E1", "E2"))
        self.assertEqual(tuple(p.coefficient for p in core.components), (2, -1j))
        self.assertIs(vertex.cores[core.identity], core)
        with self.assertRaisesRegex(ValueError, "identical ordered"):
            vertex.register_core((groups[0].factors[0], groups[1].factors[0]), ("S1", "S3"), ("E1", "E3"))

    def test_split_spin_and_elemental_keep_basis_adjoint_and_coefficient_owner(self):
        from lqcdweave.operators.single_hadron.baryon import spin_basis_matrix

        spin = SparseSpinTensor(((0, 1, 2, 0.7 - 0.3j), (3, 2, 1, 1e-200j)), "dirac_pauli")
        operator = self.operator(spin)
        rotation = spin_basis_matrix("dirac_pauli")
        expected_spin = np.einsum("abc,ai,bj,ck->ijk", spin.dense(), rotation, rotation, rotation)
        for source in (False, True):
            vertex = Baryon(self.reader, operator, source, spin_basis="degrand_rossi")
            description, = vertex.describe_factors(0)
            expected = expected_spin
            if source:
                expected = np.einsum("abc,ia,jb,kc->ijk", expected.conj(), gamma(8), gamma(8), gamma(8))
            with patch.object(self.reader, "block", side_effect=AssertionError("spin description read spatial data")):
                entries = vertex.describe_spin_entries(description)
                actual_spin = vertex.materialize_spin_factor(description)
            self.assertIsInstance(entries, tuple)
            np.testing.assert_allclose(actual_spin, expected, rtol=2e-15, atol=1e-15)
            raw = self.reader.block(self.component, 0, 0, "vvv")
            owned = vertex.materialize_unweighted_elemental(description, raw)
            expected_spatial = raw.conj() if source else raw
            np.testing.assert_array_equal(owned, expected_spatial)
            self.assertFalse(np.shares_memory(owned, raw))
            # Neither half contains the outer Operator coefficient.
            np.testing.assert_allclose(
                vertex.materialize_unweighted_factor(description, raw),
                np.multiply.outer(expected, expected_spatial), rtol=2e-15, atol=1e-15,
            )

    def test_arbitrary_full_spin_and_source(self):
        dense = self.rng.normal(size=(4, 4, 4)) + 1j*self.rng.normal(size=(4, 4, 4))
        op = self.operator(SparseSpinTensor.from_dense(dense))
        for source in (False, True):
            obj = Baryon(self.reader, op, source).load("c", usedNe=(2, 2, 3))
            spin, spatial, c = dense, self.data[0, 0, 1, :, :2, :3], 1+2j
            if source:
                spin = np.einsum("abc,ia,jb,kc->ijk", dense.conj(), gamma(8), gamma(8), gamma(8))
                spatial, c = spatial.conj(), c.conjugate()
            np.testing.assert_allclose(obj.get(1), c*np.multiply.outer(spin, spatial))
            factor = next(obj.factors(1))
            self.assertEqual(factor.field_order, (2, 1, 0) if source else (0, 1, 2))

    def test_unified_operator_metadata_is_header_only_and_keeps_endpoint_order(self):
        spin = SparseSpinTensor([(0, 1, 2, 1.0 + 0.5j)])
        term = OperatorTerm.from_baryon(
            self.component, spin, momentum=0, coefficient=0.25 - 0.75j,
            anchor_time_offset=1, endpoint_time_offsets=(0, 0, 0),
        )
        operator = Operator.from_terms("unified-baryon", (term,))
        vertex = Baryon(self.reader, operator)

        with patch.object(self.reader, "block", side_effect=AssertionError("metadata description read data")):
            description, = vertex.describe_factors(0)

        self.assertIs(description.operator_term, term)
        self.assertEqual(description.endpoint_time_offsets, (0, 0, 0))
        self.assertEqual(description.endpoint_times, (1, 1, 1))
        self.assertEqual(description.field_order, (0, 1, 2))
        self.assertEqual(description.coefficient, term.coefficient)

        asynchronous = Operator.from_terms(
            "unsupported-asynchronous-baryon",
            (OperatorTerm.from_baryon(
                self.component, spin, endpoint_time_offsets=(0, 1, 0),
            ),),
        )
        with self.assertRaises(NotImplementedError):
            Baryon(self.reader, asynchronous).describe_factors(0)
