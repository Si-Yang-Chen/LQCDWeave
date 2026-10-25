"""Numerical validation for automatic three-quark cubic irrep coupling."""

import unittest

import numpy as np
from lqcdweave.symmetry import irrep_matrices, restrict_irrep

from lqcdweave.contracts.elemental import BaryonComponent
from lqcdweave.operators.single_hadron.api import (
    Insertion,
    InsertionGaugeLink,
    Operator,
    SpinName,
    baryon_local_spin_rows,
    gamma,
)


_DIMS = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}
_PRODUCTS = {
    "G1": {
        "A1": {"G1": 1}, "A2": {"G2": 1}, "E": {"H": 1},
        "T1": {"G1": 1, "H": 1}, "T2": {"G2": 1, "H": 1},
    },
    "H": {
        "A1": {"H": 1}, "A2": {"H": 1},
        "E": {"G1": 1, "G2": 1, "H": 1},
        "T1": {"G1": 1, "G2": 1, "H": 2},
        "T2": {"G1": 1, "G2": 1, "H": 2},
    },
}
_SPINS = (SpinName.G1G, SpinName.G1U, SpinName.HG, SpinName.HU)


def _components(dimension):
    return tuple(
        BaryonComponent.paths(((i,), ((i + 1) % 6,), ((i + 2) % 6,)))
        for i in range(dimension)
    )


def _spatial_rows(dimension):
    return tuple(((1.0, component),) for component in _components(dimension))


def _coefficient_matrix(insertion, spins, components):
    spin_index = {spin.entries: i for i, spin in enumerate(spins)}
    component_index = {component: i for i, component in enumerate(components)}
    result = np.zeros((len(spins) * len(components), len(insertion.rows)), complex)
    for target_row, terms in enumerate(insertion.rows):
        for term in terms:
            product_row = spin_index[term.spin.entries] * len(components) + component_index[term.component]
            result[product_row, target_row] += term.coefficient
    return result


def _gamma4_tensor(tensor):
    g4 = gamma(8)
    return np.einsum("ia,jb,kc,abc->ijk", g4, g4, g4, tensor)


class BaryonIrrepCouplingTests(unittest.TestCase):
    def test_all_products_copies_parities_and_completeness(self):
        for spin_name in _SPINS:
            spin_kind = spin_name[:-1]
            spin_parity = 1 if spin_name.endswith("g") else -1
            spin_rows = baryon_local_spin_rows(spin_kind, spin_name[-1])
            for row in spin_rows:
                np.testing.assert_allclose(_gamma4_tensor(row.dense()),
                                           spin_parity * row.dense(), atol=1e-13, rtol=0)
            for spatial_kind, decomposition in _PRODUCTS[spin_kind].items():
                components = _components(_DIMS[spatial_kind])
                rows = _spatial_rows(_DIMS[spatial_kind])
                for spatial_parity_name in ("g", "u"):
                    spatial_parity = 1 if spatial_parity_name == "g" else -1
                    spatial_name = spatial_kind + spatial_parity_name
                    total_parity = spin_parity * spatial_parity
                    parity_name = "g" if total_parity == 1 else "u"
                    product_dimension = len(spin_rows) * len(components)
                    projectors = []
                    for target, multiplicity in decomposition.items():
                        copies = []
                        for copy_index in range(multiplicity):
                            requested = target if copy_index == 0 else target + parity_name
                            with self.subTest(spin=spin_name, spatial=spatial_name,
                                              target=target, copy=copy_index):
                                insertion = Insertion.from_irreps(
                                    spin_name, rows, spatial_name, requested,
                                    {0: (0, 0, 0)}, copy_index=copy_index,
                                )
                                coefficients = insertion.coupling_coefficients
                                self.assertEqual(insertion.multiplicity, multiplicity)
                                self.assertEqual(insertion.copy_index, copy_index)
                                self.assertEqual(insertion.parity, total_parity)
                                self.assertEqual(len(insertion.rows), {"G1": 2, "G2": 2, "H": 4}[target])
                                np.testing.assert_allclose(
                                    _coefficient_matrix(insertion, spin_rows, components),
                                    coefficients, atol=1e-12, rtol=0,
                                )
                                np.testing.assert_allclose(
                                    coefficients.conj().T @ coefficients,
                                    np.eye(coefficients.shape[1]), atol=1e-12, rtol=0,
                                )
                                copies.append(coefficients)
                                projectors.append(coefficients @ coefficients.conj().T)

                                space_parity = np.asarray(irrep_matrices(
                                    (0, 0, 0), spatial_name)["inviden"], complex)
                                target_parity_matrix = np.asarray(irrep_matrices(
                                    (0, 0, 0), target + parity_name
                                )["inviden"], complex)
                                product_parity = np.kron(
                                    spin_parity * np.eye(len(spin_rows)), space_parity
                                )
                                np.testing.assert_allclose(
                                    product_parity @ coefficients,
                                    coefficients @ target_parity_matrix,
                                    atol=1e-12, rtol=0,
                                )
                        for i, left in enumerate(copies):
                            for right in copies[i + 1:]:
                                np.testing.assert_allclose(left.conj().T @ right, 0,
                                                           atol=1e-12, rtol=0)
                    np.testing.assert_allclose(
                        sum(projectors, np.zeros((product_dimension, product_dimension), complex)),
                        np.eye(product_dimension), atol=1e-11, rtol=0,
                    )

        # A1 is the identity channel, so this also fixes the canonical CG phase.
        fixed = Insertion.from_irreps(
            SpinName.G1G, _spatial_rows(1), "A1g", "G1", {0: (0, 0, 0)}
        )
        np.testing.assert_allclose(fixed.coupling_coefficients, np.eye(2), atol=1e-13, rtol=0)
        repeated = Insertion.from_irreps(
            SpinName.G1G, _spatial_rows(1), "A1g", "G1", {0: (0, 0, 0)}
        )
        np.testing.assert_array_equal(fixed.coupling_coefficients, repeated.coupling_coefficients)

    def test_group_intertwining_from_dr_gamma_generators(self):
        # Build the physical DR field rotations from exp(-theta gamma_j gamma_k/2).
        generators = {
            "c4x": gamma(2) @ gamma(4),
            "c4y": gamma(4) @ gamma(1),
            "c4z": gamma(1) @ gamma(2),
        }
        for spin_name, targets in ((SpinName.G1G, ("G1", "H")),
                                   (SpinName.HG, ("G1", "G2", "H"))):
            spins = baryon_local_spin_rows(spin_name[:-1], spin_name[-1])
            spin_columns = np.column_stack([row.dense().reshape(-1) for row in spins])
            rows = _spatial_rows(3)
            components = _components(3)
            for target in targets:
                copies = _PRODUCTS[spin_name[:-1]]["T1"][target]
                for copy_index in range(copies):
                    insertion = Insertion.from_irreps(
                        spin_name, rows, "T1u", target, {0: (0, 0, 0)},
                        copy_index=copy_index,
                    )
                    coefficients = insertion.coupling_coefficients
                    for element, generator in generators.items():
                        field_rotation = (np.cos(np.pi / 4) * np.eye(4)
                                          - np.sin(np.pi / 4) * generator)
                        inverse_field_action = np.kron(
                            np.kron(field_rotation.conj(), field_rotation.conj()),
                            field_rotation.conj(),
                        )
                        spin_action = spin_columns.conj().T @ inverse_field_action @ spin_columns
                        np.testing.assert_allclose(
                            inverse_field_action @ spin_columns,
                            spin_columns @ spin_action, atol=1e-12, rtol=0,
                        )
                        spatial_action = np.asarray(
                            irrep_matrices((0, 0, 0), "T1")[element], complex
                        )
                        target_action = np.asarray(
                            irrep_matrices((0, 0, 0), target)[element], complex
                        )
                        product_action = np.kron(spin_action, spatial_action)
                        np.testing.assert_allclose(
                            product_action @ coefficients,
                            coefficients @ target_action,
                            atol=2e-12, rtol=0,
                        )

    def test_custom_spatial_unitary_basis_covariance(self):
        unitary = np.array([[1, 1j], [1j, 1]], complex) / np.sqrt(2)
        canonical_components = _components(2)
        canonical_rows = _spatial_rows(2)
        canonical_matrices = irrep_matrices((0, 0, 0), "E")
        spatial_matrices = {
            element: unitary.conj().T @ np.asarray(matrix, complex) @ unitary
            for element, matrix in canonical_matrices.items()
        }
        custom_rows = tuple(
            tuple((unitary[i, j], canonical_components[i])
                  for i in range(2) if abs(unitary[i, j]) > 0)
            for j in range(2)
        )
        canonical = Insertion.from_irreps(
            SpinName.G1G, canonical_rows, "Eg", "H", {0: (0, 0, 0)}
        ).coupling_coefficients
        custom = Insertion.from_irreps(
            SpinName.G1G, custom_rows, "Eg", "H", {0: (0, 0, 0)},
            spatial_matrices=spatial_matrices,
        ).coupling_coefficients
        basis_change = np.kron(np.eye(2), unitary.conj().T)
        expected = basis_change @ canonical
        phase = np.vdot(expected, custom)
        phase /= abs(phase)
        np.testing.assert_allclose(custom, phase * expected, atol=2e-12, rtol=0)
        actual_physical = np.einsum("sjt,ij->sit", custom.reshape(2, 2, 4), unitary)
        expected_physical = np.einsum("sit->sit", canonical.reshape(2, 2, 4))
        np.testing.assert_allclose(actual_physical, phase * expected_physical,
                                   atol=2e-12, rtol=0)

    def test_moving_projection_matches_bundled_projectors(self):
        spins = baryon_local_spin_rows("G1", "g")
        components = _components(3)
        rest = Insertion.from_irreps(
            SpinName.G1G, _spatial_rows(3), "T1u", "H", {0: (0, 0, 0)},
            copy_index=0,
        )
        source = rest.coupling_coefficients
        for momentum in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
            reduction = restrict_irrep(momentum, "H", "G1", rest.parity)
            projected = rest.little_group_projection(momentum, "G1", idx=0)
            expected = source @ np.column_stack(
                [np.asarray(basis[:, 0], complex).ravel() for basis in reduction["row_bases"]]
            )
            self.assertIsNot(projected, rest)
            self.assertEqual(rest.projection[-1], "H")
            np.testing.assert_allclose(
                _coefficient_matrix(projected, spins, components), expected,
                atol=2e-12, rtol=0,
            )

    def test_frontend_and_invalid_inputs(self):
        component = _components(1)[0]
        moments = {0: (0, 0, 0)}
        for gauge_name in ("A1u", "A1u+", "A1u-"):
            insertion = InsertionGaugeLink(
                SpinName.G1G, gauge_name, 0, "G1u", moments,
                {gauge_name: [[[(1, component)]]]},
            )
            operator = Operator("baryon", [insertion[0](0, 0, 0)], [2])
            self.assertEqual({term.rank for term in operator.terms}, {3})
            self.assertTrue(all(term.gamma is None and not hasattr(term, "flavors")
                                for term in operator.terms))
            self.assertEqual(insertion.charge_conjugation, None)

        local = Insertion.local_spin("G1", "g", moments)
        self.assertEqual(tuple(row[0].spin for row in local.rows),
                         baryon_local_spin_rows("G1", "g"))

        with self.assertRaises(ValueError):
            Insertion.from_irreps(SpinName.G1G, _spatial_rows(3), "T1u", "G1",
                                  moments, copy_index=1)
        with self.assertRaises(ValueError):
            Insertion.from_irreps(SpinName.G1G, _spatial_rows(1), "A1u", "G1g", moments)
        with self.assertRaises(TypeError):
            Insertion.from_irreps(SpinName.G1G, (((1, object()),),), "A1g", "G1", moments)
        with self.assertRaises(ValueError):
            InsertionGaugeLink(SpinName.G1G, "A1g+", 0, "G1g", moments,
                               {"A1g+": [[[(1, component)]]]}, profile=object())

        canonical = irrep_matrices((0, 0, 0), "E")
        missing = dict(canonical)
        missing.pop("c4x")
        with self.assertRaises(ValueError):
            Insertion.from_irreps(SpinName.G1G, _spatial_rows(2), "Eg", "H", moments,
                                  spatial_matrices=missing)
        bad_group = dict(canonical)
        bad_group["c4x"] = 2 * np.asarray(bad_group["c4x"], complex)
        with self.assertRaises(ValueError):
            Insertion.from_irreps(SpinName.G1G, _spatial_rows(2), "Eg", "H", moments,
                                  spatial_matrices=bad_group)
        nonunitary = {
            element: np.linalg.inv(np.diag([2, 1])) @ np.asarray(matrix, complex)
            @ np.diag([2, 1])
            for element, matrix in canonical.items()
        }
        with self.assertRaises(ValueError):
            Insertion.from_irreps(SpinName.G1G, _spatial_rows(2), "Eg", "H", moments,
                                  spatial_matrices=nonunitary)


if __name__ == "__main__":
    unittest.main()
