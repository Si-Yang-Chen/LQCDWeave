"""Diquark x free-quark recipes against independent gamma algebra and the 64-row catalog."""

from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from lqcdweave.symmetry import restrict_irrep

from lqcdweave.contraction.expansion.baryon_factor_network import factor_row_coefficients
from lqcdweave.operators.single_hadron.api import (
    BaryonSpinOperator, Insertion, baryon_diquark_operators, baryon_spin_operators, baryon_preset, gamma,
)

C = gamma(2) @ gamma(8)
G5, G4, SPATIAL = gamma(15), gamma(8), (gamma(1), gamma(2), gamma(4))
SCALARS = {'C': (C,), 'Cg5': (C @ G5,), 'Cg4': (C @ G4,), 'Cg4g5': (C @ G4 @ G5,)}
VECTORS = {'Cgi': tuple(C @ g for g in SPATIAL), 'Cgig5': tuple(C @ g @ G5 for g in SPATIAL),
           'Cgig4': tuple(C @ g @ G4 for g in SPATIAL), 'Cgig4g5': tuple(C @ g @ G4 @ G5 for g in SPATIAL)}
DIQUARKS = {**SCALARS, **VECTORS}
SYMMETRIC = {'Cg4', 'Cgi', 'Cgig4', 'Cgig4g5'}


def names():
    return [f'diquark_{label}_{irrep}{p}' for label in DIQUARKS
            for irrep in (('G1',) if label in SCALARS else ('G1', 'H')) for p in 'gu']


class BaryonDiquarkCatalogTests(unittest.TestCase):
    def test_charge_conjugation_and_diquark_symmetry(self):
        for index in (1, 2, 4, 8):
            np.testing.assert_allclose(C @ gamma(index) @ np.linalg.inv(C), -gamma(index).T, atol=1e-15)
        for label, matrices in DIQUARKS.items():
            for matrix in matrices:
                np.testing.assert_allclose(matrix.T, (1 if label in SYMMETRIC else -1) * matrix, atol=1e-15)
        # The 16 C Gamma matrices are trace-orthogonal, so the recipes span the full space.
        stacked = np.stack([m.reshape(16) for matrices in DIQUARKS.values() for m in matrices])
        np.testing.assert_allclose(stacked.conj() @ stacked.T, 4 * np.eye(16), atol=1e-14)

    def test_recipes_are_exact_catalog_combinations_and_a_complete_basis(self):
        recipes = baryon_diquark_operators()
        self.assertEqual([r.name for r in recipes], names())
        catalog = {r.name: r for r in baryon_spin_operators()}
        for recipe in recipes:
            self.assertIsInstance(recipe, BaryonSpinOperator)
            self.assertEqual((len(recipe.rows), recipe.basis), ({'G1': 2, 'H': 4}[recipe.irrep], 'degrand_rossi'))
            weights = dict(recipe.weights)
            self.assertAlmostEqual(sum(abs(w) ** 2 for w in weights.values()), 1, places=13)
            first = recipe.weights[0][1]
            self.assertAlmostEqual(first.imag, 0, places=14)
            self.assertGreater(first.real, 0)
            for name in weights:
                self.assertEqual((catalog[name].irrep, catalog[name].parity), (recipe.irrep, recipe.parity))
            for r, spin in enumerate(recipe.rows):
                expected = sum(w * catalog[name].rows[r].dense() for name, w in weights.items())
                np.testing.assert_allclose(spin.dense(), expected, atol=1e-14)
        matrix = np.stack([spin.dense().reshape(64) for recipe in recipes for spin in recipe.rows])
        np.testing.assert_allclose(matrix.conj() @ matrix.T, np.eye(64), atol=5e-14)

    def test_rows_are_diquarks_times_parity_projected_free_spinors(self):
        parity = np.kron(np.kron(gamma(8), gamma(8)), gamma(8)).conj()
        for recipe in baryon_diquark_operators():
            label = recipe.name.split('_')[1]
            matrices = DIQUARKS[label]
            values = np.stack([spin.dense() for spin in recipe.rows])
            for spin in values:
                free = np.einsum('iab,abs->is', np.conj(matrices), spin) / 4
                np.testing.assert_allclose(spin, np.einsum('iab,is->abs', np.asarray(matrices), free), atol=1e-14)
                np.testing.assert_allclose(parity @ spin.reshape(64), recipe.parity * spin.reshape(64),
                                           atol=1e-14)
            # Exact split at the free-quark slot: rank 1 for scalar, rank 3 for vector diquarks.
            factor = factor_row_coefficients(values)
            self.assertEqual((factor.rank, factor.slot, factor.pairs), (len(matrices), 2, (0, 1)))

    def test_cg5_reproduces_reference_program_g13_combination(self):
        # The LL3/XiN frontends rebuilt the reference G13 x e_r from these catalog rows.
        recipes = {r.name: r for r in baryon_diquark_operators()}
        for suffix, sign in (('g', 1), ('u', -1)):
            reference = np.array([np.sqrt(2 / 3), sign * np.sqrt(2), 2 / np.sqrt(3)]) / 2
            weights = dict(recipes[f'diquark_Cg5_G1{suffix}'].weights)
            actual = [weights[f'spin_G1{suffix}_{name}'] for name in ('A_copy0', 'MA_copy0', 'MA_copy2')]
            self.assertEqual(len(weights), 3)
            np.testing.assert_allclose(actual, reference, atol=1e-14)

    def test_frozen_resource_rebuild(self):
        root = Path(__file__).resolve().parents[3]
        with TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(root / 'tools/build_baryon_diquark.py'),
                                     '--output-dir', directory], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), dict(recipes=24))
            stored = files('lqcdweave.operators.single_hadron').joinpath('presets', 'baryon_diquark_v1.json')
            rebuilt = json.loads((Path(directory) / 'baryon_diquark_v1.json').read_bytes())
            frozen = json.loads(stored.read_bytes())
            # SVD/BLAS may differ in the last bits across platforms. Keep all
            # schema, ordering and support exact; compare only coefficients
            # at the same absolute precision as the independent algebra tests.
            rebuilt_coefficients, frozen_coefficients = [], []
            for document, coefficients in ((rebuilt, rebuilt_coefficients), (frozen, frozen_coefficients)):
                for recipe in document['recipes']:
                    for term in [*recipe['weights'], *(term for row in recipe['rows'] for term in row)]:
                        coefficients.append(term['coefficient'])
                        term['coefficient'] = None
            self.assertEqual(rebuilt, frozen)
            np.testing.assert_allclose(rebuilt_coefficients, frozen_coefficients, rtol=0, atol=1e-14)

if __name__ == '__main__':
    unittest.main()
