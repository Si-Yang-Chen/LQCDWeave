"""Temporary (slow, about 44 s on CSNS CPU) case moved from ``tests/correlator/operators/test_baryon_spin_catalog.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from correlator.operators.test_baryon_spin_catalog import BaryonSpinCatalogTests
from correlator.operators.test_baryon_spin_catalog import Insertion, InsertionGaugeLink, baryon_preset, baryon_spin_operators, np, restrict_irrep  # noqa: F401


class TemporaryBaryonSpinCatalogTests(BaryonSpinCatalogTests):
    def test_catalog_recipes_couple_through_existing_spatial_and_moving_interfaces(self):
        component = baryon_preset()[0]['component']
        spatial = [[(1,component)]]
        momenta = {0:(0,0,0),1:(0,0,1)}
        for recipe in baryon_spin_operators():
            insertion = Insertion.from_irreps(recipe,spatial,'A1g',recipe.irrep,momenta)
            via_dictionary = InsertionGaugeLink(recipe,'A1g',0,recipe.irrep,momenta,{'A1g':[spatial]})
            self.assertEqual(insertion.rows,via_dictionary.rows)
            for i,row in enumerate(insertion.rows):
                actual = sum((term.coefficient*term.spin.dense() for term in row),np.zeros((4,4,4),complex))
                np.testing.assert_allclose(actual,recipe.rows[i].dense(),atol=3e-14)
            target = 'G1'
            subduction = restrict_irrep((0,0,1),recipe.irrep,target,recipe.parity)
            for occurrence in range(subduction['multiplicity']):
                moving = recipe.insertion(momenta).little_group_projection((0,0,1),target,occurrence)
                for r,basis in enumerate(subduction['row_bases']):
                    actual = sum((t.coefficient*t.spin.dense() for t in moving.rows[r]),np.zeros((4,4,4),complex))
                    expected = sum((complex(w)*s.dense() for w,s in zip(basis[:,occurrence],recipe.rows)),np.zeros((4,4,4),complex))
                    np.testing.assert_allclose(actual,expected,atol=1e-14)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryBaryonSpinCatalogTests("test_catalog_recipes_couple_through_existing_spatial_and_moving_interfaces")])
