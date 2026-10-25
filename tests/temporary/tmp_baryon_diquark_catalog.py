"""Temporary (slow, about 44 s on CSNS CPU) case moved from ``tests/correlator/operators/test_baryon_diquark_catalog.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from correlator.operators.test_baryon_diquark_catalog import BaryonDiquarkCatalogTests
from correlator.operators.test_baryon_diquark_catalog import Insertion, baryon_diquark_operators, baryon_preset, np, restrict_irrep  # noqa: F401


class TemporaryBaryonDiquarkCatalogTests(BaryonDiquarkCatalogTests):
    def test_recipes_use_catalog_insertion_and_moving_interfaces(self):
        component = baryon_preset()[0]['component']
        momenta = {0: (0, 0, 0), 1: (0, 0, 1)}
        for recipe in baryon_diquark_operators():
            insertion = Insertion.from_irreps(recipe, [[(1, component)]], 'A1g', recipe.irrep, momenta)
            for r, row in enumerate(insertion.rows):
                actual = sum((term.coefficient * term.spin.dense() for term in row), np.zeros((4, 4, 4), complex))
                np.testing.assert_allclose(actual, recipe.rows[r].dense(), atol=3e-14)
            subduction = restrict_irrep((0, 0, 1), recipe.irrep, 'G1', recipe.parity)
            for occurrence in range(subduction['multiplicity']):
                moving = recipe.insertion(momenta).little_group_projection((0, 0, 1), 'G1', occurrence)
                for r, basis in enumerate(subduction['row_bases']):
                    actual = sum((t.coefficient * t.spin.dense() for t in moving.rows[r]),
                                 np.zeros((4, 4, 4), complex))
                    expected = sum((complex(w) * s.dense() for w, s in zip(basis[:, occurrence], recipe.rows)),
                                   np.zeros((4, 4, 4), complex))
                    np.testing.assert_allclose(actual, expected, atol=1e-14)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryBaryonDiquarkCatalogTests("test_recipes_use_catalog_insertion_and_moving_interfaces")])
