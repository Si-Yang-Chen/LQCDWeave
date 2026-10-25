"""Frozen baryon recipes checked against determinants and exchange matrices."""

from collections import Counter
from itertools import permutations
import unittest

import numpy as np

from lqcdweave.operators.single_hadron.api import baryon_preset
from lqcdweave.production.elemental.api import BaryonElementalGenerator
from production.elemental.test_elemental import fixture
from test_baryon_elemental import determinant_reference


class BaryonCatalogTests(unittest.TestCase):
    def test_directory_contents_and_selection_keep_global_numbers(self):
        table = baryon_preset()
        self.assertEqual(Counter(row['kind'] for row in table),
                         {'raw_paths':343, 'distance':8, 'projected':1680})
        self.assertEqual(len(table), 2031)
        self.assertEqual(table[0]['name'], 'local')
        selected = table.select(('distance/0,0,1', 'local'))
        self.assertEqual([row['id'] for row in selected], [344, 0])
        self.assertEqual(selected[0]['component'].key, ('catalog', 'baryon-vvv', 'v1', 344))
        with self.assertRaises(ValueError):
            baryon_preset(version='unknown')

    def test_all_recipe_forms_match_independent_spatial_determinants(self):
        sizes, links, vectors, _ = fixture()
        names = ('local', 'paths/0,1,2', 'distance/0,0,1', 'distance/1,1,1',
                 's3/0,1,2/S', 's3/0,1,2/A',
                 's3/0,1,2/mixed0/MS', 's3/0,1,2/mixed0/MA',
                 's2/02/0,1,2/S', 's2/02/0,1,2/A')
        table = baryon_preset().select(names)
        p = (1, -1, 0)
        generator = BaryonElementalGenerator(links, vectors, (p,), table)
        actual = generator.calc_path_blocks(1)['vvv'][:, 0].copy()
        for index, row in enumerate(table):
            expected = sum(complex(*term['coefficient'])*determinant_reference(
                links, (vectors, vectors, vectors), term['paths'], 1, p) for term in row['terms'])
            np.testing.assert_allclose(actual[index], expected, atol=3e-12, rtol=3e-12)

    def test_s3_mixed_and_s2_rows_have_declared_spatial_exchange_including_color(self):
        from lqcdweave.operators.correlator.api import QuarkExchange
        _, links, vectors, _ = fixture()
        names = tuple('s3/0,1,2/'+suffix for suffix in
                      ('S','A','mixed0/MS','mixed0/MA','mixed1/MS','mixed1/MA'))
        table = baryon_preset().select(names)
        values = BaryonElementalGenerator(links, vectors, ((0,0,0),), table).calc_path_blocks(0)['vvv'][:,0]
        for p in permutations(range(3)):
            inverse = tuple(p.index(i) for i in range(3))
            np.testing.assert_allclose(values[0].transpose(inverse), values[0], atol=3e-12)
            sign = (-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
            np.testing.assert_allclose(values[1].transpose(inverse), sign*values[1], atol=3e-12)
            matrix = np.asarray(QuarkExchange().matrices[p], complex)
            for offset in (2,4):
                moved = np.array([value.transpose(inverse) for value in values[offset:offset+2]])
                np.testing.assert_allclose(moved, np.einsum('ba,bijk->aijk',matrix,values[offset:offset+2]), atol=3e-12)
        for slots in ((0,1),(0,2),(1,2)):
            table = baryon_preset().select(tuple(f's2/{slots[0]}{slots[1]}/0,1,2/{name}' for name in ('S','A')))
            pair = BaryonElementalGenerator(links,vectors,((0,0,0),),table).calc_path_blocks(0)['vvv'][:,0]
            for value, sign in zip(pair,(1,-1)):
                np.testing.assert_allclose(value.swapaxes(*slots),sign*value,atol=3e-12)


if __name__ == '__main__':
    unittest.main()
