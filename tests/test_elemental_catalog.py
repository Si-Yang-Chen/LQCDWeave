"""Versioned recipe references shared by meson and baryon production."""

from copy import deepcopy
import unittest

from lqcdweave.contracts.elemental import ElementalComponent, BaryonComponent, elemental_operator_rows
from lqcdweave.contracts.elemental_catalog import (
    ElementalCatalog, catalog_definition, stored_catalog_components,
)


class ElementalCatalogTests(unittest.TestCase):
    def test_both_ranks_reference_directory_version_and_number(self):
        for rank, kind in ((2, ElementalComponent), (3, BaryonComponent)):
            rows = [dict(terms=[dict(path=(() if rank == 2 else ((), (), ())), coefficient=1)])]
            catalog = ElementalCatalog.from_rows('example', 'v1', rows, rank=rank)
            component = catalog.component(0)
            self.assertIsInstance(component, kind)
            self.assertEqual(component.key, ('catalog', 'example', 'v1', 0))
            self.assertEqual(kind.catalog('example', 'v1', 0), component)
            self.assertNotEqual(kind.catalog('example', 'v2', 0), component)
            self.assertEqual(ElementalCatalog.from_document(catalog.document), catalog)
            self.assertEqual(elemental_operator_rows(catalog.table(), rank=rank)[0][0], component)
        self.assertFalse(hasattr(ElementalComponent, 'projection'))

    def test_selection_roundtrip_and_conflicting_recipe_rejection(self):
        catalog = ElementalCatalog.from_rows('example', 'v1', [
            dict(terms=[dict(path=(i,), coefficient=[1, i])]) for i in range(3)])
        table = catalog.table().select((2, 0))
        self.assertEqual([row['id'] for row in table[:1]], [2])
        selection = catalog_definition(table)
        rows = [dict(id=i, terms=row['terms']) for i, row in enumerate(table)]
        self.assertEqual(stored_catalog_components(selection, rows),
                         (catalog.component(2), catalog.component(0)))
        wrong = deepcopy(table)
        wrong[0]['terms'][0]['coefficient'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            elemental_operator_rows(wrong)
        rows[0]['terms'][0]['coefficient'] = [99, 0]
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            stored_catalog_components(selection, rows)


if __name__ == '__main__':
    unittest.main()
