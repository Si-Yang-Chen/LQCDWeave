"""Versioned spatial recipe directories shared by two- and three-leg elementals.

Components reference (directory, version, number). Paths and coefficients live
once in the directory, and are never part of an individual component key.
"""

from copy import deepcopy
from dataclasses import dataclass


def catalog_key(name, version, number):
    from .elemental import integer
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError("catalog name and version must be nonempty strings")
    return ("catalog", name, version, integer(number, "catalog component", 0))


@dataclass(frozen=True)
class ElementalCatalog:
    """Immutable directory of ordered spatial recipes, independent of datasets."""

    name: str
    version: str
    rank: int
    rows: tuple

    @classmethod
    def from_rows(cls, name, version, rows, *, rank=2):
        from .elemental import elemental_operator_rows
        catalog_key(name, version, 0)
        # Directory numbering is independent of a selected table's array slots.
        records = tuple(dict(id=i, terms=row["terms"]) for i, row in enumerate(rows))
        parsed = elemental_operator_rows(records, rank=rank)
        if any(step >= 6 for _, terms in parsed for path, _ in terms
               for p in ((path,) if rank == 2 else path) for step in p.steps):
            raise ValueError("elemental catalogs require spatial paths")
        return cls(name, version, rank, tuple(terms for _, terms in parsed))

    @classmethod
    def from_document(cls, document):
        if document.get("schema") != "elemental-catalog/v1":
            raise ValueError("unsupported elemental catalog schema")
        rows = document["rows"]
        if any(row.get("id") != i or type(row.get("id")) is not int for i, row in enumerate(rows)):
            raise ValueError("catalog numbers must be consecutive from zero")
        return cls.from_rows(document["name"], document["version"], rows, rank=document["rank"])

    def component(self, number):
        from .elemental import ElementalComponent, BaryonComponent
        key = catalog_key(self.name, self.version, number)
        if key[-1] >= len(self.rows):
            raise IndexError("component is outside the elemental catalog")
        return (ElementalComponent if self.rank == 2 else BaryonComponent)(key)

    def table(self, numbers=None, *, metadata=None):
        numbers = tuple(range(len(self.rows))) if numbers is None else tuple(numbers)
        if not numbers or len(set(numbers)) != len(numbers):
            raise ValueError("select distinct nonempty catalog component numbers")
        records = []
        for number in numbers:
            component = self.component(number)
            record = {} if metadata is None else deepcopy(metadata[number])
            record.update(id=number, component=component, terms=self._terms(number))
            records.append(record)
        return _CatalogTable(records, self)

    def _terms(self, number):
        return [dict(**({"path": list(path.steps)} if self.rank == 2 else
                        {"paths": [list(p.steps) for p in path]}),
                     coefficient=[coefficient.real, coefficient.imag])
                for path, coefficient in self.rows[number]]

    @property
    def document(self):
        return dict(schema="elemental-catalog/v1", name=self.name, version=self.version,
                    rank=self.rank, rows=[dict(id=i, terms=self._terms(i)) for i in range(len(self.rows))])


class _CatalogTable(list):
    """A selected table keeps its directory when sliced or reordered."""

    def __init__(self, records, catalog):
        super().__init__(records)
        self.catalog = catalog

    def __getitem__(self, index):
        value = super().__getitem__(index)
        return type(self)(value, self.catalog) if isinstance(index, slice) else value

    def select(self, numbers):
        records = {row["id"]: row for row in self}
        records.update({row["name"]: row for row in self if "name" in row})
        chosen = [deepcopy(records[number]) for number in numbers]
        if len({row["id"] for row in chosen}) != len(chosen) or not chosen:
            raise ValueError("select distinct nonempty catalog components")
        return type(self)(chosen, self.catalog)


def catalog_definition(table):
    """Dataset metadata for a selected directory, or None for raw legacy tables."""
    catalog = getattr(table, "catalog", None)
    if catalog is None:
        return None
    return dict(directory=catalog.document, components=[row["id"] for row in table])


def embedded_catalog(rows, *, rank=2):
    """Import an explicit custom table as one self-contained versioned directory."""
    from .dataset import stable_digest
    catalog = ElementalCatalog.from_rows("custom", "pending", rows, rank=rank)
    return ElementalCatalog("custom", stable_digest(catalog.document["rows"]), rank, catalog.rows)


def stored_catalog_components(selection, rows, *, rank=2):
    """Decode a directory selection and verify its stored numerical recipe.

    Old explicit custom tables are imported as a self-contained directory.
    A preset is never inferred from row count, path resemblance, or array shape.
    """
    if selection is None:
        catalog = embedded_catalog(rows, rank=rank)
        return tuple(catalog.component(i) for i in range(len(catalog.rows)))
    catalog = ElementalCatalog.from_document(selection["directory"])
    numbers = selection["components"]
    if catalog.rank != rank or len(numbers) != len(rows) or len(set(numbers)) != len(numbers):
        raise ValueError("stored elemental catalog selection differs from its rows")
    actual = ElementalCatalog.from_rows("stored", "check", rows, rank=rank)
    if tuple(catalog.rows[i] for i in numbers) != actual.rows:
        raise ValueError("stored recipe conflicts with its catalog version and component numbers")
    return tuple(catalog.component(i) for i in numbers)


def stored_catalog_table(selection, rows, *, rank=2):
    stored_catalog_components(selection, rows, rank=rank)
    catalog = ElementalCatalog.from_document(selection["directory"])
    return catalog.table(selection["components"])
