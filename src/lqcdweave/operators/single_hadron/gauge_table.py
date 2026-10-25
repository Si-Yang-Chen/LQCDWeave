"""Paired production and insertion tables using the existing elemental contract."""

from copy import deepcopy
from functools import lru_cache

from lqcdweave.contracts.elemental import (
    GaugePath, integer, spatial_projection_rows,
)
from lqcdweave.contracts.elemental_catalog import ElementalCatalog, embedded_catalog
from ._gauge_presets import GAUGE_PRESETS


def _recipe(terms):
    return tuple((path.steps, coefficient) for path, coefficient in terms)


@lru_cache(maxsize=1)
def gauge_link_catalog():
    """The frozen v1 union of the existing length 0/1/2/3 preset rows."""
    rows, seen = [], set()
    for length in range(4):
        for copies in GAUGE_PRESETS[str(length)].values():
            for occurrence in copies:
                for terms in occurrence:
                    parsed = spatial_projection_rows([dict(id=0, terms=terms)])[0]
                    key = _recipe(parsed)
                    if key not in seen:
                        seen.add(key)
                        rows.append(dict(terms=terms))
    return ElementalCatalog.from_rows("gauge-link", "v1", rows)


def gauge_link_preset(max_length=1):
    """Return independent prebuilt irrep/copy/row/path-coefficient records.

    Presets 0, 1, 2, 3 cover spatial non-backtracking paths up to that length.
    They retain the established cubic rows, signs and normalization. There is
    no temporal-path guess and no runtime enumeration of additional lengths.
    """
    max_length = integer(max_length, "max_length", 0)
    if str(max_length) not in GAUGE_PRESETS:
        raise ValueError("available gauge-link preset lengths are 0, 1, 2, 3")
    return deepcopy(GAUGE_PRESETS[str(max_length)])


def gauge_link_tables(insertion_dict):
    """Pair explicit projected production rows with matching InsertionGaugeLink rows.

    Adapted from the old gen_gauge_list row-flattening/remapping algorithm.
    Input is irrep -> copies -> rows -> records containing explicit path and
    coefficient. Select/reorder irrep copies before this call to produce only
    needed elementals. Output is (operator_table, insertion_dict); both use
    the same versioned catalog and component numbers, independent of row slot.
    Unknown integer path IDs are rejected: import them with their actual paths.
    """
    selected, insertions = [], {}
    for name, copies in insertion_dict.items():
        insertions[name] = []
        for rows in copies:
            mapped = []
            for row in rows:
                if any(not isinstance(term.get("path"), (list, tuple, GaugePath))
                       for term in row):
                    raise ValueError("gauge table requires explicit path steps, not unknown integer IDs")
                records = [dict(path=list(GaugePath.coerce(term["path"]).steps),
                                coefficient=term["coefficient"]) for term in row]
                terms = spatial_projection_rows([dict(id=0, terms=records)])[0]
                key = _recipe(terms)
                if key not in selected:
                    selected.append(key)
                mapped.append([[1, selected.index(key)]])
            insertions[name].append(mapped)
    if not selected:
        raise ValueError("select at least one nonempty gauge-link row")
    catalog = gauge_link_catalog()
    lookup = {_recipe(terms): i for i, terms in enumerate(catalog.rows)}
    if all(key in lookup for key in selected):
        numbers = [lookup[key] for key in selected]
    else:
        # An edited table is a custom directory, never a mislabeled preset.
        rows = [dict(terms=[dict(path=list(path), coefficient=[c.real, c.imag]) for path, c in key])
                for key in selected]
        catalog = embedded_catalog(rows)
        numbers = list(range(len(rows)))
    table = catalog.table(numbers)
    for copies in insertions.values():
        for occurrence in copies:
            for row in occurrence:
                row[0][1] = catalog.component(numbers[row[0][1]])
    return table, insertions
