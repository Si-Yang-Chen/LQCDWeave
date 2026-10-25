"""Versioned 55-row meson tables and their matching ordered momentum preset.

Rows and couplings are data, not code: they are read from the self-contained
``presets/gauge/meson55_<version>.json`` document (catalog plus couplings).
The default v2 table retains 25 ordinary rows, 27 magnetic rows, and 3
electric candidates with source metadata and full path coefficients. The
``csns`` version preserves the original table and Clifford coupling.
Provenance is documented in ``docs/modules/operators/meson55.md``.
"""

from dataclasses import dataclass
from functools import lru_cache

from lqcdweave.contracts.elemental import (
    spatial_projection_rows,
)
from lqcdweave.contracts.elemental_catalog import ElementalCatalog

from .recipes import gauge_recipe, momentum_recipe


@lru_cache(maxsize=2)
def _document(version):
    return gauge_recipe(version)


@lru_cache(maxsize=2)
def _historical_momentum_dict():
    """The meson55 momentum axis, frozen in ``presets/momentum/mom4.json``.

    It is a data file, not a code constant: already-produced data depends on
    its exact ordering, so it is loaded verbatim and never regenerated.
    """
    return momentum_recipe("mom4").as_mapping()


# Transitional alias for call sites that still pass the axis explicitly.
# New code passes ``momenta=`` or ``momentum_dict=`` from the consumed data.
momDict_meson55 = _historical_momentum_dict()


def _validate_version(version):
    if not isinstance(version, str) or version not in ("v2", "csns"):
        raise ValueError("meson55 version must be 'v2' or 'csns'")
    return version


def meson55_preset(*, version="v2"):
    """Return independent operator rows with complete path/coefficient identity."""
    version = _validate_version(version)
    source_rows = [dict(row) for row in _document(version).rows]
    return ElementalCatalog.from_rows("meson55", version, source_rows).table(metadata=source_rows)


@lru_cache(maxsize=2)
def _components(version):
    return _document(_validate_version(version)).components()


@dataclass(frozen=True)
class GaugeOperator:
    """One frozen, canonically coupled rest-frame multiplet.

    Coefficients have axes (irrep row, Clifford component, spatial row).
    ``kernel_ids`` refer to the original catalog, never reader array slots.
    """

    name: str
    irrep: str
    parity: int
    c_factor: int
    reflection: int
    gamma_name: str
    masks: tuple
    phases: tuple
    kernel_ids: tuple
    kernel_label: str
    kernel_origin: tuple
    coefficients: tuple
    version: str = "v2"

    def insertion(self, momentum_dict=None):
        """Create the coupled insertion, retaining full component identity.

        ``momentum_dict`` is the ordered momentum axis of the data this
        operator is bound to. Omitting it uses the frozen historical meson55
        axis (``presets/momentum/mom4.json``); pass the real axis when the
        data was produced with a different one.
        """
        from .insertion import Insertion
        from .operator import InsertionTerm

        momentum_dict = _historical_momentum_dict() if momentum_dict is None else momentum_dict
        components = _components(self.version)
        rows = []
        for coefficients in self.coefficients:
            rows.append([
                InsertionTerm(components[kernel_id], gamma=mask, coefficient=weight)
                for i, mask in enumerate(self.masks)
                for j, kernel_id in enumerate(self.kernel_ids)
                if abs(weight := self.phases[i] * coefficients[i][j]) > 1e-12
            ])
        insertion = Insertion.from_rows(rows, momentum_dict)
        insertion.momentum_identity = True
        insertion.parity = self.parity
        insertion.charge_conjugation = self.c_factor
        insertion.projection = [None, None, self.irrep]
        insertion._rest_irrep = self.irrep
        return insertion


@lru_cache(maxsize=2)
def _load_meson55_operators(version):
    document = _document(_validate_version(version))
    return tuple(GaugeOperator(
        name=row["name"], irrep=row["irrep"], parity=row["parity"],
        c_factor=row["c_factor"], reflection=row["reflection"],
        gamma_name=row["gamma"], masks=tuple(row["masks"]), phases=tuple(row["phases"]),
        kernel_ids=tuple(row["kernel_ids"]), kernel_label=row["kernel_label"],
        kernel_origin=tuple(row["kernel_origin"]),
        coefficients=tuple(tuple(tuple(complex(re, im) for re, im in zip(real, imag))
                                 for real, imag in zip(real_row, imag_row))
                           for real_row, imag_row in zip(row["coefficients_real"],
                                                        row["coefficients_imag"])),
        version=version,
    ) for row in document.multiplets)


def meson55_operators(*, version="v2"):
    """Return immutable meson55 operator multiplets for a named table.

    The table includes all sixteen Clifford components coupled to all 55
    spatial rows. ``version="csns"`` selects the original frozen CSNS table;
    ``version="v2"`` selects the corrected default table. Moving projections
    use ``Insertion.little_group_projection``.
    """
    return _load_meson55_operators(_validate_version(version))
