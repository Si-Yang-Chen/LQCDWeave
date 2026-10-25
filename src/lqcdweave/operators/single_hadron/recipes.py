"""Recipes are data documents; this module only loads and reads them.

Two document kinds exist:

``gauge-recipe/1``
    Self-contained spatial kernel catalog plus the frozen couplings that
    reference it. One document is one recipe table (for example ``meson55``
    version ``v2``). Catalog rows are keyed by their original id and keep
    their original order; couplings keep their original list order.

``momentum-recipe/1``
    An ordered momentum axis. The order is an identity of already-produced
    data and is never regenerated from a q-squared rule.

Both documents record integrity digests. Loading never reorders, merges or
repairs a document; a mismatch is an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path

import numpy as np

from lqcdweave.contracts.elemental import ElementalComponent

GAUGE_SCHEMA = "gauge-recipe/1"
MOMENTUM_SCHEMA = "momentum-recipe/1"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value) -> str:
    return sha256(_canonical(value)).hexdigest()


def _read(path) -> dict:
    if isinstance(path, (str, Path)):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _package_text(*parts: str) -> str:
    resource = files("lqcdweave.operators.single_hadron").joinpath(*parts)
    return resource.read_text(encoding="utf-8")


@dataclass(frozen=True)
class MomentumRecipe:
    """An ordered momentum axis loaded from one document."""

    name: str
    momenta: tuple
    note: str = ""
    sha256: str = ""

    @classmethod
    def from_document(cls, document: dict) -> "MomentumRecipe":
        if document.get("schema") != MOMENTUM_SCHEMA:
            raise ValueError(f"expected {MOMENTUM_SCHEMA}, got {document.get('schema')!r}")
        entries = document["momenta"]
        momenta = []
        for position, entry in enumerate(entries):
            if int(entry["index"]) != position:
                raise ValueError(
                    f"momentum slot {position} has index {entry['index']}; "
                    "an ordered axis must be consecutive from zero")
            xyz = tuple(int(value) for value in entry["xyz"])
            if len(xyz) != 3:
                raise ValueError("momentum entries must be XYZ triples")
            momenta.append(xyz)
        if len(set(momenta)) != len(momenta):
            raise ValueError("momentum coordinates must be distinct")
        expected = _digest([list(value) for value in momenta])
        recorded = document.get("integrity", {}).get("momenta")
        if recorded is not None and recorded != expected:
            raise ValueError("momentum recipe digest does not match its content")
        return cls(str(document.get("name", "")), tuple(momenta),
                   str(document.get("note", "")), expected)

    @classmethod
    def load(cls, path) -> "MomentumRecipe":
        return cls.from_document(_read(path))

    @classmethod
    def named(cls, name: str) -> "MomentumRecipe":
        return cls.from_document(json.loads(_package_text("presets", "momentum", f"{name}.json")))

    @classmethod
    def from_values(cls, name, momenta) -> "MomentumRecipe":
        """Freeze an explicit axis (ordered XYZ triples)."""
        values = tuple(tuple(int(value) for value in momentum) for momentum in momenta)
        if not values or any(len(momentum) != 3 for momentum in values):
            raise ValueError("nonempty XYZ momenta are required")
        if len(set(values)) != len(values):
            raise ValueError("momentum coordinates must be distinct")
        return cls(str(name), values, "", _digest([list(value) for value in values]))

    def index(self, momentum) -> int:
        """Return the slot of an XYZ triple in this axis."""
        vector = tuple(int(value) for value in momentum)
        try:
            return self.momenta.index(vector)
        except ValueError:
            raise ValueError(f"momentum {vector} is not in recipe {self.name!r}") from None

    def as_mapping(self) -> dict:
        """Legacy view: ordered slot -> 'x y z'."""
        return {slot: " ".join(str(value) for value in momentum)
                for slot, momentum in enumerate(self.momenta)}

    def __len__(self) -> int:
        return len(self.momenta)


@dataclass(frozen=True)
class GaugeRecipe:
    """One frozen gauge recipe: catalog rows plus couplings that reference them."""

    name: str
    version: str
    convention: dict
    rows: tuple
    multiplets: tuple
    sources: dict
    sha256: str = ""

    @classmethod
    def from_document(cls, document: dict) -> "GaugeRecipe":
        if document.get("schema") != GAUGE_SCHEMA:
            raise ValueError(f"expected {GAUGE_SCHEMA}, got {document.get('schema')!r}")
        catalog = document["catalog"]
        rows = tuple(catalog[str(position)] for position in range(len(catalog)))
        for position, row in enumerate(rows):
            if int(row["id"]) != position:
                raise ValueError(f"catalog position {position} has id {row['id']}")
            if any(int(kernel) >= len(rows) for kernel in _kernel_ids(row)):
                raise ValueError("catalog row kernel ids are out of range")
        multiplets = tuple(document["multiplets"])
        for multiplet in multiplets:
            if any(int(kernel) >= len(rows) for kernel in multiplet["kernel_ids"]):
                raise ValueError(
                    f"{multiplet['name']}: kernel id is out of catalog range")
        expected = {"catalog": _digest(list(rows)),
                    "multiplets": _digest(list(multiplets))}
        recorded = document.get("integrity")
        if recorded is not None and recorded != expected:
            raise ValueError("gauge recipe digest does not match its content")
        return cls(str(document.get("name", "")), str(document.get("version", "")),
                   dict(document.get("convention", {})), rows, multiplets,
                   dict(document.get("sources", {})), _digest(document))

    @classmethod
    def load(cls, path) -> "GaugeRecipe":
        return cls.from_document(_read(path))

    @classmethod
    def named(cls, name: str, version: str) -> "GaugeRecipe":
        return cls.from_document(
            json.loads(_package_text("presets", "gauge", f"{name}_{version}.json")))

    def row(self, kernel_id: int) -> dict:
        return self.rows[int(kernel_id)]

    def multiplet(self, name: str) -> dict:
        for multiplet in self.multiplets:
            if multiplet["name"] == name:
                return multiplet
        raise KeyError(f"unknown recipe {name!r}")

    def components(self) -> tuple:
        return tuple(ElementalComponent.catalog(self.name, self.version, position)
                     for position in range(len(self.rows)))

    def names(self) -> tuple:
        return tuple(multiplet["name"] for multiplet in self.multiplets)

    @property
    def coupling_sha256(self) -> str:
        return _digest(list(self.multiplets))

    @property
    def catalog_sha256(self) -> str:
        return _digest(list(self.rows))


def _kernel_ids(row) -> tuple:
    return tuple(row.get("kernel_ids", ()))


def momentum_recipe(name: str) -> MomentumRecipe:
    """Load a packaged momentum recipe by name (``mom1`` .. ``mom9``)."""
    return MomentumRecipe.named(name)


def gauge_recipe(version: str, name: str = "meson55") -> GaugeRecipe:
    """Load a packaged gauge recipe by name and version (``v2``, ``csns``)."""
    return GaugeRecipe.named(name, version)


__all__ = [
    "GAUGE_SCHEMA", "MOMENTUM_SCHEMA",
    "GaugeRecipe", "MomentumRecipe", "gauge_recipe", "momentum_recipe",
]
