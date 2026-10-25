"""Freeze gauge and momentum recipes into self-contained JSON documents.

The documents are data, not code: the ordered momentum axis and the frozen
coupling coefficients are identities of already-produced data, so this tool
copies them verbatim and records integrity hashes. It never regenerates an
ordering from a q-squared rule and never sorts a catalog.

Raw inputs (the historical preset files) live in ``tools/presets_raw``; the
package only ships ``presets/gauge`` and ``presets/momentum``.

Usage:
    python tools/build_recipe_documents.py [--check]

``--check`` verifies the committed documents still match their sources and
their recorded integrity digests.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESET_DIR = ROOT / "src" / "lqcdweave" / "operators" / "single_hadron" / "presets"
GAUGE_DIR = PRESET_DIR / "gauge"
MOMENTUM_DIR = PRESET_DIR / "momentum"
RAW_DIR = ROOT / "tools" / "presets_raw"

GAUGE_PACKAGES = (
    # (version, catalog filename, coupling filename)
    ("v2", "meson55_v2.json", "meson55_bindings_v2.json"),
    ("csns", "meson55.json", "meson55_bindings.json"),
)

# Momentum documents are frozen data; they are verified, never regenerated.
MOMENTUM_NAMES = ("mom1", "mom2", "mom3", "mom4", "mom9", "mom_test")

# Historical phase/boundary convention of the meson55 catalog. Kept here so the
# binding never has to guess it.
GAUGE_CONVENTION = {
    "phase": "midpoint",
    "boundary": "periodic",
    "components": "meson55",
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value) -> str:
    return sha256(_canonical(value)).hexdigest()


def _load_raw(name: str) -> dict:
    return json.loads((RAW_DIR / name).read_text(encoding="utf-8"))


def build_gauge(version: str, catalog_name: str, coupling_name: str) -> dict:
    catalog = _load_raw(catalog_name)
    coupling = _load_raw(coupling_name)

    rows = {}
    for position, row in enumerate(catalog["rows"]):
        if int(row["id"]) != position:
            raise ValueError(f"{catalog_name}: row {position} has id {row['id']}")
        rows[str(position)] = row

    multiplets = list(coupling["multiplets"])
    names = [multiplet["name"] for multiplet in multiplets]
    if len(set(names)) != len(names):
        raise ValueError(f"{coupling_name}: duplicate multiplet names")

    document = {
        "schema": "gauge-recipe/1",
        "name": "meson55",
        "version": version,
        "convention": dict(GAUGE_CONVENTION),
        "catalog": rows,
        "multiplets": multiplets,
        "sources": {
            "catalog": catalog_name,
            "couplings": coupling_name,
            "catalog_sha256": sha256((RAW_DIR / catalog_name).read_bytes()).hexdigest(),
            "couplings_sha256": sha256((RAW_DIR / coupling_name).read_bytes()).hexdigest(),
        },
    }
    document["integrity"] = {
        # Digest the ordered row list, not the keyed mapping: the reader
        # flattens the catalog back into its original order.
        "catalog": _digest(list(rows.values())),
        "multiplets": _digest(multiplets),
    }
    return document


def build_momentum(name: str, mapping, note: str) -> dict:
    momenta = []
    for slot, text in mapping.items():
        xyz = [int(value) for value in text.split()]
        if len(xyz) != 3:
            raise ValueError(f"{name}: slot {slot} is not an XYZ triple")
        momenta.append({"index": int(slot), "xyz": xyz})

    # Keep mapping insertion order; slots must be consecutive from zero.
    indices = [entry["index"] for entry in momenta]
    if indices != list(range(len(momenta))):
        raise ValueError(f"{name}: slots must be consecutive from zero")

    document = {
        "schema": "momentum-recipe/1",
        "name": name,
        "note": note,
        "momenta": momenta,
    }
    document["integrity"] = {"momenta": _digest([entry["xyz"] for entry in momenta])}
    return document


def check_momentum(path: Path) -> bool:
    """Verify a frozen momentum document against its own digest and schema."""
    if not path.exists():
        print(f"missing: {path}")
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    momenta = [entry["xyz"] for entry in document["momenta"]]
    if document.get("schema") != "momentum-recipe/1":
        print(f"bad schema: {path}")
        return False
    if document.get("integrity", {}).get("momenta") != _digest(momenta):
        print(f"digest mismatch: {path}")
        return False
    indices = [entry["index"] for entry in document["momenta"]]
    if indices != list(range(len(indices))):
        print(f"non-consecutive slots: {path}")
        return False
    return True


def write(path: Path, document: dict, *, check: bool) -> bool:
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            print(f"stale: {path}")
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    ok = True
    for version, catalog_name, coupling_name in GAUGE_PACKAGES:
        document = build_gauge(version, catalog_name, coupling_name)
        ok &= write(GAUGE_DIR / f"meson55_{version}.json", document, check=args.check)

    for name in MOMENTUM_NAMES:
        path = MOMENTUM_DIR / f"{name}.json"
        if args.check:
            ok &= check_momentum(path)
        else:
            ok &= check_momentum(path)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
