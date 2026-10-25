"""Check the reviewed public export inventory; never import private modules by accident.

Run with the CPU documentation extras installed. ``--write-inventory`` is an
explicit maintainer operation: inspect its diff before accepting an API change.
The inventory is a review gate, not a declaration of semantic-version stability.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "tools" / "public_api.json"
# Reviewed user-facing leaf modules. Do not promote every implementation
# module merely because it happens to define an internal __all__ list.
PUBLIC_LEAF_MODULES = {
    "lqcdweave.preset",
    "lqcdweave.contraction.correlator",
}
PUBLIC_ALIASES = {
    "lqcdweave.contraction.correlator": {"calculate": "calc_diagram_eval"},
}
# These __all__ lists organize implementation helpers, not user entry points.
INTERNAL_FACADES = {
    "lqcdweave.contraction.execution.operations",
    "lqcdweave.contraction.planning.optimizers",
}


def discover_modules(package_root):
    """Find canonical boundaries, public facades and curated leaf frontends."""
    result = {}
    for path in sorted(Path(package_root).rglob("*.py")):
        parts = path.relative_to(package_root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(("lqcdweave", *parts))
        if module in INTERNAL_FACADES:
            continue
        if path.name not in {"api.py", "__init__.py"} and module not in PUBLIC_LEAF_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        declares_exports = any(
            isinstance(node, ast.Name) and node.id == "__all__"
            and isinstance(node.ctx, ast.Store) for node in ast.walk(tree)
        )
        declares_exports = declares_exports or any(
            isinstance(node, ast.ImportFrom) and any(
                (alias.asname or alias.name) == "__all__" for alias in node.names
            ) for node in ast.walk(tree)
        )
        if module in PUBLIC_LEAF_MODULES:
            result[module] = "frontend"
        elif path.name == "api.py" or declares_exports:
            result[module] = "boundary" if path.name == "api.py" else "facade"
    return result


def module_errors(module, expected):
    """Validate one module without treating incidental imports as public API."""
    errors = []
    exports = getattr(module, "__all__", None)
    if not isinstance(exports, (tuple, list)) or not all(isinstance(n, str) for n in exports):
        return [f"{module.__name__}: __all__ must be a list/tuple of names"]
    if len(exports) != len(set(exports)):
        errors.append(f"{module.__name__}: duplicate __all__ entries")
    if any(n.startswith("_") or not n.isidentifier() for n in exports):
        errors.append(f"{module.__name__}: private/invalid export name")
    actual, wanted = set(exports), set(expected)
    if len(expected) != len(wanted):
        errors.append(f"{module.__name__}: duplicate inventory entries")
    if actual != wanted:
        errors.append(f"{module.__name__}: exports changed: added={sorted(actual-wanted)}, "
                      f"removed={sorted(wanted-actual)}")
    for name in exports:
        try:
            getattr(module, name)
        except (AttributeError, ImportError) as error:
            errors.append(f"{module.__name__}.{name}: cannot resolve export: {error}")
    return errors


def build_inventory(package_root):
    return {
        "schema_version": 1,
        "modules": {
            name: {"kind": kind, "exports": sorted(set(importlib.import_module(name).__all__))}
            for name, kind in discover_modules(package_root).items()
        },
    }


def check_inventory(inventory, package_root):
    if inventory.get("schema_version") != 1:
        return ["unsupported public API inventory schema"]
    errors = []
    discovered = discover_modules(package_root)
    expected = inventory["modules"]
    if set(discovered) != set(expected):
        errors.append(f"public modules changed: added={sorted(set(discovered)-set(expected))}, "
                      f"removed={sorted(set(expected)-set(discovered))}")
    loaded = {}
    for name, entry in expected.items():
        if name in discovered and discovered[name] != entry["kind"]:
            errors.append(f"{name}: inventory kind differs from source boundary")
        try:
            module = loaded[name] = importlib.import_module(name)
        except ImportError as error:
            errors.append(f"{name}: cannot import public module: {error}")
            continue
        errors.extend(module_errors(module, entry["exports"]))
    # A supported short facade must preserve the canonical object's identity.
    # Facades may deliberately expose a smaller set than their api.py module.
    for name, module in loaded.items():
        canonical = loaded.get(name + ".api")
        if canonical is None:
            continue
        for symbol in expected[name]["exports"]:
            if symbol not in expected[name + ".api"]["exports"]:
                errors.append(f"{name}.{symbol}: facade has no canonical api export")
            elif getattr(module, symbol, None) is not getattr(canonical, symbol, None):
                errors.append(f"{name}.{symbol}: facade does not preserve canonical identity")
    for name, aliases in PUBLIC_ALIASES.items():
        module = loaded.get(name)
        if module is None:
            continue
        for alias, canonical_name in aliases.items():
            if getattr(module, alias, None) is not getattr(module, canonical_name, None):
                errors.append(f"{name}.{alias}: public alias differs from {canonical_name}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-inventory", action="store_true")
    args = parser.parse_args()
    import lqcdweave
    package_root = Path(lqcdweave.__file__).parent
    if args.write_inventory:
        INVENTORY.write_text(json.dumps(build_inventory(package_root), indent=2) + "\n")
        print(f"Wrote {INVENTORY}; review all changes before accepting")
        return 0
    inventory = json.loads(INVENTORY.read_text())
    errors = check_inventory(inventory, package_root)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Public exports verified: {len(inventory['modules'])} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
