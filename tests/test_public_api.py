"""Reviewed public exports and canonical facade identities (CPU import gate)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "tools" / "check_public_api.py"
spec = importlib.util.spec_from_file_location("public_api_check_tool", path)
api_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_check)
INVENTORY = api_check.INVENTORY
check_inventory = api_check.check_inventory
discover_modules = api_check.discover_modules
module_errors = api_check.module_errors


class PublicApiTests(unittest.TestCase):
    def test_reviewed_exports_and_facade_identity(self):
        import lqcdweave
        package_root = Path(lqcdweave.__file__).parent
        inventory = json.loads(INVENTORY.read_text())
        errors = check_inventory(inventory, package_root)
        self.assertEqual(errors, [], "\n".join(errors))

    def test_public_calculate_alias_preserves_identity(self):
        import lqcdweave
        inventory = json.loads(INVENTORY.read_text())
        with patch("lqcdweave.contraction.correlator.calculate", object()):
            errors = check_inventory(inventory, Path(lqcdweave.__file__).parent)
        self.assertTrue(any("public alias differs from calc_diagram_eval" in error for error in errors))

    def test_export_guard_rejects_missing_added_private_and_duplicate_names(self):
        module = SimpleNamespace(__name__="example", __all__=["valid"], valid=object())
        self.assertEqual(module_errors(module, ["valid"]), [])
        module.__all__ = ["valid", "valid", "missing", "_private"]
        failures = "\n".join(module_errors(module, ["valid", "removed"]))
        for expected in ("duplicate", "private/invalid", "added=", "removed=", "cannot resolve"):
            self.assertIn(expected, failures)

    def test_incidental_module_names_are_not_public_exports(self):
        module = SimpleNamespace(__name__="example", __all__=["public"],
                                 public=object(), implementation_detail=object())
        self.assertEqual(module_errors(module, ["public"]), [])

    def test_inventory_includes_canonical_boundaries_not_internal_utility_facades(self):
        import lqcdweave
        discovered = discover_modules(Path(lqcdweave.__file__).parent)
        self.assertIn("lqcdweave.production.eigenvector.api", discovered)
        self.assertIn("lqcdweave.contraction.binding", discovered)
        self.assertEqual(discovered["lqcdweave.preset"], "frontend")
        self.assertEqual(discovered["lqcdweave.contraction.correlator"], "frontend")
        self.assertNotIn("lqcdweave.adapters.preset", discovered)
        self.assertNotIn("lqcdweave.contraction.execution.operations", discovered)
        self.assertNotIn("lqcdweave.contraction.planning.optimizers", discovered)


if __name__ == "__main__":
    unittest.main()
