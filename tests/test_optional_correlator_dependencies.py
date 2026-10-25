"""Optional Correlator imports fail clearly without blocking base APIs."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


_SPEC = importlib.util.find_spec("lqcdweave")
if _SPEC is None or _SPEC.origin is None:
    raise RuntimeError("lqcdweave must be importable to run these subprocess tests")
_PACKAGE_IMPORT_ROOT = Path(_SPEC.origin).resolve().parent.parent


def _subprocess_environment():
    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    roots = [str(_PACKAGE_IMPORT_ROOT)]
    roots.extend(item for item in previous.split(os.pathsep) if item and item not in roots)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["ED_EXPECTED_PACKAGE_IMPORT_ROOT"] = str(_PACKAGE_IMPORT_ROOT)
    return env


class OptionalCorrelatorDependencyTests(unittest.TestCase):
    def run_child(self, source, *args):
        result = subprocess.run(
            [sys.executable, "-c", source, *args],
            cwd=tempfile.gettempdir(),
            env=_subprocess_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"subprocess failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result

    def test_base_workflows_and_operator_api_import_without_optional_modules(self):
        self.run_child(
            r'''
import importlib.abc
import importlib
import os
from pathlib import Path
import sys

expected_root = Path(os.environ["ED_EXPECTED_PACKAGE_IMPORT_ROOT"]).resolve()
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for root in ("sympy", "scipy", "lazy_object_proxy"):
            if fullname == root or fullname.startswith(root + "."):
                raise ModuleNotFoundError("blocked optional root", name=root)
sys.meta_path.insert(0, BlockOptional())
for name in (
    "lqcdweave",
    "lqcdweave.runtime.datasets",
    "lqcdweave.production.elemental.planning",
    "lqcdweave.operators.single_hadron.api",
    "lqcdweave.contraction.binding.api",
):
    module = importlib.import_module(name)
    if name == "lqcdweave":
        assert Path(module.__file__).resolve().parent.parent == expected_root
from lqcdweave.operators.single_hadron.api import meson55_operators
from lqcdweave.contraction.binding.api import RecipeVertex
recipes = meson55_operators()
assert len(recipes) == 368
assert sum(len(recipe.coefficients) for recipe in recipes) == 880
assert len(RecipeVertex(object(), momenta=((0, 0, 0),)).recipes) == 368
'''
        )

    def test_symbolic_entrypoints_name_the_missing_extra_dependency(self):
        source = r'''
import importlib.abc
import importlib
import os
from pathlib import Path
import sys

root, module_name = sys.argv[1:3]
expected_root = Path(os.environ["ED_EXPECTED_PACKAGE_IMPORT_ROOT"]).resolve()
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == root or fullname.startswith(root + "."):
            raise ModuleNotFoundError("blocked optional root", name=root)
sys.meta_path.insert(0, BlockOptional())
try:
    importlib.import_module(module_name)
except ModuleNotFoundError as exc:
    assert type(exc) is ModuleNotFoundError, type(exc)
    assert exc.name == root, exc.name
    assert f"Missing optional dependency {root!r}." in str(exc), str(exc)
    assert "Python 3.11+" in str(exc), str(exc)
    assert "python -m pip install -e '.[correlator]'" in str(exc), str(exc)
    assert "docs/modules/correlator/correlator_frontend.md" in str(exc), str(exc)
    assert type(exc.__cause__) is ModuleNotFoundError, type(exc.__cause__)
    assert exc.__cause__.name == root, exc.__cause__.name
else:
    raise AssertionError("optional Correlator entrypoint unexpectedly imported")
'''
        for root in ("sympy", "lazy_object_proxy"):
            for module_name in (
                "lqcdweave.operators.correlator",
                "lqcdweave.contraction.diagram_measurement",
                "lqcdweave.contraction.vsv_graph",
            ):
                with self.subTest(root=root, module=module_name):
                    self.run_child(source, root, module_name)

    def test_unrelated_import_failures_are_propagated_unchanged(self):
        source = r'''
import importlib.abc
import sys

kind = sys.argv[1]
if kind == "transitive":
    expected = ModuleNotFoundError(
        "unrelated transitive import",
        name="lqcdweave.symmetry.sympy_utils")
elif kind == "import":
    expected = ImportError("cannot import name 'missing_symbol' from optional dependency")
elif kind == "runtime":
    expected = RuntimeError("broken optional package initialization")
else:
    raise AssertionError(kind)
class FailApi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lqcdweave.operators.correlator.api":
            raise expected
sys.meta_path.insert(0, FailApi())
try:
    import lqcdweave.operators.correlator
except BaseException as caught:
    assert caught is expected
    assert type(caught) is type(expected)
    if kind == "transitive":
        assert caught.name == "lqcdweave.symmetry.sympy_utils"
    assert caught.__cause__ is None
else:
    raise AssertionError("injected import failure unexpectedly succeeded")
'''
        for kind in ("transitive", "import", "runtime"):
            with self.subTest(kind=kind):
                self.run_child(source, kind)

    def test_namespace_exports_keep_api_object_identity(self):
        self.run_child(
            r'''
import os
from pathlib import Path
import lqcdweave
import lqcdweave.operators.correlator as package
from lqcdweave.operators.correlator import api

expected_root = Path(os.environ["ED_EXPECTED_PACKAGE_IMPORT_ROOT"]).resolve()
assert Path(lqcdweave.__file__).resolve().parent.parent == expected_root
assert package.__all__ is api.__all__
for name in api.__all__:
    assert getattr(package, name) is getattr(api, name), name
'''
        )

    def test_dependency_free_early_paths(self):
        source = r'''
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.single_hadron.api import (
    DerivativeName, GammaName, Insertion, InsertionTerm, ProjectionName,
)
static = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, {})
assert static.little_group_projection((0, 0, 0), "A_1") is static

generic = Insertion.from_rows(
    [[InsertionTerm(ElementalComponent(0), gamma=0)]],
    {0: (0, 0, 1)},
)
try:
    generic.little_group_projection((0, 0, 1), "A_1")
except ValueError as exc:
    assert "generic rows must be projected before Insertion.from_rows" in str(exc)
else:
    raise AssertionError("generic row projection unexpectedly succeeded")
'''
        self.run_child(source)

    def test_builtin_symmetry_module_works_when_external_import_is_blocked(self):
        self.run_child(
            r'''
import importlib.abc
import sys

class BlockExternalPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lattice_symmetry" or fullname.startswith("lattice_symmetry."):
            raise ModuleNotFoundError("external package is unavailable", name="lattice_symmetry")
        return None

sys.meta_path.insert(0, BlockExternalPackage())
from lqcdweave.symmetry import irrep_matrices

rotations = irrep_matrices((0, 0, 0), "G1")
assert rotations
assert all(matrix.shape == (2, 2) for matrix in rotations.values())
assert "lattice_symmetry" not in sys.modules
'''
        )


if __name__ == "__main__":
    unittest.main()
