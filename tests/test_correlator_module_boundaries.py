"""Architecture gates for the correlator module handoff boundaries."""

import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "src" / "lqcdweave"
FOLDERS = {
    "S": "operators/correlator", "O": "operators/single_hadron",
    "B": "contraction/binding", "E": "contraction/expansion",
    "P": "contraction/planning", "X": "contraction/execution",
    "R": "contraction/results", "M": "contraction/performance",
    "A": "storage/contraction_plans", "C": "contracts/correlator",
    "W": "contraction",
}
# W is a logical frontend role inside contraction, not a separate package.
FRONTEND_MODULES = {
    "lqcdweave.contraction._cfg_plan_worker",
    "lqcdweave.contraction.calculation",
    "lqcdweave.contraction.calculation_compat",
    "lqcdweave.contraction.contraction_session",
    "lqcdweave.contraction.correlator",
    "lqcdweave.contraction.correlator_batch_output",
    "lqcdweave.contraction.correlator_cfg",
    "lqcdweave.contraction.correlator_coordinates",
    "lqcdweave.contraction.correlator_dt_output",
    "lqcdweave.contraction.correlator_inputs",
    "lqcdweave.contraction.correlator_output",
    "lqcdweave.contraction.correlator_postprocessing",
    "lqcdweave.contraction.correlator_replay",
    "lqcdweave.contraction.correlator_result_metadata",
    "lqcdweave.contraction.correlator_stream",
    "lqcdweave.contraction.correlator_target",
    "lqcdweave.contraction.correlator_window_output",
    "lqcdweave.contraction.correlator_window_run",
    "lqcdweave.contraction.diagram_measurement",
    "lqcdweave.contraction.directed_current_measurement",
    "lqcdweave.contraction.elemental_preparation",
    "lqcdweave.contraction.external_current_measurement",
    "lqcdweave.contraction.meson_measurement",
    "lqcdweave.contraction.replay",
}
ALLOWED = {
    "S": {"C"}, "O": {"C"}, "B": {"S", "O", "C"},
    "E": {"S", "B", "R", "C"}, "P": {"C", "M"},
    "X": {"C", "P", "M"}, "R": {"C", "S"}, "M": {"C"},
    "A": {"C", "P", "M"}, "C": set(),
    "W": {"S", "O", "B", "E", "P", "X", "R", "M", "A", "C"},
}


def _module_and_package(path, root, prefix):
    relative = path.relative_to(root).with_suffix("").parts
    is_package = relative[-1] == "__init__"
    if is_package:
        relative = relative[:-1]
    module = ".".join((prefix, *relative))
    package = module if is_package else module.rpartition(".")[0]
    return module, package


def _import_targets(tree, package):
    """Yield statically named and literal dynamic imports from one module."""
    importlib_names = {"importlib"}
    import_module_names = {"import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
            else:
                base = node.module or ""
            if base:
                yield node, base
            for alias in node.names:
                if alias.name != "*":
                    yield node, base + "." + alias.name if base else alias.name
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            is_builtin_import = (
                isinstance(function, ast.Name) and function.id == "__import__"
            )
            is_import_module = (
                isinstance(function, ast.Name) and function.id in import_module_names
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "import_module"
                and isinstance(function.value, ast.Name)
                and function.value.id in importlib_names
            )
            if not (is_builtin_import or is_import_module):
                continue
            target = node.args[0]
            if not isinstance(target, ast.Constant) or not isinstance(target.value, str):
                continue
            name = target.value
            if is_import_module and name.startswith("."):
                explicit_package = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "package"),
                    None,
                )
                if isinstance(explicit_package, ast.Constant) and isinstance(explicit_package.value, str):
                    dynamic_package = explicit_package.value
                else:
                    dynamic_package = package
                try:
                    name = importlib.util.resolve_name(name, dynamic_package)
                except (ImportError, ValueError):
                    continue
            yield node, name




def _is_module_import(target):
    if not target.startswith("lqcdweave."):
        return False
    path = ROOT
    for part in target.removeprefix("lqcdweave.").split("."):
        path = path / part
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def owner(module):
    for key, folder in FOLDERS.items():
        if key == "W" and module not in FRONTEND_MODULES:
            continue
        prefix = "lqcdweave." + folder.replace("/", ".")
        if module == prefix or module.startswith(prefix + "."):
            return key, prefix
    return None, None


def _capacity_probes(tree, relative_path):
    """Distinguish owned idle-pool accounting from runtime capacity probes."""
    owned_reads = set()
    if relative_path == "contraction/execution/run_resources.py":
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name != "RunResources":
                continue
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef) or method.name != "reusable_bytes":
                    continue
                for node in ast.walk(method):
                    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                        continue
                    pool = node.func.value
                    if (node.func.attr == "free_bytes" and not node.args and not node.keywords
                            and isinstance(pool, ast.Attribute) and pool.attr == "_pool"
                            and isinstance(pool.value, ast.Name) and pool.value.id == "self"):
                        owned_reads.add(id(node))
    return [(node.lineno, node.func.attr) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"memGetInfo", "free_bytes"} and id(node) not in owned_reads]


class CorrelatorBoundaryTests(unittest.TestCase):


    def test_cross_module_imports_use_public_interfaces_and_form_a_dag(self):
        violations, edges = [], {key: set() for key in FOLDERS}
        for source, folder in FOLDERS.items():
            directory = ROOT / folder
            self.assertTrue(directory.is_dir(), f"missing module {source}: {folder}")
            for path in directory.rglob("*.py"):
                module, package = _module_and_package(path, ROOT, "lqcdweave")
                if owner(module)[0] != source:
                    continue  # Submodules keep B/E/P/X/R/M ownership inside contraction.
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                for node, imported in _import_targets(tree, package):
                    names = tuple(alias.name for alias in node.names) if isinstance(node, ast.ImportFrom) else ()
                    if isinstance(node, ast.ImportFrom) and not _is_module_import(imported):
                        continue
                    if isinstance(node, ast.Import):
                        target, _ = owner(imported)
                        if target and target != source:
                            violations.append(f"{module}:{node.lineno}: module import {imported}; use api")
                        continue
                    if source == "C" and imported.startswith("lqcdweave.") and not imported.startswith("lqcdweave.contracts."):
                        violations.append(f"{module}:{node.lineno}: contract imports implementation {imported}")
                    target, prefix = owner(imported)
                    if not target or target == source:
                        continue
                    edges[source].add(target)
                    if target not in ALLOWED[source]:
                        violations.append(f"{module}:{node.lineno}: forbidden {source}->{target}")
                    if target != "C" and imported != prefix + ".api":
                        violations.append(f"{module}:{node.lineno}: internal import {imported}")
                    if any(name.startswith("_") for name in names):
                        violations.append(f"{module}:{node.lineno}: cross-module private import {names}")
        self.assertEqual(violations, [], "\n".join(violations))
        active, done = set(), set()
        def visit(key):
            self.assertNotIn(key, active, f"dependency cycle at {key}")
            if key in done:
                return
            active.add(key)
            for target in edges[key]:
                visit(target)
            active.remove(key)
            done.add(key)
        for key in edges:
            visit(key)

    def test_execution_does_not_probe_driver_or_pool_capacity(self):
        # Only the resource owner's idle pool is credited through the runtime
        # provider. Global/default-pool and driver capacity remain runtime-owned.
        errors = []
        for path in (ROOT / "contraction" / "execution").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for line, method in _capacity_probes(tree, path.relative_to(ROOT).as_posix()):
                errors.append(f"{path}:{line}: direct capacity probe {method}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_capacity_probe_rule_distinguishes_owned_pool_accounting(self):
        source = """class RunResources:
    def reusable_bytes(self, backend):
        self._pool.free_bytes()
        backend.get_default_memory_pool().free_bytes()
        backend.cuda.runtime.memGetInfo()
    def other(self):
        self._pool.free_bytes()
"""
        tree = ast.parse(source)
        self.assertEqual(_capacity_probes(tree, "contraction/execution/run_resources.py"),
                         [(4, "free_bytes"), (5, "memGetInfo"), (7, "free_bytes")])
        self.assertEqual(len(_capacity_probes(tree, "contraction/execution/other.py")), 4)
        self.assertEqual(len(_capacity_probes(ast.parse(source.replace("RunResources", "Other")),
                         "contraction/execution/run_resources.py")), 4)

    def test_legacy_contract_exports_keep_the_same_type_identity(self):
        from lqcdweave.contracts.correlator import api
        from lqcdweave import contraction
        from lqcdweave.contraction.expansion import api as expansion_api
        from lqcdweave.contraction.planning import api as planning_api
        from lqcdweave.contraction.results import api as results_api
        for legacy, names in (
            (contraction, ("TensorSpec", "ContractionTerm", "ContractionPlan")),
        ):
            for name in names:
                self.assertIs(getattr(legacy, name), getattr(api, name), name)
        self.assertIs(expansion_api.ExpandedCorrelator, api.ExpandedCorrelator)
        self.assertIs(planning_api.PlannedContraction, api.PlannedContraction)
        self.assertIs(results_api.OutputNode, api.OutputNode)
        self.assertIs(results_api.OutputRecipe, api.OutputRecipe)

    def test_spatial_baryon_component_identity_is_flavor_independent(self):
        from lqcdweave.contracts.elemental import BaryonComponent
        canonical = BaryonComponent.paths()
        legacy = BaryonComponent.paths(flavors=("u", "d", "u"))
        self.assertEqual(canonical, legacy)
        self.assertEqual(hash(canonical), hash(legacy))


if __name__ == "__main__":
    unittest.main()
