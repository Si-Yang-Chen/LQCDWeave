"""Static checks for the module boundaries documented in ARCHITECTURE.md.

These checks deliberately cover imports and a narrow, provable form of private
cross-component access. They do not replace review of numerical conventions.
"""

import ast
import importlib.util
from pathlib import Path
import unittest


PACKAGE = "lqcdweave"
CORE = {"symmetry", "contracts", "sampling", "operators", "production", "contraction",
        "planning", "storage", "runtime"}
FORBIDDEN_BACKENDS = ("pyquda", "pyquda_utils", "mpi4py", "reframe", "slurm")
ALLOWED_INTERNAL = {
    "symmetry": {"symmetry"},
    "contracts": {"contracts"},
    "sampling": {"contracts", "sampling"},
    "operators": {"contracts", "operators"},
    "production": {"contracts", "production"},
    "contraction": {"contracts", "operators", "contraction"},
    "planning": {"contracts", "planning"},
    "storage": {"contracts", "storage"},
    "runtime": {"contracts", "planning", "storage", "runtime"},
}
# Correlator's L0/L1 assigns profile-file IO to M and plan archives to A.
# These are exact public service edges, not blanket contraction/storage access.
# The fixed local-source runtime invokes SS only through its public API.
ALLOWED_MODULE_IMPORTS = {
    # Operators consume the symmetry module through these exact edges.
    ("operators/correlator/group_projection.py", "lqcdweave.symmetry"),
    ("operators/correlator/group_projection.py", "lqcdweave.symmetry.sympy_utils"),
    ("operators/correlator/spatial_structure.py", "lqcdweave.symmetry"),
    ("operators/single_hadron/irrep_coupling.py", "lqcdweave.symmetry"),
    ("operators/single_hadron/insertion.py", "lqcdweave.symmetry"),
    # EV construction delegates only explicit backend wiring to its adapter.
    ("production/eigenvector/generator.py", "lqcdweave.adapters.laplace"),
    # PR owns Gamma5; the old contraction name is a single public alias.
    ("contraction/propagator_views.py", "lqcdweave.production.perambulator.api"),
    ("runtime/memory.py", "lqcdweave.contraction.planning.api"),
    ("contraction/execution/scheduled.py", "lqcdweave.runtime.memory"),
    ("contraction/execution/run_resources.py", "lqcdweave.runtime.memory"),
    ("contraction/two_point.py", "lqcdweave.sampling.temporal.api"),
    ("contraction/performance/profile_io.py", "lqcdweave.storage.scene_cost"),
    ("contraction/performance/profile_io.py", "lqcdweave.runtime.measurement"),
    ("contraction/performance/targets.py", "lqcdweave.storage.dataset"),
    ("storage/contraction_plans/api.py", "lqcdweave.contraction.planning.api"),
    ("storage/contraction_plans/api.py", "lqcdweave.contraction.performance.api"),
    ("storage/contraction_plans/cfg_archive.py", "lqcdweave.contraction.planning.api"),
}

# High-level data services compose existing public APIs. Numerical kernels keep
# the restrictive area rules above; moving the frontend does not authorize new
# dependencies for other production/contraction modules.
ALLOWED_MODULE_IMPORTS |= {
    ('contraction/contraction_session.py', 'lqcdweave.runtime.environment'),
    ('contraction/correlator.py', 'lqcdweave.adapters.execution_target'),
    ('contraction/correlator.py', 'lqcdweave.adapters.hardware_budget'),
    ('contraction/correlator.py', 'lqcdweave.runtime.memory'),
    ('contraction/correlator.py', 'lqcdweave.runtime.scheduler'),
    ('contraction/correlator.py', 'lqcdweave.storage.contraction_plans.api'),
    ('contraction/correlator.py', 'lqcdweave.storage.correlator_results'),
    ('contraction/correlator_cfg.py', 'lqcdweave.adapters.execution_target'),
    ('contraction/correlator_cfg.py', 'lqcdweave.adapters.hardware_budget'),
    ('contraction/correlator_cfg.py', 'lqcdweave.storage.contraction_plans.api'),
    ('contraction/correlator_coordinates.py', 'lqcdweave.adapters.execution_target'),
    ('contraction/correlator_coordinates.py', 'lqcdweave.adapters.hardware_budget'),
    ('contraction/correlator_inputs.py', 'lqcdweave.adapters.preset'),
    ('contraction/correlator_output.py', 'lqcdweave.storage.correlator_results'),
    ('contraction/correlator_postprocessing.py', 'lqcdweave.sampling.temporal.api'),
    ('contraction/correlator_target.py', 'lqcdweave.adapters.execution_target'),
    ('contraction/correlator_target.py', 'lqcdweave.adapters.hardware_budget'),
    ('contraction/correlator_window_output.py', 'lqcdweave.storage.contraction_plans.api'),
    ('contraction/correlator_window_run.py', 'lqcdweave.adapters.execution_target'),
    ('contraction/correlator_window_run.py', 'lqcdweave.adapters.hardware_budget'),
    ('contraction/correlator_window_run.py', 'lqcdweave.storage.contraction_plans.api'),
    ('contraction/correlator_window_run.py', 'lqcdweave.storage.correlator_results'),
    ('contraction/diagram_measurement.py', 'lqcdweave.adapters.baryon_dataset'),
    ('contraction/diagram_measurement.py', 'lqcdweave.adapters.meson_dataset'),
    ('contraction/diagram_measurement.py', 'lqcdweave.planning.dataset'),
    ('contraction/diagram_measurement.py', 'lqcdweave.planning.diagram_measurement'),
    ('contraction/diagram_measurement.py', 'lqcdweave.planning.production'),
    ('contraction/diagram_measurement.py', 'lqcdweave.production.perambulator.highmode_planning'),
    ('contraction/diagram_measurement.py', 'lqcdweave.runtime.datasets'),
    # The single-current expert kernel reuses PR high-point extraction and EL transport.
    ('contraction/external_current.py', 'lqcdweave.production.elemental.transport'),
    ('contraction/external_current.py', 'lqcdweave.production.perambulator.highmode'),
    ('contraction/directed_current_measurement.py', 'lqcdweave.adapters.directed_current_dataset'),
    ('contraction/directed_current_measurement.py', 'lqcdweave.planning.dataset'),
    ('contraction/directed_current_measurement.py', 'lqcdweave.planning.directed_current'),
    ('contraction/directed_current_measurement.py', 'lqcdweave.runtime.datasets'),
    ('contraction/elemental_preparation.py', 'lqcdweave.adapters.dilution_prefix'),
    ('contraction/elemental_preparation.py', 'lqcdweave.adapters.elemental'),
    ('contraction/elemental_preparation.py', 'lqcdweave.adapters.hardware_budget'),
    ('contraction/elemental_preparation.py', 'lqcdweave.adapters.production_dataset'),
    ('contraction/elemental_preparation.py', 'lqcdweave.adapters.selection'),
    ('contraction/elemental_preparation.py', 'lqcdweave.planning.dataset'),
    ('contraction/elemental_preparation.py', 'lqcdweave.planning.production'),
    ('contraction/elemental_preparation.py', 'lqcdweave.production.elemental.api'),
    ('contraction/elemental_preparation.py', 'lqcdweave.production.gauge'),
    ('contraction/elemental_preparation.py', 'lqcdweave.runtime.datasets'),
    ('contraction/elemental_preparation.py', 'lqcdweave.runtime.dispatch'),
    ('contraction/elemental_preparation.py', 'lqcdweave.runtime.provenance'),
    ('contraction/elemental_preparation.py', 'lqcdweave.storage.dataset'),
    ('contraction/meson_measurement.py', 'lqcdweave.adapters.meson_dataset'),
    ('contraction/meson_measurement.py', 'lqcdweave.planning.meson'),
    ('production/eigenvector/planning.py', 'lqcdweave.planning.dataset'),
    ('production/eigenvector/planning.py', 'lqcdweave.planning.laplace'),
    ('production/eigenvector/planning.py', 'lqcdweave.planning.production'),
    ('production/eigenvector/planning.py', 'lqcdweave.runtime.datasets'),
    ('production/elemental/baryon_planning.py', 'lqcdweave.planning.baryon'),
    ('production/elemental/baryon_planning.py', 'lqcdweave.planning.dataset'),
    ('production/elemental/baryon_planning.py', 'lqcdweave.planning.production'),
    ('production/elemental/baryon_planning.py', 'lqcdweave.runtime.datasets'),
    ('production/elemental/planning.py', 'lqcdweave.adapters.production_dataset'),
    ('production/elemental/planning.py', 'lqcdweave.planning.dataset'),
    ('production/elemental/planning.py', 'lqcdweave.planning.production'),
    ('production/elemental/planning.py', 'lqcdweave.runtime.datasets'),
    ('production/elemental/planning.py', 'lqcdweave.runtime.provenance'),
    ('production/overlap_planning.py', 'lqcdweave.planning.dataset'),
    ('production/overlap_planning.py', 'lqcdweave.planning.overlap'),
    ('production/overlap_planning.py', 'lqcdweave.runtime.datasets'),
    ('production/perambulator/highmode_planning.py', 'lqcdweave.planning.dataset'),
    ('production/perambulator/highmode_planning.py', 'lqcdweave.planning.highmode'),
    ('production/perambulator/highmode_planning.py', 'lqcdweave.planning.production'),
    ('production/perambulator/highmode_planning.py', 'lqcdweave.runtime.datasets'),
    ('production/perambulator/planning.py', 'lqcdweave.planning.dataset'),
    ('production/perambulator/planning.py', 'lqcdweave.planning.production'),
    ('production/perambulator/planning.py', 'lqcdweave.runtime.datasets'),
    ('production/spatial_planning.py', 'lqcdweave.planning.dataset'),
    ('production/spatial_planning.py', 'lqcdweave.planning.spatial_points'),
    ('production/spatial_planning.py', 'lqcdweave.runtime.datasets'),
    ('production/wilson_reference.py', 'lqcdweave.adapters.dense_wilson'),
    ('production/wilson_reference.py', 'lqcdweave.adapters.production_dataset'),
    ('production/wilson_reference.py', 'lqcdweave.contraction.meson_measurement'),
    ('production/wilson_reference.py', 'lqcdweave.operators.single_hadron.api'),
    ('production/wilson_reference.py', 'lqcdweave.planning.dataset'),
    ('production/wilson_reference.py', 'lqcdweave.planning.laplace'),
    ('production/wilson_reference.py', 'lqcdweave.planning.meson'),
    ('production/wilson_reference.py', 'lqcdweave.planning.production'),
    ('production/wilson_reference.py', 'lqcdweave.runtime.dataset_stage'),
    ('production/wilson_reference.py', 'lqcdweave.runtime.datasets'),
    ('production/wilson_reference.py', 'lqcdweave.runtime.provenance'),
    ('production/wilson_reference.py', 'lqcdweave.storage.dataset'),
}

# Only these named pure services are admitted; module imports, dynamic imports,
# selection/search functions and private TS imports remain forbidden.
TEMPORAL_REDUCTION_IMPORTS = {
    "storage/correlator_results.py": {"pair_time_differences", "count_time_pairs", "mean_time_pairs"},
    "storage/_cfg_correlator_output.py": {"TimePairAccumulator"},
}


def _resolved_import(node, module):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    package = module.removesuffix(".__init__") if module.endswith(".__init__") else module.rpartition(".")[0]
    parts = package.split(".")
    if node.level:
        parts = parts[:len(parts) - node.level + 1]
        prefix = ".".join(parts)
        target = ".".join(filter(None, (prefix, node.module)))
    else:
        target = node.module or ""
    return [target]



def _component(module):
    """Use the documented scientific responsibility, not each Python file."""
    components = (
        "symmetry",
        "contraction.binding", "contraction.expansion", "contraction.planning",
        "contraction.execution", "contraction.results", "contraction.performance",
        "production.eigenvector", "production.noisevector", "production.perambulator",
        "production.elemental", "sampling.temporal", "sampling.spatial",
        "operators.correlator", "operators.single_hadron", "storage.contraction_plans",
    )
    for component in components:
        prefix = PACKAGE + "." + component
        if module == prefix or module.startswith(prefix + "."):
            return prefix
    return module

def _private_accesses(tree, imported_classes):
    """Find private reads on objects constructed from another module's class."""
    instances = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if not isinstance(value.func, ast.Name) or value.func.id not in imported_classes:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                instances.add(target.id)
            elif (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                  and target.value.id == "self"):
                instances.add("self." + target.attr)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not node.attr.startswith("_"):
            continue
        owner = node.value
        if isinstance(owner, ast.Name) and owner.id in instances:
            yield node.lineno, owner.id + "." + node.attr
        elif (isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name)
              and owner.value.id == "self" and "self." + owner.attr in instances):
            yield node.lineno, "self." + owner.attr + "." + node.attr


def architecture_violations(sources):
    """Return reproducible diagnostics for {package-relative path: source}."""
    errors = []
    for path, source in sorted(sources.items()):
        module = PACKAGE + "." + path.replace("\\", "/").removesuffix(".py").replace("/", ".")
        area = path.replace("\\", "/").split("/")[0]
        tree = ast.parse(source, filename=path)
        imported_classes = set()
        importlib_aliases = {"importlib"}
        import_module_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                importlib_aliases.update(alias.asname or alias.name for alias in node.names
                                         if alias.name == "importlib")
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
                import_module_aliases.update(alias.asname or alias.name for alias in node.names
                                            if alias.name == "import_module")
        def check_target(target, line, dynamic=False):
            if target.startswith(PACKAGE + ".") and area in ALLOWED_INTERNAL:
                imported_area = target.split(".")[1]
                if (imported_area not in ALLOWED_INTERNAL[area]
                        and (path.replace("\\", "/"), target) not in ALLOWED_MODULE_IMPORTS):
                    errors.append(f"{path}:{line}: {area} imports {target}")
            if area in CORE and any(target == name or target.startswith(name + ".")
                                    for name in FORBIDDEN_BACKENDS):
                qualifier = "dynamically imports" if dynamic else "imports"
                errors.append(f"{path}:{line}: core {qualifier} backend {target}")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _resolved_import(node, module):
                if (target == "lqcdweave.sampling.temporal.api"
                        and isinstance(node, ast.ImportFrom)
                        and path.replace("\\", "/") in TEMPORAL_REDUCTION_IMPORTS
                        and {alias.name for alias in node.names} <=
                            TEMPORAL_REDUCTION_IMPORTS[path.replace("\\", "/")]):
                    continue
                check_target(target, node.lineno)
            if isinstance(node, ast.ImportFrom):
                target = _resolved_import(node, module)[0]
                if target == PACKAGE:
                    for alias in node.names:
                        check_target(PACKAGE + "." + alias.name, node.lineno)
                if (target.startswith(PACKAGE + ".")
                        and _component(target) != _component(module)):
                    imported_classes.update(alias.asname or alias.name for alias in node.names
                                            if alias.name[:1].isupper())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            module_import = (isinstance(function, ast.Name) and function.id in import_module_aliases) or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
                and isinstance(function.value, ast.Name) and function.value.id in importlib_aliases)
            dynamic_import = module_import or (isinstance(function, ast.Name) and function.id == "__import__")
            if dynamic_import and node.args and isinstance(node.args[0], ast.Constant):
                target = node.args[0].value
                if isinstance(target, str):
                    if module_import and target.startswith("."):
                        package = module.removesuffix(".__init__") if module.endswith(".__init__") else module.rpartition(".")[0]
                        package_arg = next((keyword.value for keyword in node.keywords if keyword.arg == "package"), None)
                        if len(node.args) > 1:
                            package_arg = node.args[1]
                        if isinstance(package_arg, ast.Constant) and isinstance(package_arg.value, str):
                            package = package_arg.value
                        elif not isinstance(package_arg, ast.Name) or package_arg.id != "__package__":
                            continue
                        target = importlib.util.resolve_name(target, package)
                    check_target(target, node.lineno, dynamic=True)
        for line, access in _private_accesses(tree, imported_classes):
            errors.append(f"{path}:{line}: private cross-component access {access}")
    return errors


class ArchitectureTests(unittest.TestCase):
    def test_result_reduction_service_edges_are_narrow(self):
        sources = {
            "storage/correlator_results.py": (
                "from lqcdweave.sampling.temporal.api import mean_time_pairs\n"
                "from lqcdweave.sampling.temporal.reconstruction import mean_time_pairs\n"
                "from lqcdweave.sampling.spatial.api import sample_spatial_points\n"
                "from lqcdweave.sampling.temporal.api import select_t\n"
                "import lqcdweave.sampling.temporal.api as temporal\n"
                "import importlib; importlib.import_module('lqcdweave.sampling.temporal.api')\n"
            ),
            "storage/_cfg_correlator_output.py": (
                "from lqcdweave.sampling.temporal.api import TimePairAccumulator\n"
            ),
            "storage/other.py": (
                "from lqcdweave.sampling.temporal.api import mean_time_pairs\n"
            ),
        }
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 6, errors)
        self.assertTrue(any("temporal.reconstruction" in error for error in errors))
        self.assertTrue(any("sampling.spatial" in error for error in errors))
        self.assertTrue(any("storage/other.py" in error for error in errors))

    def test_memory_resource_service_edges_are_narrow(self):
        sources = {
            "runtime/memory.py": (
                "import lqcdweave.contraction.planning.api as planning_api\n"
                "from lqcdweave.contraction.planning.memory import resolve_budget_limits\n"
            ),
            "runtime/other.py": (
                "import lqcdweave.contraction.planning.api as planning_api\n"
            ),
            "contraction/execution/scheduled.py": (
                "from lqcdweave.runtime.memory import fresh_execution_limits\n"
            ),
            "contraction/execution/run_resources.py": (
                "from lqcdweave.runtime.memory import reusable_device_memory\n"
            ),
            "contraction/execution/other.py": (
                "from lqcdweave.runtime.memory import fresh_execution_limits\n"
            ),
        }
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 3, errors)
        self.assertTrue(any("planning.memory" in error for error in errors))
        self.assertTrue(any("runtime/other.py" in error for error in errors))
        self.assertTrue(any("execution/other.py" in error for error in errors))

    def test_runtime_has_no_builtin_sampler_service_edge(self):
        sources = {
            "runtime/local_sources.py": (
                "from lqcdweave.sampling.spatial.api import sample_spatial_points\n"
                "from lqcdweave.sampling.spatial.points import sample_spatial_points\n"
                "from lqcdweave.sampling.spatial import sample_spatial_points\n"
            ),
            "runtime/other.py": (
                "from lqcdweave.sampling.spatial.api import sample_spatial_points\n"
            ),
        }
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 4, errors)
        self.assertTrue(any("spatial.points" in error for error in errors))
        self.assertTrue(any("runtime/other.py" in error for error in errors))

    def test_source_boundaries(self):
        root = Path(__file__).resolve().parents[1] / "src" / PACKAGE
        sources = {str(path.relative_to(root)): path.read_text(encoding="utf-8")
                   for path in root.rglob("*.py")}
        self.assertEqual(architecture_violations(sources), [])

    def test_illegal_imports_are_detected(self):
        sources = {
            "contracts/bad.py": "from ..storage.dataset import DatasetReader\n",
            "operators/bad.py": "from lqcdweave.production import PerambulatorGenerator\n",
            "production/bad.py": "import pyquda\n",
            "production/dilution_bad.py": "from lqcdweave.sampling.mode_weights import dilution_pair_weights\n",
            "contraction/bad.py": "import importlib\nimportlib.import_module('mpi4py.MPI')\n",
            "contracts/root_bad.py": "from lqcdweave import storage\n",
        }
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 6, errors)
        self.assertTrue(any("contracts imports" in error for error in errors))
        self.assertTrue(any("operators imports" in error for error in errors))
        self.assertTrue(any("core imports backend" in error for error in errors))
        self.assertTrue(any("dynamically imports backend" in error for error in errors))
        self.assertTrue(any("production imports lqcdweave.sampling.mode_weights" in error for error in errors))

    def test_dynamic_import_aliases_and_relative_packages_are_detected(self):
        sources = {
            "contracts/__init__.py":
                "from importlib import import_module as load\n"
                "load('..storage.dataset', package=__package__)\n",
            "production/bad.py":
                "import importlib as loader\nloader.import_module('pyquda')\n",
            "operators/bad.py":
                "from importlib import import_module\n"
                "import_module('..production', 'lqcdweave.operators')\n",
            "contracts/member.py": "from .. import storage\n",
        }
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 4, errors)
        self.assertTrue(any("contracts imports lqcdweave.storage.dataset" in error for error in errors))
        self.assertTrue(any("operators imports lqcdweave.production" in error for error in errors))
        self.assertTrue(any("dynamically imports backend pyquda" in error for error in errors))

    def test_correlator_service_edges_do_not_allow_internal_planner_access(self):
        sources = {
            "storage/contraction_plans/api.py":
                "from lqcdweave.contraction.planning.api import rebind_scene_path\n"
                "from lqcdweave.contraction.planning.state import ContractionState\n",
            "contraction/performance/profile_io.py":
                "from lqcdweave.runtime.measurement import load_measurement\n"
                "from lqcdweave.runtime.scheduler import execute_request\n",
        }
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(any("planning.state" in error for error in errors))
        self.assertTrue(any("runtime.scheduler" in error for error in errors))

    def test_private_access_respects_component_ownership(self):
        sources = {
            "contraction/planning/verification.py": (
                "from .optimizers.integer_space import IntegerSpace\n"
                "space = IntegerSpace()\n"
                "snapshot = space._group_weights\n"
            ),
            "contraction/execution/device.py": (
                "from .reduction_workspace import ReductionWorkspace\n"
                "workspace = ReductionWorkspace()\n"
                "stream = workspace._current()\n"
            ),
            "contraction/execution/foreign.py": (
                "from lqcdweave.contraction.planning.api import ContractionState\n"
                "state = ContractionState()\n"
                "snapshot = state._private_table\n"
            ),
        }
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("private cross-component access state._private_table", errors[0])

    def test_other_producers_private_state_is_detected(self):
        sources = {"production/bad.py": (
            "from .elemental.generator import ElementalGenerator\n"
            "class Baryon:\n"
            "    def __init__(self):\n"
            "        self.meson = ElementalGenerator()\n"
            "        self.own = None\n"
            "    def run(self):\n"
            "        return self.meson._workspace, self.own\n")}
        errors = architecture_violations(sources)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("private cross-component access self.meson._workspace", errors[0])


if __name__ == "__main__":
    unittest.main()
