"""Data handoff boundaries of temporal/spatial and basic production modules."""
import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "src" / "lqcdweave"
MODULES = {
    "TS": "sampling/temporal", "SS": "sampling/spatial",
    "EV": "production/eigenvector", "EL": "production/elemental",
    "NV": "production/noisevector",
    "PR": "production/perambulator",
}


class ProductionModuleBoundaryTests(unittest.TestCase):
    def test_modules_consume_contracts_not_other_generators(self):
        errors = []
        prefixes = {key: "lqcdweave." + folder.replace("/", ".")
                    for key, folder in MODULES.items()}
        legacy = {
            "lqcdweave.sampling.select_t": "TS",
            "lqcdweave.sampling.time_rules": "TS",
            "lqcdweave.sampling.time_objectives": "TS",
            "lqcdweave.contraction.two_point": "TS",
        }
        for owner, folder in MODULES.items():
            directory = ROOT / folder
            self.assertTrue((directory / "api.py").exists(), f"missing public API: {folder}")
            for path in directory.rglob("*.py"):
                module = "lqcdweave." + ".".join(path.relative_to(ROOT).with_suffix("").parts)
                package = module.rpartition(".")[0]
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                    if isinstance(node, ast.ImportFrom):
                        target = node.module or ""
                        if node.level:
                            target = importlib.util.resolve_name("." * node.level + target, package)
                        targets = [target]
                    elif isinstance(node, ast.Import):
                        targets = [alias.name for alias in node.names]
                    else:
                        continue
                    for target in targets:
                        for other, prefix in prefixes.items():
                            if other != owner and (target == prefix or target.startswith(prefix + ".")):
                                errors.append(f"{module}:{node.lineno}: imports other producer/sampler {other}")
                        if target in legacy and legacy[target] != owner:
                            errors.append(f"{module}:{node.lineno}: legacy import bypasses handoff {target}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_public_compatibility_types_remain_unique(self):
        from lqcdweave.contracts.time_selection import TimeSelection
        from lqcdweave.sampling.temporal.api import TimeSelection as TemporalSelection
        from lqcdweave.sampling.spatial.api import SpatialPointSet
        from lqcdweave.contracts.spatial_sampling import SpatialPointSet as PointContract
        from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
        from lqcdweave.production import LaplaceEigenvectorGenerator as PublicEigenvector
        from lqcdweave.production.perambulator.api import PerambulatorGenerator
        from lqcdweave.production import PerambulatorGenerator as PublicPerambulator
        self.assertIs(TimeSelection, TemporalSelection)
        self.assertIs(SpatialPointSet, PointContract)
        self.assertIs(LaplaceEigenvectorGenerator, PublicEigenvector)
        self.assertIs(PerambulatorGenerator, PublicPerambulator)


if __name__ == "__main__":
    unittest.main()
