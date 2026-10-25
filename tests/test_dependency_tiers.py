"""Keep the install tiers and checked-in dependency snapshots honest."""

import json
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

try:
    import tomllib
except ImportError:  # Python 3.10 developer environments receive tomli via pytest.
    import tomli as tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def constraints(name):
    records = {}
    for raw in (ROOT / "constraints" / name).read_text(encoding="utf-8").splitlines():
        text = raw.partition("#")[0].strip()
        if not text:
            continue
        requirement = Requirement(text)
        pins = tuple(requirement.specifier)
        if len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
            raise AssertionError(f"snapshot entry must be one exact version: {text}")
        name = canonicalize_name(requirement.name)
        if name in records:
            raise AssertionError(f"duplicate snapshot entry: {name}")
        records[name] = pins[0].version
    return records


class DependencyTierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        cls.project = cls.metadata["project"]
        cls.extras = cls.project["optional-dependencies"]

    def test_minimal_install_does_not_pull_optional_platforms_or_tools(self):
        names = {canonicalize_name(Requirement(text).name) for text in self.project["dependencies"]}
        self.assertEqual(names, {"numpy", "opt-einsum"})
        self.assertEqual(self.project["requires-python"], ">=3.10")

    def test_cpu_tier_is_exactly_the_existing_cpu_feature_union(self):
        self.assertEqual(set(self.extras["cpu"]), set(self.extras["laplace"] + self.extras["correlator"]))
        for name in ("laplace", "correlator", "cpu", "mpi", "dev", "perf", "perf-hpc"):
            self.assertIn(name, self.extras)
        self.assertEqual({Requirement(text).name for text in self.extras["dev"]}, {"pytest"})
        self.assertEqual({Requirement(text).name for text in self.extras["perf"]}, {"asv", "pyperf"})
        self.assertEqual({Requirement(text).name for text in self.extras["perf-hpc"]}, {"reframe-hpc"})

    def test_correlator_and_cpu_extras_declare_symmetry_runtime_dependencies(self):
        required = {"scipy", "sympy", "lazy-object-proxy"}
        for extra in ("correlator", "cpu"):
            names = {
                canonicalize_name(Requirement(text).name)
                for text in self.extras[extra]
            }
            with self.subTest(extra=extra):
                self.assertTrue(required <= names)
                self.assertNotIn("lattice-symmetry", names)
        proxy = next(
            Requirement(text) for text in self.extras["correlator"]
            if canonicalize_name(Requirement(text).name) == "lazy-object-proxy"
        )
        self.assertEqual(str(proxy.specifier), "")

    def test_linux_cpu_snapshot_satisfies_active_cpu_and_dev_declarations(self):
        pins = constraints("cpu-py312-linux.txt")
        self.assertNotIn("lattice-symmetry", pins)
        environment = {**default_environment(), "python_version": "3.12", "python_full_version": "3.12.14", "sys_platform": "linux", "platform_system": "Linux"}
        for text in self.project["dependencies"] + self.extras["cpu"] + self.extras["dev"]:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            name = canonicalize_name(requirement.name)
            with self.subTest(requirement=text):
                self.assertIn(name, pins)
                self.assertIn(pins[name], requirement.specifier)
        self.assertFalse({"cupy", "pyquda", "mpi4py", "psutil", "asv", "pyperf", "reframe-hpc"} & pins.keys())

    def test_build_and_performance_snapshots_satisfy_their_declarations(self):
        build = constraints("build-py312-linux.txt")
        for text in self.metadata["build-system"]["requires"]:
            requirement = Requirement(text)
            self.assertIn(build[canonicalize_name(requirement.name)], requirement.specifier)
        performance = constraints("perf-py312-linux.txt")
        for text in self.extras["perf"] + self.extras["perf-hpc"]:
            requirement = Requirement(text)
            self.assertIn(performance[canonicalize_name(requirement.name)], requirement.specifier)
        cpu = constraints("cpu-py312-linux.txt")
        for name in cpu.keys() & performance.keys():
            self.assertEqual(cpu[name], performance[name], name)

    def test_ci_keeps_flexible_and_pinned_cpu_lanes(self):
        workflow = (ROOT / ".github/workflows/cpu-tests.yml").read_text(encoding="utf-8")
        self.assertRegex(workflow, r"3\.11")
        self.assertRegex(workflow, r"3\.12")
        self.assertIn("constraints/cpu-py312-linux.txt", workflow)
        self.assertIn("constraints/build-py312-linux.txt", workflow)
        self.assertIn("--no-build-isolation", workflow)
        self.assertIn(".[cpu,dev]", workflow)
        self.assertIn("python -m pip check", workflow)
        for tier in ("cpu-unit", "cpu-integration", "scientific-reference"):
            self.assertIn(f"python tools/test_tiers.py --tier {tier}", workflow)
        self.assertNotIn("python -m unittest discover -s tests -v", workflow)
        self.assertNotIn("--find-links .github/test-wheels", workflow)


if __name__ == "__main__":
    unittest.main()
