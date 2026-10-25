"""Acceptance checks record path inventories and runtime settings."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile


path = Path(__file__).resolve().parents[1] / "tools/acceptance.py"
spec = importlib.util.spec_from_file_location("acceptance_tool", path)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


class AcceptanceEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        (self.repo / "src/lqcdweave").mkdir(parents=True)
        self.code = self.repo / "src/lqcdweave/example.py"
        self.code.write_text("value = 1\n")
        self.resource = self.repo / "src/lqcdweave/example.cu"
        self.resource.write_text("kernel\n")
        self.readme = self.repo / "README.md"
        self.readme.write_text("Build metadata\n")
        self.profile = self.root / "dependency"
        self.profile.mkdir()
        (self.profile / "library.py").write_text("version = 1\n")

    def freeze(self):
        with patch.object(acceptance, "runtime_profile", side_effect=lambda roots: {
                "dependencies": {str(path):acceptance.tree_paths(path) for path in roots}}):
            return dict(dependency_roots=[str(self.profile)],
                        snapshot=acceptance.snapshot(self.repo, [self.profile]))

    def check(self, frozen):
        with patch.object(acceptance, "runtime_profile", side_effect=lambda roots: {
                "dependencies": {str(path):acceptance.tree_paths(path) for path in roots}}):
            return acceptance.verify_snapshot(self.repo, frozen)

    def test_content_edits_do_not_change_path_inventory_but_added_paths_do(self):
        frozen = self.freeze()
        self.check(frozen)
        for path in (self.code, self.resource, self.readme, self.profile / "library.py"):
            with self.subTest(path=path.name):
                old = path.read_bytes()
                path.write_bytes(old + b"changed\n")
                self.check(frozen)
                added = self.code.parent / (path.stem + ".extra.py")
                added.write_bytes(b"new path\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.check(frozen)
                added.unlink()
                path.write_bytes(old)
                self.check(frozen)

    def test_docs_and_generated_python_cache_do_not_invalidate_cpu_evidence(self):
        frozen = self.freeze()
        (self.repo / "TODO.md").write_text("updated acceptance counts\n")
        cache = self.code.parent / "__pycache__"
        cache.mkdir()
        (cache / "example.pyc").write_bytes(b"cache")
        self.check(frozen)

    def test_evidence_is_plain_json_and_missing_dependencies_are_rejected(self):
        record = self.root / "evidence.json"
        acceptance.write_record(record, {"status":"passed"})
        payload = json.loads(record.read_text())
        payload["status"] = "failed"
        record.write_text(json.dumps(payload))
        self.assertEqual(acceptance.read_record(record)["status"], "failed")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            acceptance.tree_paths(self.root / "missing")

    def test_actual_test_process_is_not_reused(self):
        (self.code.parent / "__init__.py").write_text("version = 1\n")
        tests = self.repo / "tests"
        tests.mkdir()
        (tests / "test_example.py").write_text(
            "import unittest\nclass Example(unittest.TestCase):\n"
            "    def test_value(self):\n        self.assertEqual(1 + 1, 2)\n")
        frozen = self.freeze()
        phase = self.root / "phase"
        phase.mkdir()
        package = {"installed": str(self.root / "installed")}
        with patch.object(acceptance, "runtime_profile", side_effect=lambda roots: {
                "dependencies": {str(path):acceptance.tree_paths(path) for path in roots}}):
            first = acceptance.run_suite(self.repo, phase, frozen, "source", package)
            self.assertEqual(first["exit_code"], 0)
            self.assertIn("Ran 1 test", Path(first["log"]).read_text())
            self.assertIn("Package origins verified: 1", Path(first["log"]).read_text())
            self.assertFalse((phase / "source-unfinished.json").exists())
            with patch.object(acceptance.subprocess, "run", side_effect=AssertionError("must not restart")):
                with self.assertRaisesRegex(ValueError, "result already exists"):
                    acceptance.run_suite(self.repo, phase, frozen, "source", package)

    def test_wheel_requires_expected_source_resource_paths(self):
        wheel = self.root / "package.whl"
        installed = self.root / "installed"
        installed.mkdir()
        with zipfile.ZipFile(wheel, "w") as archive:
            for file in (self.code, self.resource):
                name = file.relative_to(self.repo / "src").as_posix()
                archive.write(file, name)
                target = installed / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(file.read_bytes())
        record = acceptance.verify_wheel(self.repo, wheel, installed, [])
        self.assertEqual(len(record["package_files"]), 2)
        with self.assertRaisesRegex(ValueError, "retired"):
            acceptance.verify_wheel(self.repo, wheel, installed, ["lqcdweave.example"])
        (installed / "lqcdweave/example.cu").write_text("wrong")
        self.assertEqual(acceptance.verify_wheel(self.repo, wheel, installed, []), record)
        (installed / "lqcdweave/unexpected.py").write_bytes(b"unexpected = 1\n")
        with self.assertRaisesRegex(ValueError, "installed"):
            acceptance.verify_wheel(self.repo, wheel, installed, [])

    def test_wheel_member_and_metadata_paths_are_checked(self):
        wheel = self.root / "package.whl"
        installed = self.root / "installed"
        installed.mkdir()
        with zipfile.ZipFile(wheel, "w") as archive:
            for file in (self.code, self.resource):
                name = file.relative_to(self.repo / "src").as_posix()
                archive.write(file, name)
                target = installed / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(file.read_bytes())
            archive.writestr("lqcdweave-0.1.dist-info/METADATA", "Name: lqcdweave\n")
        metadata = installed / "lqcdweave-0.1.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_bytes(b"Name: lqcdweave\n")
        package = acceptance.verify_wheel(self.repo, wheel, installed, [])
        acceptance.verify_package(self.repo, package, [])
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("lqcdweave-0.1.dist-info/NOTICE", "changed metadata")
        with self.assertRaisesRegex(ValueError, "installed"):
            acceptance.verify_package(self.repo, package, [])
        (installed / "lqcdweave-0.1.dist-info/NOTICE").write_text(
            "different installed metadata", encoding="utf-8")
        package = acceptance.verify_wheel(self.repo, wheel, installed, [])
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("unexpected.py", "raise RuntimeError()")
        with self.assertRaisesRegex(ValueError, "unexpected wheel"):
            acceptance.verify_wheel(self.repo, wheel, installed, [])

    def test_full_installed_entry_rejects_environment_test_filter(self):
        for name, value in (("PYTEST_ADDOPTS", "-k one_test"),
                            ("PYTEST_PLUGINS", "filter_plugin")):
            with self.subTest(name=name), patch.dict(os.environ, {
                    "PYTEST_ADDOPTS":"", "PYTEST_PLUGINS":"", name:value}):
                with self.assertRaisesRegex(ValueError, name):
                    acceptance.run_suite(self.repo, self.root, {}, "installed", {})

    def test_actual_scoped_commit_handles_deleted_code_and_preserves_other_work(self):
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()
        git("init", "-q")
        git("config", "user.name", "Acceptance fixture")
        git("config", "user.email", "acceptance@example.invalid")
        git("config", "core.autocrlf", "false")
        git("config", "commit.gpgsign", "false")
        unrelated = self.repo / "notes.md"
        unrelated.write_text("old notes\n")
        git("add", ".")
        git("commit", "-q", "-m", "Fixture baseline")
        with self.assertRaisesRegex(ValueError, "explicit regular-file"):
            acceptance.commit_paths(self.repo, self.root, {}, ["."], "Reject broad scope")
        self.assertEqual(git("diff", "--cached", "--name-only"), "")
        self.code.unlink()
        replacement = self.code.with_name("replacement.py")
        replacement.write_bytes(b"value = 2\n")
        unrelated.write_text("user work to preserve\n")
        paths = sorted([self.code.relative_to(self.repo).as_posix(),
                        replacement.relative_to(self.repo).as_posix()])
        phase = self.root / "phase"
        phase.mkdir()
        frozen = self.freeze()
        with patch.object(acceptance, "runtime_profile", side_effect=lambda roots: {
                "dependencies": {str(path):acceptance.tree_paths(path) for path in roots}}):
            head = acceptance.commit_paths(self.repo, phase, frozen, paths, "Replace retired code")
        record = acceptance.read_record(phase / "commit.json")
        self.assertEqual(record["commit"], head)
        self.assertEqual(record["files"], paths)
        self.assertEqual(git("status", "--porcelain"), "M notes.md")
        self.assertEqual(unrelated.read_text(), "user work to preserve\n")


if __name__ == "__main__":
    unittest.main()
