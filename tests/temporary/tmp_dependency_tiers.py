"""Temporary (slow, about 204 s on CSNS CPU) case moved from ``tests/test_dependency_tiers.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from test_dependency_tiers import DependencyTierTests
from test_dependency_tiers import Path, ROOT, json, shutil, subprocess, sys, tarfile, tempfile, zipfile  # noqa: F401


class TemporaryDependencyTierTests(DependencyTierTests):
    def test_wheel_and_sdist_include_all_symmetry_modules_and_mit_license(self):
        symmetry = ROOT / "src/lqcdweave/symmetry"
        source_files = {
            path.relative_to(symmetry).as_posix(): path.read_bytes()
            for path in symmetry.rglob("*")
            if path.is_file() and path.suffix == ".py"
        }
        module_names = {name for name in source_files if name.endswith(".py")}
        self.assertEqual(len(module_names), 15)
        self.assertIn("api.py", source_files)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            project = workspace / "project"
            (project / "src/lqcdweave").mkdir(parents=True)
            shutil.copy2(ROOT / "pyproject.toml", project / "pyproject.toml")
            shutil.copy2(ROOT / "README.md", project / "README.md")
            shutil.copy2(ROOT / "LICENSE", project / "LICENSE")
            shutil.copy2(
                ROOT / "src/lqcdweave/__init__.py",
                project / "src/lqcdweave/__init__.py",
            )
            shutil.copytree(
                ROOT / "src/lqcdweave/symmetry",
                project / "src/lqcdweave/symmetry",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

            wheel_dir = workspace / "wheel"
            wheel_dir.mkdir()
            sdist_dir = workspace / "sdist"
            sdist_dir.mkdir()
            build_script = """
import json
from pathlib import Path
import sys
from setuptools.build_meta import build_sdist, build_wheel

kind, destination, manifest = sys.argv[1:]
build = build_wheel if kind == "wheel" else build_sdist
name = build(destination)
Path(manifest).write_text(json.dumps({"name": name}), encoding="utf-8")
"""

            def build_artifact(kind, destination):
                manifest = workspace / f"{kind}-build.json"
                result = subprocess.run(
                    [sys.executable, "-c", build_script, kind,
                     str(destination), str(manifest)],
                    cwd=project, capture_output=True, text=True, check=False,
                )
                files = sorted(path.name for path in destination.iterdir())
                diagnostics = (
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                    f"files in {destination}: {files}"
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"{kind} build failed\n{diagnostics}",
                )
                self.assertTrue(
                    manifest.is_file(),
                    f"{kind} build did not write its artifact manifest\n{diagnostics}",
                )
                try:
                    name = json.loads(manifest.read_text(encoding="utf-8"))["name"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    self.fail(
                        f"{kind} build wrote an invalid artifact manifest: {exc}\n"
                        f"{diagnostics}"
                    )
                self.assertIsInstance(name, str, f"invalid {kind} artifact name\n{diagnostics}")
                artifact = destination / name
                self.assertTrue(
                    artifact.is_file(),
                    f"{kind} build returned missing artifact {name!r}\n{diagnostics}",
                )
                return artifact

            wheel = build_artifact("wheel", wheel_dir)
            with zipfile.ZipFile(wheel) as archive:
                license_name = next(
                    name for name in archive.namelist()
                    if name.endswith(".dist-info/licenses/LICENSE")
                )
                self.assertEqual(archive.read(license_name), (ROOT / "LICENSE").read_bytes())
                metadata_name = next(
                    name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
                )
                metadata = archive.read(metadata_name).decode("utf-8")
                self.assertIn("License-Expression: MIT", metadata.splitlines())
                self.assertIn("License-File: LICENSE", metadata.splitlines())
                for name, expected in source_files.items():
                    archive_name = f"lqcdweave/symmetry/{name}"
                    with self.subTest(distribution="wheel", file=name):
                        self.assertEqual(archive.read(archive_name), expected)

            sdist = build_artifact("sdist", sdist_dir)
            symmetry_path = "src/lqcdweave/symmetry/"

            def relative_symmetry_path(path):
                normalized = path.replace("\\", "/").lstrip("./")
                if symmetry_path not in normalized:
                    return None
                return normalized.split(symmetry_path, 1)[1]

            if zipfile.is_zipfile(sdist):
                with zipfile.ZipFile(sdist) as archive:
                    root_license = next(
                        name for name in archive.namelist()
                        if name.count("/") == 1 and name.endswith("/LICENSE")
                    )
                    self.assertEqual(archive.read(root_license), (ROOT / "LICENSE").read_bytes())
                    members = {
                        relative: path
                        for path in archive.namelist()
                        if (relative := relative_symmetry_path(path)) is not None
                    }
                    read_member = archive.read
                    for name, expected in source_files.items():
                        with self.subTest(distribution="sdist", file=name):
                            self.assertIn(name, members)
                            self.assertEqual(read_member(members[name]), expected)
            elif tarfile.is_tarfile(sdist):
                with tarfile.open(sdist, "r:*") as archive:
                    root_license = next(
                        member for member in archive.getmembers()
                        if member.isfile() and member.name.count("/") == 1
                        and member.name.endswith("/LICENSE")
                    )
                    self.assertEqual(
                        archive.extractfile(root_license).read(), (ROOT / "LICENSE").read_bytes()
                    )
                    members = {
                        relative: member
                        for member in archive.getmembers()
                        if member.isfile()
                        and (relative := relative_symmetry_path(member.name)) is not None
                    }
                    for name, expected in source_files.items():
                        with self.subTest(distribution="sdist", file=name):
                            self.assertIn(name, members)
                            extracted = archive.extractfile(members[name])
                            self.assertIsNotNone(extracted)
                            self.assertEqual(extracted.read(), expected)
            else:
                self.fail(f"unsupported sdist archive format: {sdist}")


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryDependencyTierTests("test_wheel_and_sdist_include_all_symmetry_modules_and_mit_license")])
