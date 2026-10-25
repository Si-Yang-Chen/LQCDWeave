"""Reusable local acceptance evidence; datasets always live outside the repository.

Each phase records code/test path inventories, interpreter details and
explicitly supplied dependency paths. It does not capture file contents.
This tool never pushes, submits jobs, installs dependencies or changes project
instructions.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile


IGNORED = {"__pycache__", ".pytest_cache", ".git"}
CODE_ROOTS = ("src", "tests", "scripts", "benchmarks", "tools", ".github")


def tree_paths(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"dependency/code directory unavailable: {root}")
    return sorted(path.relative_to(root).as_posix()
                  for path in root.rglob("*")
                  if path.is_file() and not any(part in IGNORED or part.endswith(".egg-info")
                                               for part in path.relative_to(root).parts)
                  and path.suffix not in (".pyc", ".pyo"))


def code_paths(repo):
    repo = Path(repo)
    files = {folder + "/" + name
             for folder in CODE_ROOTS if (repo / folder).exists()
             for name in tree_paths(repo / folder)}
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (repo / name).exists():
            files.add(name)
    return sorted(files)


def runtime_profile(dependency_roots):
    roots = {Path(path).resolve() for path in dependency_roots}
    roots.add(Path(sysconfig.get_path("stdlib")).resolve())
    dlls = Path(sys.base_prefix) / "DLLs"
    if dlls.is_dir():
        roots.add(dlls.resolve())
    return dict(python=str(Path(sys.executable).resolve()), version=sys.version,
                platform=platform.platform(),
                environment={name:os.environ.get(name) for name in
                    ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTEST_ADDOPTS",
                     "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                     "ED_WEAK_FIELD_DIR", "ED_RUN_GPU_BENCHMARK")},
                dependencies={str(Path(path).resolve()): tree_paths(path)
                              for path in sorted(roots)})


def snapshot(repo, dependency_roots):
    return dict(files=code_paths(repo), runtime=runtime_profile(dependency_roots))


def read_record(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def verify_snapshot(repo, frozen):
    current = snapshot(repo, frozen["dependency_roots"])
    if current != frozen["snapshot"]:
        raise ValueError("frozen path inventory or runtime profile changed; use a new phase")
    return current


def verify_wheel(repo, wheel, installed, retired):
    repo, installed = Path(repo), Path(installed)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        if len(names) != len(archive.namelist()):
            raise ValueError("duplicate wheel members")
        for name in names:
            top = name.split("/", 1)[0]
            if (not name.startswith("lqcdweave/") and not
                    (top.startswith("lqcdweave-") and top.endswith(".dist-info"))):
                raise ValueError(f"unexpected wheel member: {name}")
        for module in retired:
            name = module.replace(".", "/")
            if name + ".py" in names or any(item.startswith(name + "/") for item in names):
                raise ValueError(f"retired module in wheel: {module}")
        archive_files = sorted(name for name in names if not name.endswith("/"))
        packaged = sorted(name for name in archive_files
                          if name.startswith("lqcdweave/"))
    expected = ["lqcdweave/" + name
                for name in tree_paths(repo / "src/lqcdweave")]
    if packaged != expected:
        raise ValueError("wheel package paths differ from delivered source/resources")
    if tree_paths(installed) != archive_files:
        raise ValueError("installed package paths differ from wheel members")
    parsed = 0
    for name in code_paths(repo):
        if name.endswith(".py"):
            ast.parse((repo / name).read_text(encoding="utf-8-sig"), feature_version=(3, 10))
            parsed += 1
    return dict(wheel=str(Path(wheel).resolve()),
                installed=str(installed.resolve()), package_files=packaged,
                archive_files=archive_files,
                python310_ast_files=parsed,
                retired_modules=retired)


def verify_package(repo, package, retired):
    return verify_wheel(repo, package["wheel"], package["installed"], retired)


def run_suite(repo, phase, frozen, suite, package):
    if suite == "installed" and os.environ.get("PYTEST_ADDOPTS", "").strip():
        raise ValueError("full installed regression requires empty PYTEST_ADDOPTS; use explicit focused tests separately")
    if suite == "installed" and os.environ.get("PYTEST_PLUGINS", "").strip():
        raise ValueError("full installed regression requires empty PYTEST_PLUGINS; external plugins may filter collection")
    verify_snapshot(repo, frozen)
    result_path = phase / (suite + "-result.json")
    if result_path.exists():
        raise ValueError("a suite result already exists; use a new phase to run tests again")
    running = phase / (suite + "-unfinished.json")
    try:
        with running.open("x", encoding="utf-8") as stream:
            json.dump(dict(suite=suite, wrapper_pid=os.getpid(),
                           instruction="Inspect the actual process/tool handle; this marker alone is not liveness evidence."), stream)
    except FileExistsError as exc:
        raise ValueError("unfinished suite evidence exists; inspect its actual process handle before any restart") from exc
    expected = Path(package["installed"]) if suite == "installed" else repo / "src"
    roots = [str(expected), *frozen["dependency_roots"], str(repo), str(repo / "tests")]
    config = dict(expected=str(expected.resolve()), roots=roots, suite=suite)
    runner = phase / (suite + "-runner.py")
    runner.write_text("""import json, pathlib, sys
config = json.loads(CONFIG)
sys.path[:0] = config['roots']
import lqcdweave
expected = pathlib.Path(config['expected'])
assert pathlib.Path(lqcdweave.__file__).resolve().is_relative_to(expected)
if config['suite'] == 'installed':
    import pytest
    result = pytest.main(['tests', '-q'])
else:
    import unittest
    tests = unittest.defaultTestLoader.discover('tests')
    result = 0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1
origins = 0
for name, module in tuple(sys.modules.items()):
    if name == 'lqcdweave' or name.startswith('lqcdweave.'):
        source = getattr(module, '__file__', None)
        if source:
            assert pathlib.Path(source).resolve().is_relative_to(expected), (name, source)
            origins += 1
print('Package origins verified:', origins, flush=True)
raise SystemExit(result)
""".replace("CONFIG", repr(json.dumps(config))), encoding="utf-8")
    log = phase / (suite + ".log")
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(roots))
    with log.open("wb") as stream:
        process = subprocess.run([sys.executable, str(runner)], cwd=repo,
                                 stdout=stream, stderr=subprocess.STDOUT, env=environment)
    verify_snapshot(repo, frozen)
    record = dict(suite=suite, exit_code=process.returncode,
                  terminal=True, log=str(log.resolve()),
                  runner=str(runner.resolve()))
    write_record(result_path, record)
    running.unlink()
    print(f"{suite} exit {process.returncode}; evidence: {result_path}")
    return record


def commit_paths(repo, phase, frozen, paths, message):
    """Commit only explicit additions, changes and deletions."""
    repo = Path(repo).resolve()
    for name in paths:
        path = repo / name
        resolved = path.resolve()
        if (not resolved.is_relative_to(repo) or resolved == repo or
                path.is_dir() or path.is_symlink() or
                path.name.casefold() in ("agents.md", "skill.md") or
                any(part.casefold() == ".git" for part in path.parts)):
            raise ValueError(f"commit requires explicit regular-file or deleted-file paths: {name}")
    git = lambda *a: subprocess.check_output(["git", "--literal-pathspecs", *a], cwd=repo, text=True).strip()
    if git("diff", "--cached", "--name-only"):
        raise ValueError("preserve existing staged work; commit requires an empty index")
    for name in paths:
        if not (repo / name).is_file() and not git("ls-files", "--", name):
            raise ValueError(f"deleted commit path was never tracked: {name}")
    subprocess.run(["git", "--literal-pathspecs", "add", "--", *paths], cwd=repo, check=True)
    if set(git("diff", "--cached", "--name-only").splitlines()) != set(paths):
        raise ValueError("staged scope differs from explicit paths")
    subprocess.run(["git", "diff", "--cached", "--check"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True)
    head = git("rev-parse", "HEAD")
    if set(git("show", "--format=", "--name-only", head).splitlines()) != set(paths):
        raise ValueError("committed scope differs")
    for name in paths:
        if not (repo / name).exists() and git("ls-tree", "--name-only", head, "--", name):
            raise ValueError(f"committed path state differs: {name}")
    verify_snapshot(repo, frozen)
    write_record(phase / "commit.json", dict(commit=head, files=paths, pushed=False))
    print(f"Scoped local commit audited: {head}; no push.")
    return head


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check", "build", "source", "installed", "audit", "commit"))
    parser.add_argument("phase")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--dependency-root", type=Path, action="append", default=[])
    parser.add_argument("--retired-report", type=Path)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--message")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", args.phase):
        parser.error("phase must be a simple lowercase name")
    repo = args.repo.resolve()
    evidence_root = (args.evidence_root or repo.parent / "acceptance-evidence").resolve()
    if evidence_root.is_relative_to(repo):
        parser.error("evidence must be outside the repository")
    phase = evidence_root / args.phase
    freeze_path = phase / "freeze.json"
    if args.command == "freeze":
        if freeze_path.exists():
            raise ValueError("phase is already frozen; use check or a new phase")
        dependencies = [str(path.resolve()) for path in args.dependency_root]
        retired = json.loads(args.retired_report.read_text())["retired_modules_verified"] if args.retired_report else []
        write_record(freeze_path, dict(repo=str(repo), dependency_roots=dependencies,
            snapshot=snapshot(repo, dependencies), retired=retired))
        print(f"Recorded {len(code_paths(repo))} code/input paths: {freeze_path}")
        return 0
    frozen = read_record(freeze_path)
    if frozen["repo"] != str(repo):
        raise ValueError("phase belongs to another repository")
    verify_snapshot(repo, frozen)
    if args.command == "check":
        print("Code/test path inventories and the declared runtime profile match the phase record.")
        return 0
    package_path = phase / "package.json"
    if args.command == "build":
        if package_path.exists():
            package = read_record(package_path)
            verify_package(repo, package, frozen["retired"])
            raise ValueError("package evidence already exists; use a new phase for a fresh build")
        with tempfile.TemporaryDirectory(dir=phase) as temporary:
            if not Path(temporary).resolve().is_relative_to(phase.resolve()):
                raise ValueError("temporary build directory escaped the named evidence phase")
            snapshot_dir = Path(temporary) / "package"
            shutil.copytree(repo / "src", snapshot_dir / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                if (repo / name).exists():
                    shutil.copy2(repo / name, snapshot_dir / name)
            wheels = phase / "wheels"
            wheels.mkdir(exist_ok=False)
            subprocess.run([sys.executable, "-m", "pip", "wheel", str(snapshot_dir), "--no-deps",
                            "--no-build-isolation", "--wheel-dir", str(wheels)], check=True)
        wheel, = wheels.glob("*.whl")
        installed = phase / "installed"
        installed.mkdir(exist_ok=False)
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if (Path(name).is_absolute() or ".." in Path(name).parts
                        or "\\" in name or ":" in name):
                    raise ValueError(f"unsafe wheel member: {name}")
            archive.extractall(installed)
        verify_snapshot(repo, frozen)
        write_record(package_path, verify_wheel(repo, wheel, installed, frozen["retired"]))
        print(f"Recorded wheel member and installed path inventories: {package_path}")
        return 0
    package = read_record(package_path)
    verify_package(repo, package, frozen["retired"])
    if args.command in ("source", "installed"):
        return run_suite(repo, phase, frozen, args.command, package)["exit_code"]
    for suite in ("source", "installed"):
        result_path = phase / (suite + "-result.json")
        if not result_path.exists():
            raise ValueError(f"successful {suite} result required")
        result = read_record(result_path)
        if result.get("suite") != suite or result.get("exit_code") != 0:
            raise ValueError(f"successful {suite} result required")
        for field in ("log", "runner"):
            path = Path(result[field]).resolve()
            if not path.is_relative_to(phase.resolve()) or not path.is_file():
                raise ValueError(f"{suite} result {field} path is missing or outside the phase")
    if args.command == "audit":
        print("Both suites passed; path inventories and runtime profile match the phase record.")
        return 0
    if not args.file or not args.message:
        parser.error("commit requires explicit --file arguments and --message")
    paths = []
    for name in args.file:
        path = (repo / name).resolve()
        if (not path.is_relative_to(repo) or path.name in ("AGENTS.md", "SKILL.md")
                or ".git" in path.relative_to(repo).parts or (repo / name).is_symlink()):
            raise ValueError(f"unapproved commit path: {name}")
        paths.append(path.relative_to(repo).as_posix())
    paths = sorted(set(paths))
    commit_paths(repo, phase, frozen, paths, args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
