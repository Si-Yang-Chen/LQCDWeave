"""Offline documentation guards: local links, real imports and call signatures.

Python fences are call scripts, potentially requiring caller-supplied inputs.
Signature outlines use text fences. Only the explicitly selected CPU scripts in
``test_documentation.py`` are executed; parsing/binding is not numerical evidence.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import html
import importlib
import inspect
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^(`{3,}|~{3,})([^\n]*)\n(.*?)^\1\s*$", re.M | re.S)
LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")


@dataclass(frozen=True)
class Snippet:
    path: Path
    line: int
    code: str


def documentation_paths(root=ROOT):
    root = Path(root)
    paths = {root / name for name in ("README.md", "ARCHITECTURE.md", "AGENTS.md", "TODO.md")}
    for folder in ("docs", ".agents/skills", "tests", "benchmarks", "tools", ".github", "constraints"):
        paths.update((root / folder).rglob("*.md"))
    return sorted(path for path in paths if path.is_file())


def python_snippets(path):
    text = Path(path).read_text(encoding="utf-8")
    for match in FENCE.finditer(text):
        if match.group(2).strip() in {"python", "py", "python3"}:
            yield Snippet(Path(path), text[:match.start(3)].count("\n") + 1, match.group(3))


def without_code(text):
    # Retain line numbers so a failed link points to the source line.
    text = FENCE.sub(lambda match: "\n" * match.group(0).count("\n"), text)
    return re.sub(r"(`+)(.*?)\1", lambda match: " " * len(match.group(0)), text)


def markdown_anchors(path):
    text = FENCE.sub("", Path(path).read_text(encoding="utf-8"))
    result = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", text, re.M):
        heading = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"<[^>]+>", "", html.unescape(heading))
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        occurrence = counts.get(slug, 0)
        counts[slug] = occurrence + 1
        result.add(slug + (f"-{occurrence}" if occurrence else ""))
    return result


def check_links(paths, root=ROOT):
    errors, checked, external = [], 0, 0
    root = Path(root).resolve()
    anchor_cache = {}
    for path in paths:
        path = Path(path)
        text = without_code(path.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match.group(1).strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                external += 1
                continue
            destination = (path.parent / unquote(url.path)).resolve() if url.path else path.resolve()
            # Migration evidence is intentionally outside source distributions.
            # Never require or claim to verify files outside this checkout.
            if not destination.is_relative_to(root):
                external += 1
                continue
            checked += 1
            location = f"{path}:{text[:match.start()].count(chr(10)) + 1}"
            if not destination.exists():
                errors.append(f"{location}: missing local link target {target}")
            elif url.fragment and destination.suffix == ".md":
                if destination not in anchor_cache:
                    anchor_cache[destination] = markdown_anchors(destination)
                if unquote(url.fragment) not in anchor_cache[destination]:
                    errors.append(f"{location}: missing Markdown anchor {target}")
    return errors, checked, external


def imported_symbol(module_name, symbol):
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol)
    except AttributeError:
        # ``from package import submodule`` is a valid Python import even if the
        # package has not already imported the child into its namespace.
        return importlib.import_module(module_name + "." + symbol)


def resolve_reference(node, names):
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = resolve_reference(node.value, names)
        if parent is not None:
            return getattr(parent, node.attr)
    return None


def check_snippet(snippet):
    errors, imports, calls = [], 0, 0
    try:
        tree = ast.parse(snippet.code, filename=str(snippet.path))
    except SyntaxError as error:
        return [f"{snippet.path}:{snippet.line + error.lineno - 1}: {error.msg}"], 0, 0
    names = {}
    for node in ast.walk(tree):
        location = f"{snippet.path}:{snippet.line + getattr(node, 'lineno', 1) - 1}"
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "lqcdweave" or node.module.startswith("lqcdweave.")
        ):
            for alias in node.names:
                if alias.name == "*":
                    errors.append(f"{location}: name public imports explicitly instead of '*'")
                    continue
                try:
                    names[alias.asname or alias.name] = imported_symbol(node.module, alias.name)
                    imports += 1
                except (ImportError, AttributeError) as error:
                    errors.append(f"{location}: invalid import {node.module}.{alias.name}: {error}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not (alias.name == "lqcdweave" or alias.name.startswith("lqcdweave.")):
                    continue
                try:
                    module = importlib.import_module(alias.name)
                    names[alias.asname or "lqcdweave"] = (
                        module if alias.asname else importlib.import_module("lqcdweave")
                    )
                    imports += 1
                except ImportError as error:
                    errors.append(f"{location}: invalid import {alias.name}: {error}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            function = resolve_reference(node.func, names)
        except AttributeError as error:
            location = f"{snippet.path}:{snippet.line + node.lineno - 1}"
            errors.append(f"{location}: invalid callable {ast.unparse(node.func)}: {error}")
            continue
        if function is None or not callable(function):
            continue
        # Ellipses and unpacked arguments are intentional partial templates.
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ) or any(isinstance(arg, ast.Constant) and arg.value is Ellipsis for arg in ast.walk(node)):
            continue
        try:
            signature = inspect.signature(function)
        except (ValueError, TypeError):
            continue
        calls += 1
        try:
            signature.bind(*[None for arg in node.args], **{key.arg: None for key in node.keywords})
        except TypeError as error:
            location = f"{snippet.path}:{snippet.line + node.lineno - 1}"
            errors.append(f"{location}: invalid call {ast.unparse(node.func)}{signature}: {error}")
    return errors, imports, calls


def check_documentation(root=ROOT):
    paths = documentation_paths(root)
    errors, links, external = check_links(paths, root)
    imports = calls = snippets = 0
    source_snippets = [snippet for path in paths for snippet in python_snippets(path)]
    # Reusable skill assets share the import gate. Local calculation projects
    # under scripts/ are untracked and have their own task-specific checks.
    source_snippets.extend(Snippet(path, 1, path.read_text(encoding="utf-8"))
                           for path in sorted((Path(root) / ".agents/skills").rglob("*.py")))
    for snippet in source_snippets:
        failures, imported, bound = check_snippet(snippet)
        errors.extend(failures)
        snippets += 1
        imports += imported
        calls += bound
    return errors, dict(documents=len(paths), snippets=snippets, local_links=links,
                        external_links_unchecked=external, imports=imports, call_signatures=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors, counts = check_documentation()
    if errors:
        print("\n".join(errors))
        return 1
    print("Documentation guards passed: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
