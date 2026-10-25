"""Docs drift guards for links, imports and call signatures."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "tools" / "check_docs.py"
spec = importlib.util.spec_from_file_location("documentation_check_tool", path)
doc_check = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = doc_check
spec.loader.exec_module(doc_check)
ROOT, Snippet = doc_check.ROOT, doc_check.Snippet
check_documentation = doc_check.check_documentation
check_links, check_snippet = doc_check.check_links, doc_check.check_snippet


class DocumentationGuardTests(unittest.TestCase):
    def test_local_calculation_projects_do_not_enter_repository_doc_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("# Maintained documentation\n", encoding="utf-8")
            project = root / "scripts" / "local_measurement"
            project.mkdir(parents=True)
            (project / "notes.md").write_text("[task-only evidence](not-copied.md)\n", encoding="utf-8")
            (project / "worker.py").write_text("not valid Python !\n", encoding="utf-8")
            assets = root / ".agents" / "skills" / "reusable" / "scripts"
            assets.mkdir(parents=True)
            (assets / "helper.py").write_text("from lqcdweave.no_such_reusable_module import helper\n", encoding="utf-8")

            errors, counts = check_documentation(root)
            self.assertEqual(counts["documents"], 1)
            self.assertEqual(counts["snippets"], 1)
            self.assertEqual(len(errors), 1)
            self.assertIn("no_such_reusable_module", errors[0])

    def test_links_imports_and_call_signatures(self):
        errors, counts = check_documentation()
        self.assertGreater(counts["imports"], 0)
        self.assertGreater(counts["call_signatures"], 0)
        self.assertEqual(errors, [], "\n".join(errors))

    def test_guard_rejects_bad_import_and_signature_without_running_snippet(self):
        errors, _, _ = check_snippet(Snippet(Path("example.md"), 12, '''
from lqcdweave.sampling.spatial.api import sample_spatial_points, no_such_symbol
sample_spatial_points(design, misspelled_seed=7)
raise RuntimeError("must never execute a documentation fragment")
'''))
        self.assertEqual(len(errors), 2)
        self.assertIn("invalid import", errors[0])
        self.assertIn("invalid call", errors[1])
        self.assertIn("example.md:14", errors[1])

    def test_guard_rejects_removed_public_method(self):
        errors, _, _ = check_snippet(Snippet(Path("example.md"), 1, '''
from lqcdweave import preset
preset.no_such_function("input.npy")
'''))
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid callable preset.no_such_function", errors[0])

    def test_guard_resolves_package_submodules_and_aliases(self):
        errors, imports, calls = check_snippet(Snippet(Path("example.md"), 1, '''
from lqcdweave import preset
from lqcdweave.sampling.spatial.api import sample_spatial_points as sample
preset.load("caller-supplied.npy", kind="gauge")
sample(design, seed=7)
'''))
        self.assertEqual(errors, [])
        self.assertEqual((imports, calls), (2, 2))

    def test_link_guard_checks_fragments_and_ignores_code_and_external_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "page.md"
            page.write_text('''# 示例 Heading
# 示例 Heading
[good](#示例-heading)
[duplicate](#示例-heading-1)
[bad anchor](#absent)
[bad file](missing.md)
[external historical evidence](../migration-notes/report.md)
`[not a link](missing-inline.md)`
```text
[not a link](missing-code.md)
```
''', encoding="utf-8")
            errors, checked, external = check_links([page], root)
            self.assertEqual((len(errors), checked, external), (2, 4, 1))
            self.assertIn("missing Markdown anchor", errors[0])
            self.assertIn("missing local link target", errors[1])

    def test_bad_python_fence_is_not_silently_treated_as_signature_notation(self):
        errors, _, _ = check_snippet(Snippet(Path("example.md"), 8, "function(value, *, option=None)"))
        self.assertEqual(len(errors), 1)
        self.assertIn("example.md:8", errors[0])


if __name__ == "__main__":
    unittest.main()
