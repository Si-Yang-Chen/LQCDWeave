"""Optional correlator dependencies stay out of pure result assembly."""

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np


class OptionalResultDependencyTests(unittest.TestCase):
    def _run_child(self, script, *arguments):
        package = importlib.import_module("lqcdweave")
        package_parent = str(Path(package.__file__).resolve().parent.parent)
        environment = os.environ.copy()
        inherited = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            [package_parent] + ([inherited] if inherited else []))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script), package_parent, *arguments],
            capture_output=True, text=True, env=environment, check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            f"isolated child failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result.stdout.strip()

    def test_result_api_workload_assembly_and_numeric_recipe_need_no_optional_roots(self):
        output = self._run_child(r"""
            import importlib.abc
            import sys
            from pathlib import Path

            blocked = {"sympy", "scipy", "lazy_object_proxy"}
            class BlockOptionalRoots(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    root = fullname.split(".", 1)[0]
                    if root in blocked:
                        raise ModuleNotFoundError(
                            f"blocked optional dependency {root}", name=root)
                    return None

            sys.meta_path.insert(0, BlockOptionalRoots())
            import lqcdweave
            assert Path(lqcdweave.__file__).resolve().parent.parent == Path(sys.argv[1]).resolve()

            import numpy as np
            from types import SimpleNamespace
            from lqcdweave.contracts.correlator.api import (
                ContractionTerm, TensorSpec, WorkloadSpec, describe_coordinates)
            from lqcdweave.contracts.execution import TermValue
            from lqcdweave.contraction.results.api import (
                OutputNode, OutputRecipe, assemble_correlator,
                assemble_workload_outputs, estimate_output_retention_bytes)

            assert "sympy" not in sys.modules
            assert "scipy" not in sys.modules
            assert "lazy_object_proxy" not in sys.modules
            assert "lqcdweave.symmetry" not in sys.modules
            assert "lqcdweave.operators.correlator.api" not in sys.modules

            value = np.array([1.0, 2.0])
            generic = WorkloadSpec(
                specs={"input": TensorSpec((2,), np.float64)},
                terms=(ContractionTerm(("input",), "i->i"),),
            )
            generic_values = (value,)
            restored = assemble_workload_outputs(generic, generic_values, object())
            assert type(restored) is tuple and restored[0] is value
            assert restored == generic_values

            names = ("left", "middle", "right")
            meson = WorkloadSpec(
                specs={name: TensorSpec((1,), np.complex128) for name in names},
                terms=tuple(ContractionTerm((name,), "i->") for name in names),
                convention="meson-simple-v1", output_groups=((0, 1), (2,)),
                metadata={"adapter": "meson-simple-v1",
                          "output_recipes": [[0, 1], [2]],
                          "assembly_extra_bytes": 96},
            )
            class Backend:
                __name__ = "numeric-test-backend"
                def __init__(self):
                    self.arrays = []
                    self.stacks = []
                def asarray(self, value):
                    self.arrays.append(value)
                    return np.asarray(value)
                def stack(self, values):
                    self.stacks.append(tuple(values))
                    return np.stack(values)

            backend = Backend()
            terms = (2 + 3j, -1 + .5j, 7 - 2j)
            assembled_workload = assemble_workload_outputs(meson, terms, backend)
            expected_workload = np.array([terms[0] * terms[1], terms[2]])
            np.testing.assert_array_equal(assembled_workload, expected_workload)
            assert assembled_workload.shape == (2,)
            assert backend.arrays == [terms[0] * terms[1], terms[2]]
            assert len(backend.stacks) == 1 and len(backend.stacks[0]) == 2

            coordinates, _ = describe_coordinates(time_axes={"sink": (4, 1)})
            recipe = OutputRecipe(
                OutputNode("diagram_sum", (("term", 2 + 1j),)),
                ("term",), coordinates)
            values = tuple(TermValue("term", point.key,
                           dict(point.values)["sink"] + 3j)
                           for point in coordinates.points)
            expanded = SimpleNamespace(output_recipe=recipe, coordinates=coordinates)
            restored_values = assemble_correlator(expanded, values)
            np.testing.assert_array_equal(restored_values,
                np.array([(2 + 1j) * (sink + 3j) for sink in (4, 1)]))
            assert restored_values.shape == (2,)
            problem = SimpleNamespace(
                terms=(ContractionTerm(("input",), "i->"),),
                specs={"input": TensorSpec((1,), np.complex128)},
            )
            retained = estimate_output_retention_bytes(recipe, problem, (None,), coordinates)
            assert retained == 96
            assert "sympy" not in sys.modules
            assert "scipy" not in sys.modules
            assert "lazy_object_proxy" not in sys.modules
            assert "lqcdweave.symmetry" not in sys.modules
            assert "lqcdweave.operators.correlator.api" not in sys.modules
            print("pure-result-paths-ok", retained)
        """)
        self.assertEqual(output, "pure-result-paths-ok 96")

    def test_symbolic_entries_defer_missing_dependency_error_to_the_s_boundary(self):
        for root in ("sympy", "lazy_object_proxy"):
            for entry in ("collect", "recipe"):
                with self.subTest(root=root, entry=entry):
                    output = self._run_child(r"""
                        import importlib.abc
                        import sys
                        from pathlib import Path

                        blocked = sys.argv[2]
                        entry = sys.argv[3]
                        class BlockOptionalRoot(importlib.abc.MetaPathFinder):
                            def find_spec(self, fullname, path=None, target=None):
                                root = fullname.split(".", 1)[0]
                                if root == blocked:
                                    raise ModuleNotFoundError(
                                        f"blocked optional dependency {root}", name=root)
                                return None

                        sys.meta_path.insert(0, BlockOptionalRoot())
                        import lqcdweave
                        assert Path(lqcdweave.__file__).resolve().parent.parent == Path(sys.argv[1]).resolve()
                        from lqcdweave.contraction.results.api import (
                            build_output_recipe, collect_diagrams)
                        from lqcdweave.contracts.correlator.api import describe_coordinates
                        assert blocked not in sys.modules
                        coordinates, _ = describe_coordinates()

                        try:
                            if entry == "collect":
                                collect_diagrams(1)
                            else:
                                build_output_recipe(1, {}, (), coordinates)
                        except ModuleNotFoundError as exc:
                            message = str(exc).lower()
                            cause = exc.__cause__
                            assert exc.name == blocked, (exc.name, str(exc))
                            assert isinstance(cause, ModuleNotFoundError)
                            assert cause.name == blocked
                            for required in (blocked, "correlator", "python 3.11",
                                             "docs/modules/correlator/correlator_frontend.md",
                                             "pip install -e '.[correlator]'",
                                             ".[correlator]"):
                                assert required in message, (required, str(exc))
                        else:
                            raise AssertionError("symbolic entry did not require the optional dependency")
                        print("deferred-with-install-hint", entry, blocked)
                    """, root, entry)
                    self.assertEqual(output, f"deferred-with-install-hint {entry} {root}")

    def test_installed_symbolic_entries_keep_the_public_s_diagram_type(self):
        if (importlib.util.find_spec("sympy") is None or
                importlib.util.find_spec("lazy_object_proxy") is None or
                importlib.util.find_spec("scipy") is None):
            self.skipTest("the optional correlator extra is not installed")

        from lqcdweave.contracts.correlator.api import (
            TermContribution, describe_coordinates)
        from lqcdweave.contraction.results.api import (
            OutputRecipe, build_output_recipe, collect_diagrams)
        from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram

        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]),
                          [0, 1], ["a", "b"], [None, "q"])
        self.assertIs(type(diagram), Diagram)
        self.assertEqual(collect_diagrams([diagram, diagram]), (diagram,))
        coordinates, _ = describe_coordinates()
        recipe = build_output_recipe(
            [diagram, diagram],
            {diagram: (TermContribution("shared", 1.25 - .5j),)},
            ("shared",), coordinates,
        )
        self.assertIs(type(recipe), OutputRecipe)
        self.assertEqual(recipe.root.kind, "list")
        self.assertEqual(len(recipe.root.children), 2)
        self.assertEqual(recipe.root.children[0].kind, "diagram_sum")
        self.assertEqual(recipe.root.children[0].value,
                         (("shared", 1.25 - .5j),))


if __name__ == "__main__":
    unittest.main()
