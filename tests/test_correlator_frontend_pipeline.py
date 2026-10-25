import inspect
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.adapters.vsv_diagram import VSVData
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.correlator.api import (
    BatchSpec, CoordinateSpec, DiagramBindings, DiagramInputs, TermContribution,
    PlannedContraction, TensorSpec, describe_coordinates,
)
from lqcdweave.contraction.results.api import (
    OutputNode, OutputRecipe, TermValue, assemble_correlator as assemble_recipe,
    build_output_recipe,
)
from lqcdweave.contraction.vsv_graph import evaluate_vsv_diagram
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.contraction.correlator import (
    assemble_correlator, calc_diagram_eval, execute_contraction_path,
    expand_correlator, find_contraction_path,
)
from lqcdweave.contraction.performance.api import save_cost_profile


class CorrelatorFrontendPipelineTests(unittest.TestCase):
    def binding(self):
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ["a", "b"], [None, "q"])
        rng = np.random.default_rng(105)
        tensors = [rng.normal(size=(4, 4, 1, 1)) + 1j * rng.normal(size=(4, 4, 1, 1))
                   for _ in range(4)]
        reads = []
        data = VSVData(
            {0: 0, 1: 1},
            lambda *args: TensorSpec((4, 4, 1, 1), "complex128"),
            lambda *args: TensorSpec((4, 4, 1, 1), "complex128"),
            lambda row, time: reads.append(("v", row, time)) or tensors[0 if row == "a" else 1],
            lambda tag, sink, source: reads.append(("p", tag, sink, source)) or tensors[2 if source == 0 else 3],
            "registered-test-input",
        )
        return diagram, data, reads

    def test_workflow_resolves_profile_objects_names_and_paths_before_planning(self):
        diagram, adapter, _ = self.binding()
        explicit = replace(CostProfile(), version="workflow-profile")
        with TemporaryDirectory() as directory, patch.dict("os.environ", {"LQCDWEAVE_PROFILE_DIR": directory}):
            path = save_cost_profile(explicit, Path(directory) / "profile.json")
            for reference, expected_version in (
                (None, CostProfile().version),
                (explicit, "workflow-profile"),
                ("default", CostProfile().version),
                (path, "workflow-profile"),
            ):
                result = calc_diagram_eval(
                    diagram, diagram_adapter=adapter, profile=reference,
                    memory_limit=1 << 20, optimize="none", execute=False,
                )
                self.assertEqual(result.path.profile.version, expected_version)

    def test_four_stages_use_explicit_recipe_and_given_path(self):
        diagram, adapter, reads = self.binding()
        expanded = expand_correlator(diagram + 2j * diagram, adapter=adapter,
                                     bindings=DiagramBindings())
        self.assertEqual(len(expanded.terms), 1)
        self.assertIsInstance(expanded.output_recipe, OutputRecipe)
        self.assertFalse(hasattr(expanded.inputs, "assemble"))
        self.assertEqual(reads, [])
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="anytime")
        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("execution must not search")):
            execution = execute_contraction_path(expanded, path)
        value = assemble_correlator(expanded, execution.term_values)
        self.assertEqual(len(reads), 4)
        self.assertIsInstance(execution.term_values[0], TermValue)
        np.testing.assert_allclose(value, (1 + 2j) * execution.term_values[0].value)

    def test_public_entry_accepts_legacy_bindings_without_requiring_vsvdata(self):
        parameters = inspect.signature(calc_diagram_eval).parameters
        for name in ("propagator_map", "vertex_map", "configuration", "usedNe",
                     "time_axes", "time_pairs", "batch", "optimize", "storage", "execute"):
            self.assertIn(name, parameters)
        diagram, _, _ = self.binding()
        component = ElementalComponent(0)
        operator = Operator.from_terms("local", [OperatorTerm(0, component, 0)])
        elemental = ArrayElemental(
            {"v2v": np.ones((1, 1, 2, 1, 1), dtype=np.complex128)},
            (component,), phase_convention="anchor", cfg="cfg",
            data_identity="cfg/elemental",
        )
        propagator = ArrayPropagator(
            {"vsv": np.ones((2, 2, 4, 4, 1, 1), dtype=np.complex128)},
            time_convention="absolute", cfg="cfg", data_identity="cfg/peram",
        )
        elemental.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("plan read elemental data"))
        propagator.block = lambda *a, **k: (_ for _ in ()).throw(AssertionError("plan read propagator data"))
        vertex_map = {row: Meson(elemental, operator, phase_convention="anchor")
                      for row in ("a", "b")}
        planned = calc_diagram_eval(
            diagram, propagator_map={"q": propagator}, vertex_map=vertex_map,
            configuration="cfg", usedNe=1, optimize="none", execute=False,
        )
        self.assertEqual(planned.execution_mode, "direct")
        self.assertEqual(len(planned.expanded.terms), 1)

        with self.assertRaisesRegex(TypeError, "propagator_map values must expose"):
            calc_diagram_eval(
                diagram, propagator_map={"q": object()}, vertex_map=vertex_map,
                configuration="cfg", usedNe=1, optimize="none", execute=False,
            )

    def test_nested_containers_and_empty_constants_restore_from_recipe(self):
        diagram, adapter, _ = self.binding()
        matrix = np.array([[diagram, 2j * diagram], [diagram ** 2, -diagram]], dtype=object)
        expression = {"matrix": matrix, "nested": [diagram, (3 * diagram,)],
                      "empty_dict": {}, "empty_list": [], "empty_tuple": ()}
        result = calc_diagram_eval(expression, diagram_adapter=adapter,
                                   memory_limit=1 << 20, optimize="anytime")
        self.assertEqual(tuple(result.value), tuple(expression))
        self.assertEqual(result.value["matrix"].shape, (2, 2))
        self.assertEqual(result.value["matrix"].dtype.kind, "c")
        self.assertEqual(type(result.value["nested"]), list)
        self.assertEqual(type(result.value["nested"][1]), tuple)
        self.assertEqual(result.value["empty_dict"], {})
        self.assertEqual(result.value["empty_list"], [])
        self.assertEqual(result.value["empty_tuple"], ())

    def test_constant_and_empty_correlators_need_no_adapter_or_budget(self):
        for value in ({"empty": [], "nested": ((), {})}, np.empty((0, 2), dtype=object), 3 + 4j):
            result = calc_diagram_eval(value, execute=True)
            if not isinstance(value, np.ndarray):
                self.assertEqual(result.value, value)
            else:
                self.assertEqual(result.value.shape, value.shape)
                self.assertEqual(result.value.dtype, object)

    def test_constant_and_empty_none_mode_resolves_before_direct_dispatch(self):
        for value in (3 + 4j, {}, [], ((), {})):
            result = calc_diagram_eval(value, optimize="none", execute=True,
                                       storage="bounded")
            self.assertEqual(result.execution_mode, "empty")
            self.assertEqual(result.value, value)

    def test_fixed_and_scanned_time_axes_have_distinct_axis_metadata(self):
        coordinates, batch = describe_coordinates(
            time_axes={0: 3, 1: range(8)}, batch={"axis": 1, "size": 2})
        self.assertEqual(coordinates.fixed_axes, (0,))
        self.assertEqual(coordinates.scan_axes, (1,))
        self.assertEqual(coordinates.scan_shape, (8,))
        self.assertEqual(coordinates.points[0].values, ((0, 3), (1, 0)))
        self.assertEqual(coordinates.points[-1].values, ((0, 3), (1, 7)))
        self.assertEqual(batch.axis, 1)
        self.assertEqual(batch.size, 2)
        fixed, fixed_batch = describe_coordinates(time_axes={0: 3, 1: 7})
        self.assertEqual(fixed.scan_shape, ())
        self.assertEqual(fixed.scan_axes, ())
        self.assertEqual(fixed_batch.axis, None)
        pairs, pair_batch = describe_coordinates(
            time_pairs=({0: 1, 1: 4}, {0: 2, 1: 5}), batch={"axis": "pair", "size": 1})
        self.assertEqual(pairs.scan_axes, ("pair",))
        self.assertEqual(pairs.scan_shape, (2,))
        self.assertEqual(pair_batch.axis, "pair")
        with self.assertRaisesRegex(ValueError, "not a scanned time axis"):
            describe_coordinates(time_axes={0: 3, 1: range(8)}, batch={"axis": 0, "size": 2})
        with self.assertRaisesRegex(ValueError, "requires an explicit axis"):
            describe_coordinates(time_axes={0: range(3), 1: range(4)}, batch=2)

    def test_diagram_inputs_freeze_mapping_copies(self):
        coordinates, batch = describe_coordinates()
        blocks = {"raw": object()}
        preparations = {"tensor": object()}
        contributions, recipes = {}, {}
        inputs = DiagramInputs(None, blocks, preparations, object(), coordinates, batch,
                               (), contributions, recipes)
        blocks["later"] = object()
        preparations.clear()
        contributions["later"] = ()
        recipes["later"] = object()
        self.assertEqual(set(inputs.blocks), {"raw"})
        self.assertEqual(set(inputs.preparations), {"tensor"})
        self.assertEqual(dict(inputs.diagram_terms), {})
        self.assertEqual(dict(inputs.diagram_recipes), {})
        with self.assertRaises(TypeError):
            inputs.blocks["raw"] = object()

    def test_one_diagram_recipe_can_sum_products_of_independent_terms(self):
        diagram, _, _ = self.binding()
        coordinates, _ = describe_coordinates()
        local_recipe = OutputNode("add", children=(
            OutputNode("multiply", children=(
                OutputNode("constant", 2),
                OutputNode("term", TermContribution("T1")),
                OutputNode("term", TermContribution("T2")),
            )),
            OutputNode("term", TermContribution("T3")),
        ))
        recipe = build_output_recipe(
            diagram, {}, ("T1", "T2", "T3"), coordinates,
            diagram_recipes={diagram: local_recipe},
        )
        expanded = type("Expanded", (), {
            "output_recipe": recipe,
            "coordinates": coordinates,
        })()
        value = assemble_recipe(expanded, (
            TermValue("T1", "bound", 2),
            TermValue("T2", "bound", 3),
            TermValue("T3", "bound", 5),
        ))
        self.assertEqual(value, 17)

    def test_none_accepts_profile_and_valid_budget_but_rejects_invalid_values(self):
        diagram, adapter, _ = self.binding()
        profile = CostProfile()
        planned = calc_diagram_eval(
            diagram, diagram_adapter=adapter, optimize="none", execute=False,
            memory_limit=1 << 20, host_memory_limit=1 << 30,
            reserved_host_bytes=128, reserved_device_bytes=0, profile=profile,
        )
        self.assertIs(planned.path.profile, profile)
        self.assertEqual(planned.path.reserved_host_bytes,
                         128 + planned.expanded.inputs.assembly_host_bytes)
        self.assertTrue(planned.plan.feasible)
        self.assertGreaterEqual(planned.prediction["total_seconds"], 0)
        for kwargs in ({"memory_limit": -1}, {"memory_limit": True},
                       {"memory_limit": float("nan")},
                       {"host_memory_limit": float("nan")},
                       {"reserved_host_bytes": -1}, {"reserved_device_bytes": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                calc_diagram_eval(diagram, diagram_adapter=adapter,
                                  optimize="none", execute=False, **kwargs)

    def test_coordinates_are_mutually_exclusive_and_adapter_must_acknowledge(self):
        diagram, adapter, reads = self.binding()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            calc_diagram_eval(diagram, diagram_adapter=adapter, time_axes={0: [0]},
                              time_pairs=[{0: 0}], execute=False)
        with self.assertRaisesRegex(NotImplementedError, "one explicit time_map"):
            calc_diagram_eval(diagram, diagram_adapter=adapter,
                              time_axes={0: [0, 1], 1: [2, 3]}, execute=False)
        self.assertEqual(reads, [])
        fixed = calc_diagram_eval(diagram, diagram_adapter=adapter,
                                  time_axes={0: 0, 1: 1}, optimize="anytime",
                                  memory_limit=1 << 20)
        self.assertEqual(np.asarray(fixed.value).shape, ())
        self.assertEqual(len(reads), 4)

    def test_raw_or_foreign_plan_rejected_before_getters(self):
        from lqcdweave.contraction.execution.api import (
            execute_contraction_path as execute_bound_path,
        )
        diagram, adapter, reads = self.binding()
        expanded = expand_correlator(diagram, adapter=adapter)
        path = find_contraction_path(expanded, memory_limit=1 << 20, optimize="anytime")
        foreign = expand_correlator(diagram, adapter=adapter)
        invalid = (
            (expanded, path.plan, TypeError, "PlannedContraction"),
            (foreign, path, ValueError, "different ExpandedCorrelator"),
            (expanded, PlannedContraction(
                expanded, None, None, {}, 0.0, "planned", 1 << 20, None, None),
             ValueError, "audited SceneStoragePlan"),
        )
        with patch("lqcdweave.contraction.elemental_preparation.prepare_elemental_inputs",
                   side_effect=AssertionError("invalid path prepared inputs")), \
             patch("lqcdweave.contraction.execution.domain.device_execution",
                   side_effect=AssertionError("invalid path activated device")), \
             patch("lqcdweave.contraction.execution.binding_cache.ExecutionBindingCache.scope",
                   side_effect=AssertionError("invalid path accessed execution cache")):
            for entry in (execute_contraction_path, execute_bound_path):
                for bound, descriptor, error, message in invalid:
                    with self.subTest(entry=entry.__module__, message=message):
                        with self.assertRaisesRegex(error, message):
                            entry(bound, descriptor)
        self.assertEqual(reads, [])

    def test_optimize_none_never_calls_search_and_direct_capability_is_explicit(self):
        diagram, adapter, reads = self.binding()
        reference = evaluate_vsv_diagram(
            diagram, adapter.vertex_getter, adapter.propagator_getter, adapter.time_map,
        )
        reads.clear()

        original_einsum = np.einsum

        def no_search_einsum(*arguments, **kwargs):
            self.assertIs(kwargs.get("optimize"), False)
            return original_einsum(*arguments, **kwargs)

        with patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("none must not search")), \
             patch("lqcdweave.contraction.planning.path_planning.solve_storage",
                   side_effect=AssertionError("none must not search")), \
             patch("opt_einsum.contract_path",
                   side_effect=AssertionError("none must not ask opt_einsum for a path")), \
             patch("numpy.einsum", new=no_search_einsum):
            planned = calc_diagram_eval(diagram, diagram_adapter=adapter,
                                       optimize="none", execute=False)
            executed = planned.run()
        self.assertEqual(planned.execution_mode, "direct")
        self.assertTrue(planned.plan.feasible)
        self.assertTrue(planned.plan.state.done)
        self.assertAlmostEqual(executed.value, reference)
        self.assertEqual(len(reads), 4)
        self.assertCountEqual(
            reads,
            [("p", "q", 1, 0), ("p", "q", 0, 1), ("v", "a", 0), ("v", "b", 1)],
        )

        class UnsupportedAdapter:
            def describe(self, correlator, *, bindings, coordinates, batch):
                inputs = adapter.describe(correlator, bindings=bindings,
                                          coordinates=coordinates, batch=batch)
                return replace(inputs, loader=object())

        reads.clear()
        with self.assertRaisesRegex(NotImplementedError, "loader.read_block"):
            calc_diagram_eval(
                diagram, diagram_adapter=UnsupportedAdapter(), optimize="none", execute=False,
            )
        self.assertEqual(reads, [])

    def test_runtime_request_cannot_return_an_untyped_value_report_tuple(self):
        from lqcdweave.runtime.scheduler import execute_request

        class UntypedRequest:
            def execute(self, *, backend, measure):
                return (1, 2)

        with self.assertRaisesRegex(TypeError, "typed DirectExecution or PathExecution"):
            execute_request(UntypedRequest(), backend=np, measure=False)

    def test_none_audits_storage_but_rejects_search_controls(self):
        diagram, adapter, _ = self.binding()
        planned = calc_diagram_eval(diagram, diagram_adapter=adapter, optimize="none",
                                   storage="bounded", execute=False)
        self.assertTrue(planned.plan.feasible)
        with self.assertRaisesRegex(ValueError, "search controls"):
            calc_diagram_eval(diagram, diagram_adapter=adapter, optimize="none",
                              search={"deadline_seconds": 1}, execute=False)


if __name__ == "__main__":
    unittest.main()
