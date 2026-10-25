import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.contraction.correlator import (
    calc_diagram_eval,
    save_calculation_plan,
    load_calculation_plan,
)
from lqcdweave.contraction.correlator_cfg import complete_calculation


class WorkflowPlanImportTests(unittest.TestCase):
    def test_direct_scan_archive_with_complex_operator_recipe_replays(self):
        diagram, propagator, elemental, _ = self._fixture()
        operator = Operator.from_terms('complex-path', (
            OperatorTerm(1, ElementalComponent(0), 0, .7 + .2j),))
        vertices = {row: Meson(elemental, operator, source=(row == 'a'), phase_convention='anchor')
                    for row in ('a', 'b')}
        result = calc_diagram_eval(-diagram, vertex_map=vertices, propagator_map={'q': propagator},
            configuration='cfg', time_axes={0: range(3), 1: range(3)},
            memory_limit=1 << 20, optimize='none', profile='default')
        self.assertEqual(result.value.shape, (3, 3))
        self.assertGreater(float(np.linalg.norm(result.value)), 1e-8)
        self.assertTrue(result.path.batch_paths)
        with TemporaryDirectory() as directory:
            archive = Path(directory) / 'direct-scan.json'
            save_calculation_plan(archive, result)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('replay must not search')):
                loaded = load_calculation_plan(archive, expected=result)
            np.testing.assert_allclose(loaded.run().value, result.value, rtol=2e-12, atol=2e-12)

    def _fixture(self, cfg="cfg", *, seed=7319):
        diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1], ["a", "b"], [None, "q"])
        component = ElementalComponent(0)
        operator = Operator.from_terms("local", [OperatorTerm(0, component, 0)])
        rng = np.random.default_rng(seed)
        elemental = ArrayElemental(
            {"v2v": (rng.normal(size=(1, 1, 3, 1, 1))
                     + 1j * rng.normal(size=(1, 1, 3, 1, 1)))},
            (component,), phase_convention="anchor", cfg=cfg,
            data_identity=f"plan-import-elemental-{cfg}",
        )
        propagator = ArrayPropagator(
            {"vsv": (rng.normal(size=(3, 3, 4, 4, 1, 1))
                     + 1j * rng.normal(size=(3, 3, 4, 4, 1, 1)))},
            time_convention="absolute", cfg=cfg, data_identity=f"plan-import-peram-{cfg}",
        )
        vertex_map = {row: Meson(elemental, operator, phase_convention="anchor")
                      for row in ("a", "b")}
        return diagram, propagator, elemental, vertex_map

    def test_complete_archive_stores_all_pairs_and_loads_raw_readers(self):
        diagram, propagator, elemental, vertex_map = self._fixture()
        planning_options = dict(
            propagator_map={"q": propagator},
            vertex_map=vertex_map,
            configuration="cfg",
            usedNe=1,
            memory_limit=1 << 20,
            optimize="anytime",
            search={
                "root_width": 1,
                "branch_width": 1,
                "scorer": ("sharing_count", "element_reduction"),
                "deadline_seconds": 10,
                "max_states": 10000,
            },
            execute=False,
        )
        pairs = ({0: 2, 1: 0}, {0: 1, 1: 0})
        selected = calc_diagram_eval(
            diagram,time_axes={0:1,1:0},**planning_options,
        )
        coordinates, _ = describe_coordinates(time_pairs=pairs)
        complete = complete_calculation(selected,coordinates=coordinates,
            readers=selected.inputs.loader.input_readers)
        other_diagram, other_propagator, other_elemental, other_vertex_map = self._fixture(
            "cfg-2", seed=9102,
        )
        direct = calc_diagram_eval(
            other_diagram,time_pairs=pairs,
            propagator_map={"q":other_propagator},vertex_map=other_vertex_map,
            configuration="cfg-2",usedNe=1,memory_limit=1<<20,
            optimize="none",execute=True,
        )
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "complete-pairs.json.gz"
            save_calculation_plan(archive,complete)
            with patch(
                "lqcdweave.contraction.correlator.find_contraction_path",
                side_effect=AssertionError("complete archive load must not search"),
            ):
                rebound = calc_diagram_eval(
                    plan_archive=archive,configuration="cfg-2",
                    propagator_map={"q":other_propagator},vertex_map=other_elemental,
                    memory_limit=1<<20,execute=False,
                )
            rebound_value = rebound.run().value

        self.assertEqual(len(rebound.plan.instances),len(pairs))
        self._assert_values_equal(rebound_value,direct.value)
        self.assertTrue(self._has_distinct_pair_values(rebound_value))

    def _assert_values_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(tuple(actual), tuple(expected))
            for key in expected:
                self._assert_values_equal(actual[key], expected[key])
            return
        np.testing.assert_allclose(actual, expected)

    def _has_distinct_pair_values(self, value):
        if isinstance(value, dict):
            return any(self._has_distinct_pair_values(child) for child in value.values())
        array = np.asarray(value)
        return array.ndim > 0 and array.shape[-1] == 2 and not np.allclose(array[..., 0], array[..., 1])

    def test_complete_archive_load_needs_no_target_hardware(self):
        diagram, propagator, elemental, vertex_map = self._fixture()
        selected = calc_diagram_eval(
            diagram,time_axes={0:1,1:0},propagator_map={"q":propagator},
            vertex_map=vertex_map,configuration="cfg",usedNe=1,
            memory_limit=1<<20,execute=False,
        )
        complete = complete_calculation(selected,
            readers=selected.inputs.loader.input_readers)
        expected = complete.run().value
        with TemporaryDirectory() as directory:
            archive = Path(directory)/"complete.json.gz"
            save_calculation_plan(archive,complete)
            loaded = calc_diagram_eval(plan_archive=archive,configuration="cfg",
                propagator_map={"q":propagator},vertex_map=elemental,
                memory_limit=1<<20,execute=False)
            np.testing.assert_allclose(loaded.run().value,expected)


if __name__ == "__main__":
    unittest.main()
