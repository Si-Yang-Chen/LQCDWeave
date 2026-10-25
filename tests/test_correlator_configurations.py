"""One frontend definition, independent cfg bindings and delayed numerical runs."""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import runpy
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave import preset
from lqcdweave.adapters.vsv_diagram import VSVData
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, gamma
from lqcdweave.contraction.correlator import calculate


class ConfigurationFrontendTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        rng = np.random.default_rng(671)
        self.arrays = {}
        for cfg in ("a", "b"):
            e = rng.normal(size=(1, 1, 2, 2, 2)) + 1j * rng.normal(size=(1, 1, 2, 2, 2))
            p = rng.normal(size=(2, 2, 4, 4, 2, 2)) + 1j * rng.normal(size=(2, 2, 4, 4, 2, 2))
            np.save(self.root / f"e_{cfg}.npy", e)
            np.save(self.root / f"p_{cfg}.npy", p)
            self.arrays[cfg] = e, p
        self.elemental = preset.ElementalNpy(str(self.root / "e_"))
        self.perambulator = preset.PerambulatorNpy(str(self.root / "p_"))
        for declaration in (self.elemental, self.perambulator):
            original = declaration.load
            def tracked_load(cfg, original=original):
                loaded = original(cfg)
                self.addCleanup(loaded.close)
                return loaded
            declaration.load = tracked_load
        self.operator = Operator.from_terms("local", [OperatorTerm(1, ElementalComponent(0), 0, .3 + .7j)])
        self.source = Meson(self.elemental, self.operator, True)
        self.sink = Meson(self.elemental, self.operator)
        self.diagram = Diagram(QuarkDiagram([[0, 1], [1, 0]]), [0, 1],
                               ["source", "sink"], [None, "q"])
        self.compute = partial(
            calculate, self.diagram,
            propagator_map={"q": self.perambulator, "same-source": self.perambulator},
            vertex_map={"source": self.source, "sink": self.sink},
            usedNe=2, time_axes={0: range(2), 1: range(2)},
            batch={"axis": 1, "size": 1}, optimize="none", memory_limit=16 * 2**20,
        )

    def expected(self, cfg, *, gamma_index=1, coefficient=.3 + .7j):
        e, p = self.arrays[cfg]
        result = np.empty((2, 2), complex)
        for source in range(2):
            for sink in range(2):
                a = np.kron(gamma(8), np.eye(2)) @ np.kron(
                    coefficient * gamma(gamma_index), e[0, 0, source]).conj().T @ np.kron(gamma(8), np.eye(2))
                b = np.kron(coefficient * gamma(gamma_index), e[0, 0, sink])
                reverse = p[sink, (source - sink) % 2].transpose(0, 2, 1, 3).reshape(8, 8)
                forward = p[source, (sink - source) % 2].transpose(0, 2, 1, 3).reshape(8, 8)
                result[source, sink] = np.trace(a @ reverse @ b @ forward)
        return result

    def test_public_prepared_example_anytime_matches_both_cfg_references(self):
        from tests.fixtures.legacy_two_point import prepare_two_point as prepare
        compute = prepare(self.elemental, self.perambulator, Lt=2, Ne=2)
        for cfg in ("a", "b"):
            actual = compute(configuration=cfg, optimize="anytime", memory_limit=16 * 2**20).value
            np.testing.assert_allclose(actual, -self.expected(cfg, gamma_index=15, coefficient=1),
                                       rtol=1e-12, atol=1e-12)

    def test_unbound_structured_matrix_uses_existing_lowering_for_both_cfgs(self):
        from test_structured_correlator import task
        compact = task(structured=True, names=("source", "sink"))
        symbolic = task(names=("source", "sink"))
        vertices = {(name, dagger): Meson(self.elemental, self.operator, dagger)
                    for name in ("source", "sink") for dagger in (False, True)}
        compute = partial(calculate,
                          propagator_map={"S^q": self.perambulator,
                                          "S^q_\\mathrm{local}": self.perambulator},
                          vertex_map=lambda row: vertices[row.hadron_name, row.dagger],
                          usedNe=2, time_axes={0: range(2), 1: range(2)},
                          optimize="none", memory_limit=16 * 2**20)
        for cfg in ("a", "b"):
            expected = compute(symbolic, configuration=cfg).value
            # W must not ask a plain-Diagram collector to lower structured requests.
            with patch("lqcdweave.contraction.correlator_inputs.collect_diagrams",
                       side_effect=AssertionError("premature structured lowering")):
                actual = compute(compact, configuration=cfg).value
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(all(vertex.reader is None for vertex in vertices.values()))

    def test_definitions_outside_cfg_loop_shared_loads_and_old_plan_lifetime(self):
        plans = []
        with patch.object(self.elemental, "load", wraps=self.elemental.load) as load_e, patch.object(
                self.perambulator, "load", wraps=self.perambulator.load) as load_p:
            for cfg in ("a", "b"):
                plans.append(self.compute(configuration=cfg, execute=False))
                self.assertEqual(load_e.call_count, len(plans))
                self.assertEqual(load_p.call_count, len(plans))
        self.assertIsNone(self.source.reader)
        self.assertIsNone(self.sink.reader)
        self.assertIsNone(self.source.key)
        # Even an explicit later load of the original template cannot retarget a plan.
        self.source.load("b")
        for index in (1, 0, 1, 0):
            np.testing.assert_allclose(plans[index].run().value, self.expected(("a", "b")[index]),
                                       rtol=1e-12, atol=1e-12)

    def test_callable_vertices_and_planning_do_not_read_numerical_blocks(self):
        e, p = self.elemental.load("a"), self.perambulator.load("a")
        calls = []
        templates = {"source": self.source, "sink": self.sink}
        def vertex(key):
            calls.append(key)
            return templates[key]
        with patch.object(e, "block", side_effect=AssertionError("elemental read while planning")), patch.object(
                p, "block", side_effect=AssertionError("propagator read while planning")):
            planned = self.compute(configuration="a", vertex_map=vertex, execute=False)
        self.assertEqual(calls, ["source", "sink"])
        np.testing.assert_allclose(planned.run().value, self.expected("a"), rtol=1e-12, atol=1e-12)

    def test_explicit_reader_stays_bound_and_failed_load_does_not_change_template(self):
        planned = self.compute(configuration="a", execute=False)
        with self.assertRaises(FileNotFoundError):
            self.compute(configuration="missing", execute=False)
        self.assertIsNone(self.source.reader)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.compute(execute=False)
        explicit = self.perambulator.load("a")
        with self.assertRaisesRegex(ValueError, "configuration|cfg"):
            self.compute(configuration="b", propagator_map={"q": explicit}, execute=False)
        np.testing.assert_allclose(planned.run().value, self.expected("a"), rtol=1e-12, atol=1e-12)

    def test_custom_adapter_bypasses_native_resolution_and_constants_do_not_load(self):
        tensor = np.ones((4, 4, 1, 1), complex)
        adapter = VSVData({0: 0, 1: 1},
                          lambda *args: TensorSpec(tensor.shape, "complex128"),
                          lambda *args: TensorSpec(tensor.shape, "complex128"),
                          lambda *args: tensor.copy(), lambda *args: tensor.copy(), "explicit")
        with patch.object(self.elemental, "load", side_effect=AssertionError("implicit elemental load")), patch.object(
                self.perambulator, "load", side_effect=AssertionError("implicit propagator load")):
            # This adapter explicitly rejects maps, and keeps that existing contract.
            with self.assertRaisesRegex(NotImplementedError, "handle maps"):
                self.compute(configuration="a", diagram_adapter=adapter, time_axes=None, batch="auto")
            value = calculate(self.diagram, diagram_adapter=adapter, optimize="none",
                              memory_limit=16 * 2**20).value
            self.assertEqual(value, 256)
            constant = calculate(3j, propagator_map={"q": self.perambulator},
                                 vertex_map={"source": self.source}, optimize="none").value
            self.assertEqual(constant, 3j)


if __name__ == "__main__":
    unittest.main()
