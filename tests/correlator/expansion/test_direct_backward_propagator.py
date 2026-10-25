"""Ordinary reversed edges read S(source,sink), without gamma5 reconstruction."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, QuarkDiagram
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calculate


class DirectBackwardPropagatorTests(unittest.TestCase):
    def test_reversed_physical_times_are_read_directly_with_selected_modes(self):
        rng = np.random.default_rng(6708)
        lt, ne, source_time, sink_time, used = 5, 3, 1, 4, 2
        # Every ordered pair is independently complex and non-Hermitian. A
        # gamma5-derived backward array cannot accidentally satisfy this oracle.
        physical = rng.normal(size=(lt, lt, 4, 4, ne, ne)) + 1j * rng.normal(
            size=(lt, lt, 4, 4, ne, ne))
        spatial = rng.normal(size=(1, 1, lt, ne, ne)) + 1j * rng.normal(size=(1, 1, lt, ne, ne))
        component = ElementalComponent.path(())
        operator = Operator.from_terms("local", [OperatorTerm(0, component, 0)])
        elemental = ArrayElemental({"v2v": spatial}, (component,), phase_convention="anchor")
        graph = Diagram(QuarkDiagram([[0, 1], [2, 0]]), [0, 1],
                        ["source", "sink"], [None, "forward", "backward"])
        vertices = {
            "source": Meson(elemental, operator, source=True, phase_convention="anchor"),
            "sink": Meson(elemental, operator, phase_convention="anchor"),
        }
        def matrix(value):
            return value[:, :, :used, :used].transpose(0, 2, 1, 3).reshape(4 * used, 4 * used)
        source = np.kron(np.eye(4), spatial[0, 0, source_time, :used, :used].conj().T)
        sink = np.kron(np.eye(4), spatial[0, 0, sink_time, :used, :used])
        forward = matrix(physical[sink_time, source_time])
        backward = matrix(physical[source_time, sink_time])
        expected = np.trace(source @ backward @ sink @ forward)
        gamma5 = np.kron(np.diag([1, 1, -1, -1]), np.eye(used))
        derived = np.trace(source @ (gamma5 @ forward.conj().T @ gamma5) @ sink @ forward)
        self.assertGreater(abs(expected - derived), 1)

        for convention in ("absolute", "relative"):
            with self.subTest(time_convention=convention):
                stored = np.empty_like(physical)
                for src in range(lt):
                    for snk in range(lt):
                        second = snk if convention == "absolute" else (snk - src) % lt
                        stored[src, second] = physical[snk, src]
                # Both graph roles deliberately use the very same reader.
                reader = ArrayPropagator({"vsv": stored}, time_convention=convention)
                calls = []
                original = reader.block
                def read(snk, src, sector="vsv", **selection):
                    calls.append((snk, src, sector, selection.copy()))
                    value = original(snk, src, sector, **selection)
                    np.testing.assert_array_equal(value, physical[snk, src, :, :, :used, :used])
                    return value
                with patch.object(reader, "block", side_effect=read):
                    result = calculate(graph, vertex_map=vertices,
                        propagator_map={"forward": reader, "backward": reader},
                        time_axes={0: source_time, 1: sink_time}, usedNe=used,
                        optimize="none", profile="default")
                self.assertEqual({(snk, src) for snk, src, _, _ in calls},
                                 {(sink_time, source_time), (source_time, sink_time)})
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(sec == "vsv" and selection == {"usedNe": used}
                                    for _, _, sec, selection in calls))
                np.testing.assert_allclose(result.value, expected, rtol=3e-13, atol=3e-13)
                prepared = result.expanded.inputs.preparations
                expected_pairs = {"forward": (sink_time, source_time),
                                  "backward": (source_time, sink_time)}
                for key, spec in result.expanded.problem.specs.items():
                    if spec.description.role == "propagator":
                        bindings = spec.description.bindings
                        self.assertEqual((bindings["sink_time"], bindings["source_time"]),
                            expected_pairs[spec.description.semantics["symbol"]])
                        self.assertEqual(spec.shape, (4, 4, used, used))
                        recipe = prepared[key].recipe
                        self.assertEqual(recipe.kind, "identity")
                        self.assertFalse(recipe.conjugate)
                        self.assertEqual(recipe.permutation, ())
                        self.assertEqual(recipe.constants, ())
