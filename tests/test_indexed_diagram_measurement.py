"""Read-only Diagram planning preserves registered scientific views."""

from contextlib import ExitStack
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contraction.binding.api import Meson, operator_term_semantics
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, SpectralProfile
from lqcdweave.operators.single_hadron.operator import OperatorPath
from lqcdweave.contraction.diagram_measurement import plan_indexed_diagram_measurement
from lqcdweave.contraction import diagram_measurement as planning_workflow
import test_published_double_current_diagram as published_fixture


class IndexedDiagramPlanningTests(unittest.TestCase):
    def _fixture(self):
        fixture = published_fixture.PublishedDoubleCurrentDiagramTests(
            "test_published_double_current_full_time_trace_uses_real_psp",
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        stack = ExitStack()
        self.addCleanup(stack.close)
        fixture._opened = {}
        source_symbol, sink_symbol = Symbol("source-axis"), Symbol("sink-axis")
        diagram, propagators, vertices = fixture._bindings_and_diagram(
            stack, source_symbol, sink_symbol,
        )
        bindings = DiagramBindings(
            propagators, vertices, "cfg-double-current", 1, fixture.point_count,
        )
        return fixture, diagram, bindings, source_symbol, sink_symbol

    def test_plan_uses_verified_actual_views_and_detached_science_snapshot(self):
        fixture, diagram, bindings, source_symbol, sink_symbol = self._fixture()
        caller_elementals = {
            id(vertex.elemental): vertex.elemental
            for vertex in bindings.vertex_map.values()
        }
        with ExitStack() as patches:
            for propagator in bindings.propagator_map.values():
                patches.enter_context(patch.object(
                    propagator, "block",
                    side_effect=AssertionError("metadata planning read propagator data"),
                ))
            for elemental in caller_elementals.values():
                patches.enter_context(patch.object(
                    elemental, "block",
                    side_effect=AssertionError("metadata planning read elemental data"),
                ))
            result = plan_indexed_diagram_measurement(
                fixture.index, diagram, bindings=bindings,
                source_time_symbol=source_symbol, sink_time_symbol=sink_symbol,
            )

        spec = result["measurement_spec"]
        self.assertEqual(spec.document["kind"], "diagram_measurement")
        self.assertEqual(spec.axis_names, ("source_time", "sink_time"))
        self.assertEqual(spec.coordinates("source_time"), list(fixture.times))
        self.assertEqual(spec.coordinates("sink_time"), list(fixture.times))
        self.assertEqual(spec.shape, (len(fixture.times), len(fixture.times)))
        input_nodes = set(result["input_roles"].values())
        self.assertTrue(input_nodes)
        self.assertEqual(input_nodes, set(result["graph"]["order"]) - {"measurement"})
        self.assertTrue(all(
            result["graph"]["stages"][node]["action"] == "reuse"
            for node in input_nodes
        ))
        self.assertEqual(set(spec.document["definition"]["input_views"]), input_nodes)
        self.assertEqual(spec.document["definition"]["science"]["usedNp"],
                         fixture.point_count)
        self.assertEqual(
            spec.document["definition"]["science"]["groups_independent"],
            "single-explicit-pool",
        )
        strict = plan_indexed_diagram_measurement(
            fixture.index, diagram, bindings=bindings,
            source_time_symbol=source_symbol, sink_time_symbol=sink_symbol,
            source_policy="require_known",
        )
        self.assertEqual(strict["graph"]["action"], "blocked")
        self.assertTrue(strict["graph"]["source_reasons"])
        prop_roles = [role for role in result["input_roles"]
                      if role.startswith("propagator-")]
        self.assertEqual({role.rsplit("-", 1)[1] for role in prop_roles},
                         {"vsv", "psv", "vsp", "psp"})
        self.assertEqual({role.split("-")[1] for role in prop_roles}, {"0", "1"})

        vertex = next(iter(bindings.vertex_map.values()))
        original_operator = vertex.operator
        changed_term = original_operator.terms[0]
        vertex.operator = OperatorPath(
            original_operator.name,
            ((changed_term.gamma, changed_term.component.key,
              changed_term.momentum, changed_term.coefficient + 0.125j,
              changed_term.anchor_time_offset),),
            current_terms=original_operator.current_terms,
            current_term_ranges=original_operator.current_term_ranges,
        )
        changed = plan_indexed_diagram_measurement(
            fixture.index, diagram, bindings=bindings,
            source_time_symbol=source_symbol, sink_time_symbol=sink_symbol,
        )
        self.assertNotEqual(spec.definition_id, changed["measurement_spec"].definition_id)
        self.assertEqual(
            spec.document["definition"]["science"]["vertices"][0]
            ["operator"]["terms"][0]["coefficient"], [0.6, 0.4],
        )
        vertex.operator = original_operator

    def test_custom_profile_identity_is_checked_during_planning(self):
        fixture, diagram, bindings, source_symbol, sink_symbol = self._fixture()
        vertex = next(iter(bindings.vertex_map.values()))
        original = vertex.operator
        original_term = original.terms[0]

        def canonical_profile_snapshot(weight):
            profile = SpectralProfile(
                np.full((1, 3, 1), weight, dtype=np.complex128),
                cfg="cfg-double-current", name="same-canonical-profile",
            )
            term = OperatorTerm(
                original_term.gamma, original_term.component,
                original_term.momentum, original_term.coefficient,
                original_term.anchor_time_offset, profile,
                current_term=original_term.current_term,
            )
            operator = Operator.from_terms(original.name, [term])
            meson = Meson(
                vertex.elemental, operator, source=vertex.dagger,
                phase_convention=vertex.phase_convention,
                boundary=vertex.boundary, array_module=vertex.xp,
            )
            return planning_workflow._operator_snapshot(meson)["terms"][0]["profile"]

        first = canonical_profile_snapshot(0.5)
        second = canonical_profile_snapshot(2.0)
        self.assertEqual(first["type"], "SpectralProfile")
        self.assertNotEqual(first["sha256"], second["sha256"])

        class CollidingProfile:
            def __init__(self, factor):
                self.factor = factor

            def __repr__(self):
                return "CollidingProfile(shared-repr)"

            def apply_block(self, value, *, times, cfg, array_module=np):
                return array_module.asarray(value) * self.factor

        profiles = (CollidingProfile(0.5), CollidingProfile(2.0))
        terms = [OperatorTerm(
            original_term.gamma, original_term.component, original_term.momentum,
            original_term.coefficient, original_term.anchor_time_offset,
            profile, current_term=original_term.current_term,
        ) for profile in profiles]
        self.assertEqual(operator_term_semantics(terms[0])["profile"],
                         operator_term_semantics(terms[1])["profile"])
        probe = np.ones((1, 1), dtype=np.complex128)
        self.assertFalse(np.array_equal(
            profiles[0].apply_block(probe, times=(0, 0), cfg="cfg-double-current"),
            profiles[1].apply_block(probe, times=(0, 0), cfg="cfg-double-current"),
        ))
        vertex.operator = Operator.from_terms(original.name, [terms[0]])
        with self.assertRaisesRegex(ValueError, "canonical profile identity"):
            plan_indexed_diagram_measurement(
                fixture.index, diagram, bindings=bindings,
                source_time_symbol=source_symbol, sink_time_symbol=sink_symbol,
            )


if __name__ == "__main__":
    unittest.main()
