"""Native E tensors expose logical axes and physical binding identities."""

import unittest
import pickle
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.elemental import ArrayBaryonElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contraction.binding.api import Baryon, Meson
from lqcdweave.contraction.expansion.baryon_diagram import (
    BaryonDiagramBinding as ImplementationBaryonDiagramBinding,
)
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding
from lqcdweave.contraction.expansion.legacy_diagram import LegacyDiagramBinding
from lqcdweave.contracts.correlator.api import DiagramBindings, describe_coordinates
from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.api import (
    Operator, OperatorTerm, SparseSpinTensor,
)


class TensorDescriptionCatalogTests(unittest.TestCase):
    def test_public_baryon_binding_preserves_class_and_pickle_module(self):
        self.assertIs(BaryonDiagramBinding, ImplementationBaryonDiagramBinding)
        self.assertEqual(
            BaryonDiagramBinding.__module__,
            "lqcdweave.contraction.expansion.baryon_diagram",
        )
        original = BaryonDiagramBinding(sector_scope="dense-v")
        restored = pickle.loads(pickle.dumps(original))
        self.assertIs(type(restored), BaryonDiagramBinding)

    def test_legacy_provider_describes_every_logical_tensor_without_reads(self):
        component = ElementalComponent.path(())
        operator = Operator.from_terms(
            "local", (OperatorTerm(0, component, 0, 0.5 + 0.25j),),
        )
        elemental = ArrayElemental(
            {"v2v": np.zeros((1, 1, 3, 1, 1), dtype=np.complex128)},
            (component,), phase_convention="anchor", cfg="cfg-a",
            data_identity="cfg-a/elemental/basis-ordered",
        )
        propagator = ArrayPropagator(
            {"vsv": np.zeros((3, 3, 4, 4, 1, 1), dtype=np.complex128)},
            time_convention="absolute", cfg="cfg-a",
            data_identity="cfg-a/perambulator/basis-ordered",
        )
        vertices = {
            "src": Meson(elemental, operator, source=True, phase_convention="anchor"),
            "snk": Meson(elemental, operator, phase_convention="anchor"),
        }
        graph = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]),
            [0, 2], ["src", "snk"], [None, "p-forward", "p-backward"],
        )
        bindings = DiagramBindings(
            {"p-forward": propagator, "p-backward": propagator},
            vertices, "cfg-a", usedNe=1,
        )
        coordinates, batch = describe_coordinates()

        with patch.object(elemental, "block", side_effect=AssertionError("E read elemental values")), \
             patch.object(propagator, "block", side_effect=AssertionError("E read propagator values")):
            inputs = LegacyDiagramBinding().describe(
                graph,
                bindings=bindings,
                coordinates=coordinates,
                batch=batch,
            )

        self.assertTrue(inputs.problem.terms)
        self.assertEqual(len(inputs.problem.specs), len(inputs.tensor_requests))
        for tensor_id, spec in inputs.problem.specs.items():
            description = spec.description
            self.assertIsNotNone(description, tensor_id)
            self.assertEqual(len(description.axes), len(spec.shape))
            self.assertTrue(description.role)
            self.assertTrue(description.source_bindings)
            self.assertEqual(description.bindings["configuration"], "cfg-a")
        vertex_specs = [spec for spec in inputs.problem.specs.values()
                        if spec.description.role == "elemental"]
        propagator_specs = [spec for spec in inputs.problem.specs.values()
                            if spec.description.role == "propagator"]
        self.assertTrue(vertex_specs)
        self.assertTrue(all(spec.description.axes[:2] == ("spin_field", "spin_bar")
                            for spec in vertex_specs))
        self.assertTrue(all(spec.description.data_identity == "cfg-a/elemental/basis-ordered"
                            for spec in vertex_specs))
        self.assertTrue(propagator_specs)
        self.assertTrue(all(spec.description.bindings["source_time"] in {0, 2}
                            for spec in propagator_specs))
        self.assertTrue(all(
            check["status"] == "unknown" and check["knowledge"] == "both-unbound"
            for spec in propagator_specs
            for check in spec.description.bindings["endpoint_axis_bindings"].values()
        ))

    def _bound_mode_fixture(self, *, mismatch=None):
        component = ElementalComponent.path(())
        operator = Operator.from_terms(
            "local", (OperatorTerm(0, component, 0),),
        )
        elemental = ArrayElemental(
            {"v2v": np.zeros((1, 1, 3, 2, 2), dtype=np.complex128)},
            (component,), phase_convention="anchor", cfg="cfg-axis",
            data_identity="cfg-axis/elemental",
        )
        propagator = ArrayPropagator(
            {"vsv": np.zeros((3, 3, 4, 4, 2, 2), dtype=np.complex128)},
            time_convention="absolute", cfg="cfg-axis",
            data_identity="cfg-axis/perambulator",
        )
        elemental_describe = elemental.describe_block
        propagator_describe = propagator.describe_block

        def describe_elemental(*args, **kwargs):
            metadata = elemental_describe(*args, **kwargs)
            return replace(metadata, axes=("sink_mode", "source_mode"), axis_bindings={
                "sink_mode": ("basis-sink", (1, 2)),
                "source_mode": ("basis-source", (7, 3)),
            })

        def describe_propagator(sink_time, source_time, sector="vsv", **kwargs):
            metadata = propagator_describe(sink_time, source_time, sector, **kwargs)
            actual_id = "basis-source" if (sink_time, source_time) == (2, 0) else "basis-sink"
            coordinates = (7, 3) if actual_id == "basis-source" else (1, 2)
            if mismatch == "mode-id" and (sink_time, source_time) == (2, 0):
                actual_id = "different-actual-mode-basis"
            if mismatch == "mode-order" and (sink_time, source_time) == (2, 0):
                coordinates = (3, 7)
            return replace(metadata, axis_bindings={
                "sink_mode": (actual_id, coordinates),
                "source_mode": (actual_id, coordinates),
            })

        elemental.describe_block = describe_elemental
        propagator.describe_block = describe_propagator
        vertices = {
            "src": Meson(elemental, operator, source=True, phase_convention="anchor"),
            "snk": Meson(elemental, operator, phase_convention="anchor"),
        }
        quark_forward = Propagator("u", Tag(0, 0), Tag(1, 2))
        quark_backward = Propagator("u", Tag(1, 2), Tag(0, 0))
        graph = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]),
            [0, 2], ["src", "snk"], [None, quark_forward, quark_backward],
        )
        bindings = DiagramBindings(
            {quark_forward: propagator, quark_backward: propagator},
            vertices, "cfg-axis", usedNe=2,
        )
        coordinates, batch = describe_coordinates()
        return graph, elemental, propagator, bindings, coordinates, batch

    def test_daggered_field_and_bar_bindings_match_opposite_raw_axes(self):
        graph, elemental, propagator, bindings, coordinates, batch = self._bound_mode_fixture()
        with patch.object(elemental, "block", side_effect=AssertionError("E read elemental values")), \
             patch.object(propagator, "block", side_effect=AssertionError("E read propagator values")):
            inputs = LegacyDiagramBinding().describe(
                graph, bindings=bindings, coordinates=coordinates, batch=batch,
            )
        propagator_specs = [spec for spec in inputs.problem.specs.values()
                            if spec.description.role == "propagator"]
        self.assertTrue(propagator_specs)
        self.assertTrue(all(
            check["status"] == "verified"
            for spec in propagator_specs
            for check in spec.description.bindings["endpoint_axis_bindings"].values()
        ))
        source = next(spec for spec in propagator_specs
                      if spec.description.bindings["source_time"] == 0)
        self.assertEqual(source.description.bindings["endpoint_axis_bindings"]["source"]["meson_axis"],
                         "source_mode")

    def test_same_shape_but_different_actual_mode_id_is_rejected_before_reads(self):
        graph, elemental, propagator, bindings, coordinates, batch = self._bound_mode_fixture(
            mismatch="mode-id",
        )
        with patch.object(elemental, "block", side_effect=AssertionError("E read elemental values")), \
             patch.object(propagator, "block", side_effect=AssertionError("E read propagator values")):
            with self.assertRaisesRegex(ValueError, "endpoint actual dataset ID"):
                LegacyDiagramBinding().describe(
                    graph, bindings=bindings, coordinates=coordinates, batch=batch,
                )

    def test_same_shape_but_different_selected_mode_order_is_rejected(self):
        graph, elemental, propagator, bindings, coordinates, batch = self._bound_mode_fixture(
            mismatch="mode-order",
        )
        with patch.object(elemental, "block", side_effect=AssertionError("E read elemental values")), \
             patch.object(propagator, "block", side_effect=AssertionError("E read propagator values")):
            with self.assertRaisesRegex(ValueError, "endpoint actual dataset ID or ordered coordinates"):
                LegacyDiagramBinding().describe(
                    graph, bindings=bindings, coordinates=coordinates, batch=batch,
                )

    def test_one_unbound_endpoint_remains_unknown_in_description(self):
        graph, elemental, propagator, bindings, coordinates, batch = self._bound_mode_fixture()
        original_describe = elemental.describe_block

        def describe_unbound(*args, **kwargs):
            metadata = original_describe(*args, **kwargs)
            return replace(metadata, axis_bindings={})

        elemental.describe_block = describe_unbound
        with patch.object(elemental, "block", side_effect=AssertionError("E read elemental values")), \
             patch.object(propagator, "block", side_effect=AssertionError("E read propagator values")):
            inputs = LegacyDiagramBinding().describe(
                graph, bindings=bindings, coordinates=coordinates, batch=batch,
            )
        propagator_specs = [spec for spec in inputs.problem.specs.values()
                            if spec.description.role == "propagator"]
        checks = [check for spec in propagator_specs
                  for check in spec.description.bindings["endpoint_axis_bindings"].values()]
        self.assertTrue(checks)
        self.assertTrue(all(check["status"] == "unknown"
                            and check["knowledge"] == "one-unbound" for check in checks))

    def test_baryon_tensor_description_keeps_ordered_spin_modes_and_real_times(self):
        component = BaryonComponent.paths()
        spin = SparseSpinTensor([(0, 1, 2, 1.0 + 0.25j)])
        operator = Operator.from_terms(
            "baryon", (OperatorTerm.from_baryon(component, spin),),
        )
        source_reader = ArrayBaryonElemental(
            {"vvv": np.zeros((1, 1, 3, 1, 1, 1), dtype=np.complex128)},
            (component,), cfg="cfg-b", data_identity="cfg-b/source-baryon",
        )
        sink_reader = ArrayBaryonElemental(
            {"vvv": np.zeros((1, 1, 3, 1, 1, 1), dtype=np.complex128)},
            (component,), cfg="cfg-b", data_identity="cfg-b/sink-baryon",
        )
        source = Baryon(source_reader, operator, source=True)
        sink = Baryon(sink_reader, operator)
        propagator = ArrayPropagator(
            {"vsv": np.zeros((3, 3, 4, 4, 1, 1), dtype=np.complex128)},
            time_convention="absolute", cfg="cfg-b",
            data_identity="cfg-b/perambulator",
        )
        zero = [[0, 0, 0] for _ in range(3)]
        edges = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        graph = Diagram(
            QuarkDiagram([[zero, edges], [zero, zero]]),
            [0, 2], ["source", "sink"], [None, "p0", "p1", "p2"],
        )
        bindings = DiagramBindings(
            {name: propagator for name in ("p0", "p1", "p2")},
            {"source": source, "sink": sink}, "cfg-b", usedNe=1,
        )
        coordinates, batch = describe_coordinates()

        with patch.object(source_reader, "block", side_effect=AssertionError("E read baryon values")), \
             patch.object(sink_reader, "block", side_effect=AssertionError("E read baryon values")), \
             patch.object(propagator, "block", side_effect=AssertionError("E read propagator values")):
            inputs = BaryonDiagramBinding(sector_scope="dense-v", vertices="joined").describe(
                graph, bindings=bindings, coordinates=coordinates, batch=batch,
            )

        specs = tuple(inputs.problem.specs.values())
        self.assertTrue(all(spec.description is not None for spec in specs))
        baryon_specs = [spec for spec in specs
                        if spec.description.role == "baryon-elemental"]
        self.assertEqual(len(baryon_specs), 2)
        source_spec = next(spec for spec in baryon_specs
                           if spec.description.bindings["endpoint_times"] == (0, 0, 0))
        self.assertEqual(
            source_spec.description.axes,
            ("mode0", "mode1", "mode2"),
        )
        self.assertEqual(source_spec.description.bindings["field_order"], (2, 1, 0))
        sink_spec = next(spec for spec in baryon_specs
                         if spec.description.bindings["endpoint_times"] == (2, 2, 2))
        self.assertEqual(sink_spec.description.bindings["field_order"], (0, 1, 2))
        self.assertEqual(source_spec.description.semantics["operator_term"]["component"]["rank"], 3)
        spin_specs = [spec for spec in specs if spec.description.role == "baryon-spin"]
        self.assertEqual(len(spin_specs), 2)
        self.assertTrue(all(spec.shape == (4, 4, 4) for spec in spin_specs))
        self.assertTrue(all(spec.axes == ("spin_slot0", "spin_slot1", "spin_slot2")
                            for spec in spin_specs))
        self.assertTrue(all(spec.data_identity is None for spec in spin_specs))
        self.assertEqual(len(inputs.problem.baryon_joins), 2)
        for join in inputs.problem.baryon_joins:
            spin_spec = inputs.problem.specs[join.spin]
            self.assertEqual(join.sparse_spin, spin_spec.description.semantics["entries"])
            self.assertEqual(inputs.preparations[join.spin].blocks, ())
            self.assertEqual(inputs.problem.sources[join.spin].read_bytes, 0)
            # Constant preparation has no reader dependency and preserves every
            # exact coefficient, independently of the surrounding spatial block.
            with patch.object(source_reader, "block", side_effect=AssertionError("spin read elemental")), \
                 patch.object(sink_reader, "block", side_effect=AssertionError("spin read elemental")), \
                 patch.object(source, "describe_spin_entries", side_effect=AssertionError("spin recomputed its basis")), \
                 patch.object(sink, "describe_spin_entries", side_effect=AssertionError("spin recomputed its basis")):
                dense = inputs.loader.materialize(join.spin, inputs.problem.sources[join.spin], {})
            expected = np.zeros((4, 4, 4), complex)
            for a, b, c, coefficient in join.sparse_spin:
                expected[a, b, c] = coefficient
            np.testing.assert_array_equal(dense, expected)


if __name__ == "__main__":
    unittest.main()
