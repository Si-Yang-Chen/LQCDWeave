"""Read-only Diagram planning and direct calculations over published VVV inputs."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

from lqcdweave.adapters.baryon_dataset import DatasetBaryonElemental
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contraction.binding.api import Baryon, Meson
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.baryon import SparseSpinTensor
from lqcdweave.operators.single_hadron.baryon import BaryonOperatorTerm
from lqcdweave.operators.single_hadron.current import LocalVectorCurrent
from lqcdweave.operators.single_hadron.operator import (
    Operator, OperatorTerm,
)
from lqcdweave.storage.dataset import DatasetIndex, DatasetReader, DatasetWriter
from lqcdweave.contraction import diagram_measurement as workflow
from lqcdweave.contraction.correlator import calc_diagram_eval

from test_baryon_dataset_binding import (
    _CONFIG, _axis, _baryon_spec, _base_specs, _publish,
)
from test_baryon_diagram_frontend import _mixed_case


def _spin(offset):
    return SparseSpinTensor(((0, 1, 2, 0.5 + 0.25j * offset),
                             (3, 2, 1, -0.25 + 0.125j * offset)))


def _vsv_spec(gauge_id, basis_id):
    labels = [7]
    return DatasetSpec(dict(
        schema=1, kind="vsv", configuration=_CONFIG,
        axes=[_axis("source_time", range(2)), _axis("sink_time", range(2)),
              _axis("sink_spin", range(4)), _axis("source_spin", range(4)),
              _axis("sink_mode", labels), _axis("source_mode", labels)],
        dtype="complex128",
        bases={"sink_mode": dict(data_id=basis_id, indices=labels),
               "source_mode": dict(data_id=basis_id, indices=labels)},
        definition=dict(gauge_id=gauge_id, solver_tolerance=1e-9,
                        dirac={"action": "Wilson", "mass": 0.17}),
        conventions=dict(normalization="unscaled", spin_basis="degrand_rossi",
                         temporal_boundary="antiperiodic"),
        parents=[basis_id, gauge_id],
    ))


def _diagram(source_symbol, sink_symbol):
    zero = [[0, 0, 0] for _ in range(3)]
    edge = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    return Diagram(
        QuarkDiagram([[zero, edge], [zero, zero]]),
        [source_symbol, sink_symbol], ["source", "sink"],
        [None,
         Propagator("u", Tag(0, 0), Tag(1, 1)),
         Propagator("d", Tag(0, 1), Tag(1, 2)),
         Propagator("s", Tag(0, 2), Tag(1, 3))],
    )


class PublishedBaryonDiagramTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.gauge, self.gauge_id, self.basis, self.basis_id = _base_specs(
            self.root, self.index,
        )
        self.rng = np.random.default_rng(5128)
        self.opened = ExitStack()
        self.addCleanup(self.opened.close)

    def _publish_baryon_vertex(self, name, offset):
        spec = _baryon_spec(self.gauge, self.gauge_id, self.basis, self.basis_id)
        values = (self.rng.normal(size=spec.shape) +
                  1j * self.rng.normal(size=spec.shape)).astype(np.complex128)
        identity = _publish(self.root, self.index, name, spec, values)
        reader = self.opened.enter_context(self.index.open(identity))
        elemental = DatasetBaryonElemental(reader)
        component = elemental.components[0]
        term = BaryonOperatorTerm(
            _spin(offset), component, 0.7 + 0.2j * offset, momentum=0,
        )
        operator = Operator.from_terms(name, (term,))
        vertex = Baryon(
            elemental, operator, source=(name == "baryon-source"),
            phase_convention="anchor", spin_basis="degrand_rossi",
            boundary="periodic", array_module=np,
        )
        return vertex, values

    def _publish_propagators(self):
        propagators = {}
        for position, (flavor, source_tag, sink_tag) in enumerate(
                (("u", 0, 1), ("d", 1, 2), ("s", 2, 3))):
            spec = _vsv_spec(self.gauge_id, self.basis_id)
            values = (self.rng.normal(size=spec.shape) +
                      1j * self.rng.normal(size=spec.shape)).astype(np.complex128)
            identity = _publish(self.root, self.index, f"vsv-{position}", spec, values)
            reader = self.opened.enter_context(self.index.open(identity))
            symbol = Propagator(flavor, Tag(0, source_tag), Tag(1, sink_tag))
            propagators[symbol] = DatasetPropagator(reader)
        return propagators

    def _publish_mixed_inputs(self):
        """Publish the existing in-memory mixed fixture under exact VVV/VV/VSV schemas."""
        case = _mixed_case((0, 1, 2))
        configuration = dict(
            ensemble="published-baryon-mixed",
            configuration="cfg-baryon-mixed",
            lattice=[1, 1, 1, 5],
        )
        times = list(range(5))
        gauge = DatasetSpec(dict(
            schema=1, kind="gauge", configuration=configuration,
            axes=[_axis("time", times), _axis("z", [0]), _axis("y", [0]),
                  _axis("x", [0]), _axis("direction", range(4)),
                  _axis("color_row", range(3)), _axis("color_column", range(3))],
            dtype="complex128", bases={}, definition={},
            conventions={"boundary": "periodic"}, parents=[],
        ))
        gauge_id = _publish(
            self.root, self.index, "mixed-gauge", gauge,
            np.zeros(gauge.shape, dtype=np.complex128),
        )
        labels = [7, 2]
        basis = DatasetSpec(dict(
            schema=1, kind="explicit_basis", configuration=configuration,
            axes=[_axis("time", times), _axis("mode", labels), _axis("z", [0]),
                  _axis("y", [0]), _axis("x", [0]), _axis("color", range(3))],
            dtype="complex128", bases={}, definition={"basis_source": "mixed-fixture"},
            conventions={}, parents=[gauge_id],
        ))
        basis_id = _publish(
            self.root, self.index, "mixed-basis", basis,
            np.zeros(basis.shape, dtype=np.complex128),
        )

        def publish_rows(name, spec, rows):
            directory = self.root / name
            with DatasetWriter(directory, spec, provenance={"test": "published-baryon-mixed"}) as writer:
                for position, time in enumerate(spec.coordinates(spec.axis_names[0])):
                    writer.write(position, np.asarray(rows[time], dtype=spec.document["dtype"]))
                identity = writer.publish()
            self.index.register(directory)
            reader = self.opened.enter_context(self.index.open(identity))
            return reader

        def publish_baryon(name, vertex):
            reader = vertex.elemental
            component = reader.components[0]
            encoded = dict(encoding="ordered_path_triple_v1",
                           paths=[list(path.steps) for path in component.key])
            definition = dict(
                gauge_id=gauge_id, gauge_purpose="elemental_path_transport",
                product="raw_paths", component_encoding="ordered_path_triples_v1",
                input_roles={"gauge": gauge_id, "basis_leg0": basis_id,
                             "basis_leg1": basis_id, "basis_leg2": basis_id},
                input_views={"gauge": gauge.document, "basis_leg0": basis.document,
                             "basis_leg1": basis.document, "basis_leg2": basis.document},
            )
            spec = DatasetSpec(dict(
                schema=1, kind="elemental_baryon_vvv", configuration=configuration,
                axes=[_axis("time", times), _axis("component", [encoded]),
                      _axis("momentum", [[0, 0, 0]]),
                      _axis("mode0", labels), _axis("mode1", labels), _axis("mode2", labels)],
                dtype="complex128",
                bases={f"mode{leg}": dict(data_id=basis_id, indices=labels)
                       for leg in range(3)},
                definition=definition,
                conventions=dict(normalization="unscaled", phase="anchor",
                                 path_boundary="periodic", color_contraction="epsilon",
                                 vector_conjugation="none"),
                parents=[gauge_id, basis_id],
            ))
            values = reader.sectors["vvv"]
            rows = {time: values[:, :, time % values.shape[2]] for time in times}
            return publish_rows(name, spec, rows)

        old_vertices = case["vertices"]
        source_reader = publish_baryon("mixed-baryon-source", old_vertices["source"])
        sink_reader = publish_baryon("mixed-baryon-sink", old_vertices["sink"])
        source = Baryon(
            DatasetBaryonElemental(source_reader), old_vertices["source"].operator,
            source=old_vertices["source"].dagger,
            phase_convention=old_vertices["source"].phase_convention,
            spin_basis=old_vertices["source"].spin_basis,
            boundary=old_vertices["source"].boundary,
        )
        sink = Baryon(
            DatasetBaryonElemental(sink_reader), old_vertices["sink"].operator,
            source=old_vertices["sink"].dagger,
            phase_convention=old_vertices["sink"].phase_convention,
            spin_basis=old_vertices["sink"].spin_basis,
            boundary=old_vertices["sink"].boundary,
        )

        current_operator = LocalVectorCurrent().operator(0)
        current_components = tuple(dict.fromkeys(
            term.component for term in current_operator.terms
        ))
        component_coordinates = [dict(path=list(component.key.steps))
                                 for component in current_components]
        v2v_spec = DatasetSpec(dict(
            schema=1, kind="elemental_vv", configuration=configuration,
            axes=[_axis("time", times), _axis("component", component_coordinates),
                  _axis("momentum", [[0, 0, 0]]),
                  _axis("sink_mode", labels), _axis("source_mode", labels)],
            dtype="complex128",
            bases={name: dict(data_id=basis_id, indices=labels)
                   for name in ("sink_mode", "source_mode")},
            definition=dict(gauge_id=gauge_id, path_boundary="periodic"),
            conventions=dict(normalization="unscaled", phase="anchor"),
            parents=[gauge_id, basis_id],
        ))
        rng = np.random.default_rng(90124)
        v2v_values = (rng.normal(size=(len(current_components), 1, 5, 2, 2)) +
                      1j * rng.normal(size=(len(current_components), 1, 5, 2, 2)))
        v2v_values = v2v_values.astype(np.complex128)
        v2v_reader = publish_rows(
            "mixed-meson-v2v", v2v_spec,
            {time: v2v_values[:, :, time] for time in times},
        )
        current = Meson(
            DatasetElemental(v2v_reader), current_operator,
            phase_convention="anchor",
        )

        propagators = {}
        readers_by_name = {}
        for name in case["propagators"]:
            array = case["arrays"][name.removeprefix("p-")]
            spec = DatasetSpec(dict(
                schema=1, kind="vsv", configuration=configuration,
                axes=[_axis("source_time", times), _axis("sink_time", times),
                      _axis("sink_spin", range(4)), _axis("source_spin", range(4)),
                      _axis("sink_mode", labels), _axis("source_mode", labels)],
                dtype="complex128",
                bases={name: dict(data_id=basis_id, indices=labels)
                       for name in ("sink_mode", "source_mode")},
                definition=dict(gauge_id=gauge_id, solver_tolerance=1e-9,
                                dirac={"action": "Wilson", "mass": 0.17}),
                conventions=dict(normalization="unscaled", spin_basis="degrand_rossi",
                                 temporal_boundary="antiperiodic"),
                parents=[basis_id, gauge_id],
            ))
            rows = {time: array[time] for time in times}
            reader = publish_rows(f"mixed-{name}", spec, rows)
            readers_by_name[name] = DatasetPropagator(reader)

        p_in = Propagator("u", Tag(2, 0), Tag(0, 1))
        p_direct0 = Propagator("u", Tag(0, 0), Tag(1, 2))
        p_direct1 = Propagator("d", Tag(1, 0), Tag(2, 2))
        p_out = Propagator("u", Tag(0, 1), Tag(3, 2))
        propagator_symbols = {
            "p-in": p_in, "p-direct0": p_direct0,
            "p-direct1": p_direct1, "p-out": p_out,
        }
        propagators = {propagator_symbols[name]: reader
                       for name, reader in readers_by_name.items()}

        source_time, sink_time = Symbol("mixed-source-time"), Symbol("mixed-sink-time")
        original = case["diagram"]
        diagram = Diagram(
            original.diagram, [source_time, 1, sink_time],
            original.vertex_list, [None, p_in, p_direct0, p_direct1, p_out],
        )
        vertices = {"source": source, "current": current, "sink": sink}
        bindings = DiagramBindings(
            propagators, vertices, configuration["configuration"],
            usedNe=(1, 1, 1),
        )
        return diagram, bindings, source_time, sink_time

    def _fixture(self):
        source, _source_values = self._publish_baryon_vertex("baryon-source", 2)
        sink, _sink_values = self._publish_baryon_vertex("baryon-sink", 5)
        propagators = self._publish_propagators()
        source_time, sink_time = Symbol("baryon-source-time"), Symbol("baryon-sink-time")
        diagram = _diagram(source_time, sink_time)
        bindings = DiagramBindings(
            propagators, {"source": source, "sink": sink},
            _CONFIG["configuration"], usedNe=1,
        )
        return diagram, bindings, source_time, sink_time

    def test_baryon_plan_preserves_full_time_contract_for_direct_calculation(self):
        diagram, bindings, source_time, sink_time = self._fixture()
        planned = workflow.plan_indexed_diagram_measurement(
            self.index, diagram, bindings=bindings,
            source_time_symbol=source_time, sink_time_symbol=sink_time,
        )
        spec = planned["measurement_spec"]
        self.assertEqual(spec.axis_names, ("source_time", "sink_time"))
        self.assertEqual(spec.shape, (2, 2))
        self.assertEqual(spec.document["definition"]["science"]["vertices"][0]
                         ["operator"]["spin_basis"], "degrand_rossi")
        self.assertTrue(any(
            record["definition"].get("product") == "raw_paths"
            for vertex in spec.document["definition"]["science"]["vertices"]
            for record in vertex["sectors"]
        ))
        values = np.empty(spec.shape, dtype=np.complex128)
        for source_value in spec.coordinates("source_time"):
            for sink_value in spec.coordinates("sink_time"):
                values[source_value, sink_value] = calc_diagram_eval(
                    diagram,
                    propagator_map=bindings.propagator_map,
                    vertex_map=bindings.vertex_map,
                    configuration=bindings.configuration,
                    usedNe=bindings.usedNe,
                    time_axes={source_time: source_value, sink_time: sink_value},
                    optimize="none", backend=np,
                ).value
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(np.linalg.norm(values), 0)

    def test_baryon_endpoint_identity_is_rejected_during_read_only_planning(self):
        diagram, bindings, source_time, sink_time = self._fixture()
        symbol = next(iter(bindings.propagator_map))
        wrong = _vsv_spec(self.gauge_id, "different-actual-basis")
        values = np.zeros(wrong.shape, dtype=np.complex128)
        identity = _publish(self.root, self.index, "wrong-vsv", wrong, values)
        wrong_reader = self.opened.enter_context(self.index.open(identity))
        wrong_propagator = DatasetPropagator(wrong_reader)
        bad_map = dict(bindings.propagator_map)
        bad_map[symbol] = wrong_propagator
        bad_bindings = DiagramBindings(
            bad_map, bindings.vertex_map, bindings.configuration, bindings.usedNe,
        )
        with patch.object(DatasetReader, "read",
                          side_effect=AssertionError("planning read numeric dataset data")):
            with self.assertRaisesRegex(ValueError, "endpoint actual dataset ID"):
                workflow.plan_indexed_diagram_measurement(
                    self.index, diagram, bindings=bad_bindings,
                    source_time_symbol=source_time, sink_time_symbol=sink_time,
                )

    def test_mixed_mesons_reduce_equal_three_leg_mode_prefix_during_planning(self):
        diagram, bindings, source_time, sink_time = self._publish_mixed_inputs()
        planned = workflow.plan_indexed_diagram_measurement(
            self.index, diagram, bindings=bindings,
            source_time_symbol=source_time, sink_time_symbol=sink_time,
        )
        self.assertEqual(planned["measurement_spec"].shape, (5, 5))
        values = np.empty((5, 5), dtype=np.complex128)
        for source_value in range(5):
            for sink_value in range(5):
                values[source_value, sink_value] = calc_diagram_eval(
                    diagram,
                    propagator_map=bindings.propagator_map,
                    vertex_map=bindings.vertex_map,
                    configuration=bindings.configuration,
                    usedNe=bindings.usedNe,
                    time_axes={source_time: source_value, sink_time: sink_value},
                    optimize="none", backend=np,
                ).value
        self.assertTrue(np.isfinite(values).all())

        unequal = DiagramBindings(
            bindings.propagator_map, bindings.vertex_map, bindings.configuration,
            usedNe=(1, 2, 1),
        )
        with self.assertRaisesRegex(ValueError, "equal per-leg usedNe"):
            workflow.plan_indexed_diagram_measurement(
                self.index, diagram, bindings=unequal,
                source_time_symbol=source_time, sink_time_symbol=sink_time,
            )

        from lqcdweave.contracts.correlator.api import BatchSpec

        batched = workflow.plan_indexed_diagram_measurement(
            self.index, diagram, bindings=bindings,
            source_time_symbol=source_time, sink_time_symbol=sink_time,
            sink_batch=BatchSpec(1, sink_time),
        )
        self.assertEqual(batched["measurement_spec"].shape, planned["measurement_spec"].shape)


if __name__ == "__main__":
    unittest.main()
