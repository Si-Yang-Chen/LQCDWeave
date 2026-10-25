"""Independent scalar references for explicit Diagram sink batches."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

import test_point_elemental_diagram as point_fixture
import test_published_double_current_diagram as sparse_fixture
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contracts.correlator.api import BatchSpec
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calc_diagram_eval


class PersistentDiagramSinkBatchTests(unittest.TestCase):
    def _dense_vv_fixture(self):
        fixture = point_fixture.PublishedPointDiagramTests(
            "test_all_published_sectors_against_explicit_full_sample_trace"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        stack = ExitStack()
        self.addCleanup(stack.close)
        production = fixture.fixture
        elemental_reader = stack.enter_context(
            production.index.open(fixture.elemental_ids["v2v"])
        )
        propagator_reader = stack.enter_context(
            production.index.open(fixture.propagator_ids["vsv"])
        )
        elemental = DatasetElemental(elemental_reader)
        propagator = DatasetPropagator(propagator_reader)
        component = ElementalComponent.path(())
        normal = Operator.from_terms(
            "normal", [OperatorTerm(0, component, 0, 1)]
        )
        quark = Propagator("u", Tag(0, 0), Tag(1, 1))
        source_time, sink_time = Symbol("batch-source"), Symbol("batch-sink")
        diagram = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]),
            [source_time, sink_time], ["normal", "normal"], [None, quark],
        )
        meson = Meson(elemental, normal, phase_convention="anchor")
        bindings = dict(
            propagator_map={quark: propagator},
            vertex_map={"normal": meson},
            configuration=production.configuration["configuration"],
            usedNe=2,
        )
        return fixture, diagram, bindings, source_time, sink_time

    def _scalar_matrix(self, diagram, bindings, source_time, sink_time):
        values = []
        quark, propagator = next(iter(bindings["propagator_map"].items()))
        for source in range(3):
            row = []
            for sink in range(3):
                result = calc_diagram_eval(
                    diagram, propagator_map={quark: propagator},
                    vertex_map=bindings["vertex_map"],
                    configuration=bindings["configuration"],
                    usedNe=bindings["usedNe"],
                    time_axes={source_time: source, sink_time: sink},
                    optimize="none", backend=np,
                )
                value = np.asarray(result.value)
                self.assertEqual(value.shape, ())
                row.append(complex(value))
            values.append(row)
        return np.asarray(values, dtype=np.complex128)

    def test_registered_vv_sink_batches_match_scalar_cells(self):
        _fixture, diagram, bindings, source_time, sink_time = self._dense_vv_fixture()
        scalar_reference = self._scalar_matrix(
            diagram, bindings, source_time, sink_time,
        )
        batch = BatchSpec(2, sink_time)
        batched_values = np.empty_like(scalar_reference)
        layouts = []
        slabs = {"elemental": 0, "propagator": 0}
        elemental_read_slab = DatasetElemental.read_slab
        propagator_read_slab = DatasetPropagator.read_slab

        def count_elemental(self, *args, **kwargs):
            slabs["elemental"] += 1
            return elemental_read_slab(self, *args, **kwargs)

        def count_propagator(self, *args, **kwargs):
            slabs["propagator"] += 1
            return propagator_read_slab(self, *args, **kwargs)

        quark, propagator = next(iter(bindings["propagator_map"].items()))
        for source in range(3):
            with patch.object(DatasetElemental, "read_slab", new=count_elemental), \
                    patch.object(DatasetPropagator, "read_slab", new=count_propagator):
                result = calc_diagram_eval(
                    diagram, propagator_map={quark: propagator},
                    vertex_map=bindings["vertex_map"],
                    configuration=bindings["configuration"],
                    usedNe=bindings["usedNe"],
                    time_axes={source_time: source, sink_time: (0, 1, 2)},
                    batch=batch, optimize="none", backend=np,
                )
            row = np.asarray(result.value)
            self.assertEqual(row.shape, (3,))
            batched_values[source] = row
            layouts.append(tuple(
                item.coordinate_batch.actual_size
                for item in result.expanded.inputs.task_batches
            ))

        self.assertEqual(layouts, [(2, 1)] * 3)
        self.assertGreater(slabs["elemental"], 0)
        self.assertGreater(slabs["propagator"], 0)
        np.testing.assert_allclose(
            batched_values, scalar_reference, atol=2e-10, rtol=2e-10,
        )

    def test_sparse_p2p_sink_batches_match_full_time_reference(self):
        fixture = sparse_fixture.PublishedDoubleCurrentDiagramTests(
            "test_published_double_current_full_time_trace_uses_real_psp"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        stack = ExitStack()
        self.addCleanup(stack.close)
        fixture._opened = {}
        source_time, sink_time = Symbol("sparse-source"), Symbol("sparse-sink")
        diagram, propagators, vertices = fixture._bindings_and_diagram(
            stack, source_time, sink_time,
        )
        expected = fixture._independent_trace_values()
        batch = BatchSpec(2, sink_time)
        actual = np.empty((len(fixture.times), len(fixture.times)),
                          dtype=np.complex128)
        layouts = []

        for source in fixture.times:
            result = calc_diagram_eval(
                diagram, propagator_map=propagators, vertex_map=vertices,
                configuration="cfg-double-current", usedNe=1,
                usedNp=fixture.point_count,
                time_axes={source_time: source, sink_time: fixture.times},
                batch=batch, optimize="none", backend=np,
            )
            row = np.asarray(result.value)
            self.assertEqual(row.shape, (len(fixture.times),))
            actual[source] = row
            layouts.extend(
                item.coordinate_batch.actual_size
                for item in result.expanded.inputs.task_batches
            )

        self.assertIn(2, layouts)
        self.assertIn(1, layouts)
        for source in fixture.times:
            for sink in fixture.times:
                with self.subTest(times=(source, sink)):
                    np.testing.assert_allclose(
                        actual[source, sink], expected[source, sink],
                        rtol=3e-10, atol=3e-10,
                    )

    def test_sparse_sink_batch_anytime_measured_matches_independent_row(self):
        fixture = sparse_fixture.PublishedDoubleCurrentDiagramTests(
            "test_published_double_current_full_time_trace_uses_real_psp"
        )
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        stack = ExitStack()
        self.addCleanup(stack.close)
        fixture._opened = {}
        source_time, sink_time = Symbol("measured-source"), Symbol("measured-sink")
        diagram, propagators, vertices = fixture._bindings_and_diagram(
            stack, source_time, sink_time,
        )
        expected = fixture._independent_trace_values()
        calculation = calc_diagram_eval(
            diagram, propagator_map=propagators, vertex_map=vertices,
            configuration="cfg-double-current", usedNe=1, usedNp=fixture.point_count,
            time_axes={source_time: fixture.times[0], sink_time: fixture.times},
            batch=BatchSpec(2, sink_time), optimize="anytime", backend=np,
            measure=True,
        )
        value = np.asarray(calculation.value)
        self.assertEqual(value.shape, (len(fixture.times),))
        self.assertTrue(calculation.options["measure"])
        self.assertIsNotNone(calculation.report)
        np.testing.assert_allclose(
            value,
            np.asarray([expected[fixture.times[0], sink] for sink in fixture.times]),
            rtol=3e-10, atol=3e-10,
        )


if __name__ == "__main__":
    unittest.main()
