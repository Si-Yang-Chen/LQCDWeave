"""Published point products in a two-current Diagram, checked independently."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from test_wilson_pipeline import random_gauge, stencil
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.current import CurrentTerm
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import OperatorPath
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from fixtures.c_group_production import (
    publish_elemental_products,
    publish_perambulators,
)
from lqcdweave.contraction.correlator import calc_diagram_eval


class PublishedDoubleCurrentDiagramTests(unittest.TestCase):
    lattice = (4, 1, 1, 3)
    times = (0, 1, 2)
    time_order = (2, 0, 1)
    point_count = 4
    mode_label = 41
    pool_id = "explicit-census"
    coefficient = 0.6 + 0.4j
    mass = 0.4
    boundary = -1

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.links = random_gauge(self.lattice, seed=2501)
        self.configuration = dict(
            ensemble="published-point-reference",
            configuration="cfg-double-current",
            lattice=list(self.lattice),
        )
        self.basis_values = self._make_basis_values()
        self.gauge_id = self._publish_gauge()
        self.basis_id = self._publish_basis()
        self.points_a_id = self._publish_points(
            "points-a", labels=(9, 3, 17, 1),
        )
        self.points_b_id = self._publish_points(
            "points-b", labels=(4, 8, 2, 6),
        )
        self.assertNotEqual(self.points_a_id, self.points_b_id)
        with self.index.open(self.points_a_id) as points_a, self.index.open(self.points_b_id) as points_b:
            self.assertNotEqual(points_a.spec.coordinates("point"),
                                points_b.spec.coordinates("point"))
            for time in self.times:
                np.testing.assert_array_equal(points_a.read(time=time), points_b.read(time=time))
                np.testing.assert_array_equal(points_a.read(time=time)[:, 0],
                                              np.arange(self.point_count))

        self.v2v_id = self._publish_v2v()
        self.elemental_a_ids = self._publish_point_elementals("elementals-a", self.points_a_id)
        self.elemental_b_ids = self._publish_point_elementals("elementals-b", self.points_b_id)
        self.propagator_ab_ids = self._publish_propagators(
            "propagators-a-to-b", sink_points=self.points_b_id,
            source_points=self.points_a_id,
        )
        self.propagator_ba_ids = self._publish_propagators(
            "propagators-b-to-a", sink_points=self.points_a_id,
            source_points=self.points_b_id,
        )
        self.assertEqual(set(self.propagator_ab_ids) & {"vsv", "psv", "vsp", "psp"},
                         {"vsv", "psv", "vsp", "psp"})
        self.assertEqual(set(self.propagator_ba_ids) & {"vsv", "psv", "vsp", "psp"},
                         {"vsv", "psv", "vsp", "psp"})

        self.current = OperatorPath(
            "complex-x-path-current",
            ((1, GaugePath((0,)), 1, self.coefficient),),
            current_terms=(CurrentTerm(
                self.coefficient, 0, (1, 0, 0, 0), 1,
            ),),
            current_term_ranges=((0,),),
        )

    def _make_basis_values(self):
        rng = np.random.default_rng(2502)
        result = {}
        for time in self.times:
            vector = rng.normal(size=(1, 1, 1, 4, 3)) + 1j * rng.normal(size=(1, 1, 1, 4, 3))
            vector /= np.linalg.norm(vector)
            result[time] = vector.astype(np.complex128)
        return result

    def _publish(self, name, spec, rows):
        directory = self.root / name
        with DatasetWriter(directory, spec, provenance={"test": "independent-double-current"}) as writer:
            for position, time in enumerate(spec.coordinates("time")):
                writer.write(position, np.asarray(rows[time], dtype=spec.document["dtype"]))
            identity = writer.publish()
        self.index.register(directory)
        return identity

    def _publish_gauge(self):
        axes = [dict(name=name, coordinates=list(range(size))) for name, size in zip(
            ("time", "z", "y", "x", "direction", "color_row", "color_column"),
            self.links.shape,
            strict=True,
        )]
        spec = DatasetSpec(dict(
            schema=1, kind="gauge", configuration=self.configuration,
            axes=axes, dtype="complex128", bases={}, parents=[], definition={},
            conventions=dict(boundary="periodic"),
        ))
        return self._publish("gauge", spec, {time: self.links[time] for time in self.times})

    def _publish_basis(self):
        spec = DatasetSpec(dict(
            schema=1, kind="explicit_basis", configuration=self.configuration,
            axes=[
                dict(name="time", coordinates=list(self.time_order)),
                dict(name="mode", coordinates=[self.mode_label]),
                dict(name="z", coordinates=[0]), dict(name="y", coordinates=[0]),
                dict(name="x", coordinates=list(range(4))),
                dict(name="color", coordinates=[0, 1, 2]),
            ],
            dtype="complex128", bases={}, parents=[self.gauge_id],
            definition=dict(realization="actual-registered-complex-mode"),
            conventions=dict(normalization="orthonormal"),
        ))
        return self._publish("basis", spec, self.basis_values)

    def _publish_points(self, name, *, labels):
        spec = DatasetSpec(dict(
            schema=1, kind="spatial_points", configuration=self.configuration,
            axes=[
                dict(name="time", coordinates=list(self.time_order)),
                dict(name="point", coordinates=list(labels)),
                dict(name="xyz", coordinates=["x", "y", "z"]),
            ],
            dtype="int64", bases={}, parents=[], definition=dict(selection="explicit-census"),
            conventions=dict(representation="coordinate_delta", normalization="unit_delta"),
        ))
        rows = {time: np.asarray([[x, 0, 0] for x in range(4)], dtype=np.int64)
                for time in self.times}
        return self._publish(name, spec, rows)

    def _publish_v2v(self):
        return publish_elemental_products(
            self.index, self.root, self.gauge_id, self.basis_id,
            output_name="elementals-v2v", sectors=("v2v",),
            paths=((), (0,)), momentum_list=((0, 0, 0), (1, 0, 0)),
            usedNe=1, phase_convention="anchor",
        )["v2v"]

    def _publish_point_elementals(self, output, points_id):
        result = publish_elemental_products(
            self.index, self.root, self.gauge_id, self.basis_id, points_id,
            output_name=output, paths=((), (0,)),
            momentum_list=((0, 0, 0), (1, 0, 0)), usedNe=1,
            usedNp=self.point_count, phase_convention="anchor",
        )
        return dict(result, v2v=self.v2v_id)

    def _publish_propagators(self, output, *, sink_points, source_points):
        return publish_perambulators(
            self.index, self.root, self.gauge_id, output_name=output,
            basis_id=self.basis_id,
            points_snk_id=sink_points, points_src_id=source_points,
            products=("VSV", "PSV", "VSP", "PSP"), Ne=1,
            Np_snk=self.point_count, Np_src=self.point_count,
            mass=self.mass, t_boundary=self.boundary,
        )

    def _reader_bundle(self, stack, elemental_ids, point_id):
        opened = getattr(self, "_opened", None)
        if opened is None:
            opened = self._opened = {}
        def open_once(identity):
            if identity not in opened:
                opened[identity] = stack.enter_context(self.index.open(identity))
            return opened[identity]

        sectors = {name: open_once(elemental_ids[name])
                   for name in ("v2v", "v2p", "p2v", "p2p")}
        point_dataset = open_once(point_id)
        component = self.current.terms[0].component
        pool = SamplingPoolSpec(self.pool_id, 4, 4)
        sampling_pool_ids = {
            (component, 1, time, side): self.pool_id
            for time in self.times for side in ("bar", "field")
        }
        elemental = DatasetElemental(
            sectors, point_dataset=point_dataset,
            sampling_pool_ids=sampling_pool_ids,
            sampling_groups={self.pool_id: pool},
        )
        return elemental

    def _bindings_and_diagram(self, stack, source_time, sink_time):
        elemental_a = self._reader_bundle(
            stack, self.elemental_a_ids, self.points_a_id,
        )
        elemental_b = self._reader_bundle(
            stack, self.elemental_b_ids, self.points_b_id,
        )
        for identity in (*self.propagator_ab_ids.values(),
                         *self.propagator_ba_ids.values()):
            if identity not in self._opened:
                self._opened[identity] = stack.enter_context(self.index.open(identity))
        propagator_ab = DatasetPropagator({
            name: self._opened[self.propagator_ab_ids[name]]
            for name in ("vsv", "psv", "vsp", "psp")
        })
        propagator_ba = DatasetPropagator({
            name: self._opened[self.propagator_ba_ids[name]]
            for name in ("vsv", "psv", "vsp", "psp")
        })
        quark_ab = Propagator("u", Tag(0, 0), Tag(1, 1))
        quark_ba = Propagator("u", Tag(1, 1), Tag(0, 0))
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [source_time, sink_time],
            ["current-a", "current-b"], [None, quark_ab, quark_ba],
        )
        vertices = {
            "current-a": Meson(elemental_a, self.current, phase_convention="anchor"),
            "current-b": Meson(elemental_b, self.current, source=True,
                               phase_convention="anchor"),
        }
        return diagram, {quark_ab: propagator_ab, quark_ba: propagator_ba}, vertices

    def _independent_local_basis(self, time):
        """Build the spin-lifted V/P basis from fields and point coordinates."""
        vector = self.basis_values[time].reshape(12, 1)
        spatial_basis = np.concatenate((vector, np.eye(12, dtype=np.complex128)), axis=1)
        width = spatial_basis.shape[1]
        local_dimension = 4 * 12
        augmented_local = np.zeros((local_dimension, 4 * width), dtype=np.complex128)
        for site in range(4):
            for spin in range(4):
                for color in range(3):
                    row = (site * 4 + spin) * 3 + color
                    augmented_local[row, spin * width:(spin + 1) * width] = spatial_basis[
                        site * 3 + color
                    ]
        return augmented_local

    def _independent_projectors(self, time):
        """Embed a time-slice V/P basis in the full Wilson field space."""
        augmented_local = self._independent_local_basis(time)
        local_dimension = augmented_local.shape[0]
        full_shape = (self.lattice[3], 1, 1, 4, 4, 3)
        full_dimension = int(np.prod(full_shape))
        embedding = np.zeros((full_dimension, local_dimension), dtype=np.complex128)
        for site in range(4):
            for spin in range(4):
                for color in range(3):
                    local_row = (site * 4 + spin) * 3 + color
                    full_row = np.ravel_multi_index((time, 0, 0, site, spin, color), full_shape)
                    embedding[full_row, local_row] = 1
        return embedding @ augmented_local

    def _independent_vertex_matrices(self):
        """Project a separately assembled dense path-current matrix into V/P."""
        gamma_x = gamma(1)
        gamma4 = gamma(8)
        coefficient = self.coefficient
        matrices = {}
        for time in self.times:
            local = np.zeros((48, 48), dtype=np.complex128)
            for x in range(4):
                phase = np.exp(2j * np.pi * x / 4)
                link = self.links[time, 0, 0, x, 0]
                x_next = (x + 1) % 4
                for spin_bar in range(4):
                    for spin_field in range(4):
                        for color_bar in range(3):
                            for color_field in range(3):
                                row = (x * 4 + spin_bar) * 3 + color_bar
                                column = (x_next * 4 + spin_field) * 3 + color_field
                                local[row, column] = (
                                    gamma_x[spin_bar, spin_field] * phase
                                    * link[color_bar, color_field]
                                )
            local_basis = self._independent_local_basis(time)
            matrices[time] = coefficient * (local_basis.conj().T @ local @ local_basis)

            gamma4_local = np.zeros((48, 48), dtype=np.complex128)
            for site in range(4):
                for color in range(3):
                    indices = [((site * 4 + spin) * 3 + color) for spin in range(4)]
                    gamma4_local[np.ix_(indices, indices)] = gamma4
            source_local = (coefficient.conjugate() * gamma4_local
                            @ local.conj().T @ gamma4_local)
            matrices[("source", time)] = local_basis.conj().T @ source_local @ local_basis
        return matrices

    def _independent_trace_values(self):
        """Invert the full Wilson stencil and form every dense V/P block trace."""
        field_shape = (self.lattice[3], 1, 1, 4, 4, 3)
        dimension = int(np.prod(field_shape))
        identity = np.eye(dimension, dtype=np.complex128)
        dirac = np.column_stack([
            stencil(self.links, identity[:, column].reshape(field_shape),
                    self.mass, self.boundary).ravel()
            for column in range(dimension)
        ])
        inverse = np.linalg.inv(dirac)
        projectors = {time: self._independent_projectors(time) for time in self.times}
        vertices = self._independent_vertex_matrices()
        result = {}
        for source_time in self.times:
            for sink_time in self.times:
                s_t_s = projectors[source_time].conj().T @ inverse @ projectors[sink_time]
                s_s_t = projectors[sink_time].conj().T @ inverse @ projectors[source_time]
                result[source_time, sink_time] = np.trace(
                    vertices[source_time] @ s_t_s
                    @ vertices[("source", sink_time)] @ s_s_t
                )
        return result

    def test_published_double_current_full_time_trace_uses_real_psp(self):
        expected = self._independent_trace_values()
        with ExitStack() as stack:
            self._opened = {}
            self.addCleanup(lambda: setattr(self, "_opened", None))
            values = {}
            for source_time in self.times:
                for sink_time in self.times:
                    diagram, propagators, vertices = self._bindings_and_diagram(
                        stack, source_time, sink_time,
                    )
                    forward = propagators[diagram.propagator_list[1]]
                    reverse = propagators[diagram.propagator_list[2]]
                    with patch.object(forward, "block", wraps=forward.block) as read_ab, \
                         patch.object(reverse, "block", wraps=reverse.block) as read_ba:
                        calculation = calc_diagram_eval(
                            diagram, propagator_map=propagators, vertex_map=vertices,
                            configuration="cfg-double-current", usedNe=1, usedNp=4,
                            optimize="none",
                        )
                    values[source_time, sink_time] = calculation.value
                    descriptions = [spec.description
                                    for spec in calculation.inputs.problem.specs.values()]
                    propagator_descriptions = [description for description in descriptions
                                               if description.role == "propagator"]
                    self.assertTrue(propagator_descriptions)
                    self.assertTrue(all(
                        all(check["status"] == "verified" for check in
                            description.bindings["endpoint_axis_bindings"].values())
                        for description in propagator_descriptions
                    ))
                    if source_time == self.times[0] and sink_time == self.times[0]:
                        bridge_counts = {}
                        for bridge in calculation.expanded.problem.sparse_bridges:
                            bridge_counts[bridge.scene] = bridge_counts.get(bridge.scene, 0) + 1
                        self.assertTrue(any(count == 2 for count in bridge_counts.values()))
                    self.assertTrue(any(call.args[2] == "psp" for call in read_ab.call_args_list))
                    self.assertTrue(any(call.args[2] == "psp" for call in read_ba.call_args_list))

            for times, value in values.items():
                with self.subTest(source_time=times[0], sink_time=times[1]):
                    np.testing.assert_allclose(value, expected[times], rtol=3e-10, atol=3e-10)



if __name__ == "__main__":
    unittest.main()
