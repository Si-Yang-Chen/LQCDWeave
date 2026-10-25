"""Registered temporal V/P Diagram chains against an independent Wilson inverse."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

import test_directed_current_point_diagram as fixture_module
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, ElementalComponent, GaugePath,
)
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import OperatorPath
from lqcdweave.contraction.diagram_measurement import plan_indexed_diagram_measurement
from lqcdweave.contraction.correlator import calc_diagram_eval
from fixtures.c_group_production import publish_elemental_products, publish_perambulators


class RegisteredTemporalPointPathDiagramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture_module.RegisteredTemporalPointCurrentTests("runTest")
        cls.fixture.setUp()
        cls.fixture._reset_to_census_current_fixture()
        cls.f = cls.fixture
        cls.f.momentum_list = ((0, 0, 0),)
        cls.path = GaugePath((6, 0))
        cls.temporal_ids = publish_elemental_products(
            cls.f.index, cls.f.root, cls.f.gauge_id, cls.f.basis_id, cls.f.left_id,
            output_name="temporal-diagram-periodic", paths=(cls.path,),
            momentum_list=cls.f.momentum_list, sectors=("v2p", "p2v"),
            usedNe=2, usedNp=3, points_right_id=cls.f.right_id,
            point_right_labels=cls.f.right_labels, phase_convention="midpoint",
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA, path_boundary="periodic",
        )
        products = publish_perambulators(
            cls.f.index, cls.f.root, cls.f.gauge_id,
            output_name="temporal-diagram-wilson", basis_id=cls.f.basis_id,
            points_snk_id=cls.f.right_id, points_src_id=cls.f.right_id,
            products=("VSV", "PSP"), Ne=2,
            Np_snk=3, Np_src=3, mass=0.4, t_boundary=+1,
        )
        cls.wilson_ids = {name: products[name] for name in ("vsv", "psp")}
        cls.mass = 0.4
        cls.sink_coefficient = 0.7 + 0.2j
        cls.source_coefficient = -0.3 + 0.4j

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def _diagram_bindings(self, elemental_reader, prop_readers, path, *, boundary="periodic",
                          source_elemental_reader=None, sink_point_id=None,
                          sink_point_labels=None, source_point_id=None,
                          source_point_labels=None):
        component = ElementalComponent.path(path)
        sink_point_id = self.f.right_id if sink_point_id is None else sink_point_id
        sink_point_labels = (self.f.right_labels if sink_point_labels is None
                             else tuple(sink_point_labels))
        source_point_id = sink_point_id if source_point_id is None else source_point_id
        source_point_labels = (sink_point_labels if source_point_labels is None
                               else tuple(source_point_labels))
        source_elemental_reader = (elemental_reader if source_elemental_reader is None
                                   else source_elemental_reader)

        def bind_elemental(reader, point_id, labels, role):
            pool_id = f"full-point-pool-{tuple(path.steps)}-{boundary}-{point_id}-{labels}"
            pool_ids = {
                (component, 0, time, side): pool_id
                for time in range(self.f.lattice[3])
                for side in ("bar", "field")
            }
            return DatasetElemental(
                {"v2p": reader},
                sampling_pool_ids=pool_ids,
                sampling_groups={pool_id: SamplingPoolSpec(
                    pool_id, spatial_sites=3, sampled_points=len(labels),
                )},
            )

        sink_elemental = bind_elemental(
            elemental_reader, sink_point_id, sink_point_labels, "sink")
        source_elemental = bind_elemental(
            source_elemental_reader, source_point_id, source_point_labels, "source")
        sink_operator = OperatorPath("sink-v2p", ((1, path, 0, self.sink_coefficient),))
        source_operator = OperatorPath("source-dagger-p2v", ((3, path, 0, self.source_coefficient),))
        self.assertEqual(source_operator.current_terms, ())
        sink = Meson(
            sink_elemental, sink_operator, phase_convention="midpoint", boundary=boundary,
            requested_sectors=("v2p",),
        )
        source = Meson(
            source_elemental, source_operator, source=True,
            phase_convention="midpoint", boundary=boundary,
            requested_sectors=("p2v",),
        )
        q = Propagator("u", Tag(0, 0), Tag(1, 1))
        propagator = DatasetPropagator(prop_readers)
        vertices = {"sink": sink, "source": source}
        bindings = DiagramBindings(
            {q: propagator}, vertices, self.f.configuration["configuration"],
            2, 3, True,
        )
        source_time, sink_time = Symbol("source-time"), Symbol("sink-time")
        diagram = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), [sink_time, source_time],
            ["sink", "source"], [None, q],
        )
        return diagram, bindings, source_time, sink_time, q, vertices, sink_elemental

    def _evaluate_grid(self, diagram, bindings, source_time, sink_time):
        result = np.empty((self.f.lattice[3], self.f.lattice[3]), dtype=np.complex128)
        for source in range(self.f.lattice[3]):
            for sink in range(self.f.lattice[3]):
                calculation = calc_diagram_eval(
                    diagram,
                    propagator_map=bindings.propagator_map,
                    vertex_map=bindings.vertex_map,
                    configuration=bindings.configuration,
                    usedNe=bindings.usedNe,
                    usedNp=bindings.usedNp,
                    groups_independent=bindings.groups_independent,
                    time_axes={source_time: source, sink_time: sink},
                    optimize="none", backend=np,
                )
                result[source, sink] = calculation.value
        return result

    def _basis_at_time(self, time, *, point_coordinates=None, point_labels=None):
        """Build V and P columns with explicit site/spin/color loops."""
        f = self.f
        point_coordinates = (f.right_coordinates if point_coordinates is None
                             else point_coordinates)
        point_labels = f.right_labels if point_labels is None else tuple(point_labels)
        modes = f.mode_values[time].reshape(2, -1).T
        spatial = np.zeros((3 * f.lattice[0], 2 + 3 * len(point_labels)), dtype=np.complex128)
        spatial[:, :2] = modes
        for point, coordinate in enumerate(point_coordinates[time]):
            x, y, z = map(int, coordinate)
            site = (z * f.lattice[1] + y) * f.lattice[0] + x
            spatial[3 * site:3 * site + 3, 2 + 3 * point:2 + 3 * point + 3] = np.eye(3)
        width = spatial.shape[1]
        full = np.zeros((12 * f.lattice[0], 4 * width), dtype=np.complex128)
        vector = np.zeros((12 * f.lattice[0], 4 * 2), dtype=np.complex128)
        point = np.zeros((12 * f.lattice[0], 4 * 3 * len(point_labels)), dtype=np.complex128)
        for site in range(f.lattice[0]):
            for spin in range(4):
                row = (site * 4 + spin) * 3
                full[row:row + 3, spin * width:(spin + 1) * width] = spatial[3 * site:3 * site + 3]
                vector[row:row + 3, spin * 2:(spin + 1) * 2] = spatial[3 * site:3 * site + 3, :2]
                point[row:row + 3, spin * 9:(spin + 1) * 9] = spatial[3 * site + 0:3 * site + 3, 2:]
        return vector, point

    def _projected_inverse_blocks(self, *, sink_coordinates=None, source_coordinates=None,
                                  sink_labels=None, source_labels=None):
        """Return independent projected Wilson blocks from a dense inverse."""
        f = self.f
        sink_coordinates = (f.right_coordinates if sink_coordinates is None
                            else sink_coordinates)
        source_coordinates = (f.right_coordinates if source_coordinates is None
                              else source_coordinates)
        sink_labels = f.right_labels if sink_labels is None else tuple(sink_labels)
        source_labels = f.right_labels if source_labels is None else tuple(source_labels)
        inverse = fixture_module.dense_wilson_inverse(f.links, self.mass)
        local = inverse.reshape(f.lattice[3], 12 * f.lattice[0],
                                f.lattice[3], 12 * f.lattice[0])
        sink_bases = [self._basis_at_time(
            time, point_coordinates=sink_coordinates, point_labels=sink_labels)
            for time in range(f.lattice[3])]
        source_bases = [self._basis_at_time(
            time, point_coordinates=source_coordinates, point_labels=source_labels)
            for time in range(f.lattice[3])]
        vv, pp = {}, {}
        for sink_time in range(f.lattice[3]):
            for source_time in range(f.lattice[3]):
                block = local[sink_time, :, source_time, :]
                q_sink_v, q_sink_p = sink_bases[sink_time]
                q_source_v, q_source_p = source_bases[source_time]
                v = q_sink_v.conj().T @ block @ q_source_v
                p = q_sink_p.conj().T @ block @ q_source_p
                vv[sink_time, source_time] = v.reshape(4, 2, 4, 2).transpose(0, 2, 1, 3)
                pp[sink_time, source_time] = p.reshape(4, 3, 3, 4, 3, 3).transpose(0, 3, 1, 2, 4, 5)
        return vv, pp

    def _independent_v2p(self, path, anchor, *, point_coordinates=None):
        """Ordered site/link loop independent of the production transport kernel."""
        f = self.f
        point_coordinates = (f.right_coordinates if point_coordinates is None
                             else point_coordinates)
        dx, dy, dz, dt = path.displacement
        endpoint = (anchor + dt) % f.lattice[3]
        result = np.zeros((2, len(point_coordinates[endpoint]), 3), dtype=np.complex128)
        for point, coord in enumerate(point_coordinates[endpoint]):
            end = tuple(map(int, coord))
            start = ((end[0] - dx) % f.lattice[0],
                     (end[1] - dy) % f.lattice[1],
                     (end[2] - dz) % f.lattice[2])
            offset = [0, 0, 0, 0]
            transport = np.eye(3, dtype=np.complex128)
            for step in path.steps:
                axis = GaugePath.AXES[step]
                sign = GaugePath.SIGNS[step]
                if sign < 0:
                    offset[axis] -= 1
                position = tuple((start[i] + offset[i]) % f.lattice[i] for i in range(3))
                time = (anchor + offset[3]) % f.lattice[3]
                link = f.links[(time, position[2], position[1], position[0], axis)]
                if sign < 0:
                    link = link.conj().T
                transport = transport @ link
                if sign > 0:
                    offset[axis] += 1
            for mode in range(2):
                vector = f.mode_values[anchor][mode, 0, 0, start[0]]
                result[mode, point] = vector.conj() @ transport
        return result

    def _expected_grid(self, path, *, sink_coordinates=None, source_coordinates=None,
                       sink_labels=None, source_labels=None):
        sink_coordinates = (self.f.right_coordinates if sink_coordinates is None
                            else sink_coordinates)
        source_coordinates = (self.f.right_coordinates if source_coordinates is None
                              else source_coordinates)
        vv, pp = self._projected_inverse_blocks(
            sink_coordinates=sink_coordinates,
            source_coordinates=source_coordinates,
            sink_labels=sink_labels, source_labels=source_labels)
        result = np.empty((self.f.lattice[3], self.f.lattice[3]), dtype=np.complex128)
        for source_time in range(self.f.lattice[3]):
            raw_source = self._independent_v2p(
                path, source_time, point_coordinates=source_coordinates)
            spatial_source = raw_source.conj().transpose(1, 2, 0)
            spin_source = gamma(8) @ gamma(3).conj().T @ gamma(8)
            factor_source = np.einsum("ab,...->ab...", spin_source, spatial_source, optimize=False)
            for sink_time in range(self.f.lattice[3]):
                raw_sink = self._independent_v2p(
                    path, sink_time, point_coordinates=sink_coordinates)
                spin_sink = gamma(1)
                factor_sink = np.einsum("ab,...->ab...", spin_sink, raw_sink, optimize=False)
                result[source_time, sink_time] = (
                    self.sink_coefficient * self.source_coefficient.conjugate()
                    * np.einsum(
                        "ijab,klmnop,jkbmn,liopa->",
                        # Both raw V axes belong to the path anchors;
                        # both P axes belong to the transported endpoints.
                        vv[source_time, sink_time],
                        pp[(sink_time + path.displacement[3]) % self.f.lattice[3],
                           (source_time + path.displacement[3]) % self.f.lattice[3]],
                        factor_sink,
                        factor_source,
                        optimize=False,
                    )
                )
        return result

    def test_registered_temporal_diagram_matches_dense_independent_inverse_and_restores_identity(self):
        with ExitStack() as stack:
            elemental_reader = stack.enter_context(self.f.index.open(self.temporal_ids["v2p"]))
            propagator_readers = {
                name: stack.enter_context(self.f.index.open(identity))
                for name, identity in self.wilson_ids.items()
            }
            diagram, bindings, source_time, sink_time, _q, vertices, elemental = self._diagram_bindings(
                elemental_reader, propagator_readers, self.path,
            )
            sink_metadata = vertices["sink"].describe_factors(1, "v2p")[0]
            source_metadata = vertices["source"].describe_factors(1, "p2v")[0]
            self.assertEqual(sink_metadata.logical_sector, "v2p")
            self.assertEqual(sink_metadata.raw_sector, "v2p")
            self.assertFalse(sink_metadata.dagger)
            self.assertEqual(sink_metadata.field_time, 2)
            self.assertEqual(sink_metadata.bar_time, 1)
            self.assertEqual(sink_metadata.tensor.axis_bindings["mode_left"],
                             (self.f.basis_id, (17, 5)))
            self.assertEqual(sink_metadata.tensor.axis_bindings["point_right"],
                             (self.f.right_id, tuple(self.f.right_labels)))
            self.assertEqual(source_metadata.logical_sector, "p2v")
            self.assertEqual(source_metadata.raw_sector, "v2p")
            self.assertTrue(source_metadata.dagger)
            self.assertEqual(source_metadata.field_time, 1)
            self.assertEqual(source_metadata.bar_time, 2)
            self.assertEqual(source_metadata.tensor.axis_bindings["point_right"],
                             (self.f.right_id, tuple(self.f.right_labels)))
            with self.f.index.open(self.temporal_ids["v2p"]) as raw_reader:
                self.assertEqual(raw_reader.spec.document["bases"]["mode_left"]["data_id"],
                                 self.f.basis_id)
                self.assertEqual(raw_reader.spec.document["bases"]["point_right"]["data_id"],
                                 self.f.right_id)
                self.assertEqual(raw_reader.spec.coordinates("point_right"), list(self.f.right_labels))

            plan = plan_indexed_diagram_measurement(
                self.f.index, diagram, bindings=bindings,
                source_time_symbol=source_time, sink_time_symbol=sink_time,
            )
            self.assertEqual(tuple(plan["source_times"]), tuple(range(self.f.lattice[3])))
            self.assertEqual(tuple(plan["sink_times"]), tuple(range(self.f.lattice[3])))
            self.assertEqual(plan["measurement_spec"].coordinates("source_time"),
                             list(range(self.f.lattice[3])))
            actual = self._evaluate_grid(diagram, bindings, source_time, sink_time)
            np.testing.assert_allclose(actual, self._expected_grid(self.path),
                                       rtol=4e-10, atol=4e-10)

    def test_source_dagger_uses_distinct_registered_point_and_psp_endpoints(self):
        source_ids = publish_elemental_products(
            self.f.index, self.f.root, self.f.gauge_id, self.f.basis_id,
            self.f.left_id, output_name="temporal-diagram-source-points", paths=(self.path,),
            momentum_list=self.f.momentum_list, sectors=("v2p",), usedNe=2,
            usedNp=len(self.f.left_labels),
            phase_convention="midpoint", temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="periodic",
        )
        distinct_wilson_ids = publish_perambulators(
            self.f.index, self.f.root, self.f.gauge_id,
            output_name="temporal-diagram-distinct-psp", basis_id=self.f.basis_id,
            points_snk_id=self.f.right_id, points_src_id=self.f.left_id,
            products=("VSV", "PSP"), Ne=2,
            Np_snk=len(self.f.right_labels), Np_src=len(self.f.left_labels),
            mass=self.mass, t_boundary=+1,
        )
        with ExitStack() as stack:
            sink_reader = stack.enter_context(self.f.index.open(self.temporal_ids["v2p"]))
            source_reader = stack.enter_context(self.f.index.open(source_ids["v2p"]))
            prop_readers = {
                name: stack.enter_context(self.f.index.open(identity))
                for name, identity in distinct_wilson_ids.items()
                if name in {"vsv", "psp"}
            }
            self.assertEqual(set(prop_readers), {"vsv", "psp"})
            diagram, bindings, source_time, sink_time, _q, vertices, _elemental = (
                self._diagram_bindings(
                    sink_reader, prop_readers, self.path,
                    source_elemental_reader=source_reader,
                    sink_point_id=self.f.right_id,
                    sink_point_labels=self.f.right_labels,
                    source_point_id=self.f.left_id,
                    source_point_labels=self.f.left_labels,
                )
            )
            sink_metadata = vertices["sink"].describe_factors(1, "v2p")[0]
            source_metadata = vertices["source"].describe_factors(1, "p2v")[0]
            self.assertEqual(
                sink_metadata.tensor.axis_bindings["point_right"],
                (self.f.right_id, tuple(self.f.right_labels)),
            )
            self.assertEqual(
                source_metadata.tensor.axis_bindings["point_right"],
                (self.f.left_id, tuple(self.f.left_labels)),
            )
            self.assertTrue(source_metadata.dagger)

            with self.f.index.open(distinct_wilson_ids["psp"]) as psp:
                bases = psp.spec.document["bases"]
                self.assertEqual(bases["sink_point"]["data_id"], self.f.right_id)
                self.assertEqual(bases["sink_point"]["indices"], list(self.f.right_labels))
                self.assertEqual(bases["source_point"]["data_id"], self.f.left_id)
                self.assertEqual(bases["source_point"]["indices"], list(self.f.left_labels))

            actual = self._evaluate_grid(diagram, bindings, source_time, sink_time)
            expected = self._expected_grid(
                self.path,
                sink_coordinates=self.f.right_coordinates,
                source_coordinates=self.f.left_coordinates,
                sink_labels=self.f.right_labels,
                source_labels=self.f.left_labels,
            )
            np.testing.assert_allclose(actual, expected, rtol=4e-10, atol=4e-10)

    def test_open_path_endpoint_out_of_range_rejects_before_getters_or_output(self):
        path = GaugePath((6, 0))
        open_ids = publish_elemental_products(
            self.f.index, self.f.root, self.f.gauge_id, self.f.basis_id,
            self.f.right_id,
            output_name="temporal-diagram-open-endpoint-out-of-range", paths=(path,),
            momentum_list=self.f.momentum_list, sectors=("v2p",), usedNe=2,
            usedNp=len(self.f.right_labels),
            phase_convention="midpoint", temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="open",
        )
        with ExitStack() as stack:
            elemental_reader = stack.enter_context(self.f.index.open(open_ids["v2p"]))
            prop_readers = {
                name: stack.enter_context(self.f.index.open(identity))
                for name, identity in self.wilson_ids.items()
            }
            diagram, bindings, source_time, sink_time, _q, _vertices, _elemental = (
                self._diagram_bindings(
                    elemental_reader, prop_readers, path, boundary="open")
            )
            with patch.object(
                    Meson, "read_factor_block",
                    side_effect=AssertionError("open preflight reached an elemental getter")) as reads:
                with patch.object(
                        DatasetPropagator, "block",
                        side_effect=AssertionError("open preflight reached a propagator getter")) as props:
                    with self.assertRaisesRegex(ValueError, "open temporal domain"):
                        plan_indexed_diagram_measurement(
                            self.f.index, diagram, bindings=bindings,
                            source_time_symbol=source_time, sink_time_symbol=sink_time,
                        )
            self.assertEqual(reads.call_count, 0)
            self.assertEqual(props.call_count, 0)

    def test_open_net_zero_excursions_are_zero_in_the_persistent_diagram(self):
        for invalid_anchor, path in ((2, GaugePath((6, 7))), (0, GaugePath((7, 6)))):
            with self.subTest(path=path.steps):
                open_ids = publish_elemental_products(
                    self.f.index, self.f.root, self.f.gauge_id, self.f.basis_id,
                    self.f.left_id,
                    output_name=f"temporal-diagram-open-excursion-{path.idx}",
                    paths=(path,), momentum_list=self.f.momentum_list,
                    sectors=("v2p", "p2v"), usedNe=2, usedNp=3,
                    points_right_id=self.f.right_id,
                    point_right_labels=self.f.right_labels,
                    phase_convention="midpoint",
                    temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
                    path_boundary="open",
                )
                self.assertEqual(path.displacement[3], 0)
                self.assertEqual(path.time_bounds[0] < 0 or path.time_bounds[1] > 0, True)
                with ExitStack() as stack:
                    elemental_reader = stack.enter_context(self.f.index.open(open_ids["v2p"]))
                    propagator_readers = {
                        name: stack.enter_context(self.f.index.open(identity))
                        for name, identity in self.wilson_ids.items()
                    }
                    diagram, bindings, source_time, sink_time, _q, _vertices, _elemental = self._diagram_bindings(
                        elemental_reader, propagator_readers, path, boundary="open",
                    )
                    actual = self._evaluate_grid(diagram, bindings, source_time, sink_time)
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_array_equal(
                        actual[invalid_anchor], np.zeros(3, dtype=np.complex128),
                    )
                    np.testing.assert_array_equal(
                        actual[:, invalid_anchor], np.zeros(3, dtype=np.complex128),
                    )


if __name__ == "__main__":
    unittest.main()
