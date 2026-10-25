"""Open temporal path-boundary Diagram measurements against dense references.

The Wilson/Dirac inverse in this test retains its declared periodic temporal
fermion boundary.  ``path_boundary="open"`` applies to the registered
elemental transport path only.
"""

from contextlib import ExitStack
import unittest

import numpy as np

import test_temporal_point_path_diagram as temporal_fixture
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA,
    GaugePath,
)
from lqcdweave.operators.single_hadron.gamma import gamma
from fixtures.c_group_production import publish_elemental_products, publish_perambulators


class OpenTemporalPointPathDiagramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_class = temporal_fixture.RegisteredTemporalPointPathDiagramTests
        cls.fixture_class.setUpClass()
        cls.helper = cls.fixture_class("runTest")
        cls.f = cls.helper.f

    @classmethod
    def tearDownClass(cls):
        cls.fixture_class.tearDownClass()

    def _produce_open_v2p(self, name, points_id, point_labels):
        result = publish_elemental_products(
            self.f.index, self.f.root, self.f.gauge_id, self.f.basis_id, points_id,
            output_name=name, paths=(self.path,),
            momentum_list=self.f.momentum_list, sectors=("v2p",),
            usedNe=2, usedNp=len(point_labels),
            phase_convention="midpoint",
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="open",
        )
        self.assertEqual(set(result), {"v2p"})
        return result["v2p"]

    def _independent_open_v2p(self, anchor, point_coordinates):
        """Evaluate one V/P path by explicit ordered site/link multiplication."""
        f = self.f
        path = self.path
        lower, upper = path.time_bounds
        result = np.zeros(
            (2, len(point_coordinates[anchor]), 3), dtype=np.complex128)
        if anchor + lower < 0 or anchor + upper >= f.lattice[3]:
            return result

        dx, dy, dz, dt = path.displacement
        endpoint_time = anchor + dt
        if not 0 <= endpoint_time < f.lattice[3]:
            return result
        for point, coordinate in enumerate(point_coordinates[endpoint_time]):
            end = tuple(int(value) for value in coordinate)
            start = (
                (end[0] - dx) % f.lattice[0],
                (end[1] - dy) % f.lattice[1],
                (end[2] - dz) % f.lattice[2],
            )
            offset = [0, 0, 0, 0]
            transport = np.eye(3, dtype=np.complex128)
            for step in path.steps:
                axis = GaugePath.AXES[step]
                sign = GaugePath.SIGNS[step]
                if sign < 0:
                    offset[axis] -= 1
                xyz = tuple(
                    (start[index] + offset[index]) % f.lattice[index]
                    for index in range(3)
                )
                link_time = anchor + offset[3]
                if not 0 <= link_time < f.lattice[3]:
                    raise AssertionError("valid open excursion selected an external link")
                link = f.links[(link_time, xyz[2], xyz[1], xyz[0], axis)]
                if sign < 0:
                    link = link.conj().T
                transport = transport @ link
                if sign > 0:
                    offset[axis] += 1
            for mode in range(2):
                vector = f.mode_values[anchor][mode, 0, 0, start[0]]
                result[mode, point] = vector.conj() @ transport
        return result

    def _independent_open_grid(self):
        """Dense Wilson inverse plus independent open-path site/link factors."""
        f = self.f
        vv, pp = self.helper._projected_inverse_blocks(
            sink_coordinates=f.right_coordinates,
            source_coordinates=f.left_coordinates,
            sink_labels=f.right_labels,
            source_labels=f.left_labels,
        )
        lt = f.lattice[3]
        dt = self.path.displacement[3]
        result = np.zeros((lt, lt), dtype=np.complex128)
        spin_sink = gamma(1)
        spin_source = gamma(8) @ gamma(3).conj().T @ gamma(8)
        for source_time in range(lt):
            raw_source = self._independent_open_v2p(
                source_time, f.left_coordinates)
            factor_source = np.einsum(
                "ab,...->ab...",
                spin_source,
                raw_source.conj().transpose(1, 2, 0),
                optimize=False,
            )
            for sink_time in range(lt):
                raw_sink = self._independent_open_v2p(
                    sink_time, f.right_coordinates)
                factor_sink = np.einsum(
                    "ab,...->ab...", spin_sink, raw_sink, optimize=False)
                result[source_time, sink_time] = (
                    self.helper.sink_coefficient
                    * self.helper.source_coefficient.conjugate()
                    * np.einsum(
                        "ijab,klmnop,jkbmn,liopa->",
                        vv[(source_time + dt) % lt, (sink_time + dt) % lt],
                        pp[sink_time, source_time],
                        factor_sink,
                        factor_source,
                        optimize=False,
                    )
                )
        return result

    def test_open_path_vp_pv_full_grid_matches_dense_reference(self):
        self.path = GaugePath((6, 0, 7))
        self.assertEqual(self.f.lattice[3], 3)
        self.assertEqual(self.path.displacement[3], 0)
        self.assertEqual(self.path.time_bounds, (0, 1))
        self.assertNotEqual(self.f.left_id, self.f.right_id)
        self.assertNotEqual(tuple(self.f.left_labels), tuple(self.f.right_labels))

        sink_id = self._produce_open_v2p(
            "open-time-vp-sink", self.f.right_id, self.f.right_labels)
        source_id = self._produce_open_v2p(
            "open-time-pv-source", self.f.left_id, self.f.left_labels)
        wilson_products = publish_perambulators(
            self.f.index, self.f.root, self.f.gauge_id,
            output_name="open-time-wilson-periodic-fermion-boundary",
            basis_id=self.f.basis_id,
            points_snk_id=self.f.right_id,
            points_src_id=self.f.left_id,
            products=("VSV", "PSP"),
            Ne=2,
            Np_snk=len(self.f.right_labels),
            Np_src=len(self.f.left_labels),
            mass=self.helper.mass,
            t_boundary=+1,
        )
        wilson_ids = {name: wilson_products[name] for name in ("vsv", "psp")}

        with ExitStack() as stack:
            sink_reader = stack.enter_context(self.f.index.open(sink_id))
            source_reader = stack.enter_context(self.f.index.open(source_id))
            propagator_readers = {
                name: stack.enter_context(self.f.index.open(identity))
                for name, identity in wilson_ids.items()
                if name in {"vsv", "psp"}
            }
            diagram, bindings, source_time, sink_time, _q, vertices, _elemental = (
                self.helper._diagram_bindings(
                    sink_reader,
                    propagator_readers,
                    self.path,
                    boundary="open",
                    source_elemental_reader=source_reader,
                    sink_point_id=self.f.right_id,
                    sink_point_labels=self.f.right_labels,
                    source_point_id=self.f.left_id,
                    source_point_labels=self.f.left_labels,
                )
            )

            sink_factor = vertices["sink"].describe_factors(0, "v2p")[0]
            source_factor = vertices["source"].describe_factors(0, "p2v")[0]
            self.assertEqual(sink_factor.logical_sector, "v2p")
            self.assertEqual(sink_factor.raw_sector, "v2p")
            self.assertFalse(sink_factor.dagger)
            self.assertEqual(source_factor.logical_sector, "p2v")
            self.assertEqual(source_factor.raw_sector, "v2p")
            self.assertTrue(source_factor.dagger)
            self.assertEqual(
                sink_factor.tensor.axis_bindings["point_right"],
                (self.f.right_id, tuple(self.f.right_labels)),
            )
            self.assertEqual(
                source_factor.tensor.axis_bindings["point_right"],
                (self.f.left_id, tuple(self.f.left_labels)),
            )
            for identity in (sink_id, source_id):
                with self.f.index.open(identity) as reader:
                    definition = reader.spec.document["definition"]
                    self.assertEqual(definition["path_boundary"], "open")
                    self.assertEqual(
                        definition["temporal_path_schema"], TEMPORAL_POINT_PATH_SCHEMA)
                    self.assertEqual(reader.spec.document["parents"][0], self.f.gauge_id)
            wilson_conventions = propagator_readers["vsv"].spec.document["conventions"]
            self.assertEqual(wilson_conventions["temporal_boundary"], "periodic")

            actual = self.helper._evaluate_grid(
                diagram, bindings, source_time, sink_time)
            expected = self._independent_open_grid()
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-10)
            np.testing.assert_array_equal(actual[2], np.zeros(3, dtype=np.complex128))
            np.testing.assert_array_equal(actual[:, 2], np.zeros(3, dtype=np.complex128))
            self.assertGreater(float(np.linalg.norm(actual[:2, :2])), 1e-8)


if __name__ == "__main__":
    unittest.main()
