"""Registered temporal point paths checked against independent site-link loops."""

from contextlib import ExitStack
from tempfile import TemporaryDirectory
import gc
import weakref
import unittest
from unittest.mock import patch

import numpy as np

import test_directed_current_point_diagram as temporal_fixture
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, ElementalComponent, GaugePath,
)
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator
from lqcdweave.storage.dataset import DatasetWriter
from lqcdweave.production.elemental.planning import plan_indexed_point_elementals
from fixtures.c_group_production import publish_elemental_products


PATHS = tuple(GaugePath(path) for path in (
    (6, 0), (0, 6), (6, 6), (6, 7), (7, 6),
))


class TemporalPointPathDatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = temporal_fixture.RegisteredTemporalPointCurrentTests(
            "test_registered_temporal_vp_pv_pp_match_independent_time_endpoint_loops"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture

    def _produce(self, paths=PATHS, *, boundary="periodic", output="temporal-path"):
        identities = publish_elemental_products(
            self.f.index, self.f.root, self.f.gauge_id, self.f.basis_id,
            self.f.left_id, output_name=output, paths=paths,
            momentum_list=self.f.momentum_list, sectors=("v2p", "p2v"),
            usedNe=2, usedNp=2, points_right_id=self.f.right_id,
            point_right_labels=self.f.right_labels, phase_convention="midpoint",
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA, path_boundary=boundary,
        )
        return {"datasets": identities}

    def _link_product(self, anchor, path, start_x, *, boundary="periodic"):
        """Independent scalar-site ordered product; it does not call path_transport."""
        lo = hi = tau = 0
        for step in path.steps:
            if step == 6:
                tau += 1
            elif step == 7:
                tau -= 1
            lo, hi = min(lo, tau), max(hi, tau)
        if boundary == "open" and (anchor + lo < 0 or anchor + hi >= self.f.lattice[3]):
            return None
        x = int(start_x)
        tau = 0
        result = np.eye(3, dtype=np.complex128)
        for step in path.steps:
            if step == 0:
                link_direction, sign = 0, 1
            elif step == 3:
                link_direction, sign = 0, -1
                x = (x - 1) % self.f.lattice[0]
            elif step == 6:
                link_direction, sign = 3, 1
            elif step == 7:
                link_direction, sign = 3, -1
                tau -= 1
            else:
                raise AssertionError(f"unexpected direction in test oracle: {step}")
            time = anchor + tau
            if boundary == "periodic":
                time %= self.f.lattice[3]
            link = self.f.links[time, 0, 0, x, link_direction]
            if sign < 0:
                link = link.conj().T
            result = result @ link
            if step == 0:
                x = (x + 1) % self.f.lattice[0]
            elif step == 6:
                tau += 1
        return result

    def _phase(self, momentum, x, displacement, convention="midpoint"):
        dx = displacement[0] / 2 if convention == "midpoint" else 0
        return np.exp(2j * np.pi * momentum[0] * (x + dx) / self.f.lattice[0])

    def _expected_v2p(self, path, anchor, momentum, *, boundary="periodic"):
        f = self.f
        result = np.zeros((2, len(f.right_labels), 3), dtype=np.complex128)
        lo, hi = self._time_bounds(path)
        if boundary == "open" and (anchor + lo < 0 or anchor + hi >= f.lattice[3]):
            return result
        dx = path.displacement[0]
        for point, coordinate in enumerate(f.right_coordinates[(anchor + path.displacement[3]) % f.lattice[3]]):
            endpoint_x = int(coordinate[0])
            base_x = (endpoint_x - dx) % f.lattice[0]
            transport = self._link_product(anchor, path, base_x, boundary=boundary)
            if transport is None:
                return result
            phase = self._phase(momentum, base_x, path.displacement)
            for mode in range(2):
                vector = f.mode_values[anchor][mode, 0, 0, base_x]
                result[mode, point] = phase * (vector.conj() @ transport)
        return result

    def _expected_p2v(self, path, anchor, momentum, *, boundary="periodic"):
        f = self.f
        result = np.zeros((2, 3, 2), dtype=np.complex128)
        lo, hi = self._time_bounds(path)
        if boundary == "open" and (anchor + lo < 0 or anchor + hi >= f.lattice[3]):
            return result
        endpoint = (anchor + path.displacement[3]) % f.lattice[3]
        dx = path.displacement[0]
        for point, coordinate in enumerate(f.left_coordinates[anchor][:2]):
            base_x = int(coordinate[0])
            transport = self._link_product(anchor, path, base_x, boundary=boundary)
            if transport is None:
                return result
            target_x = (base_x + dx) % f.lattice[0]
            phase = self._phase(momentum, base_x, path.displacement)
            for mode in range(2):
                vector = f.mode_values[endpoint][mode, 0, 0, target_x]
                result[point, :, mode] = phase * (transport @ vector)
        return result

    @staticmethod
    def _time_bounds(path):
        time = lo = hi = 0
        for step in path.steps:
            if step == 6:
                time += 1
            elif step == 7:
                time -= 1
            lo, hi = min(lo, time), max(hi, time)
        return lo, hi

    def test_registered_vp_pv_temporal_paths_match_independent_ordered_site_loops(self):
        result = self._produce()
        self.assertEqual(set(result["datasets"]), {"v2p", "p2v"})
        with ExitStack() as stack:
            readers = {sector: stack.enter_context(self.f.index.open(identity))
                       for sector, identity in result["datasets"].items()}
            elemental = DatasetElemental(readers)
            for sector, reader in readers.items():
                definition = reader.spec.document["definition"]
                self.assertEqual(definition["temporal_path_schema"], TEMPORAL_POINT_PATH_SCHEMA)
                self.assertNotIn("directed_current_schema", definition)
                self.assertEqual(definition["path_boundary"], "periodic")
                self.assertEqual(reader.spec.coordinates("time"), [0, 1, 2])
                self.assertEqual(reader.spec.document["parents"],
                                 [self.f.gauge_id, self.f.basis_id,
                                  self.f.right_id if sector == "v2p" else self.f.left_id])
            self.assertEqual(readers["v2p"].spec.coordinates("component"),
                             [dict(path=list(path.steps)) for path in PATHS])
            self.assertEqual(readers["p2v"].spec.coordinates("component"),
                             [dict(path=list(path.steps)) for path in PATHS])
            self.assertEqual(readers["v2p"].spec.coordinates("point_right"),
                             list(self.f.right_labels))
            self.assertEqual(readers["p2v"].spec.coordinates("point_left"),
                             list(self.f.left_labels[:2]))
            self.assertEqual(readers["v2p"].spec.document["bases"]["mode_left"]["data_id"],
                             self.f.basis_id)
            self.assertEqual(readers["p2v"].spec.document["bases"]["mode_right"]["data_id"],
                             self.f.basis_id)

            for component_index, path in enumerate(PATHS):
                component = ElementalComponent.path(path)
                for anchor in range(self.f.lattice[3]):
                    for momentum_index, momentum in enumerate(self.f.momentum_list):
                        np.testing.assert_allclose(
                            elemental.block(component, momentum_index, anchor, "v2p",
                                            usedNe=2, usedNp=3),
                            self._expected_v2p(path, anchor, momentum),
                            rtol=4e-13, atol=4e-13,
                        )
                        np.testing.assert_allclose(
                            elemental.block(component, momentum_index, anchor, "p2v",
                                            usedNe=2, usedNp=2),
                            self._expected_p2v(path, anchor, momentum),
                            rtol=4e-13, atol=4e-13,
                        )
                self.assertEqual(component.key, path)
                self.assertEqual(component.time_offsets, (0, path.displacement[3]))
            # These paths have the same displacement but distinct ordered link products.
            self.assertEqual(PATHS[0].displacement, PATHS[1].displacement)
            self.assertFalse(np.allclose(
                self._expected_v2p(PATHS[0], 1, self.f.momentum_list[1]),
                self._expected_v2p(PATHS[1], 1, self.f.momentum_list[1]),
            ))
            ordered_first = plan_indexed_point_elementals(
                self.f.index, self.f.gauge_id, self.f.basis_id, None,
                paths=(PATHS[0],), momentum_list=(self.f.momentum_list[1],),
                sectors=("v2p",), usedNe=2, points_right_id=self.f.right_id,
                point_right_labels=self.f.right_labels,
                temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            )["stages"]["v2p"]["spec"]
            ordered_second = plan_indexed_point_elementals(
                self.f.index, self.f.gauge_id, self.f.basis_id, None,
                paths=(PATHS[1],), momentum_list=(self.f.momentum_list[1],),
                sectors=("v2p",), usedNe=2, points_right_id=self.f.right_id,
                point_right_labels=self.f.right_labels,
                temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            )["stages"]["v2p"]["spec"]
            self.assertNotEqual(ordered_first.definition_id, ordered_second.definition_id)

    def test_open_net_zero_excursions_publish_zero_endpoint_blocks(self):
        paths = (GaugePath((6, 7)), GaugePath((7, 6)))
        result = self._produce(paths, boundary="open", output="open-excursions")
        with ExitStack() as stack:
            readers = {sector: stack.enter_context(self.f.index.open(identity))
                       for sector, identity in result["datasets"].items()}
            elemental = DatasetElemental(readers)
            for component_index, (path, anchor) in enumerate(zip(paths, (2, 0), strict=True)):
                component = ElementalComponent.path(path)
                self.assertEqual(path.displacement[3], 0)
                self.assertTrue(0 <= anchor < self.f.lattice[3])
                self.assertIsNone(self._link_product(anchor, path, 0, boundary="open"))
                for momentum_index in range(len(self.f.momentum_list)):
                    np.testing.assert_array_equal(
                        elemental.block(component, momentum_index, anchor, "v2p",
                                        usedNe=2, usedNp=3),
                        np.zeros((2, 3, 3), dtype=np.complex128),
                    )
                    np.testing.assert_array_equal(
                        elemental.block(component, momentum_index, anchor, "p2v",
                                        usedNe=2, usedNp=2),
                        np.zeros((2, 3, 2), dtype=np.complex128),
                    )

    def test_temporal_schema_is_opt_in_and_rejects_current_mix_but_accepts_pp(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            plan_indexed_point_elementals(
                self.f.index, self.f.gauge_id, self.f.basis_id, self.f.left_id,
                paths=(GaugePath((6, 0)),),
                momentum_list=self.f.momentum_list, sectors=("v2p",), usedNe=2,
                points_right_id=self.f.right_id, point_right_labels=self.f.right_labels,
                phase_convention="midpoint", current_schema="lattice.current.point/v1",
                temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            )
        pp_plan = plan_indexed_point_elementals(
            self.f.index, self.f.gauge_id, None, self.f.left_id,
            paths=(GaugePath((6, 0)),), momentum_list=((0, 0, 0),),
            sectors=("p2p",), usedNp=2,
            points_right_id=self.f.right_id,
            point_right_labels=self.f.right_labels,
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
        )
        self.assertEqual(
            pp_plan["stages"]["p2p"]["spec"].document["definition"]["temporal_path_schema"],
            TEMPORAL_POINT_PATH_SCHEMA,
        )

        with self.assertRaisesRegex(ValueError, "spatial paths only"):
            plan_indexed_point_elementals(
                self.f.index, self.f.gauge_id, self.f.basis_id, None,
                paths=(GaugePath((6, 0)),), momentum_list=((0, 0, 0),),
                sectors=("v2p",), usedNe=2, points_right_id=self.f.right_id,
                point_right_labels=self.f.right_labels,
            )

        plan = plan_indexed_point_elementals(
            self.f.index, self.f.gauge_id, self.f.basis_id, None,
            paths=(GaugePath((6, 0)),), momentum_list=((0, 0, 0),),
            sectors=("v2p",), usedNe=2, points_right_id=self.f.right_id,
            point_right_labels=self.f.right_labels,
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
        )
        planned = plan["stages"]["v2p"]["spec"]
        self.assertEqual(planned.document["definition"]["temporal_path_schema"],
                         TEMPORAL_POINT_PATH_SCHEMA)
        self.assertNotIn("directed_current_schema", planned.document["definition"])
        self.assertEqual(planned.document["definition"]["path_boundary"], "periodic")

    def test_writer_does_not_retain_array_after_producer_lifetime(self):
        refs = []
        buffer_ids = {"v2p": [], "p2v": []}
        original_calc = ElementalGenerator.calc_path_blocks
        original_publish = DatasetWriter.publish

        def observe_calc(generator, time, blocks=("v2v",), **kwargs):
            result = original_calc(generator, time, blocks, **kwargs)
            sector = blocks[0]
            refs.append((sector, weakref.ref(result[sector])))
            buffer_ids[sector].append(id(result[sector]))
            return result

        def observe_publish(writer, *args, **kwargs):
            gc.collect()
            kind = writer.spec.document["kind"]
            sector = {"elemental_v2p": "v2p", "elemental_p2v": "p2v"}.get(kind)
            if sector is not None:
                # The DatasetWriter remains active here. A live weakref would
                # mean it retained the producer's borrowed workspace array.
                self.assertTrue(all(reference() is None for name, reference in refs
                                    if name == sector))
            return original_publish(writer, *args, **kwargs)

        with patch.object(ElementalGenerator, "calc_path_blocks", observe_calc), \
                patch.object(DatasetWriter, "publish", observe_publish):
            self._produce(output="writer-lifetime")
        gc.collect()
        self.assertGreater(len(refs), 0)
        self.assertTrue(all(reference() is None for _, reference in refs))
        self.assertEqual(len(buffer_ids["v2p"]), self.f.lattice[3])
        self.assertEqual(len(buffer_ids["p2v"]), self.f.lattice[3])
        self.assertEqual(len(set(buffer_ids["v2p"])), 1)
        self.assertEqual(len(set(buffer_ids["p2v"])), 1)


if __name__ == "__main__":
    unittest.main()
