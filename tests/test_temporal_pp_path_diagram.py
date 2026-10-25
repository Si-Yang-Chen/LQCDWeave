from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

import test_directed_current_point_diagram as fixture_module
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, ElementalComponent, GaugePath,
)
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.gamma import gamma
from lqcdweave.operators.single_hadron.operator import OperatorPath
from lqcdweave.contraction.diagram_measurement import plan_indexed_diagram_measurement
from lqcdweave.contraction.correlator import calc_diagram_eval
from lqcdweave.runtime.datasets import open_dataset_view
from fixtures.c_group_production import (
    publish_directed_current, publish_elemental_products, publish_perambulators,
)


class TemporalPointPointDiagramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture_module.RegisteredTemporalPointCurrentTests("runTest")
        cls.fixture.setUp()
        cls.fixture._reset_to_census_current_fixture()
        cls.f = cls.fixture
        cls.f.momentum_list = ((0, 0, 0),)
        cls.path = GaugePath((6, 0))
        cls.point_path_ids = publish_elemental_products(
            cls.f.index, cls.f.root, cls.f.gauge_id, points_id=cls.f.left_id,
            output_name="temporal-pp-left-right", paths=(cls.path,),
            momentum_list=cls.f.momentum_list, sectors=("p2p",),
            usedNp=len(cls.f.left_labels), points_right_id=cls.f.right_id,
            point_right_labels=cls.f.right_labels,
            phase_convention="midpoint", temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="periodic",
        )
        cls.vv_ids = {"current": publish_directed_current(
            cls.f.index, cls.f.root, cls.f.gauge_id, cls.f.basis_id,
            output_name="temporal-pp-normal-vv",
            momentum_list=cls.f.momentum_list, usedNe=2, boundary="periodic",
        )}
        cls.mixed_ids = publish_perambulators(
            cls.f.index, cls.f.root, cls.f.gauge_id,
            output_name="temporal-pp-mixed-wilson", basis_id=cls.f.basis_id,
            points_snk_id=cls.f.right_id, points_src_id=cls.f.left_id,
            products=("PSV", "VSP"), Ne=2,
            Np_snk=len(cls.f.right_labels), Np_src=len(cls.f.left_labels),
            mass=0.4, t_boundary=+1,
        )
        cls.inverse = fixture_module.dense_wilson_inverse(cls.f.links, 0.4)
        cls._prepare_dual_cycle_inputs()

    @classmethod
    def _prepare_dual_cycle_inputs(cls):
        f = cls.f
        cls.cycle_labels = {
            "a": (101, 103, 107), "b": (211, 223, 227),
            "c": (307, 311, 313), "d": (401, 409, 419),
        }
        permutations = {
            "a": (0, 1, 2), "b": (1, 2, 0),
            "c": (2, 0, 1), "d": (0, 2, 1),
        }
        cls.cycle_point_ids = {}
        for name, order in permutations.items():
            coordinates = {
                time: np.asarray([[((x + time) % 3), 0, 0] for x in order], dtype=np.int64)
                for time in range(f.lattice[3])
            }
            cls.cycle_point_ids[name] = f._publish_points(
                f"temporal-pp-cycle-points-{name}", cls.cycle_labels[name], coordinates,
            )
        # Ordinary vertex: raw left is its field endpoint a, raw right its bar b.
        cls.cycle_p2p0 = publish_elemental_products(
            f.index, f.root, f.gauge_id, points_id=cls.cycle_point_ids["a"],
            output_name="temporal-pp-cycle-p2p0", paths=(cls.path,),
            momentum_list=cls.f.momentum_list, sectors=("p2p",), usedNp=3,
            points_right_id=cls.cycle_point_ids["b"],
            point_right_labels=cls.cycle_labels["b"],
            phase_convention="midpoint", temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="periodic",
        )["p2p"]
        # Source-dagger vertex: logical field is raw right d, logical bar raw left c.
        cls.cycle_p2p1 = publish_elemental_products(
            f.index, f.root, f.gauge_id, points_id=cls.cycle_point_ids["c"],
            output_name="temporal-pp-cycle-p2p1", paths=(cls.path,),
            momentum_list=cls.f.momentum_list, sectors=("p2p",), usedNp=3,
            points_right_id=cls.cycle_point_ids["d"],
            point_right_labels=cls.cycle_labels["d"],
            phase_convention="midpoint", temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="periodic",
        )["p2p"]
        # PSP edges are bound to the physical point identities declared above.
        cls.cycle_psp0 = publish_perambulators(
            f.index, f.root, f.gauge_id, output_name="temporal-pp-cycle-psp0",
            points_snk_id=cls.cycle_point_ids["c"],
            points_src_id=cls.cycle_point_ids["a"], products=("PSP",),
            Np_snk=3, Np_src=3, mass=0.4, t_boundary=+1,
        )["psp"]
        cls.cycle_psp1 = publish_perambulators(
            f.index, f.root, f.gauge_id, output_name="temporal-pp-cycle-psp1",
            points_snk_id=cls.cycle_point_ids["b"],
            points_src_id=cls.cycle_point_ids["d"], products=("PSP",),
            Np_snk=3, Np_src=3, mass=0.4, t_boundary=+1,
        )["psp"]

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def _basis_matrices(self, time, point_coordinates):
        f = self.f
        lx, ly, lz, _lt = f.lattice
        volume = lx * ly * lz
        modes = f.mode_values[time]
        mode_count = modes.shape[0]
        point_count = len(point_coordinates)
        vector = np.zeros((12 * volume, 4 * mode_count), dtype=np.complex128)
        point = np.zeros((12 * volume, 4 * point_count * 3), dtype=np.complex128)
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    site = (z * ly + y) * lx + x
                    for spin in range(4):
                        row = (site * 4 + spin) * 3
                        vector[row:row + 3, spin * mode_count:(spin + 1) * mode_count] = (
                            modes[:, z, y, x, :].T
                        )
                        for pindex, coordinate in enumerate(point_coordinates):
                            px, py, pz = map(int, coordinate)
                            if (px, py, pz) == (x, y, z):
                                column = spin * point_count * 3 + pindex * 3
                                point[row:row + 3, column:column + 3] = np.eye(3)
        return vector, point

    def _projected_wilson(self, sector, sink_time, source_time,
                          sink_points=None, source_points=None):
        f = self.f
        volume = int(np.prod(f.lattice[:3]))
        slice_size = 12 * volume
        block = self.inverse[
            sink_time * slice_size:(sink_time + 1) * slice_size,
            source_time * slice_size:(source_time + 1) * slice_size,
        ]
        sink_v, sink_p = self._basis_matrices(
            sink_time, () if sink_points is None else sink_points,
        )
        source_v, source_p = self._basis_matrices(
            source_time, () if source_points is None else source_points,
        )
        if sector == "vsp":
            projected = sink_v.conj().T @ block @ source_p
            return projected.reshape(4, 2, 4, len(source_points), 3).transpose(0, 2, 1, 3, 4)
        if sector == "psv":
            projected = sink_p.conj().T @ block @ source_v
            return projected.reshape(4, len(sink_points), 3, 4, 2).transpose(0, 3, 1, 2, 4)
        if sector == "psp":
            projected = sink_p.conj().T @ block @ source_p
            return projected.reshape(
                4, len(sink_points), 3, 4, len(source_points), 3,
            ).transpose(0, 3, 1, 2, 4, 5)
        raise AssertionError(f"unsupported independent projection {sector!r}")

    def _independent_transport(self, anchor, start, path):
        f = self.f
        extent = np.asarray(f.lattice[:3], dtype=np.int64)
        offset = [0, 0, 0, 0]
        value = np.eye(3, dtype=np.complex128)
        position0 = np.asarray(start, dtype=np.int64)
        for step in path.steps:
            axis, sign = GaugePath.AXES[step], GaugePath.SIGNS[step]
            if sign < 0:
                offset[axis] -= 1
            spatial = tuple(((position0 + np.asarray(offset[:3])) % extent).tolist())
            time = (anchor + offset[3]) % f.lattice[3]
            x, y, z = spatial
            link = f.links[time, z, y, x, axis]
            if sign < 0:
                link = link.conj().T
            value = value @ link
            if sign > 0:
                offset[axis] += 1
        return value

    def _independent_p2p_record(self, path, anchor, left_coordinates, right_coordinates,
                                *, boundary="periodic"):
        f = self.f
        dx, dy, dz, dt = path.displacement
        lo, hi = path.time_bounds
        if boundary == "open" and (anchor + lo < 0 or anchor + hi >= f.lattice[3]):
            return {
                "indices": np.empty((0, 2), dtype=np.int64),
                "values": np.empty((0, 3, 3), dtype=np.complex128),
                "point_shape": (len(left_coordinates[anchor]),
                                len(right_coordinates[anchor + dt])),
            }
        endpoint = (anchor + dt) % f.lattice[3]
        target_by_coordinate = {
            tuple(map(int, coordinate)): index
            for index, coordinate in enumerate(right_coordinates[endpoint])
        }
        rows, columns, values = [], [], []
        shift = np.asarray((dx, dy, dz), dtype=np.int64)
        extent = np.asarray(f.lattice[:3], dtype=np.int64)
        for row, coordinate in enumerate(left_coordinates[anchor]):
            target = tuple(((coordinate + shift) % extent).tolist())
            column = target_by_coordinate.get(target)
            if column is None:
                continue
            rows.append(row)
            columns.append(column)
            values.append(self._independent_transport(anchor, coordinate, path))
        return {
            "indices": np.column_stack((rows, columns)).astype(np.int64),
            "values": np.asarray(values, dtype=np.complex128).reshape(-1, 3, 3),
            "point_shape": (len(left_coordinates[anchor]), len(right_coordinates[endpoint])),
        }

    def _independent_vv(self, path, anchor):
        f = self.f
        endpoint = (anchor + path.displacement[3]) % f.lattice[3]
        result = np.zeros((2, 2), dtype=np.complex128)
        for left_mode in range(2):
            for right_mode in range(2):
                for z in range(f.lattice[2]):
                    for y in range(f.lattice[1]):
                        for x in range(f.lattice[0]):
                            start = (x, y, z)
                            transported = self._independent_transport(anchor, start, path)
                            left = f.mode_values[anchor][left_mode, z, y, x].conj()
                            right = f.mode_values[endpoint][right_mode, z, y, x]
                            result[left_mode, right_mode] += left @ transported @ right
        return result

    @staticmethod
    def _dense_p2p(record):
        rows, columns = record["point_shape"]
        dense = np.zeros((rows, 3, columns, 3), dtype=np.complex128)
        for (left, right), value in zip(record["indices"], record["values"], strict=True):
            dense[left, :, right, :] += value
        return dense

    def _full_field_path_matrix(self, path, anchor, spin, *,
                                left_coordinates=None, right_coordinates=None,
                                project_modes=False, boundary="periodic"):
        """Place site/link entries directly in full (t,site,spin,color) space.

        This oracle uses neither sparse records nor the Diagram's generated
        indices, endpoint metadata or projected propagator blocks.
        """
        f = self.f
        lx, ly, lz, lt = f.lattice
        slice_size = 12 * lx * ly * lz
        result = np.zeros_like(self.inverse)
        lower, upper = path.time_bounds
        if boundary == "open" and (anchor + lower < 0 or anchor + upper >= lt):
            return result
        dx, dy, dz, dt = path.displacement
        endpoint = (anchor + dt) % lt
        left_sites = (None if left_coordinates is None else
                      {tuple(map(int, xyz)) for xyz in left_coordinates[anchor]})
        right_sites = (None if right_coordinates is None else
                       {tuple(map(int, xyz)) for xyz in right_coordinates[endpoint]})
        local = np.zeros((slice_size, slice_size), dtype=np.complex128)
        for z in range(lz):
            for y in range(ly):
                for x in range(lx):
                    start = (x, y, z)
                    end = ((x + dx) % lx, (y + dy) % ly, (z + dz) % lz)
                    if ((left_sites is not None and start not in left_sites) or
                            (right_sites is not None and end not in right_sites)):
                        continue
                    row = 12 * ((z * ly + y) * lx + x)
                    column = 12 * ((end[2] * ly + end[1]) * lx + end[0])
                    local[row:row + 12, column:column + 12] = np.kron(
                        spin, self._independent_transport(anchor, start, path),
                    )
        if project_modes:
            left, _ = self._basis_matrices(anchor, ())
            right, _ = self._basis_matrices(endpoint, ())
            local = left @ (left.conj().T @ local @ right) @ right.conj().T
        result[anchor * slice_size:(anchor + 1) * slice_size,
               endpoint * slice_size:(endpoint + 1) * slice_size] = local
        return result

    def _single_bridge_full_field_trace(self, source, sink, *, path=None,
                                        boundary="periodic"):
        point = self._full_field_path_matrix(
            self.path if path is None else path, source, gamma(1),
            left_coordinates=self.f.left_coordinates,
            right_coordinates=self.f.right_coordinates, boundary=boundary,
        )
        normal = self._full_field_path_matrix(
            GaugePath((6,)), sink, gamma(3), project_modes=True,
        )
        return ((0.7 + 0.2j) * (-0.3 + 0.4j)
                * np.trace(point @ self.inverse @ normal @ self.inverse))

    def _single_bridge_oracle(self, source_time, sink_time, *, path=None, boundary="periodic",
                              point_coefficient=0.7 + 0.2j,
                              normal_coefficient=-0.3 + 0.4j):
        path = self.path if path is None else path
        point_record = self._independent_p2p_record(
            path, source_time, self.f.left_coordinates, self.f.right_coordinates,
            boundary=boundary,
        )
        dense_point = self._dense_p2p(point_record)
        vv = self._independent_vv(GaugePath((6,)), sink_time)
        point_endpoint = (source_time + path.displacement[3]) % self.f.lattice[3]
        normal_endpoint = (sink_time + 1) % self.f.lattice[3]
        # Each raw matrix's row belongs to its anchor and its column to the
        # transported endpoint. The inverse connects a column to the next row.
        psv = self._projected_wilson(
            "psv", point_endpoint, sink_time,
            sink_points=self.f.right_coordinates[point_endpoint],
        )
        point_factor = np.einsum("AD,naod->ADnaod", gamma(1), dense_point, optimize=False)
        vv_factor = np.einsum("CB,PO->CBPO", gamma(3), vv, optimize=False)
        coefficient = point_coefficient * normal_coefficient
        # Fixed spin/color/point/mode contraction, independent of the target E
        # graph's generated request list and equation.
        return coefficient * np.einsum(
            "BAOna,DCodP,ADnaod,CBPO->",
            self._projected_wilson(
                "vsp", normal_endpoint, source_time,
                source_points=self.f.left_coordinates[source_time],
            ), psv, point_factor, vv_factor, optimize=False,
        )

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

    def _open_point_elemental_for(self, stack, data_id, left_id, right_id,
                                  left_labels, right_labels, *, path=None):
        path = self.path if path is None else path
        reader = stack.enter_context(self.f.index.open(data_id))
        left = stack.enter_context(self.f.index.open(left_id))
        right = stack.enter_context(self.f.index.open(right_id))
        pool_ids = {}
        pools = {}
        for side, identity, labels in (
                ("bar", left_id, left_labels),
                ("field", right_id, right_labels)):
            group_id = f"temporal-pp-{identity}"
            pools[group_id] = SamplingPoolSpec(
                group_id, spatial_sites=3, sampled_points=len(labels),
            )
            for time in range(self.f.lattice[3]):
                pool_ids[(ElementalComponent.path(path), 0, time, side)] = group_id
        return DatasetElemental(
            {"p2p": reader}, point_dataset=left, point_right_dataset=right,
            sampling_pool_ids=pool_ids, sampling_groups=pools,
        )

    def _open_point_elemental(self, stack):
        return self._open_point_elemental_for(
            stack, self.point_path_ids["p2p"], self.f.left_id, self.f.right_id,
            self.f.left_labels, self.f.right_labels,
        )

    def _open_cycle_point_elemental(self, stack, data_id, left_name, right_name):
        return self._open_point_elemental_for(
            stack, data_id, self.cycle_point_ids[left_name],
            self.cycle_point_ids[right_name], self.cycle_labels[left_name],
            self.cycle_labels[right_name],
        )

    def _build_single_bridge(self, stack, *, path=None, boundary="periodic",
                             point_data_id=None,
                             point_coefficient=0.7 + 0.2j,
                             normal_coefficient=-0.3 + 0.4j):
        path = self.path if path is None else path
        p2p = (self._open_point_elemental(stack) if point_data_id is None else
               self._open_point_elemental_for(
                   stack, point_data_id, self.f.left_id, self.f.right_id,
                   self.f.left_labels, self.f.right_labels, path=path,
               ))
        with self.f.index.open(self.vv_ids["current"]) as full_vv:
            vv_document = full_vv.spec.document
            for axis in vv_document["axes"]:
                if axis["name"] == "component":
                    axis["coordinates"] = [{"path": [6]}, {"path": [7]}]
            vv_view_spec = DatasetSpec(vv_document)
        vv_reader = stack.enter_context(open_dataset_view(
            self.f.index, vv_view_spec, subset_axes=("component",),
            data_id=self.vv_ids["current"],
        ))
        vv = DatasetElemental({"v2v": vv_reader})
        prop_readers = {
            sector: stack.enter_context(self.f.index.open(self.mixed_ids[sector]))
            for sector in ("psv", "vsp")
        }
        propagator = DatasetPropagator(prop_readers)
        q_forward = Propagator("u", Tag(0, 0), Tag(1, 1))
        q_backward = Propagator("u", Tag(1, 1), Tag(0, 0))
        point_operator = OperatorPath(
            "temporal-pp",
            ((1, path, 0, point_coefficient),),
        )
        normal_operator = OperatorPath(
            "normal-one-link-vv",
            ((3, GaugePath((6,)), 0, normal_coefficient),),
        )
        vertices = {
            "point-path": Meson(
                p2p, point_operator, phase_convention="midpoint",
                boundary=boundary, requested_sectors=("p2p",),
            ),
            "normal-vv": Meson(
                vv, normal_operator, phase_convention="midpoint",
                boundary="periodic", requested_sectors=("v2v",),
            ),
        }
        source_time, sink_time = Symbol("source-time"), Symbol("sink-time")
        qdiagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [source_time, sink_time],
            ["point-path", "normal-vv"], [None, q_forward, q_backward],
        )
        bindings = DiagramBindings(
            {q_forward: propagator, q_backward: propagator}, vertices,
            self.f.configuration["configuration"], 2, len(self.f.left_labels), True,
        )
        return qdiagram, bindings, source_time, sink_time, vertices, p2p, propagator

    def _build_dual_bridge_cycle(self, stack):
        p2p0 = self._open_cycle_point_elemental(
            stack, self.cycle_p2p0, "a", "b",
        )
        p2p1 = self._open_cycle_point_elemental(
            stack, self.cycle_p2p1, "c", "d",
        )
        psp0 = stack.enter_context(self.f.index.open(self.cycle_psp0))
        psp1 = stack.enter_context(self.f.index.open(self.cycle_psp1))
        q_forward = Propagator("u", Tag(0, 0), Tag(1, 1))
        q_backward = Propagator("u", Tag(1, 1), Tag(0, 0))
        vertices = {
            "pp-normal": Meson(
                p2p0, OperatorPath("cycle-pp-normal", ((1, self.path, 0, 0.7 + 0.2j),)),
                phase_convention="midpoint", boundary="periodic",
                requested_sectors=("p2p",),
            ),
            "pp-source-dagger": Meson(
                p2p1, OperatorPath("cycle-pp-source", ((3, self.path, 0, -0.3 + 0.4j),)),
                source=True, phase_convention="midpoint", boundary="periodic",
                requested_sectors=("p2p",),
            ),
        }
        source_time, sink_time = Symbol("source-time"), Symbol("sink-time")
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [source_time, sink_time],
            ["pp-normal", "pp-source-dagger"], [None, q_forward, q_backward],
        )
        bindings = DiagramBindings(
            {q_forward: DatasetPropagator(psp0), q_backward: DatasetPropagator(psp1)},
            vertices, self.f.configuration["configuration"], None, 3, True,
        )
        return diagram, bindings, source_time, sink_time, vertices

    def test_explicit_temporal_pp_single_bridge_matches_independent_trace(self):
        with ExitStack() as stack:
            diagram, bindings, source_time, sink_time, vertices, elemental, propagator = (
                self._build_single_bridge(stack)
            )
            metadata = vertices["point-path"].describe_factors(1, "p2p")[0]
            self.assertEqual(metadata.logical_sector, "p2p")
            self.assertEqual(metadata.raw_sector, "p2p")
            self.assertEqual(metadata.field_time, 2)
            self.assertEqual(metadata.bar_time, 1)
            plan = plan_indexed_diagram_measurement(
                self.f.index, diagram, bindings=bindings,
                source_time_symbol=source_time, sink_time_symbol=sink_time,
            )
            p2p_roles = [role for role in plan["input_roles"] if role.endswith("-p2p")]
            self.assertEqual(len(p2p_roles), 1)
            self.assertTrue(any(role.endswith("-v2v") for role in plan["input_roles"]))
            self.assertTrue(any(role.endswith("-psv") for role in plan["input_roles"]))
            self.assertTrue(any(role.endswith("-vsp") for role in plan["input_roles"]))
            values = self._evaluate_grid(diagram, bindings, source_time, sink_time)
            self.assertEqual(values.shape, (self.f.lattice[3], self.f.lattice[3]))
            self.assertTrue(np.all(np.isfinite(values)))
            expected = np.asarray([
                [self._single_bridge_oracle(source, sink)
                 for sink in range(self.f.lattice[3])]
                for source in range(self.f.lattice[3])
            ])
            full_field = np.asarray([
                [self._single_bridge_full_field_trace(source, sink)
                 for sink in range(self.f.lattice[3])]
                for source in range(self.f.lattice[3])
            ])
            np.testing.assert_allclose(expected, full_field, rtol=5e-10, atol=5e-10)
            np.testing.assert_allclose(values, expected, rtol=5e-10, atol=5e-10)

    def test_backward_temporal_pp_matches_full_field_trace(self):
        path = GaugePath((7, 0))
        point_id = publish_elemental_products(
            self.f.index, self.f.root, self.f.gauge_id, points_id=self.f.left_id,
            output_name="temporal-pp-backward", paths=(path,),
            momentum_list=self.f.momentum_list, sectors=("p2p",),
            usedNp=len(self.f.left_labels), points_right_id=self.f.right_id,
            point_right_labels=self.f.right_labels, phase_convention="midpoint",
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA, path_boundary="periodic",
        )["p2p"]
        with ExitStack() as stack:
            diagram, bindings, source_time, sink_time, *_ = self._build_single_bridge(
                stack, path=path, point_data_id=point_id,
            )
            values = self._evaluate_grid(diagram, bindings, source_time, sink_time)
        expected = np.asarray([
            [self._single_bridge_full_field_trace(source, sink, path=path)
             for sink in range(self.f.lattice[3])]
            for source in range(self.f.lattice[3])
        ])
        np.testing.assert_allclose(values, expected, rtol=5e-10, atol=5e-10)

    def test_open_net_zero_internal_excursion_has_zero_rows_and_endpoint_escape_rejects(self):
        for invalid_anchor, net_zero in ((2, GaugePath((6, 7))), (0, GaugePath((7, 6)))):
            with self.subTest(path=net_zero.steps):
                open_ids = publish_elemental_products(
                    self.f.index, self.f.root, self.f.gauge_id,
                    points_id=self.f.left_id,
                    output_name=f"temporal-pp-open-net-zero-{net_zero.idx}",
                    paths=(net_zero,), momentum_list=self.f.momentum_list,
                    sectors=("p2p",), usedNp=len(self.f.left_labels),
                    points_right_id=self.f.right_id,
                    point_right_labels=self.f.right_labels,
                    phase_convention="midpoint", temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
                    path_boundary="open",
                )["p2p"]
                with ExitStack() as stack:
                    diagram, bindings, source_time, sink_time, _vertices, _elemental, _propagator = (
                        self._build_single_bridge(
                            stack, path=net_zero, boundary="open", point_data_id=open_ids,
                        )
                    )
                    values = self._evaluate_grid(diagram, bindings, source_time, sink_time)
                    expected = np.asarray([
                        [self._single_bridge_oracle(
                            source, sink, path=net_zero, boundary="open")
                         for sink in range(self.f.lattice[3])]
                        for source in range(self.f.lattice[3])
                    ])
                    full_field = np.asarray([
                        [self._single_bridge_full_field_trace(
                            source, sink, path=net_zero, boundary="open")
                         for sink in range(self.f.lattice[3])]
                        for source in range(self.f.lattice[3])
                    ])
                    np.testing.assert_allclose(expected, full_field, rtol=5e-10, atol=5e-10)
                    np.testing.assert_allclose(values, expected, rtol=5e-10, atol=5e-10)
                    np.testing.assert_array_equal(
                        values[invalid_anchor], np.zeros(self.f.lattice[3]),
                    )

        invalid = GaugePath((6, 0))
        invalid_id = publish_elemental_products(
            self.f.index, self.f.root, self.f.gauge_id,
            points_id=self.f.left_id,
            output_name="temporal-pp-open-endpoint-escape",
            paths=(invalid,), momentum_list=self.f.momentum_list,
            sectors=("p2p",), usedNp=len(self.f.left_labels),
            points_right_id=self.f.right_id,
            point_right_labels=self.f.right_labels,
            phase_convention="midpoint", temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="open",
        )["p2p"]
        with ExitStack() as stack:
            diagram, bindings, source_time, sink_time, _vertices, _elemental, _propagator = (
                self._build_single_bridge(
                    stack, path=invalid, boundary="open", point_data_id=invalid_id,
                )
            )
            with patch.object(Meson, "read_factor_block",
                              side_effect=AssertionError("open preflight read elemental")) as meson_read:
                with patch.object(DatasetPropagator, "block",
                                  side_effect=AssertionError("open preflight read propagator")) as prop_read:
                    with self.assertRaisesRegex(ValueError, "open temporal domain"):
                        plan_indexed_diagram_measurement(
                            self.f.index, diagram, bindings=bindings,
                            source_time_symbol=source_time, sink_time_symbol=sink_time,
                        )
            self.assertEqual(meson_read.call_count, 0)
            self.assertEqual(prop_read.call_count, 0)

    def test_temporal_pp_dual_bridge_cycle_preflights_existing_sparse_contract(self):
        with ExitStack() as stack:
            diagram, bindings, source_time, sink_time, vertices = self._build_dual_bridge_cycle(stack)
            normal_factor = vertices["pp-normal"].describe_factors(0, "p2p")[0]
            source_factor = vertices["pp-source-dagger"].describe_factors(0, "p2p")[0]
            self.assertFalse(normal_factor.dagger)
            self.assertTrue(source_factor.dagger)
            self.assertEqual(
                normal_factor.tensor.axis_bindings["point_left"],
                (self.cycle_point_ids["a"], self.cycle_labels["a"]),
            )
            self.assertEqual(
                normal_factor.tensor.axis_bindings["point_right"],
                (self.cycle_point_ids["b"], self.cycle_labels["b"]),
            )
            self.assertEqual(
                source_factor.tensor.axis_bindings["point_left"],
                (self.cycle_point_ids["c"], self.cycle_labels["c"]),
            )
            self.assertEqual(
                source_factor.tensor.axis_bindings["point_right"],
                (self.cycle_point_ids["d"], self.cycle_labels["d"]),
            )
            raw_record = vertices["pp-source-dagger"].read_factor_block(source_factor)
            try:
                source_parts = vertices["pp-source-dagger"].materialize_sparse_bridge_part(
                    source_factor, raw_record,
                )
                logical_record = source_parts[0][2]
                np.testing.assert_array_equal(
                    logical_record["indices"], raw_record["indices"][:, ::-1],
                )
                np.testing.assert_array_equal(
                    logical_record["values"],
                    raw_record["values"].conj().swapaxes(-1, -2),
                )
            finally:
                vertices["pp-source-dagger"].release_factor_block(
                    source_factor, raw_record,
                )
            # The source dagger reads the raw right endpoint as logical field
            # and the raw left endpoint as logical bar; their actual identities
            # are independently bound by the two PSP datasets.
            for data_id, sink_name, source_name in (
                    (self.cycle_psp0, "c", "a"),
                    (self.cycle_psp1, "b", "d")):
                with self.f.index.open(data_id) as psp:
                    bases = psp.spec.document["bases"]
                    self.assertEqual(bases["sink_point"]["data_id"],
                                     self.cycle_point_ids[sink_name])
                    self.assertEqual(bases["sink_point"]["indices"],
                                     list(self.cycle_labels[sink_name]))
                    self.assertEqual(bases["source_point"]["data_id"],
                                     self.cycle_point_ids[source_name])
                    self.assertEqual(bases["source_point"]["indices"],
                                     list(self.cycle_labels[source_name]))
            plan = plan_indexed_diagram_measurement(
                self.f.index, diagram, bindings=bindings,
                source_time_symbol=source_time, sink_time_symbol=sink_time,
            )
            self.assertTrue(plan["measurement_spec"].definition_id)
            values = self._evaluate_grid(diagram, bindings, source_time, sink_time)
            self.assertEqual(values.shape, (self.f.lattice[3], self.f.lattice[3]))
            self.assertTrue(np.all(np.isfinite(values)))
            expected = np.empty_like(values)
            full_field = np.empty_like(values)
            spin_adjoint = np.kron(
                np.eye(self.inverse.shape[0] // 12), np.kron(gamma(8), np.eye(3)),
            )
            for source in range(self.f.lattice[3]):
                for sink in range(self.f.lattice[3]):
                    rec0 = self._independent_p2p_record(
                        self.path, source,
                        self._cycle_coordinates("a"), self._cycle_coordinates("b"),
                    )
                    rec1 = self._independent_p2p_record(
                        self.path, sink,
                        self._cycle_coordinates("c"), self._cycle_coordinates("d"),
                    )
                    dense0 = self._dense_p2p(rec0)
                    dense1 = self._dense_p2p(rec1).conj().transpose(2, 3, 0, 1)
                    spin1 = gamma(8) @ gamma(3).conj().T @ gamma(8)
                    factor0 = np.einsum("AD,naob->ADnaob", gamma(1), dense0, optimize=False)
                    factor1 = np.einsum("CB,pcqd->CBpcqd", spin1, dense1, optimize=False)
                    psp0 = self._projected_wilson(
                        # Raw left a/c rows live at the two anchors, including
                        # the column of the source-daggered second vertex.
                        "psp", sink, source,
                        sink_points=self._cycle_coordinates("c")[sink],
                        source_points=self._cycle_coordinates("a")[source],
                    )
                    source_endpoint = (source + self.path.displacement[3]) % self.f.lattice[3]
                    sink_endpoint = (sink + self.path.displacement[3]) % self.f.lattice[3]
                    psp1 = self._projected_wilson(
                        # Raw right b/d columns live at transported endpoints.
                        "psp", source_endpoint, sink_endpoint,
                        sink_points=self._cycle_coordinates("b")[source_endpoint],
                        source_points=self._cycle_coordinates("d")[sink_endpoint],
                    )
                    expected[source, sink] = (
                        (0.7 + 0.2j) * (-0.3 + 0.4j).conjugate()
                        * np.einsum(
                            "BAqdna,DCobpc,ADnaob,CBpcqd->",
                            psp0, psp1, factor0, factor1, optimize=False,
                        )
                    )
                    kernel0 = self._full_field_path_matrix(
                        self.path, source, gamma(1),
                        left_coordinates=self._cycle_coordinates("a"),
                        right_coordinates=self._cycle_coordinates("b"),
                    )
                    kernel1 = self._full_field_path_matrix(
                        self.path, sink, gamma(3),
                        left_coordinates=self._cycle_coordinates("c"),
                        right_coordinates=self._cycle_coordinates("d"),
                    )
                    dagger1 = spin_adjoint @ kernel1.conj().T @ spin_adjoint
                    full_field[source, sink] = (
                        (0.7 + 0.2j) * (-0.3 + 0.4j).conjugate()
                        * np.trace(kernel0 @ self.inverse @ dagger1 @ self.inverse)
                    )
            np.testing.assert_allclose(expected, full_field, rtol=8e-10, atol=8e-10)
            np.testing.assert_allclose(values, expected, rtol=8e-10, atol=8e-10)

            # Deliberately attach the reverse edge's real PSP to the forward
            # edge.  The actual endpoint IDs/ordered labels disagree, so the
            # public planner must refuse before payload getters or output creation.
            q_forward, q_backward = tuple(bindings.propagator_map)
            wrong_forward = bindings.propagator_map[q_backward]
            wrong_bindings = DiagramBindings(
                {q_forward: wrong_forward, q_backward: wrong_forward},
                vertices, self.f.configuration["configuration"], None, 3, True,
            )
            with patch.object(Meson, "read_factor_block",
                              side_effect=AssertionError("endpoint preflight read elemental")):
                with patch.object(DatasetPropagator, "block",
                                  side_effect=AssertionError("endpoint preflight read propagator")):
                    with self.assertRaisesRegex(ValueError, "actual dataset ID or ordered coordinates"):
                        plan_indexed_diagram_measurement(
                            self.f.index, diagram, bindings=wrong_bindings,
                            source_time_symbol=source_time, sink_time_symbol=sink_time,
                        )

    def _cycle_coordinates(self, name):
        return {
            time: np.asarray([[(x + time) % 3, 0, 0]
                              for x in ({"a": (0, 1, 2), "b": (1, 2, 0),
                                         "c": (2, 0, 1), "d": (0, 2, 1)}[name])],
                             dtype=np.int64)
            for time in range(self.f.lattice[3])
        }


if __name__ == "__main__":
    unittest.main()
