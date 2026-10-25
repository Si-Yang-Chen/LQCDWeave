"""Registered temporal point-current raw blocks and exact endpoint identities."""

from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack
from numbers import Integral
import unittest
from unittest.mock import patch

import numpy as np
from sympy import Symbol

from test_wilson_pipeline import random_gauge
from lqcdweave.adapters.meson_dataset import DatasetElemental, DatasetPropagator
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.contracts.current import CURRENT_DIRECTED_POINT_SCHEMA
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, ElementalComponent, GaugePath,
)
from lqcdweave.contracts.spatial_sampling import SamplingPoolSpec
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.single_hadron.current import ConservedVectorCurrent
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.runtime.datasets import open_dataset_view
from lqcdweave.contraction import diagram_measurement
from lqcdweave.contraction.correlator import calc_diagram_eval
from fixtures.c_group_production import (
    publish_directed_current,
    publish_elemental_products,
    publish_perambulators,
)


class RegisteredTemporalPointCurrentTests(unittest.TestCase):
    lattice = (4, 1, 1, 3)
    left_labels = (41, 7)
    right_labels = (19, 5, 12)
    momentum_list = ((0, 0, 0), (1, 0, 0))
    paths = (GaugePath((6,)), GaugePath((7,)))

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.configuration = dict(
            ensemble="temporal-current-reference",
            configuration="cfg-temporal-current",
            lattice=list(self.lattice),
        )
        self.links = random_gauge(self.lattice, seed=73041)
        self.gauge_id = self._publish_gauge()
        self.mode_values = self._mode_values()
        for time in range(self.lattice[3]):
            flattened = self.mode_values[time].reshape(2, -1)
            np.testing.assert_allclose(flattened @ flattened.conj().T,
                                       np.eye(2), rtol=2e-14, atol=2e-14)
        self.basis_id = self._publish_basis()
        self.left_coordinates = self._left_coordinates()
        self.right_coordinates = self._right_coordinates()
        self.left_id = self._publish_points("left-points", self.left_labels,
                                            self.left_coordinates)
        self.right_id = self._publish_points("right-points", self.right_labels,
                                             self.right_coordinates)
        self.assertNotEqual(self.left_id, self.right_id)

    def _publish(self, name, spec, rows):
        directory = self.root / name
        with DatasetWriter(directory, spec, provenance={"test": "temporal-point-current"}) as writer:
            for position, time in enumerate(spec.coordinates("time")):
                writer.write(position, np.asarray(rows[time], dtype=spec.document["dtype"]))
            identity = writer.publish()
        self.index.register(directory)
        return identity

    def _time_axis(self):
        # Store time rows out of physical order so every consumer must honor labels.
        return dict(name="time", coordinates=[2, 0, 1])

    def _publish_gauge(self):
        axes = [self._time_axis()]
        axes.extend(dict(name=name, coordinates=list(range(size)))
                    for name, size in zip(
                        ("z", "y", "x", "direction", "color_row", "color_column"),
                        self.links.shape[1:], strict=True))
        spec = DatasetSpec(dict(
            schema=1, kind="gauge", configuration=self.configuration,
            axes=axes, dtype="complex128", bases={}, parents=[], definition={},
            conventions=dict(boundary="periodic"),
        ))
        return self._publish("gauge", spec,
                             {time: self.links[time] for time in range(self.lattice[3])})

    def _mode_values(self):
        rng = np.random.default_rng(73042)
        result = {}
        for time in range(self.lattice[3]):
            matrix = (rng.normal(size=(self.lattice[0] * 3, 2))
                      + 1j * rng.normal(size=(self.lattice[0] * 3, 2)))
            orthonormal, _ = np.linalg.qr(matrix)
            result[time] = orthonormal[:, :2].T.reshape(
                2, 1, 1, self.lattice[0], 3).astype(np.complex128)
        return result

    def _publish_basis(self):
        spec = DatasetSpec(dict(
            schema=1, kind="explicit_basis", configuration=self.configuration,
            axes=[self._time_axis(),
                  dict(name="mode", coordinates=[17, 5]),
                  dict(name="z", coordinates=[0]), dict(name="y", coordinates=[0]),
                  dict(name="x", coordinates=list(range(self.lattice[0]))),
                  dict(name="color", coordinates=[0, 1, 2])],
            dtype="complex128", bases={}, parents=[self.gauge_id],
            definition=dict(realization="time-varying-complex-mode-reference"),
            conventions=dict(normalization="orthonormal"),
        ))
        return self._publish("basis", spec, self.mode_values)

    def _left_coordinates(self):
        if self.lattice[0] == 4:
            positions = lambda time: ((0 + time) % 4, (2 + time) % 4)
        else:
            positions = lambda time: tuple(
                (point + time) % self.lattice[0]
                for point in range(len(self.left_labels)))
        return {time: np.asarray([[x, 0, 0] for x in positions(time)], dtype=np.int64)
                for time in range(self.lattice[3])}

    def _right_coordinates(self):
        if self.lattice[0] == 4:
            positions = lambda time: tuple((point + time) % 4 for point in (3, 0, 1))
        else:
            positions = lambda time: tuple(
                (2 * point + time + 1) % self.lattice[0]
                for point in range(len(self.right_labels)))
        return {time: np.asarray([[x, 0, 0] for x in positions(time)], dtype=np.int64)
                for time in range(self.lattice[3])}

    def _publish_points(self, name, labels, coordinates):
        spec = DatasetSpec(dict(
            schema=1, kind="spatial_points", configuration=self.configuration,
            axes=[self._time_axis(), dict(name="point", coordinates=list(labels)),
                  dict(name="xyz", coordinates=["x", "y", "z"])],
            dtype="int64", bases={}, parents=[],
            definition=dict(selection="time-varying-explicit-ordered-points"),
            conventions=dict(representation="coordinate_delta", normalization="unit_delta"),
        ))
        return self._publish(name, spec, coordinates)

    def _produce_point_current(self, boundary="periodic", name="temporal-current", *,
                               left_id=None, right_id=None,
                               left_labels=None, right_labels=None):
        left_id = self.left_id if left_id is None else left_id
        right_id = self.right_id if right_id is None else right_id
        left_labels = self.left_labels if left_labels is None else tuple(left_labels)
        right_labels = self.right_labels if right_labels is None else tuple(right_labels)
        identities = publish_elemental_products(
            self.index, self.root, self.gauge_id, self.basis_id, left_id,
            output_name=name, paths=self.paths, momentum_list=self.momentum_list,
            sectors=("v2p", "p2v", "p2p"), usedNe=2,
            usedNp=len(left_labels), points_right_id=right_id,
            point_right_labels=right_labels,
            phase_convention="midpoint", current_schema=CURRENT_DIRECTED_POINT_SCHEMA,
            path_boundary=boundary,
        )
        return {"datasets": identities}

    def _reset_to_census_current_fixture(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.lattice = (3, 1, 1, 3)
        self.left_labels = (17, 4, 12)
        self.right_labels = (9, 23, 8)
        self.configuration = dict(
            ensemble="temporal-current-reference",
            configuration="cfg-temporal-current-census",
            lattice=list(self.lattice),
        )
        self.links = random_gauge(self.lattice, seed=73051)
        self.gauge_id = self._publish_gauge()
        self.mode_values = self._mode_values()
        for time in range(self.lattice[3]):
            flattened = self.mode_values[time].reshape(2, -1)
            np.testing.assert_allclose(flattened @ flattened.conj().T,
                                       np.eye(2), rtol=2e-14, atol=2e-14)
        self.basis_id = self._publish_basis()
        self.left_coordinates = self._left_coordinates()
        self.right_coordinates = self._right_coordinates()
        self.left_id = self._publish_points("census-points-a", self.left_labels,
                                            self.left_coordinates)
        self.right_id = self._publish_points("census-points-b", self.right_labels,
                                             self.right_coordinates)

    def _vector_link(self, anchor, step, x):
        if step == 6:
            return self.links[anchor, 0, 0, x, 3]
        return self.links[(anchor - 1) % self.lattice[3], 0, 0, x, 3].conj().T

    def _phase(self, momentum, x):
        return np.exp(2j * np.pi * momentum[0] * x / self.lattice[0])

    def test_registered_temporal_vp_pv_pp_match_independent_time_endpoint_loops(self):
        result = self._produce_point_current()
        self.assertEqual(set(result["datasets"]), {"v2p", "p2v", "p2p"})
        with (self.index.open(result["datasets"]["v2p"]) as v2p,
              self.index.open(result["datasets"]["p2v"]) as p2v,
              self.index.open(result["datasets"]["p2p"]) as p2p,
              self.index.open(self.left_id) as left_points,
              self.index.open(self.right_id) as right_points):
            elemental = DatasetElemental(
                {"v2p": v2p, "p2v": p2v, "p2p": p2p},
                point_dataset=left_points, point_right_dataset=right_points,
            )
            for sector, reader in (("v2p", v2p), ("p2v", p2v), ("p2p", p2p)):
                self.assertEqual(
                    reader.spec.document["definition"]["directed_current_schema"],
                    CURRENT_DIRECTED_POINT_SCHEMA,
                )
                self.assertEqual(reader.spec.document["definition"]["path_boundary"],
                                 "periodic")
                self.assertIn(self.gauge_id, reader.spec.document["parents"])
                if sector == "v2p":
                    self.assertEqual(reader.spec.document["bases"]["mode_left"]["data_id"],
                                     self.basis_id)
                    self.assertEqual(reader.spec.document["bases"]["point_right"]["data_id"],
                                     self.right_id)
                    self.assertEqual(reader.spec.coordinates("point_right"), list(self.right_labels))
                elif sector == "p2v":
                    self.assertEqual(reader.spec.document["bases"]["point_left"]["data_id"],
                                     self.left_id)
                    self.assertEqual(reader.spec.coordinates("point_left"), list(self.left_labels))
                    self.assertEqual(reader.spec.document["bases"]["mode_right"]["data_id"],
                                     self.basis_id)
                else:
                    self.assertEqual(reader.spec.document["bases"]["point_left"]["data_id"],
                                     self.left_id)
                    self.assertEqual(reader.spec.document["definition"]["point_right_id"],
                                     self.right_id)
                    self.assertEqual(reader.spec.document["definition"]["point_right_labels"],
                                     list(self.right_labels))

            for step in (6, 7):
                component = ElementalComponent.path(GaugePath((step,)))
                delta = 1 if step == 6 else -1
                for anchor in range(self.lattice[3]):
                    endpoint = (anchor + delta) % self.lattice[3]
                    left = self.left_coordinates[anchor]
                    right = self.right_coordinates[endpoint]
                    for momentum_index, momentum in enumerate(self.momentum_list):
                        expected_v2p = np.zeros((2, 3, 3), dtype=np.complex128)
                        for mode in range(2):
                            for point, (x, _y, _z) in enumerate(right):
                                expected_v2p[mode, point] = (
                                    self._phase(momentum, int(x))
                                    * self.mode_values[anchor][mode, 0, 0, x].conj()
                                    @ self._vector_link(anchor, step, int(x))
                                )
                        actual_v2p = elemental.block(
                            component, momentum_index, anchor, "v2p", usedNe=2, usedNp=3,
                        )
                        np.testing.assert_allclose(actual_v2p, expected_v2p,
                                                   rtol=3e-13, atol=3e-13)

                        expected_p2v = np.zeros((2, 3, 2), dtype=np.complex128)
                        for point, (x, _y, _z) in enumerate(left):
                            for mode in range(2):
                                expected_p2v[point, :, mode] = (
                                    self._phase(momentum, int(x))
                                    * self._vector_link(anchor, step, int(x))
                                    @ self.mode_values[endpoint][mode, 0, 0, x]
                                )
                        actual_p2v = elemental.block(
                            component, momentum_index, anchor, "p2v", usedNe=2, usedNp=2,
                        )
                        np.testing.assert_allclose(actual_p2v, expected_p2v,
                                                   rtol=3e-13, atol=3e-13)

                        expected_pairs = []
                        expected_values = []
                        for left_index, left_xyz in enumerate(left):
                            for right_index, right_xyz in enumerate(right):
                                if np.array_equal(left_xyz, right_xyz):
                                    x = int(left_xyz[0])
                                    expected_pairs.append((left_index, right_index))
                                    expected_values.append(
                                        self._phase(momentum, x)
                                        * self._vector_link(anchor, step, x)
                                    )
                        actual_p2p = elemental.block(component, momentum_index, anchor, "p2p")
                        self.assertEqual(list(map(tuple, actual_p2p["indices"].tolist())),
                                         expected_pairs)
                        expected_values = (np.stack(expected_values) if expected_values else
                                           np.empty((0, 3, 3), dtype=np.complex128))
                        np.testing.assert_allclose(actual_p2p["values"], expected_values,
                                                   rtol=3e-13, atol=3e-13)

    def test_composite_point_sectors_reject_mixed_product_schemas(self):
        current_ids = self._produce_point_current(name="current-schema-inputs")["datasets"]
        temporal_ids = publish_elemental_products(
            self.index, self.root, self.gauge_id, self.basis_id, self.left_id,
            output_name="temporal-path-schema-input", paths=self.paths,
            momentum_list=self.momentum_list, sectors=("p2v",), usedNe=2,
            usedNp=len(self.left_labels), phase_convention="midpoint",
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA,
            path_boundary="periodic",
        )

        with (self.index.open(current_ids["v2p"]) as current_v2p,
              self.index.open(temporal_ids["p2v"]) as temporal_p2v):
            with self.assertRaisesRegex(
                    ValueError, "point-ended elemental sectors must share"):
                DatasetElemental({
                    "v2p": current_v2p,
                    "p2v": temporal_p2v,
                })

    def test_open_temporal_pp_production_keeps_invalid_endpoint_support_empty(self):
        result = self._produce_point_current(boundary="open", name="open-temporal-current")
        with (self.index.open(result["datasets"]["p2p"]) as p2p,
              self.index.open(self.left_id) as left_points,
              self.index.open(self.right_id) as right_points):
            elemental = DatasetElemental(
                {"p2p": p2p}, point_dataset=left_points,
                point_right_dataset=right_points,
            )
            invalid = elemental.block(ElementalComponent.path(GaugePath((6,))), 0, 2, "p2p")
            self.assertEqual(invalid["indices"].shape, (0, 2))
            self.assertEqual(invalid["values"].shape, (0, 3, 3))

    def test_registered_eight_link_vv_opens_exact_temporal_component_view(self):
        actual_id = publish_directed_current(
            self.index, self.root, self.gauge_id, self.basis_id,
            output_name="directed-vv", momentum_list=((0, 0, 0),),
            usedNe=2, boundary="periodic",
        )
        with self.index.open(actual_id) as full:
            self.assertEqual(full.spec.coordinates("component"),
                             [{"path": [index]} for index in range(8)])
            definition = full.spec.document["definition"]
            self.assertEqual(definition["path_boundary"], "periodic")
            self.assertEqual(definition["directed_basis"],
                             "lattice.current.directed-one-link-basis/v1")
            selected_document = full.spec.document
            for axis in selected_document["axes"]:
                if axis["name"] == "component":
                    axis["coordinates"] = [{"path": [6]}, {"path": [7]}]
            selected = DatasetSpec(selected_document)

        with open_dataset_view(
                self.index, selected, subset_axes=("component",),
                data_id=actual_id) as view:
            self.assertEqual(view.data_id, actual_id)
            self.assertEqual(view.spec.coordinates("component"),
                             [{"path": [6]}, {"path": [7]}])
            elemental = DatasetElemental({"v2v": view})
            for time in range(self.lattice[3]):
                for step in (6, 7):
                    if step == 6:
                        field_time = (time + 1) % self.lattice[3]
                        link = self.links[time, 0, 0, :, 3]
                    else:
                        field_time = (time - 1) % self.lattice[3]
                        link = self.links[field_time, 0, 0, :, 3].conj().swapaxes(-1, -2)
                    expected = np.einsum(
                        "eza, zac, fzc->ef",
                        self.mode_values[time][:, 0, 0].conj(), link,
                        self.mode_values[field_time][:, 0, 0],
                    )
                    actual = elemental.block(
                        ElementalComponent.path(GaugePath((step,))), 0, time,
                        "v2v", usedNe=2,
                    )
                    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def _produce_census_diagram_inputs(self):
        self._reset_to_census_current_fixture()
        self.momentum_list = ((0, 0, 0),)
        elemental_ids = {}
        for pool, identity in (("a", self.left_id), ("b", self.right_id)):
            result = self._produce_point_current(
                name=f"point-current-{pool}", left_id=identity, right_id=identity,
                left_labels=(self.left_labels if pool == "a" else self.right_labels),
                right_labels=(self.left_labels if pool == "a" else self.right_labels),
            )
            elemental_ids[pool] = dict(result["datasets"])
        vv_id = publish_directed_current(
            self.index, self.root, self.gauge_id, self.basis_id,
            output_name="directed-vv", momentum_list=self.momentum_list,
            usedNe=2, boundary="periodic",
        )
        wilson_ids = {}
        for name, points_sink, points_source in (
                ("ab", self.right_id, self.left_id),
                ("ba", self.left_id, self.right_id)):
            wilson_ids[name] = publish_perambulators(
                self.index, self.root, self.gauge_id,
                output_name=f"wilson-{name}", basis_id=self.basis_id,
                points_snk_id=points_sink, points_src_id=points_source,
                products=("VSV", "PSV", "VSP", "PSP"),
                Ne=2, Np_snk=3, Np_src=3, mass=0.4, t_boundary=+1,
            )
        return vv_id, elemental_ids, wilson_ids

    def _open_census_diagram(self, stack, vv_id, elemental_ids, wilson_ids):
        with self.index.open(vv_id) as full:
            view_document = full.spec.document
            for axis in view_document["axes"]:
                if axis["name"] == "component":
                    axis["coordinates"] = [{"path": [6]}, {"path": [7]}]
            view_spec = DatasetSpec(view_document)
        vv_view = stack.enter_context(open_dataset_view(
            self.index, view_spec, subset_axes=("component",), data_id=vv_id,
        ))
        readers = {}

        def open_once(identity):
            if identity not in readers:
                readers[identity] = stack.enter_context(self.index.open(identity))
            return readers[identity]

        operators = {name: ConservedVectorCurrent().operator(3) for name in ("a", "b")}
        vertices = {}
        callers = {"vv_view": vv_view, "elementals": [], "propagators": []}
        for name in ("a", "b"):
            sector_readers = {
                sector: (vv_view if sector == "v2v"
                         else open_once(elemental_ids[name][sector]))
                for sector in ("v2v", "v2p", "p2v", "p2p")
            }
            point_id = self.left_id if name == "a" else self.right_id
            points = open_once(point_id)
            pool_ids, pools = {}, {}
            for term in operators[name].terms:
                for time in range(self.lattice[3]):
                    pool_id = f"census-pool-{name}-endpoint-{time}"
                    pools[pool_id] = SamplingPoolSpec(
                        pool_id, spatial_sites=3, sampled_points=3,
                    )
                    for side in ("bar", "field"):
                        # Meson.describe_factors queries point_pool_id at the
                        # already resolved physical endpoint time (including
                        # source-dagger side exchange), so time is not shifted here.
                        pool_ids[(term.component, 0, time, side)] = pool_id
            elemental = DatasetElemental(
                sector_readers, point_dataset=points, point_right_dataset=points,
                sampling_pool_ids=pool_ids,
                sampling_groups=pools,
            )
            callers["elementals"].append(elemental)
            vertices[f"current-{name}"] = Meson(
                elemental, operators[name], source=(name == "b"),
                phase_convention="midpoint", boundary="periodic",
            )

        prop_symbols = {
            "ab": Propagator("u", Tag(0, 0), Tag(1, 1)),
            "ba": Propagator("u", Tag(1, 1), Tag(0, 0)),
        }
        propagators = {}
        for orientation, symbol in prop_symbols.items():
            sector_readers = {
                sector: open_once(wilson_ids[orientation][sector])
                for sector in ("vsv", "psv", "vsp", "psp")
            }
            propagator = DatasetPropagator(sector_readers)
            callers["propagators"].append(propagator)
            propagators[symbol] = propagator
        source_time, sink_time = Symbol("source-time"), Symbol("sink-time")
        diagram = Diagram(
            QuarkDiagram([[0, 1], [2, 0]]), [source_time, sink_time],
            ["current-a", "current-b"],
            [None, prop_symbols["ab"], prop_symbols["ba"]],
        )
        bindings = DiagramBindings(
            propagators, vertices, self.configuration["configuration"],
            2, 3, True,
        )
        return diagram, bindings, source_time, sink_time, callers

    def test_four_sector_census_chain_matches_dense_independent_oracle(self):
        vv_id, elemental_ids, wilson_ids = self._produce_census_diagram_inputs()
        modes = np.stack([self.mode_values[time] for time in range(self.lattice[3])])
        points_a = np.stack([self.left_coordinates[time] for time in range(self.lattice[3])])
        points_b = np.stack([self.right_coordinates[time] for time in range(self.lattice[3])])
        expected = projected_current_trace_grid(
            self.links, modes, modes, points_a, points_b, 0.4,
            momentum=(0, 0, 0),
        )
        with ExitStack() as stack:
            diagram, bindings, source_time, sink_time, callers = self._open_census_diagram(
                stack, vv_id, elemental_ids, wilson_ids,
            )
            plan = diagram_measurement.plan_indexed_diagram_measurement(
                self.index, diagram, bindings=bindings,
                source_time_symbol=source_time, sink_time_symbol=sink_time,
            )
            self.assertEqual(set(plan["source_times"]), {0, 1, 2})
            self.assertEqual(set(plan["sink_times"]), {0, 1, 2})
            for role in ("vertex-0-v2v", "vertex-0-v2p", "vertex-0-p2v", "vertex-0-p2p",
                         "vertex-1-v2v", "vertex-1-v2p", "vertex-1-p2v", "vertex-1-p2p"):
                self.assertIn(role, plan["input_roles"])
                node = plan["input_roles"][role]
                self.assertEqual(plan["graph"]["stages"][node]["action"], "reuse")
                if role.endswith("-v2v"):
                    selected = DatasetSpec(plan["requests"][node].template)
                    self.assertEqual(selected.coordinates("component"),
                                     [{"path": [6]}, {"path": [7]}])
                    self.assertIn("component", plan["requests"][node].subset_axes)
            for orientation in wilson_ids.values():
                for sector in ("vsv", "psv", "vsp", "psp"):
                    identity = orientation[sector]
                    with self.index.open(identity) as product:
                        self.assertEqual(product.spec.document["conventions"]["temporal_boundary"],
                                         "periodic")
            observed_elemental_sectors = []
            observed_propagator_sectors = []
            original_elemental_block = DatasetElemental.block
            original_propagator_block = DatasetPropagator.block

            def observe_elemental_block(reader, component, momentum, time,
                                        sector, **selection):
                observed_elemental_sectors.append(sector)
                return original_elemental_block(
                    reader, component, momentum, time, sector, **selection,
                )

            def observe_propagator_block(reader, sink_time, source_time,
                                         sector="vsv", **selection):
                observed_propagator_sectors.append(sector)
                return original_propagator_block(
                    reader, sink_time, source_time, sector, **selection,
                )

            actual = np.empty_like(expected)
            with patch.object(DatasetElemental, "block", new=observe_elemental_block), \
                 patch.object(DatasetPropagator, "block", new=observe_propagator_block):
                for source in range(self.lattice[3]):
                    for sink in range(self.lattice[3]):
                        calculation = calc_diagram_eval(
                            diagram, propagator_map=bindings.propagator_map,
                            vertex_map=bindings.vertex_map,
                            configuration=bindings.configuration,
                            usedNe=bindings.usedNe, usedNp=bindings.usedNp,
                            groups_independent=bindings.groups_independent,
                            time_axes={source_time: source, sink_time: sink},
                            optimize="none", backend=np,
                        )
                        actual[source, sink] = calculation.value
            self.assertEqual(set(observed_elemental_sectors), {"v2v", "v2p", "p2v", "p2p"})
            self.assertEqual(set(observed_propagator_sectors), {"vsv", "psv", "vsp", "psp"})
            np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-10)

# Independent NumPy formula copied from parent-mixed-current-dense-oracle.py;
# these helpers have no 本软件 imports and are test-local.
_GAMMA = np.asarray(
    [
        [
            [0, 0, 0, 1j],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [-1j, 0, 0, 0],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ],
        [
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [-1j, 0, 0, 0],
            [0, 1j, 0, 0],
        ],
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
    ],
    dtype=np.complex128,
)
_ID4 = np.eye(4, dtype=np.complex128)
_ID3 = np.eye(3, dtype=np.complex128)


def _positive_lattice(links):
    raw = np.asarray(links)
    if raw.ndim != 7 or raw.shape[-3:] != (4, 3, 3):
        raise ValueError("links must have shape (Lt,Lz,Ly,Lx,4,3,3)")
    if any(type(size) is not int or size < 1 for size in raw.shape[:4]):
        raise ValueError("all temporal and spatial extents must be positive integers")
    if raw.dtype.kind not in "fciu":
        raise TypeError("links must be a real or complex numeric array")
    value = np.asarray(raw, dtype=np.complex128)
    if not np.isfinite(value).all():
        raise ValueError("links must contain only finite values")
    return value


def _mass_value(mass):
    if isinstance(mass, (bool, np.bool_)) or not np.isscalar(mass):
        raise TypeError("mass must be a finite real scalar")
    if not np.isreal(mass) or not np.isfinite(mass):
        raise ValueError("mass must be a finite real scalar")
    return float(mass)


def _dense_wilson_matrix(links, mass):
    """Assemble the periodic isotropic r=1 Wilson matrix by site loops."""
    links = _positive_lattice(links)
    mass = _mass_value(mass)
    lt, lz, ly, lx = links.shape[:4]
    site_count = lt * lz * ly * lx
    dimension = 12 * site_count
    matrix = (mass + 4.0) * np.eye(dimension, dtype=np.complex128)
    sizes = (lt, lz, ly, lx)
    # Mu order is x,y,z,t while the field layout is t,z,y,x.
    axes = (3, 2, 1, 0)
    for point in np.ndindex(sizes):
        site = int(np.ravel_multi_index(point, sizes))
        row = slice(12 * site, 12 * site + 12)
        for mu, axis in enumerate(axes):
            ahead = list(point)
            ahead[axis] = (ahead[axis] + 1) % sizes[axis]
            behind = list(point)
            behind[axis] = (behind[axis] - 1) % sizes[axis]
            forward_site = int(np.ravel_multi_index(tuple(ahead), sizes))
            backward_site = int(np.ravel_multi_index(tuple(behind), sizes))
            forward = np.kron(_ID4 - _GAMMA[mu], links[(*point, mu)])
            backward = np.kron(
                _ID4 + _GAMMA[mu],
                links[(*behind, mu)].conj().T,
            )
            matrix[row, 12 * forward_site:12 * forward_site + 12] -= 0.5 * forward
            matrix[row, 12 * backward_site:12 * backward_site + 12] -= 0.5 * backward
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Wilson matrix contains nonfinite values")
    return matrix


def dense_wilson_inverse(links, mass):
    """Return S=D_W^-1 as a 2-D matrix in flattened (t,z,y,x,spin,color) order.

    The stencil uses periodic fermion boundary sign +1 in all four directions,
    isotropic r=1, diagonal term (mass + 4), and the literal gamma matrices in
    this module. It does not add the common antiperiodic temporal sign. A
    singular matrix raises NumPy's LinAlgError.
    """
    matrix = _dense_wilson_matrix(links, mass)
    inverse = np.linalg.inv(matrix)
    if not np.isfinite(inverse).all():
        raise FloatingPointError("Wilson inverse contains nonfinite values")
    return inverse


def _point_rows(points, *, lt, lz, ly, lx, name):
    raw = np.asarray(points)
    if raw.ndim != 3 or raw.shape[0] != lt or raw.shape[2] != 3 or raw.shape[1] < 1:
        raise ValueError(f"{name} must have shape (Lt,Np,3) with Np positive")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} coordinates must be integers")
    value = np.asarray(raw, dtype=np.int64)
    limits_xyz = np.asarray((lx, ly, lz), dtype=np.int64)
    if np.any(value < 0) or np.any(value >= limits_xyz):
        raise ValueError(f"{name} coordinates are outside the spatial lattice")
    for time in range(lt):
        if len(np.unique(value[time], axis=0)) != value.shape[1]:
            raise ValueError(f"{name} coordinates must be unique within every time slice")
    return value


def _mode_rows(mode_values, *, lt, lz, ly, lx, name):
    raw = np.asarray(mode_values)
    if (raw.ndim != 6 or raw.shape[0] != lt or raw.shape[1] < 1
            or raw.shape[2:] != (lz, ly, lx, 3)):
        raise ValueError(
            f"{name} must have shape (Lt,Ne,Lz,Ly,Lx,3) with Ne positive"
        )
    if raw.dtype.kind not in "fciu":
        raise TypeError(f"{name} must be a real or complex numeric array")
    value = np.asarray(raw, dtype=np.complex128)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _global_basis(mode_values, points, *, lt, lz, ly, lx):
    """Build block-diagonal Q_t with ordered V columns followed by ordered P columns."""
    mode_count = mode_values.shape[1]
    point_count = points.shape[1]
    spatial_width = mode_count + 3 * point_count
    local_rows = 12 * lz * ly * lx
    local_columns = 4 * spatial_width
    result = np.zeros((lt * local_rows, lt * local_columns), dtype=np.complex128)
    spatial_sites = lz * ly * lx
    for time in range(lt):
        spatial = np.zeros((3 * spatial_sites, spatial_width), dtype=np.complex128)
        spatial[:, :mode_count] = mode_values[time].reshape(mode_count, -1).T
        for point_index, xyz in enumerate(points[time]):
            x, y, z = (int(v) for v in xyz)
            site = (z * ly + y) * lx + x
            start = mode_count + 3 * point_index
            spatial[3 * site:3 * site + 3, start:start + 3] = _ID3

        local = np.zeros((local_rows, local_columns), dtype=np.complex128)
        for site in range(spatial_sites):
            for spin in range(4):
                row = (site * 4 + spin) * 3
                column = spin * spatial_width
                local[row:row + 3, column:column + spatial_width] = spatial[
                    3 * site:3 * site + 3
                ]
        row_start = time * local_rows
        column_start = time * local_columns
        result[row_start:row_start + local_rows,
               column_start:column_start + local_columns] = local
    return result


def _gamma4_full(site_count):
    return np.kron(np.eye(site_count, dtype=np.complex128),
                   np.kron(_GAMMA[3], _ID3))


def _current_kernel(links, momentum, insertion_time):
    """Full-field K_q for the temporal conserved Wilson current, including phase."""
    lt, lz, ly, lx = links.shape[:4]
    q = insertion_time
    next_time = (q + 1) % lt
    dimension = 12 * lt * lz * ly * lx
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    sizes_xyz = (lx, ly, lz)
    for z, y, x in np.ndindex((lz, ly, lx)):
        phase = np.exp(
            2j * np.pi * sum(
                momentum[axis] * coordinate / sizes_xyz[axis]
                for axis, coordinate in enumerate((x, y, z))
            )
        )
        q_site = int(np.ravel_multi_index((q, z, y, x), (lt, lz, ly, lx)))
        next_site = int(np.ravel_multi_index(
            (next_time, z, y, x), (lt, lz, ly, lx)
        ))
        row_q = slice(12 * q_site, 12 * q_site + 12)
        row_next = slice(12 * next_site, 12 * next_site + 12)
        link = links[q, z, y, x, 3]
        forward = -0.5 * np.kron(_ID4 - _GAMMA[3], link) * phase
        backward = +0.5 * np.kron(_ID4 + _GAMMA[3], link.conj().T) * phase
        result[row_q, 12 * next_site:12 * next_site + 12] += forward
        result[row_next, 12 * q_site:12 * q_site + 12] += backward
    return result


def projected_current_trace_grid(
    links,
    mode_values_A,
    mode_values_B,
    points_A,
    points_B,
    mass,
    momentum=(0, 0, 0),
):
    """Return a complex (Lt,Lt) projected two-current trace reference.

    The first grid axis is current insertion q on identity A; the second is
    insertion r on identity B. The mathematical definition is
    Tr[J_A,q G_AB J_B,r(src) G_BA], where J_A=Q_A^H K_q Q_A,
    J_B(src)=Q_B^H Gamma4 K_r^H Gamma4 Q_B,
    G_AB=Q_A^H D_W^-1 Q_B and G_BA=Q_B^H D_W^-1 Q_A.

    Inputs use full physical time coverage. points_X has shape (Lt,Np_X,3)
    and preserves the supplied ordered point labels. mode_values_X has shape
    (Lt,Ne_X,Lz,Ly,Lx,3). The routine performs no sampling correction; a
    partial point selection does not equal the population trace.
    """
    links = _positive_lattice(links)
    _mass_value(mass)
    lt, lz, ly, lx = links.shape[:4]
    modes_a = _mode_rows(mode_values_A, lt=lt, lz=lz, ly=ly, lx=lx,
                         name="mode_values_A")
    modes_b = _mode_rows(mode_values_B, lt=lt, lz=lz, ly=ly, lx=lx,
                         name="mode_values_B")
    points_a = _point_rows(points_A, lt=lt, lz=lz, ly=ly, lx=lx, name="points_A")
    points_b = _point_rows(points_B, lt=lt, lz=lz, ly=ly, lx=lx, name="points_B")
    try:
        momentum_values = tuple(momentum)
    except TypeError as exc:
        raise TypeError("momentum must be three integer spatial components") from exc
    if (len(momentum_values) != 3
            or any(isinstance(value, (bool, np.bool_))
                   or not isinstance(value, Integral) for value in momentum_values)):
        raise TypeError("momentum must be three integer spatial components")
    momentum_values = tuple(int(value) for value in momentum_values)

    inverse = dense_wilson_inverse(links, mass)
    basis_a = _global_basis(modes_a, points_a, lt=lt, lz=lz, ly=ly, lx=lx)
    basis_b = _global_basis(modes_b, points_b, lt=lt, lz=lz, ly=ly, lx=lx)
    gamma4 = _gamma4_full(lt * lz * ly * lx)
    g_ab = basis_a.conj().T @ inverse @ basis_b
    g_ba = basis_b.conj().T @ inverse @ basis_a

    grid = np.empty((lt, lt), dtype=np.complex128)
    for q in range(lt):
        kernel_a = _current_kernel(links, momentum_values, q)
        current_a = basis_a.conj().T @ kernel_a @ basis_a
        for r in range(lt):
            kernel_b_source = gamma4 @ _current_kernel(
                links, momentum_values, r
            ).conj().T @ gamma4
            current_b_source = basis_b.conj().T @ kernel_b_source @ basis_b
            grid[q, r] = np.trace(current_a @ g_ab @ current_b_source @ g_ba)
    if not np.isfinite(grid).all():
        raise FloatingPointError("projected current trace contains nonfinite values")
    return grid

if __name__ == "__main__":
    unittest.main()
