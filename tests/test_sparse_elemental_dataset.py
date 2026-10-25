"""Sparse producer -> shared DatasetWriter -> reload, with full physical oracles."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import TEMPORAL_POINT_PATH_SCHEMA
from lqcdweave.contracts.time_selection import (
    TimeSelectionRecord, dataset_time_selection, bind_time_selections,
)
from lqcdweave.production.elemental.api import BaryonElementalKernel
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter, register_npy
from lqcdweave.production.elemental.baryon_planning import plan_indexed_baryon_elementals
from lqcdweave.production.elemental.planning import plan_indexed_directed_current, plan_indexed_point_elementals
from fixtures.elemental_script import produce_current, produce_point_elemental
from production.elemental.test_elemental import reference_matrix, point_basis
from test_baryon_elemental import determinant_reference
from tests.fixtures.laplace_reference import random_gauge


class SparseElementalDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.size = (3, 2, 2, 6)
        self.cfg = dict(ensemble="synthetic", configuration="sparse", lattice=list(self.size))
        self.gauge = random_gauge(self.size, seed=164)
        rng = np.random.default_rng(163)
        self.vectors = rng.normal(size=(6, 2, 2, 2, 3, 3)) + 1j*rng.normal(size=(6, 2, 2, 2, 3, 3))
        self.points = np.array([[(t % 3, 0, 0), ((t+1) % 3, 0, 0), (0, 1, 1)] for t in range(6)])
        self.momenta = ((0, 0, 0), (1, -1, 0))
        self.gauge_id = self.publish("gauge", "gauge", self.gauge,
            [("time", range(6)), ("z", range(2)), ("y", range(2)), ("x", range(3)),
             ("direction", range(4)), ("color_row", range(3)), ("color_column", range(3))])

    def publish(self, name, kind, values, axes, *, selection=None, conventions=None):
        document = dict(schema=1, kind=kind, configuration=self.cfg,
            axes=[dict(name=axis, coordinates=list(coords)) for axis, coords in axes],
            dtype=str(values.dtype), bases={}, definition={}, parents=[],
            conventions={} if conventions is None else conventions)
        if selection is not None:
            document = bind_time_selections(document, {"time": selection})
        source = self.root / (name + ".npy")
        np.save(source, values)
        identity = register_npy(self.root / name, source, DatasetSpec(document),
                                provenance={"test": "bounded synthetic sparse production"})
        self.index.register(self.root / name)
        return identity

    def publish_basis(self, name, selection, values=None):
        values = self.vectors if values is None else values
        return self.publish(name, "explicit_basis", values[list(selection.times)],
            [("time", selection.times), ("mode", range(values.shape[1])),
             ("z", range(2)), ("y", range(2)), ("x", range(3)), ("color", range(3))],
            selection=selection)

    def publish_points(self, name, selection):
        return self.publish(name, "spatial_points", self.points[list(selection.times)].astype(np.int64),
            [("time", selection.times), ("point", (4, 1, 9)), ("xyz", ("x", "y", "z"))],
            selection=selection, conventions=dict(representation="coordinate_delta", normalization="unit_delta"))

    def test_temporal_point_products_recover_compact_rows_and_reject_missing_endpoints(self):
        anchors = TimeSelectionRecord(6, (5, 1))
        basis_id = self.publish_basis("basis", TimeSelectionRecord(6, (2, 5, 0, 1)))
        points_id = self.publish_points("points", TimeSelectionRecord(6, (1, 0, 5, 2)))
        paths = ((6, 0),)
        plan = plan_indexed_point_elementals(self.index, self.gauge_id, basis_id, points_id,
            paths=paths, momentum_list=self.momenta, selection=anchors,
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA, phase_convention="midpoint")
        # Build complete physical-time oracles first, then select output anchors.
        references = {sector: [] for sector in ("v2p", "p2v", "p2p")}
        for t in range(6):
            end = (t+1) % 6
            left = {"v": self.vectors[t].reshape(2, -1).T,
                    "p": point_basis(self.points[t], self.size)}
            right = {"v": self.vectors[end].reshape(2, -1).T,
                     "p": point_basis(self.points[end], self.size)}
            for sector in references:
                rows = []
                for momentum in self.momenta:
                    matrix = reference_matrix(self.gauge, paths[0], t, momentum)
                    value = left[sector[0]].conj().T @ matrix @ right[sector[-1]]
                    if sector == "v2p": value = value.reshape(2, 3, 3)
                    elif sector == "p2v": value = value.reshape(3, 3, 2)
                    else: value = value.reshape(3, 3, 3, 3).sum(axis=2)
                    rows.append(value)
                references[sector].append(np.asarray([rows]))
        with self.index.open(self.gauge_id) as gauge, self.index.open(basis_id) as basis, self.index.open(points_id) as points:
            for sector in references:
                spec = plan["stages"][sector]["spec"]
                self.assertEqual(spec.coordinates("time"), [0, 1])
                self.assertEqual(dataset_time_selection(spec), anchors)
                destination = self.root / ("output-" + sector)
                # A completed first slot survives interruption and is not overwritten.
                with DatasetWriter(destination, spec, provenance={"test": True}) as writer:
                    writer.write(0, references[sector][5])
                with DatasetWriter(destination, spec, provenance={"test": True}, resume=True) as writer:
                    self.assertEqual(writer.missing, (1,))
                    produce_point_elemental(gauge, basis if sector != "p2p" else None, points, writer)
                    identity = writer.publish()
                self.index.register(destination)
                with self.index.open(identity) as reader:
                    self.assertEqual(reader.spec.shape[0], anchors.Nt)
                    self.assertEqual(dataset_time_selection(reader.spec), anchors)
                    for slot, physical in enumerate(anchors.times):
                        np.testing.assert_allclose(reader.read(time=slot), references[sector][physical], atol=3e-12)
        missing = self.publish_basis("missing-basis", TimeSelectionRecord(6, (5, 0, 1)))
        with self.assertRaises(KeyError):
            plan_indexed_point_elementals(self.index, self.gauge_id, missing, points_id,
                paths=paths, selection=anchors, temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA)

    def test_distinct_left_anchor_and_right_endpoint_point_maps(self):
        anchors = TimeSelectionRecord(6, (5, 1))
        basis_id = self.publish_basis("asym-basis", TimeSelectionRecord(6, (2, 5, 0, 1)))
        left_id = self.publish_points("left-points", anchors)
        right_id = self.publish_points("right-points", TimeSelectionRecord(6, (2, 0)))
        plan = plan_indexed_point_elementals(self.index, self.gauge_id, basis_id, left_id,
            paths=((6, 0),), momentum_list=self.momenta, selection=anchors,
            points_right_id=right_id, point_right_labels=(9, 4),
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA, phase_convention="midpoint")
        with ExitStack() as stack:
            gauge = stack.enter_context(self.index.open(self.gauge_id))
            basis = stack.enter_context(self.index.open(basis_id))
            left = stack.enter_context(self.index.open(left_id))
            right = stack.enter_context(self.index.open(right_id))
            for sector in ("v2p", "p2v", "p2p"):
                spec = plan["stages"][sector]["spec"]
                destination = self.root / ("asymmetric-" + sector)
                with DatasetWriter(destination, spec, provenance={"test": True}) as writer:
                    produce_point_elemental(gauge, basis if sector != "p2p" else None,
                                               left, writer, points_right=right)
                    identity = writer.publish()
                self.index.register(destination)
                with self.index.open(identity) as reader:
                    self.assertEqual(dataset_time_selection(reader.spec), anchors)
                    for slot, time in enumerate(anchors.times):
                        endpoint = (time+1) % 6
                        left_basis = (self.vectors[time].reshape(2, -1).T if sector == "v2p" else
                                      point_basis(self.points[time], self.size))
                        right_basis = (self.vectors[endpoint].reshape(2, -1).T if sector == "p2v" else
                                       point_basis(self.points[endpoint, [2, 0]], self.size))
                        for mi, momentum in enumerate(self.momenta):
                            expected = left_basis.conj().T @ reference_matrix(
                                self.gauge, (6, 0), time, momentum) @ right_basis
                            if sector == "p2p":
                                expected = expected.reshape(3, 3, 2, 3).sum(axis=2)
                            actual = reader.read(time=slot)[0, mi]
                            np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=3e-12)

    def test_directed_current_writer_uses_physical_neighbours_and_compact_slots(self):
        anchors = TimeSelectionRecord(6, (5,))
        basis_id = self.publish_basis("current-basis", TimeSelectionRecord(6, (0, 5, 4)))
        plan = plan_indexed_directed_current(self.index, self.gauge_id, basis_id,
                                             momentum_list=self.momenta, selection=anchors)
        spec = plan["stages"]["current"]["spec"]
        with self.index.open(self.gauge_id) as gauge, self.index.open(basis_id) as basis:
            with DatasetWriter(self.root/"current", spec, provenance={"test": True}) as writer:
                produce_current(gauge, basis, writer)
                identity = writer.publish()
        self.index.register(self.root/"current")
        with self.index.open(identity) as reader:
            self.assertEqual(dataset_time_selection(reader.spec), anchors)
            row = reader.read(time=0)
            for direction in range(8):
                end = (5 + (1 if direction == 6 else -1 if direction == 7 else 0)) % 6
                for mi, momentum in enumerate(self.momenta):
                    expected = self.vectors[5].reshape(2, -1).conj() @ reference_matrix(
                        self.gauge, (direction,), 5, momentum) @ self.vectors[end].reshape(2, -1).T
                    np.testing.assert_allclose(row[direction, mi], expected, atol=3e-12)

    def test_baryon_planner_preserves_three_independent_maps_and_persistent_slots(self):
        anchors = TimeSelectionRecord(6, (5, 1))
        full = (self.vectors, self.vectors * (0.7 + 0.2j), self.vectors + 0.3j)
        selections = (anchors, TimeSelectionRecord(6, (1, 5)), TimeSelectionRecord(6, (3, 1, 5)))
        identities = tuple(self.publish_basis("leg"+str(i), selection, values)
                           for i, (selection, values) in enumerate(zip(selections, full)))
        paths = ((0,), (), (1,))
        plan = plan_indexed_baryon_elementals(self.index, self.gauge_id, identities,
            components=(paths,), momentum_list=self.momenta, selection=anchors)
        self.assertEqual(plan["times"], (0, 1))
        spec = plan["dataset_spec"]
        with ExitStack() as stack:
            gauge = stack.enter_context(self.index.open(self.gauge_id))
            bases = [stack.enter_context(self.index.open(identity)) for identity in identities]
            generator = BaryonElementalKernel(self.size, DatasetFieldHandle(gauge),
                tuple(DatasetFieldHandle(basis) for basis in bases), paths=(paths,),
                momentum_list=self.momenta, selection=anchors, stream_eigenvectors=True).load("sparse")
            with DatasetWriter(self.root/"baryon", spec, provenance={"test": True}) as writer:
                for slot in writer.missing:
                    writer.write(slot, generator.calc_path_blocks(anchors.physical_time(slot))["vvv"])
                identity = writer.publish()
        self.index.register(self.root/"baryon")
        with self.index.open(identity) as reader:
            self.assertEqual(dataset_time_selection(reader.spec), anchors)
            for slot, t in enumerate(anchors.times):
                for mi, momentum in enumerate(self.momenta):
                    expected = determinant_reference(self.gauge, full, paths, t, momentum)
                    np.testing.assert_allclose(reader.read(time=slot)[0, mi], expected, atol=3e-12)
