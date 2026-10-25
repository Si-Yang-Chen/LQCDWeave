"""Registered point-ended elemental production and independent site oracle."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from production.elemental.test_elemental import fixture, point_basis, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from fixtures.c_group_production import publish_elemental_products
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetWriter, inspect_dataset, register_npy,
)
from lqcdweave.production.elemental.planning import plan_indexed_point_elementals
from fixtures.elemental_script import produce_point_elemental

from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


def axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


class PointElementalProductionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.sizes, self.links, self.vectors, points = fixture()
        self.cfg = dict(ensemble="synthetic", configuration="cfg",
                        lattice=list(self.sizes))
        self.points = points.copy()
        self.labels = [8, 2, 5]
        self.gauge_id = self.save("gauge", self.make_spec("gauge", (
            ("time", range(3)), ("z", range(2)), ("y", range(2)), ("x", range(3)),
            ("direction", range(4)), ("color_row", range(3)),
            ("color_column", range(3))), dtype="complex128"), self.links)
        self.basis_id = self.save("basis", self.make_spec("explicit_basis", (
            ("time", range(3)), ("mode", [4, 1, 9]), ("z", range(2)),
            ("y", range(2)), ("x", range(3)), ("color", range(3))),
            dtype="complex128"), self.vectors)
        self.points_id = self.save_points("points", self.points, self.labels)
        self.paths = ((), (0,), (3,))
        self.momenta = ((1, -1, 0), (0, 0, 0))

    def make_spec(self, kind, axes, *, dtype, conventions=None, definition=None):
        return DatasetSpec(dict(schema=1, kind=kind, configuration=self.cfg,
            axes=[axis(name, values) for name, values in axes], dtype=dtype, bases={},
            parents=[], definition={} if definition is None else definition,
            conventions={} if conventions is None else conventions))

    def save(self, name, spec, values):
        source = self.root / f"{name}.npy"
        np.save(source, values)
        identity = register_npy(self.root / name, source, spec,
                                provenance=dict(test="point elemental production"))
        self.index.register(self.root / name)
        return identity

    def save_points(self, name, values, labels):
        spec = self.make_spec("spatial_points", (
            ("time", range(self.sizes[3])), ("point", labels),
            ("xyz", ("x", "y", "z"))), dtype="int64",
            conventions=dict(representation="coordinate_delta", normalization="unit_delta"))
        return self.save(name, spec, np.asarray(values, dtype=np.int64))

    def plan(self, *, sectors=("v2p", "p2v", "p2p"), basis_id=None,
             points_id=None, **options):
        if basis_id is None and set(sectors) & {"v2p", "p2v"}:
            basis_id = self.basis_id
        return plan_indexed_point_elementals(self.index, self.gauge_id,
            basis_id,
            self.points_id if points_id is None else points_id,
            paths=self.paths, momentum_list=self.momenta, sectors=sectors, **options)

    def run_products(self, name="out", *, sectors=("v2p", "p2v", "p2p"), basis_id=None,
                     points_id=None, **options):
        if basis_id is None and set(sectors) & {"v2p", "p2v"}:
            basis_id = self.basis_id
        identities = publish_elemental_products(
            self.index, self.root, self.gauge_id, basis_id,
            self.points_id if points_id is None else points_id,
            output_name=name, paths=self.paths, momentum_list=self.momenta,
            sectors=sectors, **options)
        return {"datasets": identities}

    def assert_independent_reference(self, result, *, sectors, modes=3, points=3,
                                     phase="midpoint", atol=2e-12):
        for product in sectors:
            with self.index.open(result["datasets"][product]) as output:
                self.assertEqual(output.spec.coordinates("time"), [0, 1, 2])
                expected_kind = {"v2p": "elemental_v2p", "p2v": "elemental_p2v",
                                 "p2p": "elemental_p2p_row_sparse"}[product]
                self.assertEqual(output.spec.document["kind"], expected_kind)
                self.assertEqual(output.spec.document["conventions"]["phase"], phase)
                labels_axis = {"v2p": "point_right", "p2v": "point_left",
                               "p2p": "point_left"}[product]
                self.assertEqual(output.spec.coordinates(labels_axis), self.labels[:points])
                for time in range(self.sizes[3]):
                    vectors = self.vectors[time, :modes].reshape(modes, -1).T
                    actual_points = self.points[time, :points]
                    points_dense = point_basis(actual_points, self.sizes)
                    for path_index, path in enumerate(self.paths):
                        displacement = np.zeros(3, dtype=int)
                        steps_axes = (0, 1, 2, 0, 1, 2)
                        signs = (1, 1, 1, -1, -1, -1)
                        for step in path:
                            displacement[steps_axes[step]] += signs[step]
                        for momentum_index, momentum in enumerate(self.momenta):
                            matrix = reference_matrix(self.links, path, time, momentum,
                                                      phase=phase)
                            if product == "v2p":
                                expected = vectors.conj().T @ matrix @ points_dense
                                actual = output.read(time=time)[path_index, momentum_index]
                                np.testing.assert_allclose(actual.reshape(expected.shape), expected,
                                                           atol=atol, rtol=atol)
                            elif product == "p2v":
                                expected = points_dense.conj().T @ matrix @ vectors
                                actual = output.read(time=time)[path_index, momentum_index]
                                np.testing.assert_allclose(actual.reshape(expected.shape), expected,
                                                           atol=atol, rtol=atol)
                            else:
                                expected = points_dense.conj().T @ matrix @ points_dense
                                actual = output.read(time=time)[path_index, momentum_index]
                                lookup = {tuple(xyz): index for index, xyz in enumerate(actual_points)}
                                for left, xyz in enumerate(actual_points):
                                    destination = tuple((xyz + displacement) % np.asarray(self.sizes[:3]))
                                    row = actual[left]
                                    if destination in lookup:
                                        right = lookup[destination]
                                        expected_row = expected[left*3:(left+1)*3,
                                                                right*3:(right+1)*3]
                                    else:
                                        expected_row = np.zeros((3, 3), dtype=np.complex128)
                                    np.testing.assert_allclose(row, expected_row,
                                                               atol=atol, rtol=atol)

    def test_three_products_match_independent_full_site_projection_and_contract(self):
        for phase in ("anchor", "midpoint", "none"):
            with self.subTest(phase=phase):
                with patch.object(ElementalGenerator, "calc_path_blocks",
                                  side_effect=AssertionError("planning computed")):
                    preview = self.plan(usedNe=2, usedNp=2,
                                        phase_convention=phase)
                self.assertEqual({preview["stages"][p]["action"]
                                  for p in ("v2p", "p2v", "p2p")}, {"compute"})
                result = self.run_products(f"all-{phase}", usedNe=2, usedNp=2,
                                  phase_convention=phase)
                self.assertEqual(set(result["datasets"]), {"v2p", "p2v", "p2p"})
                self.assert_independent_reference(result, sectors=("v2p", "p2v", "p2p"),
                                                  modes=2, points=2, phase=phase)
                with self.index.open(result["datasets"]["v2p"]) as v2p, \
                     self.index.open(result["datasets"]["p2v"]) as p2v, \
                     self.index.open(result["datasets"]["p2p"]) as p2p:
                    self.assertEqual(v2p.spec.axis_names,
                        ("time", "component", "momentum", "mode_left", "point_right", "color_right"))
                    self.assertEqual(p2v.spec.axis_names,
                        ("time", "component", "momentum", "point_left", "color_left", "mode_right"))
                    self.assertEqual(p2p.spec.axis_names,
                        ("time", "component", "momentum", "point_left", "color_left", "color_right"))
                    self.assertEqual(p2p.spec.document["definition"]["point_right_id"], self.points_id)
                    self.assertEqual(p2p.spec.document["bases"]["point_left"],
                                     dict(data_id=self.points_id, indices=self.labels[:2]))
                    self.assertEqual(p2p.spec.document["parents"], [self.gauge_id, self.points_id])

    def test_point_only_p2p_has_no_basis_dependency_or_ne(self):
        preview = self.plan(sectors=("p2p",), basis_id=None, usedNp=2)
        self.assertEqual(set(preview["stages"]), {"gauge", "points", "p2p"})
        result = self.run_products("point-only", sectors=("p2p",), basis_id=None, usedNp=2)
        self.assert_independent_reference(result, sectors=("p2p",), points=2, phase="anchor")
        with self.assertRaisesRegex(ValueError, "rejects usedNe"):
            self.plan(sectors=("p2p",), basis_id=None, usedNp=2, usedNe=1)
        with self.assertRaisesRegex(ValueError, "requires basis_id=None"):
            self.plan(sectors=("p2p",), basis_id=self.basis_id, usedNp=2)

    def test_changed_point_parent_invalidates_all_products(self):
        changed = self.points.copy()
        changed[:, :, 0] = (changed[:, :, 0] + 1) % self.sizes[0]
        changed_id = self.save_points("changed-points", changed, self.labels)
        with patch.object(ElementalGenerator, "load", side_effect=AssertionError("planning computed")):
            preview = self.plan(points_id=changed_id)
        for product in ("v2p", "p2v", "p2p"):
            self.assertEqual(preview["stages"][product]["action"], "compute")
        result = self.run_products("changed", points_id=changed_id)
        for product in ("v2p", "p2v", "p2p"):
            self.assertNotEqual(result["datasets"][product],
                                self.run_products(f"original-{product}", sectors=(product,))["datasets"][product])

    def test_actual_parent_mismatch_is_refused_before_kernel(self):
        preview = self.plan(sectors=("v2p",), usedNe=2, usedNp=2)
        with self.index.open(self.gauge_id) as gauge, self.index.open(self.basis_id) as basis, \
             self.index.open(self.points_id) as points:
            document = preview["stages"]["v2p"]["spec"].document
            document["definition"]["gauge_id"] = "another-actual-gauge"
            document["parents"][0] = "another-actual-gauge"
            output = DatasetSpec(document)
            with DatasetWriter(self.root / "wrong-parent", output,
                                provenance=dict(test="wrong parent")) as writer, \
                 patch.object(ElementalGenerator, "load",
                              side_effect=AssertionError("computed")):
                with self.assertRaisesRegex(ValueError, "actual input IDs"):
                    produce_point_elemental(gauge, basis, points, writer)
                self.assertEqual(writer.missing, (0, 1, 2))
        output = self.root / "wrong-parent"
        self.assertNotEqual(inspect_dataset(output, cfg="cfg")["state"], "complete")
        self.assertFalse((output / "cfg.npy").exists())
        self.assertFalse((output / "cfg.h5").exists())


if __name__ == "__main__":
    unittest.main()
