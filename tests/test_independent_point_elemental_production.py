"""Independent left/right point-pool production and recovery regression."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from production.elemental.test_elemental import fixture, point_basis, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from fixtures.c_group_production import publish_elemental_products
from lqcdweave.storage.dataset import DatasetIndex, register_npy
from lqcdweave.production.elemental.planning import (
    plan_indexed_point_elementals,
)


def axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


class IndependentPointElementalProductionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.sizes, self.links, self.vectors, self.left_values = fixture()
        self.left_values = self.left_values.copy()
        self.left_labels = [8, 2, 5]
        self.cfg = dict(ensemble="synthetic", configuration="cfg",
                        lattice=list(self.sizes))
        self.gauge_id = self.save("gauge", self.spec("gauge", (
            ("time", range(3)), ("z", range(2)), ("y", range(2)),
            ("x", range(3)), ("direction", range(4)),
            ("color_row", range(3)), ("color_column", range(3))),
            "complex128"), self.links)
        self.basis_id = self.save("basis", self.spec("explicit_basis", (
            ("time", range(3)), ("mode", [4, 1, 9]), ("z", range(2)),
            ("y", range(2)), ("x", range(3)), ("color", range(3))),
            "complex128"), self.vectors)
        self.left_id = self.save_points("points-left", self.left_values,
                                        self.left_labels)
        changed_left = self.left_values.copy()
        changed_left[:, :, 0] = (changed_left[:, :, 0] + 1) % self.sizes[0]
        self.changed_left_id = self.save_points("points-left-changed", changed_left,
                                                self.left_labels)
        self.right_labels = [7, 2, 9, 4]
        self.right_values = np.asarray([
            [(0, 0, 0), (2, 0, 1), (1, 1, 1), (1, 0, 0)],
            [(2, 0, 0), (1, 1, 0), (0, 1, 1), (0, 0, 1)],
            [(1, 0, 1), (0, 1, 0), (2, 1, 1), (2, 0, 0)],
        ], dtype=np.int64)
        self.right_id = self.save_points("points-right", self.right_values,
                                         self.right_labels)
        changed = self.right_values.copy()
        changed[:, :, 0] = (changed[:, :, 0] + 1) % self.sizes[0]
        self.changed_right_id = self.save_points("points-right-changed", changed,
                                                 self.right_labels)
        self.paths = ((), (0,), (3,))
        self.momenta = ((1, -1, 0), (0, 0, 0))

    def spec(self, kind, axes, dtype, conventions=None):
        return DatasetSpec(dict(schema=1, kind=kind, configuration=self.cfg,
            axes=[axis(name, coordinates) for name, coordinates in axes],
            dtype=dtype, bases={}, parents=[], definition={},
            conventions={} if conventions is None else conventions))

    def save(self, name, spec, values):
        source = self.root / f"{name}.npy"
        np.save(source, values)
        identity = register_npy(self.root / name, source, spec,
                                provenance=dict(test="independent point elemental"))
        self.index.register(self.root / name)
        return identity

    def save_points(self, name, values, labels):
        spec = self.spec("spatial_points", (
            ("time", range(self.sizes[3])), ("point", labels),
            ("xyz", ("x", "y", "z"))), "int64",
            conventions=dict(representation="coordinate_delta",
                             normalization="unit_delta"))
        return self.save(name, spec, np.asarray(values, dtype=np.int64))

    def plan(self, *, sectors=("v2p", "p2v", "p2p"), left_id=None,
             right_id=None, **options):
        basis_id = self.basis_id if set(sectors) & {"v2p", "p2v"} else None
        return plan_indexed_point_elementals(self.index, self.gauge_id,
            basis_id, self.left_id if left_id is None else left_id,
            paths=self.paths, momentum_list=self.momenta, sectors=sectors,
            points_right_id=right_id, **options)

    def produce(self, name, *, sectors=("v2p", "p2v", "p2p"), left_id=None,
                right_id=None, **options):
        basis_id = self.basis_id if set(sectors) & {"v2p", "p2v"} else None
        identities = publish_elemental_products(
            self.index, self.root, self.gauge_id, basis_id,
            self.left_id if left_id is None else left_id,
            output_name=name, paths=self.paths, momentum_list=self.momenta,
            sectors=sectors, points_right_id=right_id, **options)
        return {"datasets": identities}

    def _assert_site_link_reference(self, result, *, sectors, left_labels,
                                    right_labels, modes=2, phase="midpoint"):
        left_position = {label: i for i, label in enumerate(self.left_labels)}
        right_position = {label: i for i, label in enumerate(self.right_labels)}
        left_input_positions = [left_position[label] for label in left_labels]
        right_input_positions = [right_position[label] for label in right_labels]
        for product in sectors:
            with self.index.open(result["datasets"][product]) as output:
                self.assertEqual(output.spec.coordinates("time"), [0, 1, 2])
                if product == "v2p":
                    self.assertEqual(output.spec.coordinates("point_right"), right_labels)
                elif product == "p2v":
                    self.assertEqual(output.spec.coordinates("point_left"), left_labels)
                else:
                    self.assertEqual(output.spec.coordinates("point_left"), left_labels)
                    self.assertEqual(output.spec.document["definition"]["point_right_labels"],
                                     right_labels)
                for time in range(self.sizes[3]):
                    matrix_modes = self.vectors[time, :modes].reshape(modes, -1).T
                    left_points = self.left_values[time, left_input_positions]
                    right_points = self.right_values[time, right_input_positions]
                    left_basis = point_basis(left_points, self.sizes)
                    right_basis = point_basis(right_points, self.sizes)
                    for path_index, path in enumerate(self.paths):
                        displacement = np.zeros(3, dtype=int)
                        step_axes, step_signs = (0, 1, 2, 0, 1, 2), (1, 1, 1, -1, -1, -1)
                        for step in path:
                            displacement[step_axes[step]] += step_signs[step]
                        for momentum_index, momentum in enumerate(self.momenta):
                            matrix = reference_matrix(self.links, path, time,
                                momentum, phase=phase)
                            actual = output.read(time=time)[path_index, momentum_index]
                            if product == "v2p":
                                expected = matrix_modes.conj().T @ matrix @ right_basis
                                np.testing.assert_allclose(actual.reshape(expected.shape), expected,
                                                           atol=2e-12, rtol=2e-12)
                            elif product == "p2v":
                                expected = left_basis.conj().T @ matrix @ matrix_modes
                                np.testing.assert_allclose(actual.reshape(expected.shape), expected,
                                                           atol=2e-12, rtol=2e-12)
                            else:
                                expected = left_basis.conj().T @ matrix @ right_basis
                                right_lookup = {tuple(point): index
                                                for index, point in enumerate(right_points)}
                                spatial_size = np.asarray(self.sizes[:3])
                                for left_index, coordinate in enumerate(left_points):
                                    destination = tuple((coordinate + displacement) % spatial_size)
                                    expected_row = np.zeros((3, 3), dtype=np.complex128)
                                    if destination in right_lookup:
                                        right_index = right_lookup[destination]
                                        expected_row = expected[
                                            left_index * 3:(left_index + 1) * 3,
                                            right_index * 3:(right_index + 1) * 3]
                                    np.testing.assert_allclose(actual[left_index], expected_row,
                                                               atol=2e-12, rtol=2e-12)

    def test_different_rectangular_pools_and_reordered_right_labels_match_site_link_oracle(self):
        preview = self.plan(usedNe=2, usedNp=2, usedNp_right=3,
                            right_id=self.right_id, phase_convention="midpoint")
        self.assertEqual(preview["stages"]["points"]["action"], "reuse")
        self.assertEqual(preview["stages"]["points_right"]["action"], "reuse")
        result = self.produce("rectangular", usedNe=2, usedNp=2, usedNp_right=3,
                              right_id=self.right_id, phase_convention="midpoint")
        self._assert_site_link_reference(result,
            sectors=("v2p", "p2v", "p2p"), left_labels=self.left_labels[:2],
            right_labels=self.right_labels[:3], modes=2)
        with self.index.open(result["datasets"]["p2p"]) as p2p:
            document = p2p.spec.document
            self.assertEqual(document["definition"]["point_right_id"], self.right_id)
            self.assertEqual(document["definition"]["point_right_selection"],
                             "explicit_ordered_labels")
            self.assertEqual(document["parents"],
                             [self.gauge_id, self.left_id, self.right_id])

        reordered = [4, 7]
        changed_plan = self.plan(sectors=("p2p",), usedNp=2,
            right_id=self.right_id, point_right_labels=reordered)
        self.assertEqual(changed_plan["stages"]["p2p"]["action"], "compute")
        reordered_result = self.produce("reordered", sectors=("p2p",), usedNp=2,
            right_id=self.right_id, point_right_labels=reordered)
        self._assert_site_link_reference(reordered_result, sectors=("p2p",),
            left_labels=self.left_labels[:2], right_labels=reordered, modes=2,
            phase="anchor")

    def test_same_pool_identity_stays_legacy_and_right_only_change_invalidates_right_products(self):
        legacy = self.plan(sectors=("p2p",), usedNp=2)
        explicit_same = self.plan(sectors=("p2p",), usedNp=2,
                                  right_id=self.left_id, usedNp_right=2)
        self.assertEqual(legacy["stages"]["p2p"]["spec"].document,
                         explicit_same["stages"]["p2p"]["spec"].document)
        self.assertEqual(legacy["stages"]["p2p"]["spec"].document["definition"],
            dict(gauge_id=self.gauge_id, path_boundary="periodic",
                 point_right_id=self.left_id,
                 point_right_selection="same_ordered_prefix_as_point_left"))

        self.produce("all-right-a", usedNe=2, usedNp=2, usedNp_right=3,
                     right_id=self.right_id)
        changed_right = self.plan(usedNe=2, usedNp=2, usedNp_right=3,
                                  right_id=self.changed_right_id)
        self.assertEqual(changed_right["stages"]["v2p"]["action"], "compute")
        self.assertEqual(changed_right["stages"]["p2p"]["action"], "compute")
        self.assertEqual(changed_right["stages"]["p2v"]["action"], "reuse")
        changed_left = self.plan(usedNe=2, usedNp=2, usedNp_right=3,
                                 left_id=self.changed_left_id, right_id=self.right_id)
        self.assertEqual(changed_left["stages"]["v2p"]["action"], "reuse")
        self.assertEqual(changed_left["stages"]["p2v"]["action"], "compute")
        self.assertEqual(changed_left["stages"]["p2p"]["action"], "compute")


if __name__ == "__main__":
    unittest.main()
