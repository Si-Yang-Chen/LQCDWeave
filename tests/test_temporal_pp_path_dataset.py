from contextlib import ExitStack
import unittest

import numpy as np

import test_directed_current_point_diagram as fixture_module
from lqcdweave.adapters.meson_dataset import DatasetElemental
from lqcdweave.contracts.elemental import (
    TEMPORAL_POINT_PATH_SCHEMA, ElementalComponent, GaugePath,
)
from fixtures.c_group_production import publish_elemental_products


PATHS = tuple(GaugePath(path) for path in (
    (6, 0), (0, 6), (6, 7), (7, 6),
))


class TemporalPointPointDatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.RegisteredTemporalPointCurrentTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture

    def _produce(self, *, output="temporal-pp", paths=PATHS, boundary="periodic"):
        identities = publish_elemental_products(
            self.f.index, self.f.root,
            self.f.gauge_id, points_id=self.f.left_id,
            output_name=output, paths=paths, momentum_list=self.f.momentum_list,
            sectors=("p2p",), usedNp=2, points_right_id=self.f.right_id,
            point_right_labels=self.f.right_labels, phase_convention="midpoint",
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA, path_boundary=boundary,
        )
        return {"datasets": identities}

    def _transport(self, anchor, path, start, *, boundary="periodic"):
        lo, hi = path.time_bounds
        if boundary == "open" and (
                anchor + lo < 0 or anchor + hi >= self.f.lattice[3]):
            return None
        offset = [0, 0, 0, 0]
        result = np.eye(3, dtype=np.complex128)
        for step in path.steps:
            axis, sign = GaugePath.AXES[step], GaugePath.SIGNS[step]
            if sign < 0:
                offset[axis] -= 1
            position = tuple((int(start[i]) + offset[i]) % self.f.lattice[i]
                             for i in range(3))
            time = (anchor + offset[3]) % self.f.lattice[3]
            x, y, z = position
            link = self.f.links[time, z, y, x, axis]
            if sign < 0:
                link = link.conj().T
            result = result @ link
            if sign > 0:
                offset[axis] += 1
        return result

    def _oracle(self, path, anchor, momentum, *, boundary="periodic"):
        left_coordinates = self.f.left_coordinates[anchor][:2]
        endpoint_unwrapped = anchor + path.displacement[3]
        endpoint = endpoint_unwrapped % self.f.lattice[3]
        right_coordinates = self.f.right_coordinates[endpoint]
        if boundary == "open" and (
                anchor + path.time_bounds[0] < 0
                or anchor + path.time_bounds[1] >= self.f.lattice[3]):
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 3, 3), dtype=np.complex128)
        lookup = {tuple(map(int, point)): index
                  for index, point in enumerate(right_coordinates)}
        rows, columns, values = [], [], []
        spatial = np.asarray(self.f.lattice[:3], dtype=np.int64)
        dx, dy, dz, _dt = path.displacement
        for left_index, coordinate in enumerate(left_coordinates):
            target = tuple(((coordinate + (dx, dy, dz)) % spatial).tolist())
            right_index = lookup.get(target)
            if right_index is None:
                continue
            x, y, z = map(int, coordinate)
            transport = self._transport(anchor, path, coordinate, boundary=boundary)
            if transport is None:
                continue
            phase = np.exp(2j * np.pi * sum(
                p * (c + d / 2) / n
                for p, c, d, n in zip(momentum, (x, y, z), (dx, dy, dz), spatial, strict=True)
            ))
            rows.append(left_index)
            columns.append(right_index)
            values.append(phase * transport)
        return (np.asarray(list(zip(rows, columns)), dtype=np.int64).reshape(-1, 2),
                np.asarray(values, dtype=np.complex128).reshape(-1, 3, 3))

    def test_registered_temporal_pp_reopens_and_matches_full_time_site_link_oracle(self):
        produced = self._produce()
        identity = produced["datasets"]["p2p"]
        with ExitStack() as stack:
            reader = stack.enter_context(self.f.index.open(identity))
            left = stack.enter_context(self.f.index.open(self.f.left_id))
            right = stack.enter_context(self.f.index.open(self.f.right_id))
            elemental = DatasetElemental(
                {"p2p": reader}, point_dataset=left, point_right_dataset=right,
            )
            spec = reader.spec
            definition = spec.document["definition"]
            self.assertEqual(spec.document["kind"], "elemental_p2p_row_sparse")
            self.assertEqual(spec.document["dtype"], "complex128")
            self.assertEqual(spec.document["parents"],
                             [self.f.gauge_id, self.f.left_id, self.f.right_id])
            self.assertEqual(spec.coordinates("time"), [0, 1, 2])
            self.assertEqual(spec.coordinates("point_left"), list(self.f.left_labels))
            self.assertEqual(spec.document["definition"]["point_right_id"], self.f.right_id)
            self.assertEqual(spec.document["definition"]["point_right_labels"],
                             list(self.f.right_labels))
            self.assertEqual(definition["temporal_path_schema"], TEMPORAL_POINT_PATH_SCHEMA)
            self.assertNotIn("directed_current_schema", definition)
            self.assertEqual(definition["path_boundary"], "periodic")
            for component_index, path in enumerate(PATHS):
                for time in range(self.f.lattice[3]):
                    for momentum_index, momentum in enumerate(self.f.momentum_list):
                        actual = elemental.block(
                            ElementalComponent.path(path), momentum_index, time, "p2p",
                        )
                        indices, values = self._oracle(path, time, momentum)
                        np.testing.assert_array_equal(actual["indices"], indices)
                        np.testing.assert_allclose(actual["values"], values,
                                                   rtol=4e-13, atol=4e-13)

    def test_open_net_zero_excursions_are_empty_in_saved_rows_and_adapter(self):
        paths = (GaugePath((6, 7)), GaugePath((7, 6)))
        produced = self._produce(output="open-temporal-pp", paths=paths, boundary="open")
        with ExitStack() as stack:
            reader = stack.enter_context(self.f.index.open(produced["datasets"]["p2p"]))
            left = stack.enter_context(self.f.index.open(self.f.left_id))
            right = stack.enter_context(self.f.index.open(self.f.right_id))
            elemental = DatasetElemental(
                {"p2p": reader}, point_dataset=left, point_right_dataset=right,
            )
            for path, time in ((paths[0], 2), (paths[1], 0)):
                self.assertEqual(path.displacement[3], 0)
                indices, values = self._oracle(path, time, self.f.momentum_list[0], boundary="open")
                self.assertEqual(indices.shape, (0, 2))
                self.assertEqual(values.shape, (0, 3, 3))
                actual = elemental.block(ElementalComponent.path(path), 0, time, "p2p")
                self.assertEqual(actual["indices"].shape, (0, 2))
                self.assertEqual(actual["values"].shape, (0, 3, 3))



if __name__ == "__main__":
    unittest.main()
