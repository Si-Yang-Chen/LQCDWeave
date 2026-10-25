"""Automatic point-diagonal declarations from registered production recipes."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from production.elemental.test_elemental import fixture
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import TEMPORAL_POINT_PATH_SCHEMA
from lqcdweave.storage.dataset import DatasetIndex, register_npy
from lqcdweave.production.elemental.planning import plan_indexed_point_elementals


def _axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


class PointDiagonalPlanningTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.sizes, links, _vectors, points = fixture()
        self.cfg = dict(ensemble="synthetic", configuration="cfg",
                        lattice=list(self.sizes))
        self.gauge_id = self._save("gauge", "gauge", (
            ("time", range(self.sizes[3])), ("z", range(self.sizes[2])),
            ("y", range(self.sizes[1])), ("x", range(self.sizes[0])),
            ("direction", range(4)), ("color_row", range(3)),
            ("color_column", range(3))), "complex128", links)
        self.points = points.copy()
        self.point_labels = [8, 2, 5]
        self.points_id = self._save_points("points", self.points,
                                           self.point_labels)

    def _save(self, name, kind, axes, dtype, values, conventions=None):
        spec = DatasetSpec(dict(schema=1, kind=kind, configuration=self.cfg,
            axes=[_axis(axis_name, coordinates)
                  for axis_name, coordinates in axes], dtype=dtype,
            bases={}, parents=[], definition={},
            conventions={} if conventions is None else conventions))
        source = self.root / f"{name}.npy"
        np.save(source, values)
        identity = register_npy(self.root / name, source, spec,
                                provenance=dict(test="point diagonal planning"))
        self.index.register(self.root / name)
        return identity

    def _save_points(self, name, values, labels):
        return self._save(name, "spatial_points", (
            ("time", range(self.sizes[3])), ("point", labels),
            ("xyz", ("x", "y", "z"))), "int64", values,
            conventions=dict(representation="coordinate_delta",
                             normalization="unit_delta"))

    def _document(self, paths, **options):
        preview = plan_indexed_point_elementals(
            self.index, self.gauge_id, None, self.points_id,
            paths=paths, momentum_list=((0, 0, 0),), sectors=("p2p",),
            **options)
        return preview["stages"]["p2p"]["spec"].document

    def test_same_pool_local_closed_and_winding_paths_are_declared(self):
        paths = ((), (0, 1, 3, 4), (0, 0, 0), (6,), (6, 6, 6), (0,))
        document = self._document(paths,
            temporal_path_schema=TEMPORAL_POINT_PATH_SCHEMA)

        self.assertEqual(document["conventions"]["point_diagonal_components"],
                         [0, 1, 2, 4])
        self.assertEqual(document["axes"][1]["coordinates"],
                         [dict(path=list(path)) for path in paths])

    def test_off_diagonal_only_recipe_omits_empty_declaration(self):
        document = self._document(((0,), (3,)))

        self.assertNotIn("point_diagonal_components", document["conventions"])

    def test_distinct_right_pool_does_not_claim_diagonality(self):
        right_id = self._save_points("points-alias", self.points.copy(),
                                     self.point_labels)
        document = self._document(((),), points_right_id=right_id)

        self.assertNotIn("point_diagonal_components", document["conventions"])

    def test_reordered_right_selection_does_not_claim_diagonality(self):
        document = self._document(((),),
                                  point_right_labels=tuple(reversed(self.point_labels)))

        self.assertNotIn("point_diagonal_components", document["conventions"])


if __name__ == "__main__":
    unittest.main()
