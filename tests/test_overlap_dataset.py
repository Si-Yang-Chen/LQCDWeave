"""Overlap planning contracts against scalar XYZ indexing."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.planning.overlap import overlap_dataset_document
from lqcdweave.storage.dataset import DatasetIndex, register_npy
from lqcdweave.production.overlap_planning import plan_indexed_overlap
from test_project_source_provenance import import_provenance


def _axis(name, values):
    return dict(name=name, coordinates=list(values))


def _documents(dtype="complex128"):
    cfg = dict(ensemble="overlap-synthetic", configuration="cfg", lattice=[2, 3, 2, 3])
    basis = dict(schema=1, kind="explicit_basis", configuration=cfg,
        axes=[_axis("time", [2, 0, 1]), _axis("mode", [8, 3, 9, 1]),
              _axis("z", range(2)), _axis("y", range(3)), _axis("x", range(2)),
              _axis("color", range(3))], dtype=dtype, bases={},
        definition=dict(selection="explicit"), conventions=dict(normalization="unknown"), parents=[])
    points = dict(schema=1, kind="spatial_points", configuration=cfg,
        axes=[_axis("time", [1, 2, 0]), _axis("point", [9, 2, 7]), _axis("xyz", ["x", "y", "z"])],
        dtype="int64", bases={}, definition=dict(selection="explicit"),
        conventions=dict(representation="coordinate_delta", normalization="unit_delta"), parents=[])
    return DatasetSpec(basis), DatasetSpec(points)


class OverlapDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        rng = np.random.default_rng(10421)
        self.vectors = rng.normal(size=(3, 4, 2, 3, 2, 3)) + 1j * rng.normal(size=(3, 4, 2, 3, 2, 3))
        self.coordinates = np.asarray([
            [[1, 2, 1], [0, 0, 0], [1, 0, 1]],
            [[0, 1, 0], [1, 2, 0], [0, 2, 1]],
            [[1, 1, 1], [0, 1, 1], [1, 0, 0]],
        ], dtype=np.int64)
        self.basis_spec, self.point_spec = _documents()
        self.basis_id = self._register("basis", self.basis_spec, self.vectors)
        self.points_id = self._register("points", self.point_spec, self.coordinates)

    def _register(self, name, spec, values, *, known=False):
        path = self.root / (name + ".npy")
        # The payload is in its own declared stored time order, not physical
        # time order; the oracle below always uses the physical time directly.
        np.save(path, np.stack([values[t] for t in spec.coordinates("time")]))
        directory = self.root / name
        register_npy(directory, path, spec,
            provenance=(import_provenance(producer="synthetic-fixture") if known else dict(test=name)),
            provenance_mode="external_import" if known else "legacy")
        return self.index.register(directory)



    def test_builder_and_graph_bind_actual_prefixes_without_reordering(self):
        left, right, doc, times = overlap_dataset_document(self.basis_spec, self.point_spec,
            basis_id={"request": "basis"}, points_id={"request": "points"}, usedNe=2, usedNp=2)
        request = DatasetRequest(doc)
        self.assertEqual(set(request.dependencies), {"basis", "points"})
        spec = request.resolve(dict(basis=self.basis_id, points=self.points_id))
        self.assertEqual(times, (0, 1, 2))
        self.assertEqual(left.coordinates("time"), [2, 0, 1])
        self.assertEqual(right.coordinates("time"), [1, 2, 0])
        self.assertEqual(spec.coordinates("mode"), [8, 3])
        self.assertEqual(spec.coordinates("point"), [9, 2])
        self.assertEqual(spec.document["parents"], [self.basis_id, self.points_id])
        self.assertEqual(spec.document["conventions"]["vector_conjugation"], "none")
        for basis_ref, points_ref in (({"request": "points"}, {"request": "basis"}),
                                      ({"request": "other"}, {"request": "points"})):
            with self.assertRaisesRegex(ValueError, "role request"):
                overlap_dataset_document(self.basis_spec, self.point_spec,
                    basis_id=basis_ref, points_id=points_ref)
        with patch("lqcdweave.production.eigenvector.api.calc_overlap_row",
                   side_effect=AssertionError("planning gathered")):
            preview = plan_indexed_overlap(self.index, self.basis_id, self.points_id, usedNe=2, usedNp=2)
        self.assertEqual(preview["dataset_spec"].definition_id, spec.definition_id)
        self.assertFalse((self.root / "output").exists())













if __name__ == "__main__":
    unittest.main()
