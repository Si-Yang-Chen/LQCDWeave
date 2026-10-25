"""Laplace planning and test-owned production script lifecycle checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from tests.fixtures import laplace_reference as laplace_fixture

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.production.eigenvector.planning import plan_indexed_laplace_basis
from fixtures.laplace_script import produce_basis, produce_eigenvalues


def _axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


@unittest.skipIf(laplace_fixture.scipy is None, "SciPy required for Laplace production")
class _LaplaceDatasetFixture(unittest.TestCase):
    sizes = (2, 1, 1, 3)

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.links = laplace_fixture.random_gauge(self.sizes, seed=94)
        self.index = DatasetIndex(self.root / "index")
        self.gauge_id = self._register_gauge("periodic", boundary="periodic")

    def _register_gauge(self, name, *, boundary, provenance=None):
        lx, ly, lz, lt = self.sizes
        spec = DatasetSpec(dict(schema=1, kind="gauge", configuration=dict(
            ensemble="laplace-test", configuration="cfg-laplace",
            lattice=[lx, ly, lz, lt]), axes=[
                _axis("time", range(lt)), _axis("z", range(lz)),
                _axis("y", range(ly)), _axis("x", range(lx)),
                _axis("direction", range(4)), _axis("color_row", range(3)),
                _axis("color_column", range(3)),
            ], dtype="complex128", bases={}, parents=[], definition={},
            conventions=dict(boundary=boundary)))
        path = self.root / name
        with DatasetWriter(path, spec,
                provenance=provenance or dict(test="registered-laplace-gauge")) as writer:
            for time in range(self.sizes[3]):
                writer.write(time, self.links[time])
            writer.publish()
        return self.index.register(path)

class LaplaceDatasetTests(_LaplaceDatasetFixture):
    def test_basis_row_temporaries_are_released_before_next_solve(self):
        owner = self

        class Writer:
            missing = (0, 1, 2)

            def __init__(self):
                self.positions = []

            def write(self, position, value):
                self.positions.append(position)
                self.last_shape = value.shape

        class Laplace:
            xp = np
            backend = None

            def __init__(self):
                self.references = []

            def load(self, configuration):
                return self

            def calc(self, time, **kwargs):
                gc.collect()
                if self.references:
                    owner.assertTrue(all(ref() is None for ref in self.references[-1]))
                vectors = np.full((2, 1, 1, 2, 3), time + 1,
                                  dtype=np.complex128)
                values = np.array([time + 1, time + 2], dtype=np.float64)
                self.references.append((weakref.ref(vectors), weakref.ref(values)))
                return vectors, values

        writer, laplace = Writer(), Laplace()
        produce_basis(writer, laplace, (0, 1, 2), "cfg", {})
        gc.collect()
        self.assertEqual(writer.positions, [0, 1, 2])
        self.assertTrue(all(ref() is None for pair in laplace.references for ref in pair))

    def test_eigenvalue_fallback_rows_release_recomputed_arrays(self):
        owner = self

        class Writer:
            missing = (0, 1, 2)

            def __init__(self):
                self.positions = []

            def write(self, position, value):
                self.positions.append(position)
                self.last_shape = value.shape

        class SavedBasis:
            def read(self, *, time):
                value = np.zeros((2, 1, 1, 2, 3), dtype=np.complex128)
                value[0, 0, 0, 0, 0] = 1
                value[1, 0, 0, 1, 0] = 1
                return value

        class Laplace:
            xp = np
            backend = None

            def __init__(self):
                self.references = []

            def load(self, configuration):
                return self

            def calc(self, time, **kwargs):
                gc.collect()
                if self.references:
                    owner.assertTrue(all(ref() is None for ref in self.references[-1]))
                candidate = SavedBasis().read(time=time)
                values = np.array([0.0, 1.0], dtype=np.float64)
                self.references.append((weakref.ref(candidate), weakref.ref(values)))
                return candidate, values

        writer, laplace = Writer(), Laplace()
        produce_eigenvalues(writer, laplace, (0, 1, 2), "cfg", 2,
            SavedBasis(), {})
        gc.collect()
        self.assertEqual(writer.positions, [0, 1, 2])
        self.assertTrue(all(ref() is None for pair in laplace.references for ref in pair))

    def test_plan_is_full_time_metadata_preview_without_solver_or_output(self):
        output = self.root / "preview-output"
        with patch.object(LaplaceEigenvectorGenerator, "calc",
                          side_effect=AssertionError("plan called eigensolver")):
            graph = plan_indexed_laplace_basis(self.index, self.gauge_id)
        self.assertEqual(graph["stages"]["gauge"]["action"], "reuse")
        self.assertEqual(graph["stages"]["basis"]["action"], "compute")
        self.assertEqual(graph["stages"]["basis"]["spec"].coordinates("time"), [0, 1, 2])
        self.assertEqual(graph["stages"]["eigenvalues"]["action"], "blocked")
        self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
