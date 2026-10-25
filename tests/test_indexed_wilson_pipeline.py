"""Read-only Wilson planning checked against registered metadata."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.fixtures.laplace_reference import random_gauge
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator
from lqcdweave.production.perambulator.api import PerambulatorGenerator
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.production.elemental.planning import ElementalGenerator
from lqcdweave.production.wilson_reference import plan_indexed_wilson_reference


class IndexedWilsonPlanningTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        links = random_gauge((2, 1, 1, 3), seed=94)
        self.index = DatasetIndex(self.root / "index")
        axes = [dict(name=name, coordinates=list(range(size))) for name, size in zip(
            ("time", "z", "y", "x", "direction", "color_row", "color_column"),
            links.shape,
            strict=True,
        )]
        spec = DatasetSpec(dict(
            schema=1,
            kind="gauge",
            configuration=dict(ensemble="test", configuration="cfg",
                               lattice=[2, 1, 1, 3]),
            axes=axes,
            dtype="complex128",
            bases={},
            parents=[],
            definition={},
            conventions=dict(boundary="periodic"),
        ))
        gauge_directory = self.root / "gauge"
        with DatasetWriter(gauge_directory, spec,
                            provenance=dict(test="independent")) as writer:
            for time in writer.missing:
                writer.write(time, links[time])
            writer.publish()
        self.gauge_id = self.index.register(gauge_directory)

    def test_preview_does_not_solve_or_bind_unpublished_parents(self):
        index_bytes = self.index.path.read_bytes()
        with patch.object(LaplaceEigenvectorGenerator, "calc",
                          side_effect=AssertionError("unexpected solve")), \
             patch.object(PerambulatorGenerator, "calc",
                          side_effect=AssertionError("unexpected solve")), \
             patch.object(ElementalGenerator, "calc_path_blocks",
                          side_effect=AssertionError("unexpected production")):
            graph = plan_indexed_wilson_reference(self.index, self.gauge_id)
        self.assertEqual(graph["stages"]["gauge"]["action"], "reuse")
        self.assertEqual(graph["stages"]["basis"]["action"], "compute")
        for name in ("eigenvalues", "vsv", "elemental", "correlator"):
            self.assertEqual(graph["stages"][name]["action"], "blocked")
            self.assertIsNone(graph["stages"][name]["spec"])
        self.assertEqual(self.index.path.read_bytes(), index_bytes)
        self.assertEqual({path.name for path in self.root.iterdir()}, {"index", "gauge"})

        strict = plan_indexed_wilson_reference(
            self.index, self.gauge_id, source_policy="require_known",
        )
        self.assertEqual(strict["source_policy"], "require_known")
        self.assertEqual(strict["action"], "blocked")
        self.assertTrue(strict["source_reasons"])


if __name__ == "__main__":
    unittest.main()
