"""Read-only planning for sampled spatial-point dataset requests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lqcdweave.contracts.spatial_sampling import SpatialSamplingDesign
from lqcdweave.storage.dataset import DatasetIndex, DatasetWriter
from lqcdweave.sampling.spatial.api import sample_spatial_points
from lqcdweave.production.spatial_planning import plan_indexed_spatial_points


class SpatialPointPlanningTests(unittest.TestCase):
    def test_plan_previews_fresh_and_existing_unknown_sources(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = DatasetIndex(root / "index")
            design = SpatialSamplingDesign((2, 2, 2, 3), 3, "independent")
            configuration = dict(ensemble="sample-fixture", configuration="cfg-1",
                                 lattice=[2, 2, 2, 3])

            preview = plan_indexed_spatial_points(index, design,
                configuration=configuration, seed=17)
            strict_fresh = plan_indexed_spatial_points(index, design,
                configuration=configuration, seed=17, source_policy="require_known")

            self.assertEqual(preview["times"], (0, 1, 2))
            self.assertEqual(preview["graph"]["stages"]["points"]["action"], "compute")
            self.assertEqual(preview["dataset_spec"].coordinates("time"), [0, 1, 2])
            self.assertEqual(strict_fresh["graph"]["action"], "blocked")
            self.assertEqual(strict_fresh["source_status"], "unknown")
            self.assertEqual(index.observations(), ())
            self.assertFalse((root / "points").exists())

            points = sample_spatial_points(design, seed=17)
            output = root / "unknown-points"
            with DatasetWriter(output, preview["dataset_spec"],
                                provenance={"test": "unknown source"}) as writer:
                for row in writer.missing:
                    writer.write(row, points.coordinates[row])
                writer.publish()
            identity = index.register(output)

            existing = plan_indexed_spatial_points(index, design,
                configuration=configuration, seed=17)
            strict_existing = plan_indexed_spatial_points(index, design,
                configuration=configuration, seed=17,
                data_ids={"points": identity}, source_policy="require_known")
            self.assertEqual(existing["graph"]["stages"]["points"]["action"], "reuse")
            self.assertEqual(strict_existing["graph"]["action"], "blocked")
            self.assertEqual(strict_existing["source_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
