"""Integration check for the M-to-P profile-reference boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lqcdweave.contraction.planning.api import find_contraction_path
from lqcdweave.contracts.correlator.api import (
    BatchSpec,
    CoordinatePoint,
    CoordinateSpec,
    DiagramBindings,
    ExpandedCorrelator,
    OutputNode,
    OutputRecipe,
)


class ProfilePlanningBoundaryTests(unittest.TestCase):
    def test_low_level_planner_rejects_profile_file_references(self):
        coordinates = CoordinateSpec("bound", (), (), (), (CoordinatePoint("only", ()),), ())
        expanded = ExpandedCorrelator(
            None, None, OutputRecipe(OutputNode("empty"), (), coordinates),
            DiagramBindings(), coordinates, BatchSpec(),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            with self.assertRaisesRegex(TypeError, "resolve profile names and paths in the contraction frontend"):
                find_contraction_path(expanded, memory_limit=None, profile=path)


if __name__ == "__main__":
    unittest.main()
