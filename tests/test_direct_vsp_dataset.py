"""Direct VSP calculation checked against an independent inverse oracle."""

import unittest

from test_perambulator_dataset import (
    _CONFIGURATION,
    _DIRAC,
    _PerambulatorDatasetFixture,
)
from lqcdweave.planning.production import propagator_dataset_document


class DirectVSPNumericsTests(_PerambulatorDatasetFixture):
    def test_vsp_only_matches_inverse_for_all_physical_source_sink_times(self):
        preview = self._plan_products(("VSP",),
            basis_snk_id=self.sink_basis_id, Ne_snk=2,
            points_src_id=self.source_points_id, Np_src=1)
        self.assertEqual(set(preview["stages"]),
                         {"gauge", "basis", "points", "vsp"})
        spec = preview["stages"]["vsp"]["spec"]
        self.assertNotIn("derivation", spec.document["definition"])
        identities = self._produce_products(preview, ("VSP",))
        self.assertEqual(set(identities), {"vsp"})
        with self.index.open(identities["vsp"]) as reader:
            self.assertEqual(reader.spec.document["parents"], [
                self.sink_basis_id, self.gauge_id, self.source_points_id,
            ])
            self.assertEqual(reader.spec.coordinates("source_time"), [0, 1, 2])
            self.assertEqual(reader.spec.coordinates("sink_time"), [0, 1, 2])
            self.assertEqual(reader.spec.coordinates("sink_mode"), [11, 2])
            self.assertEqual(reader.spec.coordinates("source_point"), [9])
        self._assert_products_match_inverse(identities, ("VSP",))

    def test_direct_vsp_spec_is_not_gamma5_derived_and_aliases_are_explicit(self):
        document = propagator_dataset_document(
            _CONFIGURATION, (0, 1, 2), product="VSP",
            gauge_id=self.gauge_id, basis_snk_id=self.sink_basis_id,
            modes_snk=(11,), points_src_id=self.source_points_id,
            point_src_labels=(9,), dirac=_DIRAC,
            solver_tolerance=1e-11, temporal_boundary="antiperiodic")
        self.assertEqual(document["kind"], "vsp")
        self.assertNotIn("derivation", document["definition"])
        with self.assertRaisesRegex(ValueError, "explicit source point"):
            propagator_dataset_document(
                _CONFIGURATION, (0, 1, 2), product="VSP",
                gauge_id=self.gauge_id, basis_snk_id=self.sink_basis_id,
                modes_snk=(11,), points_id=self.source_points_id,
                point_labels=(9,), dirac=_DIRAC,
                solver_tolerance=1e-11, temporal_boundary="antiperiodic")


if __name__ == "__main__":
    unittest.main()
