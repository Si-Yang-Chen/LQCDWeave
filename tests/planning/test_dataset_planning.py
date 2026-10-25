"""Pure compatibility and coverage planning for complete datasets."""

import unittest

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.planning.dataset import plan_requirement


def make_spec(*, time=(0, 1, 2, 3), mode=(0, 1), point=(0, 1),
              ensemble="ensemble-a", configuration="cfg-17", lattice=(4, 4, 4, 4),
              kind="vsv.raw", dtype="complex128", mass=0.2,
              tolerance=1e-10, gauge="gauge-a", point_parent="points-a",
              normalization="unit", projection="raw", basis_id="basis-a"):
    return DatasetSpec(dict(
        schema=1,
        kind=kind,
        configuration=dict(ensemble=ensemble, configuration=configuration,
                           lattice=list(lattice)),
        axes=[dict(name="time", coordinates=list(time)),
              dict(name="mode", coordinates=list(mode)),
              dict(name="point", coordinates=list(point))],
        dtype=dtype,
        bases=dict(mode=dict(data_id=basis_id, indices=list(mode))),
        definition=dict(mass=mass, solver_tolerance=tolerance),
        conventions=dict(gauge_id=gauge, normalization=normalization,
                         projection=projection),
        parents=[gauge, point_parent],
    ))


def observation(spec, identity, *, state="complete", completed=None):
    if completed is None:
        completed = tuple(range(spec.shape[0])) if state == "complete" else ()
    return dict(state=state, spec=spec, provenance={"route": "test"},
                data_id=identity, completed=tuple(completed))


class DatasetRequirementPlanningTests(unittest.TestCase):
    def test_nested_dilution_identity_retains_json_distinctions_and_precise_paths(self):
        document = make_spec().document
        document['definition']['dilution'] = {
            'segments': [{'modes': list(range(200)), 'coefficient': 0.0}],
        }
        requested = DatasetSpec(document)
        self.assertEqual(plan_requirement(requested, [observation(
            DatasetSpec(document), 'identical')])['action'], 'reuse')
        for changed in (True, 1.0, -0.0):
            candidate = requested.document
            if changed == 1:
                candidate['definition']['dilution']['segments'][0]['modes'][1] = changed
                path = 'definition.dilution.segments[0].modes[1]'
            else:
                candidate['definition']['dilution']['segments'][0]['coefficient'] = changed
                path = 'definition.dilution.segments[0].coefficient'
            with self.subTest(changed=repr(changed)):
                decision = plan_requirement(requested, [observation(DatasetSpec(candidate), 'different')])
                self.assertEqual(decision['action'], 'compute')
                self.assertIn(path, ' '.join(decision['reasons']))

    def test_explicit_lower_mode_view_preserves_requested_coordinate_order(self):
        requested = make_spec(mode=(2, 0))
        broader_basis = make_spec(mode=(0, 1, 2, 3))

        plan = plan_requirement(
            requested, [observation(broader_basis, "basis-data")],
            subset_axes=("mode",),
        )

        self.assertEqual(plan["action"], "reuse")
        self.assertEqual(plan["data_id"], "basis-data")
        self.assertEqual(plan["completed"], (0, 1, 2, 3))
        self.assertEqual(plan["missing"], ())
        self.assertEqual(plan["selection"], {"mode": [2, 0], "point": [0, 1]})

    def test_partial_first_axis_returns_request_positions_and_keeps_source_id(self):
        requested = make_spec(time=(0, 1, 2, 3))
        available = make_spec(time=(3, 1))

        plan = plan_requirement(
            requested, [observation(available, "time-slices")],
            subset_axes=("time",),
        )

        self.assertEqual(plan["action"], "compute")
        self.assertEqual(plan["data_id"], "time-slices")
        self.assertEqual(plan["completed"], (1, 3))
        self.assertEqual(plan["missing"], (0, 2))
        self.assertEqual(plan["selection"], {"mode": [0, 1], "point": [0, 1]})

    def test_first_axis_partial_coverage_requires_explicit_subset_permission(self):
        requested = make_spec(time=(0, 1, 2))
        available = make_spec(time=(1, 2))

        plan = plan_requirement(requested, [observation(available, "partial-time")])

        self.assertEqual(plan["action"], "compute")
        self.assertIsNone(plan["data_id"])
        self.assertEqual(plan["completed"], ())
        self.assertEqual(plan["missing"], (0, 1, 2))
        self.assertIn("not subset-enabled", " ".join(plan["reasons"]))

    def test_mass_gauge_point_parent_convention_dtype_and_kind_must_match(self):
        requested = make_spec()
        incompatible = [
            ("definition.mass", make_spec(mass=0.3)),
            ("conventions.gauge_id", make_spec(gauge="gauge-b")),
            ("parents[1]", make_spec(point_parent="points-b")),
            ("conventions.normalization", make_spec(normalization="different")),
            ("dtype", make_spec(dtype="complex64")),
            ("kind", make_spec(kind="vsv.projected")),
            ("definition.solver_tolerance", make_spec(tolerance=1e-8)),
        ]

        plan = plan_requirement(
            requested,
            [observation(candidate, f"bad-{index}")
             for index, (_, candidate) in enumerate(incompatible)],
        )

        self.assertEqual(plan["action"], "compute")
        self.assertIsNone(plan["data_id"])
        reason_text = " ".join(plan["reasons"])
        for mismatch_path, _ in incompatible:
            self.assertIn(mismatch_path, reason_text)

    def test_actual_basis_id_is_required_even_when_mode_shape_matches(self):
        requested = make_spec(mode=(0, 1), basis_id="actual-basis-a")
        same_shape_other_basis = make_spec(mode=(0, 1), basis_id="actual-basis-b")

        plan = plan_requirement(
            requested, [observation(same_shape_other_basis, "same-shape")],
        )

        self.assertEqual(plan["action"], "compute")
        self.assertIsNone(plan["data_id"])
        self.assertIn("bases.mode.data_id", " ".join(plan["reasons"]))

    def test_unknown_configuration_basis_or_semantic_value_blocks_planning(self):
        unknown_specs = [
            make_spec(configuration=None),
            make_spec(basis_id=None),
            DatasetSpec({**make_spec().document,
                          "conventions": {"gauge_id": "gauge-a", "normalization": None,
                                          "projection": "raw"}}),
        ]

        for unknown in unknown_specs:
            with self.subTest(document=unknown.document):
                plan = plan_requirement(unknown, [])
                self.assertEqual(plan["action"], "blocked")
                self.assertIsNone(plan["data_id"])
                self.assertEqual(plan["completed"], ())
                self.assertEqual(plan["missing"], tuple(range(unknown.shape[0])))
                self.assertIn("unknown", " ".join(plan["reasons"]))

    def test_partial_observations_are_never_completed_reuse_candidates(self):
        requested = make_spec()
        partial = observation(requested, "draft", state="partial", completed=(0, 1))

        plan = plan_requirement(requested, [partial])

        self.assertEqual(plan["action"], "compute")
        self.assertIsNone(plan["data_id"])
        self.assertEqual(plan["missing"], tuple(range(requested.shape[0])))
        self.assertIn("only complete datasets", " ".join(plan["reasons"]))

    def test_best_coverage_then_data_id_make_selection_deterministic(self):
        requested = make_spec(time=(0, 1, 2, 3))
        candidates = [
            observation(make_spec(time=(0, 1, 2)), "z-three"),
            observation(make_spec(time=(1, 2, 3)), "b-three"),
            observation(make_spec(time=(0, 1)), "a-two"),
        ]

        forward = plan_requirement(requested, candidates, subset_axes=("time",))
        reverse = plan_requirement(requested, list(reversed(candidates)), subset_axes=("time",))

        self.assertEqual(forward["data_id"], "b-three")
        self.assertEqual(forward["completed"], (1, 2, 3))
        self.assertEqual(forward["missing"], (0,))
        self.assertEqual(reverse["data_id"], forward["data_id"])
        self.assertEqual(reverse["completed"], forward["completed"])

    def test_explicit_data_id_prevents_fallback_to_another_candidate(self):
        requested = make_spec(time=(0, 1, 2, 3))
        short = observation(make_spec(time=(0,)), "requested-id")
        broad = observation(make_spec(time=(0, 1, 2, 3)), "other-id")

        selected = plan_requirement(
            requested, [broad, short], subset_axes=("time",),
            data_id="requested-id",
        )
        absent = plan_requirement(
            requested, [broad, short], subset_axes=("time",),
            data_id="absent-id",
        )

        self.assertEqual(selected["data_id"], "requested-id")
        self.assertEqual(selected["completed"], (0,))
        self.assertEqual(selected["missing"], (1, 2, 3))
        self.assertEqual(absent["action"], "compute")
        self.assertIsNone(absent["data_id"])
        self.assertEqual(absent["missing"], (0, 1, 2, 3))

    def test_nonfirst_subset_must_cover_request_and_keep_actual_coordinates(self):
        requested = make_spec(point=(2, 0))
        partial_points = make_spec(point=(0,))
        complete_points = make_spec(point=(0, 1, 2, 3))

        plan = plan_requirement(
            requested,
            [observation(partial_points, "missing-point"),
             observation(complete_points, "all-points")],
            subset_axes=("point",),
        )

        self.assertEqual(plan["action"], "reuse")
        self.assertEqual(plan["data_id"], "all-points")
        self.assertEqual(plan["selection"]["point"], [2, 0])
        self.assertIn("does not cover requested coordinates", " ".join(plan["reasons"]))

    def test_tie_uses_lexicographically_smaller_data_id(self):
        requested = make_spec(time=(0, 1, 2))
        a = observation(make_spec(time=(0, 2)), "a-id")
        z = observation(make_spec(time=(1, 2)), "z-id")

        plan = plan_requirement(requested, [z, a], subset_axes=("time",))

        self.assertEqual(plan["data_id"], "a-id")
        self.assertEqual(plan["completed"], (0, 2))


if __name__ == "__main__":
    unittest.main()
