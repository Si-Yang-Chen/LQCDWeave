"""Pure logical-reference validation and dataset dependency graph planning."""

import unittest

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.planning.dataset import plan_dataset_graph


def make_spec(*, kind="vsv.raw", time=(0, 1, 2), mode=(0, 1),
              parents=(), configuration="cfg-17", basis_id="basis-a"):
    return DatasetSpec(dict(
        schema=1,
        kind=kind,
        configuration=dict(ensemble="ensemble-a", configuration=configuration,
                           lattice=[4, 4, 4, 4]),
        axes=[dict(name="time", coordinates=list(time)),
              dict(name="mode", coordinates=list(mode))],
        dtype="complex128",
        bases=dict(mode=dict(data_id=basis_id, indices=list(mode))),
        definition=dict(mass=0.2, solver_tolerance=1e-10),
        conventions=dict(gauge_id="gauge-a", normalization="unit"),
        parents=list(parents),
    ))


def observation(spec, identity, *, state="complete", completed=None):
    if completed is None:
        completed = tuple(range(spec.shape[0])) if state == "complete" else ()
    return dict(state=state, spec=spec, provenance={"route": "test"},
                data_id=identity, completed=tuple(completed))


class DatasetRequestTests(unittest.TestCase):
    def test_template_is_detached_and_resolves_only_real_scientific_ids(self):
        document = make_spec().document
        document["parents"] = [{"request": "basis"}, {"request": "basis"}]
        document["bases"]["mode"]["data_id"] = {"request": "basis"}
        document["definition"] = {
            "mass": 0.2,
            "operator_id": {"request": "basis"},
            "input_roles": {
                "left": {"request": "basis"},
                "right": {"request": "points"},
            },
            "input_views": {
                "nested": {
                    "parents": [{"request": "points"}],
                    "data_id": {"request": "basis"},
                    "identity": {"source_id": {"request": "points"}},
                },
            },
        }
        document["conventions"]["normalization_id"] = {"request": "points"}

        request = DatasetRequest(document, subset_axes=("time",))
        detached = request.template
        detached["kind"] = "changed"
        self.assertEqual(request.template["kind"], "vsv.raw")
        self.assertEqual(request.dependencies, ("basis", "points"))
        self.assertEqual(request.subset_axes, ("time",))
        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            request.resolve({"basis": "basis-actual"})

        resolved = request.resolve({"basis": "basis-actual", "points": "points-actual"})
        self.assertEqual(resolved.document["parents"], ["basis-actual", "basis-actual"])
        self.assertEqual(resolved.document["bases"]["mode"]["data_id"], "basis-actual")
        self.assertEqual(resolved.document["definition"]["operator_id"], "basis-actual")
        self.assertEqual(resolved.document["definition"]["input_roles"], {
            "left": "basis-actual", "right": "points-actual",
        })
        nested = resolved.document["definition"]["input_views"]["nested"]
        self.assertEqual(nested["parents"], ["points-actual"])
        self.assertEqual(nested["data_id"], "basis-actual")
        self.assertEqual(nested["identity"]["source_id"], "points-actual")
        self.assertEqual(resolved.document["conventions"]["normalization_id"], "points-actual")

    def test_logical_references_are_rejected_outside_identity_positions(self):
        for path in ("axis", "dtype", "mass", "malformed-axis-reference"):
            document = make_spec().document
            if path == "axis":
                document["axes"][0]["coordinates"][0] = {"request": "parent"}
            elif path == "malformed-axis-reference":
                document["axes"][0]["coordinates"][0] = {
                    "request": "parent", "extra": "not-a-logical-reference",
                }
            elif path == "dtype":
                document["dtype"] = {"request": "parent"}
            else:
                document["definition"]["mass"] = {"request": "parent"}
            with self.subTest(path=path), self.assertRaises(ValueError):
                DatasetRequest(document)

    def test_reference_shape_and_node_name_are_validated(self):
        for reference in ({"request": "parent", "extra": 1}, {"request": ""},
                          {"request": 4}):
            document = make_spec().document
            document["parents"] = [reference]
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                DatasetRequest(document)


    def test_resolved_bindings_validate_nested_ids_and_the_entire_definition(self):
        template = make_spec().document
        reference = dict(request="basis")
        template["parents"] = [reference]
        template["bases"]["mode"]["data_id"] = reference
        embedded = make_spec().document
        embedded["parents"] = [dict(request="gauge")]
        embedded["bases"]["mode"]["data_id"] = reference
        template["definition"] = dict(mass=.2, input_roles=dict(left=reference),
                                      input_views=dict(left=embedded))
        request = DatasetRequest(template)
        expected = dict(basis="actual-basis", gauge="actual-gauge")
        resolved = request.resolve(expected)
        self.assertEqual(request.resolved_bindings(resolved), expected)
        changed = resolved.document
        changed["bases"]["mode"]["data_id"] = "different-basis"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            request.resolved_bindings(DatasetSpec(changed))
        changed = resolved.document
        changed["definition"]["mass"] = .3
        with self.assertRaisesRegex(ValueError, "definition or input dataset differs"):
            request.resolved_bindings(DatasetSpec(changed))


class DatasetGraphPlanningTests(unittest.TestCase):
    def test_topological_order_and_complete_bindings_are_deterministic(self):
        spec_z = make_spec(kind="root.z")
        spec_a = make_spec(kind="root.a")
        spec_child = make_spec(kind="derived", parents=("actual-z",))
        child_template = spec_child.document
        child_template["parents"] = [{"request": "z-root"}]
        requests = {
            "b-child": DatasetRequest(child_template),
            "z-root": DatasetRequest(spec_z),
            "a-root": DatasetRequest(spec_a),
        }
        result = plan_dataset_graph(requests, [
            observation(spec_z, "actual-z"),
            observation(spec_a, "actual-a"),
            observation(spec_child, "actual-child"),
        ])

        self.assertEqual(result["order"], ("a-root", "z-root", "b-child"))
        self.assertEqual(result["bindings"], {
            "a-root": "actual-a", "z-root": "actual-z", "b-child": "actual-child",
        })
        self.assertEqual(result["stages"]["b-child"]["requires"], ("z-root",))
        self.assertEqual(result["stages"]["b-child"]["spec"].document["parents"], ["actual-z"])

    def test_partial_parent_is_not_bound_and_blocks_child_spec(self):
        root = make_spec(kind="root", time=(0, 1, 2))
        partial = make_spec(kind="root", time=(1,))
        child = make_spec(kind="derived", parents=("partial-root",))
        child_template = child.document
        child_template["parents"] = [{"request": "root"}]
        result = plan_dataset_graph(
            {
                "root": DatasetRequest(root, subset_axes=("time",)),
                "child": DatasetRequest(child_template),
            },
            [observation(partial, "partial-root")],
        )

        self.assertEqual(result["stages"]["root"]["action"], "compute")
        self.assertEqual(result["stages"]["root"]["data_id"], "partial-root")
        self.assertEqual(result["stages"]["root"]["missing"], (0, 2))
        self.assertEqual(result["bindings"], {})
        blocked = result["stages"]["child"]
        self.assertEqual(blocked["action"], "blocked")
        self.assertIsNone(blocked["spec"])
        self.assertEqual(blocked["missing"], (0, 1, 2))
        self.assertIn("root", " ".join(blocked["reasons"]))

    def test_unknown_scientific_identity_blocks_node_and_downstream(self):
        unknown = DatasetSpec({**make_spec().document,
                                "configuration": dict(ensemble="ensemble-a",
                                                      configuration=None,
                                                      lattice=[4, 4, 4, 4])})
        child = make_spec(kind="derived", parents=("unknown-id",))
        child_template = child.document
        child_template["parents"] = [{"request": "unknown"}]
        result = plan_dataset_graph({
            "unknown": DatasetRequest(unknown),
            "child": DatasetRequest(child_template),
        }, [observation(unknown, "unknown-id")])

        self.assertEqual(result["stages"]["unknown"]["action"], "blocked")
        self.assertIsInstance(result["stages"]["unknown"]["spec"], DatasetSpec)
        self.assertIsNone(result["stages"]["child"]["spec"])
        self.assertEqual(result["bindings"], {})

    def test_unknown_nodes_cycles_and_conflicting_fixed_ids_fail_early(self):
        base = make_spec()
        missing_template = base.document
        missing_template["parents"] = [{"request": "absent"}]
        with self.assertRaisesRegex(ValueError, "unknown request"):
            plan_dataset_graph({"node": DatasetRequest(missing_template)}, [])

        a = base.document
        b = base.document
        a["parents"] = [{"request": "b"}]
        b["parents"] = [{"request": "a"}]
        with self.assertRaisesRegex(ValueError, "cycle"):
            plan_dataset_graph({"a": DatasetRequest(a), "b": DatasetRequest(b)}, [])

        with self.assertRaisesRegex(ValueError, "conflict"):
            plan_dataset_graph(
                {"node": DatasetRequest(base, data_id="one")}, [],
                data_ids={"node": "two"},
            )

    def test_fixed_actual_id_requires_a_compatible_complete_observation(self):
        requested = make_spec()
        pinned = DatasetRequest(requested, data_id="right-id")
        matching = plan_dataset_graph(
            {"node": pinned}, [observation(requested, "right-id")],
        )
        self.assertEqual(matching["bindings"], {"node": "right-id"})
        with self.assertRaisesRegex(ValueError, "fully compatible"):
            plan_dataset_graph(
                {"node": pinned}, [observation(requested, "other-id")],
            )


if __name__ == "__main__":
    unittest.main()
