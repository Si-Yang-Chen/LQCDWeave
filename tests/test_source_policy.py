"""Source policy follows only declared dataset identities through metadata."""

import unittest
from unittest.mock import patch

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.runtime.datasets import apply_source_policy


_ORIGIN_FIELDS = (
    "source_system", "dataset_id", "revision", "producer", "producer_revision",
)


def known_origin():
    values = {
        "source_system": "fixture-archive",
        "dataset_id": "ensemble/cfg-1",
        "revision": "rev-1",
        "producer": "external-solver",
        "producer_revision": "solver-1",
    }
    return {
        "status": "recorded",
        "origin": {
            name: {"status": "known", "value": values[name]}
            for name in _ORIGIN_FIELDS
        },
    }


def unknown_origin(reason="legacy provenance"):
    return {"status": "unknown", "reason": reason}


def dataset_spec(*, parents=(), definition=None):
    return DatasetSpec({
        "schema": 1,
        "kind": "source_fixture",
        "dtype": "float64",
        "configuration": {
            "ensemble": "fixture",
            "configuration": "cfg-1",
            "lattice": [1, 1, 1, 2],
        },
        "axes": [
            {"name": "time", "coordinates": [0, 1]},
            {"name": "component", "coordinates": ["value"]},
        ],
        "bases": {},
        "definition": {} if definition is None else definition,
        "conventions": {},
        "parents": list(parents),
    })


def metadata_record(identity, spec, *, execution_schema=3, source_origin=None,
                    state="metadata"):
    return {
        "state": state,
        "data_id": identity,
        "spec": spec,
        "provenance": {"fixture": "explicit"},
        "execution": {"schema": execution_schema},
        "source_origin": known_origin() if source_origin is None else source_origin,
        "scientific_validation": {"status": "unknown"},
        "registration_runtime": {"status": "unknown"},
    }


class _Reader:
    def __init__(self, record):
        self.spec = record["spec"]
        self.execution = record["execution"]
        self.source_origin = record["source_origin"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _MetadataIndex:
    def __init__(self, records):
        self.records = dict(records)
        self.metadata_calls = []
        self.open_calls = []
        self.observations_calls = 0

    def observe_metadata(self, identity):
        self.metadata_calls.append(identity)
        if identity not in self.records:
            raise KeyError(identity)
        return dict(self.records[identity])

    def open(self, identity):
        self.open_calls.append(identity)
        if identity not in self.records:
            raise KeyError(identity)
        return _Reader(self.records[identity])

    def observations(self):
        self.observations_calls += 1
        raise AssertionError("source policy must not enumerate the dataset index")


def request(identity, spec):
    return DatasetRequest(spec, data_id=identity)


def planned(*bindings, action="reuse", reasons=()):
    return {
        "action": action,
        "reasons": tuple(reasons),
        "stages": {
            role: {"action": action, "data_id": identity, "spec": spec}
            for role, identity, spec in bindings
        },
    }


class SourcePolicyTests(unittest.TestCase):
    def test_allow_unknown_returns_the_preview_without_index_reads(self):
        index = _MetadataIndex({})
        preview = {"action": "reuse", "stages": {}}
        result = apply_source_policy(preview, {}, index)
        self.assertIs(result, preview)
        self.assertEqual(index.metadata_calls, [])
        self.assertEqual(index.open_calls, [])
        self.assertEqual(index.observations_calls, 0)

    def test_known_source_uses_declared_metadata_only(self):
        identity = "external-source"
        spec = dataset_spec()
        index = _MetadataIndex({identity: metadata_record(identity, spec)})
        requests = {"field": request(identity, spec)}
        result = apply_source_policy(
            planned(("field", identity, spec)), requests, index,
            source_policy="require_known")
        self.assertEqual(result["source_status"], "known")
        self.assertEqual(result["source_origins"][identity], known_origin())
        self.assertEqual(index.metadata_calls, [identity])
        self.assertEqual(index.open_calls, [])
        self.assertEqual(index.observations_calls, 0)

    def test_derived_dataset_can_reach_a_known_external_ancestor(self):
        derived_id, source_id = "derived", "external-source"
        derived_spec = dataset_spec(parents=(source_id,))
        source_spec = dataset_spec()
        index = _MetadataIndex({
            derived_id: metadata_record(derived_id, derived_spec,
                                        execution_schema=1,
                                        source_origin=unknown_origin()),
            source_id: metadata_record(source_id, source_spec),
        })
        result = apply_source_policy(
            planned(("product", derived_id, derived_spec)),
            {"product": request(derived_id, derived_spec)}, index,
            source_policy="require_known")
        self.assertEqual(result["source_status"], "known")
        self.assertEqual(set(index.metadata_calls), {derived_id, source_id})
        self.assertEqual(index.open_calls, [])

    def test_legacy_terminal_and_incomplete_external_origin_are_blocked(self):
        spec = dataset_spec()
        cases = (
            ("legacy", metadata_record("legacy", spec, execution_schema=1,
                                        source_origin=unknown_origin())),
            ("partial", metadata_record("partial", spec,
                                         source_origin=unknown_origin("revision unknown"))),
        )
        for identity, record in cases:
            with self.subTest(identity=identity):
                index = _MetadataIndex({identity: record})
                result = apply_source_policy(
                    planned(("field", identity, spec)),
                    {"field": request(identity, spec)}, index,
                    source_policy="require_known")
                self.assertEqual(result["action"], "blocked")
                self.assertEqual(result["source_status"], "unknown")
                self.assertTrue(result["source_reasons"])
                self.assertEqual(index.open_calls, [])

    def test_missing_parent_and_declared_parent_cycle_are_blocked(self):
        orphan_id, missing_id = "orphan", "missing-parent"
        orphan_spec = dataset_spec(parents=(missing_id,))
        missing_index = _MetadataIndex({orphan_id: metadata_record(
            orphan_id, orphan_spec, execution_schema=1,
            source_origin=unknown_origin())})
        result = apply_source_policy(
            planned(("product", orphan_id, orphan_spec)),
            {"product": request(orphan_id, orphan_spec)}, missing_index,
            source_policy="require_known")
        self.assertEqual(result["action"], "blocked")
        self.assertTrue(any("cannot be read" in reason
                            for reason in result["source_reasons"]))

        left_id, right_id = "cycle-left", "cycle-right"
        left_spec = dataset_spec(parents=(right_id,))
        right_spec = dataset_spec(parents=(left_id,))
        cyclic_index = _MetadataIndex({
            left_id: metadata_record(left_id, left_spec, execution_schema=1,
                                     source_origin=unknown_origin()),
            right_id: metadata_record(right_id, right_spec, execution_schema=1,
                                      source_origin=unknown_origin()),
        })
        result = apply_source_policy(
            planned(("product", left_id, left_spec)),
            {"product": request(left_id, left_spec)}, cyclic_index,
            source_policy="require_known")
        self.assertEqual(result["action"], "blocked")
        self.assertTrue(any("parent cycle" in reason
                            for reason in result["source_reasons"]))

    def test_walk_follows_declared_roles_and_views_not_arbitrary_labels(self):
        root_id, parent_id, view_id = "root", "role-parent", "view-parent"
        embedded = dataset_spec(parents=(view_id,)).document
        root_spec = dataset_spec(parents=(parent_id,), definition={
            "dataset_id": "label-is-not-an-dataset-id",
            "input_roles": {"field": parent_id},
            "input_views": {"field": embedded},
        })
        parent_spec = dataset_spec()
        view_spec = dataset_spec()
        index = _MetadataIndex({
            root_id: metadata_record(root_id, root_spec),
            parent_id: metadata_record(parent_id, parent_spec),
            view_id: metadata_record(view_id, view_spec),
        })
        result = apply_source_policy(
            planned(("product", root_id, root_spec)),
            {"product": request(root_id, root_spec)}, index,
            source_policy="require_known")
        self.assertEqual(result["source_status"], "known")
        self.assertEqual(set(index.metadata_calls), {root_id, parent_id, view_id})
        self.assertNotIn("label-is-not-an-dataset-id", index.metadata_calls)
        self.assertEqual(index.open_calls, [])

    def test_invalid_metadata_record_is_blocked_without_payload_open(self):
        identity = "source"
        spec = dataset_spec()
        mutations = (
            {"state": "complete"},
            {"data_id": "other-id"},
            {"spec": {"not": "DatasetSpec"}},
            {"source_origin": None},
            {"execution": None},
        )
        for update in mutations:
            with self.subTest(update=update):
                record = metadata_record(identity, spec)
                record.update(update)
                index = _MetadataIndex({identity: record})
                result = apply_source_policy(
                    planned(("field", identity, spec)),
                    {"field": request(identity, spec)}, index,
                    source_policy="require_known")
                self.assertEqual(result["action"], "blocked")
                self.assertEqual(result["source_status"], "unknown")
                self.assertTrue(result["source_reasons"])
                self.assertEqual(index.open_calls, [])

    def test_shared_terminal_is_visited_once_and_existing_block_is_preserved(self):
        terminal_id, left_id, right_id = "shared-terminal", "left", "right"
        terminal_spec = dataset_spec()
        left_spec = dataset_spec(parents=(terminal_id,))
        right_spec = dataset_spec(parents=(terminal_id,))
        index = _MetadataIndex({
            terminal_id: metadata_record(terminal_id, terminal_spec),
            left_id: metadata_record(left_id, left_spec, execution_schema=1,
                                     source_origin=unknown_origin()),
            right_id: metadata_record(right_id, right_spec, execution_schema=1,
                                      source_origin=unknown_origin()),
        })
        requests = {"left": request(left_id, left_spec),
                    "right": request(right_id, right_spec)}
        preview = planned(("left", left_id, left_spec),
                          ("right", right_id, right_spec))
        result = apply_source_policy(preview, requests, index,
                                     source_policy="require_known")
        self.assertEqual(result["source_status"], "known")
        self.assertEqual(index.metadata_calls.count(terminal_id), 1)
        self.assertEqual(index.open_calls, [])

        blocked_preview = planned(("left", left_id, left_spec), action="blocked",
                                   reasons=("existing plan block",))
        blocked = apply_source_policy(blocked_preview, {"left": requests["left"]},
                                      index, source_policy="require_known")
        self.assertEqual(blocked["action"], "blocked")
        self.assertIn("existing plan block", blocked["reasons"])


if __name__ == "__main__":
    unittest.main()
