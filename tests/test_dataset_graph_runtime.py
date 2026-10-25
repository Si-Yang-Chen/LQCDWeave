"""Real dependency publication, coverage reuse, and interrupted execution."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import uuid
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec, stable_digest
from lqcdweave.planning.dataset import plan_dataset_graph
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetWriter, inspect_dataset, read_dataset_metadata,
    read_json_records,
)
from lqcdweave.storage.registration import read_registration
from lqcdweave.runtime.datasets import (
    DatasetView, read_dataset_request_receipt, run_dataset_requests,
    save_dataset_request_receipt,
)
from lqcdweave.runtime import datasets as workflow_datasets


def basis_spec(times=(1, 0)):
    return DatasetSpec(dict(schema=1, kind="fixture_basis", configuration=dict(
        ensemble="test", configuration="cfg-7", lattice=[1, 1, 1, 2]),
        axes=[dict(name="time", coordinates=list(times)),
              dict(name="mode", coordinates=[4, 1, 7])], dtype="complex128",
        bases={}, definition=dict(operator="fixture"),
        conventions=dict(normalization="unscaled"), parents=[]))


def requests():
    base = basis_spec()
    child = base.document
    child.update(kind="fixture_projection", parents=[dict(request="basis")],
                 bases=dict(mode=dict(data_id=dict(request="basis"), indices=[4, 1])))
    child["axes"][1]["coordinates"] = [4, 1]
    return dict(basis=DatasetRequest(base, subset_axes=("time",)),
                child=DatasetRequest(child))


def row(time, modes=(4, 1, 7)):
    return np.asarray([10 * time + mode + 1j for mode in modes], dtype=np.complex128)


def registration_snapshot(directory, cfg="cfg-7"):
    record = read_dataset_metadata(directory, cfg=cfg)["record"]
    return dict(identity=record["identity"], data_id=record["data_id"],
                provenance=record["identity"]["provenance"],
                execution=record.get("execution"), fields=record.get("fields"),
                format=record["storage"]["format"])


def cfg_rows(directory, positions, cfg="cfg-7"):
    directory = Path(directory)
    path = directory / f"{cfg}.npy"
    if not path.exists():
        path = directory / ".state" / f"{cfg}.npy"
    payload = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return {position: np.ascontiguousarray(payload[position]).tobytes()
                for position in positions}
    finally:
        payload._mmap.close()


class DatasetGraphRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.requests = requests()
        self.outputs = {name: self.root / name for name in self.requests}
        self.provenance = {name: dict(producer="fixture-v1") for name in self.requests}
        self.calls = []

    def basis(self, spec, inputs, writer):
        self.calls.append(("basis", writer.missing))
        for position in writer.missing:
            writer.write(position, row(spec.coordinates("time")[position]))

    def child(self, spec, inputs, writer):
        self.calls.append(("child", writer.missing))
        for position in writer.missing:
            time = spec.coordinates("time")[position]
            value = inputs["basis"].read(time=time, select={"mode": [4, 1]})
            writer.write(position, 2 * value)

    def run_graph(self, *, resume=False, producers=None, provenance=None):
        return run_dataset_requests(self.requests, self.index, self.outputs,
            self.provenance if provenance is None else provenance,
            dict(basis=self.basis, child=self.child) if producers is None else producers,
            resume=resume)

    def assert_result(self, ids):
        with self.index.open(ids["child"]) as child:
            self.assertEqual(child.spec.document["parents"], [ids["basis"]])
            self.assertEqual(child.spec.document["bases"]["mode"]["data_id"], ids["basis"])
            self.assertEqual(child.spec.coordinates("time"), [1, 0])
            for time in (1, 0):
                np.testing.assert_array_equal(child.read(time=time), 2 * row(time, (4, 1)))

    def test_only_published_parents_bind_and_repeated_graph_does_no_work(self):
        preview = plan_dataset_graph(self.requests, self.index.observations())
        self.assertEqual(preview["stages"]["basis"]["action"], "compute")
        self.assertEqual(preview["stages"]["child"]["action"], "blocked")
        self.assertIsNone(preview["stages"]["child"]["spec"])
        ids = self.run_graph()
        self.assert_result(ids)
        self.assertEqual(self.calls, [("basis", (0, 1)), ("child", (0, 1))])
        observations = {name: inspect_dataset(path)
                        for name, path in self.outputs.items()}
        attempts = {name: item["execution"]["attempts"][-1]
                    for name, item in observations.items()}
        self.assertEqual({attempt["run_id"] for attempt in attempts.values()},
                         {attempts["basis"]["run_id"]})
        for name, attempt in attempts.items():
            self.assertEqual(attempt["origin"], "dataset_graph")
            self.assertEqual(attempt["stage"], name)
            self.assertTrue(attempt["producer_attempt_id"])
            self.assertEqual(attempt["status"], "complete")
        def forbidden(*args):
            raise AssertionError("published stage was recomputed")
        registrations = {name: registration_snapshot(path)
                        for name, path in self.outputs.items()}
        self.assertEqual(self.run_graph(resume=True,
            producers=dict(basis=forbidden, child=forbidden)), ids)
        self.assertEqual({name: registration_snapshot(path)
                          for name, path in self.outputs.items()}, registrations)

    def test_all_reuse_materializes_only_explicit_output_and_keeps_resume_checks(self):
        ids = self.run_graph()
        source = self.outputs["child"]
        source_registration = registration_snapshot(source)
        destination = self.root / "materialized-child"

        def forbidden(*args, **kwargs):
            raise AssertionError("all-reuse materialization called a producer")

        self.calls.clear()
        result = run_dataset_requests(self.requests, self.index,
            {"child": destination}, self.provenance,
            dict(basis=forbidden, child=forbidden), data_ids=ids,
            materialize_reused_outputs=("child",))
        self.assertEqual(result, ids)
        self.assertEqual(self.calls, [])
        self.assertEqual(registration_snapshot(destination), source_registration)
        index_record = json.loads(self.index.path.read_text())
        self.assertEqual(index_record["datasets"][ids["basis"]], [dict(
            directory=str(self.outputs["basis"].resolve()), configuration="cfg-7")])
        self.assertEqual(set(tuple(sorted(location.items()))
                             for location in index_record["datasets"][ids["child"]]), {
            tuple(sorted(dict(directory=str(self.outputs["child"].resolve()),
                              configuration="cfg-7").items())),
            tuple(sorted(dict(directory=str(destination.resolve()),
                              configuration="cfg-7").items()))})
        with self.index.open(ids["child"]) as reader:
            np.testing.assert_array_equal(reader.read(time=1), 2 * row(1, (4, 1)))

        before = registration_snapshot(destination)
        self.assertEqual(run_dataset_requests(self.requests, self.index,
            {"child": destination}, self.provenance,
            dict(basis=forbidden, child=forbidden), resume=True,
            data_ids=ids, materialize_reused_outputs=("child",)), ids)
        self.assertEqual(registration_snapshot(destination), before)
        self.assertEqual(self.calls, [])

    def test_reuse_rejects_partial_explicit_output_without_calling_producer(self):
        ids = self.run_graph()
        observation = inspect_dataset(self.outputs["child"])
        destination = self.root / "partial-materialization-target"
        with DatasetWriter(destination, observation["spec"],
                            provenance=observation["provenance"]) as writer:
            writer.write(0, 2 * row(1, (4, 1)))
        before = {path.name: path.read_bytes() for path in destination.iterdir()
                  if path.is_file()}

        def forbidden(*args, **kwargs):
            raise AssertionError("reuse conflict called a producer")

        with self.assertRaises((ValueError, FileNotFoundError)):
            run_dataset_requests(self.requests, self.index,
                {"child": destination}, self.provenance,
                dict(basis=forbidden, child=forbidden), data_ids=ids,
                materialize_reused_outputs=("child",))
        self.assertEqual({path.name: path.read_bytes() for path in destination.iterdir()
                          if path.is_file()}, before)

    def test_fixed_input_placeholder_stays_input_only(self):
        ids = self.run_graph()
        pinned = DatasetRequest(self.requests["basis"].template,
            data_id=ids["basis"])
        placeholder = self.root / "input-placeholder"
        result = run_dataset_requests({"input": pinned}, self.index,
            {"input": placeholder}, {}, {})
        self.assertEqual(result["input"], ids["basis"])
        self.assertFalse(placeholder.exists())

    def test_reused_materialization_names_are_validated_before_opening_inputs(self):
        ids = self.run_graph()

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid materialization metadata reached input planning")

        with patch.object(self.index, "open", side_effect=forbidden), \
             patch.object(self.index, "observations", side_effect=forbidden):
            for names, outputs, expected in (
                (("child", "child"), {"child": self.root / "target"}, "unique"),
                (("missing",), {"missing": self.root / "target"}, "request and output path"),
                (("child",), {}, "request and output path"),
            ):
                with self.subTest(names=names, outputs=tuple(outputs)):
                    with self.assertRaisesRegex(ValueError, expected):
                        run_dataset_requests(self.requests, self.index, outputs,
                            self.provenance, {}, data_ids=ids,
                            materialize_reused_outputs=names)
            with self.assertRaisesRegex(TypeError, "must be a tuple"):
                run_dataset_requests(self.requests, self.index, {},
                    self.provenance, {}, data_ids=ids,
                    materialize_reused_outputs=["child"])

    def test_complete_partial_coverage_is_copied_without_overwriting_source(self):
        source = self.root / "source"
        with DatasetWriter(source, basis_spec((1,)), provenance=dict(producer="external")) as writer:
            writer.write(0, row(1))
            source_id = writer.publish()
        self.index.register(source)
        original = cfg_rows(source, (0,))
        ids = self.run_graph()
        self.assertNotEqual(ids["basis"], source_id)
        self.assertEqual(self.calls[0], ("basis", (1,)))
        self.assertEqual(cfg_rows(source, (0,)), original)
        with self.index.open(ids["basis"]) as basis:
            self.assertEqual(basis.provenance["coverage_reuse"], dict(
                data_id=source_id, requested_positions=[0]))
            np.testing.assert_array_equal(basis.read(time=1), row(1))
        self.assert_result(ids)

    def test_interruption_resumes_completed_rows_and_rejects_changed_provenance(self):
        def interrupted(spec, inputs, writer):
            writer.write(0, 2 * row(1, (4, 1)))
            raise RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_graph(producers=dict(basis=self.basis, child=interrupted))
        partial = inspect_dataset(self.outputs["child"])
        self.assertEqual(partial["state"], "partial")
        self.assertEqual(partial["latest_attempt_status"], "failed")
        failed_attempt = partial["execution"]["attempts"][-1]
        self.assertEqual(failed_attempt["stage"], "child")
        self.assertTrue(failed_attempt["run_id"])
        self.assertTrue(failed_attempt["producer_attempt_id"])
        self.assertEqual(failed_attempt["status"], "failed")
        self.assertTrue(failed_attempt["failure"]["message"].endswith("interrupted"))
        self.assertEqual(len(self.index.observations()), 1)
        original = cfg_rows(self.outputs["child"], partial["completed"])
        changed = dict(self.provenance, child=dict(producer="changed"))
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.run_graph(resume=True, provenance=changed)
        self.calls.clear()
        ids = self.run_graph(resume=True)
        self.assertEqual(self.calls, [("child", (1,))])
        self.assertEqual(cfg_rows(self.outputs["child"], (0,)), original)
        completed = inspect_dataset(self.outputs["child"])
        self.assertEqual([item["status"] for item in completed["execution"]["attempts"]],
                         ["failed", "complete"])
        self.assertNotEqual(completed["execution"]["attempts"][0]["run_id"],
                            completed["execution"]["attempts"][1]["run_id"])
        self.assert_result(ids)

    def test_registration_failure_reuses_published_output_without_recomputation(self):
        register = self.index.register
        def fail_child(directory, *, cfg=None):
            if Path(directory) == self.outputs["child"]:
                raise RuntimeError("registration interrupted")
            return register(directory, cfg=cfg)
        with patch.object(self.index, "register", side_effect=fail_child):
            with self.assertRaisesRegex(RuntimeError, "registration interrupted"):
                self.run_graph()
        self.assertEqual(inspect_dataset(self.outputs["child"])["state"], "complete")
        original = registration_snapshot(self.outputs["child"])
        before_retry = inspect_dataset(self.outputs["child"])["execution"]["attempts"]
        self.calls.clear()
        ids = self.run_graph(resume=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(registration_snapshot(self.outputs["child"]), original)
        self.assertEqual(inspect_dataset(self.outputs["child"])["execution"]["attempts"],
                         before_retry)
        self.assert_result(ids)

    def test_producer_return_value_cannot_publish_missing_positions(self):
        with self.assertRaisesRegex(ValueError, "complete active production"):
            self.run_graph(producers=dict(basis=lambda *args: "success", child=self.child))
        self.assertEqual(self.index.observations(), ())
        self.assertEqual(read_registration(self.outputs["basis"]), {})
        self.assertEqual(inspect_dataset(self.outputs["basis"], cfg="cfg-7")["state"],
                         "partial")
        self.assertFalse(self.outputs["child"].exists())

    def test_failed_scientific_stage_is_blocked_before_opening_inputs_or_calling_producer(self):
        validation = dict(status="failed", method="independent-fixture-v1",
                          criteria={"observable": "child"},
                          checked_coverage={"time": [1]},
                          evidence={"reference": "fixture"})

        def reject_child(spec, inputs, writer):
            writer.write(0, 2 * row(1, (4, 1)))
            writer.record_scientific_validation(validation)

        with self.assertRaisesRegex(ValueError, "scientific validation failed"):
            self.run_graph(producers=dict(basis=self.basis, child=reject_child))
        saved = inspect_dataset(self.outputs["child"])
        self.assertEqual(saved["latest_attempt_status"], "failed")
        self.assertEqual(saved["state"], "partial")
        self.assertEqual(saved["execution"]["scientific_validation"]["status"], "failed")
        failed_attempt = saved["execution"]["attempts"][-1]
        self.assertEqual(failed_attempt["status"], "failed")
        original_rows = cfg_rows(self.outputs["child"], saved["completed"])
        calls = []

        def forbidden(*args):
            calls.append(args)
            raise AssertionError("failed scientific stage reached its producer")

        with patch.object(workflow_datasets, "open_dataset_view",
                          side_effect=AssertionError("failed stage opened an input view")):
            with self.assertRaisesRegex(ValueError, "blocked"):
                self.run_graph(resume=True,
                               producers=dict(basis=forbidden, child=forbidden))
        self.assertEqual(calls, [])
        after = inspect_dataset(self.outputs["child"])
        self.assertEqual(after["execution"]["scientific_validation"]["status"], "failed")
        self.assertEqual(cfg_rows(self.outputs["child"], saved["completed"]), original_rows)

    def test_fixed_unknown_identity_fails_before_output_creation(self):
        self.requests["basis"] = DatasetRequest(basis_spec(), data_id="missing-actual")
        with self.assertRaisesRegex(ValueError, "dataset"):
            self.run_graph()
        self.assertTrue(all(not path.exists() for path in self.outputs.values()))

    def check_resume_with_new_candidate(self, *, initial_source):
        document = basis_spec((3, 1, 0, 2)).document
        document["configuration"]["lattice"][3] = 4
        full = DatasetSpec(document)
        self.requests = dict(basis=DatasetRequest(full, subset_axes=("time",)))
        self.outputs = dict(basis=self.root / "basis")
        self.provenance = dict(basis=dict(producer="fixture-v1"))
        def publish_source(name, times):
            source_document = full.document
            source_document["axes"][0]["coordinates"] = list(times)
            directory = self.root / name
            with DatasetWriter(directory, DatasetSpec(source_document),
                                provenance=dict(producer=name)) as writer:
                for position, time in enumerate(times):
                    writer.write(position, row(time))
                identity = writer.publish()
            self.index.register(directory)
            return identity
        original_id = publish_source("original-source", (3,)) if initial_source else None
        def interrupted(spec, inputs, writer):
            position = writer.missing[0]
            writer.write(position, row(spec.coordinates("time")[position]))
            raise RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_graph(producers=dict(basis=interrupted))
        interrupted = inspect_dataset(self.outputs["basis"], cfg="cfg-7")
        before = cfg_rows(self.outputs["basis"], interrupted["completed"])
        publish_source("new-source", (3, 1, 0))
        self.calls.clear()
        ids = self.run_graph(resume=True, producers=dict(basis=self.basis))
        self.assertEqual(self.calls, [("basis", (2, 3) if initial_source else (1, 2, 3))])
        self.assertEqual(cfg_rows(self.outputs["basis"], interrupted["completed"]), before)
        with self.index.open(ids["basis"]) as basis:
            if initial_source:
                self.assertEqual(basis.provenance["coverage_reuse"], dict(
                    data_id=original_id, requested_positions=[0]))
            else:
                self.assertNotIn("coverage_reuse", basis.provenance)
            for time in (3, 1, 0, 2):
                np.testing.assert_array_equal(basis.read(time=time), row(time))

    def test_new_candidate_does_not_change_existing_partial_coverage_origin(self):
        self.check_resume_with_new_candidate(initial_source=True)

    def test_new_candidate_does_not_add_a_coverage_origin_to_existing_partial(self):
        self.check_resume_with_new_candidate(initial_source=False)

    def test_new_actual_parent_does_not_replace_interrupted_child_binding(self):
        def interrupted(spec, inputs, writer):
            writer.write(0, 2 * inputs["basis"].read(time=1, select={"mode": [4, 1]}))
            raise RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_graph(producers=dict(basis=self.basis, child=interrupted))
        prior = inspect_dataset(self.outputs["basis"])["data_id"]
        old_row = cfg_rows(self.outputs["child"], (0,))
        # Register an alternate realization that the default deterministic
        # policy prefers. Same generation definition is not actual identity.
        spec = basis_spec()
        provenance = dict(variant=0)
        for nonce in range(1, 100_000):
            realization = uuid.UUID(int=nonce)
            candidate = stable_digest(dict(schema=1, spec=spec.document,
                provenance=provenance, realization_id=realization.hex,
                state="complete"))
            if candidate < prior:
                break
        else:
            self.fail("fixture did not derive a preferred alternate identity")
        directory = self.root / "alternate-preferred"
        uuid_calls = 0
        def controlled_uuid4():
            nonlocal uuid_calls
            uuid_calls += 1
            return realization if uuid_calls == 1 else uuid.UUID(int=100_000 + uuid_calls)
        with patch("lqcdweave.storage.dataset.uuid.uuid4",
                   side_effect=controlled_uuid4):
            with DatasetWriter(directory, spec, provenance=provenance) as writer:
                for position, time in enumerate((1, 0)):
                    writer.write(position, row(time) + 2)
                self.assertEqual(writer.publish(), candidate)
        self.assertEqual(read_dataset_metadata(directory, cfg="cfg-7")["data_id"], candidate)
        self.index.register(directory)
        preview = plan_dataset_graph(self.requests, self.index.observations())
        self.assertEqual(preview["bindings"]["basis"], candidate)
        self.calls.clear()
        ids = self.run_graph(resume=True)
        self.assertEqual(ids["basis"], prior)
        self.assertEqual(self.calls, [("child", (1,))])
        self.assertEqual(cfg_rows(self.outputs["child"], (0,)), old_row)
        self.assert_result(ids)

    def seed_interrupted_child(self):
        def interrupted(spec, inputs, writer):
            writer.write(0, 2 * row(1, (4, 1)))
            raise RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_graph(producers=dict(basis=self.basis, child=interrupted))
        return inspect_dataset(self.outputs["basis"])["data_id"]

    def publish_alternate(self):
        directory = self.root / "other-basis"
        with DatasetWriter(directory, basis_spec(), provenance=dict(producer="alternate")) as writer:
            for position, time in enumerate((1, 0)):
                writer.write(position, row(time) + 2)
            identity = writer.publish()
        self.index.register(directory)
        return identity

    def test_explicit_pin_conflict_cannot_change_saved_parent(self):
        self.seed_interrupted_child()
        other = self.publish_alternate()
        original = cfg_rows(self.outputs["child"], (0,))
        with self.assertRaisesRegex(ValueError, "binding differs"):
            run_dataset_requests(self.requests, self.index, self.outputs, self.provenance,
                dict(basis=self.basis, child=self.child), resume=True,
                data_ids=dict(basis=other))
        self.assertEqual(cfg_rows(self.outputs["child"], (0,)), original)

    def test_unregistered_original_parent_cannot_be_replaced_during_resume(self):
        prior = self.seed_interrupted_child()
        self.publish_alternate()
        self.index.unregister(prior)
        with self.assertRaises(ValueError):
            self.run_graph(resume=True)
        partial = inspect_dataset(self.outputs["child"], cfg="cfg-7")
        self.assertEqual(partial["state"], "partial")
        self.assertEqual(partial["spec"].document["parents"], [prior])
        self.assertEqual(partial["spec"].document["bases"]["mode"]["data_id"], prior)

    def test_conflicting_sibling_partials_fail_before_any_resume_production(self):
        self.seed_interrupted_child()
        other = self.publish_alternate()
        self.requests["sibling"] = self.requests["child"]
        self.outputs["sibling"] = self.root / "sibling"
        self.provenance["sibling"] = dict(producer="fixture-v1")
        spec = self.requests["sibling"].resolve(dict(basis=other))
        with DatasetWriter(self.outputs["sibling"], spec,
                            provenance=self.provenance["sibling"]) as writer:
            writer.write(0, 2 * row(1, (4, 1)) + 4)
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "binding differs"):
            self.run_graph(resume=True)
        self.assertEqual(self.calls, [])

    def test_direct_view_construction_cannot_relabel_scientific_identity(self):
        ids = self.run_graph()
        with self.index.open(ids["basis"]) as reader:
            changed = reader.spec.document
            changed["definition"]["operator"] = "different-science"
            with self.assertRaisesRegex(ValueError, "scientific dataset view"):
                DatasetView(reader, DatasetSpec(changed))
            np.testing.assert_array_equal(reader.read(time=1), row(1))
            view = DatasetView(reader, reader.spec)
            np.testing.assert_array_equal(view.read(time=0), row(0))

    def test_receipts_are_immutable_records_and_new_publications_get_new_paths(self):
        ids = self.run_graph()
        path = save_dataset_request_receipt(self.root, self.requests, self.index, ids)
        original = path.read_bytes()
        records = list(read_json_records(path))
        self.assertEqual([item["node"] for item in records], ["basis", "child"])
        self.assertEqual(records[1]["spec"]["parents"], [ids["basis"]])
        self.assertEqual(records[1]["request"]["parents"], [dict(request="basis")])
        second = save_dataset_request_receipt(self.root, self.requests, self.index, ids)
        self.assertNotEqual(second, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(read_json_records(second)), records)
        path.write_bytes(original[:-12])
        corrupted = path.read_bytes()
        third = save_dataset_request_receipt(self.root, self.requests, self.index, ids)
        self.assertNotIn(third, (path, second))
        self.assertEqual(path.read_bytes(), corrupted)

    def test_receipt_reader_rejects_structurally_truncated_record_stream(self):
        ids = self.run_graph()
        path = save_dataset_request_receipt(self.root, self.requests, self.index, ids)
        path.write_bytes(path.read_bytes()[:-12])
        with self.assertRaises((EOFError, ValueError, OSError)):
            read_dataset_request_receipt(path, self.index)
