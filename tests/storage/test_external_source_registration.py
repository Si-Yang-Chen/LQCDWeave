"""Module tests split from test_project_source_provenance.py; shared numerical fixtures stay unchanged."""

from test_project_source_provenance import (
    _ExternalSourceRegistrationFixture,
    dataset_storage,
    canonical_json,
    import_provenance,
    json,
    np,
    register_binary,
    register_npy,
    register_npy_timeslices,
)
from pathlib import Path
from lqcdweave.contracts.dataset import stable_digest
from lqcdweave.contracts.source import source_origin_from_registration
from lqcdweave.storage.dataset import DatasetReader, read_dataset_metadata
from lqcdweave.storage.registration import (
    read_registration, registration_lock, write_registration,
)


class ExternalSourceRegistrationTests(_ExternalSourceRegistrationFixture):
    def test_source_identity_and_registration_runtime_are_separate(self):
        first = register_npy(self.root / "first", self.source, self.spec,
                             provenance=import_provenance(runtime={"python": "3.12"}),
                             provenance_mode="external_import")
        second = register_npy(self.root / "second", self.source, self.spec,
                              provenance=import_provenance(runtime={"python": "3.13"}),
                              provenance_mode="external_import")
        self.assertNotEqual(first, second)
        first_record = read_registration(self.root / "first")["cfg-1"]
        second_record = read_registration(self.root / "second")["cfg-1"]
        self.assertEqual(stable_digest(first_record["identity"]), first)
        self.assertEqual(first_record["identity"]["state"], "complete")
        self.assertEqual(first_record["identity"]["spec"], second_record["identity"]["spec"])
        self.assertEqual(first_record["identity"]["provenance"],
                         second_record["identity"]["provenance"])
        self.assertNotEqual(first_record["identity"]["realization_id"],
                            second_record["identity"]["realization_id"])
        self.assertEqual(first_record["execution"]["schema"], 3)
        self.assertEqual(second_record["execution"]["schema"], 3)
        self.assertEqual(first_record["execution"]["registration_event"]["runtime"],
                         {"status": "known", "value": {"python": "3.12"}})
        self.assertEqual(second_record["execution"]["registration_event"]["runtime"],
                         {"status": "known", "value": {"python": "3.13"}})
        identity_provenance = first_record["identity"]["provenance"]
        self.assertEqual(set(identity_provenance), {"schema", "kind", "origin"})
        self.assertNotIn("registration_runtime", identity_provenance)
        with DatasetReader(self.root / "first") as reader:
            self.assertEqual(reader.data_id, first)
            self.assertEqual(reader.source_origin["status"], "recorded")
            self.assertTrue(all(value["status"] == "known"
                                for value in reader.source_origin["origin"].values()))
            self.assertEqual(reader.registration_runtime,
                             {"status": "known", "value": {"python": "3.12"}})
            np.testing.assert_array_equal(reader.read(time=1), self.values[1])
        changed = import_provenance(revision="revision-2", runtime={"python": "3.12"})
        changed_id = register_npy(self.root / "changed-source", self.source, self.spec,
                                  provenance=changed, provenance_mode="external_import")
        self.assertNotEqual(first, changed_id)

    def test_all_external_registration_formats_use_event_schema_three(self):
        provenance = import_provenance()
        npy_id = register_npy(self.root / "npy", self.source, self.spec,
                              provenance=provenance, provenance_mode="external_import")
        binary_path = self.root / "input.bin"
        self.values.tofile(binary_path)
        binary_id = register_binary(self.root / "binary", binary_path, self.spec,
                                    provenance=provenance, storage_dtype="<f8",
                                    provenance_mode="external_import")
        slices = []
        for time, values in enumerate(self.values):
            path = self.root / f"time-{time}.npy"
            np.save(path, values, allow_pickle=False)
            slices.append(path)
        timeslice_id = register_npy_timeslices(self.root / "timeslices", slices,
                                               self.spec, provenance=provenance, provenance_mode="external_import")
        for name, identity, operation in (
                ("npy", npy_id, "register_npy"),
                ("binary", binary_id, "register_binary"),
                ("timeslices", timeslice_id, "register_npy_timeslices")):
            with self.subTest(operation=operation), DatasetReader(self.root / name) as reader:
                self.assertEqual(reader.execution["schema"], 3)
                self.assertEqual(reader.execution["registration_event"]["operation"], operation)
                self.assertEqual(reader.data_id, identity)

    def test_public_registration_formats_preserve_external_tree_and_locators(self):
        source_root = self.root / "archive drop 17" / "do not normalize this tree"
        source_root.mkdir(parents=True)
        nested = source_root / "odd subdirectory [A]"
        nested.mkdir()
        notes = source_root / "operator notes, not data.txt"
        notes.write_text("retain this unrelated file verbatim\n", encoding="utf-8")

        npy_source = nested / "field run-7 (original).array.npy"
        np.save(npy_source, self.values, allow_pickle=False)
        npy_alias = nested / "same payload through another name.npy"
        np.save(npy_alias, self.values, allow_pickle=False)
        binary_source = nested / "legacy field [raw bytes]"
        with binary_source.open("wb") as stream:
            self.values.astype("<f8", copy=False).tofile(stream)
        timeslice_sources = []
        for coordinate, values in enumerate(self.values):
            path = source_root / f"scan coordinate {coordinate} (original).npy"
            np.save(path, values, allow_pickle=False)
            timeslice_sources.append(path)

        def tree_snapshot():
            snapshot = {}
            for path in sorted(source_root.rglob("*")):
                relative = path.relative_to(source_root).as_posix()
                stat = path.stat()
                if path.is_dir():
                    snapshot[relative] = ("directory", stat.st_mtime_ns)
                else:
                    snapshot[relative] = (
                        "file", stat.st_size, stat.st_mtime_ns, path.read_bytes())
            return snapshot

        before = tree_snapshot()
        provenance = import_provenance(
            revision="preserve-external-tree-v1",
            dataset_id="archive/drop-17/unusual-layout")
        registrations = (
            ("npy", register_npy(self.root / "registered npy", npy_source,
                self.spec, provenance=provenance,
                provenance_mode="external_import"),
             self.root / "registered npy", [npy_source]),
            ("binary", register_binary(self.root / "registered binary",
                binary_source, self.spec, provenance=provenance,
                storage_dtype="<f8", provenance_mode="external_import"),
             self.root / "registered binary", [binary_source]),
            ("timeslices", register_npy_timeslices(
                self.root / "registered time slices", timeslice_sources,
                self.spec, provenance=provenance,
                provenance_mode="external_import"),
             self.root / "registered time slices", timeslice_sources),
            ("npy-alias", register_npy(self.root / "registered npy alias",
                npy_alias, self.spec, provenance=provenance,
                provenance_mode="external_import"),
             self.root / "registered npy alias", [npy_alias]),
        )

        # Each registered instance receives its own realization ID even when
        # source bytes and scientific inputs match another import.
        self.assertNotEqual(registrations[0][1], registrations[3][1])
        self.assertEqual(before, tree_snapshot())
        for name, identity, destination, source_files in registrations:
            with self.subTest(format=name):
                record = read_registration(destination)["cfg-1"]
                self.assertEqual(record["data_id"], identity)
                self.assertEqual(stable_digest(record["identity"]), identity)
                self.assertEqual(record["identity"]["spec"], self.spec.document)
                self.assertEqual(record["identity"]["state"], "complete")
                self.assertEqual(set(record["identity"]),
                                 {"schema", "spec", "provenance", "realization_id", "state"})
                self.assertNotIn("registration_runtime", record["identity"]["provenance"])
                self.assertNotIn("blocks", record["identity"])
                storage = record["storage"]
                self.assertEqual(storage["format"], "binary" if name == "binary" else "npy")
                metadata = read_dataset_metadata(destination, cfg="cfg-1")
                self.assertEqual(metadata["data_id"], identity)
                self.assertEqual(metadata["storage"], storage)
                if name == "timeslices":
                    self.assertEqual(storage["path"], "cfg-1.npy")
                    self.assertTrue((destination / storage["path"]).is_file())
                    self.assertEqual(record["execution"]["schema"], 3)
                    self.assertEqual(record["execution"]["registration_event"]["operation"],
                                     "register_npy_timeslices")
                else:
                    self.assertEqual(Path(storage["path"]).resolve(), source_files[0].resolve())

                with DatasetReader(destination) as reader:
                    self.assertEqual(reader.data_id, identity)
                    self.assertEqual(reader.spec.definition_id,
                                     self.spec.definition_id)
                    for time, expected in enumerate(self.values):
                        np.testing.assert_array_equal(
                            reader.read(time=time), expected)

        # Registration and subsequent reads neither rename/remove
        # entries nor rewrite source files; the adjacent note is retained too.
        self.assertEqual(before, tree_snapshot())

    def test_schema_three_event_has_one_shared_strict_validator(self):
        registered = self.root / "valid-registration"
        register_npy(registered, self.source, self.spec,
            provenance=import_provenance(), provenance_mode="external_import")
        baseline = read_registration(registered)["cfg-1"]
        malformed = (
            ("missing execution field", lambda execution: execution.pop("data_id")),
            ("extra execution field", lambda execution: execution.update(extra="no")),
            ("missing event field", lambda execution: execution["registration_event"].pop("runtime")),
            ("extra event field", lambda execution: execution["registration_event"].update(extra="no")),
            ("wrong actual ID", lambda execution: execution.update(data_id="wrong-actual-id")),
            ("wrong definition ID", lambda execution: execution.update(definition_id="wrong-definition-id")),
            ("boolean schema", lambda execution: execution.update(schema=True)),
            ("unsupported schema", lambda execution: execution.update(schema=4)),
            ("claimed science validation", lambda execution: execution.update(
                scientific_validation={"status": "passed", "method": "unverified"})),
            ("date-only timestamp", lambda execution: execution["registration_event"].update(
                started_at="2026-10-02Z", completed_at="2026-10-01Z")),
            ("backwards timestamps", lambda execution: execution["registration_event"].update(
                started_at="2026-10-02T00:00:01Z", completed_at="2026-10-02T00:00:00Z")),
            ("non-UTC timestamp", lambda execution: execution["registration_event"].update(
                started_at="2026-10-02T00:00:00+01:00Z")),
        )
        for name, mutate in malformed:
            with self.subTest(case=name):
                record = json.loads(canonical_json(baseline))
                execution = record["execution"]
                mutate(execution)
                actual_id = record["data_id"]
                provenance = record["identity"]["provenance"]

                with self.assertRaises(ValueError) as storage_error:
                    dataset_storage._validate_execution(execution, self.spec,
                        data_id=actual_id, draft=False, provenance=provenance)
                with self.assertRaises(ValueError) as source_error:
                    source_origin_from_registration(provenance, execution,
                        data_id=actual_id, definition_id=self.spec.definition_id,
                        spec_document=self.spec.document)
                if name not in ("boolean schema", "unsupported schema"):
                    self.assertEqual(str(storage_error.exception), str(source_error.exception))

                tampered = self.root / f"tampered-{name.replace(' ', '-') }"
                with registration_lock(tampered):
                    write_registration(tampered, {"cfg-1": record})
                with self.assertRaisesRegex(ValueError, str(storage_error.exception)):
                    DatasetReader(tampered)

    def test_unstructured_external_metadata_does_not_claim_known_origin(self):
        identity = register_npy(
            self.root / "unstructured-source", self.source, self.spec,
            provenance={"producer": "unstructured-import"},
        )
        record = read_registration(self.root / "unstructured-source")["cfg-1"]
        self.assertEqual(record["data_id"], identity)
        self.assertEqual(record["identity"]["provenance"], {"producer": "unstructured-import"})
        self.assertIsNone(record["execution"])
        with DatasetReader(self.root / "unstructured-source") as reader:
            self.assertEqual(reader.source_origin["status"], "unknown")

    def test_malformed_versioned_source_is_rejected_before_directory_creation(self):
        malformed = import_provenance(runtime={})
        destination = self.root / "invalid-empty-runtime"
        with self.assertRaisesRegex(ValueError, "must be nonempty"):
            register_npy(destination, self.source, self.spec, provenance=malformed,
                          provenance_mode="external_import")
        self.assertFalse(destination.exists())

        malformed = import_provenance()
        malformed["origin"]["revision"] = {"status": "known", "value": None}
        destination = self.root / "invalid-source"
        with self.assertRaisesRegex(ValueError, "nonempty string"):
            register_npy(destination, self.source, self.spec, provenance=malformed,
                          provenance_mode="external_import")
        self.assertFalse(destination.exists())
        malformed = import_provenance()
        malformed["schema"] = True
        with self.assertRaisesRegex(ValueError, "integer 1"):
            register_npy(destination, self.source, self.spec, provenance=malformed,
                          provenance_mode="external_import")
        self.assertFalse(destination.exists())
