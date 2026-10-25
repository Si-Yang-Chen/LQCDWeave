"""Explicit sequential dataset graphs reuse only verified project receipts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import DatasetRequest, DatasetSpec
from lqcdweave.storage.dataset import (
    DatasetIndex, DatasetWriter, inspect_dataset, read_dataset_metadata,
    read_json_records,
)
from lqcdweave.runtime.datasets import read_dataset_request_receipt
from lqcdweave.runtime.project_datasets import run_project_dataset_sequence


def _spec(kind, times=(0, 1, 2, 3), *, parents=(), definition=None):
    return DatasetSpec(dict(
        schema=1,
        kind=kind,
        configuration=dict(ensemble="sequence-test", configuration="cfg-1",
                           lattice=[1, 1, 1, len(times)]),
        axes=[dict(name="time", coordinates=list(times)),
              dict(name="mode", coordinates=[0, 1])],
        dtype="complex128",
        bases={},
        definition=(dict(stage=kind) if definition is None else definition),
        conventions=dict(normalization="unscaled"),
        parents=list(parents),
    ))


def _source_row(time):
    return np.asarray([time + 1 + 2j * time, 3 * time - 1j],
                      dtype=np.complex128)


def _cfg_rows(directory, positions, cfg="cfg-1"):
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


class ProjectDatasetSequenceTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = DatasetIndex(self.root / "index")
        self.project_id = "explicit-graph-sequence"

    def _step(self, requests, label, producers, *, outputs=None,
              provenance=None, receipt_directory=None, data_ids=None):
        root = self.root / label
        return dict(
            requests=requests,
            outputs=(dict(outputs) if outputs is not None else
                     {name: root / name for name in producers}),
            provenance=(dict(provenance) if provenance is not None else
                        {name: {"producer": f"sequence-test/{label}"}
                         for name in producers}),
            producers=dict(producers),
            receipt_directory=(root if receipt_directory is None else receipt_directory),
            **({} if data_ids is None else {"data_ids": dict(data_ids)}),
        )

    def _source_step(self, label="source", *, times=(0, 1, 2, 3), calls=None):
        spec = _spec("sequence_source", times)
        request = DatasetRequest(spec, subset_axes=("time",))

        def produce(actual_spec, inputs, writer):
            if calls is not None:
                calls.append(tuple(writer.missing))
            for position in writer.missing:
                time = actual_spec.coordinates("time")[position]
                writer.write(position, _source_row(time))

        return spec, lambda: self._step({"source": request}, label,
                                         {"source": produce})

    def _projection_step(self, label="projection", *, times=(0, 1, 2, 3),
                         calls=None, interrupt=None):
        source_spec = _spec("sequence_source", times)
        child_document = _spec("sequence_projection", times).document
        child_document["parents"] = [{"request": "input"}]
        child_document["definition"] = dict(
            operation="scale-and-shift",
            input_roles={"source": {"request": "input"}},
        )
        input_request = DatasetRequest(source_spec, subset_axes=("time",))
        projection_request = DatasetRequest(child_document)

        def project(actual_spec, inputs, writer):
            for position in writer.missing:
                if calls is not None:
                    calls.append(position)
                if interrupt is not None and interrupt["enabled"] and position == 2:
                    interrupt["enabled"] = False
                    raise RuntimeError("projection step interrupted")
                time = actual_spec.coordinates("time")[position]
                writer.write(position, 2 * inputs["input"].read(time=time) + (1 - 3j))

        requests = {"input": input_request, "projection": projection_request}
        root = self.root / label
        return lambda: dict(
            requests=requests,
            # The auto-reused input deliberately has no output, provenance,
            # or producer, as the single-graph public API permits.
            outputs={"projection": root / "projection"},
            provenance={"projection": {"producer": "sequence-test/projection"}},
            producers={"projection": project},
            receipt_directory=root,
        )

    def _tail_step(self, label="tail", calls=None):
        spec = _spec("sequence_tail", times=(0, 1))
        request = DatasetRequest(spec, subset_axes=("time",))

        def produce(actual_spec, inputs, writer):
            if calls is not None:
                calls.append(tuple(writer.missing))
            for position in writer.missing:
                time = actual_spec.coordinates("time")[position]
                writer.write(position, np.asarray([time + 9j, 2 - time], dtype=np.complex128))

        return lambda: self._step({"tail": request}, label, {"tail": produce})

    def _run(self, builders, *, receipts=(), resume=False,
             expected_project_id=None):
        return run_project_dataset_sequence(
            self.index, tuple(builders),
            expected_project_id=(self.project_id if expected_project_id is None
                                 else expected_project_id),
            initial_receipts=tuple(receipts), resume=resume)

    def _remove_worktree_tilde_path(self, relative_path):
        literal = Path.cwd() / relative_path
        tilde_root = (Path.cwd() / "~").resolve()
        resolved = literal.resolve(strict=False)
        if not resolved.is_relative_to(tilde_root):
            raise AssertionError("test cleanup path escaped the worktree tilde directory")
        if literal.is_dir():
            shutil.rmtree(literal)
        elif literal.exists():
            literal.unlink()

    def test_tilde_receipt_directory_is_expanded_for_write_and_read(self):
        home = self.root / "isolated-home"
        home.mkdir()
        token = f"sequence-{self.root.name}"
        receipt_directory = Path("~") / token
        literal_directory = Path.cwd() / receipt_directory
        self.addCleanup(self._remove_worktree_tilde_path, receipt_directory)

        with patch.dict(os.environ, {"USERPROFILE": str(home), "HOME": str(home)}):
            self.assertEqual(receipt_directory.expanduser(), home / token)
            _, builder = self._source_step("tilde-receipt-write")
            step = builder()
            step["receipt_directory"] = receipt_directory
            result, = self._run((lambda: step,))

            receipt_path = Path(result["receipt"])
            self.assertTrue(receipt_path.is_relative_to(home.resolve()))
            self.assertTrue(receipt_path.is_file())
            verified = read_dataset_request_receipt(
                receipt_path, self.index, expected_project_id=self.project_id)
            self.assertEqual(verified["project_id"], self.project_id)
            self.assertEqual(verified["records"][0]["data_id"],
                             result["datasets"]["source"])
            self.assertFalse(literal_directory.exists())

    def test_tilde_initial_receipt_path_conflict_rejects_before_writer(self):
        home = self.root / "isolated-home"
        home.mkdir()
        seed_directory = home / "seed-receipts"
        seed_directory.mkdir()
        with patch.dict(os.environ, {"USERPROFILE": str(home), "HOME": str(home)}):
            _, seed_builder = self._source_step("home-conflict-seed")
            seed_step = seed_builder()
            seed_step["receipt_directory"] = seed_directory
            seed, = self._run((lambda: seed_step,))
            seed_receipt = Path(seed["receipt"]).resolve()
            relative_receipt = (Path("seed-receipts") / seed_receipt.name).as_posix()
            tilde_receipt = Path("~") / relative_receipt
            literal_receipt = Path.cwd() / tilde_receipt
            self.addCleanup(self._remove_worktree_tilde_path, tilde_receipt)

            spec = _spec("different-sequence-conflict", times=(0, 1))
            request = DatasetRequest(spec, subset_axes=("time",))
            producer_calls = []

            def produce(actual_spec, inputs, writer):
                producer_calls.append(tuple(writer.missing))
                for position in writer.missing:
                    writer.write(position, _source_row(actual_spec.coordinates("time")[position]))

            step = self._step(
                {"new": request}, "home-conflict-output", {"new": produce},
                receipt_directory=tilde_receipt)
            output = step["outputs"]["new"]
            seed_bytes = seed_receipt.read_bytes()
            with patch("lqcdweave.storage.dataset.DatasetWriter.__enter__",
                       side_effect=AssertionError("writer opened before receipt path conflict")):
                with self.assertRaisesRegex(ValueError, "initial receipt file path"):
                    self._run(
                        (lambda: step,),
                        receipts=(tilde_receipt,))

            self.assertEqual(producer_calls, [])
            self.assertFalse(output.exists())
            self.assertFalse(literal_receipt.exists())
            self.assertEqual(seed_receipt.read_bytes(), seed_bytes)

    def test_two_producer_graphs_use_actual_receipts_and_ignore_unlisted_decoy(self):
        source_spec, source_builder = self._source_step("sequence-first")
        decoy_path = self.root / "unlisted-decoy"
        with DatasetWriter(decoy_path, source_spec,
                            provenance={"producer": "unlisted-decoy"}) as writer:
            for position, time in enumerate(source_spec.coordinates("time")):
                writer.write(position, np.full(2, 500 + 20j, dtype=np.complex128))
            writer.publish()
        decoy_id = self.index.register(decoy_path)
        projection_builder = self._projection_step("sequence-second")
        source_step = source_builder()
        projection_step = projection_builder()
        shared_receipt_directory = self.root / "shared-sequence-receipts"
        source_step["receipt_directory"] = shared_receipt_directory
        projection_step["receipt_directory"] = shared_receipt_directory

        with patch.object(self.index, "observations",
                          side_effect=AssertionError("sequence scanned the global index")):
            source_result, projection_result = self._run(
                (lambda: source_step, lambda: projection_step))

        source_id = source_result["datasets"]["source"]
        projected_id = projection_result["datasets"]["projection"]
        self.assertNotEqual(source_id, decoy_id)
        self.assertEqual(read_dataset_request_receipt(
            source_result["receipt"], self.index,
            expected_project_id=self.project_id)["project_id"], self.project_id)
        self.assertEqual(read_dataset_request_receipt(
            projection_result["receipt"], self.index,
            expected_project_id=self.project_id)["project_id"], self.project_id)
        with self.index.open(projected_id) as reader:
            self.assertEqual(reader.spec.document["parents"], [source_id])
            for time in range(4):
                np.testing.assert_array_equal(
                    reader.read(time=time), 2 * _source_row(time) + (1 - 3j))
        self.assertFalse((self.root / "sequence-second" / "input").exists())
        self.assertEqual(len(tuple(shared_receipt_directory.glob("graph-*.jsonl.gz"))), 2)

    def test_explicit_pin_does_not_expand_automatic_candidates(self):
        spec = _spec("sequence_pin", times=(0, 1))
        external_path = self.root / "external-pin"
        with DatasetWriter(external_path, spec,
                            provenance={"producer": "external-pin"}) as writer:
            writer.write(0, np.asarray([2 + 3j, 4 + 1j], dtype=np.complex128))
            writer.write(1, np.asarray([5 + 2j, 7 - 1j], dtype=np.complex128))
            writer.publish()
        external_id = self.index.register(external_path)
        request = DatasetRequest(spec, subset_axes=("time",))
        calls = []

        def produce(actual_spec, inputs, writer):
            calls.append(tuple(writer.missing))
            for position in writer.missing:
                writer.write(position, np.asarray([30 + position, 40j], dtype=np.complex128))

        requests = {
            "pinned": DatasetRequest(spec, subset_axes=("time",),
                                      data_id=external_id),
            "unpinned": request,
        }
        step = self._step(requests, "pin-scope", {"unpinned": produce},
                          outputs={"unpinned": self.root / "pin-scope" / "unpinned"},
                          provenance={"unpinned": {"producer": "sequence-test/pinned-scope"}})
        step["data_ids"] = {"pinned": external_id}
        with patch.object(self.index, "observations",
                          side_effect=AssertionError("sequence scanned the global index")):
            result, = self._run((lambda: step,))

        self.assertEqual(result["datasets"]["pinned"], external_id)
        self.assertNotEqual(result["datasets"]["unpinned"], external_id)
        self.assertEqual(calls, [(0, 1)])
        with self.index.open(result["datasets"]["unpinned"]) as reader:
            np.testing.assert_array_equal(reader.read(time=0), [30, 40j])

    def test_invalid_final_step_or_cycle_stops_before_any_graph_writer(self):
        _, first = self._source_step("preflight-first")
        cyclic_a = _spec("cycle_a", times=(0,))
        cyclic_b = _spec("cycle_b", times=(0,))
        doc_a, doc_b = cyclic_a.document, cyclic_b.document
        doc_a["parents"] = [{"request": "b"}]
        doc_b["parents"] = [{"request": "a"}]
        a, b = DatasetRequest(doc_a), DatasetRequest(doc_b)

        def producer(*args):
            raise AssertionError("a prior graph wrote before static validation finished")

        cycle_step = self._step({"a": a, "b": b}, "preflight-cycle",
            {"a": producer, "b": producer})
        outputs = tuple(cycle_step["outputs"].values())
        with patch("lqcdweave.storage.dataset.DatasetWriter.__enter__",
                   side_effect=AssertionError("writer opened before preflight finished")):
            with self.assertRaisesRegex(ValueError, "contains a cycle"):
                self._run((first, lambda: cycle_step))
        for output in outputs:
            self.assertFalse(output.exists())
        self.assertFalse((self.root / "preflight-first").exists())

        _, first = self._source_step("preflight-fields-first")
        malformed = dict(requests={"source": DatasetRequest(_spec("late_bad"))},
                         outputs={}, provenance={}, producers={})
        with patch("lqcdweave.storage.dataset.DatasetWriter.__enter__",
                   side_effect=AssertionError("writer opened before field validation")):
            with self.assertRaisesRegex(ValueError, "missing fields"):
                self._run((first, lambda: malformed))
        self.assertFalse((self.root / "preflight-fields-first").exists())

        occupied_receipt = self.root / "occupied-receipt-path"
        occupied_receipt.write_bytes(b"keep this existing file")
        _, first = self._source_step("preflight-receipt-path-first")
        invalid_receipt_path = first()
        invalid_receipt_path["receipt_directory"] = occupied_receipt
        with patch("lqcdweave.storage.dataset.DatasetWriter.__enter__",
                   side_effect=AssertionError("writer opened before path validation")):
            with self.assertRaises(NotADirectoryError):
                self._run((lambda: invalid_receipt_path,))
        self.assertEqual(occupied_receipt.read_bytes(), b"keep this existing file")
        self.assertFalse((self.root / "preflight-receipt-path-first").exists())

    def test_bad_initial_receipt_rejected_before_builders_or_writers(self):
        _, source_builder = self._source_step("trusted-seed")
        seed, = self._run((source_builder,))
        receipt_path = Path(seed["receipt"])
        original_receipt = receipt_path.read_bytes()
        output = self.root / "must-not-run"
        builders = []

        def forbidden_builder():
            builders.append(True)
            raise AssertionError("builder called before initial receipt validation")

        with patch("lqcdweave.storage.dataset.DatasetWriter.__enter__",
                   side_effect=AssertionError("writer started before bad anchor rejection")):
            with self.assertRaisesRegex(ValueError, "project_id"):
                self._run((forbidden_builder,), receipts=(receipt_path,),
                    expected_project_id="wrong-project")
            self.assertEqual(builders, [])
        self.assertEqual(builders, [])
        self.assertFalse(output.exists())
        self.assertEqual(receipt_path.read_bytes(), original_receipt)

        source_directory = self.root / "trusted-seed" / "source"
        source_payload = Path(read_dataset_metadata(
            source_directory, cfg="cfg-1")["storage"]["path"])
        if not source_payload.is_absolute():
            source_payload = source_directory / source_payload
        original_block = source_payload.read_bytes()
        try:
            source_payload.write_bytes(b"damaged referenced dataset file")
            with self.assertRaises((ValueError, OSError)):
                self._run((forbidden_builder,), receipts=(receipt_path,))
            self.assertEqual(builders, [])
            self.assertFalse(output.exists())
        finally:
            source_payload.write_bytes(original_block)

    def test_interrupted_middle_graph_resumes_before_future_graph(self):
        source_calls = []
        _, source_builder = self._source_step("resume-source", calls=source_calls)
        projection_calls = []
        interrupt = {"enabled": True}
        projection_builder = self._projection_step(
            "resume-projection", calls=projection_calls, interrupt=interrupt)
        tail_calls = []
        tail_builder = self._tail_step("resume-tail", calls=tail_calls)
        builders = (source_builder, projection_builder, tail_builder)

        with patch.object(self.index, "observations",
                          side_effect=AssertionError("sequence scanned the global index")):
            with self.assertRaisesRegex(RuntimeError, "projection step interrupted"):
                self._run(builders, resume=True)
        self.assertEqual(source_calls, [(0, 1, 2, 3)])
        self.assertEqual(projection_calls, [0, 1, 2])
        self.assertEqual(tail_calls, [])
        partial = self.root / "resume-projection" / "projection"
        partial_state = inspect_dataset(partial, cfg="cfg-1")
        self.assertEqual(partial_state["state"], "partial")
        self.assertEqual(partial_state["completed"], (0, 1))
        self.assertEqual(partial_state["execution"]["attempts"][-1]["status"], "failed")
        self.assertTrue((partial / ".state" / "cfg-1.npy").is_file())
        self.assertFalse((partial / "cfg-1.npy").exists())
        self.assertFalse((self.root / "resume-tail").exists())
        first_rows = _cfg_rows(partial, partial_state["completed"])
        first_receipt = next((self.root / "resume-source").glob("graph-*.jsonl.gz"))
        first_receipt_bytes = first_receipt.read_bytes()

        with patch.object(self.index, "observations",
                          side_effect=AssertionError("resume scanned the global index")):
            results = self._run(builders, resume=True)
        self.assertEqual(source_calls, [(0, 1, 2, 3)])
        self.assertEqual(projection_calls, [0, 1, 2, 2, 3])
        self.assertEqual(tail_calls, [(0, 1)])
        self.assertEqual(_cfg_rows(partial, partial_state["completed"]), first_rows)
        self.assertEqual(first_receipt.read_bytes(), first_receipt_bytes)
        self.assertEqual(len(results), 3)
        projected_id = results[1]["datasets"]["projection"]
        with self.index.open(projected_id) as reader:
            for time in range(4):
                np.testing.assert_array_equal(
                    reader.read(time=time), 2 * _source_row(time) + (1 - 3j))
        self.assertTrue(read_json_records(results[1]["receipt"]))


if __name__ == "__main__":
    unittest.main()
