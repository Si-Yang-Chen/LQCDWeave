import gzip
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from test_storage_schedule_archive import (
    CurrentDatasetLoader,
    batch,
    join_storage_batches,
    load_storage_schedule,
    replay_storage_batches,
    save_storage_schedule,
)
from lqcdweave.storage.contraction_plans.api import ARCHIVE_VERSION


class ScheduleArchiveTests(unittest.TestCase):
    def _schedule_inputs(self):
        plans = join_storage_batches((batch("a"), batch("b"), batch("a")))
        shapes = {"a": [4], "b": [4]}
        hardware = {"backend": "numpy", "memory": 10000}
        expected = tuple(plan.state.problem for plan in plans)
        blocks = {key: value for plan in plans for key, value in plan.blocks.items()}
        preparations = {key: value for plan in plans for key, value in plan.preparations.items()}
        return plans, shapes, hardware, expected, blocks, preparations

    def test_current_schedule_round_trips_and_rebinds_physical_datasets(self):
        plans, shapes, hardware, expected, blocks, preparations = self._schedule_inputs()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json.gz"
            save_storage_schedule(
                path, plans, shapes=shapes, hardware=hardware,
                metadata={"output_order": [0, 1, 2]},
            )
            document = json.loads(gzip.decompress(path.read_bytes()))

            self.assertEqual(document["format"], "easy-distillation.contraction-plan")
            self.assertEqual(document["version"], ARCHIVE_VERSION)
            self.assertEqual(document["payload_type"], "storage-schedule")
            self.assertEqual(set(document), {"format", "version", "payload_type", "payload"})
            self.assertNotIn("signature", document["payload"])
            self.assertEqual(document["payload"]["metadata"], {"output_order": [0, 1, 2]})

            rebound_blocks = {
                key: replace(value, dataset="new-" + key)
                for key, value in blocks.items()
            }
            restored, metadata = load_storage_schedule(
                path, shapes=shapes, hardware=hardware, expected=expected,
                blocks=rebound_blocks, preparations=preparations,
            )

        self.assertEqual(metadata, {"output_order": [0, 1, 2]})
        self.assertEqual(len(restored), len(plans))
        for current, original in zip(restored, plans, strict=True):
            self.assertEqual(current.state.problem, original.state.problem)
            self.assertEqual(current.state.actions, original.state.actions)
            self.assertEqual(current.events, original.events)
            self.assertEqual(current.blocks, rebound_blocks)
            self.assertEqual(current.preparations, original.preparations)
            self.assertTrue(current.feasible)
            self.assertTrue(current.state.done)

        loader = CurrentDatasetLoader()
        values = []
        replay_storage_batches(
            restored, loader, lambda _index, report: values.append(report[0]),
            prejoined=True,
        )
        self.assertEqual(values, [46.0, 46.0, 46.0])
        self.assertEqual(loader.reads, ["a", "b"])

    def test_metadata_round_trips_without_a_payload_signature(self):
        plans, shapes, hardware, expected, blocks, preparations = self._schedule_inputs()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            save_storage_schedule(path, plans, shapes=shapes, hardware=hardware)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["payload"]["metadata"] = {"output_order": [9, 8, 7]}

            _restored, metadata = load_storage_schedule(
                json.dumps(document), shapes=shapes, hardware=hardware,
                expected=expected, blocks=blocks, preparations=preparations,
            )
            self.assertEqual(metadata, {"output_order": [9, 8, 7]})

            document["payload"]["shapes"]["a"] = [999]
            with self.assertRaises(ValueError):
                load_storage_schedule(
                    json.dumps(document), shapes=shapes, hardware=hardware,
                    expected=expected, blocks=blocks, preparations=preparations,
                )

    def test_failed_save_preserves_previous_archive(self):
        plans, shapes, hardware, _expected, _blocks, _preparations = self._schedule_inputs()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json.gz"
            save_storage_schedule(path, plans, shapes=shapes, hardware=hardware)
            previous = path.read_bytes()

            with self.assertRaises((TypeError, ValueError)):
                save_storage_schedule(
                    path, plans, shapes=shapes, hardware=hardware,
                    metadata={"bad": object()},
                )

            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
