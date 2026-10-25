"""Explicit generic schedule conversion preserves saved choices and identities."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from correlator.plan_archive import test_storage_schedule_archive as fixtures
from test_storage_schedule_archive import CurrentDatasetLoader, replay_storage_batches
from lqcdweave.storage.contraction_plans import api


BLOCK_TAG = "lqcdweave.contracts.correlator.storage.StorageBlock"


def block_records(value):
    if isinstance(value, list):
        for item in value:
            yield from block_records(item)
    elif isinstance(value, dict):
        if value.get("type") == BLOCK_TAG:
            yield value["value"]
        for item in value.values():
            yield from block_records(item)


class DatasetFieldMigrationTests(unittest.TestCase):
    def fixture(self, directory):
        plans, shapes, hardware, expected, blocks, preparations = (
            fixtures.ScheduleArchiveTests()._schedule_inputs())
        current = Path(directory) / "current.json.gz"
        metadata = {"artifact": "keep this opaque name", "prediction": {"seconds": 2.5}}
        # Construct an old named-field fixture explicitly. The current writer
        # uses compact records, whose schemas never had an ``artifact`` field.
        with patch.object(api, '_encode_archive_value', api._encode_plan_value):
            api.save_storage_schedule(current, plans, shapes=shapes, hardware=hardware,
                                      metadata=metadata)
        document = json.loads(gzip.decompress(current.read_bytes()))
        old = deepcopy(document)
        for value in block_records(old["payload"]):
            value["artifact"] = value.pop("dataset")
        source = Path(directory) / "old.json.gz"
        source.write_bytes(gzip.compress(json.dumps(old).encode(), mtime=0))
        options = dict(shapes=shapes, hardware=hardware, expected=expected,
                       blocks=blocks, preparations=preparations)
        return plans, document, old, source, options

    def test_explicit_conversion_preserves_complete_schedule_and_numerical_replay(self):
        with TemporaryDirectory() as directory:
            plans, current, _old, source, options = self.fixture(directory)
            before = source.read_bytes()
            target = Path(directory) / "converted.json.gz"
            with self.assertRaisesRegex(api.PlanArchiveError, "StorageBlock fields"):
                api.load_storage_schedule(source, **options)
            with patch("lqcdweave.contraction.planning.api.find_contraction_path",
                       side_effect=AssertionError("must not search")), \
                 patch("lqcdweave.contraction.planning.api.plan_scene_storage",
                       side_effect=AssertionError("must not reschedule")):
                self.assertEqual(api.migrate_plan_archive(source, target), target)
                restored, metadata = api.load_storage_schedule(target, **options)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads(gzip.decompress(target.read_bytes())), current)
            self.assertEqual(metadata, current["payload"]["metadata"])
            for original, loaded in zip(plans, restored, strict=True):
                self.assertEqual(original.state.actions, loaded.state.actions)
                self.assertEqual(original.events, loaded.events)
                self.assertEqual(original.memory, loaded.memory)
                self.assertEqual(original.cost_profile, loaded.cost_profile)
            def replay(items):
                loader = CurrentDatasetLoader()
                values = []
                replay_storage_batches(items, loader,
                    lambda _index, result: values.append(result[0]), prejoined=True)
                return values, loader.reads
            self.assertEqual(replay(restored), replay(plans))

    def test_malformed_and_nonlegacy_inputs_preserve_existing_destination(self):
        with TemporaryDirectory() as directory:
            _plans, current, old, source, _options = self.fixture(directory)
            cases = {}
            for label, change in (
                ("both", lambda value: value.update(dataset="ambiguous")),
                ("unknown", lambda value: value.update(extra=1)),
                ("missing", lambda value: value.pop("selection")),
            ):
                document = deepcopy(old)
                change(next(block_records(document["payload"])))
                cases[label] = document
            cases["no old blocks"] = current
            cases["old complete cfg"] = dict(old, payload_type="complete-cfg-plan")
            cases["other current payload"] = dict(old, payload_type="correlator-plan")
            target = Path(directory) / "destination.json"
            target.write_bytes(b"existing archive")
            for label, document in cases.items():
                with self.subTest(case=label):
                    source.write_bytes(gzip.compress(json.dumps(document).encode(), mtime=0))
                    before = source.read_bytes()
                    with self.assertRaises(api.PlanArchiveError):
                        api.migrate_plan_archive(source, target)
                    self.assertEqual(source.read_bytes(), before)
                    self.assertEqual(target.read_bytes(), b"existing archive")


if __name__ == "__main__":
    unittest.main()
