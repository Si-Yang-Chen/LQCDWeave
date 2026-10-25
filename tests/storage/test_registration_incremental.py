"""Incremental v1 registration against the original full-table factoring."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.dataset import canonical_json
from lqcdweave.storage import dataset as storage
from lqcdweave.storage import metadata
from lqcdweave.storage import registration
from storage.test_dataset_registration import _publish, _spec


def _reference_common(values):
    first = values[0]
    if all(canonical_json(first) == canonical_json(value) for value in values[1:]):
        return first
    if all(type(value) is dict for value in values):
        keys = set(first).intersection(*(set(value) for value in values[1:]))
        return {key: _reference_common([value[key] for value in values])
                for key in first if key in keys}
    if (all(type(value) is list and len(value) == len(first) for value in values)
            and first and all(type(item) is dict for value in values for item in value)):
        return [_reference_common([value[i] for value in values]) for i in range(len(first))]
    return None


def _reference_restore(common, changes):
    value = json.loads(canonical_json(common))
    for path, item in changes:
        if not path:
            value = item
        else:
            parent = value
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = item
    return value


class IncrementalRegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def assert_roundtrip(self, records):
        document = registration.read_registration_document(self.root)
        self.assertEqual(canonical_json(document['common']),
                         canonical_json(_reference_common(list(records.values()))))
        restored = {cfg: _reference_restore(document['common'], changes)
                    for cfg, changes in document['configurations'].items()}
        self.assertEqual(canonical_json(restored), canonical_json(records))

    def test_fast_equality_keeps_canonical_json_types_and_signed_zero(self):
        values = [None, False, True, 0, 1, 1.0, -0.0, 0.0, 1e-7, '1',
                  [], {}, [1, True], [1.0, 1], {'x': 1}, {'x': 1.0},
                  {'a': [dict(n=-0.0)]}, {'a': [dict(n=0.0)]},
                  {'a': 1, 'b': 2}, {'b': 2, 'a': 1}, (1, 2), [1, 2]]
        for left in values:
            for right in values:
                with self.subTest(left=left, right=right):
                    self.assertEqual(registration._equal(left, right),
                                     canonical_json(left) == canonical_json(right))

    def test_shrinking_common_rebases_ancestor_descendant_and_list_changes(self):
        records = {}
        values = [
            dict(a=dict(x=1, y=[dict(p=2, q=3)]), removed=dict(k=4), n=1, z=-0.0),
            dict(a=dict(x=2, y=[dict(p=2, q=5)]), removed=dict(k=6), n=True, z=0.0),
            dict(a=dict(y=[dict(p=7, extra='new')]), n=1.0, z=None),
            dict(a=['changed', 'container'], n=None, z='changed'),
            ['root', dict(a=1)],
            None,
        ]
        for index, record in enumerate(values):
            cfg = str(index)
            # append_registration is the internal factoring boundary; its
            # records need not be complete scientific Dataset declarations.
            with registration.registration_lock(self.root):
                registration.append_registration(self.root,
                    registration.read_registration_document(self.root), cfg, record)
            records[cfg] = record
            self.assert_roundtrip(records)

    def test_seeded_mixed_records_match_full_table_reference(self):
        rng = random.Random(223715)

        def value(depth):
            if not depth or rng.randrange(3) == 0:
                return rng.choice([None, False, True, 0, 1, 1.0, -0.0, 0.0, 'same', 'other'])
            if rng.randrange(2):
                return {key: value(depth - 1) for key in ('a', 'b', 'c') if rng.randrange(4)}
            return [value(depth - 1) for _ in range(rng.randrange(4))]

        records = {}
        for index in range(50):
            record = dict(data_id=str(index), stable={'large': list(range(100))},
                          variable=value(3), rows=[dict(x=value(2))])
            registration.register_record(self.root, str(index), record)
            records[str(index)] = record
            self.assert_roundtrip(records)

    def test_append_to_existing_v1_does_not_expand_unrelated_records(self):
        records = {str(i): dict(data_id=str(i), shared=list(range(1000)), cfg=i)
                   for i in range(20)}
        registration.write_registration(self.root, records)
        new = dict(data_id='20', shared=list(range(1000)), cfg=20)
        with patch.object(registration, '_restore', side_effect=AssertionError('expanded old cfg')):
            registration.register_record(self.root, '20', new)
        records['20'] = new
        self.assert_roundtrip(records)
        before = (self.root / 'dataset.json').read_bytes()
        with self.assertRaises(FileExistsError):
            registration.register_record(self.root, '20', new)
        self.assertEqual((self.root / 'dataset.json').read_bytes(), before)

    def test_selected_metadata_restores_only_requested_cfg(self):
        records = {str(i): dict(data_id=str(i), shared={'nested': [1, 2]}, cfg=i)
                   for i in range(10)}
        registration.write_registration(self.root, records)
        with patch.object(registration, '_restore', wraps=registration._restore) as restore:
            key, record = registration.selected_record(self.root, '4')
        self.assertEqual(key, '4')
        self.assertEqual(record, records['4'])
        self.assertEqual(restore.call_count, 1)
        record['shared']['nested'][0] = 999
        self.assertEqual(registration.selected_record(self.root, '5')[1], records['5'])

    def test_writer_publish_rolls_back_payload_when_registration_write_fails(self):
        with storage.DatasetWriter(self.root, _spec('failed'), provenance={}) as writer:
            for index in writer.missing:
                writer.write(index, np.ones(3))
            with patch.object(registration, 'write_json_document', side_effect=OSError('write failed')):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    writer.publish()
            self.assertFalse((self.root / 'failed.npy').exists())
            self.assertTrue((self.root / '.state' / 'failed.npy').is_file())
        self.assertEqual(registration.read_registration(self.root), {})
        self.assertFalse((self.root / '.state' / 'registration.lock').exists())

    def test_atomic_metadata_write_survives_a_brief_windows_reader_handle(self):
        path = self.root / 'dataset.json'
        metadata.write_json_document(path, {'previous': True})
        original = metadata.os.replace
        denied = PermissionError('reader still has the destination open')
        denied.winerror = 32
        calls = []

        def replace(source, destination):
            calls.append((source, destination))
            if len(calls) == 1:
                self.assertEqual(json.loads(path.read_text()), {'previous': True})
                raise denied
            return original(source, destination)

        with patch.object(metadata.os, 'replace', replace):
            metadata.write_json_document(path, {'published': True})
        self.assertEqual(json.loads(path.read_text()), {'published': True})
        self.assertEqual(set(self.root.iterdir()), {path})

    def test_unrelated_permission_failure_is_not_retried(self):
        path = self.root / 'dataset.json'
        with patch.object(metadata.os, 'replace', side_effect=PermissionError('not writable')) as replace:
            with self.assertRaisesRegex(PermissionError, 'not writable'):
                metadata.write_json_document(path, {'published': True})
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_timed_out_registration_does_not_remove_the_owning_lock(self):
        state = self.root / '.state'
        state.mkdir()
        lock = state / 'registration.lock'
        lock.write_text('another-process')
        with patch.object(registration.time, 'monotonic', side_effect=[0, 61]):
            with self.assertRaisesRegex(FileExistsError, 'registration is locked'):
                with registration.registration_lock(self.root):
                    self.fail('entered a lock held by another process')
        self.assertEqual(lock.read_text(), 'another-process')

    def test_concurrent_cfg_publication_preserves_metadata_and_payloads(self):
        def publish(index):
            cfg = str(index)
            identity, expected = _publish(self.root, _spec(cfg, modes=(index + 4, 1, index + 30)))
            with storage.DatasetReader(self.root, cfg=cfg) as reader:
                np.testing.assert_array_equal(reader.read(time=2), expected[0])
                self.assertEqual(reader.data_id, identity)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(publish, range(12)))
        self.assertEqual(set(registration.read_registration(self.root)), {str(i) for i in range(12)})


if __name__ == '__main__':
    unittest.main()
