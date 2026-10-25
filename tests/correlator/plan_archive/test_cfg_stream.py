"""Large archives keep one chunk, preserve sharing, and reject partial files."""
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from correlator.execution import test_cfg_runtime as fixtures
from lqcdweave.storage.contraction_plans import cfg_stream
from lqcdweave.storage.contraction_plans.api import (
    save_cfg_plan, load_cfg_plan, PlanArchiveError,
)


class CfgStreamTests(unittest.TestCase):
    def test_gzip_preserves_stream_records_timestamp_and_legacy_readback(self):
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory)/'plain.plan'
            compressed = Path(directory)/'compressed.plan.gz'
            cfg_stream.write_stream(plain, plan)
            cfg_stream.write_stream(compressed, plan)
            raw = plain.read_bytes()
            packed = compressed.read_bytes()
            self.assertEqual(gzip.decompress(packed), raw)
            self.assertEqual(packed[4:8], b'\0'*4)  # Deterministic mtime, not wall time.
            self.assertEqual(packed[3] & 8, 0)  # No temporary filename in the gzip header.
            self.assertEqual(load_cfg_plan(compressed), plan)
            self.assertEqual(load_cfg_plan(gzip.compress(raw, compresslevel=9, mtime=0)), plan)
            cfg_stream.write_stream(compressed, plan)
            self.assertEqual(compressed.read_bytes(), packed)

    def test_full_field_v4_remains_readable_and_rejects_v5_schema_spoofing(self):
        from lqcdweave.storage.contraction_plans import api, compact
        plan,*_=fixtures.CfgReplayTests().setup_plan()
        layout=cfg_stream._layout(plan)
        encoder=compact.StreamEncoder(omit_defaults=False)
        lines=[cfg_stream._MAGIC.rstrip(b'\n'),api._json_bytes(api._make_envelope(
            'complete-cfg-plan',{'encoding':'unified-device-stream-v4','fields':layout}))]
        for name,kind,count in layout:
            value=getattr(plan,name)
            items=list(value.items() if kind=='mapping' else value if kind=='tuple' else (value,))
            for offset in range(0,count,cfg_stream._CHUNK_ITEMS):
                lines.append(api._json_bytes([name,offset,*encoder.chunk(items[offset:offset+cfg_stream._CHUNK_ITEMS])]))
        lines.append(b'{"complete":true}')
        restored=load_cfg_plan(b'\n'.join(lines)+b'\n')
        self.assertEqual(restored,plan)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'defaults.plan'
            cfg_stream.write_stream(path,plan)
            records=path.read_bytes().splitlines()
        header=json.loads(records[1])
        self.assertEqual(header['payload']['encoding'],'unified-device-stream-v5')
        header['payload']['encoding']='unified-device-stream-v4'
        records[1]=api._json_bytes(header)
        with self.assertRaisesRegex(api.PlanArchiveError,'record schema'):
            load_cfg_plan(b'\n'.join(records)+b'\n')

    def test_chunked_roundtrip_sharing_and_execution_without_planning(self):
        fixture = fixtures.CfgReplayTests()
        original, expanded, template, elemental, propagator = fixture.setup_plan()
        spec = next(iter(original.specs.values()))
        plan = replace(original, specs=dict(original.specs,
            **{f'shared-{i}': spec for i in range(600)}))
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.plan', '.plan.gz'):
                path = Path(directory) / ('cfg' + suffix)
                with patch.object(cfg_stream, '_STREAM_ITEMS', 0):
                    save_cfg_plan(path, plan)
                with (patch('lqcdweave.storage.contraction_plans.cfg_archive._read_document',
                            side_effect=AssertionError('whole document')),
                      patch('lqcdweave.contraction.planning.api.audit_cfg_plan',
                            side_effect=AssertionError('audit')),
                      patch('lqcdweave.contraction.planning.api.build_cfg_plan',
                            side_effect=AssertionError('replan'))):
                    restored = load_cfg_plan(path)
                    from_bytes = load_cfg_plan(path.read_bytes())
                self.assertEqual(restored, plan)
                self.assertEqual(from_bytes, plan)
                self.assertIs(restored.specs['shared-0'], restored.specs['shared-599'])
            # Execute the decoded arrangements through the same two-cfg fixture.
            expected = fixture.run_plan(capacity=0)[1]
            with patch.object(fixture, 'setup_plan', return_value=(
                    restored, expanded, template, elemental, propagator)):
                actual = fixture.run_plan(capacity=0)[1]
            for key, values in expected.items():
                for reference, value in zip(values, actual[key], strict=True):
                    np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)

    def test_truncation_duplicate_chunk_and_trailing_data_are_rejected(self):
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.plan'
            cfg_stream.write_stream(path, plan)
            lines = path.read_bytes().splitlines(keepends=True)
        invalid = [b''.join(lines[:-1]), b''.join(lines + [b'{}\n']),
                   b''.join(lines[:3] + lines[2:])]
        # Tampering with the version must not enter a migration or planner.
        header = json.loads(lines[1]); header['version'] = 7
        invalid.append(lines[0] + json.dumps(header).encode() + b'\n' + b''.join(lines[2:]))
        # A chunk whose advertised offset skips records is malformed.
        chunk = json.loads(lines[2]); chunk[1] += 1
        invalid.append(b''.join(lines[:2]) + json.dumps(chunk).encode() + b'\n' + b''.join(lines[3:]))
        invalid.append(gzip.compress(b''.join(lines))[:-8])
        for data in invalid:
            with self.subTest(size=len(data)):
                with self.assertRaises(PlanArchiveError):
                    load_cfg_plan(data)

    def test_failed_save_preserves_target_and_removes_temporary_file(self):
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        invalid = replace(plan, output_definition={'unsupported': object()})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.plan.gz'
            cfg_stream.write_stream(path, plan)
            before = path.read_bytes()
            with self.assertRaises(PlanArchiveError):
                cfg_stream.write_stream(path, invalid)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertEqual(load_cfg_plan(path), plan)
            cfg_stream.write_stream(path, plan)
            self.assertEqual(path.read_bytes(), before)

    def test_global_mapping_is_encoded_in_bounded_chunks(self):
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        spec = next(iter(plan.specs.values()))
        plan = replace(plan, specs={str(i): spec for i in range(1025)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.plan'
            with patch.object(cfg_stream, '_CHUNK_ITEMS', 7):
                cfg_stream.write_stream(path, plan)
                restored = load_cfg_plan(path)
            self.assertEqual(restored, plan)
            records = [json.loads(line) for line in path.read_bytes().splitlines()[2:-1]]
            chunks = [r for r in records if r[0] == 'specs']
            self.assertGreater(len(chunks), 100)
            self.assertLessEqual(max(len(r[-1]) for r in chunks), 7)
            self.assertEqual(sum(len(r[-1]) for r in chunks), 1025)

    def test_cross_chunk_sharing_preserves_distinct_equal_records(self):
        from lqcdweave.contracts.correlator.api import TensorSpec
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        first, second = TensorSpec((3,), 'float64'), TensorSpec((3,), 'float64')
        template = plan.templates[0]
        other = replace(template, plan=replace(template.plan, state=replace(template.plan.state)))
        plan = replace(plan, templates=(template, other),
                       specs=dict(plan.specs, a=first, b=second, c=first, d=second))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'identity.plan.gz'
            with patch.object(cfg_stream, '_CHUNK_ITEMS', 1):
                cfg_stream.write_stream(path, plan)
            restored = load_cfg_plan(path)
        self.assertIs(restored.specs['a'], restored.specs['c'])
        self.assertIs(restored.specs['b'], restored.specs['d'])
        self.assertIsNot(restored.specs['a'], restored.specs['b'])
        left, right = restored.templates
        self.assertIsNot(left.plan.state, right.plan.state)
        self.assertIs(left.plan.state.actions[0], right.plan.state.actions[0])
        self.assertIs(left.plan.memory[0], right.plan.memory[0])
        binding = dict(zip(left.keys, restored.instances[0].node_ids.array(), strict=True))
        key = str(int(binding['tensor:2']))
        self.assertIs(left.preparation_specs['tensor:2'], restored.specs[key])
        self.assertIs(left.preparation_closure['tensor:2'].recipe.constants[0],
                      restored.preparations[key].recipe.constants[0])

    def test_legacy_stream_remains_readable_and_bad_dictionaries_are_rejected(self):
        from lqcdweave.storage.contraction_plans import api
        plan, *_ = fixtures.CfgReplayTests().setup_plan()
        # Independently generate the former v3 framing with its original codec.
        shared = {}
        layout = cfg_stream._layout(plan)
        lines = [cfg_stream._MAGIC.rstrip(b'\n'), api._json_bytes(api._make_envelope(
            'complete-cfg-plan', {'encoding': 'unified-device-stream-v3', 'fields': layout}))]
        for name, kind, count in layout:
            value = getattr(plan, name)
            items = list(value.items() if kind == 'mapping' else value if kind == 'tuple' else (value,))
            for offset in range(0, count, cfg_stream._CHUNK_ITEMS):
                lines.append(api._json_bytes([name, offset, [api._encode_plan_value(item, shared=shared)
                    for item in items[offset:offset+cfg_stream._CHUNK_ITEMS]]]))
        lines.append(b'{"complete":true}')
        self.assertEqual(load_cfg_plan(b'\n'.join(lines)+b'\n'), plan)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'compact.plan'
            cfg_stream.write_stream(path, plan)
            records = path.read_bytes().splitlines()
            original = json.loads(records[2])
            for change in (
                lambda row: row[2].append(['unknown-record', []]),
                lambda row: row[3].append(1),
                lambda row: row.__setitem__(4, [[-10, 99999999]]),
                lambda row: row[2].extend(row[2][:1]),
            ):
                from copy import deepcopy
                broken = deepcopy(original)
                change(broken)
                modified = list(records)
                modified[2] = json.dumps(broken).encode()
                with self.assertRaises(PlanArchiveError):
                    load_cfg_plan(b'\n'.join(modified)+b'\n')


if __name__ == '__main__':
    unittest.main()
