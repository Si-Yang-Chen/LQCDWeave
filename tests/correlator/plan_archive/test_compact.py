"""Compact records preserve typed values and reject malformed descriptions."""
from copy import deepcopy
import json
import unittest

import numpy as np

from lqcdweave.storage.contraction_plans import api, compact


class CompactArchiveTests(unittest.TestCase):
    def test_stream_identity_table_keeps_objects_alive_between_chunks(self):
        import gc
        import weakref
        encoder = compact.StreamEncoder()
        value = api.contract_api.ArrayConstant.from_array(np.eye(3, dtype='complex64'))
        reference = weakref.ref(value)
        first = encoder.chunk((value,))
        del value
        gc.collect()
        self.assertIsNotNone(reference())
        # Equality does not merge distinct frozen records in a complete plan.
        other = api.contract_api.ArrayConstant.from_array(np.eye(3, dtype='complex64'))
        second = encoder.chunk((other, reference()))
        decoder = compact.StreamDecoder()
        original, = decoder.chunk(*first)
        distinct, repeated = decoder.chunk(*second)
        self.assertIsNot(original, distinct)
        self.assertIs(original, repeated)
        del encoder
        gc.collect()
        self.assertIsNone(reference())

    def test_encoder_scalar_shortcuts_keep_depth_types_and_numeric_bits(self):
        value = (None, False, 0, 2**80, -2, 'basis', -0.0,
                 np.int64(3), np.float64(-0.0), (True, 'time', 7))
        restored = self.roundtrip(value)
        for before, after in zip(value, restored, strict=True):
            self.assertIs(type(before), type(after))
            if isinstance(before, (float, np.generic)):
                self.assertEqual(np.asarray(before).tobytes(), np.asarray(after).tobytes())
            else:
                self.assertEqual(before, after)
        for scalar in (None, False, 0, 'basis'):
            nested = scalar
            for _ in range(128):
                nested = (nested,)
            compact.encode(nested)
            with self.subTest(scalar=scalar), self.assertRaisesRegex(api.PlanArchiveError, 'nesting'):
                compact.encode((nested,))

    def test_identity_preserving_root_keeps_inline_records_and_distinct_descriptions(self):
        from dataclasses import replace
        c = api.contract_api
        action = c.Action('load', 'a')
        pattern = c.AxisPattern(((0, 1), (1, 2)), (0, 2))
        distinct = replace(pattern)
        value = (action, action, pattern, distinct)
        restored = api.decode_plan_value(compact.encode(value, preserve_identity=True))
        self.assertEqual(restored, value)
        self.assertIs(restored[0], restored[1])
        self.assertIsNot(restored[2], restored[3])
        # Other archive payloads retain their existing value-dictionary policy.
        ordinary = api.decode_plan_value(compact.encode(value))
        self.assertIs(ordinary[2], ordinary[3])

    def test_old_precision_records_restore_explicit_compatible_defaults(self):
        c=api.contract_api
        coordinates,_=c.describe_coordinates()
        records=(c.TensorPreparation('identity',('raw',)),
                 c.OutputRecipe(c.OutputNode('constant',0),(),coordinates))
        for record in records:
            for version in (1,2):
                schemas,strings,visit=compact._encoder(omit_defaults=version==2)
                root=visit(record)
                row=schemas[0]
                removed=len(row[1])-1
                row[1].pop()
                if version==2:
                    row[2]=[item for item in row[2] if item[0]!=removed]
                    self.assertFalse(root[1]&(1<<removed))
                else:
                    root.pop()
                encoded=dict(type='archive.compact-v'+str(version),
                             value=dict(schemas=schemas,strings=strings,root=root))
                self.assertEqual(api.decode_plan_value(encoded),record)
                broken=deepcopy(encoded)
                broken['value']['schemas'][0][1][0]='unknown'
                with self.assertRaisesRegex(api.PlanArchiveError,'fields'):
                    api.decode_plan_value(broken)

    def test_problem_records_saved_before_factorized_vertices_are_rejected(self):
        c = api.contract_api
        problem = c.ContractionProblem({'a': c.TensorSpec((2,), np.complex128)},
                                       (c.ContractionTerm(('a',), 'i->i'),),
                                       {'a': c.SourceSpec(region='a', region_bytes=32, read_bytes=32)})
        for version in (1, 2):
            schemas, strings, visit = compact._encoder(omit_defaults=version == 2)
            root = visit(problem)
            row = next(row for row in schemas if row[0].endswith('.ContractionProblem'))
            self.assertEqual(row[1][-1], 'factorized_vertices')
            removed = len(row[1]) - 1
            row[1].pop()
            if version == 2:
                row[2] = [item for item in row[2] if item[0] != removed]
            else:
                root.pop()
            encoded = dict(type='archive.compact-v' + str(version),
                           value=dict(schemas=schemas, strings=strings, root=root))
            with self.assertRaisesRegex(api.PlanArchiveError, 'fields'):
                api.decode_plan_value(encoded)

    def test_field_masks_require_exact_range_required_bits_and_payload_count(self):
        original=compact.encode(api.contract_api.Action('load','a'))
        count=len(original['value']['schemas'][0][1])
        for mask in (True,-1,0,1,1<<count,2**120):
            broken=deepcopy(original)
            broken['value']['root'][1]=mask
            with self.subTest(mask=mask),self.assertRaisesRegex(api.PlanArchiveError,'field mask'):
                api.decode_plan_value(broken)
        for add in (False,True):
            broken=deepcopy(original)
            if add:
                broken['value']['root'].append(None)
            else:
                broken['value']['root'].pop()
            with self.subTest(add=add),self.assertRaisesRegex(api.PlanArchiveError,'wrong field count'):
                api.decode_plan_value(broken)

    def test_many_distinct_masks_keep_all_values_after_layout_cache_capacity(self):
        c=api.contract_api
        records=tuple(c.Action('load',str(bits),used_np=int(bool(bits&1)),join_id='x' if bits&2 else '',
            same_point=bool(bits&4),transpose_bridge=bool(bits&8),term_index=0 if bits&16 else -1,
            positions=(0,) if bits&32 else (),kernel_path=((0,1),) if bits&64 else ())
            for bits in range(128))
        self.assertEqual(self.roundtrip(records),records)

    def test_reused_mask_still_rejects_wrong_payload_count_and_noninteger_masks(self):
        encoder=compact.StreamEncoder()
        schemas,strings,roots=encoder.chunk((api.contract_api.Action('load','a'),))
        self.assertEqual(roots[0][0],compact._SHARED)
        inline=[roots[0][2],*roots[0][3:]]
        for invalid in (inline[:-1],[*inline,None],
                        [inline[0],True,*inline[2:]],
                        [inline[0],float(inline[1]),*inline[2:]]):
            decoder=compact.StreamDecoder()
            decoder.chunk(schemas,strings,roots)
            with self.subTest(invalid=invalid),self.assertRaises(api.PlanArchiveError):
                decoder.chunk([],[],[invalid])

    def test_reused_mask_checks_omitted_default_depth_at_each_record_position(self):
        encoder=compact.StreamEncoder()
        schemas,strings,roots=encoder.chunk((api.contract_api.Action('load','a'),))
        decoder=compact.StreamDecoder()
        decoder.chunk(schemas,strings,roots)
        nested=[roots[0][2],*roots[0][3:]]
        for _ in range(127):
            nested=[compact._TUPLE,nested]
        with self.assertRaisesRegex(api.PlanArchiveError,'nesting'):
            decoder.chunk([],[],[nested])

    def test_default_snapshots_restore_full_fields_types_and_float_bits(self):
        from dataclasses import fields
        c=api.contract_api
        records=(c.Action('load','a'),c.Action('load','b',seconds=-0.0),
                 c.Action('load','c',used_np=False),c.Action('load','d',seconds=np.float64(-0.0)))
        encoded=compact.encode(records)
        self.assertEqual(encoded['type'],'archive.compact-v2')
        restored=self.roundtrip(records)
        for before,after in zip(records,restored,strict=True):
            for field in fields(before):
                left,right=getattr(before,field.name),getattr(after,field.name)
                self.assertIs(type(left),type(right))
                if isinstance(left,(float,np.generic)):
                    self.assertEqual(np.asarray(left).tobytes(),np.asarray(right).tobytes())
                else:
                    self.assertEqual(left,right)
        action_schema=next(row for row in encoded['value']['schemas'] if row[0].endswith('.Action'))
        snapshots=dict(action_schema[2])
        self.assertIn(action_schema[1].index('left_axes'),snapshots)
        # Stored values define the omitted fields; a reader's current default
        # must not silently override the actual value saved in the archive.
        payload=compact.encode(c.Action('load','a'))['value']
        index=payload['schemas'][0][1].index('used_np')
        next(item for item in payload['schemas'][0][2] if item[0]==index)[1]=7
        self.assertEqual(compact.decode(payload).used_np,7)

    def test_defaults_reject_required_fields_mutable_nodes_duplicates_and_v1_spoofing(self):
        original=compact.encode(api.contract_api.Action('load','a'))
        for change in (
            lambda row: row[2].insert(0,[0,None]),
            lambda row: row[2].append(row[2][-1]),
            lambda row: row[2].__setitem__(0,[True,None]),
            lambda row: row[2][0].__setitem__(1,[compact._LIST]),
            lambda row: row[2][0].__setitem__(1,[compact._REF,0]),
            lambda row: row[2][0].__setitem__(1,float('nan')),
            lambda row: row[2][0].__setitem__(1,-2**120),
        ):
            broken=deepcopy(original)
            change(broken['value']['schemas'][0])
            with self.subTest(broken=broken),self.assertRaises(api.PlanArchiveError):
                api.decode_plan_value(broken)
        with self.assertRaises(api.PlanArchiveError):
            api.decode_plan_value(dict(original,type='archive.compact-v1'))
        broken=deepcopy(original)
        broken['value']['strings'].append('i,j->ij')
        index=broken['value']['schemas'][0][1].index('equation')
        next(item for item in broken['value']['schemas'][0][2] if item[0]==index)[1]=-len(broken['value']['strings'])
        with self.assertRaisesRegex(api.PlanArchiveError,'integer AxisPattern'):
            api.decode_plan_value(broken)
        broken=deepcopy(original)
        broken['value']['root']=broken['value']['root'][:2]
        with self.assertRaisesRegex(api.PlanArchiveError,'wrong field count'):
            api.decode_plan_value(broken)

    def test_omitted_nested_defaults_keep_the_original_depth_limit(self):
        value=api.contract_api.Action('load','a')
        for _ in range(126):
            value=(value,)
        self.assertEqual(self.roundtrip(value),value)
        with self.assertRaisesRegex(api.PlanArchiveError,'nesting'):
            compact.encode((value,))
        payload=compact.encode(api.contract_api.Action('load','a'))['value']
        for _ in range(127):
            payload['root']=[compact._TUPLE,payload['root']]
        with self.assertRaisesRegex(api.PlanArchiveError,'nesting'):
            compact.decode(payload)

    def roundtrip(self, value):
        # Cross the actual JSON boundary, including signed zeros and references.
        return api.decode_plan_value(json.loads(api._json_bytes(compact.encode(value))))

    def test_direct_leaves_keep_boolean_identity_integer_range_and_mapping_collisions(self):
        payload = {'schemas': [], 'strings': ['text'],
                   'root': [-2, None, False, True, 0, 2**120, [-1, -1], -1, -0.0]}
        restored = compact.decode(payload)
        self.assertIsNone(restored[0])
        self.assertIs(restored[1], False)
        self.assertIs(restored[2], True)
        self.assertEqual(restored[3:7], (0, 2**120, -1, 'text'))
        self.assertTrue(np.signbit(restored[-1]))
        for pairs in ([-4, True, 0, 1, 1], [-4, None, 0, None, 1]):
            with self.subTest(pairs=pairs), self.assertRaisesRegex(api.PlanArchiveError, 'duplicate keys'):
                compact.decode(dict(payload, root=pairs))

    def test_literal_shortcuts_preserve_depth_at_container_mapping_and_record_boundaries(self):
        def nest(node, count):
            for _ in range(count):
                node = [-2, node]
            return node

        for leaf in (None, False, True, 0, 2**120):
            for inner in ([-2, leaf], [-3, leaf], [-5, leaf], [-6, leaf], [-4, leaf, 0]):
                with self.subTest(leaf=leaf, tag=inner[0]):
                    compact.decode({'schemas': [], 'strings': [], 'root': nest(inner, 127)})
                    with self.assertRaisesRegex(api.PlanArchiveError, 'nesting'):
                        compact.decode({'schemas': [], 'strings': [], 'root': nest(inner, 128)})
        record = api.contract_api.MemoryPoint(1, 'contract', 2, 3, 4, 5, 6, 7)
        payload = compact.encode(record)['value']
        restored = compact.decode(dict(payload, root=nest(payload['root'], 127)))
        for _ in range(127):
            restored = restored[0]
        self.assertEqual(restored, record)
        with self.assertRaisesRegex(api.PlanArchiveError, 'nesting'):
            compact.decode(dict(payload, root=nest(payload['root'], 128)))

    def test_string_reference_and_finite_float_checks_survive_parent_shortcuts(self):
        for node in (
            [-2, -1], [-4, -1, 0], [-4, 0, -1],
            [-2, float('nan')], [-3, float('inf')], [-4, 0, float('-inf')],
            [-7, 0], [-7, True], [-9, False, -1],
        ):
            with self.subTest(node=node), self.assertRaises(api.PlanArchiveError):
                compact.decode({'schemas': [], 'strings': [], 'root': node})

    def test_parent_string_lookup_keeps_range_tag_and_duplicate_key_rules(self):
        strings = ['first', 'last']
        constructors = ((compact._TUPLE, tuple), (compact._LIST, list),
                        (compact._SET, set), (compact._FROZEN, frozenset))
        for reference in (-1, -2):
            expected = strings[-reference-1]
            for tag, container in constructors:
                with self.subTest(reference=reference, tag=tag):
                    self.assertEqual(compact.decode({'schemas': [], 'strings': strings,
                        'root': [tag, reference, [compact._INT, reference]]}),
                        container((expected, reference)))
            self.assertEqual(compact.decode({'schemas': [], 'strings': strings,
                'root': [compact._MAP, reference, 0, 0, reference]}), {expected: 0, 0: expected})
        for reference in (-3, -2**120):
            roots = [[tag, reference] for tag, _ in constructors]
            roots.extend(([compact._MAP, reference, 0], [compact._MAP, 0, reference]))
            for root in roots:
                with self.subTest(root=root), self.assertRaisesRegex(api.PlanArchiveError, 'string reference'):
                    compact.decode({'schemas': [], 'strings': strings, 'root': root})
        with self.assertRaisesRegex(api.PlanArchiveError, 'duplicate keys'):
            compact.decode({'schemas': [], 'strings': ['same', 'same'],
                'root': [compact._MAP, -1, 0, -2, 1]})

    def test_record_string_lookup_keeps_constructor_paths_and_depth(self):
        c = api.contract_api
        for record, name in ((c.MemoryPoint(1, 'contract', 2, 3, 4, 5, 6, 7), 'phase'),
                             (c.Action('load', 'a'), 'tensor')):
            payload = compact.encode(record)['value']
            schema = payload['root'][0]
            index = payload['schemas'][schema][1].index(name)
            offset = (2+(payload['root'][1]&((1<<index)-1)).bit_count()
                      if len(payload['schemas'][schema])==3 else index+1)
            payload['strings'].append('last')
            payload['root'][offset] = -len(payload['strings'])
            with self.subTest(record=type(record).__name__):
                self.assertEqual(getattr(compact.decode(payload), name), 'last')
                broken = deepcopy(payload)
                broken['root'][offset] -= 1
                with self.assertRaisesRegex(api.PlanArchiveError, 'string reference'):
                    compact.decode(broken)
                with self.assertRaisesRegex(api.PlanArchiveError, 'nesting'):
                    compact.decode(payload, depth=128)
        for root in ([compact._TUPLE, -1], [compact._MAP, -1, 0], [compact._MAP, 0, -1]):
            with self.subTest(root=root), self.assertRaisesRegex(api.PlanArchiveError, 'nesting'):
                compact.decode({'schemas': [], 'strings': ['first'], 'root': root}, depth=128)

    def test_records_deduplicate_complete_immutable_descriptions(self):
        c = api.contract_api
        first = c.AxisPattern(((0, 1), (1, 2)), (0, 2))
        second = c.AxisPattern(((0, 1), (1, 2)), (0, 2))
        spec = c.TensorSpec((3, 4), 'complex128')
        value = (first, second, spec, c.TensorSpec((3, 4), 'complex128'),
                 c.TensorSpec((3, 4), 'float64'),
                 c.TensorSpec((3, 4), 'complex128', description=c.TensorDescription(
                     axes=('a', 'b'), data_identity='different actual ordered basis')))
        restored = self.roundtrip(value)
        self.assertEqual(restored, value)
        self.assertIs(restored[0], restored[1])
        self.assertIs(restored[2], restored[3])
        self.assertIsNot(restored[2], restored[4])
        self.assertIsNot(restored[2], restored[5])
        encoded = compact.encode((value,) * 100)
        self.assertEqual(len(encoded['value']['strings']), len(set(encoded['value']['strings'])))
        self.assertLess(len(api._json_bytes(encoded)), len(api._json_bytes(api.encode_plan_value((value,)*100))))

    def test_all_scalar_and_container_types_and_mutable_ownership(self):
        mutable = [-1, -2**80, 2**80, -0.0, None, True, 'same text', 2-3j]
        value = ({('key', -2): mutable}, mutable, mutable,
                 frozenset((1, -3, 'same text')), set((2, 4)), np.dtype('>f8'),
                 np.float64(-0.0), np.complex128(1+2j),
                 api.contract_api.ArrayConstant.from_array(np.eye(2)))
        restored = self.roundtrip(value)
        self.assertEqual(restored, value)
        self.assertIsNot(restored[1], restored[2])
        self.assertTrue(np.signbit(restored[1][3]))
        self.assertEqual(restored[6].tobytes(), value[6].tobytes())
        self.assertEqual(restored[7].tobytes(), value[7].tobytes())
        for scalar in (np.float32(np.nan), np.float64(np.inf)):
            self.assertEqual(self.roundtrip(scalar).tobytes(), scalar.tobytes())

    def test_rejects_unknown_schema_fields_bad_counts_and_references(self):
        encoded = compact.encode(api.contract_api.TensorSpec((3,), 'float64'))
        for change in (
            lambda p: p['schemas'][0].__setitem__(0, 'os.system'),
            lambda p: p['schemas'][0][1].append('unexpected'),
            lambda p: p['root'].pop(),
            lambda p: p.__setitem__('root', [-10, 0]),
            lambda p: p.__setitem__('root', [-11, 0, 0, [-10, 0], None, None]),
            lambda p: p.__setitem__('root', -len(p['strings'])-1),
            lambda p: p.__setitem__('root', [-4, 1, None, 1, None]),
            lambda p: p.__setitem__('root', [-4, [-3], None]),
            lambda p: p.__setitem__('root', [True]),
            lambda p: p.__setitem__('root', [-99]),
            lambda p: p.__setitem__('root', float('nan')),
        ):
            broken = deepcopy(encoded)
            change(broken['value'])
            with self.subTest(payload=broken), self.assertRaises(api.PlanArchiveError):
                api.decode_plan_value(broken)

    def test_inline_and_shared_records_keep_schema_range_and_field_count_checks(self):
        c = api.contract_api
        for record in (c.MemoryPoint(1, 'contract', 2, 3, 4, 5, 6, 7), c.Action('load', 'a')):
            # Full-field v1 keeps every field. Keep its strict field-count
            # corruption gate independently from the new v2 default snapshots.
            schemas,strings,visit=compact._encoder(omit_defaults=False)
            inline=dict(schemas=schemas,strings=strings,root=visit(record))
            schemas, strings, roots = compact.StreamEncoder(omit_defaults=False).chunk((record,))
            shared = dict(schemas=schemas, strings=strings, root=roots[0])
            for payload, schema_offset in ((inline, 0), (shared, 2)):
                with self.subTest(record=type(record).__name__, schema_offset=schema_offset):
                    self.assertEqual(compact.decode(payload), record)
                    for invalid in (True, -99, len(payload['schemas']), 2**120):
                        broken = deepcopy(payload)
                        broken['root'][schema_offset] = invalid
                        with self.subTest(schema=invalid), self.assertRaises(api.PlanArchiveError):
                            compact.decode(broken)
                    for add in (False, True):
                        broken = deepcopy(payload)
                        if add:
                            broken['root'].append(None)
                        else:
                            broken['root'].pop()
                        with self.subTest(add=add), self.assertRaisesRegex(api.PlanArchiveError, 'wrong field count'):
                            compact.decode(broken)

    def test_preserves_integer_axis_guard_depth_limit_and_whitelist(self):
        for value in (np.arange(2), lambda: None, float('nan'), complex(float('inf'), 0)):
            with self.assertRaises(api.PlanArchiveError):
                compact.encode(value)
        value = None
        for _ in range(128):
            value = (value,)
        self.assertEqual(self.roundtrip(value), value)
        with self.assertRaises(api.PlanArchiveError):
            compact.encode((value,))
        node = None
        for _ in range(129):
            node = [-2, node]
        with self.assertRaises(api.PlanArchiveError):
            compact.decode({'schemas': [], 'strings': [], 'root': node})
        broken = compact.encode(api.contract_api.Factor('A', (0,)))
        # An external string must not reach the constructor's convenience path.
        broken['value']['strings'].append('old string axes')
        broken['value']['root'][-1] = -len(broken['value']['strings'])
        with self.assertRaisesRegex(api.PlanArchiveError, 'integer axes'):
            api.decode_plan_value(broken)
