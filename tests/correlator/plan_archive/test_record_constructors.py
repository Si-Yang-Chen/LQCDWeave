"""Schema-bound construction preserves hooks, constraints and Python binding."""
from copy import deepcopy
from dataclasses import InitVar, dataclass, fields
from inspect import Parameter, signature
import unittest
from unittest.mock import patch

from lqcdweave.storage.contraction_plans import api, compact


class RecordConstructorTests(unittest.TestCase):
    @staticmethod
    def bind(cls):
        names, _, _, _, integers = api._plan_record_schema(cls)
        return api._bind_positional_record_constructor(cls, names, cls.__name__, integers)

    def test_native_record_decodes_all_fields_without_keyword_reassembly(self):
        cls = api.contract_api.MemoryPoint
        values = {field.name: (index+1)*17 for index, field in enumerate(fields(cls))}
        values.update(action_index=7, phase='contract')
        record = cls(**values)
        encoded = compact.encode(record)
        with patch.object(api, '_construct_plan_record', side_effect=AssertionError('keyword reassembly')):
            restored = api.decode_plan_value(encoded)
        self.assertEqual(restored, record)
        self.assertIs(type(restored), cls)

    def test_bound_constructor_still_calls_the_actual_record_initializer(self):
        cls = api.contract_api.MemoryPoint
        constructor = self.bind(cls)
        self.assertIsNotNone(constructor)
        values = (7, 'contract', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        initializer, calls = cls.__init__, []
        def tracked(instance, *args, **kwargs):
            calls.append((args, kwargs))
            initializer(instance, *args, **kwargs)
        with patch.object(cls, '__init__', tracked):
            restored = constructor(values)
        self.assertEqual(calls, [(values, {})])
        self.assertEqual(restored, cls(*values))

    def test_positional_constructor_retains_errors_from_real_record_validation(self):
        cases = (
            (api.contract_api.ArrayConstant, {'shape': (1,), 'dtype': '<f8', 'data': ''}),
            (api.contract_api.AxisPattern, {'inputs': ((0,),), 'output': (9,)}),
        )
        for cls, decoded in cases:
            with self.subTest(record=cls.__name__):
                names = api._plan_record_schema(cls)[0]
                constructor = api._bind_positional_record_constructor(cls, names, cls.__name__, ())
                self.assertIsNotNone(constructor)
                with self.assertRaises(api.PlanArchiveError) as original:
                    api._construct_plan_record(cls, decoded, cls.__name__, ())
                with self.assertRaises(api.PlanArchiveError) as actual:
                    constructor([decoded[name] for name in names])
                self.assertEqual(str(actual.exception), str(original.exception))

    def test_keyword_only_and_archive_guarded_records_use_proven_signatures(self):
        c = api.contract_api
        for cls in (c.TensorSpec, c.Action, c.ContractionTerm, c.Factor, c.PendingTerm, c.BaryonJoinSpec):
            with self.subTest(record=cls.__name__):
                self.assertIsNotNone(self.bind(cls))
        spec = c.TensorSpec((3, 4), 'complex128')
        with patch.object(api, '_construct_plan_record', side_effect=AssertionError('keyword reassembly')):
            self.assertEqual(api.decode_plan_value(compact.encode(spec)), spec)

    def test_mixed_constructor_delivers_keyword_tail_to_actual_initializer(self):
        c = api.contract_api
        record = c.Action('contract', 'out', ('a',), c.AxisPattern(((0,),), (0,)),
                          seconds=-0.0, sparse_spin=((0, 1, 2, 1 + 2j),),
                          resource=c.ResourceTransition('tensor', 'host', 'device', 16),
                          kernel_description=c.VertexKernelDescription('meson', 'whole-matrix-v1'))
        names = api._plan_record_schema(c.Action)[0]
        values = tuple(getattr(record, name) for name in names)
        parameters = signature(c.Action).parameters
        positional = tuple(name for name in names
                           if parameters[name].kind is Parameter.POSITIONAL_OR_KEYWORD)
        keywords = tuple(name for name in names
                         if parameters[name].kind is Parameter.KEYWORD_ONLY)
        self.assertEqual(names, positional + keywords)
        self.assertEqual(keywords, ('sparse_spin', 'core', 'resource',
                                    'wide_accumulation', 'kernel_description'))
        constructor = self.bind(c.Action)
        initializer, calls = c.Action.__init__, []
        def tracked(instance, *args, **kwargs):
            calls.append((args, kwargs))
            initializer(instance, *args, **kwargs)
        with patch.object(c.Action, '__init__', tracked):
            restored = constructor(values)
        self.assertEqual(calls, [(tuple(getattr(record, name) for name in positional),
                                 {name: getattr(record, name) for name in keywords})])
        self.assertEqual(restored, record)
        self.assertEqual(restored.seconds.hex(), record.seconds.hex())
        with patch.object(api, '_construct_plan_record', side_effect=AssertionError('keyword reassembly')):
            self.assertEqual(api.decode_plan_value(compact.encode(record)), record)

    def test_positional_archive_guards_match_original_keyword_rejections(self):
        c = api.contract_api
        equation = c.AxisPattern(((0,),), (0,))
        cases = (
            (c.Action('contract', 'out', ('a',), equation), 'equation', ('i->i', False, b'i->i')),
            (c.ContractionTerm(('a',), equation), 'subscripts', ('i->i', None, False)),
            (c.Factor('a', (0,)), 'labels', ([0], (-1,), (True,), ('0',), (0.0,))),
            (c.PendingTerm((c.Factor('a', (0,)),), (0,)), 'output', ([0], (-1,), (True,), ('0',))),
            (c.BaryonJoinSpec(0, 'sink', 'G', 'E', 'ijk', 'abc'), 'spin_labels',
             ([0, 1, 2], (0, -1, 2), (False, 1, 2), ('0', 1, 2))),
        )
        for record, field_name, invalid_values in cases:
            cls = type(record)
            names, _, _, _, integer_fields = api._plan_record_schema(cls)
            constructor = self.bind(cls)
            for invalid in invalid_values:
                with self.subTest(record=cls.__name__, field=field_name, value=invalid):
                    decoded = {name: getattr(record, name) for name in names}
                    decoded[field_name] = invalid
                    with self.assertRaises(api.PlanArchiveError) as original:
                        api._construct_plan_record(cls, decoded, cls.__name__, integer_fields)
                    with self.assertRaises(api.PlanArchiveError) as actual:
                        constructor([decoded[name] for name in names])
                    self.assertEqual(str(actual.exception), str(original.exception))
        names = api._plan_record_schema(c.Action)[0]
        self.assertIsNone(api._bind_positional_record_constructor(c.Action, names, 'Action', ('positions',)))


    def test_archive_guards_reject_constructor_convenience_conversions(self):
        c = api.contract_api
        equation = c.AxisPattern(((0,),), (0,))
        cases = (
            (c.Action('contract', 'out', ('a',), equation), 'equation', 'i->i'),
            (c.ContractionTerm(('a',), equation), 'subscripts', 'i->i'),
            (c.Factor('a', (0,)), 'labels', [compact._LIST, 0]),
            (c.PendingTerm((c.Factor('a', (0,)),), (0,)), 'output', [compact._TUPLE, True]),
        )
        for record, name, replacement in cases:
            with self.subTest(record=type(record).__name__):
                encoded = deepcopy(compact.encode(record))
                payload = encoded['value']
                root = payload['root']
                shared = root[0] == compact._SHARED
                schema, offset = (root[2], 3) if shared else (root[0], 1)
                names = payload['schemas'][schema][1]
                if isinstance(replacement, str):
                    payload['strings'].append(replacement)
                    replacement = -len(payload['strings'])
                index=names.index(name)
                if len(payload['schemas'][schema])==3:
                    mask=root[offset]
                    self.assertTrue(mask&(1<<index))
                    offset+=1+(mask&((1<<index)-1)).bit_count()
                else:
                    offset+=index
                root[offset] = replacement
                with self.assertRaisesRegex(api.PlanArchiveError, 'integer axes|integer AxisPattern'):
                    api.decode_plan_value(encoded)

    def test_constructor_signature_proof_handles_initvars_and_unknown_signatures(self):
        @dataclass(frozen=True)
        class TailDefault:
            first: int
            last: int
            auxiliary: InitVar[int] = 7

            def __post_init__(self, auxiliary):
                object.__setattr__(self, 'actual_auxiliary', auxiliary)

        @dataclass(frozen=True)
        class MiddleDefault:
            first: int
            auxiliary: InitVar[int] = 7
            last: int = 0

        @dataclass(frozen=True)
        class UnknownSignature:
            first: int
            __signature__ = object()

        constructor = self.bind(TailDefault)
        self.assertIsNotNone(constructor)
        self.assertEqual(constructor((2, 3)), TailDefault(first=2, last=3))
        self.assertEqual(constructor((2, 3)).actual_auxiliary, 7)
        self.assertIsNone(self.bind(MiddleDefault))
        self.assertIsNone(self.bind(UnknownSignature))


if __name__ == '__main__':
    unittest.main()
