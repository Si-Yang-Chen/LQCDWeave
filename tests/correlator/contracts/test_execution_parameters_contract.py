"""Execution policy data remain JSON-native and detached from caller state."""
import unittest
from collections import UserDict

from lqcdweave.contracts.execution import (
    DirectExecution, TermValue, normalize_execution_parameters,
)
from lqcdweave.contracts.source import source_origin_from_registration


class ExecutionParametersContractTests(unittest.TestCase):
    def test_producer_schema_four_never_becomes_an_external_source_claim(self):
        marker = {'schema': 1, 'kind': 'external_import'}
        for schema in (1, 2, 4):
            self.assertEqual(source_origin_from_registration(marker, {'schema': schema}),
                {'status': 'unknown', 'reason': 'legacy-unstructured-provenance'})
        for execution in ({'schema': 4, 'registration_event': {}}, {'schema': 5}):
            with self.assertRaises(ValueError):
                source_origin_from_registration(marker, execution)

    def test_detaches_shared_mapping_and_lists_without_changing_values(self):
        shared = [None, True, 3, 2.5, '中文', {'width': 2}]
        source = UserDict(first=shared, second=shared)
        result = normalize_execution_parameters(source)
        self.assertEqual(result, dict(source))
        self.assertIsNot(result['first'], shared)
        self.assertIsNot(result['first'], result['second'])
        shared[-1]['width'] = 4
        result['first'][-1]['width'] = 5
        self.assertEqual(result['second'][-1]['width'], 2)
        self.assertIsNone(normalize_execution_parameters(None))
        self.assertEqual(normalize_execution_parameters({}), {})

    def test_refuses_coercion_and_nonfinite_values(self):
        for value in ([1], 1, 'width', True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_execution_parameters(value)
        for child in ((1, 2), {1: 'width'}, {1, 2}, object()):
            with self.subTest(child=repr(child)), self.assertRaises(TypeError):
                normalize_execution_parameters({'child': child})
        for child in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(child=repr(child)), self.assertRaises(ValueError):
                normalize_execution_parameters({'child': child})
        for value in ({'child': '\ud800'}, {'\udfff': 'width'}):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                normalize_execution_parameters(value)

    def test_rejects_cycles_but_keeps_existing_execution_types(self):
        mapping = {}
        mapping['self'] = mapping
        sequence = []
        sequence.append(sequence)
        for value in (mapping, {'sequence': sequence}):
            with self.assertRaises(ValueError):
                normalize_execution_parameters(value)
        value = TermValue('term', 'coordinate', 4)
        self.assertEqual(DirectExecution((value,)).term_values, (value,))


if __name__ == '__main__':
    unittest.main()
