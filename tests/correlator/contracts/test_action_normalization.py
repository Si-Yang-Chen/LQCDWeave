"""Action normalization preserves immutable native fields and extension hooks."""
from dataclasses import fields
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator import scene


class ActionNormalizationTests(unittest.TestCase):
    def test_native_records_preserve_every_field_and_object(self):
        pattern = scene.AxisPattern(((0, 1), (1, 2)), (0, 2))
        operands, core = ('left', 'right'), object()
        arguments = dict(kind='contract', tensor='out', operands=operands, equation=pattern,
                         seconds=0.125, reused=True, used_np=7, join_id='join',
                         left_axes=(-3, -2, -1), right_axes=(0, 1, 2), same_point=True,
                         transpose_bridge=True, term_index=3, positions=(1, 0),
                         kernel_path=((0, 1),), used_np_right=5, sparse_spin=None,
                         core=core, resource=None, wide_accumulation=False,
                         kernel_description=None)
        with (patch.object(scene, 'parse_axis_pattern', side_effect=AssertionError('already parsed')),
              patch.object(scene, '_sparse_spin_entries', side_effect=AssertionError('absent'))):
            action = scene.Action(**arguments)
            loaded = scene.Action('load', 'input')
        self.assertEqual({field.name: getattr(action, field.name) for field in fields(action)}, arguments)
        self.assertIs(action.operands, operands)
        self.assertIs(action.equation, pattern)
        self.assertIs(action.core, core)
        self.assertEqual(loaded.operands, ())
        self.assertIsNone(loaded.equation)
        self.assertIsNone(loaded.sparse_spin)

    def test_external_sequences_and_equations_keep_normalization(self):
        pattern = scene.AxisPattern(((97, 98), (98, 99)), (97, 99))
        for operands in (['left', 'right'], iter(('left', 'right'))):
            action = scene.Action('contract', 'out', operands, 'ab,bc->ac')
            self.assertEqual(action.operands, ('left', 'right'))
            self.assertIs(type(action.operands), tuple)
            self.assertEqual(action.equation, pattern)
        arguments = ['left', 'right']
        action = scene.Action('contract', 'out', arguments, '')
        arguments.clear()
        self.assertEqual(action.operands, ('left', 'right'))
        self.assertIsNone(action.equation)
        with self.assertRaisesRegex(TypeError, 'AxisPattern'):
            scene.Action('contract', 'out', (), 123)

    def test_optional_records_keep_conversion_and_failure_boundaries(self):
        resource = scene.ResourceTransition('tensor', 'host', 'device', 16)
        spin = [[0, 1, 2, 2], [1, 0, 3, -1j]]
        action = scene.Action('sparse_contract', 'out', sparse_spin=spin, resource=resource)
        spin[0][3] = 99
        self.assertEqual(action.sparse_spin, ((0, 1, 2, 2+0j), (1, 0, 3, -1j)))
        self.assertIs(action.resource, resource)
        with self.assertRaisesRegex(TypeError, 'ResourceTransition'):
            scene.Action('load', 'input', resource=object())
        for entries, message in (
            (((0, 0, 0, 0),), 'nonzero'),
            (((0, 0, 0, float('inf')),), 'finite'),
            (((4, 0, 0, 1),), '0..3'),
            (((0, 0, 0, 1), (0, 0, 0, 2)), 'unique'),
        ):
            with self.subTest(entries=entries), self.assertRaisesRegex(ValueError, message):
                scene.Action('sparse_contract', 'out', sparse_spin=entries)

    def test_action_and_axis_extensions_keep_original_hooks(self):
        assignments = []

        class ExtendedAction(scene.Action):
            @property
            def operands(self):
                return self._operands

            @operands.setter
            def operands(self, value):
                assignments.append(value)
                object.__setattr__(self, '_operands', value)

        class ExtendedPattern(scene.AxisPattern):
            pass

        pattern = ExtendedPattern(((0,),), ())
        with (patch.object(scene, 'parse_axis_pattern', wraps=scene.parse_axis_pattern) as parse,
              patch.object(scene, '_sparse_spin_entries', wraps=scene._sparse_spin_entries) as spin):
            action = ExtendedAction('contract', 'out', ('input',), pattern)
            native_action = scene.Action('contract', 'out', ('input',), pattern)
        self.assertEqual(assignments, [('input',), ('input',)])
        self.assertEqual(parse.call_count, 2)
        self.assertEqual(spin.call_count, 2)
        self.assertIs(action.equation, pattern)
        self.assertIs(native_action.equation, pattern)


if __name__ == '__main__':
    unittest.main()
