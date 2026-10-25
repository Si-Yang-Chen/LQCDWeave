"""Only immutable conventions are reused; fresh reader headers remain mandatory."""
from dataclasses import replace
import gc
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from lqcdweave.contracts.correlator.api import SourceReadRequest, TensorAxisMetadata
from lqcdweave.contraction.expansion import saved_inputs
from lqcdweave.contraction.expansion.api import restore_input_loader


class Reader:
    def __init__(self):
        self.cfg = 'cfg'
        self.time_convention = 'absolute'
        self.momenta = ((0, 0, 0), (1, 0, 0))
        self.time_selection_identity = (4, (3, 1))
        self.metadata = TensorAxisMetadata(('mode',), (2,), 'complex128', 'actual', 'ordered',
                                          axis_bindings={'mode': ('basis', (0, 1))})
        self.calls = []

    def describe_block(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.metadata

    def block(self, *args, **kwargs):
        return np.asarray([1 + 2j, 3 - 4j], dtype=self.metadata.dtype)


def request(reader):
    return SourceReadRequest('reader', 'propagator', (0, 1, 'vsv'), (), reader.metadata,
        conventions=(('time_convention', 'absolute'), ('momenta', reader.momenta),
                     ('time_selection_identity', reader.time_selection_identity)))


class SavedInputValidationCacheTests(unittest.TestCase):
    def restore(self, reader, req, cache, **options):
        return restore_input_loader({'raw': req}, {'reader': reader}, {}, {},
                                    configuration='cfg', convention_cache=cache, **options)

    def test_immutable_conventions_checked_once_across_binding_calls(self):
        reader = Reader()
        req = request(reader)
        cache = {}
        with patch.object(saved_inputs, '_check_conventions', wraps=saved_inputs._check_conventions) as check:
            first = self.restore(reader, req, cache)
            second = self.restore(reader, replace(req, arguments=(1, 0, 'vsv')), cache)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(len(reader.calls), 2)
        self.assertEqual(reader.calls[1][0], (1, 0, 'vsv'))
        self.assertIsNot(first.metadata, second.metadata)
        np.testing.assert_array_equal(second.read_block('raw', None), [1 + 2j, 3 - 4j])

    def test_default_loader_deduplicates_repeated_reader_conventions(self):
        reader = Reader()
        req = request(reader)
        with patch.object(saved_inputs, '_check_conventions', wraps=saved_inputs._check_conventions) as check:
            restore_input_loader({str(i): replace(req, arguments=(i, 0, 'vsv')) for i in range(6)},
                                 {'reader': reader}, {}, {}, configuration='cfg')
        self.assertEqual(check.call_count, 1)
        self.assertEqual(len(reader.calls), 6)

    def test_mutable_array_and_list_conventions_cannot_hit_by_identity(self):
        for make in (lambda: [[0, 0, 0], [1, 0, 0]],
                     lambda: np.asarray([[0, 0, 0], [1, 0, 0]])):
            with self.subTest(container=make):
                reader = Reader()
                reader.momenta = make()
                req = request(reader)
                cache = {}
                with patch.object(saved_inputs, '_check_conventions', wraps=saved_inputs._check_conventions) as check:
                    self.restore(reader, req, cache)
                    self.restore(reader, req, cache)
                    reader.momenta[0][0] = 9
                    with self.assertRaisesRegex(ValueError, 'conventions'):
                        self.restore(reader, req, cache)
                self.assertEqual(check.call_count, 3)
                self.assertEqual(cache, {})

    def test_changed_immutable_declarations_cfg_and_reader_handle_are_rechecked(self):
        for name, value in (('time_convention', 'relative'), ('momenta', ((1, 0, 0), (0, 0, 0))),
                            ('time_selection_identity', (4, (1, 3))), ('cfg', 'other')):
            with self.subTest(field=name):
                reader = Reader()
                req = request(reader)
                cache = {}
                self.restore(reader, req, cache)
                setattr(reader, name, value)
                with self.assertRaisesRegex(ValueError, 'conventions|configuration'):
                    self.restore(reader, req, cache)
                self.assertEqual(len(reader.calls), 1)
        reader = Reader()
        cache = {}
        with patch.object(saved_inputs, '_check_conventions', wraps=saved_inputs._check_conventions) as check:
            self.restore(reader, request(reader), cache)
            other = Reader()
            self.restore(other, request(other), cache)
        self.assertEqual(check.call_count, 2)

    def test_shape_dtype_axes_selection_and_identity_checks_survive_hits(self):
        for field, value, message in (
                ('shape', (3,), 'axes, shape, dtype or layout'),
                ('dtype', 'complex64', 'axes, shape, dtype or layout'),
                ('axes', ('different_mode',), 'axes, shape, dtype or layout'),
                ('axis_bindings', {'mode': ('basis', (1, 0))}, 'ordered endpoint coordinates'),
                ('data_identity', None, 'source sharing')):
            with self.subTest(field=field):
                reader = Reader()
                req = request(reader)
                cache = {}
                self.restore(reader, req, cache)
                if field in ('shape', 'axes'):
                    reader.metadata = replace(reader.metadata, **{field: value}, axis_bindings={})
                else:
                    reader.metadata = replace(reader.metadata, **{field: value})
                with self.assertRaisesRegex(ValueError, message):
                    self.restore(reader, req, cache)
                self.assertEqual(len(reader.calls), 2)

    def test_new_selection_is_described_and_cannot_reuse_prior_header(self):
        reader = Reader()
        req = request(reader)
        cache = {}
        self.restore(reader, req, cache)
        selected = replace(req, selection=(('mode_indices', (('mode', (1, 0)),)),),
                           metadata=replace(req.metadata, axis_bindings={'mode': ('basis', (1, 0))}))
        with self.assertRaisesRegex(ValueError, 'ordered endpoint coordinates'):
            self.restore(reader, selected, cache)
        self.assertEqual(reader.calls[-1][1], {'mode_indices': {'mode': (1, 0)}})
        reader.metadata = selected.metadata
        self.restore(reader, selected, cache)
        self.assertEqual(len(reader.calls), 3)

    def test_numeric_read_shape_and_dtype_remain_checked_after_cache_hit(self):
        reader = Reader()
        req = request(reader)
        cache = {}
        self.restore(reader, req, cache)
        loader = self.restore(reader, req, cache)
        for value in (np.zeros(3, dtype='complex128'), np.zeros(2, dtype='complex64')):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with patch.object(reader, 'block', return_value=value):
                    with self.assertRaisesRegex(ValueError, 'block differs'):
                        loader.read_block('raw', None)

    def test_cache_is_bounded_and_does_not_own_reader_lifetimes(self):
        cache = {}
        readers = [Reader() for _ in range(saved_inputs._ConventionChecks.limit + 7)]
        for reader in readers:
            self.restore(reader, request(reader), cache)
        self.assertLessEqual(len(cache), saved_inputs._ConventionChecks.limit)
        reference = weakref.ref(reader)
        del reader, readers
        gc.collect()
        self.assertIsNone(reference())

    def test_replaced_request_with_mutable_convention_uses_safe_fallback(self):
        class MutableConvention:
            def __init__(self, values):
                self.values = values
            def __array__(self, dtype=None, copy=None):
                return np.asarray(self.values, dtype=dtype)
            def __eq__(self, other):
                raise AssertionError('unknown equality used by convention cache')
        reader = Reader()
        req = request(reader)
        cache = {}
        self.restore(reader, req, cache)
        mutable = MutableConvention(reader.momenta)
        changed = replace(req, conventions=tuple((name, mutable if name == 'momenta' else value)
                                                for name, value in req.conventions))
        self.restore(reader, changed, cache)
        self.assertEqual(cache, {})
        mutable.values = ((9, 0, 0), (1, 0, 0))
        with self.assertRaisesRegex(ValueError, 'conventions'):
            self.restore(reader, changed, cache)

    def test_property_snapshot_is_the_value_that_was_validated(self):
        class ChangingReader(Reader):
            @property
            def time_convention(self):
                self.accesses += 1
                return 'relative' if self.accesses == 1 else 'absolute'
            @time_convention.setter
            def time_convention(self, value):
                self.accesses = 0
        reader = ChangingReader()
        req = request(reader)
        with self.assertRaisesRegex(ValueError, 'conventions'):
            self.restore(reader, req, {})
        self.assertEqual(reader.accesses, 1)


if __name__ == '__main__':
    unittest.main()
