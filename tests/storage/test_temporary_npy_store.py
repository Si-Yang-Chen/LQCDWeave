"""Owned scratch-array restores without a second full payload allocation."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from lqcdweave.storage.dataset import TemporaryNpyArrayStore


class _PayloadReader:
    def __init__(self, stream, *, chunk_size=None, before_payload=None):
        self.stream = stream
        self.descriptor = stream.fileno()
        self.chunk_size = chunk_size
        self.before_payload = before_payload
        self.targets = []

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *exc):
        return self.stream.__exit__(*exc)

    def readinto(self, target):
        if not self.targets and self.before_payload is not None:
            self.before_payload(self.stream)
        self.targets.append(target)
        if self.chunk_size is not None:
            target = target[:self.chunk_size]
        return self.stream.readinto(target)


@contextmanager
def _payload_readers(**options):
    original_open = Path.open
    readers = []

    def open_reader(path, *args, **kwargs):
        reader = _PayloadReader(original_open(path, *args, **kwargs), **options)
        readers.append(reader)
        return reader

    with mock.patch.object(Path, "open", open_reader):
        yield readers


class TemporaryNpyStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = TemporaryNpyArrayStore(directory.name)
        self.scope = self.store.open_scope()
        self.addCleanup(self.scope.close)

    def test_roundtrip_preserves_order_dtype_values_and_ownership(self):
        cases = [((2, 3, 4), dtype) for dtype in
                 ("?", "i2", "u4", "f4", "f8", "c8", "c16", ">f8", ">c16")]
        cases += [((), ">c16"), ((0, 3), "f8"), ((2, 0, 4), ">c16")]
        for shape, dtype in cases:
            for order in ("C", "F"):
                with self.subTest(shape=shape, dtype=dtype, order=order):
                    base = np.arange(int(np.prod(shape))).reshape(shape)
                    if np.dtype(dtype).kind == "c":
                        base = base + 0.25 + 1j * (base + 0.5)
                    elif np.dtype(dtype).kind == "b":
                        base = base % 2
                    elif np.dtype(dtype).kind == "f":
                        base = base + 0.25
                    values = np.array(base, dtype=dtype, order=order)
                    receipt = self.scope.write_array(values)

                    restored = self.scope.read_array(receipt, shape=shape, dtype=dtype)

                    np.testing.assert_array_equal(restored, values)
                    self.assertEqual(restored.shape, shape)
                    self.assertEqual(restored.dtype.str, values.dtype.str)
                    self.assertTrue(restored.flags.owndata)
                    self.assertIsNone(restored.base)
                    self.assertEqual(restored.flags.c_contiguous, values.flags.c_contiguous)
                    self.assertEqual(restored.flags.f_contiguous, values.flags.f_contiguous)
                    self.assertFalse(np.shares_memory(restored, values))
                    self.scope.remove_array(receipt)

    def test_payload_reads_into_the_only_allocation_on_the_validated_descriptor(self):
        original_empty = np.empty
        for order in ("C", "F"):
            with self.subTest(order=order):
                values = np.array(np.arange(24).reshape(2, 3, 4), dtype=">c16", order=order)
                receipt = self.scope.write_array(values)
                allocations = []

                def allocate(*args, **kwargs):
                    value = original_empty(*args, **kwargs)
                    allocations.append(value)
                    return value

                with _payload_readers() as readers, \
                        mock.patch("numpy.empty", side_effect=allocate) as empty, \
                        mock.patch("numpy.load", side_effect=AssertionError("loader staging")), \
                        mock.patch("numpy.fromfile", side_effect=AssertionError("flat staging")), \
                        mock.patch("numpy.frombuffer", side_effect=AssertionError("buffer staging")), \
                        mock.patch("numpy.array", side_effect=AssertionError("array copy")), \
                        mock.patch("numpy.copy", side_effect=AssertionError("payload copy")), \
                        mock.patch("os.fstat", wraps=os.fstat) as fstat:
                    restored = self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
                    empty.assert_called_once_with(values.shape, dtype=values.dtype, order=order)
                    self.assertEqual(len(readers), 1)
                    fstat.assert_called_once_with(readers[0].descriptor)
                    self.assertIs(restored, allocations[0])
                    self.assertTrue(readers[0].stream.closed)
                    self.assertTrue(readers[0].targets)
                    for target in readers[0].targets:
                        self.assertIsInstance(target, memoryview)
                        self.assertEqual(target.format, "B")
                        self.assertTrue(target.c_contiguous)
                        self.assertTrue(np.shares_memory(target.obj, restored))
                np.testing.assert_array_equal(restored, values)
                self.scope.remove_array(receipt)

    def test_partial_reads_fill_the_complete_c_or_fortran_payload(self):
        for order in ("C", "F"):
            with self.subTest(order=order):
                values = np.array(np.arange(24).reshape(2, 3, 4), dtype=">f8", order=order)
                receipt = self.scope.write_array(values)
                with _payload_readers(chunk_size=7) as readers:
                    restored = self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
                self.assertGreater(len(readers[0].targets), 1)
                self.assertEqual(readers[0].targets[0].nbytes, values.nbytes)
                self.assertEqual(readers[0].targets[1].nbytes, values.nbytes - 7)
                np.testing.assert_array_equal(restored, values)
                self.assertTrue(restored.flags.owndata)
                self.scope.remove_array(receipt)

    def test_zero_byte_restore_accepts_c_and_fortran_headers_without_payload_reads(self):
        values = np.empty((2, 0, 4), dtype=">c16", order="F")
        receipt = self.scope.write_array(values)
        for write_header in (np.lib.format.write_array_header_1_0,
                             np.lib.format.write_array_header_2_0):
            for fortran_order in (False, True):
                with self.subTest(header=write_header.__name__, fortran_order=fortran_order):
                    with (self.scope._root / receipt.filename).open("wb") as stream:
                        write_header(stream, {"descr": values.dtype.str,
                                              "fortran_order": fortran_order,
                                              "shape": values.shape})
                    with _payload_readers() as readers, \
                            mock.patch("numpy.empty", wraps=np.empty) as empty:
                        restored = self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
                    empty.assert_called_once_with(values.shape, dtype=values.dtype,
                                                  order="F" if fortran_order else "C")
                    self.assertEqual(readers[0].targets, [])
                    self.assertEqual(restored.shape, values.shape)
                    self.assertEqual(restored.nbytes, 0)
                    self.assertTrue(restored.flags.owndata)

    def test_truncated_or_extra_payload_fails_before_allocation(self):
        values = np.arange(12, dtype=np.float64)
        for change in ("truncate", "append"):
            with self.subTest(change=change):
                receipt = self.scope.write_array(values)
                path = self.scope._root / receipt.filename
                payload = path.read_bytes()
                path.write_bytes(payload[:-1] if change == "truncate" else payload + b"x")
                with mock.patch("numpy.empty", side_effect=AssertionError("unexpected allocation")):
                    with self.assertRaisesRegex(ValueError, "payload size"):
                        self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
                self.scope.remove_array(receipt)

    def test_truncation_after_size_check_rejects_a_partial_owned_array(self):
        values = np.arange(12, dtype=np.float64)
        receipt = self.scope.write_array(values)
        path = self.scope._root / receipt.filename
        original_open = Path.open

        def truncate_payload(stream):
            with original_open(path, "r+b") as writer:
                writer.truncate(stream.tell() + 13)

        with _payload_readers(before_payload=truncate_payload) as readers:
            with self.assertRaisesRegex(ValueError, "payload is truncated"):
                self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
        self.assertEqual(len(readers[0].targets), 2)
        self.assertEqual(readers[0].targets[1].nbytes, values.nbytes - 13)
        self.assertTrue(readers[0].stream.closed)

    def test_receipt_and_header_mismatches_fail_before_allocation(self):
        values = np.arange(6, dtype=np.float64).reshape(2, 3)
        for change in ("receipt_shape", "receipt_dtype", "receipt_size", "header_shape", "header_dtype"):
            with self.subTest(change=change):
                receipt = self.scope.write_array(values)
                path = self.scope._root / receipt.filename
                if change == "receipt_shape":
                    receipt = replace(receipt, shape=(3, 2))
                elif change == "receipt_dtype":
                    receipt = replace(receipt, dtype="<i8")
                elif change == "receipt_size":
                    receipt = replace(receipt, nbytes=receipt.nbytes + 1)
                elif change == "header_shape":
                    np.save(path, values.reshape(3, 2), allow_pickle=False)
                else:
                    np.save(path, values.astype(np.int64), allow_pickle=False)
                with mock.patch("numpy.empty", side_effect=AssertionError("unexpected allocation")):
                    with self.assertRaisesRegex(ValueError, "disagrees"):
                        self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
                self.scope.remove_array(receipt)

    def test_restores_have_no_shared_owner_and_survive_file_and_scope_removal(self):
        values = np.array([[1 + 2j, 3 - 4j], [-5 + 6j, 7 + 8j]], order="F")
        receipt = self.scope.write_array(values)
        first = self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
        second = self.scope.read_array(receipt, shape=values.shape, dtype=values.dtype)
        self.assertFalse(np.shares_memory(first, second))
        self.scope.remove_array(receipt)
        self.scope.close()
        np.testing.assert_array_equal(first, values)
        np.testing.assert_array_equal(second, values)
        first[0, 0] = 123j
        np.testing.assert_array_equal(second, values)
        self.assertTrue(first.flags.owndata)
        self.assertTrue(second.flags.owndata)


if __name__ == "__main__":
    unittest.main()
