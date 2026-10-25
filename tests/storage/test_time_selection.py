"""Integer-NPY persistence of actual temporal coordinates, not periodic templates."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lqcdweave.contracts.time_selection import TimeSelection, TimeSelectionRecord
from lqcdweave.storage.time_selection import load_time_selection, save_time_selection


class TimeSelectionStorageTests(unittest.TestCase):
    def test_roundtrip_keeps_wraparound_order_and_existing_file_bytes(self):
        times = (104, 105, 108, 113, 115, 120, 121, 124, 1, 3)
        record = TimeSelectionRecord(128, times)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '01.select_t5x8' / '2000.npy'
            save_time_selection(path, record)
            saved = path.read_bytes()
            array = np.load(path, allow_pickle=False)
            self.assertEqual(array.dtype, np.dtype('int64'))
            np.testing.assert_array_equal(array, times)
            loaded = load_time_selection(path, Lt=128)
            self.assertEqual(loaded, record)
            with self.assertRaises(FileExistsError):
                save_time_selection(path, TimeSelectionRecord(128, (2, 3)))
            self.assertEqual(path.read_bytes(), saved)

    def test_existing_integer_npy_requires_explicit_physical_lt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '2010.npy'
            np.save(path, np.array([109, 110, 1, 6], dtype=np.int32))
            loaded = load_time_selection(path, Lt=128)
            self.assertEqual(loaded.times, (109, 110, 1, 6))
            self.assertEqual(loaded.Lt, 128)
            self.assertEqual(loaded.Nt, 4)
            with self.assertRaises(TypeError):
                load_time_selection(path)
            with self.assertRaises(ValueError):
                load_time_selection(path, Lt=40)

    def test_periodic_template_can_be_saved_and_read_as_actual_times(self):
        template = TimeSelection(8, 4, (0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'times.npy'
            save_time_selection(path, template)
            loaded = load_time_selection(path, Lt=8)
            self.assertIsInstance(loaded, TimeSelectionRecord)
            self.assertEqual(loaded.times, template.times)

    def test_invalid_saved_arrays_and_pickle_are_rejected(self):
        invalid = [np.array([], dtype=np.int64), np.array([0, 0]), np.array([-1, 2]),
                   np.array([0, 128]), np.array([0., 1.]), np.array([True, False]),
                   np.array([[0, 1]]), np.array([{'time': 0}], dtype=object)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.npy'
            for array in invalid:
                with self.subTest(array=array):
                    np.save(path, array)
                    with self.assertRaises(ValueError):
                        load_time_selection(path, Lt=128)

    def test_record_validation_and_immutable_owned_order(self):
        source = [7, 0, 3, 1]
        record = TimeSelectionRecord(8, source)
        source[0] = 2
        self.assertEqual(record.times, (7, 0, 3, 1))
        with self.assertRaises(FrozenInstanceError):
            record.Lt = 9
        for times in ((), (0, 0), (-1,), (8,), (True,), (1.,)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                TimeSelectionRecord(8, times)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'not-created.npy'
            with self.assertRaises(TypeError):
                save_time_selection(path, (0, 1))
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
