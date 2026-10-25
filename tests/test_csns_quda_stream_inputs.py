"""Resolve the fixed QUDA comparison inputs before submitting a GPU job."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from csns_quda_stream import mode_handle


class QudaStreamInputTests(unittest.TestCase):
    def test_weak_field_key_opens_one_mode_file(self):
        fixture = Path(__file__).resolve().parent / "data" / "weak_field"
        handle = mode_handle(fixture)
        loaded = handle.load("weak_field")
        self.assertEqual(loaded.shape, (8,20,4,4,4,3))
        np.testing.assert_array_equal(handle.read_modes("weak_field",3,2,5),
                                      loaded[3,2:5])


if __name__ == "__main__":
    unittest.main()
