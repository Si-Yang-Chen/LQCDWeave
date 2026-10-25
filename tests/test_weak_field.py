"""CPU regression checks for the bundled weak_field reference data."""

import json
from pathlib import Path
import unittest

import numpy as np

from lqcdweave.production.elemental import ElementalGenerator
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "weak_field"


class ArrayHandle:
    def __init__(self, value):
        self.value = value

    def load(self, key):
        if key != "weak_field":
            raise ValueError(key)
        return self.value


class WeakFieldFixtureTests(unittest.TestCase):
    def test_array_shapes_and_dtypes_match_manifest(self):
        manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
        for filename, expected in manifest["arrays"].items():
            array = np.load(FIXTURE_DIR / filename, mmap_mode="r")
            self.assertEqual(list(array.shape), expected["shape"], filename)
            self.assertEqual(str(array.dtype), expected["dtype"], filename)

    def test_gauge_copy_and_input_basis(self):
        gauge = np.load(FIXTURE_DIR / "weak_field.npy", mmap_mode="r")
        raw_gauge = np.fromfile(FIXTURE_DIR / "weak_field.bin", dtype="<c16")
        np.testing.assert_array_equal(gauge.ravel(), raw_gauge)

        vectors = np.load(FIXTURE_DIR / "weak_field.eigenvector.input.npy", mmap_mode="r")
        self.assertEqual(vectors.shape, (8, 20, 4, 4, 4, 3))
        flattened = vectors.reshape(8, 20, -1)
        for time_slice in range(8):
            gram = flattened[time_slice].conj() @ flattened[time_slice].T
            np.testing.assert_allclose(gram, np.eye(20), atol=1e-10, rtol=0)

    def test_cpu_elemental_matches_archived_all_time_reference(self):
        gauge = np.load(FIXTURE_DIR / "weak_field.npy", mmap_mode="r")
        gauge = gauge.reshape(4, 8, 4, 4, 4, 3, 3).transpose(1, 2, 3, 4, 0, 5, 6)
        vectors = np.load(FIXTURE_DIR / "weak_field.eigenvector.input.npy", mmap_mode="r")
        vectors = vectors.astype(np.complex64)
        expected = np.load(FIXTURE_DIR / "weak_field.elemental.npy", mmap_mode="r")

        momenta = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1),
                   (0, 0, 2), (0, 1, 2), (1, 1, 2))
        generator = ElementalGenerator(
            (4, 4, 4, 8), ArrayHandle(gauge), ArrayHandle(vectors),
            num_nabla=2, momentum_list=momenta, calc_mode="calc_deriv",
            ne_block_size=20, array_module=np,
        ).load("weak_field")

        for time_slice in range(8):
            np.testing.assert_allclose(
                generator.calc_path_blocks(time_slice)["v2v"], expected[:, :, time_slice],
                atol=2e-12, rtol=2e-11,
            )


if __name__ == "__main__":
    unittest.main()
