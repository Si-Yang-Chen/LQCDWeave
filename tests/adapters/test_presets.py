"""Legacy frontend ownership, header checks and bounded mode access."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave import preset
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = str(self.root / "cfg_")

    def save(self, array, cfg="1", suffix=".npy"):
        path = Path(f"{self.prefix}{cfg}{suffix}")
        np.save(path, array)
        return path

    def test_declaration_is_lazy_and_repeated_load_reuses_reader(self):
        with patch("numpy.load", side_effect=AssertionError("declaration opened input")):
            source = preset.PerambulatorNpy(self.prefix, ".npy", (2, 2, 4, 4, 3, 3), 3)
        self.save(np.ones((2, 2, 4, 4, 3, 3), dtype=np.complex64))
        with patch("numpy.load", wraps=np.load) as loading, patch(
                "numpy.array", side_effect=AssertionError("load copied payload")):
            reader = source.load("1")
            self.assertIs(reader, source.load("1"))
            self.assertEqual(loading.call_count, 1)
            self.assertEqual(reader.dtype, np.dtype("complex64"))
            self.assertIsNone(reader.data_identity)
            metadata = reader.describe_block(0, 1, usedNe=2)
            self.assertEqual(metadata.shape, (4, 4, 2, 2))
        reader.close()

    def test_relative_time_and_old_indexing_have_distinct_contracts(self):
        shape = (3, 3, 4, 4, 2, 2)
        values = np.arange(np.prod(shape)).reshape(shape).astype(complex)
        self.save(values)
        source = preset.PerambulatorNpy(self.prefix, ".npy", shape, 2)
        reader = source.load("1")
        block = reader.block(0, 1, usedNe=1)
        np.testing.assert_array_equal(block, values[1, 2, :, :, :1, :1])
        self.assertFalse(block.flags.writeable)
        old_slice = reader[1, 2]
        self.assertTrue(old_slice.flags.owndata)
        old_slice[:] = 0
        np.testing.assert_array_equal(reader[1, 2], values[1, 2])
        reader.close()
        self.assertIsNot(source.load("1"), reader)
        source.load("1").close()

    def test_reload_keeps_previous_configuration_bound_and_rechecks_replacement(self):
        first = np.ones((2, 2, 4, 4, 1, 1), complex)
        self.save(first)
        self.save(2 * first, "2")
        source = preset.PerambulatorNpy(self.prefix)
        old = source.load("1")
        new = source.load("2")
        np.testing.assert_array_equal(old.block(0, 0), first[0, 0])
        np.testing.assert_array_equal(new.block(0, 0), 2 * first[0, 0])
        old.close()
        new.close()
        # Replacing a cached path must not conceal its changed header.
        source.load("1").close()
        self.save(np.zeros((2, 2, 3, 4, 1, 1), complex))
        with self.assertRaisesRegex(ValueError, "spin"):
            source.load("1")

    def test_backend_rejects_bad_header_shape_dtype_count_and_length(self):
        values = np.ones((2, 2, 4, 4, 2, 2), dtype=np.complex64)
        path = self.save(values)
        for options, message in ((dict(shape=(2, 2, 4, 4, 1, 1)), "shape"),
                                 (dict(dtype="complex128"), "dtype"),
                                 (dict(totNe=1), "Ne")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                preset.PerambulatorNpy(self.prefix, **options).load("1")
        with path.open("ab") as stream:
            stream.write(b"trailing bytes")
        with self.assertRaisesRegex(ValueError, "length"):
            preset.PerambulatorNpy(self.prefix).load("1")
        self.save(values.real)
        with self.assertRaisesRegex(ValueError, "complex"):
            preset.PerambulatorNpy(self.prefix).load("1")
        self.save(np.array([object()], dtype=object))
        with self.assertRaises(ValueError):
            preset.PerambulatorNpy(self.prefix).load("1")

    def test_elemental_default_rows_and_explicit_components(self):
        shape = (2, 3, 4, 2, 2)
        values = np.arange(np.prod(shape)).reshape(shape).astype(complex)
        self.save(values)
        source = preset.ElementalNpy(self.prefix, ".npy", shape, 2)
        reader = source.load("1")
        self.assertEqual(reader.components, (ElementalComponent(0), ElementalComponent(1)))
        np.testing.assert_array_equal(reader.block(ElementalComponent(1), 2, 3, "v2v"), values[1, 2, 3])
        self.assertTrue(reader[1, 2, 3].flags.owndata)
        self.assertIsNone(reader.data_identity)
        components = (ElementalComponent.path(GaugePath()), ElementalComponent.path(GaugePath((0,))))
        declared = preset.ElementalNpy(self.prefix, components=components,
                                      phase_convention="anchor").load("1")
        np.testing.assert_array_equal(declared.block(components[1], 2, 3, "v2v"), values[1, 2, 3])
        with self.assertRaisesRegex(ValueError, "shape"):
            preset.ElementalNpy(self.prefix, components=components[:1]).load("1")
        reader.close()
        declared.close()

    def test_current_elemental_uses_legacy_time_major_layout(self):
        values = np.arange(4*2*3*2*2).reshape(4, 2, 3, 2, 2).astype(complex)
        self.save(values, suffix="_v2v.npy")
        reader = preset.CurrentElementalV2V(self.prefix, shape=values.shape, Ne=2).load("1")
        self.assertEqual(reader.shape, (2, 3, 4, 2, 2))
        np.testing.assert_array_equal(reader[1, 2, 3], values[3, 1, 2])
        reader.close()

    def test_canonical_eigenvectors_borrow_only_requested_modes(self):
        values = np.arange(4*3*2*2*2*3).reshape(4, 3, 2, 2, 2, 3).astype(complex)
        self.save(values)
        source = preset.EigenvectorNpy(self.prefix, ".npy", values.shape, 3)
        with patch("numpy.array", side_effect=AssertionError("full basis copy")):
            loaded = source.load("1")
            block = source.read_modes("1", 2, 1, 3)
            self.assertTrue(np.shares_memory(block, loaded))
        np.testing.assert_array_equal(block, values[2, 1:3])
        for args in ((4, 0, 1), (0, 2, 1), (0, 0, 4)):
            with self.assertRaises(ValueError):
                source.read_modes("1", *args)
        del block, loaded, source

    def test_old_flat_eigenvectors_need_geometry_and_never_copy_whole_file(self):
        canonical = np.arange(4*3*2*2*2*3).reshape(4, 3, 2, 2, 2, 3).astype(complex)
        flat = canonical.reshape(4, 3, 8, 3).transpose(1, 0, 2, 3).copy()
        self.save(flat)
        with self.assertRaisesRegex(ValueError, "lattice_size"):
            preset.EigenvectorNpy(self.prefix).load("1")
        source = preset.EigenvectorNpy(self.prefix, ".npy", flat.shape, 3,
                                      lattice_size=(2, 2, 2, 4))
        with patch("numpy.array", side_effect=AssertionError("basis copy")):
            loaded = source.load("1")
        np.testing.assert_array_equal(loaded, canonical)
        del loaded, source
        with self.assertRaisesRegex(ValueError, "lattice_size"):
            preset.EigenvectorNpy(self.prefix, lattice_size=(4, 2, 2, 4)).load("1")


if __name__ == "__main__":
    unittest.main()
