"""Small CPU roundtrips for the purpose-first, dense NPY convenience API."""

from pathlib import Path
import os
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave import preset
from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent


class PresetIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def values(self, shape, dtype):
        data = np.arange(np.prod(shape)).reshape(shape)
        if np.dtype(dtype).kind == "c":
            data = data + 1j * (data + 2)
        return data.astype(dtype)

    def raw_save(self, path, data):
        with path.open("wb") as stream:
            np.save(stream, data)

    def close_array(self, array):
        array._mmap.close()

    def test_all_fields_roundtrip_header_and_readonly_views(self):
        cases = {
            "gauge": ((2, 1, 2, 3, 4, 3, 3), ">c8", {}),
            "eigenvector": ((2, 3, 1, 2, 4, 3), "<c16", {}),
            "eigenvalues": ((2, 3), ">f8", {}),
            "points": ((2, 4, 3), ">i4", {}),
            "overlap": ((2, 3, 4, 3), ">c16", {}),
        }
        for kind, (shape, dtype, options) in cases.items():
            with self.subTest(kind=kind):
                data = self.values(shape, dtype)
                path = self.root / kind  # No implicit .npy suffix.
                self.assertEqual(preset.save(path, data, kind=kind, **options), path)
                self.assertFalse(path.with_suffix(".npy").exists())
                with patch("numpy.array", side_effect=AssertionError("whole-array copy")):
                    loaded = preset.load(path, kind=kind, shape=shape, dtype=dtype, **options)
                self.addCleanup(self.close_array, loaded)
                self.assertIsInstance(loaded, np.memmap)
                self.assertFalse(loaded.flags.writeable)
                self.assertEqual(loaded.dtype.str, data.dtype.str)
                np.testing.assert_array_equal(loaded, data)

    def test_fields_explicit_axes_preserve_disk_order_and_return_canonical(self):
        cases = (
            ("gauge", (2, 1, 2, 3, 4, 3, 3),
             ("direction", "time", "z", "y", "x", "color_left", "color_right"),
             (4, 0, 1, 2, 3, 5, 6)),
            ("eigenvector", (2, 3, 1, 2, 4, 3),
             ("mode", "time", "z", "y", "x", "color"), (1, 0, 2, 3, 4, 5)),
            ("eigenvalues", (2, 3), ("mode", "time"), (1, 0)),
            ("overlap", (2, 3, 4, 3), ("point", "time", "mode", "color"), (2, 0, 1, 3)),
        )
        for kind, shape, order, permutation in cases:
            with self.subTest(kind=kind):
                dtype = "int64" if kind == "points" else "float32" if kind == "eigenvalues" else "complex64"
                canonical = self.values(shape, dtype)
                disk = canonical.transpose(permutation)
                path = preset.save(self.root / kind, disk, kind=kind, axis_order=order)
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), disk)
                loaded = preset.load(path, kind=kind, axis_order=order, shape=disk.shape)
                self.addCleanup(self.close_array, loaded)
                self.assertEqual(loaded.shape, canonical.shape)
                np.testing.assert_array_equal(loaded, canonical)

    def test_points_npy_uses_only_the_canonical_time_first_order(self):
        canonical = self.values((2, 4, 3), ">i8")
        path = preset.save(self.root / "points", canonical, kind="points")
        loaded = preset.load(path, kind="points")
        self.addCleanup(self.close_array, loaded)
        self.assertEqual(loaded.shape, (2, 4, 3))
        np.testing.assert_array_equal(loaded, canonical)

        same_order = ("time", "point", "xyz")
        explicit = preset.load(path, kind="points", axis_order=same_order)
        self.addCleanup(self.close_array, explicit)
        np.testing.assert_array_equal(explicit, canonical)

    def test_all_propagators_roundtrip_relative_physical_time_and_axes(self):
        tails = {"vsv": (2, 3), "psv": (2, 3, 4),
                 "vsp": (2, 4, 3), "psp": (2, 3, 4, 3)}
        for kind, tail in tails.items():
            for convention in ("relative", "absolute"):
                with self.subTest(kind=kind, convention=convention):
                    canonical = self.values((3, 3, 4, 4, *tail), ">c8")
                    disk = canonical.swapaxes(0, 1)
                    options = {"time_convention": convention,
                               "axis_order": ("sink_time", "source_time")}
                    path = self.root / f"{kind}-{convention}"
                    preset.save(path, disk, kind=kind, **options)
                    with preset.load(path, kind=kind, cfg="1000", shape=disk.shape,
                                     dtype=">c8", **options) as reader:
                        self.assertIsInstance(reader, ArrayPropagator)
                        self.assertEqual(reader.cfg, "1000")
                        self.assertIsNone(reader.data_identity)
                        self.assertEqual(reader.sectors[kind].dtype.str, ">c8")
                        index = 2 if convention == "relative" else 0
                        np.testing.assert_array_equal(reader.block(0, 1, kind), canonical[1, index])
                        self.assertFalse(reader.block(0, 1, kind).flags.writeable)
                    self.assertEqual(reader.sectors, {})
                    np.testing.assert_array_equal(np.load(path, allow_pickle=False), disk)

    def test_all_elementals_roundtrip_explicit_components_phase_and_axes(self):
        tails = {"v2v": (2, 4), "v2p": (2, 4, 3), "p2v": (2, 3, 4),
                 "p2p": (2, 3, 4, 3), "vvv": (2, 3, 4)}
        for kind, tail in tails.items():
            with self.subTest(kind=kind):
                component_cls = BaryonComponent if kind == "vvv" else ElementalComponent
                components = (component_cls("explicit-a"), component_cls("explicit-b"))
                canonical = self.values((2, 3, 4, *tail), ">c16")
                disk = canonical.transpose((2, 0, 1, *range(3, canonical.ndim)))
                options = {"components": components, "phase_convention": "anchor",
                           "axis_order": ("time", "component", "momentum")}
                path = preset.save(self.root / kind, disk, kind=kind, **options)
                with preset.load(path, kind=kind, cfg="1000", shape=disk.shape,
                                 dtype=">c16", **options) as reader:
                    cls = ArrayBaryonElemental if kind == "vvv" else ArrayElemental
                    self.assertIsInstance(reader, cls)
                    self.assertEqual(reader.components, components)
                    self.assertEqual(reader.phase_convention, "anchor")
                    self.assertEqual(reader.cfg, "1000")
                    self.assertIsNone(reader.data_identity)
                    np.testing.assert_array_equal(reader.block(components[1], 2, 3, kind),
                                                  canonical[1, 2, 3])
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), disk)

    def test_real_elementals_keep_real_dtype_and_fortran_order(self):
        data = np.asfortranarray(self.values((1, 2, 3, 4, 5), ">f4"))
        options = {"components": [ElementalComponent("known")], "phase_convention": "none",
                   "axis_order": ("component", "momentum", "time")}
        path = preset.save(self.root / "real.npy", data, kind="v2v", **options)
        with preset.load(path, kind="v2v", **options) as reader:
            value = reader.sectors["v2v"]
            self.assertTrue(value.flags.f_contiguous)
            self.assertEqual(value.dtype.str, ">f4")
            np.testing.assert_array_equal(value, data)

    def test_missing_or_irrelevant_declarations_fail_before_io(self):
        data = np.zeros((1, 1, 4, 4, 1, 1), complex)
        cases = [
            ("vsv", {}), ("psv", {"time_convention": "guessed"}),
            ("v2v", {}),
            ("v2v", {"components": [ElementalComponent("a")], "phase_convention": "none"}),
            ("v2v", {"axis_order": ("component", "momentum", "time"), "phase_convention": "none"}),
            ("v2v", {"components": [ElementalComponent("a")],
                      "axis_order": ("component", "momentum", "time")}),
            ("gauge", {"time_convention": "absolute"}),
            ("eigenvector", {"phase_convention": "none"}),
            ("eigenvalues", {"components": []}),
            ("other", {}), ("VSV", {}),
        ]
        for kind, options in cases:
            with self.subTest(kind=kind, options=options):
                path = self.root / "must-not-exist"
                with self.assertRaises(ValueError):
                    preset.load(path, kind=kind, **options)
                with self.assertRaises(ValueError):
                    preset.save(path, data, kind=kind, **options)
                self.assertFalse(path.exists())

    def test_invalid_shape_dtype_and_axes_rejected_on_read_and_write(self):
        elemental = {"components": [ElementalComponent("a")], "phase_convention": "none",
                     "axis_order": ("component", "momentum", "time")}
        cases = [
            ("gauge", np.zeros((1, 1, 1, 1, 3, 3, 3), complex), {}),
            ("gauge", np.zeros((1, 1, 1, 1, 4, 3, 3), float), {}),
            ("eigenvector", np.zeros((2, 3, 4, 3), complex), {}),
            ("eigenvector", np.zeros((2, 3, 1, 1, 1, 2), complex), {}),
            ("eigenvalues", np.zeros((2, 3), complex), {}),
            ("eigenvalues", np.zeros((0, 3), float), {}),
            ("points", np.zeros((2, 3, 3), float), {}),
            ("overlap", np.zeros((2, 3, 4, 2), complex), {}),
            ("vsv", np.zeros((2, 3, 4, 4, 1, 1), complex), {"time_convention": "absolute"}),
            ("psv", np.zeros((2, 2, 4, 4, 1, 2, 1), complex), {"time_convention": "absolute"}),
            ("vsv", np.zeros((2, 2, 3, 4, 1, 1), complex), {"time_convention": "absolute"}),
            ("v2v", np.zeros((2, 3, 4, 2, 2), complex), elemental),
            ("p2p", np.zeros((1, 2, 3, 2, 3, 4, 2), complex), elemental),
            ("v2v", np.zeros((1, 2, 3, 4, 5), int), elemental),
            ("gauge", np.zeros((1, 1, 1, 1, 4, 3, 3), complex), {"axis_order": ("time",) * 7}),
        ]
        for index, (kind, data, options) in enumerate(cases):
            with self.subTest(kind=kind, index=index):
                path = self.root / str(index)
                with self.assertRaises(ValueError):
                    preset.save(path, data, kind=kind, **options)
                self.assertFalse(path.exists())
                self.raw_save(path, data)
                with self.assertRaises(ValueError):
                    preset.load(path, kind=kind, **options)

    def test_component_definitions_duplicates_and_baryon_phase_are_checked(self):
        cases = (
            ("v2v", (2, 1, 1, 2, 2), (ElementalComponent("x"),) * 2, "none"),
            ("v2v", (1, 1, 1, 2, 2), ("not-a-component",), "none"),
            ("v2v", (1, 1, 1, 2, 2), (ElementalComponent("x"),), "invalid"),
            ("vvv", (1, 1, 1, 2, 2, 2), (ElementalComponent("x"),), "anchor"),
            ("vvv", (1, 1, 1, 2, 2, 2), (BaryonComponent("x"),), "midpoint"),
        )
        for index, (kind, shape, components, phase) in enumerate(cases):
            with self.subTest(kind=kind, index=index):
                options = {"components": components, "phase_convention": phase,
                           "axis_order": ("component", "momentum", "time")}
                data = np.zeros(shape, complex)
                path = self.root / str(index)
                with self.assertRaises(ValueError):
                    preset.save(path, data, kind=kind, **options)
                self.raw_save(path, data)
                with self.assertRaises(ValueError):
                    preset.load(path, kind=kind, **options)

    def test_optional_header_checks_use_disk_shape_and_exact_dtype(self):
        data = self.values((2, 3), ">f8")
        path = preset.save(self.root / "evals", data, kind="eigenvalues")
        for options, message in (({"shape": (3, 2)}, "shape"),
                                 ({"dtype": ">f4"}, "dtype"),
                                 ({"dtype": "<f8"}, "dtype")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                preset.load(path, kind="eigenvalues", **options)
        with self.assertRaisesRegex(ValueError, "cfg"):
            preset.load(path, kind="eigenvalues", cfg="1000")
        values = np.zeros((2, 2, 4, 4, 1, 1), complex)
        path = preset.save(self.root / "vsv", values, kind="vsv", time_convention="absolute")
        for options in ({"shape": (2, 2, 4, 4, 2, 2)}, {"dtype": "complex64"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                preset.load(path, kind="vsv", time_convention="absolute", **options)

    def test_object_sparse_container_and_non_numpy_values_are_not_accepted(self):
        path = self.root / "bad.npy"
        for data in (np.array([[object()]], dtype=object), [[1.0]],
                     {"values": np.ones((2, 3))}, np.ma.array(np.ones((2, 3)))):
            with self.subTest(type=type(data)), self.assertRaises(TypeError):
                preset.save(path, data, kind="eigenvalues")
            self.assertFalse(path.exists())
        self.raw_save(path, np.array([[object()]], dtype=object))
        with self.assertRaises(ValueError):
            preset.load(path, kind="eigenvalues")
        with path.open("wb") as stream:
            np.savez(stream, data=np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "one dense NPY"):
            preset.load(path, kind="eigenvalues")

    def test_trailing_and_truncated_payloads_are_rejected(self):
        for kind, data, options in (
            ("eigenvalues", np.ones((2, 3)), {}),
            ("vsv", np.ones((2, 2, 4, 4, 1, 1), complex), {"time_convention": "absolute"}),
            ("v2v", np.ones((1, 2, 3, 2, 2), complex),
             {"components": [ElementalComponent("a")], "phase_convention": "none",
              "axis_order": ("component", "momentum", "time")}),
        ):
            with self.subTest(kind=kind):
                path = self.root / kind
                self.raw_save(path, data)
                original = path.read_bytes()
                path.write_bytes(original + b"extra")
                with self.assertRaisesRegex(ValueError, "length"):
                    preset.load(path, kind=kind, **options)
                path.write_bytes(original[:-1])
                with self.assertRaises(ValueError):
                    preset.load(path, kind=kind, **options)

    def test_numpy_v3_numeric_header_and_object_rejection(self):
        def write_v3(path, array):
            with path.open("wb") as stream:
                header = np.lib.format.header_data_from_array_1_0(array)
                np.lib.format.write_array_header_2_0(stream, header)
                stream.seek(6)
                stream.write(b"\x03\x00")
                stream.seek(0, 2)
                if array.dtype.hasobject:
                    pickle.dump(array, stream, protocol=4)
                else:
                    stream.write(array.tobytes(order="C"))

        numeric = np.arange(6, dtype=np.float32).reshape(2, 3)
        numeric_path = self.root / "v3-numeric.npy"
        write_v3(numeric_path, numeric)
        loaded = preset.load(numeric_path, kind="eigenvalues")
        self.addCleanup(self.close_array, loaded)
        np.testing.assert_array_equal(loaded, numeric)

        objects = np.array([["not a numeric preset"]], dtype=object)
        object_path = self.root / "v3-object.npy"
        write_v3(object_path, objects)
        np.testing.assert_array_equal(np.load(object_path, allow_pickle=True), objects)
        with self.assertRaisesRegex(ValueError, "non-object"):
            preset.load(object_path, kind="eigenvalues")

    def test_overwrite_requires_explicit_opt_in_and_validation_precedes_write(self):
        data = self.values((2, 3), "float64")
        path = preset.save(self.root / "existing", data, kind="eigenvalues")
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            preset.save(path, data + 1, kind="eigenvalues")
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError):
            preset.save(path, data.astype(complex), kind="eigenvalues", overwrite=True)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(TypeError):
            preset.save(path, data, kind="eigenvalues", overwrite="yes")
        self.assertEqual(preset.save(path, data + 1, kind="eigenvalues", overwrite=True), path)
        np.testing.assert_array_equal(np.load(path), data + 1)

    def test_overwrite_from_mapping_does_not_truncate_its_input(self):
        data = self.values((2, 3), ">f8")
        path = preset.save(self.root / "mapped", data, kind="eigenvalues")
        mapped = preset.load(path, kind="eigenvalues")
        self.addCleanup(self.close_array, mapped)
        if os.name == 'nt':
            original = path.read_bytes()
            with self.assertRaisesRegex(PermissionError, 'release active file mappings'):
                preset.save(path, mapped, kind="eigenvalues", overwrite=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(self.root.iterdir()), [path])
        else:
            preset.save(path, mapped, kind="eigenvalues", overwrite=True)
        np.testing.assert_array_equal(mapped, data)
        np.testing.assert_array_equal(np.load(path), data)
        mapped._mmap.close()
        preset.save(path, data, kind="eigenvalues", overwrite=True)
        np.testing.assert_array_equal(np.load(path), data)

    def test_write_failure_cleans_new_file_and_preserves_existing_file(self):
        data = np.ones((2, 3))
        path = self.root / "output"
        with patch("numpy.save", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                preset.save(path, data, kind="eigenvalues")
        self.assertFalse(path.exists())
        preset.save(path, data, kind="eigenvalues")
        original = path.read_bytes()
        with patch("numpy.save", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                preset.save(path, data, kind="eigenvalues", overwrite=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
