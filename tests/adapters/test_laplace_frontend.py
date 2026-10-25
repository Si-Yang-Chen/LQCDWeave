"""Explicit frontend selection and header-driven NPY inputs; CPU only."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from lqcdweave.adapters import DenseFileHandle
from lqcdweave.production.eigenvector.api import LaplaceEigenvectorGenerator


class LaplaceFrontendTests(unittest.TestCase):
    def setUp(self):
        self.size = (2, 2, 2, 2)
        self.handle = Mock()

    def test_explicit_cpu_does_not_import_gpu_or_load_inputs(self):
        with patch.dict(sys.modules, {"cupy": None, "pyquda_utils": None}):
            generator = LaplaceEigenvectorGenerator(self.size, self.handle, Ne=2, tol=1e-9,
                                                   backend="numpy")
        self.assertIs(type(generator), LaplaceEigenvectorGenerator)
        self.assertIs(generator.xp, np)
        self.assertIsNone(generator.backend)
        self.handle.load.assert_not_called()

    def test_explicit_gpu_choices_only_wire_requested_backend(self):
        cupy = types.ModuleType("cupy")
        solver = Mock()
        with patch.dict(sys.modules, {"cupy": cupy}), patch(
            "lqcdweave.adapters.laplace_quda.QudaLaplaceBackend", return_value=solver,
        ) as quda:
            generator = LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9, backend="cupy")
            self.assertIs(generator.xp, cupy)
            self.assertIsNone(generator.backend)
            quda.assert_not_called()
            generator = LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9, backend="quda")
            self.assertIs(generator.xp, cupy)
            self.assertIs(generator.backend, solver)
            quda.assert_called_once_with()

    def test_unknown_backend_and_missing_dependency_do_not_fallback(self):
        with self.assertRaises(ValueError):
            LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9, backend="auto")
        with patch.dict(sys.modules, {"cupy": None}):
            for backend in ("cupy", "quda"):
                with self.subTest(backend=backend), self.assertRaises(ImportError):
                    LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9, backend=backend)

    def test_backend_is_required_and_none_is_not_an_implicit_choice(self):
        with self.assertRaises(TypeError):
            LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9)
        with self.assertRaisesRegex(ValueError, "explicit"):
            LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9, backend=None)
        with self.assertRaisesRegex(ValueError, "LaplaceBackend"):
            LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9, backend=object())

    def test_custom_solver_is_preserved_with_its_explicit_array_module(self):
        solver = Mock(spec=["load", "invalidate", "stout_smear", "solve"])
        cupy = types.ModuleType("cupy")
        with patch.dict(sys.modules, {"cupy": None, "pyquda_utils": None}):
            for xp in (np, cupy):
                generator = LaplaceEigenvectorGenerator(
                    self.size, self.handle, 2, 1e-9, backend=solver, array_module=xp)
                self.assertIs(generator.backend, solver)
                self.assertIs(generator.xp, xp)
            generator = LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9, backend=solver)
            self.assertIs(generator.xp, np)
        self.assertEqual(solver.mock_calls, [])

    def test_named_backend_rejects_conflicting_array_module_before_gpu_import(self):
        with patch.dict(sys.modules, {"cupy": None}):
            for name, xp in (("cupy", np), ("quda", np), ("numpy", types.ModuleType("cupy"))):
                with self.subTest(backend=name), self.assertRaisesRegex(ValueError, "conflicts"):
                    LaplaceEigenvectorGenerator(self.size, self.handle, 2, 1e-9,
                                               backend=name, array_module=xp)

    def test_npy_header_inference_and_explicit_checks(self):
        data = np.ones((2, 2, 2, 2, 4, 3, 3), dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "gauge.")
            np.save(prefix + "cfg.npy", data)
            for options in ({}, {"shape": data.shape}, {"dtype": data.dtype},
                            {"shape": data.shape, "dtype": data.dtype}):
                handle = DenseFileHandle(prefix, ".npy", format="npy", **options)
                array = handle.load("cfg")
                self.assertIsInstance(array, np.memmap)
                self.assertFalse(array.flags.writeable)
                np.testing.assert_array_equal(array, data)
                array._mmap.close()
            for options in ({"shape": (2, 2)}, {"dtype": "complex128"}):
                with self.assertRaisesRegex(ValueError, "declared"):
                    DenseFileHandle(prefix, ".npy", format="npy", **options).load("cfg")

    def test_header_inference_preserves_explicit_layout_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "gauge.")
            data = np.arange(24).reshape(2, 12)
            np.save(prefix + "cfg.npy", data)
            handle = DenseFileHandle(prefix, ".npy", format="npy",
                                     reshape=(2, 3, 4), transpose=(0, 2, 1))
            array = handle.load("cfg")
            np.testing.assert_array_equal(array, data.reshape(2, 3, 4).transpose(0, 2, 1))
            array._mmap.close()

    def test_raw_input_still_requires_shape_and_dtype(self):
        for options in ({}, {"shape": (2, 3)}, {"dtype": ">c16"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "explicit"):
                DenseFileHandle("unused", ".bin", format="binary", **options)
