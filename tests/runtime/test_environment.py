"""Tests for runtime hardware/software environment identity metadata."""
from __future__ import annotations

import json
import os
import platform
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import opt_einsum

from lqcdweave.runtime import environment
from lqcdweave.runtime.environment import runtime_fingerprint


class RuntimeFingerprintTests(unittest.TestCase):
    def test_numpy_identity_uses_real_cpu_environment_and_is_json_serializable(self):
        settings = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUPY_ACCELERATORS": "",
        }
        with patch.dict(os.environ, settings, clear=False):
            identity = runtime_fingerprint(np)

        self.assertEqual(environment.__all__, ["runtime_fingerprint"])
        self.assertIs(environment.runtime_fingerprint, runtime_fingerprint)
        self.assertEqual(identity["python"], platform.python_version())
        self.assertEqual(identity["system"], platform.system())
        self.assertEqual(identity["machine"], platform.machine())
        self.assertEqual(identity["processor"], platform.processor())
        self.assertEqual(identity["numpy"], np.__version__)
        self.assertEqual(identity["opt_einsum"], opt_einsum.__version__)
        self.assertEqual(identity["backend"], "numpy")
        self.assertEqual(identity["thread_settings"], settings)
        self.assertEqual(
            set(identity),
            {
                "python", "system", "machine", "processor", "numpy", "opt_einsum",
                "backend", "thread_settings",
            },
        )
        json.dumps(identity)

    def test_thread_environment_is_read_for_each_call(self):
        first = {
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "3",
            "MKL_NUM_THREADS": "4",
            "CUPY_ACCELERATORS": "first",
        }
        second = {
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "9",
            "MKL_NUM_THREADS": "10",
            "CUPY_ACCELERATORS": "second",
        }
        with patch.dict(os.environ, first, clear=False):
            first_identity = runtime_fingerprint(np)
        with patch.dict(os.environ, second, clear=False):
            second_identity = runtime_fingerprint(np)

        self.assertEqual(first_identity["thread_settings"], first)
        self.assertEqual(second_identity["thread_settings"], second)

    @staticmethod
    def _cupy_backend(device_name, failing_method=None, error=None):
        class FakeRuntime:
            def _maybe_raise(self, method):
                if method == failing_method:
                    raise error

            def getDevice(self):
                self._maybe_raise("getDevice")
                return 7

            def getDeviceProperties(self, device_id):
                self._maybe_raise("getDeviceProperties")
                if device_id != 7:
                    raise AssertionError("unexpected simulated device id")
                return {"name": device_name, "totalGlobalMem": 12_345_678_901}

            def runtimeGetVersion(self):
                self._maybe_raise("runtimeGetVersion")
                return 12040

            def driverGetVersion(self):
                self._maybe_raise("driverGetVersion")
                return 12050

        return SimpleNamespace(
            __name__="cupy",
            __version__="simulated-cupy",
            cuda=SimpleNamespace(runtime=FakeRuntime()),
        )

    def test_simulated_cupy_device_metadata_decodes_bytes_and_preserves_strings(self):
        for raw_name, expected_name in (
            (b"Simulated GPU \xe4\xb8\xad", "Simulated GPU \u4e2d"),
            ("Simulated GPU String", "Simulated GPU String"),
        ):
            with self.subTest(device_name=raw_name):
                identity = runtime_fingerprint(self._cupy_backend(raw_name))
                self.assertEqual(identity["backend"], "cupy")
                self.assertEqual(identity["cupy"], "simulated-cupy")
                self.assertEqual(identity["gpu_name"], expected_name)
                self.assertEqual(identity["gpu_total_bytes"], 12_345_678_901)
                self.assertEqual(identity["cuda_runtime"], 12040)
                self.assertEqual(identity["cuda_driver"], 12050)
                json.dumps(identity)

    def test_cupy_runtime_query_errors_propagate_unchanged(self):
        for method in (
            "getDevice", "getDeviceProperties", "runtimeGetVersion", "driverGetVersion",
        ):
            with self.subTest(method=method):
                error = RuntimeError("simulated failure from " + method)
                backend = self._cupy_backend(
                    "Simulated GPU", failing_method=method, error=error,
                )
                with self.assertRaises(RuntimeError) as caught:
                    runtime_fingerprint(backend)
                self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()


