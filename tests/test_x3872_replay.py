"""Startup diagnostics must not change backend setup or hide import failures."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


_PATH = Path(__file__).resolve().parents[1] / 'tools/campaigns/x3872_replay.py'
_SPEC = importlib.util.spec_from_file_location('x3872_replay_under_test', _PATH)
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)

_ENVIRONMENT = '''calls = dict(scan=0, dll=0, preload=0)
def _detect_duplicate_installation():
    calls['scan'] += 1
def _setup_win32_dll_directory():
    calls['dll'] += 1
def _preload_library(name):
    assert name == 'cutensor'
    calls['preload'] += 1
'''
_INITIALIZER = '''from cupy import _environment
_environment._detect_duplicate_installation()
_environment._setup_win32_dll_directory()
_environment._preload_library('cutensor')
'''


@contextmanager
def backend_package(*, known=True, fail=False):
    saved = {k:v for k,v in sys.modules.items() if k == 'cupy' or k.startswith('cupy.')}
    for name in saved:
        del sys.modules[name]
    try:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'cupy'
            package.mkdir()
            environment = _ENVIRONMENT.encode()
            (package / '_environment.py').write_bytes(environment)
            (package / '__init__.py').write_text(_INITIALIZER +
                ("raise RuntimeError('first backend failure')\n" if fail else ''), encoding='utf-8')
            digest = hashlib.sha256(environment).hexdigest() if known else 'unknown'
            with patch.object(sys, 'path', [directory, *sys.path]), patch.object(
                    replay, '_CSNS_CUPY_ENV_SHA256', digest):
                yield package
    finally:
        for name in tuple(sys.modules):
            if name == 'cupy' or name.startswith('cupy.'):
                del sys.modules[name]
        sys.modules.update(saved)


class BackendStartupTests(unittest.TestCase):
    def test_normal_start_omits_scan_preserves_setup_and_restores_diagnostic(self):
        with backend_package() as package:
            before = (package / '_environment.py').read_bytes()
            backend, omitted = replay._import_cupy()
            self.assertTrue(omitted)
            self.assertEqual(backend._environment.calls, dict(scan=0, dll=1, preload=1))
            backend._environment._detect_duplicate_installation()
            self.assertEqual(backend._environment.calls['scan'], 1)
            self.assertEqual((package / '_environment.py').read_bytes(), before)

    def test_debug_keeps_original_scan(self):
        with backend_package():
            backend, omitted = replay._import_cupy(debug=True)
            self.assertFalse(omitted)
            self.assertEqual(backend._environment.calls, dict(scan=1, dll=1, preload=1))

    def test_unknown_environment_uses_ordinary_import(self):
        with backend_package(known=False):
            backend, omitted = replay._import_cupy()
            self.assertFalse(omitted)
            self.assertEqual(backend._environment.calls, dict(scan=1, dll=1, preload=1))

    def test_failed_import_preserves_first_error_and_removes_partial_environment(self):
        with backend_package(fail=True):
            with self.assertRaisesRegex(RuntimeError, 'first backend failure'):
                replay._import_cupy()
            self.assertNotIn('cupy', sys.modules)
            self.assertNotIn('cupy._environment', sys.modules)

    def test_existing_backend_is_reused_without_reinitialization(self):
        with backend_package():
            first, _ = replay._import_cupy()
            second, omitted = replay._import_cupy()
            self.assertIs(second, first)
            self.assertFalse(omitted)
            self.assertEqual(first._environment.calls, dict(scan=0, dll=1, preload=1))


if __name__ == '__main__':
    unittest.main()
