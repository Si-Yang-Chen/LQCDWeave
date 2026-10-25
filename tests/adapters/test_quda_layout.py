"""Check checkerboard conversion against explicit site coordinates without CUDA."""

import unittest
from types import SimpleNamespace

import numpy as np

from lqcdweave.adapters.quda import QudaBackend


class LayoutTests(unittest.TestCase):
    def test_stout_transform_stays_at_adapter_boundary(self):
        backend = QudaBackend.__new__(QudaBackend)

        class Gauge:
            def __init__(self):
                self.calls = []

            def stoutSmear(self, nstep, rho, dir_ignore):
                self.calls.append((nstep, rho, dir_ignore))

        gauge = Gauge()
        self.assertIs(backend.stout_smear(gauge, nstep=2, rho=0.1, dir_ignore=3), gauge)
        self.assertIs(backend.stout_smear(gauge, nstep=1, rho=0.125, dir_ignore=4), gauge)
        self.assertEqual(gauge.calls, [(2, 0.1, 3), (1, 0.125, 4)])
        with self.assertRaisesRegex(NotImplementedError, "does not provide PyQUDA stoutSmear"):
            backend.stout_smear(object(), nstep=1, rho=0.1, dir_ignore=3)

    def test_mapping_and_source_reuse(self):
        lx, ly, lz, lt = 6, 3, 2, 3
        backend = QudaBackend.__new__(QudaBackend)
        backend.array_module = np
        backend.latt_info = SimpleNamespace(size=(lx, ly, lz, lt))
        backend._sources = []
        backend._core = SimpleNamespace(LatticeFermion=lambda info: SimpleNamespace(
            data=np.empty((2, lt, lz, ly, lx//2, 4, 3), complex)))
        # Same construction as the adapter's load; compare values with an
        # independently looped parity reference, not just a round trip.
        t, z, y, h = np.indices((lt, lz, ly, lx//2))
        backend._indices = np.stack([
            (((t*lz+z)*ly+y)*lx+2*h+(p-t-z-y) % 2) for p in (0, 1)
        ]).reshape(-1)
        rhs = np.arange(2*lt*lz*ly*lx*12).reshape(2, lt, lz, ly, lx, 4, 3).astype(complex)
        fields = backend._to_quda(rhs)
        for r in range(2):
            for t in range(lt):
                for z in range(lz):
                    for y in range(ly):
                        for x in range(lx):
                            np.testing.assert_array_equal(
                                fields[r].data[(t+z+y+x) % 2, t, z, y, x//2], rhs[r,t,z,y,x])
        out = np.full_like(rhs, np.nan)
        backend._from_quda(fields, out)
        np.testing.assert_array_equal(out, rhs)
        again = backend._to_quda(rhs[:1] * 2)
        self.assertIs(again[0], fields[0])
        backend._from_quda(again, out[:1])
        np.testing.assert_array_equal(out[:1], rhs[:1]*2)

    def test_multi_rhs_container_is_reused(self):
        class Field:
            def __init__(self):
                self.data = np.zeros((2, 1, 1, 1, 1, 4, 3), complex)

        class Multi:
            def __init__(self, _info, count):
                self.fields = [Field() for _ in range(count)]

            def __getitem__(self, index):
                return self.fields[index]

        class Dirac:
            def __init__(self):
                self.calls = 0

            def invertMultiSrc(self, fields):
                self.calls += 1
                return fields

        backend = QudaBackend.__new__(QudaBackend)
        backend.array_module = np
        backend._core = SimpleNamespace(
            LatticeFermion=Field, MultiLatticeFermion=Multi)
        backend.latt_info = SimpleNamespace(size=(2, 1, 1, 1))
        backend._sources = []
        backend._multi_sources = {}
        backend._indices = np.array([0, 1])
        backend.dirac = Dirac()
        backend._tol, backend._maxiter = 1e-9, 10
        backend.measure = False
        events = []
        backend.synchronize = lambda: events.append('sync')
        backend.to_quda_seconds = backend.solve_seconds = backend.from_quda_seconds = 0.0
        backend.solve_calls = backend.rhs_count = backend.multirhs_calls = 0
        rhs = np.arange(2 * 1 * 1 * 1 * 2 * 4 * 3, dtype=float).reshape(
            2, 1, 1, 1, 2, 4, 3).astype(complex)
        solution = np.empty_like(rhs)
        backend.solve_into(rhs, solution, tol=1e-9, maxiter=10)
        container = backend._multi_sources[2]
        backend.solve_into(rhs, solution, tol=1e-9, maxiter=10)
        self.assertEqual(backend.dirac.calls, 2)
        self.assertEqual(backend.multirhs_calls, 2)
        self.assertIs(backend._multi_sources[2], container)
        np.testing.assert_array_equal(solution, rhs)
        self.assertEqual(events, ['sync'] * 4)  # Two native dependency fences per call.
        backend.measure = True
        events.clear()
        backend.solve_into(rhs, solution, tol=1e-9, maxiter=10)
        self.assertEqual(events, ['sync'] * 4)
        np.testing.assert_array_equal(solution, rhs)


if __name__ == '__main__':
    unittest.main()
