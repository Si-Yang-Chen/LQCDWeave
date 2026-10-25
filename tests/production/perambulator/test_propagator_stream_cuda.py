"""CuPy device check for bounded source and sink mode reads."""

import unittest

import numpy as np

try:
    import cupy as cp
    try:
        DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
    except Exception:
        DEVICE_COUNT = 0
except Exception:
    cp, DEVICE_COUNT = None, 0

from lqcdweave.production.perambulator.api import PerambulatorGenerator, VectorBasis


class BlockHandle:
    def __init__(self, values):
        self.values = values
        self.Ne = values.shape[1]
        self.scratch = np.empty((2, *values.shape[2:]), values.dtype)

    def load(self, key):
        raise AssertionError("whole basis must not be loaded")

    def read_modes(self, key, time, start, stop):
        self.scratch[:stop-start] = self.values[time, start:stop]
        return self.scratch[:stop-start]


class Gauge:
    def load(self, key):
        return None


class IdentityBackend:
    def __init__(self):
        self.array_module = cp
        self.empty = cp.empty
        self.zeros = cp.zeros

    def load(self, gauge, parameters):
        return None

    def solve_into(self, rhs, solution, *, tol, maxiter):
        solution[...] = rhs


@unittest.skipUnless(cp is not None and DEVICE_COUNT > 0, "requires a working CUDA device")
class StreamCudaTests(unittest.TestCase):
    def test_identity_solver_matches_independent_mode_overlap(self):
        rng = np.random.default_rng(206)
        source = rng.normal(size=(2,3,1,1,2,3)) + 1j*rng.normal(size=(2,3,1,1,2,3))
        sink = rng.normal(size=(2,4,1,1,2,3)) + 1j*rng.normal(size=(2,4,1,1,2,3))
        source_basis=VectorBasis(BlockHandle(source))
        sink_basis=VectorBasis(BlockHandle(sink))
        producer = PerambulatorGenerator((2,1,1,2), Gauge(), .2, 1e-10, 10,
            backend=IdentityBackend(),
            stream_eigenvectors=True, sink_ne_block_size=2,
            sink_time_block_size=2, sink_cache_limit_bytes=768, MRHS=True)
        producer.load("cfg")
        for t_src in range(2):
            actual = cp.asnumpy(producer.calc(t_src,src=source_basis,sink=[sink_basis])[0].values)
            expected = np.zeros_like(actual)
            overlap = sink[t_src].reshape(4,-1).conj() @ source[t_src].reshape(3,-1).T
            for spin in range(4):
                expected[t_src,spin,spin] = overlap
            np.testing.assert_allclose(actual,expected,atol=3e-12)
        self.assertEqual(producer.sink_cache_used_bytes,768)


if __name__ == "__main__":
    unittest.main()
