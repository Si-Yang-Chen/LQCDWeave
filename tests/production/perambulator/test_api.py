"""Public perambulator exports share one producer."""

from inspect import signature
import unittest

from lqcdweave.contracts.propagator import ModeBlockHandle, PropagatorBackend
from lqcdweave.production import PerambulatorGenerator as PackageGenerator
from lqcdweave.production.perambulator.api import (
    PerambulatorGenerator, PointBasis, PropagatorResult, PropagatorSolution, VectorBasis,
)


class PerambulatorApiTests(unittest.TestCase):
    def test_public_exports_preserve_one_class(self):
        self.assertIs(PackageGenerator, PerambulatorGenerator)
        self.assertEqual(
            PerambulatorGenerator.__module__,
            "lqcdweave.production.perambulator.generator",
        )
        self.assertEqual(
            tuple(signature(PerambulatorGenerator).parameters),
            (
                "latt_size", "gauge_field", "mass", "tol", "maxiter", "xi_0", "nu",
                "clover_coeff_t", "clover_coeff_r", "t_boundary", "multigrid",
                "contract_prec", "backend", "MRHS", "rhs_batch_size",
                "stream_eigenvectors", "sink_ne_block_size", "sink_time_block_size",
                "sink_cache_limit_bytes", "vector_scratch_limit_bytes", "source_selection",
                "sink_selection", "measure", "gamma5", "gamma5_hermitian",
            ),
        )
        self.assertEqual(
            tuple(signature(PerambulatorGenerator.calc).parameters),
            ("self", "t_src", "src", "sink", "write"),
        )
        self.assertIsNotNone(VectorBasis)
        self.assertIsNotNone(PointBasis)
        self.assertIsNotNone(PropagatorResult)
        self.assertIsNotNone(PropagatorSolution)

    def test_shared_solver_and_mode_block_protocols_remain_unique(self):
        self.assertTrue(hasattr(PropagatorBackend, "solve_into"))
        self.assertTrue(hasattr(ModeBlockHandle, "read_modes"))


if __name__ == "__main__":
    unittest.main()
