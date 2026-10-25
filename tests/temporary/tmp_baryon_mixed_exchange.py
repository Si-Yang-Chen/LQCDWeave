"""Temporary (slow, about 37 s on CSNS CPU) case moved from ``tests/test_baryon_mixed_exchange.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from test_baryon_mixed_exchange import BaryonMixedExchangeTests
from test_baryon_mixed_exchange import make_case, np, request, run, sp  # noqa: F401


class TemporaryBaryonMixedExchangeTests(BaryonMixedExchangeTests):
    def test_complex_changed_basis_keeps_source_conjugation_and_nonzero_matrix_mixing(self):
        change = sp.Matrix([[1, sp.I/2], [0, 1]])
        axes, exchange, bind, propagator, _, expected = make_case(spin_change=change)
        full = run(request(axes), bind, propagator)
        np.testing.assert_allclose(full.value, expected, atol=3e-12, rtol=3e-12)
        for structured in (False, True):
            with self.subTest(structured=structured):
                reduced = run(request(axes, exchange, structured=structured), bind, propagator)
                np.testing.assert_allclose(reduced.value, expected, atol=3e-12, rtol=3e-12)
                self.assertLess(len(reduced.expanded.problem.terms), len(full.expanded.problem.terms))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryBaryonMixedExchangeTests("test_complex_changed_basis_keeps_source_conjugation_and_nonzero_matrix_mixing")])
