"""Temporary (slow, about 41 s on CSNS CPU) case moved from ``tests/correlator/symbolic/test_exchange_independence.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from correlator.symbolic.test_exchange_independence import ExchangeIndependenceTests
from correlator.symbolic.test_exchange_independence import family  # noqa: F401


class TemporaryExchangeIndependenceTests(ExchangeIndependenceTests):
    def test_identical_vertices_transport_components_even_without_flavor_preserving_leg_swaps(self):
        left = family('b', 'mixed', 'uds')
        right = tuple(h.conjugate() for h in left)
        self.assert_relation_rank((left, left, right, right), ((1, 0, 2, 3), (0, 1, 3, 2)), factorized=True)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryExchangeIndependenceTests("test_identical_vertices_transport_components_even_without_flavor_preserving_leg_swaps")])
