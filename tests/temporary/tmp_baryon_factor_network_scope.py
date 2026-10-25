"""Temporary (slow, about 12 min on CSNS CPU) scope cases moved from
``tests/correlator/expansion/test_baryon_factor_network_scope.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from correlator.expansion.test_baryon_factor_network_scope import BaryonFactorNetworkScopeTests, TEMPORARY_CASES


class TemporaryBaryonFactorNetworkScopeTests(BaryonFactorNetworkScopeTests):
    def test_slow_cases_under_every_lowering(self):
        self.check_cases(TEMPORARY_CASES)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryBaryonFactorNetworkScopeTests("test_slow_cases_under_every_lowering")])
