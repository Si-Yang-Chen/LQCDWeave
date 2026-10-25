"""Temporary (slow, about 64 s on CSNS CPU) case moved from ``tests/test_three_sector_closed_loop.py``.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from test_three_sector_closed_loop import ThreeSectorClosedLoopTests
from test_three_sector_closed_loop import combinations, np, product  # noqa: F401


class TemporaryThreeSectorClosedLoopTests(ThreeSectorClosedLoopTests):
    def test_joint_random_medium_and_high_average_to_full_wilson_trace(self):
        target = self.expected((), (0, 0, 0))
        for weighted in (False, True):
            values = [self.evaluate((a, b), ((x,), (y,)), weighted=weighted, path=(), momentum=(0, 0, 0),
                name=f'joint-{weighted}-{a}-{b}-{x}-{y}')
                for a, b, x, y in product(combinations(range(3), 2), combinations(range(3), 2), range(2), range(2))]
            mean = np.mean(values)
            np.testing.assert_allclose(mean, target, rtol=2e-11, atol=1e-13)
            print(f'three-sector joint 36 draws weighted={weighted}: mean={mean}, full={target}')


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryThreeSectorClosedLoopTests("test_joint_random_medium_and_high_average_to_full_wilson_trace")])
