"""Temporary (slow on CSNS CPU) structure cases of
``tests/correlator/expansion/test_baryon_row_structures.py``: every structure.

Not collected by default; run with ``python tools/test_tiers.py --tier temporary``.
"""
import unittest

from correlator.expansion.baryon_row_structures import STRUCTURES
from correlator.expansion.test_baryon_row_structures import REGULAR, BaryonRowStructureTests


class TemporaryBaryonRowStructureTests(BaryonRowStructureTests):
    def test_every_structure_under_every_opened_lowering(self):
        self.check_structures(spec for spec in STRUCTURES if spec not in REGULAR)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([TemporaryBaryonRowStructureTests("test_every_structure_under_every_opened_lowering")])
