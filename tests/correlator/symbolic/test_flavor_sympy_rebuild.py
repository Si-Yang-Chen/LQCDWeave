"""SymPy reconstruction of flavor leaves used by multihadron frontends."""

import unittest

from sympy import expand, factor_terms

from lqcdweave.operators.correlator.api import Flavor


class FlavorSympyRebuildTest(unittest.TestCase):
    def test_operator_rebuild_keeps_physical_metadata(self):
        cases = (
            Flavor("uds"),
            Flavor("bar{uds}", "sink"),
            Flavor("uss", "source", "symmetric"),
            Flavor("dud", 3, {(1, 0, 2): -1}),
            Flavor("", "sink"),
            Flavor("uds", "time)|quark_symmetry=inside"),
        )
        for original in cases:
            with self.subTest(original=str(original)):
                rebuilt = original.func(*original.args)
                self.assertEqual(rebuilt, original)
                self.assertEqual(str(rebuilt), str(original))
                self.assertEqual(rebuilt.flavor_str, original.flavor_str)
                self.assertEqual(rebuilt.time, original.time)
                self.assertEqual(rebuilt.quark_symmetry, original.quark_symmetry)
                self.assertEqual(rebuilt.quark_list, original.quark_list)
                self.assertEqual(rebuilt.anti_quark_list, original.anti_quark_list)

    def test_ll3_flavor_product_expands_and_factors(self):
        lam = (-2 * Flavor("uds", "sink") - Flavor("usd", "sink")
               + Flavor("dsu", "sink"))
        xin = (Flavor("uss", "source") * Flavor("dud", "source")
               + Flavor("dss", "source") * Flavor("udu", "source"))
        expanded = expand(lam * lam * xin)
        self.assertEqual(expand(factor_terms(expanded) - expanded), 0)
        self.assertTrue(expanded.has(Flavor("uds", "sink")))
        self.assertTrue(expanded.has(Flavor("uss", "source")))


if __name__ == "__main__":
    unittest.main()
