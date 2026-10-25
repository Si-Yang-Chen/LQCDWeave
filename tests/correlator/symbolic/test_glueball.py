"""Zero-quark insertions retain their factors and leave fermionic Wick unchanged."""
import unittest
import sympy as sp
from lqcdweave.operators.correlator.api import (
    Diagram, Flavor, Hadron, HadronIrrep, Tag, gen_correlator,
    iter_structured_wick_entries,
)


def particle(name, flavor, slot=0):
    return Hadron(HadronIrrep(name, (0, 0, 0), "A_1", 1, Tag(slot, 0))[0], Flavor(flavor))


class GlueballSymbolicTests(unittest.TestCase):
    def test_empty_flavor_retains_time_and_adjoint(self):
        flavor = Flavor("", "sink")
        self.assertEqual(flavor.quark_list, [])
        self.assertEqual(flavor.anti_quark_list, [])
        self.assertEqual(flavor.conjugate(), flavor)
        self.assertNotEqual(flavor, sp.S.One)
        self.assertNotEqual(flavor, Flavor("", "source"))

    def test_pure_glue_and_repeated_insertions(self):
        g = particle("G", "")
        value = gen_correlator([[g], [g]], ["sink", "source"], [False, True])[0, 0]
        self.assertEqual(len(value.atoms(Diagram)), 2)
        for diagram in value.atoms(Diagram):
            self.assertEqual(diagram.diagram.adjacency_matrix, [[0]])
            self.assertEqual(diagram.propagator_list, [None])
        repeated = Hadron(g.irrep_row**2, g.flavor_structure**2)
        one = gen_correlator([[g]], ["sink"], [False])[0]
        two = gen_correlator([[repeated]], ["sink"], [False])[0]
        self.assertEqual(sp.expand(two - one**2), 0)

    def test_glue_factor_multiplies_existing_meson_and_baryon_wick(self):
        for flavor in ("ss", "uds"):
            g, h = particle("G", "", 1), particle("H", flavor)
            mixed = Hadron(g.irrep_row*h.irrep_row, g.flavor_structure*h.flavor_structure)
            value = gen_correlator([[mixed], [h]], ["sink", "source"], [False, True])[0, 0]
            quarks = gen_correlator([[h], [h]], ["sink", "source"], [False, True])[0, 0]
            glue = gen_correlator([[g]], ["sink"], [False])[0]
            self.assertEqual(sp.expand(value - glue*quarks), 0)

    def test_structured_mixed_matrix_matches_ordinary(self):
        axis = [particle("G", ""), particle("M", "ss")]
        expected = gen_correlator([axis.copy(), axis.copy()], ["sink", "source"], [False, True])
        task = gen_correlator([axis.copy(), axis.copy()], ["sink", "source"], [False, True], structured=True)
        for index, entries in iter_structured_wick_entries(task):
            actual = sum(c*sp.Mul(*ds) for c, ds in entries)
            self.assertEqual(sp.expand(actual-expected[index]), 0)
