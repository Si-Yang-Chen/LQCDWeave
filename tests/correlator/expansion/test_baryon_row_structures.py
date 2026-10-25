"""Every opened Baryon lowering on row families of different rank, density, basis and mixing.

Each forced form (``pair`` at any rank, ``nonzero``, ``dense``) applies to
every family without fallback; ``auto`` plans the per-family rule and the
distinct uniform pair and nonzero forms, and keeps the least predicted time. Values are
checked against the closed ``joined`` lowering. Timings of the same
structures: ``benchmarks/baryon_vertex_structure.py``.
"""
import unittest

import numpy as np

from lqcdweave.contraction.correlator import calculate

import test_baryon_quark_exchange as uuu
from correlator.expansion.baryon_row_structures import (
    BASES, STRUCTURES, SYNTHETIC, TWO_PHI, VANISHING, request, structure, two_phi_reader,
)

OPENED = ("auto", "pair", "nonzero", "dense")
# Small requests with nonvanishing values: single rows (a fold, a rank-4 row,
# a row with two Phi, a fold beside an uncompressed rank-2 row auto lowers to
# nonzero entries) and two-row families (a fold beside rank 7). The four-row
# structures take minutes on CPU: tests/temporary/tmp_baryon_row_structures.py.
REGULAR = ("single:fold-p16", "single:rank6-p16", "single:fold-p16+rank6-p16", "single:fold-p16|generic-e2",
           "fold-p16-r2|spin-G1g-S0")


def _case(spec):
    _, propagator, readers, _ = uuu.fixture()
    if "+" in spec:
        readers = {side: two_phi_reader(readers[side], side) for side in ("sink", "source")}
    value, vertices = request(spec, sink_reader=readers["sink"], source_reader=readers["source"])
    return value, dict(vertex_map=vertices, propagator_map={"S^q": propagator, "S^q_\\mathrm{local}": propagator},
                       configuration="exchange-cfg", time_axes={"sink": 0, "source": 2})


def _operations(problem):
    return {dict(spec.description.transform).get("operation") for spec in problem.specs.values()
            if spec.description is not None and spec.description.role == "baryon-spin"} - {None}


def _forms(lowering, family):
    if lowering == "nonzero":
        return {"row-spin-entry-weights", "spin-entry-selection"}
    if lowering == "dense":
        return {"row-merged-spin-coefficients"}
    return {"row-spin-pair-coefficients"} | (set() if family["fold"] else {"row-spin-slot-coefficients"})


class BaryonRowStructureTests(unittest.TestCase):
    def check_structures(self, specs):
        for spec in specs:
            value, options = _case(spec)
            families = structure(spec)
            reference = np.asarray(calculate(value, **options, baryon_vertices="joined").value)
            # Otherwise the comparison below would only see rounding.
            vanishing = any(name.removeprefix("single:").split("@")[0] in VANISHING for name in spec.split("|"))
            self.assertEqual(np.abs(reference).max() < 1e-9, vanishing, spec)
            for lowering in OPENED:
                with self.subTest(structure=spec, baryon_vertices=lowering):
                    result = calculate(value, **options, baryon_vertices=lowering)
                    problem = result.expanded.problem
                    self.assertTrue(problem.factorized_vertices)
                    selected = lowering
                    if lowering == "auto":
                        selection = result.path.prediction["baryon_vertex_selection"]
                        predicted = selection["predicted_seconds"]
                        rules = {family["auto"] for family in families.values()}
                        self.assertEqual(set(predicted) | set(selection["infeasible"]),
                                         {"auto", "pair", "nonzero"} - (rules if len(rules) == 1 else set()))
                        selected = selection["selected"]
                        self.assertEqual(predicted[selected], min(predicted.values()))
                    expected = set().union(*(_forms(family["auto"] if selected == "auto" else selected, family)
                                             for family in families.values()))
                    self.assertEqual(_operations(problem), expected)
                    np.testing.assert_allclose(np.asarray(result.value), reference, rtol=3e-12, atol=1e-10)

    def test_regular_structures_under_every_opened_lowering(self):
        self.check_structures(REGULAR)

    def test_structures_span_ranks_densities_and_bases(self):
        described = {spec: structure(spec) for spec in STRUCTURES}
        for name, recipe in SYNTHETIC.items():
            with self.subTest(structure=name):
                sink = described[name]["sink"]
                self.assertEqual((sink["rank"], sink["fold"]), (recipe["rank"], recipe["row_entries"] == 1
                                                                and recipe["rank"] == 1))
        # Beyond the bounded rank only the forced pair splits.
        self.assertFalse(described["rank8-p16"]["sink"]["bounded_split"])
        self.assertFalse(described["generic-e64"]["sink"]["bounded_split"])
        for name in BASES:
            with self.subTest(basis=name):
                dr, dp = described[name]["sink"], described[name + "@dp"]["sink"]
                self.assertEqual(dr["rank"], dp["rank"])  # local unitary on each slot
                self.assertNotEqual((dr["nonzero"], dr["fold"]), (dp["nonzero"], dp["fold"]))
        # auto keeps the split while it compresses entries, else takes the entries.
        for families in described.values():
            for family in families.values():
                compressed = family["rank"] == 1 or family["rank"] < family["nonzero"]
                self.assertEqual(family["auto"], "pair" if compressed else "nonzero")
        # Mixed requests and two-Phi rows hold families that auto lowers differently.
        for spec in ("rank6-p16|generic-e1", "rank2-p16+generic-e1", "single:fold-p16|generic-e2"):
            self.assertEqual({family["auto"] for family in described[spec].values()}, {"pair", "nonzero"})
        for spec in TWO_PHI:  # one structure per part and end
            self.assertEqual(set(described[spec]), {"sink0", "sink1", "source0", "source1"})


if __name__ == "__main__":
    unittest.main()
