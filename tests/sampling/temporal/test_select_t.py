"""Independent brute-force and reference-result checks for time selection."""

import math
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
from itertools import combinations

import numpy as np

from lqcdweave.sampling.temporal.api import (
    SearchLimitExceeded, find_min_loop, iter_legal_schemes, select_best, select_t,
    TimeSelection, TimeSelectionRecord, covers_all_differences, covers_differences, exponential_pair_q,
)
from lqcdweave.contracts.time_selection import TimeSelection as ContractTimeSelection


class TimeSelectionTests(unittest.TestCase):
    def test_cfg_shift_keeps_template_order_pair_geometry_and_global_rng(self):
        options = dict(Lt=128, Nt=40, key=lambda s: s.offsets)
        template = select_t(**options)
        state = np.random.get_state()
        records = [select_t(**options, cfg=cfg) for cfg in (2000, '2000', 2010, 2020)]
        self.assertIsInstance(template, TimeSelection)
        self.assertTrue(all(isinstance(record, TimeSelectionRecord) for record in records))
        self.assertEqual(records[0], records[1])
        shifts = set()
        for record in records:
            self.assertEqual(record.Lt, 128)
            self.assertEqual(record.Nt, 40)
            shifts_for_slots = {(actual - base) % 128
                                for actual, base in zip(record.times, template.times)}
            self.assertEqual(len(shifts_for_slots), 1)
            shifts.update(shifts_for_slots)
            for i in range(40):
                for j in range(40):
                    self.assertEqual((record.times[j] - record.times[i]) % 128,
                                     (template.times[j] - template.times[i]) % 128)
        self.assertGreater(len(shifts), 1)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_cfg_shift_does_not_depend_on_python_hash_seed(self):
        import lqcdweave
        package_root = str(Path(lqcdweave.__file__).resolve().parent.parent)
        script = (
            f"import sys; sys.path.insert(0, {package_root!r}); "
            "from lqcdweave.sampling.temporal.api import select_t; "
            "print(select_t(8, 4, key=lambda s: s.offsets, cfg='2000').times)"
        )
        outputs = []
        for hash_seed in ('1', '987654'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE='1')
            outputs.append(subprocess.check_output([sys.executable, '-c', script], env=env))
        self.assertEqual(outputs[0], outputs[1])

    def test_cfg_validation_and_no_solution(self):
        for cfg in (True, False, '', 2.5, [], {}):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, 'cfg'):
                select_t(8, 4, key=lambda s: s.offsets, cfg=cfg)
        self.assertIsNone(select_t(8, 1, key=lambda s: 0, cfg=2000))

    def test_public_api_and_legacy_imports_share_one_implementation(self):
        api = importlib.import_module("lqcdweave.sampling.temporal.api")
        self.assertIs(api.TimeSelection, ContractTimeSelection)
        self.assertIs(api.TimeSelection, TimeSelection)

        legacy_search = importlib.import_module("lqcdweave.sampling.select_t")
        legacy_criteria = importlib.import_module("lqcdweave.sampling.time_rules")
        legacy_objective = importlib.import_module("lqcdweave.sampling.time_objectives")
        legacy_reconstruction = importlib.import_module("lqcdweave.contraction.two_point")
        self.assertIs(legacy_search.select_t, api.select_t)
        self.assertIs(legacy_search.TimeSelection, api.TimeSelection)
        self.assertIs(legacy_criteria.covers_all_differences, api.covers_all_differences)
        self.assertIs(legacy_objective.exponential_pair_q, api.exponential_pair_q)
        self.assertIs(legacy_reconstruction.reconstruct_two_point, api.reconstruct_two_point)

    def test_all_small_lattices_against_full_time_brute_force(self):
        for Lt in range(1, 11):
            for Nt in range(1, Lt + 1):
                expected_loops = []
                for loop in range(1, Lt + 1):
                    if Lt % loop:
                        continue
                    expected = set()
                    for times in combinations(range(Lt), Nt):
                        chosen = set(times)
                        if {(t + loop) % Lt for t in times} != chosen:
                            continue
                        if all(any((t + d) % Lt in chosen for t in times) for d in range(Lt)):
                            expected.add(tuple(t for t in times if t < loop))
                    actual = {s.offsets for s in iter_legal_schemes(Lt, Nt, loop)}
                    self.assertEqual(actual, expected, (Lt, Nt, loop))
                    if expected:
                        expected_loops.append(loop)
                self.assertEqual(find_min_loop(Lt, Nt), next(iter(expected_loops), None))

    def test_reference_counts_and_minimum(self):
        # Legacy PANEL_COUNTS_FINAL.csv reports translation orbits, not labelled sets.
        for loop, size, count in [(8, 4, 6), (16, 5, 32)]:
            schemes = list(iter_legal_schemes(128, size * 128 // loop, loop))
            orbits = {
                min(tuple(sorted((x - shift) % loop for x in s.offsets)) for shift in s.offsets)
                for s in schemes
            }
            self.assertEqual(len(orbits), count)
        for Nt, loop in [(40, 16), (64, 8), (96, 4), (128, 1)]:
            self.assertEqual(find_min_loop(128, Nt), loop)

    def test_custom_rules_do_not_inherit_coverage_or_translation_assumptions(self):
        contains_three = lambda s: 3 in s.offsets
        self.assertEqual(find_min_loop(8, 2, criteria=()), 4)
        schemes = list(iter_legal_schemes(8, 2, 4, criteria=(contains_three,)))
        self.assertEqual([s.offsets for s in schemes], [(3,)])
        rule = covers_differences([0, 4, -4])
        self.assertTrue(rule(schemes[0]))
        self.assertFalse(covers_all_differences(schemes[0]))
        self.assertEqual(list(iter_legal_schemes(8, 2, 4, criteria=(rule, contains_three))), schemes)
        with self.assertRaises(TypeError):
            list(iter_legal_schemes(8, 2, 4, criteria=(lambda s: 1,)))

    def test_rank_direction_ties_empty_and_composite_key(self):
        schemes = list(iter_legal_schemes(8, 2, 4, criteria=()))
        self.assertEqual(select_best(schemes, key=lambda s: s.offsets), schemes[0])
        self.assertEqual(select_best(schemes, key=lambda s: s.offsets, maximize=True), schemes[-1])
        self.assertEqual(select_best(schemes, key=lambda s: 1), schemes[0])
        self.assertIsNone(select_best([], key=lambda s: 0))
        self.assertEqual(select_t(8, 2, criteria=(), key=lambda s: s.offsets), schemes[0])

    def test_budget_exhaustion_cannot_claim_minimum_or_optimum(self):
        last_only = lambda s: s.offsets == (3,)
        with self.assertRaises(SearchLimitExceeded):
            find_min_loop(8, 2, criteria=(last_only,), max_candidates=1)
        with self.assertRaises(SearchLimitExceeded):
            select_t(8, 2, loop=4, criteria=(), key=lambda s: 0, max_candidates=1)
        self.assertEqual(len(list(iter_legal_schemes(8, 2, 4, criteria=(), max_candidates=4))), 4)
        self.assertEqual(len(list(iter_legal_schemes(8, 2, 4, criteria=(), max_candidates=None))), 4)

    def test_q_against_direct_definition_and_reference(self):
        ell = 2 ** (29 / 8)
        differences = tuple(range(64))
        objective = exponential_pair_q(ell=ell, differences=differences)
        panel = TimeSelection(128, 16, (0, 1, 4, 9, 11))
        times = panel.times
        baseline = sum(math.exp(-min(d, 128 - d) / ell) for d in range(128)) / 128
        values = []
        for d in differences:
            sources = [t for t in times if (t + d) % 128 in times]
            values.append(sum(
                math.exp(-min(abs(a - b), 128 - abs(a - b)) / ell)
                for a in sources for b in sources
            ) / len(sources) ** 2)
        direct = math.sqrt(sum(values) / len(values) / baseline)
        self.assertAlmostEqual(objective(panel), direct, places=12)
        self.assertAlmostEqual(objective(panel), 1.04791882, places=8)
        self.assertAlmostEqual(objective(TimeSelection(128, 1, (0,))), 1.0, places=12)
        best = select_t(128, 40, key=objective)
        self.assertAlmostEqual(objective(best), objective(panel), places=12)
        # Repeated residues carry the original difference weights, also for odd Lt.
        odd_q = exponential_pair_q(ell=2, differences=[0, 1, 1, 3])
        odd = TimeSelection(9, 3, (0, 1))
        values = []
        for d in [0, 1, 1, 3]:
            sources = [t for t in odd.times if (t + d) % 9 in odd.times]
            values.append(sum(math.exp(-min(abs(a-b), 9-abs(a-b))/2)
                              for a in sources for b in sources) / len(sources)**2)
        baseline = sum(math.exp(-min(d, 9-d)/2) for d in range(9)) / 9
        self.assertAlmostEqual(odd_q(odd), math.sqrt(sum(values)/4/baseline), places=12)

    def test_invalid_inputs_and_no_solution(self):
        for Lt, Nt, loop in [(0, 1, 1), (8, 9, 4), (8, 2, 3), (8, True, 4)]:
            with self.assertRaises(ValueError):
                list(iter_legal_schemes(Lt, Nt, loop))
        for offsets in [(0, 0), (-1,), (4,), ()]:
            with self.assertRaises(ValueError):
                TimeSelection(8, 4, offsets)
        self.assertEqual(list(iter_legal_schemes(8, 3, 4)), [])
        self.assertIsNone(find_min_loop(8, 1))
        self.assertIsNone(select_t(8, 1, key=lambda s: 0))
        for ell in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                exponential_pair_q(ell=ell, differences=[0])
        with self.assertRaises(ValueError):
            exponential_pair_q(ell=1, differences=[])
        with self.assertRaises(ValueError):
            exponential_pair_q(ell=1, differences=[1])(TimeSelection(8, 4, (0,)))


if __name__ == '__main__':
    unittest.main()
