"""Scientific scope and persistence boundaries of the task draft."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.campaigns.single_current_vdp_plan import build_plan, main, medium_columns


class SingleCurrentVDPPlanTests(unittest.TestCase):
    def test_scan_selects_each_actual_group_without_changing_high_complement(self):
        plan = build_plan()
        cases = [c for c in plan['cases'] if c['route'] == 'VDP']
        self.assertEqual(len(cases), 27)
        self.assertEqual(len({(c['columns_per_group'], c['point_count']) for c in cases}), 27)
        # Each requested matched-budget tier has one configuration per m,
        # differing by at most a single space-color sampling component.
        for target in (48,63,96):
            for m in (2,3,4):
                self.assertTrue(any(c['columns_per_group']==m and abs(c['sampled_components']-target)<=1 for c in cases))
        for case in cases:
            selected = case['d_column_positions']
            for group in range(10):
                self.assertEqual([i % 4 for i in selected if i // 4 == group],
                                 list(range(case['columns_per_group'])))
            self.assertEqual(case['p_projection'], 'Q428')
            self.assertEqual(case['sampled_components'], len(selected) + 3 * case['point_count'])
        self.assertEqual(sum(plan['medium_parent_counts']) + plan['low_modes'], 428)
        self.assertEqual(medium_columns(2)[-2:], (36, 37))
        for bad in (True, 1, 5, 2.0):
            with self.assertRaises(ValueError):
                medium_columns(bad)

    def test_scope_is_current_at_zero_and_only_forward_dt_one_to_eighteen(self):
        plan = build_plan()
        self.assertEqual(plan['source']['role'], 'current')
        self.assertEqual(plan['source']['physical_time'], 0)
        self.assertEqual(plan['sink']['physical_times'], list(range(1, 19)))
        self.assertEqual([m['channel'] for m in plan['measurements']], ['C2', 'C2p', 'C3'])
        self.assertEqual(plan['sampling']['point_candidates'], 64)
        self.assertEqual(plan['sampling']['color'], 'all-three')
        self.assertFalse(plan['statistics']['source_average'])

    def test_disk_budget_counts_current_slice_not_full_four_dimensional_solution(self):
        plan = build_plan()
        budget = plan['storage_budget']
        # Eighteen times, 100 source modes, both spin indices, full space/color.
        expected = 18 * 100 * 4 * 4 * (24**3 * 3) * 16
        self.assertEqual(budget['per_configuration_payload_bytes']['optional_current_time_solution_slices'], expected)
        self.assertEqual(budget['ensemble_payload_bytes']['optional_current_time_solution_slices'], 18 * expected)
        self.assertFalse(plan['retention']['optional_current_time_solution_slices'])
        self.assertFalse(plan['highmode']['persistent_output'])
        self.assertTrue(build_plan(retain_current_slices=True)['retention']['optional_current_time_solution_slices'])
        self.assertFalse(build_plan(retain_current_slices=True)['highmode']['persistent_output'])

    def test_cli_never_overwrites_a_reviewed_task_or_claims_numerical_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            main(['--output', str(path)])
            original = path.read_bytes()
            plan = json.loads(original)
            self.assertFalse(plan['numerical_worker_included'])
            self.assertEqual(plan['status'], 'planned_not_executed')
            with self.assertRaises(FileExistsError):
                main(['--output', str(path), '--retain-current-slices'])
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
