"""Bound planner scratch, preserving the original complete CfgPlan delivery."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from correlator.planning import test_cfg_plan as planning_fixtures
from correlator.execution import test_cfg_runtime as runtime_fixtures
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.expansion.api import describe_input_reads
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.storage.contraction_plans.api import save_cfg_plan, load_cfg_plan


class CompleteBindingWindowTests(unittest.TestCase):
    def compare(self, selected, output, requests, preparations, specs, *, mode):
        points, _ = describe_coordinates(time_pairs=[{0:1, 1:0}, {0:1, 1:0},
            {0:0, 1:1}, {0:1, 1:0}, {0:0, 1:1}])
        binding = cfg_plan._bind_cfg_template(selected, output, requests, preparations,
            specs, output_mode=mode)
        reference = binding(points.points, reuse_window=2)
        seen = []
        def bind(points, **options):
            seen.append((len(points), options.get('node_offset', 0)))
            return binding(points, **options)
        with patch.object(cfg_plan, '_bind_cfg_template', return_value=bind), \
                patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 2*len(selected.state.actions)):
            actual = cfg_plan.build_cfg_plan(selected, output, points.points, requests,
                preparations, specs, output_mode=mode)
        self.assertEqual([count for count, _ in seen], [2, 2, 1])
        self.assertEqual(seen[0][1], 0)
        self.assertTrue(all(a[1] < b[1] for a, b in zip(seen, seen[1:])))
        for field in fields(reference):
            if field.name != 'statistics':
                self.assertEqual(getattr(actual, field.name), getattr(reference, field.name), field.name)
        timings = {'binding_seconds', 'resource_seconds', 'lifecycle_seconds'}
        for key, expected in reference.statistics.items():
            if key not in timings:
                if type(expected) is float:
                    self.assertAlmostEqual(actual.statistics[key], expected, places=11, msg=key)
                else:
                    self.assertEqual(actual.statistics[key], expected, key)
        cfg_plan.audit_cfg_plan(actual)

    def test_raw_reads_preparations_reuse_events_and_output_reserve_match_full_domain(self):
        expanded, selected, *_ = planning_fixtures.selected_template()
        for mode in ('memory', 'dataset'):
            with self.subTest(mode=mode):
                self.compare(selected, expanded.output_recipe, describe_input_reads(expanded.inputs),
                    expanded.inputs.loader.preparations, expanded.inputs.loader.specs, mode=mode)

    def test_dense_batches_constants_cpu_and_offline_gpu_match_full_domain(self):
        for backend in ('numpy', 'cupy'):
            for mode in ('memory', 'dataset'):
                with self.subTest(backend=backend, mode=mode):
                    selected, preparations, specs = planning_fixtures.CfgPlanTests.dense_fixture(backend)
                    self.compare(selected, None, {}, preparations, specs, mode=mode)

    def test_complete_archive_replays_both_cfgs_without_any_planning(self):
        fixture = runtime_fixtures.CfgReplayTests()
        _, expected, *_ = fixture.run_plan(capacity=0)
        with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1):
            bounded = fixture.setup_plan()
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'full.plan.gz'
            save_cfg_plan(archive, bounded[0])
            with patch.object(cfg_plan, '_bind_cfg_template', side_effect=AssertionError('template binding')), \
                    patch.object(cfg_plan, '_bind_cfg_coordinates', side_effect=AssertionError('coordinate scheduling')), \
                    patch('lqcdweave.contraction.correlator.expand_correlator', side_effect=AssertionError('E')), \
                    patch('lqcdweave.contraction.correlator.find_contraction_path', side_effect=AssertionError('path')):
                restored = load_cfg_plan(archive)
                with patch.object(fixture, 'setup_plan', return_value=(restored, *bounded[1:])):
                    report, actual, *_ = fixture.run_plan(capacity=0)
        self.assertEqual(report['completed'], ('a', 'b'))
        self.assertEqual(actual.keys(), expected.keys())
        for key, values in actual.items():
            for value, reference in zip(values, expected[key], strict=True):
                np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
