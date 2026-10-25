"""Target budgets bound reuse without dropping coordinates or changing kernels."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from correlator.planning import test_cfg_plan as plan_fixtures
from correlator.planning.test_cfg_plan import selected_template
from correlator.execution import test_cfg_runtime as runtime_fixtures
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.expansion.api import describe_input_reads
from lqcdweave.contraction.planning import cfg_plan
from lqcdweave.storage.contraction_plans.api import save_cfg_plan, load_cfg_plan


class CfgResidencyWindowTests(unittest.TestCase):
    def test_constant_reservation_does_not_grow_with_residency_ids(self):
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                selected, preps, specs = plan_fixtures.CfgPlanTests.dense_fixture(backend)
                points, _ = describe_coordinates(time_pairs=[{0:i, 1:0} for i in range(5)])
                with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1):
                    single = cfg_plan.build_cfg_plan(selected, None, points.points[:1], {},
                        preps, specs, output_mode='dataset', debug=True)
                    multiple = cfg_plan.build_cfg_plan(selected, None, points.points, {},
                        preps, specs, output_mode='dataset', debug=True)
                for kind in ('host', 'device'):
                    key = f'constant_{kind}_bytes'
                    self.assertEqual(single.statistics[key], multiple.statistics[key])
                self.assertEqual(single.peak_device_bytes, multiple.peak_device_bytes)
                self.assertEqual(single.peak_host_bytes, multiple.peak_host_bytes)

    def test_budget_reduces_retention_and_archive_preserves_all_points(self):
        expanded, selected, *_ = selected_template()
        points, _ = describe_coordinates(time_pairs=[{0:1, 1:0}, {0:0, 1:1}, {0:1, 1:0}])
        args = (expanded.output_recipe, points.points, describe_input_reads(expanded.inputs),
                expanded.inputs.loader.preparations, expanded.inputs.loader.specs)
        full = cfg_plan.build_cfg_plan(selected, *args, output_mode='dataset', debug=True)
        with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', len(selected.state.actions)):
            single = cfg_plan.build_cfg_plan(selected, *args, output_mode='dataset', debug=True)
        self.assertLess(single.peak_host_bytes, full.peak_host_bytes)
        constrained = replace(selected, memory_limit=single.peak_host_bytes)
        with patch.object(cfg_plan, '_bind_cfg_template', wraps=cfg_plan._bind_cfg_template) as bind:
            result = cfg_plan.build_cfg_plan(constrained, *args, output_mode='dataset', debug=True)
        self.assertEqual(bind.call_count, 1)
        self.assertEqual(result.statistics['reuse_window'], 1)
        self.assertGreater(result.statistics['resource_attempts'], 1)
        self.assertEqual([item.coordinate for item in result.instances],
                         [item.coordinate for item in full.instances])
        self.assertEqual(result.templates[0].plan.state.actions, full.templates[0].plan.state.actions)
        self.assertLessEqual(result.peak_host_bytes, constrained.memory_limit)
        self.assertEqual(result.statistics['compute_nodes'], single.statistics['compute_nodes'])
        self.assertFalse(any(flag for action, flag in zip(result.templates[0].plan.state.actions,
            result.instances[2].reused.array(), strict=True) if action.kind == 'contract'))
        # Compute windows remain closed; exact prepared owners may fit in the
        # saved spare budget. Raw blocks still close within their original use.
        for instance in result.instances:
            self.assertCountEqual([e.block for e in instance.events if e.kind == 'read'],
                                  [e.block for e in instance.events if e.kind == 'release'])
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'bounded.plan.gz'
            save_cfg_plan(archive, result)
            with patch.object(cfg_plan, 'build_cfg_plan', side_effect=AssertionError('replan')):
                restored = load_cfg_plan(archive)
            self.assertEqual(restored, result)

    def test_one_point_infeasibility_is_not_hidden(self):
        expanded, selected, *_ = selected_template()
        points, _ = describe_coordinates(time_pairs=[{0:1, 1:0}, {0:0, 1:1}])
        with self.assertRaisesRegex(MemoryError, 'host bytes'):
            cfg_plan.build_cfg_plan(replace(selected, memory_limit=1), expanded.output_recipe,
                points.points, describe_input_reads(expanded.inputs),
                expanded.inputs.loader.preparations, expanded.inputs.loader.specs)

    def test_numerical_results_and_cfg_isolation_survive_window_boundaries(self):
        reference, outputs, *_ = runtime_fixtures.CfgReplayTests().run_plan(capacity=0)
        with patch.object(cfg_plan, '_CFG_BINDING_WINDOW_SLOTS', 1):
            actual, bounded, *_ = runtime_fixtures.CfgReplayTests().run_plan(capacity=0)
        self.assertEqual(actual['completed'], reference['completed'])
        self.assertEqual(set(bounded), set(outputs))
        for key, values in outputs.items():
            for expected, value in zip(values, bounded[key], strict=True):
                np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
