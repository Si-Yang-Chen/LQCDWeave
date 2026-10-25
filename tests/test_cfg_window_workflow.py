"""Whole-cfg output identity, dt means and publication across P/X windows."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import test_complete_cfg_dt_output as fixtures
from lqcdweave.contraction.correlator_cfg import complete_calculation
from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation
from lqcdweave.contraction.correlator_output import load_correlator_output
from lqcdweave.storage.dataset import read_dataset_metadata


class CfgWindowWorkflowTests(unittest.TestCase):
    def setup_case(self):
        helper, selected, coords, elemental, propagator = fixtures.PackedDtTests().selected(loops=True)
        complete = complete_calculation(selected, coordinates=coords, output_mode='dataset',
            physical_lt=128, output_layout='dt_packed')
        closed = []
        factory = helper.factory(complete.plan, elemental, propagator, [], closed, physical=True)
        return selected, coords, complete, factory, closed

    def test_window_outputs_match_complete_datasets_in_both_debug_modes(self):
        selected, coords, complete, factory, closed = self.setup_case()
        with tempfile.TemporaryDirectory() as root:
            baseline = replace(complete, options=dict(complete.options, configuration='a',
                source_factory=factory, output=str(Path(root)/'baseline'))).run()
            for debug in (False, True):
                events = []
                destination = Path(root)/str(debug)
                with patch('lqcdweave.contraction.correlator_cfg.build_cfg_plan',
                           side_effect=AssertionError('expanded a complete cfg in the parent')):
                    actual = run_cfg_window_calculation(selected, coordinates=coords,
                        configuration='a', source_factory=factory, output=destination,
                        physical_lt=128, output_layout='dt_packed', window_size=1,
                        debug=debug, progress=lambda kind, values: events.append((kind, values)))
                self.assertEqual(actual.report['completed'], ('a',))
                self.assertEqual(actual.report['coordinate_count'], 2)
                self.assertEqual(actual.report['planned_windows'], 2)
                self.assertGreater(actual.report['whole_cfg_output_extra_host_bytes'], 0)
                self.assertGreaterEqual(actual.report['peak_planned_host_bytes'],
                                        actual.report['whole_cfg_output_extra_host_bytes'])
                self.assertEqual([values['count'] for kind, values in events if kind=='point'], [1, 2])
                self.assertEqual(sum(kind=='published' for kind, values in events), 1)
                for field in ('two_point', 'vacuum_sink', 'vacuum_source'):
                    expected = load_correlator_output(Path(root)/'baseline', (field,), configuration='a')
                    value = load_correlator_output(destination, (field,), configuration='a')
                    np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-12, equal_nan=True)
                before = read_dataset_metadata(Path(root)/'baseline',cfg='a')['spec']
                after = read_dataset_metadata(destination,cfg='a')['spec']
                self.assertEqual(after.document, before.document)
        self.assertEqual(closed, ['a', 'a', 'a'])

    def test_cancel_after_first_point_never_publishes_cfg_and_closes_readers(self):
        selected, coords, complete, factory, closed = self.setup_case()
        def cancel(kind, values):
            if kind == 'point':
                raise RuntimeError('cancelled by consumer')
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'cancelled by consumer'):
                run_cfg_window_calculation(selected, coordinates=coords, configuration='a',
                    source_factory=factory, output=root, physical_lt=128,
                    output_layout='dt_packed', window_size=1, progress=cancel)
            with self.assertRaises(FileNotFoundError): read_dataset_metadata(root,cfg='a')
            self.assertFalse((Path(root)/'a.h5').exists())
        self.assertEqual(closed, ['a'])

    def test_output_budget_failure_precedes_worker_start(self):
        selected, coords, complete, factory, closed = self.setup_case()
        selected = replace(selected, path=replace(selected.path,
            plan=replace(selected.plan, host_memory_limit=64)))
        with patch('lqcdweave.contraction.correlator_window_run.plan_cfg_windows',
                   side_effect=AssertionError('started planner before output budget check')):
            with self.assertRaises(MemoryError):
                run_cfg_window_calculation(selected, coordinates=coords, configuration='a',
                    source_factory=factory, output='unused', physical_lt=128, output_layout='dt_packed')
        self.assertEqual(closed, [])

    def test_pair_output_keeps_whole_coordinate_axis(self):
        selected, coords, complete, factory, closed = self.setup_case()
        with tempfile.TemporaryDirectory() as root:
            expected = complete_calculation(selected, coordinates=coords,
                readers=selected.inputs.loader.input_readers).run().value
            run_cfg_window_calculation(selected, coordinates=coords, configuration='a',
                source_factory=factory, output=root, window_size=1)
            for field in ('two_point', 'vacuum_sink', 'vacuum_source'):
                value = load_correlator_output(Path(root), (field,), configuration='a')
                np.testing.assert_allclose(value, expected[field], rtol=3e-12, atol=3e-12)
        self.assertEqual(closed, ['a'])


if __name__ == '__main__':
    unittest.main()
