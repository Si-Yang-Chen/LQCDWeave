"""Frontend plan/result pairing; numerical storage is checked independently too."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from test_structured_correlator import fixture, task
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.expansion.api import input_readers
from lqcdweave.contraction.correlator import (
    calculate, complete_calculation, load_calculation_plan, save_calculation_plan,
    save_correlator, load_correlator,
)


class FrontendRoundtripTests(unittest.TestCase):
    def selected(self, *, time_axes=None):
        bind, elemental, propagator = fixture()
        selected = calculate(task(structured=True), vertex_map=bind,
            propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator},
            configuration='cfg', usedNe=3, optimize='none', time_axes=time_axes,
            memory_limit=1 << 30, execute=False)
        return selected, elemental, propagator

    def test_fixed_binding_save_load_pair_does_not_search(self):
        selected, _, _ = self.selected()
        expected = selected.run().value
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'fixed.json.gz'
            save_calculation_plan(path, selected)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('load must not search')):
                loaded = load_calculation_plan(path, expected=selected)
                actual = loaded.run().value
        self.assertIsNone(loaded.value)
        np.testing.assert_allclose(actual, expected)

    def test_complete_save_load_pair_and_result_coordinates(self):
        selected, elemental, propagator = self.selected(time_axes={0: 1, 1: 0})
        coordinates, _ = describe_coordinates(time_axes={0: [1, 0], 1: [0, 1]})
        complete = complete_calculation(selected, coordinates=coordinates,
            readers=input_readers(selected.inputs))
        expected = complete.run().value
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'complete.json.gz'
            save_calculation_plan(path, complete)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('load must not search')):
                loaded = load_calculation_plan(path, configuration='cfg',
                    propagator_map={'S^q': propagator, r'S^q_\mathrm{local}': propagator},
                    vertex_map=elemental).run()
            np.testing.assert_allclose(loaded.value, expected)
            save_correlator(Path(directory) / 'result', loaded)
            restored = load_correlator(Path(directory) / 'result')
            self.assertEqual(restored.axes, (0, 1))
            self.assertEqual(tuple(restored.coordinates[0]), (1, 0))
            self.assertEqual(tuple(restored.coordinates[1]), (0, 1))
            np.testing.assert_array_equal(restored.value, expected)
            self.assertIn('operator_axis_0', restored.field_coordinates[()])
            self.assertEqual(len(restored.field_coordinates[()]['operator_axis_0']), 2)
            self.assertIn('operator', restored.field_coordinates[()]['operator_axis_0'][0])

    def test_result_requires_execution(self):
        selected, _, _ = self.selected()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'run the calculation'):
                save_correlator(Path(directory) / 'result', selected)

    def test_complete_loader_rejects_ambiguous_reader_arguments(self):
        with self.assertRaisesRegex(ValueError, 'not both'):
            load_calculation_plan('unused', source_factory=lambda cfg: None, readers={})
