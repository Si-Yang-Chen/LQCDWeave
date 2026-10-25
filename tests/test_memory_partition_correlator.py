"""Storage-only mode partitions preserve the full correlator definition."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.vsv_graph import evaluate_vsv_expression
from lqcdweave.operators.correlator.api import split_meson_mode_pairs
from lqcdweave.contraction.correlator import calculate, save_calculation_plan
from test_mode_block_correlator import fixture as diagram_fixture
from test_structured_correlator import fixture, task


class MemoryPartitionTests(unittest.TestCase):
    def options(self, bind, propagator):
        return dict(vertex_map=bind, propagator_map={'S^q':propagator,
            r'S^q_\mathrm{local}':propagator}, configuration='cfg', usedNe=3,
            memory_limit=1 << 30, host_memory_limit=1 << 30)

    def test_four_unequal_blocks_match_independent_full_trace_and_keep_containers(self):
        connected, loops, vertices, elemental, propagator = diagram_fixture()
        full = -(1 + .2j)*connected + .4*loops[0]*loops[1]
        request = {'matrix':np.array([[full, 0], [2j*connected, connected]], dtype=object),
                   'loops':(loops[0], loops[1]), 'constant':[3]}
        options = dict(vertex_map=vertices, propagator_map={'q':propagator},
            configuration='cfg', usedNe=7, optimize='none', memory_limit=1 << 30,
            partitions=(0,1,3,4,7), partition_mode='memory')
        with patch.object(elemental, 'block', side_effect=AssertionError('planning IO')), \
             patch.object(propagator, 'block', side_effect=AssertionError('planning IO')):
            planned = calculate(request, execute=False, **options)
        widths = {n for spec in planned.expanded.problem.specs.values()
                  if spec.description.role == 'propagator' for n in spec.shape[-2:]}
        self.assertEqual(widths, {1,2,3})
        actual = planned.run().value
        reference = evaluate_vsv_expression(full, lambda row,t:vertices[row].get(t),
            lambda tag,s,t:propagator.block(s,t), {0:0,1:1})
        connected_reference = evaluate_vsv_expression(connected, lambda row,t:vertices[row].get(t),
            lambda tag,s,t:propagator.block(s,t), {0:0,1:1})
        self.assertEqual(actual['matrix'].shape, (2,2))
        self.assertIsInstance(actual['loops'], tuple)
        self.assertEqual(actual['constant'], [3])
        np.testing.assert_allclose(actual['matrix'][0,0], reference, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(actual['matrix'][1,0], 2j*connected_reference, rtol=3e-12, atol=3e-12)
        self.assertEqual(actual['matrix'][0,1], 0)

    def test_symbolic_and_structured_scans_match_unpartitioned_charged_I0_I1(self):
        bind, _, propagator = fixture()
        options = dict(self.options(bind, propagator), optimize='none',
                       time_pairs=({0:1,1:0}, {0:0,1:1}), batch=1)
        for flavor in ('charged','I0','I1'):
            expected = calculate(task(flavor=flavor), **options).value
            for structured in (False,True):
                with self.subTest(flavor=flavor, structured=structured):
                    request = split_meson_mode_pairs(task(structured=structured, flavor=flavor),
                        partitions=(0,1,3), partition_mode='memory')
                    self.assertEqual(request.shape, (2,2))
                    actual = calculate(request, **options).value
                    self.assertEqual(actual.shape, (2,2,2))
                    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def test_matrix_mode_preserves_values_and_reduces_output_reservation(self):
        bind, _, propagator = fixture()
        options = self.options(bind, propagator)
        for flavor in ('charged','I0','I1'):
            with self.subTest(flavor=flavor):
                request = task(structured=True, flavor=flavor)
                expected = calculate(task(flavor=flavor), optimize='none', **options).value
                memory = calculate(request, partitions=(0,1,3), partition_mode='memory',
                    optimize='meson_matrix', search={'deadline_seconds':1,'max_states':1000}, execute=False, **options)
                radial = calculate(request, partitions=(0,1,3), optimize='meson_matrix', search={'deadline_seconds':1,'max_states':1000},
                    execute=False, **options)
                self.assertEqual(len(memory.expanded.problem.terms), len(radial.expanded.problem.terms))
                self.assertLess(memory.inputs.assembly_host_bytes, radial.inputs.assembly_host_bytes)
                self.assertFalse(memory.inputs.matrix_kernels)
                np.testing.assert_allclose(memory.run().value, expected, rtol=3e-12, atol=3e-12)

    def test_radial_and_memory_requests_share_physical_block_terms(self):
        bind, _, propagator = fixture()
        full = task(structured=True, flavor='I0')
        request = {mode:split_meson_mode_pairs(full, partitions=(0,1,3), partition_mode=mode)
                   for mode in ('memory','radial')}
        options = self.options(bind, propagator)
        planned = calculate(request, optimize='meson_matrix', search={'deadline_seconds':1,'max_states':1000}, execute=False, **options)
        self.assertEqual(len(planned.expanded.problem.terms), 2**4 + 2*2**2)
        actual = planned.run().value
        self.assertEqual(actual['memory'].shape, (2,2))
        self.assertEqual(actual['radial'].shape, (2,2,3,3))
        weights = np.array([1, np.sqrt(2), 1])
        np.testing.assert_allclose(np.einsum('ijab,a,b->ij', actual['radial'], weights, weights),
                                   actual['memory'], rtol=3e-12, atol=3e-12)

    def test_anytime_ordinary_path_and_saved_cfg_keep_memory_only_output(self):
        bind, elemental, propagator = fixture()
        options = self.options(bind, propagator)
        request = task(structured=True, flavor='I0', names=('a',))
        planned = calculate(request, partitions=(0,1,3), partition_mode='memory',
            optimize='anytime', execute=False, search={'root_width':1,'branch_width':1,'max_states':5000},
            **options)
        self.assertFalse(planned.inputs.matrix_kernels)
        self.assertTrue(planned.path.plan.state.actions)
        expected = calculate(task(flavor='I0', names=('a',)), optimize='none', **options).value
        np.testing.assert_allclose(planned.run().value, expected, rtol=3e-12, atol=3e-12)
        coordinates = ({0:1,1:0}, {0:0,1:1})
        complete = calculate(request, partitions=(0,1,3), partition_mode='memory',
            optimize='meson_matrix', search={'deadline_seconds':1,'max_states':1000}, time_pairs=coordinates, execute=False, **options)
        expected = calculate(task(flavor='I0', names=('a',)), optimize='none',
                             time_pairs=coordinates, batch=1, **options).value
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'memory.json.gz'
            save_calculation_plan(archive, complete)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('archive replanned')):
                restored = calculate(plan_archive=archive, configuration='cfg', vertex_map=elemental,
                    propagator_map=options['propagator_map'], memory_limit=1 << 30)
            self.assertEqual(restored.value.shape, (1,1,2))
            np.testing.assert_allclose(restored.value, expected, rtol=3e-12, atol=3e-12)
            with self.assertRaisesRegex(ValueError, 'fixes planning options'):
                calculate(plan_archive=archive, partition_mode='memory')

    def test_invalid_modes_and_extent_fail_before_reads(self):
        bind, elemental, propagator = fixture()
        options = self.options(bind, propagator)
        with patch.object(elemental, 'block', side_effect=AssertionError('invalid IO')):
            for mode in ('other', None, True):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'partition_mode'):
                    calculate(task(), partitions=(0,1,3), partition_mode=mode, **options)
            with self.assertRaisesRegex(ValueError, 'requires partitions'):
                calculate(task(), partition_mode='memory', **options)
            with self.assertRaisesRegex(ValueError, 'usedNe'):
                calculate(task(structured=True), partitions=(0,1,4), partition_mode='memory',
                          optimize='meson_matrix', search={'deadline_seconds':1,'max_states':1000}, **options)

    def test_dataset_delivery_saves_original_axes_without_radial_fields(self):
        from lqcdweave.contraction.correlator_output import load_correlator_output

        bind, _, propagator = fixture()
        options = dict(self.options(bind, propagator), time_pairs=({0:1,1:0}, {0:0,1:1}))
        expected = calculate(task(flavor='I0'), optimize='none', batch=1, **options).value
        with tempfile.TemporaryDirectory() as directory:
            calculate(task(structured=True, flavor='I0'), partitions=(0,1,3),
                      partition_mode='memory', optimize='meson_matrix', search={'deadline_seconds':1,'max_states':1000}, output=directory, **options)
            actual = load_correlator_output(directory, configuration='cfg')
            self.assertEqual(actual.shape, (2,2,2))
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


if __name__ == '__main__':
    unittest.main()
