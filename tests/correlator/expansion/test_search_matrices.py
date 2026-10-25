"""Closed whole-meson equations preserve scalar/Wick/projection semantics."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import DiagramBindings, MesonWholeKernel, describe_coordinates
from lqcdweave.contraction.expansion.api import NativeDiagramBinding, try_expand_search_matrices
from lqcdweave.contraction.planning.api import find_contraction_path
from lqcdweave.contraction.correlator import calculate, Calculation, save_calculation_plan
from lqcdweave.contraction.correlator_cfg import complete_calculation
from test_structured_correlator import fixture, task
from lqcdweave.operators.correlator.api import split_meson_mode_pairs


class SearchMatrixTests(unittest.TestCase):
    def test_compact_whole_paths_match_symbolic_complex_wick_and_unequal_partitions(self):
        for flavor in ('charged', 'I0', 'I1'):
            with self.subTest(flavor=flavor):
                bind, _, propagator = fixture()
                options = dict(vertex_map=bind, propagator_map={'S^q':propagator,
                    r'S^q_\mathrm{local}':propagator}, configuration='cfg', usedNe=3, memory_limit=2**30)
                expected = calculate(split_meson_mode_pairs(task(flavor=flavor), partitions=(0,1,3)),
                                     optimize='none', **options).value
                compact = split_meson_mode_pairs(task(structured=True, flavor=flavor), partitions=(0,1,3))
                bindings = DiagramBindings(vertex_map=bind, propagator_map=options['propagator_map'],
                                           configuration='cfg', usedNe=3)
                expanded = try_expand_search_matrices(compact, adapter=NativeDiagramBinding(), bindings=bindings)
                self.assertFalse(expanded.inputs.matrix_kernels)
                # The compact path executes closed whole-vertex joins. The
                # scalar symbolic calculation above independently expands the
                # Wick/component algebra; the legacy matrix-PC kernel stays off.
                path = find_contraction_path(expanded, optimize='none', memory_limit=2**30)
                with patch('lqcdweave.contraction.execution.matrix_kernels.contract_meson_matrix_pc',
                           side_effect=AssertionError('fixed template invoked')):
                    actual = Calculation(expanded, path).run().value
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)

    def test_public_anytime_uses_closed_whole_kernels_and_keeps_prepared_archive(self):
        bind, elemental, propagator = fixture()
        value = split_meson_mode_pairs(task(structured=True, flavor='I0'), partitions=(0,1,3))
        planned = calculate(value, vertex_map=bind, propagator_map={'S^q':propagator,
            r'S^q_\mathrm{local}':propagator}, configuration='cfg', usedNe=3, memory_limit=2**30,
            optimize='anytime', execute=False, search={'root_width':1,'branch_width':1,'max_states':5000})
        self.assertFalse(planned.expanded.inputs.matrix_kernels)
        self.assertTrue(planned.path.plan.feasible)
        self.assertIsNone(planned.path.search)
        self.assertFalse(any(action.kind=='meson_matrix' for action in planned.path.plan.state.actions))
        # P receives whole vertices and closed joins. Component/FWR arithmetic
        # remains inside the local kernel, so P neither searches private paths
        # nor needs sum barriers on the old globally exposed component Terms.
        self.assertTrue(planned.expanded.problem.meson_cores)
        self.assertTrue(all(isinstance(term.kernel, MesonWholeKernel)
            for term in planned.expanded.problem.terms))
        planner = planned.path.prediction['planner']
        self.assertFalse(planner['search_performed'])
        self.assertFalse(planner['private_paths_searched'])
        contracts = [action for action in planned.path.plan.state.actions if action.kind == 'contract']
        self.assertTrue(contracts)
        self.assertTrue(all(isinstance(action.core, MesonWholeKernel) and action.kernel_path == ()
            for action in contracts))
        self.assertTrue(any(prep.recipe.spatial_only
            for prep in planned.inputs.loader.preparations.values() if prep.recipe))
        widths={n for spec in planned.expanded.problem.specs.values()
                if spec.description is not None and spec.description.role=='propagator'
                for n in spec.shape[-2:]}
        self.assertEqual(widths,{1,2})
        expected = calculate(value, vertex_map=bind, propagator_map={'S^q':propagator,
            r'S^q_\mathrm{local}':propagator}, configuration='cfg', usedNe=3,
            optimize='none', memory_limit=2**30).value
        np.testing.assert_allclose(planned.run().value,expected,rtol=3e-12,atol=3e-12)
        coordinates, _ = describe_coordinates(time_pairs=({0:0,1:1},{0:1,1:0}))
        complete = complete_calculation(planned, coordinates=coordinates,
                                        readers=planned.inputs.loader.input_readers)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'searched-matrix.json.gz'
            save_calculation_plan(archive, complete)
            with patch('lqcdweave.contraction.correlator.find_contraction_path',
                       side_effect=AssertionError('archive searched again')), \
                 patch('lqcdweave.contraction.expansion.api.try_expand_search_matrices',
                       side_effect=AssertionError('archive expanded again')):
                restored = calculate(None, plan_archive=archive, configuration='cfg',
                    propagator_map={'S^q':propagator,r'S^q_\mathrm{local}':propagator},
                    vertex_map=elemental, memory_limit=2**30, host_memory_limit=2**30,
                    execute=False)
                actual = restored.run().value
            reference = calculate(value, vertex_map=bind,
                propagator_map={'S^q':propagator,r'S^q_\mathrm{local}':propagator},
                configuration='cfg', usedNe=3, optimize='none', memory_limit=2**30,
                time_pairs=({0:0,1:1},{0:1,1:0}), batch=1).value
            np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)

    def test_scans_keep_existing_expansion(self):
        bind, _, propagator = fixture()
        self.assertIsNone(try_expand_search_matrices(task(structured=True),
            adapter=NativeDiagramBinding(), bindings=DiagramBindings(vertex_map=bind,
                propagator_map={'S^q':propagator},configuration='cfg',usedNe=3),
            time_axes={0:(0,1),1:(0,1)}))

    def test_compact_budget_rejects_before_numerical_reads(self):
        bind, elemental, propagator = fixture()
        with patch.object(elemental, 'read_slab', side_effect=AssertionError('elemental IO')), \
             patch.object(propagator, 'read_slab', side_effect=AssertionError('propagator IO')), \
             patch.object(elemental, 'block', side_effect=AssertionError('elemental IO')), \
             patch.object(propagator, 'block', side_effect=AssertionError('propagator IO')):
            # The fixed whole-vertex route has no search exhaustion state; its
            # physical storage pass rejects an infeasible budget before IO.
            with self.assertRaisesRegex(MemoryError, 'physical storage budget before IO'):
                calculate(task(structured=True, names=('a',)), vertex_map=bind,
                    propagator_map={'S^q':propagator}, configuration='cfg', usedNe=3,
                    memory_limit=1, host_memory_limit=1, execute=False,
                    search={'root_width':1,'branch_width':1,'max_states':5000})

    def test_exact_modes_keep_original_expansion_domain(self):
        bind, _, propagator = fixture()
        class OriginalExpansionReached(Exception):
            pass
        def inspect(expanded, **options):
            self.assertFalse(expanded.inputs.matrix_kernels)
            self.assertTrue(all(len(term.inputs) == 4 for term in expanded.problem.terms))
            raise OriginalExpansionReached
        for mode in ('exact', 'dynamic_programming'):
            with self.subTest(mode=mode), \
                 patch('lqcdweave.contraction.expansion.api.try_expand_search_matrices',
                       side_effect=AssertionError('exact domain changed')), \
                 patch('lqcdweave.contraction.correlator.find_contraction_path', side_effect=inspect):
                with self.assertRaises(OriginalExpansionReached):
                    calculate(task(structured=True, names=('a',)), vertex_map=bind,
                        propagator_map={'S^q':propagator}, configuration='cfg', usedNe=3,
                        optimize=mode, memory_limit=2**30, execute=False)
