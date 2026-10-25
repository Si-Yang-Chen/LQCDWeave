"""Explicit compact matrices: numerical semantics, IO ownership and archives."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_structured_correlator import fixture, task
from tests.correlator.planning.test_meson_matrix import multi_component_fixture
from lqcdweave.contraction.performance.api import CostProfile
import lqcdweave.contraction.planning.path_planning as path_planning
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.operators.correlator.api import remove_disconneted_diagram
from lqcdweave.contraction.correlator import calculate, save_calculation_plan


class CompactMesonMatrixTests(unittest.TestCase):
    def setUp(self):
        self.bind, self.elemental, self.propagator = multi_component_fixture()
        self.options = dict(vertex_map=self.bind,
            propagator_map={'S^q': self.propagator, 'S^q_\\mathrm{local}': self.propagator},
            configuration='cfg', usedNe=3, memory_limit=1 << 30, host_memory_limit=1 << 30)

    def compute(self, request, **options):
        if options.get('optimize', 'anytime') in ('anytime', 'meson_matrix'):
            options.setdefault('search', {'deadline_seconds': 1.0, 'max_states': 1000, 'component_workers': 1})
            # These fixtures test matrix semantics under small local budgets.
            options['search'] = {'component_workers': 1, **options['search']}
        return calculate(request, **dict(self.options, **options))

    def test_compact_rows_preserve_values_without_scalar_expansion_or_search(self):
        request = task(structured=True)
        with patch('lqcdweave.contraction.expansion.structured.expand_structured',
                   side_effect=AssertionError('scalar matrix expansion must not run')), \
             patch.object(path_planning, 'solve_storage', wraps=path_planning.solve_storage) as search:
            result = self.compute(request, optimize='meson_matrix', execute=False)
        search.assert_not_called()
        legacy = self.compute(request, optimize='none', execute=False)
        self.assertLess(len(result.inputs.term_ids), len(legacy.inputs.term_ids))
        self.assertFalse(result.prediction['planner']['search_performed'])
        self.assertTrue(result.plan.state.actions)
        np.testing.assert_allclose(result.run().value, legacy.run().value, rtol=3e-12, atol=3e-12)

    def test_flavors_complex_coefficients_four_unequal_mode_blocks(self):
        # Endpoint lengths 1/2/1/3 are deliberately different.
        from lqcdweave.adapters.elemental import ArrayElemental
        from lqcdweave.adapters.propagator import ArrayPropagator
        from lqcdweave.contraction.binding.api import Meson
        from lqcdweave.contracts.elemental import ElementalComponent
        from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
        rng = np.random.default_rng(8841)
        random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
        component = ElementalComponent(0)
        e = ArrayElemental({'v2v': random((1,1,2,7,7))}, (component,), phase_convention='anchor', cfg='cfg')
        p = ArrayPropagator({'vsv': random((2,2,4,4,7,7))}, time_convention='absolute', cfg='cfg')
        op = Operator.from_terms('a', (OperatorTerm(5,component,0,.6-.25j),))
        bound = {False: Meson(e,op,False,phase_convention='anchor'),
                 True: Meson(e,op,True,phase_convention='anchor')}
        options = dict(vertex_map=lambda row: bound[row.dagger],
                       propagator_map={'S^q':p, 'S^q_\\mathrm{local}':p}, usedNe=7,
                       partitions=[0,1,3,4,7], time_pairs=({0:0,1:1},), execute=False)
        for flavor in ('charged','I0','I1'):
            with self.subTest(flavor=flavor):
                actual = self.compute(task(structured=True,flavor=flavor,names=('a',)),
                                      # Numerical semantics, not a one-second search SLA.
                                      optimize='meson_matrix',
                                      search={'deadline_seconds': 5.0, 'max_states': 1000, 'component_workers': 1},
                                      **options).run().value
                reference = self.compute(task(flavor=flavor,names=('a',)), optimize='none', **options).run().value
                self.assertEqual(actual.shape, (1,1,10,10,1))
                np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)

    def test_batch_widths_and_output_axes(self):
        for times in (dict(time_pairs=({0:0,1:1},{0:1,1:0})), dict(time_axes={0:0,1:(0,1)})):
            reference = self.compute(task(flavor='I0'), optimize='none', partitions=[0,1,3], **times).value
            for batch in (1,2,'auto'):
                with self.subTest(times=times,batch=batch):
                    planned = self.compute(task(structured=True,flavor='I0'), optimize='meson_matrix',
                                           partitions=[0,1,3], batch=batch, execute=False, **times)
                    np.testing.assert_allclose(planned.run().value, reference, rtol=3e-12, atol=3e-12)

    def test_user_groups_nested_connected_disconnected_and_duplicate_rows(self):
        full = task(structured=True,flavor='I0')
        connected = remove_disconneted_diagram(full, ['S^q_\\mathrm{local}'])
        request = {'I0':{'full':full,'connected':connected,'disconnected':full-connected},
                   'other':task(structured=True,names=('a','copy'))}
        symbolic = task(flavor='I0')
        symbolic_connected = remove_disconneted_diagram(symbolic, ['S^q_\\mathrm{local}'])
        reference = {'I0':{'full':symbolic,'connected':symbolic_connected,
                           'disconnected':symbolic-symbolic_connected},
                     'other':task(names=('a','copy'))}
        actual = self.compute(request,optimize='meson_matrix').value
        expected = self.compute(reference,optimize='none').value
        for key in ('full','connected','disconnected'):
            np.testing.assert_allclose(actual['I0'][key],expected['I0'][key],rtol=3e-12,atol=3e-12)
        np.testing.assert_allclose(actual['other'],expected['other'],rtol=3e-12,atol=3e-12)

    def test_metadata_only_and_raw_slabs_read_once_across_pair_batches(self):
        with patch.object(self.elemental,'read_slab',wraps=self.elemental.read_slab) as e_reads, \
             patch.object(self.propagator,'read_slab',wraps=self.propagator.read_slab) as p_reads:
            planned = self.compute(task(structured=True,flavor='I0'), optimize='meson_matrix',
                partitions=[0,1,3],time_pairs=({0:0,1:1},{0:1,1:0}),batch=1,execute=False)
            self.assertEqual(e_reads.call_count+p_reads.call_count,0)
            self.assertTrue(planned.plan.requests)
            self.assertTrue(any(request.prefetch_host_bytes is not None
                                for request in planned.plan.requests.values()))
            # Stack materialization delegates all reads to the existing task loader.
            actual = planned.run().value
            for calls in (e_reads.call_args_list,p_reads.call_args_list):
                signatures = [repr((call.args,call.kwargs)) for call in calls]
                self.assertTrue(signatures)
                self.assertEqual(len(signatures),len(set(signatures)))
        expected = self.compute(task(flavor='I0'),optimize='none',partitions=[0,1,3],
            time_pairs=({0:0,1:1},{0:1,1:0}),batch=1).value
        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)

    def test_stack_workspace_and_memory_rejection_before_io(self):
        planned = self.compute(task(structured=True),optimize='meson_matrix',execute=False)
        stack_keys = [key for key in planned.inputs.preparations if key.startswith('matrix:vertices:')]
        self.assertTrue(stack_keys)
        for key in stack_keys:
            source = planned.inputs.problem.sources[key]
            prep = planned.inputs.preparations[key]
            self.assertGreater(prep.workspace_host_bytes,source.snapshot_bytes)
            self.assertGreater(prep.traffic_bytes,0)
        with patch.object(self.elemental,'read_slab',side_effect=AssertionError('IO before budget audit')):
            with self.assertRaisesRegex(MemoryError, 'storage budget'):
                self.compute(task(structured=True),optimize='meson_matrix',execute=False,
                             memory_limit=128,host_memory_limit=128)

    def test_complete_archive_stores_all_pairs_and_loads_raw_readers(self):
        options = dict(optimize='meson_matrix',partitions=[0,1,3],batch=1,execute=False,
                       profile=CostProfile(hardware_fingerprint='compact-meson-cpu'),
                       target_hardware=TargetHardwareMemory(host_available_bytes=1<<30))
        request = task(structured=True,flavor='I0')
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'rows.json.gz'
            pairs = ({0:0,1:1},{0:1,1:0})
            complete = self.compute(request,time_pairs=pairs,**options)
            save_calculation_plan(archive,complete)
            restored = calculate(None,plan_archive=archive,configuration='cfg',
                propagator_map=self.options['propagator_map'],vertex_map=self.elemental,
                memory_limit=1<<30,host_memory_limit=1<<30,execute=False)
            fresh = self.compute(request,time_pairs=pairs,**options)
            np.testing.assert_allclose(restored.run().value,fresh.run().value,rtol=3e-12,atol=3e-12)


    def test_declared_dense_sector_and_vacuum_outputs(self):
        from lqcdweave.operators.correlator.api import Hadron, gen_correlator
        full = task(structured=True, flavor='I0')
        axis = [Hadron(position, flavor) for position, flavor in full.axes[0]]
        compact = gen_correlator([axis], [0], [False], degenerate=True, structured=True)
        symbolic = gen_correlator([axis], [0], [False], degenerate=True)
        symbolic = np.asarray([value.expand().evalf() for value in symbolic.flat],
                              dtype=object).reshape(symbolic.shape)
        with patch.object(self.elemental, 'available_sectors', return_value=('v2v',), create=True):
            actual = self.compute(compact, optimize='meson_matrix', partitions=[0,1,3]).value
            reference = self.compute(symbolic, optimize='none', partitions=[0,1,3]).value
            np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)

    def test_repeated_matrix_outputs_own_separate_arrays(self):
        request = task(structured=True, flavor='I0')
        result = self.compute({'first':request, 'again':request}, optimize='meson_matrix',
                              time_pairs=({0:0,1:1},{0:1,1:0})).value
        np.testing.assert_array_equal(result['first'], result['again'])
        self.assertFalse(np.shares_memory(result['first'], result['again']))


if __name__ == '__main__':
    unittest.main()
