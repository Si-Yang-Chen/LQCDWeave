"""Root alternatives share one bounded process pool and select one route."""
from dataclasses import replace
import multiprocessing
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, SourceSpec, SourcePreparation, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.planning.optimizers.options import resolve_strategy_options
from lqcdweave.contraction.planning.optimizers.parallel_components import _worker_count
from lqcdweave.contraction.planning.optimizers import parallel_components as _parallel
from lqcdweave.contraction.planning.optimizers.protocols import SearchEvent
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerOperation, IntegerSpace
from lqcdweave.contraction.planning.path_planning import PathSearchExhausted
from correlator.planning.test_parallel_components import GatedProfile
from correlator.planning import test_tied_width as _tied_width


def chain():
    shapes = dict(A=(20, 2), B=(2, 3), C=(3, 10))
    problem = ContractionProblem(
        {key: TensorSpec(shape, 'complex128') for key, shape in shapes.items()},
        (ContractionTerm(tuple(shapes), 'ab,bc,cd->ad'),),
        {key: SourceSpec(key, 0) for key in shapes})
    return problem, {}, {key: SourcePreparation(()) for key in shapes}


def cycle():
    problem, blocks, preparations = chain()
    keys = tuple('ABCD')
    problem = replace(problem, specs={key: TensorSpec((2, 2), 'complex128') for key in keys},
        sources={key: SourceSpec(key, 0) for key in keys},
        terms=(ContractionTerm(keys, 'ab,bc,cd,da->'),))
    return problem, blocks, {key: SourcePreparation(()) for key in keys}


class ParallelRootTests(unittest.TestCase):
    def test_retained_root_statistics_count_before_worker_admission(self):
        discover = _parallel._discover_roots
        def with_large_root_statistics(*args, **kwargs):
            roots, statistics = discover(*args, **kwargs)
            # Keep an owner-only diagnostic out of the selected worker route:
            # budgeting just Actions/catalog or selected reports misses it.
            statistics[0]['retained_diagnostic'] = bytearray(2 * 2**20)
            return roots, statistics
        with patch.object(_parallel, '_discover_roots', side_effect=with_large_root_statistics), \
             patch.object(_parallel, 'ProcessPoolExecutor',
                          side_effect=AssertionError('oversized owner metadata admitted workers')):
            with self.assertRaises(PathSearchExhausted) as caught:
                solve_storage(*chain(), memory_limit=10**7, component_workers=2,
                    max_states=1000, deadline_seconds=30, search_memory_limit=2**20)
        self.assertEqual(caught.exception.search_result.stop_reason, 'search_memory_limit')
        self.assertGreaterEqual(caught.exception.search_result.statistics['sampled_peak_search_bytes'], 2 * 2**20)

    def test_double_baryon_join_process_roots_match_serial_and_independent_einsum(self):
        specs = {key:TensorSpec((2, 2, 2, 2), 'complex128') for key in ('q0', 'q1', 'q2')}
        specs.update(spin=TensorSpec((2, 2, 2), 'complex128'),
                     mode=TensorSpec((2, 2, 2), 'complex128'))
        joins = tuple(BaryonJoinSpec(0, name, 'spin', 'mode', spin, mode)
            for name,spin,mode in (('left', 'abc', 'def'), ('right', 'ABC', 'DEF')))
        problem = ContractionProblem(specs,
            (ContractionTerm(('q0', 'q1', 'q2'), 'adAD,beBE,cfCF->'),),
            {key:SourceSpec(key, 0) for key in specs}, baryon_joins=joins)
        preparations = {key:SourcePreparation(()) for key in problem.sources}
        options = dict(memory_limit=10**8, root_width=2, branch_width=1,
                       max_states=1000, deadline_seconds=30, debug=True)
        with multiprocessing.get_context('spawn').Manager() as manager:
            arrivals = manager.list()
            profile = GatedProfile(barrier=manager.Barrier(2), arrivals=arrivals, owner=os.getpid())
            parallel = solve_storage(problem, {}, preparations, component_workers=2,
                                     profile=profile, **options)
            self.assertEqual(len(set(arrivals)), 2)
        serial = solve_storage(problem, {}, preparations, component_workers=1, **options)
        stats = parallel.statistics
        self.assertEqual(stats['component_count'], 1)
        self.assertGreaterEqual(stats['component_root_counts'][0], 2)
        self.assertEqual(stats['component_roots_started'], stats['component_root_counts'])
        self.assertEqual(stats['component_roots_feasible'], stats['component_root_counts'])
        self.assertTrue(all(record['pid'] != os.getpid() for record in stats['root_branch_statistics'][0]))
        self.assertEqual(len({record['pid'] for record in stats['root_branch_statistics'][0]}), 2)
        self.assertTrue(stats['storage_schedule']['completed'])
        self.assertTrue(parallel.plan.feasible)
        rng = np.random.default_rng(20261016)
        arrays = {key:rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
                  for key,spec in specs.items()}
        expected = np.einsum('abc,def,ABC,DEF,adAD,beBE,cfCF->',
            arrays['spin'], arrays['mode'], arrays['spin'], arrays['mode'],
            arrays['q0'], arrays['q1'], arrays['q2'], optimize=False)
        for result in (parallel, serial):
            actual = replay(result.state, lambda key,source:arrays[key])[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
        self.assertAlmostEqual(stats['logical_best_seconds'], serial.statistics['logical_best_seconds'])

    def test_two_bridge_cycle_process_roots_match_serial_and_dense_oracle(self):
        from test_sparse_cycle_partitions import _SparseCyclePartitionFixture, _dense_oracle
        rng = np.random.default_rng(611)
        arrays = {key:rng.normal(size=(4, 4, 3, 3, 3, 3))+
                      1j*rng.normal(size=(4, 4, 3, 3, 3, 3)) for key in ('p0', 'p1')}
        gammas = [rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)) for _ in range(2)]
        records = [dict(type='sparse', indices=np.array([(i,j) for i in range(3) for j in range(3)]),
            values=rng.normal(size=(9, 3, 3))+1j*rng.normal(size=(9, 3, 3))) for _ in range(2)]
        arrays.update({f'r{i}':((gammas[i], 1, records[i]),) for i in range(2)})
        for partition in ((0, 1, 2, 3), (0, 1, 2, 0)):
            with self.subTest(partition=partition):
                state, profile, _, _ = _SparseCyclePartitionFixture()._planned_cycle(partition)
                preparations = {key:SourcePreparation(()) for key in state.problem.sources}
                options = dict(memory_limit=10**8, profile=profile, max_states=1000, deadline_seconds=30)
                parallel = solve_storage(state.problem, {}, preparations, component_workers=2, **options)
                serial = solve_storage(state.problem, {}, preparations, component_workers=1, **options)
                stats = parallel.statistics
                self.assertEqual(stats['component_count'], 1)
                self.assertGreaterEqual(stats['component_root_counts'][0], 1)
                self.assertEqual(stats['component_roots_started'], stats['component_root_counts'])
                self.assertEqual(stats['component_roots_feasible'], stats['component_root_counts'])
                self.assertTrue(all(record['pid'] != os.getpid()
                                    for record in stats['root_branch_statistics'][0]))
                self.assertTrue(stats['storage_schedule']['completed'])
                self.assertTrue(parallel.plan.feasible)
                expected, _ = _dense_oracle(partition, arrays['p0'], arrays['p1'],
                                           *gammas, *records, 3)
                for result in (parallel, serial):
                    actual = replay(result.state, lambda key,source:arrays[key])[0]
                    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
                self.assertAlmostEqual(stats['logical_best_seconds'], serial.statistics['logical_best_seconds'])

    def test_root_binding_distinguishes_algebraically_equal_kernel_recipes(self):
        space = IntegerSpace(chain()[0], profile=CostProfile())
        operation, cost = next(space.search_candidates(lambda:None))
        calculation = space.registry.calculations[operation.calculation]
        alternative = replace(calculation, seconds=cost+1.0, workspace=calculation.workspace+128,
            recipe=replace(calculation.recipe, kernel_path=('alternative',)))
        identity = len(space.registry.calculations)
        space.registry.calculations.append(alternative)
        other = IntegerOperation(identity, alternative.result, False, alternative.seconds)
        first_key, second_key = (_parallel._root_key(space.registry, value) for value in (operation, other))
        self.assertEqual(first_key[0], second_key[0])
        candidates = SimpleNamespace(registry=space.registry,
            search_candidates=lambda checkpoint:iter(((operation,cost),(other,alternative.seconds))))
        self.assertEqual(_parallel._bind_root(candidates, second_key, lambda:None),
                         ((other, alternative.seconds),))
        self.assertEqual(_parallel._bind_root(candidates, first_key, lambda:None), ((operation,cost),))

    def test_unused_global_reservations_are_returned(self):
        context = multiprocessing.get_context('spawn')
        counters = (context.Value('q', 32), context.Value('q', 1000))
        with patch.object(_parallel, '_WORKER_QUOTAS', counters):
            budget = _parallel._WorkerBudget(shared_quotas=True, max_states=32)
            try:
                budget.consume(SearchEvent('expand'))
                budget.consume(SearchEvent('transition'))
                self.assertEqual((budget.expanded, budget.transitions), (1, 1))
                self.assertEqual(tuple(value.value for value in counters), (16, 744))
            finally:
                budget.close()
            self.assertEqual(tuple(value.value for value in counters), (31, 999))

    def test_custom_clock_auto_falls_back_and_explicit_parallel_rejects_it(self):
        with patch('lqcdweave.contraction.planning.optimizers.parallel_components.ProcessPoolExecutor',
                   side_effect=AssertionError('custom clock cannot spawn')):
            result = solve_storage(*chain(), memory_limit=10**7, clock=lambda:0.0)
        self.assertEqual(result.statistics['parallel_fallback_reason'], 'custom_clock')
        with self.assertRaisesRegex(ValueError, 'real monotonic clock'):
            solve_storage(*chain(), memory_limit=10**7, clock=lambda:0.0, component_workers=2)

    def test_components_choose_each_best_root_then_share_equal_computations(self):
        problem, blocks, preparations = chain()
        problem = replace(problem, terms=problem.terms * 2)
        with patch('lqcdweave.contraction.planning.optimizers.graph_session.shared_components',
                   return_value=((0,), (1,))):
            result = solve_storage(problem, blocks, preparations, memory_limit=10**7,
                component_workers=2, root_width=2, branch_width=1, debug=True, deadline_seconds=30)
        stats = result.statistics
        self.assertEqual(stats['component_root_counts'], (2, 2))
        self.assertEqual(stats['component_roots_started'], (2, 2))
        self.assertEqual(stats['component_selected_roots'], (1, 1))
        costs = [records[1]['logical_best_seconds'] for records in stats['root_branch_statistics']]
        self.assertAlmostEqual(stats['logical_best_seconds'], costs[0])
        self.assertAlmostEqual(stats['cross_component_reused_seconds'], costs[1])
        self.assertEqual(result.state.results[0], result.state.results[1])

    def test_cpu_affinity_slurm_and_explicit_total_ceiling(self):
        with patch.object(os, 'sched_getaffinity', return_value={1, 3, 7}, create=True):
            with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '2'}):
                self.assertEqual(_worker_count(None, 30, 0, None), 2)
                self.assertEqual(_worker_count(1, 30, 0, None), 1)
                self.assertEqual(_worker_count(8, 30, 0, None), 2)

    def test_auto_single_root_and_single_cpu_stay_in_owner(self):
        problem, blocks, preparations = chain()
        one = replace(problem, terms=(ContractionTerm(('A', 'B'), 'ab,bc->ac'),))
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '2'}):
            result = solve_storage(one, blocks, preparations, memory_limit=10**7,
                                   max_states=1000, deadline_seconds=30)
        self.assertEqual(result.statistics['component_worker_pids'], (os.getpid(),))
        self.assertEqual(result.statistics['actual_search_workers'], 0)
        self.assertEqual(result.statistics['parallel_fallback_reason'], 'single_root_task')
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '1'}):
            with patch('lqcdweave.contraction.planning.optimizers.parallel_components.ProcessPoolExecutor',
                       side_effect=AssertionError('CPU cap one must not spawn')):
                result = solve_storage(*chain(), memory_limit=10**7, max_states=1000,
                                       deadline_seconds=30)
        self.assertEqual(result.statistics['parallel_fallback_reason'], 'cpu_limit')

    def test_auto_default_and_explicit_serial(self):
        self.assertIsNone(resolve_strategy_options()['component_workers'])
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '2'}):
            result = solve_storage(*chain(), memory_limit=10**7, max_states=1000,
                                   deadline_seconds=30)
        stats = result.statistics
        self.assertEqual(stats['component_workers'], 2)
        self.assertEqual(stats['component_roots_started'], (2,))
        reference = solve_storage(*chain(), memory_limit=10**7, max_states=1000,
                                  deadline_seconds=30, component_workers=1)
        self.assertAlmostEqual(stats['logical_best_seconds'], reference.statistics['logical_best_seconds'])

    def test_two_roots_really_overlap_and_later_cheaper_root_wins(self):
        with multiprocessing.get_context('spawn').Manager() as manager:
            arrivals = manager.list()
            profile = GatedProfile(barrier=manager.Barrier(2), arrivals=arrivals, owner=os.getpid())
            result = solve_storage(*chain(), memory_limit=10**7, component_workers=2,
                profile=profile, root_width=2, branch_width=1, deadline_seconds=30)
            self.assertEqual(len(set(arrivals)), 2)
        stats = result.statistics
        self.assertEqual(stats['component_root_counts'], (2,))
        self.assertEqual(stats['component_roots_feasible'], (2,))
        self.assertEqual(stats['component_selected_roots'], (1,))
        roots = stats['root_branch_statistics'][0]
        self.assertLess(roots[1]['logical_best_seconds'], roots[0]['logical_best_seconds'])
        self.assertEqual(stats['logical_best_seconds'], roots[1]['logical_best_seconds'])
        rng = np.random.default_rng(12)
        # Use original input specs: intermediate specs belong to the delivered catalog.
        arrays = {key:rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape)
                  for key,spec in chain()[0].specs.items()}
        actual = replay(result.state, lambda key,source:arrays[key])[0]
        np.testing.assert_allclose(actual, np.einsum('ab,bc,cd->ad', arrays['A'],arrays['B'],arrays['C']), rtol=1e-12, atol=1e-12)

    def test_explicit_roots_queue_beyond_worker_count_without_memory_walk(self):
        with patch('lqcdweave.contraction.planning.optimizers.parallel_components.retained_size',
                   side_effect=AssertionError('unrequested memory traversal')):
            result = solve_storage(*cycle(), memory_limit=10**7, component_workers=2,
                root_width=4, branch_width=1, max_states=1000, deadline_seconds=30)
        stats = result.statistics
        self.assertEqual(stats['component_root_counts'], (4,))
        self.assertEqual(stats['component_roots_started'], (4,))
        self.assertEqual(stats['component_roots_completed'], (4,))
        self.assertEqual(stats['component_roots_feasible'], (4,))
        self.assertGreaterEqual(stats['actual_width_histogram'][4], 1)
        self.assertEqual(len(stats['root_branch_statistics'][0]), 4)

    def test_exhausted_alternative_does_not_erase_complete_root(self):
        result = solve_storage(*chain(), memory_limit=10**7, component_workers=2,
            max_states=2, max_transitions=10000, deadline_seconds=30)
        stats = result.statistics
        self.assertLessEqual(stats['expanded_states'], 2)
        self.assertLessEqual(stats['evaluated_transitions'], 10000)
        # Tail compression may finish an additional root inside the same
        # productive-state budget; the retained complete root must survive.
        self.assertGreaterEqual(stats['component_roots_feasible'][0], 1)
        self.assertTrue(result.plan.feasible)
        self.assertTrue(stats['storage_schedule']['completed'])

    def test_four_eight_factor_complex_traces_and_output_coefficients(self):
        _tied_width.TiedWidthTests()._check_complex_traces(include_ties=True, component_workers=2)


if __name__ == '__main__':
    unittest.main()
