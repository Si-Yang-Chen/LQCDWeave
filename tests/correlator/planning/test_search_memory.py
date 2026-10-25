"""Static memory indexing preserves complete dynamic-root budget estimates."""
from dataclasses import dataclass
import os
import sys
import unittest
from unittest.mock import patch

from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.budget import (
    BudgetExhausted, SearchBudget, StaticMemory, retained_size,
)
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from correlator.planning.test_integer_search import make_problem, solve


@dataclass
class MutableNode:
    value: object


class SearchMemoryTests(unittest.TestCase):
    def test_unlimited_budget_does_not_traverse_or_report_a_measured_peak(self):
        budget = SearchBudget()
        with patch('lqcdweave.contraction.planning.optimizers.budget.retained_size',
                   side_effect=AssertionError('unrequested memory traversal')):
            budget.sample_memory((object(),))
        self.assertEqual(budget.peak_memory, 0)
        self.assertIsNone(budget.sampled_peak_memory)

    def test_unlimited_public_search_skips_memory_index_and_walk_but_keeps_the_plan(self):
        problem = make_problem([(('A', 'B'), 'i,i->')], {'A': (3,), 'B': (3,)})
        for optimizer in ('anytime', 'exact'):
            with self.subTest(optimizer=optimizer):
                reference = solve(problem, optimizer=optimizer, search_memory_limit=1 << 26)
                with (
                    patch('lqcdweave.contraction.planning.optimizers.budget.retained_size',
                          side_effect=AssertionError('unrequested memory traversal')),
                    patch('lqcdweave.contraction.planning.optimizers.graph_session.StaticMemory',
                          side_effect=AssertionError('unrequested memory index')),
                ):
                    actual = solve(problem, optimizer=optimizer)
                self.assertEqual(actual.state.actions, reference.state.actions)
                self.assertEqual(actual.state.results, reference.state.results)
                self.assertEqual(actual.plan.kernel_batches, reference.plan.kernel_batches)
                self.assertEqual(actual.plan.seconds, reference.plan.seconds)
                self.assertIsNone(actual.statistics['sampled_peak_search_bytes'])
                self.assertIsNotNone(reference.statistics['sampled_peak_search_bytes'])
                if optimizer == 'anytime':
                    self.assertEqual(actual.statistics['static_memory_index_bytes'], 0)
                    self.assertEqual(actual.statistics['static_registry_definition_counts'], (0, 0, 0))

    def test_unlimited_parallel_search_skips_owner_memory_walk(self):
        problem = make_problem([(('A', 'B'), 'i,i->'), (('C', 'D'), 'i,i->')],
                               {key: (3,) for key in 'ABCD'})
        with (
            patch('lqcdweave.contraction.planning.optimizers.budget.retained_size',
                  side_effect=AssertionError('unrequested memory traversal')),
            patch('lqcdweave.contraction.planning.optimizers.parallel_components.retained_size',
                  side_effect=AssertionError('unrequested owner memory traversal')),
            patch('lqcdweave.contraction.planning.optimizers.graph_session.StaticMemory',
                  side_effect=AssertionError('unrequested memory index')),
        ):
            actual = solve(problem, component_workers=2)
        self.assertTrue(actual.plan.feasible)
        # A single-core allocation still exercises a real child process.
        self.assertIn(actual.statistics['component_workers'], (1, 2))
        self.assertTrue(actual.statistics['component_worker_pids'])
        self.assertNotIn(os.getpid(), actual.statistics['component_worker_pids'])
        self.assertIsNone(actual.statistics['sampled_peak_search_bytes'])
        for statistics in actual.statistics['component_statistics']:
            self.assertIsNone(statistics['sampled_peak_search_bytes'])

    def test_static_shared_graph_is_counted_once_and_index_is_budgeted(self):
        shared = tuple(('source-' + str(i), tuple(range(i, i + 4))) for i in range(50))
        static = StaticMemory()
        static.include((shared,))
        dynamic = MutableNode([shared, {'alias': shared}])
        roots = (dynamic, shared)
        self.assertEqual(retained_size(roots, static=static), retained_size(roots) + static.overhead)
        overhead = (sys.getsizeof(static) + sys.getsizeof(static.ids)
                    + sum(sys.getsizeof(identity) for identity in static.ids)
                    + sys.getsizeof(static.anchors) + sys.getsizeof(static.total)
                    + sys.getsizeof(static.id_bytes))
        self.assertEqual(static.overhead, overhead)
        static.include((shared,))
        self.assertEqual(retained_size(roots, static=static), retained_size(roots) + static.overhead)

    def test_dynamic_growth_removal_cached_attributes_and_cycles_still_count(self):
        shared = ('stable', (1001, 1002, 1003))
        static = StaticMemory()
        static.include((shared,))
        node = MutableNode([shared])
        for values in ([shared], [shared, bytes(30000)], []):
            node.value = values
            node.cached = {'self': node, 'shared': shared}
            self.assertEqual(retained_size((node, shared), static=static),
                             retained_size((node, shared)) + static.overhead)

    def test_incremental_component_constants_keep_shared_children_unique(self):
        shared = ('input', tuple(range(20)))
        first, second = (shared, 'first'), (shared, 'second')
        static = StaticMemory()
        static.include((first,))
        static.include((second,))
        self.assertEqual(static.total, retained_size((first, second)))
        self.assertEqual(retained_size((first, second), static=static),
                         retained_size((first, second)) + static.overhead)

    def test_same_sample_enforces_limit_after_dynamic_growth(self):
        shared = ('input', tuple(range(20)))
        static = StaticMemory()
        static.include((shared,))
        dynamic = []
        size = retained_size((shared, dynamic), static=static)
        budget = SearchBudget(search_memory_limit=size)
        budget.sample_memory((shared, dynamic), static=static)
        dynamic.append(bytes(10000))
        with self.assertRaisesRegex(BudgetExhausted, 'search_memory_limit'):
            budget.sample_memory((shared, dynamic), static=static)
        self.assertGreater(budget.peak_memory, size)

    def test_integer_constants_stay_fixed_across_apply_undo_reset_and_release(self):
        problem = make_problem([(('A', 'B', 'C'), 'ij,jk,kl->il')] * 8,
                               {'A': (2, 3), 'B': (3, 4), 'C': (4, 2)})
        space = IntegerSpace(problem, profile=CostProfile())
        constants = space.memory_constants()
        static = StaticMemory()
        static.include(constants)

        def compare():
            self.assertEqual(space.memory_constants(), constants)
            self.assertEqual(static.total, retained_size(constants))
            self.assertEqual(retained_size((space,), static=static),
                             retained_size((space,)) + static.overhead)

        compare()
        operation, _ = next(space.search_candidates(lambda: None))
        token = space.apply(operation, lambda: None)
        list(space.search_candidates(lambda: None))
        compare()
        space.undo(token)
        compare()
        space.reset()
        compare()
        space.release()
        compare()

    def test_growing_registry_indexes_only_new_records_and_keeps_mutable_tables_visible(self):
        problem = make_problem([(('A', 'B', 'C'), 'ij,jk,kl->il')] * 8,
                               {'A': (2, 3), 'B': (3, 4), 'C': (4, 2)})
        space = IntegerSpace(problem, profile=CostProfile())
        registry = space.registry
        static = StaticMemory()
        static.include(space.memory_constants())
        positions, indexed_records = (0, 0, 0), set()

        def compare():
            nonlocal positions
            roots, ends = registry.memory_constants_since(positions)
            roots = tuple(roots)
            expected = tuple(registry.data[positions[0]:]) + tuple(registry.patterns[positions[1]:])
            expected += tuple(registry.calculations[positions[2]:])
            self.assertEqual(roots, expected)
            self.assertTrue(indexed_records.isdisjoint(map(id, roots)))
            indexed_records.update(map(id, roots))
            static.include(roots)
            positions = ends
            live = (space, positions)
            self.assertEqual(retained_size(live, static=static), retained_size(live) + static.overhead)

        compare()
        first_count = len(indexed_records)
        operation, _ = next(space.search_candidates(lambda: None))
        compare()
        token = space.apply(operation, lambda: None)
        list(space.search_candidates(lambda: None))
        compare()
        self.assertGreater(len(indexed_records), first_count)
        space.undo(token)
        compare()
        space.reset()
        compare()
        space.release()
        compare()

        connections = []
        registry.connection_tables.append(connections)
        compare()
        budget = SearchBudget(search_memory_limit=retained_size((space, positions), static=static))
        connections.append(bytes(30000))
        compare()
        with self.assertRaisesRegex(BudgetExhausted, 'search_memory_limit'):
            budget.sample_memory((space, positions), static=static)
        connections.clear()
        compare()

    def test_real_multicomponent_samples_match_the_full_object_walk(self):
        problem = make_problem([(('A', 'B', 'C'), 'ij,jk,kl->il')] * 4
                               + [(('D', 'E', 'F'), 'ij,jk,kl->il')] * 4,
                               {key: (2, 2) for key in 'ABCDEF'})
        original, samples = SearchBudget.sample_memory, []

        def sample(budget, roots, *, static=None):
            self.assertIsNotNone(static)
            self.assertEqual(retained_size(roots, static=static), retained_size(roots) + static.overhead)
            samples.append(roots[-1])
            return original(budget, roots, static=static)

        with patch.object(SearchBudget, 'sample_memory', sample):
            result = solve(problem, root_width=1, branch_width=1, search_memory_limit=1 << 28)
        self.assertTrue(result.plan.feasible)
        self.assertEqual(result.statistics['component_count'], 2)
        self.assertTrue(samples)
        counts = result.statistics['static_registry_definition_counts']
        self.assertEqual(samples[-1], counts)
        self.assertGreater(counts[0], len(problem.specs))
        self.assertGreater(counts[1], 0)
        self.assertGreater(counts[2], 0)


if __name__ == '__main__':
    unittest.main()
