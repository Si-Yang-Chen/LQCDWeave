"""Compare cost-bound truncation with independent complete enumeration."""
from dataclasses import replace
import math
import sys
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture, ContractionTerm
from lqcdweave.contracts.correlator.api import TensorSpec
from lqcdweave.contraction.planning.logical_index import LayoutIndex
from lqcdweave.contraction.planning.optimizers.budget import retained_size, BudgetExhausted


class BoundedStepTests(unittest.TestCase):
    def test_ranked_steps_equal_complete_preview_in_small_pressured_domains(self):
        # Check the retained domain directly, without the retired DFS ranker.
        terms = (ContractionTerm(('A', 'B', 'C'), 'i,i,i->'),
                 ContractionTerm(('A', 'C'), 'i,i->i'), ContractionTerm(('B', 'C'), 'i,i->i'))
        for cap in (264, 10000):
            space = fixture(terms, cap=cap)
            queue, seen = [space.initial_state()], set()
            while queue and len(seen) < 48:
                cursor = queue.pop()
                if space.key(cursor) in seen or cursor.state.done:
                    continue
                seen.add(space.key(cursor))
                complete = tuple(space.iter_successors(cursor))
                increment = lambda edge: space.cost(edge.state) - space.cost(cursor)
                for width in (1, 2, 4):
                    expected = sorted(complete, key=increment)[:width]
                    selected = []
                    def cutoff():
                        return increment(selected[-1]) if len(selected) == width else math.inf
                    for edge in space.iter_bounded_successors(cursor, cutoff, lambda: None):
                        selected.append(edge)
                        selected.sort(key=increment)
                        del selected[width:]
                    self.assertEqual(selected, expected)
                    for edge in selected:
                        self.assertEqual(space.materialize(cursor, edge.operation), edge)
                queue.extend(edge.state for edge in complete)
            self.assertGreater(len(seen), 3)
            self.assertGreater(space.omitted_previews, 0)


    def test_descriptions_do_not_construct_shared_updates_and_restore_exact_representatives(self):
        space = fixture((ContractionTerm(('A', 'B'), 'i,i->'), ContractionTerm(('A', 'C'), 'i,i->')))
        root = space.initial_state()
        with patch.object(space, '_materialize_logical', side_effect=AssertionError('eager effects')):
            first = tuple(space.candidates(root))
            second = tuple(space.candidates(root))
        self.assertEqual(first, second)
        self.assertEqual(space.logical_effects, 0)
        self.assertGreater(space.description_hits, 0)
        for operation in second:
            self.assertEqual(space.preview(root, operation).operation.term_index, operation.term_index)


    def test_all_memory_failures_are_checked_before_adjustments_even_with_low_cutoff(self):
        terms = (ContractionTerm(('A', 'B', 'C'), 'i,i,i->'),
                 ContractionTerm(('A', 'C'), 'i,i->i'), ContractionTerm(('B', 'C'), 'i,i->i'))
        space = fixture(terms, cap=264)
        cursor = space.successors(space.initial_state())[0].state
        expected = space.successors(cursor)
        batch = space.iter_bounded_successors(cursor, lambda: -math.inf, lambda: None)
        self.assertEqual(tuple(batch), expected)
        self.assertEqual(batch.omitted, 0)
        self.assertTrue(expected)
        self.assertTrue(all(e.operation.kind in ('discard', 'spill_disk') for e in expected))

    def test_unknown_recovery_uses_complete_preview_and_bound_is_exact_without_recovery(self):
        space = fixture((ContractionTerm(('A', 'B'), 'i,i->'), ContractionTerm(('A', 'C'), 'i,i->')))
        root = space.initial_state()
        for candidate in space._logical_candidates(root, lambda: None):
            edge = space.evaluate(root, candidate)
            self.assertEqual(space._increment_bound(root, candidate), space.cost(edge.state) - space.cost(root))
        first = space.successors(root)[0]
        cold = space.apply(first.state, space._move(first.state, 'spill_disk', 'tensor', 'A'))
        candidates = tuple(space._logical_candidates(cold, lambda: None))
        self.assertTrue(candidates)
        self.assertTrue(all(space._increment_bound(cold, c) is None for c in candidates))

    def test_index_retargeting_matches_fresh_indexes_after_backtracking_and_geometry_change(self):
        term = ContractionTerm(('A', 'B', 'C'), 'i,i,i->')
        space = fixture((term,) * 8)
        root = space.initial_state()
        siblings = space.successors(root)
        for cursor in (siblings[0].state, siblings[-1].state, root):
            current = tuple(space.logical_steps(cursor))
            space.layout_index = None
            self.assertEqual(current, tuple(space.logical_steps(cursor)))
        state = space._symbolic(root)
        index = LayoutIndex(state)
        original = index.layout(0, (0, 1))
        changed = replace(state, catalog=state.catalog.updated((('A', TensorSpec((8,), 'float64')),
                                                               ('B', TensorSpec((8,), 'float64')),
                                                               ('C', TensorSpec((8,), 'float64')))))
        index.advance(changed, ())
        self.assertEqual(index.layout(0, (0, 1)), changed._contract_layout(0, (0, 1)))
        self.assertNotEqual(original[2], index.layout(0, (0, 1))[2])

    def test_interleaved_generators_and_interrupted_retarget_do_not_share_mutable_index(self):
        term = ContractionTerm(('A', 'B', 'C'), 'i,i,i->')
        space = fixture((term,) * 8)
        root = space.initial_state()
        expected = tuple(space.logical_steps(root))
        iterator = space.logical_steps(root)
        self.assertEqual(next(iterator), expected[0])
        child = space.preview(root, expected[0].actions[0]).state
        tuple(space.logical_steps(child))
        self.assertEqual(tuple(iterator), expected[1:])
        def interrupt():
            raise BudgetExhausted('test')
        with self.assertRaises(BudgetExhausted):
            tuple(space.logical_steps(child, interrupt))
        self.assertIsNone(space.layout_index)
        self.assertFalse(space.layout_index_active)
        self.assertEqual(tuple(space.logical_steps(root)), expected)

    def test_retained_size_counts_leaf_identities_and_container_subclasses_exactly(self):
        class TupleSubclass(tuple):
            pass
        leaves = [None, True, 1000, 1.5, 2j, 'leaf', b'leaf', range(7)]
        wrapper = TupleSubclass(leaves)
        expected = sys.getsizeof(wrapper) + sum(sys.getsizeof(v) for v in leaves)
        self.assertEqual(retained_size([wrapper, wrapper]), expected)
        self.assertEqual(retained_size([wrapper, *leaves]), expected)
