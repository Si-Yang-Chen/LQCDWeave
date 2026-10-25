"""Scalar summaries ignore irrelevant prefix caches, but retain all costs."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.selected_path import materialize_dense_path
from lqcdweave.contraction.planning.optimizers.terminal_completion import expand_completion
from correlator.planning.test_integer_search import make_problem


def scalar_fixture():
    return make_problem([(('A', 'B', 'C'), 'ij,jk,ki->'),
                         (('D', 'E'), 'ij,ji->')],
                        {key: (2, 2) for key in 'ABCDE'})


class ScalarTerminalTests(unittest.TestCase):
    def test_incremental_blocks_match_reference_exactly_across_availability_and_undo(self):
        terms = [((f'A{i}', f'B{i}'), 'i,i->') for i in range(193)]
        problem = make_problem(terms, {name: (i + 1,) for i in range(193)
                                      for name in (f'A{i}', f'B{i}')})
        space = IntegerSpace(problem, profile=CostProfile())

        def compare():
            space._index(lambda: None)
            actual = space._scalar_terminals.completion(space, lambda: None)
            with patch.object(space._scalar_terminals, 'key', return_value=None):
                reference = space.terminal_completion(lambda: None)
            self.assertEqual(actual.seconds, reference.seconds)
            self.assertEqual(tuple(actual.calculations), tuple(reference.calculations))
            return actual

        first = compare()
        saved = tuple(first.calculations)
        choices = list(space.candidates(lambda: None))
        token = space.apply(choices[-1][0], lambda: None)
        second = compare()
        self.assertIs(first.calculations.chunks[0], second.calculations.chunks[0])
        self.assertEqual(tuple(first.calculations), saved)
        space.undo(token)
        self.assertEqual(tuple(compare().calculations), saved)
        identity = choices[70][0].calculation
        result = space.registry.calculations[identity].result
        space.available.add(result)
        space.digest ^= space._part(('available', result))
        space._scalar_terminals.availability(identity, space)
        compare()
        space.available.remove(result)
        space.digest ^= space._part(('available', result))
        space._scalar_terminals.availability(identity, space)
        self.assertEqual(compare().seconds, first.seconds)

    @staticmethod
    def _many_pair_space(count=600):
        terms = [((f'A{i}', f'B{i}'), 'i,i->') for i in range(count)]
        return IntegerSpace(make_problem(terms, {name: (i + 1,) for i in range(count)
                                                 for name in (f'A{i}', f'B{i}')}),
                            profile=CostProfile())

    def test_incremental_fold_reuses_the_unchanged_prefix(self):
        space = self._many_pair_space()
        index = space._scalar_terminals
        index.verify_overrides = True

        def incremental():
            space._index(lambda: None)
            return index.completion(space, lambda: None)

        first = incremental()
        blocks = len(index.chunks)
        self.assertGreater(blocks, 4)
        # The first summary owns nothing yet: it folds every block once.
        self.assertEqual(index.fold_steps, blocks)
        self.assertEqual(index.fold_reuses, 0)
        base = tuple(index.chunks)
        self.assertEqual(index.overrides(base), [])
        choices = list(space.candidates(lambda: None))
        token = space.apply(choices[-1][0], lambda: None)
        walked, reused = index.fold_steps, index.fold_reuses
        second = incremental()
        # One rebuilt block must not cost a re-walk of the whole sequence.
        self.assertLessEqual(index.fold_steps - walked, 2)
        self.assertGreaterEqual(index.fold_reuses - reused, blocks - 2)
        with patch.object(index, 'key', return_value=None):
            reference = space.terminal_completion(lambda: None)
        self.assertEqual(second.seconds, reference.seconds)
        self.assertEqual(tuple(second.calculations), tuple(reference.calculations))
        # The carried patch is exactly the difference against the retained base.
        self.assertEqual(index.overrides(base),
                         [(position, chunk) for position, chunk in enumerate(index.chunks)
                          if chunk is not base[position]])
        self.assertIsNone(index.overrides(index.chunks[:-1]))
        space.undo(token)
        walked, reused = index.fold_steps, index.fold_reuses
        third = incremental()
        with patch.object(index, 'key', return_value=None):
            reference = space.terminal_completion(lambda: None)
        self.assertEqual(third.seconds, reference.seconds)
        self.assertEqual(tuple(third.calculations), tuple(reference.calculations))
        self.assertLessEqual(index.fold_steps - walked, 2)
        self.assertGreaterEqual(index.fold_reuses - reused, blocks - 2)

    def test_interrupted_fold_resumes_at_the_first_stale_slot(self):
        space = self._many_pair_space()
        index = space._scalar_terminals
        space._index(lambda: None)
        index.completion(space, lambda: None)
        with patch.object(index, 'key', return_value=None):
            reference = space.terminal_completion(lambda: None)
        # A block whose contents did not change still restarts the fold at its
        # own slot, which is where the interrupted walk has to resume.
        index.dirty_buckets.add(index.order[2])
        reused = index.fold_reuses
        self.assertGreaterEqual(len(index.order) - 2, 2)

        def checkpoint():
            # The dirty rebuild runs first; only the fold may be interrupted.
            if index.dirty_buckets:
                return
            raise InterruptedError('fold checkpoint')

        with self.assertRaises(InterruptedError):
            index.completion(space, checkpoint)
        actual = index.completion(space, lambda: None)
        self.assertEqual(actual.seconds, reference.seconds)
        self.assertEqual(tuple(actual.calculations), tuple(reference.calculations))
        self.assertGreaterEqual(index.fold_reuses - reused, 2)

    def test_supplied_block_overrides_are_verified_in_debug(self):
        from lqcdweave.contraction.planning.optimizers.terminal_completion import (
            SharedChunks, TerminalCompletions)
        cache = TerminalCompletions()
        base = tuple((name,) for name in 'abcdefghijklmnopqrst')
        cache.sequence = base
        chunks = base[:5] + (('X',),) + base[6:]
        overrides = [(5, ('X',))]
        shared = cache._share(chunks, True, overrides)
        self.assertIsInstance(shared, SharedChunks)
        self.assertEqual(tuple(shared), chunks)
        with self.assertRaises(RuntimeError):
            cache._share(chunks, True, [])
        with self.assertRaises(RuntimeError):
            cache._share(chunks, True, [(5, ('X',)), (7, base[7])])
        # Without a supplied patch the walk still derives the same one.
        cache.sequence = base
        self.assertEqual(tuple(cache._share(chunks, True)), chunks)

    def test_interrupted_incremental_summary_can_resume(self):
        space = IntegerSpace(scalar_fixture(), profile=CostProfile())
        while any(len(term.factors) > 2 for term in space.terms if not term.done):
            op, _ = next(space.search_candidates(lambda: None))
            space.apply(op, lambda: None)
        space._index(lambda: None)
        index = space._scalar_terminals
        checks = 0
        def checkpoint():
            nonlocal checks
            checks += 1
            if checks == 2:
                raise InterruptedError('summary checkpoint')
        with self.assertRaises(InterruptedError):
            index.completion(space, checkpoint)
        actual = index.completion(space, lambda: None)
        with patch.object(index, 'key', return_value=None):
            reference = space.terminal_completion(lambda: None)
        self.assertEqual(actual.seconds, reference.seconds)
        self.assertEqual(tuple(actual.calculations), tuple(reference.calculations))

    def test_distinct_completed_prefixes_reuse_entire_tail_and_match_numpy(self):
        problem = scalar_fixture()
        space = IntegerSpace(problem, profile=CostProfile())

        def prefix(pair):
            choices = list(space.candidates(lambda: None))
            op = next(op for op, _ in choices if
                      set(space.registry.calculations[op.calculation].recipe.operands) == set(pair))
            tokens, operations = [space.apply(op, lambda: None)], [op]
            op = next(op for op, _ in space.candidates(lambda: None)
                      if 0 in space.groups[op.calculation])
            tokens.append(space.apply(op, lambda: None))
            operations.append(op)
            return tokens, operations

        tokens, first = prefix('AB')
        original = space.terminal_completion(lambda: None)
        first_key = space.fingerprint()
        first_state, _ = materialize_dense_path(problem, expand_completion(original, first, space.registry),
                                               sum(op.seconds for op in first)+original.seconds, space.registry)
        for token in reversed(tokens):
            space.undo(token)
        _, second = prefix('BC')
        with patch.object(space._terminal_completions, '_family_options',
                          side_effect=AssertionError('repeated family scan')):
            current = space.terminal_completion(lambda: None)
        self.assertIs(current, original)
        self.assertNotEqual(space.fingerprint(), first_key)
        second_state = space.audit(expand_completion(current, second, space.registry),
                                   sum(op.seconds for op in second)+current.seconds)
        rng = np.random.default_rng(125)
        arrays = {key: rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)) for key in 'ABCDE'}
        expected = (np.einsum('ij,jk,ki->', arrays['A'], arrays['B'], arrays['C'], optimize=False),
                    np.einsum('ij,ji->', arrays['D'], arrays['E'], optimize=False))
        for state in (first_state, second_state):
            values = replay(state, lambda key, source: arrays[key], shared_arrays=dict(arrays))
            np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)

    def test_relevant_availability_changes_key_and_price(self):
        problem = make_problem([(('A', 'B'), 'i,i->')], {'A': (2,), 'B': (2,)})
        space = IntegerSpace(problem, profile=CostProfile())
        original = space.terminal_completion(lambda: None)
        identity = next(iter(space.groups))
        calc = space.registry.calculations[identity]
        data = space.registry.data[calc.result]
        space.available.add(calc.result)
        space.catalog[data.name] = data.spec
        space.digest ^= space._part(('available', calc.result))
        space._scalar_terminals.availability(identity, space)
        cached = space.terminal_completion(lambda: None)
        self.assertIsNot(cached, original)
        self.assertEqual(cached.seconds, 0)
        space.available.remove(calc.result)
        del space.catalog[data.name]
        space.digest ^= space._part(('available', calc.result))
        space._scalar_terminals.availability(identity, space)
        self.assertIs(space.terminal_completion(lambda: None), original)

    def test_new_alternative_recipe_disables_semantic_key(self):
        space = IntegerSpace(scalar_fixture(), profile=CostProfile())
        # Finish the productive expression; only an independent scalar tail remains.
        while any(len(term.factors) > 2 for term in space.terms if not term.done):
            op, _ = next(space.search_candidates(lambda: None))
            space.apply(op, lambda: None)
        original = space.terminal_completion(lambda: None)
        self.assertIsNotNone(space._scalar_terminals.key(space))
        identity = next(iter(space.groups))
        calc = space.registry.calculations[identity]
        space.registry.calculation(replace(calc.recipe, seconds=calc.seconds+1),
                                   space.registry.data[calc.result].spec, calc.workspace)
        self.assertTrue(space.registry.has_alternative_calculations)
        self.assertIsNone(space._scalar_terminals.key(space))
        self.assertIsNot(space.terminal_completion(lambda: None), original)

    def test_uncovered_row_with_no_legal_pair_cannot_use_scalar_cache(self):
        lines = ('abc', 'ade', 'afg', 'bdf', 'beg', 'cdg', 'cef')
        names = tuple(f'F{i}' for i in range(7))
        problem = make_problem([(names, ','.join(lines)+'->'), (('D', 'E'), 'i,i->')],
                               {**{key: (2, 2, 2) for key in names}, 'D': (2,), 'E': (2,)})
        space = IntegerSpace(problem, profile=CostProfile())
        self.assertIsNone(space.terminal_completion(lambda: None))
        self.assertEqual(space._scalar_terminals.coverage, 1)
        self.assertEqual(space.remaining, 2)
        self.assertIsNone(space._scalar_terminals.key(space))

    def test_incremental_key_matches_rebuild_across_undo_reset_release(self):
        space = IntegerSpace(scalar_fixture(), profile=CostProfile())
        def verify():
            space._index(lambda: None)
            expected = coverage = 0
            for identity, family in space.terminal_families.items():
                reused = space.registry.calculations[identity].result in space.available
                expected ^= space._part(('scalar-terminal', identity, family, reused))
                coverage += len(space.groups[identity])
            self.assertEqual(expected, space._scalar_terminals.digest)
            self.assertEqual(coverage, space._scalar_terminals.coverage)
        def visit(depth):
            verify()
            if depth:
                for op, _ in list(space.candidates(lambda: None))[:2]:
                    token = space.apply(op, lambda: None)
                    visit(depth-1)
                    space.undo(token)
                    verify()
        visit(3)
        space.reset()
        verify()
        space.release()
        self.assertFalse(space._scalar_terminals.entries)
        self.assertFalse(space._scalar_terminals.parts)

    def test_debug_and_free_outputs_keep_state_key_reference(self):
        self.assertIsNone(IntegerSpace(scalar_fixture(), profile=CostProfile(), debug=True)._scalar_terminals)
        problem = make_problem([(('A', 'B'), 'ij,jk->ik')], {'A': (2, 2), 'B': (2, 2)})
        self.assertIsNone(IntegerSpace(problem, profile=CostProfile())._scalar_terminals)


if __name__ == '__main__':
    unittest.main()
