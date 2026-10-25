"""A retained tail shares one block sequence instead of copying it per state.

Successive retained tails repeat almost the same ordered block sequence, so a
tail keeps only the positions where its sequence differs. Iteration order, the
block objects and the delivered winner must be identical to the plain tuple
representation, the retained-size accounting must follow the shared base, and
the shared sequence check must be re-openable in debug.
"""
import unittest
from unittest.mock import patch

from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers import terminal_completion as terminal
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.terminal_completion import (
    CalculationChunks, SharedChunks, TerminalCompletions, expand_completion)
from correlator.planning.test_integer_search import make_problem

BLOCKS = 64


def fixture(count=5):
    shapes, terms = {}, []
    for i in range(count):
        a, b = f"A{i}", f"B{i}"
        shapes.update({a: (2, 3), b: (3, 4)})
        terms.extend([((a, b), "ij,jk->ki"), ((a, b), "ab,bc->ac")])
    return make_problem(terms, shapes)


def plain_sharing(chunks, debug=False):
    """The representation the release path uses when nothing is shared."""
    return chunks


def blocks(count):
    return tuple(tuple(range(4 * i, 4 * i + 4)) for i in range(count))


class BlockSequenceTests(unittest.TestCase):
    def test_completion_blocks_are_deduplicated_objects_in_cache_order(self):
        space = IntegerSpace(fixture(200), profile=CostProfile())
        original = space.terminal_completion(lambda: None)
        op, _ = next(space.search_candidates(lambda: None))
        space.apply(op, lambda: None)
        other = space.terminal_completion(lambda: None)
        ids = tuple(original.calculations)
        self.assertEqual(ids, tuple(identity for chunk in original.calculations.chunks
                                     for identity in chunk))
        self.assertEqual(len(ids), len(original.calculations))
        # Unchanged blocks stay one shared object across different tails.
        common = {id(chunk) for chunk in original.calculations.chunks}.intersection(
                 id(chunk) for chunk in other.calculations.chunks)
        self.assertTrue(common)


class SharedSequenceTests(unittest.TestCase):
    """Representation-only sharing: same length, same order, same objects."""

    def setUp(self):
        self.cache = TerminalCompletions()
        self.base = blocks(BLOCKS)

    def variant(self, position):
        return self.base[:position] + ((10 ** 6,),) + self.base[position + 1:]

    def test_few_differences_are_shared_and_replay_the_plain_sequence(self):
        self.assertIs(self.cache._share(self.base), self.base)
        variant = self.variant(3)
        shared = self.cache._share(variant)
        self.assertIsInstance(shared, SharedChunks)
        self.assertEqual(self.cache.shared_sequences, 1)
        self.assertEqual(len(shared), len(variant))
        self.assertEqual(tuple(shared), variant)
        self.assertIs(shared[3], variant[3])
        self.assertIs(shared[0], self.base[0])
        self.assertIs(shared[-1], self.base[-1])
        self.assertEqual([id(chunk) for chunk in shared],
                         [id(chunk) for chunk in variant])
        self.assertIs(self.cache.sequence, self.base)

    def test_identical_sequence_reuses_the_retained_base(self):
        self.cache._share(self.base)
        self.assertIs(self.cache._share(tuple(self.base)), self.base)
        self.assertEqual(self.cache.reused_sequences, 1)
        self.assertEqual(self.cache.shared_sequences, 0)

    def test_a_large_patch_is_retained_plainly_instead_of_indirectly(self):
        self.cache._share(self.base)
        other = tuple((1000 + i,) for i in range(BLOCKS))
        self.assertIs(self.cache._share(other), other)
        self.assertEqual(self.cache.shared_sequences, 0)
        self.assertIs(self.cache.sequence, other)

    def test_a_different_length_never_shares_a_base(self):
        self.cache._share(self.base)
        shorter = self.base[:3]
        self.assertIs(self.cache._share(shorter), shorter)
        self.assertIs(self.cache.sequence, shorter)

    def test_clearing_the_caches_drops_only_the_retained_base(self):
        self.cache._share(self.base)
        shared = self.cache._share(self.variant(3))
        self.cache.clear()
        self.assertIsNone(self.cache.sequence)
        self.assertEqual(tuple(shared), self.variant(3))

    def test_retained_size_walk_follows_a_shared_sequence(self):
        from lqcdweave.contraction.planning.optimizers.budget import retained_walk
        shared = SharedChunks(self.base, ((3, (10 ** 6,)),))
        total, objects = retained_walk((shared,), set(), 0)
        # The base blocks and the override are reachable, not only the wrapper.
        self.assertGreaterEqual(total, sum(len(chunk) * 8 for chunk in self.base))
        self.assertGreater(objects, len(self.base))

    def test_debug_reopens_the_shared_sequence_check(self):
        class Corrupt(SharedChunks):
            def __iter__(self):
                return iter(self.base)

        self.cache._share(self.base)
        variant = self.variant(3)
        with patch.object(terminal, "SharedChunks", Corrupt):
            with self.assertRaisesRegex(RuntimeError, "shared terminal block sequence"):
                self.cache._share(variant, debug=True)
        # The release path retains the same variant as a correct shared sequence.
        self.assertEqual(tuple(self.cache._share(variant)), variant)


class SharedDeliveryTests(unittest.TestCase):
    """A shared sequence is transparent to winner expansion and delivery."""

    def search(self, problem, sharing):
        space = IntegerSpace(problem, profile=CostProfile())
        engine = FingerprintDFS(space, root_width=4, branch_width=4)
        if sharing:
            result = engine.run()
        else:
            with patch.object(TerminalCompletions, "_share", staticmethod(plain_sharing)):
                result = engine.run()
        return engine, result

    def test_shared_tail_delivers_the_same_operations_as_the_plain_one(self):
        problem = fixture(6)
        shared, first = self.search(problem, True)
        plain, second = self.search(problem, False)
        self.assertEqual(first.status, "FEASIBLE")
        self.assertEqual(second.status, "FEASIBLE")
        self.assertEqual(first.statistics["logical_best_seconds"],
                         second.statistics["logical_best_seconds"])
        self.assertEqual(tuple(shared.selected_operations()),
                         tuple(plain.selected_operations()))
        self.assertTrue(shared.best_plan.done and plain.best_plan.done)

    def test_shared_sequence_expands_exactly_like_the_plain_one(self):
        space = IntegerSpace(fixture(200), profile=CostProfile())
        original = space.terminal_completion(lambda: None)
        self.assertIsNotNone(original)
        chunks = original.calculations.chunks
        plain = expand_completion(original, (), space.registry)
        for overrides in ((), ((0, chunks[0]),)):
            recipe = CalculationChunks(SharedChunks(chunks, overrides), original.calculations.size)
            self.assertEqual(len(recipe), original.calculations.size)
            self.assertEqual(tuple(recipe), tuple(original.calculations))
            self.assertEqual(expand_completion(
                type(original)(original.seconds, recipe), (), space.registry), plain)


if __name__ == "__main__":
    unittest.main()