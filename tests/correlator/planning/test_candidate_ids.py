from dataclasses import dataclass
import unittest

from correlator.planning.test_macro_steps import fixture
from lqcdweave.contraction.planning.optimizers.choices import OperationPool


@dataclass(frozen=True)
class CollidingOperation:
    name: str

    def __hash__(self):
        return 7


class CandidateIDTests(unittest.TestCase):
    def test_hash_collisions_compare_descriptors_and_equal_operations_share_id(self):
        pool = OperationPool()
        first, second = CollidingOperation("first"), CollidingOperation("second")
        a, b = pool.intern(first), pool.intern(second)
        self.assertNotEqual(a, b)
        self.assertEqual(pool.intern(CollidingOperation("first")), a)
        self.assertIs(pool[a], first)
        self.assertIs(pool[b], second)
        self.assertEqual(len(pool), 2)


    def test_macro_recipe_recreates_the_entire_fixed_step(self):
        space = fixture()
        parent = space.initial_state()
        previews = tuple(space.iter_successors(parent))
        self.assertEqual(len(previews), 3)
        for preview in previews:
            actual = space.materialize(parent, preview.operation)
            self.assertEqual(actual, preview)
            self.assertEqual(parent.state.actions, ())
            self.assertEqual(parent.resources.allocations, ())


if __name__ == "__main__":
    unittest.main()
