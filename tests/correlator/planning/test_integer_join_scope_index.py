"""The integer planner prepares each baryon join only in its owning scene."""
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, CoreComponent,
    SourceSpec, TensorCoreSpec, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.shared_components import component_problem


class IntegerJoinScopeIndexTests(unittest.TestCase):
    def test_scoped_preparation_matches_full_scan_with_interleaved_joins(self):
        shapes = {"S1": (4, 4, 4), "S2": (4, 4, 4),
                  "E1": (2, 2, 2), "E2": (2, 2, 2),
                  "P": (4, 2), "Q": (4, 2), "R": (4, 2)}
        specs = {key: TensorSpec(shape, "complex128") for key, shape in shapes.items()}
        core = TensorCoreSpec(
            "baryon", ("spin0", "spin1", "spin2", "laplace_vector0",
                       "laplace_vector1", "laplace_vector2"),
            TensorSpec((4, 4, 4, 2, 2, 2), "complex128"),
            (CoreComponent(("S1", "E1")), CoreComponent(("S2", "E2"))),
        )
        term = ContractionTerm(("P", "Q", "R"), "ax,by,cz->")
        joins = (BaryonJoinSpec(3, "last", "S1", "E1", "abc", "xyz", core=core),
                 BaryonJoinSpec(0, "first", "S1", "E1", "abc", "xyz", core=core),
                 BaryonJoinSpec(2, "middle", "S1", "E1", "abc", "xyz", core=core),
                 BaryonJoinSpec(0, "second", "S1", "E1", "abc", "xyz", core=core))
        source = ContractionProblem(
            specs, (term,) * 5, {key: SourceSpec(key, 0) for key in specs},
            baryon_joins=joins,
        )
        observed = []

        def tracked(problem, scenes, *, _scoped_joins=None):
            self.assertIs(problem, source)
            self.assertIsNotNone(_scoped_joins)
            selected = tuple(_scoped_joins)
            observed.append((scenes[0], selected))
            local = component_problem(problem, scenes, _scoped_joins=selected)
            self.assertEqual(local, component_problem(problem, scenes))
            return local

        with patch("lqcdweave.contraction.planning.optimizers.integer_space.component_problem",
                   side_effect=tracked):
            space = IntegerSpace(source, profile=CostProfile())
        self.assertEqual(tuple(scene for scene, _ in observed), tuple(range(5)))
        self.assertEqual(tuple(tuple(join.name for join in selected) for _, selected in observed),
                         (("first", "second"), (), ("middle",), ("last",), ()))
        self.assertEqual(sum(len(selected) for _, selected in observed), len(joins))
        self.assertEqual(tuple(join.name for join in source.baryon_joins),
                         ("last", "first", "middle", "second"))
        self.assertEqual(set(space.contexts[0].specs), set(specs))
        self.assertEqual(tuple(join.name for join in space.contexts[0].baryon_joins),
                         ("first", "second"))


if __name__ == "__main__":
    unittest.main()
