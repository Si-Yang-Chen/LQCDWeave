"""Selected-path scopes preserve algebra without rescanning unrelated rows."""
from dataclasses import replace
import math
import unittest

import numpy as np
import opt_einsum

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, CoreComponent,
    SourceSpec, TensorCoreSpec, TensorSpec,
)
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.canonical_operations import describe, apply_description
from lqcdweave.contraction.planning.state import ContractionState, _RowTable


class NoGlobalIteration(tuple):
    def __iter__(self):
        raise AssertionError("selected operation scanned all scene bindings")


def fixture(multicomponent=False):
    entries = ((0, 1, 2, 1 + .25j), (3, 2, 1, -.5j))
    shapes = {"G": (4, 4, 4), "H": (4, 4, 4), "E": (2, 2, 2), "F": (2, 2, 2),
              "P": (4, 4, 2, 2), "Q": (4, 4, 2, 2), "R": (4, 4, 2, 2), "X": (2,)}
    core = None
    if multicomponent:
        shapes.update(G2=(4, 4, 4), E2=(2, 2, 2))
        core = TensorCoreSpec("baryon",
            ("spin0", "spin1", "spin2", "laplace_vector0", "laplace_vector1", "laplace_vector2"),
            TensorSpec((4, 4, 4, 2, 2, 2), "complex128"),
            (CoreComponent(("G", "E"), 2 + .5j, entries),
             CoreComponent(("G2", "E2"), -.75, entries)))
    joins = tuple(BaryonJoinSpec(scene, name, spin, spatial, spins, modes,
                                sparse_spin=entries, core=core if name == "sink" else None)
                  for name, spin, spatial, spins, modes in
                  (("sink", "G", "E", "ijk", "abc"), ("source", "H", "F", "lmn", "def"))
                  for scene in range(2))
    specs = {key: TensorSpec(shape, "complex128") for key, shape in shapes.items()}
    problem = ContractionProblem(specs,
        (ContractionTerm(("P", "Q", "R", "X"), "ilad,jmbe,kncf,a->"),) * 2,
        {key: SourceSpec(key, math.prod(spec.shape) * spec.dtype.itemsize)
         for key, spec in specs.items()}, baryon_joins=joins)
    rng = np.random.default_rng(186)
    arrays = {key: rng.normal(size=shape) + 1j * rng.normal(size=shape) for key, shape in shapes.items()}
    for key in ("G", "H", "G2"):
        if key in arrays:
            arrays[key][:] = 0
            for a, b, c, value in entries:
                arrays[key][a, b, c] = value
    return problem, arrays


class ScopedSelectedOperationsTests(unittest.TestCase):
    def test_scoped_route_matches_full_scan_and_independent_numeric_reference(self):
        for multicomponent in (False, True):
            with self.subTest(multicomponent=multicomponent):
                problem, arrays = fixture(multicomponent)
                profile = CostProfile()
                original = ContractionState.initial(problem)
                for key in problem.sources:
                    original = original.load(key, profile, 1 << 28)
                original = original.with_indexed_rows()
                scoped = original
                joins = {scene: tuple(j for j in problem.baryon_joins if j.scene == scene)
                         for scene in range(2)}
                for scene in range(2):
                    # Ordinary pair preserves a mode still needed by the core;
                    # both endpoint joins then consume that row. Scene 1 shares
                    # the same intermediates but still has a distinct output.
                    for pair, join in (((0, 3), None), ((), joins[scene][0]), ((), joins[scene][1])):
                        expected = describe(original, scene, pair, join=join, profile=profile)
                        original = apply_description(original, *expected, join=join)
                        all_joins, all_bridges = problem.baryon_joins, problem.sparse_bridges
                        try:
                            object.__setattr__(problem, "baryon_joins", NoGlobalIteration(all_joins))
                            object.__setattr__(problem, "sparse_bridges", NoGlobalIteration(all_bridges))
                            actual = describe(scoped, scene, pair, join=join, profile=profile,
                                              scene_joins=joins, scoped_bridges=())
                            scoped = apply_description(scoped, *actual, join=join, scene_joins=joins)
                        finally:
                            object.__setattr__(problem, "baryon_joins", all_joins)
                            object.__setattr__(problem, "sparse_bridges", all_bridges)
                        self.assertEqual(actual, expected)
                        self.assertEqual(scoped, original)
                self.assertTrue(scoped.done)
                coefficients = ((1, "G", "E"),) if not multicomponent else (
                    (2 + .5j, "G", "E"), (-.75, "G2", "E2"))
                reference = sum(coefficient * opt_einsum.contract(
                    "ijk,abc,lmn,def,ilad,jmbe,kncf,a->",
                    *(arrays[key] for key in (spin, spatial, "H", "F", "P", "Q", "R", "X")))
                    for coefficient, spin, spatial in coefficients)
                np.testing.assert_allclose(replay(scoped.with_tuple_rows(), lambda key, source: arrays[key]),
                                           [reference, reference], rtol=1e-12, atol=1e-12)

    def test_static_scope_filters_completed_join_before_final_unary(self):
        problem, _ = fixture()
        join = problem.baryon_joins[0]
        joins = {0: (join,)}
        # A final transpose/reduction must not see a completed join as active.
        problem = replace(problem, terms=(ContractionTerm(("X",), "a->"),), baryon_joins=())
        state = ContractionState.initial(problem)
        state = replace(state, loaded=frozenset(state.catalog), completed_joins=frozenset(((0, join.name),)))
        self.assertEqual(describe(state, 0, (0,), profile=CostProfile()),
                         describe(state, 0, (0,), profile=CostProfile(), scene_joins=joins, scoped_bridges=()))

    def test_join_prediction_keeps_indexed_rows(self):
        problem, _ = fixture()
        count = _RowTable.block_size + 1
        joins = tuple(j for j in problem.baryon_joins if j.scene == 0)
        problem = replace(problem, terms=(problem.terms[0],) * count,
            baryon_joins=tuple(replace(j, scene=scene) for scene in range(count) for j in joins))
        state = ContractionState.initial(problem)
        state = replace(state, loaded=frozenset(state.catalog)).with_indexed_rows()
        self.assertIsInstance(state.pending, _RowTable)
        join = problem.baryon_joins[0]
        after = state._join_from_layout(join, state._join_layout(join), CostProfile(), 1 << 28,
                                       search_kernel_path=False)
        self.assertIsInstance(after.pending, _RowTable)
        self.assertIsInstance(after.results, _RowTable)
        self.assertIs(after.pending[1], state.pending[1])
        self.assertEqual(state.completed_joins, frozenset())


if __name__ == "__main__":
    unittest.main()
