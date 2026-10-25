"""Registered special operations deliver the same full-axis actions as debug replay."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, CoreComponent, SourceSpec,
    SparseBridgeSpec, TensorCoreSpec, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers import integer_space
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from correlator.planning.test_candidate_metadata import join_fixture
from correlator.planning.test_integer_search import make_problem


def _route(problem):
    space = IntegerSpace(problem, profile=CostProfile())
    engine = FingerprintDFS(space, root_width=1, branch_width=1)
    engine.advance()
    return space, engine.selected_operations(), engine.best_cost


def _sparse_problem():
    specs = {name: TensorSpec((1, 1, 1), "float64") for name in ("L", "R")}
    term = ContractionTerm(("L", "R"), "spc,spc->")
    bridges = (SparseBridgeSpec("L", "R", "record0", 1, 0, same_point=True, scene=0),
               SparseBridgeSpec("L", "R", "record1", 1, 0, same_point=True, scene=1))
    sources = {name: SourceSpec(name, 0) for name in specs}
    sources.update({bridge.record: SourceSpec(bridge.record, 0, snapshot_bytes=0)
                    for bridge in bridges})
    return ContractionProblem(specs, (term, term), sources, sparse_bridges=bridges)


def _barrier_problem():
    problem = make_problem([(("A", "B", "C"), "ak,ak,a->")] * 3,
                           {"A": (2, 3), "B": (2, 3), "C": (2,)})
    terms = []
    for term in problem.terms:
        a, k = term.subscripts.inputs[0]
        terms.append(replace(term, sum_barriers=(((k,), (a,)),)))
    return replace(problem, terms=tuple(terms),
                   specs={name: TensorSpec(spec.shape, "complex64")
                          for name, spec in problem.specs.items()})


def _core_join_problem():
    specs = {name: TensorSpec((2, 2), "complex128") for name in ("P", "Q", "R")}
    specs.update({name: TensorSpec((2, 2, 2), "complex128")
                  for name in ("S1", "S2", "E1", "E2")})
    core = TensorCoreSpec(
        "baryon", ("spin0", "spin1", "spin2", "laplace_vector0",
                   "laplace_vector1", "laplace_vector2"),
        TensorSpec((2,) * 6, "complex128"),
        (CoreComponent(("S1", "E1")), CoreComponent(("S2", "E2"))))
    return ContractionProblem(
        specs, (ContractionTerm(("P", "Q", "R"), "ax,by,cz->"),),
        {name: SourceSpec(name, 0) for name in specs},
        baryon_joins=(BaryonJoinSpec(0, "whole", "S1", "E1", "abc", "xyz",
                                    core=core),))


def _renamed_join_problem():
    specs = {name: TensorSpec((2, 2), "float64") for name in ("P", "Q", "R")}
    specs.update(spin=TensorSpec((2, 2, 2), "float64"),
                 mode=TensorSpec((2, 2, 2), "float64"))
    terms = (ContractionTerm(("P", "Q", "R"), "ad,be,cf->"),
             ContractionTerm(("P", "Q", "R"), "ux,vy,wz->"))
    joins = (BaryonJoinSpec(0, "first", "spin", "mode", "abc", "def"),
             BaryonJoinSpec(1, "second", "spin", "mode", "uvw", "xyz"))
    return ContractionProblem(specs, terms,
                              {name: SourceSpec(name, 0) for name in specs},
                              baryon_joins=joins)


class SelectedSpecialDeliveryTests(unittest.TestCase):
    def test_join_sparse_and_sum_barrier_paths_match_independent_audit(self):
        for problem in (join_fixture().problem, _core_join_problem(),
                        _renamed_join_problem(),
                        _sparse_problem(), _barrier_problem()):
            with self.subTest(joins=len(problem.baryon_joins),
                              bridges=len(problem.sparse_bridges),
                              barriers=bool(problem.terms[0].sum_barriers)):
                space, operations, cost = _route(problem)
                self.assertTrue(operations)
                debug = IntegerSpace(problem, profile=CostProfile(),
                                     registry=space.registry, debug=True)
                reference = debug.audit(operations, cost)
                with (patch.object(space, "_selected_description",
                                   wraps=space._selected_description) as bind,
                      patch.object(integer_space, "describe",
                                   side_effect=AssertionError("selected audit repriced")),
                      patch.object(integer_space, "replay",
                                   side_effect=AssertionError("selected audit replayed"))):
                    delivered = space.audit(operations, cost)
                self.assertGreater(bind.call_count, 0)
                self.assertEqual(len(delivered.actions), len(reference.actions))
                for index, (actual, expected) in enumerate(zip(delivered.actions,
                                                                  reference.actions)):
                    for field in actual.__dataclass_fields__:
                        self.assertEqual(getattr(actual, field), getattr(expected, field),
                                         f"action {index} field {field}")
                self.assertEqual(delivered, reference)
                self.assertEqual(space.snippets, debug.snippets)
                self.assertTrue(delivered.done)

    def test_shared_join_keeps_each_scene_name_and_renamed_equation(self):
        problem = _renamed_join_problem()
        space, operations, cost = _route(problem)
        state = space.audit(operations, cost)
        joins = [action for action in state.actions if action.kind == "baryon_join"]
        self.assertEqual([action.join_id for action in joins], ["first", "second"])
        self.assertEqual(joins[0].tensor, joins[1].tensor)
        self.assertEqual([action.reused for action in joins], [False, True])
        self.assertEqual(joins[0].equation.output, joins[1].equation.output)
        self.assertNotEqual(joins[0].equation.inputs, joins[1].equation.inputs)

    def test_many_shared_join_occurrences_do_not_grow_state_action_history(self):
        problem = join_fixture().problem
        space, operations, cost = _route(problem)
        from lqcdweave.contraction.planning import state as state_module
        original = state_module.ContractionState.with_action_history
        lengths = []

        def record(state, actions):
            lengths.append(len(actions))
            return original(state, actions)

        with patch.object(state_module.ContractionState, "with_action_history", record):
            result = space.audit(operations, cost)
        self.assertTrue(result.done)
        self.assertEqual(lengths[-1], len(result.actions))
        self.assertTrue(all(length == 0 for length in lengths[:-1]))


if __name__ == "__main__":
    unittest.main()
