"""Check inherited liveness against a closure of declared producer inputs."""
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture
from correlator.planning.test_candidate_metadata import join_fixture
from lqcdweave.contracts.correlator.api import (
    Action, ContractionProblem, ContractionTerm, Factor, PendingTerm,
    SourcePreparation, SourceSpec, StorageBlock, TensorSpec,
)


def closure(cursor):
    """Independent least fixed point; no recursive memo or library liveness."""
    state = cursor.state
    factor_inputs = {f.tensor for term in state.pending if term for f in term.factors}
    required = factor_inputs | {key for key in state.results if key is not None}
    for join in state.problem.baryon_joins:
        if state.pending[join.scene] is not None and (join.scene, join.name) not in state.completed_joins:
            required.update((join.spin, join.elemental))
    for bridge in state.problem.sparse_bridges:
        terms = state.pending if bridge.scene is None else (state.pending[bridge.scene],)
        if any(term and {bridge.left, bridge.right} <= {f.tensor for f in term.factors} for term in terms):
            required.add(bridge.record)
    implicit = required - factor_inputs
    producers = dict(cursor.producers)
    useful = set(required)
    for term in state.pending:
        if term is None:
            continue
        available = {f.tensor for f in term.factors} | implicit
        while True:
            added = {key for key, inputs in producers.items() if set(inputs) <= available} - available
            if not added:
                break
            available.update(added)
        useful.update(state.loaded & available & producers.keys())
    return frozenset(useful)


class LivenessInheritanceTests(unittest.TestCase):
    def test_dead_dense_release_and_block_close_reuse_liveness_without_a_scan(self):
        space = fixture((ContractionTerm(("A",), "i->"),))
        cursor = space.initial_state()
        for key in ("B", "C"):
            cursor = space.apply(cursor, Action("read_block", key))
            cursor = space.apply(cursor, cursor.state.load(key, space.logical_profile, 10000).actions[-1])
        expected = space.useful(cursor)
        self.assertEqual(expected, closure(cursor))
        with patch("lqcdweave.contraction.planning.optimizers.macro_space._useful_tensors",
                   side_effect=AssertionError("recomputed unchanged dependencies")):
            for key in ("B", "C"):
                cursor = space.apply(cursor, Action("release", key))
                self.assertIs(space.useful(cursor), expected)
                cursor = space.apply(cursor, Action("close_block", key))
                self.assertIs(space.useful(cursor), expected)
        self.assertEqual(space.useful(cursor), closure(cursor))

    def test_borrowed_view_release_does_not_free_its_raw_owner_or_change_liveness(self):
        from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
        specs = {"A": TensorSpec((4,), "float64")}
        problem = ContractionProblem(specs, (ContractionTerm(("A",), "i->"),),
            {"A": SourceSpec("A", 0), "record": SourceSpec("record", 0, snapshot_bytes=0)})
        blocks = {"raw": StorageBlock("dataset", "raw", 24, host_bytes=24)}
        space = MacroSearchSpace(problem, blocks,
            {"A": SourcePreparation(()), "record": SourcePreparation(("raw",))}, memory_limit=10000)
        cursor = space.apply(space.initial_state(), Action("read_block", "raw"))
        cursor = space.apply(cursor, cursor.state.load("record", space.logical_profile, 10000).actions[-1])
        useful = space.useful(cursor)
        after = space.apply(cursor, Action("release", "record"))
        self.assertIs(space.useful(after), useful)
        self.assertEqual(after.resources.allocations, cursor.resources.allocations)
        self.assertEqual(after.resources.borrowed, ())
        self.assertEqual(space.useful(after), closure(after))

    def test_random_producer_graphs_and_dead_subsets_match_independent_closure(self):
        rng = random.Random(20261004)
        space = fixture()
        initial = space.initial_state()
        for _ in range(100):
            keys = list("ABC")
            producers = []
            for i in range(8):
                key = "p" + str(i)
                producers.append((key, tuple(rng.sample(keys, rng.randint(1, min(3, len(keys)))))))
                keys.append(key)
            pending = tuple(PendingTerm(tuple(Factor(key, "i") for key in rng.sample(keys, rng.randint(1, 4))), "")
                            for _ in range(3))
            loaded = frozenset(rng.sample(keys, rng.randint(1, len(keys))))
            state = replace(initial.state, pending=pending, results=(None,) * 3, loaded=loaded)
            before = replace(initial, state=state, producers=tuple(producers))
            useful = space.useful(before)
            self.assertEqual(useful, closure(before))
            for dead in (frozenset(), loaded - useful):
                after = replace(before, state=replace(state, loaded=loaded - dead))
                before._inherit_useful(after)
                self.assertIs(after.useful_tensors, useful)
                self.assertEqual(after.useful_tensors, closure(after))

    def test_expression_results_producers_and_loaded_additions_invalidate_cache(self):
        space = fixture()
        initial = space.initial_state()
        producers = (("cached", ("A", "B")),)
        state = replace(initial.state, loaded=frozenset({"cached"}))
        before = replace(initial, state=state, producers=producers)
        useful = space.useful(before)
        self.assertIn("cached", useful)
        variants = (
            replace(before, state=replace(state, loaded=frozenset())),
            replace(before, state=replace(state, loaded=state.loaded | {"new"})),
            replace(before, state=replace(state, pending=(PendingTerm((Factor("C", "i"),), ""),))),
            replace(before, state=replace(state, results=("output",))),
            replace(before, state=replace(state, completed_joins=frozenset({(0, "changed")}))),
            replace(before, state=replace(state, problem=replace(state.problem))),
            replace(before, producers=(("cached", ("missing",)),)),
        )
        for after in variants:
            before._inherit_useful(after)
            self.assertNotIn("useful_tensors", after.__dict__)
            self.assertEqual(space.useful(after), closure(after))
        self.assertIs(space.useful(before), useful)

    def test_join_completion_invalidates_implicit_inputs_and_spill_restore_preserves_them(self):
        space = join_fixture()
        before = space.initial_state()
        useful = space.useful(before)
        joins = frozenset((join.scene, join.name) for join in space.problem.baryon_joins)
        after = replace(before, state=replace(before.state, completed_joins=joins))
        before._inherit_useful(after)
        self.assertNotIn("useful_tensors", after.__dict__)
        self.assertNotEqual(space.useful(after), useful)
        self.assertEqual(space.useful(after), closure(after))
        space = fixture(backend="cupy", host=10000)
        first = space.successors(space.initial_state())[0].state
        useful = space.useful(first)
        key = first.state.pending[0].factors[-1].tensor
        for kind in ("spill_host", "spill_disk", "restore_host", "restore_device"):
            first = space.apply(first, space._move(first, kind, "tensor", key))
            self.assertIs(space.useful(first), useful)
            self.assertEqual(space.useful(first), closure(first))

    def test_pressure_reclamation_keeps_identical_audited_steps_and_cache_cleanup(self):
        space = fixture(cap=264)
        first = space.successors(space.initial_state())[0]
        expected = space.successors(first.state)
        from lqcdweave.contraction.planning.optimizers.macro_space import MacroCursor
        with patch.object(MacroCursor, "_inherit_useful", lambda _, after: after):
            reference = space.successors(first.state)
        self.assertEqual(expected, reference)
        for after in expected:
            plan = space.plan(after.state, first.actions + after.actions)
            self.assertTrue(plan.feasible)
            self.assertAlmostEqual(plan.seconds, space.cost(after.state))
        self.assertNotIn("useful_tensors", first.state.__dict__)


if __name__ == "__main__":
    unittest.main()
