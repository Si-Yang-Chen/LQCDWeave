"""Check enumeration shortcuts against complete logical transitions."""
from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture
from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, SourcePreparation,
    SourceSpec, SparseBridgeSpec, TensorSpec, parse_axis_pattern,
)
from lqcdweave.contraction.planning.actions import _live_inputs
from lqcdweave.contraction.planning.state import ContractionState, _join_path_workspace
from lqcdweave.contraction.planning.optimizers.macro_space import (
    MacroSearchSpace, operation_identity,
)


def reference_candidates(space, cursor):
    """Original enumeration: compute every occurrence, then deduplicate effects."""
    state = replace(cursor.state, loaded=cursor.state.loaded | _live_inputs(cursor.state),
                    actions=(), peak_bytes=0, peak_host_bytes=0, peak_device_bytes=0)
    seen, actions = set(), []
    for index, term in enumerate(state.pending):
        if term is None:
            continue
        pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
        for pair in pairs:
            try:
                after = state.contract(index, tuple(pair), space.logical_profile, 1 << 100)
            except ValueError:
                continue
            action = after.actions[-1]
            key = (operation_identity(action), after.pending, after.results, after.completed_joins)
            if key not in seen:
                seen.add(key)
                actions.append(action)
    for join in space.problem.baryon_joins:
        if (join.scene, join.name) in state.completed_joins:
            continue
        try:
            after = state.contract_join(join, space.logical_profile, 1 << 100)
        except ValueError:
            continue
        action = after.actions[-1]
        key = operation_identity(action)
        if key not in seen:
            seen.add(key)
            actions.append(action)
    return tuple(actions)


def join_fixture():
    specs = {k: TensorSpec((2, 2), "float64") for k in ("a", "b", "c")}
    specs.update(spin=TensorSpec((2, 2, 2), "float64"), mode=TensorSpec((2, 2, 2), "float64"))
    term = ContractionTerm(("a", "b", "c"), "ad,be,cf->")
    joins = tuple(BaryonJoinSpec(i, "shared", "spin", "mode", labels, "def")
                  for i, labels in enumerate(("abc", "abc", "bac")))
    problem = ContractionProblem(specs, (term,) * 3, {k: SourceSpec(k, 0) for k in specs}, baryon_joins=joins)
    return MacroSearchSpace(problem, {}, {k: SourcePreparation(()) for k in specs}, memory_limit=100000)


class CandidateMetadataTests(unittest.TestCase):
    def test_join_layouts_are_resolved_once_per_node_for_enumeration_and_sharing(self):
        space = join_fixture()
        original, resolved = ContractionState._join_layout, []
        def tracked(state, join, **kwargs):
            resolved.append((join.scene, join.name))
            return original(state, join, **kwargs)
        with patch.object(ContractionState, "_join_layout", tracked):
            steps = tuple(space.logical_steps(space.initial_state()))
        self.assertEqual(resolved, [(0, "shared"), (2, "shared")])
        joins = [s for s in steps if s.actions[0].kind == "baryon_join"]
        self.assertEqual(len(joins), 2)
        self.assertEqual(joins[0].state.completed_joins, frozenset({(0, "shared"), (1, "shared")}))
        self.assertEqual(joins[1].state.completed_joins, frozenset({(2, "shared")}))
        self.assertEqual([a.reused for a in joins[0].actions], [False, True])

    def test_join_sharing_index_compares_algebra_despite_output_fingerprint_collision(self):
        space = join_fixture()
        original = ContractionState._join_layout
        def collide(state, join, **kwargs):
            layout = original(state, join, **kwargs)
            return layout[:-1] + ("node:collision",)
        with patch.object(ContractionState, "_join_layout", collide):
            steps = [s for s in space.logical_steps(space.initial_state())
                     if s.actions[0].kind == "baryon_join"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].state.completed_joins, frozenset({(0, "shared"), (1, "shared")}))
        self.assertEqual(steps[1].state.completed_joins, frozenset({(2, "shared")}))

    def test_join_layout_index_is_rebuilt_after_another_endpoint_changes_legs(self):
        specs = {k: TensorSpec((2, 2, 2, 2), "float64") for k in ("a", "b", "c")}
        specs.update(spin=TensorSpec((2, 2, 2), "float64"), mode=TensorSpec((2, 2, 2), "float64"))
        term = ContractionTerm(("a", "b", "c"), "adAD,beBE,cfCF->")
        joins = tuple(BaryonJoinSpec(scene, name, "spin", "mode", spin, mode)
                      for scene in range(3)
                      for name, spin, mode in (("left", "abc", "def"), ("right", "ABC", "DEF")))
        problem = ContractionProblem(specs, (term,) * 3, {k: SourceSpec(k, 0) for k in specs}, baryon_joins=joins)
        space = MacroSearchSpace(problem, {}, {k: SourcePreparation(()) for k in specs}, memory_limit=100000)
        cursor, actions = space.initial_state(), ()
        for name in ("left", "right"):
            before = space._symbolic(cursor)
            expected = before
            for join in joins:
                if join.name == name:
                    expected = expected.contract_join(join, space.logical_profile, 1 << 100)
            step = next(s for s in space.logical_steps(cursor) if s.actions[0].join_id == name)
            self.assertEqual(step.state.pending, expected.pending)
            self.assertEqual(step.state.results, expected.results)
            self.assertEqual(step.state.completed_joins, expected.completed_joins)
            self.assertEqual(step.actions, expected.actions)
            edge = space.evaluate(cursor, step)
            cursor, actions = edge.state, actions + edge.actions
        self.assertTrue(cursor.state.done)
        self.assertTrue(space.plan(cursor, actions).feasible)

    def test_shortcut_matches_original_enumeration_through_shared_and_repeated_terms(self):
        for terms in (
            (ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 3,
            (ContractionTerm(("A", "A", "B"), "i,i,i->"),) * 2,
            (ContractionTerm(("A", "B", "C"), "i,i,i->"),
             ContractionTerm(("A", "B", "C"), "i,i,i->i")),
        ):
            space = fixture(terms)
            frontier = [space.initial_state()]
            seen = set()
            while frontier:
                cursor = frontier.pop()
                key = space.key(cursor)
                if key in seen:
                    continue
                seen.add(key)
                self.assertEqual(tuple(space.candidates(cursor)), reference_candidates(space, cursor))
                if not cursor.state.done:
                    frontier.extend(edge.state for edge in space.successors(cursor))

    def test_join_identity_filter_preserves_full_shared_preview_and_axis_distinctions(self):
        space = join_fixture()
        cursor = space.initial_state()
        self.assertEqual(tuple(space.candidates(cursor)), reference_candidates(space, cursor))
        for action in space.candidates(cursor):
            if action.kind != "baryon_join":
                continue
            actual = space.preview(cursor, action)
            # Force every join through the old expensive check. The final
            # computed identity still determines whether to substitute it.
            with patch.object(ContractionState, "join_output_key", return_value=action.tensor):
                reference = space.preview(cursor, action)
            self.assertEqual(actual, reference)
            self.assertEqual(tuple(space.candidates(actual.state)), reference_candidates(space, actual.state))
            self.assertEqual(cursor.state.loaded, frozenset())

    def test_join_identity_requires_neither_residency_nor_costing(self):
        space = join_fixture()
        state = space.initial_state().state
        with patch.object(type(space.logical_profile), "contract_seconds", side_effect=AssertionError("costed")):
            keys = [state.join_output_key(j) for j in space.problem.baryon_joins]
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])
        with self.assertRaises(ValueError):
            state.contract_join(space.problem.baryon_joins[0], space.logical_profile, 100000)

    def test_scene_specific_sparse_contexts_are_not_merged(self):
        specs = {k: TensorSpec((1, 1, 1), "float64") for k in ("L", "R")}
        term = ContractionTerm(("L", "R"), "spc,spc->")
        bridges = (SparseBridgeSpec("L", "R", "record0", 1, 0, same_point=True, scene=0),
                   SparseBridgeSpec("L", "R", "record1", 1, 0, same_point=True, scene=1))
        sources = {k: SourceSpec(k, 0) for k in specs}
        sources.update({b.record: SourceSpec(b.record, 0, snapshot_bytes=0) for b in bridges})
        problem = ContractionProblem(specs, (term, term), sources, sparse_bridges=bridges)
        space = MacroSearchSpace(problem, {}, {k: SourcePreparation(()) for k in sources}, memory_limit=10000)
        cursor = space.initial_state()
        actual = tuple(space.candidates(cursor))
        self.assertEqual(actual, reference_candidates(space, cursor))
        self.assertEqual(len(actual), 2)
        for action in actual:
            edge = space.preview(cursor, action)
            self.assertEqual(sum(result is not None for result in edge.state.state.results), 1)

    def test_workspace_cache_uses_shapes_and_path_and_excludes_dtype(self):
        equation = parse_axis_pattern("ab,bc,cd->ad")
        shapes = ((2, 3), (3, 4), (4, 5))
        path = ((0, 1), (0, 1))
        for actual_shapes, actual_path in (
            (shapes, path), (((7, 3), (3, 4), (4, 5)), path),
            (shapes, ((1, 2), (0, 1))),
        ):
            self.assertEqual(_join_path_workspace(equation, actual_shapes, actual_path),
                             _join_path_workspace.__wrapped__(equation, actual_shapes, actual_path))
        self.assertNotEqual(_join_path_workspace(equation, shapes, path),
                            _join_path_workspace(equation, ((7, 3), (3, 4), (4, 5)), path))


if __name__ == "__main__":
    unittest.main()
