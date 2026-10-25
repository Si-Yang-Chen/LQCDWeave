"""Independent checks of occurrence indexing, layout reuse and join batching."""
from dataclasses import replace
from itertools import combinations, product
import unittest
import random
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture
from correlator.planning.test_candidate_metadata import join_fixture
from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, Factor, PendingTerm, SourceSpec,
    SourcePreparation, SparseBridgeSpec, TensorSpec,
)
from lqcdweave.contraction.planning.logical_index import FactorIndex, LayoutIndex, reuse_indexes
from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contraction.planning.contraction_index import ContractionIndex, ContractionKey


def positions(factors, selected):
    return tuple(pair for pair in combinations(range(len(factors)), len(selected))
                 if tuple(factors[i] for i in pair) == selected)


class LogicalIndexTests(unittest.TestCase):
    def test_unrelated_valid_choice_survives_reversed_protected_sparse_pair(self):
        specs = {k: TensorSpec((1, 1, 1), "float64") for k in ("L", "R")}
        specs.update({k: TensorSpec((1,), "float64") for k in ("A", "B")})
        term = ContractionTerm(("R", "L", "A", "B"), "spc,spc,i,i->")
        sources = {k: SourceSpec(k, 0) for k in specs}
        sources["record"] = SourceSpec("record", 0, snapshot_bytes=0)
        problem = ContractionProblem(specs, (term,) * 8, sources,
            sparse_bridges=(SparseBridgeSpec("L", "R", "record", 1, 1),))
        state = replace(ContractionState.initial(problem), loaded=frozenset(sources))
        profile = fixture().logical_profile
        reference = state.contract(0, (2, 3), profile, 100000)
        index = LayoutIndex(state)
        actual = state._contract_from_layout(0, (2, 3), index.layout(0, (2, 3)),
            profile, 100000, layout_index=index)
        self.assertEqual(actual, reference)
        with self.assertRaisesRegex(ValueError, "sparse bridge first"):
            index.layout(0, (0, 1))

    def test_incremental_hash_matches_scan_with_diagonals_axis_orders_and_partial_shared_terms(self):
        rng = random.Random(20261004)
        specs = {key: TensorSpec((2, 2), "float64") for key in ("A", "B", "C")}
        terms = []
        for _ in range(18):
            keys = tuple(rng.choice(tuple(specs)) for _ in range(rng.randint(1, 4)))
            labels = tuple(rng.choice(("ij", "ji", "ii", "jk", "kk")) for _ in keys)
            available = list(dict.fromkeys("".join(labels)))
            rng.shuffle(available)
            output = "".join(available[:rng.randrange(len(available) + 1)])
            terms.append(ContractionTerm(keys, ",".join(labels) + "->" + output))
        problem = ContractionProblem(specs, tuple(terms), {k: SourceSpec(k, 0) for k in specs})
        state = replace(ContractionState.initial(problem), loaded=frozenset(specs))
        index = LayoutIndex(state)
        profile = fixture().logical_profile
        while not state.done:
            chosen = None
            for i, term in enumerate(state.pending):
                if term is None:
                    continue
                pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
                for pair in pairs:
                    try:
                        expected = state.contract(i, pair, profile, 1 << 100)
                    except ValueError as error:
                        with self.assertRaises(ValueError) as actual_error:
                            index.layout(i, pair)
                        self.assertEqual(str(actual_error.exception), str(error))
                        continue
                    chosen = i, pair, expected
                    break
                if chosen is not None:
                    break
            if chosen is None:
                # Random disconnected free-axis expressions can require an
                # outer product and therefore have no route in this domain.
                break
            i, pair, expected = chosen
            changes = []
            actual = state._contract_from_layout(i, pair, index.layout(i, pair), profile,
                1 << 100, layout_index=index, changes=changes)
            self.assertEqual(actual, expected)
            index.advance(actual, (j for j, _, _ in changes))
            state = actual

    def test_fixed_storage_audit_rebuilds_hash_index_and_matches_independent_scan(self):
        from lqcdweave.contraction.planning.storage import plan_scene_storage
        space = fixture((ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 8)
        state = ContractionState.initial(space.problem)
        for key in state.problem.sources:
            state = state.load(key, space.logical_profile, 1 << 100)
        while not state.done:
            i = next(i for i, term in enumerate(state.pending) if term is not None)
            pair = (0,) if len(state.pending[i].factors) == 1 else (0, 1)
            state = state.contract(i, pair, space.logical_profile, 1 << 100)
        args = (state, space.blocks, space.preparations)
        options = dict(profile=space.profile, memory_limit=10000)
        with patch("lqcdweave.contraction.planning.storage.reuse_indexes", return_value=False):
            reference = plan_scene_storage(*args, **options)
        with patch("lqcdweave.contraction.planning.state._matching_positions",
                   side_effect=AssertionError("audit must use hash lookup")):
            actual = plan_scene_storage(*args, **options)
        self.assertEqual(actual.events, reference.events)
        self.assertEqual(actual.memory, reference.memory)
        self.assertEqual(actual.seconds, reference.seconds)
        self.assertEqual(actual.feasible, reference.feasible)

    def test_hashed_computations_resolve_collisions_and_preserve_all_ordered_effects(self):
        terms = (ContractionTerm(("A", "A", "B"), "i,i,i->"),
                 ContractionTerm(("A", "A", "B"), "i,i,i->i"),
                 ContractionTerm(("B", "A", "A"), "i,i,i->"),
                 ContractionTerm(("A",), "i->"))
        space = fixture(terms)
        state = space._symbolic(space.initial_state())
        with patch.object(ContractionKey, "__hash__", lambda self: 0):
            index = LayoutIndex(state)
            for i, term in enumerate(state.pending):
                pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
                for pair in pairs:
                    expected = state.contract(i, pair, space.logical_profile, 10000)
                    actual = state._contract_from_layout(i, pair, index.layout(i, pair),
                        space.logical_profile, 10000, layout_index=index)
                    self.assertEqual(actual, expected)
            self.assertIsInstance(index.contractions, ContractionIndex)

    def test_hash_lookup_builds_each_operand_group_once_and_updates_only_changed_terms(self):
        terms = (ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 8 + (
            ContractionTerm(("A", "B", "C"), "i,i,i->i"),)
        space = fixture(terms)
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        original = FactorIndex.matching_terms
        built = set()
        def once(factors, selected):
            self.assertNotIn(selected, built)
            built.add(selected)
            return original(factors, selected)
        with patch.object(FactorIndex, "matching_terms", once):
            while not state.done:
                i = next(i for i, term in enumerate(state.pending) if term is not None)
                term = state.pending[i]
                pair = (0,) if len(term.factors) == 1 else (0, 1)
                expected = state.contract(i, pair, space.logical_profile, 10000)
                changes = []
                actual = state._contract_from_layout(i, pair, index.layout(i, pair),
                    space.logical_profile, 10000, layout_index=index, changes=changes)
                self.assertEqual(actual, expected)
                unchanged = set(range(len(state.pending))) - {j for j, _, _ in changes}
                contexts = {j: index.contexts[j] for j in unchanged if j in index.contexts}
                index.advance(actual, (j for j, _, _ in changes))
                for j, context in contexts.items():
                    self.assertIs(index.contexts[j], context)
                with self.assertRaisesRegex(ValueError, "different pending state"):
                    state._contract_from_layout(i, pair, state._contract_layout(i, pair),
                        space.logical_profile, 10000, layout_index=index)
                state = actual
            self.assertEqual(index.contractions.groups, {})
            self.assertEqual(index.contractions.term_keys, {})

    def test_incremental_hash_index_refreshes_completed_join_constraints(self):
        space = join_fixture()
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        index.contractions = ContractionIndex(index)
        for join in state.problem.baryon_joins:
            if (join.scene, join.name) in state.completed_joins:
                continue
            state = state.contract_join(join, space.logical_profile, 1 << 100)
            index.advance(state, (join.scene,))
            reference = LayoutIndex(state)
            rebuilt = ContractionIndex(reference)
            for i, term in enumerate(state.pending):
                if term is not None:
                    pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
                    for pair in pairs:
                        try:
                            expected = reference.layout(i, pair)
                        except ValueError as error:
                            with self.assertRaises(ValueError) as actual_error:
                                index.layout(i, pair)
                            self.assertEqual(str(actual_error.exception), str(error))
                            continue
                        layout = index.layout(i, pair)
                        self.assertEqual(layout, expected)
                        self.assertEqual(tuple(index.contractions.matching(layout)),
                                         tuple(rebuilt.matching(layout)))

    def test_hash_groups_are_lazy_and_interrupted_build_publishes_nothing(self):
        space = fixture((ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 8)
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        contractions = ContractionIndex(index)
        self.assertEqual(contractions.groups, {})
        first = index.layout(0, (0, 1))
        expected = tuple((i, (0, 1)) for i in range(8))
        self.assertEqual(tuple(contractions.matching(first)), expected)
        old_groups = {key: dict(rows) for key, rows in contractions.groups.items()}
        old_keys = {i: set(keys) for i, keys in contractions.term_keys.items()}
        old_built = set(contractions.built)
        calls = 0
        def interrupt():
            nonlocal calls
            calls += 1
            if calls == 5:
                raise TimeoutError("metadata budget")
        contractions.checkpoint = interrupt
        second = index.layout(0, (0, 2))
        with self.assertRaisesRegex(TimeoutError, "metadata budget"):
            tuple(contractions.matching(second))
        self.assertEqual(contractions.groups, old_groups)
        self.assertEqual(contractions.term_keys, old_keys)
        self.assertEqual(contractions.built, old_built)
        contractions.checkpoint = lambda: None
        self.assertEqual(tuple(contractions.matching(second)), tuple((i, (0, 2)) for i in range(8)))
        with patch.object(FactorIndex, "matching_terms", side_effect=AssertionError("rebuilt group")):
            self.assertEqual(tuple(contractions.matching(first)), expected)
            self.assertEqual(tuple(contractions.matching(second)), tuple((i, (0, 2)) for i in range(8)))

    def test_multiple_built_groups_survive_updates_and_new_operands_are_built_on_demand(self):
        terms = (ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 8 + (
            ContractionTerm(("A", "B", "C"), "i,i,i->i"),)
        space = fixture(terms)
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        for pair in ((0, 1), (0, 2), (1, 2)):
            tuple(index.substitutions(index.layout(0, pair)))
        changes = []
        result = state._contract_from_layout(0, (0, 1), index.layout(0, (0, 1)),
            space.logical_profile, 10000, layout_index=index, changes=changes)
        index.advance(result, (i for i, _, _ in changes))
        for i, term in enumerate(result.pending):
            if term is None:
                continue
            pair = (0,) if len(term.factors) == 1 else (0, 1)
            actual = result._contract_from_layout(i, pair, index.layout(i, pair),
                space.logical_profile, 10000, layout_index=index)
            self.assertEqual(actual, result.contract(i, pair, space.logical_profile, 10000))

    def test_index_matches_every_ordered_occurrence_with_labels_repeats_and_hash_collisions(self):
        alphabet = (Factor("A", "i"), Factor("A", "j"), Factor("B", "i"))
        pending = (None,) + tuple(PendingTerm(factors, "") for size in range(5)
                                  for factors in product(alphabet, repeat=size))
        with patch.object(Factor, "__hash__", lambda self: 7):
            index = FactorIndex(pending)
            for size in (1, 2):
                for selected in product(alphabet, repeat=size):
                    actual = tuple((i, tuple(matches)) for i, matches in index.matching_terms(selected))
                    expected = tuple((i, matches) for i, term in enumerate(pending) if term is not None
                                     if (matches := positions(term.factors, selected)))
                    self.assertEqual(actual, expected)

    def test_layout_cache_avoids_repeated_resolution_of_equal_subexpressions(self):
        space = fixture((ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 8)
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        original, calls = ContractionState._contract_layout, []
        def tracked(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        with patch.object(ContractionState, "_contract_layout", tracked):
            for term_index in range(8):
                for pair in combinations(range(3), 2):
                    index.layout(term_index, pair)
        self.assertEqual(len(calls), 3)
        self.assertEqual(index.hits, 21)

    def test_cached_layouts_equal_authoritative_layouts_through_complete_small_domains(self):
        specs = {key: TensorSpec((2, 2), "float64") for key in ("A", "B")}
        terms = (ContractionTerm(("A", "B"), "ij,ji->ij"),
                 ContractionTerm(("A", "B"), "ij,ji->ji"),
                 ContractionTerm(("A", "B"), "ii,ii->i"))
        problem = ContractionProblem(specs, terms, {k: SourceSpec(k, 0) for k in specs})
        matrix = MacroSearchSpace(problem, {}, {k: SourcePreparation(()) for k in specs}, memory_limit=100000)
        spaces = (matrix, fixture((ContractionTerm(("A", "A", "B"), "i,i,i->"),) * 2), join_fixture())
        for space in spaces:
            frontier, seen = [space.initial_state()], set()
            while frontier:
                cursor = frontier.pop()
                if space.key(cursor) in seen:
                    continue
                seen.add(space.key(cursor))
                state = space._symbolic(cursor)
                index = LayoutIndex(state)
                for i, term in enumerate(state.pending):
                    if term is None:
                        continue
                    pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
                    for pair in pairs:
                        try:
                            expected = state._contract_layout(i, tuple(pair), scene_joins=index.scene_joins)
                        except ValueError:
                            with self.assertRaises(ValueError):
                                index.layout(i, tuple(pair))
                        else:
                            self.assertEqual(index.layout(i, tuple(pair)), expected)
                frontier.extend(e.state for e in space.successors(cursor) if not e.state.state.done)

    def test_layout_cache_keeps_scoped_sparse_records_and_all_bounds(self):
        specs = {k: TensorSpec((1, 1, 1), "float64") for k in ("L", "R")}
        term = ContractionTerm(("L", "R"), "spc,spc->")
        bridges = tuple(SparseBridgeSpec("L", "R", "record", 1, i + 1, scene=i, part_count=i + 1)
                        for i in range(2))
        sources = {k: SourceSpec(k, 0) for k in specs}
        sources["record"] = SourceSpec("record", 0, snapshot_bytes=0)
        problem = ContractionProblem(specs, (term, term), sources, sparse_bridges=bridges)
        space = MacroSearchSpace(problem, {}, {k: SourcePreparation(()) for k in sources}, memory_limit=10000)
        index = LayoutIndex(space._symbolic(space.initial_state()))
        self.assertEqual(index.layout(0, (0, 1))[5], bridges[0])
        self.assertEqual(index.layout(1, (0, 1))[5], bridges[1])
        self.assertEqual(index.misses, 2)

    def test_layout_cache_is_bounded_and_eviction_does_not_change_layout(self):
        space = fixture()
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        index.capacity = 1
        first = index.layout(0, (0, 1))
        index.layout(0, (0, 2))
        self.assertEqual(len(index.layouts), 1)
        self.assertEqual(index.layout(0, (0, 1)), first)
        self.assertEqual(index.misses, 3)

    def test_small_node_bypasses_index_setup_with_identical_effects(self):
        space = fixture()
        cursor = space.initial_state()
        with patch.object(FactorIndex, "__init__", side_effect=AssertionError("unnecessary index")):
            actual = tuple(space.logical_steps(cursor))
        original = LayoutIndex.__init__
        def force_index(index, state, checkpoint=lambda: None, **kwargs):
            return original(index, state, checkpoint, reuse=True)
        with patch.object(LayoutIndex, "__init__", force_index):
            reference = tuple(space.logical_steps(cursor))
        self.assertEqual(actual, reference)

    def test_indexed_effects_match_unindexed_effects_and_reject_stale_index(self):
        space = fixture((ContractionTerm(("A", "A", "B"), "i,i,i->"),
                         ContractionTerm(("A", "A", "B"), "i,i,i->i")))
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        for pair in combinations(range(3), 2):
            layout = index.layout(0, pair)
            actual = state._contract_from_layout(0, pair, layout, space.logical_profile, 10000,
                                                  layout_index=index)
            self.assertEqual(actual, state.contract(0, pair, space.logical_profile, 10000))
        changed = state.contract(0, (0, 1), space.logical_profile, 10000)
        with self.assertRaisesRegex(ValueError, "different pending state"):
            changed._contract_from_layout(0, (0, 1), changed._contract_layout(0, (0, 1)),
                                          space.logical_profile, 10000, factor_index=index.factors)

    def test_shared_templates_preserve_outputs_and_share_immutable_updates(self):
        terms = (ContractionTerm(("A", "A", "B"), "i,i,i->"),) * 8
        space = fixture(terms + (ContractionTerm(("A", "A", "B"), "i,i,i->i"),))
        state = space._symbolic(space.initial_state())
        index = LayoutIndex(state)
        for pair in combinations(range(3), 2):
            actual = state._contract_from_layout(0, pair, index.layout(0, pair),
                space.logical_profile, 10000, layout_index=index)
            self.assertEqual(actual, state.contract(0, pair, space.logical_profile, 10000))
            self.assertIs(actual.pending[0], actual.pending[1])

    def test_index_policy_requires_both_work_and_reuse_without_changing_choices(self):
        short = fixture().initial_state().state.pending
        long = (PendingTerm(tuple(Factor(str(i), "i") for i in range(20)), ""),)
        self.assertFalse(reuse_indexes(short))
        self.assertFalse(reuse_indexes(long))
        self.assertTrue(reuse_indexes(short * 8))

    def test_shared_joins_build_only_representatives_and_match_full_sequential_updates(self):
        space = join_fixture()
        cursor = space.initial_state()
        state = space._symbolic(cursor)
        original, calls = ContractionState._join_from_layout, []
        def tracked(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        with patch.object(ContractionState, "_join_from_layout", tracked):
            steps = [s for s in space.logical_steps(cursor) if s.actions[0].kind == "baryon_join"]
        self.assertEqual(len(calls), 2)
        for step in steps:
            expected = state
            for action in step.actions:
                join = next(j for j in space.problem.baryon_joins
                            if j.scene == action.term_index and j.name == action.join_id)
                expected = expected.contract_join(join, space.logical_profile, 1 << 100)
            self.assertEqual(step.state, replace(expected, actions=()))
            self.assertEqual(step.actions, expected.actions)


if __name__ == "__main__":
    unittest.main()
