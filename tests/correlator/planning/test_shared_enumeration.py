"""Equivalence checks for shared exact/DFS logical enumeration."""
from dataclasses import replace
from itertools import combinations, product
from contextlib import suppress
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture
from correlator.planning.test_candidate_metadata import join_fixture
from lqcdweave.contracts.correlator.api import ContractionTerm, Factor
from lqcdweave.contraction.planning import state as state_module
from lqcdweave.contraction.planning.logical_index import LayoutIndex
from lqcdweave.contraction.planning.paths import _contract_successors, _state_key
from lqcdweave.contraction.planning.optimizers.session import solve_storage


def reference_positions(factors, selected):
    return (positions for positions in combinations(range(len(factors)), len(selected))
            if tuple(factors[i] for i in positions) == selected)


def reference_successors(state, profile, memory_limit, **kwargs):
    # Keep an independent route through the public, unindexed transitions.
    for index, term in enumerate(state.pending):
        if term is None:
            continue
        pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
        for pair in pairs:
            with suppress(MemoryError, ValueError):
                yield state.contract(index, tuple(pair), profile, memory_limit)
    for join in state.problem.baryon_joins:
        if (join.scene, join.name) not in state.completed_joins:
            with suppress(MemoryError, ValueError):
                yield state.contract_join(join, profile, memory_limit)


def effects(states):
    return {(_state_key(s), s.seconds, s.peak_bytes, s.peak_host_bytes, s.peak_device_bytes)
            for s in states}


class SharedEnumerationTests(unittest.TestCase):
    def test_position_lookup_matches_ordered_combinations_with_repeated_labels(self):
        alphabet = (Factor("A", "i"), Factor("A", "j"), Factor("B", "i"))
        for count in range(6):
            for factors in product(alphabet, repeat=count):
                for size in (1, 2):
                    for selected in product(alphabet, repeat=size):
                        self.assertEqual(tuple(state_module._matching_positions(factors, selected)),
                                         tuple(reference_positions(factors, selected)))

    def test_complete_step_transitions_match_unindexed_reference_through_all_small_states(self):
        spaces = [fixture((ContractionTerm(("A", "A", "B"), "i,i,i->"),) * 2),
                  fixture((ContractionTerm(("A", "B", "C"), "i,i,i->"),
                           ContractionTerm(("A", "B", "C"), "i,i,i->i"))), join_fixture()]
        for space in spaces:
            frontier, seen = [space.initial_state()], set()
            while frontier:
                cursor = frontier.pop()
                key = space.key(cursor)
                if key in seen:
                    continue
                seen.add(key)
                original_index = LayoutIndex.__init__
                def force_index(index, state, checkpoint=lambda: None, **kwargs):
                    return original_index(index, state, checkpoint, reuse=True)
                with patch.object(LayoutIndex, "__init__", force_index):
                    actual = space.successors(cursor)
                original = state_module.ContractionState._contract_from_layout
                def unindexed(state, *args, **kwargs):
                    kwargs.pop("factor_index", None)
                    return original(state, *args, **kwargs)
                with patch.object(state_module, "_matching_positions", reference_positions), \
                     patch.object(state_module.ContractionState, "_contract_from_layout", unindexed):
                    reference = space.successors(cursor)
                self.assertEqual(actual, reference)
                frontier.extend(e.state for e in actual if not e.state.state.done)

    def test_exact_dedup_preserves_all_primitive_effects_at_every_small_residency(self):
        space = fixture((ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 3)
        initial = space.initial_state().state
        for count in range(4):
            for resident in combinations("ABC", count):
                state = replace(initial, loaded=frozenset(resident))
                original = tuple(_contract_successors(state, space.logical_profile, 10000))
                unique = tuple(_contract_successors(state, space.logical_profile, 10000, unique_terms=True))
                self.assertEqual(effects(original), effects(unique))
                if count >= 2:
                    self.assertEqual(len(original), 3 * len(unique))

    def test_shared_join_output_does_not_merge_different_completion_states(self):
        space = join_fixture()
        state = replace(space.initial_state().state, loaded=frozenset(space.problem.sources))
        original = tuple(_contract_successors(state, space.logical_profile, 100000))
        unique = tuple(_contract_successors(state, space.logical_profile, 100000, unique_terms=True))
        self.assertEqual(effects(original), effects(unique))
        joins = [s for s in unique if s.actions[-1].kind == "baryon_join"]
        self.assertEqual(len(joins), 3)
        self.assertEqual(joins[0].actions[-1].tensor, joins[1].actions[-1].tensor)
        self.assertNotEqual(joins[0].completed_joins, joins[1].completed_joins)

    def test_exact_and_layered_keep_original_optimum_under_storage_pressure(self):
        term = ContractionTerm(("A", "B", "C"), "i,i,i->")
        macro = fixture((term, term), cap=264)
        for optimizer in ("exact", "dynamic_programming"):
            controls = dict(memory_limit=264, optimizer=optimizer, deadline_seconds=20)
            actual = solve_storage(macro.problem, macro.blocks, macro.preparations, **controls)
            with patch("lqcdweave.contraction.planning.optimizers.storage_space._contract_successors", reference_successors), \
                 patch.object(state_module, "_matching_positions", reference_positions):
                reference = solve_storage(macro.problem, macro.blocks, macro.preparations, **controls)
            self.assertEqual(actual.status, "OPTIMAL")
            self.assertEqual(reference.status, "OPTIMAL")
            self.assertEqual(actual.plan.seconds, reference.plan.seconds)
            self.assertEqual(actual.plan.peak_host_bytes, reference.plan.peak_host_bytes)


if __name__ == "__main__":
    unittest.main()
