"""Incremental top-k and boundary ties match a separate full-scan oracle."""
import unittest
from unittest.mock import patch

from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.integer_scoring import IntegerScorer
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from correlator.planning.test_integer_search import make_problem


LOCAL_SCORERS = ("incremental_time", "shared_savings", "time_per_term",
                 "element_reduction", "time_per_removed_element", "sharing_count",
                 ("sharing_count", "element_reduction"), ("element_reduction", "sharing_count"),
                 ("incremental_time", "shared_savings"))


def problem():
    return make_problem([
        (("A", "B", "C"), "ij,jk,kl->il"),
        (("A", "B", "D"), "ij,jk,kl->li"),
        (("A", "E", "D"), "ij,jk,kl->il"),
        (("A", "B"), "ij,jk->ki"),
        (("X", "Y", "Z"), "ij,jk,kl->li"),
        (("B", "B", "C"), "ij,jk,kl->il"),
        (("X", "Y"), "ij,jk->ik"),
    ], {key: (3, 3) for key in "ABCDEXYZ"})


def scan(space, name, width, include_ties=True):
    scorer = IntegerScorer(name)
    choices = list(space.search_candidates(lambda: None))
    ranked = []
    for ordinal, choice in enumerate(choices):
        operation, cost = choice
        score = scorer(space, operation, cost, lambda: None)
        # Keep the oracle independent of IntegerSpace.tie_break: derive the
        # same semantic overlap directly from the current occurrence index.
        occurrences = space.groups.get(operation.calculation, {})
        overlap = len(space.recent_scenes.intersection(occurrences))
        ranked.append((score, -overlap, ordinal, choice))
    ranked.sort(key=lambda item: item[:3])
    selected = ranked[:width]
    if include_ties and selected and len(ranked) > width:
        boundary_score = selected[-1][0]
        for item in ranked[width:]:
            if item[0] != boundary_score:
                break
            selected.append(item)
    return tuple(item[3] for item in selected)


class IntegerRankingTests(unittest.TestCase):
    def assert_rank(self, space, name, width, include_ties=True):
        before = space.fingerprint()
        expected = scan(space, name, width, include_ties)
        selected = space.rank_candidates(width, IntegerScorer(name), lambda: None,
                                         include_ties=include_ties)
        self.assertEqual(selected.choices, expected)
        self.assertEqual(space.fingerprint(), before)
        self.assertEqual(space.is_done(), all(term.done for term in space.terms))
        return selected

    def test_top_k_and_stable_ties_match_scan_through_shared_branch_undo(self):
        for name in LOCAL_SCORERS:
            with self.subTest(scorer=name):
                space = IntegerSpace(problem(), profile=CostProfile())

                def visit(depth):
                    selected = self.assert_rank(space, name, 3)
                    if depth == 0 or space.is_done():
                        return
                    for operation, _ in selected.choices[:2]:
                        before = space.fingerprint()
                        before_recent = space.recent_scenes
                        token = space.apply(operation, lambda: None)
                        self.assertEqual(space.recent_scenes,
                                         frozenset(scene for scene, _ in token.terms))
                        visit(depth - 1)
                        space.undo(token)
                        self.assertEqual(space.fingerprint(), before)
                        self.assertEqual(space.recent_scenes, before_recent)
                        self.assert_rank(space, name, 3)

                visit(4)

    def test_unchanged_scores_are_not_reevaluated(self):
        space = IntegerSpace(problem(), profile=CostProfile())
        for name in ("shared_savings", ("sharing_count", "element_reduction")):
            first = self.assert_rank(space, name, 3)
            second = self.assert_rank(space, name, 3)
            self.assertGreater(first.evaluations, 0)
            self.assertEqual(second.evaluations, 0)
            self.assertEqual(first.choices, second.choices)

    def test_boundary_ties_are_included_by_default_and_can_be_disabled(self):
        for include_ties in (True, False):
            with self.subTest(include_ties=include_ties):
                space = IntegerSpace(problem(), profile=CostProfile())
                with patch.object(IntegerScorer, "__call__", return_value=1.0):
                    expected = scan(space, "time_per_term", 1, include_ties)
                    selected = space.rank_candidates(
                        1, IntegerScorer("time_per_term"), lambda: None,
                        include_ties=include_ties,
                    )
                self.assertEqual(selected.choices, expected)
                if include_ties:
                    self.assertGreater(len(selected.choices), 1)
                else:
                    self.assertEqual(len(selected.choices), 1)

    def test_recent_overlap_orders_only_equal_primary_scores(self):
        space = IntegerSpace(problem(), profile=CostProfile())
        choices = list(space.search_candidates(lambda: None))
        # Seed a valid recent-scene set directly to isolate the ranking rule;
        # apply/undo's production of this field is exercised separately below.
        # Scene 2 belongs to groups 3 and 4 but not the original earlier groups.
        recent_scene = next(scene for operation, _ in choices
                            for scene in space.groups[operation.calculation]
                            if scene == 2)
        space.recent_scenes = frozenset((recent_scene,))
        overlaps = [len(space.recent_scenes.intersection(
            space.groups.get(operation.calculation, {}))) for operation, _ in choices]
        self.assertGreater(max(overlaps), min(overlaps))
        with patch.object(IntegerScorer, "__call__", return_value=1.0):
            expected = scan(space, "shared_savings", 1, include_ties=False)
            selected = space.rank_candidates(
                1, IntegerScorer("shared_savings"), lambda: None,
                include_ties=False,
            )
        self.assertEqual(selected.choices, expected)
        self.assertEqual(len(selected.choices), 1)
        self.assertEqual(space.tie_break(selected.choices[0][0], lambda: None), max(overlaps))
        selected_ordinal = next(i for i, (candidate, _) in enumerate(choices)
                                if candidate == selected.choices[0][0])
        self.assertEqual(overlaps[selected_ordinal], max(overlaps))

    def test_interrupted_refresh_preserves_state_and_can_resume(self):
        space = IntegerSpace(problem(), profile=CostProfile())
        operation, _ = next(space.search_candidates(lambda: None))
        token = space.apply(operation, lambda: None)
        recent_before = space.recent_scenes
        expected = scan(space, "time_per_term", 3)
        before, remaining = space.fingerprint(), 3

        def checkpoint():
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                raise InterruptedError("test candidate-ranking budget")

        with self.assertRaises(InterruptedError):
            space.rank_candidates(3, IntegerScorer("time_per_term"), checkpoint,
                                  include_ties=True)
        self.assertEqual(space.fingerprint(), before)
        self.assertEqual(space.recent_scenes, recent_before)
        selected = space.rank_candidates(3, IntegerScorer("time_per_term"), lambda: None,
                                         include_ties=True)
        self.assertEqual(selected.choices, expected)
        space.undo(token)

    def test_repeated_undo_compacts_stale_entries_without_changing_top_k(self):
        space = IntegerSpace(problem(), profile=CostProfile())
        for _ in range(90):
            choices = list(space.search_candidates(lambda: None))
            self.assert_rank(space, "time_per_term", 1)
            before_recent = space.recent_scenes
            token = space.apply(choices[-1][0], lambda: None)
            self.assert_rank(space, "time_per_term", 1)
            space.undo(token)
            self.assertEqual(space.recent_scenes, before_recent)
            self.assert_rank(space, "time_per_term", 1)
            self.assertLessEqual(len(space._ranking.heap), max(64, 2 * len(space._ranking.entries)))
        self.assertGreater(space.rank_heap_compactions, 0)

    def test_global_and_custom_scorers_keep_full_scan(self):
        space = IntegerSpace(problem(), profile=CostProfile())
        for name in ("storage_delta", "sharing_lookahead",
                     ("sharing_count", "storage_delta"), ("sharing_lookahead", "element_reduction")):
            self.assertIsNone(space.rank_candidates(3, IntegerScorer(name), lambda: None))

        class CustomScorer(IntegerScorer):
            def __call__(self, space, operation, cost, checkpoint):
                return -cost

        self.assertIsNone(space.rank_candidates(3, CustomScorer("shared_savings"), lambda: None))

    def test_reset_and_release_discard_workspace_ranking(self):
        space = IntegerSpace(problem(), profile=CostProfile())
        self.assert_rank(space, "element_reduction", 3)
        operation, _ = next(space.search_candidates(lambda: None))
        space.apply(operation, lambda: None)
        self.assertTrue(space.recent_scenes)
        space.reset()
        self.assertIsNone(space._ranking)
        self.assertEqual(space.recent_scenes, frozenset())
        self.assert_rank(space, "element_reduction", 3)
        operation, _ = next(space.search_candidates(lambda: None))
        space.apply(operation, lambda: None)
        space.release()
        self.assertIsNone(space._ranking)
        self.assertEqual(space.recent_scenes, frozenset())
        self.assertEqual(space.group_ordinals, {})


if __name__ == "__main__":
    unittest.main()
