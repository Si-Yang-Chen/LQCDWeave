"""Output layout alternatives retain sharing, algebra and old plan replay."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import AxisPattern, ContractionTerm
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contraction.planning.canonical_operations import describe
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.terminal_steps import is_terminal, completion_cost
from lqcdweave.contraction.planning.api import rebind_scene_path
from lqcdweave.contraction.execution.api import replay
from lqcdweave.storage.contraction_plans.api import save_storage_schedule, load_storage_schedule
from correlator.planning.test_integer_search import make_problem, solve


class TerminalStepTests(unittest.TestCase):
    def test_large_rebind_matches_independent_transitions_and_complete_history(self):
        from lqcdweave.contraction.planning.canonical_operations import replay as replay_operation
        problem=make_problem([(('A','B'),'ij,jk->ik')]*600,{'A':(2,3),'B':(3,4)})
        profile=CostProfile()
        def reference(problem):
            state=ContractionState.initial(problem)
            for key in problem.sources:
                state=state.load(key,profile,1<<30)
            for index in range(len(problem.terms)):
                action,*_=describe(state,index,(0,1),profile=profile)
                state=replay_operation(state,action,profile,1<<30)
            return state
        template=reference(problem)
        names={'A':'new:A','B':'new:B'}
        renamed=replace(problem,specs={names[k]:v for k,v in problem.specs.items()},
            sources={names[k]:replace(v,region=names[k]) for k,v in problem.sources.items()},
            terms=tuple(replace(t,inputs=tuple(names[k] for k in t.inputs)) for t in problem.terms))
        expected=reference(renamed)
        actual=rebind_scene_path(template,renamed,names,profile=profile,memory_limit=1<<30)
        self.assertEqual(actual,expected)
        self.assertIs(type(actual.pending),tuple)
        self.assertIs(type(actual.results),tuple)
        self.assertEqual(len(actual.actions),602)
        self.assertEqual(sum(a.reused for a in actual.actions),599)

    def setUp(self):
        self.problem = make_problem([(("A", "B"), "ij,jk->ki")],
                                    {"A": (2, 3), "B": (3, 5)})
        rng = np.random.default_rng(32)
        self.arrays = {key: rng.normal(size=spec.shape) + 1j * rng.normal(size=spec.shape)
                       for key, spec in self.problem.specs.items()}

    def test_direct_final_layout_avoids_paid_transpose(self):
        space = IntegerSpace(self.problem, profile=CostProfile())
        self.assertEqual(len(list(space.candidates(lambda: None))), 2)
        selected = list(space.search_candidates(lambda: None))
        self.assertEqual(len(selected), 1)
        operation, cost = selected[0]
        token = space.apply(operation, lambda: None)
        self.assertTrue(space.is_done())
        space.undo(token)
        self.assertFalse(space.is_done())
        self.assertEqual(list(space.search_candidates(lambda: None)), selected)
        state = space.audit((operation,), cost)
        self.assertEqual(len(state.actions), 1)
        self.assertEqual(state.catalog[state.results[0]].shape, (5, 2))
        result = solve(self.problem)
        paid = [a for a in result.state.actions if a.kind == "contract" and not a.reused]
        self.assertEqual(len(paid), 1)
        np.testing.assert_allclose(replay(result.state, lambda k, s: self.arrays[k])[0],
                                   (self.arrays["A"] @ self.arrays["B"]).T)

    def test_opposite_outputs_share_one_binary_recipe(self):
        problem = replace(self.problem, terms=(self.problem.terms[0],
            ContractionTerm(("A", "B"), "ab,bc->ac")))
        result = solve(problem)
        paid = [a for a in result.state.actions if a.kind == "contract" and not a.reused]
        self.assertEqual([len(a.operands) for a in paid], [2, 1])
        values = replay(result.state, lambda k, s: self.arrays[k])
        expected = self.arrays["A"] @ self.arrays["B"]
        np.testing.assert_allclose(values[0], expected.T)
        np.testing.assert_allclose(values[1], expected)

    def test_old_canonical_layout_can_still_be_audited(self):
        space = IntegerSpace(self.problem, profile=CostProfile())
        route, cost = [], 0.0
        while not space.is_done():
            # Canonical recipe is retained as the first raw domain candidate.
            operation, seconds = next(space.candidates(lambda: None))
            space.apply(operation, lambda: None)
            route.append(operation)
            cost += seconds
        state = space.audit(route, cost)
        self.assertEqual([len(a.operands) for a in state.actions], [2, 1])
        np.testing.assert_allclose(replay(state, lambda k, s: self.arrays[k], shared_arrays=dict(self.arrays))[0],
                                   (self.arrays["A"] @ self.arrays["B"]).T)

    def test_partial_shared_recipe_is_not_deferred(self):
        problem = make_problem([(("A", "B"), "ij,jk->ik"),
                                (("A", "B", "C"), "ij,jk,kl->il")],
                               {"A": (2, 3), "B": (3, 5), "C": (5, 2)})
        space = IntegerSpace(problem, profile=CostProfile())
        candidates = list(space.search_candidates(lambda: None))
        shared = next(op for op, _ in candidates if len(space.groups[op.calculation]) == 2)
        self.assertFalse(is_terminal(space, space.groups[shared.calculation]))
        space.apply(shared, lambda: None)
        self.assertTrue(space.terms[0].done)
        self.assertFalse(space.terms[1].done)

    def test_deferred_outputs_are_budgeted_without_constructing_operations(self):
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il"),
                                (("D", "E"), "ij,jk->ki")],
                               {"A": (2, 3), "B": (3, 5), "C": (5, 7),
                                "D": (2, 3), "E": (3, 5)})
        space = IntegerSpace(problem, profile=CostProfile())
        raw = list(space.candidates(lambda: None))
        expected = [(op, seconds) for op, seconds in raw
                    if op.calculation not in space.terminal_families]
        checks = []
        from lqcdweave.contraction.planning.optimizers.integer_space import IntegerOperation
        with patch("lqcdweave.contraction.planning.optimizers.integer_space.IntegerOperation",
                   wraps=IntegerOperation) as constructor:
            actual = list(space.search_candidates(lambda: checks.append(None)))
        self.assertEqual(actual, expected)
        self.assertEqual(constructor.call_count, len(expected))
        self.assertEqual(len(checks), len(raw))
        self.assertEqual(space.terminal_deferred, len(raw) - len(expected))

    def test_independent_outputs_have_one_schedule_and_budgeted_layouts(self):
        problem = make_problem([(("A", "B"), "ij,jk->ki"),
                                (("C", "D"), "ij,jk->ki")],
                               {"A": (2, 3), "B": (3, 5), "C": (2, 3), "D": (3, 5)})
        space = IntegerSpace(problem, profile=CostProfile())
        engine = FingerprintDFS(space, root_width=20, branch_width=20)
        result = engine.run()
        self.assertEqual(result.statistics["complete_plans"], 1)
        self.assertEqual(space.audited_steps, 2)
        self.assertGreater(space.terminal_deferred, 0)
        self.assertGreater(engine.budget.transitions, 2)

    def test_terminal_budget_only_visits_the_selected_independent_family(self):
        from lqcdweave.contraction.planning.optimizers.budget import SearchBudget
        from lqcdweave.contraction.planning.optimizers.protocols import SearchEvent
        problem = make_problem([((f"A{i}", f"B{i}"), "i,i->") for i in range(64)],
                               {f"{name}{i}": (3,) for i in range(64) for name in "AB"})
        space = IntegerSpace(problem, profile=CostProfile())
        raw = list(space.candidates(lambda: None))
        budget = SearchBudget(deadline_seconds=60, max_transitions=2)
        def checkpoint():
            budget.check()
            budget.consume(SearchEvent("transition"))
        selected = list(space.search_candidates(checkpoint))
        self.assertEqual(selected, [raw[0]])
        self.assertEqual(space.terminal_deferred, 63)
        self.assertEqual(budget.transitions, 2)

    def test_unchanged_layout_families_are_retained_during_apply_and_undo(self):
        problem = make_problem([(("A", "B"), "ij,jk->ki"),
                                (("C", "D"), "ij,jk->ki")],
                               {"A": (2, 3), "B": (3, 5), "C": (2, 3), "D": (3, 5)})
        space = IntegerSpace(problem, profile=CostProfile())
        before = list(space.search_candidates(lambda: None))
        self.assertEqual(space.layout_candidates, 2)
        untouched = {identity: group for identity, group in space.groups.items() if 1 in group}
        token = space.apply(before[0][0], lambda: None)
        list(space.search_candidates(lambda: None))
        self.assertEqual(space.layout_candidates, 2)
        for identity, group in untouched.items():
            self.assertIs(space.groups[identity], group)
        space.undo(token)
        self.assertEqual(list(space.search_candidates(lambda: None)), before)
        self.assertEqual(space.layout_candidates, 3)
        for identity, group in untouched.items():
            self.assertIs(space.groups[identity], group)

    def test_terminal_family_index_matches_full_scan_after_completion_and_undo(self):
        problem = make_problem([((f"A{i}", f"B{i}"), "ij,jk->ki") for i in range(80)],
                               {f"{name}{i}": shape for i in range(80)
                                for name, shape in (("A", (2, 3)), ("B", (3, 5)))})
        space = IntegerSpace(problem, profile=CostProfile())

        def compare():
            raw = list(space.candidates(lambda: None))
            family = min(space.terminal_families.values())
            choices = [item for item in raw
                       if space.terminal_families[item[0].calculation] == family]
            expected = min(choices, key=lambda item: (
                completion_cost(space, item[0], lambda: None), item[0].calculation))
            self.assertEqual(list(space.search_candidates(lambda: None)), [expected])
            return expected[0]

        tokens = []
        for _ in range(75):
            tokens.append(space.apply(compare(), lambda: None))
        compare()
        self.assertLessEqual(len(space.terminal_heap), 64)
        for token in reversed(tokens):
            space.undo(token)
            compare()
        space.reset()
        compare()
        space.release()
        self.assertEqual(space.terminal_members, {})
        self.assertEqual(space.terminal_heap, [])

    def test_nonterminal_proof_matches_full_scan_through_apply_and_undo(self):
        from lqcdweave.contraction.planning.optimizers.terminal_steps import group_is_terminal
        problem = make_problem([(("A", "B"), "ij,jk->ki"),
                                (("A", "B", "C"), "ij,jk,kl->li")],
                               {"A": (2, 3), "B": (3, 5), "C": (5, 7)})
        space = IntegerSpace(problem, profile=CostProfile())

        def verify():
            for identity, occurrences in space.groups.items():
                self.assertEqual(group_is_terminal(space, identity),
                                 is_terminal(space, occurrences))

        choices = list(space.candidates(lambda: None))
        verify()
        for operation, _ in choices:
            token = space.apply(operation, lambda: None)
            list(space.candidates(lambda: None))
            verify()
            space.undo(token)
            list(space.candidates(lambda: None))
            verify()

    def test_debug_reopens_the_complete_terminal_scan(self):
        from lqcdweave.contraction.planning.optimizers import terminal_steps
        space = IntegerSpace(self.problem, profile=CostProfile(), debug=True)
        list(space.candidates(lambda: None))
        terminal = next(iter(space.terminal_families))
        # A forged proof must be rejected by the debug re-scan, not trusted.
        terminal_steps.terminal_proofs(space)[1][terminal] = next(iter(space.groups[terminal]))
        with self.assertRaisesRegex(RuntimeError, "disagrees with the complete scan"):
            terminal_steps.register_layouts(space, {terminal}, lambda: None)

    def test_layout_family_index_matches_full_variant_scan(self):
        from lqcdweave.contraction.planning.optimizers.terminal_steps import layout_family_index
        problem = make_problem([((f"A{i}", f"B{i}"), "ij,jk->ki") for i in range(8)],
                               {f"{name}{i}": shape for i in range(8)
                                for name, shape in (("A", (2, 3)), ("B", (3, 5)))})
        space = IntegerSpace(problem, profile=CostProfile())
        list(space.candidates(lambda: None))
        self.assertTrue(space.layout_variants)
        affected = set(space.terminal_families)
        index = layout_family_index(space)
        expected = {identity for identity in space.layout_variants
                    if space.terminal_families[identity] in affected}
        actual = set()
        for family in affected:
            actual.update(index.get(family, ()))
        self.assertEqual(actual, expected)
    def test_family_is_reclassified_when_its_shared_consumers_change(self):
        problem = make_problem([(("A", "B"), "ij,jk->ki"),
                                (("A", "B", "C"), "ij,jk,kl->li")],
                               {"A": (2, 3), "B": (3, 5), "C": (5, 7)})
        space = IntegerSpace(problem, profile=CostProfile())
        choices = list(space.candidates(lambda: None))
        shared = next(op for op, _ in choices if len(space.groups[op.calculation]) == 2)
        self.assertNotIn(shared.calculation, space.terminal_families)
        right = next(op for op, _ in choices
                     if tuple((o.scene, o.positions) for o in space.groups[op.calculation].values()) == ((1, (1, 2)),))
        token = space.apply(right, lambda: None)
        list(space.candidates(lambda: None))
        self.assertEqual(space.terminal_families[shared.calculation], shared.calculation)
        self.assertEqual(set(space.groups[shared.calculation]), {0})
        space.undo(token)
        list(space.candidates(lambda: None))
        self.assertNotIn(shared.calculation, space.terminal_families)
        self.assertEqual(set(space.groups[shared.calculation]), {0, 1})

    def test_layout_must_preserve_retained_axes_including_aligned_axis(self):
        problem = make_problem([(("A", "B"), "ij,ij->i")], {"A": (2, 3), "B": (2, 3)})
        state = ContractionState.initial(problem)
        state = replace(state, loaded=frozenset(state.catalog))
        retained = state.pending[0].output
        describe(state, 0, (0, 1), profile=CostProfile(), output_order=retained)
        for output in ((), (999,), retained + retained):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "permute the retained"):
                describe(state, 0, (0, 1), profile=CostProfile(), output_order=output)

    def test_archive_and_rebind_preserve_explicit_layout_with_renamed_axes(self):
        result = solve(self.problem)
        shapes = {key: list(spec.shape) for key, spec in self.problem.specs.items()}
        hardware = {"backend": "numpy", "memory": 10**8}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            save_storage_schedule(path, (result.plan,), shapes=shapes, hardware=hardware)
            plans, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
                expected=(self.problem,), blocks=result.plan.blocks, preparations=result.plan.preparations)
            plan = plans[0]
            np.testing.assert_allclose(replay(plan.state, lambda k, s: self.arrays[k])[0],
                                       (self.arrays["A"] @ self.arrays["B"]).T)
        # Reverse the source-name ordering as well as renaming dummy axes.
        names = {"A": "new:Z", "B": "new:A"}
        original_names = {value: key for key, value in names.items()}
        other = replace(self.problem, specs={names[k]: v for k, v in self.problem.specs.items()},
            sources={names[k]: replace(v, region=names[k]) for k, v in self.problem.sources.items()},
            terms=(ContractionTerm(tuple(names[k] for k in ("A", "B")),
                AxisPattern(((10, 11), (11, 12)), (12, 10))),))
        rebound = rebind_scene_path(result.state, other, names, profile=CostProfile(), memory_limit=10**8)
        np.testing.assert_allclose(replay(rebound, lambda k, s: self.arrays[original_names[k]])[0],
                                   (self.arrays["A"] @ self.arrays["B"]).T)


if __name__ == "__main__":
    unittest.main()
