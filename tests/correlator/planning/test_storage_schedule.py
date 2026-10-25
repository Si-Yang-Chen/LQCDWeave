from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    SparseBridgeSpec,
    StorageBlock,
    TensorSpec,
)
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.storage import logical_source_problem
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
from lqcdweave.contraction.planning.optimizers.storage_schedule import (
    _Dag,
    _Frontier,
    replay_fixed_dag,
    schedule_fixed_dag,
)
from lqcdweave.contraction.planning.optimizers.graph_session import solve_graph_storage
from lqcdweave.contraction.planning.optimizers.session import solve_storage


def _problem(*, shared=False):
    specs = {
        "A": TensorSpec((20, 30), "float64"),
        "B": TensorSpec((30, 40), "float64"),
        "C": TensorSpec((40,), "float64"),
        "D": TensorSpec((20, 30), "float64"),
        "E": TensorSpec((30, 40), "float64"),
        "F": TensorSpec((40,), "float64"),
    }
    terms = (
        ContractionTerm(("A", "B", "C"), "ik,kj,j->i"),
        ContractionTerm(("D", "E", "F"), "ik,kj,j->i"),
    )
    if shared:
        specs = {key: specs[key] for key in ("A", "B", "C", "F")}
        terms = (
            ContractionTerm(("A", "B", "C"), "ik,kj,j->i"),
            ContractionTerm(("A", "B", "F"), "ik,kj,j->i"),
        )
    sources = {key: SourceSpec(key, 0) for key in specs}
    problem = logical_source_problem(ContractionProblem(specs, terms, sources))
    arrays = {
        key: np.arange(np.prod(spec.shape), dtype=np.float64).reshape(spec.shape) / 13 + index
        for index, (key, spec) in enumerate(specs.items(), start=1)
    }
    blocks = {
        key: StorageBlock("dataset-" + key, "full", int(array.nbytes), host_bytes=int(array.nbytes))
        for key, array in arrays.items()
    }
    preparations = {key: SourcePreparation((key,)) for key in specs}
    return problem, arrays, blocks, preparations


def _fixed_path(problem, profile, steps):
    space = IntegerSpace(problem, profile=profile)
    operations = []
    for scene, required_pair in steps:
        candidates = tuple(space.candidates(lambda: None))
        selected = None
        for operation, _ in candidates:
            occurrence = space.groups.get(operation.calculation, {}).get(scene)
            if occurrence is None:
                continue
            current = space.terms[scene]
            names = tuple(space.registry.data[current.factors[position]].name
                          for position in occurrence.positions)
            if required_pair is None or frozenset(names) == frozenset(required_pair):
                selected = operation
                break
        if selected is None:
            raise AssertionError(f"no selected operation for scene {scene}, pair={required_pair!r}")
        operations.append(selected)
        space.apply(selected, lambda: None)
    state = space.audit(operations, sum(operation.seconds for operation in operations))
    return state, tuple(space.snippets), tuple(
        0 if operation.reused else space.registry.calculations[operation.calculation].workspace
        for operation in operations
    )


def _physical(problem, blocks, preparations, *, cap=1_000_000, profile=None):
    return MacroSearchSpace(problem, blocks, preparations, memory_limit=cap, profile=profile)


def _computation_signature(actions):
    return Counter(
        (action.kind, action.tensor, action.operands, action.equation, action.term_index,
         action.reused, action.seconds)
        for action in actions
        if action.kind not in ("load", "release", "read_block", "close_block")
    )


def _ordered_computation_signature(actions):
    return tuple(
        (action.kind, action.tensor, action.operands, action.equation,
         action.term_index, action.reused, action.seconds)
        for action in actions
        if action.kind not in ("load", "release", "read_block", "close_block")
    )


def _per_term_signature(actions, term_index):
    return tuple((action.kind, action.tensor, action.operands, action.reused)
                 for action in actions if action.term_index == term_index
                 and action.kind not in ("load", "release", "read_block", "close_block"))


class _ArrayLoader:
    def __init__(self, arrays):
        self.arrays = arrays

    def read_block(self, key, block):
        return self.arrays[key]

    def materialize(self, key, source, blocks):
        return blocks[source.region]


class StorageScheduleTests(unittest.TestCase):
    def test_foreign_fragments_and_debug_costs_remain_checked(self):
        problem, _, blocks, preparations = _problem(shared=True)
        profile = CostProfile()
        state, snippets, workspaces = _fixed_path(problem, profile,
            ((0, ('A', 'B')), (0, None), (1, None)))
        physical = _physical(problem, blocks, preparations, profile=profile)
        # A forged fragment must still be rejected by the untrusted entry and
        # diagnostics, even though internal emitters guarantee this grouping.
        begin, end = snippets[0]
        self.assertGreater(end-begin, 1)
        actions = list(state.actions)
        actions[begin+1] = replace(actions[begin+1], tensor='foreign-result')
        forged = replace(state, actions=tuple(actions))
        with self.assertRaisesRegex(ValueError, 'one selected shared calculation'):
            _Dag(forged, snippets, workspaces, physical)
        with self.assertRaisesRegex(ValueError, 'one selected shared calculation'):
            schedule_fixed_dag(forged, snippets, workspaces, physical, debug=True)
        actions = list(state.actions)
        actions[begin] = replace(actions[begin], seconds=float('nan'))
        with self.assertRaises(ValueError):
            schedule_fixed_dag(replace(state, actions=tuple(actions)), snippets,
                               workspaces, physical, debug=True)

    def test_audited_dag_accounts_resources_without_replaying_logical_states(self):
        problem, arrays, blocks, preparations = _problem(shared=True)
        profile = CostProfile(preparation_call_seconds=0.003)
        state, snippets, workspaces = _fixed_path(problem, profile,
            ((0, ('A', 'B')), (0, None), (1, None)))
        physical = _physical(problem, blocks, preparations, profile=profile)
        with patch('lqcdweave.contraction.planning.optimizers.storage_schedule.audit_resource_trajectory',
                   side_effect=AssertionError('repeated logical replay')):
            plan, _ = schedule_fixed_dag(state, snippets, workspaces, physical)
        reference, _ = schedule_fixed_dag(state, snippets, workspaces, physical, debug=True)
        self.assertEqual(plan.memory, reference.memory)
        self.assertEqual(plan.events, reference.events)
        self.assertEqual(plan.preparation_costs, reference.preparation_costs)
        self.assertEqual(plan.peak_host_bytes, reference.peak_host_bytes)
        self.assertEqual(plan.seconds, reference.seconds)
        expected = tuple(np.einsum('ik,kj,j->i', arrays['A'], arrays['B'], arrays[key])
                         for key in ('C', 'F'))
        for actual, value in zip(replay_scene_storage(plan, _ArrayLoader(arrays)), expected):
            np.testing.assert_allclose(actual, value)

    def test_resource_trace_matches_independent_replay_on_both_memory_domains(self):
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                problem, _, blocks, preparations = _problem()
                problem = replace(problem, backend=backend)
                if backend == 'cupy':
                    blocks = {key: replace(block, host_bytes=0, device_bytes=block.host_bytes,
                                           staging_host_bytes=16) for key, block in blocks.items()}
                else:
                    blocks = {key: replace(block, staging_host_bytes=16) for key, block in blocks.items()}
                preparations = {key: replace(prep, workspace_host_bytes=24,
                    workspace_device_bytes=32 if backend == 'cupy' else 0)
                    for key, prep in preparations.items()}
                profile = CostProfile()
                state, spans, workspaces = _fixed_path(problem, profile,
                    ((0, ('A', 'B')), (1, ('D', 'E')), (0, None), (1, None)))
                physical = _physical(problem, blocks, preparations, profile=profile)
                # debug compares every resource point/event with the general
                # independently replayed trajectory, not just its final peak.
                plan, _ = schedule_fixed_dag(state, spans, workspaces, physical, debug=True)
                self.assertTrue(plan.feasible)
                # The normal builder writes final indices once; debug retains
                # the old raw trajectory, independent replay and conversion.
                with patch('lqcdweave.contraction.planning.optimizers.storage_schedule.executable_delivery',
                           side_effect=AssertionError('redundant whole-plan conversion')):
                    direct, _ = schedule_fixed_dag(state, spans, workspaces, physical,
                                                  max_states=1, deadline_seconds=0)
                reference, _ = schedule_fixed_dag(state, spans, workspaces, physical,
                                                 max_states=1, deadline_seconds=0, debug=True)
                self.assertEqual(direct.state.actions, reference.state.actions)
                self.assertEqual(direct.memory, reference.memory)
                self.assertEqual(any(p.allocation_rounding_device_bytes for p in reference.memory),
                                 backend == 'cupy')
                self.assertEqual(direct.events, reference.events)
                self.assertEqual(direct.preparation_costs, reference.preparation_costs)
                self.assertEqual(direct.seconds, reference.seconds)
                self.assertEqual(direct.feasible, reference.feasible)

    def test_independent_storage_replay_is_debug_only(self):
        problem, arrays, blocks, preparations = _problem(shared=True)
        with patch.object(MacroSearchSpace, 'materialize_actions',
                          side_effect=AssertionError('extra state replay on normal path')):
            normal = solve_storage(problem, blocks, preparations, memory_limit=1_000_000,
                root_width=1, branch_width=1, max_states=128, deadline_seconds=10)
        original = MacroSearchSpace.materialize_actions
        calls = []
        def track(space, *args, **kwargs):
            calls.append(1)
            return original(space, *args, **kwargs)
        with patch.object(MacroSearchSpace, 'materialize_actions', track):
            diagnostic = solve_storage(problem, blocks, preparations, memory_limit=1_000_000,
                root_width=1, branch_width=1, max_states=128, deadline_seconds=10, debug=True)
        self.assertTrue(calls)
        expected = tuple(np.einsum('ik,kj,j->i', arrays['A'], arrays['B'], arrays[key])
                         for key in ('C', 'F'))
        for result in (normal, diagnostic):
            self.assertTrue(result.plan.feasible)
            self.assertGreaterEqual(result.statistics['storage_schedule']['audited_schedules'], 1)
            np.testing.assert_allclose(replay_scene_storage(result.plan, _ArrayLoader(arrays)), expected)

    def test_resident_target_stays_first_when_its_updated_priority_is_highest(self):
        shapes = {"A": (200, 30), "B": (30, 4), "C": (4, 1),
                  "D": (30, 1), "E": (200, 300), "F": (300, 1)}
        problem = logical_source_problem(ContractionProblem(
            {key: TensorSpec(shape, "float64") for key, shape in shapes.items()},
            (ContractionTerm(("A", "B", "C"), "ij,jk,kl->il"),
             ContractionTerm(("A", "D"), "ij,jl->il"),
             ContractionTerm(("E", "F"), "ij,jl->il")),
            {key: SourceSpec(key, 0) for key in shapes}))
        arrays = {key: np.arange(np.prod(shape), dtype=float).reshape(shape) / 17
                  for key, shape in shapes.items()}
        blocks = {key: StorageBlock(key, "full", a.nbytes, host_bytes=a.nbytes)
                  for key, a in arrays.items()}
        preparations = {key: SourcePreparation((key,)) for key in shapes}
        profile = CostProfile()
        state, snippets, workspaces = _fixed_path(problem, profile, (
            (1, ("A", "D")), (2, ("E", "F")), (0, ("B", "C")), (0, None)))
        physical = _physical(problem, blocks, preparations, cap=2_000_000)
        frontier = _Frontier(_Dag(state, snippets, workspaces, physical))
        frontier.advance(0)
        goal = frontier.release_goal()
        self.assertEqual(goal.key, "A")
        self.assertEqual(goal.computations, 2)
        self.assertEqual(goal.required, (1 << 2) | (1 << 3))
        original = replay_fixed_dag(state, physical)
        scheduled, stats = schedule_fixed_dag(state, snippets, workspaces, physical,
                                             baseline=original, branch_width=1)
        self.assertEqual([a.term_index for a in scheduled.state.actions if a.kind == "contract"],
                         [1, 0, 0, 2])
        self.assertEqual(stats["policy"], "peak_bounded_release_dfs")
        self.assertEqual(stats["longest_consumption_steps"], 2)
        self.assertEqual(stats["release_targets_completed"], 1)
        self.assertLess(scheduled.peak_host_bytes, original.peak_host_bytes)
        self.assertAlmostEqual(scheduled.seconds, original.seconds, places=14)
        actual = replay_scene_storage(scheduled, _ArrayLoader(arrays))
        expected = (np.einsum("ij,jk,kl->il", arrays["A"], arrays["B"], arrays["C"]),
                    arrays["A"] @ arrays["D"], arrays["E"] @ arrays["F"])
        for value, reference in zip(actual, expected):
            np.testing.assert_allclose(value, reference)

    def test_resident_release_score_uses_size_per_computation_not_size_alone(self):
        shapes = {"A": (100, 100), "B": (60, 100),
                  **{key: (100, 1) for key in ("X0", "X1", "X2", "X3", "Y0", "Y1")}}
        pairs = (("A", "X0"), ("B", "Y0"), ("A", "X1"),
                 ("A", "X2"), ("A", "X3"), ("B", "Y1"))
        problem = logical_source_problem(ContractionProblem(
            {key: TensorSpec(shape, "float64") for key, shape in shapes.items()},
            tuple(ContractionTerm(pair, "ij,jk->ik") for pair in pairs),
            {key: SourceSpec(key, 0) for key in shapes}))
        blocks = {key: StorageBlock(key, "full", int(np.prod(shape)*8), host_bytes=int(np.prod(shape)*8))
                  for key, shape in shapes.items()}
        preparations = {key: SourcePreparation((key,)) for key in shapes}
        profile = CostProfile()
        state, snippets, workspaces = _fixed_path(problem, profile,
                                                  tuple(enumerate(pairs)))
        frontier = _Frontier(_Dag(state, snippets, workspaces,
                                 _physical(problem, blocks, preparations)))
        frontier.advance(0)
        frontier.advance(1)
        goal = frontier.release_goal()
        # A is larger (80,000 bytes), but needs three calculations. B releases
        # 48,000 bytes after one; both data are already resident.
        self.assertEqual(goal.key, "B")
        self.assertEqual(goal.nbytes, 48_000)
        self.assertEqual(goal.computations, 1)
        self.assertEqual(goal.required, 1 << 5)

    def test_borrowed_views_share_one_raw_release_target(self):
        shapes = {key: (1, 1, 1) for key in ("L0", "R0", "L1", "R1")}
        problem = logical_source_problem(ContractionProblem(
            {key: TensorSpec(shape, "float64") for key, shape in shapes.items()},
            (ContractionTerm(("L0", "R0"), "spc,spc->"),
             ContractionTerm(("L1", "R1"), "spc,spc->")),
            {**{key: SourceSpec(key, 0) for key in shapes},
             **{key: SourceSpec(key, 0, snapshot_bytes=0) for key in ("record0", "record1")}},
            sparse_bridges=tuple(SparseBridgeSpec(f"L{i}", f"R{i}", f"record{i}",
                                                 1, 0, same_point=True) for i in range(2))))
        blocks = {key: StorageBlock(key, "full", 8, host_bytes=8) for key in shapes}
        blocks["shared"] = StorageBlock("shared", "full", 800, host_bytes=800)
        preparations = {key: SourcePreparation((key,)) for key in shapes}
        preparations.update({key: SourcePreparation(("shared",))
                             for key in ("record0", "record1")})
        state, snippets, workspaces = _fixed_path(problem, CostProfile(),
                                                  ((0, ("L0", "R0")), (1, ("L1", "R1"))))
        frontier = _Frontier(_Dag(state, snippets, workspaces,
                                 _physical(problem, blocks, preparations)))
        frontier.advance(0)
        self.assertEqual(frontier.host, 816)
        goal = frontier.release_goal()
        self.assertEqual((goal.domain, goal.key, goal.nbytes, goal.computations),
                         ("block", "shared", 800, 1))
        change = frontier.advance(1, reversible=True)
        self.assertNotIn("shared", frontier.raw)
        self.assertEqual(frontier.host, 32)
        self.assertEqual(frontier.live, frontier.dag.outputs)
        self.assertIsNone(frontier.release_goal())
        frontier.undo(change)
        self.assertIn("shared", frontier.raw)
        self.assertEqual(frontier.host, 816)
        frontier.advance(1)
        plan = replay_fixed_dag(state, frontier.dag.physical)
        self.assertTrue(plan.feasible)
        self.assertEqual(sum(event.kind == "read" and event.block == "shared"
                             for event in plan.events), 1)

    def test_shared_consumption_counts_one_calculation_for_multiple_terms(self):
        problem = logical_source_problem(ContractionProblem(
            {key: TensorSpec((10, 10), "float64") for key in "ABC"},
            (ContractionTerm(("A", "C"), "ij,ij->"),
             ContractionTerm(("A", "B"), "ij,ij->"),
             ContractionTerm(("A", "B"), "ij,ij->")),
            {key: SourceSpec(key, 0) for key in "ABC"}))
        blocks = {key: StorageBlock(key, "full", 800, host_bytes=800) for key in "ABC"}
        preparations = {key: SourcePreparation((key,)) for key in "ABC"}
        state, snippets, workspaces = _fixed_path(problem, CostProfile(),
                                                  ((0, ("A", "C")), (1, ("A", "B"))))
        frontier = _Frontier(_Dag(state, snippets, workspaces,
                                 _physical(problem, blocks, preparations)))
        frontier.advance(0)
        goal = frontier.release_goal()
        self.assertEqual(goal.key, "A")
        self.assertEqual(goal.computations, 1)
        self.assertEqual(goal.required, 1 << 1)
        self.assertEqual(len(frontier.dag.nodes[1].actions), 2)

    def test_interleaved_terms_are_serialized_without_changing_computation_or_values(self):
        problem, arrays, blocks, preparations = _problem()
        profile = CostProfile()
        state, snippets, workspaces = _fixed_path(problem, profile, (
            (0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None),
        ))
        physical = _physical(problem, blocks, preparations)
        original = replay_fixed_dag(state, physical)
        self.assertEqual(_ordered_computation_signature(state.actions),
                         _ordered_computation_signature(original.state.actions))
        intermediates = {
            action.tensor for action in state.actions
            if action.kind not in ("load", "release", "read_block", "close_block")
        } - set(state.results)
        released = {action.tensor for action in original.state.actions if action.kind == "release"}
        self.assertTrue(intermediates)
        self.assertTrue(intermediates <= released)
        scheduled, stats = schedule_fixed_dag(
            state, snippets, workspaces, physical, baseline=original,
        )

        self.assertEqual(_computation_signature(state.actions),
                         _computation_signature(scheduled.state.actions))
        self.assertAlmostEqual(scheduled.seconds, original.seconds, places=14)
        self.assertTrue(stats["changed_order"])
        self.assertLess(scheduled.peak_host_bytes, original.peak_host_bytes)
        for term_index in range(len(problem.terms)):
            self.assertEqual(_per_term_signature(state.actions, term_index),
                             _per_term_signature(scheduled.state.actions, term_index))

        actual = replay_scene_storage(scheduled, _ArrayLoader(arrays))
        expected = (
            np.einsum("ik,kj,j->i", arrays["A"], arrays["B"], arrays["C"], optimize=False),
            np.einsum("ik,kj,j->i", arrays["D"], arrays["E"], arrays["F"], optimize=False),
        )
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(result, reference)

    def test_shared_calculation_is_paid_once_and_term_updates_keep_order(self):
        problem, _, blocks, preparations = _problem(shared=True)
        profile = CostProfile()
        state, snippets, workspaces = _fixed_path(problem, profile, (
            (0, ("A", "B")), (0, None), (1, None),
        ))
        physical = _physical(problem, blocks, preparations)
        original = replay_fixed_dag(state, physical)
        scheduled, stats = schedule_fixed_dag(
            state, snippets, workspaces, physical, baseline=original,
        )

        shared_tensor = state.actions[snippets[0][0]].tensor
        for plan in (original, scheduled):
            actions = [action for action in plan.state.actions
                       if action.kind == "contract" and action.tensor == shared_tensor]
            self.assertEqual(sum(not action.reused for action in actions), 1)
            self.assertEqual(sum(action.reused for action in actions), 1)
        self.assertAlmostEqual(scheduled.seconds, original.seconds, places=14)
        self.assertEqual(stats["nodes"], 3)
        for term_index in range(len(problem.terms)):
            self.assertEqual(_per_term_signature(state.actions, term_index),
                             _per_term_signature(scheduled.state.actions, term_index))

    def test_deadline_and_small_state_budget_return_a_complete_legal_path(self):
        problem, _, blocks, preparations = _problem()
        profile = CostProfile()
        state, snippets, workspaces = _fixed_path(problem, profile, (
            (0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None),
        ))
        for deadline, state_budget in ((0.0, 1), (1.0, 1)):
            with self.subTest(deadline=deadline, state_budget=state_budget):
                plan, stats = schedule_fixed_dag(
                    state, snippets, workspaces,
                    _physical(problem, blocks, preparations),
                    deadline_seconds=deadline, max_states=state_budget,
                )
                self.assertTrue(plan.state.done)
                self.assertTrue(plan.feasible)
                self.assertEqual(_computation_signature(state.actions),
                                 _computation_signature(plan.state.actions))
                self.assertEqual(stats["nodes"], 4)
                self.assertGreaterEqual(stats["audited_schedules"], 1)
                original = replay_fixed_dag(state, _physical(problem, blocks, preparations))
                self.assertAlmostEqual(plan.seconds, original.seconds, places=14)

    def test_reordering_can_rescue_the_original_peak_but_a_lower_cap_is_rejected(self):
        problem, _, blocks, preparations = _problem()
        profile = CostProfile()
        state, snippets, workspaces = _fixed_path(problem, profile, (
            (0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None),
        ))
        roomy = _physical(problem, blocks, preparations)
        original = replay_fixed_dag(state, roomy)
        roomy_best, _ = schedule_fixed_dag(state, snippets, workspaces, roomy, baseline=original)

        tight = _physical(problem, blocks, preparations, cap=roomy_best.peak_host_bytes)
        rescued, stats = schedule_fixed_dag(state, snippets, workspaces, tight)
        self.assertGreater(original.peak_host_bytes, tight.memory_limit)
        self.assertTrue(rescued.feasible)
        self.assertLessEqual(rescued.peak_host_bytes, tight.memory_limit)
        self.assertTrue(stats["changed_order"])

        too_tight = _physical(problem, blocks, preparations, cap=roomy_best.peak_host_bytes - 1)
        rejected, _ = schedule_fixed_dag(state, snippets, workspaces, too_tight)
        self.assertFalse(rejected.feasible)

    def test_scheduler_preserves_modeled_costs_with_an_alternate_profile(self):
        problem, _, blocks, preparations = _problem()
        profile = CostProfile(
            contract_scale=0.4,
            preparation_call_seconds=0.003,
        )
        state, snippets, workspaces = _fixed_path(problem, profile, (
            (0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None),
        ))
        physical = _physical(problem, blocks, preparations, profile=profile)
        original = replay_fixed_dag(state, physical)
        scheduled, _ = schedule_fixed_dag(state, snippets, workspaces, physical, baseline=original)
        self.assertAlmostEqual(scheduled.seconds, original.seconds, places=14)
        self.assertAlmostEqual(sum(scheduled.preparation_costs.values()),
                               sum(original.preparation_costs.values()), places=14)
        self.assertEqual(_computation_signature(state.actions),
                         _computation_signature(scheduled.state.actions))

    def test_graph_session_keeps_source_preparation_costs_after_post_search_scheduling(self):
        problem, _, blocks, preparations = _problem()
        profile = CostProfile(preparation_call_seconds=0.003)
        result = solve_graph_storage(
            problem,
            blocks,
            preparations,
            memory_limit=1_000_000,
            profile=profile,
            deadline_seconds=1.0,
            max_states=128,
            component_workers=1,
        )
        plan = result.plan
        loads = [(index, action.tensor) for index, action in enumerate(plan.state.actions)
                 if action.kind == "load"]
        self.assertEqual({key for _, key in loads}, set(preparations))
        self.assertEqual(len(loads), len(preparations))
        expected = sum(preparation.seconds(profile) for preparation in preparations.values())
        self.assertAlmostEqual(plan.preparation_seconds, expected, places=14)
        first_index, first_source = loads[0]
        self.assertAlmostEqual(plan.preparation_costs[first_index],
                               preparations[first_source].seconds(profile), places=14)
        self.assertTrue(plan.feasible)
        self.assertEqual(result.statistics["storage_schedule"]["nodes"], 4)
        self.assertTrue(result.statistics["storage_schedule"]["required"])
        self.assertTrue(result.statistics["storage_schedule"]["completed"])

    def test_storage_optimization_finishes_after_search_state_budget_exhaustion(self):
        # Keep productive alternatives after the first complete compact tail,
        # so this still exhausts a graph-state budget rather than tail steps.
        problem, _, blocks, preparations = _problem(shared=True)
        result = solve_storage(problem, blocks, preparations, memory_limit=1_000_000,
                               max_states=1, component_workers=1)
        self.assertTrue(result.limited)
        self.assertEqual(result.stop_reason, "max_states")
        self.assertTrue(result.state.done)
        self.assertTrue(result.plan.feasible)
        scheduling = result.statistics["storage_schedule"]
        self.assertTrue(scheduling["required"])
        self.assertTrue(scheduling["completed"])
        self.assertGreaterEqual(scheduling["audited_schedules"], 1)
        self.assertEqual(scheduling["peak_host_bytes"], result.plan.peak_host_bytes)

    def test_public_delivery_rejects_a_search_that_skips_storage_optimization(self):
        problem, _, blocks, preparations = _problem()
        result = solve_storage(problem, blocks, preparations, memory_limit=1_000_000,
                               root_width=1, branch_width=1)
        for scheduling in (None, dict(required=True, completed=False)):
            stats = dict(result.statistics)
            stats.pop("storage_schedule")
            if scheduling is not None:
                stats["storage_schedule"] = scheduling
            unfinished = replace(result, statistics=stats)
            with self.subTest(scheduling=scheduling), patch(
                    "lqcdweave.contraction.planning.optimizers.graph_session.solve_graph_storage",
                    return_value=unfinished):
                with self.assertRaisesRegex(RuntimeError, "completed storage optimization"):
                    solve_storage(problem, blocks, preparations, memory_limit=1_000_000)


if __name__ == "__main__":
    unittest.main()
