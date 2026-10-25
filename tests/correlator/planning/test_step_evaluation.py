"""Independent checks for aggregate evaluation and retained state changes."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture, ContractionTerm
from correlator.planning.test_candidate_metadata import join_fixture
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contracts.correlator.api import ResourceAllocation, ResourceSnapshot


class StepEvaluationTests(unittest.TestCase):
    def test_candidate_fingerprint_collision_does_not_merge_different_algebra(self):
        space = fixture()
        original = ContractionState._contract_layout
        def collide(state, *args, **kwargs):
            layout = original(state, *args, **kwargs)
            return layout[:-2] + ("node:collision", layout[-1])
        with patch.object(ContractionState, "_contract_layout", collide):
            edges = space.successors(space.initial_state())
        self.assertEqual(len(edges), 3)
        self.assertEqual(len({e.operation.operands for e in edges}), 3)
        self.assertEqual(len(space.workspace_estimates), 3)

    def test_unique_operations_build_shared_effect_once_without_public_contract(self):
        space = fixture((ContractionTerm(("A", "B", "C"), "i,i,i->"),) * 8)
        with patch.object(ContractionState, "contract", side_effect=AssertionError("full replay")):
            edges = space.successors(space.initial_state())
        self.assertEqual(len(edges), 3)
        self.assertEqual(space.logical_effects, 3)
        self.assertEqual(space.aggregate_checks, 3)
        self.assertTrue(all(len({t.factors for t in e.state.state.pending}) == 1 for e in edges))

    def test_read_metadata_does_not_affect_step_ordering_or_modeled_cost(self):
        terms = (ContractionTerm(("A", "B"), "i,i->"), ContractionTerm(("B", "C"), "i,i->"))
        space = fixture(terms)
        # All three blocks remain mandatory for execution, but read bytes do
        # not enter the modeled computation objective.
        space.blocks["A"] = replace(space.blocks["A"], read_bytes=10**9)
        space = type(space)(space.problem, space.blocks, space.preparations,
                            memory_limit=space.memory_limit)
        # Physical cost tables are derived on first use; fix them here so the
        # patches below still prove later steps never re-price a read.
        space.materialize_physical_facts()
        initial = space.initial_state()
        self.assertEqual(space.cost(initial), 0)
        with patch.object(CostProfile, "snapshot_seconds", side_effect=AssertionError("re-costed load")), \
             patch.object(CostProfile, "preparation_seconds", side_effect=AssertionError("re-costed preparation")):
            first, other = space.successors(initial)
        self.assertAlmostEqual(space.cost(first.state) - space.cost(initial),
                               space.cost(other.state) - space.cost(initial), places=12)
        second = space.successors(first.state)[0]
        plan = space.plan(second.state, first.actions + second.actions)
        self.assertAlmostEqual(plan.seconds, space.cost(second.state))
        self.assertFalse(hasattr(plan, "read_seconds"))

    def test_selected_delta_does_not_describe_prepare_or_replay_again(self):
        space = fixture()
        parent = space.initial_state()
        preview = space.successors(parent)[0]
        delta = space.freeze_transition(parent, preview)
        with patch.object(space, "evaluate", side_effect=AssertionError("re-evaluated")), \
             patch.object(space, "preview", side_effect=AssertionError("replayed")), \
             patch.object(space, "logical_steps", side_effect=AssertionError("re-enumerated")):
            edge = space.apply_delta(parent, delta)
        self.assertEqual(space.selected_delta_applications, 1)
        self.assertEqual(edge.operation, preview.operation)
        cheaper = replace(parent, state=replace(parent.state, seconds=7.0))
        rebased = space.apply_delta(cheaper, delta)
        self.assertAlmostEqual(rebased.state.state.seconds, edge.state.state.seconds + 7)

    def test_sparse_delta_preserves_unchanged_terms_and_normalized_borrowed_tuple(self):
        terms = (ContractionTerm(("A", "B"), "i,i->"),
                 ContractionTerm(("A", "C"), "i,i->"))
        space = fixture(terms)
        before = space.initial_state()
        edge = space.successors(before)[0]
        delta = space.freeze_transition(before, edge)
        applied = space.apply_delta(before, delta)
        self.assertEqual(space.key(applied.state), space.key(edge.state))
        changed = {index for index, _ in delta.pending}
        unchanged = next(index for index in range(len(before.state.pending)) if index not in changed)
        self.assertIs(applied.state.state.pending[unchanged], before.state.pending[unchanged])

        borrowed = (("view", ("raw",)),)
        snapshot = ResourceSnapshot((ResourceAllocation("block", "raw", host_bytes=8),), borrowed)
        self.assertIs(snapshot.borrowed, borrowed)


    def test_aggregate_bound_dominates_every_audited_phase_in_small_domains(self):
        spaces = [fixture(), fixture(cap=264), join_fixture()]
        for space in spaces:
            if not space.problem.baryon_joins:
                space.blocks["A"] = replace(space.blocks["A"], staging_host_bytes=3)
                space.preparations["B"] = replace(space.preparations["B"], workspace_host_bytes=5)
            pending = [(space.initial_state(), ())]
            seen = set()
            while pending:
                cursor, actions = pending.pop()
                key = space.key(cursor)
                if key in seen:
                    continue
                seen.add(key)
                for edge in space.successors(cursor):
                    path = actions + edge.actions
                    if edge.state.state.done:
                        plan = space.plan(edge.state, path)
                        self.assertTrue(plan.feasible)
                        self.assertLessEqual(plan.peak_host_bytes, edge.state.peak_host)
                        self.assertLessEqual(plan.peak_device_bytes, edge.state.peak_device)
                    else:
                        pending.append((edge.state, path))

    def test_peak_bound_can_be_conservative_without_corrupting_audit(self):
        space = fixture()
        # Reading A needs a large staging buffer. Once B/output exist, the
        # staging buffer is gone. The aggregate model intentionally adds its
        # allowance to the complete preparation/compute owner set.
        space.blocks["A"] = replace(space.blocks["A"], staging_host_bytes=300)
        first = space.successors(space.initial_state())[0]
        second = space.successors(first.state)[0]
        plan = space.plan(second.state, first.actions + second.actions)
        self.assertGreater(second.state.peak_host, plan.peak_host_bytes)
        self.assertTrue(plan.feasible)

    def test_two_tier_restore_delta_matches_audited_overlap_and_cost(self):
        space = fixture(backend="cupy", host=10000)
        first = space.successors(space.initial_state())[0]
        key = first.operation.tensor
        host_move = space._move(first.state, "spill_host", "tensor", key)
        host = space.apply(first.state, host_move)
        disk_move = space._move(host, "spill_disk", "tensor", key)
        disk = space.apply(host, disk_move)
        edge = space.successors(disk)[0]
        delta = space.freeze_transition(disk, edge)
        rebased = space.apply_delta(disk, delta)
        self.assertEqual(space.key(rebased.state), space.key(edge.state))
        plan = space.plan(rebased.state, first.actions + (host_move, disk_move) + rebased.actions)
        self.assertEqual([a.kind for a in edge.actions if a.kind.startswith("restore_")],
                         ["restore_host", "restore_device"])
        self.assertAlmostEqual(plan.seconds, space.cost(rebased.state))
        self.assertLessEqual(plan.peak_host_bytes, rebased.state.peak_host)
        self.assertLessEqual(plan.peak_device_bytes, rebased.state.peak_device)


if __name__ == "__main__":
    unittest.main()
