from dataclasses import replace
import unittest

from lqcdweave.contracts.correlator.api import (
    Action, ContractionProblem, ContractionTerm, TensorSpec, SourceSpec,
    StorageBlock, SourcePreparation, ResourceAllocation, ResourceSnapshot,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
from lqcdweave.contraction.planning.optimizers.macro_space import UnavailableInput


def fixture(terms=None, *, cap=10000, backend="numpy", host=None, disk=None):
    specs = {k: TensorSpec((4,), "float64") for k in "ABC"}
    problem = ContractionProblem(specs, terms or (ContractionTerm(("A", "B", "C"), "i,i,i->"),),
        {k: SourceSpec(k, 0) for k in specs}, backend=backend)
    blocks = {k: StorageBlock("dataset", k, 32, host_bytes=0 if backend == "cupy" else 32,
                             device_bytes=32 if backend == "cupy" else 0) for k in specs}
    preps = {k: SourcePreparation((k,)) for k in specs}
    return MacroSearchSpace(problem, blocks, preps, memory_limit=cap,
                            host_memory_limit=host, disk_limit_bytes=disk)


class MacroStepTests(unittest.TestCase):
    def test_fixed_preparation_and_no_path_history_in_cursor(self):
        space = fixture()
        initial = space.initial_state()
        candidate = next(iter(space.candidates(initial)))
        edge = space.preview(initial, candidate)
        self.assertEqual([(a.kind, a.tensor) for a in edge.actions[:-1]],
                         [("read_block", "A"), ("load", "A"), ("read_block", "B"), ("load", "B")])
        self.assertEqual(edge.state.state.actions, ())
        self.assertEqual(initial.resources, ResourceSnapshot())
        self.assertEqual(initial.state.loaded, frozenset())

    def test_room_retains_unused_cache_and_pressure_reclaims_only_needed(self):
        roomy = fixture()
        first = roomy.successors(roomy.initial_state())[0]
        second = roomy.successors(first.state)[0]
        self.assertFalse(any(a.kind in ("release", "close_block", "discard") for a in second.actions))
        space = fixture(cap=264)
        first = space.successors(space.initial_state())[0]
        self.assertTrue(first.state.resources.allocations)
        second = space.successors(first.state)[0]
        self.assertTrue(any(a.kind in ("release", "close_block") for a in second.actions))
        self.assertTrue(space.plan(second.state, first.actions + second.actions).feasible)

    def test_failure_rolls_back_reads_materialization_and_releases(self):
        space = fixture(cap=150)
        cursor = space.initial_state()
        candidate = next(iter(space.candidates(cursor)))
        with self.assertRaises(MemoryError): space.preview(cursor, candidate)
        self.assertEqual(cursor.resources.opened_blocks, frozenset())
        self.assertEqual(cursor.state.seconds, 0)
        self.assertEqual(cursor.state.loaded, frozenset())

    def test_device_pressure_keeps_unpressured_host_cache(self):
        space = fixture(backend="cupy", cap=200)
        space.blocks = {key: replace(block, host_bytes=32, device_bytes=0) for key, block in space.blocks.items()}
        first = space.successors(space.initial_state())[0]
        second = space.successors(first.state)[0]
        self.assertTrue(any(a.kind == "release" for a in second.actions))
        self.assertFalse(any(a.kind == "close_block" for a in second.actions))
        self.assertEqual(sum(a.host_bytes for a in second.state.resources.allocations), 96)
        self.assertTrue(space.plan(second.state, first.actions + second.actions).feasible)

    def test_shared_terms_have_one_candidate_and_all_results_updated(self):
        term = ContractionTerm(("A", "B"), "i,i->")
        space = fixture((term, term))
        candidates = tuple(space.candidates(space.initial_state()))
        self.assertEqual(len(candidates), 1)
        edge = space.preview(space.initial_state(), candidates[0])
        self.assertTrue(edge.state.state.done)
        self.assertEqual(edge.state.state.results[0], edge.state.state.results[1])
        plan = space.plan(edge.state, edge.actions)
        self.assertAlmostEqual(plan.seconds, space.cost(edge.state))

    def test_staging_and_full_materialization_workspace_reject_atomically(self):
        for stage in ("read", "prepare"):
            with self.subTest(stage=stage):
                space = fixture(cap=1000)
                if stage == "read":
                    space.blocks["A"] = replace(space.blocks["A"], staging_host_bytes=1000)
                else:
                    space.preparations["A"] = replace(space.preparations["A"], workspace_host_bytes=1000)
                initial = space.initial_state()
                with self.assertRaises(MemoryError): space.preview(initial, next(iter(space.candidates(initial))))
                self.assertEqual(initial.resources, ResourceSnapshot())
                self.assertEqual(initial.state.seconds, 0)

    def test_useful_source_discard_rematerializes_without_original_reread(self):
        terms = (ContractionTerm(("A", "B"), "i,i->"), ContractionTerm(("A", "C"), "i,i->"))
        space = fixture(terms)
        first = space.successors(space.initial_state())[0]
        discard = space._move(first.state, "discard", "tensor", "A")
        saved = space.apply(first.state, discard)
        second = space.successors(saved)[0]
        plan = space.plan(second.state, first.actions + (discard,) + second.actions)
        self.assertEqual(sum(e.kind == "read" and e.block == "A" for e in plan.events), 1)
        self.assertEqual(sum(a.kind == "load" and a.tensor == "A" for a in plan.state.actions), 2)
        self.assertFalse(hasattr(plan, "read_seconds"))

    def test_borrowed_view_pins_owner_and_never_duplicates_storage(self):
        base = fixture()
        problem = replace(base.problem, sources={**base.problem.sources, "record": SourceSpec("record", 0, snapshot_bytes=24)})
        blocks = {**base.blocks, "raw": StorageBlock("record", "all", 24, host_bytes=24)}
        space = MacroSearchSpace(problem, blocks, {**base.preparations, "record": SourcePreparation(("raw",))}, memory_limit=1000)
        cursor = space.apply(space.initial_state(), Action("read_block", "raw"))
        load = cursor.state.load("record", space.logical_profile, 1000).actions[-1]
        cursor = space.apply(cursor, load)
        self.assertEqual(cursor.resources.borrowed, (("record", ("raw",)),))
        self.assertEqual(sum(a.host_bytes for a in cursor.resources.allocations), 24)
        with self.assertRaises(ValueError): space.apply(cursor, space._move(cursor, "spill_disk", "block", "raw"))
        released = space.apply(cursor, Action("release", "record"))
        self.assertEqual(released.resources.allocations, cursor.resources.allocations)
        self.assertEqual(released.resources.borrowed, ())

    def test_closed_original_block_cannot_be_read_again(self):
        space = fixture((ContractionTerm(("A",), "i->"),))
        edge = space.successors(space.initial_state())[0]
        cursor = space.apply(edge.state, Action("close_block", "A"))
        with self.assertRaises(UnavailableInput):
            space.apply(cursor, Action("read_block", "A"))

    def test_partial_memory_failure_does_not_generate_adjustments(self):
        specs = {"A": TensorSpec((10,), "float64"), "B": TensorSpec((10,), "float64"),
                 "C": TensorSpec((1,), "float64"), "D": TensorSpec((1,), "float64")}
        problem = ContractionProblem(specs, (ContractionTerm(("A", "B"), "i,i->"),
                                            ContractionTerm(("C", "D"), "j,j->")),
                                     {k: SourceSpec(k, 0) for k in specs})
        space = MacroSearchSpace(problem, {}, {k: SourcePreparation(()) for k in specs}, memory_limit=64)
        edges = space.successors(space.initial_state())
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].actions[-1].operands, ("C", "D"))
        self.assertEqual(space.resource_branches, 0)

    def test_adjustment_only_after_all_complete_steps_exceed_cap(self):
        # A and B remain live in additional terms, so dead reclamation cannot
        # make room. Only now may the planner spill or discard recoverable data.
        term = ContractionTerm(("A", "B", "C"), "i,i,i->")
        other = ContractionTerm(("A", "C"), "i,i->i")
        third = ContractionTerm(("B", "C"), "i,i->i")
        space = fixture((term, other, third), cap=264)
        first = space.successors(space.initial_state())[0]
        edges = space.successors(first.state)
        self.assertTrue(edges)
        self.assertTrue(all(len(edge.actions) == 1 for edge in edges))
        self.assertGreater(space.resource_branches, 0)
        self.assertTrue(all(edge.actions[0].kind in ("discard", "spill_host", "spill_disk") for edge in edges))

    def test_final_output_copy_is_in_atomic_step_and_can_reclaim_dead_data(self):
        terms = (ContractionTerm(("A",), "i->i"),) * 10
        space = fixture(terms, cap=360)
        edge = space.successors(space.initial_state())[0]
        self.assertTrue(any(a.kind == "release" for a in edge.actions))
        self.assertTrue(space.plan(edge.state, edge.actions).feasible)
        tight = fixture(terms, cap=351)
        cursor = tight.initial_state()
        with self.assertRaises(MemoryError): tight.preview(cursor, next(iter(tight.candidates(cursor))))
        self.assertEqual(cursor.resources.opened_blocks, frozenset())

    def test_gpu_two_tier_move_restore_preserves_capacity_without_reread(self):
        space = fixture(backend="cupy", host=10000)
        first = space.successors(space.initial_state())[0]
        key = first.state.state.pending[0].factors[-1].tensor
        host_action = space._move(first.state, "spill_host", "tensor", key)
        host = space.apply(first.state, host_action)
        disk_action = space._move(host, "spill_disk", "tensor", key)
        disk = space.apply(host, disk_action)
        candidate = next(iter(space.candidates(disk)))
        edge = space.preview(disk, candidate)
        restores = [a for a in edge.actions if a.kind.startswith("restore_")]
        self.assertEqual([a.kind for a in restores], ["restore_host", "restore_device"])
        self.assertEqual(disk.resources.allocations, space.apply(host, disk_action).resources.allocations)
        plan = space.plan(edge.state, first.actions + (host_action, disk_action) + edge.actions)
        self.assertEqual(sum(e.kind == "read" for e in plan.events), 3)
        self.assertEqual(plan.peak_disk_bytes, 32)
        self.assertFalse(hasattr(plan, "resource_seconds"))
        self.assertTrue(all(a.seconds == 0 for a in (host_action, disk_action, *restores)))

    def test_receiver_capacity_and_disk_cap_are_hard_constraints(self):
        space = fixture(backend="cupy", host=16, disk=16)
        first = space.successors(space.initial_state())[0]
        key = first.state.state.pending[0].factors[-1].tensor
        with self.assertRaises(MemoryError): space.apply(first.state, space._move(first.state, "spill_host", "tensor", key))
        cpu = fixture(disk=16)
        first = cpu.successors(cpu.initial_state())[0]
        key = first.state.state.pending[0].factors[-1].tensor
        with self.assertRaises(MemoryError): cpu.apply(first.state, cpu._move(first.state, "spill_disk", "tensor", key))

    def test_completed_output_restores_when_final_assembly_needs_it(self):
        terms = (ContractionTerm(("A",), "i->"), ContractionTerm(("B",), "i->"))
        space = fixture(terms)
        first = space.successors(space.initial_state())[0]
        output = first.state.state.results[0]
        move = space._move(first.state, "spill_disk", "tensor", output)
        saved = space.apply(first.state, move)
        second = space.successors(saved)[0]
        self.assertTrue(second.state.state.done)
        self.assertEqual([a.tensor for a in second.actions if a.kind == "restore_host"], [output])
        self.assertTrue(space.plan(second.state, first.actions + (move,) + second.actions).feasible)

    def test_intermediate_cannot_be_discarded_but_source_can_recover(self):
        space = fixture()
        first = space.successors(space.initial_state())[0]
        key = first.state.state.pending[0].factors[-1].tensor
        with self.assertRaises(ValueError): space.apply(first.state, space._move(first.state, "discard", "tensor", key))

    def test_checkpoint_covers_candidate_preparation_and_adjustment(self):
        space = fixture()
        calls = []
        def stop():
            calls.append(1)
            if len(calls) == 5:
                raise RuntimeError("budget interruption")
        cursor = space.initial_state()
        with self.assertRaisesRegex(RuntimeError, "budget interruption"):
            space.successors(cursor, stop)
        self.assertEqual(space.resource_branches, 0)
        self.assertEqual(cursor.resources.opened_blocks, frozenset())

    def test_generated_cache_reuse_liveness_tracks_opaque_current_factors(self):
        term = ContractionTerm(("A", "B", "C"), "i,i,i->")
        other = ContractionTerm(("A", "B", "C"), "i,i,i->i")
        space = fixture((term, other))
        first = space.successors(space.initial_state())[0]
        generated = first.actions[-1].tensor
        self.assertIn(generated, space.useful(first.state))
        for edge in space.successors(first.state):
            if edge.state.state.done:
                self.assertTrue(set(edge.state.state.results) <= space.useful(edge.state))

    def test_shared_baryon_join_has_one_decision_and_one_paid_kernel(self):
        from lqcdweave.contracts.correlator.api import BaryonJoinSpec
        specs = {"spin": TensorSpec((1, 1, 1), "float64"), "mode": TensorSpec((1, 1, 1), "float64"),
                 **{key: TensorSpec((1, 1), "float64") for key in ("a", "b", "c")}}
        term = ContractionTerm(("a", "b", "c"), "ad,be,cf->")
        joins = tuple(BaryonJoinSpec(scene, "shared", "spin", "mode", "abc", "def") for scene in (0, 1))
        problem = ContractionProblem(specs, (term, term), {key: SourceSpec(key, 0) for key in specs}, baryon_joins=joins)
        space = MacroSearchSpace(problem, {}, {key: SourcePreparation(()) for key in specs}, memory_limit=10000)
        initial = space.initial_state()
        candidates = [a for a in space.candidates(initial) if a.kind == "baryon_join"]
        self.assertEqual(len(candidates), 1)
        edge = space.preview(initial, candidates[0])
        self.assertTrue(edge.state.state.done)
        joins = [a for a in edge.actions if a.kind == "baryon_join"]
        self.assertEqual(len(joins), 2)
        self.assertFalse(joins[0].reused)
        self.assertTrue(joins[1].reused)
        self.assertTrue(space.plan(edge.state, edge.actions).feasible)

    def test_join_axis_bindings_are_distinct_and_replay_matches_independent_einsum(self):
        import numpy as np
        from lqcdweave.contracts.correlator.api import BaryonJoinSpec
        from lqcdweave.contraction.planning.api import ContractionState, logical_cost_profile, logical_source_problem, plan_scene_storage
        from lqcdweave.contraction.execution.api import replay_scene_storage
        specs = {key: TensorSpec((2, 2), "float64") for key in ("a", "b", "c")}
        specs.update(spin=TensorSpec((2, 2, 2), "float64"), mode=TensorSpec((2, 2, 2), "float64"))
        term = ContractionTerm(("a", "b", "c"), "ad,be,cf->")
        joins = (BaryonJoinSpec(0, "shared", "spin", "mode", "abc", "def"),
                 BaryonJoinSpec(1, "shared", "spin", "mode", "bac", "def"))
        problem = ContractionProblem(specs, (term, term), {key: SourceSpec(key, 0) for key in specs}, baryon_joins=joins)
        profile = CostProfile()
        state = ContractionState.initial(logical_source_problem(problem))
        for key in specs:
            state = state.load(key, logical_cost_profile(profile), 10000)
        for join in joins:
            state = state.contract_join(join, logical_cost_profile(profile), 10000)
        self.assertNotEqual(state.results[0], state.results[1])
        self.assertFalse(any(a.reused for a in state.actions if a.kind == "baryon_join"))
        arrays = {key: (np.arange(np.prod(spec.shape), dtype=np.float64).reshape(spec.shape) + index + 1)
                  for index, (key, spec) in enumerate(specs.items())}
        class Loader:
            def read_block(self, key, block): return arrays
            def materialize(self, key, source, blocks): return blocks["raw"][key]
        blocks = {"raw": StorageBlock("all", "all", 224, host_bytes=224)}
        preps = {key: SourcePreparation(("raw",)) for key in specs}
        plan = plan_scene_storage(state, blocks, preps, profile=profile, memory_limit=10000)
        values = replay_scene_storage(plan, Loader())
        expected = tuple(np.einsum(labels + ",def,ad,be,cf->", arrays["spin"], arrays["mode"],
                                   arrays["a"], arrays["b"], arrays["c"]) for labels in ("abc", "bac"))
        self.assertNotEqual(expected[0], expected[1])
        for actual, reference in zip(values, expected):
            np.testing.assert_allclose(actual, reference)
        space = MacroSearchSpace(problem, blocks, preps, memory_limit=10000)
        candidates = [a for a in space.candidates(space.initial_state()) if a.kind == "baryon_join"]
        self.assertEqual(len(candidates), 2)
        first = space.preview(space.initial_state(), candidates[0])
        self.assertIsNone(first.state.state.results[1])


if __name__ == "__main__":
    unittest.main()
