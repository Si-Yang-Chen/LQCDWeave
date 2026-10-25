"""Resource moves are audited as fixed trajectories, without numerical IO."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from lqcdweave.contracts.correlator.api import (
    Action, ContractionProblem, ContractionTerm, SourceSpec, TensorSpec, StorageBlock,
    SourcePreparation, ResourceTransition, ResourceAllocation, ResourceSnapshot, StorageEvent,
    validate_executable_resources,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.api import (
    ContractionState, logical_source_problem, logical_cost_profile, plan_scene_storage,
    audit_resource_trajectory, transition_resource, resource_point,
)
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.storage.contraction_plans.api import save_storage_schedule, encode_plan_value, decode_plan_value


def fixture():
    profile = CostProfile()
    problem = logical_source_problem(ContractionProblem(
        {"A": TensorSpec((2,), "float64")}, (ContractionTerm(("A",), "i->"),),
        {"A": SourceSpec("A", 0)}))
    blocks = {"raw": StorageBlock("dataset", "all", 64, host_bytes=64)}
    preps = {"A": SourcePreparation(("raw",))}
    state = ContractionState.initial(problem).load("A", logical_cost_profile(profile), 10000)
    state = state.contract(0, (0,), logical_cost_profile(profile), 10000)
    return profile, state, blocks, preps


def trajectory(spill=True):
    profile, state, blocks, preps = fixture()
    actions = [Action("read_block", "raw"), state.actions[0]]
    if spill:
        actions += [Action("spill_disk", "A", resource=ResourceTransition("tensor", "host", "disk", 16)),
                    Action("restore_host", "A", resource=ResourceTransition("tensor", "disk", "host", 16))]
    actions += [state.actions[1], Action("close_block", "raw")]
    state = replace(state, actions=tuple(actions), seconds=sum(a.seconds for a in actions))
    return profile, state, blocks, preps


class ResourceTrajectoryTests(unittest.TestCase):
    def test_owner_totals_follow_disk_overlap_and_raw_close_exactly(self):
        profile, state, blocks, preps = trajectory()
        plan = audit_resource_trajectory(state, blocks, preps, profile=profile,
            memory_limit=1000, reserved_host_bytes=7)
        # Independent ownership table: raw 64 B, source 16 B, scalar 8 B.
        # Moving a source overlaps both tiers before releasing the old tier.
        expected = [(64, 0, 0, 0, 0), (64, 0, 16, 0, 0),
                    (64, 0, 16, 0, 16), (64, 0, 16, 0, 16),
                    (64, 0, 24, 0, 0), (0, 0, 24, 0, 0),
                    (0, 0, 24, 0, 0)]
        self.assertEqual([(p.raw_host_bytes, p.raw_device_bytes,
            p.logical_host_bytes, p.logical_device_bytes, p.temporary_disk_bytes)
            for p in plan.memory], expected)
        self.assertTrue(all(p.reserved_host_bytes == 7 for p in plan.memory))

    def test_cached_reduction_after_input_discard_uses_exact_incremental_residency(self):
        from lqcdweave.contraction.planning.canonical_operations import describe, replay
        from lqcdweave.contraction.planning import resources

        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                profile = CostProfile()
                model = logical_cost_profile(profile)
                term = ContractionTerm(('A',), 'i->')
                problem = logical_source_problem(ContractionProblem(
                    {'A': TensorSpec((3,), 'float64')}, (term, term),
                    {'A': SourceSpec('A', 0)}, backend=backend))
                initial = ContractionState.initial(problem).load('A', model, 1000)
                first = describe(initial, 0, (0,), profile=model)[0]
                computed = replay(initial, first, model).release('A')
                symbolic = replace(computed, loaded=computed.loaded | {'A'})
                hit = describe(symbolic, 1, (0,), profile=model)[0]
                self.assertTrue(hit.reused)
                final = replay(symbolic, hit, model)
                discard = Action('discard', 'A', resource=ResourceTransition(
                    'tensor', 'host' if backend == 'numpy' else 'device', 'absent', 24))
                final = replace(final, loaded=computed.loaded, actions=(Action('read_block', 'raw'),)
                    + computed.actions[:-1] + (discard, hit, Action('close_block', 'raw')))
                blocks = {'raw': StorageBlock('input', 'all', 24, host_bytes=24)}
                preparations = {'A': SourcePreparation(('raw',))}
                apply = resources._apply_selected
                reused_totals = []

                def observe(state, scene, action, *args, **kwargs):
                    if action.reused:
                        self.assertIn('_resident_totals', state.__dict__)
                        # Input and scalar result are the only symbolic owners.
                        self.assertEqual(state.resident_bytes, 32)
                        self.assertEqual(state.resident_host_bytes, 32 if backend == 'numpy' else 0)
                        reused_totals.append(state.resident_bytes)
                    return apply(state, scene, action, *args, **kwargs)

                with patch.object(resources, '_apply_selected', side_effect=observe):
                    plan = audit_resource_trajectory(final, blocks, preparations,
                        profile=profile, memory_limit=1000)
                self.assertEqual(reused_totals, [32])
                self.assertEqual(plan.state.loaded, frozenset((first.tensor,)))
                self.assertTrue(plan.feasible)
                cached_point = plan.memory[4]
                self.assertEqual(cached_point.host_bytes, 32 if backend == 'numpy' else 24)
                self.assertEqual(cached_point.device_bytes, 8 if backend == 'cupy' else 0)

    def test_fixed_raw_lifetime_cost_disk_and_backward_legacy_route(self):
        profile, state, blocks, preps = trajectory()
        plan = plan_scene_storage(state, blocks, preps, profile=profile, memory_limit=1000,
                                  explicit_resources=True)
        self.assertEqual([(e.kind, e.action_index) for e in plan.events], [("read", 0), ("release", 5)])
        self.assertEqual(plan.peak_disk_bytes, 16)
        self.assertAlmostEqual(plan.seconds, state.seconds)
        self.assertTrue(plan.feasible)
        repeated = audit_resource_trajectory(plan.state, blocks, preps, profile=profile, memory_limit=1000)
        self.assertEqual(repeated.seconds, plan.seconds)
        self.assertEqual(repeated.memory, plan.memory)
        self.assertEqual(plan.memory[3].temporary_disk_bytes, 16)  # restoration overlaps source
        self.assertEqual(plan.memory[3].host_bytes, 80)
        profile, old, blocks, preps = fixture()
        legacy = plan_scene_storage(old, blocks, preps, profile=profile, memory_limit=1000)
        self.assertFalse(legacy.explicit_resources)
        self.assertEqual(legacy.peak_disk_bytes, 0)
        self.assertEqual(legacy.events[-1].action_index, 0)
        self.assertGreater(plan.peak_host_bytes, legacy.peak_host_bytes)

    def test_disk_limit_and_overlap_limits_roll_back(self):
        profile = CostProfile()
        initial = ResourceSnapshot((ResourceAllocation("tensor", "a", device_bytes=16),))
        action = Action("spill_host", "a", resource=ResourceTransition("tensor", "device", "host", 16, 4, 3))
        with self.assertRaises(MemoryError):
            transition_resource(initial, action, backend="cupy", memory_limit=18, host_memory_limit=100)
        self.assertEqual(initial.allocations[0].device_bytes, 16)
        moved, peak = transition_resource(initial, action, backend="cupy", memory_limit=19, host_memory_limit=20)
        self.assertEqual((peak.host_bytes, peak.device_bytes), (20, 19))
        self.assertEqual(moved.allocations[0].device_bytes, 0)
        profile, state, blocks, preps = trajectory()
        with self.assertRaises(MemoryError):
            audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000, disk_limit_bytes=15)
        plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
        infeasible = replace(plan, disk_limit_bytes=15, feasible=False)
        self.assertFalse(infeasible.with_reserves(reserved_host_bytes=1).feasible)

    def test_io_cost_fields_are_not_part_of_the_model(self):
        profile = CostProfile()
        self.assertFalse(hasattr(profile, "read_call_seconds"))
        self.assertFalse(hasattr(profile, "resource_seconds"))
        with self.assertRaises(TypeError):
            CostProfile(read_call_seconds=2)

    def test_no_underreported_moves_borrowed_moves_or_unrecoverable_discard(self):
        profile = CostProfile()
        initial = ResourceSnapshot((ResourceAllocation("tensor", "a", host_bytes=16),))
        forged = Action("spill_disk", "a", seconds=1.0,
                        resource=ResourceTransition("tensor", "host", "disk", 1))
        with self.assertRaises(ValueError):
            transition_resource(initial, forged, backend="numpy", memory_limit=100)
        discard = Action("discard", "a", resource=ResourceTransition("tensor", "host", "absent", 16))
        with self.assertRaises(ValueError):
            transition_resource(initial, discard, future_needed={("tensor", "a")},
                                backend="numpy", memory_limit=100)
        raw = ResourceSnapshot((ResourceAllocation("block", "raw", host_bytes=64),), (("view", ("raw",)),))
        self.assertEqual(resource_point(raw, 0, "borrow").host_bytes, 64)
        with self.assertRaises(ValueError):
            transition_resource(raw, Action("spill_disk", "raw", seconds=1.0,
                resource=ResourceTransition("block", "host", "disk", 64)), backend="numpy", memory_limit=100)
        discarded, peak = transition_resource(raw,
            Action("discard", "view", resource=ResourceTransition("tensor", "host", "absent", 0)),
            backend="numpy", memory_limit=100)
        self.assertEqual(discarded.borrowed, ())
        self.assertEqual(peak.host_bytes, 64)
        self.assertEqual(discarded.allocations, raw.allocations)
        with self.assertRaises(ValueError):
            ResourceSnapshot(raw.allocations + (ResourceAllocation("tensor", "view", host_bytes=64),), raw.borrowed)

    def test_original_raw_block_is_read_once_across_temporary_restore(self):
        profile, state, blocks, preps = trajectory()
        extra = Action("read_block", "raw")
        bad = replace(state, actions=state.actions + (extra,), seconds=state.seconds + extra.seconds)
        with self.assertRaises(ValueError):
            audit_resource_trajectory(bad, blocks, preps, profile=profile, memory_limit=1000)
        plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
        self.assertEqual(sum(e.kind == "read" for e in plan.events), 1)
        self.assertFalse(hasattr(plan, "read_seconds"))
        self.assertFalse(hasattr(plan, "resource_seconds"))

    def test_execution_archive_guards_precede_io(self):
        profile, state, blocks, preps = trajectory()
        plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
        class Trap:
            def read_block(self, *args): raise AssertionError("IO")
            def materialize(self, *args): raise AssertionError("IO")
            def close_block(self, *args): raise AssertionError("IO")
            def close_all(self): raise AssertionError("IO")
        with self.assertRaises(NotImplementedError): replay_scene_storage(plan, Trap())
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "plan.json"
            with self.assertRaises(NotImplementedError): save_storage_schedule(target, (plan,), shapes={"A": [2]}, hardware={"target": "cpu"})
            self.assertFalse(target.exists())
        profile, old, blocks, preps = fixture()
        legacy = plan_scene_storage(old, blocks, preps, profile=profile, memory_limit=1000)
        unknown = replace(legacy, events=legacy.events + (StorageEvent(0, "typo", "raw"),))
        with self.assertRaises(ValueError): replay_scene_storage(unknown, Trap())
        bad = replace(old, actions=old.actions + (Action("typo", "A"),))
        with self.assertRaises(ValueError): validate_executable_resources(bad)

    def test_legacy_archive_defaults_survive_new_fields(self):
        profile, state, blocks, preps = fixture()
        plan = plan_scene_storage(state, blocks, preps, profile=profile, memory_limit=1000)
        payload = encode_plan_value(plan)
        def strip(value):
            if isinstance(value, dict):
                fields = value.get("value")
                name = value.get("type", "").split(".")[-1]
                remove = {"Action": ("resource",), "MemoryPoint": ("temporary_disk_bytes",),
                          "SceneStoragePlan": ("peak_disk_bytes", "disk_limit_bytes", "explicit_resources")}.get(name, ())
                if isinstance(fields, dict):
                    for key in remove: fields.pop(key, None)
                for item in value.values(): strip(item)
            elif isinstance(value, list):
                for item in value: strip(item)
        strip(payload)
        restored = decode_plan_value(payload)
        self.assertEqual(restored.seconds, plan.seconds)
        self.assertEqual(restored.peak_disk_bytes, 0)


    def test_gpu_raw_move_requires_restore_before_materialization(self):
        profile, cpu, _, _ = fixture()
        problem = replace(cpu.problem, backend="cupy")
        logical = ContractionState.initial(problem).load("A", logical_cost_profile(profile), 1000)
        logical = logical.contract(0, (0,), logical_cost_profile(profile), 1000)
        blocks = {"raw": StorageBlock("dataset", "all", 16, device_bytes=16)}
        preps = {"A": SourcePreparation(("raw",))}
        prefix = (Action("read_block", "raw"),
                  Action("spill_host", "raw", resource=ResourceTransition("block", "device", "host", 16)))
        bad_actions = prefix + logical.actions + (Action("close_block", "raw"),)
        bad = replace(logical, actions=bad_actions, seconds=sum(a.seconds for a in bad_actions))
        with self.assertRaisesRegex(ValueError, "available raw"):
            audit_resource_trajectory(bad, blocks, preps, profile=profile, memory_limit=1000)
        restore = Action("restore_device", "raw", resource=ResourceTransition("block", "host", "device", 16))
        good_actions = prefix + (restore,) + logical.actions + (Action("close_block", "raw"),)
        good = replace(logical, actions=good_actions, seconds=sum(a.seconds for a in good_actions))
        plan = audit_resource_trajectory(good, blocks, preps, profile=profile, memory_limit=1000)
        self.assertEqual(plan.peak_host_bytes, 16)
        self.assertEqual(sum(e.kind == "read" for e in plan.events), 1)

    def test_close_cannot_remove_missing_source_recovery_route(self):
        profile, logical, blocks, preps = fixture()
        actions = (Action("read_block", "raw"),
                   Action("close_block", "raw")) + logical.actions
        bad = replace(logical, actions=actions, seconds=sum(a.seconds for a in actions))
        with self.assertRaisesRegex(ValueError, "recovery dependency"):
            audit_resource_trajectory(bad, blocks, preps, profile=profile, memory_limit=1000)

    def test_no_spill_trajectory_can_be_audited_again(self):
        profile, state, blocks, preps = trajectory(False)
        plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
        repeated = audit_resource_trajectory(plan.state, blocks, preps, profile=profile, memory_limit=1000)
        self.assertEqual(repeated.memory, plan.memory)
        self.assertEqual(repeated.seconds, plan.seconds)


    def test_compute_cost_report_keeps_physical_read_and_resource_metadata(self):
        from lqcdweave.contraction.performance.api import predict_plan_cost
        profile, state, blocks, preps = trajectory()
        plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
        report = predict_plan_cost((plan,), profile, workload={}, provenance={})
        expected = sum(a.seconds for a in state.actions if a.kind in ("load", "contract"))
        self.assertAlmostEqual(report['total_seconds'], expected)
        self.assertAlmostEqual(report['total_seconds'], plan.seconds)
        self.assertEqual(report['physical_reads'], 1)
        self.assertEqual(report['read_bytes'], 64)
        self.assertEqual(report['temporary_read_bytes'], 16)
        self.assertEqual(report['temporary_write_bytes'], 16)
        self.assertEqual(report['peak_disk_bytes'], 16)
        self.assertNotIn('resource_seconds', report)


    def test_cached_join_reuses_resident_output_after_original_inputs_released(self):
        profile, state, blocks, preps = self.cached_join_trajectory(False)
        plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=10000)
        reused = [a for a in plan.state.actions if a.kind == 'baryon_join' and a.reused]
        self.assertEqual(len(reused), 1)
        self.assertEqual(reused[0].seconds, 0)
        self.assertEqual(plan.state.loaded, frozenset(plan.state.results))
        # Cache substitution needs neither old input bytes nor operation scratch.
        hit = next(p for p in plan.memory if p.phase == 'baryon_join' and p.action_index == len(state.actions) - 2)
        self.assertEqual(hit.host_bytes, 48)
        self.assertEqual(hit.workspace_host_bytes, 0)

    def test_cached_join_cannot_reuse_spilled_output_without_restore(self):
        profile, state, blocks, preps = self.cached_join_trajectory(True)
        with self.assertRaisesRegex(ValueError, 'cached output must be restored'):
            audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=10000)

    @staticmethod
    def cached_join_trajectory(spill):
        from lqcdweave.contracts.correlator.api import BaryonJoinSpec
        specs = {'spin': TensorSpec((1, 1, 1), 'float64'),
                 'mode': TensorSpec((1, 1, 1), 'float64'),
                 **{key: TensorSpec((1, 1), 'float64') for key in ('a', 'b', 'c')}}
        term = ContractionTerm(('a', 'b', 'c'), 'ad,be,cf->')
        joins = tuple(BaryonJoinSpec(scene, 'shared', 'spin', 'mode', 'abc', 'def') for scene in (0, 1))
        problem = logical_source_problem(ContractionProblem(specs, (term, term),
            {key: SourceSpec(key, 0) for key in specs}, baryon_joins=joins))
        profile = CostProfile()
        state = ContractionState.initial(problem)
        for key in specs:
            state = state.load(key, logical_cost_profile(profile), 10000)
        state = state.contract_join(joins[0], logical_cost_profile(profile), 10000)
        for key in specs:
            state = state.release(key)
        prefix = tuple(Action('discard', a.tensor,
            resource=ResourceTransition('tensor', 'host', 'absent', 8))
            if a.kind == 'release' else a for a in state.actions)
        # Construct the independently known second algebraic join shape without
        # loading numeric inputs: only its existing output is a live allocation.
        symbolic = replace(state, loaded=state.loaded | set(specs))
        after = symbolic.contract_join(joins[1], logical_cost_profile(profile), 10000)
        cache_hit = after.actions[-1]
        assert cache_hit.reused
        actions = prefix
        if spill:
            actions += (Action('spill_disk', cache_hit.tensor,
                resource=ResourceTransition('tensor', 'host', 'disk', 8)),)
        block = StorageBlock('shared-input', 'all', 40, host_bytes=40)
        actions = (Action('read_block', 'raw'),) + actions
        actions += (cache_hit, Action('close_block', 'raw'))
        state = replace(after, loaded=state.loaded, actions=actions, seconds=sum(a.seconds for a in actions))
        return profile, state, {'raw': block}, {key: SourcePreparation(('raw',)) for key in specs}


    def test_empty_input_reduction_has_a_valid_zero_byte_owner(self):
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                profile, state, blocks, preps = self.empty_trajectory('i->', backend)
                plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
                self.assertTrue(plan.feasible)
                self.assertEqual(plan.state.catalog[plan.state.results[0]].shape, ())
                self.assertEqual(plan.peak_disk_bytes, 0)
                repeated = audit_resource_trajectory(plan.state, blocks, preps, profile=profile, memory_limit=1000)
                self.assertEqual(plan.seconds, repeated.seconds)

    def test_empty_output_remains_valid_for_final_copy_and_cleanup(self):
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                profile, state, blocks, preps = self.empty_trajectory('i->i', backend)
                plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
                self.assertEqual(plan.state.catalog[plan.state.results[0]].shape, (0,))
                self.assertEqual(plan.peak_host_bytes, 0)
                self.assertEqual(plan.peak_device_bytes, 0)
                self.assertTrue(plan.feasible)

    def test_zero_byte_raw_dependency_and_borrowed_record_are_valid(self):
        # The sparse record is metadata-only here; the dense calculation is
        # independent. Its explicit load/discard still audits raw ownership.
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                profile, state, blocks, preps = self.empty_trajectory('i->', backend)
                record = SourceSpec('record', 0, snapshot_bytes=0, read_bytes=0)
                problem = replace(state.problem, sources={**state.problem.sources, 'record': record})
                model = logical_cost_profile(profile)
                view_load = ContractionState.initial(problem).load('record', model, 1000).actions[-1]
                view_discard = Action('discard', 'record', resource=ResourceTransition(
                    'tensor', 'device' if backend == 'cupy' else 'host', 'absent', 0))
                actions = state.actions[:1] + (view_load, view_discard) + state.actions[1:]
                state = replace(state, problem=problem, actions=actions, seconds=sum(a.seconds for a in actions))
                preps = {**preps, 'record': SourcePreparation(('raw',))}
                plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
                self.assertEqual(sum(e.kind == 'read' for e in plan.events), 1)
                self.assertFalse(hasattr(plan, 'read_seconds'))
                self.assertTrue(plan.feasible)

    def test_zero_byte_values_are_not_moved_but_can_be_discarded(self):
        profile = CostProfile()
        for backend, tier in (('numpy', 'host'), ('cupy', 'device')):
            initial = ResourceSnapshot((ResourceAllocation('tensor', 'empty'),))
            discard = Action('discard', 'empty', resource=ResourceTransition('tensor', tier, 'absent', 0))
            absent, peak = transition_resource(initial, discard, backend=backend, memory_limit=1)
            self.assertEqual(absent.allocations, ())
            self.assertEqual(peak.host_bytes + peak.device_bytes, 0)
            move_kind, target = ('spill_disk', 'disk') if tier == 'host' else ('spill_host', 'host')
            move = Action(move_kind, 'empty',
                          resource=ResourceTransition('tensor', tier, target, 0))
            with self.assertRaisesRegex(ValueError, 'zero-byte owners'):
                transition_resource(initial, move, backend=backend, memory_limit=1)

    @staticmethod
    def empty_trajectory(equation, backend):
        problem = logical_source_problem(ContractionProblem(
            {'A': TensorSpec((0,), 'float64')}, (ContractionTerm(('A',), equation),),
            {'A': SourceSpec('A', 0)}, backend=backend))
        profile = CostProfile()
        state = ContractionState.initial(problem).load('A', logical_cost_profile(profile), 1000)
        state = state.contract(0, (0,), logical_cost_profile(profile), 1000)
        block = StorageBlock('empty', 'all', 0)
        actions = (Action('read_block', 'raw'),) + state.actions + (Action('close_block', 'raw'),)
        return profile, replace(state, actions=actions, seconds=sum(a.seconds for a in actions)), {'raw': block}, {'A': SourcePreparation(('raw',))}


    def test_sparse_bridge_consumes_a_valid_zero_byte_borrowed_owner(self):
        from lqcdweave.contracts.correlator.api import SparseBridgeSpec
        for backend in ('numpy', 'cupy'):
            with self.subTest(backend=backend):
                problem = logical_source_problem(ContractionProblem(
                    {key: TensorSpec((1, 1, 1), 'float64') for key in ('L', 'R')},
                    (ContractionTerm(('L', 'R'), 'spc,spc->'),),
                    {'L': SourceSpec('L', 0), 'R': SourceSpec('R', 0),
                     'record': SourceSpec('record', 0, snapshot_bytes=0)},
                    sparse_bridges=(SparseBridgeSpec('L', 'R', 'record', 1, 0, same_point=True),),
                    backend=backend))
                profile = CostProfile()
                state = ContractionState.initial(problem)
                for key in ('L', 'R', 'record'):
                    state = state.load(key, logical_cost_profile(profile), 1000)
                state = state.contract(0, (0, 1), logical_cost_profile(profile), 1000)
                actions = (Action('read_block', 'raw'),) + state.actions
                actions += (Action('discard', 'record', resource=ResourceTransition(
                    'tensor', 'device' if backend == 'cupy' else 'host', 'absent', 0)),
                    Action('close_block', 'raw'))
                state = replace(state, loaded=state.loaded - {'record'}, actions=actions, seconds=sum(a.seconds for a in actions))
                blocks = {'raw': StorageBlock('empty-coo', 'all', 0)}
                preps = {key: SourcePreparation(('raw',)) for key in problem.sources}
                plan = audit_resource_trajectory(state, blocks, preps, profile=profile, memory_limit=1000)
                self.assertTrue(plan.feasible)
                self.assertEqual(sum(a.kind == 'sparse_bridge' for a in plan.state.actions), 1)
                self.assertEqual(plan.events[-1].kind, 'release')
