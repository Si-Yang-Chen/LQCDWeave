"""Full-axis sharing, integer ownership, core recipes and independent replay."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, CoreComponent, TensorCoreSpec,
    TensorSpec, SourceSpec, SourcePreparation,
    SparseBridgeSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace, IntegerRegistry, TermRecord
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.budget import SearchBudget
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.planning.optimizers.shared_components import shared_components
from lqcdweave.contraction.execution.api import replay
from lqcdweave.contraction.planning.api import rebind_scene_path
from lqcdweave.storage.contraction_plans.api import save_storage_schedule, load_storage_schedule


def make_problem(terms, shapes, **options):
    return ContractionProblem({key: TensorSpec(shape, "complex128") for key, shape in shapes.items()},
        tuple(ContractionTerm(inputs, equation) for inputs, equation in terms),
        {key: SourceSpec(key, 0) for key in shapes}, **options)


def solve(problem, **options):
    if options.get('optimizer', 'anytime') == 'anytime':
        options.setdefault('component_workers', 1)
    return solve_storage(problem, {}, {key: SourcePreparation(()) for key in problem.sources},
                         memory_limit=10**8, **options)


class IntegerSearchTests(unittest.TestCase):
    def test_shared_row_transition_compiles_once_and_does_not_rehash_old_rows(self):
        problem = make_problem([(('A', 'B', 'C'), 'ij,jk,kl->il')] * 24,
                               {'A': (2, 3), 'B': (3, 4), 'C': (4, 5)})
        space = IntegerSpace(problem, profile=CostProfile())
        operation, _ = next(space.candidates(lambda: None))
        with patch.object(space.registry, 'pattern', wraps=space.registry.pattern) as pattern, \
             patch.object(space, '_part', wraps=space._part) as part:
            space.apply(operation, lambda: None)
        self.assertEqual(pattern.call_count, 1)
        self.assertEqual(part.call_count, 25)  # New rows and one new available result.

    def test_cached_row_hashes_equal_full_recomputation_across_nested_undo(self):
        problem = make_problem([(('A', 'B', 'C'), 'ij,jk,kl->il'),
                                (('B', 'A', 'C'), 'jk,ij,kl->li')],
                               {k: (2, 2) for k in 'ABC'})
        space = IntegerSpace(problem, profile=CostProfile())
        space.enable_shared_fingerprint()

        def verify():
            local = shared = 0
            for scene, term in enumerate(space.terms):
                local ^= space._part(('term', scene, term))
                shared ^= space._shared_term(scene, term)
            for data in space.available:
                local ^= space._part(('available', data))
                shared ^= space._shared_data(data)
            self.assertEqual(space.digest, local)
            self.assertEqual(space.shared_digest, shared)

        def visit(depth):
            verify()
            if not depth:
                return
            for operation, _ in list(space.candidates(lambda: None))[:2]:
                token = space.apply(operation, lambda: None)
                visit(depth - 1)
                space.undo(token)
                verify()
        visit(4)
        space.reset()
        verify()

    def test_renamed_free_and_summed_axes_share_one_actual_computation(self):
        problem = make_problem([(("A", "B"), "ij,ij->i"), (("A", "B"), "ab,ab->a")],
                               {"A": (2, 3), "B": (2, 3)})
        space = IntegerSpace(problem, profile=CostProfile())
        self.assertTrue(all(isinstance(term, TermRecord) for term in space.terms))
        choices = list(space.candidates(lambda: None))
        self.assertEqual(len(choices), 1)
        original = space.fingerprint()
        token = space.apply(choices[0][0], lambda: None)
        self.assertTrue(space.is_done())
        self.assertEqual(space.terms[0].factors, space.terms[1].factors)
        space.undo(token)
        self.assertEqual(space.fingerprint(), original)
        result = solve(problem)
        compute = [a for a in result.state.actions if a.kind == "contract"]
        self.assertEqual([a.reused for a in compute], [False, True])
        arrays = {"A": np.arange(6).reshape(2, 3) + 1j, "B": np.arange(6).reshape(2, 3) - 2j}
        values = replay(result.state, lambda key, source: arrays[key])
        expected = np.einsum("ij,ij->i", arrays["A"], arrays["B"])
        for value in values:
            np.testing.assert_allclose(value, expected)

    def test_retained_axis_alignment_is_part_of_computation_identity(self):
        problem = make_problem([(("A", "B"), "ij,ij->i"), (("A", "B"), "ij,kj->ik")],
                               {"A": (2, 3), "B": (2, 3)})
        space = IntegerSpace(problem, profile=CostProfile())
        self.assertEqual(len(list(space.candidates(lambda: None))), 2)
        # Initial operations have different retained-axis alignment.
        self.assertEqual(shared_components(problem), ((0,), (1,)))
        result = solve(problem)
        arrays = {"A": np.arange(6).reshape(2, 3) + 1j, "B": np.arange(6).reshape(2, 3) - 2j}
        values = replay(result.state, lambda key, source: arrays[key])
        np.testing.assert_allclose(values[0], np.einsum("ij,ij->i", arrays["A"], arrays["B"]))
        np.testing.assert_allclose(values[1], np.einsum("ij,kj->ik", arrays["A"], arrays["B"]))

    def test_debug_final_replay_rejects_a_forced_computation_hash_collision(self):
        problem = make_problem([(("A", "B"), "ij,ij->i"), (("A", "B"), "ij,ji->i")],
                               {"A": (3, 3), "B": (3, 3)})
        with patch("lqcdweave.contraction.planning.canonical_operations.result_name", return_value="node:canonical:collision"):
            with self.assertRaisesRegex(RuntimeError, "different full computations"):
                solve(problem, debug=True)

    def test_join_workspace_prediction_does_not_include_historical_peak(self):
        from lqcdweave.contraction.planning.state import ContractionState
        from lqcdweave.contraction.planning.canonical_operations import describe
        from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem
        problem = sparse_spin_problem()
        state = ContractionState.initial(problem)
        state = replace(state, loaded=frozenset(state.catalog))
        first = describe(state, 0, join=problem.baryon_joins[0], profile=CostProfile())
        inflated = replace(state, peak_bytes=10**12, peak_host_bytes=10**12)
        other = describe(inflated, 0, join=problem.baryon_joins[0], profile=CostProfile())
        self.assertEqual(first[:3], other[:3])

    def test_diagonal_and_output_permutation_have_correct_values(self):
        problem = make_problem([(("A",), "ii->i"), (("A",), "ij->ji")], {"A": (3, 3)})
        arrays = {"A": np.arange(9, dtype=np.complex128).reshape(3, 3)}
        values = replay(solve(problem).state, lambda key, source: arrays[key])
        np.testing.assert_array_equal(values[0], arrays["A"].diagonal())
        np.testing.assert_array_equal(values[1], arrays["A"].T)

    def test_branch_registration_does_not_grant_cached_availability(self):
        problem = make_problem([(("A", "B"), "i,i->")], {"A": (2,), "B": (2,)})
        registry = IntegerRegistry(problem)
        left, right = (IntegerSpace(problem, profile=CostProfile(), registry=registry) for _ in range(2))
        op = next(left.candidates(lambda: None))[0]
        left.apply(op, lambda: None)
        self.assertIn(op.result, left.available)
        self.assertNotIn(op.result, right.available)
        self.assertFalse(next(right.candidates(lambda: None))[0].reused)

    def test_apply_revalidates_registered_identity_only_in_debug(self):
        from lqcdweave.contraction.planning.optimizers.integer_space import IntegerOperation
        problem = make_problem([(("A", "B"), "ij,jk->ik")], {"A": (2, 3), "B": (3, 4)})
        fast = IntegerSpace(problem, profile=CostProfile())
        reference = IntegerSpace(problem, profile=CostProfile(), debug=True)
        operation, _ = next(fast.candidates(lambda: None))
        next(reference.candidates(lambda: None))
        self.assertFalse(operation.reused)
        # Producers own the exact operation; only debug re-derives its cost.
        stale = replace(operation, seconds=operation.seconds + 1.0)
        fast.apply(stale, lambda: None)
        with self.assertRaisesRegex(ValueError, "changed its registered cost"):
            reference.apply(stale, lambda: None)

    def test_intern_cost_consistency_is_debug_only(self):
        from lqcdweave.contraction.planning.optimizers.choices import OperationPool
        class Engine:
            def __init__(self, debug):
                self.operations, self.operation_costs, self.debug = OperationPool(), [], debug
        for debug, raises in ((False, False), (True, True)):
            engine = Engine(debug)
            self.assertEqual(FingerprintDFS._intern(engine, "op", 1.0), 0)
            if raises:
                with self.assertRaisesRegex(ValueError, "fixed incremental cost"):
                    FingerprintDFS._intern(engine, "op", 2.0)
            else:
                self.assertEqual(FingerprintDFS._intern(engine, "op", 2.0), 0)
                self.assertEqual(engine.operation_costs, [1.0])

    def test_only_changed_rows_are_rebuilt(self):
        problem = make_problem([(("A", "B"), "i,i->"), (("C", "D"), "i,i->")],
                               {key: (2,) for key in "ABCD"})
        space = IntegerSpace(problem, profile=CostProfile())
        op = next(space.candidates(lambda: None))[0]
        count = space.row_rebuilds
        token = space.apply(op, lambda: None)
        list(space.candidates(lambda: None))
        self.assertEqual(space.row_rebuilds, count)
        # The changed row is now complete and requires no algebra rebuilding.
        space.undo(token)
        list(space.candidates(lambda: None))
        self.assertEqual(space.row_rebuilds, count + 1)

    def test_prediction_snapshot_is_shared_and_invalidated_on_apply_undo(self):
        problem = make_problem([(("A", "B"), "ij,jk->ik"),
                                (("C", "D"), "ij,jk->ik")],
                               {"A": (2, 3), "B": (3, 5), "C": (2, 3), "D": (3, 5)})
        space = IntegerSpace(problem, profile=CostProfile())
        left, right = space._state(0), space._state(1)
        self.assertIs(left.catalog, right.catalog)
        self.assertIs(left.loaded, right.loaded)
        op = next(space.candidates(lambda: None))[0]
        name = space.registry.data[op.result].name
        token = space.apply(op, lambda: None)
        changed = space._state(1)
        self.assertIn(name, changed.catalog)
        self.assertNotIn(name, left.catalog)
        space.undo(token)
        restored = space._state(0)
        self.assertNotIn(name, restored.catalog)
        self.assertIn(name, changed.catalog)
        self.assertEqual(restored.pending, left.pending)
        space.release()
        self.assertIsNone(space._prediction_catalog)
        self.assertIsNone(space._prediction_loaded)

    def test_dense_pair_filter_matches_full_description_domain(self):
        equations = ("ij,kl->ijkl", "ij,ik->ijk", "ij,ij->ij",
                     "ij,jk,kl->il", "ii,j->j", ",i->i", "ij,jk,ij->ik")
        for equation in equations:
            with self.subTest(equation=equation):
                inputs = equation.split("->")[0].split(",")
                names = tuple(f"T{i}" for i in range(len(inputs)))
                problem = make_problem([(names, equation)],
                    {name: (2,) * len(labels) for name, labels in zip(names, inputs)})
                filtered = IntegerSpace(problem, profile=CostProfile())
                ordinary = IntegerSpace(problem, profile=CostProfile())
                def allow_all(pattern_id, checkpoint):
                    count = len(ordinary.registry.patterns[pattern_id].inputs)
                    return (1 << (count * (count - 1) // 2)) - 1
                with patch.object(ordinary.registry, "dense_pair_mask", side_effect=allow_all):
                    reference = list(ordinary.candidates(lambda: None))
                self.assertEqual(list(filtered.candidates(lambda: None)), reference)
                self.assertEqual(filtered.registry.data, ordinary.registry.data)
        # A suspended computation must not publish a partial legality table.
        registry = IntegerRegistry(problem)
        identity = registry.pattern(((0, 1), (1, 2), (2, 3)), (0, 3))
        def stop():
            raise RuntimeError("stop")
        with self.assertRaisesRegex(RuntimeError, "stop"):
            registry.dense_pair_mask(identity, stop)
        self.assertNotIn(identity, registry.dense_pair_masks)

    def test_components_get_first_solutions_before_improvement(self):
        problem = make_problem([(("A", "B", "C"), "i,i,i->"), (("D", "E", "F"), "i,i,i->")],
                               {key: (2,) for key in "ABCDEF"})
        calls = []
        original = FingerprintDFS.advance
        def advance(engine, **kwargs):
            calls.append((tuple(engine.space.problem.specs), kwargs.get("first_solution", False)))
            return original(engine, **kwargs)
        with patch.object(FingerprintDFS, "advance", advance):
            result = solve(problem)
        self.assertEqual([first for _, first in calls[:4]], [True, True, False, False])
        self.assertEqual(result.statistics["component_count"], 2)
        self.assertIsNotNone(result.first_solution_seconds)

    def test_global_memory_sampling_does_not_rescan_for_each_component(self):
        terms = [((f"A{i}", f"B{i}"), "i,i->") for i in range(30)]
        problem = make_problem(terms, {key: (2,) for inputs, _ in terms for key in inputs})
        sampled_components = []
        original = SearchBudget.sample_memory
        def sample(budget, roots, **options):
            sampled_components.append(len(roots[0]))
            return original(budget, roots, **options)
        with patch.object(SearchBudget, "sample_memory", sample):
            result = solve(problem, root_width=1, branch_width=1, search_memory_limit=1 << 28)
        self.assertEqual(result.statistics["component_count"], 30)
        self.assertLessEqual(len(sampled_components), 4)
        self.assertEqual(sampled_components[-1], 30)
        self.assertGreater(result.statistics["sampled_peak_search_bytes"], 0)

    def test_complex_core_executes_sparse_components_without_dense_rank_six_input(self):
        entries1 = ((0, 1, 2, 1 + 2j), (3, 2, 1, -.25j))
        entries2 = ((1, 0, 3, -.5),)
        core = TensorCoreSpec("baryon", ("spin0", "spin1", "spin2", "laplace_vector0", "laplace_vector1", "laplace_vector2"),
            TensorSpec((4, 4, 4, 2, 2, 2), "complex64"),
            (CoreComponent(("S1", "E1"), 2 + .5j, entries1), CoreComponent(("S2", "E2"), -1, entries2)))
        shapes = {"S1": (4, 4, 4), "S2": (4, 4, 4), "E1": (2, 2, 2), "E2": (2, 2, 2),
                  "P": (4, 2), "Q": (4, 2), "R": (4, 2)}
        join = BaryonJoinSpec(0, "complex", "S1", "E1", "abc", "xyz", sparse_spin=entries1, core=core)
        problem = make_problem([(("P", "Q", "R"), "ax,by,cz->")], shapes, baryon_joins=(join,))
        space = IntegerSpace(problem, profile=CostProfile())
        self.assertEqual(sum(space.registry.data[i].core is not None for i in space.terms[0].factors), 1)
        result = solve(problem)
        actions = [a for a in result.state.actions if a.kind == "baryon_join"]
        self.assertEqual(len(actions), 1)
        self.assertIsNotNone(actions[0].core)
        self.assertEqual(result.state.catalog[actions[0].tensor].dtype, np.dtype("complex128"))
        self.assertTrue(all(len(spec.shape) != 6 for spec in result.state.catalog.values()))
        rng = np.random.default_rng(501)
        arrays = {key: rng.normal(size=shape) + 1j * rng.normal(size=shape) for key, shape in shapes.items()}
        for key, entries in (("S1", entries1), ("S2", entries2)):
            arrays[key] = np.zeros(shapes[key], dtype=np.complex128)
            for a, b, c, coefficient in entries:
                arrays[key][a, b, c] = coefficient
        expected = sum(c.coefficient * np.einsum("abc,xyz,ax,by,cz->", arrays[c.inputs[0]], arrays[c.inputs[1]],
                                                arrays["P"], arrays["Q"], arrays["R"]) for c in core.components)
        np.testing.assert_allclose(replay(result.state, lambda key, source: arrays[key])[0], expected, rtol=1e-12, atol=1e-12)
        shapes = {key: list(spec.shape) for key, spec in problem.specs.items()}
        hardware = {"backend": "numpy", "memory": 10**8}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "core.gz"
            save_storage_schedule(path, (result.plan,), shapes=shapes, hardware=hardware)
            restored, _ = load_storage_schedule(path, shapes=shapes, hardware=hardware,
                expected=(problem,), blocks=result.plan.blocks, preparations=result.plan.preparations)
        self.assertEqual(restored[0].state.actions, result.state.actions)
        np.testing.assert_allclose(replay(restored[0].state, lambda key, source: arrays[key])[0], expected, rtol=1e-12, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "fused logical cores"):
            solve(problem, optimizer="exact")

    def test_canonical_path_rebind_keeps_shared_computations(self):
        problem = make_problem([(("A", "B"), "ij,ij->i"), (("A", "B"), "ab,ab->a")],
                               {"A": (2, 3), "B": (2, 3)})
        names = {key: "renamed:" + key for key in problem.sources}
        other = replace(problem, specs={names[k]: v for k, v in problem.specs.items()},
            sources={names[k]: replace(v, region=names[k]) for k, v in problem.sources.items()},
            terms=tuple(replace(t, inputs=tuple(names[k] for k in t.inputs)) for t in problem.terms))
        rebound = rebind_scene_path(solve(problem).state, other, names, profile=CostProfile(), memory_limit=10**8)
        self.assertTrue(rebound.done)
        self.assertEqual(rebound.results[0], rebound.results[1])
        arrays = {names[k]: np.arange(6).reshape(2, 3) + (1j if k == "A" else -2j) for k in names}
        expected = np.einsum("ij,ij->i", arrays[names["A"]], arrays[names["B"]])
        for value in replay(rebound, lambda key, source: arrays[key]):
            np.testing.assert_allclose(value, expected)

    def test_p2p_core_keeps_color_matrices_parts_and_asymmetric_point_axes(self):
        shapes = {"L": (2, 4, 2, 3), "R": (4, 3, 3, 5)}
        bridge = SparseBridgeSpec("L", "R", "J", 2, 2, left_axes=(1, 2, 3),
                                  right_axes=(0, 1, 2), used_np_right=3, part_count=2)
        problem = make_problem([(("L", "R"), "iapc,bqdj->ij"),
                                (("L", "R"), "tapc,bqdu->tu")], shapes)
        problem = replace(problem, sources=dict(problem.sources, J=SourceSpec("J", 0)), sparse_bridges=(bridge,))
        space = IntegerSpace(problem, profile=CostProfile())
        self.assertEqual(len(list(space.candidates(lambda: None))), 1)
        self.assertEqual(space.registry.data[space.terms[0].factors[-1]].core.spec.shape, (4, 2, 3, 4, 3, 3))
        # Source-only records have no TensorSpec; prediction must agree with
        # the uncached resident accounting, without charging unloaded sources.
        predicted = space._state(0)
        self.assertEqual(predicted._resident_totals, replace(predicted)._resident_totals)
        rng = np.random.default_rng(141)
        arrays = {key: rng.normal(size=shape) + 1j * rng.normal(size=shape) for key, shape in shapes.items()}
        gamma = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        values = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
        record = dict(type="sparse", indices=np.array(((0, 2), (1, 0)), dtype=np.int64), values=values)
        parts = ((gamma, 1, record), (gamma.T, -.3j, record))
        arrays["J"] = parts
        dense = np.zeros((2, 3, 3, 3), dtype=np.complex128)
        dense[0, 2], dense[1, 0] = values
        expected = sum(weight * np.einsum("iapc,ab,pqcd,bqdj->ij", arrays["L"], spin, dense, arrays["R"])
                       for spin, weight, _ in parts)
        result = solve(problem)
        for value in replay(result.state, lambda key, source: arrays[key]):
            np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)

    def test_two_p2p_cores_close_cycle_and_indexed_join(self):
        from test_sparse_cycle_partitions import _SparseCyclePartitionFixture, _dense_oracle
        rng = np.random.default_rng(611)
        arrays = {key: rng.normal(size=(4, 4, 3, 3, 3, 3)) + 1j * rng.normal(size=(4, 4, 3, 3, 3, 3))
                  for key in ("p0", "p1")}
        gammas = [rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) for _ in range(2)]
        records = [dict(type="sparse", indices=np.array([(i, j) for i in range(3) for j in range(3)]),
            values=rng.normal(size=(9, 3, 3)) + 1j * rng.normal(size=(9, 3, 3))) for _ in range(2)]
        arrays.update({f"r{i}": ((gammas[i], 1, records[i]),) for i in range(2)})
        for partition in ((0, 1, 2, 3), (0, 1, 2, 0)):
            with self.subTest(partition=partition):
                state, _, _, _ = _SparseCyclePartitionFixture()._planned_cycle(partition)
                workspace = IntegerSpace(state.problem, profile=CostProfile())
                self.assertEqual(sum(workspace.registry.data[i].core is not None for i in workspace.terms[0].factors), 2)
                result = solve(state.problem)
                expected, _ = _dense_oracle(partition, arrays["p0"], arrays["p1"], *gammas, *records, 3)
                np.testing.assert_allclose(replay(result.state, lambda key, source: arrays[key])[0], expected,
                                           rtol=1e-11, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
