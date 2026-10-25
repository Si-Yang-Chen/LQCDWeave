"""Independent logical branches run in processes and merge through the same IR."""
from dataclasses import dataclass, replace
import multiprocessing
import os
import pickle
import time
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionProblem, ContractionTerm, SourceSpec, SourcePreparation,
    StorageBlock, TensorSpec,
)
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.planning.path_planning import PathSearchExhausted
from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.planning.optimizers.parallel_components import (
    _merge_routes, _pack, _search_component, _worker_count, _weighted_shares,
)
from lqcdweave.contraction.planning.optimizers.shared_components import component_problem, shared_components
from lqcdweave.contraction.planning.optimizers.budget import retained_size, SearchBudget, BudgetExhausted


@dataclass(frozen=True)
class GatedProfile(CostProfile):
    barrier: object = None
    arrivals: object = None
    owner: int = 0

    def contract_seconds(self, *args, **kwargs):
        if os.getpid() != self.owner and not getattr(self, "_arrived", False):
            object.__setattr__(self, "_arrived", True)
            self.arrivals.append(os.getpid())
            # Both workers must price a candidate before either may proceed.
            # This is a concurrency check, not a timing benchmark.
            self.barrier.wait(timeout=20)
        return super().contract_seconds(*args, **kwargs)


@dataclass(frozen=True)
class FailingProfile(CostProfile):
    def contract_seconds(self, *args, **kwargs):
        if multiprocessing.current_process().name != "MainProcess":
            raise RuntimeError("worker cost model failed")
        return super().contract_seconds(*args, **kwargs)


class Loader:
    def __init__(self):
        self.reads = []

    def read_block(self, key, block):
        self.reads.append(key)
        return np.arange(4, dtype=np.float64) + ord(key) - ord("A")

    def materialize(self, key, source, blocks):
        return blocks[key]


def fixture():
    keys = ("A", "B", "C", "D")
    problem = ContractionProblem(
        {key: TensorSpec((4,), "float64") for key in keys},
        (ContractionTerm(("A", "B"), "i,i->"),
         ContractionTerm(("C", "D"), "i,i->"),
         ContractionTerm(("A", "B"), "i,i->")),
        {key: SourceSpec(key, 32) for key in keys},
    )
    blocks = {key: StorageBlock(key, key, 32, 32) for key in keys}
    preparations = {key: SourcePreparation((key,)) for key in keys}
    return problem, blocks, preparations


class ParallelComponentTests(unittest.TestCase):
    def test_transitively_shared_terms_keep_their_own_wide_boundaries_in_owner_matching(self):
        from lqcdweave.contraction.execution.api import replay
        for dtype in ('float32', 'complex64', 'float64', 'complex128'):
            term = ContractionTerm(('A', 'B', 'C'), 'ak,ak,a->')
            a, k = term.subscripts.inputs[0]
            wide = replace(term, sum_barriers=(((k,), (a,)),))
            specs = {'A': TensorSpec((2, 3), dtype), 'B': TensorSpec((2, 3), dtype),
                     'C': TensorSpec((2,), dtype)}
            source = ContractionProblem(specs, (term, wide),
                {key: SourceSpec(key, 0, read_bytes=0) for key in specs})
            # The shared A*C operation legitimately connects both Terms.
            # Their final pair has identical axes but different precision.
            self.assertEqual(shared_components(source), ((0, 1),))
            rng = np.random.default_rng(261051)
            complex_dtype = np.dtype(dtype).kind == 'c'
            arrays = {key: (rng.normal(size=spec.shape) +
                      (1j*rng.normal(size=spec.shape) if complex_dtype else 0)).astype(dtype)
                      for key, spec in specs.items()}
            expected = np.einsum('ak,ak,a->', *(array.astype('complex128' if complex_dtype else 'float64')
                                                for array in arrays.values()))
            for workers in (1, 2):
                with self.subTest(dtype=dtype, workers=workers):
                    selected = solve_storage(source, {}, {key: SourcePreparation(()) for key in specs},
                        memory_limit=1 << 20, host_memory_limit=1 << 20,
                        component_workers=workers, root_width=1, branch_width=1, include_ties=False,
                        deadline_seconds=10, max_states=1000)
                    self.assertTrue(selected.plan.feasible)
                    actions = [action for action in selected.state.actions if action.kind == 'contract']
                    self.assertEqual(any(action.wide_accumulation and action.term_index == 1 for action in actions),
                                     dtype in ('float32', 'complex64'))
                    self.assertFalse(any(action.wide_accumulation and action.term_index == 0 for action in actions))
                    if workers == 2:
                        self.assertTrue(all(record['pid'] != os.getpid()
                            for group in selected.statistics['root_branch_statistics'] for record in group))
                    for value in replay(selected.state, lambda key, _source: arrays[key]):
                        limit = 5e-5*max(abs(value), abs(expected), 1e-12)+2e-8
                        self.assertLessEqual(abs(value-expected), limit)

    @patch('lqcdweave.contraction.planning.optimizers.shared_components._PARALLEL_CLASSIFICATION_MIN_TERMS', 256)
    def test_parallel_classification_preserves_cross_chunk_shared_classes(self):
        problem, _, _ = fixture()
        problem = replace(problem, terms=problem.terms * 200)
        expected = shared_components(problem)
        budget = SearchBudget(deadline_seconds=30, max_transitions=10000, search_memory_limit=8 * 2**20)
        statistics = {}
        actual = shared_components(problem, budget.check, component_workers=2,
                                   budget=budget, statistics=statistics)
        self.assertEqual(actual, expected)
        self.assertEqual(statistics['classification_workers'], 2)
        self.assertTrue(statistics['classification_worker_pids'])
        self.assertLessEqual(budget.transitions, 10000)
        with self.assertRaises(BudgetExhausted):
            shared_components(problem, component_workers=2,
                              budget=SearchBudget(deadline_seconds=0))

    def test_worker_ceiling_and_weighted_quotas_respect_owner_resources(self):
        with patch('os.sched_getaffinity', return_value=set(range(16)), create=True), \
             patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '6'}):
            self.assertEqual(_worker_count(14, 30, 100, None), 6)
            self.assertEqual(_worker_count(14, 30, 100, 700), 3)
            self.assertEqual(_worker_count(14, 2, 100, None), 2)
        for total in (0, 2, 3, 1001):
            shares = _weighted_shares(total, (1, 10, 100))
            self.assertEqual(sum(shares), total)
            self.assertTrue(all(v >= 0 for v in shares))
            self.assertGreaterEqual(shares[-1], shares[0])

    @patch('lqcdweave.contraction.planning.optimizers.shared_components._PARALLEL_CLASSIFICATION_MIN_TERMS', 256)
    def test_parallel_classification_preserves_scoped_baryon_joins(self):
        from correlator.planning.test_candidate_metadata import join_fixture
        original = join_fixture().problem
        n = len(original.terms)
        problem = replace(original, terms=original.terms * 100,
                          baryon_joins=tuple(replace(j, scene=j.scene + repeat * n)
                                             for repeat in range(100) for j in original.baryon_joins))
        expected = shared_components(problem)
        actual = shared_components(problem, component_workers=2,
                                   budget=SearchBudget(deadline_seconds=30, search_memory_limit=64 * 2**20))
        self.assertEqual(actual, expected)

    def test_frozen_routes_assemble_without_repricing_and_match_debug_replay(self):
        problem, _, _ = fixture()
        # Deliberately split equal terms to check cross-component reuse too.
        groups = ((0,), (1,), (2,))
        options = dict(root_width=1, branch_width=1, debug=False,
                       scorer='incremental_time', profile_scoring=False,
                       budget=dict(max_states=1000, max_transitions=None, search_memory_limit=None))
        reports = [pickle.loads(_search_component(_pack((component_problem(problem, scenes),
                    CostProfile(), options, time.monotonic() + 10)))) for scenes in groups]
        reference, spans, workspaces = _merge_routes(problem, reports, groups, CostProfile(), debug=True)
        with patch('lqcdweave.contraction.planning.optimizers.parallel_components.describe',
                   side_effect=AssertionError('owner repriced frozen worker actions')):
            actual, actual_spans, actual_workspaces = _merge_routes(problem, reports, groups, CostProfile())
        self.assertEqual(actual.actions, reference.actions)
        self.assertEqual(actual.results, reference.results)
        self.assertEqual(actual.seconds, reference.seconds)
        self.assertEqual((actual_spans, actual_workspaces), (spans, workspaces))
        self.assertTrue(actual.actions[-1].reused)
        self.assertEqual(actual.actions[-1].seconds, 0)

    def test_fixed_owner_metadata_is_sized_once_and_memory_remains_bounded(self):
        import lqcdweave.contraction.planning.optimizers.parallel_components as parallel
        with patch.object(parallel, 'retained_size', wraps=retained_size) as size:
            result = solve_storage(*fixture(), memory_limit=4096,
                component_workers=2, deadline_seconds=30, max_states=1000,
                search_memory_limit=2**20)
        fixed_samples = [call for call in size.call_args_list
            if hasattr(call.args[0][0], 'logical_profile')]
        self.assertEqual(len(fixed_samples), 1)
        self.assertTrue(result.plan.feasible)
        self.assertLessEqual(result.statistics['sampled_peak_search_bytes'], 2**20)

    def test_process_branches_share_computations_only_inside_their_class(self):
        with multiprocessing.get_context("spawn").Manager() as manager:
            arrivals = manager.list()
            profile = GatedProfile(barrier=manager.Barrier(2), arrivals=arrivals, owner=os.getpid())
            result = solve_storage(*fixture(), memory_limit=4096, profile=profile,
                                   component_workers=2, root_width=1, branch_width=1,
                                   deadline_seconds=30, max_states=1000)
            self.assertEqual(len(set(arrivals)), 2)
        stats = result.statistics
        self.assertEqual(stats["component_scenes"], ((0, 2), (1,)))
        self.assertEqual(stats["component_workers"], 2)
        self.assertEqual(len(set(stats["component_worker_pids"])), 2)
        self.assertLessEqual(stats["expanded_states"], 1000)
        self.assertTrue(result.plan.state.done)
        self.assertTrue(result.plan.feasible)
        self.assertTrue(stats["storage_schedule"]["required"])
        self.assertTrue(stats["storage_schedule"]["completed"])
        self.assertGreaterEqual(stats["storage_schedule"]["audited_schedules"], 1)
        loader = Loader()
        values = replay_scene_storage(result.plan, loader)
        first = np.dot(np.arange(4), np.arange(4) + 1)
        second = np.dot(np.arange(4) + 2, np.arange(4) + 3)
        np.testing.assert_allclose(values, (first, second, first))
        self.assertEqual(set(loader.reads), set("ABCD"))
        self.assertEqual(len(loader.reads), 4)

    def test_zero_state_budget_reports_no_complete_path(self):
        with self.assertRaises(PathSearchExhausted) as caught:
            solve_storage(*fixture(), memory_limit=4096, component_workers=2,
                          deadline_seconds=30, max_states=0)
        result = caught.exception.search_result
        self.assertEqual(result.status, "UNSOLVED")
        self.assertEqual(result.statistics["expanded_states"], 0)

    def test_ordered_scorer_pair_survives_process_handoff(self):
        pair = ("sharing_count", "element_reduction")
        result = solve_storage(*fixture(), memory_limit=4096, component_workers=2,
                               root_width=1, branch_width=1, scorer=list(pair),
                               deadline_seconds=30, max_states=1000)
        self.assertEqual(result.diagnostic["scorer"], pair)
        self.assertEqual(result.statistics["component_workers"], 2)
        values = replay_scene_storage(result.plan, Loader())
        first = np.dot(np.arange(4), np.arange(4) + 1)
        second = np.dot(np.arange(4) + 2, np.arange(4) + 3)
        np.testing.assert_allclose(values, (first, second, first))

    def test_exact_rejects_parallel_heuristic_option(self):
        with self.assertRaisesRegex(ValueError, "unsupported options for exact"):
            solve_storage(*fixture(), memory_limit=4096, optimizer="exact", component_workers=2)

    def test_global_transition_quota_and_memory_limit(self):
        result = solve_storage(*fixture(), memory_limit=4096, component_workers=2,
                               deadline_seconds=30, max_states=1000, max_transitions=1000,
                               search_memory_limit=2**20)
        self.assertLessEqual(result.statistics["evaluated_transitions"], 1000)
        self.assertLessEqual(result.statistics["sampled_peak_search_bytes"], 2**20)
        with self.assertRaises(PathSearchExhausted) as caught:
            solve_storage(*fixture(), memory_limit=4096, component_workers=2,
                          deadline_seconds=30, search_memory_limit=1)
        self.assertEqual(caught.exception.search_result.stop_reason, "search_memory_limit")

    def test_worker_failure_reaches_owner_and_all_workers_close(self):
        before = {child.pid for child in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "worker cost model failed"):
            solve_storage(*fixture(), memory_limit=4096, profile=FailingProfile(),
                          component_workers=2, deadline_seconds=30)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_final_owner_audit_rejects_cross_branch_hash_collision(self):
        problem, _, _ = fixture()
        groups = ((0, 2), (1,))
        options = dict(root_width=1, branch_width=1, debug=False,
                       scorer="incremental_time", profile_scoring=False,
                       budget=dict(max_states=1000, max_transitions=None,
                                   search_memory_limit=None))
        reports = [pickle.loads(_search_component(_pack((
            component_problem(problem, scenes), CostProfile(), options, time.monotonic() + 10,
        )))) for scenes in groups]
        for report in reports:
            report["actions"] = tuple(replace(action, tensor="forced-hash")
                                      for action in report["actions"])
        with patch("lqcdweave.contraction.planning.canonical_operations.result_name",
                   return_value="forced-hash"):
            with self.assertRaisesRegex(RuntimeError, "under one hash"):
                _merge_routes(problem, reports, groups, CostProfile())


if __name__ == "__main__":
    unittest.main()
