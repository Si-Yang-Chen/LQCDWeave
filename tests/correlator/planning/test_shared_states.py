"""Cross-process completed-work reuse and strict root quotas."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import multiprocessing
import pickle
import time
import unittest

from lqcdweave.contracts.correlator.api import ContractionTerm, SourceSpec
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.session import solve_storage
from lqcdweave.contraction.planning.optimizers.budget import SearchBudget
from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
from lqcdweave.contraction.planning.optimizers.parallel_components import (
    _initialize_worker, _pack, _root_key, _search_component,
)
from lqcdweave.contraction.planning.optimizers.shared_states import SharedStates
from correlator.planning.test_parallel_roots import cycle
from correlator.planning.test_tied_width import FallbackGraph


def publish_in_process(table, key, cost):
    client = table.client(0)
    client.mark_feasible()
    client.publish(key, cost)


class SharedStateTests(unittest.TestCase):
    def test_roots_are_strict_with_either_tie_setting_and_descendants_keep_ties(self):
        for ties in (False, True):
            with self.subTest(ties=ties):
                graph = FallbackGraph({'root': tuple((f'r{i}', 0) for i in range(8)),
                    'r0': (('dead', 1), ('goal', 1))}, goals=('goal',))
                engine = FingerprintDFS(graph, root_width=4, branch_width=1, include_ties=ties)
                result = engine.run()
                self.assertEqual(len(engine.branch_ticks), 4)
                self.assertNotIn('r4', graph.applied)
                self.assertEqual(result.status, 'FEASIBLE' if ties else 'UNSOLVED')
                if ties:
                    self.assertIn('goal', graph.applied)

    def test_cross_process_table_bounds_namespace_and_eviction(self):
        context = multiprocessing.get_context('spawn')
        table = SharedStates(context, 2, slots=4)
        key = hashlib.sha256(b'state').digest()
        client = table.client(0)
        client.publish(key, 10)  # No retained solution: do not publish.
        self.assertFalse(client.dominated(key, 11))
        process = context.Process(target=publish_in_process, args=(table, key, 10))
        process.start()
        process.join(timeout=30)
        self.assertEqual(process.exitcode, 0)
        self.assertTrue(client.dominated(key, 10))
        self.assertTrue(client.dominated(key, 11))
        self.assertFalse(client.dominated(key, 9))
        self.assertEqual(client.cross_process_hits, 2)
        self.assertFalse(table.client(1).dominated(key, 20))
        space = IntegerSpace(cycle()[0], profile=CostProfile())
        space.enable_shared_fingerprint()
        operations = tuple(op for op, _ in space.search_candidates(lambda: None))
        key0 = client.key(space, operations)
        peer = table.client(1)
        key1 = peer.key(space, operations)
        self.assertNotEqual(key0, key1)
        peer.mark_feasible()
        client.publish(key0, 10)
        self.assertFalse(peer.dominated(key1, 20))
        for i in range(20):
            client.publish(hashlib.sha256(str(i).encode()).digest(), i)
        self.assertGreater(client.evictions, 0)
        self.assertFalse(client.dominated(hashlib.sha256(b'missing').digest(), 100))

    def test_parallel_root_discovery_obeys_strict_quota_with_default_ties(self):
        problem, blocks, preparations = cycle()
        keys = 'ABCDEFGH'
        labels = 'abcdefgh'
        problem = replace(problem,
            specs={key: problem.specs['A'] for key in keys},
            sources={key: SourceSpec(key, 0) for key in keys},
            terms=(ContractionTerm(tuple(keys), ','.join(
                labels[i]+labels[(i+1)%len(keys)] for i in range(len(keys)))+'->'),))
        preparations = {key: preparations['A'] for key in keys}
        for ties in (False, True):
            with self.subTest(ties=ties):
                result = solve_storage(problem, blocks, preparations, memory_limit=10**7, root_width=4,
                    branch_width=1, component_workers=4, include_ties=ties,
                    max_states=1000, deadline_seconds=30)
                stats = result.statistics
                self.assertEqual(stats['component_root_counts'], (4,))
                self.assertEqual(stats['component_roots_started'], (4,))
                self.assertEqual(stats['shared_state_reuse'], ties)
                self.assertTrue(result.plan.feasible)

    def test_registry_independent_fingerprint_handles_commuted_factors_and_undo(self):
        problem, _, _ = cycle()
        a, b = (IntegerSpace(problem, profile=CostProfile()) for _ in range(2))
        b.registry.register_data('unrelated-registration')
        b.registry.pattern(((42, 43),), ())
        for space in (a, b):
            space.enable_shared_fingerprint()
        initial = a.shared_fingerprint()
        self.assertEqual(initial, b.shared_fingerprint())
        def apply_pair(space, pair):
            choices = list(space.search_candidates(lambda: None))
            op = next(op for op, _ in choices
                      if set(space.registry.calculations[op.calculation].recipe.operands) == set(pair))
            return space.apply(op, lambda: None)
        ta = [apply_pair(a, 'AB'), apply_pair(a, 'CD')]
        tb = [apply_pair(b, 'CD'), apply_pair(b, 'AB')]
        self.assertNotEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.shared_fingerprint(), b.shared_fingerprint())
        table = SharedStates(multiprocessing.get_context('spawn'), 1, slots=4)
        ca = tuple(op for op, _ in a.search_candidates(lambda: None))
        cb = tuple(op for op, _ in b.search_candidates(lambda: None))
        self.assertEqual(table.client(0).key(a, ca), table.client(0).key(b, cb))
        # Availability is part of the identity, not just the remaining equation.
        b.shared_digest ^= b._shared_data(next(iter(b.available)))
        self.assertNotEqual(a.shared_fingerprint(), b.shared_fingerprint())
        for space, tokens in ((a, ta), (b, tb)):
            for token in reversed(tokens):
                space.undo(token)
            self.assertEqual(space.shared_fingerprint(), initial)
            space.reset()
            self.assertEqual(space.shared_fingerprint(), initial)

    def test_real_workers_reuse_closed_work_and_retain_a_valid_route(self):
        problem, _, _ = cycle()
        space = IntegerSpace(problem, profile=CostProfile())
        candidates = list(space.search_candidates(lambda: None))
        roots = {frozenset(space.registry.calculations[op.calculation].recipe.operands):
                 _root_key(space.registry, op) for op, _ in candidates}
        context = multiprocessing.get_context('spawn')
        table = SharedStates(context, 1, slots=1024)
        reports = []
        for pair in ('AB', 'CD'):
            options = dict(root_width=4, branch_width=1, include_ties=True, debug=False,
                scorer='time_per_term', profile_scoring=False, root_key=roots[frozenset(pair)],
                shared_component=0, budget=dict(max_states=10000))
            with ProcessPoolExecutor(1, mp_context=context, initializer=_initialize_worker,
                                     initargs=(context.Event(), None, table)) as pool:
                packet = _pack((problem, CostProfile(), options, time.monotonic()+30))
                reports.append(pickle.loads(pool.submit(_search_component, packet).result(timeout=40)))
        first, second = reports
        self.assertEqual(first['status'], 'FEASIBLE')
        self.assertGreater(second['statistics']['shared_state_cross_process_hits'], 0)
        # The first process retains the actual audited route; a cache entry
        # alone is never promoted to a deliverable route in the second process.
        self.assertTrue(first['actions'])
        reference = pickle.loads(_search_component(_pack((problem, CostProfile(),
            dict(options, shared_component=999), time.monotonic()+30))))
        self.assertEqual(reference['status'], 'FEASIBLE')
        best = min(r['statistics']['logical_best_seconds'] for r in reports if r['status']=='FEASIBLE')
        self.assertLessEqual(best, reference['statistics']['logical_best_seconds'] + 1e-14)

    def test_budget_interrupted_frames_are_not_published(self):
        class Recorder:
            def __init__(self): self.published = []
            def key(self, space, operations): return space.fingerprint()
            def dominated(self, key, cost): return False
            def publish(self, key, cost): self.published.append(key)
            def mark_feasible(self): pass
            def statistics(self): return {}
        graph = FallbackGraph({'root': (('next', 1),), 'next': (('goal', 1),)}, goals=('goal',))
        recorder = Recorder()
        engine = FingerprintDFS(graph, shared_states=recorder,
                                budget=SearchBudget(max_states=1))
        result = engine.run()
        self.assertTrue(result.limited)
        self.assertEqual(recorder.published, [])

    def test_cheaper_label_propagation_reopens_a_shared_pruned_descendant(self):
        class Cache:
            def key(self, space, operations): return space.fingerprint()
            def dominated(self, key, cost):
                return key == FallbackGraph.identity('X') and cost >= 10
            def publish(self, key, cost): pass
            def mark_feasible(self): pass
            def statistics(self): return {}
        graph = FallbackGraph({'root': (('entry', 0),), 'entry': (('A', 10), ('B', 1)), 'A': (('P', 0),),
            'B': (('P', 0),), 'P': (('X', 0),), 'X': (('goal', 1),)}, goals=('goal',))
        engine = FingerprintDFS(graph, root_width=2, branch_width=2, shared_states=Cache(),
            root_choices=((('root', 'entry', 0), 0),),
            scorer=lambda space, op, cost, check: {'A': 0, 'B': 1}.get(op[1], cost))
        result = engine.run()
        self.assertEqual(result.status, 'FEASIBLE')
        self.assertEqual(engine.best_plan, 2)
        self.assertGreater(engine.reconstructions, 0)


if __name__ == '__main__':
    unittest.main()
