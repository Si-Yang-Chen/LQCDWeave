"""Independent full enumeration versus bounded transactional row maintenance."""
from dataclasses import replace
from itertools import combinations
import random
import unittest
from unittest.mock import patch

from correlator.planning.test_macro_steps import fixture, ContractionTerm
from correlator.planning.test_candidate_metadata import join_fixture, reference_candidates
from lqcdweave.contracts.correlator.api import (
    ContractionProblem, SourceSpec, SparseBridgeSpec, SourcePreparation, TensorSpec,
)
from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
from lqcdweave.contraction.planning.contraction_index import ContractionKey
from lqcdweave.contraction.planning.logical_index import LayoutIndex, bridge_identity
from lqcdweave.contraction.planning.optimizers.candidate_catalog import CandidateCatalog, CatalogOverflow
from lqcdweave.contraction.planning.optimizers.budget import BudgetExhausted


def reference(state):
    result, seen = [], set()
    for i, term in enumerate(state.pending):
        if term is None:
            continue
        pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
        for pair in pairs:
            try:
                layout = state._contract_layout(i, pair)
            except ValueError:
                continue
            identity = ('contract', layout[0], layout[1], bridge_identity(layout[5]), bridge_identity(layout[6]))
            if identity not in seen:
                result.append((i, pair, layout, identity))
                seen.add(identity)
    return tuple(result)


def values(entries):
    return tuple((i, pair, layout, key.values) for i, pair, layout, key in entries)


def old_bound(space, cursor, step):
    # Independent fixed-order recipe resolution, with no fee recipe cache.
    operation = step.actions[0]
    keys = {operation.tensor} if operation.reused else set(operation.operands)
    costs, opened = [], set()
    preparation = 0.0
    for key in sorted(keys):
        if space.usable(cursor, 'tensor', key):
            continue
        if key in cursor.state.loaded or key not in space.preparations:
            return None
        prep = space.preparations[key]
        for dep in sorted(set(prep.blocks)):
            if dep in opened or space.usable(cursor, 'block', dep):
                continue
            if space.owner(cursor, 'block', dep) is not None or dep in cursor.resources.opened_blocks:
                return None
            opened.add(dep)
        costs.append(space.load_actions[key].seconds)
        preparation += space.preparation_costs[key]
    costs.extend(a.seconds for a in step.actions)
    price = cursor.state.seconds + sum(costs) + cursor.preparation_seconds + preparation
    return price - space.cost(cursor)


class CandidateCatalogTests(unittest.TestCase):
    def test_retargeted_catalog_matches_scan_through_children_siblings_and_backtracking(self):
        term = ContractionTerm(('A', 'B', 'C'), 'i,i,i->')
        space = fixture((term,) * 6 + (ContractionTerm(('A', 'C'), 'i,i->i'),) * 6)
        root = space.initial_state()
        siblings = space.successors(root)
        catalog = CandidateCatalog(space.problem)
        state = space._symbolic(root)
        index = LayoutIndex(state)
        for cursor in (root, siblings[0].state, siblings[-1].state, root, siblings[1].state):
            state = space._symbolic(cursor)
            index.retarget(state)
            self.assertEqual(values(catalog.layouts(state, index, lambda: None)), reference(state))
        self.assertGreater(catalog.row_reuses, 0)

    def test_hash_collisions_and_repeated_positions_preserve_first_representative(self):
        terms = (ContractionTerm(('A', 'A', 'B'), 'i,i,i->'),
                 ContractionTerm(('A', 'B', 'C'), 'i,i,i->i')) * 4
        space = fixture(terms)
        state = space._symbolic(space.initial_state())
        with patch.object(ContractionKey, '__hash__', lambda self: 0):
            actual = CandidateCatalog(space.problem).layouts(state, LayoutIndex(state), lambda: None)
            self.assertEqual(values(actual), reference(state))
        self.assertEqual(len({key.values for _, _, _, key in actual}), len(actual))

    def test_existing_geometry_changes_update_unchanged_terms_and_compacted_catalogs(self):
        term = ContractionTerm(('A', 'B', 'C'), 'i,i,i->')
        space = fixture((term,) * 8)
        old = space._symbolic(space.initial_state())
        index, catalog = LayoutIndex(old), CandidateCatalog(space.problem)
        catalog.layouts(old, index, lambda: None)
        new_catalog = old.catalog.updated(tuple((key, TensorSpec((8,), 'float64')) for key in 'ABC'))
        new_catalog = new_catalog.updated(tuple((f'extra{i}', TensorSpec((1,), 'float64')) for i in range(70)))
        state = replace(old, catalog=new_catalog)
        self.assertEqual(state.catalog.changed_keys_since(old.catalog), frozenset('ABC'))
        index.retarget(state)
        self.assertEqual(index.changed_rows, frozenset(range(8)))
        self.assertEqual(values(catalog.layouts(state, index, lambda: None)), reference(state))
        index.retarget(old)
        self.assertEqual(values(catalog.layouts(old, index, lambda: None)), reference(old))

    def test_join_completion_with_unchanged_pending_rows_invalidates_context(self):
        space = join_fixture()
        root = space.initial_state()
        old = space._symbolic(root)
        catalog, index = CandidateCatalog(space.problem), LayoutIndex(old)
        original = catalog.layouts(old, index, lambda: None)
        marker = space.problem.baryon_joins[0]
        state = replace(old, completed_joins=frozenset({(marker.scene, marker.name)}))
        index.retarget(state)
        self.assertIn(marker.scene, index.changed_rows)
        actual = catalog.layouts(state, index, lambda: None)
        self.assertEqual(values(actual), reference(state))
        self.assertNotEqual(values(original), values(actual))
        for cursor in (root, *[e.state for e in space.successors(root)], root):
            self.assertEqual(tuple(space.candidates(cursor)), reference_candidates(space, cursor))

    def test_every_interrupted_update_leaves_last_complete_catalog_unchanged(self):
        space = fixture((ContractionTerm(('A','B','C'), 'i,i,i->'),) * 8)
        root = space.initial_state()
        old = space._symbolic(root)
        new = space._symbolic(space.successors(root)[0].state)
        baseline = CandidateCatalog(space.problem)
        index = LayoutIndex(old)
        baseline.layouts(old, index, lambda: None)
        index.retarget(new)
        calls = []
        baseline.layouts(new, index, lambda: calls.append(1))
        for stop in range(1, len(calls) + 1):
            candidate = CandidateCatalog(space.problem)
            current = LayoutIndex(old)
            expected = candidate.layouts(old, current, lambda: None)
            before = candidate.rows, candidate.groups, candidate.count, candidate.context
            current.retarget(new)
            ticks = 0
            def interrupt():
                nonlocal ticks
                ticks += 1
                if ticks == stop:
                    raise BudgetExhausted('test')
            with self.assertRaises(BudgetExhausted):
                candidate.layouts(new, current, interrupt)
            self.assertIs(candidate.rows, before[0])
            self.assertIs(candidate.groups, before[1])
            self.assertEqual((candidate.count, candidate.context), before[2:])
            self.assertEqual(candidate.layouts(old, LayoutIndex(old), lambda: None), expected)
            self.assertEqual(values(candidate.layouts(new, current, lambda: None)), reference(new))

    def test_capacity_failure_falls_back_without_removing_any_candidate(self):
        space = fixture((ContractionTerm(('A','B','C'), 'i,i,i->'),) * 8)
        root = space.initial_state()
        expected = space.successors(root)
        with patch.object(CandidateCatalog, 'capacity', 1):
            actual = space.successors(root)
        self.assertEqual(actual, expected)
        self.assertIsNone(space.candidate_catalog)
        self.assertGreater(space.catalog_fallbacks, 0)
        state = space._symbolic(root)
        catalog = CandidateCatalog(space.problem)
        with patch.object(catalog, 'capacity', 1), self.assertRaises(CatalogOverflow):
            catalog.layouts(state, LayoutIndex(state), lambda: None)
        self.assertEqual((catalog.rows, catalog.groups, catalog.count), ({}, {}, 0))

    def test_wrong_retarget_context_uses_full_row_validation(self):
        space = fixture((ContractionTerm(('A','B','C'), 'i,i,i->'),) * 8)
        root = space.initial_state()
        old = space._symbolic(root)
        catalog = CandidateCatalog(space.problem)
        catalog.layouts(old, LayoutIndex(old), lambda: None)
        child = space._symbolic(space.successors(root)[0].state)
        mismatched = LayoutIndex(child)
        mismatched.retarget(child)
        self.assertEqual(values(catalog.layouts(child, mismatched, lambda: None)), reference(child))
        with self.assertRaisesRegex(ValueError, 'another problem'):
            catalog.layouts(replace(child, problem=replace(child.problem)), mismatched, lambda: None)

    def test_scoped_sparse_bridges_survive_retarget_and_do_not_share_distinct_records(self):
        specs = {key: TensorSpec((1, 1, 1), 'float64') for key in ('L', 'R')}
        term = ContractionTerm(('L', 'R'), 'spc,spc->')
        bridges = tuple(SparseBridgeSpec('L', 'R', f'record{i}', 1, 0, same_point=True, scene=i)
                        for i in range(2))
        sources = {key: SourceSpec(key, 0) for key in specs}
        sources.update({b.record: SourceSpec(b.record, 0, snapshot_bytes=0) for b in bridges})
        problem = ContractionProblem(specs, (term, term), sources, sparse_bridges=bridges)
        space = MacroSearchSpace(problem, {}, {key: SourcePreparation(()) for key in sources}, memory_limit=1000)
        root = space.initial_state()
        siblings = space.successors(root)
        self.assertEqual(len(siblings), 2)
        catalog, index = CandidateCatalog(space.problem), LayoutIndex(space._symbolic(root))
        for cursor in (root, siblings[0].state, siblings[1].state, root):
            state = space._symbolic(cursor)
            index.retarget(state)
            self.assertEqual(values(catalog.layouts(state, index, lambda: None)), reference(state))
            self.assertEqual(tuple(space.candidates(cursor)), reference_candidates(space, cursor))

    def test_cached_fee_recipes_equal_full_resolution_across_residency_and_recovery(self):
        terms = (ContractionTerm(('A','B'), 'i,i->'), ContractionTerm(('A','C'), 'i,i->'))
        space = fixture(terms)
        root = space.initial_state()
        first = space.successors(root)[0]
        discarded = space.apply(first.state, space._move(first.state, 'discard', 'tensor', 'A'))
        cold = space.apply(first.state, space._move(first.state, 'spill_disk', 'tensor', 'A'))
        for cursor in (root, first.state, discarded, cold, root):
            for step in space._logical_candidates(cursor, lambda: None):
                self.assertEqual(space._increment_bound(cursor, step), old_bound(space, cursor, step))
        self.assertTrue(space.bound_recipes)
        self.assertLessEqual(len(space.bound_recipes), 2048)

    def test_shared_raw_block_fee_is_counted_once_and_cached_recipes_do_not_cache_readiness(self):
        base = fixture((ContractionTerm(('A','B'), 'i,i->'), ContractionTerm(('A','C'), 'i,i->')))
        space = MacroSearchSpace(base.problem, base.blocks,
            {key: SourcePreparation(('A',)) for key in 'ABC'}, memory_limit=10000)
        cursor = space.initial_state()
        actions = ()
        for _ in range(2):
            for step in space._logical_candidates(cursor, lambda: None):
                bound = space._increment_bound(cursor, step)
                self.assertEqual(bound, old_bound(space, cursor, step))
                edge = space.evaluate(cursor, step)
                self.assertEqual(bound, space.cost(edge.state) - space.cost(cursor))
            edge = space.successors(cursor)[0]
            cursor, actions = edge.state, actions + edge.actions
        plan = space.plan(cursor, actions)
        self.assertEqual(sum(a.kind == 'read_block' for a in actions), 1)
        self.assertAlmostEqual(plan.seconds, space.cost(cursor))

    def test_seeded_partial_shared_updates_keep_catalog_and_scan_agreeing(self):
        rng = random.Random(22673)
        for _ in range(25):
            terms = []
            for _ in range(12):
                size = rng.randint(2, 4)
                terms.append(ContractionTerm(tuple(rng.choices('ABC', k=size)),
                             ','.join('i' for _ in range(size)) + '->' + rng.choice(('', 'i'))))
            space = fixture(tuple(terms))
            cursor = space.initial_state()
            catalog = CandidateCatalog(space.problem)
            index = LayoutIndex(space._symbolic(cursor))
            for _ in range(8):
                state = space._symbolic(cursor)
                index.retarget(state)
                self.assertEqual(values(catalog.layouts(state, index, lambda: None)), reference(state))
                edges = space.successors(cursor)
                if not edges or cursor.state.done:
                    break
                cursor = rng.choice(edges).state
