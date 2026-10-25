"""Setup-prefix construction: deferred physical facts and O(classes) initial rows.

These tests pin the three optimizations of the pre-steady-state prefix:

* MacroSearchSpace derives its physical scheduling facts (initial expression,
  bridge index, read/load actions, preparation costs) on first use instead of
  during construction, while `problem` and `logical_profile` stay unchanged;
* IntegerSpace compiles repeated initial expressions once per TermRecord class
  straight from the static problem, without materializing a full
  ContractionState;
* dense classification compares cheap canonical tuples that partition the
  scenes exactly like the hashed result names they replaced.

A reference implementation of the pre-change hashed classification is kept
below so the contrast is explicit rather than asserted by construction.
"""
from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import (
    Action, BaryonJoinSpec, ContractionProblem, ContractionTerm, SourcePreparation,
    SourceSpec, SparseBridgeSpec, StorageBlock, TensorSpec,
)
from lqcdweave.contraction.planning.optimizers.budget import SearchBudget
from lqcdweave.contraction.planning.actions import _live_inputs
from lqcdweave.contraction.planning.canonical_operations import (
    _result_name, logical_result_name,
)
from lqcdweave.contraction.planning.logical_index import BridgeIndex
from lqcdweave.contraction.planning.optimizers.dense_patterns import compile_dense_choices
from lqcdweave.contraction.planning.optimizers.integer_space import (
    IntegerRegistry, IntegerSpace, TermRecord,
)
from lqcdweave.contraction.planning.optimizers.macro_space import MacroSearchSpace
from lqcdweave.contraction.planning.optimizers.shared_components import (
    shared_components, shared_operation_keys,
)
from lqcdweave.contraction.planning.state import ContractionState
from lqcdweave.contraction.planning.storage import (
    logical_cost_profile, logical_source_problem,
)
from lqcdweave.contraction.performance.api import CostProfile
from correlator.planning.test_candidate_metadata import join_fixture
from correlator.planning.test_integer_search import make_problem
from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem


def dense_reduction_problem():
    return make_problem([(("A", "B", "C"), "ij,jk,kl->il"),
                         (("A", "B", "D"), "ij,jk,kl->li")],
                        {"A": (2, 3), "B": (3, 4), "C": (4, 5), "D": (4, 5)})


def macro_fixture(problem=None):
    problem = dense_reduction_problem() if problem is None else problem
    blocks = {key: StorageBlock("dataset", key, 32, host_bytes=32)
              for key in problem.sources}
    preparations = {key: SourcePreparation((key,)) for key in problem.sources}
    return problem, blocks, preparations


def state_based_rows(problem, registry):
    """The pre-change per-scene construction, through ContractionState."""
    state = ContractionState.initial(problem)
    symbols, rows = [], []
    for term in state.pending:
        labels = tuple(dict.fromkeys(
            tuple(c for f in term.factors for c in f.labels) + term.output))
        numbers = {c: i for i, c in enumerate(labels)}
        axes = tuple(tuple(numbers[c] for c in f.labels) for f in term.factors)
        symbols.append(labels)
        rows.append(TermRecord(tuple(registry.names[f.tensor] for f in term.factors),
                               registry.pattern(axes, tuple(numbers[c] for c in term.output))))
    return tuple(symbols), tuple(rows)


def hashed_operation_keys(problem, checkpoint=lambda: None):
    """Pre-change classification: one blake2b result name per candidate."""
    state = None
    joins = {}
    for join in problem.baryon_joins:
        checkpoint()
        joins.setdefault(join.scene, []).append(join)
    bridges = BridgeIndex(problem.sparse_bridges)
    dense_keys, patterns = {}, {}
    for scene, logical in enumerate(problem.terms):
        checkpoint()
        scoped = bridges.scoped(scene)
        dense = not joins.get(scene) and not scoped
        if dense:
            row_key = logical.inputs, logical.subscripts
            if row_key not in dense_keys:
                if logical.subscripts not in patterns:
                    patterns[logical.subscripts] = compile_dense_choices(
                        logical.subscripts, checkpoint)
                keys = []
                for positions, variants in patterns[logical.subscripts]:
                    checkpoint()
                    inputs = tuple(logical.inputs[i] for i in positions)
                    if len(inputs) == 1:
                        relation = variants[0][1][0]
                    else:
                        left, right = inputs
                        reverse = left > right
                        relation = variants[reverse][1][2 if left == right else int(reverse)]
                        if reverse:
                            inputs = (right, left)
                    keys.append(_result_name(("contract", (inputs, *relation))))
                dense_keys[row_key] = tuple(keys)
            for key in dense_keys[row_key]:
                checkpoint()
                yield key, scene
            continue
        if state is None:
            state = ContractionState.initial(problem)
        term = state.pending[scene]
        positions = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
        for selected in positions:
            checkpoint()
            try:
                layout = state._contract_layout(scene, selected, scene_joins=joins,
                                                active_bridges=scoped)
                key = logical_result_name(state, scene, selected, layout=layout)
            except ValueError:
                continue
            yield key, scene
        for join in joins.get(scene, ()):
            checkpoint()
            try:
                layout = state._join_layout(join, scene_joins=joins, scoped_bridges=scoped)
                key = logical_result_name(state, scene, join=join, layout=layout)
            except ValueError:
                continue
            yield key, scene


def scene_classes(pairs):
    classes = {}
    return tuple((classes.setdefault(key, len(classes)), scene) for key, scene in pairs)


def groups_from(pairs, scenes):
    parent = list(range(scenes))

    def root(scene):
        while parent[scene] != scene:
            parent[scene] = parent[parent[scene]]
            scene = parent[scene]
        return scene

    operations = {}
    for key, scene in pairs:
        representative = operations.setdefault(key, scene)
        left, right = root(representative), root(scene)
        if left != right:
            parent[max(left, right)] = min(left, right)
    groups = {}
    for scene in range(scenes):
        groups.setdefault(root(scene), []).append(scene)
    return tuple(tuple(value) for value in groups.values())

def dense_classes_problem():
    terms = [(("A", "B", "C"), "ij,jk,kl->il"),
             (("A", "B", "D"), "ij,jk,kl->li"),
             (("A", "B"), "ij,jk->ki"),
             (("B", "B", "C"), "ij,jk,kl->il")]
    return make_problem(terms * 3, {key: (2, 2) for key in "ABCD"})


def sparse_bridge_problem():
    specs = {"L": TensorSpec((2, 2, 2), "float64"),
             "R": TensorSpec((2, 2, 2), "float64"),
             "other": TensorSpec((2,), "float64")}
    terms = (ContractionTerm(("L", "R"), "apj,bqj->ab"),
             ContractionTerm(("other",), "i->"),
             ContractionTerm(("L", "R"), "apj,bqj->ab"))
    bridges = (SparseBridgeSpec("L", "R", "coo", 2, 4, scene=0),
               SparseBridgeSpec("L", "R", "coo", 2, 4, scene=2))
    sources = {key: SourceSpec(key, 0) for key in (*specs, "coo")}
    return ContractionProblem(specs, terms, sources, sparse_bridges=bridges)


class DeferredPhysicalFactsTests(unittest.TestCase):
    def test_physical_facts_are_deferred_and_match_the_eager_tables(self):
        problem, blocks, preparations = macro_fixture()
        profile = CostProfile()
        space = MacroSearchSpace(problem, blocks, preparations,
                                 memory_limit=1 << 30, profile=profile)
        for name in ("_initial_expression", "bridge_index", "mandatory_blocks",
                     "read_actions", "load_actions", "preparation_costs"):
            self.assertNotIn(name, space.__dict__)
        initial = ContractionState.initial(space.problem)
        required = _live_inputs(initial) & preparations.keys()
        facts = space.materialize_physical_facts()
        self.assertEqual(len(facts), 5)
        self.assertEqual(space.mandatory_blocks, frozenset(
            dep for key in required for dep in preparations[key].blocks))
        self.assertEqual(space.read_actions,
                         {key: Action("read_block", key) for key in blocks})
        self.assertEqual(space.load_actions,
                         {key: Action("load", key, seconds=space.logical_profile.snapshot_seconds(
                             initial._snapshot_bytes(key))) for key in preparations})
        self.assertEqual(space.preparation_costs,
                         {key: prep.seconds(space.profile) for key, prep in preparations.items()})
        self.assertEqual(space.bridge_index.scoped(0),
                         BridgeIndex(space.problem.sparse_bridges).scoped(0))
        for name in ("_initial_expression", "bridge_index", "mandatory_blocks",
                     "read_actions", "load_actions", "preparation_costs"):
            self.assertIn(name, space.__dict__)

    def test_problem_and_logical_profile_stay_field_wise_unchanged(self):
        problem, blocks, preparations = macro_fixture()
        profile = CostProfile()
        space = MacroSearchSpace(problem, blocks, preparations,
                                 memory_limit=1 << 30, profile=profile)
        reference = logical_source_problem(problem)
        self.assertIsNot(space.problem, problem)
        for name in ("backend", "specs", "sources", "terms", "sparse_bridges", "baryon_joins"):
            self.assertEqual(getattr(space.problem, name), getattr(reference, name))
        self.assertEqual(space.logical_profile, logical_cost_profile(profile))

    def test_bridge_index_is_deferred_and_still_resolves_one_scene(self):
        problem = sparse_bridge_problem()
        blocks = {key: StorageBlock("dataset", key, 32, host_bytes=32)
                  for key in problem.sources}
        preparations = {key: SourcePreparation((key,)) for key in problem.sources}
        space = MacroSearchSpace(problem, blocks, preparations, memory_limit=1 << 30)
        self.assertNotIn("bridge_index", space.__dict__)
        self.assertEqual(len(space.bridge_index.scoped(0)), 1)
        self.assertEqual(len(space.bridge_index.scoped(1)), 0)


class DedupedInitialRowsTests(unittest.TestCase):
    def test_repeated_initial_expressions_compile_once_per_class(self):
        classes = [(("A", "B"), "ij,jk->ik"), (("A", "C"), "ij,jk->ik")]
        problem = make_problem([classes[i % 2] for i in range(5000)],
                               {"A": (2, 3), "B": (3, 4), "C": (3, 4)})
        registry = IntegerRegistry(problem)
        with patch.object(registry, "pattern", wraps=registry.pattern) as pattern, \
             patch.object(ContractionState, "initial",
                          side_effect=AssertionError("temporary state")):
            space = IntegerSpace(problem, profile=CostProfile(), registry=registry)
        self.assertEqual(pattern.call_count, 2)
        self.assertEqual(len(space.initial_terms), 2)
        self.assertEqual(tuple(tuple(group) for group in space.scene_members),
                         (tuple(range(0, 5000, 2)), tuple(range(1, 5000, 2))))

    def test_dense_rows_match_the_state_based_reference(self):
        problem = make_problem([(("A", "B", "C"), "ij,jk,kl->il")] * 12,
                               {"A": (2, 3), "B": (3, 4), "C": (4, 5)})
        registry = IntegerRegistry(problem)
        with patch.object(ContractionState, "initial",
                          side_effect=AssertionError("temporary state")):
            space = IntegerSpace(problem, profile=CostProfile(), registry=registry)
        self.assertIsNone(space.scene_members)
        reference_symbols, reference_rows = state_based_rows(problem, registry)
        self.assertEqual(space.symbols, reference_symbols)
        self.assertEqual(space.initial_terms, reference_rows)

    def test_dense_rows_agree_with_the_debug_state_path(self):
        problem = dense_classes_problem()
        fast = IntegerSpace(problem, profile=CostProfile())
        debug = IntegerSpace(problem, profile=CostProfile(), debug=True)
        self.assertEqual(fast.initial_terms, debug.initial_terms)
        self.assertEqual(fast.symbols, debug.symbols)
        self.assertIsNotNone(debug.contexts[0])
        self.assertIsNone(fast.contexts[0])


class ClassificationContrastTests(unittest.TestCase):
    def test_new_and_reference_hashed_keys_give_identical_groups(self):
        problems = (dense_classes_problem(), join_fixture().problem,
                    sparse_bridge_problem(), sparse_spin_problem())
        for problem in problems:
            with self.subTest(terms=len(problem.terms)):
                fresh = list(shared_operation_keys(problem))
                hashed = list(hashed_operation_keys(problem))
                scenes = len(problem.terms)
                self.assertEqual(scene_classes(fresh), scene_classes(hashed))
                self.assertEqual(groups_from(fresh, scenes), groups_from(hashed, scenes))
                self.assertEqual(shared_components(problem), groups_from(hashed, scenes))
                dense = not problem.baryon_joins and not problem.sparse_bridges
                self.assertTrue(all(isinstance(key, tuple) for key, _ in fresh) or not dense)

    def test_dense_problem_keeps_more_than_one_operation_class(self):
        problem = dense_classes_problem()
        self.assertGreater(len({key for key, _ in shared_operation_keys(problem)}), 1)


def mixed_parallel_problem():
    """Dense scenes in early chunks, baryon joins only in a later chunk."""
    specs = {key: TensorSpec((2, 2), "float64") for key in ("a", "b", "c")}
    specs.update(spin=TensorSpec((2, 2, 2), "float64"),
                 mode=TensorSpec((2, 2, 2), "float64"),
                 A=TensorSpec((2,), "float64"), B=TensorSpec((2,), "float64"))
    joined = ContractionTerm(("a", "b", "c"), "ad,be,cf->")
    dense = ContractionTerm(("A", "B"), "i,i->")
    terms = [dense] * 520 + [joined] * 3 + [dense] * 200
    joins = (BaryonJoinSpec(520, "shared", "spin", "mode", "abc", "def"),)
    return ContractionProblem(specs, tuple(terms),
                              {key: SourceSpec(key, 0) for key in specs},
                              baryon_joins=joins)


class ParallelClassificationConsistencyTests(unittest.TestCase):
    @patch("lqcdweave.contraction.planning.optimizers.shared_components"
           "._PARALLEL_CLASSIFICATION_MIN_TERMS", 256)
    def test_parallel_chunks_agree_with_the_serial_partition_on_a_mixed_task(self):
        problem = mixed_parallel_problem()
        expected = shared_components(problem)
        self.assertGreater(len(expected), 1)
        statistics = {}
        actual = shared_components(
            problem, component_workers=2, statistics=statistics,
            budget=SearchBudget(deadline_seconds=30, search_memory_limit=64 * 2**20))
        self.assertGreaterEqual(statistics["classification_workers"], 1)
        self.assertTrue(statistics["classification_worker_pids"])
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
