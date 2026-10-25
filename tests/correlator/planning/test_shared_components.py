"""Exact initial-operation grouping, without future-sharing assumptions."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from lqcdweave.contracts.correlator.api import (
    BaryonJoinSpec, ContractionProblem, ContractionTerm, SourceSpec,
    SourcePreparation, SparseBridgeSpec, TensorSpec,
)
from lqcdweave.contraction.planning.optimizers.shared_components import (
    component_problem, shared_components,
)
from lqcdweave.contraction.planning.state import ContractionState
from correlator.planning.test_candidate_metadata import join_fixture
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.canonical_operations import describe, logical_result_name
from itertools import combinations


def problem(terms, specs=None, **kwargs):
    specs = specs or {key: TensorSpec((2,), "float64") for term in terms for key in term.inputs}
    return ContractionProblem(specs, tuple(terms), {key: SourceSpec("same-raw-block", 0)
                              for key in specs}, **kwargs)


class SharedComponentTests(unittest.TestCase):
    def test_common_input_or_physical_region_alone_does_not_connect(self):
        source = problem((ContractionTerm(("A", "B"), "i,i->"),
                          ContractionTerm(("A", "C"), "i,i->")))
        self.assertEqual(shared_components(source), ((0,), (1,)))

    def test_actual_output_boundary_distinguishes_computation(self):
        source = problem((ContractionTerm(("A", "B"), "i,i->"),
                          ContractionTerm(("A", "B"), "i,i->i")))
        self.assertEqual(shared_components(source), ((0,), (1,)))

    def test_transitive_sharing_and_stable_scene_order(self):
        source = problem((ContractionTerm(("A", "B", "C"), "i,i,i->"),
                          ContractionTerm(("D", "E"), "i,i->"),
                          ContractionTerm(("A", "B", "F"), "i,i,i->"),
                          ContractionTerm(("B", "F", "G"), "i,i,i->")))
        self.assertEqual(shared_components(source), ((0, 2, 3), (1,)))

    def test_identical_initial_pair_with_repeated_operands_is_grouped(self):
        source = problem((ContractionTerm(("A", "A", "B", "C"), "i,i,i,i->"),
                          ContractionTerm(("A", "A", "B", "D"), "i,i,i,i->")))
        self.assertEqual(shared_components(source), ((0, 1),))
        self.assert_initial_operation_matches(source)

    def test_unary_identity_does_not_predict_future_use_inside_a_longer_scene(self):
        source = problem((ContractionTerm(("A",), "i->i"),
                          ContractionTerm(("A", "B"), "i,i->"),
                          ContractionTerm(("C",), "i->")))
        self.assertEqual(shared_components(source), ((0,), (1,), (2,)))
        self.assert_initial_operation_matches(source)

    def test_unary_reduction_boundary_is_not_the_original_leaf(self):
        source = problem((ContractionTerm(("A",), "i->"),
                          ContractionTerm(("A", "B"), "i,i->")))
        self.assertEqual(shared_components(source), ((0,), (1,)))
        self.assert_initial_operation_matches(source)

    def test_full_axis_wiring_and_dummy_renaming(self):
        specs = {k: TensorSpec((2, 2), "float64") for k in ("A", "B")}
        source = problem((ContractionTerm(("A", "B"), "ij,ij->i"),
                          ContractionTerm(("A", "B"), "ij,ji->i"),
                          ContractionTerm(("B", "A"), "ab,ab->a")), specs)
        self.assertEqual(shared_components(source), ((0, 2), (1,)))
        self.assert_initial_operation_matches(source)

    def test_outer_product_does_not_connect_otherwise_separate_terms(self):
        specs = {k: TensorSpec((2, 2), "float64") for k in ("A", "B", "C", "D")}
        source = problem((ContractionTerm(("A", "B", "C"), "ij,kl,jk->il"),
                          ContractionTerm(("A", "B", "D"), "ij,kl,jk->il")), specs)
        self.assertEqual(shared_components(source), ((0,), (1,)))
        state = ContractionState.initial(source)
        with self.assertRaisesRegex(ValueError, "outer products"):
            state._contract_layout(0, (0, 1))
        self.assert_initial_operation_matches(source)

    def test_initial_classification_does_not_price_or_materialize_operations(self):
        source = problem((ContractionTerm(("A", "B"), "i,i->"),) * 2)
        with patch.object(CostProfile, "contract_seconds", side_effect=AssertionError("priced")), \
             patch.object(ContractionState, "contract", side_effect=AssertionError("applied")), \
             patch.object(ContractionState, "_join_from_layout", side_effect=AssertionError("joined")):
            self.assertEqual(shared_components(source), ((0, 1),))
            self.assertEqual(shared_components(join_fixture().problem), ((0, 1), (2,)))

    def test_single_leg_join_includes_full_implicit_input_identity(self):
        specs = {"leg": TensorSpec((2,) * 6, "float64"),
                 "spin": TensorSpec((2,) * 3, "float64"),
                 "other_spin": TensorSpec((2,) * 3, "float64"),
                 "mode": TensorSpec((2,) * 3, "float64")}
        term = ContractionTerm(("leg",), "abcdef->")
        joins = (BaryonJoinSpec(0, "J", "spin", "mode", "abc", "def"),
                 BaryonJoinSpec(1, "J", "spin", "mode", "abc", "def"),
                 BaryonJoinSpec(2, "J", "other_spin", "mode", "abc", "def"))
        source = problem((term,) * 3, specs, baryon_joins=joins)
        self.assertEqual(shared_components(source), ((0, 1), (2,)))
        self.assert_initial_operation_matches(source)

    def test_sparse_operation_matches_and_records_are_retained(self):
        specs = {"L": TensorSpec((2, 2, 2), "float64"),
                 "R": TensorSpec((2, 2, 2), "float64"),
                 "other": TensorSpec((2,), "float64")}
        terms = (ContractionTerm(("L", "R"), "apj,bqj->ab"),
                 ContractionTerm(("other",), "i->"),
                 ContractionTerm(("L", "R"), "apj,bqj->ab"))
        bridges = (SparseBridgeSpec("L", "R", "coo", 2, 4, scene=0),
                   SparseBridgeSpec("L", "R", "coo", 2, 4, scene=2))
        sources = {key: SourceSpec(key, 0) for key in (*specs, "coo")}
        source = ContractionProblem(specs, terms, sources, sparse_bridges=bridges)
        self.assertEqual(shared_components(source), ((0, 2), (1,)))
        local = component_problem(source, (0, 2))
        self.assertEqual(tuple(b.scene for b in local.sparse_bridges), (0, 1))
        self.assertEqual(set(local.sources), {"L", "R", "coo"})
        self.assert_initial_operation_matches(source)

    def test_component_remaps_join_scenes_and_removes_unrelated_specs(self):
        source = join_fixture().problem
        local = component_problem(source, (2, 0))
        self.assertEqual(local.terms, (source.terms[2], source.terms[0]))
        self.assertEqual(tuple(j.scene for j in local.baryon_joins), (1, 0))
        self.assertEqual(tuple(j.spin_labels for j in local.baryon_joins),
                         (tuple(map(ord, "abc")), tuple(map(ord, "bac"))))
        self.assertEqual(local.backend, source.backend)
        dense = problem((ContractionTerm(("A", "B"), "i,i->"),
                         ContractionTerm(("C", "D"), "i,i->")), single_read_regions=True)
        local = component_problem(dense, (1,))
        self.assertEqual(set(local.specs), {"C", "D"})
        self.assertEqual(set(local.sources), {"C", "D"})
        self.assertTrue(local.single_read_regions)
        for invalid in ((), (0, 0), (2,), (-1,), (True,)):
            with self.subTest(scenes=invalid), self.assertRaises(ValueError):
                component_problem(dense, invalid)

    def test_cancelled_preprocessing_does_not_publish_partial_partition(self):
        source = problem((ContractionTerm(("A", "B"), "i,i->"),) * 20)
        calls = 0
        def checkpoint():
            nonlocal calls
            calls += 1
            if calls == 10:
                raise InterruptedError("budget")
        with self.assertRaises(InterruptedError):
            shared_components(source, checkpoint)

    def test_component_reuses_frozen_spin_while_remapping_scope(self):
        from correlator.planning.test_sparse_spin_join_path import sparse_spin_problem
        from lqcdweave.storage.contraction_plans.api import encode_plan_value, decode_plan_value

        base = sparse_spin_problem()
        joins = tuple(replace(base.baryon_joins[0], scene=index) for index in range(3))
        source = replace(base, terms=base.terms * 3, baryon_joins=joins)
        with patch('lqcdweave.contracts.correlator.scene._sparse_spin_entries',
                   side_effect=AssertionError('component reparsed frozen coefficients')):
            local = component_problem(source, (2, 0))
        self.assertEqual(tuple(join.scene for join in source.baryon_joins), (0, 1, 2))
        self.assertEqual(tuple(join.scene for join in local.baryon_joins), (1, 0))
        for actual, original in zip(local.baryon_joins, (joins[0], joins[2])):
            self.assertEqual(actual, replace(original, scene=actual.scene))
            self.assertIs(actual.sparse_spin, original.sparse_spin)
            self.assertEqual(decode_plan_value(encode_plan_value(actual)), actual)

    def assert_initial_operation_matches(self, source):
        """Compare unpriced descriptors to the independently priced initial actions."""
        state = ContractionState.initial(source)
        state = replace(state, loaded=frozenset(state.catalog) | frozenset(source.sources))
        owner = {scene: group for group, scenes in enumerate(shared_components(source)) for scene in scenes}
        origins = {}
        for scene, term in enumerate(state.pending):
            pairs = ((0,),) if len(term.factors) == 1 else combinations(range(len(term.factors)), 2)
            for pair in pairs:
                try:
                    action, *_ = describe(state, scene, tuple(pair), profile=CostProfile())
                except ValueError:
                    continue
                self.assertEqual(logical_result_name(state, scene, tuple(pair)), action.tensor)
                self.assertEqual(origins.setdefault(action.tensor, owner[scene]), owner[scene])
            for join in source.baryon_joins:
                if join.scene != scene:
                    continue
                try:
                    action, *_ = describe(state, scene, join=join, profile=CostProfile())
                except ValueError:
                    continue
                self.assertEqual(logical_result_name(state, scene, join=join), action.tensor)
                self.assertEqual(origins.setdefault(action.tensor, owner[scene]), owner[scene])


if __name__ == "__main__":
    unittest.main()
