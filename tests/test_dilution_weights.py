"""D-column inclusion coefficients against finite populations and complex QR."""

import json
from fractions import Fraction
from itertools import combinations, product
import unittest

import numpy as np

from lqcdweave.contracts.dilution import DilutionBasis, DilutionGroup, DilutionSource
from lqcdweave.contraction.expansion.dilution_weights import (
    dilution_component_weight, dilution_weight_scenes,
)


class DilutionSourceTests(unittest.TestCase):
    def source(self):
        return DilutionSource("actual-D", "actual-V", (
            DilutionGroup((2, 5, 8), ((0, 0), (0, 1))),
            DilutionGroup((9, 10), ((1, 0), (1, 1)), "exact"),
            DilutionGroup((13, 17, 19), ((2, 0),), "u1"),
        ), ((0, "realization-t0"), (3, "realization-t3")))

    def test_persistent_actual_source_and_missing_legacy(self):
        source = self.source()
        self.assertIs(DilutionBasis, DilutionSource)
        restored = DilutionSource.from_document(json.loads(json.dumps(source.document)))
        self.assertEqual(restored, source)
        self.assertIsNone(DilutionSource.from_document(None))
        self.assertEqual(source.ordered_indices, (0, 1, 2, 3, 4))
        self.assertEqual(source.first_moment_weights, (1.5, 1.5, 1, 1, 1))
        self.assertEqual(source.group_indices(1), (2, 3))
        self.assertIs(source.group_for_label((0, 1)), source.groups[0])
        self.assertEqual(source.pool_id(3), "realization-t3")
        self.assertIsNone(source.pool_id(1))
        self.assertIsNone(source.pool_key(1, 0))
        self.assertNotEqual(source.pool_key(0, 0), source.pool_key(0, 1))
        self.assertNotEqual(source.pool_key(0, 0), source.pool_key(3, 0))
        # Copies of one random realization share a pool independently of
        # the publication ID; explicit group parent ordering is scientific.
        copied = DilutionSource("copied-D", source.parent_basis_id, source.groups, source.time_pools)
        self.assertEqual(copied.pool_key(0, 0), source.pool_key(0, 0))

    def test_shared_column_selection_across_physical_times(self):
        source = self.source()
        shared = DilutionSource(source.data_id, source.parent_basis_id, source.groups,
                                ((0, "shared-columns"), (3, "shared-columns"),
                                 (5, "independently-selected-columns")))
        self.assertEqual(shared.pool_key(0, 0), shared.pool_key(3, 0))
        self.assertNotEqual(shared.pool_key(0, 0), shared.pool_key(5, 0))
        self.assertNotEqual(shared.pool_key(0, 0), shared.pool_key(3, 1))
        restored = DilutionSource.from_document(json.loads(json.dumps(shared.document)))
        self.assertEqual(restored.pool_key(0, 0), restored.pool_key(3, 0))

    def test_complete_group_selection_and_explicit_labels(self):
        source = self.source()
        self.assertEqual(source.validate_selection((3, 2, 1, 0)), (3, 2, 1, 0))
        for labels in ((0,), (2, 2), (5,)):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                source.validate_selection(labels)
        document = source.document
        document["ordered_indices"] = [1, 0, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "ordered indices"):
            DilutionSource.from_document(document)

    def test_description_requires_completed_actual_parent(self):
        source = self.source()
        description = {
            "parent_basis_id": source.parent_basis_id,
            "mode_groups": tuple(g.parent_modes for g in source.groups),
            "parent_counts": tuple(g.n for g in source.groups),
            "vector_counts": tuple(g.m for g in source.groups),
            "vector_labels": source.vector_labels,
            "branches": tuple(g.branch for g in source.groups),
            "covariance_factors": tuple(g.covariance_factor for g in source.groups),
            "complete": True, "normalization": "raw", "algorithm": source.algorithm,
        }
        restored = DilutionSource.from_description(description, data_id=source.data_id,
                                                   time_pools=source.time_pools)
        self.assertEqual(restored, source)
        for update in ({"parent_basis_id": None}, {"complete": False},
                       {"parent_counts": (4, 2, 3)}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                DilutionSource.from_description(dict(description, **update), data_id="D")

    def test_covariance_is_raw_and_group_provenance_is_checked(self):
        for arguments in (
            ((0, 1), ((0, 0),), "qr", 1),
            ((0, 1), ((0, 0),), "exact", None),
            ((0, 1), ((0, 0), (0, 1)), "u1", None),
            ((0, 0), ((0, 0),), "qr", None),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                DilutionGroup(*arguments)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            DilutionSource("D", "V", (DilutionGroup((0, 1), ((0, 0),)),
                                         DilutionGroup((1, 2), ((1, 0),))))
        with self.assertRaisesRegex(ValueError, "actual identity"):
            DilutionSource("D", None, (DilutionGroup((0, 1), ((0, 0),)),))


class DilutionComponentWeightTests(unittest.TestCase):
    @staticmethod
    def group(n, m, branch="qr"):
        return DilutionGroup(tuple(range(n)), tuple((0, i) for i in range(m)), branch)

    def test_actual_equal_and_distinct_component_weights(self):
        group = self.group(7, 3)
        self.assertEqual(dilution_component_weight(group, 0), 1)
        self.assertEqual(dilution_component_weight(group, 1), Fraction(7, 3))
        self.assertEqual(dilution_component_weight(group, 2), 7)
        self.assertEqual(dilution_component_weight(group, 3), 35)
        scenes = dilution_weight_scenes(group, ("left", "right"))
        terms = {scene.partitions["dilution"]: scene.coefficient for scene in scenes}
        self.assertEqual(terms[(0, 1)], 7)
        self.assertEqual(terms[(0, 0)], Fraction(-14, 3))
        # A single local D/D matrix already contains H1 on its diagonal and
        # H2 off-diagonal. Its raw free-sum residual is exactly one.
        weighted = dilution_weight_scenes(group, ("left", "right"),
            local_entries=(((0, 1), 3),))
        self.assertEqual(len(weighted), 1)
        self.assertEqual(weighted[0].partitions["dilution"], (0, 1))
        self.assertEqual(weighted[0].coefficient, 1)

    def test_complete_equality_patterns_up_to_five_legs(self):
        n, m = 8, 5
        group = self.group(n, m)
        for k in range(1, 6):
            with self.subTest(k=k):
                scenes = dilution_weight_scenes(group, tuple(f"leg{i}" for i in range(k)))
                # Enumerate assignments independently. Every actual equality
                # pattern must accumulate H_r from compatible free-sum scenes.
                for labels in product(range(k), repeat=k):
                    distinct = len(set(labels))
                    expected = Fraction(1)
                    for step in range(distinct):
                        expected *= Fraction(n - step, m - step)
                    actual = sum((scene.coefficient for scene in scenes
                        if all(labels[i] == labels[j]
                            for i in range(k) for j in range(i)
                            if scene.partitions["dilution"][i] == scene.partitions["dilution"][j])), Fraction(0))
                    self.assertEqual(actual, expected)

    def test_proven_diagonal_quotient_supports_more_occurrences_than_rank(self):
        group = self.group(6, 1)
        scenes = dilution_weight_scenes(group, ("a", "b", "c", "d"),
            equalities=((0, 1), (1, 2), (2, 3)))
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].partitions["dilution"], (0, 0, 0, 0))
        self.assertEqual(scenes[0].coefficient, 6)
        weighted = dilution_weight_scenes(group, ("a", "b", "c", "d"),
            local_entries=(((0, 1), 1), ((2, 3), 1)), equalities=((0, 1), (1, 2), (2, 3)))
        self.assertEqual(weighted[0].coefficient, Fraction(1, 6))
        with self.assertRaisesRegex(ValueError, "inclusion probability is zero"):
            dilution_weight_scenes(group, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "inclusion probability is zero"):
            dilution_component_weight(group, 2)

    def test_exact_groups_are_unrestricted_without_order_limit(self):
        for branch in ("qr", "exact"):
            group = self.group(1, 1, branch)
            scene, = dilution_weight_scenes(group, tuple(f"leg{i}" for i in range(80)),
                local_entries=(((0, 1, 2), 1),), equalities=((0, 1),))
            self.assertEqual(scene.coefficient, 1)
            self.assertEqual(scene.partitions["dilution"], tuple(range(80)))
            self.assertEqual(dilution_component_weight(group, 80), 1)
        legacy = self.group(5, 1, "u1")
        self.assertEqual(dilution_component_weight(legacy, 1), 1)
        self.assertEqual(dilution_weight_scenes(legacy, ("one",))[0].coefficient, 1)
        with self.assertRaisesRegex(ValueError, "u1.*unsupported"):
            dilution_weight_scenes(legacy, ("a", "b"), equalities=((0, 1),))
        with self.assertRaisesRegex(ValueError, "u1.*unsupported"):
            dilution_component_weight(legacy, 2)

    @staticmethod
    def finite_average(function, group, scenes, local_entries=()):
        """All fixed-size subsets, with a literal free component-label sum."""
        total = 0j
        draws = tuple(combinations(range(group.n), group.m))
        k = function.ndim
        for sample in draws:
            for scene in scenes:
                pattern = scene.partitions["dilution"]
                for values in product(sample, repeat=len(set(pattern))):
                    actual = tuple(values[block] for block in pattern)
                    local = Fraction(1)
                    for positions, original_m in local_entries:
                        r = len({actual[position] for position in positions})
                        for step in range(r):
                            local *= Fraction(group.n - step, original_m - step)
                    total += float(scene.coefficient * local) * function[actual]
        return total / len(draws)

    def test_finite_all_subsets_arbitrary_complex_component_oracle(self):
        rng = np.random.default_rng(871)
        for k in range(1, 6):
            with self.subTest(k=k):
                n, m = k + 1, k
                function = rng.normal(size=(n,) * k) + 1j * rng.normal(size=(n,) * k)
                group = self.group(n, m)
                scenes = dilution_weight_scenes(group, tuple(f"leg{i}" for i in range(k)))
                observed = self.finite_average(function, group, scenes)
                np.testing.assert_allclose(observed, np.sum(function), atol=2e-10)
        # A rank-one draw can estimate a tensor known to live only on its
        # complete label diagonal even though the tensor has four legs.
        n, k = 4, 4
        function = np.zeros((n,) * k, dtype=complex)
        for i in range(n):
            function[(i,) * k] = rng.normal() + 1j * rng.normal()
        group = self.group(n, 1)
        scenes = dilution_weight_scenes(group, tuple(f"leg{i}" for i in range(k)),
            equalities=((0, 1), (1, 2), (2, 3)))
        np.testing.assert_allclose(self.finite_average(function, group, scenes), np.sum(function), atol=1e-13)

    def test_mixed_local_preweight_residual_matches_literal_subsets(self):
        rng = np.random.default_rng(535)
        group = self.group(5, 3)
        function = rng.normal(size=(5, 5, 5)) + 1j * rng.normal(size=(5, 5, 5))
        legs = ("current-left", "current-right", "other-current")
        local_entries = (((0, 1), 3), ((2,), 3))
        raw = dilution_weight_scenes(group, legs)
        weighted = dilution_weight_scenes(group, legs, local_entries=local_entries)
        raw_average = self.finite_average(function, group, raw)
        weighted_average = self.finite_average(function, group, weighted, local_entries)
        np.testing.assert_allclose(raw_average, np.sum(function), atol=2e-12)
        np.testing.assert_allclose(weighted_average, np.sum(function), atol=2e-12)
        # With a known local current diagonal, collapse its two labels before
        # inverse inclusion or local-weight division. Sample rank two suffices.
        diagonal_group = self.group(5, 2)
        diagonal_function = function.copy()
        i, j, _ = np.indices(diagonal_function.shape)
        diagonal_function[i != j] = 0
        entries = (((0, 1), 2), ((2,), 2))
        scenes = dilution_weight_scenes(diagonal_group, legs, local_entries=entries, equalities=((0, 1),))
        np.testing.assert_allclose(self.finite_average(diagonal_function, diagonal_group, scenes, entries),
                                   np.sum(diagonal_function), atol=2e-12)

    def test_complex_qr_same_and_distinct_column_moments(self):
        # A complete QR basis is randomly rotated; its columns are exchangeable.
        # Thus the first m columns have the same inclusion law as a uniform
        # m-subset of all n columns. The full closed contraction Tr(A B) is
        # basis invariant, so no Haar-projector inverse is needed.
        rng = np.random.default_rng(1981)
        n, m, count = 5, 3, 18000
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        q, _ = np.linalg.qr(rng.normal(size=(count, n, n)) + 1j * rng.normal(size=(count, n, n)))
        a_q = q.conj().transpose(0, 2, 1) @ a @ q
        b_q = q.conj().transpose(0, 2, 1) @ b @ q
        components = a_q * b_q.transpose(0, 2, 1)
        equal = components[:, 0, 0]
        distinct = components[:, 0, 1]
        trace_ab = np.trace(a @ b)
        trace_a_b = np.trace(a) * np.trace(b)
        equal_reference = (trace_a_b + trace_ab) / (n * (n + 1))
        distinct_reference = (n * trace_ab - trace_a_b) / (n * (n * n - 1))
        for values, reference in ((equal, equal_reference), (distinct, distinct_reference)):
            error = np.std(values, ddof=1) / np.sqrt(count)
            self.assertLess(abs(np.mean(values) - reference), 5 * error)
        np.testing.assert_allclose(np.sum(components, axis=(1, 2)), trace_ab, atol=8e-14)
        group = self.group(n, m)
        sampled = components[:, :m, :m]
        diagonal = np.trace(sampled, axis1=1, axis2=2)
        unrestricted = np.sum(sampled, axis=(1, 2))
        estimate = float(dilution_component_weight(group, 2)) * (unrestricted - diagonal)
        estimate += float(dilution_component_weight(group, 1)) * diagonal
        error = np.std(estimate, ddof=1) / np.sqrt(count)
        self.assertLess(abs(np.mean(estimate) - trace_ab), 5 * error)
        # Independently exhaust every subset of one complex rotated basis:
        # the inclusion identity holds conditionally, not just in Monte Carlo.
        np.testing.assert_allclose(self.finite_average(components[0], group,
            dilution_weight_scenes(group, ("a", "b"))), trace_ab, atol=4e-14)


if __name__ == "__main__":
    unittest.main()
