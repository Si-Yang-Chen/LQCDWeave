"""Sampling scene contracts checked against finite-population enumeration."""

from fractions import Fraction
from itertools import combinations, product
import unittest

import numpy as np

from lqcdweave.contraction.expansion.sampling import (
    SamplingGroup,
    expand_sampling_scenes,
)
from lqcdweave.contraction.expansion.sampling_coefficients import (
    CompensationScheme,
    blocks_of,
    expected_weight,
    labelled_set_partitions,
    observed_weight,
    scene_coefficients,
)


def _complex_field(coordinates):
    """An intentionally nonsymmetric complex-valued function of labelled legs."""
    real = sum((leg + 2) * (coordinate + 1) ** (leg + 1)
               for leg, coordinate in enumerate(coordinates))
    imag = sum((leg + 1) * (coordinate + 2) ** (len(coordinates) - leg)
               for leg, coordinate in enumerate(coordinates))
    return complex(real, imag)


def _free_sum(partition, sample, field):
    """Enumerate all unrestricted block-label assignments for one scene."""
    blocks = blocks_of(partition)
    total = 0j
    for block_coordinates in product(sample, repeat=len(blocks)):
        coordinates = tuple(block_coordinates[block] for block in partition)
        total += field(coordinates)
    return total


def _single_pool_estimate(partitions, coefficients, sample, field):
    return sum(
        (coefficient * _free_sum(partition, sample, field)
         for partition, coefficient in zip(partitions, coefficients, strict=True)),
        0j,
    )


class SamplingSceneExpansionTests(unittest.TestCase):
    def test_labelled_partition_counts_and_exact_r2_formula(self):
        self.assertEqual(
            [len(labelled_set_partitions(r)) for r in range(5)],
            [1, 1, 2, 5, 15],
        )

        M, Np = 6, 3
        w1 = Fraction(M, Np)
        w2 = Fraction(M * (M - 1), Np * (Np - 1))
        self.assertEqual(scene_coefficients(2, M, Np), [w1 - w2, w2])
        self.assertEqual(scene_coefficients(2, M, Np), [Fraction(-3), Fraction(5)])

    def test_expected_coefficients_match_exhaustive_complex_oracle_for_r2_to_r4(self):
        # Enumerate every fixed-size draw and every free block assignment. The
        # target is independently summed over the full coordinate population.
        population = tuple(range(5))
        Np = 4
        for r in (2, 3, 4):
            with self.subTest(r=r):
                partitions = labelled_set_partitions(r)
                coefficients = scene_coefficients(r, len(population), Np)
                samples = tuple(combinations(population, Np))
                estimate_mean = sum(
                    (_single_pool_estimate(partitions, coefficients, sample, _complex_field)
                     for sample in samples),
                    0j,
                ) / len(samples)
                exact_target = sum(
                    (_complex_field(coordinates)
                     for coordinates in product(population, repeat=r)),
                    0j,
                )
                self.assertAlmostEqual(estimate_mean.real, exact_target.real, places=8)
                self.assertAlmostEqual(estimate_mean.imag, exact_target.imag, places=8)

    def test_r4_random_nonseparable_complex_array_matches_exhaustive_oracle(self):
        rng = np.random.default_rng(123)
        field_values = (
            rng.standard_normal((5,) * 4)
            + 1j * rng.standard_normal((5,) * 4)
        )
        self.assertGreater(np.linalg.matrix_rank(field_values.reshape(5, -1)), 1)
        field = lambda coordinates: field_values[coordinates]

        partitions = labelled_set_partitions(4)
        coefficients = scene_coefficients(4, M=5, Np=4)
        samples = tuple(combinations(range(5), 4))
        estimates = [
            _single_pool_estimate(partitions, coefficients, sample, field)
            for sample in samples
        ]
        estimate_mean = sum(estimates, 0j) / len(estimates)
        exact_target = complex(field_values.sum())
        self.assertLessEqual(abs(estimate_mean - exact_target), 1e-12)

    def test_cross_diagram_legs_and_different_times_share_explicit_pool(self):
        group = SamplingGroup(
            "shared-point-set",
            ["diagram-A/left@t0", "diagram-B/right@t7"],
            6,
            3,
        )
        scenes = expand_sampling_scenes([group])
        self.assertEqual(len(scenes), 2)
        self.assertEqual(scenes[0].partitions["shared-point-set"], (0, 0))
        self.assertEqual(scenes[0].point_labels["diagram-A/left@t0"],
                         scenes[0].point_labels["diagram-B/right@t7"])
        self.assertEqual(scenes[1].partitions["shared-point-set"], (0, 1))
        self.assertNotEqual(scenes[1].point_labels["diagram-A/left@t0"],
                            scenes[1].point_labels["diagram-B/right@t7"])
        # Different symbolic labels in the free sum do not encode an inequality.
        self.assertEqual(scenes[1].coefficient, Fraction(5))

    def test_independent_groups_multiply_even_when_coordinates_overlap(self):
        M, Np = 3, 2
        groups = (
            SamplingGroup("source-pool", ("source-leg",), M, Np),
            SamplingGroup("sink-pool", ("sink-leg",), M, Np),
        )
        scenes = expand_sampling_scenes(groups, groups_independent=True)
        self.assertEqual(len(scenes), 1)
        scene = scenes[0]
        self.assertEqual(scene.coefficient, Fraction(9, 4))
        self.assertNotEqual(scene.point_labels["source-leg"], scene.point_labels["sink-leg"])

        sample_sets = tuple(combinations(range(M), Np))
        self.assertTrue(any(left == right for left in sample_sets for right in sample_sets))
        estimates = []
        for source_sample, sink_sample in product(sample_sets, repeat=2):
            value = sum(
                (_complex_field((source, sink))
                 for source, sink in product(source_sample, sink_sample)),
                0j,
            )
            estimates.append(scene.coefficient * value)
        estimate_mean = sum(estimates, 0j) / len(estimates)
        exact_target = sum(
            (_complex_field(coordinates) for coordinates in product(range(M), repeat=2)),
            0j,
        )
        self.assertAlmostEqual(estimate_mean.real, exact_target.real, places=8)
        self.assertAlmostEqual(estimate_mean.imag, exact_target.imag, places=8)

    def test_multiple_groups_require_independence_declaration(self):
        groups = (
            SamplingGroup("a", ("a0",), 5, 3),
            SamplingGroup("b", ("b0",), 5, 3),
        )
        with self.assertRaisesRegex(ValueError, "explicit independence"):
            expand_sampling_scenes(groups)
        with self.assertRaisesRegex(NotImplementedError, "joint inclusion"):
            expand_sampling_scenes(groups, groups_independent=False)
        with self.assertRaises(TypeError):
            expand_sampling_scenes(groups, groups_independent=1)

    def test_empty_and_low_mode_groups_have_unit_weight(self):
        no_sample_scenes = expand_sampling_scenes([])
        self.assertEqual(len(no_sample_scenes), 1)
        self.assertEqual(no_sample_scenes[0].coefficient, Fraction(1))
        empty_group = SamplingGroup("no-point-legs", (), 0, 0)
        scenes = expand_sampling_scenes((empty_group,))
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].coefficient, Fraction(1))
        self.assertEqual(scenes[0].partitions["no-point-legs"], ())
        self.assertEqual(dict(scenes[0].point_labels), {})
        self.assertEqual(scene_coefficients(0, 0, 0), [Fraction(1)])

    def test_observed_scheme_uses_current_ordered_count_and_shared_scenes(self):
        M, current_Np = 6, 3
        expected = scene_coefficients(2, M, current_Np)
        # k=1 has all three current selected points; only four ordered k=2
        # records are present in this observed selection.
        observed = SamplingGroup(
            "observed-pool",
            ("left", "right"),
            M,
            current_Np,
            compensation=CompensationScheme.OBSERVED,
            observed_counts={1: current_Np, 2: 4},
        )
        scenes = expand_sampling_scenes((observed,))
        expected_scenes = expand_sampling_scenes((
            SamplingGroup("observed-pool", ("left", "right"), M, current_Np),
        ))
        self.assertEqual([scene.partitions["observed-pool"] for scene in scenes],
                         labelled_set_partitions(2))
        self.assertEqual([scene.partitions for scene in scenes],
                         [scene.partitions for scene in expected_scenes])
        self.assertEqual([scene.coefficient for scene in scenes],
                         [Fraction(-11, 2), Fraction(15, 2)])
        self.assertNotEqual([scene.coefficient for scene in scenes], expected)
        self.assertEqual(observed_weight(M, current_Np, 2, 6), expected_weight(M, current_Np, 2))

        # Counts from the maximum backing block cannot exceed the ordered tuples
        # available in the current selection.
        # The low-level weighting function remains generic; the pool API knows
        # this count belongs to the current selection and applies its bound.
        self.assertEqual(observed_weight(M, current_Np, 2, 7), Fraction(30, 7))
        oversized = SamplingGroup(
            "oversized-count", ("a", "b"), M, current_Np,
            compensation=CompensationScheme.OBSERVED,
            observed_counts={1: current_Np, 2: 12},
        )
        with self.assertRaisesRegex(ValueError, "current selection"):
            expand_sampling_scenes((oversized,))

    def test_zero_counts_missing_counts_and_zero_inclusion_are_errors(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            observed_weight(6, 3, 2, 0)
        with self.assertRaisesRegex(ValueError, "not estimable"):
            scene_coefficients(4, 64, 3)
        with self.assertRaisesRegex(ValueError, "not estimable"):
            expected_weight(64, 3, 4)
        too_few_points = SamplingGroup("too-small", ("a", "b", "c", "d"), 64, 3)
        with self.assertRaisesRegex(ValueError, "not estimable"):
            expand_sampling_scenes((too_few_points,))

        missing = SamplingGroup(
            "missing-count", ("a", "b"), 6, 3,
            compensation=CompensationScheme.OBSERVED,
            observed_counts={1: 3},
        )
        with self.assertRaisesRegex(ValueError, "missing current-selection"):
            expand_sampling_scenes((missing,))

    def test_unsupported_designs_and_invalid_pool_maps_fail_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "only"):
            SamplingGroup("stratified", ("p",), 8, 4, sampling_design="stratified")
        with self.assertRaises((TypeError, ValueError)):
            SamplingGroup("bad-bool", ("p",), True, 1)
        with self.assertRaises(ValueError):
            SamplingGroup("bad-size", ("p",), 4, 5)
        with self.assertRaises(ValueError):
            expand_sampling_scenes((
                SamplingGroup("same", ("a",), 4, 2),
                SamplingGroup("same", ("b",), 4, 2),
            ), groups_independent=True)
        with self.assertRaises(ValueError):
            expand_sampling_scenes((
                SamplingGroup("a", ("same-leg",), 4, 2),
                SamplingGroup("b", ("same-leg",), 4, 2),
            ), groups_independent=True)

    def test_nested_inputs_and_returned_scene_mappings_are_frozen(self):
        legs = ["left"]
        observed_counts = {1: 1}
        group = SamplingGroup(
            "frozen", legs, 4, 2,
            compensation=CompensationScheme.OBSERVED,
            observed_counts=observed_counts,
        )
        legs.append("late-mutation")
        observed_counts[1] = 2
        self.assertEqual(group.legs, ("left",))
        self.assertEqual(group.observed_counts[1], 1)
        scenes = expand_sampling_scenes((group,))
        with self.assertRaises(TypeError):
            group.observed_counts[1] = 3
        with self.assertRaises(TypeError):
            scenes[0].point_labels["new"] = "pool:frozen:point:0"
        with self.assertRaises(TypeError):
            scenes[0].partitions["frozen"] = (0,)


if __name__ == "__main__":
    unittest.main()
