"""Point-equality quotienting for sampled P-end scene expansion."""

from fractions import Fraction
import unittest

from lqcdweave.contraction.expansion.sampling import (
    SamplingGroup,
    expand_sampling_scenes,
)
from lqcdweave.contracts.sampling_weights import (
    blocks_of,
    expected_weight,
    is_finer,
    labelled_set_partitions,
    scene_coefficients,
)


class DiagonalSamplingSceneTests(unittest.TestCase):
    def test_single_diagonal_pair_needs_one_point_and_has_w1(self):
        group = SamplingGroup("pair", ("left", "right"), 9, 1)

        scenes = expand_sampling_scenes(
            (group,), point_equalities={"pair": ((0, 1),)}
        )

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].partitions["pair"], (0, 0))
        self.assertEqual(scenes[0].point_labels["left"], scenes[0].point_labels["right"])
        self.assertEqual(scenes[0].coefficient, expected_weight(9, 1, 1))
        self.assertEqual(scenes[0].coefficient, Fraction(9))

    def test_two_diagonal_pairs_reduce_four_legs_to_two_scenes(self):
        group = SamplingGroup(
            "four", ("left-a", "left-b", "right-a", "right-b"), 9, 2
        )

        scenes = expand_sampling_scenes(
            (group,), point_equalities={"four": ((0, 1), (2, 3))}
        )

        self.assertEqual(len(scenes), 2)
        self.assertEqual(
            [scene.partitions["four"] for scene in scenes],
            [(0, 0, 0, 0), (0, 0, 1, 1)],
        )
        self.assertEqual(
            [scene.coefficient for scene in scenes], scene_coefficients(2, 9, 2)
        )
        self.assertEqual([scene.coefficient for scene in scenes], [Fraction(-63, 2), Fraction(36)])
        for scene in scenes:
            self.assertEqual(
                scene.point_labels["left-a"], scene.point_labels["left-b"]
            )
            self.assertEqual(
                scene.point_labels["right-a"], scene.point_labels["right-b"]
            )

    def test_local_residuals_match_exact_patterns_at_current_and_smaller_n(self):
        M, current_n = 9, 3
        group = SamplingGroup(
            "four", ("left-a", "left-b", "right-a", "right-b"), M, current_n
        )
        equalities = {"four": ((0, 1), (2, 3))}
        exact_patterns = tuple(
            (partition[0], partition[0], partition[1], partition[1])
            for partition in labelled_set_partitions(2)
        )

        for local_n in (current_n, 2):
            with self.subTest(local_n=local_n):
                local_entries = (((0,), local_n), ((2,), local_n))
                scenes = expand_sampling_scenes(
                    (group,),
                    point_equalities=equalities,
                    local_weights={"four": local_entries},
                )
                self.assertEqual(len(scenes), 2)

                for exact_pattern in exact_patterns:
                    residual_sum = sum(
                        (
                            scene.coefficient
                            for scene in scenes
                            if is_finer(
                                blocks_of(scene.partitions["four"]),
                                blocks_of(exact_pattern),
                            )
                        ),
                        Fraction(0),
                    )
                    local_inbuilt = Fraction(1)
                    for positions, original_n in local_entries:
                        local_k = len({exact_pattern[position] for position in positions})
                        local_inbuilt *= expected_weight(M, original_n, local_k)
                    quotient_k = len({exact_pattern[position] for position in (0, 2)})
                    target = expected_weight(M, current_n, quotient_k)
                    self.assertEqual(local_inbuilt * residual_sum, target)

    def test_local_entries_may_collapse_onto_the_same_quotient_class(self):
        M, current_n = 9, 3
        group = SamplingGroup("pair", ("left", "right"), M, current_n)
        scenes = expand_sampling_scenes(
            (group,),
            point_equalities={"pair": ((0, 1),)},
            local_weights={"pair": (((0,), current_n), ((1,), current_n))},
        )

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].partitions["pair"], (0, 0))
        local_inbuilt = expected_weight(M, current_n, 1) ** 2
        self.assertEqual(local_inbuilt * scenes[0].coefficient, expected_weight(M, current_n, 1))

    def test_unconstrained_scene_list_is_unchanged(self):
        group = SamplingGroup("plain", ("a", "b", "c"), 8, 4)
        scenes = expand_sampling_scenes((group,))

        self.assertEqual(
            [scene.partitions["plain"] for scene in scenes],
            labelled_set_partitions(3),
        )
        self.assertEqual(
            [scene.coefficient for scene in scenes], scene_coefficients(3, 8, 4)
        )
        self.assertEqual(
            scenes,
            expand_sampling_scenes((group,), point_equalities=None),
        )

    def test_observed_compensation_ignores_equalities(self):
        group = SamplingGroup(
            "observed",
            ("a", "b", "c"),
            8,
            3,
            compensation="observed",
            observed_counts={1: 3, 2: 6, 3: 6},
        )
        baseline = expand_sampling_scenes((group,))
        constrained = expand_sampling_scenes(
            (group,), point_equalities={"observed": ((0, 1), (1, 2))}
        )
        self.assertEqual(constrained, baseline)
        self.assertEqual(len(constrained), len(labelled_set_partitions(3)))

    def test_invalid_point_equality_mappings_and_pairs_are_rejected(self):
        group = SamplingGroup("pool", ("a", "b", "c"), 5, 3)

        with self.assertRaises(TypeError):
            expand_sampling_scenes((group,), point_equalities=())
        with self.assertRaises(TypeError):
            expand_sampling_scenes((group,), point_equalities={1: ((0, 1),)})
        with self.assertRaises(ValueError):
            expand_sampling_scenes((group,), point_equalities={"missing": ((0, 1),)})
        with self.assertRaises(TypeError):
            expand_sampling_scenes((group,), point_equalities={"pool": (1, 2)})
        with self.assertRaises(ValueError):
            expand_sampling_scenes((group,), point_equalities={"pool": ((0,),)})
        with self.assertRaises(TypeError):
            expand_sampling_scenes((group,), point_equalities={"pool": ((0, True),)})
        with self.assertRaises(ValueError):
            expand_sampling_scenes((group,), point_equalities={"pool": ((0, 3),)})


if __name__ == "__main__":
    unittest.main()
