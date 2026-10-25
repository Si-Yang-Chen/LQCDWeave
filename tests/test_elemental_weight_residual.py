"""Exact residual scene coefficients after elemental sampling compensation."""

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


class ElementalWeightResidualTests(unittest.TestCase):
    def test_fully_local_single_and_double_current_compensation(self):
        M, Np = 9, 5

        one_current = SamplingGroup("one", ("leg-0",), M, Np)
        one_scenes = expand_sampling_scenes(
            (one_current,),
            local_weights={"one": (((0,), Np),)},
        )
        self.assertEqual(one_scenes[0].coefficient, Fraction(1))

        two_currents = SamplingGroup("two", ("leg-0", "leg-1"), M, Np)
        two_scenes = expand_sampling_scenes(
            (two_currents,),
            local_weights={"two": (((0, 1), Np),)},
        )
        coefficients = {
            scene.partitions["two"]: scene.coefficient for scene in two_scenes
        }
        self.assertEqual(coefficients[(0, 1)], Fraction(1))
        self.assertEqual(coefficients[(0, 0)], Fraction(0))

        # The product of the inbuilt elemental weight and residual scene sum
        # recovers the original group target for each exact coincidence pattern.
        partitions = labelled_set_partitions(2)
        for exact_pattern in partitions:
            residual_sum = sum(
                coefficient
                for scene_pattern, coefficient in coefficients.items()
                if is_finer(blocks_of(scene_pattern), blocks_of(exact_pattern))
            )
            local_k = len({exact_pattern[position] for position in (0, 1)})
            inbuilt = expected_weight(M, Np, local_k)
            target = expected_weight(M, Np, len(set(exact_pattern)))
            self.assertEqual(inbuilt * residual_sum, target)

    def test_r4_two_local_pairs_leave_exact_residual_coefficients(self):
        M, current_n = 7, 4
        local_entries = (((0, 1), 5), ((2, 3), 6))
        group = SamplingGroup(
            "four",
            ("leg-0", "leg-1", "leg-2", "leg-3"),
            M,
            current_n,
        )
        scenes = expand_sampling_scenes(
            (group,),
            local_weights={"four": local_entries},
        )
        residual_coefficients = {
            scene.partitions["four"]: scene.coefficient for scene in scenes
        }

        for exact_pattern in labelled_set_partitions(4):
            residual_sum = sum(
                coefficient
                for scene_pattern, coefficient in residual_coefficients.items()
                if is_finer(blocks_of(scene_pattern), blocks_of(exact_pattern))
            )
            inbuilt = Fraction(1)
            for positions, original_n in local_entries:
                local_k = len({exact_pattern[position] for position in positions})
                inbuilt *= expected_weight(M, original_n, local_k)
            target = expected_weight(M, current_n, len(set(exact_pattern)))
            self.assertEqual(inbuilt * residual_sum, target)

    def test_default_coefficients_and_compatibility_exports_are_unchanged(self):
        from lqcdweave.contraction.expansion.sampling_coefficients import (
            scene_coefficients as compatibility_scene_coefficients,
        )

        group = SamplingGroup("plain", ("left", "right"), 8, 4)
        scenes = expand_sampling_scenes((group,))
        self.assertEqual(
            tuple(scene.coefficient for scene in scenes),
            tuple(scene_coefficients(2, 8, 4)),
        )
        self.assertIs(compatibility_scene_coefficients, scene_coefficients)


if __name__ == "__main__":
    unittest.main()
