"""Reference-order and canonical point-basis contract checks."""

import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.contracts.spatial_sampling import (
    SamplingPoolSpec as ContractSamplingPoolSpec,
    SpatialPointSet as ContractSpatialPointSet,
    SpatialSamplingDesign as ContractSpatialSamplingDesign,
)
from lqcdweave.sampling.spatial import (
    SamplingPoolSpec, SpatialPointSet, SpatialSamplingDesign,
    sample_spatial_points,
)
from lqcdweave.sampling.spatial.api import (
    sample_spatial_points as api_sample_spatial_points,
)


class SpatialSamplingTests(unittest.TestCase):
    def test_segment_lengths_against_ordered_list_reference(self):
        lattice, count, seed = (2, 3, 4, 7), 19, 81
        for length in (1, 2, 3, 7):
            with self.subTest(length=length):
                rng = np.random.Generator(np.random.PCG64(seed))
                expected = []
                for start in range(0, lattice[3], length):
                    pool, row = list(range(24)), []
                    for _ in range(count):
                        index = int(rng.integers(len(pool)))
                        flat = pool[index]
                        pool[index] = pool[-1]
                        pool.pop()
                        z, yx = divmod(flat, 6)
                        y, x = divmod(yx, 2)
                        row.append((x, y, z))
                    expected.extend([row] * min(length, lattice[3] - start))
                points = sample_spatial_points(SpatialSamplingDesign(lattice, count, length), seed=seed)
                np.testing.assert_array_equal(points.coordinates, expected)
                self.assertEqual(points.coordinates.shape, (lattice[3], count, 3))
                self.assertEqual(points.coordinates.dtype, np.int64)

    def test_numeric_endpoints_preserve_string_policy_sequences(self):
        lattice = (4, 3, 2, 7)
        for length, alias in ((1, "independent"), (7, "same")):
            numeric = sample_spatial_points(SpatialSamplingDesign(lattice, 5, length), seed=123)
            original = sample_spatial_points(SpatialSamplingDesign(lattice, 5, alias), seed=123)
            np.testing.assert_array_equal(numeric.coordinates, original.coordinates)
            self.assertEqual(original.design.temporal_policy, alias)

    def test_selected_segments_use_physical_times_and_full_rng_stream(self):
        from lqcdweave.contracts.time_selection import TimeSelectionRecord

        selection = TimeSelectionRecord(7, (6, 1, 0, 5, 3))
        for length in (1, 2, 3, 7):
            with self.subTest(length=length):
                design = SpatialSamplingDesign((2, 3, 4, 7), 5, length)
                full = sample_spatial_points(design, seed=42)
                compact = sample_spatial_points(design, seed=42, selection=selection)
                self.assertEqual(compact.times, selection.times)
                np.testing.assert_array_equal(compact.coordinates, full.coordinates[list(selection.times)])
                self.assertEqual(compact.coordinates.shape, (len(selection.times), 5, 3))

    def test_segment_realization_checks_ordered_points_by_physical_segment(self):
        design = SpatialSamplingDesign((2, 2, 2, 7), 2, 3)
        coords = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]]])
        # Adjacent stored rows in different physical segments can differ.
        SpatialPointSet(design.lattice_size, (5, 2), coords, design)
        SpatialPointSet(design.lattice_size, (2, 0), np.repeat(coords[:1], 2, axis=0), design)
        with self.assertRaises(ValueError):
            SpatialPointSet(design.lattice_size, (2, 0), coords, design)
        reordered = np.stack([coords[0], coords[0, ::-1]])
        with self.assertRaises(ValueError):
            SpatialPointSet(design.lattice_size, (2, 0), reordered, design)

    def test_segment_design_roundtrips_dataset_and_local_source_contract(self):
        from lqcdweave.adapters.spatial_points import SpatialPointHandle
        from lqcdweave.contracts.source import validate_local_spatial_point_spec
        from lqcdweave.contracts.time_selection import TimeSelectionRecord
        from lqcdweave.planning.spatial_points import sampled_spatial_points_spec
        from lqcdweave.storage.dataset import DatasetWriter

        design = SpatialSamplingDesign((2, 3, 4, 7), 5, np.int64(3))
        configuration = dict(ensemble="test", configuration="cfg-1", lattice=list(design.lattice_size))
        for selection in (None, TimeSelectionRecord(7, (6, 2, 0, 5, 3))):
            with self.subTest(selection=selection), TemporaryDirectory() as temporary:
                points = sample_spatial_points(design, seed=42, selection=selection)
                spec = sampled_spatial_points_spec(design, configuration=configuration, seed=42, selection=selection)
                validate_local_spatial_point_spec(spec.document)
                self.assertEqual(spec.document["definition"]["realization"]["design"]["temporal_policy"], 3)
                directory = Path(temporary) / "points"
                with DatasetWriter(directory, spec, provenance={}) as writer:
                    for slot in writer.missing:
                        writer.write(slot, points.coordinates[slot])
                    writer.publish()
                loaded = SpatialPointHandle.from_dataset(directory, cfg="cfg-1")
                self.assertEqual(loaded.points.design, design)
                self.assertEqual(loaded.points.times, points.times)
                np.testing.assert_array_equal(loaded.points.coordinates, points.coordinates)
                for policy in (0, 8, True, 1.0, "bad"):
                    invalid = spec.document
                    invalid["definition"]["realization"]["design"]["temporal_policy"] = policy
                    with self.assertRaises(ValueError):
                        validate_local_spatial_point_spec(invalid)

    def test_public_package_reexports_canonical_contracts(self):
        self.assertIs(SpatialPointSet, ContractSpatialPointSet)
        self.assertIs(SpatialSamplingDesign, ContractSpatialSamplingDesign)
        self.assertIs(SamplingPoolSpec, ContractSamplingPoolSpec)
        self.assertIs(sample_spatial_points, api_sample_spatial_points)

    def test_independent_permutation_reference(self):
        design = SpatialSamplingDesign((2, 3, 4, 5), 19)
        actual = sample_spatial_points(design, seed=81)
        rng = np.random.Generator(np.random.PCG64(81))
        expected = []
        for _ in range(5):
            pool = list(range(24))
            row = []
            for _ in range(19):
                i = int(rng.integers(len(pool)))
                flat = pool[i]
                pool[i] = pool[-1]
                pool.pop()
                z, yx = divmod(flat, 6)
                y, x = divmod(yx, 2)
                row.append((x, y, z))
            expected.append(row)
        np.testing.assert_array_equal(actual.coordinates, expected)
        self.assertEqual(actual.times, tuple(range(5)))
        self.assertFalse(np.array_equal(actual.coordinates[0], actual.coordinates[1]))
        self.assertEqual(actual.seed, 81)
        self.assertEqual(actual.algorithm, "numpy.PCG64/partial-fisher-yates-v1")

    def test_full_volume_and_same_policy(self):
        for policy in ("same", "independent"):
            result = sample_spatial_points(
                SpatialSamplingDesign((2, 3, 4, 3), 24, policy), seed=8
            )
            expected = set(itertools.product(range(2), range(3), range(4)))
            for row in result.coordinates:
                self.assertEqual(set(map(tuple, row)), expected)
            if policy == "same":
                np.testing.assert_array_equal(result.coordinates[0], result.coordinates[2])

    def test_reproducibility_and_global_rng_isolation(self):
        before = np.random.get_state()
        design = SpatialSamplingDesign((4, 3, 2, 7), 5)
        a = sample_spatial_points(design, seed=123)
        b = sample_spatial_points(design, seed=123)
        unseeded = sample_spatial_points(design)
        np.testing.assert_array_equal(a.coordinates, b.coordinates)
        self.assertIsNone(unseeded.seed)
        after = np.random.get_state()
        for x, y in zip(before, after):
            np.testing.assert_equal(x, y)

    def test_canonical_time_first_layout_and_basis_order(self):
        coordinates = np.array(
            [[[1, 2, 0], [0, 1, 1]], [[0, 0, 1], [1, 0, 0]]], dtype=np.int32
        )
        points = SpatialPointSet((2, 3, 2, 2), (0, 1), coordinates)
        np.testing.assert_array_equal(points.coordinates, coordinates)
        self.assertIsNone(points.design)
        self.assertIsNone(points.seed)
        self.assertIsNone(points.algorithm)
        coordinates[:] = 0
        self.assertEqual(tuple(points.coordinates[0, 0]), (1, 2, 0))
        with self.assertRaises(ValueError):
            points.coordinates.setflags(write=True)
        shifted = SpatialPointSet(points.lattice_size, (1, 0), points.coordinates)
        self.assertEqual(shifted.times, (1, 0))
        self.assertEqual(shifted.coordinates.shape, (2, 2, 3))

    def test_pool_descriptor_remains_the_contract_type(self):
        pool = SamplingPoolSpec("explicit-pool", 24, 5)
        self.assertEqual(pool.group_id, "explicit-pool")
        self.assertEqual((pool.spatial_sites, pool.sampled_points), (24, 5))

    def test_invalid_design_and_seed(self):
        for sizes, n, policy in [
            ((2, 3, 4), 1, "same"),
            ((0, 3, 4, 5), 1, "same"),
            ((2.0, 3, 4, 5), 1, "same"),
            ((True, 3, 4, 5), 1, "same"),
            ((2, 3, 4, 5), 25, "same"),
            ((2, 3, 4, 5), 0, "same"),
            ((2, 3, 4, 5), 1.5, "same"),
            ((2, 3, 4, 5), 1, "bad"),
            ((2, 3, 4, 5), 1, 0),
            ((2, 3, 4, 5), 1, 6),
            ((2, 3, 4, 5), 1, -1),
            ((2, 3, 4, 5), 1, True),
            ((2, 3, 4, 5), 1, np.bool_(False)),
            ((2, 3, 4, 5), 1, 1.0),
        ]:
            with self.subTest(sizes=sizes, n=n, policy=policy), self.assertRaises(ValueError):
                SpatialSamplingDesign(sizes, n, policy)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                sample_spatial_points(SpatialSamplingDesign((1, 1, 1, 1), 1), seed=seed)

    def test_invalid_realization(self):
        for times, coords in [
            ((0, 0), np.zeros((2, 1, 3), int)),
            ((2,), np.zeros((1, 1, 3), int)),
            ((0,), np.zeros((1, 2, 3), int)),
            ((0,), np.zeros((1, 1, 3), float)),
            ((0,), np.full((1, 1, 3), -1)),
            ((0,), np.full((1, 1, 3), 2)),
            ((0,), np.zeros((1, 0, 3), int)),
        ]:
            with self.subTest(times=times, coords=coords), self.assertRaises(ValueError):
                SpatialPointSet((2, 2, 2, 2), times, coords)
        with self.assertRaises(ValueError):
            SpatialPointSet(
                (2, 2, 2, 2), (0, 1), np.array([[[0, 0, 0]], [[1, 0, 0]]]),
                SpatialSamplingDesign((2, 2, 2, 2), 1, "same"),
            )


if __name__ == "__main__":
    unittest.main()
