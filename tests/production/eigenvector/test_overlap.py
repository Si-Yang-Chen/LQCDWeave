"""Independent scalar checks for indexed point/eigenvector overlap gathers."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
from lqcdweave.production.eigenvector import overlap

from lqcdweave.production.eigenvector.api import (
    calc_overlap_row,
    get_overlap_matrix,
)


def _fixture(dtype=np.complex128):
    rng = np.random.default_rng(82031)
    vectors = (
        rng.normal(size=(3, 4, 2, 3, 4, 3))
        + 1j * rng.normal(size=(3, 4, 2, 3, 4, 3))
    ).astype(dtype)
    # Shape is (time, point, xyz). Point order is unrelated to XYZ sort order,
    # and coordinates move between physical times. Nt != Np catches axis swaps.
    points = np.asarray(
        [
            [(3, 2, 1), (0, 0, 0), (1, 2, 0), (2, 1, 1)],
            [(0, 1, 1), (3, 1, 0), (2, 0, 1), (1, 2, 1)],
            [(1, 0, 0), (2, 2, 1), (0, 1, 1), (3, 0, 0)],
        ],
        dtype=np.int32,
    )
    return vectors, points


def _scalar_row(vectors, points, used_ne, used_np):
    """Scalar indexing oracle independent of the production gather expression."""
    output = np.empty((used_ne, used_np, 3), dtype=vectors.dtype)
    for mode in range(used_ne):
        for point in range(used_np):
            x, y, z = (int(value) for value in points[point])
            for color in range(3):
                output[mode, point, color] = vectors[mode, z, y, x, color]
    return output


def _scalar_full(vectors, points, used_ne, used_np):
    output = np.empty((vectors.shape[0], used_ne, used_np, 3), dtype=vectors.dtype)
    for time in range(vectors.shape[0]):
        output[time] = _scalar_row(
            vectors[time], points[time, :used_np], used_ne, used_np,
        )
    return output


class PointOverlapTests(unittest.TestCase):
    def test_full_coordinates_checked_once_per_time_before_output_allocation(self):
        vectors, points = _fixture()
        with patch.object(overlap, '_selected_coordinates',
                          wraps=overlap._selected_coordinates) as coordinates, patch.object(
                              np, 'isfinite', wraps=np.isfinite,
                          ) as finite:
            actual = get_overlap_matrix(vectors, points)
        self.assertEqual(coordinates.call_count, vectors.shape[0])
        self.assertEqual(finite.call_count, vectors.shape[0])
        np.testing.assert_array_equal(actual, _scalar_full(vectors, points, 4, 4))

        bad = points.copy()
        bad[0, -1, 0] = vectors.shape[-2]
        array_module = Mock(wraps=np)
        array_module.empty.side_effect = AssertionError('allocated before coordinates')
        with self.assertRaisesRegex(ValueError, 'outside'):
            get_overlap_matrix(vectors, bad, array_module=array_module)

    def test_public_row_revalidates_coordinates_and_keeps_finite_check(self):
        vectors, points = _fixture()
        row = points[0].copy()
        with patch.object(overlap, '_selected_coordinates',
                          wraps=overlap._selected_coordinates) as coordinates:
            actual = calc_overlap_row(vectors[0], row)
        self.assertEqual(coordinates.call_count, 1)
        np.testing.assert_array_equal(actual, _scalar_row(vectors[0], row, 4, 4))
        for xyz, message in (((-1, 0, 0), 'non-negative'), ((4, 0, 0), 'outside'),
                             (tuple(row[1]), 'unique')):
            changed = row.copy()
            changed[0] = xyz
            with self.subTest(xyz=xyz), self.assertRaisesRegex(ValueError, message):
                calc_overlap_row(vectors[0], changed)
        vectors[0, 0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            calc_overlap_row(vectors[0], row)

    def test_full_time_gather_matches_scalar_oracle_without_conjugation(self):
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                vectors, points = _fixture(dtype)
                vectors_before = vectors.copy()
                points_before = points.copy()
                actual = get_overlap_matrix(vectors, points, array_module=np)
                expected = _scalar_full(vectors, points, 4, 4)
                self.assertEqual(actual.shape, (3, 4, 4, 3))
                self.assertEqual(actual.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(actual, expected)
                self.assertFalse(np.shares_memory(actual, vectors))
                np.testing.assert_array_equal(vectors, vectors_before)
                np.testing.assert_array_equal(points, points_before)
                self.assertGreater(float(np.max(np.abs(actual.imag))), 1e-4)
                self.assertFalse(np.allclose(actual, expected.conj()))

    def test_equal_time_and_point_extents_keep_asymmetric_coordinate_meaning(self):
        vectors, _ = _fixture()
        vectors = np.concatenate((vectors, vectors[:1]), axis=0)
        points = np.asarray([
            [(3, 2, 1), (0, 0, 0), (1, 2, 0), (2, 1, 1)],
            [(0, 1, 1), (3, 1, 0), (2, 0, 1), (1, 2, 1)],
            [(1, 0, 0), (2, 2, 1), (0, 1, 1), (3, 0, 0)],
            [(2, 0, 1), (1, 1, 0), (3, 2, 0), (0, 2, 1)],
        ], dtype=np.int64)
        actual = get_overlap_matrix(vectors, points)
        self.assertEqual(points.shape[:2], (vectors.shape[0], points.shape[1]))
        self.assertFalse(np.array_equal(points[0, 1], points[1, 0]))
        np.testing.assert_array_equal(actual, _scalar_full(vectors, points, 4, 4))

    def test_row_gather_preserves_selected_order_and_returns_owned_values(self):
        vectors, points = _fixture()
        time = 1
        expected = _scalar_row(vectors[time], points[time], 3, 4)
        actual = calc_overlap_row(
            vectors[time], points[time], usedNe=3, usedNp=4,
            array_module=np,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (3, 4, 3))
        self.assertFalse(np.shares_memory(actual, vectors))
        self.assertFalse(np.shares_memory(actual, points))
        before = vectors.copy()
        actual[0, 0, 0] += 100
        np.testing.assert_array_equal(vectors, before)

    def test_requested_prefix_is_validated_and_unrequested_suffix_is_not_claimed(self):
        vectors, points = _fixture()
        vectors[:, 2:] = np.nan + 1j * np.nan
        points[:, 2:, :] = np.asarray((99, -1, 0), dtype=np.int32)

        actual = get_overlap_matrix(vectors, points, usedNe=2, usedNp=2)
        expected = _scalar_full(vectors, points, 2, 2)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "finite"):
            get_overlap_matrix(vectors, points, usedNp=2)
        with self.assertRaisesRegex(ValueError, "coordinates"):
            get_overlap_matrix(vectors, points, usedNe=2)

    def test_same_points_across_times_are_allowed_and_coordinates_are_physical(self):
        vectors, points = _fixture()
        repeated = np.repeat(points[:1, :, :], vectors.shape[0], axis=0)
        actual = get_overlap_matrix(vectors, repeated)
        expected = _scalar_full(vectors, repeated, 4, 4)
        np.testing.assert_array_equal(actual, expected)

    def test_counts_require_positive_non_boolean_in_range_integers(self):
        vectors, points = _fixture()
        for name, call, extent in (
            ("usedNe", lambda value: get_overlap_matrix(
                vectors, points, usedNe=value), vectors.shape[1]),
            ("usedNp", lambda value: get_overlap_matrix(
                vectors, points, usedNp=value), points.shape[1]),
        ):
            with self.subTest(name=name):
                self.assertEqual(call(np.int64(2)).shape[1 if name == "usedNe" else 2], 2)
                for value in (True, 1.5):
                    with self.subTest(value=value):
                        with self.assertRaises(TypeError):
                            call(value)
                for value in (0, -1, extent + 1):
                    with self.subTest(value=value):
                        with self.assertRaises(ValueError):
                            call(value)

    def test_rejects_invalid_vector_rank_extent_dtype_and_nonfinite_prefix(self):
        vectors, points = _fixture()
        invalid = (
            (vectors[0], "rank"),
            (vectors[..., :2], "three color"),
            (vectors.real, "complex64 or complex128"),
            (vectors.astype(np.complex64)[:, :, :, :, :, :2], "three color"),
        )
        for data, message in invalid:
            with self.subTest(shape=data.shape, message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    get_overlap_matrix(data, points)

        nonfinite = vectors.copy()
        nonfinite[1, 0, 0, 0, 0, 0] = np.inf + 0j
        with self.assertRaisesRegex(ValueError, "finite"):
            get_overlap_matrix(nonfinite, points)

        with self.assertRaisesRegex(ValueError, "time extents"):
            get_overlap_matrix(vectors, points[:2, :, :])

    def test_rejects_noninteger_boolean_duplicate_and_out_of_lattice_coordinates(self):
        vectors, points = _fixture()
        invalid = (
            (points.astype(np.float64), "integer dtype"),
            (points.astype(bool), "integer dtype"),
        )
        for data, message in invalid:
            with self.subTest(dtype=data.dtype):
                with self.assertRaisesRegex(TypeError, message):
                    get_overlap_matrix(vectors, data)

        duplicate = points.copy()
        duplicate[0, 1] = duplicate[0, 0]
        with self.assertRaisesRegex(ValueError, "unique"):
            get_overlap_matrix(vectors, duplicate)

        for xyz in ((4, 0, 0), (-1, 0, 0), (0, 3, 0), (0, 0, 2)):
            bad = points.copy()
            bad[0, 0] = xyz
            with self.subTest(xyz=xyz):
                with self.assertRaisesRegex(ValueError, "coordinates"):
                    get_overlap_matrix(vectors, bad)

        unsigned = points.astype(np.uint32)
        np.testing.assert_array_equal(
            get_overlap_matrix(vectors, unsigned), _scalar_full(vectors, unsigned, 4, 4),
        )

    def test_row_rejects_bad_spatial_extent_and_bad_point_count(self):
        vectors, points = _fixture()
        row_vectors = vectors[0]
        row_points = points[0]
        with self.assertRaisesRegex(ValueError, "outside"):
            calc_overlap_row(row_vectors, np.asarray(((0, 0, 2),), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "positive"):
            calc_overlap_row(row_vectors, row_points, usedNp=0)


if __name__ == "__main__":
    unittest.main()
