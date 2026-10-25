"""Coordinate batching preserves request order and records true tail sizes."""

import unittest

from lqcdweave.contracts.correlator.api import BatchSpec
from lqcdweave.contraction.expansion.api import (
    group_coordinate_batches, describe_coordinates,
)


class CoordinateBatchTests(unittest.TestCase):
    def test_named_axis_groups_by_outer_coordinates_and_keeps_real_tail(self):
        coordinates, batch = describe_coordinates(
            time_axes={"source": (2, 0), "sink": (4, 1, 3)},
            batch={"axis": "sink", "size": 2},
        )
        groups = group_coordinate_batches(coordinates, batch)
        self.assertEqual([item.point_indices for item in groups],
                         [(0, 1), (2,), (3, 4), (5,)])
        self.assertEqual([item.outer_values for item in groups], [
            (("source", 2),), (("source", 2),),
            (("source", 0),), (("source", 0),),
        ])
        self.assertEqual([item.point_keys for item in groups], [
            ("0", "1"), ("2",), ("3", "4"), ("5",),
        ])
        self.assertEqual([item.actual_size for item in groups], [2, 1, 2, 1])
        self.assertEqual([item.batch_size for item in groups], [2, 2, 2, 2])
        self.assertEqual([item.is_tail for item in groups], [False, True, False, True])

    def test_pairs_keep_unsorted_order_across_batches(self):
        pairs = (
            {"source": 3, "sink": 1},
            {"source": 0, "sink": 4},
            {"source": 3, "sink": 0},
        )
        coordinates, batch = describe_coordinates(time_pairs=pairs, batch=2)
        groups = group_coordinate_batches(coordinates, batch)
        self.assertEqual([item.axis for item in groups], ["pair", "pair"])
        self.assertEqual([item.point_indices for item in groups], [(0, 1), (2,)])
        self.assertEqual([item.points[0].values for item in groups], [
            (("source", 3), ("sink", 1)),
            (("source", 3), ("sink", 0)),
        ])
        self.assertEqual(groups[1].actual_size, 1)
        self.assertTrue(groups[1].is_tail)

    def test_auto_size_accepts_planner_resolution_without_padding(self):
        coordinates, batch = describe_coordinates(
            time_axes={"sink": (7, 2, 5, 1, 4)}, batch="auto",
        )
        groups = group_coordinate_batches(coordinates, batch, resolved_size=2)
        self.assertEqual([item.point_keys for item in groups], [
            ("0", "1"), ("2", "3"), ("4",),
        ])
        self.assertEqual([item.actual_size for item in groups], [2, 2, 1])
        self.assertEqual(groups[-1].batch_size, 2)

    def test_unspecified_axis_batches_the_flattened_scan_order(self):
        coordinates, batch = describe_coordinates(
            time_axes={"source": (4, 1), "sink": (3, 0, 2)},
            batch="auto",
        )
        groups = group_coordinate_batches(coordinates, batch, resolved_size=4)
        self.assertEqual([item.point_indices for item in groups], [(0, 1, 2, 3), (4, 5)])
        self.assertTrue(all(item.axis is None for item in groups))
        self.assertEqual([point.values for point in groups[0].points], [
            (("source", 4), ("sink", 3)),
            (("source", 4), ("sink", 0)),
            (("source", 4), ("sink", 2)),
            (("source", 1), ("sink", 3)),
        ])

    def test_fixed_coordinates_form_one_nonpadded_batch(self):
        coordinates, batch = describe_coordinates(time_axes={"source": 3, "sink": 6})
        (item,) = group_coordinate_batches(coordinates, batch)
        self.assertIsNone(item.axis)
        self.assertEqual(item.point_indices, (0,))
        self.assertEqual(item.actual_size, item.batch_size)
        self.assertFalse(item.is_tail)

    def test_invalid_axis_or_batch_override_fails_before_binding(self):
        coordinates, batch = describe_coordinates(
            time_axes={"source": (0, 1), "sink": (0, 1)}, batch="auto",
        )
        with self.assertRaisesRegex(ValueError, "explicit axis"):
            group_coordinate_batches(coordinates, BatchSpec(2))
        with self.assertRaisesRegex(ValueError, "not a scanned"):
            group_coordinate_batches(coordinates, BatchSpec(2, "missing"))
        with self.assertRaisesRegex(ValueError, "cannot override"):
            group_coordinate_batches(coordinates, BatchSpec(2, "sink"), resolved_size=1)


if __name__ == "__main__":
    unittest.main()
