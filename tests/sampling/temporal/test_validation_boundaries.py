"""Public geometry checks and checked internal reductions use one scan each."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.sampling.temporal import reconstruction as temporal


def _pair_oracle(data, differences, Lt, valid=None):
    sums = np.zeros(data.shape[:-2] + (Lt,), dtype=np.complex128)
    counts = np.zeros(Lt, dtype=np.int64)
    for row in range(differences.shape[0]):
        for column in range(differences.shape[1]):
            if valid is None or valid[row, column]:
                dt = int(differences[row, column])
                sums[..., dt] += data[..., row, column]
                counts[dt] += 1
    means = np.full(sums.shape, np.nan, dtype=np.complex128)
    for dt in range(Lt):
        if counts[dt]:
            means[..., dt] = sums[..., dt] / counts[dt]
    return means, counts


class TemporalValidationBoundaryTests(unittest.TestCase):
    def test_count_and_mean_each_validate_external_geometry_once(self):
        geometry = np.asarray(((0, 1, 3), (2, 0, 1)), dtype=np.int64)
        data = (np.arange(12).reshape(2, 2, 3) + 1j).astype(np.complex64)
        valid = np.asarray(((True, False, True), (True, True, True)))
        expected, counts = _pair_oracle(data, geometry, 5, valid)
        with patch.object(temporal, '_coordinates', wraps=temporal._coordinates) as scan:
            counted = temporal.count_time_pairs(geometry, Lt=5, valid=valid)
        self.assertEqual(scan.call_count, 1)
        np.testing.assert_array_equal(counted, counts)
        with patch.object(temporal, '_coordinates', wraps=temporal._coordinates) as scan:
            actual = temporal.mean_time_pairs(
                data, geometry, Lt=5, source_axis=1, other_axis=2,
                valid=valid, missing='nan',
            )
        self.assertEqual(scan.call_count, 1)
        np.testing.assert_array_equal(actual.counts, counts)
        np.testing.assert_allclose(actual.values, expected, equal_nan=True)
        self.assertEqual(actual.values.dtype, np.dtype('complex128'))
        self.assertEqual(actual.dt_axis, 1)

    def test_reconstruction_scans_pair_geometry_once_for_all_coordinate_frontends(self):
        sources = np.asarray((5, 0, 3, 1), dtype=np.int64)
        geometry = np.asarray([[(int(t) - int(s)) % 8 for t in sources] for s in sources])
        data = np.arange(32).reshape(2, 4, 4) + 0.25j
        expected, counts = _pair_oracle(data, geometry, 8)
        cyclic = np.empty_like(data)
        for row in range(4):
            for difference in range(4):
                cyclic[:, row, difference] = data[:, row, (row + difference) % 4]
        record = TimeSelectionRecord(8, tuple(int(t) for t in sources))
        calls = (
            lambda: temporal.reconstruct_two_point_from_coordinates(
                data, Lt=8, tsrc_times=sources, dt_values=geometry, tsrc_axis=1, dt_axis=2),
            lambda: temporal.reconstruct_two_point_from_coordinates(
                data, record, tsrc_axis=1, dt_axis=2),
            lambda: temporal.reconstruct_two_point_from_coordinates(
                cyclic, record, tsrc_axis=1, dt_axis=2, time_layout='cyclic_difference'),
            lambda: temporal.reconstruct_two_point(
                cyclic, Lt=8, selected_times=sources, tsrc_axis=1, dt_axis=2),
        )
        for entry, call in enumerate(calls):
            with self.subTest(entry=entry), patch.object(
                temporal, '_coordinates', wraps=temporal._coordinates,
            ) as scan:
                result = call()
            geometry_scans = [item for item in scan.call_args_list
                              if np.asarray(item.args[0]).ndim == 2]
            self.assertEqual(len(geometry_scans), 1)
            np.testing.assert_allclose(result.values, expected)
            np.testing.assert_array_equal(result.counts, counts)

    def test_public_coordinate_errors_remain_errors_after_successful_call(self):
        data = np.ones((2, 2))
        geometry = np.asarray(((0, 1), (1, 0)))
        for operation in (
            lambda values: temporal.count_time_pairs(values, Lt=2),
            lambda values: temporal.mean_time_pairs(
                data, values, Lt=2, source_axis=0, other_axis=1),
            lambda values: temporal.reconstruct_two_point_from_coordinates(
                data, Lt=2, tsrc_times=[0, 1], dt_values=values),
        ):
            operation(geometry)
            for bad in (np.asarray(((-1, 1), (1, 0))),
                        np.asarray(((0, 2), (1, 0))), geometry.astype(float),
                        geometry.astype(bool)):
                with self.subTest(operation=operation, geometry=bad):
                    with self.assertRaisesRegex(ValueError, 'integer coordinates'):
                        operation(bad)

    def test_public_mask_axis_dtype_and_missing_errors_remain_checked(self):
        geometry = np.asarray(((0, 1), (1, 0)))
        data = np.ones((2, 2))
        for mask in (np.ones((2, 2), dtype=int), np.ones((1, 2), dtype=bool)):
            with self.assertRaisesRegex(ValueError, 'boolean and match geometry'):
                temporal.count_time_pairs(geometry, Lt=2, valid=mask)
        for changes in ({'source_axis': True}, {'other_axis': 0}, {'other_axis': 2},
                        {'data': np.asarray([['a', 'b'], ['c', 'd']])},
                        {'valid': np.ones((1, 2), dtype=bool)}, {'missing': 'zero'},
                        {'differences': geometry[:, :1]}):
            kwargs = dict(data=data, differences=geometry, Lt=2, source_axis=0, other_axis=1)
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                temporal.mean_time_pairs(**kwargs)
        with self.assertRaisesRegex(ValueError, 'no source pairs'):
            temporal.mean_time_pairs(data, geometry, Lt=3, source_axis=0, other_axis=1)


if __name__ == '__main__':
    unittest.main()
