"""Known periodic signals and independently enumerated source-pair means."""

import unittest

import numpy as np

from lqcdweave.sampling.temporal.api import (
    reconstruct_two_point as reconstruct_sparse,
    reconstruct_two_point_from_coordinates as reconstruct_two_point,
    TimeSelection,
    TimeSelectionRecord,
)


class TwoPointTests(unittest.TestCase):
    def test_saved_cfg_wraparound_order_matches_independent_physical_pairs(self):
        # The ordered integer table in CSNS 01.select_t5x8/2000.npy.
        times = (104, 105, 108, 113, 115, 120, 121, 124, 1, 3, 8, 9, 12, 17,
                 19, 24, 25, 28, 33, 35, 40, 41, 44, 49, 51, 56, 57, 60,
                 65, 67, 72, 73, 76, 81, 83, 88, 89, 92, 97, 99)
        selection = TimeSelectionRecord(128, times)
        rng = np.random.default_rng(314)
        data = rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))
        before = data.copy()
        result = reconstruct_two_point(data, selection, time_layout='cyclic_difference')
        sums = np.zeros(128, dtype=np.complex128)
        counts = np.zeros(128, dtype=np.int64)
        for source_slot, source_time in enumerate(times):
            for sink_slot, sink_time in enumerate(times):
                dt = (sink_time - source_time) % 128
                sums[dt] += data[source_slot, (sink_slot - source_slot) % 40]
                counts[dt] += 1
        np.testing.assert_allclose(result.values, sums / counts)
        np.testing.assert_array_equal(result.counts, counts)
        np.testing.assert_array_equal(data, before)
        self.assertEqual(selection.times, times)
        # Sorting the record without moving its data changes the estimator.
        sorted_result = reconstruct_two_point(data, TimeSelectionRecord(128, tuple(sorted(times))), time_layout='cyclic_difference')
        self.assertFalse(np.allclose(sorted_result.values, result.values))

    def test_selection_frontend_40_slots_matches_physical_pair_oracle(self):
        selection = TimeSelection(128, 16, (0, 1, 4, 9, 11))
        rng = np.random.default_rng(17)
        data = rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))
        before = data.copy()
        result = reconstruct_two_point(data, selection, time_layout='cyclic_difference')
        positions = {time: i for i, time in enumerate(selection.times)}
        expected = []
        counts = []
        for physical_dt in range(selection.Lt):
            pairs = []
            for source_time, i in positions.items():
                sink_time = (source_time + physical_dt) % selection.Lt
                if sink_time in positions:
                    j = (positions[sink_time] - i) % selection.Nt
                    pairs.append(data[i, j])
            counts.append(len(pairs))
            expected.append(sum(pairs) / len(pairs))
        np.testing.assert_allclose(result.values, expected)
        np.testing.assert_array_equal(result.counts, counts)
        np.testing.assert_array_equal(data, before)
        self.assertEqual(result.dt_axis, 0)

    def test_selection_frontend_preserves_all_spectator_axis_placements(self):
        selection = TimeSelection(8, 8, (0, 1, 3, 5))
        rng = np.random.default_rng(21)
        original = rng.normal(size=(2, 3, 4, 4))
        expected = reconstruct_two_point(original, selection, tsrc_axis=2, dt_axis=3, time_layout='cyclic_difference')
        for source_axis in range(4):
            for difference_axis in range(4):
                if source_axis == difference_axis:
                    continue
                data = np.moveaxis(original, (-2, -1), (source_axis, difference_axis))
                result = reconstruct_two_point(
                    data, selection=selection,
                    tsrc_axis=source_axis - 4, dt_axis=difference_axis - 4,
                 time_layout='cyclic_difference')
                np.testing.assert_allclose(
                    result.values, np.moveaxis(expected.values, -1, result.dt_axis),
                )
                np.testing.assert_array_equal(result.counts, expected.counts)

    def test_full_selection_reduces_to_logical_source_mean(self):
        selection = TimeSelection(5, 1, (0,))
        data = np.arange(25).reshape(5, 5)
        result = reconstruct_two_point(data, selection, time_layout='cyclic_difference')
        np.testing.assert_allclose(result.values, data.mean(axis=0))
        np.testing.assert_array_equal(result.counts, np.full(5, 5))

    def test_selection_frontend_missing_policy(self):
        selection = TimeSelection(8, 8, (0, 1))
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'no source pairs'):
            reconstruct_two_point(data, selection, time_layout='cyclic_difference')
        result = reconstruct_two_point(data, selection, missing='nan', time_layout='cyclic_difference')
        np.testing.assert_allclose(
            result.values, [2, 2, np.nan, np.nan, np.nan, np.nan, np.nan, 4],
            equal_nan=True,
        )
        np.testing.assert_array_equal(result.counts, [2, 1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(result.values.dtype, np.dtype('float64'))

    def test_selection_frontend_rejects_ambiguous_or_invalid_inputs(self):
        selection = TimeSelection(8, 8, (0, 1, 3, 5))
        data = np.zeros((4, 4))
        for coordinates in (
            {'Lt': 8}, {'tsrc_times': selection.times},
            {'dt_values': range(4)}, {'sink_times': selection.times},
        ):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                reconstruct_two_point(data, selection, **coordinates, time_layout='cyclic_difference')
        for invalid in (False, selection.times, {'Lt': 8, 'times': selection.times}):
            with self.subTest(selection=invalid), self.assertRaises(TypeError):
                reconstruct_two_point(data, invalid, time_layout='cyclic_difference')
        with self.assertRaises(TypeError):
            reconstruct_two_point(data)
        for shape in ((3, 4), (4, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                reconstruct_two_point(np.zeros(shape), selection, time_layout='cyclic_difference')
        for axes in ({'tsrc_axis': True}, {'dt_axis': 2}, {'dt_axis': 0}):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                reconstruct_two_point(data, selection, **axes, time_layout='cyclic_difference')

    def test_sparse_index_difference_40_by_40_to_128(self):
        # Keep an unsorted cyclic order, as in shifted production selections.
        times = (np.array(TimeSelection(128, 16, (0, 1, 4, 9, 11)).times) + 101) % 128
        data = np.empty((40, 40), dtype=complex)
        buckets = [[] for _ in range(128)]
        for i in range(40):
            for j in range(40):
                dt = (int(times[(i+j) % 40]) - int(times[i])) % 128
                value = 2 * dt + i + 1j * (dt - i)
                data[i, j] = value
                buckets[dt].append(value)
        result = reconstruct_sparse(data, Lt=128, selected_times=times, tsrc_axis=0, dt_axis=1)
        self.assertEqual(result.values.shape, (128,))
        np.testing.assert_allclose(result.values, [sum(b)/len(b) for b in buckets])
        np.testing.assert_array_equal(result.counts, [len(b) for b in buckets])
        # dt before source, with an extra operator axis.
        moved = np.stack([data.T, 2 * data.T], axis=1)
        result = reconstruct_sparse(moved, Lt=128, selected_times=times, tsrc_axis=-1, dt_axis=0)
        np.testing.assert_allclose(result.values[:, 1], 2 * result.values[:, 0])
        self.assertEqual(result.values.shape, (128, 2))
        with self.assertRaises(ValueError):
            reconstruct_sparse(data[:, :-1], Lt=128, selected_times=times, tsrc_axis=0, dt_axis=1)

    def test_full_time_mean_with_arbitrary_axes_and_complex_data(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(2, 5, 3, 5)) + 1j * rng.normal(size=(2, 5, 3, 5))
        result = reconstruct_two_point(data, Lt=5, tsrc_axis=1, dt_axis=-1,
                                      tsrc_times=range(5), dt_values=range(5))
        np.testing.assert_allclose(result.values, data.mean(axis=1))
        np.testing.assert_array_equal(result.counts, np.full(5, 5))
        self.assertEqual(result.dt_axis, 2)
        transposed = data.transpose(3, 2, 0, 1)
        result = reconstruct_two_point(transposed, Lt=5, tsrc_axis=-1, dt_axis=0,
                                      tsrc_times=range(5), dt_values=range(5))
        np.testing.assert_allclose(result.values, transposed.mean(axis=-1))
        self.assertEqual(result.dt_axis, 0)

    def test_sparse_pairs_and_unsorted_coordinates(self):
        Lt = 8
        sources = np.array([5, 0, 3, 1])
        # Columns here are arbitrary stored sink coordinates; explicitly convert to dt.
        sinks = np.array([1, 5, 0, 3])
        differences = (sinks[None, :] - sources[:, None]) % Lt
        data = sources[:, None] + 10 * differences + 1j * differences
        result = reconstruct_two_point(data, Lt=Lt, tsrc_axis=0, dt_axis=1,
                                      tsrc_times=sources, dt_values=differences)
        for d in range(Lt):
            entries = [s + 10*d + 1j*d for s in sources if (s+d) % Lt in sinks]
            self.assertEqual(result.counts[d], len(entries))
            self.assertAlmostEqual(result.values[d], sum(entries)/len(entries))
        self.assertGreater(len(set(result.counts)), 1)

    def test_sink_filter_matches_sparse_storage_and_preserves_spectator_axes(self):
        sources = np.array([5, 0, 3, 1])
        differences = np.arange(8)
        full = sources[:, None] + differences[None, :] * 10
        data = np.stack([full, full + 100], axis=1)
        result = reconstruct_two_point(data, Lt=8, tsrc_axis=0, dt_axis=2,
                                      tsrc_times=sources, dt_values=differences, sink_times=sources)
        mapping = (sources[None, :] - sources[:, None]) % 8
        sparse = sources[:, None] + mapping * 10
        expected = reconstruct_two_point(sparse, Lt=8, tsrc_axis=0, dt_axis=1,
                                        tsrc_times=sources, dt_values=mapping)
        np.testing.assert_allclose(result.values[0], expected.values)
        np.testing.assert_allclose(result.values[1], expected.values + 100)
        np.testing.assert_array_equal(result.counts, expected.counts)

    def test_missing_is_not_silently_zero_filled(self):
        kwargs = dict(Lt=4, tsrc_axis=0, dt_axis=1, tsrc_times=[0], dt_values=[0, 2])
        with self.assertRaisesRegex(ValueError, 'no source pairs'):
            reconstruct_two_point([[4, 8]], **kwargs)
        result = reconstruct_two_point([[4, 8]], missing='nan', **kwargs)
        np.testing.assert_allclose(result.values, [4, np.nan, 8, np.nan], equal_nan=True)
        np.testing.assert_array_equal(result.counts, [1, 0, 1, 0])

    def test_known_signal_reconstructed_for_every_axis_placement(self):
        sources = np.array([0, 1, 3, 5])
        dt = (sources[None, :] - sources[:, None]) % 8
        signal = np.exp(-np.arange(8)) + np.exp(-(8 - np.arange(8)))
        original = np.broadcast_to(signal[dt], (2, 3, 4, 4)).copy()
        for src_axis in range(4):
            for dt_axis in range(4):
                if src_axis == dt_axis:
                    continue
                data = np.moveaxis(original, (-2, -1), (src_axis, dt_axis))
                before = data.copy()
                result = reconstruct_two_point(data, Lt=8, tsrc_axis=src_axis, dt_axis=dt_axis,
                                              tsrc_times=sources, dt_values=dt)
                expected = np.moveaxis(np.broadcast_to(signal, (2, 3, 8)), -1, result.dt_axis)
                np.testing.assert_allclose(result.values, expected)
                np.testing.assert_array_equal(data, before)

    def test_validation(self):
        kwargs = dict(Lt=4, tsrc_axis=0, dt_axis=1, tsrc_times=[0, 1], dt_values=range(4))
        data = np.zeros((2, 4))
        for change in [dict(tsrc_axis=1), dict(dt_axis=2), dict(tsrc_axis=True),
                       dict(tsrc_times=[0, 0]), dict(tsrc_times=[0]),
                       dict(dt_values=[0, 1, 1, 3]), dict(dt_values=[0, 1, 2, 4]),
                       dict(dt_values=[0., 1., 2., 3.]), dict(sink_times=[0, 0]),
                       dict(missing='zero'), dict(Lt=0)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconstruct_two_point(data, **(kwargs | change))


if __name__ == '__main__':
    unittest.main()
