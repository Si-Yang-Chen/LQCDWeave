"""Absolute compact slots use independently enumerated physical pairs."""
import unittest
import numpy as np
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.sampling.temporal.api import reconstruct_two_point_from_coordinates


class SlotReconstructionTests(unittest.TestCase):
    def test_absolute_default_wrap_order_asymmetric_and_spectator_axes(self):
        source = TimeSelectionRecord(8, (6, 0, 1, 2))
        sink = TimeSelectionRecord(8, (1, 6, 3))
        full = np.arange(2*8*3*8).reshape(2, 8, 3, 8) + 1j
        sparse = full[:, source.times, :, :][:, :, :, sink.times]
        result = reconstruct_two_point_from_coordinates(sparse, source, sink_selection=sink,
            tsrc_axis=1, dt_axis=3, missing='nan')
        expected = np.full((2, 3, 8), np.nan, complex)
        counts = []
        for dt in range(8):
            pairs = [(s, t) for s in source.times for t in sink.times if (t-s)%8 == dt]
            counts.append(len(pairs))
            if pairs:
                expected[..., dt] = sum(full[:, s, :, t] for s, t in pairs)/len(pairs)
        np.testing.assert_allclose(result.values, expected, equal_nan=True)
        np.testing.assert_array_equal(result.counts, counts)
        self.assertEqual(result.dt_axis, 2)

    def test_explicit_cyclic_matches_absolute_but_is_not_physical_dt(self):
        record = TimeSelectionRecord(8, (6, 0, 1, 2))
        absolute = np.arange(16).reshape(4, 4) + 1j*np.arange(16, 32).reshape(4, 4)
        cyclic = np.array([[absolute[i, (i+j)%4] for j in range(4)] for i in range(4)])
        expected = reconstruct_two_point_from_coordinates(absolute, record)
        actual = reconstruct_two_point_from_coordinates(cyclic, record, time_layout='cyclic_difference')
        np.testing.assert_allclose(actual.values, expected.values)
        implicit = reconstruct_two_point_from_coordinates(cyclic, record)
        self.assertFalse(np.allclose(implicit.values, expected.values))
        with self.assertRaisesRegex(ValueError, 'identical ordered'):
            reconstruct_two_point_from_coordinates(cyclic, record,
                sink_selection=TimeSelectionRecord(8, (0, 6, 1, 2)), time_layout='cyclic_difference')
