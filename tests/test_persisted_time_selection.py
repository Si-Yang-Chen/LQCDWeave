"""TS production -> NPY storage -> later reconstruction, using independent pairs."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from lqcdweave.sampling.temporal.api import select_t, reconstruct_two_point_from_coordinates
from lqcdweave.storage.time_selection import load_time_selection, save_time_selection


class PersistedTimeSelectionTests(unittest.TestCase):
    def test_cfg_selection_saved_and_reloaded_before_independent_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            for cfg in (2000, 2010):
                path = Path(directory) / f'{cfg}.npy'
                produced = select_t(8, 4, key=lambda s: s.offsets, cfg=cfg)
                save_time_selection(path, produced)
                selection = load_time_selection(path, Lt=8)
                np.testing.assert_array_equal(selection.times, produced.times)
                data = np.arange(16).reshape(4, 4) + 1j * np.arange(16, 32).reshape(4, 4)
                result = reconstruct_two_point_from_coordinates(data, selection, time_layout='cyclic_difference')
                for dt in range(8):
                    values = []
                    for i, source in enumerate(selection.times):
                        for k, sink in enumerate(selection.times):
                            if (sink - source) % 8 == dt:
                                values.append(data[i, (k - i) % 4])
                    self.assertEqual(result.counts[dt], len(values))
                    self.assertAlmostEqual(result.values[dt], sum(values) / len(values))


if __name__ == '__main__':
    unittest.main()
