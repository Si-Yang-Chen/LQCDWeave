"""Indexed raw V/V Elemental budget admission and identity checks."""

import unittest
import weakref

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.memory import MemoryEstimate
from lqcdweave.planning.production import (
    _elemental_phase_cache_peak_bytes,
    elemental_dataset_document,
    estimate_elemental_memory,
)
from lqcdweave.production.elemental.transport import momentum_phase
from lqcdweave.storage.dataset import DatasetIndex
from test_production_dataset import _ProductionDatasetFixture



class ElementalProductionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _ProductionDatasetFixture("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.index = DatasetIndex(self.fixture.root / "dataset-index")
        self.gauge_id = self.index.register(self.fixture.root / "gauge")
        self.basis_id = self.index.register(self.fixture.root / "basis")
        self.paths = ((), (0,))
        self.momenta = ((0, 0, 0),)

    def _metadata_specs_and_output(self, *, paths=None, momenta=None, used_ne=2):
        gauge = self.index.observe_metadata(self.gauge_id)["spec"]
        basis = self.index.observe_metadata(self.basis_id)["spec"]
        selected_paths = self.paths if paths is None else paths
        selected_momenta = self.momenta if momenta is None else momenta
        modes = basis.coordinates("mode")[:used_ne]
        output = DatasetSpec(elemental_dataset_document(
            gauge, basis, gauge_id=self.gauge_id, basis_id=self.basis_id,
            components=[dict(path=list(path)) for path in selected_paths],
            momenta=selected_momenta, modes=modes, phase_convention="anchor"))
        return {self.gauge_id: gauge, self.basis_id: basis}, output

    def test_none_phase_miss_counts_live_indices_and_complex_ones(self):
        class NumpyAllocationProbe:
            def __init__(self):
                self.indices_ref = None
                self.overlap_bytes = None

            def indices(self, shape):
                coordinates = np.indices(shape)
                self.indices_ref = weakref.ref(coordinates)
                return coordinates

            def ones(self, shape, *, dtype):
                coordinates = self.indices_ref()
                self.assert_live_coordinates = coordinates is not None
                array = np.ones(shape, dtype=dtype)
                # momentum_phase has unpacked z/y/x but has not returned yet;
                # their shared indices base and this result are simultaneously
                # live at this allocation point.
                self.overlap_bytes = coordinates.nbytes + array.nbytes
                return array

        probe = NumpyAllocationProbe()
        phase = momentum_phase(
            (3, 2, 2, 3), (1, -1, 2), convention="none", array_module=probe)
        volume = 3 * 2 * 2
        self.assertTrue(probe.assert_live_coordinates)
        self.assertEqual(probe.overlap_bytes, (3 * 8 + 16) * volume)
        self.assertEqual(probe.overlap_bytes, 40 * volume)
        self.assertEqual(phase.shape, (2, 2, 3))
        self.assertEqual(phase.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(phase, np.ones((2, 2, 3), dtype=np.complex128))

        # At the 64th distinct key, 63 cached arrays overlap construction;
        # after saturation, a miss overlaps all 64 old entries.
        self.assertEqual(_elemental_phase_cache_peak_bytes(volume, 1, "none"),
                         40 * volume)
        self.assertEqual(_elemental_phase_cache_peak_bytes(volume, 64, "none"),
                         (63 * 16 + 40) * volume)
        self.assertEqual(_elemental_phase_cache_peak_bytes(volume, 65, "none"),
                         (64 * 16 + 40) * volume)

    def test_hand_calculated_known_lifetimes_and_full_complex128_output_row(self):
        input_specs, output = self._metadata_specs_and_output()
        host, device = estimate_elemental_memory(
            input_specs, output, gauge_id=self.gauge_id, basis_id=self.basis_id,
            ne_block_size=1, target="cpu")

        # This fixture is (Lx,Ly,Lz,Lt)=(3,2,2,3), Ne=3, complex128.
        # Both complete parent payloads are mapped: gauge=20,736 B and basis=
        # 5,184 B. Workspace.load overlaps the 20,736 B full gauge stack with
        # its 20,736 B owned complex128 links. That phase is the largest known
        # phase; row assembly is smaller for this fixture.
        self.assertEqual(host.input_bytes, 25_920)
        self.assertEqual(host.snapshot_bytes, 20_736)
        self.assertEqual(host.intermediate_bytes, 20_736)
        self.assertEqual(host.output_bytes, 0)
        self.assertEqual(host.logical_peak_bytes, 67_392)
        self.assertEqual(device, MemoryEstimate())

        # Production emits all components/momenta/mode pairs in one
        # (component,momentum,sink_mode,source_mode) complex128 row.
        self.assertEqual(output.shape, (3, 2, 1, 2, 2))
        row_bytes = 2 * 1 * 2 * 2 * np.dtype(np.complex128).itemsize
        self.assertEqual(row_bytes, 128)
        estimate, _ = estimate_elemental_memory(
            input_specs, output, gauge_id=self.gauge_id, basis_id=self.basis_id,
            ne_block_size=1, target="cpu")
        self.assertEqual(estimate.logical_peak_bytes, 67_392)


if __name__ == "__main__":
    unittest.main()
