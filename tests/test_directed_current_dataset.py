"""Registered format adaptation and physical compatibility checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave.adapters.directed_current_dataset import load_registered_directed_current
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.current import DirectedCurrentBasis
from lqcdweave.contraction.directed_current import contract_directed_current_pair_v2v
from lqcdweave.operators.single_hadron.current import ConservedVectorInsertion
from lqcdweave.planning.directed_current import validate_directed_current_datasets
from lqcdweave.storage.dataset import DatasetReader, register_npy


def axis(name, coordinates):
    return dict(name=name, coordinates=list(coordinates))


class DirectedCurrentDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cfg = dict(ensemble="test", configuration="cfg", lattice=[1, 1, 1, 3])
        self.basis = dict(data_id="actual-basis", indices=[5, 2])
        self.momenta = [[1, 0, 0], [0, 0, 0]]
        self.components = [{"path": []}] + [{"path": [direction]} for direction in reversed(range(8))]
        self.current_document = dict(
            schema=1, kind="elemental_vv", configuration=self.cfg,
            axes=[axis("time", [2, 0, 1]), axis("component", self.components),
                  axis("momentum", self.momenta), axis("sink_mode", [5, 2]),
                  axis("source_mode", [5, 2])],
            dtype="complex128", bases=dict(sink_mode=self.basis, source_mode=self.basis),
            definition=dict(gauge_id="elemental-gauge", path_boundary="periodic",
                            directed_basis=DirectedCurrentBasis.SCHEMA),
            conventions=dict(phase="midpoint", normalization="unscaled"),
            parents=["elemental-gauge", "actual-basis"],
        )
        self.vsv_document = dict(
            schema=1, kind="vsv", configuration=self.cfg,
            axes=[axis("source_time", [1, 2, 0]), axis("sink_time", [2, 0, 1]),
                  axis("sink_spin", range(4)), axis("source_spin", range(4)),
                  axis("sink_mode", [5, 2]), axis("source_mode", [5, 2])],
            dtype="complex128", bases=dict(sink_mode=self.basis, source_mode=self.basis),
            definition=dict(gauge_id="dirac-gauge"),
            conventions=dict(spin_basis="degrand_rossi", temporal_boundary="periodic",
                             normalization="unscaled"),
            parents=["dirac-gauge", "actual-basis"],
        )

    def _registered(self):
        values = np.empty((3, len(self.components), 2, 2, 2), dtype=np.complex128)
        for ti, time in enumerate([2, 0, 1]):
            for di, component in enumerate(self.components):
                for mi in range(2):
                    direction = component["path"][0] if component["path"] else -1
                    values[ti, di, mi] = (direction + 10*time + 100*mi + 1j) * np.eye(2)
        vsv = np.ones((3, 3, 4, 4, 2, 2), dtype=np.complex128)
        np.save(self.root / "current.npy", values)
        np.save(self.root / "vsv.npy", vsv)
        register_npy(self.root / "current", self.root / "current.npy",
                     DatasetSpec(self.current_document), provenance={"test": True})
        register_npy(self.root / "vsv", self.root / "vsv.npy",
                     DatasetSpec(self.vsv_document), provenance={"test": True})

    def test_registered_named_axes_are_reordered_and_contracted(self):
        self._registered()
        with DatasetReader(self.root / "current") as current, DatasetReader(self.root / "vsv") as vsv:
            adapted = load_registered_directed_current(current, vsv)
            self.assertEqual(adapted["raw"]["v2v"].shape, (8, 3, 2, 2, 2))
            self.assertEqual(adapted["momenta"], self.momenta)
            self.assertEqual(adapted["basis"], self.basis)
            for direction in range(8):
                for time in range(3):
                    for momentum in range(2):
                        np.testing.assert_array_equal(
                            adapted["raw"]["v2v"][direction, time, momentum],
                            (direction + 10*time + 100*momentum + 1j) * np.eye(2))
            insertion = ConservedVectorInsertion(3, momentum_index=1)
            result = contract_directed_current_pair_v2v(
                insertion, adapted["raw"], adapted["contract"],
                insertion, adapted["raw"], adapted["contract"], adapted["vsv"][0],
                first_anchor_time=0, second_anchor_time=1,
                first_field_ne=2, first_bar_ne=2,
                second_field_ne=2, second_bar_ne=2)
            self.assertTrue(np.isfinite(result["value"]))
            self.assertEqual(result["momenta"], {"first": 1, "second": 1})

    def test_unknown_or_incompatible_metadata_is_rejected(self):
        current = DatasetSpec(self.current_document)
        vsv = DatasetSpec(self.vsv_document)
        self.assertEqual(validate_directed_current_datasets(current, vsv)["basis"], self.basis)
        for field, value, pattern in (
            ("conventions", dict(phase="anchor", normalization="unscaled"), "midpoint"),
            ("definition", dict(gauge_id="elemental-gauge", path_boundary="periodic"), "directed basis"),
            ("axes", [axis("time", [2, 0, 1]), axis("component", self.components[:-1]),
                      axis("momentum", self.momenta), axis("sink_mode", [5, 2]),
                      axis("source_mode", [5, 2])], "eight"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                validate_directed_current_datasets(
                    DatasetSpec({**self.current_document, field: value}), vsv)
        wrong_basis = dict(self.basis, data_id="other-basis")
        wrong_vsv = DatasetSpec({**self.vsv_document,
                                  "bases": dict(sink_mode=wrong_basis, source_mode=wrong_basis)})
        with self.assertRaisesRegex(ValueError, "basis identity"):
            validate_directed_current_datasets(current, wrong_vsv)
        with self.assertRaisesRegex(ValueError, "VSV temporal boundary"):
            validate_directed_current_datasets(
                current, DatasetSpec({**self.vsv_document, "conventions":
                    dict(spin_basis="degrand_rossi", temporal_boundary="open",
                         normalization="unscaled")}))
        with self.assertRaisesRegex(ValueError, "unscaled"):
            validate_directed_current_datasets(
                current, DatasetSpec({**self.vsv_document, "conventions":
                    dict(spin_basis="degrand_rossi", temporal_boundary="periodic",
                         normalization="unknown")}))

    def test_registered_data_is_checked_against_shape_and_dtype(self):
        self._registered()
        path = self.root / "current.npy"
        np.save(path, np.zeros((1,), dtype=np.complex128), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            DatasetReader(self.root / "current")


if __name__ == "__main__":
    unittest.main()
