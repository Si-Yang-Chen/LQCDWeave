"""Sparse physical coordinates survive the existing dataset/recovery boundary."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.time_selection import (
    TimeSelection, TimeSelectionRecord, dataset_physical_times,
    dataset_time_selection, bind_time_selections, normalize_time_selection,
)
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter
from lqcdweave.storage.time_selection import save_time_selection, load_time_selection
from lqcdweave.sampling.temporal.api import realize_time_selection, select_t
from lqcdweave.sampling.spatial.api import SpatialSamplingDesign, sample_spatial_points
from lqcdweave.planning.laplace import laplace_basis_document, laplace_eigenvalues_document
from lqcdweave.planning.spatial_points import sampled_spatial_points_spec
from lqcdweave.adapters.production_dataset import DatasetFieldHandle, DatasetPointHandle


class SparseTimeDatasetsTests(unittest.TestCase):
    def setUp(self):
        self.selection = TimeSelectionRecord(5, (4, 0, 2))
        self.cfg = dict(ensemble='test', configuration='cfg', lattice=[2, 1, 1, 5])

    def basis_spec(self, selection=None):
        selection = self.selection if selection is None else selection
        return DatasetSpec(laplace_basis_document(self.cfg, selection.times, 'gauge', 1, 1e-8, selection=selection))

    def test_mapping_rejects_missing_and_preserves_wrap(self):
        self.assertEqual(self.selection.require_times((0, 4, 2)), (1, 0, 2))
        self.assertEqual(self.selection.physical_time(0), 4)
        self.assertEqual(TimeSelectionRecord.from_document(self.selection.document), self.selection)
        with self.assertRaises(KeyError):
            self.selection.slot(1)
        with self.assertRaises(ValueError):
            self.selection.physical_time(3)
        self.assertEqual(normalize_time_selection(None, Lt=5).times, tuple(range(5)))

    def test_definition_identity_includes_order_and_extent(self):
        spec = self.basis_spec()
        self.assertEqual(spec.shape[0], 3)
        self.assertEqual(spec.document['configuration']['lattice'][3], 5)
        self.assertEqual(dataset_physical_times(spec), (4, 0, 2))
        self.assertEqual(dataset_time_selection(spec), self.selection)
        reordered = self.basis_spec(TimeSelectionRecord(5, (0, 2, 4)))
        self.assertNotEqual(spec.definition_id, reordered.definition_id)
        malformed = spec.document
        malformed['conventions']['time_coordinates']['selections']['time']['Lt'] = 6
        with self.assertRaises(ValueError):
            DatasetSpec(malformed)
        with self.assertRaises(ValueError):
            bind_time_selections(spec.document, {'time': TimeSelectionRecord(5, (0, 2, 4))})

    def test_existing_writer_interruption_resume_and_physical_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'basis'
            spec = self.basis_spec()
            full = np.arange(5 * 6).reshape(5, 1, 1, 1, 2, 3).astype(complex)
            with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                with DatasetWriter(path, spec, provenance={'test': 'sparse'}) as writer:
                    writer.write(0, full[4])
                    raise RuntimeError('interrupt')
            with self.assertRaises(ValueError):
                with DatasetWriter(path, self.basis_spec(TimeSelectionRecord(5, (0, 2, 4))), provenance={'test': 'sparse'}, resume=True):
                    pass
            with DatasetWriter(path, spec, provenance={'test': 'sparse'}, resume=True) as writer:
                self.assertEqual(tuple(writer.missing), (1, 2))
                for slot in writer.missing:
                    writer.write(slot, full[self.selection.times[slot]])
                writer.publish()
            with DatasetReader(path) as reader:
                self.assertEqual(dataset_time_selection(reader.spec), self.selection)
                handle = DatasetFieldHandle(reader)
                self.assertEqual(handle.time_selection, self.selection)
                np.testing.assert_array_equal(np.asarray(handle.load('cfg')), full[[4, 0, 2]])
                np.testing.assert_array_equal(handle.read_modes('cfg', 0, 0, 1), full[0])
                with self.assertRaises(KeyError):
                    handle.read_modes('cfg', 1, 0, 1)

    def test_template_realization_reuses_exact_cfg_algorithm(self):
        template = TimeSelection(5, 5, (0, 1, 3))
        direct = select_t(5, 3, key=lambda value: value.offsets, cfg='cfg')
        searched = select_t(5, 3, key=lambda value: value.offsets)
        self.assertEqual(realize_time_selection(searched, cfg='cfg'), direct)
        realized = realize_time_selection(template, cfg='cfg')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.npy'
            save_time_selection(path, realized)
            self.assertEqual(load_time_selection(path, Lt=5), realized)
            with self.assertRaises(FileExistsError):
                save_time_selection(path, realized)

    def test_selected_sampling_keeps_fixed_seed_full_time_reference(self):
        design = SpatialSamplingDesign((2, 2, 1, 5), 2)
        full = sample_spatial_points(design, seed=12)
        sparse = sample_spatial_points(design, seed=12, selection=self.selection)
        np.testing.assert_array_equal(sparse.coordinates, full.coordinates[[4, 0, 2]])
        self.assertEqual(sparse.coordinates.shape, (3, 2, 3))
        spec = sampled_spatial_points_spec(design, configuration=dict(self.cfg, lattice=[2, 2, 1, 5]), seed=12, selection=self.selection)
        self.assertEqual(dataset_time_selection(spec), self.selection)
        self.assertEqual(spec.shape, (3, 2, 3))
        same = SpatialSamplingDesign((2, 2, 1, 5), 2, 'same')
        np.testing.assert_array_equal(sample_spatial_points(same, seed=12, selection=self.selection).coordinates,
                                      sample_spatial_points(same, seed=12).coordinates[[4, 0, 2]])

    def test_sampled_points_write_same_compact_record(self):
        design = SpatialSamplingDesign((2, 1, 1, 5), 1)
        spec = sampled_spatial_points_spec(design, configuration=self.cfg, seed=7, selection=self.selection)
        full = sample_spatial_points(design, seed=7)
        sampled = sample_spatial_points(design, seed=7, selection=self.selection)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "points"
            with DatasetWriter(directory, spec, provenance={"test": "points"}) as writer:
                for row in writer.missing:
                    writer.write(row, np.array(sampled.coordinates[row], dtype=np.int64, copy=True))
                writer.publish()
            with DatasetReader(directory) as reader:
                handle = DatasetPointHandle(reader)
                np.testing.assert_array_equal(np.asarray(handle.load("cfg")), full.coordinates[[4, 0, 2]])
                np.testing.assert_array_equal(handle.read_points("cfg", 0, 0, 1), full.coordinates[0])

    def test_gamma5_derived_contract_swaps_asymmetric_endpoint_records(self):
        from lqcdweave.planning.production import propagator_dataset_document, vsp_dataset_document
        source = TimeSelectionRecord(5, (4, 0))
        sink = TimeSelectionRecord(5, (2, 4, 1))
        raw = propagator_dataset_document(self.cfg, range(5), product="PSV",
            gauge_id="gauge", basis_id="basis", modes=(0,), points_id="points", point_labels=(0,),
            dirac=dict(action="wilson", mass=0.4, r=1, lattice_spacing=1, clover=0),
            solver_tolerance=1e-10, temporal_boundary="periodic",
            source_selection=source, sink_selection=sink)
        result = DatasetSpec(vsp_dataset_document(raw, psv_id="psv", gauge_id="gauge", basis_id="basis", points_id="points"))
        self.assertEqual(result.shape[:2], (3, 2))
        self.assertEqual(dataset_time_selection(result, "source_time"), sink)
        self.assertEqual(dataset_time_selection(result, "sink_time"), source)

    def test_gauge_cannot_relabel_physical_axis_as_slots(self):
        from types import SimpleNamespace
        from lqcdweave.planning.production import validate_gauge_field, validate_elemental_fields
        selection = TimeSelectionRecord(5, (4, 0, 1, 2, 3))
        axes = [("time", selection.times), ("z", [0]), ("y", [0]), ("x", [0, 1]),
                ("direction", range(4)), ("color_row", range(3)), ("color_column", range(3))]
        document = dict(schema=1, kind="gauge", configuration=self.cfg,
            axes=[dict(name=name, coordinates=list(values)) for name, values in axes],
            dtype="complex128", bases={}, definition={}, conventions={}, parents=[])
        spec = DatasetSpec(bind_time_selections(document, {"time": selection}))
        with self.assertRaisesRegex(ValueError, "gauge fields"):
            validate_gauge_field(spec)
        with self.assertRaisesRegex(ValueError, "gauge fields"):
            validate_elemental_fields(spec, self.basis_spec(), selection=self.selection)
        with self.assertRaisesRegex(ValueError, "gauge fields"):
            DatasetFieldHandle(SimpleNamespace(spec=spec, data_id="gauge"))

    def test_eigenvalue_mapping_inherited(self):
        basis = self.basis_spec()
        document = laplace_eigenvalues_document(self.cfg, self.selection.times, 1, 'basis', 'gauge', basis.document)
        values = DatasetSpec(document)
        self.assertEqual(dataset_time_selection(values), self.selection)
        self.assertEqual(values.coordinates('time'), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
