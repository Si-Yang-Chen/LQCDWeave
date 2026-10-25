"""Object sampling and canonical point handles through real PR consumption."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from lqcdweave.sampling.spatial import (
    sample_spatial_points, SpatialSamplingDesign,
)
from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.contracts.spatial_sampling import SpatialPointSet
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.adapters.legacy_fields import DenseFileHandle
from lqcdweave.adapters.dense_dirac import DenseDiracBackend
from lqcdweave.production import PerambulatorGenerator
from lqcdweave.production.perambulator.api import PointBasis
from lqcdweave.preset import save
from lqcdweave.planning.spatial_points import (
    sampled_spatial_points_spec, validate_complete_spatial_point_set,
)
from lqcdweave.storage.dataset import DatasetReader, DatasetWriter


class SpatialFrontendTests(unittest.TestCase):
    def test_shared_dataset_selects_each_cfg_without_changing_points_or_identity(self):
        design = SpatialSamplingDesign((3, 2, 1, 5), 3)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "points"
            expected = {}
            for cfg, seed in (("2000", 73), ("2004", 19)):
                points = sample_spatial_points(design, seed=seed)
                spec = sampled_spatial_points_spec(design, configuration={
                    "ensemble": "test", "configuration": cfg,
                    "lattice": list(design.lattice_size)}, seed=seed)
                with DatasetWriter(directory, spec, provenance={}) as writer:
                    for slot in writer.missing:
                        writer.write(slot, points.coordinates[slot])
                    identity = writer.publish()
                expected[cfg] = (points, identity)
            with self.assertRaises(KeyError):
                SpatialPointHandle.from_dataset(directory)
            for cfg, (points, identity) in expected.items():
                handle = SpatialPointHandle.from_dataset(directory, cfg=cfg)
                self.assertEqual(handle.cfg, cfg)
                self.assertEqual(handle.data_id, identity)
                self.assertEqual(handle.points.seed, points.seed)
                np.testing.assert_array_equal(handle.points.coordinates, points.coordinates)

    def test_sampler_returns_time_first_coordinates(self):
        for policy in ('same', 'independent'):
            design = SpatialSamplingDesign((3, 2, 4, 5), 8, policy)
            points = sample_spatial_points(design, seed=81)
            self.assertEqual(points.coordinates.shape, (5, 8, 3))
            self.assertEqual(points.coordinates.dtype, np.dtype('int64'))
            self.assertFalse(points.coordinates.flags.writeable)
        np.testing.assert_array_equal(
            sample_spatial_points(SpatialSamplingDesign((3, 2, 4, 5), 8), seed=81).coordinates,
            sample_spatial_points(SpatialSamplingDesign((3, 2, 4, 5), 8, 'independent'), seed=81).coordinates)
        self.assertEqual(sample_spatial_points(SpatialSamplingDesign((1, 1, 1, 3), 1)).coordinates.shape,
                         (3, 1, 3))

    def test_npy_roundtrip_and_handle_ownership(self):
        design = SpatialSamplingDesign((3, 2, 4, 5), 8)
        points = sample_spatial_points(design, seed=81)
        handle = SpatialPointHandle(points, cfg='c1')
        with self.assertRaises(ValueError):
            handle.load('c2')
        with self.assertRaises(ValueError):
            handle.load('c1').flags.writeable = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'seed123_same.npy'
            save(path, points.coordinates, kind='points', overwrite=True)
            imported = SpatialPointHandle.from_npy(path, lattice_size=design.lattice_size)
            np.testing.assert_array_equal(imported.load('anything'), handle.load('c1'))
            self.assertIsNone(imported.points.design)
            self.assertIsNone(imported.points.seed)
            self.assertIsNone(imported.points.algorithm)
            path.unlink()  # copied input does not require an open file
            self.assertEqual(imported.Np, 8)
        selected = SpatialPointHandle(type(points)(points.lattice_size, (1,), points.coordinates[:1]))
        self.assertEqual(selected.time_selection, TimeSelectionRecord(5, (1,)))
        np.testing.assert_array_equal(selected.read_points('anything', 1), points.coordinates[0])
        with self.assertRaises(KeyError):
            selected.read_points('anything', 0)

    def test_equal_time_and_point_extents_keep_asymmetric_time_first_coordinates(self):
        # Equal extents cannot reveal an axis swap from shape alone.
        lattice = (3, 2, 2, 2)
        coordinates = np.asarray(
            [[[0, 0, 0], [2, 1, 1]], [[1, 0, 1], [0, 1, 0]]], dtype=np.int64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'equal-extents.npy'
            save(path, coordinates, kind='points')
            restored = SpatialPointHandle.from_npy(path, lattice_size=lattice)
            self.assertEqual(restored.load('cfg').shape, (2, 2, 3))
            np.testing.assert_array_equal(restored.load('cfg'), coordinates)
            np.testing.assert_array_equal(restored.read_points('cfg', 0), coordinates[0])
            np.testing.assert_array_equal(restored.read_points('cfg', 1), coordinates[1])
            self.assertFalse(np.array_equal(coordinates[0], coordinates[1]))

    def test_dataset_roundtrip_retains_explicit_provenance(self):
        for policy in ('same', 'independent'):
            design = SpatialSamplingDesign((3, 2, 4, 5), 8, policy)
            points = sample_spatial_points(design, seed=81)
            spec = sampled_spatial_points_spec(design, configuration={
                'ensemble': 'test', 'configuration': 'c1', 'lattice': [3,2,4,5]}, seed=81)
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "dataset"
                with DatasetWriter(directory, spec, provenance={'numpy_version': np.__version__}) as writer:
                    for row in writer.missing:
                        writer.write(row, points.coordinates[row])
                    writer.publish()
                handle = SpatialPointHandle.from_dataset(directory)
                self.assertEqual(handle.points.design, design)
                self.assertEqual(handle.points.seed, 81)
                self.assertEqual(handle.points.algorithm, points.algorithm)
                self.assertEqual(handle.provenance, {'numpy_version': np.__version__})
                with DatasetReader(directory) as reader:
                    self.assertEqual(handle.source_origin, reader.source_origin)
                np.testing.assert_array_equal(handle.load('c1'), points.coordinates)
                with self.assertRaises(ValueError):
                    handle.load('wrong')

    def test_imported_dataset_keeps_unknown_history_and_npy_checks_geometry(self):
        design = SpatialSamplingDesign((3, 2, 4, 5), 8)
        sampled = sample_spatial_points(design, seed=81)
        # These coordinates came from a sampler, but this imported caller set
        # intentionally carries no declared design or generation history.
        points = SpatialPointSet(sampled.lattice_size, sampled.times, sampled.coordinates)
        configuration = {'ensemble': 'test', 'configuration': 'c1', 'lattice': [3,2,4,5]}
        spec = validate_complete_spatial_point_set(points, configuration=configuration)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'dataset'
            with DatasetWriter(directory, spec, provenance={'note': 'imported'}) as writer:
                for row in writer.missing:
                    writer.write(row, points.coordinates[row])
                writer.publish()
            handle = SpatialPointHandle.from_dataset(directory)
            self.assertIsNone(handle.points.seed)
            self.assertIsNone(handle.points.design)
            self.assertIsNone(handle.points.algorithm)
            self.assertEqual(handle.provenance, {'note': 'imported'})
            np.testing.assert_array_equal(handle.points.coordinates, points.coordinates)
            path = Path(tmp) / 'points.npy'
            save(path, points.coordinates, kind='points')
            with self.assertRaises(ValueError):
                SpatialPointHandle.from_npy(path, lattice_size=(3,2,4,6))

    def test_sampled_caller_dataset_roundtrip_retains_sampling_metadata(self):
        design = SpatialSamplingDesign((3, 2, 4, 6), 8, temporal_policy=2)
        points = sample_spatial_points(design, seed=81)
        configuration = {
            'ensemble': 'test', 'configuration': 'caller-cfg',
            'lattice': list(design.lattice_size),
        }
        spec = validate_complete_spatial_point_set(points, configuration=configuration)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'caller-dataset'
            with DatasetWriter(directory, spec, provenance={'note': 'caller sampled set'}) as writer:
                for row in writer.missing:
                    writer.write(row, points.coordinates[row])
                writer.publish()

            restored = SpatialPointHandle.from_dataset(directory, cfg='caller-cfg')
            self.assertEqual(restored.points.design, design)
            self.assertEqual(restored.points.seed, points.seed)
            self.assertEqual(restored.points.algorithm, points.algorithm)
            self.assertEqual(restored.points.pool_ids, points.pool_ids)
            self.assertEqual(restored.points.sampling_groups, points.sampling_groups)
            self.assertEqual(restored.provenance, {'note': 'caller sampled set'})
            np.testing.assert_array_equal(restored.points.coordinates, points.coordinates)

    def test_local_dataset_pool_ids_are_stable_per_cfg_and_segment(self):
        design = SpatialSamplingDesign((3, 2, 4, 6), 8, temporal_policy=2)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'local-dataset'
            for cfg, seed in (('cfg-a', 81), ('cfg-b', 82)):
                points = sample_spatial_points(design, seed=seed)
                configuration = {
                    'ensemble': 'test', 'configuration': cfg,
                    'lattice': list(design.lattice_size),
                }
                spec = sampled_spatial_points_spec(
                    design, configuration=configuration, seed=seed,
                )
                with DatasetWriter(directory, spec, provenance={}) as writer:
                    for row in writer.missing:
                        writer.write(row, points.coordinates[row])
                    writer.publish()

            first = SpatialPointHandle.from_dataset(directory, cfg='cfg-a')
            repeated = SpatialPointHandle.from_dataset(directory, cfg='cfg-a')
            other = SpatialPointHandle.from_dataset(directory, cfg='cfg-b')
            self.assertEqual(first.data_id, repeated.data_id)
            self.assertNotEqual(first.data_id, other.data_id)
            self.assertEqual(first.points.design, design)
            self.assertEqual(first.points.seed, 81)
            self.assertEqual(first.points.algorithm, 'numpy.PCG64/partial-fisher-yates-v1')
            self.assertEqual(other.points.seed, 82)

            for handle in (first, repeated, other):
                pool_ids = handle.points.pool_ids
                self.assertEqual(len(pool_ids), design.lattice_size[3])
                self.assertEqual(pool_ids[0], pool_ids[1])
                self.assertEqual(pool_ids[2], pool_ids[3])
                self.assertEqual(pool_ids[4], pool_ids[5])
                self.assertEqual(len({pool_ids[0], pool_ids[2], pool_ids[4]}), 3)
            self.assertEqual(first.points.pool_ids, repeated.points.pool_ids)
            self.assertTrue(set(first.points.pool_ids).isdisjoint(other.points.pool_ids))

    def test_real_perambulator_accepts_object_and_saved_point_handles(self):
        sizes = (2, 1, 1, 2)
        points = sample_spatial_points(SpatialSamplingDesign(sizes, 2), seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.save(path / 'gauge.npy', 2 * np.eye(48, dtype=complex))
            save(path / 'points.npy', points.coordinates, kind='points')
            handles = [SpatialPointHandle(points), SpatialPointHandle.from_npy(
                path / 'points.npy', lattice_size=sizes)]
            results = []
            for handle in handles:
                gauge_file = DenseFileHandle(path / 'gauge', '.npy', format='npy')
                generator = PerambulatorGenerator(sizes,
                    gauge_file,
                    mass=.2, tol=1e-12, maxiter=10,
                    backend=DenseDiracBackend())
                try:
                    generator.load('')
                    source=PointBasis(handle)
                    sink=PointBasis(handle)
                    result = generator.calc(0,src=source,sink=[sink])[0].values.copy()
                    results.append(result)
                    expected = np.zeros_like(result)
                    # Identity dense Dirac inverse: same source time, spin, color and site.
                    for spin in range(4):
                        for p in range(2):
                            for color in range(3):
                                expected[0, spin, spin, p, color, p, color] = .5
                    np.testing.assert_allclose(result, expected)
                finally:
                    gauge = generator.gauge_field_smear
                    if gauge is not None:
                        gauge._mmap.close()
            np.testing.assert_array_equal(*results)


    def test_selected_point_handles_preserve_wrapped_order_through_files(self):
        design = SpatialSamplingDesign((3,2,1,5),3)
        selection = TimeSelectionRecord(5,(4,0,2))
        points = sample_spatial_points(design,seed=73,selection=selection)
        full = sample_spatial_points(design,seed=73)
        np.testing.assert_array_equal(points.coordinates,full.coordinates[[4,0,2]])
        original = SpatialPointHandle(points,cfg='sparse')
        self.assertEqual(original.load('sparse').shape,(3,3,3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'points.npy'
            save(path,points.coordinates,kind='points',overwrite=True)
            restored = SpatialPointHandle.from_npy(path,lattice_size=design.lattice_size,
                cfg='sparse',time_selection=selection)
            self.assertEqual(restored.time_selection,selection)
            np.testing.assert_array_equal(restored.load('sparse'),original.load('sparse'))
            with self.assertRaises(ValueError):
                SpatialPointHandle.from_npy(path,lattice_size=design.lattice_size)
            spec = sampled_spatial_points_spec(design,configuration={
                'ensemble':'test','configuration':'sparse','lattice':list(design.lattice_size)},
                seed=73,selection=selection)
            directory = Path(tmp)/'dataset'
            with DatasetWriter(directory,spec,provenance={'numpy_version':np.__version__}) as writer:
                for slot in writer.missing: writer.write(slot,points.coordinates[slot])
                writer.publish()
            restored = SpatialPointHandle.from_dataset(directory)
            self.assertEqual(restored.time_selection,selection)
            self.assertEqual(restored.points.times,(4,0,2))
            self.assertEqual(restored.points.design,design)
            self.assertEqual(restored.basis_id,restored.data_id)
            for time in selection.times:
                np.testing.assert_array_equal(restored.read_points('sparse',time),full.coordinates[time])


if __name__ == '__main__':
    unittest.main()
