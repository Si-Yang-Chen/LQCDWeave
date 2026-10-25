"""NPY metadata and spatial-point adapter handoff to the public producer."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lqcdweave import preset
from lqcdweave.adapters.dense_dirac import DenseDiracBackend
from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.contracts.spatial_sampling import SpatialPointSet
from lqcdweave.production import PerambulatorGenerator
from lqcdweave.production.perambulator.api import PointBasis, VectorBasis


class MatrixHandle:
    def load(self, key):
        return 2 * np.eye(24, dtype=complex)


class PropagatorFrontendTests(unittest.TestCase):
    def test_npy_metadata_resolved_before_construction(self):
        with TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'vectors_')
            values = np.arange(12).reshape(2, 2, 1, 1, 1, 3).astype(complex) / 5
            np.save(prefix + 'cfg.npy', values)
            for declared_shape, stream in ((None, False), (None, True),
                                            (values.shape, False), (values.shape, True)):
                with self.subTest(shape=declared_shape, stream=stream):
                    basis = preset.EigenvectorNpy(prefix, shape=declared_shape)
                    basis_input=VectorBasis(basis)
                    options = dict(latt_size=(1, 1, 1, 2), gauge_field=MatrixHandle(),
                                   mass=0, tol=1e-12, maxiter=20, backend=DenseDiracBackend(),
                                   stream_eigenvectors=stream)
                    gen = PerambulatorGenerator(**options)
                    gen.load('cfg')
                    if declared_shape is None:
                        with self.assertRaisesRegex(ValueError, 'Ne is unknown; load'):
                            gen.calc(1,src=basis_input,sink=[basis_input])
                        basis.load('cfg')
                    self.assertEqual(basis.Ne, 2)
                    result=gen.calc(1,src=basis_input,sink=[basis_input])[0]
                    actual=result.values
                    expected = np.zeros_like(actual)
                    for spin in range(4):
                        expected[1, spin, spin] = values[1].reshape(2, 3).conj() @ values[1].reshape(2, 3).T / 2
                    np.testing.assert_allclose(actual, expected, atol=1e-14)
                    basis._loaded._mmap.close()

    def test_npy_declared_axes_infer_count_and_header_still_validates(self):
        with TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'vectors_')
            shape = (3, 2, 1, 1, 1, 3)
            axes = ('mode', 'time', 'z', 'y', 'x', 'color')
            basis = preset.EigenvectorNpy(prefix, shape=shape, axis_order=axes)
            self.assertEqual(basis.Ne, 3)
            np.save(prefix + 'cfg.npy', np.ones(shape, complex))
            loaded = basis.load('cfg')
            self.assertEqual(loaded.shape, (2, 3, 1, 1, 1, 3))
            loaded._mmap.close()
            np.save(prefix + 'bad.npy', np.ones((2, 2, 1, 1, 1, 3), complex))
            with self.assertRaisesRegex(ValueError, 'shape'):
                basis.load('bad')

    def test_spatial_handle_vsp_only_without_source_modes(self):
        points = SpatialPointHandle(SpatialPointSet(
            (1, 1, 1, 2), (0, 1), np.zeros((2, 1, 3), dtype=np.int32)), cfg='cfg')
        with TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'v_')
            values = np.ones((2, 1, 1, 1, 1, 3), complex)
            np.save(prefix + 'cfg.npy', values)
            basis = preset.EigenvectorNpy(prefix, shape=values.shape)
            src_basis=PointBasis(points)
            sink_basis=VectorBasis(basis,count=1)
            gen = PerambulatorGenerator((1, 1, 1, 2), MatrixHandle(), 0, 1e-12, 20,
                backend=DenseDiracBackend())
            gen.load('cfg')
            vsp = gen.calc(1,src=src_basis,sink=[sink_basis])[0].values
            expected = np.zeros_like(vsp)
            for spin in range(4):
                expected[1, spin, spin, 0, 0, :] = .5
            np.testing.assert_allclose(vsp, expected)
            gen.load('other')
            self.assertTrue(gen.describe()['loaded'])
            basis._loaded._mmap.close()
