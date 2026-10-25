"""Independent small-matrix checks for LapH eigensystem production."""

import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.production.eigenvector.api import (
    LaplaceEigenvectorGenerator, spatial_laplacian,
)
from tests.fixtures.laplace_reference import GaugeHandle, dense_laplacian, random_gauge, scipy


def coordinate_stout_reference(links, nstep, rho):
    """Independent periodic-site oracle for the three spatial stout directions."""
    result = np.array(links, copy=True)
    _, lt, lz, ly, lx = result.shape[:5]
    extents = (lz, ly, lx)
    spatial_axes = (2, 1, 0)

    def shifted(point, direction, amount):
        value = list(point)
        axis = spatial_axes[direction]
        value[axis] = (value[axis] + amount) % extents[axis]
        return tuple(value)

    for _ in range(nstep):
        previous = result.copy()
        for time in range(lt):
            for z in range(lz):
                for y in range(ly):
                    for x in range(lx):
                        point = (z, y, x)
                        for mu in range(3):
                            staple = np.zeros((3, 3), dtype=complex)
                            for nu in range(3):
                                if nu == mu:
                                    continue
                                point_plus_nu = shifted(point, nu, 1)
                                point_plus_mu = shifted(point, mu, 1)
                                point_minus_nu = shifted(point, nu, -1)
                                point_minus_nu_plus_mu = shifted(point_minus_nu, mu, 1)

                                staple += (
                                    previous[(nu, time, *point)]
                                    @ previous[(mu, time, *point_plus_nu)]
                                    @ previous[(nu, time, *point_plus_mu)].conj().T
                                )
                                staple += (
                                    previous[(nu, time, *point_minus_nu)].conj().T
                                    @ previous[(mu, time, *point_minus_nu)]
                                    @ previous[(nu, time, *point_minus_nu_plus_mu)]
                                )

                            omega = rho * staple @ previous[(mu, time, *point)].conj().T
                            q = 0.5j * (omega.conj().T - omega)
                            q -= np.trace(q) * np.eye(3) / 3
                            values, vectors = np.linalg.eigh(q)
                            exponential = (vectors * np.exp(1j * values)[None, :]) @ vectors.conj().T
                            result[(mu, time, *point)] = exponential @ previous[(mu, time, *point)]
    return result


@unittest.skipIf(scipy is None, "SciPy is required for Laplace eigensystem tests")
class LaplaceEigenvectorTests(unittest.TestCase):
    def setUp(self):
        self.latt_size = (2, 2, 2, 2)
        self.gauge = GaugeHandle(random_gauge(self.latt_size))

    def test_public_api_preserves_class_identity(self):
        from lqcdweave.production import LaplaceEigenvectorGenerator as PublicGenerator
        self.assertIs(PublicGenerator, LaplaceEigenvectorGenerator)

    def test_low_modes_match_independent_dense_matrix(self):
        generator = LaplaceEigenvectorGenerator(self.latt_size, self.gauge, 3, 1e-11,
                                                backend="numpy")
        generator.load("cfg")
        self.assertEqual(self.gauge.calls, 1)
        self.assertIs(generator._links.base, None)
        self.assertIsNotNone(self.gauge.data)
        for t in range(2):
            vectors, values = generator.calc(t)
            links = self.gauge.data[t, :, :, :, :3].transpose(3, 0, 1, 2, 4, 5)
            dense = dense_laplacian(links)
            reference = np.linalg.eigvalsh(dense)[:3]
            np.testing.assert_allclose(values, reference, atol=1e-9)
            self.assertEqual(vectors.shape, (3, 2, 2, 2, 3))
            np.testing.assert_allclose(vectors.reshape(3, -1).conj() @ vectors.reshape(3, -1).T,
                                       np.eye(3), atol=1e-8)
            for mode in range(3):
                np.testing.assert_allclose(dense @ vectors[mode].reshape(-1),
                                           values[mode] * vectors[mode].reshape(-1), atol=1e-8)
        self.assertEqual(generator.describe()["gauge_id"], "synthetic:gauge")

    def test_operator_matches_dense_for_multiple_rhs(self):
        links = self.gauge.data[0, :, :, :, :3].transpose(3, 0, 1, 2, 4, 5)
        dense = dense_laplacian(links)
        field = np.arange(48).reshape(2, 2, 2, 3, 2).astype(complex)
        expected = (dense @ field.reshape(24, 2)).reshape(field.shape)
        np.testing.assert_allclose(spatial_laplacian(field, links), expected)

    def test_degenerate_unit_links_return_valid_zero_modes(self):
        links = np.broadcast_to(np.eye(3), (2, 2, 2, 2, 4, 3, 3)).astype(complex).copy()
        generator = LaplaceEigenvectorGenerator(self.latt_size, GaugeHandle(links), 2, 1e-10,
                                                backend="numpy").load("unit")
        vectors, values = generator.calc(0)
        np.testing.assert_allclose(values, 0, atol=1e-9)
        operator = dense_laplacian(generator._links[:3, 0])
        for mode in range(2):
            np.testing.assert_allclose(operator @ vectors[mode].reshape(-1),
                                       values[mode] * vectors[mode].reshape(-1), atol=1e-9)

    def test_spatial_transforms_preserve_temporal_links_and_su3(self):
        raw = self.gauge.data.copy()
        raw[..., :3, :, :] *= 1.08
        self.gauge.data = raw
        generator = LaplaceEigenvectorGenerator(self.latt_size, self.gauge, 2, 1e-9,
                                                backend="numpy").load("cfg")
        temporal = generator._links[3].copy()
        generator.project_SU3()
        spatial = generator._links[:3]
        np.testing.assert_allclose(np.linalg.det(spatial), 1, atol=1e-12)
        np.testing.assert_allclose(spatial @ spatial.conj().swapaxes(-1, -2),
                                   np.broadcast_to(np.eye(3), spatial.shape), atol=1e-12)
        generator.stout_smear(1, 0.1)
        np.testing.assert_allclose(generator._links[3], temporal)
        vectors, values = generator.calc(0)
        self.assertEqual(vectors.shape[0], 2)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(len(generator.describe()["transforms"]), 2)

    def test_stout_matches_independent_coordinate_oracle_and_preserves_inputs(self):
        latt_size = (3, 2, 1, 2)
        raw = random_gauge(latt_size, seed=31)
        raw_snapshot = raw.copy()
        for nstep in (1, 3):
            with self.subTest(nstep=nstep):
                handle = GaugeHandle(raw)
                generator = LaplaceEigenvectorGenerator(
                    latt_size, handle, 2, 1e-9, backend="numpy"
                ).load("cfg")
                before = generator._links.copy()
                expected = coordinate_stout_reference(
                    before, nstep, 0.073
                )

                generator.stout_smear(nstep, 0.073)

                np.testing.assert_allclose(generator._links, expected, atol=3e-13, rtol=3e-13)
                np.testing.assert_array_equal(generator._links[3], before[3])
                np.testing.assert_array_equal(handle.data, raw_snapshot)
                self.assertFalse(np.shares_memory(generator._links, handle.data))
                self.assertEqual(generator.describe()["transforms"][-1],
                                 {"kind": "stout", "nstep": nstep, "rho": 0.073})

    def test_failed_in_place_stout_invalidates_loaded_links(self):
        generator = LaplaceEigenvectorGenerator(
            self.latt_size, self.gauge, 2, 1e-9, backend="numpy"
        ).load("cfg")

        def fail_after_partial_write(links, nstep, rho, **kwargs):
            links[0, 0, 0, 0, 0, 0, 0] = 0
            raise RuntimeError("synthetic transform failure")

        with patch(
            "lqcdweave.production.eigenvector.generator.stout_smear",
            side_effect=fail_after_partial_write,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic transform failure"):
                generator.stout_smear(1, 0.1)

        self.assertIsNone(generator._links)
        self.assertFalse(generator.describe()["loaded"])
        with self.assertRaises(RuntimeError):
            generator.calc(0)

    def test_validation_and_failed_load_reset(self):
        with self.assertRaises(ValueError):
            LaplaceEigenvectorGenerator(self.latt_size, self.gauge, 23, 1e-9,
                                        backend="numpy")
        generator = LaplaceEigenvectorGenerator(self.latt_size, self.gauge, 2, 1e-9,
                                                backend="numpy")
        with self.assertRaises(RuntimeError):
            generator.calc(0)
        generator.load("cfg")
        for time in (-1, 2, True):
            with self.assertRaises(ValueError):
                generator.calc(time)
        with self.assertRaises(ValueError):
            generator.calc(0, polynomial_degree=10)
        with self.assertRaises(NotImplementedError):
            generator.calc(0, polynomial_degree=10, lambda_cut=1.0)
        class CupyMarker:
            pass
        CupyMarker.__name__ = "cupy"
        generator.xp = CupyMarker
        with self.assertRaisesRegex(NotImplementedError, "requires an explicit QudaLaplaceBackend"):
            generator.stout_smear(1, 0.1)
        self.assertTrue(generator.describe()["loaded"])
        generator.xp = np
        self.gauge.data = self.gauge.data[..., :2, :2]
        with self.assertRaises(ValueError):
            generator.load("bad")
        self.assertFalse(generator.describe()["loaded"])


if __name__ == "__main__":
    unittest.main()
