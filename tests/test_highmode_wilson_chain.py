"""Nine physical-time pairs of Q_sink S Q_source against a dense Wilson oracle."""

from unittest.mock import patch
import unittest

import numpy as np

from test_perambulator_dataset import _PerambulatorDatasetFixture
from lqcdweave.contraction.propagator_views import (
    project_psp_highmode,
    project_psv_highmode,
    project_vsp_highmode,
)
from lqcdweave.production.eigenvector.api import calc_overlap_row


def _orthonormal_vectors(instance, *, seed, count):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(3):
        matrix = rng.normal(size=(6, count)) + 1j * rng.normal(size=(6, count))
        columns, _ = np.linalg.qr(matrix)
        np.testing.assert_allclose(columns.conj().T @ columns, np.eye(count), atol=1e-14)
        rows.append(columns.T.reshape(count, 1, 1, 2, 3))
    return np.stack(rows)


class HighmodeWilsonOracleTests(_PerambulatorDatasetFixture):
    def setUp(self):
        with patch.object(_PerambulatorDatasetFixture, "_basis_values",
                          _orthonormal_vectors):
            super().setUp()

    def _overlap_rows(self, basis_id, points_id):
        with self.index.open(basis_id) as basis, self.index.open(points_id) as points:
            return {
                time: calc_overlap_row(basis.read(time=time), points.read(time=time))
                for time in range(3)
            }

    def test_projected_sectors_match_full_space_complements_at_every_time_pair(self):
        products = ("VSV", "PSV", "VSP", "PSP")
        preview = self._plan_products(products,
            basis_src_id=self.source_basis_id, basis_snk_id=self.sink_basis_id,
            points_src_id=self.source_points_id, points_snk_id=self.sink_points_id,
            Ne_src=3, Ne_snk=2, Np_src=1, Np_snk=2)
        raw = self._produce_products(preview, products)
        source_overlap = self._overlap_rows(self.source_basis_id, self.source_points_id)
        sink_overlap = self._overlap_rows(self.sink_basis_id, self.sink_points_id)

        dimension = self.inverse.shape[0]
        identity = np.eye(dimension, dtype=np.complex128)
        with self.index.open(raw["vsv"]) as vsv_reader, \
             self.index.open(raw["psv"]) as psv_reader, \
             self.index.open(raw["vsp"]) as vsp_reader, \
             self.index.open(raw["psp"]) as psp_reader:
            for reader in (vsv_reader, psv_reader, vsp_reader, psp_reader):
                self.assertEqual(reader.spec.coordinates("source_time"), [0, 1, 2])
                self.assertEqual(reader.spec.coordinates("sink_time"), [0, 1, 2])

            for source_time in range(3):
                source_vectors = self._basis_columns(
                    self.source_basis_values, source_time, 3)
                source_points = self._point_columns(
                    self.source_point_rows, source_time, 1)
                q_source = identity - source_vectors @ source_vectors.conj().T
                for sink_time in range(3):
                    sink_vectors = self._basis_columns(
                        self.sink_basis_values, sink_time, 2)
                    sink_points = self._point_columns(
                        self.sink_point_rows, sink_time, 2)
                    q_sink = identity - sink_vectors @ sink_vectors.conj().T

                    raw_vsv = vsv_reader.read(source_time=source_time,
                                              sink_time=sink_time)
                    raw_psv = psv_reader.read(source_time=source_time,
                                              sink_time=sink_time)
                    raw_vsp = vsp_reader.read(source_time=source_time,
                                              sink_time=sink_time)
                    raw_psp = psp_reader.read(source_time=source_time,
                                              sink_time=sink_time)

                    actual = {
                        "psv": project_psv_highmode(
                            raw_vsv, raw_psv, sink_overlap[sink_time]),
                        "vsp": project_vsp_highmode(
                            raw_vsv, raw_vsp, source_overlap[source_time]),
                        "psp": project_psp_highmode(
                            raw_vsv, raw_psv, raw_vsp, raw_psp,
                            sink_overlap[sink_time], source_overlap[source_time]),
                    }
                    expected = {
                        "psv": (sink_points.conj().T @ q_sink @ self.inverse @ source_vectors)
                            .reshape(4, 2, 3, 4, 3).transpose(0, 3, 1, 2, 4),
                        "vsp": (sink_vectors.conj().T @ self.inverse @ q_source @ source_points)
                            .reshape(4, 2, 4, 1, 3).transpose(0, 2, 1, 3, 4),
                        "psp": (sink_points.conj().T @ q_sink @ self.inverse @ q_source @ source_points)
                            .reshape(4, 2, 3, 4, 1, 3).transpose(0, 3, 1, 2, 4, 5),
                    }
                    for product in expected:
                        with self.subTest(product=product, source=source_time,
                                          sink=sink_time):
                            np.testing.assert_allclose(
                                actual[product], expected[product],
                                atol=3e-12, rtol=3e-12,
                            )


if __name__ == "__main__":
    unittest.main()
