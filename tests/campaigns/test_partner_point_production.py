"""CPU checks for the C24P29 partner point-source production recipe."""
from __future__ import annotations

import builtins
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lqcdweave.adapters.dense_dirac import DenseDiracBackend
from lqcdweave.adapters.spatial_points import SpatialPointHandle
from lqcdweave.contracts.propagator import PropagatorEndpoint
from lqcdweave.contracts.time_selection import TimeSelectionRecord
from lqcdweave.production.perambulator.api import PropagatorReader
from lqcdweave.storage.dataset import DatasetReader

from tools.campaigns import partner_point_production as campaign


def _fixture_manifest(root: Path) -> dict:
    return {
        "schema": 1,
        "ensemble": "C24P29",
        "lattice": [2, 1, 1, 4],
        "cfgs": ["10000"],
        "root": str(root),
        "gauge_pattern": "/test/gauge/cfg_{cfg}.lime",
        "legacy_vsv_root": "/test/legacy-basis",
        "basis_entries": {"10000": []},
        "basis_ne": 2,
        "sampling": {"num_points": 2, "temporal_policy": 4, "seed_base": 20261011},
        "dirac": {
            "mass": -0.2, "tol": 1e-12, "maxiter": 5000,
            "xi_0": 1.0, "nu": 1.0,
            "clover_coeff_t": 1.0, "clover_coeff_r": 1.0,
            "t_boundary": -1, "multigrid": None,
            "stout": {"nstep": 1, "rho": 0.125, "dir_ignore": 4},
        },
        "storage": {"PSP_time_triangle": "sink>=source", "dtype": "complex128"},
    }


def _gamma5_hermitian_matrix(lattice, seed=4321):
    lx, ly, lz, lt = lattice
    shape = (lt, lz, ly, lx, 4, 3)
    size = int(np.prod(shape))
    rng = np.random.default_rng(seed)
    random = (rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size)))
    random /= np.sqrt(2 * size)
    hermitian = random.conj().T @ random + 2.0*np.eye(size)
    site_count = lt*lz*ly*lx
    gamma_diagonal = np.tile(np.repeat([1.0, 1.0, -1.0, -1.0], 3), site_count)
    return gamma_diagonal[:, None] * hermitian


def _flat(lattice, time, x, spin, color):
    lx, ly, lz, lt = lattice
    return np.ravel_multi_index((time, 0, 0, x, spin, color),
                                (lt, lz, ly, lx, 4, 3))


class _FixtureBasis:
    def __init__(self, lattice):
        self.lattice = lattice
        self.Ne = 2
        self.basis_id = "fixture-basis-ne2"
        self.endpoint = PropagatorEndpoint("V", self.basis_id, range(self.Ne))
        self.time_selection = TimeSelectionRecord(lattice[3], tuple(range(lattice[3])))
        lx, ly, lz, lt = lattice
        self.values = np.zeros((lt, 2, lz, ly, lx, 3), dtype=np.complex128)
        self.values[:, 0, 0, 0, 0, 0] = 1
        self.values[:, 1, 0, 0, 1, 1] = 1

    def load(self, key):
        return self.values

    def read_modes(self, key, time, start, stop):
        return self.values[time, start:stop]


class _FixtureGauge:
    def __init__(self, matrix, gauge_id):
        self.matrix = matrix
        self.gauge_id = gauge_id
        self.smear_calls = []

    def load(self, key):
        return self.matrix


class _FixtureBackend(DenseDiracBackend):
    def __init__(self, *, max_dimension=2048):
        super().__init__(max_dimension=max_dimension)
        self.rhs_total = 0
        self.smear_calls = []

    def stout_smear(self, gauge, *, nstep, rho, dir_ignore):
        self.smear_calls.append((nstep, rho, dir_ignore))
        return gauge

    def solve_into(self, rhs, solution, *, tol, maxiter):
        self.rhs_total += len(rhs)
        return super().solve_into(rhs, solution, tol=tol, maxiter=maxiter)


def _publish_fixture_points(manifest):
    campaign.prepare_campaign(manifest, campaign_profile=False)
    return SpatialPointHandle.from_dataset(
        Path(manifest["root"]) / "inputs" / "points", cfg="10000")


def _reference_psp(inverse, lattice, points, sink_time, source_time):
    result = np.empty((4, 4, points.Np, 3, points.Np, 3), dtype=np.complex128)
    for sink_spin in range(4):
        for source_spin in range(4):
            for sink_point, sink_xyz in enumerate(points.points.coordinates[sink_time]):
                for sink_color in range(3):
                    sink_index = np.ravel_multi_index(
                        (sink_time, 0, 0, int(sink_xyz[0]), sink_spin, sink_color),
                        (lattice[3], lattice[2], lattice[1], lattice[0], 4, 3))
                    for source_point, source_xyz in enumerate(points.points.coordinates[source_time]):
                        for source_color in range(3):
                            source_index = np.ravel_multi_index(
                                (source_time, 0, 0, int(source_xyz[0]), source_spin, source_color),
                                (lattice[3], lattice[2], lattice[1], lattice[0], 4, 3))
                            result[sink_spin, source_spin, sink_point, sink_color,
                                   source_point, source_color] = inverse[sink_index, source_index]
    return result


def _reference_psv(inverse, lattice, points, basis, point_time, mode_time):
    result = np.empty((4, 4, points.Np, 3, basis.Ne), dtype=np.complex128)
    vectors = basis.values[mode_time]
    for sink_spin in range(4):
        for source_spin in range(4):
            for sink_point, sink_xyz in enumerate(points.points.coordinates[point_time]):
                for sink_color in range(3):
                    sink_index = np.ravel_multi_index(
                        (point_time, 0, 0, int(sink_xyz[0]), sink_spin, sink_color),
                        (lattice[3], lattice[2], lattice[1], lattice[0], 4, 3))
                    for mode in range(basis.Ne):
                        total = 0j
                        for x in range(lattice[0]):
                            for color in range(3):
                                source_index = np.ravel_multi_index(
                                    (mode_time, 0, 0, x, source_spin, color),
                                    (lattice[3], lattice[2], lattice[1], lattice[0], 4, 3))
                                total += inverse[sink_index, source_index] * vectors[mode, 0, 0, x, color]
                        result[sink_spin, source_spin, sink_point, sink_color, mode] = total
    return result


class PartnerPointProductionTests(unittest.TestCase):
    def test_quda_single_rhs_residual_reads_only_active_slot(self):
        maximum = 1e-6
        self.assertEqual(
            campaign._first_active_true_residual(9.544279e-8, maximum=maximum),
            9.544279e-8,
        )
        padded = np.full(144, np.nan, dtype=np.float64)
        padded[0] = 9.544279e-8
        self.assertEqual(
            campaign._first_active_true_residual(padded, maximum=maximum),
            9.544279e-8,
        )
        with self.assertRaisesRegex(RuntimeError, "active true residual"):
            campaign._first_active_true_residual(
                np.concatenate(([1.01e-6], np.zeros(143))), maximum=maximum)
        with self.assertRaisesRegex(RuntimeError, "active true residual"):
            campaign._first_active_true_residual(np.array([np.nan, 0.0]), maximum=maximum)
        with self.assertRaisesRegex(RuntimeError, "empty true residual"):
            campaign._first_active_true_residual(np.array([], dtype=np.float64), maximum=maximum)

    def test_solver_accuracy_is_provenance_not_physical_dirac(self):
        manifest = _fixture_manifest(Path("/unused"))
        alternate = dict(manifest)
        alternate["dirac"] = {
            **manifest["dirac"], "tol": 1e-7, "maxiter": 9000,
            "multigrid": {"setup": "alternate-solver"},
        }
        points = types.SimpleNamespace(data_id="fixture-points")
        basis = types.SimpleNamespace(basis_id="fixture-basis")

        physical = campaign._dirac_definition(manifest)
        alternate_physical = campaign._dirac_definition(alternate)
        self.assertEqual(physical, alternate_physical)
        self.assertNotIn("tol", physical)
        self.assertNotIn("maxiter", physical)
        self.assertNotIn("multigrid", physical)

        precise_solver = campaign._output_provenance(
            manifest, "10000", points, basis, "PSP")["solver"]
        alternate_solver = campaign._output_provenance(
            alternate, "10000", points, basis, "PSP")["solver"]
        self.assertEqual(precise_solver, {
            "tol": 1e-12, "maxiter": 5000, "multigrid": None,
        })
        self.assertEqual(alternate_solver, {
            "tol": 1e-7, "maxiter": 9000,
            "multigrid": {"setup": "alternate-solver"},
        })
        self.assertNotEqual(precise_solver, alternate_solver)

    def test_gauge_handle_lazily_returns_native_reader_payload(self):
        manifest = _fixture_manifest(Path("/unused"))
        source = Path(campaign.__file__)
        spec = importlib.util.spec_from_file_location("partner_campaign_import_probe", source)
        probe = importlib.util.module_from_spec(spec)
        real_import = builtins.__import__

        def reject_top_level_gpu_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"pyquda", "pyquda_utils"}:
                raise AssertionError(f"unexpected eager GPU import: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_top_level_gpu_import):
            spec.loader.exec_module(probe)

        native_gauge = object()
        read_gauge = mock.Mock(return_value=native_gauge)
        pyquda_utils = types.ModuleType("pyquda_utils")
        pyquda_utils.io = types.SimpleNamespace(readQIOGauge=read_gauge)
        with mock.patch.dict(sys.modules, {"pyquda_utils": pyquda_utils}):
            handle = probe._GaugeHandle(manifest)
            result = handle.load("10000")

        self.assertIs(result, native_gauge)
        read_gauge.assert_called_once_with("/test/gauge/cfg_10000.lime")
        self.assertEqual(handle.gauge_id, probe.gauge_identity(manifest, "10000"))

    def test_prepare_records_policy4_points_and_coordinate_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = _fixture_manifest(Path(temporary))
            audit = campaign.prepare_campaign(manifest, campaign_profile=False)[0]
            points = SpatialPointHandle.from_dataset(
                Path(temporary) / "inputs" / "points", cfg="10000")
            self.assertEqual(points.points.design.temporal_policy, 4)
            for time in range(1, 4):
                np.testing.assert_array_equal(points.points.coordinates[0],
                                             points.points.coordinates[time])
            self.assertEqual(audit["coordinate_sha256"],
                             campaign.hashlib.sha256(
                                 np.asarray(points.points.coordinates, dtype="<i8").tobytes()).hexdigest())
            self.assertTrue((Path(temporary) / "inputs" / "points" / "10000.json").is_file())
            self.assertEqual((Path(temporary) / "cfgs.txt").read_text(), "10000\n")

    def test_prepare_reuses_verified_points_when_only_solver_tolerance_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = _fixture_manifest(root)
            first_audit = campaign.prepare_campaign(original, campaign_profile=False)[0]
            first_state = campaign.inspect_dataset(root / "inputs" / "points", cfg="10000")

            updated = dict(original)
            updated["dirac"] = {**original["dirac"], "tol": 1e-7}
            second_audit = campaign.prepare_campaign(updated, campaign_profile=False)[0]
            second_state = campaign.inspect_dataset(root / "inputs" / "points", cfg="10000")
            stored_manifest = campaign.load_manifest(root / "manifest.json",
                                                     campaign_profile=False)

            self.assertEqual(first_audit["data_id"], second_audit["data_id"])
            self.assertEqual(first_audit["coordinate_sha256"],
                             second_audit["coordinate_sha256"])
            self.assertEqual(first_state["provenance"], second_state["provenance"])
            self.assertNotEqual(
                second_state["provenance"]["manifest_id"],
                campaign._point_provenance(
                    updated, "10000", campaign._cfg_seed(
                        updated["sampling"]["seed_base"], "10000"))["manifest_id"],
            )
            self.assertEqual(stored_manifest["dirac"]["tol"], 1e-7)

    def test_production_campaign_profile_accepts_legacy_solver_tolerance(self):
        cfgs = [str(10000 + index * 100) for index in range(18)]
        manifest = {
            "schema": 1, "ensemble": "C24P29", "lattice": [24, 24, 24, 72],
            "cfgs": cfgs, "root": "/campaign", "gauge_pattern": "/gauge/cfg_{cfg}.lime",
            "legacy_vsv_root": "/legacy/vsv", "basis_ne": 100,
            "basis_entries": {cfg: [{"time": time} for time in range(72)] for cfg in cfgs},
            "sampling": {"num_points": 64, "temporal_policy": 4, "seed_base": 20261011},
            "dirac": {
                "mass": -0.277, "tol": 1e-7, "maxiter": 5000, "xi_0": 1.0,
                "nu": 1.0, "clover_coeff_t": 1.160920226,
                "clover_coeff_r": 1.160920226, "t_boundary": -1,
                "multigrid": None, "stout": {"nstep": 1, "rho": 0.125, "dir_ignore": 4},
            },
            "storage": {"PSP_time_triangle": "sink>=source", "dtype": "complex128"},
        }
        self.assertIs(campaign.validate_manifest(manifest), manifest)

    def test_partial_resume_publishes_full_psv_and_psp_triangle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _fixture_manifest(root)
            points = _publish_fixture_points(manifest)
            lattice = tuple(manifest["lattice"])
            matrix = _gamma5_hermitian_matrix(lattice)
            inverse = np.linalg.inv(matrix)
            basis = _FixtureBasis(lattice)
            gauge = _FixtureGauge(matrix, campaign.gauge_identity(manifest, "10000"))
            backend = _FixtureBackend()

            first = campaign.produce_cfg(
                manifest, "10000", backend=backend, gauge_handle=gauge,
                basis_handle=basis, point_handle=points,
                source_times=(0,), max_source_times=1,
                campaign_profile=False,
            )
            self.assertEqual(first["state"], "partial")
            self.assertEqual(first["remaining_psv_pairs"], 12)
            self.assertEqual(first["remaining_psp_pairs"], 6)
            self.assertEqual(backend.rhs_total, 2*12)

            second = campaign.produce_cfg(
                manifest, "10000", backend=backend, gauge_handle=gauge,
                basis_handle=basis, point_handle=points,
                source_times=(0, 1), max_source_times=1,
                campaign_profile=False,
            )
            self.assertEqual(second["state"], "partial")
            self.assertEqual(second["source_times"], [1])
            self.assertEqual(backend.rhs_total, 4*12)

            final = campaign.produce_cfg(
                manifest, "10000", backend=backend, gauge_handle=gauge,
                basis_handle=basis, point_handle=points,
                source_times=(2, 3), campaign_profile=False,
            )
            self.assertEqual(final["state"], "complete")
            self.assertEqual(backend.rhs_total, 8*12)
            self.assertEqual(backend.smear_calls, [(1, 0.125, 4)]*3)

            point_endpoint = PropagatorEndpoint("P", points.data_id, range(points.Np))
            vector_endpoint = basis.endpoint
            with DatasetReader(root / "data" / "PSP", cfg="10000") as psp_reader, \
                    DatasetReader(root / "data" / "PSV", cfg="10000") as psv_reader:
                reader = PropagatorReader([psp_reader, psv_reader])
                # The stored PSP triangle contains (sink=3, source=0); reading
                # (0,3) must use the declared Gamma5 relation.
                reverse = reader.read("PSP", 0, 3, sink=point_endpoint, source=point_endpoint)
                np.testing.assert_allclose(
                    reverse.values, _reference_psp(inverse, lattice, points, 0, 3),
                    rtol=1e-11, atol=1e-11,
                )
                psv = reader.read("PSV", 1, 2, sink=point_endpoint, source=vector_endpoint)
                np.testing.assert_allclose(
                    psv.values, _reference_psv(inverse, lattice, points, basis, 1, 2),
                    rtol=1e-11, atol=1e-11,
                )

    def test_full_declaration_is_not_narrowed_by_partial_source_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _fixture_manifest(root)
            points = _publish_fixture_points(manifest)
            lattice = tuple(manifest["lattice"])
            matrix = _gamma5_hermitian_matrix(lattice)
            backend = _FixtureBackend()
            result = campaign.produce_cfg(
                manifest, "10000", backend=backend,
                gauge_handle=_FixtureGauge(matrix, campaign.gauge_identity(manifest, "10000")),
                basis_handle=_FixtureBasis(lattice), point_handle=points,
                source_times=(1,), campaign_profile=False,
            )
            self.assertEqual(result["state"], "partial")
            psv_state = campaign.inspect_dataset(root / "data" / "PSV", cfg="10000")
            self.assertEqual(psv_state["state"], "partial")
            self.assertEqual(len(psv_state["spec"].coordinates("time_pair")), 16)
            self.assertEqual(len(psv_state["completed"]), 4)

    def test_finalize_resumes_partial_dataset_with_no_missing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _fixture_manifest(root)
            points = _publish_fixture_points(manifest)
            basis = _FixtureBasis(tuple(manifest["lattice"]))
            _, _, psv_spec, _ = campaign.make_pair_specs(
                manifest, "10000", points, basis,
                psv_pairs=[(0, 0)], psp_pairs=[(0, 0)],
            )
            provenance = campaign._output_provenance(
                manifest, "10000", points, basis, "PSV")
            directory = root / "data" / "PSV"

            # Model interruption after the final row was flushed but before
            # DatasetWriter.publish() moved the partial file into registration.
            from lqcdweave.storage.dataset import DatasetWriter, inspect_dataset
            with DatasetWriter(directory, psv_spec, provenance=provenance,
                               format="npy") as writer:
                writer.write(0, np.zeros(psv_spec.shape[1:], dtype=np.complex128))
            partial = inspect_dataset(directory, cfg="10000")
            self.assertEqual(partial["state"], "partial")
            self.assertEqual(len(partial["completed"]), 1)

            output = campaign._PairOutput(
                directory, psv_spec, provenance, cfg="10000")
            self.assertFalse(output.missing_pairs)
            self.assertTrue(output.finalize())
            self.assertEqual(inspect_dataset(directory, cfg="10000")["state"], "complete")


def run_cpu_self_test():
    """Standalone bounded self-test used by the campaign CLI."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PartnerPointProductionTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("partner point CPU self-test failed")


if __name__ == "__main__":
    unittest.main()
