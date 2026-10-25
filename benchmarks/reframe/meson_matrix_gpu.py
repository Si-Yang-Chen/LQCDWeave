"""CSNS V100 route comparison for the fixed Ne120/Ne70 meson matrix workload."""

import json
import os
from pathlib import Path
import statistics

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import performance_function, sanity_function


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_NAME = "meson-matrix-gpu-report.json"
METHODS = ("bulk_sync", "row_sync", "row_prefetch")
RUN_ROOT = os.environ.get(
    "CSNS_MESON_ROW_REFRAME_ROOT",
    "/dg_workfs/LQCD/chensiyang/codex_jobs/meson-row-reframe-20261006",
)
MESON55_DIR = os.environ.get("CSNS_MESON55_DIR", f"{RUN_ROOT}/external/meson55")
LATTICE_SCATTERING_ROOT = os.environ.get(
    "LATTICE_SCATTERING_ROOT", f"{RUN_ROOT}/external/lattice-scattering"
)
T40_DATA_ROOT = os.environ.get(
    "CSNS_T40_DATA_ROOT",
    "/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104",
)
PYTHONPATH_ENTRIES = [
    str(PROJECT_ROOT / "src"),
    MESON55_DIR,
    f"{LATTICE_SCATTERING_ROOT}/scripts/spectroscopy",
]
if os.environ.get("PYTHONPATH"):
    PYTHONPATH_ENTRIES.append(os.environ["PYTHONPATH"])


@rfm.simple_test
class MesonMatrixGpuStorage(rfm.RunOnlyRegressionTest):
    descr = "Compare bulk, row, and row-prefetch storage replay on CSNS V100"
    valid_systems = ["csns:dggpu"]
    valid_prog_environs = ["python311-cupy"]
    time_limit = "30m"
    num_tasks = 1
    extra_resources = {
        "gpu": {"num_gpus_per_node": 1},
        "memory": {"size": "8G"},
    }
    env_vars = {
        "CSNS_MESON55_DIR": MESON55_DIR,
        "LATTICE_SCATTERING_ROOT": LATTICE_SCATTERING_ROOT,
        "CSNS_T40_DATA_ROOT": T40_DATA_ROOT,
        "PYTHONPATH": os.pathsep.join(PYTHONPATH_ENTRIES),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    executable = "python3"
    executable_opts = [
        str(PROJECT_ROOT / "benchmarks" / "csns_meson_matrix.py"),
        "--workload", "all",
        "--backend", "cupy",
        "--ne", "120",
        "--repeats", "3",
        "--time-pairs", "2",
        "--cfg", "2000",
        "--host-memory-bytes", "4294967296",
        "--json", REPORT_NAME,
    ]

    @property
    def _report(self):
        with open(Path(self.stagedir) / REPORT_NAME, encoding="utf-8") as stream:
            return json.load(stream)

    def _result(self, workload):
        return next(item for item in self._report["results"]
                    if item.get("workload") == workload)

    @property
    def _synthetic(self):
        return self._result("synthetic")

    @property
    def _real(self):
        return self._result("cfg2000_real_t40")

    @staticmethod
    def _median_elapsed(result, method):
        return statistics.median(
            row["elapsed_seconds_including_replay_and_assembly"]
            for row in result["variants"][method]
        )

    @sanity_function
    def validate_report(self):
        report = self._report
        synthetic, real = self._synthetic, self._real
        cpu_ref, gpu_ref = report.get("cpu_reference", {}), report.get("gpu_reference", {})
        variants_valid = all(
            len(synthetic.get("variants", {}).get(method, ())) == 3
            and len(real.get("variants", {}).get(method, ())) == 3
            for method in METHODS
        )
        return sn.assert_true(
            "V100" in report.get("runtime_environment", {}).get("device_name", "")
            and synthetic.get("ne") == 120
            and synthetic.get("terms_per_batch") == 224
            and synthetic.get("computations_per_batch", 0) > 0
            and synthetic.get("computations_per_batch") == (
                synthetic.get("schedule_audit", [{}])[0].get("nodes")
            )
            and synthetic.get("all_methods_allclose") == "passed"
            and cpu_ref.get("ne") == 2
            and cpu_ref.get("independent_numpy_reference") == "passed"
            and cpu_ref.get("all_methods_allclose") == "passed"
            and gpu_ref.get("ne") == 2
            and gpu_ref.get("independent_numpy_reference") == "passed"
            and gpu_ref.get("all_methods_allclose") == "passed"
            and real.get("configuration") == "2000"
            and real.get("operator_block") == "A1_Pm_Cp_Rm"
            and real.get("operator_count") == 6
            and real.get("mode_partitioning") is None
            and real.get("terms_per_pair") == 224
            and real.get("computations_per_pair", 0) > 0
            and real.get("computations_per_pair") == (
                real.get("schedule_audit", [{}])[0].get("nodes")
            )
            and len(real.get("time_pairs_slots", ())) == 2
            and real.get("all_methods_allclose") == "passed"
            and real.get("assembled_output_allclose") == "passed"
            and variants_valid
        )

    @performance_function("s")
    def synthetic_bulk_sync(self):
        return self._median_elapsed(self._synthetic, "bulk_sync")

    @performance_function("s")
    def synthetic_row_sync(self):
        return self._median_elapsed(self._synthetic, "row_sync")

    @performance_function("s")
    def synthetic_row_prefetch(self):
        return self._median_elapsed(self._synthetic, "row_prefetch")

    @performance_function("s")
    def real_bulk_sync(self):
        return self._median_elapsed(self._real, "bulk_sync")

    @performance_function("s")
    def real_row_sync(self):
        return self._median_elapsed(self._real, "row_sync")

    @performance_function("s")
    def real_row_prefetch(self):
        return self._median_elapsed(self._real, "row_prefetch")
