"""CSNS V100 correctness and performance regression check for Elemental."""

import json
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import performance_function, sanity_function


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_NAME = "elemental-gpu-report.json"


@rfm.simple_test
class ElementalGpuPerformance(rfm.RunOnlyRegressionTest):
    descr = "Elemental CPU/GPU correctness, throughput, and memory regression"
    valid_systems = ["csns:dggpu"]
    valid_prog_environs = ["python311-cupy"]
    time_limit = "30m"
    num_tasks = 1
    extra_resources = {"gpu": {"num_gpus_per_node": 1}}
    executable = "python3"
    executable_opts = [
        str(PROJECT_ROOT / "tests" / "run_elemental_cuda.py"),
        "--report",
        REPORT_NAME,
    ]
    # The previous V100 reference used synthetic 16^3 x 4 input. Record the
    # weak_field series before setting a regression threshold for this workload.
    require_reference = False

    @property
    def _report(self):
        with open(Path(self.stagedir) / REPORT_NAME, encoding="utf-8") as stream:
            return json.load(stream)

    @property
    def _benchmark_rows(self):
        return {
            row["ne_block_size"]: row
            for row in self._report["benchmark"]["results"]
        }

    @sanity_function
    def validate_report(self):
        report = self._report
        return sn.assert_true(
            report.get("failures") == 0
            and report.get("errors") == 0
            and report.get("skipped") == 0
            and "V100" in report.get("gpu", "")
            and report.get("benchmark", {}).get("fixture") == "weak_field"
        )

    @performance_function("pairs/s")
    def throughput_ne_8(self):
        return self._benchmark_rows[8]["site_mode_pairs_per_second"]

    @performance_function("pairs/s")
    def throughput_ne_16(self):
        return self._benchmark_rows[16]["site_mode_pairs_per_second"]

    @performance_function("pairs/s")
    def throughput_ne_20(self):
        return self._benchmark_rows[20]["site_mode_pairs_per_second"]

    @performance_function("pairs/s")
    def cpu_throughput_ne_20(self):
        return self._benchmark_rows[20]["cpu_site_mode_pairs_per_second"]
