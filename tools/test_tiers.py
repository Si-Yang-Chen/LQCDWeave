"""Select existing unittest cases and report outcomes without changing them.

This is a thin discovery/reporting CLI, not a scheduler or replacement test
framework. QUDA, real MPI and performance use their existing external runners.
The normal `python -m unittest discover -s tests -v` entry remains unchanged.
"""
from collections import Counter
import argparse
import errno
import json
from pathlib import Path
import sys
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[1]
# Curated cross-module independent reference gates. Numerical module-unit
# tests remain in cpu-unit; this label is not a claim that other tests lack an oracle.
REFERENCE_MODULES = frozenset({
    "test_small_scientific_pipeline", "test_wilson_pipeline", "test_weak_field",
    "test_sparse_persisted_pipeline",
    "test_highmode_wilson_chain", "test_independent_point_diagram_numeric",
    "test_baryon_production_measurement", "test_operator_dataset_reuse",
    "test_dilution_correlator",
    "test_dilution_component_reference",
    "test_weak_field_dilution_subset",
    "test_three_sector_closed_loop",
    "test_propagator_gamma5_production",
})
# Root-level numerical module tests are otherwise treated as integration tests.
UNIT_MODULES = frozenset({"test_dilution_weights"})
GPU_PREFIXES = (
    # Real-device oracles are separate from CPU protocol and host-kernel tests.
    "correlator.execution.test_bound_output_views.RealCuPyBoundOutputViewTests.",
    "correlator.execution.test_bound_output_views.RealCuPyOutputSnapshotTests.",
    "correlator.execution.test_cartesian_group_execution.CartesianGroupPublicCuPyTests.",
    "correlator.execution.test_cartesian_group_packing.CartesianGroupPackingCuPyTests.",
    "correlator.execution.test_coalesced_scalar_batch.CoalescedScalarBatchCuPyTests.",
    "correlator.execution.test_deferred_result_views.RealCuPyDeferredPackingTests.",
    "correlator.execution.test_deferred_result_views.RealCuPyDeferredResultViewTests.",
    "correlator.execution.test_dense_store.DenseStoreTests.test_real_cupy_unary_snapshot_and_binary_store_rebind",
    "correlator.execution.test_gemm_group_device.CuPyDenseGemmGroupTests.",
    "correlator.execution.test_gemm_group_packing.GroupPackingCupyTests.",
    "correlator.execution.test_gemm_pointer_arguments.GemmPointerArgumentsCuPyTests.",
    "correlator.execution.test_matrix_reduction_group.MatrixReductionGroupCuPyTests.",
    "correlator.execution.test_matrix_reduction_sources.MatrixReductionSourcesCuPyTests.",
    "correlator.execution.test_mixed_stride_stack_packing.MixedStrideStackPackingCuPyTests.",
    "correlator.execution.test_output_snapshot_dedup.RealCuPyOutputSnapshotDedupTests.",
    "correlator.execution.test_precise_batch_reduction.PreciseBatchReductionCuPyTests.",
    "correlator.execution.test_preparation_input_gather.PreparationInputGatherCuPyTests.",
    "correlator.execution.test_raw_packing.RawPackingCupyTests.",
    "correlator.execution.test_raw_precise_reduction.RawPreciseReductionCuPyTests.",
    "correlator.execution.test_read_ahead_admission.ReadAheadAdmissionCuPyTests.",
    "correlator.execution.test_reduction_group_output_owner.SharedReductionOutputCuPyTests.",
    "correlator.execution.test_reduction_workspace_binding.ReductionWorkspaceBindingCuPyTests.",
    "correlator.execution.test_reference_matrix.ReferenceMatrixCupyTests.",
    "correlator.execution.test_scalar_group_execution.ScalarGroupPublicCuPyTests.",
    "correlator.execution.test_scalar_group_width32.ScalarGroupWidth32CuPyTests.",
    "correlator.execution.test_scalar_reduction_group.ScalarReductionGroupCuPyTests.",
    "correlator.execution.test_stack_packing.StackPackingCuPyTests.",
    "correlator.execution.test_tiled_matrix_reduction.TiledMatrixReductionCuPyTests.",
    "correlator.execution.test_wide_output_copy.WideOutputCopyCuPyTests.",
    "correlator.planning.test_dense_batch_subgrids.DenseBatchSubgridCuPyTests.",
    "correlator.planning.test_reduction_batches.ReductionBatchCupyTests.",
    "correlator.execution.test_scene_kernel_reuse.CuPySceneKernelReuseTests.",
    "workflows.test_elemental_backend.ElementalBackendCuPyTests.",
    "correlator.execution.test_elemental_backend_routing.ElementalBackendRoutingCuPyTests.",
    "correlator.execution.test_device_output_program.RealCuPyOutputTests.",
    "correlator.execution.test_output_owner_snapshot.RealCuPyOwnerSnapshotTests.",
    "sampling.temporal.test_device_reconstruction.DeviceAccumulatorCuPyTests.",
    "test_gpu_output_postprocessing.RealCuPyDatasetPostprocessingTests.",
    "correlator.execution.test_fused_preparation.FusedPreparationCuPyTests.",
    "correlator.execution.test_fused_preparation.FusedOutputCuPyTests.",
    "correlator.execution.test_upload_lifecycle.RealCuPyUploadLifecycleTests.",
    "correlator.execution.test_input_retirement.RealCuPyInputRetirementTests.",
    "correlator.execution.test_iterator_output_queue.RealCuPyIteratorOutputQueueTests.",
    "correlator.execution.test_prebound_preparation.CuPyPreboundPreparationTests.",
    "correlator.execution.test_raw_preparation._CuPyCase.",
    "correlator.execution.test_cfg_template_preparation.CuPyCfgTemplatePreparationTests.",
    "correlator.execution.test_cfg_instance_preparation.CuPyCfgInstancePreparationTests.",
    "correlator.execution.test_raw_consumer_arguments.RawConsumerArgumentsCuPyTests.",
    "correlator.execution.test_output_queue.RealCuPyOutputQueueTests.",
    "test_elemental_cuda.",
    "production.perambulator.test_propagator_stream_cuda.",
    "production.elemental.test_baryon_elemental.BaryonProductionTests.test_cupy_",
    "correlator.execution.test_sparse_retained_accumulation.RetainedSparseGPUChecks.",
    "correlator.execution.test_device.RealCuPyOptionalTests.",
    "correlator.execution.test_dense_atomic_kernel.DenseAtomicCupyTests.",
    "correlator.execution.test_prepared_layouts.PreparedLayoutCupyTests.",
    "correlator.execution.test_pointer_matrix_product.PointerMatrixCuPyTests.",
    "correlator.execution.test_packed_matrix_product.PackedMatrixProductCuPyTests.",
    "correlator.execution.test_direct_matrix_packing.DirectMatrixPackingCuPyTests.",
    "correlator.execution.test_raw_driver_arguments.RawDriverArgumentsCuPyTests.",
    "correlator.planning.test_dense_batches.DenseBatchCupyTests.",
)
LOCAL_TIERS = ("cpu-unit", "cpu-integration", "scientific-reference", "gpu", "slow", "temporary")
EXTERNAL_TIERS = {
    "quda": "Allocated CUDA/PyQUDA node: python benchmarks/csns_quda_stream.py --out /external/quda-stream.json",
    "mpi": "Site-matched MPI launcher: python benchmarks/launch_mpi.py --old-root /path/to/legacy --data-root /path/to/weak_field --output /external/mpi-quda.json; single rank only, multi-rank correctness remains unvalidated",
    "performance": "Same-machine CPU: asv check; asv machine; asv continuous BASE_COMMIT HEAD. Allocated GPU: reframe -C SITE_CONFIG -c benchmarks/reframe --system csns:dggpu --repeat 3 --run",
}


def tier_for(test_id):
    if test_id.startswith("slow."):
        return "slow"
    if test_id.startswith("temporary."):
        return "temporary"
    if test_id.startswith(GPU_PREFIXES):
        return "gpu"
    module = test_id.split(".")[0]
    if module in REFERENCE_MODULES:
        return "scientific-reference"
    if module in UNIT_MODULES:
        return "cpu-unit"
    return "cpu-integration" if module.startswith("test_") else "cpu-unit"


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def skip_category(reason):
    """Annotate only recognizable prerequisite skips; retain original reason."""
    lower = reason.lower()
    if any(word in lower for word in ("cuda", "cupy", "gpu node")):
        return "environment-platform"
    if any(word in lower for word in ("scipy", "sympy", "lazy_object_proxy", "optional correlator extra", "dependencies are unavailable",
                                      "host c++ compiler unavailable")):
        return "environment-dependency"
    if any(word in lower for word in ("platform-specific", "temporary junction", "symlinks are unavailable",
                                      "posix filesystem required")):
        return "environment-filesystem"
    if "reference fixture is not available" in lower:
        return "environment-fixture"
    return "unclassified-skip"


def error_category(error):
    """Conservative diagnosis only: every failure/error stays unsuccessful.

    Socket denial in multiprocessing is distinguishable from a numerical
    mismatch or the subsequent worker-cleanup assertion. The latter remains
    test-failure and needs investigation even when both occur in one run.
    """
    _, exception, tb = error
    frames = traceback.extract_tb(tb)
    files = [frame.filename.replace("\\", "/") for frame in frames]
    if (isinstance(exception, OSError) and exception.errno in (errno.EACCES, errno.EPERM)
            and any("/multiprocessing/" in name for name in files)
            and any(name.endswith("/socket.py") for name in files)):
        return "environment-restriction"
    if isinstance(exception, ModuleNotFoundError) and (exception.name or "").split(".")[0] in {
            "scipy", "sympy", "lazy_object_proxy", "cupy", "pyquda", "mpi4py"}:
        return "environment-dependency"
    return "test-failure"


class TierResult(unittest.TextTestResult):
    """Keep unittest's actual categories and count successes directly."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []
        self.diagnostics = []
        self.audits = {}

    def _record_audits(self, test):
        metrics = getattr(test, "audit_metrics", None)
        if isinstance(metrics, dict):
            self.audits[test.id()] = dict(metrics)

    def addSuccess(self, test):
        self.passed.append(test.id())
        self._record_audits(test)
        super().addSuccess(test)

    def addSkip(self, test, reason):
        self._record_audits(test)
        self.diagnostics.append(dict(test=test.id(), outcome="skipped",
                                     classification=skip_category(reason), reason=reason))
        super().addSkip(test, reason)

    def _diagnose(self, test, outcome, error):
        self._record_audits(test)
        self.diagnostics.append(dict(test=test.id(), outcome=outcome,
            classification=error_category(error), reason=str(error[1])))

    def addError(self, test, err):
        self._diagnose(test, "error", err)
        super().addError(test, err)

    def addFailure(self, test, err):
        self._diagnose(test, "failure", err)
        super().addFailure(test, err)

    def addSubTest(self, test, subtest, err):
        self._record_audits(test)
        if err is not None:
            self._diagnose(subtest, "failure" if issubclass(err[0], test.failureException) else "error", err)
        super().addSubTest(test, subtest, err)

    def addExpectedFailure(self, test, err):
        self._record_audits(test)
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self._record_audits(test)
        super().addUnexpectedSuccess(test)


def summarize(result, *, tier, selected):
    return dict(tier=tier, selected_cases=selected, tests_run=result.testsRun,
        passed=len(result.passed), skipped=len(result.skipped),
        failures=len(result.failures), errors=len(result.errors),
        expected_failures=len(result.expectedFailures), unexpected_successes=len(result.unexpectedSuccesses),
        successful=result.wasSuccessful(), diagnostics=result.diagnostics, audits=result.audits)


def exit_code(report):
    if not report["successful"]:
        return 1
    # Explicit gates are incomplete with skipped cases, even if unittest says
    # OK(skipped=N). Never treat missing GPU/dependencies as platform success.
    return 2 if report["skipped"] or report["expected_failures"] or not report["tests_run"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("all", *LOCAL_TIERS, *EXTERNAL_TIERS), default="all")
    parser.add_argument("--list", action="store_true", help="list cases/commands without running tests")
    parser.add_argument("--json", type=Path, help="write machine-readable outcome counts and diagnostics")
    args = parser.parse_args(argv)
    if args.tier in EXTERNAL_TIERS:
        print(f"{args.tier}: NOT RUN. {EXTERNAL_TIERS[args.tier]}")
        print("Requires the named environment and explicit job authorization; see tests/README.md.")
        return 0 if args.list else 2
    # Match `python -m unittest` from the repository root without forcing src
    # ahead of an intentionally selected installed package.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    # Keep the normal discovery pattern/default unchanged. The separately
    # named slow pipeline and the temporary cases are collected only through
    # their explicit tiers.
    suite = loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT / "tests"))
    cases = list(flatten(suite))
    if args.tier == "slow":
        slow_suite = loader.discover(
            str(ROOT / "tests" / "slow"),
            pattern="*_pipeline.py",
            top_level_dir=str(ROOT / "tests"),
        )
        cases.extend(flatten(slow_suite))
    if args.tier == "temporary":
        cases.extend(flatten(loader.discover(
            str(ROOT / "tests" / "temporary"),
            pattern="tmp_*.py",
            top_level_dir=str(ROOT / "tests"),
        )))
    # Collection errors must fail every gate; filtering cannot conceal them.
    failed_imports = [case for case in cases if isinstance(case, unittest.loader._FailedTest)]
    if loader.errors:
        for error in loader.errors:
            print(error, file=sys.stderr)
        if args.list:
            return 1
    selected = [case for case in cases if args.tier == "all" or tier_for(case.id()) == args.tier]
    selected += [case for case in failed_imports if case not in selected]
    if args.list:
        for case in selected:
            print(f"{tier_for(case.id())}\t{case.id()}")
        print(f"Selected {len(selected)} cases: {dict(Counter(tier_for(case.id()) for case in selected))}")
        return 0 if selected else 2
    result = unittest.TextTestRunner(verbosity=2, resultclass=TierResult).run(unittest.TestSuite(selected))
    report = summarize(result, tier=args.tier, selected=len(selected))
    report["exit_code"] = exit_code(report)
    report["status"] = {0: "passed", 1: "failed", 2: "incomplete"}[report["exit_code"]]
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
