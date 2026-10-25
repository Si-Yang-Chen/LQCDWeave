"""Profile IO, prediction fusion and standalone fit/merge workflow checks."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import lqcdweave

from lqcdweave.contraction.performance.api import (
    CostProfile,
    ProfileFitReport,
    evaluate_cost_profile,
    execution_cost_samples,
    fit_cost_profile_sources,
    load_cost_profile,
    load_cost_profile_document,
    save_cost_profile,
    weighted_profile_merge,
)
from lqcdweave.contracts.measurement import (
    ActionTiming,
    FileIOEvent,
    PhaseTiming,
    ReplayReport,
)
from lqcdweave.runtime.measurement import save_measurement
from correlator._paths import PROJECT_ROOT


def mergeable_profiles():
    first = CostProfile(
        version="observed-a", hardware_fingerprint="device-a", sample_count=10,
        snapshot_call_seconds=.005, snapshot_seconds_per_byte=.006,
        contract_scale=2.0, kernel_call_seconds=10.,
        kernel_call_seconds_float32=11., kernel_call_seconds_float64=12.,
        kernel_call_seconds_complex64=13., kernel_call_seconds_complex128=14.,
        flop_seconds=.2, traffic_seconds_per_byte=.3,
        sparse_bridge_scale=6., sparse_kernel_call_scale=.5,
        sparse_flop_seconds=.4, sparse_cycle_flop_seconds=.5,
        sparse_traffic_seconds_per_byte=.6, baryon_join_scale=1.,
        preparation_call_seconds=.07, preparation_flop_seconds=None,
    )
    second = CostProfile(
        version="observed-b", hardware_fingerprint="device-a", sample_count=30,
        snapshot_call_seconds=.015, snapshot_seconds_per_byte=.016,
        contract_scale=4.0, kernel_call_seconds=20.,
        kernel_call_seconds_float32=21., kernel_call_seconds_float64=22.,
        kernel_call_seconds_complex64=23., kernel_call_seconds_complex128=24.,
        flop_seconds=.8, traffic_seconds_per_byte=.9,
        sparse_bridge_scale=12., sparse_kernel_call_scale=.5,
        sparse_flop_seconds=.8, sparse_cycle_flop_seconds=.9,
        sparse_traffic_seconds_per_byte=1.0, baryon_join_scale=2.,
        preparation_call_seconds=.17, preparation_flop_seconds=.7,
    )
    return first, second


def coverage_metadata(source, **statuses):
    categories = {
        "source_preparation", "logical_snapshot", "dense_contraction",
        "sparse_contraction", "baryon_join",
    }
    coverage = {
        category: {"status": statuses.get(category, "base-unobserved"),
                   "prediction_available": True}
        for category in categories
    }
    return {
        "source": source,
        "prior_provenance": {
            "operation": "fit",
            "fit_diagnostics": {"operation_coverage": coverage},
        },
    }


def measured_report(multiplier, offset):
    timings = (
        ActionTiming(
            "contract", .01, .02 * multiplier + offset, 0, "dense-result",
            model_features={"kernel_calls": 1, "flops": 2_000,
                            "traffic_bytes": 4_000, "dtype": "complex128"},
        ),
        ActionTiming(
            "load", .02, .03 * multiplier + offset, 1, "source",
            phases=(PhaseTiming("materialize", .03 * multiplier + offset),),
            model_features={"storage_preparation": True, "kernel_calls": 1,
                            "flops": 100, "traffic_bytes": 200, "dtype": "complex128"},
        ),
    )
    events = (FileIOEvent(1, "source", f"input-{offset}", "npy_whole_read",
                          100 + offset, .001 * multiplier + offset / 1000),)
    return ReplayReport((), timings, None, None, io_events=events)


def linear_report(observed, count=1, *, baryon_observed=None, offset=0):
    timings = [
        ActionTiming(
            "contract", .01, float(observed), index, f"dense-{offset}-{index}",
            model_features={"kernel_calls": 1, "flops": 0,
                            "traffic_bytes": 0, "dtype": "complex128"},
        )
        for index in range(count)
    ]
    if baryon_observed is not None:
        timings.append(ActionTiming(
            "baryon_join", .01, float(baryon_observed), count,
            f"baryon-{offset}", operands=("a", "b", "c"),
            model_features={"kernel_calls": 1, "flops": 0,
                            "traffic_bytes": 0, "dtype": "complex128"},
        ))
    return ReplayReport((), tuple(timings), None, None)


class ProfileWorkflowTests(unittest.TestCase):
    def test_default_object_name_path_and_compute_only_profile_schema(self):
        default = load_cost_profile()
        self.assertEqual(default, CostProfile())
        self.assertIs(load_cost_profile(default), default)
        self.assertEqual(load_cost_profile("default"), default)
        default_document = load_cost_profile_document()
        self.assertEqual(default_document["provenance"]["calibration_status"], "unmeasured")
        self.assertEqual(default_document["source"], str(
            Path(lqcdweave.__file__).parent / "contraction" / "performance" / "profiles" / "default.json"
        ))
        self.assertEqual(default_document["calibration_status"], "unmeasured")
        self.assertEqual(load_cost_profile_document(default)["calibration_status"], "unmeasured")
        node_document = load_cost_profile_document("csns-v100-gcn006-dense")
        self.assertEqual(node_document["calibration_status"], "has-samples")
        self.assertEqual(node_document["profile"].sample_count, 189)
        training_report = node_document["provenance"]["fit"]["training_report"]
        self.assertEqual(training_report["run_id"], "v100-24f7fc4f92af5b45-light-train")
        self.assertEqual(training_report["source_weight"], 1.0)
        self.assertTrue(Path(node_document["source"]).as_posix().endswith(
            "contraction/performance/profiles/csns-v100-gcn006-dense.json"
        ))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            # Current documents may omit provenance, but old cost domains and
            # stale IO coefficients are rejected explicitly.
            document = {
                "format": "easy-distillation.scene-compute-cost-profile",
                "profile": {"version": "legacy-v1", "hardware_fingerprint": "legacy", "sample_count": 4},
            }
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = load_cost_profile(path)
            self.assertEqual(loaded.version, "legacy-v1")
            self.assertEqual(loaded.sample_count, 4)
            self.assertEqual(loaded.flop_seconds, CostProfile().flop_seconds)
            self.assertEqual(load_cost_profile_document(path)["calibration_status"], "has-samples")
            document["profile"]["read_call_seconds"] = 0.1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown CostProfile fields"):
                load_cost_profile(path)
            document["format"] = "easy-distillation.scene-cost-profile"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "format"):
                load_cost_profile(path)

    def test_merged_profile_status_is_not_reported_as_unmeasured(self):
        merged = weighted_profile_merge(mergeable_profiles())
        document = load_cost_profile_document(merged)
        self.assertEqual(document["calibration_status"], "merged")
        self.assertIn("CostProfile object", document["source"])

    def test_profile_fields_load_and_failed_atomic_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            save_cost_profile(CostProfile(), path)
            original = path.read_bytes()
            document = json.loads(path.read_text(encoding="utf-8"))
            document["profile"]["flop_seconds"] *= 2
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_cost_profile(path).flop_seconds, CostProfile().flop_seconds * 2)
            document["format"] = "unsupported"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "format"):
                load_cost_profile(path)

            save_cost_profile(CostProfile(), path)
            original = path.read_bytes()
            with self.assertRaises((TypeError, ValueError)):
                save_cost_profile(CostProfile(), path, provenance={"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_weighted_merge_matches_weighted_predictions_after_factor_normalization(self):
        first, second = mergeable_profiles()
        weights = (0.25, 0.75)
        mixed = weighted_profile_merge(((first, weights[0]), (second, weights[1])))

        def weighted(a, b):
            return weights[0] * a + weights[1] * b

        self.assertAlmostEqual(
            mixed.snapshot_seconds(300),
            weighted(first.snapshot_seconds(300), second.snapshot_seconds(300)),
        )
        for dtype in (None, "float32", "float64", "complex64", "complex128"):
            expected = weighted(
                first.preparation_seconds(300, 400, kernel_calls=3, dtype=dtype),
                second.preparation_seconds(300, 400, kernel_calls=3, dtype=dtype),
            )
            self.assertAlmostEqual(mixed.preparation_seconds(300, 400, kernel_calls=3, dtype=dtype), expected)
            for kind, records in (("contract", 1), ("sparse_bridge", 1),
                                  ("sparse_bridge", 2), ("baryon_join", 1)):
                expected = weighted(
                    first.contract_seconds(2_000, 400, kind=kind, dtype=dtype,
                                           kernel_calls=3, sparse_records=records),
                    second.contract_seconds(2_000, 400, kind=kind, dtype=dtype,
                                            kernel_calls=3, sparse_records=records),
                )
                self.assertAlmostEqual(
                    mixed.contract_seconds(2_000, 400, kind=kind, dtype=dtype,
                                           kernel_calls=3, sparse_records=records), expected
                )
        self.assertEqual(mixed.sample_count, 0)
        self.assertEqual(mixed.hardware_fingerprint, "device-a")

    def test_coverage_aware_merge_excludes_large_unobserved_dense_and_snapshot_bases(self):
        dense = CostProfile(
            version="dense-covered", hardware_fingerprint="same-device",
            snapshot_call_seconds=900., snapshot_seconds_per_byte=80.,
            contract_scale=3., kernel_call_seconds=2., kernel_call_seconds_float32=3.,
            kernel_call_seconds_float64=4., kernel_call_seconds_complex64=5.,
            kernel_call_seconds_complex128=6., flop_seconds=.25,
            traffic_seconds_per_byte=.125, preparation_call_seconds=.3,
            preparation_kernel_call_seconds=.4, preparation_flop_seconds=.5,
            preparation_traffic_seconds_per_byte=.6,
        )
        snapshot = replace(
            dense, version="snapshot-covered",
            snapshot_call_seconds=7., snapshot_seconds_per_byte=.02,
            contract_scale=11., kernel_call_seconds=700.,
            kernel_call_seconds_float32=800., kernel_call_seconds_float64=900.,
            kernel_call_seconds_complex64=1000., kernel_call_seconds_complex128=1100.,
            flop_seconds=1200., traffic_seconds_per_byte=1300.,
        )
        mixed, provenance = weighted_profile_merge(
            ((dense, 1), (snapshot, 3)), return_provenance=True,
            source_metadata=(
                coverage_metadata("dense-report", dense_contraction="fitted"),
                coverage_metadata("snapshot-report", logical_snapshot="fitted"),
            ),
        )
        dense_sample = {"category": "dense_contraction", "dtype": "complex128",
                        "fields": {"kernel_call_seconds_complex128": 2.,
                                   "flop_seconds": 100.,
                                   "traffic_seconds_per_byte": 200.},
                        "observed_seconds": 0.}
        snapshot_sample = {"category": "logical_snapshot",
                           "fields": {"snapshot_call_seconds": 1.,
                                      "snapshot_seconds_per_byte": 300.},
                           "observed_seconds": 0.}
        self.assertAlmostEqual(
            evaluate_cost_profile(mixed, (dense_sample,))["dense_contraction"]["predicted_seconds"],
            evaluate_cost_profile(dense, (dense_sample,))["dense_contraction"]["predicted_seconds"],
        )
        self.assertAlmostEqual(
            evaluate_cost_profile(mixed, (snapshot_sample,))["logical_snapshot"]["predicted_seconds"],
            evaluate_cost_profile(snapshot, (snapshot_sample,))["logical_snapshot"]["predicted_seconds"],
        )
        coverage = provenance["operation_coverage"]
        self.assertEqual(coverage["dense_contraction"]["weight_policy"],
                         "covered-sources-renormalized")
        self.assertEqual(coverage["dense_contraction"]["contributors"], [
            {"index": 0, "source": "dense-report", "normalized_weight": 1.,
             "coverage_status": "fitted"},
        ])
        self.assertEqual(coverage["logical_snapshot"]["contributors"], [
            {"index": 1, "source": "snapshot-report", "normalized_weight": 1.,
             "coverage_status": "fitted"},
        ])
        self.assertEqual(coverage["sparse_contraction"]["status"], "base-unobserved")
        self.assertFalse(coverage["sparse_contraction"]["calibrated"])
        self.assertEqual(coverage["sparse_contraction"]["weight_policy"],
                         "base-fallback-global-weights")

    def test_execution_io_observations_are_not_cost_calibration_samples(self):
        report = measured_report(1., 1)
        samples = execution_cost_samples(report)
        self.assertEqual({sample["category"] for sample in samples},
                         {"dense_contraction", "source_preparation"})
        self.assertTrue(report.io_events)
        with self.assertRaisesRegex(ValueError, "unsupported compute-cost sample category"):
            evaluate_cost_profile(CostProfile(), ({
                "category": "p2p_read", "fields": {}, "observed_seconds": 1.
            },))

    def test_read_and_view_waits_are_excluded_but_preparation_and_snapshot_are_sampled(self):
        load = ActionTiming(
            "load", 1., 901.3, 0, "source",
            phases=(PhaseTiming("read_block", 900., bytes=128),
                    PhaseTiming("view_or_read", 1., bytes=128),
                    PhaseTiming("materialize", .2),
                    PhaseTiming("snapshot", .1, bytes=256)),
            model_features={"storage_preparation": True, "kernel_calls": 1,
                            "flops": 100, "traffic_bytes": 200,
                            "dtype": "complex128"},
        )
        report = ReplayReport((), (load,), None, None, io_events=(
            FileIOEvent(0, "source", "input.npy", "npy_whole_read", 128, 900.),
        ))
        samples = execution_cost_samples(report)
        by_category = {sample["category"]: sample for sample in samples}
        self.assertEqual(set(by_category), {"source_preparation", "logical_snapshot"})
        self.assertEqual(by_category["source_preparation"]["observed_seconds"], .2)
        self.assertEqual(by_category["logical_snapshot"]["observed_seconds"], .1)

    def test_profiles_without_weights_merge_equally_and_mixed_forms_are_rejected(self):
        first, second = mergeable_profiles()
        equal = weighted_profile_merge((first, second))
        explicit = weighted_profile_merge(((first, 1), (second, 1)))
        self.assertEqual(equal, explicit)
        _, provenance = weighted_profile_merge((first, second), return_provenance=True)
        self.assertEqual(provenance["input_weight_mode"], "equal")
        self.assertEqual([item["normalized_weight"] for item in provenance["inputs"]], [.5, .5])
        self.assertEqual(
            provenance["operation_coverage"]["dense_contraction"]["weight_policy"],
            "legacy-complete-assumed",
        )
        self.assertEqual([item["version"] for item in provenance["inputs"]],
                         [first.version, second.version])
        with self.assertRaisesRegex(ValueError, "do not mix"):
            weighted_profile_merge((first, (second, 1)))

    def test_weight_validation_zero_weight_exclusion_and_overflow_safe_sum(self):
        first, second = mergeable_profiles()
        merged = weighted_profile_merge(((first, 1e308), (second, 1e308)))
        self.assertAlmostEqual(merged.snapshot_seconds(300),
                               (first.snapshot_seconds(300) + second.snapshot_seconds(300)) / 2)
        ignored = replace(first, hardware_fingerprint="unrelated-device")
        same = weighted_profile_merge(((object(), 0), (ignored, 0), (first, 7)))
        self.assertEqual(same.hardware_fingerprint, first.hardware_fingerprint)
        self.assertAlmostEqual(same.flop_seconds, first.contract_scale * first.flop_seconds)
        with self.assertRaisesRegex(ValueError, "finite"):
            weighted_profile_merge(((first, float("nan")),))
        with self.assertRaisesRegex(ValueError, "finite"):
            weighted_profile_merge(((first, float("inf")),))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            weighted_profile_merge(((first, -1),))
        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_profile_merge(((first, 0), (second, 0)))
        with self.assertRaisesRegex(ValueError, "at least one"):
            weighted_profile_merge(())

    def test_hardware_and_nonrepresentable_feature_ratios_are_explicit(self):
        first, second = mergeable_profiles()
        other_hardware = replace(second, hardware_fingerprint="device-b")
        with self.assertRaisesRegex(ValueError, "different hardware"):
            weighted_profile_merge(((first, 1), (other_hardware, 1)))
        expected = weighted_profile_merge(((first, 1), (other_hardware, 1)),
                                          allow_cross_hardware=True)
        self.assertTrue(expected.hardware_fingerprint.startswith("expected-mix:"))

        bad_sparse_ratio = replace(second, sparse_kernel_call_scale=1.0)
        with self.assertRaisesRegex(ValueError, "sparse kernel-call"):
            weighted_profile_merge(((first, 1), (bad_sparse_ratio, 1)))

        incompatible_prep = replace(second, preparation_kernel_call_seconds=100.0)
        with self.assertRaisesRegex(ValueError, "preparation_kernel_call_seconds"):
            weighted_profile_merge(((first, 1), (incompatible_prep, 1)))

    def test_profile_provenance_roundtrip_preserves_sources_and_weights(self):
        first, second = mergeable_profiles()
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.json"
            b = Path(directory) / "b.json"
            merged_path = Path(directory) / "merged.json"
            save_cost_profile(first, a)
            save_cost_profile(second, b)
            from lqcdweave.contraction.performance.api import merge_profile_files

            profile, provenance = merge_profile_files(((a, 1), (b, 3)), return_provenance=True)
            save_cost_profile(profile, merged_path, provenance=provenance)
            loaded = load_cost_profile(merged_path)
            self.assertEqual(loaded, profile)
            metadata = load_cost_profile_document(merged_path)["provenance"]
            self.assertEqual([item["source"] for item in metadata["inputs"]], [str(a), str(b)])
            self.assertEqual([item["normalized_weight"] for item in metadata["inputs"]], [.25, .75])

    def test_standalone_fit_and_merge_cli_roundtrip_with_held_out_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = save_measurement(measured_report(1., 1), root / "training.json")
            validation_path = save_measurement(measured_report(1.1, 2), root / "validation.json")
            fitted_path = root / "fitted.json"
            fit_command = [
                sys.executable, "-m", "lqcdweave.contraction.performance.profile", "fit",
                "--training-report", str(training_path),
                "--validation-report", str(validation_path),
                "--training-run-id", "run-training-01",
                "--validation-run-id", "run-validation-02",
                "--hardware", "cpu-test", "--verified", "--output", str(fitted_path),
            ]
            fit_result = subprocess.run(fit_command, cwd=PROJECT_ROOT, text=True,
                                        capture_output=True, check=False)
            self.assertEqual(fit_result.returncode, 0, fit_result.stderr)
            fitted = load_cost_profile(fitted_path)
            self.assertEqual(fitted.hardware_fingerprint, "cpu-test")
            fit_metadata = load_cost_profile_document(fitted_path)["provenance"]
            self.assertEqual(fit_metadata["training_sources"][0]["run_id"], "run-training-01")
            self.assertEqual(fit_metadata["validation_sources"][0]["run_id"], "run-validation-02")
            self.assertIn("dense_contraction", fit_metadata["fit_diagnostics"]["validation"])
            self.assertEqual(
                fit_metadata["fit_diagnostics"]["operation_coverage"]["sparse_contraction"]["status"],
                "base-unobserved",
            )
            self.assertGreater(
                fitted.contract_seconds(100, 50, kind="sparse_bridge"), 0,
                "an operation absent from measurements retains its base model, not a zero prediction",
            )

            other = replace(fitted, flop_seconds=fitted.flop_seconds * 2)
            other_path = root / "other.json"
            save_cost_profile(other, other_path, provenance=fit_metadata)
            merged_path = root / "merged.json"
            merge_command = [
                sys.executable, "-m", "lqcdweave.contraction.performance.profile", "merge",
                "--component", str(fitted_path), "1e308",
                "--component", str(other_path), "1e308",
                "--output", str(merged_path),
            ]
            merge_result = subprocess.run(merge_command, cwd=PROJECT_ROOT, text=True,
                                          capture_output=True, check=False)
            self.assertEqual(merge_result.returncode, 0, merge_result.stderr)
            merged = load_cost_profile(merged_path)
            self.assertAlmostEqual(merged.flop_seconds,
                                   (fitted.contract_scale * fitted.flop_seconds
                                    + other.contract_scale * other.flop_seconds) / 2)
            metadata = load_cost_profile_document(merged_path)["provenance"]
            self.assertEqual([item["weight"] for item in metadata["inputs"]], [1e308, 1e308])
            self.assertEqual(len(metadata["inputs"]), 2)
            self.assertEqual([item["version"] for item in metadata["inputs"]],
                             [fitted.version, other.version])
            self.assertEqual(
                metadata["operation_coverage"]["dense_contraction"]["weight_policy"],
                "base-fallback-global-weights",
            )
            self.assertFalse(metadata["operation_coverage"]["dense_contraction"]["calibrated"])

            equal_path = root / "equal.json"
            equal_command = [
                sys.executable, "-m", "lqcdweave.contraction.performance.profile", "merge",
                "--profile", str(fitted_path), "--profile", str(other_path),
                "--output", str(equal_path),
            ]
            equal_result = subprocess.run(equal_command, cwd=PROJECT_ROOT, text=True,
                                          capture_output=True, check=False)
            self.assertEqual(equal_result.returncode, 0, equal_result.stderr)
            equal_metadata = load_cost_profile_document(equal_path)["provenance"]
            self.assertEqual(equal_metadata["input_weight_mode"], "equal")
            self.assertAlmostEqual(load_cost_profile(equal_path).flop_seconds,
                                   merged.flop_seconds)
            self.assertNotEqual(metadata["inputs"][0]["source"], metadata["inputs"][1]["source"])

    def test_cli_fits_weighted_sources_and_holds_out_shape_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_a = save_measurement(linear_report(2, offset=1), root / "train-a.json")
            train_b = save_measurement(linear_report(10, count=8, offset=2), root / "train-b.json")
            heldout = save_measurement(linear_report(5, offset=3), root / "heldout.json")
            output = root / "weighted-fit.json"
            command = [
                sys.executable, "-m", "lqcdweave.contraction.performance.profile", "fit",
                "--training-source", str(train_a), "1", "source-a", "run-a", "family-a",
                "--training-source", str(train_b), "3", "source-b", "run-b", "family-b",
                "--validation-source", str(heldout), "validation", "run-heldout", "family-heldout",
                "--validation-by", "shape_family",
                "--hardware", "cpu-test", "--verified", "--output", str(output),
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, text=True,
                                    capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            fitted = load_cost_profile(output)
            self.assertAlmostEqual(fitted.kernel_call_seconds_complex128, 8.)
            provenance = load_cost_profile_document(output)["provenance"]
            self.assertEqual([source["weight"] for source in provenance["training_sources"]], [1., 3.])
            self.assertEqual(provenance["validation_by"], "shape_family")
            coverage = provenance["fit_diagnostics"]["operation_coverage"]["dense_contraction"]
            self.assertEqual(coverage["status"], "fitted")
            self.assertEqual(coverage["source_weights"], {"source-a": .25, "source-b": .75})

    def test_cli_does_not_overwrite_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = mergeable_profiles()
            a = save_cost_profile(first, root / "a.json")
            b = save_cost_profile(second, root / "b.json")
            output = root / "existing.json"
            output.write_text("keep", encoding="utf-8")
            command = [
                sys.executable, "-m", "lqcdweave.contraction.performance.profile", "merge",
                "--component", str(a), "1", "--component", str(b), "1",
                "--output", str(output),
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, text=True,
                                    capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_fit_rejects_same_report_and_requires_distinct_run_declarations(self):
        from lqcdweave.contraction.performance.api import fit_cost_profile

        report = measured_report(1., 1)
        with self.assertRaisesRegex(ValueError, "run identities"):
            fit_cost_profile(CostProfile(), report, measured_report(1.1, 2),
                             verified=True, hardware_fingerprint="cpu",
                             training_run_id="same", validation_run_id="same")
        with self.assertRaisesRegex(ValueError, "separate run"):
            fit_cost_profile(CostProfile(), report, report, verified=True,
                             hardware_fingerprint="cpu", training_run_id="train",
                             validation_run_id="validation")

    def test_fit_explicitly_rejects_direct_reports_marked_ineligible(self):
        direct = replace(
            measured_report(1., 1),
            execution_profile={"calibration_eligible": False},
        )
        training = (ProfileFitReport(
            direct, "direct-training", run_id="run-direct", shape_family="family-direct",
            weight=1.,
        ),)
        validation = (ProfileFitReport(
            measured_report(1.1, 2), "heldout", run_id="run-heldout",
            shape_family="family-heldout",
        ),)
        with self.assertRaisesRegex(ValueError, "direct contraction reports are not calibration-eligible"):
            fit_cost_profile_sources(
                CostProfile(), training, validation, verified=True,
                hardware_fingerprint="cpu-test", validation_by="run",
            )

    def test_fit_files_reject_reused_path_and_report(self):
        from lqcdweave.contraction.performance.api import fit_profile_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = save_measurement(measured_report(1., 1), root / "one.json")
            duplicate = root / "copy.json"
            duplicate.write_bytes(report.read_bytes())
            kwargs = dict(verified=True, hardware_fingerprint="cpu-test",
                          training_run_id="run-a", validation_run_id="run-b")
            with self.assertRaisesRegex(ValueError, "paths must differ"):
                fit_profile_files(None, report, report, **kwargs)
            with self.assertRaisesRegex(ValueError, "contents are identical"):
                fit_profile_files(None, report, duplicate, **kwargs)

    def test_source_total_weights_are_invariant_to_repeating_one_sources_samples(self):
        base = CostProfile()
        validation = (ProfileFitReport(
            linear_report(5, offset=99), "heldout", run_id="run-heldout",
            shape_family="family-heldout",
        ),)

        def fit(second_count):
            training = (
                ProfileFitReport(linear_report(2, offset=1), "source-a", run_id="run-a",
                                 shape_family="family-a", weight=1),
                ProfileFitReport(linear_report(10, second_count, baryon_observed=12, offset=2),
                                 "source-b", run_id="run-b", shape_family="family-b", weight=3),
            )
            return fit_cost_profile_sources(
                base, training, validation, verified=True,
                hardware_fingerprint="cpu-test", validation_by="run",
            )

        once, once_diagnostics = fit(1)
        repeated, repeated_diagnostics = fit(10)
        # Independent oracle: each source contributes total weight 1 or 3,
        # regardless of how many operation samples that source contains.
        self.assertAlmostEqual(once.kernel_call_seconds_complex128, (1 * 2 + 3 * 10) / 4)
        self.assertAlmostEqual(repeated.kernel_call_seconds_complex128,
                               once.kernel_call_seconds_complex128)
        self.assertEqual(once_diagnostics["operation_coverage"]["dense_contraction"]["source_weights"],
                         {"source-a": .25, "source-b": .75})
        self.assertEqual(once_diagnostics["operation_coverage"]["dense_contraction"]["source_sample_counts"],
                         {"source-a": 1, "source-b": 1})
        self.assertEqual(repeated_diagnostics["operation_coverage"]["dense_contraction"]["source_weights"],
                         {"source-a": .25, "source-b": .75})
        self.assertEqual(repeated_diagnostics["operation_coverage"]["dense_contraction"]["source_sample_counts"],
                         {"source-a": 1, "source-b": 10})
        weighted_error = once_diagnostics["operation_coverage"]["dense_contraction"]
        self.assertAlmostEqual(weighted_error["weighted_training_mean_absolute_error_seconds"], 3.)
        self.assertAlmostEqual(
            weighted_error["validation_errors_by_report"]["heldout@run-heldout"]["absolute_error_seconds"],
            3.,
        )
        self.assertEqual(once_diagnostics["source_sample_counts"]["source-b"], 2)
        self.assertEqual(repeated_diagnostics["source_sample_counts"]["source-b"], 11)
        self.assertEqual(once_diagnostics["operation_coverage"]["sparse_contraction"]["status"],
                         "base-unobserved")
        self.assertGreater(once.contract_seconds(100, 50, kind="sparse_bridge"), 0)
        self.assertEqual(once_diagnostics["operation_coverage"]["baryon_join"]["status"],
                         "unsupported")
        self.assertEqual(once.baryon_join_scale, base.baryon_join_scale)

    def test_fit_holdout_may_be_disjoint_by_run_or_shape_family(self):
        training = (ProfileFitReport(
            linear_report(2, offset=1), "training", run_id="shared-run",
            shape_family="train-family", weight=1,
        ),)
        validation = (ProfileFitReport(
            linear_report(5, offset=2), "heldout", run_id="shared-run",
            shape_family="heldout-family",
        ),)
        args = dict(verified=True, hardware_fingerprint="cpu-test")
        with self.assertRaisesRegex(ValueError, "run_id groups overlap"):
            fit_cost_profile_sources(CostProfile(), training, validation,
                                     validation_by="run", **args)
        _, family_diagnostics = fit_cost_profile_sources(
            CostProfile(), training, validation, validation_by="shape_family", **args
        )
        self.assertEqual(family_diagnostics["validation_by"], "shape_family")
        self.assertIn("heldout-family", family_diagnostics["validation_groups"])

        same_family = (replace(validation[0], shape_family="train-family", run_id="other-run"),)
        with self.assertRaisesRegex(ValueError, "shape_family groups overlap"):
            fit_cost_profile_sources(CostProfile(), training, same_family,
                                     validation_by="shape_family", **args)

    def test_operation_coverage_marks_rank_deficient_and_unsupported_without_zeroing_base(self):
        dense_rank_deficient = ReplayReport((), (
            ActionTiming(
                "contract", .01, 2.0, 0, "dense",
                model_features={"kernel_calls": 1, "flops": 100,
                                "traffic_bytes": 200, "dtype": "complex128"},
            ),
            ActionTiming(
                "baryon_join", .01, 3.0, 1, "baryon", operands=("a", "b", "c"),
                model_features={"kernel_calls": 1, "flops": 100,
                                "traffic_bytes": 200, "dtype": "complex128"},
            ),
        ), None, None)
        validation = (ProfileFitReport(
            linear_report(9, offset=87), "validation", run_id="validation-run",
            shape_family="validation-family",
        ),)
        base = CostProfile()
        fitted, diagnostics = fit_cost_profile_sources(
            base,
            (ProfileFitReport(dense_rank_deficient, "training", run_id="training-run",
                              shape_family="training-family", weight=1),),
            validation, verified=True, hardware_fingerprint="cpu-test",
            validation_by="run",
        )
        coverage = diagnostics["operation_coverage"]
        self.assertEqual(coverage["dense_contraction"]["status"], "base-insufficient")
        self.assertNotIn("p2p_read", coverage)
        self.assertEqual(coverage["baryon_join"]["status"], "unsupported")
        self.assertEqual(fitted.kernel_call_seconds_complex128, base.kernel_call_seconds_complex128)
        self.assertEqual(fitted.flop_seconds, base.flop_seconds)
        self.assertEqual(fitted.baryon_join_scale, base.baryon_join_scale)


if __name__ == "__main__":
    unittest.main()
