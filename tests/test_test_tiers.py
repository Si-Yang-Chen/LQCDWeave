"""Tier reporting must not hide skips, collection errors or real regressions."""
import errno
import importlib.util
import io
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "tools" / "test_tiers.py"
spec = importlib.util.spec_from_file_location("test_tiers_tool", path)
tiers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiers)


class TestTierTests(unittest.TestCase):
    def test_unit_integration_reference_and_gpu_membership(self):
        for identity, expected in (
            ("storage.test_dataset.DatasetTests.test_read", "cpu-unit"),
            ("test_dataset_workflow.WorkflowTests.test_handoff", "cpu-integration"),
            ("test_dilution_weights.DilutionComponentWeightTests.test_actual_equal_and_distinct_component_weights", "cpu-unit"),
            ("test_dilution_correlator.DilutionCorrelatorTests.test_same_pool_second_moment_complex_spin_and_source_dagger", "scientific-reference"),
            ("test_dilution_component_reference.DilutionComponentReferenceTests.test_three_disconnected_currents_match_all_component_equalities", "scientific-reference"),
            ("test_weak_field_dilution_subset.WeakFieldDilutionSubsetTests.test_actual_nonzero_momentum_source_dd_fixed_vv_subset_mean", "scientific-reference"),
            ("adapters.test_dilution_readers.DilutionElementalReaderTests.test_registered_dilution_sources_survive_view_reorder_and_persistence", "cpu-unit"),
            ("production.elemental.test_dilution_sectors.DilutionSectorProductionTests.test_nine_sectors_independent_spatial_temporal_transport_reference", "cpu-unit"),
            ("test_small_scientific_pipeline.SmallScientificPipelineTests.test_default_full_time_pipeline_and_save_load", "scientific-reference"),
            ("production.elemental.test_baryon_elemental.BaryonProductionTests.test_cupy_paths_projection_and_streaming", "gpu"),
            ("production.elemental.test_baryon_elemental.BaryonProductionTests.test_paths_all_legs_against_determinants_all_times", "cpu-unit"),
            ("correlator.execution.test_sparse_retained_accumulation.RetainedSparseGPUChecks.test_mixed_precision_preserves_output_precision", "gpu"),
            ("correlator.execution.test_raw_preparation._CuPyCase.test_native_parameter_slots_follow_current_stream_and_chunk_offsets", "gpu"),
            ("correlator.execution.test_cfg_instance_preparation.CfgInstancePreparationTests.test_current_ids_bind_once_order_duplicates_and_replaced_blocks_remain_current", "cpu-unit"),
            ("correlator.execution.test_cfg_instance_preparation.CuPyCfgInstancePreparationTests.test_current_instance_binding_matches_numpy_and_debug_on_changed_strides", "gpu"),
            ("correlator.execution.test_cfg_template_preparation.CuPyCfgTemplatePreparationTests.test_nested_profile_dagger_casts_match_reference_and_debug_for_two_gid_sets", "gpu"),
            ("correlator.execution.test_raw_consumer_arguments.RawConsumerArgumentsCuPyTests.test_concat_and_snapshot_use_fresh_deferred_addresses_on_current_stream", "gpu"),
            ("correlator.execution.test_pointer_matrix_product.PointerMatrixCuPyTests.test_cuda_fixed_geometry_current_stream_fresh_owners_and_fallbacks", "gpu"),
            ("correlator.execution.test_packed_matrix_product.PackedMatrixProductCuPyTests.test_cuda_required_packing_current_stream_and_fresh_results_without_dot", "gpu"),
            ("correlator.execution.test_direct_matrix_packing.DirectMatrixPackingCuPyTests.test_cuda_original_strides_current_stream_and_fresh_outputs_without_input_views", "gpu"),
            ("correlator.execution.test_raw_driver_arguments.RawDriverArgumentsCuPyTests.test_signed_pointer_constant_table_reuse_on_nondefault_stream", "gpu"),
        ):
            with self.subTest(identity=identity):
                self.assertEqual(tiers.tier_for(identity), expected)

    def test_device_oracles_do_not_capture_cpu_siblings_or_host_compilers(self):
        for identity, expected in (
            ("correlator.execution.test_bound_output_views.RealCuPyBoundOutputViewTests.test_view", "gpu"),
            ("correlator.execution.test_cartesian_group_execution.CartesianGroupPublicCuPyTests.test_public", "gpu"),
            ("correlator.execution.test_dense_store.DenseStoreTests.test_real_cupy_unary_snapshot_and_binary_store_rebind", "gpu"),
            ("correlator.execution.test_dense_store.DenseStoreTests.test_cpu_store", "cpu-unit"),
            ("correlator.execution.test_device_output_program.GeneratedOutputCodeTests.test_empty_scatter_zero_and_literal_arrays", "cpu-unit"),
            ("correlator.execution.test_fused_preparation.FusedPreparationHostTests.test_kernel", "cpu-unit"),
            ("correlator.execution.test_replay_hotpath.PackingOracleTests.test_packing", "cpu-unit"),
            ("correlator.planning.test_dense_batch_subgrids.DenseBatchSubgridCuPyTests.test_device", "gpu"),
        ):
            with self.subTest(identity=identity):
                self.assertEqual(tiers.tier_for(identity), expected)

    def _report(self, *cases):
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=tiers.TierResult).run(
            unittest.TestSuite(cases))
        return tiers.summarize(result, tier="test", selected=len(cases))

    def test_class_skip_does_not_subtract_from_executed_successes(self):
        class Passed(unittest.TestCase):
            def runTest(self):
                pass
        class MissingPlatform(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("requires a working CUDA device")
            def runTest(self):
                self.fail("must not run")
        report = self._report(Passed(), MissingPlatform())
        self.assertEqual((report["tests_run"], report["passed"], report["skipped"]), (1, 1, 1))
        self.assertEqual(report["diagnostics"][0]["classification"], "environment-platform")
        self.assertEqual(tiers.exit_code(report), 2)

    def test_skip_is_incomplete_and_unrecognized_reason_stays_unclassified(self):
        class Skipped(unittest.TestCase):
            def runTest(self):
                self.skipTest("an unspecified condition")
        report = self._report(Skipped())
        self.assertEqual((report["passed"], report["skipped"]), (0, 1))
        self.assertEqual(report["diagnostics"][0]["classification"], "unclassified-skip")
        self.assertEqual(tiers.exit_code(report), 2)

    def test_explicit_compiler_and_filesystem_prerequisites_remain_incomplete(self):
        for reason, expected in (
            ('host C++ compiler unavailable; this is not a GPU test', 'environment-dependency'),
            ('POSIX filesystem required', 'environment-filesystem'),
        ):
            with self.subTest(reason=reason):
                class MissingPrerequisite(unittest.TestCase):
                    def runTest(self):
                        self.skipTest(reason)
                report = self._report(MissingPrerequisite())
                self.assertEqual(report['diagnostics'][0]['classification'], expected)
                self.assertEqual(report['diagnostics'][0]['reason'], reason)
                self.assertEqual(report['skipped'], 1)
                self.assertEqual(report['passed'], 0)
                self.assertEqual(tiers.exit_code(report), 2)

    def test_failure_and_subtest_failure_are_not_converted_to_skips(self):
        class NumericalBug(unittest.TestCase):
            def runTest(self):
                with self.subTest(coordinate=0):
                    self.assertEqual(1, 2, "numerical mismatch")
        class CleanupBug(unittest.TestCase):
            def runTest(self):
                self.fail("worker cleanup assertion")
        report = self._report(NumericalBug(), CleanupBug())
        self.assertEqual((report["passed"], report["failures"], report["skipped"]), (0, 2, 0))
        self.assertTrue(all(item["classification"] == "test-failure" for item in report["diagnostics"]))
        self.assertEqual(tiers.exit_code(report), 1)

    def test_socket_denial_is_annotated_but_keeps_error_outcome(self):
        # Synthetic traceback frames exercise diagnosis without creating a real
        # socket or depending on the host sandbox's permission settings.
        namespace = {"errno": errno}
        exec(compile("def denied():\n    raise PermissionError(errno.EACCES, 'socket denied')\n",
                     "/stdlib/socket.py", "exec"), namespace)
        exec(compile("def manager():\n    denied()\n",
                     "/stdlib/multiprocessing/connection.py", "exec"), namespace)
        class Restricted(unittest.TestCase):
            def runTest(self):
                namespace["manager"]()
        report = self._report(Restricted())
        self.assertEqual((report["errors"], report["skipped"]), (1, 0))
        self.assertEqual(report["diagnostics"][0]["classification"], "environment-restriction")
        self.assertEqual(tiers.exit_code(report), 1)

    def test_unknown_import_and_permission_errors_are_not_assumed_environmental(self):
        for exception in (ModuleNotFoundError("wrong import", name="lqcdweave.typo"),
                          PermissionError(errno.EACCES, "not a multiprocessing socket"),
                          AssertionError("CUDA numerical mismatch")):
            with self.subTest(exception=exception):
                self.assertEqual(tiers.error_category((type(exception), exception, None)), "test-failure")
        missing = ModuleNotFoundError("missing scipy", name="scipy")
        self.assertEqual(tiers.error_category((type(missing), missing, None)), "environment-dependency")
        missing_proxy = ModuleNotFoundError(
            "missing lazy object proxy", name="lazy_object_proxy")
        self.assertEqual(
            tiers.error_category((type(missing_proxy), missing_proxy, None)),
            "environment-dependency",
        )

    def test_collection_error_cannot_be_hidden_by_a_tier_filter(self):
        from contextlib import redirect_stderr, redirect_stdout
        from unittest.mock import patch
        loader = unittest.TestLoader()
        failed = loader.loadTestsFromName("missing_module_for_tier_reporting_test")
        with patch.object(tiers.unittest, "TestLoader", return_value=loader), \
                patch.object(loader, "discover", return_value=failed), \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(tiers.main(["--tier", "scientific-reference"]), 1)

    def test_slow_tier_is_not_part_of_default_or_all_discovery(self):
        from contextlib import redirect_stdout
        from unittest.mock import patch

        class RegularCase(unittest.TestCase):
            __module__ = "test_regular_tier_case"
            def runTest(self):
                pass

        tests_dir = str(tiers.ROOT / "tests")
        for arguments in ([], ["--tier", "all"]):
            loader = unittest.TestLoader()
            with self.subTest(arguments=arguments), \
                    patch.object(tiers.unittest, "TestLoader", return_value=loader), \
                    patch.object(loader, "discover", return_value=unittest.TestSuite([RegularCase()])) as discover, \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(tiers.main(arguments + ["--list"]), 0)
            discover.assert_called_once_with(tests_dir, top_level_dir=tests_dir)
            self.assertNotIn("slow.", output.getvalue())

    def test_slow_discovery_runs_only_for_explicit_tier(self):
        from contextlib import redirect_stdout
        from unittest.mock import patch

        class SlowCase(unittest.TestCase):
            __module__ = "slow.weak_field_pipeline"
            def runTest(self):
                pass

        loader = unittest.TestLoader()
        tests_dir = str(tiers.ROOT / "tests")
        slow_dir = str(tiers.ROOT / "tests" / "slow")

        def discover(start_dir, **kwargs):
            if start_dir == tests_dir:
                return unittest.TestSuite()
            if start_dir == slow_dir:
                return unittest.TestSuite([SlowCase()])
            self.fail(f"unexpected discovery path: {start_dir}")

        with patch.object(tiers.unittest, "TestLoader", return_value=loader), \
                patch.object(loader, "discover", side_effect=discover) as discover_mock, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(tiers.main(["--tier", "slow", "--list"]), 0)
        self.assertEqual(discover_mock.call_count, 2)
        self.assertEqual(discover_mock.call_args_list[0].args, (tests_dir,))
        self.assertEqual(discover_mock.call_args_list[0].kwargs, {"top_level_dir": tests_dir})
        self.assertEqual(discover_mock.call_args_list[1].args, (slow_dir,))
        self.assertEqual(discover_mock.call_args_list[1].kwargs, {
            "pattern": "*_pipeline.py", "top_level_dir": tests_dir,
        })
        self.assertIn("slow\tslow.weak_field_pipeline.", output.getvalue())
        self.assertIn("SlowCase.runTest", output.getvalue())

    def test_temporary_discovery_runs_only_for_explicit_tier(self):
        from contextlib import redirect_stdout
        from unittest.mock import patch

        class TemporaryCase(unittest.TestCase):
            __module__ = "temporary.tmp_example"
            def runTest(self):
                pass

        loader = unittest.TestLoader()
        tests_dir = str(tiers.ROOT / "tests")
        temporary_dir = str(tiers.ROOT / "tests" / "temporary")

        def discover(start_dir, **kwargs):
            if start_dir == tests_dir:
                return unittest.TestSuite()
            if start_dir == temporary_dir:
                return unittest.TestSuite([TemporaryCase()])
            self.fail(f"unexpected discovery path: {start_dir}")

        with patch.object(tiers.unittest, "TestLoader", return_value=loader), \
                patch.object(loader, "discover", side_effect=discover) as discover_mock, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(tiers.main(["--tier", "temporary", "--list"]), 0)
        self.assertEqual(discover_mock.call_args_list[1].args, (temporary_dir,))
        self.assertEqual(discover_mock.call_args_list[1].kwargs, {
            "pattern": "tmp_*.py", "top_level_dir": tests_dir,
        })
        self.assertIn("temporary\ttemporary.tmp_example.", output.getvalue())
        with patch.object(tiers.unittest, "TestLoader", return_value=loader), \
                patch.object(loader, "discover", side_effect=discover) as discover_mock, \
                redirect_stdout(io.StringIO()):
            tiers.main(["--tier", "all", "--list"])
        self.assertEqual(discover_mock.call_count, 1)

    def test_slow_entry_keeps_regular_discovery_import_errors(self):
        from contextlib import redirect_stderr, redirect_stdout
        from unittest.mock import patch

        loader = unittest.TestLoader()
        failed = loader.loadTestsFromName("missing_module_for_slow_tier_test")
        tests_dir = str(tiers.ROOT / "tests")
        slow_dir = str(tiers.ROOT / "tests" / "slow")

        def discover(start_dir, **kwargs):
            if start_dir == tests_dir:
                return failed
            if start_dir == slow_dir:
                return unittest.TestSuite()
            self.fail(f"unexpected discovery path: {start_dir}")

        with patch.object(tiers.unittest, "TestLoader", return_value=loader), \
                patch.object(loader, "discover", side_effect=discover) as discover_mock, \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(tiers.main(["--tier", "slow"]), 1)
        self.assertEqual(discover_mock.call_count, 2)
        self.assertIn('"errors": 1', output.getvalue())

    def test_audit_metrics_are_reported_for_success_failure_and_error(self):
        class PassedWithAudit(unittest.TestCase):
            def runTest(self):
                self.audit_metrics = {"checks": 4, "max_residual": 1e-12}

        class FailedWithAudit(unittest.TestCase):
            def runTest(self):
                self.audit_metrics = {"checks": 3, "stage": "comparison"}
                self.fail("audit mismatch")

        class ErrorWithAudit(unittest.TestCase):
            def runTest(self):
                self.audit_metrics = {"checks": 2, "stage": "load"}
                raise RuntimeError("audit input error")

        cases = (PassedWithAudit(), FailedWithAudit(), ErrorWithAudit())
        report = self._report(*cases)
        self.assertEqual((report["passed"], report["failures"], report["errors"]), (1, 1, 1))
        self.assertEqual(report["audits"], {
            cases[0].id(): {"checks": 4, "max_residual": 1e-12},
            cases[1].id(): {"checks": 3, "stage": "comparison"},
            cases[2].id(): {"checks": 2, "stage": "load"},
        })

    def test_zero_tests_is_not_a_pass(self):
        self.assertEqual(tiers.exit_code(self._report()), 2)

    def test_external_tiers_are_explicitly_not_run(self):
        from contextlib import redirect_stdout
        for tier in ("quda", "mpi", "performance"):
            with self.subTest(tier=tier), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(tiers.main(["--tier", tier]), 2)
            self.assertIn("NOT RUN", output.getvalue())


if __name__ == "__main__":
    unittest.main()
