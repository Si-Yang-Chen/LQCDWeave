"""Module tests split from test_storage_cost_calibration.py; shared numerical fixtures stay unchanged."""

from test_storage_cost_calibration import (
    ActionTiming,
    CostProfile,
    PhaseTiming,
    ReplayReport,
    _CalibrationFixture,
    calibrate_cost_profile,
    replace,
)


class CalibrationTests(_CalibrationFixture):
    def test_independent_coefficients_and_held_out_prediction(self):
        fitted,diagnostics=calibrate_cost_profile(CostProfile(),[self.report()],verified=True,hardware_fingerprint='test')
        self.assertAlmostEqual(fitted.preparation_call_seconds,.01)
        self.assertAlmostEqual(fitted.preparation_kernel_call_seconds,.003)
        self.assertAlmostEqual(fitted.preparation_flop_seconds,.00002)
        self.assertAlmostEqual(fitted.kernel_call_seconds_complex128,.02)
        self.assertAlmostEqual(fitted.flop_seconds,.00003)
        self.assertAlmostEqual(fitted.preparation_seconds(50,400,kernel_calls=3),.0204)
        self.assertNotIn('read_and_host_pack', diagnostics['groups'])
        self.assertNotIn('p2p_read', diagnostics['groups'])
        self.assertNotIn('host_to_device', diagnostics['groups'])
        for row in diagnostics['training'].values():
            self.assertAlmostEqual(row['absolute_error_seconds'],0)

    def test_rank_deficient_and_invalid_evidence(self):
        report=self.report()
        for verified,hardware in [(False,'test'),(True,'')]:
            with self.assertRaises(ValueError):
                calibrate_cost_profile(CostProfile(),[report],verified=verified,hardware_fingerprint=hardware)
        with self.assertRaises(ValueError):
            calibrate_cost_profile(CostProfile(hardware_fingerprint='other'),[report],verified=True,hardware_fingerprint='test')
        single=replace(report,timings=(report.timings[-1],),io_events=())
        fitted,diagnostic=calibrate_cost_profile(CostProfile(),[single],verified=True,hardware_fingerprint='test')
        self.assertFalse(diagnostic['groups']['dense_contraction']['fitted'])
        self.assertEqual(fitted.flop_seconds,CostProfile().flop_seconds)
        invalid=replace(report,timings=(replace(report.timings[0],observed_seconds=float('nan'),
            phases=(PhaseTiming('materialize',float('nan')),)),))
        with self.assertRaises(ValueError):
            calibrate_cost_profile(CostProfile(),[invalid],verified=True,hardware_fingerprint='test')

    def test_explicit_source_weights_ignore_source_sample_count(self):
        def dense_report(observed, count, label):
            timings = tuple(ActionTiming(
                'contract', 1., observed, index, f'{label}-{index}',
                model_features=dict(kernel_calls=1, flops=0, traffic_bytes=0,
                                    dtype='complex128'))
                for index in range(count))
            return ReplayReport((), timings, None, None)

        sources = [dense_report(2., 1, 'a'), dense_report(10., 1, 'b')]
        repeated = [dense_report(2., 1, 'a'), dense_report(10., 10, 'b')]
        options = dict(verified=True, hardware_fingerprint='test',
                       source_weights=(1., 3.), source_ids=('source-a', 'source-b'))
        fitted, diagnostics = calibrate_cost_profile(CostProfile(), sources, **options)
        repeated_fit, repeated_diagnostics = calibrate_cost_profile(
            CostProfile(), repeated, **options)
        expected = (1. * 2. + 3. * 10.) / 4.
        self.assertAlmostEqual(fitted.kernel_call_seconds_complex128, expected)
        self.assertAlmostEqual(repeated_fit.kernel_call_seconds_complex128, expected)
        self.assertEqual(diagnostics['operation_coverage']['dense_contraction']['source_weights'],
                         {'source-a': .25, 'source-b': .75})
        self.assertEqual(repeated_diagnostics['operation_coverage']['dense_contraction']['source_weights'],
                         {'source-a': .25, 'source-b': .75})
        self.assertEqual(diagnostics['source_sample_counts'], {'source-a': 1, 'source-b': 1})
        self.assertEqual(repeated_diagnostics['source_sample_counts'], {'source-a': 1, 'source-b': 10})
