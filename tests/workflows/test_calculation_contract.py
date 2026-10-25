"""W returns one replay handle without flattening outputs or forcing cfg IO."""
from collections.abc import Mapping
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import ContractionTerm, ExecutionResult
from lqcdweave.contraction.execution.api import execute_plan
from lqcdweave.contraction.correlator import Calculation, save_calculation_plan, load_calculation_plan, save_correlator
from lqcdweave.contraction.correlator_cfg import CompleteCalculation, complete_calculation
from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation, CfgWindowResult
from lqcdweave.contraction.contraction_session import ContractionSession, CfgExecutionResult
import test_complete_cfg_output as fixtures


@pytest.fixture(scope='module')
def case():
    helper = fixtures.CompleteOutputTests()
    return helper, helper.selected()


def assert_contract(result, *, status='completed', mode=None):
    assert type(result) is Calculation
    assert isinstance(result.report, Mapping)
    assert result.report['status'] == status
    assert result.report['execution_mode'] == result.execution_mode
    assert result.report['output_mode'] == result.output_mode
    assert result.values is result.value
    if mode is not None:
        assert result.execution_mode == mode


def test_fixed_and_complete_share_contract_caches_and_archive_boundaries(case, tmp_path):
    _, (selected, coords, _, _) = case
    assert_contract(selected, status='planned')
    assert selected.value is None
    assert bool(selected)
    with (patch('lqcdweave.contraction.correlator.expand_correlator', side_effect=AssertionError('expanded')),
          patch('lqcdweave.contraction.correlator.find_contraction_path', side_effect=AssertionError('searched'))):
        first = selected.run()
        second = first.run(debug=True)
    assert_contract(first)
    assert_contract(second)
    assert first._execution_bindings is second._execution_bindings is selected._execution_bindings
    assert first._execution_resources is second._execution_resources is selected._execution_resources
    np.testing.assert_array_equal(first.value['two_point'], second.value['two_point'])
    assert selected.value is None and selected.report['status'] == 'planned'
    first.close()
    np.testing.assert_array_equal(first.value['two_point'], selected.run().value['two_point'])
    complete = complete_calculation(selected, coordinates=coords, readers=selected.inputs.loader.input_readers)
    assert_contract(complete, status='planned', mode='complete_cfg')
    assert CompleteCalculation is Calculation
    for old_constructor in (CompleteCalculation(complete.plan, options=complete.options),
                            CompleteCalculation(plan=complete.plan, options=complete.options)):
        actual = old_constructor.run()
        assert_contract(actual)
        assert actual.value['two_point'].shape[-1] == 2
        np.testing.assert_array_equal(actual.value['two_point'][..., 0], first.value['two_point'])
    changed = replace(complete, plan=replace(complete.plan))
    assert changed.plan is not complete.plan
    archive = tmp_path / 'full.json.gz'
    save_calculation_plan(archive, complete)
    restored = load_calculation_plan(archive, readers=complete.options['readers'], configuration='cfg')
    assert_contract(restored, status='planned')
    assert_contract(restored.run())
    save_correlator(tmp_path / 'memory-output', complete.run())


def test_measured_report_is_mapping_and_keeps_original_evidence(case):
    _, (selected, _, _, _) = case
    measured = selected.run(measure=True)
    assert_contract(measured)
    assert measured.report['execution'] is not None
    assert measured.report.execution_profile is measured.report['execution'].execution_profile


def test_lazy_cfg_status_and_close_do_not_claim_extra_cfgs(case):
    helper, (selected, coords, elemental, propagator) = case
    complete = complete_calculation(selected, coordinates=coords)
    records, closed, claimed = [], [], []
    factory = helper.factory(complete.plan, elemental, propagator, records, closed)
    def configurations():
        for cfg in ('a', 'b'):
            claimed.append(cfg)
            yield cfg
    def stream():
        return replace(complete, options=dict(complete.options,
            configuration=configurations(), source_factory=factory)).run()
    result = stream()
    assert_contract(result, status='pending')
    assert claimed == records == closed == []
    report = result.report
    result.value.close()
    assert report['status'] == 'closed' and claimed == []
    result = stream()
    report = result.report
    # Reports are evidence, never execution controls.
    report['status'] = 'closed'
    values = list(result.value)
    assert result.report is report
    assert_contract(result)
    assert [cfg for cfg, _ in values] == ['a', 'b']
    assert closed == ['a', 'b']
    np.testing.assert_allclose(values[1][1]['two_point'], 16 * values[0][1]['two_point'])


def test_lazy_failure_marks_report_and_releases_reader(case):
    helper, (selected, coords, elemental, propagator) = case
    complete = complete_calculation(selected, coordinates=coords)
    closed = []
    factory = helper.factory(complete.plan, elemental, propagator, [], closed)
    result = replace(complete, options=dict(complete.options,
        configuration=iter(('a',)), source_factory=factory)).run()
    with patch('lqcdweave.contraction.correlator_cfg.stack_point_results', side_effect=RuntimeError('assembly failed')):
        with pytest.raises(RuntimeError, match='assembly failed'):
            next(result.value)
    assert result.report['status'] == 'failed'
    result.value.close()
    assert result.report['status'] == 'failed' and closed == ['a']


def test_dataset_and_window_replay_return_calculation_with_full_coverage(case, tmp_path):
    helper, (selected, coords, elemental, propagator) = case
    complete = complete_calculation(selected, coordinates=coords, output_mode='dataset')
    factory = helper.factory(complete.plan, elemental, propagator, [], [])
    dataset = replace(complete, options=dict(complete.options, configuration='a',
        source_factory=factory, output=tmp_path / 'complete')).run()
    assert_contract(dataset)
    window = run_cfg_window_calculation(selected, coordinates=coords, configuration='a',
        source_factory=factory, output=tmp_path / 'window', window_size=1)
    assert_contract(window, mode='cfg_windows')
    assert isinstance(window, CfgWindowResult)
    assert window.value['a']['directory'] == str(tmp_path / 'window')
    with (patch('lqcdweave.contraction.correlator.expand_correlator', side_effect=AssertionError('expanded')),
          patch('lqcdweave.contraction.correlator.find_contraction_path', side_effect=AssertionError('searched'))):
        again = window.run(output=tmp_path / 'again')
    assert_contract(again)
    assert again.report['coordinate_count'] == window.report['coordinate_count'] == 2
    assert again.report['planned_windows'] == 2
    from lqcdweave.contraction.correlator_output import load_correlator_output
    reference = load_correlator_output(tmp_path / 'complete', ('two_point',), configuration='a')
    for directory in ('window', 'again'):
        np.testing.assert_array_equal(load_correlator_output(tmp_path / directory,
            ('two_point',), configuration='a'), reference)
    with pytest.raises(ValueError, match='not a complete archive'):
        save_calculation_plan(tmp_path / 'partial.json.gz', window)
    with pytest.raises(ValueError, match='already publish'):
        save_correlator(tmp_path / 'republish', window)


def test_session_replays_selected_ir_without_recompile_and_keeps_x_contract(tmp_path):
    session = ContractionSession()
    inputs = {'a': np.arange(3, dtype=np.float64)}
    result = session.tensors(inputs, (ContractionTerm(('a',), 'i->i'),))
    assert_contract(result, mode='session')
    before = result.value[0].copy()
    inputs['a'][:] += 2
    with patch('lqcdweave.contraction.contraction_session.compile_plan_cached', side_effect=AssertionError('recompiled')):
        again = result.run(debug=True)
    assert_contract(again)
    np.testing.assert_array_equal(result.value[0], before)
    np.testing.assert_array_equal(again.value[0], inputs['a'])
    assert result.stats.operation_count == result.report['executed_operations']
    assert type(execute_plan(result.plan, {0: inputs['a']})) is ExecutionResult
    original_stats = {'old': 1}
    old = CfgExecutionResult(values=(np.array(2),), stats=original_stats)
    assert_contract(old)
    assert old.stats['old'] == old.report['old'] == 1
    assert original_stats == {'old': 1}
    from lqcdweave.contraction.correlator import load_correlator
    save_correlator(tmp_path / 'session-values', again, axes=('i',),
        coordinates={'i': (0, 1, 2)}, configuration='test')
    saved = load_correlator(tmp_path / 'session-values', configuration='test')
    np.testing.assert_array_equal(saved.value[0], again.value[0])
    terminal = CfgWindowResult({'cfg': {}}, {'complete': True})
    assert_contract(terminal)
    with pytest.raises(ValueError, match='no selected work'):
        terminal.run()


def test_legacy_array_and_pair_protocols_stay_in_compatibility_layer():
    from lqcdweave.contraction.calculation_compat import session_result
    from lqcdweave.contraction.contraction_session import ContractionSession
    value = np.arange(3.)
    stats = ContractionSession._empty_execution_stats()
    result = session_result(value, stats, None, array=True)
    assert_contract(result)
    np.testing.assert_array_equal(result, value)
    np.testing.assert_array_equal(result + 1, value + 1)
    assert result.shape == value.shape and result.dtype == value.dtype
    result[1] = 4
    assert value[1] == 4
    pair = session_result(value, stats, None, array=True, pair=True)
    unpacked, old_stats = pair
    assert unpacked is pair.value is pair[0]
    assert old_stats is pair.stats is pair[1]


def test_persisted_session_result_replays_prepared_bundle_with_fresh_loader_values():
    from test_contraction_session import ContractionSessionStorageInjectionTests
    from lqcdweave.storage.contraction_plans.api import validate_bundle
    session = ContractionSession.from_plan(ContractionSessionStorageInjectionTests._bundle(),
        validate_bundle=validate_bundle)
    calls = []
    def loader(cfg, key, metadata):
        calls.append((cfg, key))
        return np.full(2, len(calls), dtype=np.float64)
    original = session.execute_cfg('cfg', loader)
    assert_contract(original)
    with patch('lqcdweave.contraction.execution.api.prepare_bundle_execution',
               side_effect=AssertionError('prepared twice')):
        again = original.run()
    assert_contract(again)
    np.testing.assert_array_equal(original.value[0], np.ones(2))
    np.testing.assert_array_equal(again.value[0], 2 * np.ones(2))
    assert len(calls) == 2
    assert 'status' not in original.stats
    assert 'session_seconds' in again.report['timings']
