"""Task ownership must not introduce implicit planning, IO or stale results."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.correlator import Calculation, ContractionTask
from lqcdweave.contraction.correlator_cfg import complete_calculation
import test_complete_cfg_output as fixtures


@pytest.fixture
def case():
    helper = fixtures.CompleteOutputTests()
    return helper, helper.selected()


def task_for(selected):
    task = ContractionTask(object())
    with patch('lqcdweave.contraction.correlator_task_plan.plan_task', return_value=selected) as plan:
        assert task.plan() is task
        assert task.plan() is task
        plan.assert_called_once()
    return task


def test_declaration_is_inert_and_failed_planning_is_retryable(case):
    _, (selected, _, _, _) = case
    with patch('lqcdweave.contraction.correlator_task_plan.plan_task', side_effect=RuntimeError('planning')) as plan:
        task = ContractionTask(object())
        plan.assert_not_called()
        with pytest.raises(RuntimeError, match='task.plan'):
            task.run()
        with pytest.raises(RuntimeError, match='task.plan'):
            task.save_plan('unused.json')
        with pytest.raises(RuntimeError, match='planning'):
            task.plan()
        assert task._selected is None
    with patch('lqcdweave.contraction.correlator_task_plan.plan_task', return_value=selected):
        task.plan()
    assert task.plan_report['status'] == 'planned'
    with pytest.raises(TypeError):
        task.plan_report['status'] = 'changed'
    task.close()
    task.close()
    with pytest.raises(RuntimeError, match='closed'):
        task.run()


def test_run_failure_cleans_up_retries_and_preserves_delivered_values(case):
    _, (selected, _, _, _) = case
    task = task_for(selected)
    with (patch('lqcdweave.contraction.correlator.expand_correlator', side_effect=AssertionError('expanded')),
          patch('lqcdweave.contraction.correlator.find_contraction_path', side_effect=AssertionError('searched'))):
        with patch.object(Calculation, '_run', side_effect=RuntimeError('execution')):
            with patch.object(Calculation, 'close', autospec=True) as close:
                with pytest.raises(RuntimeError, match='execution'):
                    task.run()
                close.assert_called_once_with(selected)
        first, second = task.run(), task.run()
    assert type(first) is Calculation and first is not second
    assert first.report is not second.report
    assert task._selected.value is None
    expected = first.value['two_point'].copy()
    task.close()
    np.testing.assert_array_equal(first.value['two_point'], expected)
    np.testing.assert_array_equal(second.value['two_point'], expected)


def test_busy_task_cannot_be_reentered_or_closed(case):
    _, (selected, _, _, _) = case
    task = task_for(selected)
    original = Calculation._run
    def busy(calculation, **options):
        with pytest.raises(RuntimeError, match='in use'):
            task.run()
        with pytest.raises(RuntimeError, match='in use'):
            task.close()
        return original(calculation, **options)
    with patch.object(Calculation, '_run', busy):
        assert task.run().report['status'] == 'completed'
    task.close()


def test_stream_holds_ownership_until_closed_or_exhausted(case, tmp_path):
    helper, (selected, coords, elemental, propagator) = case
    complete = complete_calculation(selected, coordinates=coords)
    records, closed = [], []
    factory = helper.factory(complete.plan, elemental, propagator, records, closed)
    task = task_for(replace(complete, options=dict(complete.options,
        configuration=iter(('a', 'b')), source_factory=factory)))
    result = task.run()
    assert result.report['status'] == 'pending' and records == []
    with pytest.raises(RuntimeError, match='in use'):
        task.run()
    result.value.close()
    result.value.close()
    assert result.report['status'] == 'closed' and records == []
    again = task.run()
    values = list(again.value)
    assert again.report['status'] == 'completed'
    assert [cfg for cfg, _ in values] == ['a', 'b']
    assert closed == ['a', 'b']
    np.testing.assert_allclose(values[1][1]['two_point'], 16 * values[0][1]['two_point'])
    task.save_plan(tmp_path / 'complete.json.gz')
    task.close()


def test_task_close_closes_unstarted_stream_and_stream_failure_releases_task(case):
    helper, (selected, coords, elemental, propagator) = case
    complete = complete_calculation(selected, coordinates=coords)
    closed = []
    factory = helper.factory(complete.plan, elemental, propagator, [], closed)
    def make_task():
        return task_for(replace(complete, options=dict(complete.options,
            configuration=iter(('a',)), source_factory=factory)))
    task = make_task()
    pending = task.run()
    task.close()
    assert pending.report['status'] == 'closed' and closed == []
    task = make_task()
    result = task.run()
    with patch('lqcdweave.contraction.correlator_cfg.stack_point_results', side_effect=RuntimeError('assembly')):
        with pytest.raises(RuntimeError, match='assembly'):
            next(result.value)
    assert result.report['status'] == 'failed' and closed == ['a']
    assert task.plan() is task
    task.close()


def test_archive_load_and_run_need_no_symbolic_definition(case, tmp_path):
    _, (selected, coords, _, _) = case
    complete = complete_calculation(selected, coordinates=coords, readers=selected.inputs.loader.input_readers)
    task = task_for(complete)
    expected = task.run().value['two_point'].copy()
    archive = tmp_path / 'complete.json.gz'
    with (patch('lqcdweave.contraction.correlator.expand_correlator', side_effect=AssertionError('expanded')),
          patch('lqcdweave.contraction.correlator.find_contraction_path', side_effect=AssertionError('searched'))):
        task.save_plan(archive)
        with ContractionTask.load_plan(archive, readers=complete.options['readers'], configuration='cfg') as restored:
            assert restored.plan() is restored
            np.testing.assert_array_equal(restored.run().value['two_point'], expected)
    task.close()
