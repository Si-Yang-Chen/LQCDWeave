"""Empirical allocator allowance stays inside the hard cap, without CUDA."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import json

import pytest

from lqcdweave.contracts.correlator.api import device_allocator_headroom
from lqcdweave.contracts.memory import DEFAULT_PLANNING_FRACTION, MemoryLimits
from lqcdweave.contraction.correlator_coordinates import select_coordinate_replay


GIB = 1024 ** 3


def test_empirical_allowance_keeps_pool_limit_without_claiming_device_calibration():
    policy = device_allocator_headroom(28*GIB, device_name='Tesla V100-PCIE-32GB')
    assert policy['reserve_bytes'] == GIB
    assert policy['hard_limit_bytes'] == 28*GIB
    assert policy['numerical_limit_bytes'] == 27*GIB
    assert policy['calibration_status'] == 'empirical_default'
    assert policy['source']['kind'] == 'library_default'
    assert policy['source']['runtime_calibrated'] is False
    json.dumps(policy)


def test_small_budget_uses_existing_slack_cpu_zero_and_unknown_device_is_explicit():
    cap = 128*1024**2
    policy = device_allocator_headroom(cap, device_name='V100')
    assert policy['reserve_bytes'] == cap-int(cap*DEFAULT_PLANNING_FRACTION)
    assert 0 < policy['reserve_bytes'] < cap
    assert policy['calibration_status'] == 'small_budget_fallback'
    assert device_allocator_headroom(28*GIB)['calibration_status'] == 'empirical_default'
    cpu = device_allocator_headroom(cap, backend='numpy')
    assert cpu['reserve_bytes'] == 0
    assert cpu['numerical_limit_bytes'] == cap
    assert cpu['calibration_status'] == 'not_applicable'


@pytest.mark.parametrize('cap', [True, 0, -1, 1.5])
def test_invalid_caps_are_rejected_before_any_runtime(cap):
    with pytest.raises(ValueError, match='positive integer'):
        device_allocator_headroom(cap)


def test_runtime_prefetch_cannot_spend_headroom_already_in_plan_peak():
    from lqcdweave.contraction.execution.read_ahead import _resolve_device_capacity
    cap = 28*GIB
    allocator = device_allocator_headroom(cap)['reserve_bytes']
    # P's peak contains 26 GiB of numerical owners plus the allocator allowance.
    plan = SimpleNamespace(state=SimpleNamespace(problem=SimpleNamespace(backend='cupy')),
        memory_limit=cap, peak_device_bytes=26*GIB+allocator)
    device = SimpleNamespace(limit=cap, constant_reserve=1024)
    with patch('lqcdweave.contraction.execution.device.active_device_execution', return_value=device):
        assert _resolve_device_capacity(plan, None) == GIB-1024


def gpu_calculation(previous):
    from correlator.execution.test_coordinate_replay import selected
    calculation, readers = selected()
    plan = calculation.plan
    problem = replace(plan.state.problem, backend='cupy')
    plan = replace(plan, state=replace(plan.state, problem=problem), memory_limit=28*GIB,
        host_memory_limit=48*GIB, reserved_device_bytes=4096+previous)
    expanded = replace(calculation.expanded, inputs=replace(calculation.inputs, problem=problem))
    path = replace(calculation.path, plan=plan, expanded=expanded, memory_limit=28*GIB,
                   host_memory_limit=48*GIB)
    options = dict(calculation.options, execution_target=SimpleNamespace(name='Tesla V100-PCIE-32GB'))
    if previous:
        options['coordinate_allocator_headroom'] = dict(reserve_bytes=previous)
    calculation = replace(calculation, expanded=expanded, path=path, options=options)
    return calculation, readers


@pytest.mark.parametrize('previous', [0, GIB])
def test_public_coordinate_selection_passes_allowance_once_without_loading_values(previous):
    calculation, readers = gpu_calculation(previous)

    class Planned(Exception):pass
    def capture(template, *args, **kwargs):
        assert template.reserved_device_bytes == 4096
        assert kwargs['allocator_reserve_bytes'] == GIB
        assert kwargs['memory_limit'] == 28*GIB
        raise Planned

    module = 'lqcdweave.contraction.correlator_coordinates.'
    with patch(module+'check_execution_target', return_value=SimpleNamespace(__name__='cupy')), \
         patch(module+'resolve_diagram_memory_limits', return_value=MemoryLimits(48*GIB, 28*GIB)), \
         patch(module+'plan_coordinate_batches', side_effect=capture):
        with pytest.raises(Planned):
            select_coordinate_replay(calculation, time_axes={'source':3, 'sink':range(40)})
    assert sum(reader.reads for reader in readers.values()) == 0


@pytest.mark.parametrize('previous', [0, GIB])
def test_window_replay_uses_same_allowance_once_before_reading_values(previous,tmp_path):
    from lqcdweave.contracts.correlator.api import describe_coordinates
    from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation
    calculation, readers = gpu_calculation(previous)
    coordinates,_=describe_coordinates(time_axes={'source':3,'sink':range(40)})
    class Planned(Exception):pass
    def capture(template,*args,**kwargs):
        assert template.reserved_device_bytes == 4096+GIB
        assert template.memory_limit == 28*GIB
        raise Planned
    module='lqcdweave.contraction.correlator_window_run.'
    with patch(module+'check_execution_target',return_value=SimpleNamespace(__name__='cupy')), \
         patch(module+'resolve_diagram_memory_limits',return_value=MemoryLimits(48*GIB,28*GIB)), \
         patch(module+'plan_cfg_windows',side_effect=capture):
        with pytest.raises(Planned):
            run_cfg_window_calculation(calculation,coordinates=coordinates,
                readers=readers,output=tmp_path)
    assert sum(reader.reads for reader in readers.values()) == 0
