"""Uploaded identity inputs share storage until both planned lifetimes end.

The NumPy-backed protocol proves ownership/order, not real CUDA performance.
"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contraction.execution.device import DeviceExecution, device_execution
from lqcdweave.contraction.execution import preparation
from correlator.execution.test_device import _FakeCupy, _Tracker, _targets


@pytest.mark.parametrize('raw_first', [False, True])
def test_raw_and_logical_release_order_waits_for_last_borrower(raw_first):
    tracker = _Tracker()
    cp = _FakeCupy(tracker)
    closed = []
    host = np.arange(6, dtype=np.float32)
    physical = SimpleNamespace(preparations={}, specs={}, read_block=lambda *a: host,
        close_block=lambda key, owner: closed.append((key, owner)))
    with device_execution(cp, device_memory_limit=1024) as device:
        loader = device.bind_loader(physical)
        value = loader.read_block('raw', None)
        assert loader.borrow_uploaded_block('raw', 'first', value)
        assert loader.borrow_uploaded_block('raw', 'second', value)
        if raw_first:
            loader.close_block('raw', value)
            assert closed and tracker.release_events_recorded == 0
        loader.release_uploaded_borrow('raw', 'first')
        assert tracker.release_events_recorded == 0
        loader.release_uploaded_borrow('raw', 'second')
        if not raw_first:
            assert tracker.release_events_recorded == 0
            loader.close_block('raw', value)
        assert tracker.release_events_recorded == 1
        assert not loader._borrowed_uploads and not loader.host_owners
        np.testing.assert_array_equal(value, host)


def test_injected_non_host_array_cannot_be_adopted():
    tracker = _Tracker()
    cp = _FakeCupy(tracker)
    value = object()
    physical = SimpleNamespace(preparations={}, specs={}, read_block=lambda *a: value,
        close_block=lambda *a: None)
    with device_execution(cp, device_memory_limit=1024) as device:
        loader = device.bind_loader(physical)
        actual = loader.read_block('raw', None)
        assert actual is value
        assert not loader.borrow_uploaded_block('raw', 'logical', value)
        loader.close_block('raw', value)


def _selected(*, mixed=False):
    from test_structured_correlator import fixture, task
    from lqcdweave.contraction.correlator import calculate
    binding, elemental, propagator = fixture()
    if mixed:
        elemental.sectors['v2v'] = elemental.sectors['v2v'].astype(np.complex64)
    selected = calculate(task(structured=True, names=('a',)), vertex_map=binding,
        propagator_map={'S^q': propagator}, configuration='cfg', usedNe=3,
        time_axes={0: 1, 1: 0}, optimize='exact', memory_limit=1 << 26,
        host_memory_limit=1 << 27, target_hardware=_targets()[1], execute=False)
    return selected, propagator


def test_public_gpu_protocol_avoids_same_dtype_identity_copy_and_observes_new_inputs():
    selected, propagator = _selected()
    assert not selected.inputs.task_batches
    tracker = _Tracker()
    cp = _FakeCupy(tracker)
    original_download = DeviceExecution.download
    def download(device, values):
        tracker.download_depth += 1
        try:
            return original_download(device, values)
        finally:
            tracker.download_depth -= 1
    with (patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                return_value=1 << 40),
          patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                return_value=(1 << 30, 1 << 30)),
          patch.object(DeviceExecution, 'download', download),
          patch.object(preparation, '_convert_precision', wraps=preparation._convert_precision) as cast):
        first = selected.run(backend=cp)
        assert not any(call.kwargs.get('copy') is True for call in cast.call_args_list)
        cast.reset_mock()
        reference = selected.run(backend=cp, debug=True)
        assert any(call.kwargs.get('copy') is True for call in cast.call_args_list)
        np.testing.assert_allclose(first.value, reference.value, rtol=1e-13, atol=1e-13)
        propagator.sectors['vsv'] *= 2
        second = selected.run(backend=cp)
        np.testing.assert_allclose(second.value, 4 * first.value, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(reference.value, first.value, rtol=1e-13, atol=1e-13)


def test_body_failure_closes_borrowers_and_uploads():
    selected, _ = _selected()
    tracker = _Tracker()
    cp = _FakeCupy(tracker)
    live_loaders = []
    original = DeviceExecution.bind_loader
    def observe(device, *args, **kw):
        loader = original(device, *args, **kw)
        live_loaders.append(loader)
        return loader
    with (patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                return_value=1 << 40),
          patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                return_value=(1 << 30, 1 << 30)),
          patch.object(DeviceExecution, 'bind_loader', observe),
          patch.object(cp, 'dot', side_effect=RuntimeError('kernel failed'), create=True)):
        with pytest.raises(RuntimeError, match='kernel failed'):
            selected.run(backend=cp)
    assert live_loaders
    assert all(not loader.host_owners and not loader._borrowed_uploads for loader in live_loaders)
    assert all(not loader.device._uploads for loader in live_loaders)


def test_actual_precision_conversion_still_allocates_and_matches_reference():
    selected, _ = _selected(mixed=True)
    tracker = _Tracker()
    cp = _FakeCupy(tracker)
    # This protocol backend enforces that downloads occur inside delivery.
    tracker.download_depth = 1
    with (patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                return_value=1 << 40),
          patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                return_value=(1 << 30, 1 << 30)),
          patch.object(preparation, '_convert_precision', wraps=preparation._convert_precision) as cast):
        first = selected.run(backend=cp)
        assert any(call.kwargs.get('copy') is True for call in cast.call_args_list)
        reference = selected.run(backend=cp, debug=True)
    assert first.value.dtype == np.dtype('complex64')
    np.testing.assert_allclose(first.value, reference.value, rtol=2e-5, atol=1e-6)
