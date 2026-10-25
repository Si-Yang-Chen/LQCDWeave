"""Immutable constants reuse; numerical inputs and unknown allocations do not."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.contracts.correlator.api import ArrayConstant
from lqcdweave.contraction.execution.constant_cache import DeviceConstantCache
from correlator.execution.test_device import _FakeCupy, _Tracker
from correlator.execution.test_uploaded_input_handoff import _selected


class _PooledArray(np.ndarray):
    @property
    def data(self):
        return SimpleNamespace(mem=SimpleNamespace(size=((self.nbytes + 511) // 512) * 512))


def test_extent_budget_catalog_change_and_unknown_allocations():
    item = ArrayConstant.from_array(np.eye(4, dtype=np.complex64))
    other = ArrayConstant.from_array(2 * np.eye(4, dtype=np.complex64))
    cache = DeviceConstantCache()
    create = lambda: item.array().copy().view(_PooledArray)
    cache.configure(frozenset((item,)), 511)
    assert cache.get(item, create) is not cache.get(item, create)
    assert cache.nbytes == 0
    cache.configure(frozenset((item,)), 512)
    first = cache.get(item, create)
    assert cache.get(item, create) is first and cache.nbytes == 512
    cache.configure(frozenset((other,)), 512)
    assert not cache.values and cache.nbytes == 0
    cache.configure(frozenset((item,)), 512)
    assert cache.get(item, lambda: item.array()) is not None
    assert not cache.values  # NumPy has no device allocation extent contract.
    cache.clear()
    assert not cache.items and cache.nbytes == cache.capacity == 0


def test_public_replay_reuses_constants_but_uploads_current_inputs_and_clears_on_failure():
    selected, propagator = _selected()
    tracker, calls = _Tracker(), []
    tracker.download_depth = 1
    cp = _FakeCupy(tracker)
    cp.cuda.driver = SimpleNamespace(ctxGetCurrent=lambda: 10)
    def asarray(value, **kwargs):
        calls.append(tuple(value.shape))
        return np.array(value, copy=True, **kwargs).view(_PooledArray)
    cp.asarray = asarray
    with (patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                return_value=1 << 40),
          patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                return_value=(1 << 30, 1 << 30))):
        first = selected.run(backend=cp)
        cache = selected._execution_resources._constants
        assert cache.values and cache.nbytes <= cache.capacity
        raw = len(selected.plan.blocks)
        before = len(calls)
        propagator.sectors['vsv'] *= 2
        second = selected.run(backend=cp)
        assert len(calls) - before == raw
        np.testing.assert_allclose(second.value, 4 * first.value, rtol=1e-13, atol=1e-13)
        with patch.object(cp, 'dot', side_effect=RuntimeError('kernel failed'), create=True):
            with pytest.raises(RuntimeError, match='kernel failed'):
                selected.run(backend=cp)
        assert not cache.values and cache.nbytes == 0
        restored = selected.run(backend=cp)
        np.testing.assert_array_equal(restored.value, second.value)
        selected.close()
        assert not cache.values and cache.nbytes == 0
