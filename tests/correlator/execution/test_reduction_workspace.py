"""Scope/budget protocol tests; native CUDA use is a separately gated tier."""
from types import SimpleNamespace
from threading import Thread
from unittest.mock import patch
import os
import gc
import weakref

import numpy as np
import pytest

from lqcdweave.contraction.execution.device import device_execution
from lqcdweave.contraction.execution.scene_runtime import _prepare_reduction_workspace
from lqcdweave.contraction.execution.reduction_workspace import ReductionWorkspace, _FAILED_WORKSPACES
from correlator.execution.test_device import _FakeCupy, _Tracker
from correlator.execution.test_device_constant_cache import _PooledArray


def backend():
    tracker = _Tracker()
    cp = _FakeCupy(tracker)
    state = dict(device=0, context=10, stream=cp.cuda.compute_stream)
    state['stream'].ptr = 1
    cp.cuda.runtime.getDevice = lambda: state['device']
    cp.cuda.driver = SimpleNamespace(ctxGetCurrent=lambda: state['context'])
    cp.cuda.get_current_stream = lambda: state['stream']
    cp.empty = lambda shape, dtype: np.empty(shape, dtype=dtype).view(_PooledArray)
    return cp, tracker, state


def test_hip_uses_ordinary_partials_without_querying_context():
    cp,tracker,_=backend()
    cp.cuda.runtime.is_hip=True
    with patch.object(cp.cuda.driver,'ctxGetCurrent',side_effect=RuntimeError('unsupported')) as context:
        with device_execution(cp,device_memory_limit=4096) as device:
            assert device.reserve_reduction_workspace(2048)==0
            assert device.reduction_partial(32,np.complex128) is None
            assert device.reduction_workspace.allocations==0
        assert tracker.pools[-1].freed
        context.assert_not_called()


def test_single_buffer_fresh_scope_dtype_and_thread_stream_context_boundaries():
    cp, tracker, state = backend()
    with device_execution(cp, device_memory_limit=4096) as device:
        assert device.reserve_reduction_workspace(1024, output_reserve=512) == 2560
        first = device.reduction_partial(256, np.complex128)
        assert first.ndim == 1 and first.nbytes == 2560
        assert device.reduction_partial(1024, np.complex128) is first
        assert device.reduction_partial(256, np.float64) is None
        assert device.reduction_partial(2561, np.complex128) is None
        for key, other in (('device', 1), ('context', 11),
                           ('stream', SimpleNamespace(ptr=2))):
            previous = state[key]
            state[key] = other
            assert device.reduction_partial(256, np.complex128) is None
            state[key] = previous
        seen = []
        thread = Thread(target=lambda: seen.append(device.reduction_partial(256, np.complex128)))
        thread.start(); thread.join(timeout=5)
        assert not thread.is_alive() and seen == [None]
        reference = weakref.ref(first)
        del first
        assert reference() is not None
    assert reference() is None
    assert tracker.stream_syncs.count('compute') == 1
    with device_execution(cp, device_memory_limit=4096) as other:
        assert other.reduction_workspace.allocations == 0


def test_tight_budget_unknown_allocations_and_optional_oom_use_original_allocation():
    cp, _, _ = backend()
    with device_execution(cp, device_memory_limit=4096) as device:
        device.constant_reserve = 512
        assert device.reserve_reduction_workspace(3072, output_reserve=512) == 0
        assert device.reduction_partial(16, np.complex128) is None
        device.reserve_reduction_workspace(2048, output_reserve=512)
        with patch.object(cp, 'empty', side_effect=MemoryError('optional arena')):
            assert device.reduction_partial(16, np.complex128) is None
        assert device.reduction_workspace.buffer is None
        device.reserve_reduction_workspace(2048, output_reserve=512)
        with patch.object(cp, 'empty', side_effect=lambda shape, dtype: np.empty(shape, dtype)):
            assert device.reduction_partial(16, np.complex128) is None


def test_changed_admission_and_failures_fence_before_releasing_owner():
    cp, tracker, state = backend()
    with pytest.raises(RuntimeError, match='numerical failure'):
        with device_execution(cp, device_memory_limit=4096) as device:
            device.reserve_reduction_workspace(1024)
            first = device.reduction_partial(32, np.complex128)
            ref = weakref.ref(first)
            del first
            seen = []
            stream = state['stream']
            original = stream.synchronize
            def synchronize():
                seen.append(ref() is not None)
                original()
            stream.synchronize = synchronize
            device.reserve_reduction_workspace(3584)
            assert seen == [True] and ref() is None
            second = device.reduction_partial(32, np.complex128)
            ref = weakref.ref(second)
            del second
            raise RuntimeError('numerical failure')
    assert seen == [True, True] and ref() is None


def test_close_after_stream_switch_waits_for_original_compute_stream():
    cp, tracker, state = backend()
    original = state['stream']
    with device_execution(cp, device_memory_limit=4096) as device:
        device.reserve_reduction_workspace(1024)
        device.reduction_partial(32, np.complex128)
        state['stream'] = SimpleNamespace(ptr=2, synchronize=lambda: tracker.stream_syncs.append('other'))
    assert tracker.stream_syncs.index('other') < tracker.stream_syncs.index('compute')
    assert device.reduction_workspace.buffer is None


@pytest.mark.parametrize('failure', ['context', 'synchronize'])
def test_failed_close_retains_unsynchronized_owner_after_scope_gc(failure):
    cp, tracker, state = backend()
    original_sync = state['stream'].synchronize
    before = tuple(_FAILED_WORKSPACES)
    with pytest.raises(RuntimeError):
        with device_execution(cp, device_memory_limit=4096) as device:
            device.reserve_reduction_workspace(1024)
            pool_reference = weakref.ref(device.pool)
            partial = device.reduction_partial(32, np.complex128)
            reference = weakref.ref(partial)
            del partial
            if failure == 'context':
                state['context'] = 11
            else:
                def fail():
                    raise RuntimeError('synchronization failed')
                state['stream'].synchronize = fail
    workspace = device.reduction_workspace
    tracker.pools.clear()  # Remove the protocol fixture's diagnostic ownership.
    del device
    gc.collect()
    assert reference() is not None and pool_reference() is not None
    assert workspace in _FAILED_WORKSPACES
    state['context'] = 10
    state['stream'].synchronize = original_sync
    assert workspace.borrow(32, np.complex128) is None
    # Explicit recovery for this protocol fixture; production never retries
    # retained failures during another calculation or workspace borrow.
    workspace.close()
    gc.collect()
    assert reference() is None and pool_reference() is None
    assert tuple(_FAILED_WORKSPACES) == before


def test_normal_close_does_not_retain_private_pool_with_workspace():
    cp, tracker, _ = backend()
    before = tuple(_FAILED_WORKSPACES)
    with device_execution(cp, device_memory_limit=4096) as device:
        pool_reference = weakref.ref(device.pool)
        device.reserve_reduction_workspace(1024)
        device.reduction_partial(32, np.complex128)
        # Shrinking a reservation closes its old array and must attach the
        # same pool to the newly admitted replacement before another borrow.
        device.reserve_reduction_workspace(2048)
        device.reduction_partial(32, np.complex128)
        workspace = device.reduction_workspace
        assert workspace._pool is device.pool
        workspace.close()
        assert workspace._pool is None
        device.reserve_reduction_workspace(1024)
        device.reduction_partial(32, np.complex128)
        assert workspace._pool is device.pool
    tracker.pools.clear()
    del device
    gc.collect()
    assert pool_reference() is None and workspace._pool is None
    assert tuple(_FAILED_WORKSPACES) == before


def test_scene_admission_deducts_constants_outputs_and_prefetch_before_actions():
    cp, _, _ = backend()
    plan = SimpleNamespace(peak_device_bytes=1024, memory_limit=4096,
        retained_before=(), retained_after=(), explicit_resources=False)
    queue = SimpleNamespace(device_capacity=3072)
    execution = SimpleNamespace(plan=plan, owns_read_ahead=True, owns_storage=True,
                                read_ahead_started=False, read_ahead=queue)
    with device_execution(cp, device_memory_limit=4096) as device:
        device.constant_reserve = 512
        with patch('lqcdweave.contraction.execution.output_queue.active_output_queue',
                   return_value=SimpleNamespace(reserve=512)):
            _prepare_reduction_workspace(execution)
        assert device.reduction_workspace.capacity == 2048
        assert queue.device_capacity == 0
    with device_execution(cp, device_memory_limit=4096) as device:
        execution.read_ahead_started = True
        _prepare_reduction_workspace(execution)
        assert device.reduction_workspace.capacity == 0


def test_cfg_admission_deducts_same_reserve_from_continuous_queue():
    from correlator.execution.test_device import _targets, _complete
    from lqcdweave.contraction.execution.cfg_runtime import _iter_execute_cfg_plan
    _, target = _targets()
    _, complete = _complete(target)
    plan = complete.plan
    cp, _, _ = backend()
    report = {}
    limit = plan.peak_device_bytes + 4096
    with device_execution(cp, device_memory_limit=limit) as device:
        device.constant_reserve = 512
        with (patch('lqcdweave.contraction.execution.cfg_runtime.active_output_queue',
                    return_value=object()),
              patch('lqcdweave.contraction.execution.cfg_runtime.output_queue_reserve',
                    return_value=512)):
            assert list(_iter_execute_cfg_plan(plan, (), None, device_memory_limit=limit, report=report)) == []
        assert device.reduction_workspace.capacity == 3072
        assert report['prefetch_device_capacity_bytes'] == 0
        assert report['device']['reduction_workspace_reserved_bytes'] == 3072


@pytest.mark.skipif(os.environ.get('LQCDWEAVE_TEST_CUDA') != '1', reason='explicit CUDA gate')
def test_cuda_partial_reuse_keeps_results_independent_and_rejects_other_stream():
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA unavailable')
    # The kernel handoff uses only the allocation pointer; shape is private.
    first = cp.RawKernel('extern "C" __global__ void fill(double* p, double x) { p[0]=x; }', 'fill')
    finish = cp.RawKernel('extern "C" __global__ void read(const double* p, double* y) { y[0]=p[0]; }', 'read')
    with device_execution(cp, device_memory_limit=1 << 26) as device:
        device.reserve_reduction_workspace(1 << 20)
        outputs = []
        addresses = []
        for value in range(5):
            partial = device.reduction_partial(8, np.float64)
            addresses.append(partial.data.ptr)
            result = cp.empty(1, cp.float64)
            first((1,), (1,), (partial, np.float64(value)))
            finish((1,), (1,), (partial, result))
            outputs.append(result)
        assert len(set(addresses)) == 1
        with cp.cuda.Stream(non_blocking=True):
            assert device.reduction_partial(8, np.float64) is None
    np.testing.assert_array_equal(cp.asnumpy(cp.concatenate(outputs)), np.arange(5))
