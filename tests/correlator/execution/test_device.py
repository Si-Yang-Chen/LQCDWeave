"""Generic device-scope and cfg upload protocol tests.

The fake backend exercises X's device protocol with NumPy arrays. It does not
measure or claim GPU performance; a real CuPy smoke test is optional below.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from correlator.execution.test_cfg_runtime import SafeReader
from lqcdweave.contracts.correlator.api import ExecutionTarget
from lqcdweave.contracts.memory import TargetHardwareMemory
from lqcdweave.contraction.execution.device import (
    active_device_execution,
    device_execution,
)
from lqcdweave.contraction.execution.cfg_runtime import iter_execute_cfg_plan
from lqcdweave.contraction.correlator import calculate
from lqcdweave.contraction.correlator_cfg import CfgSources, complete_calculation
from test_execution_target_frontend import inputs


class _Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.stream_syncs = []
        self.events_recorded = 0
        self.event_waits = 0
        self.input_events_recorded = 0
        self.release_events_recorded = 0
        self.upload_event_dependencies = []
        self.pools = []
        self.asnumpy_calls = 0
        self.download_depth = 0
        self.local = threading.local()
        self.upload_cfg_by_contents = {}
        self.background_b_read_entered = threading.Event()
        self.release_background_b = threading.Event()
        self.kernel_started = threading.Event()
        self.upload_during_kernel = threading.Event()
        self.fail_background_b_upload = False
        self.gate_background_b = False


class _Pool:
    def __init__(self, tracker):
        self.tracker = tracker
        self.limit = None
        self.freed = False
        tracker.pools.append(self)

    def set_limit(self, *, size):
        self.limit = size

    def malloc(self, size):
        return bytearray(size)

    def free_all_blocks(self):
        self.freed = True


class _Stream:
    def __init__(self, tracker, name):
        self.tracker, self.name = tracker, name

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def synchronize(self):
        with self.tracker.lock:
            self.tracker.stream_syncs.append(self.name)

    def wait_event(self, event):
        assert self.name == 'upload'
        assert event.stream is not None and event.stream.name == 'compute'
        with self.tracker.lock:
            self.tracker.upload_event_dependencies.append((self.name, event.stream.name))


class _Event:
    def __init__(self, tracker):
        self.tracker = tracker
        self.stream = None

    def record(self, stream):
        assert stream.name in ('upload', 'compute')
        self.stream = stream
        with self.tracker.lock:
            self.tracker.events_recorded += 1
            if stream.name == 'upload':
                self.tracker.input_events_recorded += 1
            else:
                self.tracker.release_events_recorded += 1

    def synchronize(self):
        assert self.stream is not None and self.stream.name == 'upload'
        with self.tracker.lock:
            self.tracker.event_waits += 1


class _Device:
    id = 0

    def __init__(self, *_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Cuda:
    def __init__(self, tracker, *, fail_stream=False):
        self.tracker = tracker
        self.runtime = type('Runtime', (), {'is_hip': False})()
        self.fail_stream = fail_stream
        self.compute_stream = _Stream(tracker, 'compute')

    def MemoryPool(self):
        return _Pool(self.tracker)

    def using_allocator(self, _allocator):
        return nullcontext()

    def Device(self, *args):
        return _Device(*args)

    def Stream(self, *, non_blocking):
        assert non_blocking is True
        if self.fail_stream:
            raise RuntimeError('fake stream setup failed')
        return _Stream(self.tracker, 'upload')

    def Event(self):
        return _Event(self.tracker)

    def alloc_pinned_memory(self, size):
        return bytearray(size)

    def get_current_stream(self):
        return self.compute_stream


class _FakeCupy:
    """NumPy-backed implementation of the small CuPy API X uses here."""
    __name__ = 'cupy'
    __version__ = 'protocol-test'
    ndarray = np.ndarray

    def __init__(self, tracker, *, fail_stream=False):
        self.tracker = tracker
        self.cuda = _Cuda(tracker, fail_stream=fail_stream)

    def __getattr__(self, name):
        return getattr(np, name)

    def asarray(self, value, dtype=None, order=None, *, copy=None):
        tracker = self.tracker
        background = threading.current_thread().name.startswith(
            ('contraction-io', 'contraction-prepare'))
        # The staged upload can run on another worker. Match the actual host
        # payload, not thread-local reader state, to retain the cfg assertion.
        cfg = (tracker.upload_cfg_by_contents.get((value.shape, value.dtype.str, value.tobytes()))
               if background else None)
        if background and cfg == 'b':
            if tracker.kernel_started.is_set():
                tracker.upload_during_kernel.set()
            if tracker.fail_background_b_upload:
                raise RuntimeError('fake background upload failed')
        if copy is True:
            return np.array(value, dtype=dtype, order=order, copy=True)
        return np.asarray(value, dtype=dtype, order=order)

    def array(self, value, *args, **kwargs):
        return np.array(value, *args, **kwargs)

    def asnumpy(self, value):
        if self.tracker.download_depth <= 0:
            raise AssertionError('device-to-host conversion occurred outside output delivery')
        self.tracker.asnumpy_calls += 1
        return np.asarray(value)


class _ObservedReader(SafeReader):
    def __init__(self, reader, cfg, scale, records, tracker):
        super().__init__(reader, cfg, scale, records)
        self.tracker = tracker

    def block(self, *args, **kwargs):
        tracker = self.tracker
        if threading.current_thread().name.startswith('contraction-io'):
            tracker.local.cfg = self.cfg
            if self.cfg == 'b' and tracker.gate_background_b:
                tracker.background_b_read_entered.set()
                if not tracker.release_background_b.wait(10):
                    raise RuntimeError('test did not release the background cfg read')
        value = super().block(*args, **kwargs)
        if threading.current_thread().name.startswith('contraction-io'):
            tracker.upload_cfg_by_contents[value.shape, value.dtype.str, value.tobytes()] = self.cfg
        return value


def _targets():
    host = 1 << 31
    return (
        ExecutionTarget('offline-numpy', 'numpy', 'cpu',
                        TargetHardwareMemory(host_available_bytes=host)),
        ExecutionTarget('offline-cupy', 'cupy', 'cuda', TargetHardwareMemory(
            host_available_bytes=host, device_total_bytes=1 << 30,
            device_free_bytes=1 << 30)),
    )


def _complete(target):
    options = inputs()
    # These cases exercise execution ownership/queues, not anytime search.
    # A sub-second search deadline depends on worker startup and CPU load.
    options.pop('search', None)
    options['optimize'] = 'none'
    selected = calculate(**options, target_hardware=target)
    complete = complete_calculation(
        selected,
        readers=selected.expanded.inputs.loader.input_readers,
    )
    return selected, complete


def _factory(complete, selected, tracker, closed):
    base_readers = selected.expanded.inputs.loader.input_readers

    @contextmanager
    def bind(cfg):
        wrappers = {}
        readers = {}
        for request in complete.plan.requests.values():
            for value in (request, *request.aliases):
                original = base_readers[value.reader_role]
                wrapper = wrappers.get(id(original))
                if wrapper is None:
                    wrapper = _ObservedReader(original, cfg, 1 if cfg == 'a' else 2,
                                              [], tracker)
                    wrappers[id(original)] = wrapper
                readers[value.reader_role] = wrapper
        try:
            yield CfgSources(readers)
        finally:
            closed.append(cfg)

    return bind


def _output_worker(values):
    if isinstance(values, dict):
        return {key: _output_worker(value) for key, value in values.items()}
    if isinstance(values, (tuple, list)):
        return type(values)(_output_worker(value) for value in values)
    return np.asarray(values)


def _assert_tree_allclose(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_allclose(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree_allclose(left, right)
        return
    np.testing.assert_allclose(_output_worker(actual), _output_worker(expected),
                               rtol=3e-12, atol=3e-12)


def _run(complete, selected, backend, tracker, cfgs=('a', 'b'), *,
         prefetch_device_bytes=None, source_factory=None, device_memory_limit=None):
    closed = []
    factory = source_factory or _factory(complete, selected, tracker, closed)
    options = dict(complete.options, configuration=iter(cfgs), source_factory=factory,
                   readers=None)
    if prefetch_device_bytes is not None:
        options['prefetch_device_bytes'] = prefetch_device_bytes
    calculation = replace(complete, options=options)
    from lqcdweave.contraction.execution.device import DeviceExecution
    original_download = DeviceExecution.download

    def output_only_download(device, values):
        tracker.download_depth += 1
        try:
            return original_download(device, values)
        finally:
            tracker.download_depth -= 1

    with (patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                return_value=1 << 40),
          patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                return_value=(1 << 30, 1 << 30)),
          patch.object(DeviceExecution, 'download', output_only_download)):
        result = calculation.run(backend=backend, device_memory_limit=device_memory_limit)
        values = dict(result.value)
    return values, result.report, closed


class DeviceProtocolTests(unittest.TestCase):
    def test_scope_restores_context_and_frees_pool_on_setup_and_body_failures(self):
        tracker = _Tracker()
        backend = _FakeCupy(tracker)
        with self.assertRaisesRegex(RuntimeError, 'body failed'):
            with device_execution(backend, device_memory_limit=4096) as device:
                self.assertIs(active_device_execution(), device)
                raise RuntimeError('body failed')
        self.assertIsNone(active_device_execution())
        self.assertTrue(tracker.pools[0].freed)

        setup_tracker = _Tracker()
        with self.assertRaisesRegex(RuntimeError, 'setup failed'):
            with device_execution(_FakeCupy(setup_tracker, fail_stream=True),
                                  device_memory_limit=4096):
                self.fail('failed setup must not enter the scope')
        self.assertIsNone(active_device_execution())
        self.assertTrue(setup_tracker.pools[0].freed)

    def test_invalid_quota_precedes_backend_setup_or_loader_reads(self):
        tracker = _Tracker()
        backend = _FakeCupy(tracker)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            with device_execution(backend, device_memory_limit=0):
                self.fail('invalid quota must not enter device scope')
        self.assertEqual(tracker.pools, [])
        with device_execution(backend, device_memory_limit=4096):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                with device_execution(backend, device_memory_limit=0):
                    self.fail('nested invalid quota must be rejected')
        self.assertIsNone(active_device_execution())

        _, gpu_complete = _complete(_targets()[1])
        factory_calls = []

        @contextmanager
        def factory(cfg):
            factory_calls.append(cfg)
            raise AssertionError('invalid quota must precede source binding')
            yield

        read_tracker = _Tracker()
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            list(iter_execute_cfg_plan(
                gpu_complete.plan, ('a',), factory, backend=_FakeCupy(read_tracker),
                device_memory_limit=0,
            ))
        self.assertEqual(factory_calls, [])
        self.assertEqual(read_tracker.pools, [])

    def test_offline_cpu_and_gpu_cfg_outputs_match_without_per_action_sync(self):
        cpu_target, gpu_target = _targets()
        cpu_selected, cpu_complete = _complete(cpu_target)
        gpu_selected, gpu_complete = _complete(gpu_target)
        self.assertEqual(cpu_complete.plan.backend, 'numpy')
        self.assertEqual(gpu_complete.plan.backend, 'cupy')
        self.assertEqual(cpu_complete.plan.coordinate_shape,
                         gpu_complete.plan.coordinate_shape)

        cpu_tracker = _Tracker()
        cpu_outputs, _, cpu_closed = _run(
            cpu_complete, cpu_selected, np, cpu_tracker)
        self.assertEqual(cpu_closed, ['a', 'b'])

        tracker = _Tracker()
        backend = _FakeCupy(tracker)
        from lqcdweave.contraction.execution.device import DeviceExecution
        original_download = DeviceExecution.download

        def output_only_download(device, values):
            tracker.download_depth += 1
            try:
                return original_download(device, values)
            finally:
                tracker.download_depth -= 1

        with patch.object(DeviceExecution, 'download', output_only_download):
            gpu_outputs, report, closed = _run(
                gpu_complete, gpu_selected, backend, tracker)
        self.assertEqual(closed, ['a', 'b'])
        self.assertEqual(report['completed'], ('a', 'b'))
        self.assertEqual(tracker.stream_syncs.count('compute'), 1)
        self.assertEqual(tracker.stream_syncs.count('upload'), 1)
        self.assertGreater(tracker.input_events_recorded, 0)
        self.assertEqual(tracker.event_waits, tracker.input_events_recorded)
        self.assertGreater(tracker.release_events_recorded, 0)
        self.assertEqual(len(tracker.upload_event_dependencies),
                         tracker.release_events_recorded)
        self.assertTrue(all(dependency == ('upload', 'compute')
                            for dependency in tracker.upload_event_dependencies))
        self.assertEqual(tracker.events_recorded,
                         tracker.input_events_recorded + tracker.release_events_recorded)
        expected_d2h = sum(
            len({(gpu_complete.plan.templates[instance.template].plan.state.catalog[key].shape,
                  gpu_complete.plan.templates[instance.template].plan.state.catalog[key].dtype)
                 for key in gpu_complete.plan.templates[instance.template].plan.state.results})
            for instance in gpu_complete.plan.instances
        ) * 2
        self.assertEqual(tracker.asnumpy_calls, expected_d2h)
        self.assertEqual(report['device']['output_download_calls'], expected_d2h)
        for cfg in ('a', 'b'):
            _assert_tree_allclose(gpu_outputs[cfg], cpu_outputs[cfg])

    def test_background_upload_overlaps_compute_and_cfgs_remain_isolated(self):
        cpu_target, gpu_target = _targets()
        cpu_selected, cpu_complete = _complete(cpu_target)
        gpu_selected, gpu_complete = _complete(gpu_target)
        cpu_tracker = _Tracker()
        cpu_outputs, _, _ = _run(cpu_complete, cpu_selected, np, cpu_tracker)

        tracker = _Tracker()
        tracker.gate_background_b = True
        backend = _FakeCupy(tracker)
        from lqcdweave.contraction.execution.device import DeviceExecution
        from lqcdweave.contraction.execution import scene_runtime
        original_kernel = scene_runtime.contract_dense
        first_kernel = True

        def wait_for_background_upload(*args, **kwargs):
            nonlocal first_kernel
            if first_kernel:
                first_kernel = False
                if not tracker.background_b_read_entered.wait(10):
                    raise AssertionError('next cfg read was not started')
                tracker.kernel_started.set()
                tracker.release_background_b.set()
                if not tracker.upload_during_kernel.wait(10):
                    raise AssertionError('background upload did not overlap the kernel')
            return original_kernel(*args, **kwargs)

        original_download = DeviceExecution.download

        def output_only_download(device, values):
            tracker.download_depth += 1
            try:
                return original_download(device, values)
            finally:
                tracker.download_depth -= 1

        with (patch.object(DeviceExecution, 'download', output_only_download),
              patch.object(scene_runtime, 'contract_dense', wait_for_background_upload)):
            gpu_outputs, report, closed = _run(
                gpu_complete, gpu_selected, backend, tracker)
        self.assertTrue(tracker.upload_during_kernel.is_set())
        self.assertGreater(report['prefetched_reads'], 0)
        self.assertEqual(closed, ['a', 'b'])
        for cfg in ('a', 'b'):
            _assert_tree_allclose(gpu_outputs[cfg], cpu_outputs[cfg])

    def test_zero_device_prefetch_is_correct_and_below_peak_fails_before_factory(self):
        _, gpu_target = _targets()
        selected, complete = _complete(gpu_target)
        tracker = _Tracker()
        outputs, report, closed = _run(
            complete, selected, _FakeCupy(tracker), tracker,
            prefetch_device_bytes=0)
        self.assertEqual(report['prefetch_device_capacity_bytes'], 0)
        self.assertEqual(report['prefetched_reads'], 0)
        self.assertEqual(closed, ['a', 'b'])
        self.assertEqual(set(outputs), {'a', 'b'})

        called = []

        @contextmanager
        def forbidden_factory(cfg):
            called.append(cfg)
            raise AssertionError('source factory called before device admission')
            yield

        cap = complete.plan.peak_device_bytes - 1
        self.assertGreater(cap, 0)
        with (patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                    return_value=1 << 40),
              patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                    return_value=(1 << 30, 1 << 30))):
            with self.assertRaisesRegex(MemoryError, 'below the saved cfg requirement'):
                complete.run(backend=_FakeCupy(_Tracker()), device_memory_limit=cap)
        self.assertEqual(called, [])

    def test_background_upload_failure_joins_queue_closes_cfgs_and_restores_scope(self):
        _, gpu_target = _targets()
        selected, complete = _complete(gpu_target)
        tracker = _Tracker()
        tracker.gate_background_b = True
        tracker.fail_background_b_upload = True
        backend = _FakeCupy(tracker)
        closed = []
        factory = _factory(complete, selected, tracker, closed)
        from lqcdweave.contraction.execution.device import DeviceExecution
        from lqcdweave.contraction.execution import scene_runtime
        original_kernel = scene_runtime.contract_dense

        def release_background(*args, **kwargs):
            if not tracker.kernel_started.is_set():
                if not tracker.background_b_read_entered.wait(10):
                    raise AssertionError('next cfg read was not started')
                tracker.kernel_started.set()
                tracker.release_background_b.set()
            return original_kernel(*args, **kwargs)

        original_download = DeviceExecution.download

        def output_only_download(device, values):
            tracker.download_depth += 1
            try:
                return original_download(device, values)
            finally:
                tracker.download_depth -= 1

        from lqcdweave.contraction.expansion.saved_inputs import SavedPhysicalLoader
        close_prefetch_calls = []
        original_close_prefetch = SavedPhysicalLoader.close_prefetch

        def record_close_prefetch(loader, key, value):
            close_prefetch_calls.append((key, threading.current_thread().name))
            return original_close_prefetch(loader, key, value)

        with (patch.object(DeviceExecution, 'download', output_only_download),
              patch.object(scene_runtime, 'contract_dense', release_background),
              patch.object(SavedPhysicalLoader, 'close_prefetch', record_close_prefetch),
              patch('lqcdweave.adapters.hardware_budget._effective_local_host_available_bytes',
                    return_value=1 << 40),
              patch('lqcdweave.adapters.hardware_budget._observed_device_memory',
                    return_value=(1 << 30, 1 << 30))):
            calculation = replace(complete, options=dict(
                complete.options, configuration=iter(('a', 'b')),
                source_factory=factory, readers=None))
            with self.assertRaisesRegex(RuntimeError, 'background upload failed'):
                result = calculation.run(backend=backend)
                dict(result.value)
        self.assertEqual(closed, ['a', 'b'])
        self.assertTrue(close_prefetch_calls)
        self.assertIsNone(active_device_execution())
        self.assertTrue(tracker.pools[0].freed)
        self.assertFalse(any(t.name.startswith(('contraction-io', 'contraction-prepare'))
                             for t in threading.enumerate()))


@unittest.skipUnless(__import__('importlib').util.find_spec('cupy'),
                     'optional real CuPy device smoke test requires CuPy')
class RealCuPyOptionalTests(unittest.TestCase):
    def test_import_only_runtime_smoke(self):
        # This optional case only verifies that CuPy can initialize; it does
        # not run a workload, compare performance, or affect fake-backend tests.
        import cupy
        self.assertEqual(cupy.__name__, 'cupy')


if __name__ == '__main__':
    unittest.main()
