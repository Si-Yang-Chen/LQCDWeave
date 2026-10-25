"""Observable read-ahead behavior at the public scene-replay boundary."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from lqcdweave.contraction.execution.api import replay_scene_storage
from lqcdweave.contraction.execution.read_ahead import ReadAhead, _ReadTicket
from lqcdweave.contraction.performance.api import CostProfile
from lqcdweave.contraction.planning.optimizers.storage_schedule import replay_fixed_dag
from lqcdweave.adapters.scene_storage import NpyBlockBinding, NpySceneBlockLoader
from lqcdweave.adapters.elemental import ArrayElemental
from lqcdweave.adapters.propagator import ArrayPropagator
from lqcdweave.contracts.correlator.api import (
    AxisPattern, ContractionProblem, ContractionTerm, SourcePreparation, SourceSpec,
    parse_axis_pattern,
    StorageBlock, TensorSpec,
)
from lqcdweave.contraction.binding.api import Meson
from lqcdweave.contraction.execution.api import replay_storage_batches
from lqcdweave.contraction.planning.api import (
    ContractionState, join_storage_batches, logical_cost_profile,
    logical_source_problem, plan_scene_storage,
)
from lqcdweave.contracts.elemental import ElementalComponent
from lqcdweave.operators.correlator.api import Diagram, Propagator, QuarkDiagram, Tag
from lqcdweave.operators.correlator.mode_blocks import ModeBlockedDiagram
from lqcdweave.operators.single_hadron.operator import Operator, OperatorTerm
from lqcdweave.contraction.correlator import calc_diagram_eval
from correlator.planning.test_storage_schedule import _fixed_path, _physical, _problem


# These waits only prevent a broken test from hanging forever. Assertions use
# event handshakes, never elapsed-time thresholds.
_EVENT_TIMEOUT = 10


def _integer_einsum_pattern(test, arguments):
    """Read an X interleaved integer-axis call without rendering a string."""
    test.assertTrue(arguments)
    test.assertNotIsInstance(arguments[0], str)
    operand_count = (len(arguments) - 1) // 2
    test.assertEqual(len(arguments), 2 * operand_count + 1)
    return AxisPattern(
        tuple(tuple(arguments[2 * index + 1]) for index in range(operand_count)),
        tuple(arguments[-1]),
    ).normalized()


def _assert_integer_einsum(test, arguments, equation):
    actual = _integer_einsum_pattern(test, arguments)
    test.assertEqual(actual, parse_axis_pattern(equation).normalized())


def _two_dot_plan(*, memory_limit=1_000_000):
    problem, arrays, blocks, preparations = _problem()
    profile = CostProfile()
    state, _, _ = _fixed_path(problem, profile, (
        (0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None),
    ))
    physical = _physical(problem, blocks, preparations, cap=memory_limit)
    return replay_fixed_dag(state, physical), arrays


def _two_dot_cupy_plan():
    problem, arrays, blocks, preparations = _problem()
    problem = replace(problem, backend="cupy")
    blocks = {
        key: replace(block, device_bytes=block.read_bytes,
                     transfer_bytes=block.read_bytes,
                     staging_host_bytes=block.read_bytes)
        for key, block in blocks.items()
    }
    profile = CostProfile()
    state, _, _ = _fixed_path(problem, profile, (
        (0, ("A", "B")), (1, ("D", "E")), (0, None), (1, None),
    ))
    physical = _physical(problem, blocks, preparations, cap=1_000_000)
    return replay_fixed_dag(state, physical), arrays


def _pair_batch_plan(pair, *, memory_limit=1_000_000):
    block_a, block_b = f"{pair}:A", f"{pair}:B"
    shape = (4, 4)
    nbytes = int(np.prod(shape)) * np.dtype("float64").itemsize
    problem = logical_source_problem(ContractionProblem(
        {"A": TensorSpec(shape, "float64"), "B": TensorSpec(shape, "float64")},
        (ContractionTerm(("A", "B"), "ik,kj->ij"),),
        {"A": SourceSpec(block_a, 0), "B": SourceSpec(block_b, 0)},
    ))
    profile = CostProfile()
    model = logical_cost_profile(profile)
    state = ContractionState.initial(problem)
    state = state.load("A", model, 1_000_000).load("B", model, 1_000_000)
    state = state.contract(0, (0, 1), model, 1_000_000).release("A").release("B")
    blocks = {
        block_a: StorageBlock(f"dataset-{block_a}", "full", nbytes, host_bytes=nbytes),
        block_b: StorageBlock(f"dataset-{block_b}", "full", nbytes, host_bytes=nbytes),
    }
    preparations = {
        "A": SourcePreparation((block_a,)),
        "B": SourcePreparation((block_b,)),
    }
    plan = plan_scene_storage(
        state, blocks, preparations, profile=profile,
        memory_limit=memory_limit, host_memory_limit=memory_limit,
    )
    base = int(pair[-1]) * 100
    arrays = {
        block_a: (np.arange(nbytes // 8, dtype=np.float64).reshape(shape) + base),
        block_b: np.eye(shape[0], dtype=np.float64) * (int(pair[-1]) + 1),
    }
    return plan, arrays


def _expected_values(arrays):
    return (
        np.einsum("ik,kj,j->i", arrays["A"], arrays["B"], arrays["C"], optimize=False),
        np.einsum("ik,kj,j->i", arrays["D"], arrays["E"], arrays["F"], optimize=False),
    )


class _ProtocolLoader:
    """Small host loader whose future result is finalized on the caller thread."""

    def __init__(self, arrays, *, prefetchable=("D",), gate_first_dot=False,
                 read_error=None, close_prefetch_error=None, close_all_error=None):
        self.arrays = arrays
        self.prefetchable = set(prefetchable)
        self.gate_first_dot = gate_first_dot
        self.read_error = read_error
        self.close_prefetch_error = close_prefetch_error
        self.close_all_error = close_all_error
        self.read_counts = Counter()
        self.prefetch_counts = Counter()
        self.finish_counts = Counter()
        self.close_prefetch_counts = Counter()
        self.read_threads = []
        self.prefetch_threads = []
        self.finish_threads = []
        self.close_events = []
        self.max_active_workers = 0
        self._active_workers = 0
        self._lock = threading.Lock()

        self.prefetch_entered = threading.Event()
        self.kernel_started = threading.Event()
        self.read_active_during_kernel = threading.Event()
        self.kernel_finished = threading.Event()
        self.d_prefetch_finished = threading.Event()

    def read_block(self, key, block):
        self.read_counts[key] += 1
        self.read_threads.append(threading.current_thread())
        return self.arrays[key]

    def materialize(self, key, source, blocks):
        return blocks[source.region]

    def close_block(self, key, value):
        self.close_events.append(("close_block", key))

    def can_prefetch_block(self, key, block):
        return key in self.prefetchable

    def prefetch_host_bytes(self, key, block):
        return block.host_bytes + block.staging_host_bytes

    def prefetch_block(self, key, block):
        thread = threading.current_thread()
        with self._lock:
            self._active_workers += 1
            self.max_active_workers = max(self.max_active_workers, self._active_workers)
            self.prefetch_threads.append(thread)
            self.prefetch_counts[key] += 1
            self.read_counts[key] += 1
        try:
            if self.read_error is not None and key == "D":
                self.prefetch_entered.set()
                raise self.read_error
            if self.gate_first_dot and key == "D":
                self.prefetch_entered.set()
                if not self.kernel_started.wait(_EVENT_TIMEOUT):
                    raise AssertionError("first dot did not start while the read was pending")
                # Keep the worker inside the read callback until the actual
                # main-thread einsum finishes. This makes overlap deterministic.
                self.read_active_during_kernel.set()
                if not self.kernel_finished.wait(_EVENT_TIMEOUT):
                    raise AssertionError("first dot did not finish while the read was active")
            return (key, self.arrays[key])
        finally:
            with self._lock:
                self._active_workers -= 1
            if key == "D":
                self.d_prefetch_finished.set()

    def finish_prefetch(self, key, block, host_value):
        self.finish_counts[key] += 1
        self.finish_threads.append(threading.current_thread())
        if host_value[0] != key:
            raise AssertionError("host read token belongs to a different block")
        return host_value[1]

    def close_prefetch(self, key, host_value):
        self.close_prefetch_counts[key] += 1
        self.close_events.append(("close_prefetch", key))
        if self.close_prefetch_error is not None:
            raise self.close_prefetch_error

    def close_all(self):
        joined = all(not thread.is_alive() for thread in self.prefetch_threads)
        self.close_events.append(("close_all", joined))
        if self.close_all_error is not None:
            raise self.close_all_error


class _SynchronousLoader:
    """Legacy loader without the optional read-ahead capability."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.read_counts = Counter()
        self.read_threads = []
        self.closed = False

    def read_block(self, key, block):
        self.read_counts[key] += 1
        self.read_threads.append(threading.current_thread())
        return self.arrays[key]

    def materialize(self, key, source, blocks):
        return blocks[source.region]

    def close_block(self, key, value):
        pass

    def close_all(self):
        self.closed = True


class _FinishFailingLoader(_ProtocolLoader):
    def __init__(self, arrays, *, finish_error, **kwargs):
        super().__init__(arrays, **kwargs)
        self.finish_error = finish_error

    def finish_prefetch(self, key, block, host_value):
        self.finish_counts[key] += 1
        self.finish_threads.append(threading.current_thread())
        raise self.finish_error


class _TrackingNpyLoader(NpySceneBlockLoader):
    def __init__(self, bindings, materializers):
        super().__init__(bindings, materializers)
        self.read_counts = Counter()
        self.read_threads = []
        self.prefetch_threads = []
        self.finish_threads = []

    def read_block(self, key, block):
        self.read_counts[key] += 1
        self.read_threads.append(threading.current_thread())
        return super().read_block(key, block)

    def prefetch_block(self, key, block):
        self.read_counts[key] += 1
        self.prefetch_threads.append(threading.current_thread())
        return super().prefetch_block(key, block)

    def finish_prefetch(self, key, block, value):
        self.finish_threads.append(threading.current_thread())
        return super().finish_prefetch(key, block, value)


class _HostToken:
    def __init__(self, key, array):
        self.key = key
        self.array = array
        self.events = []
        self.released = False

    def release(self):
        self.events.append("release")
        self.array = None
        self.released = True


class _FakeCupyBackend:
    """NumPy-backed transfer/stream recorder; this does not emulate a GPU."""

    __name__ = "cupy"

    def __init__(self):
        self.call_threads = []
        self._pending_token = None
        self.cuda = SimpleNamespace(get_current_stream=lambda: self)

    def __getattr__(self, name):
        return getattr(np, name)

    def asarray(self, value):
        self.call_threads.append(("asarray", threading.current_thread()))
        if isinstance(value, _HostToken):
            self._pending_token = value
            value.events.append("asarray")
            value = value.array
        return np.asarray(value)

    def synchronize(self):
        self.call_threads.append(("synchronize", threading.current_thread()))
        if self._pending_token is not None:
            self._pending_token.events.append("synchronize")
            self._pending_token = None


class _FakeCupyLoader(_ProtocolLoader):
    def __init__(self, arrays, backend):
        super().__init__(arrays, prefetchable=arrays)
        self.backend = backend
        self.host_tokens = []

    def read_block(self, key, block):
        super().read_block(key, block)
        value = self.backend.asarray(self.arrays[key])
        self.backend.cuda.get_current_stream().synchronize()
        return value

    def prefetch_block(self, key, block):
        key_and_array = super().prefetch_block(key, block)
        token = _HostToken(key_and_array[0], key_and_array[1])
        self.host_tokens.append(token)
        return token

    def finish_prefetch(self, key, block, host_value):
        self.finish_counts[key] += 1
        self.finish_threads.append(threading.current_thread())
        if host_value.key != key:
            raise AssertionError("host read token belongs to a different block")
        value = self.backend.asarray(host_value)
        self.backend.cuda.get_current_stream().synchronize()
        host_value.release()
        return value

    def close_prefetch(self, key, host_value):
        self.close_prefetch_counts[key] += 1
        self.close_events.append(("close_prefetch", key))
        if not host_value.released:
            host_value.release()


class _TaskScopedLoader(_ProtocolLoader):
    prefetch_across_batches = True

    def __init__(self, arrays, *, gate_future=False):
        super().__init__(arrays, prefetchable=arrays)
        self.active_batch = None
        self.materialization_batches = []
        self.finish_batches = {}
        self.gate_future = gate_future
        self.future_batch_read_entered = threading.Event()
        self.future_batch_read_active = threading.Event()
        self.future_batch_read_finished = threading.Event()
        self.first_batch_kernel_started = threading.Event()
        self.first_batch_kernel_finished = threading.Event()

    @contextmanager
    def activate(self, batch):
        if self.active_batch is not None:
            raise AssertionError("batch activation cannot be nested")
        self.active_batch = batch
        try:
            yield
        finally:
            self.active_batch = None

    def read_block(self, key, block):
        self.read_counts[key] += 1
        self.read_threads.append(threading.current_thread())
        return self.arrays[key]

    def prefetch_block(self, key, block):
        if self.active_batch is None:
            self.future_batch_read_entered.set()
        value = super().prefetch_block(key, block)
        if key == "pair1:A":
            self.future_batch_read_entered.set()
            if self.gate_future:
                if not self.first_batch_kernel_started.wait(_EVENT_TIMEOUT):
                    raise AssertionError("second batch read did not start during the first batch")
                self.future_batch_read_active.set()
                if not self.first_batch_kernel_finished.wait(_EVENT_TIMEOUT):
                    raise AssertionError("second batch read did not remain active during the first kernel")
            self.future_batch_read_finished.set()
        return value

    def finish_prefetch(self, key, block, host_value):
        if self.active_batch is None:
            raise AssertionError("future block finalize requires its batch to be active")
        self.finish_batches[key] = self.active_batch
        return super().finish_prefetch(key, block, host_value)

    def materialize(self, key, source, blocks):
        self.materialization_batches.append((key, self.active_batch))
        return blocks[f"pair{self.active_batch}:{key}"]

    def close_all(self):
        joined = all(not thread.is_alive() for thread in self.prefetch_threads)
        self.close_events.append(("close_all", joined))


@contextmanager
def _patch_binary_kernel(operation):
    # Fixed CPU replay may use BLAS; keep the same gate/error injection at
    # either numerical entry, with the explicit 2-D equation for assertions.
    def matmul(left,right):
        return operation(left,[0,1],right,[1,2],[0,2])
    with patch.object(np,'einsum',side_effect=operation),patch.object(np,'matmul',side_effect=matmul):
        yield


class ReadAheadReplayTests(unittest.TestCase):
    def test_future_read_overlaps_the_first_dot_for_two_independent_tasks(self):
        plan, arrays = _two_dot_plan()
        loader = _ProtocolLoader(arrays, gate_first_dot=True)
        original_einsum = np.einsum
        first_dot_seen = False

        def controlled_einsum(*arguments, **kwargs):
            nonlocal first_dot_seen
            if not first_dot_seen:
                first_dot_seen = True
                _assert_integer_einsum(self, arguments, "ik,kj->ij")
                self.assertTrue(loader.prefetch_entered.wait(_EVENT_TIMEOUT))
                loader.kernel_started.set()
                self.assertTrue(loader.read_active_during_kernel.wait(_EVENT_TIMEOUT))
                result = original_einsum(*arguments, **kwargs)
                # The worker cannot finish its read before this calculation
                # ends, so both operations were live concurrently.
                self.assertFalse(loader.d_prefetch_finished.is_set())
                loader.kernel_finished.set()
                return result
            return original_einsum(*arguments, **kwargs)

        with _patch_binary_kernel(controlled_einsum):
            actual = replay_scene_storage(plan, loader)

        for result, expected in zip(actual, _expected_values(arrays), strict=True):
            np.testing.assert_allclose(result, expected)
        self.assertEqual(len(plan.state.problem.terms), 2)
        self.assertEqual(loader.prefetch_counts, Counter({"D": 1}))
        self.assertEqual(loader.finish_counts, Counter({"D": 1}))
        self.assertEqual(loader.read_counts, Counter({key: 1 for key in arrays}))
        self.assertEqual(loader.max_active_workers, 1)
        self.assertEqual(len({thread.ident for thread in loader.prefetch_threads}), 1)
        self.assertTrue(all(thread is threading.current_thread() for thread in loader.finish_threads))
        self.assertEqual(loader.close_events[-1], ("close_all", True))

    def test_multiple_queued_reads_stay_within_reserved_host_budget(self):
        roomy_plan, arrays = _two_dot_plan()
        # E and F together need 9,920 bytes. This leaves enough room for two
        # future reads after an earlier queued value is consumed, while still
        # forcing the queue to honor a finite extra-memory reservation.
        capacity = 9_920
        memory_limit = roomy_plan.peak_host_bytes + capacity
        plan, _ = _two_dot_plan(memory_limit=memory_limit)
        self.assertTrue(plan.feasible)
        loader = _ProtocolLoader(arrays, prefetchable=arrays)
        observations = []
        original_start_after = ReadAhead.start_after

        def record_queue(self, index):
            original_start_after(self, index)
            observations.append((len(self.pending), self.reserved_bytes, self.capacity))

        with patch.object(ReadAhead, "start_after", record_queue):
            actual = replay_scene_storage(plan, loader)

        for result, expected in zip(actual, _expected_values(arrays), strict=True):
            np.testing.assert_allclose(result, expected)
        self.assertGreater(max(pending for pending, _, _ in observations), 1)
        self.assertTrue(all(reserved <= allowed for _, reserved, allowed in observations))
        self.assertTrue(all(allowed <= memory_limit - plan.peak_host_bytes
                            for _, _, allowed in observations))
        self.assertEqual(loader.read_counts, Counter({key: 1 for key in arrays}))
        self.assertEqual(loader.max_active_workers, 1)

    def test_npy_scene_loader_prefetches_real_mmap_input_and_finalizes_on_main_thread(self):
        plan, arrays = _two_dot_plan()
        with tempfile.TemporaryDirectory() as directory:
            bindings = {}
            materializers = {}
            for key, array in arrays.items():
                path = Path(directory) / f"{key}.npy"
                np.save(path, array)
                bindings[key] = NpyBlockBinding(
                    path, tuple(slice(None) for _ in array.shape),
                    array.shape, array.dtype.str,
                )
                materializers[key] = lambda metadata, blocks, key=key: blocks[key]

            loader = _TrackingNpyLoader(bindings, materializers)
            actual = replay_scene_storage(plan, loader)

        for result, expected in zip(actual, _expected_values(arrays), strict=True):
            np.testing.assert_allclose(result, expected)
        self.assertEqual(loader.read_counts, Counter({key: 1 for key in arrays}))
        self.assertTrue(loader.prefetch_threads)
        self.assertTrue(all(thread is not threading.current_thread()
                            for thread in loader.prefetch_threads))
        self.assertTrue(all(thread is threading.current_thread()
                            for thread in loader.finish_threads))

    def test_fake_cupy_keeps_prefetch_host_only_and_finalizes_transfer_on_main_thread(self):
        plan, arrays = _two_dot_cupy_plan()
        backend = _FakeCupyBackend()
        loader = _FakeCupyLoader(arrays, backend)

        actual = replay_scene_storage(plan, loader, backend=backend)

        for result, expected in zip(actual, _expected_values(arrays), strict=True):
            np.testing.assert_allclose(result, expected)
        self.assertTrue(loader.prefetch_counts)
        self.assertTrue(loader.finish_threads)
        self.assertTrue(all(thread is threading.current_thread()
                            for thread in loader.finish_threads))
        self.assertTrue(all(thread is threading.current_thread()
                            for _, thread in backend.call_threads))
        self.assertTrue(all(token.released
                            and token.events == ["asarray", "synchronize", "release"]
                            for token in loader.host_tokens))
        worker_ids = {thread.ident for thread in loader.prefetch_threads}
        self.assertFalse(any(thread.ident in worker_ids
                            for _, thread in backend.call_threads))
        self.assertEqual(loader.max_active_workers, 1)

    def test_task_read_ahead_overlaps_pair0_kernel_and_finishes_pair1_when_activated(self):
        self.check_pair_overlap(streaming=False)

    def test_streamed_window_overlaps_pair0_kernel_and_finishes_pair1_when_activated(self):
        self.check_pair_overlap(streaming=True)

    def check_pair_overlap(self, *, streaming):
        plan0, arrays0 = _pair_batch_plan("pair0")
        plan1, arrays1 = _pair_batch_plan("pair1")
        self.assertEqual(len(plan0.state.problem.terms), 1)
        self.assertEqual(len(plan1.state.problem.terms), 1)
        joined = join_storage_batches((plan0, plan1))
        arrays = arrays0 | arrays1
        loader = _TaskScopedLoader(arrays, gate_future=True)
        active = {id(joined[0]): 0, id(joined[1]): 1}
        outputs = []
        original_einsum = np.einsum
        first_kernel_seen = False

        def controlled_einsum(*arguments, **kwargs):
            nonlocal first_kernel_seen
            if not first_kernel_seen:
                first_kernel_seen = True
                _assert_integer_einsum(self, arguments, "ik,kj->ij")
                self.assertTrue(loader.future_batch_read_entered.wait(_EVENT_TIMEOUT))
                loader.first_batch_kernel_started.set()
                self.assertTrue(loader.future_batch_read_active.wait(_EVENT_TIMEOUT))
                result = original_einsum(*arguments, **kwargs)
                self.assertFalse(loader.future_batch_read_finished.is_set())
                loader.first_batch_kernel_finished.set()
                return result
            return original_einsum(*arguments, **kwargs)

        with _patch_binary_kernel(controlled_einsum):
            replay_storage_batches(
                joined, loader,
                lambda index, report: outputs.append((index, np.array(report[0], copy=True))),
                prejoined=True,
                streaming=streaming,
                batch_context=lambda plan: loader.activate(active[id(plan)]),
            )

        expected0 = arrays["pair0:A"] @ arrays["pair0:B"]
        expected1 = arrays["pair1:A"] @ arrays["pair1:B"]
        self.assertEqual([index for index, _ in outputs], [0, 1])
        np.testing.assert_allclose(outputs[0][1], expected0)
        np.testing.assert_allclose(outputs[1][1], expected1)
        self.assertEqual(loader.read_counts, Counter({key: 1 for key in arrays}))
        self.assertEqual(loader.finish_batches["pair1:A"], 1)
        self.assertIn(("A", 1), loader.materialization_batches)
        self.assertEqual(loader.max_active_workers, 1)
        self.assertEqual(loader.close_events[-1], ("close_all", True))

    def test_streamed_plan_and_source_iterators_are_bounded_by_window(self):
        # Six plans ensure advancing past several window boundaries.
        plans_and_arrays = [_pair_batch_plan(f'pair{i}') for i in range(6)]
        joined = join_storage_batches(tuple(p for p, _ in plans_and_arrays))
        loader = _TaskScopedLoader({k:v for _, arrays in plans_and_arrays for k,v in arrays.items()})
        outputs, planned, sources = [], [], []
        active = {id(p):i for i,p in enumerate(joined)}
        def plan_stream():
            for i,p in enumerate(joined):
                self.assertEqual(len(outputs), i - i % 2)
                planned.append(i)
                yield p
        def source_stream():
            for i,p in enumerate(joined):
                self.assertEqual(len(outputs), i - i % 2)
                sources.append(i)
                yield p.state.problem.sources
        replay_storage_batches(plan_stream(), loader,
            lambda i,r:outputs.append(np.array(r[0],copy=True)),
            prejoined=True, streaming=True, source_descriptions=source_stream(),
            batch_context=lambda p:loader.activate(active[id(p)]))
        self.assertEqual(planned,list(range(6)))
        self.assertEqual(sources,list(range(6)))
        for result,(_,arrays) in zip(outputs,plans_and_arrays,strict=True):
            a,b = arrays.values()
            np.testing.assert_allclose(result,a@b)
        self.assertEqual(loader.read_counts,Counter({k:1 for k in loader.arrays}))
        self.assertEqual(loader.close_events[-1],('close_all',True))

    def test_streamed_consumer_failure_stops_future_windows_and_joins_worker(self):
        plans_and_arrays = [_pair_batch_plan(f'pair{i}') for i in range(4)]
        joined = join_storage_batches(tuple(p for p,_ in plans_and_arrays))
        loader = _TaskScopedLoader({k:v for _,arrays in plans_and_arrays for k,v in arrays.items()})
        planned=[]
        def stream():
            for p in joined:
                planned.append(p)
                yield p
        active={id(p):i for i,p in enumerate(joined)}
        def failed_writer(*args):
            raise RuntimeError('writer failed')
        with self.assertRaisesRegex(RuntimeError,'writer failed'):
            replay_storage_batches(stream(),loader,failed_writer,
                prejoined=True,streaming=True,batch_context=lambda p:loader.activate(active[id(p)]))
        self.assertEqual(len(planned),2)
        self.assertFalse(any(key.startswith('pair2:') for key in loader.read_counts))
        self.assertEqual(loader.close_events[-1],('close_all',True))

    def test_streamed_invalid_future_plan_is_rejected_before_window_reads(self):
        plan0,arrays0=_pair_batch_plan('pair0')
        plan1,arrays1=_pair_batch_plan('pair1')
        joined=join_storage_batches((plan0,plan1))
        loader=_TaskScopedLoader(arrays0|arrays1)
        invalid=replace(joined[1],feasible=False)
        with self.assertRaises(MemoryError):
            replay_storage_batches(iter((joined[0],invalid)),loader,lambda *args:None,
                prejoined=True,streaming=True)
        self.assertFalse(loader.read_counts)
        self.assertEqual(loader.close_events[-1],('close_all',True))

    def test_streamed_windows_release_completed_plan_descriptions(self):
        import gc
        import weakref
        arrays={k:v for i in range(6) for k,v in _pair_batch_plan(f'pair{i}')[1].items()}
        loader=_TaskScopedLoader(arrays)
        refs=[]
        def stream():
            for i in range(6):
                if i and i%2==0:
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in refs))
                p,_=_pair_batch_plan(f'pair{i}')
                refs.append(weakref.ref(p))
                yield p
                del p
        def activate(p):
            key=next(iter(p.blocks))
            return loader.activate(int(key.split(':')[0][4:]))
        replay_storage_batches(stream(),loader,lambda *args:None,
            prejoined=True,streaming=True,batch_context=activate)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_streamed_no_headroom_runs_synchronously_with_correct_values(self):
        plans_and_arrays=[_pair_batch_plan(f'pair{i}') for i in range(2)]
        joined=join_storage_batches(tuple(p for p,_ in plans_and_arrays))
        cap=max(p.peak_host_bytes for p in joined)
        joined=tuple(replace(p,memory_limit=cap,host_memory_limit=cap) for p in joined)
        loader=_TaskScopedLoader({k:v for _,arrays in plans_and_arrays for k,v in arrays.items()})
        active={id(p):i for i,p in enumerate(joined)}
        outputs=[]
        replay_storage_batches(iter(joined),loader,lambda i,r:outputs.append(np.array(r[0],copy=True)),
            prejoined=True,streaming=True,batch_context=lambda p:loader.activate(active[id(p)]))
        self.assertFalse(loader.prefetch_counts)
        for actual,(_,arrays) in zip(outputs,plans_and_arrays,strict=True):
            a,b=arrays.values()
            np.testing.assert_allclose(actual,a@b)

    def test_streamed_runtime_source_count_mismatch_is_explicit(self):
        p0,a0=_pair_batch_plan('pair0')
        p1,a1=_pair_batch_plan('pair1')
        joined=join_storage_batches((p0,p1))
        loader=_TaskScopedLoader(a0|a1)
        with self.assertRaisesRegex(ValueError,'one runtime source description'):
            replay_storage_batches(iter(joined),loader,lambda *args:None,prejoined=True,streaming=True,
                source_descriptions=iter((p0.state.problem.sources,)))
        self.assertFalse(loader.read_counts)
        loader=_TaskScopedLoader(a0)
        values=[]
        with self.assertRaisesRegex(ValueError,'more runtime source descriptions'):
            replay_storage_batches(iter((p0,)),loader,lambda i,r:values.append(r[0]),
                prejoined=True,streaming=True,batch_context=lambda p:loader.activate(0),
                source_descriptions=iter((p0.state.problem.sources,p1.state.problem.sources)))
        self.assertEqual(len(values),1)
        self.assertEqual(loader.close_events[-1],('close_all',True))

    def test_streamed_lookahead_requires_positive_integer_before_reading_iterator(self):
        def stream():
            raise AssertionError('must validate before consuming plans')
            yield
        for width in (0,-1,True,1.5):
            with self.subTest(width=width),self.assertRaisesRegex(ValueError,'positive integer'):
                replay_storage_batches(stream(),_TaskScopedLoader({}),lambda *args:None,
                    prejoined=True,streaming=True,lookahead=width)

    def test_streamed_disabled_measurement_and_missing_capability_pull_one_plan(self):
        p0,a0=_pair_batch_plan('pair0')
        p1,a1=_pair_batch_plan('pair1')
        joined=join_storage_batches((p0,p1))
        for name,options in (('disabled',{'io_overlap':False}),('measurement',{'measure':True}),
                             ('missing',{})):
            with self.subTest(mode=name):
                loader=_TaskScopedLoader(a0|a1)
                if name=='missing':
                    loader.finish_prefetch=None
                consumed=[]
                def stream():
                    for i,p in enumerate(joined):
                        self.assertEqual(len(consumed),i)
                        yield p
                active={id(p):i for i,p in enumerate(joined)}
                replay_storage_batches(stream(),loader,lambda i,r:consumed.append(i),
                    prejoined=True,streaming=True,batch_context=lambda p:loader.activate(active[id(p)]),
                    **options)
                self.assertEqual(consumed,[0,1])
                self.assertFalse(loader.prefetch_counts)

    def test_streamed_writer_error_preserved_when_future_cleanup_fails(self):
        p0,a0=_pair_batch_plan('pair0')
        p1,a1=_pair_batch_plan('pair1')
        joined=join_storage_batches((p0,p1))
        loader=_TaskScopedLoader(a0|a1)
        loader.close_prefetch_error=RuntimeError('future cleanup failed')
        primary=RuntimeError('writer failed')
        def fail(*args):
            self.assertTrue(loader.future_batch_read_finished.wait(_EVENT_TIMEOUT))
            raise primary
        with self.assertRaisesRegex(RuntimeError,'writer failed') as caught:
            replay_storage_batches(iter(joined),loader,fail,prejoined=True,streaming=True,
                batch_context=lambda p:loader.activate(0))
        self.assertIs(caught.exception,primary)
        self.assertTrue(any('future cleanup failed' in note for note in getattr(primary,'__notes__',())))
        self.assertEqual(loader.close_events[-1],('close_all',True))

    def test_streamed_raw_lifetime_spans_window_boundary_without_reread(self):
        p,arrays=_pair_batch_plan('pair0')
        joined=join_storage_batches((p,p,p))
        loader=_TaskScopedLoader(arrays)
        values=[]
        replay_storage_batches(iter(joined),loader,lambda i,r:values.append(np.array(r[0],copy=True)),
            prejoined=True,streaming=True,batch_context=lambda p:loader.activate(0))
        self.assertEqual(len(values),3)
        for value in values:
            np.testing.assert_allclose(value,arrays['pair0:A']@arrays['pair0:B'])
        self.assertEqual(loader.read_counts,Counter({key:1 for key in arrays}))
        self.assertEqual(Counter(key for kind,key in loader.close_events if kind=='close_block'),
                         Counter({key:1 for key in arrays}))

    def test_native_frontend_prefetches_second_pair_source_during_first_pair_compute(self):
        rng = np.random.default_rng(261005)
        component = ElementalComponent(0)

        class TrackedPropagator(ArrayPropagator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.gate_future_source = False
                self.future_source_entered = threading.Event()
                self.first_kernel_started = threading.Event()
                self.future_source_active = threading.Event()
                self.future_source_finished = threading.Event()
                self.first_kernel_finished = threading.Event()

            def read_slab_in_background(self, sink_times, source_time, sector="vsv", **kwargs):
                if self.gate_future_source and source_time == 2:
                    self.future_source_entered.set()
                    if not self.first_kernel_started.wait(_EVENT_TIMEOUT):
                        raise AssertionError("second pair source read did not overlap the first pair")
                    self.future_source_active.set()
                    value = super().read_slab_in_background(
                        sink_times, source_time, sector, **kwargs,
                    )
                    self.future_source_finished.set()
                    if not self.first_kernel_finished.wait(_EVENT_TIMEOUT):
                        raise AssertionError("second pair source read outlived the first kernel")
                    return value
                return super().read_slab_in_background(
                    sink_times, source_time, sector, **kwargs,
                )

        elemental = ArrayElemental(
            {"v2v": rng.normal(size=(1, 1, 3, 1, 1))
                     + 1j * rng.normal(size=(1, 1, 3, 1, 1))},
            (component,), phase_convention="anchor", data_identity="io-overlap/native-el",
        )
        propagator = TrackedPropagator(
            {"vsv": rng.normal(size=(3, 3, 4, 4, 1, 1))
                     + 1j * rng.normal(size=(3, 3, 4, 4, 1, 1))},
            time_convention="absolute", data_identity="io-overlap/native-peram",
        )
        quark = Propagator("u", Tag(0, 0), Tag(1, 1))
        graph = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["a", "b"], [None, quark],
        )
        operator = Operator.from_terms("native-overlap", [
            OperatorTerm(0, component, 0, 1.0 + 0.25j),
        ])
        vertices = {
            name: Meson(elemental, operator, phase_convention="anchor")
            for name in ("a", "b")
        }
        pairs = ({"sink": 1, "source": 0}, {"sink": 2, "source": 1})
        expected = np.asarray([
            calc_diagram_eval(
                graph, propagator_map={quark: propagator}, vertex_map=vertices,
                usedNe=1, time_axes=pair, optimize="none",
            ).value
            for pair in pairs
        ])

        calculation = calc_diagram_eval(
            graph, propagator_map={quark: propagator}, vertex_map=vertices,
            usedNe=1, time_pairs=pairs, batch={"axis": "pair", "size": 1},
            optimize="none", execute=False,
        )
        self.assertEqual(len(calculation.inputs.task_batches), 2)
        loader = calculation.inputs.loader
        source2_block = next(
            key for key, identity in loader.slab_identity_by_block.items()
            if identity[0] == "propagator" and identity[2] == 2
        )
        finish_events = []
        materialize_events = []
        original_finish = loader.finish_prefetch
        original_materialize = loader.materialize

        def tracked_finish(key, block, value):
            if key == source2_block:
                finish_events.append((loader.active_batch, threading.current_thread()))
            return original_finish(key, block, value)

        def tracked_materialize(key, source, blocks):
            if source2_block in blocks:
                materialize_events.append((loader.active_batch, threading.current_thread()))
            return original_materialize(key, source, blocks)

        loader.finish_prefetch = tracked_finish
        loader.materialize = tracked_materialize
        propagator.gate_future_source = True
        original_einsum = np.einsum
        first_kernel_seen = False

        def controlled_einsum(*arguments, **kwargs):
            nonlocal first_kernel_seen
            # Meson factor setup also uses einsum. Gate on the first actual
            # contraction kernel after those main-thread materializations.
            is_factor_setup = (isinstance(arguments[0], str)
                               and arguments[0] == "ab,...->ab...")
            if not first_kernel_seen and not is_factor_setup:
                first_kernel_seen = True
                _integer_einsum_pattern(self, arguments)
                self.assertTrue(propagator.future_source_entered.wait(_EVENT_TIMEOUT))
                propagator.first_kernel_started.set()
                self.assertTrue(propagator.future_source_active.wait(_EVENT_TIMEOUT))
                result = original_einsum(*arguments, **kwargs)
                self.assertTrue(propagator.future_source_finished.is_set())
                propagator.first_kernel_finished.set()
                return result
            return original_einsum(*arguments, **kwargs)

        with _patch_binary_kernel(controlled_einsum):
            actual = calculation.run().value

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(finish_events), 1)
        self.assertEqual(finish_events[0][0], 1)
        self.assertIs(finish_events[0][1], threading.current_thread())
        # Native dense lanes are prepared by descriptive recipes, not the opaque
        # loader callback; the prefetched slab feeds a lane recipe instead.
        self.assertTrue(any(
            prep.blocks == (source2_block,) and prep.recipe.raw_index is not None
            for prep in loader.preparations.values()))
        self.assertTrue(all(batch == 1 and thread is threading.current_thread()
                            for batch, thread in materialize_events))

    def test_fixed_native_mode_gathers_declare_workspace_and_keep_budget_until_finish(self):
        rng = np.random.default_rng(261006)
        component = ElementalComponent(0)

        class WorkspacePropagator(ArrayPropagator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.workspace_requests = []
                self.background_calls = []
                self.worker_reservations = []
                self.fail_block = False
                self.test_loader = None
                self.target_source_time = None
                self.target_take_waiting = threading.Event()

            def read_slab_workspace_bytes(self, metadata, *, mode_indices=None, **kwargs):
                self.workspace_requests.append((metadata, mode_indices))
                if (not isinstance(mode_indices, dict)
                        or set(mode_indices) != {"sink_mode", "source_mode"}):
                    raise AssertionError("both propagator mode axes must be selected")
                return 8_192

            def read_slab_in_background(self, sink_times, source_time, sector="vsv", **kwargs):
                selection = dict(kwargs.get("mode_indices") or {})
                self.background_calls.append((source_time, selection, threading.current_thread()))
                queue = self.test_loader._test_read_ahead
                if source_time == self.target_source_time:
                    if not self.target_take_waiting.wait(_EVENT_TIMEOUT):
                        raise AssertionError("storage replay did not wait for the selected slab")
                self.worker_reservations.append(queue.reserved_bytes)
                return super().read_slab_in_background(
                    sink_times, source_time, sector, **kwargs,
                )

            def block(self, *args, **kwargs):
                if self.fail_block and args[1] == self.target_source_time:
                    raise AssertionError("fixed background reads must use the selected slab reader")
                return super().block(*args, **kwargs)

        elemental = ArrayElemental(
            {"v2v": rng.normal(size=(1, 1, 3, 4, 4))
                     + 1j * rng.normal(size=(1, 1, 3, 4, 4))},
            (component,), phase_convention="anchor", data_identity="io-overlap/fixed-el",
        )
        propagator = WorkspacePropagator(
            {"vsv": rng.normal(size=(3, 3, 4, 4, 4, 4))
                     + 1j * rng.normal(size=(3, 3, 4, 4, 4, 4))},
            time_convention="absolute", data_identity="io-overlap/fixed-peram",
        )
        quark = Propagator("u", Tag(0, 0), Tag(1, 1))
        base = Diagram(
            QuarkDiagram([[0, 1], [1, 0]]), ["sink", "source"],
            ["a", "b"], [None, quark],
        )
        selections = (((2, 0), (3, 1)), ((1, 3), (0, 2)))
        graph = ModeBlockedDiagram(base, selections, mode_extent=4)
        operator = Operator.from_terms("fixed-mode-overlap", [
            OperatorTerm(0, component, 0, 1.0 - 0.125j),
        ])
        vertices = {
            name: Meson(elemental, operator, phase_convention="anchor")
            for name in ("a", "b")
        }
        calculation = calc_diagram_eval(
            graph, propagator_map={quark: propagator}, vertex_map=vertices,
            usedNe=4, time_axes={"sink": 1, "source": 0}, optimize="anytime",
            execute=False, memory_limit=10_000_000, host_memory_limit=10_000_000,
        )
        self.assertFalse(calculation.inputs.task_batches)
        loader = calculation.inputs.loader
        propagator.test_loader = loader
        expected = calculation.run(measure=True).value

        prop_keys = {
            key for key, block in calculation.inputs.blocks.items()
            if block.dataset.startswith("propagator:")
        }
        self.assertEqual(len(prop_keys), 2)
        # block:0 is needed at the first action and therefore stays synchronous;
        # block:1 is read after the first contraction and can overlap it.
        target_key = sorted(prop_keys)[1]
        target_slab = next(iter(loader.background_slabs()[target_key].values()))[0]
        propagator.target_source_time = target_slab.identity[2]
        finish_reservations = {}
        original_finish = loader.finish_prefetch
        original_start_after = ReadAhead.start_after
        original_ticket_result = _ReadTicket.result

        def attach_queue(queue, index):
            queue.loader._test_read_ahead = queue
            original_start_after(queue, index)
            if target_key in queue.pending:
                queue.loader._test_target_future = queue.pending[target_key]

        def observe_target_wait(future, *args, **kwargs):
            if future is loader._test_target_future:
                propagator.target_take_waiting.set()
            return original_ticket_result(future, *args, **kwargs)

        def tracked_finish(key, block, value):
            if key in prop_keys:
                queue = loader._test_read_ahead
                finish_reservations[key] = queue.reserved_bytes
            return original_finish(key, block, value)

        loader.finish_prefetch = tracked_finish
        propagator.fail_block = True
        with patch.object(ReadAhead, "start_after", attach_queue), \
                patch.object(_ReadTicket, "result", observe_target_wait):
            actual = calculation.run().value

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        expected_mode_selections = {((3, 1), (1, 3))}
        self.assertEqual(
            {(tuple(selection["sink_mode"]), tuple(selection["source_mode"]))
             for _, selection, _ in propagator.background_calls},
            expected_mode_selections,
        )
        self.assertTrue(all(thread is not threading.current_thread()
                            for _, _, thread in propagator.background_calls))
        self.assertTrue(all(value >= 8_192 for value in propagator.worker_reservations))
        self.assertTrue(propagator.target_take_waiting.is_set())
        for metadata, selection in propagator.workspace_requests:
            self.assertEqual(metadata.axes[0], "coordinate_batch")
            self.assertEqual(metadata.shape, (1, 4, 4, 2, 2))
            self.assertEqual(set(selection), {"sink_mode", "source_mode"})
        block = calculation.inputs.blocks[target_key]
        self.assertGreaterEqual(
            finish_reservations[target_key], 2 * block.host_bytes + 8_192,
        )

    def test_task_read_ahead_falls_back_when_any_batch_has_no_spare_host_budget(self):
        roomy, _ = _pair_batch_plan("pair0")
        tight_cap = roomy.peak_host_bytes
        plan0, arrays0 = _pair_batch_plan("pair0", memory_limit=tight_cap)
        plan1, arrays1 = _pair_batch_plan("pair1", memory_limit=tight_cap)
        self.assertEqual(plan0.peak_host_bytes, tight_cap)
        self.assertEqual(plan1.peak_host_bytes, tight_cap)
        joined = join_storage_batches((plan0, plan1))
        arrays = arrays0 | arrays1
        loader = _TaskScopedLoader(arrays)
        active = {id(joined[0]): 0, id(joined[1]): 1}
        outputs = []

        replay_storage_batches(
            joined, loader,
            lambda index, report: outputs.append((index, np.array(report[0], copy=True))),
            prejoined=True,
            batch_context=lambda plan: loader.activate(active[id(plan)]),
        )

        self.assertEqual([index for index, _ in outputs], [0, 1])
        self.assertFalse(loader.prefetch_counts)
        self.assertTrue(all(thread is threading.current_thread()
                            for thread in loader.read_threads))
        self.assertEqual(loader.read_counts, Counter({key: 1 for key in arrays}))
        self.assertEqual(loader.close_events[-1], ("close_all", True))

    def test_first_batch_kernel_failure_joins_and_closes_unconsumed_second_batch_future(self):
        plan0, arrays0 = _pair_batch_plan("pair0")
        plan1, arrays1 = _pair_batch_plan("pair1")
        joined = join_storage_batches((plan0, plan1))
        arrays = arrays0 | arrays1
        loader = _TaskScopedLoader(arrays)
        active = {id(joined[0]): 0, id(joined[1]): 1}
        primary = RuntimeError("pair0 kernel failed")

        def fail_first_kernel(*arguments, **kwargs):
            _assert_integer_einsum(self, arguments, "ik,kj->ij")
            self.assertTrue(loader.future_batch_read_entered.wait(_EVENT_TIMEOUT))
            self.assertTrue(loader.future_batch_read_finished.wait(_EVENT_TIMEOUT))
            raise primary

        with _patch_binary_kernel(fail_first_kernel):
            with self.assertRaisesRegex(RuntimeError, "pair0 kernel failed") as caught:
                replay_storage_batches(
                    joined, loader, lambda *args: None, prejoined=True,
                    batch_context=lambda plan: loader.activate(active[id(plan)]),
                )

        self.assertIs(caught.exception, primary)
        self.assertGreaterEqual(loader.close_prefetch_counts["pair1:A"], 1)
        self.assertEqual(loader.read_counts["pair1:A"], 1)
        self.assertEqual(loader.close_events[-1], ("close_all", True))

    def test_disabled_measurement_missing_capability_and_tight_budget_are_synchronous(self):
        roomy_plan, arrays = _two_dot_plan()
        tight_plan, _ = _two_dot_plan(memory_limit=roomy_plan.peak_host_bytes)
        self.assertTrue(tight_plan.feasible)

        cases = (
            ("disabled", roomy_plan, _ProtocolLoader(arrays), {"io_overlap": False}),
            ("measurement", roomy_plan, _ProtocolLoader(arrays), {"measure": True}),
            ("tight", tight_plan, _ProtocolLoader(arrays), {}),
            ("legacy", roomy_plan, _SynchronousLoader(arrays), {}),
        )
        for name, plan, loader, kwargs in cases:
            with self.subTest(name=name):
                result = replay_scene_storage(plan, loader, **kwargs)
                values = result.values if hasattr(result, "values") else result
                for actual, expected in zip(values, _expected_values(arrays), strict=True):
                    np.testing.assert_allclose(actual, expected)
                if isinstance(loader, _ProtocolLoader):
                    self.assertFalse(loader.prefetch_counts)
                    self.assertTrue(all(thread is threading.current_thread()
                                        for thread in loader.read_threads))
                else:
                    self.assertTrue(loader.closed)

    def test_background_read_error_is_propagated_after_worker_join(self):
        plan, arrays = _two_dot_plan()
        primary = RuntimeError("prefetch read failed")
        loader = _ProtocolLoader(arrays, read_error=primary)

        with self.assertRaisesRegex(RuntimeError, "prefetch read failed") as caught:
            replay_scene_storage(plan, loader)

        self.assertIs(caught.exception, primary)
        self.assertEqual(loader.close_events[-1], ("close_all", True))
        self.assertEqual(loader.close_prefetch_counts, Counter())

    def test_kernel_error_closes_unconsumed_host_result_after_worker_join(self):
        plan, arrays = _two_dot_plan()
        primary = RuntimeError("main kernel failed")
        cleanup = RuntimeError("closing prefetched host data failed")
        loader = _ProtocolLoader(arrays, close_prefetch_error=cleanup)

        def fail_first_kernel(*arguments, **kwargs):
            _assert_integer_einsum(self, arguments, "ik,kj->ij")
            self.assertTrue(loader.d_prefetch_finished.wait(_EVENT_TIMEOUT))
            raise primary

        with _patch_binary_kernel(fail_first_kernel):
            with self.assertRaisesRegex(RuntimeError, "main kernel failed") as caught:
                replay_scene_storage(plan, loader)

        self.assertIs(caught.exception, primary)
        self.assertEqual(loader.close_prefetch_counts, Counter({"D": 1}))
        self.assertEqual(loader.close_events[-1], ("close_all", True))

    def test_finish_and_cleanup_errors_preserve_primary_on_python310_style_exception(self):
        plan, arrays = _two_dot_plan()

        class Python310StyleError(RuntimeError):
            add_note = None

        primary = Python310StyleError("finalizing prefetched data failed")
        cleanup = RuntimeError("closing prefetched data failed")
        loader = _FinishFailingLoader(
            arrays, finish_error=primary, close_prefetch_error=cleanup,
        )

        with self.assertRaisesRegex(Python310StyleError, "finalizing prefetched data failed") as caught:
            replay_scene_storage(plan, loader)

        self.assertIs(caught.exception, primary)
        self.assertEqual(loader.finish_counts, Counter({"D": 1}))
        self.assertEqual(loader.close_prefetch_counts, Counter({"D": 1}))
        self.assertEqual(loader.close_events[-1], ("close_all", True))


if __name__ == "__main__":
    unittest.main()
