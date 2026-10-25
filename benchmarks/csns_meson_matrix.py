"""Unified-route meson matrix storage benchmark for CSNS.

The synthetic fixture has 224 complete terms. The current unified planner
selects its operation DAG, after which the benchmark compares storage schedules
without changing that computation. The optional cfg2000 workload imports the
user's meson55 workflow and only executes the requested six-operator,
unpartitioned t40 task.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Mapping

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SRC = REPOSITORY / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lqcdweave.contracts.correlator.api import (  # noqa: E402
    ContractionProblem,
    ContractionTerm,
    SourcePreparation,
    SourceSpec,
    StorageBlock,
    TensorDescription,
    TensorSpec,
)
from lqcdweave.contracts.execution import TermValue  # noqa: E402
from lqcdweave.contraction.execution.api import replay_storage_batches  # noqa: E402
from lqcdweave.contraction.performance.api import CostProfile  # noqa: E402
from lqcdweave.contraction.planning.api import join_storage_batches  # noqa: E402
from lqcdweave.contraction.planning.canonical_operations import (  # noqa: E402
    describe,
    replay,
)
from lqcdweave.contraction.planning.optimizers.macro_space import (  # noqa: E402
    MacroSearchSpace,
)
from lqcdweave.contraction.planning.optimizers.session import (  # noqa: E402
    solve_storage,
)
from lqcdweave.contraction.planning.optimizers.storage_schedule import (  # noqa: E402
    _Dag,
    _Frontier,
    _plan,
)
from lqcdweave.contraction.planning.state import ContractionState  # noqa: E402
from lqcdweave.contraction.results.api import assemble_correlator  # noqa: E402


_OPERATORS = 14
_EXPECTED_TERMS = 2 * _OPERATORS + _OPERATORS * _OPERATORS
_HOST_LIMIT = 64 * 2**30
_DEVICE_LIMIT = 12 * 2**30


@dataclass(frozen=True)
class WorkBatch:
    problem: object
    blocks: Mapping
    preparations: Mapping
    arrays: Mapping
    task: object = None
    selected_path: object = None


class TraceLoader:
    """Observe the existing loader protocol without changing its behavior."""

    def __init__(self, loader):
        self.loader = loader
        self.events = []
        self.batch_windows = []
        self._active_batch_index = None
        self._consumer_batch_by_block = {}
        self._lock = threading.Lock()

    def set_batch_consumers(self, values):
        self._consumer_batch_by_block = dict(values)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def _timed(self, name, callback, *args):
        start = time.perf_counter_ns()
        try:
            return callback(*args)
        finally:
            event = {
                "event": name,
                "thread": threading.current_thread().name,
                "thread_id": threading.get_ident(),
                "start_ns": start,
                "end_ns": time.perf_counter_ns(),
                "key": str(args[0]) if args else None,
                "active_batch_index": self._active_batch_index,
                "consumer_batch_index": self._consumer_batch_by_block.get(args[0]) if args else None,
            }
            event["seconds"] = (event["end_ns"] - start) / 1e9
            event["read_ahead_for_future_batch"] = (
                event["consumer_batch_index"] is not None
                and event["active_batch_index"] is not None
                and event["consumer_batch_index"] > event["active_batch_index"]
            )
            with self._lock:
                self.events.append(event)

    def read_block(self, key, block):
        return self._timed("read_block", self.loader.read_block, key, block)

    def materialize(self, key, source, raw_blocks):
        return self._timed("materialize", self.loader.materialize, key, source, raw_blocks)

    def can_prefetch_block(self, key, block):
        callback = getattr(self.loader, "can_prefetch_block", None)
        return bool(callback(key, block)) if callable(callback) else False

    def prefetch_host_bytes(self, key, block):
        callback = getattr(self.loader, "prefetch_host_bytes", None)
        return callback(key, block) if callable(callback) else block.host_bytes

    def prefetch_block(self, key, block):
        callback = getattr(self.loader, "prefetch_block", None)
        if not callable(callback):
            raise TypeError("loader advertises prefetch but has no prefetch_block")
        return self._timed("prefetch_block", callback, key, block)

    def finish_prefetch(self, key, block, value):
        callback = getattr(self.loader, "finish_prefetch", None)
        if callable(callback):
            return callback(key, block, value)
        return value

    def close_prefetch(self, key, value):
        callback = getattr(self.loader, "close_prefetch", None)
        if callable(callback):
            return callback(key, value)

    def close_block(self, key, value):
        callback = getattr(self.loader, "close_block", None)
        if callable(callback):
            return callback(key, value)

    def close_all(self):
        callback = getattr(self.loader, "close_all", None)
        if callable(callback):
            return callback()

    def activate(self, index):
        callback = getattr(self.loader, "activate", None)
        @contextmanager
        def observe():
            before = self._active_batch_index
            started = time.perf_counter_ns()
            self._active_batch_index = index
            try:
                with (null_context() if not callable(callback) else callback(index)):
                    yield
            finally:
                ended = time.perf_counter_ns()
                window = {"batch_index": index, "start_ns": started,
                          "end_ns": ended, "seconds": (ended - started) / 1e9}
                with self._lock:
                    self.batch_windows.append(window)
                self._active_batch_index = before
        return observe()


@contextmanager
def null_context():
    yield


class SyntheticLoader:
    """In-memory input loader used to exercise the production replay path."""

    prefetch_across_batches = True

    def __init__(self, arrays):
        self.arrays = arrays
        self.active = None
        self.read_counts = Counter()

    @contextmanager
    def activate(self, index):
        before, self.active = self.active, index
        try:
            yield self
        finally:
            self.active = before

    def read_block(self, key, block):
        self.read_counts[key] += 1
        return self.arrays[key]

    def materialize(self, key, source, raw_blocks):
        return raw_blocks[source.metadata["block_key"]]

    def can_prefetch_block(self, key, block):
        return True

    def prefetch_host_bytes(self, key, block):
        return block.host_bytes + block.staging_host_bytes

    def prefetch_block(self, key, block):
        self.read_counts[key] += 1
        return self.arrays[key]

    def finish_prefetch(self, key, block, value):
        return value

    def close_prefetch(self, key, value):
        return None

    def close_block(self, key, value):
        return None

    def close_all(self):
        self.active = None


def _source_description(role, key):
    return TensorDescription(
        axes=("spin0", "spin1", "mode0", "mode1"),
        role=role,
        data_identity=f"synthetic:{key}",
        selection_identity="full-small-fixture",
        source_bindings=(key,),
    )


def _make_synthetic_batch(ne, pair_index, *, seed):
    rng = np.random.default_rng(seed)
    dtype = np.dtype("complex128")
    tensor_shape = (4, 4, ne, ne)
    arrays, specs, sources, blocks, preparations = {}, {}, {}, {}, {}
    keys = ("P0", "P1", *(f"V{i}" for i in range(_OPERATORS)),
            *(f"W{j}" for j in range(_OPERATORS)))
    for key in keys:
        role = "propagator" if key in ("P0", "P1") else "elemental"
        block_key = f"pair{pair_index}:{key}"
        value = (rng.standard_normal(tensor_shape) + 1j * rng.standard_normal(tensor_shape)).astype(dtype)
        arrays[block_key] = value
        size = int(value.nbytes)
        specs[key] = TensorSpec(tensor_shape, dtype, description=_source_description(role, key))
        sources[key] = SourceSpec(
            region=f"logical:{block_key}", region_bytes=0, snapshot_bytes=size,
            metadata={"block_key": block_key},
        )
        blocks[block_key] = StorageBlock(
            f"synthetic:{key}", f"pair{pair_index}:all", size,
            host_bytes=size, staging_host_bytes=size,
        )
        preparations[key] = SourcePreparation((block_key,), dtype=dtype.str)

    terms = [
        ContractionTerm(("P0", f"V{i}"), "BAON,ABNO->")
        for i in range(_OPERATORS)
    ]
    terms.extend(
        ContractionTerm(("P1", f"W{j}"), "BAON,ABNO->")
        for j in range(_OPERATORS)
    )
    terms.extend(
        ContractionTerm(("P0", "P1", f"V{i}", f"W{j}"),
                        "BAON,DCQP,ADNQ,CBPO->")
        for i in range(_OPERATORS)
        for j in range(_OPERATORS)
    )
    problem = ContractionProblem(specs, tuple(terms), sources, backend="numpy")
    return WorkBatch(problem, blocks, preparations, arrays)


def _cupy_problem(problem):
    """Keep host reads and preparation bounds; model each logical copy on GPU."""
    sources = {}
    for key, source in problem.sources.items():
        size = int(np.prod(problem.specs[key].shape)) * problem.specs[key].dtype.itemsize
        sources[key] = replace(
            source,
            snapshot_bytes=size,
            transfer_bytes=max(source.transfer_bytes, size),
        )
    return replace(problem, backend="cupy", sources=sources)


def _build_physical(batch, *, backend_name, profile, host_limit, device_limit,
                    reserved_host=0, reserved_device=0):
    problem = batch.problem if backend_name == "numpy" else _cupy_problem(batch.problem)
    blocks = dict(batch.blocks)
    preparations = dict(batch.preparations)
    if backend_name == "cupy":
        # Benchmark-only placement: raw inputs and preparation stay on host;
        # logical snapshots are accounted on device by the normal GPU planner.
        blocks = {key: replace(block, device_bytes=0, staging_device_bytes=0)
                  for key, block in blocks.items()}
    physical = MacroSearchSpace(
        problem, blocks, preparations,
        profile=profile,
        memory_limit=host_limit if backend_name == "numpy" else device_limit,
        host_memory_limit=host_limit,
        reserved_host_bytes=reserved_host,
        reserved_device_bytes=reserved_device,
    )
    return physical


def _compute_actions(storage_plan):
    return tuple(
        action for action in storage_plan.state.actions
        if action.kind in ("contract", "sparse_bridge", "baryon_join")
    )


def _reconstruct_dag(physical, storage_plan, *, expected_computations=None):
    actions = _compute_actions(storage_plan)
    state = ContractionState.initial(physical.problem)
    state = replace(state, loaded=frozenset(state.catalog) | set(state.problem.sources))
    snippets, workspaces = [], []
    rebuilt_actions = []
    begin = 0
    current_tensor = None
    current_workspace = 0
    for index, action in enumerate(actions):
        described, _, workspace, _, _ = describe(
            state, action.term_index, action.positions,
            profile=physical.logical_profile,
        )
        if replace(action, seconds=described.seconds) != described:
            raise AssertionError("selected computation changed while rebinding its backend")
        action = described
        rebuilt_actions.append(action)
        state = replay(state, action, physical.logical_profile)
        if action.tensor != current_tensor:
            if current_tensor is not None:
                snippets.append((begin, index))
                workspaces.append(current_workspace)
            begin, current_tensor, current_workspace = index, action.tensor, workspace
        else:
            current_workspace = max(current_workspace, workspace)
    if current_tensor is not None:
        snippets.append((begin, len(actions)))
        workspaces.append(current_workspace)
    if not state.done:
        raise AssertionError("selected action route did not complete every Term")
    state = replace(state, actions=tuple(rebuilt_actions))
    dag = _Dag(state, tuple(snippets), tuple(workspaces), physical)
    if expected_computations is not None and len(dag.nodes) != expected_computations:
        raise AssertionError(
            f"expected {expected_computations} selected computations, got {len(dag.nodes)}"
        )
    return dag


def _make_schedules(batch, *, backend_name, profile=None, host_limit=_HOST_LIMIT,
                    device_limit=_DEVICE_LIMIT, reserved_host=0, reserved_device=0,
                    expected_computations=None, selected_plan=None,
                    selected_diagnostic=None):
    profile = CostProfile() if profile is None else profile
    physical = _build_physical(
        batch, backend_name=backend_name, profile=profile,
        host_limit=host_limit, device_limit=device_limit,
        reserved_host=reserved_host, reserved_device=reserved_device,
    )
    selected = None
    if selected_plan is None:
        selected = solve_storage(
            physical.problem, physical.blocks, physical.preparations,
            memory_limit=physical.memory_limit,
            profile=profile,
            host_memory_limit=physical.host_memory_limit,
            reserved_host_bytes=physical.reserved_host_bytes,
            reserved_device_bytes=physical.reserved_device_bytes,
            optimizer="anytime",
            deadline_seconds=15.0,
            max_states=2_000,
            root_width=2,
            branch_width=2,
        )
        if selected is None or selected.plan is None or not selected.plan.feasible:
            raise RuntimeError("the unified planner did not return a feasible storage plan")
        selected_plan = selected.plan
        selected_engine = selected.diagnostic.get("engine")
        search_performed = True
    else:
        if not selected_plan.feasible:
            raise MemoryError("the selected calculation path is not storage-feasible")
        if selected_diagnostic is None:
            raise AssertionError("selected calculation path is missing planner diagnostics")
        selected_engine = selected_diagnostic.get("engine")
        search_performed = selected_diagnostic.get("search_performed")

    dag = _reconstruct_dag(physical, selected_plan,
                           expected_computations=expected_computations)
    row = _plan(dag, range(len(dag.nodes)))
    frontier = _Frontier(dag)
    categories = {}
    for index, node in enumerate(dag.nodes):
        if len(node.actions) > 1:
            categories[index] = 0  # shared K_i / H_j intermediates
        else:
            term = dag.state.problem.terms[node.actions[0].term_index]
            categories[index] = 1 if len(term.inputs) == 4 else 2
    order = []
    while frontier.ready:
        index = min(frontier.ready, key=lambda item: (
            categories[item], frontier.estimate(item), item,
        ))
        frontier.advance(index)
        order.append(index)
    bulk = _plan(dag, order)
    _assert_same_computations(row, bulk)
    selected_digest = _computation_signature_sha256(selected_plan)
    row_digest = _computation_signature_sha256(row)
    bulk_digest = _computation_signature_sha256(bulk)
    if selected_digest != row_digest or selected_digest != bulk_digest:
        raise AssertionError("fixed schedules changed the selected computation multiset")
    return bulk, row, {
        "nodes": len(dag.nodes),
        "terms": len(batch.problem.terms),
        "cost_seconds": row.seconds,
        "row_peak_host_bytes": row.peak_host_bytes,
        "row_peak_device_bytes": row.peak_device_bytes,
        "bulk_peak_host_bytes": bulk.peak_host_bytes,
        "bulk_peak_device_bytes": bulk.peak_device_bytes,
        "selected_engine": selected_engine,
        "selected_search_performed": search_performed,
        "selected_path_computation_multiset_sha256": selected_digest,
        "bulk_computation_multiset_sha256": bulk_digest,
        "row_computation_multiset_sha256": row_digest,
    }


def _computation_signature(plan):
    return Counter(
        (action.term_index, action.kind, action.operands, action.equation,
         action.tensor, action.reused)
        for action in plan.state.actions
        if action.kind not in ("load", "release", "read_block", "close_block")
    )


def _assert_same_computations(left, right):
    if _computation_signature(left) != _computation_signature(right):
        raise AssertionError("storage scheduling changed the selected operation multiset")


def _computation_signature_sha256(plan):
    signature = _computation_signature(plan)
    rows = [
        {"action": list(key), "count": count}
        for key, count in sorted(signature.items(), key=lambda item: repr(item[0]))
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_numpy(value, xp):
    if xp.__name__ == "cupy":
        return xp.asnumpy(value)
    return np.array(value, copy=True)


def _collect_result(index, report, *, batches, backend, output):
    values = tuple(report.values) if hasattr(report, "values") else tuple(report)
    output[index] = tuple(_to_numpy(value, backend) for value in values)


def _assemble_real(expanded, batches, output):
    values = []
    for index, batch in enumerate(batches):
        task = batch.task
        if task is None:
            raise AssertionError("real result assembly requires its selected DiagramInputs batch")
        points = (task.coordinate_batch.points if task.coordinate_batch is not None
                  else expanded.coordinates.points)
        term_ids = task.term_ids
        axes = task.batch_output_axes or tuple(None for _ in term_ids)
        raw_values = output[index]
        if len(raw_values) != len(term_ids) or len(axes) != len(term_ids):
            raise AssertionError("batch result count differs from its selected Terms")
        for term_id, value, output_axis in zip(term_ids, raw_values, axes, strict=True):
            if output_axis is None:
                for point in points:
                    values.append(TermValue(term_id, point.key, np.array(value, copy=True)))
            else:
                if value.shape[output_axis] != len(points):
                    raise AssertionError("batch result coordinate axis has the wrong size")
                for lane, point in enumerate(points):
                    values.append(TermValue(
                        term_id, point.key,
                        np.array(np.take(value, lane, axis=output_axis), copy=True),
                    ))
    return assemble_correlator(expanded, tuple(values))


class PoolSampler:
    def __init__(self, backend, interval=0.01):
        self.backend, self.interval = backend, interval
        self.stop = threading.Event()
        self.thread = None
        self.peak_used = self.peak_total = 0

    def _sample(self):
        pool = self.backend.get_default_memory_pool()
        self.peak_used = max(self.peak_used, int(pool.used_bytes()))
        self.peak_total = max(self.peak_total, int(pool.total_bytes()))

    def __enter__(self):
        if self.backend.__name__ != "cupy":
            return self
        self._sample()

        def poll():
            while not self.stop.wait(self.interval):
                try:
                    self._sample()
                except Exception:
                    return

        self.thread = threading.Thread(target=poll, name="cupy-pool-sampler", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        if self.thread is not None:
            self.stop.set()
            self.thread.join()
            self._sample()

    def result(self):
        if self.backend.__name__ != "cupy":
            return None
        return {
            "pool": "cupy.get_default_memory_pool()",
            "sampling_interval_seconds": self.interval,
            "sampled_peak_used_bytes": self.peak_used,
            "sampled_peak_reserved_bytes": self.peak_total,
        }


def _synchronize(backend):
    if backend.__name__ == "cupy":
        backend.cuda.get_current_stream().synchronize()


def _run_replay(plans, batches, loader, backend, *, io_overlap, expanded=None):
    joined = join_storage_batches(tuple(plans))
    traced = loader if isinstance(loader, TraceLoader) else TraceLoader(loader)
    traced.events.clear()
    traced.batch_windows.clear()
    consumer_by_block = {}
    for batch_index, plan in enumerate(joined):
        for action in plan.state.actions:
            if action.kind == "load":
                for raw in plan.preparations[action.tensor].blocks:
                    consumer_by_block.setdefault(raw, batch_index)
    traced.set_batch_consumers(consumer_by_block)
    indices = {id(plan): index for index, plan in enumerate(joined)}
    results = [None] * len(joined)

    def activate(plan):
        try:
            index = indices[id(plan)]
        except KeyError as exc:
            raise AssertionError("joined plan identity changed unexpectedly") from exc
        return traced.activate(index)

    def consume(index, report):
        _collect_result(index, report, batches=batches, backend=backend, output=results)

    _synchronize(backend)
    started = time.perf_counter()
    with PoolSampler(backend) as pool:
        replay_storage_batches(
            joined, traced, consume, backend=backend, measure=False,
            prejoined=True, batch_context=activate,
            source_descriptions=tuple(batch.problem.sources for batch in batches),
            io_overlap=io_overlap,
        )
        assembled = None if expanded is None else _assemble_real(expanded, batches, results)
        _synchronize(backend)
    elapsed = time.perf_counter() - started
    events = tuple(traced.events)
    for event in events:
        if event["event"] not in ("read_block", "prefetch_block"):
            continue
        consumer = event["consumer_batch_index"]
        overlap = next((window["batch_index"] for window in traced.batch_windows
                        if consumer is not None and window["batch_index"] < consumer
                        and event["start_ns"] < window["end_ns"]
                        and event["end_ns"] > window["start_ns"]), None)
        event["read_ahead_during_batch_index"] = overlap
        event["read_ahead_for_future_batch"] = overlap is not None
    record = {
        "elapsed_seconds_including_replay_and_assembly": elapsed,
        "io_overlap_enabled": bool(io_overlap),
        "read_callback_count": sum(event["event"] in ("read_block", "prefetch_block") for event in events),
        "read_callback_seconds": sum(event["seconds"] for event in events
                                      if event["event"] in ("read_block", "prefetch_block")),
        "materialize_callback_count": sum(event["event"] == "materialize" for event in events),
        "materialize_callback_seconds": sum(event["seconds"] for event in events
                                             if event["event"] == "materialize"),
        "loader_timeline": list(events),
        "batch_execution_windows": list(traced.batch_windows),
        "read_ahead_for_future_batch_count": sum(
            event["event"] in ("read_block", "prefetch_block")
            and event["read_ahead_for_future_batch"] for event in events
        ),
        "first_consumer_batch_by_raw_block": consumer_by_block,
        "cupy_pool": pool.result(),
        "term_values_by_batch": results,
        "assembled": assembled,
    }
    return record, traced


def _assert_results_close(reference, candidate, *, rtol=None, atol=None):
    if len(reference) != len(candidate):
        raise AssertionError("variant returned a different number of task batches")
    for batch_index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        if len(left) != len(right):
            raise AssertionError(f"batch {batch_index} returned a different Term count")
        for term_index, (expected, actual) in enumerate(zip(left, right, strict=True)):
            _assert_array_close(expected, actual,
                                err_msg=f"batch {batch_index}, Term {term_index}",
                                rtol=rtol, atol=atol)


def _assert_array_close(expected, actual, *, err_msg=None, rtol=None, atol=None):
    expected = _to_host_numpy(expected)
    actual = _to_host_numpy(actual)
    dtype = np.result_type(np.asarray(expected).dtype, np.asarray(actual).dtype)
    if rtol is None or atol is None:
        if dtype.kind == "c" and dtype.itemsize <= 8 or dtype.kind == "f" and dtype.itemsize <= 4:
            rtol, atol = 1e-5, 1e-6
        else:
            rtol, atol = 1e-10, 1e-8
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)


def _to_host_numpy(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp
        return cp.asnumpy(value)
    return np.asarray(value)


def _assert_nested_close(expected, actual):
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or expected.keys() != actual.keys():
            raise AssertionError("assembled output mappings differ")
        for key in expected:
            _assert_nested_close(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise AssertionError("assembled output sequences differ")
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_close(left, right)
    else:
        _assert_array_close(expected, actual, err_msg="assembled output")


def _synthetic_reference(batch):
    arrays = batch.arrays
    prefix = "pair0:"
    p0, p1 = arrays[prefix + "P0"], arrays[prefix + "P1"]
    out = []
    for i in range(_OPERATORS):
        out.append(np.einsum("BAON,ABNO->", p0, arrays[prefix + f"V{i}"], optimize=True))
    for j in range(_OPERATORS):
        out.append(np.einsum("BAON,ABNO->", p1, arrays[prefix + f"W{j}"], optimize=True))
    for i in range(_OPERATORS):
        for j in range(_OPERATORS):
            out.append(np.einsum(
                "BAON,DCQP,ADNQ,CBPO->", p0, p1,
                arrays[prefix + f"V{i}"], arrays[prefix + f"W{j}"],
                optimize=True,
            ))
    return tuple(np.asarray(value) for value in out)


def _run_synthetic(args, backend):
    profile = CostProfile()
    batches = tuple(_make_synthetic_batch(args.ne, index, seed=20261006 + index)
                    for index in range(2))
    schedules = [_make_schedules(batch, backend_name=backend.__name__, profile=profile,
                                 expected_computations=None)
                 for batch in batches]
    bulk_plans = tuple(item[0] for item in schedules)
    row_plans = tuple(item[1] for item in schedules)
    if any(len(batch.problem.terms) != _EXPECTED_TERMS for batch in batches):
        raise AssertionError("synthetic fixture does not have 224 Terms")
    loader = SyntheticLoader({key: value for batch in batches for key, value in batch.arrays.items()})
    computation_counts = [item[2]["nodes"] for item in schedules]
    if len(set(computation_counts)) != 1:
        raise AssertionError("synthetic batches selected different computation DAG sizes")
    output = {"workload": "synthetic", "ne": args.ne, "terms_per_batch": _EXPECTED_TERMS,
              "computations_per_batch": computation_counts[0],
              "schedule_audit": [item[2] for item in schedules], "variants": {}}
    modes = (
        ("bulk_sync", bulk_plans, False),
        ("row_sync", row_plans, False),
        ("row_prefetch", row_plans, True),
    )
    # Warm the loader/replay path once, then interleave measured variants.
    _, loader = _run_replay(row_plans, batches, loader, backend, io_overlap=False)
    results = {name: [] for name, _, _ in modes}
    for round_index in range(args.repeats):
        offset = round_index % len(modes)
        for name, plans, overlap in modes[offset:] + modes[:offset]:
            record, loader = _run_replay(plans, batches, loader, backend, io_overlap=overlap)
            results[name].append(record)
    reference = results["bulk_sync"][0]["term_values_by_batch"]
    for name, records in results.items():
        for record in records:
            _assert_results_close(reference, record["term_values_by_batch"])
        output["variants"][name] = [
            {key: value for key, value in record.items()
             if key not in ("term_values_by_batch", "assembled")}
            for record in records
        ]
    if args.ne <= 4:
        reference = _synthetic_reference(batches[0])
        _assert_results_close((reference,),
                              (results["bulk_sync"][0]["term_values_by_batch"][0],))
        output["independent_numpy_reference"] = "passed"
    else:
        output["independent_numpy_reference"] = "not-run-for-this-size-or-backend"
    output["computation_multiset"] = "identical across bulk/row/prefetch"
    output["all_methods_allclose"] = "passed"
    if backend.__name__ == "cupy":
        output["gpu_placement_transfer_bytes"] = gpu_placement_transfer_bytes(batches)
    return output


def gpu_placement_transfer_bytes(batches):
    byte_count = 0
    for batch in batches:
        for key, source in batch.problem.sources.items():
            if key not in batch.problem.specs:
                continue
            snapshot = source.snapshot_bytes
            if snapshot is None:
                spec = batch.problem.specs[key]
                snapshot = int(np.prod(spec.shape)) * spec.dtype.itemsize
            byte_count += int(snapshot)
    return {
        "logical_host_to_device_bytes": byte_count,
        "note": "benchmark-only GPU placement; wall time includes actual runtime copies",
    }


def _import_mesons(args):
    import_paths = [args.meson55_dir, os.environ.get("LATTICE_SCATTERING_ROOT")]
    for candidate in import_paths:
        if not candidate:
            continue
        directory = Path(candidate).resolve()
        candidates = (directory, directory / "scripts" / "spectroscopy")
        for candidate_path in candidates:
            path = str(candidate_path)
            if candidate_path.exists() and path not in sys.path:
                sys.path.insert(0, path)
    script_name = os.environ.get("CSNS_MESON55_MODULE", "calculate_meson55")
    data_name = os.environ.get("CSNS_MESON55_DATA_MODULE", "meson55_projected_data")
    script = importlib.import_module(script_name)
    data = importlib.import_module(data_name)
    for module, symbol in ((script, "calculate_matrices"), (script, "build_basis"),
                           (data, "open_t40_inputs")):
        if not callable(getattr(module, symbol, None)):
            raise ImportError(f"{module.__name__} must provide {symbol}()")
    return script, data


@contextmanager
def _real_batches(args, backend):
    script, data_module = _import_mesons(args)
    with data_module.open_t40_inputs(root=args.data_root, cfg=args.cfg) as inputs:
        all_multiplets = tuple(script.build_basis())
        basis = tuple(item for item in all_multiplets
                      if getattr(item, "block", None) == "A1_Pm_Cp_Rm")
        if len(basis) != 6:
            raise AssertionError(f"A1_Pm_Cp_Rm must select six operators; got {len(basis)}")
        mode_boundaries = args.mode_boundaries
        pairs = ({0: 1, 1: 0}, {0: 2, 1: 0}, {0: 0, 1: 1})[:args.time_pairs]
        for pair in pairs:
            if (pair[0] == pair[1]
                    or inputs.physical_time(pair[0]) == inputs.physical_time(pair[1])):
                raise AssertionError("selected physical time pair must have nonzero dt")
        calculation_started = time.perf_counter()
        print(
            f"csns_meson_matrix phase=calculate_matrices status=start cfg={args.cfg} "
            f"ne=70 operators=6 time_pairs={len(pairs)} "
            f"mode_partitions={mode_boundaries}",
            file=sys.stderr, flush=True,
        )
        calculation = script.calculate_matrices(
            inputs.propagator, inputs.elemental,
            multiplets=basis, ne=70, time_pairs=pairs,
            batch={"axis": "pair", "size": 1},
            mode_partitions=mode_boundaries, optimize="anytime", execute=False, measure=False,
        )
        calculation_wall_seconds = time.perf_counter() - calculation_started
        print(
            f"csns_meson_matrix phase=calculate_matrices status=complete "
            f"wall_seconds={calculation_wall_seconds:.6f} "
            f"planning_seconds={float(calculation.planning_seconds):.6f} "
            f"terms={len(calculation.expanded.terms)}",
            file=sys.stderr, flush=True,
        )
        expanded = calculation.expanded
        task_batches = tuple(expanded.inputs.task_batches)
        paths = tuple(calculation.path.batch_paths)
        if len(task_batches) != len(pairs) or len(paths) != len(pairs):
            raise AssertionError("T40 workflow did not produce one selected plan per pair")
        if mode_boundaries is None and len(expanded.terms) != _EXPECTED_TERMS:
            raise AssertionError(
                f"real selected route expected {_EXPECTED_TERMS} Terms, got {len(expanded.terms)}"
            )
        batches = []
        plans = []
        schedule_audit = []
        for task, path in zip(task_batches, paths, strict=True):
            original_plan = path.plan
            profile = path.profile or CostProfile()
            planner_diagnostic = path.prediction.get("planner")
            if not isinstance(planner_diagnostic, Mapping):
                raise AssertionError("selected calculation path has no planner diagnostics")
            selected_plan = original_plan
            if not selected_plan.feasible or not _compute_actions(selected_plan):
                raise AssertionError("real T40 task has no feasible unified operation plan")
            # Use the already-selected P route; only re-audit it with device
            # placement enabled. Input loader callbacks continue to return host arrays.
            batch = WorkBatch(task.problem, task.blocks, task.preparations, {}, task, path)
            bulk, row, audit = _make_schedules(
                batch, backend_name=backend.__name__, profile=profile,
                host_limit=args.host_memory_bytes,
                device_limit=args.device_memory_bytes,
                reserved_host=path.reserved_host_bytes,
                reserved_device=path.reserved_device_bytes,
                expected_computations=None,
                selected_plan=selected_plan,
                selected_diagnostic=planner_diagnostic,
            )
            _assert_same_computations(selected_plan, row)
            batches.append(batch)
            plans.append((bulk, row))
            schedule_audit.append(audit)
        yield (calculation, inputs, expanded, task_batches, batches, plans,
               schedule_audit, pairs, calculation_wall_seconds, mode_boundaries)


def _run_real(args, backend):
    with _real_batches(args, backend) as prepared:
        return _run_real_open(args, backend, *prepared)


def _run_real_open(args, backend, calculation, inputs, expanded, task_batches,
                   batches, schedules, audit, pairs, calculation_wall_seconds,
                   mode_boundaries):
    loader = TraceLoader(expanded.inputs.loader)
    bulk_plans = tuple(item[0] for item in schedules)
    row_plans = tuple(item[1] for item in schedules)
    output = {
        "workload": "cfg2000_real_t40",
        "configuration": args.cfg,
        "ne": 70,
        "operator_block": "A1_Pm_Cp_Rm",
        "operator_count": 6,
        "mode_partitioning": None if mode_boundaries is None else list(mode_boundaries),
        "time_pairs_slots": [dict(pair) for pair in pairs],
        "terms_per_pair": len(expanded.terms),
        "computations_per_pair": (audit[0]["nodes"] if len({item["nodes"] for item in audit}) == 1
                                   else None),
        "terms_by_batch": [item["terms"] for item in audit],
        "computations_by_batch": [item["nodes"] for item in audit],
        "schedule_audit": audit,
        "variants": {},
        "calculate_matrices_wall_seconds": calculation_wall_seconds,
        "planning_seconds": float(calculation.planning_seconds),
    }
    if backend.__name__ == "cupy":
        output["gpu_placement_transfer_bytes"] = gpu_placement_transfer_bytes(batches)
    if mode_boundaries is None and len(expanded.terms) != _EXPECTED_TERMS:
        output["expected_224_term_route"] = "not-reproduced"
        raise AssertionError("real workload does not match the selected 224-Term fixture")

    # One unrecorded replay warms the shared filesystem page cache. No system
    # cache is cleared; all reported runs use this warm-cache boundary.
    _, loader = _run_replay(row_plans, batches, loader, backend, io_overlap=False,
                            expanded=expanded)
    modes = (
        ("bulk_sync", bulk_plans, False),
        ("row_sync", row_plans, False),
        ("row_prefetch", row_plans, True),
    )
    raw = {}
    for round_index in range(args.repeats):
        rotated = modes[round_index % len(modes):] + modes[:round_index % len(modes)]
        for name, plans, overlap in rotated:
            record, loader = _run_replay(plans, batches, loader, backend,
                                         io_overlap=overlap, expanded=expanded)
            raw.setdefault(name, []).append(record)
    reference = raw["bulk_sync"][0]["term_values_by_batch"]
    reference_assembled = raw["bulk_sync"][0]["assembled"]
    for name in ("bulk_sync", "row_sync", "row_prefetch"):
        for record in raw[name]:
            _assert_results_close(reference, record["term_values_by_batch"])
            _assert_nested_close(reference_assembled, record["assembled"])
        output["variants"][name] = [
            {key: value for key, value in record.items()
             if key not in ("term_values_by_batch", "assembled")}
            for record in raw[name]
        ]
    output["computation_multiset"] = "identical across bulk/row/prefetch"
    output["all_methods_allclose"] = "passed"
    output["assembled_output_allclose"] = "passed"
    output["page_cache_boundary"] = "one row-sync warm-up; no cache clear; timed runs are warm-cache"
    output["planning_seconds_excluded"] = float(calculation.planning_seconds)
    return output


def _resolve_backend(name):
    if name == "numpy":
        return np
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is required for --backend cupy") from exc
    cp.cuda.Device().use()
    cp.cuda.get_current_stream().synchronize()
    return cp


def _runtime_environment(backend):
    info = {"python": sys.version.split()[0], "numpy": np.__version__,
            "backend": backend.__name__}
    if backend.__name__ == "cupy":
        import cupy as cp
        device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        info.update({"cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                     "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                     "device_id": device_id, "device_name": name,
                     "device_total_memory_bytes": int(properties["totalGlobalMem"])})
    return info


def _parse_mode_boundaries(value):
    if value is None:
        return None
    try:
        boundaries = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mode boundaries must be comma-separated integers") from exc
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != 70
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))):
        raise argparse.ArgumentTypeError(
            "Ne70 mode boundaries must start at 0, end at 70, and increase strictly"
        )
    return boundaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("synthetic", "real", "all"), default="synthetic")
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--ne", type=int, default=2,
                        help="synthetic Ne; use 120 for the CSNS GPU run")
    parser.add_argument("--repeats", type=int, default=1,
                        help="timed interleaved rounds per method")
    parser.add_argument("--time-pairs", type=int, choices=(2, 3), default=3)
    parser.add_argument("--mode-boundaries", type=_parse_mode_boundaries, default=None,
                        help="optional CSV Ne70 mode partition, e.g. 0,16,32,48,70")
    parser.add_argument("--cfg", default="2000")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--meson55-dir", default=os.environ.get("CSNS_MESON55_DIR"))
    parser.add_argument("--host-memory-bytes", type=int, default=_HOST_LIMIT)
    parser.add_argument("--device-memory-bytes", type=int, default=_DEVICE_LIMIT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.ne < 1 or args.repeats < 1:
        parser.error("Ne and repeats must be positive")
    if args.mode_boundaries is not None and args.workload not in ("real", "all"):
        parser.error("--mode-boundaries applies only to real T40 input")
    if args.workload in ("real", "all") and not args.data_root:
        args.data_root = os.environ.get(
            "CSNS_T40_DATA_ROOT",
            "/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104",
        )
    backend = _resolve_backend(args.backend)
    output = {"backend": backend.__name__, "pid": os.getpid(),
              "runtime_environment": _runtime_environment(backend), "results": []}
    if args.workload in ("synthetic", "all"):
        if backend.__name__ == "cupy" and args.ne > 4:
            cpu_args = argparse.Namespace(**vars(args))
            cpu_args.ne = 2
            cpu_args.repeats = 1
            output["cpu_reference"] = _run_synthetic(cpu_args, np)
            gpu_reference_args = argparse.Namespace(**vars(args))
            gpu_reference_args.ne = 2
            gpu_reference_args.repeats = 1
            output["gpu_reference"] = _run_synthetic(gpu_reference_args, backend)
        output["results"].append(_run_synthetic(args, backend))
    if args.workload in ("real", "all"):
        output["results"].append(_run_real(args, backend))
    encoded = json.dumps(output, sort_keys=True, default=str)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
        summaries = []
        for result in output["results"]:
            variants = result.get("variants", {})
            summaries.append({
                "workload": result.get("workload"),
                "terms": result.get("terms_per_batch", result.get("terms_per_pair")),
                "computations": result.get("computations_per_batch",
                                             result.get("computations_per_pair")),
                "all_methods_allclose": result.get("all_methods_allclose"),
                "elapsed_seconds": {
                    name: [item["elapsed_seconds_including_replay_and_assembly"]
                           for item in rows]
                    for name, rows in variants.items()
                },
            })
        print(json.dumps({"status": "complete", "report": str(args.json),
                          "runtime_environment": output["runtime_environment"],
                          "results": summaries}, sort_keys=True), flush=True)
    else:
        print(encoded, flush=True)


if __name__ == "__main__":
    main()
