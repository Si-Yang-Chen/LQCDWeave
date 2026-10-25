"""Execute a shared-block storage schedule using the existing numerical replay."""

from __future__ import annotations
from lqcdweave.contracts.correlator.api import validate_executable_resources

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from contextlib import nullcontext
from math import prod
from types import MappingProxyType
from weakref import ref

import numpy as np

from lqcdweave.contracts.correlator.api import (
    PlannedContraction,
    SceneStoragePlan,
    SourceLoader,
    SourceSpec,
    SourcePreparation,
    TensorPreparation,
)
from lqcdweave.contracts.measurement import ReplayReport

from .scene_runtime import (
    _coordinate_batch_sparse_workspace_bytes,
    replay, bind_scene_execution,
)
from .read_ahead import prepare_batch_read_ahead, prepare_read_ahead, supports_batch_read_ahead
from .preparation import bind_prepared_loader, PreparedSourceLoader
from .device import active_device_execution, DeviceSourceLoader
from .diagnostics import debug_enabled
from .binding_cache import bind_cached, native_storage_binding, native_resource_audit_binding


def _validate_static_resources(plan, backend):
    """Reuse a completed audit only for this exact immutable plan/backend."""
    return bind_cached('resource-audit', plan, backend,
        lambda: validate_executable_resources(plan),
        lambda: native_resource_audit_binding(plan))


def _bind_loader_for_plan(loader, plan, backend):
    if getattr(loader, "preparation_bound", False) is True:
        return loader
    if (type(loader) in (PreparedSourceLoader, DeviceSourceLoader)
            and loader.backend is backend
            and (previous := getattr(loader, '_storage_plan', None)) is not None
            and previous() is plan):
        return loader
    preparations = dict(getattr(loader, "preparations", {}) or {})
    preparations.update(plan.preparations)
    specs = dict(getattr(loader, "specs", {}) or {})
    specs.update(plan.state.catalog)
    device = active_device_execution()
    if device is not None:
        if (isinstance(loader, DeviceSourceLoader) and loader.device is device
                and getattr(loader, '_batch_preparation_views', False)):
            loader.bind_batch_preparations(preparations, specs)
            bound = loader
        else:
            bound = device.bind_loader(loader, preparations, specs)
    else:
        bound = bind_prepared_loader(loader, preparations, specs, backend=backend)
    bound._storage_plan = ref(plan)
    return bound


def _bind_batch_loader(loader, plans, backend):
    """Bind a complete, compatible task closure to one raw-buffer owner.

    A joined raw block can remain live across plan boundaries. Rebinding a
    DeviceSourceLoader then loses the upload/host-owner table (and is refused
    by the device scope). Give read-ahead and every block execution the same
    loader instead. Repeated plan keys must have identical definitions; older
    opaque/activation-based batches with conflicting names keep per-plan
    preparation views, while a device task retains one raw-buffer owner.
    Streaming cannot use this complete-task proof.
    """
    if getattr(loader, 'preparation_bound', False) is True:
        return loader

    def bind_first_view():
        bound = _bind_loader_for_plan(loader, plans[0], backend)
        if isinstance(bound, DeviceSourceLoader):
            bound._batch_preparation_views = True
        return bound

    preparations = dict(getattr(loader, 'preparations', {}) or {})
    specs = dict(getattr(loader, 'specs', {}) or {})
    selected_preparations, selected_specs = {}, {}
    for plan in plans:
        for key, value in plan.preparations.items():
            previous = selected_preparations.setdefault(key, value)
            if previous != value:
                return bind_first_view()
        for key, value in plan.state.catalog.items():
            previous = selected_specs.setdefault(key, value)
            if previous != value:
                return bind_first_view()
    # As in the ordinary single-plan binder, P's selected top-level records
    # override matching loader entries while its full scalar closure survives.
    preparations.update(selected_preparations)
    specs.update(selected_specs)
    device = active_device_execution()
    bound = (device.bind_loader(loader, preparations, specs) if device is not None
             else bind_prepared_loader(loader, preparations, specs, backend=backend))
    # Private execution-owned proof: tables cover every accepted task plan.
    # In particular, read-ahead's replaced resource plans need no new wrapper.
    bound.preparation_bound = True
    return bound


def _read_ahead_loader(physical, bound):
    # Host readers and their instrumentation keep their physical identity on
    # CPU. An active device scope must expose its upload-aware reader instead.
    return bound if active_device_execution() is not None else _physical_loader(physical)


def _prepend_plan(first, remaining):
    yield first
    first = None
    yield from remaining


def _physical_loader(loader):
    """Unwrap execution-local bindings when checking physical capabilities."""
    while isinstance(loader, (PreparedSourceLoader, DeviceSourceLoader)):
        loader = loader.prepared if isinstance(loader, DeviceSourceLoader) else loader.physical
    return loader


def _identity_upload_lifetimes(plan):
    """Pure roles/release positions; mutable recipes are rechecked at materialization."""
    if not native_storage_binding(plan):
        return frozenset(), {}
    keys = frozenset(key for key, prep in plan.preparations.items()
        if type(prep) is SourcePreparation and type(prep.recipe) is TensorPreparation
        and prep.recipe.kind == 'identity' and prep.recipe.compute_dtype is not None
        and not prep.recipe.conjugate and prep.recipe.raw_index is None
        and not prep.recipe.index_permutations)
    releases = {index: action.tensor for index, action in enumerate(plan.state.actions)
                if action.kind == 'release' and not action.reused and action.tensor in keys}
    return keys, releases


def _coordinate_batch_workspace_shortfall(plan, backend):
    """Independently compare the saved workspace with X's NumPy kernel ABI."""
    problem = plan.state.problem
    marked = {
        key for key, source in problem.sources.items()
        if isinstance(source.metadata, Mapping)
        and source.metadata.get("family") == "p2p-coordinate-batch"
    }
    if not marked:
        return 0
    points_by_action = {}
    for point in plan.memory:
        if point.phase == "sparse_bridge":
            points_by_action.setdefault(point.action_index, []).append(point)
    seen = set()
    correction = 0
    executed_marked = False
    for index, action in enumerate(plan.state.actions):
        if action.kind != "sparse_bridge":
            continue
        from .operations.sparse_bridge import action_bridges

        bridges = tuple(
            bridge for bridge in action_bridges(plan.state, action)
            if bridge is not None
        )
        marked_records = tuple(
            bridge.record for bridge in bridges
            if (isinstance(problem.sources[bridge.record].metadata, Mapping)
                and problem.sources[bridge.record].metadata.get("family")
                == "p2p-coordinate-batch")
        )
        if not marked_records:
            continue
        if len(marked_records) != len(bridges):
            raise ValueError("coordinate-batched sparse actions cannot mix marked and scalar records")
        seen.update(marked_records)
        if action.reused:
            continue
        executed_marked = True
        if (problem.backend != "numpy"
                or getattr(backend, "__name__", None) != "numpy"):
            raise NotImplementedError(
                "coordinate-batched sparse workspace admission currently requires NumPy"
            )
        workspace = _coordinate_batch_sparse_workspace_bytes(plan.state, action)
        if workspace is None:
            raise ValueError("marked coordinate-batch action lost its source contract")
        points = points_by_action.get(index, ())
        if len(points) != 1:
            raise ValueError("audited plan has no unique sparse action memory point")
        correction = max(
            correction,
            max(0, workspace - points[0].workspace_host_bytes),
        )
    if seen != marked:
        raise ValueError("every coordinate-batched source must have one executed sparse action")
    if not executed_marked:
        return 0

    return correction


def validate_coordinate_batch_workspace(plan, backend):
    """Reject an incomplete saved envelope before any source IO, without mutation."""
    shortfall = _coordinate_batch_workspace_shortfall(plan, backend)
    if shortfall:
        raise MemoryError(
            f"saved coordinate-batched sparse workspace is short by {shortfall} host bytes; "
            "prepare the storage plan through planning before execution"
        )


def audit_execution_memory(path, *, backend=np):
    """Audit X's known workspace across a complete selected path, without I/O.

    The returned immutable mapping summarizes every final joined batch plan.
    It does not search, activate a loader, read a getter, or mutate the
    supplied ``PlannedContraction``. Saved peaks/reserves are reported unchanged;
    missing known sparse workspace makes the saved descriptor inadmissible.
    This is a modeled admission envelope, not an RSS or native-scratch bound.
    """
    if not isinstance(path, PlannedContraction):
        raise TypeError("path must be a PlannedContraction")

    if path.plan is None:
        if path.execution_mode != "empty":
            raise ValueError("only an empty path may omit its storage plan")
        inputs = getattr(path.expanded, "inputs", None)
        for name in ("reserved_host_bytes", "reserved_device_bytes"):
            value = getattr(path, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"empty path {name} must be nonnegative integer bytes")
        peak_host = path.reserved_host_bytes + (
            0 if inputs is None else inputs.assembly_host_bytes
        )
        peak_device = path.reserved_device_bytes + (
            0 if inputs is None else inputs.assembly_device_bytes
        )
        backend_name = backend if isinstance(backend, str) else getattr(backend, "__name__", None)
        primary_peak = peak_device if str(backend_name).lower() == "cupy" else peak_host
        feasible = (
            (path.memory_limit is None or primary_peak <= path.memory_limit)
            and (path.host_memory_limit is None or peak_host <= path.host_memory_limit)
        )
        return MappingProxyType({
            "peak_host_bytes": peak_host,
            "peak_device_bytes": peak_device,
            "reserved_host_bytes": path.reserved_host_bytes,
            "reserved_device_bytes": path.reserved_device_bytes,
            "feasible": feasible,
            "batches": (),
        })

    if path.plan_identity != id(path.plan):
        raise ValueError("path plan object no longer matches its selected descriptor")
    selected = path.batch_paths or (path,)
    if path.batch_paths and path.batch_paths[0].plan is not path.plan:
        raise ValueError("task path does not expose its first joined batch plan")

    records = []
    peak_host = peak_device = 0
    feasible = True
    for index, item in enumerate(selected):
        if item.plan is None or item.plan_identity != id(item.plan):
            raise ValueError("one selected batch has no matching storage plan")
        correction = _coordinate_batch_workspace_shortfall(item.plan, backend)
        record = MappingProxyType({
            "batch_index": index,
            "plan_peak_host_bytes": item.plan.peak_host_bytes,
            "plan_peak_device_bytes": item.plan.peak_device_bytes,
            "x_workspace_correction_host_bytes": correction,
            "corrected_peak_host_bytes": item.plan.peak_host_bytes,
            "corrected_peak_device_bytes": item.plan.peak_device_bytes,
            "reserved_host_bytes": item.plan.reserved_host_bytes,
            "reserved_device_bytes": item.plan.reserved_device_bytes,
            "feasible": (item.plan.feasible and not correction),
        })
        records.append(record)
        peak_host = max(peak_host, item.plan.peak_host_bytes)
        peak_device = max(peak_device, item.plan.peak_device_bytes)
        feasible = feasible and (item.plan.feasible and not correction)
    return MappingProxyType({
        "peak_host_bytes": peak_host,
        "peak_device_bytes": peak_device,
        "feasible": feasible,
        "batches": tuple(records),
    })


class _BlockExecution:
    def __init__(
        self,
        plan: SceneStoragePlan,
        loader: SourceLoader,
        backend: object,
        shared: tuple[dict[str, object], set[str]] | None = None,
        retained: dict[str, object] | None = None,
        sources: Mapping[str, SourceSpec] | None = None,
        temporary_scope=None,
        io_overlap=True,
        measure=False,
        task_read_ahead=None,
        action_offset=0,
        execution_tables=None,
        read_ahead_loader=None,
        owned_preparations=None,
        packed_value_cache=None,
    ):
        if debug_enabled():
            validate_executable_resources(plan, allow_temporary_array=temporary_scope is not None)
        if plan.kernel_batches or plan.scalar_reduction_groups or plan.cartesian_reduction_groups:
            if getattr(backend, '__name__', '') != plan.state.problem.backend:
                raise ValueError("execution backend disagrees with batched storage plan")
        physical_loader = loader if read_ahead_loader is None else read_ahead_loader
        loader = _bind_loader_for_plan(loader, plan, backend)
        if debug_enabled():
            _validate_loader(loader, plan)
        self.plan, self.loader = plan, loader
        self.backend = backend
        self._owned_preparations = owned_preparations
        if (self._owned_preparations is None
                and isinstance(loader, (PreparedSourceLoader, DeviceSourceLoader))
                and loader.backend is backend):
            self._owned_preparations = loader.materializes_owned
        self._borrowed_sources = {}
        self._identity_keys, self._identity_releases = (), {}
        if (type(loader) is DeviceSourceLoader and not debug_enabled()
                and shared is None and execution_tables is None
                and not plan.retained_before and not plan.retained_after
                and not plan.explicit_resources):
            self._identity_keys, self._identity_releases = bind_cached(
                'identity-upload-lifetimes', plan, backend,
                lambda: _identity_upload_lifetimes(plan), lambda: native_storage_binding(plan))
        self.execution_tables = (bind_cached('storage', plan, backend,
                                    lambda: bind_scene_execution(plan.state, plan.kernel_batches, backend,
                                        events=plan.events, retained=plan.retained_after,
                                        tensor_release_indices=tuple(self._identity_releases),
                                        scalar_reduction_groups=plan.scalar_reduction_groups,
                                        cartesian_reduction_groups=plan.cartesian_reduction_groups),
                                    lambda: native_storage_binding(plan),
                                    signature=tuple(self._identity_releases) or None)
                                 if execution_tables is None else execution_tables)
        self.temporary_scope = temporary_scope
        self._disk_receipts = {}
        self.sources = (plan.state.problem.sources if sources is None
                        else MappingProxyType(dict(sources)))
        if sources is not None and set(self.sources) != set(plan.state.problem.sources):
            raise ValueError("runtime source descriptions do not match the audited plan sources")
        self.retained = {} if retained is None else retained
        self.keep = set(plan.retained_after)
        self.owns_storage = shared is None
        self.live, self.opened = ({}, set()) if shared is None else shared
        self.before, self.after = defaultdict(list), defaultdict(list)
        self.costs = None  # Only explicit measurement consumes per-action costs.
        for event in plan.events:
            (self.before if event.kind == "read" else self.after)[event.action_index].append(event)
        self.read_ahead_started = False
        self._waiting_read = None
        self.owns_read_ahead = task_read_ahead is None
        self.action_offset = action_offset
        if task_read_ahead is None:
            self.plan, self.read_ahead, self.io_reserve = prepare_read_ahead(
                self.plan, _read_ahead_loader(physical_loader, self.loader),
                enabled=io_overlap, measure=measure,
            )
        else:
            self.read_ahead, self.io_reserve = task_read_ahead, 0
        from .packed_values import PackedValueCache
        self.owns_packed_value_cache = packed_value_cache is None
        self.packed_value_cache = packed_value_cache
        if self.owns_packed_value_cache:
            capacity = (self.plan.packed_value_cache_device_bytes
                        if not debug_enabled() and self.plan.state.problem.backend == 'cupy'
                        and all(prep.recipe is not None for prep in self.plan.preparations.values())
                        else 0)
            self.packed_value_cache = PackedValueCache(
                capacity, array_type=getattr(backend, 'ndarray', None))

    def reads_ready(self, index):
        """Inspect only this saved call's input tickets; do not read or plan."""
        for event in self.before.get(index, ()):
            ticket = self.read_ahead.pending_read(event.block)
            if ticket is not None and not ticket.done():
                self._waiting_read = ticket
                return False
        self._waiting_read = None
        return True

    def wait_for_reads(self):
        ticket = self._waiting_read
        # The cursor sets this only for an admitted, running dependency.
        # IO and CPU resource generators continue while this consumer waits.
        ticket.result()
        self._waiting_read = None

    def begin_action(self, index, timed):
        events = self.before.get(index, ())
        for event in events:
            if event.block in self.opened:
                raise RuntimeError("storage schedule attempts a second physical read")
            block = self.plan.blocks[event.block]
            found, value = ((False, None) if self.read_ahead is None else
                            self.read_ahead.take(event.block, block))
            self.live[event.block] = value if found else timed(
                "read_block:" + event.block,
                partial(self.loader.read_block, event.block, block),
                nbytes=block.read_bytes, device=bool(block.device_bytes))
            self.opened.add(event.block)
        if (self.read_ahead is not None
                and (not self.read_ahead_started or events)):
            self.read_ahead.start_after(self.action_offset + index)
            self.read_ahead_started = True

    def materialize(self, key, source):
        preparation = self.plan.preparations[key]
        dependencies = preparation.blocks
        if key in self._identity_keys:
            current = self.loader.preparations[key]
            recipe = current.recipe
            if (current is preparation and type(recipe) is TensorPreparation
                    and recipe.kind == 'identity' and recipe.compute_dtype is not None
                    and not recipe.conjugate
                    and recipe.raw_index is None
                    and not recipe.index_permutations
                    and len(dependencies) == 1 and recipe.inputs == dependencies):
                block, = dependencies
                value = self.live[block]
                if (getattr(value, 'dtype', None) == recipe.compute_dtype
                        and self.loader.borrow_uploaded_block(block, key, value)):
                    self._borrowed_sources[key] = block
                    return value
        return self.loader.materialize(
            key,
            self.sources[key],
            MappingProxyType({k: self.live[k] for k in dependencies}),
        )

    def materializes_owned(self, key):
        return self._owned_preparations is not None and self._owned_preparations(key)

    def source(self, key):
        return self.sources[key]

    def end_action(self, index, timed):
        events = self.after.get(index, ())
        if (events and debug_enabled()
                and getattr(self.backend, "__name__", "") == "cupy"):
            timed("owner_stream_sync", self._synchronize_device, device=True)
        for event in events:
            timed("release_block:" + event.block, partial(self._release, event.block))
        if self._identity_releases:
            key = self._identity_releases.get(index)
            if key in self._borrowed_sources:
                self._release_borrowed_source(key)

    def _release_borrowed_source(self, key):
        self.loader.release_uploaded_borrow(self._borrowed_sources[key], key)
        del self._borrowed_sources[key]

    def _synchronize_device(self):
        self.backend.cuda.get_current_stream().synchronize()

    def _release(self, key):
        value = self.live.pop(key)
        close = getattr(self.loader, "close_block", None)
        if close is not None:
            close(key, value)

    def has_retained(self, key):
        return key in self.retained

    def retained_value(self, key):
        return self.retained[key]

    def retains(self, key):
        return key in self.keep

    def retain_tensor(self, key, value):
        if key in self.keep:
            self.retained[key] = value

    def model_features(self, key):
        prep = self.plan.preparations[key]
        source = self.sources[key]
        spec = self.plan.state.catalog.get(key)
        snapshot = (source.snapshot_bytes if source.snapshot_bytes is not None else
                    (0 if spec is None else int(np.prod(spec.shape)) * spec.dtype.itemsize))
        return dict(storage_preparation=True, first_region=True,
                    read_kind=("p2p" if isinstance(source.metadata, Mapping)
                               and source.metadata.get("family") == "p2p" else "array"),
                    read_bytes=(source.region_bytes if source.read_bytes is None else source.read_bytes),
                    transfer_bytes=source.transfer_bytes, snapshot_bytes=snapshot,
                    flops=prep.flops,
                    traffic_bytes=prep.traffic_bytes, kernel_calls=prep.kernel_calls,
                    dtype=prep.dtype)

    def action_seconds(self, index):
        if self.costs is None:
            self.costs = [a.seconds for a in self.plan.state.actions]
            for batch in self.plan.kernel_batches:
                for member in batch.action_indices:
                    self.costs[member] = 0.0
                self.costs[batch.action_indices[0]] = batch.seconds
            for group in self.execution_tables.scalar_groups:
                for member in group.action_indices:
                    self.costs[member] = 0.0
                self.costs[group.action_indices[0]] = group.seconds
            for group in self.execution_tables.cartesian_groups:
                for child in group.batch_action_indices:
                    for member in child:
                        self.costs[member] = 0.0
                self.costs[group.batch_action_indices[0][0]] = group.seconds
            for member, seconds in self.plan.preparation_costs.items():
                self.costs[member] += seconds
        return self.costs[index]

    def validate_state(self, state):
        if state is not self.plan.state:
            raise ValueError("resource replay state differs from its audited storage plan")

    def move_tensor(self, action, arrays, timed):
        """Apply one already-audited, single-path CPU resource action."""
        from lqcdweave.contracts.correlator.api import ResourceTransition

        if self.temporary_scope is None:
            raise NotImplementedError("temporary array moves require an execution-scoped scratch store")
        transition = action.resource
        if not isinstance(transition, ResourceTransition):
            raise ValueError("resource action has no audited transition")
        if action.kind == "spill_disk":
            if transition.domain != "tensor" or transition.source != "host" or transition.target != "disk":
                raise NotImplementedError("only owned CPU tensor spills are supported")
            if action.tensor not in arrays or action.tensor in self._disk_receipts:
                raise ValueError("spill requires one resident tensor with no existing scratch receipt")
            value = arrays[action.tensor]
            spec = self.plan.state.catalog.get(action.tensor)
            if (not isinstance(value, np.ndarray) or spec is None
                    or tuple(value.shape) != spec.shape or value.dtype != spec.dtype
                    or value.nbytes != transition.nbytes or not value.flags.owndata
                    or not (value.flags.c_contiguous or value.flags.f_contiguous)):
                raise ValueError("spill requires an owned contiguous array matching the audited tensor")
            aliases = [key for key, other in arrays.items() if key != action.tensor
                       and isinstance(other, np.ndarray) and np.shares_memory(value, other)]
            retained_aliases = [key for key, other in self.retained.items()
                                if isinstance(other, np.ndarray) and np.shares_memory(value, other)]
            if aliases or retained_aliases:
                raise ValueError("cannot spill an array with live aliases or retained owners")
            receipt = timed("spill_disk:" + action.tensor,
                            lambda: self.temporary_scope.write_array(value),
                            nbytes=value.nbytes)
            if receipt is None:
                raise TypeError("temporary array store returned no receipt")
            self._disk_receipts[action.tensor] = receipt
            del arrays[action.tensor]
            return

        if action.kind == "restore_host":
            if transition.domain != "tensor" or transition.source != "disk" or transition.target != "host":
                raise NotImplementedError("only CPU tensor restores are supported")
            if action.tensor in arrays or action.tensor not in self._disk_receipts:
                raise ValueError("restore requires one spilled tensor without a resident alias")
            spec = self.plan.state.catalog.get(action.tensor)
            if spec is None or transition.nbytes != prod(spec.shape) * spec.dtype.itemsize:
                raise ValueError("restore size disagrees with the audited tensor")
            receipt = self._disk_receipts[action.tensor]
            value = timed("restore_host:" + action.tensor,
                          lambda: self.temporary_scope.read_array(
                              receipt, shape=spec.shape, dtype=spec.dtype.str,
                          ), nbytes=transition.nbytes)
            if (not isinstance(value, np.ndarray) or tuple(value.shape) != spec.shape
                    or value.dtype != spec.dtype or not value.flags.owndata
                    or not (value.flags.c_contiguous or value.flags.f_contiguous)):
                raise ValueError("temporary restore did not return a matching owned contiguous array")
            self.temporary_scope.remove_array(receipt)
            del self._disk_receipts[action.tensor]
            arrays[action.tensor] = value
            return

        if action.kind == "discard":
            if transition.domain != "tensor" or transition.source != "host" or transition.target != "absent":
                raise NotImplementedError("only resident CPU tensor discards are supported")
            value = arrays.get(action.tensor)
            spec = self.plan.state.catalog.get(action.tensor)
            if (not isinstance(value, np.ndarray) or spec is None
                    or tuple(value.shape) != spec.shape or value.dtype != spec.dtype
                    or value.nbytes != transition.nbytes or not value.flags.owndata):
                raise ValueError("discard requires the complete resident array owner")
            aliases = [key for key, other in arrays.items() if key != action.tensor
                       and isinstance(other, np.ndarray) and np.shares_memory(value, other)]
            retained_aliases = [key for key, other in self.retained.items()
                                if isinstance(other, np.ndarray) and np.shares_memory(value, other)]
            if aliases or retained_aliases:
                raise ValueError("cannot discard an array with live aliases")
            del arrays[action.tensor]
            return
        raise NotImplementedError(f"resource action {action.kind!r} is not executable")

    def close(self):
        # Includes exceptions during a read/materialization/kernel.
        self._waiting_read = None
        cleanup_errors = []
        if self.owns_packed_value_cache:
            self.packed_value_cache.clear()
        if self.read_ahead is not None and self.owns_read_ahead:
            try:
                self.read_ahead.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if not self.owns_storage:
            if cleanup_errors:
                raise cleanup_errors[0]
            return
        try:
            if (self.live and debug_enabled()
                    and getattr(self.backend, "__name__", "") == "cupy"):
                try:
                    self._synchronize_device()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for key in tuple(self.live):
                try:
                    self._release(key)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        finally:
            for key in tuple(self._borrowed_sources):
                try:
                    self._release_borrowed_source(key)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            self.retained.clear()
            self.live.clear()
            self._disk_receipts.clear()
            if self.temporary_scope is not None:
                try:
                    self.temporary_scope.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            raise cleanup_errors[0]


def replay_scene_storage(
    plan: SceneStoragePlan,
    loader: SourceLoader,
    *,
    backend=np,
    measure=False,
    source_descriptions=None,
    io_overlap=True,
    execution_cache=None,
):
    """Replay fixed computations and releases with bounded early host reads.

    Model peaks are in plan. Measured device peaks in ReplayReport are CuPy
    pool reservations sampled at phase/action boundaries, not a hardware high
    water mark; CPU runs do not claim measured GPU results.
    ``io_overlap=True`` advances eligible host reads only within the spare
    audited host budget. ``measure=True`` retains synchronous calibration.
    """
    if execution_cache is not None:
        with execution_cache.scope(plan, backend):
            return replay_scene_storage(plan, loader, backend=backend, measure=measure,
                source_descriptions=source_descriptions, io_overlap=io_overlap)
    resource_actions = any(action.kind in ("spill_disk", "restore_host")
                           for action in plan.state.actions)
    validate_coordinate_batch_workspace(plan, backend)
    if not plan.feasible:
        raise MemoryError("the supplied storage plan is infeasible; select a feasible plan before replay")
    if plan.retained_before:
        raise ValueError('this plan requires the preceding retained batches')
    temporary_store = None
    temporary_scope = None
    temporary_file_peak_bytes = None
    if resource_actions:
        if backend is not np:
            raise NotImplementedError("temporary array resource replay requires the NumPy CPU backend")
        temporary_store = getattr(loader, "temporary_array_store", None)
        if temporary_store is None:
            raise NotImplementedError("temporary resource replay requires an explicit scratch-store capability")
        for name in ("open_scope", "storage_bytes"):
            if not callable(getattr(temporary_store, name, None)):
                raise TypeError(f"temporary_array_store.{name} must be callable")
        reserve = getattr(temporary_store, "host_reserve_bytes", 0)
        if type(reserve) is not int or reserve < 0:
            raise TypeError("temporary_array_store.host_reserve_bytes must be nonnegative integer bytes")
        plan = plan.with_reserves(reserved_host_bytes=plan.reserved_host_bytes + reserve)
        validate_executable_resources(plan, allow_temporary_array=True)
        scratch_file_bytes_by_tensor = {}
        scratch_file_bytes = 0
        scratch_file_peak_bytes = 0
        for action in plan.state.actions:
            if action.kind == "spill_disk":
                spec = plan.state.catalog[action.tensor]
                file_bytes = temporary_store.storage_bytes(spec.shape, spec.dtype.str)
                if type(file_bytes) is not int or file_bytes < action.resource.nbytes:
                    raise ValueError("temporary array store returned an invalid scratch-file byte bound")
                scratch_file_bytes_by_tensor[action.tensor] = file_bytes
                scratch_file_bytes += file_bytes
                scratch_file_peak_bytes = max(scratch_file_peak_bytes, scratch_file_bytes)
            elif action.kind == "restore_host":
                if action.tensor not in scratch_file_bytes_by_tensor:
                    raise ValueError("restore has no scratch-file size reservation")
                scratch_file_bytes -= scratch_file_bytes_by_tensor.pop(action.tensor)
        if scratch_file_bytes_by_tensor or scratch_file_bytes:
            raise ValueError("temporary resource plan leaves scratch-file bytes unaccounted")
        if scratch_file_peak_bytes > plan.disk_limit_bytes:
            raise MemoryError("NPY file headers and payload exceed the declared scratch-file byte cap")
        temporary_file_peak_bytes = scratch_file_peak_bytes
    else:
        _validate_static_resources(plan, backend)
    if not plan.feasible:
        raise MemoryError("the supplied storage plan is infeasible; select a feasible plan before replay")
    if plan.retained_before:
        raise ValueError('this plan requires the preceding retained batches')
    physical_loader = loader
    loader = _bind_loader_for_plan(loader, plan, backend)
    _validate_loader(loader, plan)
    primary_error = None
    execution = None
    try:
        if resource_actions:
            temporary_scope = temporary_store.open_scope()
            for name in ("write_array", "read_array", "remove_array", "close"):
                if not callable(getattr(temporary_scope, name, None)):
                    raise TypeError(f"temporary array scope.{name} must be callable")
        execution = _BlockExecution(plan, loader, backend, sources=source_descriptions,
                                    temporary_scope=temporary_scope,
                                    io_overlap=io_overlap, measure=measure,
                                    read_ahead_loader=physical_loader)
        temporary_scope = None  # the execution object now owns the scope
        return _replay_storage(
            plan, loader, backend, measure, execution,
            temporary_file_peak_bytes=temporary_file_peak_bytes,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if temporary_scope is not None:
            # Scope creation or execution construction can fail before ownership
            # transfers to _BlockExecution. The scope still belongs to this call.
            try:
                temporary_scope.close()
            except BaseException as cleanup_error:
                _preserve_or_raise(primary_error, cleanup_error, "temporary scope close")
        try:
            _close_loader(loader)
        except BaseException as cleanup_error:
            _preserve_or_raise(primary_error, cleanup_error, "loader close_all")


def _replay_storage(plan, loader, backend, measure, execution, *, temporary_file_peak_bytes=None):
    plan = execution.plan
    measurement_hook = None
    primary_error = None
    try:
        if not plan.feasible:
            raise MemoryError(f"storage plan exceeds memory limit: host={plan.peak_host_bytes}, "
                              f"device={plan.peak_device_bytes}")
        if getattr(backend, "__name__", "") != plan.state.problem.backend:
            raise ValueError("execution backend disagrees with storage plan")
        measurement_hook = getattr(loader, "set_measurement", None) if measure else None
        if measurement_hook is not None:
            if not callable(measurement_hook):
                raise TypeError("loader.set_measurement must be callable when present")
            measurement_hook(bool(measure))
        report = replay(plan.state, loader, backend=backend, measure=measure, source_execution=execution)
        if measure and isinstance(report, ReplayReport):
            profile = dict(report.execution_profile or {})
            profile.update(_storage_deviation(plan, report))
            if plan.kernel_batches:
                # Aggregate packing and contraction timings may contain mixed
                # dtypes. The ordinary single-kernel fitter is not applicable.
                profile.update(calibration_eligible=False,
                               calibration_reason="cartesian_batch_aggregate")
            if execution.execution_tables.scalar_groups:
                profile.update(calibration_eligible=False,
                               calibration_reason='scalar_reduction_group_aggregate')
            if execution.execution_tables.cartesian_groups:
                groups = execution.execution_tables.cartesian_groups
                profile.update(calibration_eligible=False,
                    calibration_reason='cartesian_reduction_group_aggregate',
                    cartesian_reduction_groups=len(groups),
                    cartesian_reduction_child_batches=sum(len(g.batch_action_indices) for g in groups),
                    cartesian_reduction_kernel_calls=2*len(groups))
            if temporary_file_peak_bytes is not None:
                profile["temporary_file_peak_bytes"] = temporary_file_peak_bytes
            report = replace(report, execution_profile=profile)
        return report
    except BaseException as exc:
        if isinstance(exc,MemoryError) and getattr(backend,'__name__','')=='cupy':
            from .allocation_failure import annotate_allocation_failure
            annotate_allocation_failure(exc,execution)
        primary_error = exc
        raise
    finally:
        cleanup_errors = []
        try:
            execution.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        finally:
            if measurement_hook is not None:
                try:
                    measurement_hook(False)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            _preserve_or_raise(primary_error, cleanup_errors[0], "storage replay cleanup")


def _streamed_execution_windows(plans, sources, loader, backend, *, lookahead, io_overlap, measure):
    """Hold a finite audited window and close its read queue before advancing."""
    source_iterator = None if sources is None else iter(sources)
    width = (lookahead if io_overlap and not measure
             and supports_batch_read_ahead(loader) else 1)
    while True:
        records = []
        for _ in range(width):
            try:
                context_plan = next(plans)
            except StopIteration:
                break
            plan = context_plan
            validate_coordinate_batch_workspace(plan, backend)
            validate_executable_resources(plan)
            if not plan.feasible:
                raise MemoryError('streamed batch exceeds memory limits')
            if plan.state.problem.backend != backend.__name__:
                raise ValueError('execution backend disagrees')
            sources = None
            if source_iterator is not None:
                try:
                    sources = next(source_iterator)
                except StopIteration as exc:
                    raise ValueError('one runtime source description is required per storage plan') from exc
            records.append((context_plan, plan, sources))
        if not records:
            if source_iterator is not None:
                try:
                    next(source_iterator)
                except StopIteration:
                    pass
                else:
                    raise ValueError('more runtime source descriptions than storage plans')
            return
        _, read_ahead, offsets, _ = prepare_batch_read_ahead(
            tuple(record[1] for record in records), loader, enabled=io_overlap, measure=measure,
        )
        try:
            yield records, read_ahead, offsets
        except GeneratorExit:
            primary_error = None
            raise
        except BaseException as exc:
            primary_error = exc
            raise
        else:
            primary_error = None
        finally:
            if read_ahead is not None:
                try:
                    read_ahead.close()
                except BaseException as exc:
                    _preserve_or_raise(primary_error, exc, 'streamed read-ahead cleanup')
        # Do not retain the preceding window while constructing the next one.
        records.clear()
        del context_plan, plan, sources, read_ahead, offsets


def replay_storage_batches(plans, loader: SourceLoader, consume, *, backend=np, measure=False, prejoined=True, streaming=False,
                           batch_context=None, source_descriptions=None, io_overlap=True, lookahead=2,
                           execution_cache=None):
    """Consume frozen paths and outputs with one shared raw lifetime.

    ``prejoined`` must be True. This service only executes an audited/saved
    complete schedule and never joins lifetimes or increases sparse workspace.

    With ``streaming=True, prejoined=True``, consume an incremental audited
    iterator instead. Each batch is checked before its IO; later failures may
    occur after earlier outputs were consumed. The caller owns recovery.
    A task-scoped prefetch loader admits at most ``lookahead`` complete plan
    descriptions at a time (default 2), using the existing bounded read queue.
    Other loaders and calibration measurements advance one plan at a time.
    ``source_descriptions`` may also be incremental in this mode.

    ``consume(index, report)`` must finish writing or copying results to storage
    before returning, and must not retain device arrays across batches. Its
    work is outside replay timing. The caller owns any output publication and
    recovery. No batch is executed if the joined memory accounting is infeasible.
    """
    if execution_cache is not None and not streaming:
        plans = tuple(plans)
        with execution_cache.scope(plans, backend):
            return replay_storage_batches(plans, loader, consume, backend=backend, measure=measure,
                prejoined=prejoined, batch_context=batch_context, source_descriptions=source_descriptions,
                io_overlap=io_overlap, lookahead=lookahead)
    if prejoined is not True:
        raise ValueError('execution requires prejoined=True; use contraction.replay for convenience preparation')
    if type(lookahead) is not int or lookahead < 1:
        raise ValueError('lookahead must be a positive integer')
    if type(io_overlap) is not bool:
        raise TypeError('io_overlap must be bool')
    plans = iter(plans) if streaming else tuple(plans)
    first_stream_plan = None
    if streaming:
        plan_iterator = iter(plans)
        try:
            first_stream_plan = next(plan_iterator)
        except StopIteration:
            plans = iter(())
        else:
            plans = _prepend_plan(first_stream_plan, plan_iterator)
    context_plans = None if streaming else tuple(plans)
    if not streaming:
        for plan in plans:
            validate_coordinate_batch_workspace(plan, backend)
    sources_by_batch = (source_descriptions if streaming else
                       None if source_descriptions is None else tuple(source_descriptions))
    if sources_by_batch is not None and not streaming and len(sources_by_batch) != len(plans):
        raise ValueError("one runtime source description is required per storage plan")
    if not streaming:
        for plan in plans:
            _validate_static_resources(plan, backend)
    if not streaming and any(not plan.feasible for plan in plans):
        raise MemoryError("cross-batch raw lifetimes exceed memory limits; replan")
    if not streaming and any(plan.state.problem.backend != backend.__name__ for plan in plans):
        raise ValueError("execution backend disagrees with batch plans")
    physical_loader = loader
    if streaming and first_stream_plan is not None:
        loader = _bind_loader_for_plan(loader, first_stream_plan, backend)
        _validate_loader(loader, first_stream_plan)
        first_stream_plan = None
    elif not streaming and plans:
        loader = _bind_batch_loader(loader, plans, backend)
        _validate_loader(loader, plans)
    else:
        _validate_loader(loader)
    read_ahead_loader = _read_ahead_loader(physical_loader, loader)
    task_read_ahead = None
    action_offsets = ()
    if not streaming:
        plans, task_read_ahead, action_offsets, _ = prepare_batch_read_ahead(
            plans, read_ahead_loader, enabled=io_overlap, measure=measure,
        )
    if streaming:
        windows = _streamed_execution_windows(
            plans, sources_by_batch, read_ahead_loader, backend,
            lookahead=lookahead, io_overlap=io_overlap, measure=measure,
        )
    else:
        records = tuple((context_plans[i], plan,
                         None if sources_by_batch is None else sources_by_batch[i])
                        for i, plan in enumerate(plans))
        windows = ((records, task_read_ahead, action_offsets),)
    shared = ({}, set())
    retained = {}
    try:
        index = 0
        for window, window_read_ahead, offsets in windows:
            for local_index, (context_plan, plan, sources) in enumerate(window):
                if debug_enabled():
                    validate_executable_resources(plan)
                if not plan.feasible: raise MemoryError('streamed batch exceeds memory limits')
                if plan.state.problem.backend!=backend.__name__: raise ValueError('execution backend disagrees')
                # These are explicit boundary evictions recorded by the planner.
                for key in plan.retained_releases:
                    if key not in retained: raise RuntimeError('planned cache eviction has no resident tensor')
                    del retained[key]
                if set(retained) != set(plan.retained_before):
                    raise RuntimeError('retained tensor schedule is incomplete')
                execution = _BlockExecution(plan, loader, backend, shared, retained, sources=sources,
                                            io_overlap=io_overlap, measure=measure,
                                            task_read_ahead=window_read_ahead,
                                            action_offset=0 if window_read_ahead is None else offsets[local_index],
                                            read_ahead_loader=physical_loader)
                # Activation consumes the same immutable descriptor delivered by P.
                with nullcontext() if batch_context is None else batch_context(context_plan):
                    report = _replay_storage(plan, loader, backend, measure, execution)
                    consume(index, report)
                if debug_enabled() and set(retained) != set(plan.retained_after):
                    raise RuntimeError('retained tensor schedule does not match the audited batch plan')
                index += 1
                del report
            if window:
                del context_plan, plan, sources, execution
            del window, window_read_ahead, offsets
    except BaseException as exc:
        primary_error = exc
        raise
    else:
        primary_error = None
    finally:
        cleanup_errors = []
        if streaming:
            try:
                windows.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if task_read_ahead is not None:
            try:
                task_read_ahead.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        close = getattr(loader, "close_block", None)
        try:
            if (shared[0] and debug_enabled()
                    and getattr(backend, "__name__", "") == "cupy"):
                try:
                    backend.cuda.get_current_stream().synchronize()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for key, value in tuple(shared[0].items()):
                if close is not None:
                    try:
                        close(key, value)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
        finally:
            retained.clear()
            shared[0].clear()
            try:
                _close_loader(loader)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            _preserve_or_raise(primary_error, cleanup_errors[0], "batch storage cleanup")


def _close_loader(loader):
    close_all = getattr(loader, "close_all", None)
    if close_all is not None:
        if not callable(close_all):
            raise TypeError("loader.close_all must be callable when present")
        close_all()


def _validate_loader(loader, plans=None):
    for name in ("read_block",):
        if not callable(getattr(loader, name, None)):
            raise TypeError(f"loader.{name} must be callable")
    if plans is None:
        plan_items = ()
    elif isinstance(plans, SceneStoragePlan):
        plan_items = (plans,)
    else:
        plan_items = tuple(plans)
    preparations = dict(getattr(loader, "preparations", {}) or {})
    for plan in plan_items:
        preparations.update(plan.preparations)
    needs_materializer = any(preparation.recipe is None
                             for preparation in preparations.values())
    physical = _physical_loader(loader)
    if needs_materializer and not callable(getattr(physical, "materialize", None)):
        raise TypeError("opaque source preparations require loader.materialize")
    if not callable(getattr(loader, "materialize", None)) and needs_materializer:
        raise TypeError("opaque source preparations require loader.materialize")
    for name in ("close_block", "close_all", "set_measurement", "drain_io_events"):
        method = getattr(loader, name, None)
        if method is not None and not callable(method):
            raise TypeError(f"loader.{name} must be callable when present")


def _storage_deviation(plan, report):
    host_delta = (None if report.peak_host_rss_bytes is None
                  else report.peak_host_rss_bytes - plan.peak_host_bytes)
    device_delta = (None if report.peak_device_bytes is None
                    else report.peak_device_bytes - plan.peak_device_bytes)
    exceeded = any(value is not None and value > 0 for value in (host_delta, device_delta))
    return {
        "planned_peak_host_bytes": plan.peak_host_bytes,
        "planned_peak_device_bytes": plan.peak_device_bytes,
        "observed_minus_planned_host_bytes": host_delta,
        "observed_minus_planned_device_bytes": device_delta,
        "observed_storage_budget_exceeded": exceeded,
    }


def _preserve_or_raise(primary_error, cleanup_error, context):
    if primary_error is None:
        raise cleanup_error
    add_note = getattr(primary_error, "add_note", None)
    if add_note is not None:
        add_note(f"{context} also failed: {cleanup_error!r}")
