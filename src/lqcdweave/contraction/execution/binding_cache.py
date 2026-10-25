"""Owner-scoped structural bindings; runtime arrays never enter this cache."""
from contextlib import contextmanager
from contextvars import ContextVar

from .diagnostics import debug_enabled


_active_bindings = ContextVar('contraction_bindings', default=None)
_factorized_gram_tiling = ContextVar('factorized_gram_tiling', default=False)


def factorized_gram_tiling_enabled():
    return _factorized_gram_tiling.get()


def binding_cache_active():
    return not debug_enabled() and _active_bindings.get() is not None


class ExecutionBindingCache:
    """Keep one selected plan/backend's bindings for the lifetime of its owner.

    Publication contains complete bindings only. Concurrent first calls may
    duplicate binding work; every replay keeps its own numerical state.
    """
    def __init__(self, *, factorized_gram_tiling=False):
        if type(factorized_gram_tiling) is not bool:
            raise TypeError('factorized_gram_tiling must be bool')
        self.factorized_gram_tiling = factorized_gram_tiling
        self._binding = None

    @contextmanager
    def factory_scope(self):
        """Apply this owner's execution choice only while binding its code."""
        token = _factorized_gram_tiling.set(self.factorized_gram_tiling)
        try:
            yield
        finally:
            _factorized_gram_tiling.reset(token)

    def _lookup(self, plan, backend):
        previous = None if debug_enabled() else self._binding
        if previous is not None and previous[1] is backend and (
                previous[0] is plan or (type(plan) is tuple and type(previous[0]) is tuple
                    and len(plan) == len(previous[0])
                    and all(a is b for a, b in zip(plan, previous[0])))):
            return previous[2:]
        return None

    def _publish(self, plan, backend, *values):
        self._binding = (plan, backend, *values)

    @contextmanager
    def scope(self, plan, backend):
        previous = self._lookup(plan, backend)
        entries = {} if previous is None else previous[0]
        token = _active_bindings.set(entries if not debug_enabled() else None)
        completed = False
        try:
            with self.factory_scope():
                yield
            completed = True
        finally:
            _active_bindings.reset(token)
            # Rejected extension descriptions can contain runtime owners.
            # Keep no root at all unless an immutable binding was admitted.
            if completed and previous is None and entries and not debug_enabled():
                self._publish(plan, backend, entries)


def bind_cached(namespace, description, backend, build, reusable, *, slot=None, signature=None):
    """Bind a supported immutable descriptor once in the current owner scope."""
    entries = None if debug_enabled() else _active_bindings.get()
    if entries is None:
        return build()
    key = (namespace, id(description) if slot is None else slot, id(backend))
    previous = entries.get(key)
    if (previous is not None and previous[0] is description and previous[1] is backend
            and previous[3] == signature):
        return previous[2]
    value = build()
    if reusable():
        entries[key] = (description, backend, value, signature)
    return value


def native_state_binding(state):
    """Admit native frozen descriptions once, on a binding miss only.

    Extension protocols and mutable Action fields keep the original rebind
    path. Storage blocks/preparations are deliberately not cached: each run
    reads their current descriptions through its new execution context.
    """
    from lqcdweave.contraction.planning.api import ContractionState
    from lqcdweave.contracts.correlator.api import (
        Action, ContractionProblem, MatrixKernelSpec, FactorizedWideReduction, MesonWholeKernel, TensorSpec, SourceSpec,
        SparseBridgeSpec, BaryonJoinSpec,
    )
    from lqcdweave.contracts.correlator.cores import TensorCoreSpec, CoreComponent
    def native_core(core):
        return (core is None or type(core) in (MatrixKernelSpec, FactorizedWideReduction, MesonWholeKernel)
            or (type(core) is TensorCoreSpec and type(core.spec) is TensorSpec
                and all(type(component) is CoreComponent for component in core.components)))
    return (type(state) is ContractionState and type(state.problem) is ContractionProblem
        and all(type(spec) is TensorSpec for spec in state.catalog.values())
        and all(type(source) is SourceSpec for source in state.problem.sources.values())
        and all(type(bridge) is SparseBridgeSpec for bridge in state.problem.sparse_bridges)
        and all(type(join) is BaryonJoinSpec and native_core(join.core) for join in state.problem.baryon_joins)
        and all(type(action) is Action
                and native_core(action.core)
                and type(action.operands) is tuple and type(action.positions) is tuple
                and type(action.left_axes) is tuple and type(action.right_axes) is tuple
                and type(action.kernel_path) is tuple
                and all(type(step) is tuple for step in action.kernel_path)
                for action in state.actions))


def native_storage_binding(plan):
    from lqcdweave.contracts.correlator.api import SceneStoragePlan, StorageEvent, ScalarReductionGroup, CartesianReductionGroup
    from .scene_runtime import _native_batch_description
    return (type(plan) is SceneStoragePlan and type(plan.events) is tuple
        and all(type(event) is StorageEvent for event in plan.events)
        and type(plan.retained_after) is tuple and type(plan.kernel_batches) is tuple
        and all(_native_batch_description(batch) for batch in plan.kernel_batches)
        and type(plan.scalar_reduction_groups) is tuple
        and all(type(group) is ScalarReductionGroup and type(group.action_indices) is tuple
                and all(type(index) is int for index in group.action_indices)
                and type(group.workspace_bytes) is int and type(group.seconds) in (int, float)
                for group in plan.scalar_reduction_groups)
        and type(plan.cartesian_reduction_groups) is tuple
        and all(type(group) is CartesianReductionGroup and type(group.batch_action_indices) is tuple
                and all(type(child) is tuple and all(type(index) is int for index in child)
                        for child in group.batch_action_indices)
                and type(group.workspace_bytes) is int and type(group.seconds) in (int,float)
                for group in plan.cartesian_reduction_groups)
        and native_state_binding(plan.state))


def native_resource_audit_binding(plan):
    """Prove audit inputs immutable, once on an owner cache miss.

    The weaker execution-table proof deliberately omits memory trajectories.
    This proof covers every field read by the ordinary C resource audit, but
    not mutable loader/preparation bindings: those are checked anew by replay.
    Explicit resource moves and core/extension descriptions retain full audit.
    No current array, reader, device state or numerical result is captured.
    """
    from dataclasses import fields
    from lqcdweave.contraction.planning.api import has_native_catalog
    from lqcdweave.contracts.correlator.api import (
        Action, AxisPattern, MemoryPoint, StorageEvent, DenseKernelBatch,
        DenseBatchOperand, DenseBatchResult, DenseGemmGroup,
        ScalarReductionGroup, CartesianReductionGroup,
    )

    if (not native_storage_binding(plan) or plan.explicit_resources
            or not has_native_catalog(plan.state)):
        return False
    from lqcdweave.contracts.correlator.meson_core import MesonWholeKernel
    from lqcdweave.contracts.correlator.vertex_kernel import VertexKernelDescription
    records = frozenset((MesonWholeKernel, VertexKernelDescription, Action, AxisPattern, MemoryPoint, StorageEvent,
        DenseKernelBatch, DenseBatchOperand, DenseBatchResult, DenseGemmGroup,
        ScalarReductionGroup, CartesianReductionGroup))

    def immutable(value):
        kind = type(value)
        if kind in (str, int, float, bool, complex, type(None)):
            return True
        if kind is tuple:
            return all(immutable(item) for item in value)
        if kind in records:
            return all(immutable(getattr(value, field.name)) for field in fields(kind))
        return False

    # TensorSpec normalizes shape/dtype, and the native catalog owns its map.
    # Semantic descriptions are not read by the resource validators.
    if any(type(key) is not str for key in plan.state.catalog):
        return False
    return immutable((plan.state.problem.backend, plan.state.actions,
        plan.events, plan.memory, plan.kernel_batches,
        plan.scalar_reduction_groups, plan.cartesian_reduction_groups,
        plan.retained_before, plan.retained_after, plan.retained_releases,
        plan.retention_planned, plan.explicit_resources, plan.feasible,
        plan.peak_host_bytes, plan.peak_device_bytes, plan.peak_disk_bytes,
        plan.memory_limit, plan.host_memory_limit, plan.disk_limit_bytes))
