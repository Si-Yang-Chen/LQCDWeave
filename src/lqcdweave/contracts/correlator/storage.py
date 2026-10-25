"""Pure physical storage and executable memory-plan descriptors."""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import prod
from typing import Mapping
from .scene import Action
from .dense_batch import DenseKernelBatch
from .gemm_group import DenseGemmGroup
from .scalar_reduction_group import ScalarReductionGroup
from .cartesian_reduction_group import CartesianReductionGroup
from .preparation import TensorPreparation
from .protocols import CostModel as CostProfile

@dataclass(frozen=True)
class StorageBlock:
    """One read, with separate persistent and temporary allocation bounds.

    ``device_bytes`` is zero for CPU blocks; CPU storage uses ``host_bytes``.
    ``selection`` is an explicit adapter-owned, immutable selection identity.
    """

    dataset: str
    selection: str
    read_bytes: int
    host_bytes: int = 0
    device_bytes: int = 0
    transfer_bytes: int = 0
    staging_host_bytes: int = 0
    staging_device_bytes: int = 0
    read_kind: str = "dense"

    def __post_init__(self):
        if not self.dataset or not self.selection:
            raise ValueError("blocks require dataset and selection identities")
        if self.read_kind not in ("dense", "p2p"):
            raise ValueError("unknown block read kind")
        for name in ("read_bytes", "host_bytes", "device_bytes", "transfer_bytes",
                     "staging_host_bytes", "staging_device_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

@dataclass(frozen=True)
class SourcePreparation:
    """Logical source dependencies; materialization itself performs no I/O.

    Dense results are snapshotted by replay. Sparse results may borrow from
    these blocks, so their blocks survive until the logical source is released.
    Workspace excludes the final owned snapshot already counted by scene_state;
    it MUST include temporary materialization results when they do not alias a
    raw block. Transformation work is explicitly billed through CostProfile.
    """

    blocks: tuple[str, ...]
    workspace_host_bytes: int = 0
    workspace_device_bytes: int = 0
    flops: int = 0
    traffic_bytes: int = 0
    kernel_calls: int = 0
    dtype: str = "complex128"
    recipe: TensorPreparation | None = None

    def __post_init__(self):
        if self.recipe is not None and not isinstance(self.recipe, TensorPreparation):
            raise TypeError("recipe must be a TensorPreparation or None")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if len(set(self.blocks)) != len(self.blocks):
            raise ValueError("duplicate source block dependency")
        for name in ("workspace_host_bytes", "workspace_device_bytes", "flops", "traffic_bytes", "kernel_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def seconds(self, profile: CostProfile) -> float:
        return profile.preparation_seconds(self.flops, self.traffic_bytes,
                                        kernel_calls=self.kernel_calls, dtype=self.dtype)


@dataclass(frozen=True)
class StorageEvent:
    action_index: int
    kind: str
    block: str


@dataclass(frozen=True)
class MemoryPoint:
    action_index: int
    phase: str
    raw_host_bytes: int
    raw_device_bytes: int
    logical_host_bytes: int
    logical_device_bytes: int
    workspace_host_bytes: int
    workspace_device_bytes: int
    reserved_host_bytes: int = 0
    reserved_device_bytes: int = 0
    retained_host_bytes: int = 0
    retained_device_bytes: int = 0

    temporary_disk_bytes: int = 0
    packed_value_cache_device_bytes: int = 0
    # Known per-owner CuPy request granularity. Payload, caller reserves and
    # allocator fragmentation remain separate concepts.
    allocation_rounding_device_bytes: int = 0
    scalar_reduction_device_bytes: int = 0
    cartesian_reduction_device_bytes: int = 0

    @property
    def host_bytes(self):
        return self.raw_host_bytes + self.logical_host_bytes + self.workspace_host_bytes + self.reserved_host_bytes + self.retained_host_bytes

    @property
    def device_bytes(self):
        return (self.raw_device_bytes + self.logical_device_bytes + self.workspace_device_bytes
                + self.reserved_device_bytes + self.retained_device_bytes
                + self.packed_value_cache_device_bytes + self.allocation_rounding_device_bytes
                + self.scalar_reduction_device_bytes + self.cartesian_reduction_device_bytes)


@dataclass(frozen=True)
class SceneStoragePlan:
    state: object
    blocks: Mapping[str, StorageBlock]
    preparations: Mapping[str, SourcePreparation]
    events: tuple[StorageEvent, ...]
    memory: tuple[MemoryPoint, ...]
    preparation_seconds: float
    preparation_costs: Mapping[int, float]
    peak_host_bytes: int
    peak_device_bytes: int
    feasible: bool
    memory_limit: int
    host_memory_limit: int | None
    reserved_host_bytes: int = 0
    reserved_device_bytes: int = 0
    cost_profile: CostProfile | None = None
    retained_before: tuple[str, ...] = ()
    retained_after: tuple[str, ...] = ()
    retention_planned: bool = False
    retained_releases: tuple[str, ...] = ()

    peak_disk_bytes: int = 0
    disk_limit_bytes: int | None = None
    explicit_resources: bool = False
    kernel_batches: tuple[DenseKernelBatch | DenseGemmGroup, ...] = ()
    packed_value_cache_device_bytes: int = 0
    scalar_reduction_groups: tuple[ScalarReductionGroup, ...] = ()
    cartesian_reduction_groups: tuple[CartesianReductionGroup, ...] = ()

    @property
    def kernel_cost_adjustment(self):
        adjustment = sum(batch.seconds - sum(self.state.actions[i].seconds for i in batch.action_indices)
                         for batch in (*self.kernel_batches, *self.scalar_reduction_groups))
        if self.cartesian_reduction_groups:
            costs = {batch.action_indices: batch.seconds for batch in self.kernel_batches}
            adjustment += sum(group.seconds - sum(costs[indices] for indices in group.batch_action_indices)
                              for group in self.cartesian_reduction_groups)
        return adjustment

    @property
    def compute_seconds(self):
        return sum(a.seconds for a in self.state.actions
                   if a.kind in ("contract", "sparse_bridge", "baryon_join")) + self.kernel_cost_adjustment

    @property
    def seconds(self):
        return self.state.seconds + self.preparation_seconds + self.kernel_cost_adjustment

    def with_reserves(self, *, reserved_host_bytes=None, reserved_device_bytes=None):
        """Reaudit this schedule with caller-declared host/device reserves.

        Each reserve is added uniformly to every memory-trajectory point, so
        the returned peaks and feasibility include it without changing actions,
        physical lifetimes, or cost identity. Callers may use this for known
        external arrays or caller-owned opaque workspace. It is a planning
        declaration, not a measurement or proof of process RSS/device usage.
        An increased reserve may make the schedule infeasible; callers must
        reject it or search again before performing any tensor IO.
        """
        host = self.reserved_host_bytes if reserved_host_bytes is None else reserved_host_bytes
        device = self.reserved_device_bytes if reserved_device_bytes is None else reserved_device_bytes
        if any(type(value) is not int or value < 0 for value in (host, device)):
            raise ValueError("reserved memory must be nonnegative integer bytes")
        if self.state.problem.backend == "numpy" and device:
            raise ValueError("CPU plans cannot reserve device memory")
        if host == self.reserved_host_bytes and device == self.reserved_device_bytes:
            return self
        packed = min(self.packed_value_cache_device_bytes,
                     max(0, self.memory_limit - (self.peak_device_bytes
                         - self.packed_value_cache_device_bytes
                         + device - self.reserved_device_bytes)))
        memory = tuple(replace(point, reserved_host_bytes=host, reserved_device_bytes=device,
                               packed_value_cache_device_bytes=packed)
                       for point in self.memory)
        peak_host = max(point.host_bytes for point in memory)
        peak_device = max(point.device_bytes for point in memory)
        primary = peak_device if self.state.problem.backend == "cupy" else peak_host
        return replace(self, memory=memory, reserved_host_bytes=host, reserved_device_bytes=device,
                       packed_value_cache_device_bytes=packed,
                       peak_host_bytes=peak_host, peak_device_bytes=peak_device,
                       feasible=primary <= self.memory_limit and
                       (self.host_memory_limit is None or peak_host <= self.host_memory_limit) and
                       (self.disk_limit_bytes is None or self.peak_disk_bytes <= self.disk_limit_bytes))



RESOURCE_ACTION_KINDS = frozenset(("discard", "spill_host", "spill_disk", "restore_host", "restore_device", "read_block", "close_block"))
COMPUTE_ACTION_KINDS = frozenset(("load", "release", "contract", "sparse_bridge", "baryon_join"))


def validate_executable_resources(value, *, allow_temporary_array=False):
    """Reject unsupported resource trajectories before execution or archive IO.

    The default remains metadata-only. Callers may explicitly opt into the
    currently supported CPU tensor spill/restore descriptor; that descriptor
    still does not imply a runtime scratch-store capability is available.
    """
    if type(allow_temporary_array) is not bool:
        raise TypeError("allow_temporary_array must be bool")
    state = getattr(value, "state", value)
    if getattr(value, "kernel_batches", ()):
        from .dense_batch import validate_dense_batches
        validate_dense_batches(value)
    if (getattr(value, "scalar_reduction_groups", ())
            or any(getattr(point, 'scalar_reduction_device_bytes', 0) for point in getattr(value, 'memory', ()))):
        from .scalar_reduction_group import validate_scalar_reduction_groups
        validate_scalar_reduction_groups(value)
    if (getattr(value, "cartesian_reduction_groups", ())
            or any(getattr(point, 'cartesian_reduction_device_bytes', 0) for point in getattr(value, 'memory', ()))):
        from .cartesian_reduction_group import validate_cartesian_reduction_groups
        validate_cartesian_reduction_groups(value)
    actions = tuple(state.actions)
    if allow_temporary_array and any(
            action.kind in ("spill_disk", "restore_host") for action in actions):
        _validate_temporary_array_plan(value, state, actions)
        return
    for action in actions:
        if action.kind in RESOURCE_ACTION_KINDS or action.resource is not None:
            raise NotImplementedError("resource trajectories currently support metadata audit only")
        if action.kind not in COMPUTE_ACTION_KINDS:
            raise ValueError(f"unknown action {action.kind!r}")
    if getattr(value, "explicit_resources", False):
        raise NotImplementedError("explicit resource schedules currently support metadata audit only")
    for event in getattr(value, "events", ()):
        if event.kind not in ("read", "release"):
            raise ValueError(f"unknown storage event {event.kind!r}")


def _validate_temporary_array_plan(value, state, actions):
    """Validate the one format/backend-neutral resource subpath X can replay."""
    if not isinstance(value, SceneStoragePlan):
        raise NotImplementedError("temporary array moves require a complete SceneStoragePlan")
    if (not value.explicit_resources or not getattr(state, "done", False)
            or state.problem.backend != "numpy"):
        raise NotImplementedError("only complete explicit CPU resource plans are supported")
    if value.cost_profile is None:
        raise ValueError("temporary resource plans require their audited cost profile")
    if type(value.disk_limit_bytes) is not int or value.disk_limit_bytes <= 0:
        raise NotImplementedError("temporary resource execution requires an explicit positive disk cap")
    if value.peak_disk_bytes <= 0:
        raise ValueError("temporary resource plans require a positive payload disk peak")
    if value.peak_disk_bytes > value.disk_limit_bytes:
        raise MemoryError("temporary resource payload exceeds the declared disk cap")
    if not value.feasible:
        raise MemoryError("temporary resource plan is infeasible")
    if value.retained_before or value.retained_after or value.retained_releases:
        raise NotImplementedError("temporary array moves do not support retained batch state")
    if any(block.device_bytes or block.staging_device_bytes for block in value.blocks.values()):
        raise ValueError("CPU temporary resource plans cannot contain device blocks")
    if any(preparation.workspace_device_bytes for preparation in value.preparations.values()):
        raise ValueError("CPU temporary resource plans cannot contain device preparation workspace")

    disk_owners = {}
    supported = False
    for action in actions:
        kind = action.kind
        transition = action.resource
        if kind in ("read_block", "close_block"):
            if transition is not None or action != Action(kind, action.tensor, seconds=action.seconds):
                raise ValueError("raw lifecycle actions must be plain block events")
            if action.tensor not in value.blocks:
                raise ValueError("raw lifecycle action references an unknown block")
            block = value.blocks[action.tensor]
            if action.seconds != 0.0:
                raise ValueError("raw lifecycle actions have no predicted time cost")
            continue
        if kind == "spill_disk":
            if transition is None or (transition.domain, transition.source, transition.target) != (
                    "tensor", "host", "disk"):
                raise NotImplementedError("only owned CPU tensor host-to-disk moves are supported")
            spec = state.catalog.get(action.tensor)
            if spec is None:
                raise NotImplementedError("borrowed or undescribed tensors cannot be spilled")
            nbytes = prod(spec.shape) * spec.dtype.itemsize
            if nbytes <= 0 or transition.nbytes != nbytes or action.tensor in disk_owners:
                raise ValueError("spill size or owner state disagrees with the tensor description")
            if action != Action(kind, action.tensor, seconds=action.seconds, resource=transition):
                raise ValueError("spill actions cannot contain compute fields")
            if action.seconds != 0.0:
                raise ValueError("spill actions have no predicted time cost")
            disk_owners[action.tensor] = nbytes
            supported = True
            continue
        if kind == "restore_host":
            if transition is None or (transition.domain, transition.source, transition.target) != (
                    "tensor", "disk", "host"):
                raise NotImplementedError("only CPU tensor disk-to-host restores are supported")
            nbytes = disk_owners.get(action.tensor)
            if nbytes is None or transition.nbytes != nbytes:
                raise ValueError("restore does not match a preceding spill owner")
            if action != Action(kind, action.tensor, seconds=action.seconds, resource=transition):
                raise ValueError("restore actions cannot contain compute fields")
            if action.seconds != 0.0:
                raise ValueError("restore actions have no predicted time cost")
            del disk_owners[action.tensor]
            continue
        if kind == "discard":
            if transition is None or (transition.domain, transition.source, transition.target) != (
                    "tensor", "host", "absent"):
                raise NotImplementedError("only host-owned tensor discards are supported")
            spec = state.catalog.get(action.tensor)
            if (spec is None or action.tensor in disk_owners
                    or transition.nbytes != prod(spec.shape) * spec.dtype.itemsize):
                raise ValueError("discard requires the complete resident tensor allocation")
            if action != Action(kind, action.tensor, seconds=action.seconds, resource=transition):
                raise ValueError("discard actions cannot contain compute fields")
            if action.seconds != 0.0:
                raise ValueError("discard actions have no predicted time cost")
            continue
        if kind in COMPUTE_ACTION_KINDS:
            if transition is not None:
                raise ValueError("compute actions cannot carry a resource move")
            if action.tensor in disk_owners or any(key in disk_owners for key in action.operands):
                raise ValueError("a spilled tensor must be restored before it is used")
            continue
        raise NotImplementedError(f"resource action {kind!r} is not executable")
    if not supported or disk_owners:
        raise ValueError("every spill must have a matching restore in the same single path")

    points = tuple(value.memory)
    if (len(points) != len(actions) + 1
            or any(point.action_index != index for index, point in enumerate(points))):
        raise ValueError("resource memory points do not cover every action and final output phase")
    disk = 0
    for index, action in enumerate(actions):
        if action.kind == "spill_disk":
            disk += action.resource.nbytes
        expected_disk = disk
        if points[index].temporary_disk_bytes != expected_disk:
            raise ValueError("resource memory point disagrees with disk owner transitions")
        if action.kind == "restore_host":
            disk -= action.resource.nbytes
    if points[-1].temporary_disk_bytes != disk or disk != 0:
        raise ValueError("resource plan leaves a disk allocation unaccounted")
    expected_events = tuple(
        StorageEvent(index, "read" if action.kind == "read_block" else "release",
                     action.tensor)
        for index, action in enumerate(actions) if action.kind in ("read_block", "close_block")
    )
    if tuple(value.events) != expected_events:
        raise ValueError("raw lifecycle events disagree with explicit resource actions")

    # Check the recorded allocation axes, not only the plan's peak fields. A
    # restore point includes the new owned host value while the disk source is
    # still live; spill/discard points include their source owner until the
    # action completes. The P transition audit uses exactly these boundaries.
    host_tensors = {}
    host_blocks = {}
    disk_by_tensor = {}

    def compare_point(index):
        point = points[index]
        if (point.logical_host_bytes != sum(host_tensors.values())
                or point.raw_host_bytes != sum(host_blocks.values())
                or point.temporary_disk_bytes != sum(disk_by_tensor.values())):
            raise ValueError("resource memory point disagrees with tensor/block owner lifetimes")
        if (point.reserved_host_bytes != value.reserved_host_bytes
                or point.reserved_device_bytes != value.reserved_device_bytes
                or point.retained_host_bytes or point.retained_device_bytes
                or point.device_bytes):
            raise ValueError("CPU resource memory point has inconsistent reserve or device ownership")

    for index, action in enumerate(actions):
        kind = action.kind
        if kind == "read_block":
            host_blocks[action.tensor] = value.blocks[action.tensor].host_bytes
            if points[index].workspace_host_bytes != value.blocks[action.tensor].staging_host_bytes:
                raise ValueError("raw read staging workspace disagrees with its block declaration")
        elif kind == "close_block":
            if action.tensor not in host_blocks:
                raise ValueError("raw block closes without a live host owner")
            del host_blocks[action.tensor]
            if points[index].workspace_host_bytes:
                raise ValueError("raw close cannot declare a staging workspace")
        elif kind == "spill_disk":
            if action.tensor not in host_tensors:
                raise ValueError("spill has no live owned host tensor")
            if points[index].workspace_host_bytes != action.resource.workspace_host_bytes:
                raise ValueError("spill workspace disagrees with its resource transition")
            disk_by_tensor[action.tensor] = action.resource.nbytes
            compare_point(index)
            del host_tensors[action.tensor]
            continue
        elif kind == "restore_host":
            if action.tensor not in disk_by_tensor:
                raise ValueError("restore has no live disk tensor owner")
            if points[index].workspace_host_bytes != action.resource.workspace_host_bytes:
                raise ValueError("restore workspace disagrees with its resource transition")
            host_tensors[action.tensor] = action.resource.nbytes
            compare_point(index)
            del disk_by_tensor[action.tensor]
            continue
        elif kind == "discard":
            if action.tensor not in host_tensors:
                raise ValueError("discard has no live owned host tensor")
            if points[index].workspace_host_bytes != action.resource.workspace_host_bytes:
                raise ValueError("discard workspace disagrees with its resource transition")
            compare_point(index)
            del host_tensors[action.tensor]
            continue
        elif kind == "load":
            spec = state.catalog.get(action.tensor)
            if spec is not None:
                host_tensors[action.tensor] = prod(spec.shape) * spec.dtype.itemsize
            preparation = value.preparations[action.tensor]
            if points[index].workspace_host_bytes != preparation.workspace_host_bytes:
                raise ValueError("load workspace disagrees with its source preparation")
        elif kind == "release":
            host_tensors.pop(action.tensor, None)
            if points[index].workspace_host_bytes:
                raise ValueError("logical release cannot declare a staging workspace")
        elif kind in COMPUTE_ACTION_KINDS:
            spec = state.catalog.get(action.tensor)
            if spec is not None and not action.reused:
                host_tensors[action.tensor] = prod(spec.shape) * spec.dtype.itemsize
        if kind in ("spill_disk", "restore_host"):
            if points[index].workspace_host_bytes != action.resource.workspace_host_bytes:
                raise ValueError("temporary move workspace disagrees with its resource transition")
        compare_point(index)
    if disk_by_tensor:
        raise ValueError("resource owner replay leaves disk tensors resident")
    if set(host_tensors) != set(state.loaded).intersection(state.catalog):
        raise ValueError("resource owner replay disagrees with the final logical loaded set")
    final_point = points[-1]
    if (final_point.logical_host_bytes != sum(host_tensors.values())
            or final_point.raw_host_bytes != sum(host_blocks.values())
            or final_point.temporary_disk_bytes != 0):
        raise ValueError("final resource memory point disagrees with live owners")
    if (final_point.reserved_host_bytes != value.reserved_host_bytes
            or final_point.reserved_device_bytes != value.reserved_device_bytes
            or final_point.retained_host_bytes or final_point.retained_device_bytes
            or final_point.device_bytes or final_point.workspace_device_bytes):
        raise ValueError("CPU final memory point has inconsistent reserves or device ownership")
    peak_host = max(point.host_bytes for point in points)
    peak_device = max(point.device_bytes for point in points)
    peak_disk = max(point.temporary_disk_bytes for point in points)
    if (value.peak_host_bytes != peak_host or value.peak_device_bytes != peak_device
            or value.peak_disk_bytes != peak_disk):
        raise ValueError("resource peak fields disagree with the audited memory points")
    feasible = all(point.host_bytes <= value.memory_limit
                   and (value.host_memory_limit is None
                        or point.host_bytes <= value.host_memory_limit)
                   and point.temporary_disk_bytes <= value.disk_limit_bytes
                   for point in points)
    if not feasible:
        raise MemoryError("temporary resource memory or disk cap is exceeded")


@dataclass(frozen=True)
class ResourceAllocation:
    """One owner; borrowed logical views have no allocation of their own."""
    domain: str
    key: str
    host_bytes: int = 0
    device_bytes: int = 0
    disk_bytes: int = 0

    def __post_init__(self):
        if self.domain not in ("tensor", "block") or not isinstance(self.key, str) or not self.key:
            raise ValueError("allocation needs a domain and nonempty identity")
        for name in ("host_bytes", "device_bytes", "disk_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative integer bytes")


@dataclass(frozen=True)
class ResourceSnapshot:
    allocations: tuple[ResourceAllocation, ...] = ()
    borrowed: tuple[tuple[str, tuple[str, ...]], ...] = ()
    opened_blocks: frozenset[str] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "allocations", tuple(self.allocations))
        borrowed = self.borrowed
        if not isinstance(borrowed, tuple) or any(
                not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[1], tuple)
                for item in borrowed):
            borrowed = tuple((key, tuple(owners)) for key, owners in borrowed)
            object.__setattr__(self, "borrowed", borrowed)
        object.__setattr__(self, "opened_blocks", frozenset(self.opened_blocks))
        identities = [(a.domain, a.key) for a in self.allocations]
        if len(set(identities)) != len(identities):
            raise ValueError("one resource owner must have one allocation")
        if len({key for key, _ in self.borrowed}) != len(self.borrowed):
            raise ValueError("one borrowed view must have one dependency record")
        owners = {a.key for a in self.allocations if a.domain == "block"}
        if any(not set(deps) <= owners for _, deps in self.borrowed):
            raise ValueError("borrowed views require live block owners")
        if any(("tensor", key) in identities for key, _ in self.borrowed):
            raise ValueError("borrowed views cannot own a second allocation")
