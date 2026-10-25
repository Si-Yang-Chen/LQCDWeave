"""Injected service interfaces used by pure descriptors."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import ContextManager, Protocol
from .inputs import BatchSpec, CoordinateSpec, DiagramBindings, DiagramInputs

class DiagramBindingProvider(Protocol):
    """Metadata-only provider for Diagram input and output descriptions.

    ``describe`` may inspect registered reader metadata, but must not invoke
    numeric block getters, load data, or produce missing inputs. Numeric access
    stays in the execution capability carried by the returned request object.
    """

    def describe(
        self,
        correlator,
        *,
        bindings: DiagramBindings,
        coordinates: CoordinateSpec,
        batch: BatchSpec,
    ) -> DiagramInputs: ...
from .scene import Action, ContractionProblem
from .ir import TensorSpec

class CostModel(Protocol):
    version: str
    hardware_fingerprint: str
    def snapshot_seconds(self, nbytes: int) -> float: ...
    def preparation_seconds(self, flops, traffic_bytes, *, kernel_calls=0, dtype=None) -> float: ...
    def contract_seconds(self, flops: int, traffic_bytes: int, *, kind="contract", dtype=None, kernel_calls=1, sparse_records=1) -> float: ...
    def kernel_call_for_dtype(self, dtype=None) -> float: ...

class StateDescriptor(Protocol):
    """Read-only state facts consumed by plan inspection and execution."""
    problem: ContractionProblem
    actions: tuple[Action, ...]
    catalog: Mapping[str, TensorSpec]
    results: tuple[str | None, ...]
    seconds: float
    done: bool
    peak_bytes: int
    peak_host_bytes: int
    peak_device_bytes: int

class SourceLoader(Protocol):
    """Runtime-only numeric access injected by bindings, never archived.

    ``read_block`` returns storage owned by the loader for this execution.
    ``materialize`` may return views borrowing those supplied blocks, but does
    not open another source. ``close_block`` and ``close_all`` release only
    task-local blocks/handles created by this loader; they do not close the
    caller's reader object.
    """

    def read_block(self, key: str, block) -> object: ...
    def materialize(self, key: str, source, blocks: Mapping[str, object]) -> object: ...
    def close_block(self, key: str, block: object) -> None: ...
    def close_all(self) -> None: ...


class TemporaryArrayScope(Protocol):
    """One execution's private temporary-array lifetime.

    Receipts are opaque runtime values. ``read_array`` returns a new owned
    numeric array with the requested shape and dtype; ``write_array`` must
    finish publishing a complete value before it returns. ``close`` removes
    every remaining value in this scope, including files left by failed writes.
    The protocol carries neither paths nor a scientific dataset identity.
    """

    def write_array(self, array: object) -> object: ...
    def read_array(self, receipt: object, *, shape: tuple[int, ...], dtype: str) -> object: ...
    def remove_array(self, receipt: object) -> None: ...
    def close(self) -> None: ...


class TemporaryArrayStore(Protocol):
    """Optional, backend-neutral capability for execution-scoped scratch IO."""

    @property
    def host_reserve_bytes(self) -> int:
        """Additional host workspace needed while writing or restoring arrays."""

    def storage_bytes(self, shape: tuple[int, ...], dtype: str) -> int:
        """Upper bound on scratch-file bytes for one array, including format overhead."""

    def open_scope(self) -> TemporaryArrayScope:
        """Create an isolated scope whose close removes every scratch file."""


class BatchLoader(Protocol):
    """Optional task-scoped loader selection for a preplanned compute batch."""

    def activate(self, batch_index: int) -> ContextManager[None]: ...


class MeasurementAwareLoader(Protocol):
    """Optional measurement switch for adapters that expose per-read events."""

    def set_measurement(self, enabled: bool) -> None: ...


class IOReportingLoader(Protocol):
    """Optional adapter telemetry drained at a specific execution action."""

    def drain_io_events(self) -> Iterable[Mapping[str, object]]: ...


class BackgroundBlockLoader(Protocol):
    """Optional host-only read capability, independent of source preparation.

    Descriptor methods perform no I/O. One worker may call ``prefetch_block``
    concurrently with main-thread materialization, kernels, and the release
    of other consumed blocks. It must not
    touch a device or a mutable buffer borrowed by those operations. Its byte
    bound includes the returned host value and all transient read workspace.
    ``finish_prefetch`` runs on the owner thread and consumes the host value
    into the same result as ``read_block``; device transfer belongs there.
    ``close_prefetch`` disposes an unconsumed host value on that owner thread.
    Readers remain open until the worker has stopped.
    Task-scoped loaders may additionally declare ``prefetch_across_batches =
    True`` when these callbacks do not depend on the currently activated batch.
    """

    def can_prefetch_block(self, key: str, block) -> bool: ...
    def prefetch_host_bytes(self, key: str, block) -> int: ...
    def prefetch_block(self, key: str, block) -> object: ...
    def finish_prefetch(self, key: str, block, value: object) -> object: ...
    def close_prefetch(self, key: str, value: object) -> None: ...
