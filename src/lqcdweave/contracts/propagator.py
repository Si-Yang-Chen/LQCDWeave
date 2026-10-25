"""Backend boundary for propagator production; no device initialization."""

from typing import Any, Protocol
from dataclasses import dataclass
from numbers import Integral
import json

from .dataset import canonical_json


@dataclass(frozen=True, init=False)
class PropagatorEndpoint:
    """Actual ordered spatial endpoint, including raw dilution provenance.

    D indices address the flattened vector axis. Its scientific block labels,
    parent-mode groups, realization/pool and normalization belong in metadata.
    Unknown identities remain None; persistence requires actual data IDs.
    """

    kind: str
    data_id: str | None
    indices: tuple
    _metadata_json: str

    def __init__(self, kind, data_id, indices, *, metadata=None):
        kind = str(kind).upper()
        if kind not in ('V', 'P', 'D'):
            raise ValueError('endpoint kind must be V, P or D')
        if data_id is not None and (not isinstance(data_id, str) or not data_id):
            raise ValueError('endpoint data_id must be an actual ID or None')
        indices = tuple(indices)
        if (not indices or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0
                               for i in indices) or len(set(indices)) != len(indices)):
            raise ValueError('endpoint indices must be distinct nonnegative integers')
        metadata = {} if metadata is None else dict(metadata)
        object.__setattr__(self, 'kind', kind)
        object.__setattr__(self, 'data_id', data_id)
        object.__setattr__(self, 'indices', tuple(int(i) for i in indices))
        object.__setattr__(self, '_metadata_json', canonical_json(metadata))

    @property
    def metadata(self):
        return json.loads(self._metadata_json)

    @property
    def document(self):
        return dict(kind=self.kind, data_id=self.data_id,
                    indices=list(self.indices), metadata=self.metadata)

    @classmethod
    def from_document(cls, document):
        return cls(**document)


@dataclass(frozen=True)
class PropagatorBlock:
    """One physical time pair; values have spin, spin, sink, source axes.

    Values may borrow a generator/reader buffer. Metadata is immutable; a
    Gamma5-derived block owns its new array and records its actual parent ID.
    """

    product: str
    sink_time: int
    source_time: int
    sink: PropagatorEndpoint
    source: PropagatorEndpoint
    values: Any
    parent_id: str | None = None
    derived: bool = False

    def __post_init__(self):
        if not isinstance(self.sink, PropagatorEndpoint) or not isinstance(self.source, PropagatorEndpoint):
            raise TypeError('block endpoints must be PropagatorEndpoint records')
        product = str(self.product).upper()
        if product != self.sink.kind + 'S' + self.source.kind:
            raise ValueError('block product differs from its actual endpoints')
        for time in (self.sink_time, self.source_time):
            if isinstance(time, bool) or not isinstance(time, Integral) or time < 0:
                raise ValueError('block times must be nonnegative physical integers')
        object.__setattr__(self, 'product', product)


class PropagatorBackend(Protocol):
    """A backend instance belongs to one generator.

    load receives a gauge payload and a parameter dictionary. solve_into solves
    D x = rhs and fills the caller-owned solution buffer, or raises on failure.
    It must not modify rhs. tol and maxiter describe the requested solve.
    Adapters may declare stream_ordered_solve=True when native solver/array
    dependencies are handled internally and solution is usable on return on
    the caller's current array stream. Without this optional capability the
    producer retains the legacy synchronize() hook at stage boundaries.
    Device arrays need not be host-complete; callers synchronize before
    measuring completion or consuming them outside that stream.

    The current single-process numerical path uses logical fields shaped
    (rhs,t,z,y,x,spin,color). The adapter owns conversions to solver-native
    layouts; array_module, empty and zeros keep arrays on the selected device.
    The t extent is always physical Lt, even for selected source/sink products.
    Optional gamma5_hermitian=True and gamma5 declare the action's spin-basis
    convention for built-in physical pair reversal. A generic solver must not
    claim this property for an arbitrary matrix. Native Wilson/clover adapters
    provide the convention; custom actions may explicitly supply it at setup.
    MPI domain ownership and reductions are not implemented by this interface.
    """

    array_module: Any

    def load(self, gauge, parameters: dict) -> None: ...
    def solve_into(self, rhs, solution, *, tol: float, maxiter: int) -> None: ...
    def empty(self, shape, dtype): ...
    def zeros(self, shape, dtype): ...


class ModeBlockHandle(Protocol):
    """Read an ordered physical-time/modal block for propagator production.

    read_modes(key, time, start, stop) returns (stop-start,z,y,x,color).
    Returned memory may be borrowed and expire on the next handle call or close;
    the producer must copy it into bounded scratch before making another read.
    The handle reports the available mode count, not a guessed basis identity.
    Optional time_selection is the shared TimeSelectionRecord describing compact
    load() row order and physical Lt. If omitted, load() uses full canonical Lt.
    read_modes still accepts physical time: the adapter maps it to a stored slot.
    Selected production endpoints must be covered; no shape-based inference or
    implicit temporal resampling is permitted.
    """

    Ne: int

    def read_modes(self, key, time: int, start: int, stop: int): ...


class PropagatorGaugeTransform(Protocol):
    """Optional adapter-owned gauge transformation for a producer.

    The core treats the payload as opaque and never inspects solver-native
    methods. The adapter returns the transformed payload; it may preserve
    legacy in-place ownership semantics, which must be explicitly documented.
    A failed transformation invalidates the producer until its next load.
    """

    def stout_smear(self, gauge, *, nstep: int, rho: float, dir_ignore: int) -> Any: ...
