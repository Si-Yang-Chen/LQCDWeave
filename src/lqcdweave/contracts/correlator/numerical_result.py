"""Owned numerical result and explicit output-axis metadata; no I/O."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorrelatorResult:
    """A saved numerical tree, never a symbolic correlator or an execution plan.

    ``axes`` are the common trailing scan axes. ``field_axes`` and
    ``field_coordinates`` describe each numerical leaf by its container path.
    Arrays are owned by the caller. Frozen record attributes do not make the
    numerical arrays or detached metadata dictionaries immutable.
    """

    value: object
    axes: tuple = ()
    coordinates: dict = field(default_factory=dict)
    field_axes: dict = field(default_factory=dict)
    field_coordinates: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    reduction: dict | None = None
    provenance: dict = field(default_factory=dict)
    data_id: str | None = None
