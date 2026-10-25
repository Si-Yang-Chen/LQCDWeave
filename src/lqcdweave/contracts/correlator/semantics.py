"""Pure logical tensor semantics attached to the unique TensorSpec catalog."""
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import math
from types import MappingProxyType

import numpy as np


def _freeze_description(value, known=None):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, complex)):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("tensor semantics require finite numeric values")
        return value
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, Mapping):
        if known is not None and known.get(id(value)) is value:
            return value
        if any(not isinstance(key, str) for key in value):
            raise TypeError("tensor semantics mapping keys must be strings")
        return MappingProxyType({key: _freeze_description(item, known) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        if known is not None and known.get(id(value)) is value:
            return value
        return tuple(_freeze_description(item, known) for item in value)
    raise TypeError("tensor description cannot contain arrays, callbacks, or live objects")


@dataclass(frozen=True)
class TensorDescription:
    axes: tuple[str, ...] = ()
    data_identity: str | None = None
    selection_identity: str | None = None
    role: str = ""
    source_bindings: tuple[str, ...] = ()
    transform: tuple = ()
    semantics: Mapping = field(default_factory=dict)
    bindings: Mapping = field(default_factory=dict)
    _metadata_sources: InitVar[tuple] = ()

    def __post_init__(self, _metadata_sources):
        # Only a previously validated TensorDescription proves deep immutability.
        # Arbitrary mapping proxies can still refer to mutable external dicts.
        known = None
        if _metadata_sources:
            if any(type(source) is not TensorDescription for source in _metadata_sources):
                raise TypeError('metadata sources require validated TensorDescription records')
            known = {id(value): value for source in _metadata_sources
                     for value in (source.transform, source.semantics, source.bindings)}
        axes, sources = tuple(self.axes), tuple(self.source_bindings)
        if any(not isinstance(name, str) or not name for name in axes + sources):
            raise TypeError("tensor axes and source bindings must be nonempty strings")
        if len(set(axes)) != len(axes):
            raise ValueError("logical tensor axis names must be unique")
        for identity in (self.data_identity, self.selection_identity):
            if identity is not None and (not isinstance(identity, str) or not identity):
                raise TypeError("unknown tensor identities are None; known identities are nonempty strings")
        if not isinstance(self.role, str):
            raise TypeError("tensor role must be a string")
        if not isinstance(self.semantics, Mapping) or not isinstance(self.bindings, Mapping):
            raise TypeError("tensor semantics and actual bindings must be mappings")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "source_bindings", sources)
        object.__setattr__(self, "transform", _freeze_description(tuple(self.transform), known))
        object.__setattr__(self, "semantics", _freeze_description(self.semantics, known))
        object.__setattr__(self, "bindings", _freeze_description(self.bindings, known))
