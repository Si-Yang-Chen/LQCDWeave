"""Typed execution values shared by contraction, workflow, and runtime modules."""

from __future__ import annotations

from collections.abc import Mapping
import math
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable


@dataclass(frozen=True)
class TermValue:
    """One raw contraction term result at one requested coordinate."""

    term_id: str
    coordinate_key: str
    value: object

    def __post_init__(self):
        if not isinstance(self.term_id, str) or not self.term_id:
            raise ValueError("TermValue needs a nonempty term_id")
        if not isinstance(self.coordinate_key, str) or not self.coordinate_key:
            raise ValueError("TermValue needs a nonempty coordinate key")


def _term_value_tuple(values, owner: str) -> tuple[TermValue, ...]:
    result = tuple(values)
    if any(not isinstance(value, TermValue) for value in result):
        raise TypeError(f"{owner}.term_values must contain TermValue objects")
    return result


@dataclass(frozen=True)
class DirectExecution:
    """Raw per-term values from a direct executor, before output assembly."""

    term_values: tuple[TermValue, ...]
    report: object = None

    def __post_init__(self):
        object.__setattr__(self, "term_values", _term_value_tuple(self.term_values, "DirectExecution"))


@dataclass(frozen=True)
class PathExecution:
    """Typed result of executing a selected contraction descriptor."""

    term_values: tuple[TermValue, ...]
    report: object = None

    def __post_init__(self):
        object.__setattr__(self, "term_values", _term_value_tuple(self.term_values, "PathExecution"))


ExecutionOutcome: TypeAlias = DirectExecution | PathExecution


@runtime_checkable
class ExecutionRequest(Protocol):
    """Explicit executor injected into the generic runtime dispatch boundary."""

    def execute(self, *, backend: object, measure: bool) -> ExecutionOutcome: ...


__all__ = [
    "DirectExecution",
    "ExecutionOutcome",
    "ExecutionRequest",
    "PathExecution",
    "TermValue",
    "normalize_execution_parameters",
]


from .dataset import canonical_json


def normalize_execution_parameters(value):
    """Return a detached JSON object, or None for no explicit parameters.

    Mapping keys must be strings. Values may contain only JSON-native scalars,
    lists and mappings; tuples, arrays, cycles and nonfinite floats are refused
    rather than coerced. Shared containers are detached independently.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError('execution parameters must be a JSON object or None')
    active = set()

    def detach(item):
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError('execution parameters cannot contain nonfinite numbers')
            return item
        if isinstance(item, Mapping) or type(item) is list:
            identity = id(item)
            if identity in active:
                raise ValueError('execution parameters cannot contain cycles')
            active.add(identity)
            try:
                if isinstance(item, Mapping):
                    if any(not isinstance(key, str) for key in item):
                        raise TypeError('execution parameter keys must be strings')
                    return {key:detach(child) for key, child in item.items()}
                return [detach(child) for child in item]
            finally:
                active.remove(identity)
        raise TypeError('execution parameters must contain JSON-native values')

    result = detach(value)
    # Check the actual JSON encoding while this is still a pure contract.
    canonical_json(result).encode('utf-8')
    return result
