"""Pure, numerical-free records for explicit three-line baryon contractions."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
import numbers
import operator
from typing import Any

import numpy as np


def _check_hashable_key(key: Hashable, *, name: str) -> None:
    try:
        hash(key)
    except TypeError as exc:
        raise TypeError(f"{name} key {key!r} is not hashable") from exc


def _normalise_three_keys(value: Any, *, name: str) -> tuple[Hashable, Hashable, Hashable]:
    try:
        keys = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain exactly three keys") from exc
    if len(keys) != 3:
        raise ValueError(f"{name} must contain exactly three keys")
    for index, key in enumerate(keys):
        _check_hashable_key(key, name=f"{name}[{index}]")
    return keys  # type: ignore[return-value]


def _normalise_source_slots(value: Any) -> tuple[int, int, int]:
    try:
        slots = tuple(value)
    except TypeError as exc:
        raise TypeError("source_slots must be a permutation of (0, 1, 2)") from exc
    if len(slots) != 3:
        raise ValueError("source_slots must be a permutation of (0, 1, 2)")

    normalized: list[int] = []
    for index, slot in enumerate(slots):
        if isinstance(slot, (bool, np.bool_)):
            raise TypeError(f"source_slots[{index}] must be an integer, not bool")
        try:
            normalized.append(operator.index(slot))
        except TypeError as exc:
            raise TypeError(
                f"source_slots[{index}] must be an integer in {{0, 1, 2}}"
            ) from exc
    result = tuple(normalized)
    if set(result) != {0, 1, 2}:
        raise ValueError("source_slots must be a permutation of (0, 1, 2)")
    return result  # type: ignore[return-value]


def _validate_coefficient(coefficient: Any) -> None:
    if not isinstance(coefficient, (numbers.Number, np.bool_)):
        raise TypeError("term coefficient must be a scalar number")


@dataclass(frozen=True, slots=True)
class BaryonVertex:
    """A source or sink vertex represented by its spin and elemental keys."""

    spin: Hashable
    elemental: Hashable

    def __post_init__(self) -> None:
        _check_hashable_key(self.spin, name="spin")
        _check_hashable_key(self.elemental, name="elemental")


@dataclass(frozen=True, slots=True)
class BaryonTerm:
    """One explicit source-to-sink three-propagator contraction term."""

    source: BaryonVertex
    sink: BaryonVertex
    propagators: tuple[Hashable, Hashable, Hashable]
    source_slots: tuple[int, int, int] = (0, 1, 2)
    coefficient: Any = 1

    def __post_init__(self) -> None:
        if not isinstance(self.source, BaryonVertex):
            raise TypeError("source must be a BaryonVertex")
        if not isinstance(self.sink, BaryonVertex):
            raise TypeError("sink must be a BaryonVertex")
        object.__setattr__(
            self,
            "propagators",
            _normalise_three_keys(self.propagators, name="propagators"),
        )
        object.__setattr__(self, "source_slots", _normalise_source_slots(self.source_slots))
        _validate_coefficient(self.coefficient)


@dataclass(frozen=True, slots=True)
class BaryonLine:
    """One directed propagator edge and its source/sink slots."""

    propagator: int
    source_slot: int
    sink_slot: int

    def __post_init__(self) -> None:
        for name, value in (
            ("propagator", self.propagator),
            ("source_slot", self.source_slot),
            ("sink_slot", self.sink_slot),
        ):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer, not bool")
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            object.__setattr__(self, name, normalized)
        if self.propagator <= 0:
            raise ValueError("propagator IDs must be positive")
        if self.source_slot not in (0, 1, 2):
            raise ValueError("source_slot must be one of 0, 1, or 2")
        if self.sink_slot not in (0, 1, 2):
            raise ValueError("sink_slot must be one of 0, 1, or 2")


@dataclass(frozen=True, slots=True)
class BaryonTopology:
    """A validated two-vertex topology with lines ordered by sink slot."""

    source_vertex: int
    sink_vertex: int
    lines: tuple[BaryonLine, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("source_vertex", self.source_vertex),
            ("sink_vertex", self.sink_vertex),
        ):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer, not bool")
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            object.__setattr__(self, name, normalized)
        if self.source_vertex not in (0, 1) or self.sink_vertex not in (0, 1):
            raise ValueError("baryon vertices must be 0 or 1")
        if self.source_vertex == self.sink_vertex:
            raise ValueError("source and sink vertices must be distinct")
        lines = tuple(self.lines)
        if len(lines) != 3 or any(not isinstance(line, BaryonLine) for line in lines):
            raise ValueError("baryon topology must contain three BaryonLine objects")
        if tuple(line.sink_slot for line in lines) != (0, 1, 2):
            raise ValueError("baryon lines must be ordered by sink slot")
        if tuple(sorted(line.source_slot for line in lines)) != (0, 1, 2):
            raise ValueError("baryon source slots must be a bijection")
        object.__setattr__(self, "lines", lines)


__all__ = ["BaryonLine", "BaryonTerm", "BaryonTopology", "BaryonVertex"]

