"""Immutable event/report values exchanged by runtime and numerical executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhaseTiming:
    """One phase nested under the action timing that includes its wall time."""

    name: str
    observed_seconds: float
    device_seconds: float | None = None
    bytes: int | None = None
    parent_action_index: int | None = None
    stream_id: str | None = None
    timing_mode: str | None = None


@dataclass(frozen=True)
class ActionTiming:
    kind: str
    estimated_seconds: float
    observed_seconds: float
    action_index: int = -1
    tensor: str = ""
    phases: tuple[PhaseTiming, ...] = ()
    operands: tuple[str, ...] = ()
    region: str | None = None
    model_features: dict[str, int | bool | str] | None = None


@dataclass(frozen=True)
class FileIOEvent:
    action_index: int
    tensor: str
    path: str
    operation: str
    bytes: int
    observed_seconds: float


@dataclass(frozen=True)
class ReplayReport:
    values: tuple[Any, ...]
    timings: tuple[ActionTiming, ...]
    peak_device_bytes: int | None
    peak_host_rss_bytes: int | None
    output_timings: tuple[ActionTiming, ...] = ()
    io_events: tuple[FileIOEvent, ...] = ()
    execution_profile: dict[str, Any] | None = None


__all__ = ["ActionTiming", "FileIOEvent", "PhaseTiming", "ReplayReport"]
