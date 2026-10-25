"""Immutable selected executable plan and prediction envelope."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PlannedContraction:
    expanded: object
    plan: object
    search: object
    prediction: dict
    planning_seconds: float
    execution_mode: str = "planned"
    memory_limit: int | None = None
    host_memory_limit: int | None = None
    plan_identity: int | None = None
    profile: object = None
    reserved_host_bytes: int = 0
    reserved_device_bytes: int = 0
    batch_paths: tuple["PlannedContraction", ...] = ()
    batch_shape_keys: tuple[object, ...] = ()

