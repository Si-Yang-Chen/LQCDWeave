"""Neutral numeric memory records shared by planning and runtime callers.

This module defines value contracts and small derived properties only. It does
not estimate a plan, inspect arrays, probe process/device resources, or import
a numerical backend.
"""

from dataclasses import dataclass
import numbers
from typing import Any


DEFAULT_PLANNING_FRACTION = 0.8


class BudgetError(RuntimeError):
    """Base class for an infeasible or invalid memory budget."""


class BudgetExceededError(BudgetError, MemoryError):
    """Raised before or during execution when the declared budget is exceeded."""


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Caller-modeled logical byte accounting for one execution batch.

    This record has no allocation-domain label: the caller assigns its values
    to a host or device accounting context. It does not split or combine those
    domains. ``input_bytes`` counts one owned allocation for each distinct
    input represented by the estimate. A separately retained loader/executor
    snapshot is another owned copy and belongs in ``snapshot_bytes`` too.
    ``intermediate_bytes`` and ``output_bytes`` describe intermediate and
    output storage modeled as live at the batch peak, not cumulative
    allocations over the whole run. ``assembly_extra_bytes`` is the caller's
    declaration of additional assembly storage at that peak. ``workspace_bytes``
    is an unverified caller-supplied reserve or estimate for temporary
    workspace, not an observation or proof that every backend/native temporary
    is covered.

    ``logical_peak_bytes`` sums these categories according to the caller's
    model. It is not RSS, a complete bound on allocator/native-library scratch,
    or a host/device limit by itself.
    """

    input_bytes: int = 0
    snapshot_bytes: int = 0
    intermediate_bytes: int = 0
    output_bytes: int = 0
    assembly_extra_bytes: int = 0
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "input_bytes",
            "snapshot_bytes",
            "intermediate_bytes",
            "output_bytes",
            "assembly_extra_bytes",
            "workspace_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def logical_peak_bytes(self) -> int:
        """Return the caller-modeled sum; this is not a measured RSS value."""

        return int(
            self.input_bytes
            + self.snapshot_bytes
            + self.intermediate_bytes
            + self.output_bytes
            + self.assembly_extra_bytes
            + self.workspace_bytes
        )

    @property
    def peak_bytes(self) -> int:
        """Compatibility alias for :attr:`logical_peak_bytes`."""

        return self.logical_peak_bytes

    def as_dict(self) -> dict[str, int]:
        return {
            "input_bytes": self.input_bytes,
            "snapshot_bytes": self.snapshot_bytes,
            "intermediate_bytes": self.intermediate_bytes,
            "output_bytes": self.output_bytes,
            "assembly_extra_bytes": self.assembly_extra_bytes,
            "workspace_bytes": self.workspace_bytes,
            "logical_peak_bytes": self.logical_peak_bytes,
        }


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    """Resolved host/device limits and observations for one process."""

    host_bytes: int | None
    device_bytes: int | None
    host_available_bytes: int | None = None
    device_free_bytes: int | None = None
    device_total_bytes: int | None = None
    device_reserve_bytes: int | None = None
    device_driver_free_bytes: int | None = None
    device_pool_free_bytes: int | None = None

    def limit_for(self, backend: Any) -> int | None:
        name = getattr(backend, "__name__", backend)
        return self.device_bytes if str(name).lower() == "cupy" else self.host_bytes


def _memory_bytes(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a nonnegative integer or None")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return int(value)


@dataclass(frozen=True, slots=True)
class TargetHardwareMemory:
    """Declared memory available on a target when planning away from it.

    ``host_available_bytes`` is the amount available to the job after its
    process/job limits. Device ``free`` means memory available before the new
    Diagram task, so existing allocations are already reflected. Supply either
    ``device_free_bytes`` or ``device_used_bytes`` together with
    ``device_total_bytes``; if both are supplied they must agree. A target
    reservation that has not yet been allocated must be reflected in the
    declared free amount. Planner ``reserved_*_bytes`` remain separate modeled
    task memory and are counted by the existing planning layer.
    """

    host_available_bytes: int | None = None
    device_total_bytes: int | None = None
    device_free_bytes: int | None = None
    device_used_bytes: int | None = None

    def __post_init__(self):
        host = _memory_bytes(self.host_available_bytes, "host_available_bytes")
        total = _memory_bytes(self.device_total_bytes, "device_total_bytes")
        free = _memory_bytes(self.device_free_bytes, "device_free_bytes")
        used = _memory_bytes(self.device_used_bytes, "device_used_bytes")
        if any(value is not None for value in (free, used)) and total is None:
            raise ValueError("device_total_bytes is required with device free/used memory")
        if total is not None:
            if free is None and used is None:
                raise ValueError("device_free_bytes or device_used_bytes is required with device_total_bytes")
            if free is None:
                free = total - used
                if free < 0:
                    raise ValueError("device_used_bytes cannot exceed device_total_bytes")
            elif free > total:
                raise ValueError("device_free_bytes cannot exceed device_total_bytes")
            if used is None:
                used = total - free
            elif used > total or free + used != total:
                raise ValueError("device free and used bytes must sum to device_total_bytes")
        object.__setattr__(self, "host_available_bytes", host)
        object.__setattr__(self, "device_total_bytes", total)
        object.__setattr__(self, "device_free_bytes", free)
        object.__setattr__(self, "device_used_bytes", used)


__all__ = [
    "DEFAULT_PLANNING_FRACTION",
    "BudgetError",
    "BudgetExceededError",
    "MemoryEstimate",
    "MemoryLimits",
    "TargetHardwareMemory",
]
