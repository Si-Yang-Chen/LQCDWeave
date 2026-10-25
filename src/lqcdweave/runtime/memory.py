"""Fresh host/device memory observations for runtime execution guards.

Capacity resolution belongs to planning. This module normalizes the requested
backend, obtains only observations the caller did not supply, and delegates
the pure cap calculation through ``contraction.planning.api``.
"""

from __future__ import annotations

from contextlib import suppress, contextmanager
from contextvars import ContextVar
from dataclasses import replace
import importlib
import os
from typing import Any

from lqcdweave.contracts.memory import (
    BudgetError,
    BudgetExceededError,
    MemoryEstimate,
    MemoryLimits,
)
import lqcdweave.contraction.planning.api as _planning_memory

_reusable_memory = ContextVar('execution_reusable_memory', default=None)


@contextmanager
def reusable_device_memory(provider):
    """Expose only the current owner's idle bytes to fresh admission checks."""
    token = _reusable_memory.set(provider)
    try:
        yield
    finally:
        _reusable_memory.reset(token)


def reusable_device_bytes(backend):
    provider = _reusable_memory.get()
    return 0 if provider is None else provider(_backend_module(backend))


def _backend_module(backend: Any) -> Any:
    if backend is None:
        return importlib.import_module("numpy")
    if isinstance(backend, str):
        lowered = backend.lower()
        if lowered not in ("numpy", "cupy"):
            raise ValueError("backend must be NumPy or CuPy")
        return importlib.import_module(lowered)
    name = getattr(backend, "__name__", "").lower()
    if name not in ("numpy", "cupy"):
        raise ValueError("backend must be NumPy or CuPy")
    return backend


def device_memory_info(backend: Any) -> tuple[int | None, int | None]:
    """Return ``(free, total)`` for CuPy, or ``(None, None)`` otherwise."""

    xp = _backend_module(backend)
    if getattr(xp, "__name__", "").lower() != "cupy":
        return None, None
    try:
        free, total = xp.cuda.runtime.memGetInfo()
    except Exception as exc:  # pragma: no cover - backend/device dependent
        raise BudgetError("unable to query CuPy device memory") from exc
    return int(free), int(total)


def host_available_memory() -> int | None:
    """Return available host memory where a reliable query is available."""

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        # Avoid fabricating a host-memory guarantee. The static logical guard
        # remains usable when psutil is not installed.
        pass
    if os.name == "posix":  # pragma: no cover - platform dependent fallback
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (OSError, ValueError, AttributeError):
            pass
    return None


def resolve_memory_limits(
    budget: Any,
    backend: Any = None,
    *,
    host_available: int | None = None,
    device_info: tuple[int, int] | None = None,
) -> MemoryLimits:
    """Resolve current host/device observations through the pure planner API.

    Unspecified observations are queried on every call. Injected observations
    remain deterministic and never cause a probe. CPU resolution preserves the
    legacy behavior of ignoring device observations.
    """

    xp = _backend_module(backend)
    is_gpu = getattr(xp, "__name__", "").lower() == "cupy"
    observed_host = host_available if host_available is not None else host_available_memory()
    observed_device = device_info
    if is_gpu and observed_device is None:
        free, total = device_memory_info(xp)
        observed_device = (min(total, free + reusable_device_bytes(xp)), total)
    return _planning_memory.resolve_budget_limits(
        budget,
        target="gpu" if is_gpu else "cpu",
        host_available=observed_host,
        device_info=observed_device if is_gpu else None,
    )


def ensure_within_budget(
    estimate: MemoryEstimate,
    budget: Any,
    *,
    backend: Any = None,
    limits: MemoryLimits | None = None,
    label: str = "execution batch",
) -> MemoryLimits:
    """Validate an estimate against a supplied or freshly resolved limit."""

    if not isinstance(estimate, MemoryEstimate):
        raise TypeError("estimate must be a MemoryEstimate")
    resolved = limits if limits is not None else resolve_memory_limits(budget, backend)
    xp = _backend_module(backend)
    limit = resolved.limit_for(xp)
    if limit is not None and estimate.logical_peak_bytes > limit:
        raise BudgetExceededError(
            f"{label} requires {estimate.logical_peak_bytes} logical bytes, "
            f"but the resolved {('device' if xp.__name__ == 'cupy' else 'host')} "
            f"budget is {limit} bytes"
        )
    return resolved


def actual_available_bytes(backend: Any = None) -> tuple[int | None, int | None]:
    """Return fresh ``(host_available, device_free)`` observations."""

    xp = _backend_module(backend)
    host = host_available_memory()
    device, _ = device_memory_info(xp)
    return host, device


def fresh_execution_limits(
    budget: Any,
    backend: Any = None,
) -> MemoryLimits:
    """Resolve a fresh execution guard, including CuPy pool headroom.

    A GPU call queries driver memory once and adds free bytes in CuPy's reusable
    pool. Pool inspection is opportunistic: if it fails, pool headroom is zero.
    Driver query failures keep ``device_memory_info``'s canonical ``BudgetError``;
    a cap violation remains the separate ``BudgetExceededError`` decision made
    by the caller's guard. Host memory is sampled once by
    :func:`resolve_memory_limits`; this avoids the former discarded
    ``actual_available_bytes`` sample and duplicate driver query.
    """

    xp = _backend_module(backend)
    if getattr(xp, "__name__", "").lower() != "cupy":
        return resolve_memory_limits(budget, xp)

    driver_free, total = device_memory_info(xp)
    pool_free = reusable_device_bytes(xp)
    with suppress(Exception):
        pool_free += int(xp.get_default_memory_pool().free_bytes())
    limits = resolve_memory_limits(
        budget,
        xp,
        device_info=(min(int(total), int(driver_free) + pool_free), int(total)),
    )
    return replace(
        limits,
        device_driver_free_bytes=int(driver_free),
        device_pool_free_bytes=int(pool_free),
    )


__all__ = [
    "actual_available_bytes",
    "device_memory_info",
    "ensure_within_budget",
    "fresh_execution_limits",
    "host_available_memory",
    "resolve_memory_limits",
]
