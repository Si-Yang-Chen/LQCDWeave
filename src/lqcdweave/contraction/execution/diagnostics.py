"""Context-local controls for optional execution diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_EXECUTION_DEBUG: ContextVar[bool] = ContextVar("execution_debug", default=False)


@contextmanager
def execution_debug(enabled: bool) -> Iterator[None]:
    """Temporarily select whether this execution performs debug audits."""
    if type(enabled) is not bool:
        raise TypeError("enabled must be bool")
    token = _EXECUTION_DEBUG.set(enabled)
    try:
        yield
    finally:
        _EXECUTION_DEBUG.reset(token)


def debug_enabled() -> bool:
    """Return whether the current execution context enables debug audits."""
    return _EXECUTION_DEBUG.get()


__all__ = ["debug_enabled", "execution_debug"]
