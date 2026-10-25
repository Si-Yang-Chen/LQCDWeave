"""Generic runtime dispatch for explicitly injected execution requests."""

from __future__ import annotations

from ..contracts.execution import (
    DirectExecution,
    ExecutionOutcome,
    ExecutionRequest,
    PathExecution,
)


def execute_request(
    request: ExecutionRequest,
    *,
    backend: object,
    measure: bool = False,
) -> ExecutionOutcome:
    """Dispatch an injected executor without inspecting its domain payload."""
    if not isinstance(request, ExecutionRequest):
        raise TypeError("request must implement contracts.execution.ExecutionRequest")
    if type(measure) is not bool:
        raise TypeError("measure must be bool")
    result = request.execute(backend=backend, measure=measure)
    if not isinstance(result, (DirectExecution, PathExecution)):
        raise TypeError("execution request must return a typed DirectExecution or PathExecution")
    return result


__all__ = [
    "DirectExecution",
    "PathExecution",
    "execute_request",
]
