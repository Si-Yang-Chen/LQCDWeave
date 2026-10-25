"""Execution-only errors for explicit saved-bundle binding."""

class UnsupportedLoaderError(TypeError):
    """Raised when a loader cannot satisfy the persisted-input contract."""

class SourceIdentityError(ValueError):
    """Raised when a persisted source descriptor cannot be verified."""

class PlanMismatchError(ValueError):
    """Raised when a current configuration does not match a persisted plan."""

__all__ = ["UnsupportedLoaderError", "SourceIdentityError", "PlanMismatchError"]
