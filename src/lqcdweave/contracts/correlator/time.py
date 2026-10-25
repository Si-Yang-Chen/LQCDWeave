"""Symbolic time keys, kept separate from nonnegative physical coordinates."""

from operator import index

import numpy as np


def normalize_time_key(value, *, name="time key"):
    """Accept named variables or existing nonnegative integer time labels."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        return value
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonempty string or nonnegative integer")
    try:
        value = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a nonempty string or nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def time_key_order(value):
    """A stable total order for symbolic canonicalization, never output axes."""
    value = normalize_time_key(value)
    return (1, value) if isinstance(value, str) else (0, value)


__all__ = ["normalize_time_key", "time_key_order"]
