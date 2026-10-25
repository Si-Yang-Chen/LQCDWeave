"""Cfg-bound getter loader capability for persisted Meson workloads."""
from __future__ import annotations

from collections.abc import Mapping


def bind_meson_getters(cfg, *, vertices, propagators):
    """Bind handle sequences already opened for ``cfg`` to a loader closure.

    Binding borrows the supplied sequences and performs no getter reads or
    closes. The caller keeps the handles alive through each session execution.
    The execution layer owns immediate snapshots of returned arrays.
    """
    def load(request_cfg, key, source):
        try:
            same_cfg = request_cfg == cfg
            same_cfg = bool(same_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError("request configuration cannot be compared with the bound cfg") from exc
        if not same_cfg:
            raise ValueError("getter objects are bound to another configuration")
        if not isinstance(source, Mapping):
            raise ValueError("persisted Meson input source must be a mapping")
        if source.get("method") != "get":
            raise ValueError("unsupported persisted getter method")
        kind = source.get("kind")
        if kind == "vertex":
            handles = vertices
            expected_time_count = 1
        elif kind == "propagator":
            handles = propagators
            expected_time_count = 2
        else:
            raise ValueError("unsupported persisted input kind")

        handle = source.get("handle")
        if type(handle) is not int or handle < 0:
            raise ValueError("persisted getter handle must be a non-negative integer")
        times = source.get("times")
        if not isinstance(times, (tuple, list)) or len(times) != expected_time_count:
            raise ValueError(f"persisted {kind} getter requires {expected_time_count} time value(s)")
        if any(type(value) is not int for value in times):
            raise ValueError("persisted getter times must be integer labels")
        try:
            getter = handles[handle]
        except (IndexError, KeyError, TypeError) as exc:
            raise IndexError(f"no {kind} getter handle at index {handle}") from exc
        method = getattr(getter, "get", None)
        if not callable(method):
            raise ValueError(f"{kind} getter handle at index {handle} has no callable get method")
        return method(*times)

    return load


__all__ = ["bind_meson_getters"]