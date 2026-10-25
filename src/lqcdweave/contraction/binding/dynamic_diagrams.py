"""Read dynamic simple and three-line baryon diagram inputs.

The request routing and snapshot rules were migrated from the now-retired
``contraction.diagram_adapter`` and ``contraction.baryon_adapter`` sources.
This module owns only getter access and input bindings; it does not compile or
execute contractions.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import operator
from typing import Any

import numpy as np

from lqcdweave.contracts.correlator.api import BaryonTopology, TensorSpec


_SIMPLE_METHODS = {
    "VSV": "get",
    "VSP": "get_v2p",
    "PSV": "get_p2v",
    "PSP": "get_p2p",
}


def _backend_module(backend: Any) -> tuple[Any, str]:
    if isinstance(backend, str):
        name = backend.lower()
        if name not in {"numpy", "cupy"}:
            raise NotImplementedError("dynamic diagram inputs support NumPy or CuPy")
        try:
            backend = importlib.import_module(name)
        except ImportError as exc:
            raise NotImplementedError(f"{name} is not installed") from exc
    if backend is np:
        return backend, "numpy"
    if getattr(backend, "__name__", "").rsplit(".", 1)[-1].lower() == "cupy":
        try:
            actual_cupy = importlib.import_module("cupy")
        except ImportError as exc:
            raise NotImplementedError("dynamic diagram inputs support CuPy only when installed") from exc
        if backend is actual_cupy:
            return backend, "cupy"
    raise NotImplementedError("dynamic diagram inputs support the NumPy backend or CuPy only")


def _sequence(value: Any, role: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{role} must be a sequence, not text")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{role} must be a sequence") from exc


def _index(value: Any, role: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{role} must be an integer index, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{role} must be an integer index") from exc


def _table_item(table: Any, value: Any, role: str) -> tuple[int, Any]:
    index = _index(value, role)
    try:
        item = table[index]
    except (IndexError, KeyError) as exc:
        raise ValueError(f"{role} {index} is outside the available inputs") from exc
    except TypeError as exc:
        raise TypeError(f"{role} source must support integer indexing") from exc
    return index, item


def _time_token(value: Any, role: str) -> tuple[str, str, int]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise NotImplementedError(f"dynamic simple diagrams support scalar integer times only; got {role}={value!r}")
    scalar = value.item() if isinstance(value, np.generic) else value
    return type(value).__module__, type(value).__qualname__, int(scalar)


def read_simple_diagram_inputs(
    diagrams,
    time_list,
    vertex_list,
    propagator_list,
    *,
    backend,
):
    """Read simple diagram getter inputs and immediately snapshot each request.

    Returns ``(specs, bindings, keys_by_occurrence)``. The last value is nested
    by diagram and contraction group, then holds operand keys in einsum order
    (all propagators followed by all vertices).
    """
    xp, backend_name = _backend_module(backend)
    diagram_values = _sequence(diagrams, "diagrams")
    time_values = time_list
    vertex_values = vertex_list
    propagator_values = propagator_list

    source_refs: dict[int, Any] = {}
    request_cache: dict[tuple[Any, ...], tuple[str, Any]] = {}
    specs: dict[str, TensorSpec] = {}
    bindings: dict[str, Any] = {}
    keys_by_occurrence = []

    def snapshot(role, handle, method, args, requested_type):
        source_id = id(handle)
        source_refs.setdefault(source_id, handle)
        tokens = tuple(_time_token(arg, f"{role} getter argument") for arg in args)
        request = (role, source_id, method, requested_type, tokens)
        if request in request_cache:
            return request_cache[request][0]
        value = getattr(handle, method)(*args)
        if role == "propagator" and requested_type in {"VSP", "PSV", "PSP"}:
            value, _effective_type = value
        if (
            backend_name == "cupy"
            and isinstance(value, xp.ndarray)
            and int(value.device.id) != int(xp.cuda.runtime.getDevice())
        ):
            raise ValueError("dynamic diagram inputs must be on the current CuPy device")
        copied = xp.array(value, copy=True)
        key = "dynamic-simple:" + ":".join(
            (role, str(source_id), method, requested_type or "-")
        ) + ":" + repr(tokens)
        bindings[key] = copied
        specs[key] = TensorSpec(tuple(copied.shape), copied.dtype)
        request_cache[request] = (key, None)
        return key

    for diagram_index, diagram in enumerate(diagram_values):
        try:
            groups = diagram.operands
            equations = diagram.subscripts
        except AttributeError as exc:
            raise TypeError(f"diagram {diagram_index} must declare operands and subscripts") from exc
        diagram_keys = []
        for group_index, (operands, _subscripts) in enumerate(zip(groups, equations)):
            input_keys = []
            if hasattr(diagram, "propagator_types") and diagram.propagator_types:
                propagator_types = diagram.propagator_types[group_index]
            else:
                propagator_types = ["VSV"] * len(operands[0])

            for prop_index, item in enumerate(operands[0]):
                route_index, propagator = _table_item(
                    propagator_values,
                    item[0],
                    f"propagator handle index in diagram {diagram_index}, group {group_index}",
                )
                prop_type = propagator_types[prop_index]
                try:
                    method = _SIMPLE_METHODS[prop_type]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"Unknown propagator type: {prop_type}") from exc
                source_position, source_time = _table_item(
                    time_values,
                    item[1],
                    f"source-time position in diagram {diagram_index}, group {group_index}",
                )
                sink_position, sink_time = _table_item(
                    time_values,
                    item[2],
                    f"sink-time position in diagram {diagram_index}, group {group_index}",
                )
                try:
                    input_key = snapshot(
                        "propagator", propagator, method,
                        (source_time, sink_time), prop_type,
                    )
                except (AttributeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Error getting propagator[{route_index}] type {prop_type}: {exc}\n"
                        f"  Propagator type: {type(propagator).__name__}"
                    ) from exc
                input_keys.append(input_key)

            for item in operands[1]:
                vertex_index, vertex = _table_item(
                    vertex_values,
                    item,
                    f"vertex position in diagram {diagram_index}, group {group_index}",
                )
                _time_position, vertex_time = _table_item(
                    time_values,
                    item,
                    f"vertex time position in diagram {diagram_index}, group {group_index}",
                )
                input_keys.append(snapshot("vertex", vertex, "get", (vertex_time,), None))
            diagram_keys.append(tuple(input_keys))
        keys_by_occurrence.append(tuple(diagram_keys))
    return specs, bindings, tuple(keys_by_occurrence)


def _baryon_time(value: Any, role: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{role} must be a scalar integer time")
    return int(value)


def _table_item_baryon(table: Any, index: int, role: str):
    try:
        value = table[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise KeyError(f"missing {role} ID {index}") from exc
    return value


def _require_getter(handle: Any, method: str, role: str, index: int) -> None:
    if handle is None:
        raise TypeError(f"missing {role} handle {index!r}")
    if not callable(getattr(handle, method, None)):
        raise TypeError(f"{role} handle {index!r} must provide {method}()")


def _baryon_key(role: str, handle: Any, *times: int) -> str:
    return f"dynamic-baryon:{role}:{id(handle)}:" + ":".join(str(value) for value in times)


def read_baryon_diagram_inputs(
    topologies,
    time_list,
    vertex_list,
    propagator_list,
    *,
    backend,
):
    """Read already-validated baryon topology inputs with call-local reuse."""
    xp, backend_name = _backend_module(backend)
    topology_values = _sequence(topologies, "topologies")
    times = _sequence(time_list, "time_list")
    vertices = _sequence(vertex_list, "vertex_list")
    if topology_values:
        if len(times) != 2 or len(vertices) != 2:
            raise ValueError("time_list and vertex_list must each have length two")
        times = tuple(_baryon_time(value, f"time_list[{index}]") for index, value in enumerate(times))
    elif (len(times), len(vertices)) not in {(0, 0), (2, 2)}:
        raise ValueError("empty baryon inputs require empty or two-element time and vertex lists")
    elif len(times) == 2:
        times = tuple(_baryon_time(value, f"time_list[{index}]") for index, value in enumerate(times))
    if any(not isinstance(topology, BaryonTopology) for topology in topology_values):
        raise TypeError("topologies must contain BaryonTopology records")
    if not topology_values:
        return {}, {}, ()

    for index, handle in enumerate(vertices):
        _require_getter(handle, "get_components", "vertex", index)

    prepared = []
    for topology in topology_values:
        source_index, sink_index = topology.source_vertex, topology.sink_vertex
        source_handle = vertices[source_index]
        sink_handle = vertices[sink_index]
        source_time, sink_time = times[source_index], times[sink_index]
        props = []
        for line in topology.lines:
            handle = _table_item_baryon(propagator_list, line.propagator, "propagator")
            _require_getter(handle, "get", "propagator", line.propagator)
            props.append((handle, line.propagator))
        prepared.append((source_handle, source_time, sink_handle, sink_time, tuple(props), topology))

    source_refs: dict[int, Any] = {}
    cache: dict[tuple[Any, ...], tuple[str, ...] | str] = {}
    specs: dict[str, TensorSpec] = {}
    bindings: dict[str, Any] = {}

    def snapshot_vertex(handle, time):
        source_id = id(handle)
        source_refs.setdefault(source_id, handle)
        request = ("vertex", source_id, time)
        if request in cache:
            return cache[request]
        value = handle.get_components(time)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("vertex getter must return (spin, elemental)")
        arrays = tuple(value)
        if backend_name == "numpy":
            if any(not isinstance(array, np.ndarray) for array in arrays):
                raise TypeError("NumPy backend requires NumPy getter arrays")
        else:
            if any(not isinstance(array, xp.ndarray) for array in arrays):
                raise TypeError("CuPy backend requires CuPy getter arrays")
            if any(int(array.device.id) != int(xp.cuda.runtime.getDevice()) for array in arrays):
                raise ValueError("CuPy getter arrays must use the current CUDA device")
        copied = tuple(xp.array(array, copy=True) for array in arrays)
        keys = (
            _baryon_key("vertex-spin", handle, time),
            _baryon_key("vertex-elemental", handle, time),
        )
        for key, array in zip(keys, copied, strict=True):
            bindings[key] = array
            specs[key] = TensorSpec(tuple(array.shape), array.dtype)
        cache[request] = keys
        return keys

    def snapshot_propagator(handle, source_time, sink_time):
        source_id = id(handle)
        source_refs.setdefault(source_id, handle)
        request = ("propagator", source_id, source_time, sink_time)
        if request in cache:
            return cache[request]
        array = handle.get(source_time, sink_time)
        if backend_name == "numpy":
            if not isinstance(array, np.ndarray):
                raise TypeError("NumPy backend requires NumPy getter arrays")
        else:
            if not isinstance(array, xp.ndarray):
                raise TypeError("CuPy backend requires CuPy getter arrays")
            if int(array.device.id) != int(xp.cuda.runtime.getDevice()):
                raise ValueError("CuPy getter arrays must use the current CUDA device")
        copied = xp.array(array, copy=True)
        key = _baryon_key("propagator", handle, source_time, sink_time)
        bindings[key] = copied
        specs[key] = TensorSpec(tuple(copied.shape), copied.dtype)
        cache[request] = key
        return key

    term_input_keys = []
    for source_handle, source_time, sink_handle, sink_time, props, topology in prepared:
        source_keys = snapshot_vertex(source_handle, source_time)
        sink_keys = snapshot_vertex(sink_handle, sink_time)
        prop_keys = tuple(
            snapshot_propagator(handle, source_time, sink_time)
            for handle, _propagator_id in props
        )
        term_input_keys.append((source_keys, sink_keys, prop_keys))
    return specs, bindings, tuple(term_input_keys)


__all__ = ["read_simple_diagram_inputs", "read_baryon_diagram_inputs"]
