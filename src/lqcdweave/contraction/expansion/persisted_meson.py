"""Getter-free descriptions for persisted scalar VSV Meson workloads."""
from __future__ import annotations

from collections.abc import Mapping
import operator

import numpy as np

from lqcdweave.contracts.correlator.api import (
    ContractionTerm,
    parse_axis_pattern,
    TensorSpec,
    WorkloadSpec,
)


def _integer(value, role):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{role} must be a non-boolean integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{role} must be an integer scalar") from exc


def _times(values):
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise TypeError("times must be a sequence of scalar integer labels") from exc
    return tuple(_integer(value, "time") for value in supplied)


def _position(value, *, role, diagram_index, group_index, times):
    index = _integer(value, f"{role} position in diagram {diagram_index}, group {group_index}")
    if not 0 <= index < len(times):
        raise ValueError(
            f"{role} position {index} in diagram {diagram_index}, group {group_index} "
            f"must be within [0, {len(times)})"
        )
    return index


def _sequence(value, role):
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{role} must be a sequence, not text")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{role} must be a sequence") from exc


def describe_meson_diagrams(
    diagrams,
    times,
    *,
    vertex_specs,
    propagator_specs,
    vertex_sources,
    propagator_sources,
):
    """Describe scalar-output VSV diagrams without reading numerical values.

    ``vertex_specs`` maps ``(vertex_handle_index, actual_time)`` to
    ``TensorSpec``; ``propagator_specs`` maps
    ``(propagator_handle_index, source_time, sink_time)`` to ``TensorSpec``.
    The corresponding source mappings are caller-declared, non-empty JSON
    metadata keyed by handle index. Distinct handle indexes remain distinct
    even when their shapes and declarations agree.
    """
    times = _times(times)
    for name, table in (
        ("vertex_specs", vertex_specs),
        ("propagator_specs", propagator_specs),
        ("vertex_sources", vertex_sources),
        ("propagator_sources", propagator_sources),
    ):
        if not isinstance(table, Mapping):
            raise TypeError(f"{name} must be a mapping")

    specs, sources, terms, recipes = {}, {}, [], []

    def request(kind, index_value, args, *, diagram_index, group_index):
        index = _integer(
            index_value,
            f"{kind} handle index in diagram {diagram_index}, group {group_index}",
        )
        if index < 0:
            raise ValueError(
                f"negative {kind} handle index {index} in diagram {diagram_index}, group {group_index}"
            )
        table = vertex_specs if kind == "vertex" else propagator_specs
        origin = vertex_sources if kind == "vertex" else propagator_sources
        spec_key = (index, *args)
        try:
            spec = table[spec_key]
        except KeyError as exc:
            raise KeyError(f"missing {kind} TensorSpec declaration for {spec_key!r}") from exc
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"declared {kind} inputs must use TensorSpec")
        try:
            declared_source = origin[index]
        except KeyError as exc:
            raise KeyError(f"missing {kind} source declaration for handle {index}") from exc
        if not isinstance(declared_source, Mapping) or not declared_source:
            raise ValueError(f"{kind} handle {index} requires non-empty JSON source metadata")
        key = f"{kind}:{index}:" + ":".join(str(arg) for arg in args)
        specs[key] = spec
        sources[key] = {
            "kind": kind,
            "handle": index,
            "method": "get",
            "times": list(args),
            "source": dict(declared_source),
        }
        return key

    try:
        diagram_list = tuple(diagrams)
    except TypeError as exc:
        raise TypeError("diagrams must be an iterable of Diagram-like records") from exc

    for diagram_index, diagram in enumerate(diagram_list):
        try:
            operands = _sequence(diagram.operands, f"diagram {diagram_index} operands")
            equations = _sequence(diagram.subscripts, f"diagram {diagram_index} equations")
        except AttributeError as exc:
            raise TypeError(f"diagram {diagram_index} must declare operands and subscripts") from exc
        if len(operands) != len(equations):
            raise ValueError(f"diagram {diagram_index} operands and equations differ in length")

        prop_types = getattr(diagram, "propagator_types", None)
        if prop_types is None:
            declared_types = None  # The simple-Diagram convention assumes VSV.
        else:
            declared_types = _sequence(prop_types, f"diagram {diagram_index} propagator_types")
            if len(declared_types) != len(operands):
                raise ValueError(
                    f"diagram {diagram_index} requires one propagator_types group per contraction group"
                )

        recipe = []
        for group_index, (raw_group, equation) in enumerate(zip(operands, equations)):
            try:
                group = _sequence(raw_group, f"diagram {diagram_index} group {group_index}")
            except TypeError as exc:
                raise TypeError(
                    f"diagram {diagram_index} group {group_index} must contain propagators and vertices"
                ) from exc
            if len(group) != 2:
                raise ValueError(
                    f"diagram {diagram_index} group {group_index} must contain propagators and vertices"
                )
            try:
                propagator_edges = _sequence(
                    group[0], f"diagram {diagram_index} group {group_index} propagator edges"
                )
                vertex_handles = _sequence(
                    group[1], f"diagram {diagram_index} group {group_index} vertex handles"
                )
            except TypeError as exc:
                raise TypeError(f"diagram {diagram_index} group {group_index} has malformed operands") from exc

            if declared_types is not None:
                edge_types = _sequence(
                    declared_types[group_index],
                    f"diagram {diagram_index} group {group_index} propagator types",
                )
                if len(edge_types) != len(propagator_edges):
                    raise ValueError(
                        f"diagram {diagram_index} group {group_index} requires one propagator type per edge"
                    )
                if any(kind != "VSV" for kind in edge_types):
                    raise NotImplementedError("persisted Meson frontend supports VSV propagators only")

            keys = []
            for edge_index, edge in enumerate(propagator_edges):
                edge = _sequence(
                    edge,
                    f"diagram {diagram_index} group {group_index} propagator edge {edge_index}",
                )
                if len(edge) != 3:
                    raise ValueError(
                        f"diagram {diagram_index} group {group_index} propagator edge {edge_index} "
                        "must contain handle, source position, and sink position"
                    )
                handle, source_position, sink_position = edge
                source_position = _position(
                    source_position, role="source-time", diagram_index=diagram_index,
                    group_index=group_index, times=times,
                )
                sink_position = _position(
                    sink_position, role="sink-time", diagram_index=diagram_index,
                    group_index=group_index, times=times,
                )
                keys.append(request(
                    "propagator", handle,
                    (times[source_position], times[sink_position]),
                    diagram_index=diagram_index, group_index=group_index,
                ))

            for vertex_value in vertex_handles:
                vertex = _integer(
                    vertex_value,
                    f"vertex position in diagram {diagram_index}, group {group_index}",
                )
                if vertex < 0:
                    raise ValueError(
                        f"negative vertex position {vertex} in diagram {diagram_index}, group {group_index}"
                    )
                if vertex >= len(times):
                    raise ValueError(
                        f"vertex position {vertex} in diagram {diagram_index}, group {group_index} "
                        f"must be within [0, {len(times)})"
                    )
                keys.append(request(
                    "vertex", vertex, (times[vertex],),
                    diagram_index=diagram_index, group_index=group_index,
                ))

            if not isinstance(equation, str):
                raise TypeError(f"diagram {diagram_index} group {group_index} equation must be a string")
            equation = "".join(equation.split())
            if not equation:
                raise ValueError(f"diagram {diagram_index} group {group_index} equation is empty")
            pattern = parse_axis_pattern(equation)
            if pattern.output:
                raise NotImplementedError(
                    "persisted Meson frontend requires scalar group outputs"
                )
            recipe.append(len(terms))
            terms.append(ContractionTerm(tuple(keys), pattern))
        recipes.append(recipe)

    if not terms:
        raise ValueError("describe at least one nonempty contraction group")
    return WorkloadSpec(
        specs=specs,
        terms=tuple(terms),
        input_sources=sources,
        convention="meson-simple-v1",
        output_groups=tuple(tuple(recipe) for recipe in recipes if recipe),
        metadata={
            "adapter": "meson-simple-v1",
            "output_recipes": recipes,
            # Per-term outputs remain live during neutral multiplication and
            # final stacking. Reserve three complex128 scalar arrays per diagram.
            "assembly_extra_bytes": 48 * len(recipes),
        },
    )


__all__ = ["describe_meson_diagrams"]
