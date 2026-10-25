"""Getter-free expansion for the dynamic simple and baryon diagram frontends.

The simple einsum normalization was migrated from the now-retired
``contraction.diagram_adapter`` source. The baryon graph parser and metadata
checks were migrated from ``contraction.baryon_adapter``, constructing numerical-free
records from ``contracts.correlator``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numbers
import operator
from typing import Any

import numpy as np

from lqcdweave.contracts.correlator.api import (
    AxisPattern,
    BaryonLine,
    BaryonTerm,
    BaryonTopology,
    BaryonVertex,
    ContractionTerm,
    parse_axis_pattern,
    TensorSpec,
)


def _sequence(value: Any, role: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{role} must be a sequence, not text")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{role} must be a sequence") from exc


def _normalise_subscripts(text: Any, operand_count: int) -> AxisPattern:
    if not isinstance(text, str):
        raise TypeError(f"contraction subscripts must be a string, got {type(text)!r}")
    text = "".join(text.split())
    if "..." in text:
        raise NotImplementedError("dynamic simple diagrams do not support ellipses in subscripts")
    pattern = parse_axis_pattern(text)
    if len(pattern.inputs) != operand_count:
        raise ValueError(
            f"subscript operand count {len(pattern.inputs)} does not match loaded operand count "
            f"{operand_count}: {text!r}"
        )
    return pattern


def _simple_group_count(diagram: Any, diagram_index: int) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    try:
        groups = _sequence(diagram.operands, f"diagram {diagram_index} operands")
        equations = _sequence(diagram.subscripts, f"diagram {diagram_index} subscripts")
    except AttributeError as exc:
        raise TypeError(f"diagram {diagram_index} must declare operands and subscripts") from exc
    return groups, equations


def expand_simple_diagrams(diagrams, keys_by_occurrence, specs):
    """Turn simple diagram equations into terms and ordered output recipes.

    ``keys_by_occurrence`` is nested by diagram and contraction group. Each
    group contains keys in the legacy operand order: propagators, then
    vertices. ``output_recipes`` groups term indexes by diagram; result
    assembly multiplies the term outputs inside each recipe.
    """
    diagram_values = _sequence(diagrams, "diagrams")
    occurrence_values = _sequence(keys_by_occurrence, "keys_by_occurrence")
    if len(occurrence_values) != len(diagram_values):
        raise ValueError("keys_by_occurrence must contain one entry per diagram")
    if not isinstance(specs, Mapping):
        raise TypeError("specs must be a mapping of input keys to TensorSpec objects")
    for key, spec in specs.items():
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError(f"input key {key!r} is not hashable") from exc
        if not isinstance(spec, TensorSpec):
            raise TypeError(f"spec for input key {key!r} must be a TensorSpec")

    terms: list[ContractionTerm] = []
    output_recipes = []
    for diagram_index, (diagram, raw_keys) in enumerate(zip(diagram_values, occurrence_values, strict=True)):
        groups, equations = _simple_group_count(diagram, diagram_index)
        group_keys = _sequence(raw_keys, f"diagram {diagram_index} input-key groups")
        matched_group_count = min(len(groups), len(equations))
        if len(group_keys) != matched_group_count:
            raise ValueError(
                f"diagram {diagram_index} has {matched_group_count} matched groups but received "
                f"{len(group_keys)} input-key groups"
            )
        recipe = []
        for group_index, (raw_group, equation, raw_inputs) in enumerate(
            zip(
                groups[:matched_group_count],
                equations[:matched_group_count],
                group_keys,
                strict=True,
            )
        ):
            group = _sequence(raw_group, f"diagram {diagram_index} group {group_index}")
            if len(group) < 2:
                raise ValueError(
                    f"diagram {diagram_index} group {group_index} must contain propagators and vertices"
                )
            prop_edges = _sequence(
                group[0], f"diagram {diagram_index} group {group_index} propagator edges"
            )
            vertex_positions = _sequence(
                group[1], f"diagram {diagram_index} group {group_index} vertex positions"
            )
            input_keys = _sequence(raw_inputs, f"diagram {diagram_index} group {group_index} input keys")
            expected_count = len(prop_edges) + len(vertex_positions)
            if len(input_keys) != expected_count:
                raise ValueError(
                    f"diagram {diagram_index} group {group_index} has {expected_count} operands "
                    f"but received {len(input_keys)} input keys"
                )
            missing = [key for key in input_keys if key not in specs]
            if missing:
                raise KeyError(f"missing TensorSpec declaration for input key(s): {missing!r}")
            terms.append(
                ContractionTerm(tuple(input_keys), _normalise_subscripts(equation, len(input_keys)))
            )
            recipe.append(len(terms) - 1)
        output_recipes.append(tuple(recipe))
    return tuple(terms), tuple(output_recipes)


def _is_zero(value: Any) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value == 0
    )


def _cell_values(cell: Any, location: tuple[int, int]) -> list[list[Any]] | None:
    if _is_zero(cell):
        return None
    if isinstance(cell, (str, bytes)):
        raise TypeError(f"adjacency cell {location} must be a 3x3 integer block or zero")
    try:
        rows = list(cell)
    except TypeError as exc:
        raise TypeError(f"adjacency cell {location} must be a 3x3 integer block or zero") from exc
    if len(rows) != 3:
        raise ValueError(f"adjacency cell {location} must have shape 3x3")
    result = []
    for row in rows:
        try:
            values = list(row)
        except TypeError as exc:
            raise ValueError(f"adjacency cell {location} must have shape 3x3") from exc
        if len(values) != 3:
            raise ValueError(f"adjacency cell {location} must have shape 3x3")
        result.append(values)
    if all(_is_zero(value) for row in result for value in row):
        return None
    return result


def extract_baryon_topology(adjacency_matrix: Any) -> BaryonTopology:
    """Parse the two-vertex graph into the unique correlator topology record."""
    try:
        outer = list(adjacency_matrix)
    except TypeError as exc:
        raise TypeError("adjacency matrix must be a 2x2 matrix") from exc
    if len(outer) != 2 or any(
        not isinstance(row, Sequence) and not hasattr(row, "__iter__") for row in outer
    ):
        raise ValueError("adjacency matrix must be a 2x2 matrix")
    rows = [list(row) for row in outer]
    if any(len(row) != 2 for row in rows):
        raise ValueError("adjacency matrix must be a 2x2 matrix")

    edge_cells: list[tuple[int, int, list[list[Any]]]] = []
    for source_vertex, row in enumerate(rows):
        for sink_vertex, cell in enumerate(row):
            block = _cell_values(cell, (source_vertex, sink_vertex))
            if block is not None:
                edge_cells.append((source_vertex, sink_vertex, block))
    if len(edge_cells) != 1:
        raise ValueError("baryon graph must contain exactly one directed edge cell")
    source_vertex, sink_vertex, block = edge_cells[0]
    if source_vertex == sink_vertex:
        raise ValueError("self loops are not valid baryon graphs")

    edges = []
    for source_slot, row in enumerate(block):
        for sink_slot, value in enumerate(row):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("boolean propagator IDs are invalid")
            if not isinstance(value, (int, np.integer)):
                raise TypeError("propagator IDs must be integers")
            if value < 0:
                raise ValueError("propagator IDs must be non-negative")
            if value:
                edges.append(BaryonLine(int(value), source_slot, sink_slot))
    if len(edges) != 3:
        raise ValueError("baryon graph must contain exactly three lines")
    if {line.source_slot for line in edges} != {0, 1, 2} or {line.sink_slot for line in edges} != {0, 1, 2}:
        raise ValueError("baryon slots must be a bijection")
    edges.sort(key=lambda line: line.sink_slot)
    return BaryonTopology(source_vertex, sink_vertex, tuple(edges))


def _baryon_metadata(diagram: Any) -> None:
    for name in ("sampling_times", "temporal_extent", "usedNp"):
        if getattr(diagram, name, None) is not None:
            raise NotImplementedError(f"{name} metadata is not supported by the baryon dynamic frontend")
    value = getattr(diagram, "vertex_list", None)
    if value is not None:
        try:
            active = any(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("vertex_list metadata must be a scalar flag sequence") from exc
        if active:
            raise NotImplementedError("vertex_list metadata is not supported by the baryon dynamic frontend")
    offsets = getattr(diagram, "leg_offsets", None)
    if offsets is not None:
        try:
            active = any(item is not None for item in offsets)
        except (TypeError, ValueError) as exc:
            raise TypeError("leg_offsets metadata must be a sequence") from exc
        if active:
            raise NotImplementedError("leg_offsets metadata is not supported by the baryon dynamic frontend")
    types = getattr(diagram, "propagator_types", None)
    if types:
        if isinstance(types, str):
            groups = ((types,),)
        else:
            try:
                groups = tuple(types)
            except TypeError as exc:
                raise TypeError("propagator_types metadata must be a sequence") from exc
            if groups and all(isinstance(item, str) for item in groups):
                groups = (groups,)
        for group in groups:
            try:
                if any(item != "VSV" for item in group):
                    raise NotImplementedError("only VSV propagators are supported by baryon diagrams")
            except TypeError as exc:
                raise TypeError("propagator_types metadata must contain sequences") from exc


def _topology(diagram: Any) -> BaryonTopology:
    _baryon_metadata(diagram)
    getter = getattr(diagram, "get_baryon_topology", None)
    if getter is not None:
        if not callable(getter):
            raise TypeError("get_baryon_topology must be callable")
        result = getter()
    else:
        try:
            adjacency = diagram.adjacency_matrix
        except AttributeError as exc:
            raise TypeError("diagram must provide get_baryon_topology() or adjacency_matrix") from exc
        result = extract_baryon_topology(adjacency)
    if not isinstance(result, BaryonTopology):
        raise TypeError("get_baryon_topology() must return a BaryonTopology")
    return result


def _integer_time(value: Any, role: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{role} must be a scalar integer time")
    return int(value)


def _normalise_coefficients(coefficients: Any, count: int) -> tuple[Any, ...]:
    if coefficients is None:
        result = (1,) * count
    else:
        try:
            result = tuple(coefficients)
        except TypeError as exc:
            raise TypeError("coefficients must be an iterable with one value per diagram") from exc
        if len(result) != count:
            raise ValueError("coefficients must have one value per diagram")
    for coefficient in result:
        if not isinstance(coefficient, (numbers.Number, np.bool_)):
            raise TypeError("term coefficient must be a scalar number")
    return result


def validate_baryon_diagram_batch(
    diagrams,
    times,
    vertex_count,
    *,
    coefficients=None,
):
    """Preflight baryon topology, time arity, and coefficients without reading arrays."""
    diagram_values = _sequence(diagrams, "diagrams")
    raw_times = _sequence(times, "times")
    if isinstance(vertex_count, (bool, np.bool_)):
        raise TypeError("vertex_count must be an integer")
    try:
        normalized_vertex_count = operator.index(vertex_count)
    except TypeError as exc:
        raise TypeError("vertex_count must be an integer") from exc

    if diagram_values:
        valid_lengths = len(raw_times) == 2 and normalized_vertex_count == 2
    else:
        valid_lengths = (len(raw_times), normalized_vertex_count) in {(0, 0), (2, 2)}
    if not valid_lengths:
        raise ValueError("times and vertex list must each have length two")
    normalized_times = tuple(
        _integer_time(value, f"times[{index}]") for index, value in enumerate(raw_times)
    )
    normalized_coefficients = _normalise_coefficients(coefficients, len(diagram_values))
    topologies = tuple(_topology(diagram) for diagram in diagram_values)
    return topologies, normalized_times, normalized_coefficients


def expand_baryon_diagrams(topologies, term_input_keys, coefficients):
    """Create the unique C baryon terms from validated topology and binding keys."""
    topology_values = _sequence(topologies, "topologies")
    input_values = _sequence(term_input_keys, "term_input_keys")
    coefficient_values = _normalise_coefficients(coefficients, len(topology_values))
    if len(input_values) != len(topology_values):
        raise ValueError("term_input_keys must contain one entry per topology")
    terms = []
    for index, (topology, raw_keys, coefficient) in enumerate(
        zip(topology_values, input_values, coefficient_values, strict=True)
    ):
        if not isinstance(topology, BaryonTopology):
            raise TypeError(f"topologies[{index}] must be a BaryonTopology")
        keys = _sequence(raw_keys, f"term_input_keys[{index}]")
        if len(keys) != 3:
            raise ValueError(f"term_input_keys[{index}] must contain source, sink, and propagator keys")
        source_keys = _sequence(keys[0], f"term_input_keys[{index}].source")
        sink_keys = _sequence(keys[1], f"term_input_keys[{index}].sink")
        propagator_keys = _sequence(keys[2], f"term_input_keys[{index}].propagators")
        if len(source_keys) != 2 or len(sink_keys) != 2 or len(propagator_keys) != 3:
            raise ValueError(f"term_input_keys[{index}] must have two vertex keys and three propagator keys")
        terms.append(
            BaryonTerm(
                BaryonVertex(source_keys[0], source_keys[1]),
                BaryonVertex(sink_keys[0], sink_keys[1]),
                tuple(propagator_keys),
                tuple(line.source_slot for line in topology.lines),
                coefficient,
            )
        )
    return tuple(terms)


__all__ = [
    "expand_simple_diagrams",
    "extract_baryon_topology",
    "validate_baryon_diagram_batch",
    "expand_baryon_diagrams",
]
