"""Bounded CPU reference for equal-time meson graphs with VSV quark lines."""

import numpy as np
from lqcdweave.operators.correlator.api import Diagram
from lqcdweave.contracts.correlator.api import AxisPattern
from sympy import Add, Mul, Pow


# The index pairing and tensor axis order follow the legacy v2v graph analyser.


def _einsum_arguments(pattern, operands):
    """Build NumPy's integer-list einsum arguments without an X dependency."""
    local = pattern.normalized()
    arguments = []
    for operand, axes in zip(operands, local.inputs, strict=True):
        arguments.extend((operand, list(axes)))
    arguments.append(list(local.output))
    return tuple(arguments)


def evaluate_vsv_diagram(diagram, vertex_getter, propagator_getter, time_map):
    """Contract one meson-only Diagram using existing, explicitly supplied blocks.

    vertex_getter(row, time) returns (left_spin,right_spin,left_mode,right_mode).
    propagator_getter(tag, sink_time, source_time) returns
    (sink_spin,source_spin,sink_mode,source_mode). Inputs own their data lifetime.
    This reference path supports one incoming and outgoing VSV line per vertex.
    """
    if not isinstance(diagram, Diagram):
        raise TypeError("a symbolic Diagram is required")
    matrix = diagram.diagram.adjacency_matrix
    count = len(diagram.vertex_list)
    if len(matrix) != count or len(diagram.time_list) != count or any(len(row) != count for row in matrix):
        raise ValueError("diagram vertices and adjacency matrix differ")
    times = []
    for symbol in diagram.time_list:
        if symbol not in time_map or type(time_map[symbol]) is not int:
            raise ValueError("each symbolic vertex time needs one physical integer time")
        times.append(time_map[symbol])

    edges = []
    outgoing, incoming = [0] * count, [0] * count
    for source, row in enumerate(matrix):
        for sink, label in enumerate(row):
            if label == 0:
                continue
            if type(label) is not int or not 0 < label < len(diagram.propagator_list):
                raise ValueError("this reference accepts one integer VSV label per edge")
            outgoing[source] += 1
            incoming[sink] += 1
            edges.append((label, source, sink))
    if not edges or any(left != 1 or right != 1 for left, right in zip(outgoing, incoming)):
        raise ValueError("each meson vertex requires one incoming and outgoing VSV line")
    if 4 * len(edges) > 52:
        raise ValueError("diagram exceeds NumPy integer-index capacity")

    operands, input_axes = [], []
    vertex_left, vertex_right = [None] * count, [None] * count
    mode_offset = 2 * len(edges)
    for edge_index, (label, source, sink) in enumerate(edges):
        left, right = 2 * edge_index, 2 * edge_index + 1
        vertex_left[source] = (left, mode_offset + left)
        vertex_right[sink] = (right, mode_offset + right)
        value = np.asarray(propagator_getter(diagram.propagator_list[label],
                                              times[sink], times[source]))
        if value.ndim != 4 or value.shape[:2] != (4, 4) or value.shape[2] != value.shape[3]:
            raise ValueError("VSV block must have axes (sink_spin,source_spin,sink_mode,source_mode)")
        operands.append(value)
        input_axes.append((right, left, mode_offset + right, mode_offset + left))

    mode_count = operands[0].shape[2]
    for index, row in enumerate(diagram.vertex_list):
        value = np.asarray(vertex_getter(row, times[index]))
        if value.shape != (4, 4, mode_count, mode_count):
            raise ValueError("meson vertex and VSV mode/spin axes differ")
        left_spin, left_mode = vertex_left[index]
        right_spin, right_mode = vertex_right[index]
        operands.append(value)
        input_axes.append((left_spin, right_spin, left_mode, right_mode))
    pattern = AxisPattern(tuple(input_axes), ())
    return complex(np.einsum(*_einsum_arguments(pattern, operands),
                             optimize=False, dtype=np.complex128))


def evaluate_vsv_expression(expr, vertex_getter, propagator_getter, time_map):
    """Evaluate one SymPy scalar expression of VSV Diagram atoms."""
    if isinstance(expr, Diagram):
        return evaluate_vsv_diagram(expr, vertex_getter, propagator_getter, time_map)
    if isinstance(expr, Add):
        return sum((evaluate_vsv_expression(arg, vertex_getter, propagator_getter, time_map)
                    for arg in expr.args), 0j)
    if isinstance(expr, Mul):
        result = 1 + 0j
        for arg in expr.args:
            result *= evaluate_vsv_expression(arg, vertex_getter, propagator_getter, time_map)
        return result
    if isinstance(expr, Pow):
        if not expr.exp.is_Integer or expr.exp < 0:
            raise ValueError("only nonnegative integer graph powers are supported")
        return evaluate_vsv_expression(expr.base, vertex_getter, propagator_getter,
                                       time_map) ** int(expr.exp)
    if getattr(expr, "is_number", False):
        return complex(expr)
    raise ValueError(f"unsupported symbolic graph factor {type(expr).__name__}")


def evaluate_vsv_matrix(symbolic_matrix, vertex_getter, propagator_getter, time_map):
    """Evaluate every operator pair without changing its symbolic matrix axes."""
    if not isinstance(symbolic_matrix, np.ndarray) or symbolic_matrix.dtype != object:
        raise ValueError("expected a NumPy object matrix from gen_correlator")
    result = np.empty(symbolic_matrix.shape, dtype=np.complex128)
    for index in np.ndindex(symbolic_matrix.shape):
        result[index] = evaluate_vsv_expression(symbolic_matrix[index], vertex_getter,
                                                 propagator_getter, time_map)
    return result
