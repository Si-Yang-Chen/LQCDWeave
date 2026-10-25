"""Immutable integer axis connections shared by expansion and planning.

Axis IDs describe equality, not execution order or physical axis names. This
record replaces the search workspace's private AxisPattern definition.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator
import re


def _axes(values) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("axis IDs must be integers")
    result = []
    for value in values:
        if type(value) is int:
            axis = value
        else:
            if isinstance(value, bool):
                raise TypeError("boolean values cannot be axis IDs")
            axis = operator.index(value)
        if axis < 0:
            raise ValueError("axis IDs must be non-negative")
        result.append(axis)
    return tuple(result)


def axis_ids(values) -> tuple[int, ...]:
    """Normalize explicitly supplied legacy axis labels at the input boundary."""
    if isinstance(values, str):
        if not re.fullmatch(r"[A-Za-z]*", values):
            raise ValueError("external axis labels must be ASCII letters")
        return tuple(map(ord, values))
    return _axes(values)


def parse_axis_pattern(value) -> AxisPattern:
    """Parse an external equation once, including its implicit output rule."""
    if isinstance(value, AxisPattern):
        return value
    if not isinstance(value, str):
        raise TypeError("contraction relation must be AxisPattern or an external string")
    text = "".join(value.split())
    if text.count("->") > 1 or "..." in text:
        raise ValueError("external equation requires explicit axes without ellipsis")
    if "->" in text:
        left, output = text.split("->")
        return AxisPattern(tuple(axis_ids(axes) for axes in left.split(",")), axis_ids(output))
    inputs = tuple(axis_ids(axes) for axes in text.split(","))
    occurrences = {}
    for axes in inputs:
        for axis in axes:
            occurrences[axis] = occurrences.get(axis, 0) + 1
    return AxisPattern(inputs, tuple(sorted(axis for axis, count in occurrences.items() if count == 1)))


@dataclass(frozen=True, slots=True)
class AxisPattern:
    """Ordered operand axes and output axes, with no einsum string.

Repeated input IDs express diagonals, shared IDs express alignment, and IDs
absent from output are summed. IDs are unbounded non-negative integers.
"""

    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]

    def __post_init__(self):
        inputs = tuple(_axes(axes) for axes in self.inputs)
        output = _axes(self.output)
        if not inputs:
            raise ValueError("an axis pattern needs at least one operand")
        if output:
            output_ids = set(output)
            if len(output_ids) != len(output):
                raise ValueError("output axis IDs must be unique")
            if not output_ids <= set().union(*inputs):
                raise ValueError("output axis IDs must occur in an input")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "output", output)

    def normalized(self) -> AxisPattern:
        """Rename dummy IDs by first occurrence, preserving every position."""
        ids = {}

        def encode(axes):
            return tuple(ids.setdefault(axis, len(ids)) for axis in axes)

        return AxisPattern(tuple(encode(axes) for axes in self.inputs), encode(self.output))

    def as_dict(self) -> dict:
        """Pure JSON description, retaining ordered integer connections."""
        return {"inputs": [list(axes) for axes in self.inputs], "output": list(self.output)}

    def dimensions(self, shapes) -> dict[int, int]:
        """Validate ranks, diagonal extents and aligned extents without arrays."""
        shapes = tuple(tuple(shape) for shape in shapes)
        if len(shapes) != len(self.inputs):
            raise ValueError("one shape is required per operand")
        dimensions = {}
        for axes, shape in zip(self.inputs, shapes, strict=True):
            if len(axes) != len(shape):
                raise ValueError("operand rank does not match its axis IDs")
            for axis, size in zip(axes, shape, strict=True):
                if isinstance(size, bool):
                    raise TypeError("dimensions must be non-negative integers")
                size = operator.index(size)
                if size < 0:
                    raise ValueError("dimensions must be non-negative")
                if dimensions.setdefault(axis, size) != size:
                    raise ValueError(f"axis {axis} has incompatible dimensions")
        return dimensions

    def output_shape(self, shapes) -> tuple[int, ...]:
        dimensions = self.dimensions(shapes)
        return tuple(dimensions[axis] for axis in self.output)
