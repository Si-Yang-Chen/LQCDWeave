"""Explicit-index equation validation copied from the existing compiler."""
from __future__ import annotations
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from .ir import TensorSpec, ContractionTerm
from .indices import AxisPattern

@dataclass(frozen=True, slots=True)
class ParsedTerm:
    source: ContractionTerm
    equation: AxisPattern
    input_labels: tuple[tuple[int, ...], ...]
    output_labels: tuple[int, ...]
    input_keys: tuple[Hashable, ...]
    output_shape: tuple[int, ...]
    label_dimensions: Mapping[int, int]


def parse_term(term: ContractionTerm, specs: Mapping[Hashable, TensorSpec], term_index: int) -> ParsedTerm:
    if not isinstance(term, ContractionTerm):
        raise TypeError(f"term {term_index} must be a ContractionTerm")
    equation = term.subscripts
    input_labels, output = equation.inputs, equation.output
    if len(input_labels) != len(term.inputs):
        raise ValueError(
            f"term {term_index}: equation has {len(input_labels)} operands but "
            f"{len(term.inputs)} input keys were supplied"
        )

    label_dimensions: dict[int, int] = {}
    for operand_index, (key, labels) in enumerate(zip(term.inputs, input_labels, strict=True)):
        if key not in specs:
            raise KeyError(f"term {term_index}: no TensorSpec was supplied for input key {key!r}")
        spec = specs[key]
        if len(labels) != len(spec.shape):
            raise ValueError(
                f"term {term_index}, operand {operand_index} ({key!r}): rank {len(spec.shape)} "
                f"does not match subscript length {len(labels)}"
            )
        local_dimensions: dict[int, int] = {}
        for label, dimension in zip(labels, spec.shape, strict=True):
            previous_local = local_dimensions.get(label)
            if previous_local is not None and previous_local != dimension:
                raise ValueError(
                    f"term {term_index}, operand {operand_index}: repeated label {label!r} "
                    f"has dimensions {previous_local} and {dimension}"
                )
            local_dimensions[label] = dimension
            previous_global = label_dimensions.get(label)
            if previous_global is not None and previous_global != dimension:
                raise ValueError(
                    f"term {term_index}: label {label!r} has incompatible dimensions "
                    f"{previous_global} and {dimension}"
                )
            label_dimensions[label] = dimension

    missing_output = [label for label in output if label not in label_dimensions]
    if missing_output:
        raise ValueError(
            f"term {term_index}: output label(s) {missing_output!r} do not appear in inputs"
        )
    output_shape = tuple(label_dimensions[label] for label in output)
    return ParsedTerm(
        source=term,
        equation=equation,
        input_labels=input_labels,
        output_labels=output,
        input_keys=term.inputs,
        output_shape=output_shape,
        label_dimensions=label_dimensions,
    )

